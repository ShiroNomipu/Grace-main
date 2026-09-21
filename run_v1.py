"""
V1: Qwen2-VL-2B + LoRA（4-bit QLoRA），纯文本 ERC。F1 地基，过 R1 No-Go 闸门。

训练范式（生成式，InstructERC 风格）：
  输入 = system + user(对话上下文 + 问题)；目标 = 情感词（assistant）。
  CE loss 只算在答案 token 上（prompt 全部 mask 为 -100）。
评估：复用 infer.evaluate 的候选打分（与 V0 同口径，可比）。

质量护栏：
  1. 断言 LoRA 可训练参数 > 0 且占比合理
  2. tiny-overfit sanity：8 样本过拟合，loss 必须降到接近 0（证明梯度通、能学）
  3. 固定种子；dev weighted-F1 early stopping；指标+adapter 落盘
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

import config as C
from dataset_spec import get_spec
from infer import load_model_and_tokenizer, evaluate, print_metrics


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_train_example(tokenizer, sample, max_len=C.MAX_SEQ_LEN):
    """构造 (input_ids, labels)。labels 只在答案 token 上有效，其余 -100。"""
    msgs = sample["messages"]
    prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    full_ids = tokenizer.apply_chat_template(
        msgs + [{"role": "assistant", "content": sample["label_word"]}],
        add_generation_prompt=False, tokenize=True)

    # 左截断保护：绝不截断答案；超长时从 prompt 左侧砍
    if len(full_ids) > max_len:
        ans_len = len(full_ids) - len(prompt_ids)
        keep_prompt = max_len - ans_len
        assert keep_prompt > 0, "answer longer than max_len, impossible for emotion word"
        full_ids = full_ids[-max_len:]
        prompt_ids = prompt_ids[:keep_prompt]   # 仅用于确定 mask 边界
        prompt_keep = max_len - ans_len
    else:
        prompt_keep = len(prompt_ids)

    labels = [-100] * prompt_keep + full_ids[prompt_keep:]
    assert len(labels) == len(full_ids)
    assert any(l != -100 for l in labels), "all labels masked!"
    return full_ids, labels


def collate_batch(tokenizer, batch_examples):
    """把若干 (input_ids, labels) 右 padding 成 batch tensor。
    pad 位置：input_ids 用 pad_id，attention_mask=0，labels=-100（不计 loss）。"""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    maxlen = max(len(ids) for ids, _ in batch_examples)
    input_ids, attn, labels = [], [], []
    for ids, labs in batch_examples:
        p = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * p)
        attn.append([1] * len(ids) + [0] * p)
        labels.append(labs + [-100] * p)
    return (torch.tensor(input_ids), torch.tensor(attn), torch.tensor(labels))


def forward_batch(model, batch, device):
    input_ids, attn, labels = (t.to(device) for t in batch)
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    return out.loss


def setup_lora(model, rank=None, alpha=None):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    rank = C.LORA_R if rank is None else rank
    alpha = (C.LORA_ALPHA if rank == C.LORA_R else 2 * rank) if alpha is None else alpha
    lora = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=C.LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM",
        # 仅命中语言模型的线性层（Qwen2-VL 视觉塔用 qkv/proj 命名，不会被误中）
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] trainable params: {trainable:,} ({100*trainable/total:.3f}% of {total:,})")
    assert trainable > 0, "no trainable params! LoRA not attached"
    assert trainable < total * 0.05, "trainable ratio too high, target_modules likely wrong"
    return model


def tiny_overfit_sanity(model, tokenizer, samples, n=8, steps=60):
    """过拟合 n 个样本，loss 必须显著下降——证明梯度通、batch 路径正确、模型能学。"""
    print(f"\n[SANITY] tiny-overfit on {n} samples (batch={C.BATCH_SIZE}) for {steps} steps ...")
    device = model.device
    subset = samples[:n]
    examples = [build_train_example(tokenizer, s) for s in subset]
    batch = collate_batch(tokenizer, examples[:C.BATCH_SIZE])   # 固定一个 batch 反复过拟合
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    model.train()
    first_loss, last_loss = None, None
    for step in range(steps):
        loss = forward_batch(model, batch, device)
        loss.backward()
        opt.step(); opt.zero_grad()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        if step % 10 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={loss.item():.4f}")

    print(f"[SANITY] first_loss={first_loss:.4f} -> last_loss={last_loss:.4f}")
    assert last_loss < first_loss * 0.3, (
        f"overfit failed: loss didn't drop enough ({first_loss:.3f}->{last_loss:.3f}). "
        f"梯度可能没通/LoRA没生效/数据有问题")
    print("[SANITY] PASS: model can learn the task (gradients flow).\n")


def train_full(model, tokenizer, D, spec, ctx=None, max_len=None, tag="v1"):
    from transformers import get_linear_schedule_with_warmup
    device = model.device
    max_len = C.MAX_SEQ_LEN if max_len is None else max_len
    train = D.load_split("train", context_window=ctx)
    dev = D.load_split("dev", context_window=ctx)
    print(f"train={len(train)}  dev={len(dev)}  ctx={ctx or C.CONTEXT_WINDOW} max_len={max_len}")

    examples = [build_train_example(tokenizer, s, max_len=max_len) for s in train]

    n_batches = len(examples) // C.BATCH_SIZE
    total_steps = (n_batches // C.GRAD_ACCUM) * C.EPOCHS
    warmup = int(total_steps * C.WARMUP_RATIO)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=C.LR)
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    best_dev_f1 = -1.0
    best_dir = os.path.join(C.OUTPUT_DIR, f"{tag}_lora_best")
    os.makedirs(best_dir, exist_ok=True)
    loss_log, dev_log = [], []     # 训练曲线落盘（loss / dev-F1）
    global_batch = 0

    for epoch in range(C.EPOCHS):
        model.train()
        idxs = list(range(len(examples)))
        random.shuffle(idxs)
        # 切成 batch（丢弃最后不足一个 batch 的尾巴，保持稳定）
        batches = [idxs[i:i + C.BATCH_SIZE]
                   for i in range(0, len(idxs) - C.BATCH_SIZE + 1, C.BATCH_SIZE)]
        opt.zero_grad()
        pbar = tqdm(batches, desc=f"epoch {epoch}")
        for bi, batch_idx in enumerate(pbar):
            batch = collate_batch(tokenizer, [examples[j] for j in batch_idx])
            raw_loss = forward_batch(model, batch, device)          # 原始 loss（未除梯度累积）
            (raw_loss / C.GRAD_ACCUM).backward()
            global_batch += 1
            cur_lr = sched.get_last_lr()[0]
            loss_log.append((epoch, global_batch, float(raw_loss.item()), float(cur_lr)))
            pbar.set_postfix(loss=f"{raw_loss.item():.3f}", lr=f"{cur_lr:.2e}")
            if (bi + 1) % C.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], C.MAX_GRAD_NORM)
                opt.step(); sched.step(); opt.zero_grad()

        # epoch 末在 dev 上评估（候选打分）
        model.eval()
        m = evaluate(model, tokenizer, dev, desc=f"dev e{epoch}", spec=spec)
        print_metrics(m, title=f"DEV epoch {epoch}", spec=spec)
        dev_log.append({"epoch": epoch, "weighted_f1": m["weighted_f1"],
                        "accuracy": m["accuracy"], "per_class_f1": m["per_class_f1"]})
        if m["weighted_f1"] > best_dev_f1:
            best_dev_f1 = m["weighted_f1"]
            model.save_pretrained(best_dir)
            print(f"  >>> new best dev wf1={best_dev_f1*100:.2f}, saved adapter -> {best_dir}")

    # 训练曲线落盘（per-step loss CSV + per-epoch dev JSON）
    import csv as _csv
    loss_csv = os.path.join(C.OUTPUT_DIR, f"{tag}_train_loss.csv")
    with open(loss_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["epoch", "global_batch", "loss", "lr"])
        w.writerows(loss_log)
    dev_json = os.path.join(C.OUTPUT_DIR, f"{tag}_dev_curve.json")
    with open(dev_json, "w", encoding="utf-8") as f:
        json.dump(dev_log, f, indent=2, ensure_ascii=False)
    print(f"训练曲线已存: {loss_csv} | {dev_json}")

    return best_dir, best_dev_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity_only", action="store_true", help="只跑 tiny-overfit 验证")
    ap.add_argument("--ctx", type=int, default=None, help="上下文窗口（默认 config CONTEXT_WINDOW=8）")
    ap.add_argument("--rank", type=int, default=None, help="LoRA rank（默认 config LORA_R=8，alpha=2*rank）")
    ap.add_argument("--max_len", type=int, default=None, help="最大序列长度（默认 config MAX_SEQ_LEN=1024）")
    ap.add_argument("--dataset", default="iemocap", choices=["iemocap", "meld"],
                    help="数据集（iemocap 默认；meld 第二数据集）")
    ap.add_argument("--tag", default=None, help="输出目录/产物前缀（默认 v1 / v1_meld）")
    ap.add_argument("--grad_ckpt", action="store_true", help="梯度检查点（长 ctx/大 rank 治显存）")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 C.SEED；跨骨干对照需与 adapter 侧同种子（123）")
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    D = spec.get_data_module()
    tag = args.tag or ("v1" if args.dataset == "iemocap" else f"v1_{args.dataset}")

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    run_seed = args.seed if args.seed is not None else C.SEED
    seed_all(run_seed)

    print(f"Loading {os.path.basename(os.path.normpath(C.MODEL_PATH))} (4-bit QLoRA)... "
          f"[dataset={args.dataset} tag={tag} ctx={args.ctx} seed={run_seed} "
          f"rank={args.rank} max_len={args.max_len} | {spec.num_classes} classes]")
    model, tok, _ = load_model_and_tokenizer(for_training=True)
    model = setup_lora(model, rank=args.rank)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
        print("[mem] gradient checkpointing enabled")

    train = D.load_split("train", context_window=args.ctx)
    tiny_overfit_sanity(model, tok, train)

    if args.sanity_only:
        print("sanity_only: stop here.")
        return

    best_dir, best_dev_f1 = train_full(model, tok, D, spec, ctx=args.ctx, max_len=args.max_len, tag=tag)
    print(f"\nBest dev wf1 = {best_dev_f1*100:.2f}; best adapter at {best_dir}")

    # 用最佳 adapter 在 test 上评估
    from peft import PeftModel
    print("\nReloading best adapter for test eval...")
    base, tok2, _ = load_model_and_tokenizer(for_training=True)
    model_best = PeftModel.from_pretrained(base, best_dir)
    model_best.eval()
    test = D.load_split("test", context_window=args.ctx)
    m = evaluate(model_best, tok2, test, desc=f"{tag} test", spec=spec)
    print_metrics(m, title=f"{tag} LoRA (test)", spec=spec)
    with open(os.path.join(C.OUTPUT_DIR, f"{tag}_lora_test.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"saved -> {os.path.join(C.OUTPUT_DIR, f'{tag}_lora_test.json')}")


if __name__ == "__main__":
    main()
