"""
推理 + 评估模块（V0 zero-shot / V1 LoRA / 后续 CP 共用）。

核心方法：候选打分（candidate scoring / cloze）。
  对 6 个情感词，分别计算 LLM 在给定 prompt 下生成该词的平均 token 对数似然，
  取最高者为预测。优点：无生成解析失败；天然产出每类分数（softmax 后即类别概率，CP 可直接用）。

评估指标：weighted-F1（主）、accuracy、per-class F1、混淆矩阵。
"""
import numpy as np
import torch
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             confusion_matrix, f1_score)

import config as C
from dataset_spec import default_spec


def _resolve_spec(spec):
    """spec=None 时回退到 IEMOCAP 全局口径（向后兼容既有调用）。"""
    return default_spec() if spec is None else spec


def load_model_and_tokenizer(for_training=False):
    """加载骨干。推理用 bf16；训练用 4-bit（省显存给优化器）。

    默认 Qwen2-VL-2B（走 Qwen2VLForConditionalGeneration + AutoProcessor）。
    跨骨干对照把 C.MODEL_PATH 指向纯文本 LLM（如 Llama-3.2-3B-Instruct）时，
    改走 AutoModelForCausalLM，processor 为 None——离线模态特征从 hook 进入，
    不经过 processor，所以这条路径不需要它。
    """
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    cfg = AutoConfig.from_pretrained(C.MODEL_PATH)
    is_qwen_vl = getattr(cfg, "model_type", "") in ("qwen2_vl", "qwen2_5_vl")

    if is_qwen_vl:
        from transformers import Qwen2VLForConditionalGeneration as ModelCls
        processor = AutoProcessor.from_pretrained(C.MODEL_PATH)
        tokenizer = processor.tokenizer
    else:
        from transformers import AutoModelForCausalLM as ModelCls
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(C.MODEL_PATH)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(torch_dtype=torch.bfloat16, device_map={"": 0})
    if for_training and C.USE_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = ModelCls.from_pretrained(C.MODEL_PATH, **kwargs)
    return model, tokenizer, processor


def _build_candidate_token_ids(tokenizer, messages, spec=None):
    """返回 (prompt_ids, list[(label, answer_token_ids)])。
    answer_token_ids = 该情感词在 chat 模板里紧跟 assistant 前缀的真实 token 序列（已剥尾部特殊token）。"""
    spec = _resolve_spec(spec)
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True)
    special = set(tokenizer.all_special_ids)

    cands = []
    for lab in spec.label_list:
        word = spec.label2word[lab]
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": word}],
            add_generation_prompt=False, tokenize=True)
        ans = full[len(prompt_ids):]
        # 剥掉尾部特殊 token（<|im_end|> 及换行等），只保留情感词本身
        while ans and ans[-1] in special:
            ans = ans[:-1]
        assert len(ans) > 0, f"empty answer tokens for word={word}"
        cands.append((lab, ans))
    return prompt_ids, cands


@torch.no_grad()
def score_candidates(model, tokenizer, messages, return_scores=False, spec=None):
    """对一个样本打分，返回预测 label_id（return_scores 时附带各类归一化概率）。"""
    spec = _resolve_spec(spec)
    prompt_ids, cands = _build_candidate_token_ids(tokenizer, messages, spec=spec)
    device = model.device

    avg_logps = []
    for lab, ans in cands:
        seq = prompt_ids + ans
        input_ids = torch.tensor([seq], device=device)
        logits = model(input_ids=input_ids).logits[0]      # (L, V)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        # ans 的第 k 个 token 位于 seq 索引 len(prompt_ids)+k，由前一位置 logits 预测
        total = 0.0
        for k, tok_id in enumerate(ans):
            pos = len(prompt_ids) + k - 1
            total += logprobs[pos, tok_id].item()
        avg_logps.append(total / len(ans))                 # 长度归一，避免偏向短词

    avg_logps = np.array(avg_logps)
    pred_idx = int(avg_logps.argmax())
    pred_label_id = spec.label2id[spec.label_list[pred_idx]]

    if return_scores:
        probs = np.exp(avg_logps - avg_logps.max())
        probs = probs / probs.sum()
        # 按 label_id 顺序排列概率
        score_by_id = np.zeros(spec.num_classes)
        for i, lab in enumerate(spec.label_list):
            score_by_id[spec.label2id[lab]] = probs[i]
        return pred_label_id, score_by_id
    return pred_label_id


def compute_metrics(preds, golds, spec=None):
    """由预测/金标签计算指标 dict（V0/V1/V2 共用，保证口径一致）。"""
    spec = _resolve_spec(spec)
    preds, golds = np.array(preds), np.array(golds)
    wf1 = f1_score(golds, preds, average="weighted", zero_division=0)
    acc = accuracy_score(golds, preds)
    nc = spec.num_classes
    per_class = precision_recall_fscore_support(
        golds, preds, labels=list(range(nc)), zero_division=0)
    cm = confusion_matrix(golds, preds, labels=list(range(nc)))
    return {
        "weighted_f1": float(wf1),
        "accuracy": float(acc),
        "per_class_f1": {spec.id2label[i]: float(per_class[2][i]) for i in range(nc)},
        "support": {spec.id2label[i]: int(per_class[3][i]) for i in range(nc)},
        "confusion_matrix": cm.tolist(),
        "n": len(preds),
    }


def evaluate(model, tokenizer, samples, desc="eval", limit=None, verbose=False, spec=None):
    """在样本集上评估，返回 metrics dict。"""
    from tqdm import tqdm
    spec = _resolve_spec(spec)
    if limit:
        samples = samples[:limit]

    preds, golds = [], []
    for s in tqdm(samples, desc=desc):
        pred = score_candidates(model, tokenizer, s["messages"], spec=spec)
        preds.append(pred)
        golds.append(s["label_id"])
        if verbose:
            print(f"  gold={spec.id2label[s['label_id']]:4} pred={spec.id2label[pred]:4} "
                  f"| {s['dialogue_id']}_{s['utt_id']}")

    return compute_metrics(preds, golds, spec=spec)


def print_metrics(m, title="METRICS", spec=None):
    spec = _resolve_spec(spec)
    print(f"\n===== {title} =====")
    print(f"  weighted-F1 : {m['weighted_f1']*100:.2f}")
    print(f"  accuracy    : {m['accuracy']*100:.2f}")
    print(f"  n           : {m['n']}")
    print(f"  per-class F1: " + "  ".join(
        f"{lab}={m['per_class_f1'][lab]*100:.1f}(n={m['support'][lab]})" for lab in spec.label_list))
    print(f"  confusion matrix (rows=gold {spec.label_list}):")
    for i, row in enumerate(m["confusion_matrix"]):
        print(f"    {spec.label_list[i]:4} {row}")


# ───────────────── sanity: 3 样本打分诊断 ─────────────────
if __name__ == "__main__":
    import data_iemocap as D
    print("Loading model (bf16 inference)...")
    model, tok, _ = load_model_and_tokenizer(for_training=False)
    model.eval()
    print("Model loaded. Running 3-sample scoring diagnostic...\n")

    samples = D.load_split("test")
    for idx in [0, 50, 100]:
        s = samples[idx]
        pred_id, scores = score_candidates(model, tok, s["messages"], return_scores=True)
        print(f"[test {idx}] {s['dialogue_id']}_{s['utt_id']}")
        print(f"  gold = {C.ID2LABEL[s['label_id']]}")
        print(f"  pred = {C.ID2LABEL[pred_id]}")
        print(f"  scores: " + "  ".join(
            f"{lab}={scores[C.LABEL2ID[lab]]:.3f}" for lab in C.LABEL_LIST))
        print()
    print("Sanity diagnostic done.")
