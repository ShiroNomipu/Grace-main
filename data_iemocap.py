"""
IEMOCAP 数据模块（V0/V1/V2/V3 共用）。

数据源（L10 修订）：InstructERC 公开 IEMOCAP（COSMIC 预处理），与 LLM-ERC SOTA 同款协议。
  - 6 类全量（含 script），带 speaker（'M'/'F'），Ses5 做 test。
  - split 由 pkl 自带：train(108对话/5163句) / valid→dev(12/647) / test(31/1623=Ses5)。
  - 对话顺序即 pkl 内顺序（COSMIC 已按时序组织），无需再按时间排序。

职责：读 pkl → 按对话分组 → 为每个 utterance 构造带上下文的对话样本 →
      输出 chat messages（system/user）+ label。
不做 tokenize（保持与具体模型解耦）；tokenize 在训练脚本里用模型 chat template 完成。

每个 sample 是 dict：
  {
    "dialogue_id", "utt_id", "speaker", "text",
    "label": str(如 'fru'), "label_id": int, "label_word": str(如 'frustrated'),
    "messages": [{"role":"system",...}, {"role":"user",...}],   # 不含 assistant 答案
    "context_len": int,                                          # 上下文句数
  }
"""
import csv
import os
import pickle
from pathlib import Path

import config as C


# ───────────────────────── 通用小工具（保留供调试脚本复用）─────────────────────────
def fix_path(raw: str) -> str:
    """规范化混用斜杠的路径为当前系统路径。"""
    return str(Path(raw.replace("/", os.sep).replace("\\", os.sep)))


def open_csv(path):
    """编码容错读取 CSV，返回 list[dict]。"""
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode CSV: {path}")


def norm_emotion(raw: str) -> str:
    """去引号去空格小写。"""
    return raw.strip().strip("'\"").lower()


# ───────────────────────── prompt 构造 ─────────────────────────
SYSTEM_PROMPT = (
    "You are an expert in emotion recognition in conversations. "
    "Read the conversation and identify the emotion of the LAST utterance. "
    "Respond with exactly one word chosen from: "
    + ", ".join(C.LABEL2WORD[lab] for lab in C.LABEL_LIST) + "."
)


def build_messages(context_turns, target_speaker, target_text):
    """构造 chat messages（不含答案）。context_turns: list[(speaker, text)]，含目标句在最后。"""
    lines = [f"Speaker_{spk}: {txt}" for spk, txt in context_turns]
    convo = "\n".join(lines)
    user = (
        f"Conversation:\n{convo}\n\n"
        f"Question: What is the emotion of the last utterance spoken by Speaker_{target_speaker}? "
        f"Answer with one word."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ───────────────────────── V1R：检索增强 + 两个辅助任务的 prompt 构造 ─────────────────────────
# speaker 在 IEMOCAP 仅 M/F（性别角色）→ speaker identification 辅助任务退化为角色预测，
# 仍服务 InstructERC 原意（预热对话角色建模）。
SPEAKER2WORD = {"M": "Male", "F": "Female"}
SPEAKER_WORDS = ["Male", "Female"]

SPK_SYSTEM = (
    "You are an expert in conversation analysis. Identify which speaker role "
    "spoke the given utterance. Respond with exactly one word from: Male, Female."
)


def build_messages_demo(context_turns, target_speaker, demo_str):
    """主任务 + 检索 demonstration（InstructERC §2.2）。demo_str 为空时退化为普通 V1 prompt。"""
    lines = [f"Speaker_{spk}: {txt}" for spk, txt in context_turns]
    convo = "\n".join(lines)
    demo_block = f"\n\nFor reference, {demo_str}" if demo_str else ""
    user = (
        f"Conversation:\n{convo}\n\n"
        f"Question: What is the emotion of the last utterance spoken by Speaker_{target_speaker}? "
        f"Answer with one word.{demo_block}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_speaker_messages(target_text):
    """辅助任务①：给一句话预测说话人角色（M/F）。不含历史（InstructERC：无视上下文）。"""
    user = (
        f'Utterance: "{target_text}"\n\n'
        f"Question: Which speaker role said this utterance? Answer with one word."
    )
    return [
        {"role": "system", "content": SPK_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_emotion_impact_messages(hist_turns, target_speaker):
    """辅助任务②：仅给历史（不含当前句），预测当前句情感倾向（建模未来情感）。
    hist_turns: list[(speaker, text)]，不含当前句；为空表示无历史。"""
    if hist_turns:
        convo = "\n".join(f"Speaker_{spk}: {txt}" for spk, txt in hist_turns)
    else:
        convo = "(no prior context)"
    user = (
        f"Conversation history:\n{convo}\n\n"
        f"Question: Based only on the history above, predict the emotion of the NEXT "
        f"utterance spoken by Speaker_{target_speaker}. Answer with one word."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ───────────────────────── InstructERC pkl 读取 ─────────────────────────
_PKL_CACHE = None


def _load_pkl():
    """加载并缓存 InstructERC iemocap.pkl。
    结构：list[6] = [speakers, labels, sentences, train_ids, test_ids, valid_ids]，
    前三者为 {dialogue_id: list}（按对话内时序对齐），后三者为对话 id 列表。"""
    global _PKL_CACHE
    if _PKL_CACHE is None:
        with open(C.INSTRUCTERC_PKL, "rb") as f:
            _PKL_CACHE = pickle.load(f)
        assert isinstance(_PKL_CACHE, (list, tuple)) and len(_PKL_CACHE) == 6, \
            f"unexpected pkl structure: {type(_PKL_CACHE)}"
    return _PKL_CACHE


# 本项目 split 名 → pkl 索引（dev = pkl 的 valid）
_SPLIT_ID_INDEX = {"train": 3, "test": 4, "dev": 5}


def load_split(split: str, context_window: int = None):
    """加载一个 split，返回 list[sample]。split ∈ {train, dev, test}。"""
    if context_window is None:
        context_window = C.CONTEXT_WINDOW
    spk_d, lab_d, txt_d, train_ids, test_ids, valid_ids = _load_pkl()
    split_ids = {"train": train_ids, "test": test_ids, "dev": valid_ids}[split]

    samples = []
    for did in split_ids:
        speakers, labels, texts = spk_d[did], lab_d[did], txt_d[did]
        assert len(speakers) == len(labels) == len(texts), f"{did}: misaligned lists"
        for t in range(len(texts)):
            emo = C.COSMIC_ID2EMO[labels[t]]
            if emo not in C.LABEL2ID:          # 安全：6 类外跳过（COSMIC 即 6 类，理论不触发）
                continue
            ctx_start = max(0, t - context_window + 1)
            context_turns = [(speakers[j], texts[j].strip()) for j in range(ctx_start, t + 1)]
            context_feat_keys = [f"{did}|{j}" for j in range(ctx_start, t + 1)]
            samples.append({
                "dialogue_id": did,
                "utt_id": str(t),              # 对话内序号（pkl 无显式 utt 文件名；V2 音频对齐时再映射）
                "speaker": speakers[t],
                "text": texts[t].strip(),
                "label": emo,
                "label_id": C.LABEL2ID[emo],
                "label_word": C.LABEL2WORD[emo],
                "messages": build_messages(context_turns, speakers[t], texts[t].strip()),
                "context_turns": context_turns,   # 含当前句（V1R 辅助任务/demo 注入用）
                "context_feat_keys": context_feat_keys,  # 多模态特征对齐 key（与 MELD context_clip_ids 对称）
                "context_len": len(context_turns),
            })
    return samples


# ───────────────────────── 自检 ─────────────────────────
def _self_test():
    from collections import Counter

    # 标签映射防呆：用 test(=Ses5) 标签分布对照已知 Ses5 全量 6 类分布锚点
    SES5_ANCHOR = {"ang": 170, "exc": 299, "fru": 381, "hap": 144, "neu": 384, "sad": 245}

    print("SYSTEM PROMPT:\n  " + SYSTEM_PROMPT + "\n")
    print(f"COSMIC_ID2EMO = {C.COSMIC_ID2EMO}\n")

    all_dialogue_ids = {}
    for split in ["train", "dev", "test"]:
        samples = load_split(split)
        n = len(samples)
        labels = Counter(s["label"] for s in samples)
        ids = set(s["label_id"] for s in samples)
        ctx_lens = [s["context_len"] for s in samples]

        # 断言 1：数量匹配 InstructERC 论文锚点
        exp = C.EXPECTED_COUNTS[split]
        assert n == exp, f"{split}: count {n} != expected {exp}"
        # 断言 2：label_id 全在 0..5
        assert ids.issubset(set(range(C.NUM_CLASSES))), f"{split}: bad label ids {ids}"
        # 断言 3：train 6 类齐全
        if split == "train":
            assert set(labels.keys()) == set(C.LABEL_LIST), f"train missing classes: {labels.keys()}"
        # 断言 4：test 标签分布与 Ses5 锚点一致（验证 COSMIC_ID2EMO 映射正确）
        if split == "test":
            assert dict(labels) == SES5_ANCHOR, (
                f"test label dist {dict(sorted(labels.items()))} != Ses5 anchor {SES5_ANCHOR}; "
                f"COSMIC_ID2EMO 映射可能错了")

        all_dialogue_ids[split] = set(s["dialogue_id"] for s in samples)
        print(f"[OK] {split}: {n} samples (expect {exp}) | "
              f"label dist={dict(sorted(labels.items()))} | "
              f"ctx_len min/max/avg={min(ctx_lens)}/{max(ctx_lens)}/{sum(ctx_lens)/len(ctx_lens):.1f}")

    # 断言 5：train/dev/test 对话无交集（防泄露）
    tr, dv, te = all_dialogue_ids["train"], all_dialogue_ids["dev"], all_dialogue_ids["test"]
    assert tr.isdisjoint(te), f"train/test overlap: {tr & te}"
    assert dv.isdisjoint(te), f"dev/test overlap: {dv & te}"
    assert tr.isdisjoint(dv), f"train/dev overlap: {tr & dv}"
    print(f"[OK] train/dev/test dialogues disjoint (train={len(tr)}, dev={len(dv)}, test={len(te)})")

    # 断言 6：test 全部是 Ses5
    assert all(d[3:5] == "05" for d in te), "test 含非 Ses5 对话"
    print(f"[OK] test 全部为 Ses5（{len(te)} 对话，说话人独立）")

    # 断言 7：上下文确实含说话人交替（M/F 双人对话）
    samples = load_split("test")
    multi_spk = sum(
        1 for s in samples
        if len({line.split(":")[0] for m in s["messages"] if m["role"] == "user"
                for line in m["content"].splitlines() if line.startswith("Speaker_")}) >= 2
    )
    ratio = multi_spk / len(samples)
    assert ratio > 0.5, f"too few multi-speaker contexts ({ratio:.2f})"
    print(f"[OK] {ratio*100:.0f}% of test contexts have >=2 speakers (turn alternation OK)")

    # 打印 2 个样例
    for idx in [0, 50]:
        s = samples[idx]
        print(f"\n{'='*60}\nSAMPLE test[{idx}]  dialogue={s['dialogue_id']} utt={s['utt_id']}")
        for m in s["messages"]:
            print(f"--- {m['role']} ---\n{m['content']}")
        print(f"--- TARGET (assistant) ---\n{s['label_word']}  (label_id={s['label_id']})")

    print("\nAll data-module self-tests passed.")


if __name__ == "__main__":
    _self_test()
