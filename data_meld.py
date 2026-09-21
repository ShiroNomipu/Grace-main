"""
MELD 数据模块（第二数据集，7类；与 data_iemocap 同接口，V1/V2/V3.5 可复用）。

数据源：MAGTKD 原始 `MELD/{train,dev,test}_sent_emo.csv`（官方文本+7类标签）+ 每句 mp4
  切片（命名 dia{Dialogue_ID}_utt{Utterance_ID}.mp4）。**特征我们自提**（音频 emotion2vec、
  视频自选编码器）→ 纯度干净无 R5（不碰 MAGTKD 已微调特征）。

与 IEMOCAP 的差异：①7 类（neutral/joy/sadness/anger/surprise/fear/disgust）②speaker 为真实
  角色名（Chandler/Joey…）非 M/F ③对话内顺序 = Utterance_ID 升序 ④无 script 切分问题。

每个 sample 是 dict（同 data_iemocap 接口）：
  { "dialogue_id"(int), "utt_id"(str), "clip_id"(=dia{D}_utt{U}, mp4/特征对齐用),
    "speaker", "text", "label"(str), "label_id"(int), "label_word"(str),
    "messages":[system,user], "context_turns", "context_len" }
"""
import csv as _csv
from collections import Counter, defaultdict

import config as C

LABEL_LIST = C.MELD_LABEL_LIST
LABEL2ID = C.MELD_LABEL2ID
LABEL2WORD = C.MELD_LABEL2WORD

SYSTEM_PROMPT = (
    "You are an expert in emotion recognition in conversations. "
    "Read the conversation and identify the emotion of the LAST utterance. "
    "Respond with exactly one word chosen from: "
    + ", ".join(LABEL2WORD[lab] for lab in LABEL_LIST) + "."
)

# Phase 2: Persona-enhanced system prompt for MELD (Friends characters)
PERSONA_INFO = {
    "Chandler": "sarcastic, uses humor as defense, insecure",
    "Joey": "simple, enthusiastic, loyal, food-loving",
    "Monica": "competitive, perfectionist, caring, control-oriented",
    "Phoebe": "eccentric, free-spirited, brutally honest",
    "Rachel": "dramatic, emotional, fashion-oriented",
    "Ross": "intellectual, neurotic, romantic, easily frustrated",
}

SYSTEM_PROMPT_PERSONA = (
    "You are an expert in emotion recognition in conversations from a TV series. "
    "The main characters have distinct personalities: "
    + "; ".join(f"{k} is {v}" for k, v in PERSONA_INFO.items())
    + ". Consider each speaker's personality when inferring emotion. "
    "Identify the emotion of the LAST utterance. "
    "Respond with exactly one word chosen from: "
    + ", ".join(LABEL2WORD[lab] for lab in LABEL_LIST) + "."
)


def _open_csv(path):
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(_csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode CSV: {path}")


def _norm_emotion(raw: str) -> str:
    return raw.strip().strip("'\"").lower()


def build_messages(context_turns, target_speaker, target_text, use_persona=False):
    """构造 chat messages（不含答案）。context_turns: list[(speaker, text)]，含目标句在最后。"""
    convo = "\n".join(f"{spk}: {txt}" for spk, txt in context_turns)
    if use_persona:
        sys_prompt = SYSTEM_PROMPT_PERSONA
        user = (
            f"Conversation:\n{convo}\n\n"
            f"Question: Considering {target_speaker}'s personality and the conversational dynamics, "
            f"what is the emotion of the last utterance? Answer with one word."
        )
    else:
        sys_prompt = SYSTEM_PROMPT
        user = (
            f"Conversation:\n{convo}\n\n"
            f"Question: What is the emotion of the last utterance spoken by {target_speaker}? "
            f"Answer with one word."
        )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]


def load_split(split: str, context_window: int = None, use_persona: bool = False):
    """加载一个 split，返回 list[sample]。split ∈ {train, dev, test}。"""
    if context_window is None:
        context_window = C.CONTEXT_WINDOW
    rows = _open_csv(C.MELD_CSV[split])

    # 按 Dialogue_ID 分组，组内按 Utterance_ID 升序（保证对话内时序）
    by_dia = defaultdict(list)
    for r in rows:
        by_dia[int(r["Dialogue_ID"])].append(r)
    for did in by_dia:
        by_dia[did].sort(key=lambda r: int(r["Utterance_ID"]))

    samples = []
    for did in sorted(by_dia.keys()):
        turns = by_dia[did]
        speakers = [r["Speaker"].strip() for r in turns]
        texts = [r["Utterance"].strip() for r in turns]
        emos = [_norm_emotion(r["Emotion"]) for r in turns]
        uids = [int(r["Utterance_ID"]) for r in turns]
        for t in range(len(turns)):
            emo = emos[t]
            assert emo in LABEL2ID, f"{split} dia{did} utt{uids[t]}: unknown emotion {emo!r}"
            ctx_start = max(0, t - context_window + 1)
            context_turns = [(speakers[j], texts[j]) for j in range(ctx_start, t + 1)]
            context_clip_ids = [f"dia{did}_utt{uids[j]}" for j in range(ctx_start, t + 1)]
            samples.append({
                "dialogue_id": did,
                "utt_id": str(uids[t]),
                "clip_id": f"dia{did}_utt{uids[t]}",
                "speaker": speakers[t],
                "text": texts[t],
                "label": emo,
                "label_id": LABEL2ID[emo],
                "label_word": LABEL2WORD[emo],
                "messages": build_messages(context_turns, speakers[t], texts[t],
                                           use_persona=use_persona),
                "context_turns": context_turns,
                "context_clip_ids": context_clip_ids,
                "context_len": len(context_turns),
            })
    return samples


def _self_test():
    print("SYSTEM PROMPT:\n  " + SYSTEM_PROMPT + "\n")
    all_clip_ids = {}
    for split in ["train", "dev", "test"]:
        samples = load_split(split)
        n = len(samples)
        labels = Counter(s["label"] for s in samples)
        ids = set(s["label_id"] for s in samples)
        ctx_lens = [s["context_len"] for s in samples]

        exp = C.MELD_EXPECTED_COUNTS[split]
        assert n == exp, f"{split}: count {n} != expected {exp}"
        assert ids.issubset(set(range(C.MELD_NUM_CLASSES))), f"{split}: bad label ids {ids}"
        if split == "train":
            assert set(labels.keys()) == set(LABEL_LIST), \
                f"train missing classes: {set(LABEL_LIST) - set(labels.keys())}"
        # clip_id 唯一性
        clips = [s["clip_id"] for s in samples]
        assert len(clips) == len(set(clips)), f"{split}: duplicate clip_id"
        all_clip_ids[split] = set(clips)
        print(f"[OK] {split}: {n} samples (expect {exp}) | "
              f"label dist={dict(sorted(labels.items(), key=lambda x: -x[1]))} | "
              f"ctx min/max/avg={min(ctx_lens)}/{max(ctx_lens)}/{sum(ctx_lens)/len(ctx_lens):.1f}")

    # 对话 id 在不同 split 间会重复（都从 0 计），但 split 各自独立无泄露（不同 csv）
    print(f"[OK] clip_id unique within each split "
          f"(train={len(all_clip_ids['train'])}, dev={len(all_clip_ids['dev'])}, "
          f"test={len(all_clip_ids['test'])})")

    samples = load_split("test")
    multi_spk = sum(1 for s in samples
                    if len({spk for spk, _ in s["context_turns"]}) >= 2)
    print(f"[OK] {multi_spk/len(samples)*100:.0f}% of test contexts have >=2 speakers")

    for idx in [0, 100]:
        s = samples[idx]
        print(f"\n{'='*60}\nSAMPLE test[{idx}]  clip={s['clip_id']} speaker={s['speaker']}")
        for m in s["messages"]:
            print(f"--- {m['role']} ---\n{m['content']}")
        print(f"--- TARGET ---\n{s['label_word']}  (label_id={s['label_id']})")

    print("\nAll MELD data-module self-tests passed.")


if __name__ == "__main__":
    _self_test()
