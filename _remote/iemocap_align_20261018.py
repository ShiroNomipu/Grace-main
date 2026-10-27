"""2026-10-18: align our IEMOCAP utterances (InstructERC/COSMIC pkl: dialogue_id + position) to IEMOCAP
turn names, and attach the per-evaluator annotations.

Matching is one-to-one inside each dialogue on (speaker letter, transcript text, label, relative position);
the pkl order differs from start-time order where turns overlap. Every pair is checked on text similarity
and label; unmatched utterances are reported.
Read-only on the IEMOCAP tree; output runs/iemocap_align_20261018.json.
"""
import difflib
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())
import config as C  # noqa: E402
import data_iemocap as D  # noqa: E402

ROOT = "/root/iemocap_raw/x/iemocap_raw"
SIX = {"hap", "sad", "neu", "ang", "exc", "fru"}
FULL = {"Neutral": "neu", "Neutral state": "neu", "Frustration": "fru", "Anger": "ang", "Sadness": "sad",
        "Happiness": "hap", "Excited": "exc", "Surprise": "sur", "Fear": "fea", "Disgust": "dis", "Other": "oth"}
NAME = re.compile(r"^(Ses\d\d[FM]_\w+?)_([FM])(\d{3})$")


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).split()


def parse_emo(path):
    turns, cur = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        m = re.match(r"^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\S+)\t(\w+)\t\[([\d.]+), ([\d.]+), ([\d.]+)\]", line)
        if m:
            cur = m.group(3)
            turns[cur] = {"start": float(m.group(1)), "end": float(m.group(2)), "label": m.group(4),
                          "vad_mean": [float(m.group(i)) for i in (5, 6, 7)], "cat": {}, "dim": {}}
            continue
        if cur is None:
            continue
        m = re.match(r"^C-(\w+):\t(.*?)\t", line)
        if m:
            labs = [FULL.get(x.strip().split(" (")[0], "oth:" + x.strip()) for x in m.group(2).split(";") if x.strip()]
            turns[cur]["cat"][m.group(1)] = labs
            continue
        m = re.match(r"^A-(\w+):\tval (\d+); act (\d+); dom\s+(\d+);", line)
        if m:
            turns[cur]["dim"][m.group(1)] = [int(m.group(i)) for i in (2, 3, 4)]
    return turns


def parse_trans(path):
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^(\S+) \[[\d.]+-[\d.]+\]: (.*)$", line.strip())
        if m and NAME.match(m.group(1)):
            out[m.group(1)] = m.group(2)
    return out


spk_d, lab_d, txt_d, tr_ids, te_ids, dv_ids = D._load_pkl()
split_of = {**{d: "train" for d in tr_ids}, **{d: "dev" for d in dv_ids}, **{d: "test" for d in te_ids}}
result, report = {}, {"dialogues": 0, "rule_ok": 0, "fallback": [], "utts": 0, "text_sim_lt_0.6": 0}
for did in tr_ids + dv_ids + te_ids:
    ses = int(did[3:5])
    emo = parse_emo(f"{ROOT}/Session{ses}/dialog/EmoEvaluation/{did}.txt")
    trans = parse_trans(f"{ROOT}/Session{ses}/dialog/transcriptions/{did}.txt")
    # one-to-one matching within the dialogue: same speaker letter, text similarity, label agreement and
    # relative position (the pkl order differs from start-time order where turns overlap)
    names_all = sorted([n for n in set(emo) | set(trans) if NAME.match(n)], key=lambda n: emo.get(n, {}).get("start", 1e9))
    labs = [C.COSMIC_ID2EMO[x] for x in lab_d[did]]
    spks, txts = spk_d[did], [t.strip() for t in txt_d[did]]
    report["dialogues"] += 1
    cands = []
    for i in range(len(labs)):
        a = norm(txts[i])
        for k, n in enumerate(names_all):
            if NAME.match(n).group(2) != spks[i]:
                continue
            sim = difflib.SequenceMatcher(None, norm(trans.get(n, "")), a).ratio()
            if sim < 0.5:
                continue
            lm = emo.get(n, {}).get("label") == labs[i]
            pos = abs(i / max(len(labs) - 1, 1) - k / max(len(names_all) - 1, 1))
            cands.append((sim + 0.05 * lm - 0.2 * pos, i, n))
    cands.sort(reverse=True)
    got, used = {}, set()
    for sc, i, n in cands:
        if i in got or n in used:
            continue
        got[i] = n
        used.add(n)
    pairs = [(i, got.get(i)) for i in range(len(labs))]
    miss = sum(n is None for _, n in pairs)
    if miss == 0:
        report["rule_ok"] += 1
    else:
        report["fallback"].append({"dialogue": did, "n_pkl": len(labs), "unmatched": miss})
    for i, n in pairs:
        report["utts"] += 1
        rec = {"split": split_of[did], "speaker": spks[i], "label": labs[i], "name": n}
        if n is not None:
            sim = difflib.SequenceMatcher(None, norm(trans.get(n, "")), norm(txts[i])).ratio()
            report["text_sim_lt_0.6"] += sim < 0.6
            rec.update({"wav": f"Session{ses}/sentences/wav/{did}/{n}.wav", "text_sim": round(sim, 3),
                        "label_match": emo[n]["label"] == labs[i], **{k: emo[n][k] for k in ("start", "end", "vad_mean", "cat", "dim")}})
            rec["wav_exists"] = os.path.exists(f"{ROOT}/{rec['wav']}")
        result[f"{did}|{i}"] = rec
report["missing_wav"] = sum(1 for r in result.values() if r.get("name") and not r["wav_exists"])
report["label_mismatch"] = sum(1 for r in result.values() if r.get("name") and not r["label_match"])
json.dump({"report": report, "utts": result}, open("runs/iemocap_align_20261018.json", "w"), ensure_ascii=False)
print(json.dumps(report, ensure_ascii=False))
