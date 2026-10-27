"""Completeness analyses on the lock batch, CPU only (2026-10-25; B5 / B3 items of the lock card §9; descriptive,
written after the lock results). Inputs: lock dumps via runs/lock_20261022/lock_runs.json, dev alpha from J2
(R20261022-211/lock_full.json). Test split, 5 seeds unless noted.
 1. Where the future helps: future fixes / breaks of the dual model (online -> offline text) grouped by number of future
    turns available, whether the next turn is by the same speaker, whether the target speaker speaks again within the 2
    future turns, and (MELD) the number of speakers in the dialogue.
 2. The price of waiting in seconds: time from the end of the target utterance to the end of the last future turn used
    (k = 2 labelled turns), IEMOCAP from the alignment file, MELD from the official CSV timestamps; overall and for the
    20% utterances the online + E2 rule would wait on.
 3. Confusion matrices (summed over seeds) of dual online, dual offline, dual online + E2.
 4. Case candidates: IEMOCAP repeated scripted lines (gold differs from the training majority) where the text says the
    training label and online + E2 is right in >= 4/5 seeds; MELD utterances where E3 turns a wrong text emotion into a
    right neutral, and where it turns a right text emotion into a wrong neutral (>= 4/5 seeds).
Usage (repo root): python <this> --out <json>"""
import argparse, collections, csv, glob, json, re
import numpy as np
R = "outputs/runs"; NC = {"iemocap": 6, "meld": 7}
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
LK = json.load(open("runs/lock_20261022/lock_runs.json")); J2 = json.load(open(f"{R}/R20261022-211/lock_full.json"))["datasets"]
def rows(p): return [json.loads(l) for l in open(p)]
def sm(s): z = np.exp(s - s.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)
def r2(x): return round(float(x), 2)
out = {"design": __doc__}
for ds in ("iemocap", "meld"):
    C = NC[ds]; lk = LK[ds]; al = J2[ds]["alpha_dev"]
    T = {m: [{f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{x}/eval/preds_test.jsonl")} for x in lk["text"]["dual"][m]["test"]] for m in ("online", "full")}
    keys = sorted(T["online"][0]); ref = T["online"][0]; y = np.array([ref[k]["gold"] for k in keys])
    So = [np.array([d[k]["scores"]["clean"] for k in keys]) for d in T["online"]]; Sf = [np.array([d[k]["scores"]["clean"] for k in keys]) for d in T["full"]]
    E2 = [np.array([d[k]["logp"] for k in keys]) for d in ({r["key"]: r for r in rows(f"{R}/{x}/preds_test.jsonl")} for x in lk["experts"]["e2"])]
    E3 = [np.array([d[k]["logp"] for k in keys]) for d in ({r["key"]: r for r in rows(f"{R}/{x}/preds_test.jsonl")} for x in lk["experts"]["e3"])]
    Po = [s.argmax(1) for s in So]; Pf = [s.argmax(1) for s in Sf]
    Fo = [s + al["dual_online+e2"] * e for s, e in zip(So, E2)]; Pfo = [f.argmax(1) for f in Fo]
    # dialogue structure
    dl = collections.defaultdict(list)
    for i, k in enumerate(keys):
        dl[k.split("|")[0]].append((int(k.split("|")[1]), i))
    nxt_same, spk_again, nspk = np.zeros(len(keys), int) - 1, np.zeros(len(keys), int), np.zeros(len(keys), int)
    order = {}
    for d, lst in dl.items():
        lst.sort(); idx = [i for _, i in lst]; sp = [ref[keys[i]]["speaker"] for i in idx]; ns = len(set(sp))
        for j, i in enumerate(idx):
            order[i] = (d, j, idx); nspk[i] = ns
            if j + 1 < len(idx):
                nxt_same[i] = int(sp[j + 1] == sp[j])
            spk_again[i] = int(sp[j] in sp[j + 1:j + 3])
    nf = np.array([ref[k]["n_future"] for k in keys])
    groups = {"n_future=0": nf == 0, "n_future=1": nf == 1, "n_future=2": nf == 2,
              "next_turn_same_speaker": nxt_same == 1, "next_turn_other_speaker": nxt_same == 0,
              "target_speaker_speaks_again_in_2": spk_again == 1, "target_speaker_silent_in_2": (spk_again == 0) & (nf > 0)}
    if ds == "meld":
        groups.update({"dialogue_2_speakers": nspk <= 2, "dialogue_3plus_speakers": nspk >= 3})
    fut = {}
    for g, m in groups.items():
        fx = np.mean([((p != y) & (q == y) & m).sum() for p, q in zip(Po, Pf)]); br = np.mean([((p == y) & (q != y) & m).sum() for p, q in zip(Po, Pf)])
        ch = np.mean([((p != q) & m).sum() for p, q in zip(Po, Pf)])
        fut[g] = {"share_pct": r2(100 * m.mean()), "n": int(m.sum()), "changed_per100": r2(100 * ch / max(m.sum(), 1)),
                  "fixes": r2(fx), "breaks": r2(br), "net_per100": r2(100 * (fx - br) / max(m.sum(), 1))}
    o = {"future_by_group": fut}
    # wait seconds
    if ds == "iemocap":
        alg = json.load(open("runs/iemocap_align_20261018.json"))["utts"]; end = np.array([alg[k]["end"] for k in keys])
    else:
        tt = {}
        for r in csv.DictReader(open("datasets/meld_csv/test_sent_emo.csv", encoding="utf-8")):
            h, mi, rest = r["EndTime"].split(":"); s_, ms = rest.split(","); tt[f"{int(r['Dialogue_ID'])}|{int(r['Utterance_ID'])}"] = int(h) * 3600 + int(mi) * 60 + int(s_) + int(ms) / 1000
        end = np.array([tt.get(k, np.nan) for k in keys])
    wait = np.zeros(len(keys))
    for i in range(len(keys)):
        d, j, idx = order[i]; last = idx[min(j + 2, len(idx) - 1)]; wait[i] = end[last] - end[i]
    bad = int(np.sum(~np.isfinite(wait) | (wait < 0))); w = wait[np.isfinite(wait) & (wait >= 0)]
    sel = []
    for f in Fo:
        q = np.zeros(len(keys), bool); q[np.argsort(sm(f).max(1), kind="stable")[:int(round(0.2 * len(keys)))]] = True; sel.append(q)
    ws = np.concatenate([wait[q & np.isfinite(wait) & (wait >= 0)] for q in sel])
    o["wait_seconds"] = {"all": {"median": r2(np.median(w)), "p90": r2(np.percentile(w, 90)), "mean": r2(w.mean())},
                         "waited_20pct_online+e2": {"median": r2(np.median(ws)), "p90": r2(np.percentile(ws, 90))}, "invalid_timestamps": bad}
    # confusion matrices (rows = gold, columns = predicted, summed over seeds)
    def cm(P): M = np.zeros((C, C), int); [np.add.at(M, (y, p), 1) for p in P]; return M.tolist()
    o["confusion_sum5"] = {"dual_online": cm(Po), "dual_full": cm(Pf), "dual_online+e2": cm(Pfo)}
    # cases
    cases = {}
    if ds == "iemocap":
        RAW = "/root/iemocap_raw/x/iemocap_raw"; K = ["ang", "exc", "fru", "hap", "neu", "sad"]; txt = {}
        for f in glob.glob(f"{RAW}/Session*/dialog/transcriptions/*.txt"):
            for line in open(f, errors="ignore"):
                mm = re.match(r"^(\S+) \[[\d.]+-[\d.]+\]: (.*)$", line.strip())
                if mm:
                    txt[mm.group(1)] = mm.group(2)
        norm = lambda s: " ".join(re.sub(r"[^a-z' ]", " ", s.lower()).split()); sid = lambda k: re.sub(r"^Ses0\d[FM]_", "", k.split("|")[0])
        tl = collections.defaultdict(list)
        for k, u in alg.items():
            if u["split"] == "train" and "script" in k and u["name"] in txt and len(norm(txt[u["name"]]).split()) >= 3:
                tl[(sid(k), norm(txt[u["name"]]))].append(u["label"])
        lst = []
        for i, k in enumerate(keys):
            labs = tl.get((sid(k), norm(txt.get(alg[k]["name"], "")))) if "script" in k else None
            if not labs:
                continue
            maj = K.index(collections.Counter(labs).most_common(1)[0][0])
            if maj == y[i]:
                continue
            copy = sum(int(p[i] == maj) for p in Po); fixed = sum(int(p[i] == y[i]) for p in Pfo)
            if copy >= 4 and fixed >= 4:
                lst.append({"key": k, "text": txt[alg[k]["name"]], "gold": K[y[i]], "train_majority": K[maj], "train_labels": dict(collections.Counter(labs)),
                            "text_says_train_label_seeds": copy, "online+e2_right_seeds": fixed})
        cases["repeated_line_fixed_by_voice"] = lst
    else:
        U = {f"{int(r['Dialogue_ID'])}|{int(r['Utterance_ID'])}": (r["Speaker"], r["Utterance"]) for r in csv.DictReader(open("datasets/meld_csv/test_sent_emo.csv", encoding="utf-8"))}
        L = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
        Q = [e.argmax(1) for e in E3]; resc, harm = [], []
        for i, k in enumerate(keys):
            if y[i] == 0 and sum(int(p[i] != 0 and q[i] == 0) for p, q in zip(Po, Q)) >= 4:
                resc.append({"key": k, "speaker": U[k][0], "text": U[k][1], "gold": L[y[i]], "text_pred_mode": L[collections.Counter(int(p[i]) for p in Po).most_common(1)[0][0]]})
            if y[i] != 0 and sum(int(p[i] == y[i] and q[i] == 0) for p, q in zip(Po, Q)) >= 4:
                harm.append({"key": k, "speaker": U[k][0], "text": U[k][1], "gold": L[y[i]]})
        cases["e3_neutral_rescue"] = {"n": len(resc), "first": resc[:12]}; cases["e3_neutral_harm"] = {"n": len(harm), "first": harm[:12]}
    o["cases"] = cases
    out[ds] = o
json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
for ds in ("iemocap", "meld"):
    print("=====", ds); print(json.dumps(out[ds]["future_by_group"], indent=0)); print(json.dumps(out[ds]["wait_seconds"]))
    print({k: (v["n"] if isinstance(v, dict) else len(v)) for k, v in out[ds]["cases"].items()})
