"""Dev checks of descriptive test-set patterns before they are written (2026-10-27, CPU only; written after the lock
results, descriptive). Rule: a test-set pattern enters the paper only after it is checked on dev. Inputs: lock dumps via
runs/lock_20261022/lock_runs.json, dev alpha from R20261022-211/lock_full.json (the lock judge output).
 1. future_by_group on dev and test: fixes / breaks of the dual model (online -> offline text) by group; same groups and
    arithmetic as lock_cpu_analyses_20261025.py section 1 (the test numbers must reproduce that file).
 2. MELD, dev and test: per-class rescue / harm of each expert against dual online text and the share of neutral
    predictions; same arithmetic as lock_meld_perclass_20261025.py (the test numbers must reproduce that file).
 3. IEMOCAP, dev and test, scripted vs improvised dialogues: wF1 of dual online and of dual online + E2 (dev alpha) on each
    subset, per seed; net fixes of the fusion (text wrong -> fused right minus text right -> fused wrong, mean over
    seeds) per subset; the number of dialogues whose seed-averaged net fixes are > 0, = 0, < 0.
Usage (repo root): python <this> --out <json>"""
import argparse, collections, json
import numpy as np
R = "outputs/runs"; NC = {"iemocap": 6, "meld": 7}
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
LK = json.load(open("runs/lock_20261022/lock_runs.json")); J2 = json.load(open(f"{R}/R20261022-211/lock_full.json"))["datasets"]
def rows(p): return [json.loads(l) for l in open(p)]
def r2(x): return round(float(x), 2)
def wf1(y, p, C):
    out = 0.0
    for c in range(C):
        n = (y == c).sum(); tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum()
        out += n / len(y) * (2 * tp / (2 * tp + fp + n - tp) if tp else 0.0)
    return 100 * float(out)


def load(ds, sp):
    lk = LK[ds]
    T = {m: [{f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{x}/eval/preds_{sp}.jsonl")} for x in lk["text"]["dual"][m][sp]]
         for m in ("online", "full")}
    keys = sorted(T["online"][0]); ref = T["online"][0]; y = np.array([ref[k]["gold"] for k in keys])
    for m in T:
        for d in T[m]:
            assert set(d) == set(keys) and all(d[k]["gold"] == ref[k]["gold"] for k in keys), (ds, sp, m)
    S = {m: [np.array([d[k]["scores"]["clean"] for k in keys]) for d in T[m]] for m in T}
    E = {}
    for ex, ids in lk["experts"].items():
        E[ex] = []
        for x in ids:
            d = {r["key"]: r for r in rows(f"{R}/{x}/preds_{sp}.jsonl")}
            assert set(d) == set(keys) and all(d[k]["gold"] == ref[k]["gold"] for k in keys), (ds, sp, ex, x)
            E[ex].append(np.array([d[k]["logp"] for k in keys]))
    return keys, ref, y, S, E


out = {"design": __doc__}
for ds in ("iemocap", "meld"):
    C = NC[ds]; al = J2[ds]["alpha_dev"]; o = {}
    for sp in ("dev", "test"):
        keys, ref, y, S, E = load(ds, sp)
        Po = [s.argmax(1) for s in S["online"]]; Pf = [s.argmax(1) for s in S["full"]]
        # 1. future_by_group (copied arithmetic)
        dl = collections.defaultdict(list)
        for i, k in enumerate(keys):
            dl[k.split("|")[0]].append((int(k.split("|")[1]), i))
        nxt_same, spk_again, nspk = np.zeros(len(keys), int) - 1, np.zeros(len(keys), int), np.zeros(len(keys), int)
        for d, lst in dl.items():
            lst.sort(); idx = [i for _, i in lst]; spk = [ref[keys[i]]["speaker"] for i in idx]; ns = len(set(spk))
            for j, i in enumerate(idx):
                nspk[i] = ns
                if j + 1 < len(idx):
                    nxt_same[i] = int(spk[j + 1] == spk[j])
                spk_again[i] = int(spk[j] in spk[j + 1:j + 3])
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
        o[sp] = {"future_by_group": fut}
        # 2. MELD per-class rescue / harm (copied arithmetic)
        if ds == "meld":
            pc = {"n_per_class": np.bincount(y, minlength=C).tolist(),
                  "text_acc": [round(float(np.mean([100 * (p[y == c] == c).mean() for p in Po])), 1) for c in range(C)]}
            for ex in ("e1", "e2", "e3"):
                Q = [l.argmax(1) for l in E[ex]]; res, harm = [], []
                for c in range(C):
                    m = y == c
                    res.append(round(float(np.mean([100 * (q[m & (p != y)] == c).mean() if (m & (p != y)).any() else 0 for p, q in zip(Po, Q)])), 1))
                    harm.append(round(float(np.mean([100 * (q[m & (p == y)] != c).mean() if (m & (p == y)).any() else 0 for p, q in zip(Po, Q)])), 1))
                pc[ex] = {"rescue": res, "harm": harm, "pred_share_neutral": round(float(np.mean([100 * (q == 0).mean() for q in Q])), 1)}
            o[sp]["perclass"] = pc
        # 3. IEMOCAP scripted vs improvised
        if ds == "iemocap":
            Pfo = [(s + al["dual_online+e2"] * e).argmax(1) for s, e in zip(S["online"], E["e2"])]
            dlg = np.array([k.split("|")[0] for k in keys]); scr = np.array(["script" in d for d in dlg])
            sub = {}
            for name, m in (("scripted", scr), ("improvised", ~scr)):
                t = [wf1(y[m], p[m], C) for p in Po]; f = [wf1(y[m], p[m], C) for p in Pfo]
                net = np.mean([((p != y) & (q == y) & m).sum() - ((p == y) & (q != y) & m).sum() for p, q in zip(Po, Pfo)])
                sub[name] = {"n_utts": int(m.sum()), "n_dialogues": int(len(set(dlg[m]))), "text_wf1": r2(np.mean(t)), "fused_wf1": r2(np.mean(f)),
                             "gain_per_seed": [r2(b - x) for x, b in zip(t, f)], "gain_mean": r2(np.mean(f) - np.mean(t)), "net_fixes": r2(net)}
            per_d = {}
            for d in sorted(set(dlg)):
                m = dlg == d
                per_d[d] = float(np.mean([((p != y) & (q == y) & m).sum() - ((p == y) & (q != y) & m).sum() for p, q in zip(Po, Pfo)]))
            for name, isscr in (("scripted", True), ("improvised", False)):
                v = [x for d, x in per_d.items() if ("script" in d) == isscr]
                sub[name]["dialogues_net_pos_zero_neg"] = [int(sum(x > 0 for x in v)), int(sum(x == 0 for x in v)), int(sum(x < 0 for x in v))]
            o[sp]["script_vs_impro"] = sub
    out[ds] = o
json.dump(out, open(a.out, "w"), indent=1)
for ds in ("iemocap", "meld"):
    for sp in ("dev", "test"):
        print(ds, sp, json.dumps({k: v for k, v in out[ds][sp].items()})[:1500])
