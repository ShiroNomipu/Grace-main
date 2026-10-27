"""B1 follow-up on dev (2026-10-27, CPU only; written after the B1 judge R20261022-340, descriptive). On IEMOCAP test,
GRACE + E2 (74.66) is above dual online + E2 (74.17), -0.49 for ours, 1/5 seeds. Rule: a test-set pattern must be checked on
dev before it is written. Same run lists, inputs and fusion as _remote/b1_grace_judge_20261022.py (5b7ed4d8, unchanged);
the dev alpha is read from the judge output.
Reports per dataset, dev split: mean wF1 of GRACE on / off, dual online text, GRACE + E2 and dual online + E2 at the dev
alpha, per-seed dev difference (dual online + E2) - (GRACE + E2), and both alpha curves on dev and test (mean over seeds),
so the comparison can be read at every alpha, not only at the chosen one.
Usage (repo root): python <this> --out <json>"""
import argparse, json
import numpy as np
R = "outputs/runs"; ALPHAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2]; NC = {"iemocap": 6, "meld": 7}
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
B1 = json.load(open("runs/lock_20261022/b1_runs.json")); LK = json.load(open("runs/lock_20261022/lock_runs.json"))
J = json.load(open(f"{R}/R20261022-340/b1_grace.json"))["datasets"]
def rows(p): return [json.loads(l) for l in open(p)]
def r2(x): return round(float(x), 2)
def wf1(y, p, C):
    out = 0.0
    for c in range(C):
        n = (y == c).sum(); tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum()
        out += n / len(y) * (2 * tp / (2 * tp + fp + n - tp) if tp else 0.0)
    return 100 * float(out)
out = {"design": __doc__}
for ds in ("iemocap", "meld"):
    C = NC[ds]; o = {}
    al = J[ds]["alpha_dev"]
    for sp in ("dev", "test"):
        G = [{f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{d}/eval/preds_{sp}.jsonl")} for d in B1[ds]["dump"][sp]]
        T = [{f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{x}/eval/preds_{sp}.jsonl")} for x in LK[ds]["text"]["dual"]["online"][sp]]
        E = [{r["key"]: r for r in rows(f"{R}/{x}/preds_{sp}.jsonl")} for x in LK[ds]["experts"]["e2"]]
        keys = sorted(G[0]); y = np.array([G[0][k]["gold"] for k in keys])
        for s in G + T + E:
            assert set(s) == set(keys) and all(s[k]["gold"] == G[0][k]["gold"] for k in keys), (ds, sp)
        gon = [np.array([g[k]["scores"]["clean"] for k in keys]) for g in G]
        goff = [np.array([g[k]["scores"]["drop"] for k in keys]) for g in G]
        dual = [np.array([t[k]["scores"]["clean"] for k in keys]) for t in T]
        e2 = [np.array([e[k]["logp"] for k in keys]) for e in E]
        f = lambda S, L, x: [wf1(y, np.argmax(s + x * l, 1), C) for s, l in zip(S, L)]
        ge, de = f(gon, e2, al["grace+e2"]), f(dual, e2, al["dual_online+e2"])
        o[sp] = {"grace_audio_on": r2(np.mean([wf1(y, s.argmax(1), C) for s in gon])),
                 "grace_audio_off": r2(np.mean([wf1(y, s.argmax(1), C) for s in goff])),
                 "dual_online_text": r2(np.mean([wf1(y, s.argmax(1), C) for s in dual])),
                 "grace+e2": r2(np.mean(ge)), "dual_online+e2": r2(np.mean(de)),
                 "dual_online+e2 - grace+e2": {"mean": r2(np.mean(np.subtract(de, ge))), "per_seed": [r2(x) for x in np.subtract(de, ge)],
                                               "n_pos": int((np.subtract(de, ge) > 0).sum())},
                 "alpha_curve": {"alphas": ALPHAS, "grace+e2": [r2(np.mean(f(gon, e2, x))) for x in ALPHAS],
                                 "dual_online+e2": [r2(np.mean(f(dual, e2, x))) for x in ALPHAS]}}
    o["alpha_dev"] = al
    out[ds] = o
json.dump(out, open(a.out, "w"), indent=1)
for ds in ("iemocap", "meld"):
    for sp in ("dev", "test"):
        x = out[ds][sp]; print(ds, sp, {k: v for k, v in x.items() if k != "alpha_curve"})
        print("   curve", x["alpha_curve"])
