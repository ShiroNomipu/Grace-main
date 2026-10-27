"""Ensemble row (2026-10-25; planned since 10-22 as a separately labelled row, never compared with single models).
IEMOCAP lock dumps: mean of the 5 seeds' log-softmax text scores (dual online / dual full) + alpha * mean of the 5 E2
log-probs; alpha chosen on dev (grid, ties -> smaller); waiting 20% by the online ensemble's confidence. Test reported.
Usage (repo root): python <this> --out <json>"""
import argparse, json
import numpy as np
R = "outputs/runs"; C = 6; A = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2]
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
LK = json.load(open("runs/lock_20261022/lock_runs.json"))["iemocap"]
def rows(p): return [json.loads(l) for l in open(p)]
def ls(s): s = s - s.max(1, keepdims=True); return s - np.log(np.exp(s).sum(1, keepdims=True))
def wf1(y, p):
    s = 0.0
    for c in range(C):
        n = (y == c).sum(); tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum()
        s += n * (2 * tp / (2 * tp + fp + n - tp) if tp else 0.0)
    return 100 * s / len(y)
D = {}
for sp in ("dev", "test"):
    T = {m: [{f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{x}/eval/preds_{sp}.jsonl")} for x in LK["text"]["dual"][m][sp]] for m in ("online", "full")}
    keys = sorted(T["online"][0]); y = np.array([T["online"][0][k]["gold"] for k in keys])
    S = {m: np.mean([ls(np.array([d[k]["scores"]["clean"] for k in keys])) for d in L], 0) for m, L in T.items()}
    E = np.mean([np.array([d[k]["logp"] for k in keys]) for d in ({r["key"]: r for r in rows(f"{R}/{x}/preds_{sp}.jsonl")} for x in LK["experts"]["e2"])], 0)
    D[sp] = (y, S, E)
def pick(m):
    y, S, E = D["dev"]; v = [wf1(y, (S[m] + al * E).argmax(1)) for al in A]; return A[int(np.argmax(np.array(v) - 1e-9 * np.arange(len(v))))]
ao, af = pick("online"), pick("full")
y, S, E = D["test"]; fo, ff = S["online"] + ao * E, S["full"] + af * E
def sm(s): z = np.exp(s - s.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)
w = np.zeros(len(y), bool); w[np.argsort(sm(fo).max(1), kind="stable")[:int(round(0.2 * len(y)))]] = True
out = {"design": __doc__, "alpha_dev": {"online": ao, "full": af}, "test": {
    "ens_online_text": round(wf1(y, S["online"].argmax(1)), 2), "ens_full_text": round(wf1(y, S["full"].argmax(1)), 2),
    "ens_online+e2": round(wf1(y, fo.argmax(1)), 2), "ens_full+e2": round(wf1(y, ff.argmax(1)), 2),
    "ens_wait20+e2": round(wf1(y, np.where(w, ff.argmax(1), fo.argmax(1))), 2)}}
json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out["alpha_dev"]), json.dumps(out["test"]))
