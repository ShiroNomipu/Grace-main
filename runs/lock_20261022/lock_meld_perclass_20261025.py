"""B5 item (2026-10-25): per-class rescue / harm of the audio experts against the LOCK MELD dual-online text model
(test, 5 seeds; run ids from runs/lock_20261022/lock_runs.json). Rescue = P(expert right | text wrong), harm =
P(expert wrong | text right), per gold class. Descriptive. Usage (repo root): python <this> --out <json>"""
import argparse, json
import numpy as np
R = "outputs/runs"; C = 7
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
LK = json.load(open("runs/lock_20261022/lock_runs.json"))["meld"]
def rows(p): return [json.loads(l) for l in open(p)]
T = [{f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{x}/eval/preds_test.jsonl")} for x in LK["text"]["dual"]["online"]["test"]]
keys = sorted(T[0]); y = np.array([T[0][k]["gold"] for k in keys]); P = [np.array([t[k]["scores"]["clean"] for k in keys]).argmax(1) for t in T]
out = {"design": __doc__, "n_per_class": np.bincount(y, minlength=C).tolist(), "text_acc": [round(float(np.mean([100 * (p[y == c] == c).mean() for p in P])), 1) for c in range(C)]}
for ex in ("e1", "e2", "e3"):
    Q = []
    for x in LK["experts"][ex]:
        d = {r["key"]: r for r in rows(f"{R}/{x}/preds_test.jsonl")}; assert set(d) == set(keys); Q.append(np.array([d[k]["logp"] for k in keys]).argmax(1))
    res, harm = [], []
    for c in range(C):
        m = y == c
        res.append(round(float(np.mean([100 * (q[m & (p != y)] == c).mean() if (m & (p != y)).any() else 0 for p, q in zip(P, Q)])), 1))
        harm.append(round(float(np.mean([100 * (q[m & (p == y)] != c).mean() if (m & (p == y)).any() else 0 for p, q in zip(P, Q)])), 1))
    out[ex] = {"rescue": res, "harm": harm, "pred_share_neutral": round(float(np.mean([100 * (q == 0).mean() for q in Q])), 1)}
json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "design"}))
