"""Post-hoc MELD epoch-selection sensitivity (2026-10-25; written AFTER seeing the lock results; option A chosen by the
user: the main table keeps the registered last-epoch rule, this is a labelled extra row). Same 15 MELD lock runs; the
dev-best adapter (<tag>_lora_best, saved by run_v1.py by dev wF1; dual runs evaluate dev in the full mode).
Dual online / full: dumps R20261025-01..05 / -06..10 (smoke_textdump_future_split_which_20261025.py --which best).
Online-only / full-only: the runs' own best-adapter test metric <tag>_lora_test.json (same scoring path as the dumps).
Reported, not judged: mean / sd / per seed; dual full - dual online paired with dialogue bootstrap (B = 10000, seed
20261022); the last-epoch values from J2 (R20261022-211) next to them. Usage (repo root): python <this> --out <json>"""
import argparse, json
import numpy as np
R = "outputs/runs"; C = 7
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
LK = json.load(open("runs/lock_20261022/lock_runs.json")); SEEDS = LK["seeds"]; M = LK["meld"]
J2 = json.load(open(f"{R}/R20261022-211/lock_full.json"))["datasets"]["meld"]["table"]["test"]
DUMP = {"online": [f"R20261025-{i:02d}" for i in range(1, 6)], "full": [f"R20261025-{i:02d}" for i in range(6, 11)]}
def rows(p): return [json.loads(l) for l in open(p)]
def r2(x): return round(float(x), 2)
def summ(v): v = np.asarray(v, float); return {"mean": r2(v.mean()), "sd": r2(v.std(ddof=1)), "per_seed": [r2(x) for x in v]}
def f1s(y, p):
    out = []
    for c in range(C):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); n = (y == c).sum()
        out.append(2 * tp / (2 * tp + fp + n - tp) if tp else 0.0)
    return np.array(out)
def wf1(y, p): return 100 * float((f1s(y, p) * np.bincount(y, minlength=C)).sum() / len(y))
def boot(keys, y, PA, PB, n=10000, seed=20261022):
    dl = np.array([k.split("|")[0] for k in keys]); uq = sorted(set(dl)); idx = np.searchsorted(uq, dl)
    W = np.random.default_rng(seed).multinomial(len(uq), np.full(len(uq), 1 / len(uq)), size=n)
    def bw(p):
        Mx = np.zeros((len(uq), 3 * C))
        np.add.at(Mx, (idx, y), 1); np.add.at(Mx, (idx, C + p), 1); np.add.at(Mx, (idx[y == p], 2 * C + y[y == p]), 1)
        S = W @ Mx; sup, pr, tp = S[:, :C], S[:, C:2 * C], S[:, 2 * C:]
        f1 = np.where(sup + pr > 0, 2 * tp / np.maximum(sup + pr, 1e-9), 0); return 100 * (sup * f1).sum(1) / sup.sum(1)
    D = np.mean([bw(p) for p in PA], 0) - np.mean([bw(p) for p in PB], 0)
    return {"ci95": [r2(np.percentile(D, 2.5)), r2(np.percentile(D, 97.5))], "p_two_sided": round(float(min(1, 2 * min((D <= 0).mean(), (D >= 0).mean()))), 4)}
out = {"design": __doc__, "provenance": []}
P = {}
for mode, ids in DUMP.items():
    P[mode] = []
    for i, x in enumerate(ids):
        info = json.load(open(f"{R}/{x}/eval/run_info.json"))
        assert info["src"] == M["train"]["dual"][i] and info["args"]["which"] == "best" and info["args"]["mode"] == mode and info["args"]["split"] == "test", (x, info["args"])
        assert int(info["src_args"]["--seed"]) == SEEDS[i] and info["src_args"]["--dataset"] == "meld"
        out["provenance"].append({"dump": x, "src": info["src"], "seed": SEEDS[i], "mode": mode, "adapter_sha256": info["sha256"]["adapter_model.safetensors"]})
        P[mode].append({f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{x}/eval/preds_test.jsonl")})
keys = sorted(P["online"][0]); y = np.array([P["online"][0][k]["gold"] for k in keys])
pred = {m: [np.array([d[k]["pred"]["clean"] for k in keys]) for d in L] for m, L in P.items()}
for m in pred:
    for d in P[m]:
        assert set(d) == set(keys)
def own(var):
    v = []
    for t in M["train"][var]:
        g = open(f"{R}/{t}/args.txt").read().split(); tag = g[g.index("--tag") + 1]
        v.append(100 * json.load(open(f"{R}/{t}/{tag}_lora_test.json"))["weighted_f1"])
    return v
out["best_dev_epoch"] = {"dual_online": summ([wf1(y, p) for p in pred["online"]]), "dual_full": summ([wf1(y, p) for p in pred["full"]]),
                         "online_only": summ(own("online")), "full_only": summ(own("full"))}
d = np.array([wf1(y, q) - wf1(y, p) for p, q in zip(pred["online"], pred["full"])])
out["dual_full_minus_online_best"] = {**summ(d), "n_pos": int((d > 0).sum()), **boot(keys, y, pred["full"], pred["online"])}
out["last_epoch_from_J2"] = {k: J2[k]["mean"] for k in ("dual_online", "dual_full", "online_online", "full_full")}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("design", "provenance")}, indent=1))
