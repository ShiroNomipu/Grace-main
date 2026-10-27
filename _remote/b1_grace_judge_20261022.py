"""B1 judgment and ledger (2026-10-22; card runs/锁定批_20261022/B1_GRACE修复重跑_卡_2026-10-22.md; line logged in
charter/decisions.md before any B1 training ran). GRACE = audio injected into the LLM (dropout-mask fix, 15 past turns),
seeds 123/42/3047/7/2026 paired with the lock batch by seed; test wF1; last epoch. Run ids only from
runs/lock_20261022/b1_runs.json and lock_runs.json.
Line: B1-L1 (IEMOCAP) late fusion beats injection: (dual online + E2, dev alpha) - GRACE (audio on), mean >= +0.5 and
>= 4/5 seeds > 0.
Reported, not judged, both datasets: GRACE with audio on / off (same model; "drop" = the training-time dropped state);
GRACE vs online-only text and vs dual online text; GRACE + E2 (alpha on dev); E2 rescue rate on GRACE errors; dialogue
bootstrap (B = 10000, seed 20261022) of each paired difference; per-class F1; each dump's clean wF1 against the run's own
last-epoch test metric (|diff| > 0.05 points -> error).
Usage (repo root): python <this> --out <json> [--dry]
  --dry: code test only, IEMOCAP, with the old locked GRACE dumps (R20261017-27.., -28..) and the old text16 dumps
  (R20261021-11..20) standing in for both text variants; provenance and dump checks skipped. Must reproduce gate 1:
  GRACE 72.31, GRACE + E2 73.39 (alpha 0.2), text16 + E2 74.24 (alpha 0.3)."""
import argparse, json, os
import numpy as np
R = "outputs/runs"; ALPHAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2]; NC = {"iemocap": 6, "meld": 7}
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
B1 = json.load(open("runs/lock_20261022/b1_runs.json")); LK = json.load(open("runs/lock_20261022/lock_runs.json")); SEEDS = B1["seeds"]
assert SEEDS == LK["seeds"]


def rows(p): return [json.loads(l) for l in open(p)]
def r2(x): return round(float(x), 2)
def summ(v): v = np.asarray(v, float); return {"mean": r2(v.mean()), "sd": r2(v.std(ddof=1)), "per_seed": [r2(x) for x in v]}
def ok(run): e = f"{R}/{run}/exit_code"; return os.path.exists(e) and open(e).read().strip() == "0"
def args_of(run):
    t = open(f"{R}/{run}/args.txt").read().split()
    return {t[i]: (t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True) for i in range(len(t)) if t[i].startswith("--")}
def f1s(y, p, C):
    out = []
    for c in range(C):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); n = (y == c).sum()
        out.append(2 * tp / (2 * tp + fp + n - tp) if tp else 0.0)
    return np.array(out)
def wf1(y, p, C): return 100 * float((f1s(y, p, C) * np.bincount(y, minlength=C)).sum() / len(y))
def boot(keys, y, PA, PB, C, n=10000, seed=20261022):
    dl = np.array([k.split("|")[0] for k in keys]); uq = sorted(set(dl)); idx = np.searchsorted(uq, dl)
    W = np.random.default_rng(seed).multinomial(len(uq), np.full(len(uq), 1 / len(uq)), size=n)
    def bw(p):
        M = np.zeros((len(uq), 3 * C))
        np.add.at(M, (idx, y), 1); np.add.at(M, (idx, C + p), 1); np.add.at(M, (idx[y == p], 2 * C + y[y == p]), 1)
        S = W @ M; sup, pr, tp = S[:, :C], S[:, C:2 * C], S[:, 2 * C:]
        f1 = np.where(sup + pr > 0, 2 * tp / np.maximum(sup + pr, 1e-9), 0); return 100 * (sup * f1).sum(1) / sup.sum(1)
    D = np.mean([bw(p) for p in PA], 0) - np.mean([bw(p) for p in PB], 0)
    return {"ci95": [r2(np.percentile(D, 2.5)), r2(np.percentile(D, 97.5))], "p_two_sided": round(float(min(1, 2 * min((D <= 0).mean(), (D >= 0).mean()))), 4)}
def paired(y, keys, PA, PB, C):
    d = np.array([wf1(y, p, C) - wf1(y, q, C) for p, q in zip(PA, PB)])
    return {**summ(d), "n_pos": int((d > 0).sum()), **boot(keys, y, PA, PB, C)}
def pick(y, S, L, C):
    curve = [float(np.mean([wf1(y, np.argmax(s + al * l, 1), C) for s, l in zip(S, L)])) for al in ALPHAS]
    best = 0
    for i in range(1, len(ALPHAS)):
        if curve[i] > curve[best] + 1e-9:
            best = i
    return ALPHAS[best]


out = {"design": __doc__, "datasets": {}, "provenance": []}
for ds in ("iemocap", "meld"):
    C = NC[ds]; b = B1[ds]; lk = LK[ds]; D = {}
    if a.dry:
        if ds == "meld":
            continue
        b = {"dump": {"dev": [f"R20261017-{i}" for i in (27, 31, 35, 39, 43)], "test": [f"R20261017-{i}" for i in (28, 32, 36, 40, 44)]}}
        t16 = {"dev": [f"R20261021-{i}" for i in range(11, 16)], "test": [f"R20261021-{i}" for i in range(16, 21)]}
        lk = {"text": {"online": {"online": t16}, "dual": {"online": t16}}, "experts": lk["experts"]}
    for sp in ("dev", "test"):
        G = []
        for i, s in enumerate(SEEDS):
            d = b["dump"][sp][i]
            if not a.dry:
                t = b["train"][i]; ga, da = args_of(t), args_of(d)
                assert ok(t) and ok(d) and int(ga["--seed"]) == s and ga["--dataset"] == ds and ga.get("--mdrop_consistent") and ga.get("--ctx") == "16", (t, ga)
                assert da["--split"] == sp and da["--checkpoint"] == f"outputs/runs/{t}/{b['tag'][i]}_last" and da.get("--ctx") == "16", (d, da)
            G.append({f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{d}/eval/preds_{sp}.jsonl")})
            out["provenance"].append({"dump": d, "train": None if a.dry else t, "seed": s, "split": sp})
        def text(var, mode):
            return [{f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{x}/eval/preds_{sp}.jsonl")} for x in lk["text"][var][mode][sp]]
        T = {"online_only": text("online", "online"), "dual_online": text("dual", "online")}
        E2 = [{r["key"]: r for r in rows(f"{R}/{x}/preds_{sp}.jsonl")} for x in lk["experts"]["e2"]]
        keys = sorted(G[0]); y = np.array([G[0][k]["gold"] for k in keys])
        for src in G + T["online_only"] + T["dual_online"] + E2:
            assert set(src) == set(keys) and all(src[k]["gold"] == G[0][k]["gold"] for k in keys), (ds, sp)
        D[sp] = {"keys": keys, "y": y,
                 "g_on": [np.array([g[k]["scores"]["clean"] for k in keys]) for g in G],
                 "g_off": [np.array([g[k]["scores"]["drop"] for k in keys]) for g in G],
                 "online_only": [np.array([t[k]["scores"]["clean"] for k in keys]) for t in T["online_only"]],
                 "dual_online": [np.array([t[k]["scores"]["clean"] for k in keys]) for t in T["dual_online"]],
                 "e2": [np.array([e[k]["logp"] for k in keys]) for e in E2]}
    # each dump's clean wF1 must equal the run's own last-epoch test metric
    chk = []
    for i, s in enumerate([] if a.dry else SEEDS):
        mf = f"{R}/{b['train'][i]}/{b['tag'][i]}_test_last_metrics.json"
        ref = 100 * json.load(open(mf))["weighted_f1"]; got = wf1(D["test"]["y"], D["test"]["g_on"][i].argmax(1), C)
        assert abs(ref - got) <= 0.05, (b["train"][i], ref, got)
        chk.append({"seed": s, "train_metric": r2(ref), "dump": r2(got)})
    a_ge = pick(D["dev"]["y"], D["dev"]["g_on"], D["dev"]["e2"], C)
    a_de = pick(D["dev"]["y"], D["dev"]["dual_online"], D["dev"]["e2"], C)
    y, k = D["test"]["y"], D["test"]["keys"]
    P = {"grace_audio_on": [s.argmax(1) for s in D["test"]["g_on"]], "grace_audio_off": [s.argmax(1) for s in D["test"]["g_off"]],
         "online_only_text": [s.argmax(1) for s in D["test"]["online_only"]], "dual_online_text": [s.argmax(1) for s in D["test"]["dual_online"]],
         "grace+e2": [np.argmax(s + a_ge * e, 1) for s, e in zip(D["test"]["g_on"], D["test"]["e2"])],
         "dual_online+e2": [np.argmax(s + a_de * e, 1) for s, e in zip(D["test"]["dual_online"], D["test"]["e2"])]}
    o = {"alpha_dev": {"grace+e2": a_ge, "dual_online+e2": a_de}, "dump_check": chk,
         "test": {n: {**summ([wf1(y, p, C) for p in ps]), "per_class_f1": [r2(100 * x) for x in np.mean([f1s(y, p, C) for p in ps], 0)]} for n, ps in P.items()},
         "paired_test": {f"{p} - {q}": paired(y, k, P[p], P[q], C) for p, q in (
             ("grace_audio_on", "grace_audio_off"), ("grace_audio_on", "online_only_text"), ("grace_audio_on", "dual_online_text"),
             ("dual_online+e2", "grace_audio_on"), ("grace+e2", "grace_audio_on"), ("dual_online+e2", "grace+e2"))},
         "e2_rescue_on_grace_errors": r2(np.mean([100 * (e.argmax(1)[g.argmax(1) != y] == y[g.argmax(1) != y]).mean() for g, e in zip(D["test"]["g_on"], D["test"]["e2"])]))}
    if ds == "iemocap":
        l1 = o["paired_test"]["dual_online+e2 - grace_audio_on"]
        o["verdict_B1_L1"] = {"mean": l1["mean"], "n_pos": l1["n_pos"], "ci95": l1["ci95"], "pass": bool(l1["mean"] >= 0.5 and l1["n_pos"] >= 4)}
    out["datasets"][ds] = o
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({ds: {"alpha_dev": o["alpha_dev"], "test": {n: x["mean"] for n, x in o["test"].items()}, "verdict": o.get("verdict_B1_L1")} for ds, o in out["datasets"].items()}, indent=1))
