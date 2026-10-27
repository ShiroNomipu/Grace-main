"""Waiting rule with its threshold fixed on dev, and the dialogue bootstrap of (GRACE + E2) - (dual online + E2)
(2026-10-27, CPU only; written after the lock results and labelled so in the paper).
1. Dev threshold. For each dataset, seed and system (text: dual online -> dual offline; + E2: fused online -> fused
   offline at the dev alphas of the lock judge R20261022-211), the threshold is the confidence (largest softmax
   probability of the online score) of the dev utterance at rank round(0.2 n_dev) in ascending order; an utterance
   waits when its confidence is below it, so 20% of dev waits. The same threshold is applied to test. Reports the share
   waiting, wF1 per seed, and paired tests (lock judge arithmetic: dialogue bootstrap, B = 10000, seed 20261022) against
   the online system and against the test-ranked rule of the lock judge (q = 0.2). Also the median wait in seconds of
   the waited test utterances, with the timing of lock_cpu_analyses_20261025.py.
   Reproduction checks first: the test-ranked rule must give the lock judge means (dual_wait20, dual_wait20+e2) and,
   on IEMOCAP, the waited median of lock_cpu_analyses.json.
2. IEMOCAP and MELD, dev and test: paired (GRACE + E2) - (dual online + E2) with the bootstrap; same inputs and B1 judge
   alphas as b1_dev_check_20261027.py, whose means it must reproduce.
Usage (repo root): python <this> --out <json>"""
import argparse, collections, csv, json
import numpy as np
R = "outputs/runs"; NC = {"iemocap": 6, "meld": 7}; QW = 0.2
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
LK = json.load(open("runs/lock_20261022/lock_runs.json")); J2 = json.load(open(f"{R}/R20261022-211/lock_full.json"))["datasets"]
B1 = json.load(open("runs/lock_20261022/b1_runs.json")); JB = json.load(open(f"{R}/R20261022-340/b1_grace.json"))["datasets"]
CPU = json.load(open("runs/lock_20261022/lock_cpu_analyses.json")); BD = json.load(open("runs/lock_20261022/b1_dev_check.json"))
def rows(p): return [json.loads(l) for l in open(p)]
def r2(x): return round(float(x), 2)
def summ(v): v = np.asarray(v, float); return {"mean": r2(v.mean()), "sd": r2(v.std(ddof=1)), "per_seed": [r2(x) for x in v]}
def sm(s): z = np.exp(s - s.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)
def f1s(y, p, C):  # as in lock_judge_20261022.py
    out = np.zeros(C)
    for c in range(C):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        out[c] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return out
def wf1(y, p, C): return 100 * float((f1s(y, p, C) * np.bincount(y, minlength=C)).sum() / len(y))
def boot(keys, y, PA, PB, C, n=10000, seed=20261022):  # copied from lock_judge_20261022.py
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
    return {**summ(d), "n_pos": int((d > 0).sum()), "n_nonneg": int((d >= 0).sum()), **boot(keys, y, PA, PB, C)}
def wait_ranked(a, b, q):  # lock_judge_20261022.py wait()
    w = np.zeros(len(a), bool); w[np.argsort(sm(a).max(1), kind="stable")[:int(round(q * len(a)))]] = True
    return w
def load_split(ds, sp):
    lk = LK[ds]
    T = {m: [{f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{x}/eval/preds_{sp}.jsonl")} for x in lk["text"]["dual"][m][sp]]
         for m in ("online", "full")}
    keys = sorted(T["online"][0]); ref = T["online"][0]; y = np.array([ref[k]["gold"] for k in keys])
    E = [{r["key"]: r for r in rows(f"{R}/{x}/preds_{sp}.jsonl")} for x in lk["experts"]["e2"]]
    for d in T["online"] + T["full"] + E:
        assert set(d) == set(keys) and all(d[k]["gold"] == ref[k]["gold"] for k in keys), (ds, sp)
    S = {m: [np.array([d[k]["scores"]["clean"] for k in keys]) for d in T[m]] for m in T}
    return keys, y, ref, S, [np.array([d[k]["logp"] for k in keys]) for d in E]
def wait_seconds(ds, keys, ref):  # lock_cpu_analyses_20261025.py, test split
    dl = collections.defaultdict(list)
    for i, k in enumerate(keys):
        dl[k.split("|")[0]].append((int(k.split("|")[1]), i))
    order = {}
    for d, lst in dl.items():
        lst.sort(); idx = [i for _, i in lst]
        for j, i in enumerate(idx):
            order[i] = (j, idx)
    if ds == "iemocap":
        alg = json.load(open("runs/iemocap_align_20261018.json"))["utts"]; end = np.array([alg[k]["end"] for k in keys])
    else:
        tt = {}
        for r in csv.DictReader(open("datasets/meld_csv/test_sent_emo.csv", encoding="utf-8")):
            h, mi, rest = r["EndTime"].split(":"); s_, ms = rest.split(","); tt[f"{int(r['Dialogue_ID'])}|{int(r['Utterance_ID'])}"] = int(h) * 3600 + int(mi) * 60 + int(s_) + int(ms) / 1000
        end = np.array([tt.get(k, np.nan) for k in keys])
    w = np.zeros(len(keys))
    for i in range(len(keys)):
        j, idx = order[i]; w[i] = end[idx[min(j + 2, len(idx) - 1)]] - end[i]
    return w, np.isfinite(w) & (w >= 0)


out = {"design": __doc__, "datasets": {}}
for ds in ("iemocap", "meld"):
    C = NC[ds]; al = J2[ds]["alpha_dev"]; o = {"alpha_dev": {k: al[k] for k in ("dual_online+e2", "dual_full+e2")}}
    D = {sp: load_split(ds, sp) for sp in ("dev", "test")}
    sysd = {}
    for sp, (keys, y, ref, S, E2) in D.items():
        sysd[sp] = {"text": (S["online"], S["full"]),
                    "e2": ([s + al["dual_online+e2"] * e for s, e in zip(S["online"], E2)], [s + al["dual_full+e2"] * e for s, e in zip(S["full"], E2)])}
    keys, y, ref, _, _ = D["test"]; ydev = D["dev"][1]
    ws, okw = wait_seconds(ds, keys, ref)
    o["dev_threshold"] = {}
    for name, lock_row in (("text", "dual_wait20"), ("e2", "dual_wait20+e2")):
        on_d, off_d = sysd["dev"][name]; on_t, off_t = sysd["test"][name]
        Pon = [s.argmax(1) for s in on_t]
        Wr = [wait_ranked(s, t, QW) for s, t in zip(on_t, off_t)]
        Prank = [np.where(w, t.argmax(1), s.argmax(1)) for w, s, t in zip(Wr, on_t, off_t)]
        rep = r2(np.mean([wf1(y, p, C) for p in Prank]))
        assert rep == J2[ds]["table"]["test"][lock_row]["mean"], (ds, name, rep, J2[ds]["table"]["test"][lock_row]["mean"])
        if name == "e2":
            med = r2(np.median(np.concatenate([ws[w & okw] for w in Wr])))
            assert ds != "iemocap" or med == CPU[ds]["wait_seconds"]["waited_20pct_online+e2"]["median"], (ds, med)
        tau, Wd, Wt = [], [], []
        for sd, td in zip(on_d, on_t):
            cd, ct = sm(sd).max(1), sm(td).max(1)
            k = int(round(QW * len(cd))); th = float(np.sort(cd, kind="stable")[k]); tau.append(th)
            Wd.append(cd < th); Wt.append(ct < th)
        Pdev = [np.where(w, t.argmax(1), s.argmax(1)) for w, s, t in zip(Wd, on_d, off_d)]
        Pthr = [np.where(w, t.argmax(1), s.argmax(1)) for w, s, t in zip(Wt, on_t, off_t)]
        o["dev_threshold"][name] = {
            "threshold_per_seed": [round(t, 4) for t in tau],
            "dev_share_waiting_pct": summ([100 * w.mean() for w in Wd]),
            "test_share_waiting_pct": summ([100 * w.mean() for w in Wt]),
            "dev_wf1": summ([wf1(ydev, p, C) for p in Pdev]),
            "test_wf1": summ([wf1(y, p, C) for p in Pthr]),
            "test_ranked_wf1_reproduced": rep,
            "test_waited_seconds": {"median": r2(np.median(np.concatenate([ws[w & okw] for w in Wt]))),
                                    "p90": r2(np.percentile(np.concatenate([ws[w & okw] for w in Wt]), 90))},
            "paired_test": {"devthr-online": paired(y, keys, Pthr, Pon, C), "devthr-test_ranked": paired(y, keys, Pthr, Prank, C)}}
        if name == "e2":
            o["dev_threshold"][name]["test_ranked_waited_median_s_reproduced"] = med
    # 2. injection + E2 against ours, dev and test
    alb = JB[ds]["alpha_dev"]; o["b1"] = {"alpha_dev": {k: alb[k] for k in ("grace+e2", "dual_online+e2")}}
    for sp in ("dev", "test"):
        G = [{f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{d}/eval/preds_{sp}.jsonl")} for d in B1[ds]["dump"][sp]]
        k2, y2, _, S2, E2 = D[sp]
        for g in G:
            assert set(g) == set(k2) and all(g[k]["gold"] == int(y2[i]) for i, k in enumerate(k2)), (ds, sp)
        gon = [np.array([g[k]["scores"]["clean"] for k in k2]) for g in G]
        Pg = [np.argmax(s + alb["grace+e2"] * e, 1) for s, e in zip(gon, E2)]
        Pd = [np.argmax(s + alb["dual_online+e2"] * e, 1) for s, e in zip(S2["online"], E2)]
        pr = paired(y2, k2, Pg, Pd, C)
        assert pr["mean"] == -BD[ds][sp]["dual_online+e2 - grace+e2"]["mean"], (ds, sp, pr["mean"])
        o["b1"][sp] = {"grace+e2-dual_online+e2": pr}
    out["datasets"][ds] = o
json.dump(out, open(a.out, "w"), indent=1)
for ds, o in out["datasets"].items():
    for name, x in o["dev_threshold"].items():
        print(ds, name, "share dev/test", x["dev_share_waiting_pct"]["mean"], x["test_share_waiting_pct"]["mean"], "wf1 test", x["test_wf1"]["mean"],
              "vs online", {k: x["paired_test"]["devthr-online"][k] for k in ("mean", "n_pos", "ci95", "p_two_sided")},
              "vs ranked", {k: x["paired_test"]["devthr-test_ranked"][k] for k in ("mean", "n_pos", "ci95")}, "wait s", x["test_waited_seconds"])
    for sp in ("dev", "test"):
        p = o["b1"][sp]["grace+e2-dual_online+e2"]; print(ds, "b1", sp, {k: p[k] for k in ("mean", "n_pos", "ci95", "p_two_sided")})
