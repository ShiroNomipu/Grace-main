"""Lock batch judgment and ledger (2026-10-22; card runs/锁定批_20261022/锁定批_卡_2026-10-22.md; lines logged in
charter/decisions.md before any lock-batch training ran). Every main-axis number of the paper is read from this script's
JSON; run ids come only from runs/lock_20261022/lock_runs.json (lock_jobs_gen_20261022.py).
Seeds 123/42/3047/7/2026, paired by seed. Test wF1 (IEMOCAP 6 classes, MELD 7); last-epoch adapters; text = one LLM
(C15 recipe: 15 past turns, 2 future turns). Selection on dev only: alpha per (dataset, text variant, expert) by the mean
dev wF1 over the 5 seeds, grid ALPHAS, ties -> smaller. Waiting: the 20% least confident utterances by the softmax max of
the online (fused) score take the offline (fused) prediction; the fraction 0.2 is fixed in advance.
Lines (verdicts):
  L1 future helps the text model, per dataset: dual offline - dual online, mean >= +0.5 and >= 4/5 seeds > 0.
  L2 one model, two modes, per dataset (stage full): dual online - online-only >= -0.3 and dual offline - offline-only
     >= -0.3 (means; the DM-gate margin of 10-21).
  L3 voice replaces waiting, IEMOCAP with E2: (a) of the online text errors that the offline mode fixes, online + E2
     also fixes >= 50%, in >= 4/5 seeds; (b) (dual online + E2) - (dual offline text), mean >= 0 and >= 4/5 seeds >= 0.
  L4 MELD audio stays closed unless (dual online + E2) - (dual online), dev alpha, has mean >= +0.8 and >= 4/5 > 0.
Reported, not judged: main table rows per seed / mean / sd / per-class F1, dev and test; dialogue bootstrap (B = 10000,
seed 20261022) of every paired difference; E1 / E2 / E3 alone, rescue rate, harm rate, dev alpha, fused test, per dataset;
voice-vs-future fix overlap (dev alpha and fixed 0.3) and the waiting curve, dev and test; repeated scripted lines
(IEMOCAP); dev alpha curves; provenance of every run (seed, mode, split checked against args; adapter sha256).
Usage (repo root): python <this> --stage main|full --out <json> [--dry]
  --dry: use only the seeds whose runs exist (to test the code on the reused C15 seeds before the batch runs)."""
import argparse, collections, glob, json, os, re
import numpy as np
R = "outputs/runs"
ALPHAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2]
Q = [0, 0.1, 0.2, 0.3, 0.5, 1.0]; QW = 0.2
NC = {"iemocap": 6, "meld": 7}
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["main", "full"])
ap.add_argument("--out", required=True); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
RUNS = json.load(open("runs/lock_20261022/lock_runs.json")); SEEDS = RUNS["seeds"]
VARS = ["dual"] + (["online", "full"] if a.stage == "full" else [])


def rows(p): return [json.loads(l) for l in open(p)]
def r2(x): return round(float(x), 2)
def summ(v): v = np.asarray(v, float); return {"mean": r2(v.mean()), "sd": r2(v.std(ddof=1)) if len(v) > 1 else None, "per_seed": [r2(x) for x in v]}
def sm(s): z = np.exp(s - s.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)
def f1s(y, p, C):
    out = []
    for c in range(C):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); n = (y == c).sum()
        out.append(2 * tp / (2 * tp + fp + n - tp) if tp else 0.0)
    return np.array(out)
def wf1(y, p, C): return 100 * float((f1s(y, p, C) * np.bincount(y, minlength=C)).sum() / len(y))
def ok(run): e = f"{R}/{run}/exit_code"; return os.path.exists(e) and open(e).read().strip() == "0"
def args_of(run):
    t = open(f"{R}/{run}/args.txt").read().split()
    return {t[i]: t[i + 1] for i in range(len(t) - 1) if t[i].startswith("--") and not t[i + 1].startswith("--")}


prov = []
def load_text(ds, var, mode, split, run, seed):
    info = json.load(open(f"{R}/{run}/eval/run_info.json")); sa = info["src_args"]
    assert ok(run) and ok(info["src"]), run
    assert sa.get("--dataset", "iemocap") == ds and int(sa["--seed"]) == seed and sa["--future_mode"] == var, (run, sa)
    assert info["args"]["mode"] == mode and info["args"].get("split", "test") == split and sa.get("--ctx") == "16", (run, info["args"])
    prov.append({"run": run, "src": info["src"], "dataset": ds, "variant": var, "mode": mode, "split": split, "seed": seed,
                 "adapter_sha256": info.get("sha256", {}).get("adapter_model.safetensors"), "wf1_dump": r2(100 * info["weighted_f1"])})
    return {f"{r['dialogue_id']}|{r['utt_id']}": r for r in rows(f"{R}/{run}/eval/preds_{split}.jsonl")}
def load_expert(ds, ex, split, run, seed):
    g = args_of(run); assert ok(run) and g["--dataset"] == ds and int(g["--seed"]) == seed and g["--expert"] == ex, (run, g)
    prov.append({"run": run, "dataset": ds, "expert": ex, "split": split, "seed": seed})
    return {r["key"]: r for r in rows(f"{R}/{run}/preds_{split}.jsonl")}


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
def paired(y, keys, PA, PB, C, nonneg=False):
    d = np.array([wf1(y, p, C) - wf1(y, q, C) for p, q in zip(PA, PB)])
    return {**summ(d), "n_pos": int((d > 0).sum()), "n_nonneg": int((d >= 0).sum()), **boot(keys, y, PA, PB, C)}
def pick(y, S, L, C):
    curve = [float(np.mean([wf1(y, np.argmax(s + al * l, 1), C) for s, l in zip(S, L)])) for al in ALPHAS]
    best = 0
    for i in range(1, len(ALPHAS)):
        if curve[i] > curve[best] + 1e-9:
            best = i
    return ALPHAS[best], [round(c, 3) for c in curve]
def wait(a, b, q):
    w = np.zeros(len(a), bool); w[np.argsort(sm(a).max(1), kind="stable")[:int(round(q * len(a)))]] = True
    return np.where(w, b.argmax(1), a.argmax(1))


out = {"design": __doc__, "stage": a.stage, "dry": a.dry, "datasets": {}}
for ds in ("iemocap", "meld"):
    C = NC[ds]; rr = RUNS[ds]
    if a.dry:
        use = [i for i, s in enumerate(SEEDS) if all(ok(rr["text"][v][m][sp][i]) for v in VARS for m in rr["text"][v] for sp in ("dev", "test"))]
        if not use:
            continue
    else:
        use = list(range(len(SEEDS)))
    seeds = [SEEDS[i] for i in use]
    D = {}
    for sp in ("dev", "test"):
        T = {}
        for v in VARS:
            for m, by_split in rr["text"][v].items():
                T[f"{v}_{m}"] = [load_text(ds, v, m, sp, by_split[sp][i], SEEDS[i]) for i in use]
        ref = T["dual_online"][0]; keys = sorted(ref); y = np.array([ref[k]["gold"] for k in keys])
        S = {}
        for name, lst in T.items():
            S[name] = []
            for d in lst:
                assert set(d) == set(keys) and all(d[k]["gold"] == ref[k]["gold"] for k in keys), (ds, sp, name)
                S[name].append(np.array([d[k]["scores"]["clean"] for k in keys]))
        E = {}
        for ex, ids in rr["experts"].items():
            E[ex] = []
            for i in use:
                d = load_expert(ds, ex, sp, ids[i], SEEDS[i])
                assert set(d) == set(keys) and all(d[k]["gold"] == ref[k]["gold"] for k in keys), (ds, sp, ex, ids[i])
                E[ex].append(np.array([d[k]["logp"] for k in keys]))
        D[sp] = {"keys": keys, "y": y, "S": S, "E": E}
    o = {"seeds": seeds, "n": {sp: len(D[sp]["y"]) for sp in D}}
    # dev alpha per (text variant x mode, expert)
    alpha, curves = {}, {}
    for name in D["dev"]["S"]:
        for ex in D["dev"]["E"]:
            alpha[f"{name}+{ex}"], curves[f"{name}+{ex}"] = pick(D["dev"]["y"], D["dev"]["S"][name], D["dev"]["E"][ex], C)
    o["alpha_dev"] = alpha; o["alpha_dev_curves"] = curves
    # predictions of the main rows, per split
    P = {}
    for sp in ("dev", "test"):
        S, E, y = D[sp]["S"], D[sp]["E"], D[sp]["y"]
        fo = [s + alpha["dual_online+e2"] * e for s, e in zip(S["dual_online"], E["e2"])]
        ff = [s + alpha["dual_full+e2"] * e for s, e in zip(S["dual_full"], E["e2"])]
        P[sp] = {name: [s.argmax(1) for s in lst] for name, lst in S.items()}
        P[sp]["dual_wait20"] = [wait(s, t, QW) for s, t in zip(S["dual_online"], S["dual_full"])]
        P[sp]["dual_online+e2"] = [f.argmax(1) for f in fo]
        P[sp]["dual_full+e2"] = [f.argmax(1) for f in ff]
        P[sp]["dual_wait20+e2"] = [wait(f, g, QW) for f, g in zip(fo, ff)]
        D[sp]["fo"], D[sp]["ff"] = fo, ff
    o["table"] = {sp: {name: {**summ([wf1(D[sp]["y"], p, C) for p in ps]),
                              "per_class_f1": [r2(100 * x) for x in np.mean([f1s(D[sp]["y"], p, C) for p in ps], 0)]}
                       for name, ps in P[sp].items()} for sp in ("dev", "test")}
    yt, kt, Pt = D["test"]["y"], D["test"]["keys"], P["test"]
    # verdicts
    v = {}
    l1 = paired(yt, kt, Pt["dual_full"], Pt["dual_online"], C)
    v["L1_future_helps_text"] = {**l1, "pass": bool(l1["mean"] >= 0.5 and l1["n_pos"] >= 4)}
    if a.stage == "full":
        d_on = paired(yt, kt, Pt["dual_online"], Pt["online_online"], C)
        d_off = paired(yt, kt, Pt["dual_full"], Pt["full_full"], C)
        v["L2_one_model_two_modes"] = {"dual_online_minus_online_only": d_on, "dual_offline_minus_offline_only": d_off,
                                       "pass": bool(d_on["mean"] >= -0.3 and d_off["mean"] >= -0.3)}
    # voice vs future overlap (per split; dev alpha and fixed 0.3)
    ov = {}
    for sp in ("dev", "test"):
        y, S, E = D[sp]["y"], D[sp]["S"], D[sp]["E"]
        for tag, al in (("alpha_dev", alpha["dual_online+e2"]), ("alpha_0.3", 0.3)):
            lst = []
            for so, sf, e in zip(S["dual_online"], S["dual_full"], E["e2"]):
                To, Tf, Ao = so.argmax(1) == y, sf.argmax(1) == y, (so + al * e).argmax(1) == y
                fut, voi = ~To & Tf, ~To & Ao
                lst.append({"future_fixes": int(fut.sum()), "voice_also": int((fut & Ao).sum()), "share_voice_also": r2(100 * (fut & Ao).sum() / max(fut.sum(), 1)),
                            "voice_fixes": int(voi.sum()), "future_also": int((voi & Tf).sum()),
                            "future_breaks": int((To & ~Tf).sum()), "voice_breaks": int((To & ~Ao).sum())})
            ov[f"{sp}_{tag}"] = lst
    o["voice_vs_future"] = ov
    if ds == "iemocap":
        sh = [x["share_voice_also"] for x in ov["test_alpha_dev"]]
        l3b = paired(yt, kt, Pt["dual_online+e2"], Pt["dual_full"], C)
        v["L3_voice_replaces_waiting"] = {"a_share_per_seed": sh, "a_pass": bool(sum(x >= 50 for x in sh) >= 4), "b": l3b,
                                          "b_pass": bool(l3b["mean"] >= 0 and l3b["n_nonneg"] >= 4)}
        v["L3_voice_replaces_waiting"]["pass"] = v["L3_voice_replaces_waiting"]["a_pass"] and v["L3_voice_replaces_waiting"]["b_pass"]
    else:
        l4 = paired(yt, kt, Pt["dual_online+e2"], Pt["dual_online"], C)
        v["L4_meld_audio"] = {**l4, "alpha_dev": alpha["dual_online+e2"], "verdict": "REOPEN" if (l4["mean"] >= 0.8 and l4["n_pos"] >= 4) else "CLOSED"}
    o["verdicts"] = v
    # every paired difference of the main rows (reported)
    comp = [("dual_full", "dual_online"), ("dual_wait20", "dual_online"), ("dual_online+e2", "dual_online"),
            ("dual_full+e2", "dual_online+e2"), ("dual_wait20+e2", "dual_online+e2"), ("dual_online+e2", "dual_full"),
            ("dual_wait20+e2", "dual_full")]
    if a.stage == "full":
        comp += [("dual_online", "online_online"), ("dual_full", "full_full"), ("full_full", "online_online")]
    o["paired_test"] = {f"{p} - {q}": paired(yt, kt, Pt[p], Pt[q], C) for p, q in comp}
    # experts: alone, rescue / harm against dual online, fusion with dev alpha
    ex_out = {}
    for ex in D["test"]["E"]:
        r = {"alpha_dev_online": alpha[f"dual_online+{ex}"], "alpha_dev_offline": alpha[f"dual_full+{ex}"]}
        for sp in ("dev", "test"):
            y, S, L = D[sp]["y"], D[sp]["S"]["dual_online"], D[sp]["E"][ex]
            To = [s.argmax(1) == y for s in S]; Ae = [l.argmax(1) == y for l in L]
            r[sp] = {"alone": summ([wf1(y, l.argmax(1), C) for l in L]),
                     "rescue_P(expert_right|text_wrong)": r2(np.mean([100 * e[~t].mean() for t, e in zip(To, Ae)])),
                     "harm_P(expert_wrong|text_right)": r2(np.mean([100 * (~e[t]).mean() for t, e in zip(To, Ae)])),
                     "fused_online": summ([wf1(y, np.argmax(s + r["alpha_dev_online"] * l, 1), C) for s, l in zip(S, L)])}
        r["d_test_online"] = paired(yt, kt, [np.argmax(s + r["alpha_dev_online"] * l, 1) for s, l in zip(D["test"]["S"]["dual_online"], D["test"]["E"][ex])], Pt["dual_online"], C)
        ex_out[ex] = r
    o["experts"] = ex_out
    # waiting curve (text and with E2), dev and test
    o["waiting_curve"] = {sp: {"q": Q, "text": [r2(np.mean([wf1(D[sp]["y"], wait(s, t, q), C) for s, t in zip(D[sp]["S"]["dual_online"], D[sp]["S"]["dual_full"])])) for q in Q],
                               "with_e2": [r2(np.mean([wf1(D[sp]["y"], wait(f, g, q), C) for f, g in zip(D[sp]["fo"], D[sp]["ff"])])) for q in Q]}
                          for sp in ("dev", "test")}
    # repeated scripted lines (IEMOCAP): gold differs from the training-set majority label of the same line
    if ds == "iemocap":
        RAW = "/root/iemocap_raw/x/iemocap_raw"; K = ["ang", "exc", "fru", "hap", "neu", "sad"]
        txt = {}
        for f in glob.glob(f"{RAW}/Session*/dialog/transcriptions/*.txt"):
            for line in open(f, errors="ignore"):
                mm = re.match(r"^(\S+) \[[\d.]+-[\d.]+\]: (.*)$", line.strip())
                if mm:
                    txt[mm.group(1)] = mm.group(2)
        al = json.load(open("runs/iemocap_align_20261018.json"))["utts"]
        norm = lambda s: " ".join(re.sub(r"[^a-z' ]", " ", s.lower()).split())
        sid = lambda k: re.sub(r"^Ses0\d[FM]_", "", k.split("|")[0])
        tl = collections.defaultdict(list)
        for k, u in al.items():
            if u["split"] == "train" and "script" in k and u["name"] in txt and len(norm(txt[u["name"]]).split()) >= 3:
                tl[(sid(k), norm(txt[u["name"]]))].append(u["label"])
        rep = {}
        for sp in ("dev", "test"):
            keys, y = D[sp]["keys"], D[sp]["y"]
            lk = np.array([K.index(collections.Counter(tl[(sid(k), norm(txt.get(al[k]["name"], "")))]).most_common(1)[0][0])
                           if "script" in k and tl.get((sid(k), norm(txt.get(al[k]["name"], "")))) else -1 for k in keys])
            m = (lk >= 0) & (lk != y)
            rep[sp] = {"n": int(m.sum())}
            for name in ("dual_online", "dual_online+e2"):
                rep[sp][name] = {"acc": r2(np.mean([100 * (p[m] == y[m]).mean() for p in P[sp][name]])),
                                 "says_train_label": r2(np.mean([100 * (p[m] == lk[m]).mean() for p in P[sp][name]]))}
            rep[sp]["e2_alone"] = {"acc": r2(np.mean([100 * (l.argmax(1)[m] == y[m]).mean() for l in D[sp]["E"]["e2"]])),
                                   "says_train_label": r2(np.mean([100 * (l.argmax(1)[m] == lk[m]).mean() for l in D[sp]["E"]["e2"]]))}
        o["repeated_lines"] = rep
    out["datasets"][ds] = o
out["provenance"] = prov
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
short = {ds: {"seeds": o["seeds"], "verdicts": {k: {kk: vv for kk, vv in x.items() if kk in ("mean", "n_pos", "ci95", "pass", "verdict", "a_share_per_seed", "a_pass", "b_pass", "alpha_dev")}
                                               for k, x in o["verdicts"].items()},
              "alpha_dev": {k: x for k, x in o["alpha_dev"].items() if "e2" in k},
              "test": {k: x["mean"] for k, x in o["table"]["test"].items()}} for ds, o in out["datasets"].items()}
print(json.dumps(short, indent=1))
