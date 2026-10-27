"""Layer weights of the WavLM Base+ speech expert on IEMOCAP (2026-10-27, CPU; descriptive, written after the lock
results). The expert combines the 13 hidden states of WavLM Base+ (index 0: the output of the convolutional encoder
after projection; 1-12: transformer layers) with softmax(head.lw), learned from zeros (uniform start, 1/13 each).
Reads head.lw from expert_bf16.pt of the five lock-batch E2 runs (lock_runs.json, iemocap/experts/e2), records the
sha256 of each checkpoint, and reports per seed and over seeds: the weight of every state in percent, the state with
the largest weight, and the shares of states 0-4, 5-8 and 9-12. MELD E2 checkpoints were not kept, so MELD is absent.
Usage (repo root): python <this> --out <json>"""
import argparse, hashlib, json
import torch
R = "outputs/runs"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
LK = json.load(open("runs/lock_20261022/lock_runs.json")); runs = LK["iemocap"]["experts"]["e2"]; seeds = LK["seeds"]
r2 = lambda x: round(float(x), 2)
per, sha = [], {}
for run in runs:
    p = f"{R}/{run}/expert_bf16.pt"; sha[run] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    sd = torch.load(p, map_location="cpu", weights_only=True)
    lw = sd["head.lw"].float(); assert lw.numel() == 13, lw.shape
    per.append(torch.softmax(lw, 0) * 100)
W = torch.stack(per)                                    # (5, 13), percent
groups = {"0-4": slice(0, 5), "5-8": slice(5, 9), "9-12": slice(9, 13)}
out = {"design": __doc__, "runs": runs, "seeds": seeds, "sha256": sha, "uniform_pct": r2(100 / 13),
       "per_seed_pct": [[r2(x) for x in w] for w in W], "mean_pct": [r2(x) for x in W.mean(0)],
       "sd_pct": [r2(x) for x in W.std(0)], "argmax_state_per_seed": [int(w.argmax()) for w in W],
       "group_share_pct": {g: {"mean": r2(W[:, s].sum(1).mean()), "per_seed": [r2(x) for x in W[:, s].sum(1)]} for g, s in groups.items()},
       "max_minus_min_pct_per_seed": [r2(w.max() - w.min()) for w in W]}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("design", "sha256")}, indent=1))
