"""2026-10-20: per-sample scores of a text-only LoRA run (run_v1.py) on dev or test, from the best-dev or the
last-epoch adapter. Extends _remote/smoke_textdump_20261018.py (test / last only) with --split and --which, for M1
(MELD complementarity check, runs/MELD无增益_对策与二区判断_2026-10-19.md §2): the locked MELD text-only runs use the
best-dev protocol (R20261015-03/07/11, R20261017-52/54, <tag>_lora_best).

Scoring path unchanged from the 10-18 script: load_model_and_tokenizer(for_training=True), PeftModel on the adapter,
fast_scoring.candidate_avg_logps, infer._pred_from_avg_logps. The source runs were evaluated without --fast_eval, so
the reference check allows a small gap: |dump wF1 - reference| <= --tol points (default 0.25), else exit code 3.
Reference: test -> <tag>_lora_test.json (best) or <tag>_lora_test_last.json (last); dev -> the run's dev curve
(<tag>_dev_curve.json), its maximum (best) or last entry (last).

Named _remote/smoke_* only so that the resident queue counts it as a GPU slot; it is not a smoke test.
"""
import argparse
import hashlib
import json
import os
import shlex
import sys
import time

sys.path.insert(0, os.getcwd())
import infer  # noqa: E402
from dataset_spec import get_spec  # noqa: E402
from fast_scoring import candidate_avg_logps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="text-only run id, e.g. R20261015-03")
ap.add_argument("--split", required=True, choices=["dev", "test"])
ap.add_argument("--which", required=True, choices=["best", "last"])
ap.add_argument("--out_dir", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tol", type=float, default=0.25, help="allowed |wF1 gap| to the reference, in points")
a = ap.parse_args()

src = f"outputs/runs/{a.src}"
toks = shlex.split(open(f"{src}/args.txt").read())
arg = {toks[i]: toks[i + 1] for i in range(len(toks) - 1) if toks[i].startswith("--") and not toks[i + 1].startswith("--")}
ds, tag, ctx = arg.get("--dataset", "iemocap"), arg["--tag"], int(arg["--ctx"]) if "--ctx" in arg else None
adapter = f"{src}/{tag}_lora_{a.which}"
assert os.path.isdir(adapter), adapter
if a.split == "test":
    ref = 100 * json.load(open(f"{src}/{tag}_lora_test{'' if a.which == 'best' else '_last'}.json"))["weighted_f1"]
else:
    curve = [e["weighted_f1"] for e in json.load(open(f"{src}/{tag}_dev_curve.json"))]
    ref = 100 * (max(curve) if a.which == "best" else curve[-1])

spec = get_spec(ds)
D = spec.get_data_module()
data = D.load_split(a.split, context_window=ctx)
if a.limit:
    data = data[: a.limit]

from peft import PeftModel  # noqa: E402

t0 = time.time()
base, tok, _ = infer.load_model_and_tokenizer(for_training=True)
model = PeftModel.from_pretrained(base, adapter)
model.eval()
if os.path.exists(a.out_dir) and os.listdir(a.out_dir):
    raise SystemExit(f"out_dir exists and is not empty: {a.out_dir}")
os.makedirs(a.out_dir, exist_ok=True)
preds, golds = [], []
with open(os.path.join(a.out_dir, f"preds_{a.split}.jsonl"), "w") as f:
    for s in data:
        prompt_ids, cands = infer._build_candidate_token_ids(tok, s["messages"], spec=spec)
        assert [c[0] for c in cands] == list(spec.label_list)
        avg = candidate_avg_logps(model, prompt_ids, cands, model.device)
        pid = infer._pred_from_avg_logps(avg, spec, False)
        preds.append(pid)
        golds.append(s["label_id"])
        f.write(json.dumps({"dialogue_id": s["dialogue_id"], "utt_id": s["utt_id"], "clip_id": s.get("clip_id"),
                            "speaker": s["speaker"], "gold": s["label_id"],
                            "scores": {"clean": [float(x) for x in avg]}, "pred": {"clean": pid}}) + "\n")
m = infer.compute_metrics(preds, golds, spec=spec)
w = 100 * m["weighted_f1"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
info = {"src": a.src, "args": vars(a), "src_args": arg, "adapter": adapter, "n": len(data),
        "label_list": list(spec.label_list), "weighted_f1": round(w, 4), "ref_weighted_f1": round(ref, 4),
        "gap_points": round(w - ref, 4), "elapsed_sec": round(time.time() - t0, 1),
        "sha256": {"script": sha(__file__), "adapter_model.safetensors": sha(f"{adapter}/adapter_model.safetensors")}}
json.dump(m, open(os.path.join(a.out_dir, f"metrics_{a.split}.json"), "w"), indent=1,
          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
json.dump(info, open(os.path.join(a.out_dir, "run_info.json"), "w"), indent=1)
print(f"[textdump] {a.src} {a.split}/{a.which} n={len(data)} wF1={w:.2f} ref={ref:.2f} gap={w - ref:+.2f}")
if not a.limit and abs(w - ref) > a.tol:
    print("[textdump] MISMATCH with the run's own metric beyond --tol")
    sys.exit(3)
