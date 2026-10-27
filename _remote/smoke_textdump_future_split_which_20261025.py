"""2026-10-25: same as smoke_textdump_future_split_20261022.py (unchanged, 5a8017c0) plus --which {last,best}, for the
post-hoc MELD epoch-selection sensitivity (runs/锁定批_20261022/锁定批判定_2026-10-25.md §5, user chose option A).
--which best reads <tag>_lora_best (the dev-best epoch saved by run_v1.py). Reference check: last -> as before; best ->
<tag>_lora_test.json, which run_v1.py evaluated in the trained mode (dual -> full), so for a dual run with --mode online
there is no reference and the check is skipped (ref_weighted_f1 = null in run_info.json).

Docstring of smoke_textdump_future_split_20261022.py:
2026-10-22: same as smoke_textdump_future_20261021.py (unchanged, sha kept) plus --split {dev,test}; written for the
dev replication of the voice-vs-future check (runs/C15KD与闸门V结果_20261022/论文故事_听代替等_2026-10-22.md §4, lines
logged in charter/decisions.md before any dev dump existed). Outputs preds_<split>.jsonl / metrics_<split>.json. The
reference-metric check runs only for --split test --limit 0 (no last-epoch dev metric is saved per mode).

Original docstring:
2026-10-21: per-sample test scores of a text-only run trained with --future_mode (F / DM gates), last-epoch
adapter, in the mode given by --mode. Derived from smoke_textdump_20261018.py; the original is unchanged.
Reference metric: <tag>_lora_test_last.json for online/full runs, <tag>_lora_test_last_<mode>.json for dual runs.

Original docstring of smoke_textdump_20261018.py:
2026-10-18: per-sample test scores of a text-only LoRA run (run_v1.py), last-epoch adapter.

Needed for the A' proxy in the revised error analysis (runs/错误分析改方案_2026-10-18.md): which
errors the locked text-only models (R20261017-11..15, ctx 16) make too.

Same scoring path as run_v1.py's test eval with --fast_eval: load_model_and_tokenizer(for_training=True),
PeftModel on <tag>_lora_last, fast_scoring.candidate_avg_logps, infer._pred_from_avg_logps.
With --limit 0 the weighted F1 is checked against the run's own <tag>_lora_test_last.json; a gap
above 0.05 points exits with code 3.

The file is named _remote/smoke_* only so that the resident queue (_remote/qd.sh, n_train) counts it
as a GPU slot; it is not a smoke test.
"""
import argparse
import hashlib
import json
import os
import shlex
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
import infer  # noqa: E402
from dataset_spec import get_spec  # noqa: E402
from fast_scoring import candidate_avg_logps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="text-only run id, e.g. R20261017-11")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--mode", required=True, choices=["online", "full"])
ap.add_argument("--split", default="test", choices=["dev", "test"])
ap.add_argument("--which", default="last", choices=["last", "best"])
a = ap.parse_args()

src = f"outputs/runs/{a.src}"
toks = shlex.split(open(f"{src}/args.txt").read())
arg = {toks[i]: toks[i + 1] for i in range(len(toks) - 1) if toks[i].startswith("--") and not toks[i + 1].startswith("--")}
ds, tag, ctx = arg.get("--dataset", "iemocap"), arg["--tag"], int(arg["--ctx"]) if "--ctx" in arg else None
adapter = f"{src}/{tag}_lora_{a.which}"
fmode, fk = arg["--future_mode"], int(arg.get("--future_k", 2))
assert fmode == "dual" or fmode == a.mode, (fmode, a.mode)
if a.which == "last":
    ref = json.load(open(f"{src}/{tag}_lora_test_last_{a.mode}.json" if fmode == "dual" else f"{src}/{tag}_lora_test_last.json"))
else:
    ref = json.load(open(f"{src}/{tag}_lora_test.json")) if (fmode == a.mode or (fmode == "dual" and a.mode == "full")) else None
assert os.path.isdir(adapter), adapter

spec = get_spec(ds)
D = spec.get_data_module()
import future_ctx  # noqa: E402
test = future_ctx.with_mode(future_ctx.apply(D.load_split(a.split, context_window=ctx), spec, "dual", fk), a.mode)
if a.limit:
    test = test[: a.limit]

from peft import PeftModel  # noqa: E402

t0 = time.time()
base, tok, _ = infer.load_model_and_tokenizer(for_training=True)
model = PeftModel.from_pretrained(base, adapter)
model.eval()
os.makedirs(a.out_dir, exist_ok=True)
preds, golds = [], []
with open(os.path.join(a.out_dir, f"preds_{a.split}.jsonl"), "w") as f:
    for s in test:
        prompt_ids, cands = infer._build_candidate_token_ids(tok, s["messages"], spec=spec)
        assert [c[0] for c in cands] == list(spec.label_list)
        avg = candidate_avg_logps(model, prompt_ids, cands, model.device)
        pid = infer._pred_from_avg_logps(avg, spec, False)
        preds.append(pid)
        golds.append(s["label_id"])
        f.write(json.dumps({"dialogue_id": s["dialogue_id"], "utt_id": s["utt_id"], "speaker": s["speaker"],
                            "gold": s["label_id"], "scores": {"clean": [float(x) for x in avg]},
                            "pred": {"clean": pid}, "mode": a.mode, "n_future": s["n_future"]}) + "\n")
m = infer.compute_metrics(preds, golds, spec=spec)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
info = {"src": a.src, "args": vars(a), "src_args": arg, "adapter": adapter, "n": len(test),
        "label_list": list(spec.label_list), "weighted_f1": m["weighted_f1"],
        "which": a.which, "ref_weighted_f1": None if ref is None else ref["weighted_f1"], "elapsed_sec": round(time.time() - t0, 1),
        "sha256": {"script": sha(__file__), "adapter_model.safetensors": sha(f"{adapter}/adapter_model.safetensors")}}
json.dump(m, open(os.path.join(a.out_dir, f"metrics_{a.split}.json"), "w"), indent=1)
json.dump(info, open(os.path.join(a.out_dir, "run_info.json"), "w"), indent=1)
print(f"[textdump] {a.src} which={a.which} split={a.split} n={len(test)} wF1={100 * m['weighted_f1']:.2f} ref={None if ref is None else round(100 * ref['weighted_f1'], 2)}")
if ref is not None and a.split == "test" and not a.limit and abs(m["weighted_f1"] - ref["weighted_f1"]) > 0.0005:
    print("[textdump] MISMATCH with the run's own last-epoch test metric")
    sys.exit(3)
