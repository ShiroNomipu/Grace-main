"""Completeness item (2026-10-27; listed in runs/锁定批_20261022/完整性_CPU部分_2026-10-25.md §6, user approved queuing
the remaining GPU items): per-utterance inference latency of the final system and of the injection baseline, for the
real-time setting. Derived from _remote/smoke_efficiency_20261020.py (777f8f80, unchanged). Evaluation only; named
_remote/smoke_* only so that the resident queue counts it as a GPU slot.

Parent mode (no --config): waits until no other process holds the GPU (up to --wait_h hours; the snapshot is recorded
either way), then runs each configuration in its own subprocess, one after another, and writes out_dir/efficiency.json.
Configurations (seed 123 checkpoints; backbone 4-bit NF4; batch 1):
  dual_online_iemocap  C15 dual LoRA R20261021-102 _lora_last, online prompt (future turns "(not available)"), ctx 16
  dual_full_iemocap    same adapter, 2 future turns
  e2_iemocap           E2 R20261021-21 expert_bf16.pt (WavLM Base+ fine-tuned), raw 16 kHz wav of the target (<= 15 s),
                       bf16 autocast, model code from _remote/smoke_g1expert_20261021.py (the script that trained it)
  grace_iemocap        B1 fixed GRACE R20261022-306 _last, hooks on, ctx 16; emotion2vec features precomputed
  dual_online_meld     MELD dual LoRA R20261022-121 _lora_last, online prompt
  dual_full_meld       same adapter, 2 future turns
Utterances: round(linspace(0, n_test - 1, --n_utts)) over the test list of each dataset (dual-model order); every
configuration of a dataset times the same utterances, matched by key dialogue_id|utt_id.
Per utterance: one full prediction (text and GRACE: fast candidate scoring of all labels; E2: wav array -> log-probabilities,
incl. host-to-device copy, excl. disk read), 1 warm-up then --iters timed, cuda synchronize around each.
Reported per configuration: mean / median over utterances of the per-utterance mean ms, mean prompt tokens, mean audio
seconds (e2), torch.cuda.max_memory_allocated (includes the weights), sha256 of the checkpoint file.
Derived (parent, IEMOCAP, per utterance then averaged): online + E2 run one after the other (sum) and side by side (max).
Not included: emotion2vec feature extraction for GRACE (no funasr; no package is installed for this), wav decoding.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())

CONFIGS = {
    "dual_online_iemocap": ("iemocap", "R20261021-102", "online"),
    "dual_full_iemocap": ("iemocap", "R20261021-102", "full"),
    "e2_iemocap": ("iemocap", "R20261021-21", "e2"),
    "grace_iemocap": ("iemocap", "R20261022-306", "grace"),
    "dual_online_meld": ("meld", "R20261022-121", "online"),
    "dual_full_meld": ("meld", "R20261022-121", "full"),
}
FUTURE_K, CTX = 2, 16


def gpu_apps():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return [l for l in out.splitlines() if l.strip()]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def src_args(rid):
    toks = shlex.split(open(f"outputs/runs/{rid}/args.txt").read())
    return {toks[i]: toks[i + 1] for i in range(len(toks) - 1) if toks[i].startswith("--") and not toks[i + 1].startswith("--")}


def child(a):
    import numpy as np
    import torch
    import config as C
    import future_ctx
    from dataset_spec import get_spec
    from infer import load_model_and_tokenizer, _build_candidate_token_ids
    from run_v1 import seed_all
    ds, rid, kind = CONFIGS[a.config]
    seed_all(C.SEED)
    spec = get_spec(ds)
    D = spec.get_data_module()
    key = lambda s: f"{s['dialogue_id']}|{s['utt_id']}"
    order = [key(s) for s in D.load_split("test", context_window=CTX)]
    keys = [order[int(round(x))] for x in np.linspace(0, len(order) - 1, a.n_utts)]
    dev = torch.device("cuda")
    if kind in ("online", "full"):
        from fast_scoring import candidate_avg_logps
        from peft import PeftModel
        arg = src_args(rid)
        assert arg["--future_mode"] == "dual" and int(arg["--ctx"]) == CTX and int(arg.get("--future_k", 2)) == FUTURE_K, arg
        ckpt = f"outputs/runs/{rid}/{arg['--tag']}_lora_last"
        data = future_ctx.with_mode(future_ctx.apply(D.load_split("test", context_window=CTX), spec, "dual", FUTURE_K), kind)
        by = {key(s): s for s in data}
        base, tok, _ = load_model_and_tokenizer(for_training=True)
        model = PeftModel.from_pretrained(base, ckpt)
        model.eval()
        ck_file = f"{ckpt}/adapter_model.safetensors"

        def make(k):
            s = by[k]
            prompt, cands = _build_candidate_token_ids(tok, s["messages"], spec=spec)
            return (lambda: candidate_avg_logps(model, prompt, cands, model.device)), {"prompt_tokens": len(prompt), "n_future": s["n_future"]}
    elif kind == "grace":
        import backbone
        import eval_dump_preds as E
        from run_v1 import setup_lora
        from run_pilot3_dialogue_visual import (DialogueVisualAdapter, load_modality_feats, install_hooks,
                                                get_dialogue_feats, N_FRAMES_AUDIO)
        arg = src_args(rid)
        assert int(arg["--ctx"]) == CTX and arg["--n_blocks"] == "2" and arg["--n_prev_turns"] == "1", arg
        ckpt = glob.glob(f"outputs/runs/{rid}/pilot3_*_last")
        assert len(ckpt) == 1, ckpt
        ckpt = ckpt[0]
        model, tok, _ = load_model_and_tokenizer(for_training=True)
        model = setup_lora(model)
        model.load_adapter(ckpt, adapter_name="default")
        adapter = DialogueVisualAdapter(n_blocks=2, dim=backbone.hidden_size(model), n_prev_turns=1,
                                        use_frame_attn=False).to(model.device).float()
        adapter.load_state_dict(torch.load(os.path.join(ckpt, "visual_adapter.pt"), weights_only=False))
        install_hooks(model, adapter, [9, 18])
        model.eval(); adapter.eval()
        E.FAST = True
        by = {key(s): s for s in D.load_split("test", context_window=CTX)}
        feat_map = load_modality_feats(spec, "test", "audio")
        ck_file = os.path.join(ckpt, "visual_adapter.pt")

        def make(k):
            s = by[k]
            f = get_dialogue_feats(s, feat_map, 1, N_FRAMES_AUDIO)
            prompt, _ = _build_candidate_token_ids(tok, s["messages"], spec=spec)
            return (lambda: E.candidate_logps(model, adapter, tok, s["messages"], f, model.device, spec)), {"prompt_tokens": len(prompt)}
    else:  # e2
        assert open(f"outputs/runs/{rid}/script.txt").read().strip() == "_remote/smoke_g1expert_20261021.py"
        m = importlib.util.spec_from_file_location("g1", "_remote/smoke_g1expert_20261021.py")
        G = importlib.util.module_from_spec(m)
        m.loader.exec_module(G)
        align = json.load(open(G.ALIGN))["utts"]
        items = {it["key"]: it for it in G.load_split_items(spec, D, "test", "e2", align, None)}
        model = G.Expert("e2", spec.num_classes)
        ck_file = f"outputs/runs/{rid}/expert_bf16.pt"
        model.load_state_dict(torch.load(ck_file, weights_only=True))
        model.to(dev).eval()

        def make(k):
            it = items[k]
            w, _ = G.read_wav(it["wav"])
            cache = {k: w}

            def fn():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(G.make_batch([it], "e2", cache, dev))
                return torch.log_softmax(logits.float(), -1).cpu()
            return fn, {"audio_sec": round(len(w) / G.SR, 3)}

    per = []
    sync = torch.cuda.synchronize
    with torch.no_grad():
        for k in keys:
            fn, info = make(k)
            fn()
            sync()
            ts = []
            for _ in range(a.iters):
                t = time.perf_counter(); fn(); sync(); ts.append(1000 * (time.perf_counter() - t))
            per.append({"key": k, **info, "predict_ms": round(float(np.mean(ts)), 3)})
    ms = [r["predict_ms"] for r in per]
    res = {"config": a.config, "run": rid, "kind": kind, "checkpoint_file": ck_file, "checkpoint_sha256": sha(ck_file),
           "n_utts": len(per), "iters": a.iters, "predict_ms_mean": round(float(np.mean(ms)), 2),
           "predict_ms_median": round(float(np.median(ms)), 2),
           "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
           "gpu": torch.cuda.get_device_name(0), "per_utt": per}
    for f in ("prompt_tokens", "audio_sec"):
        if f in per[0]:
            res[f"{f}_mean"] = round(float(np.mean([r[f] for r in per])), 2)
    print("RESULT " + json.dumps(res), flush=True)


def parent(a):
    import numpy as np
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    while True:
        apps = gpu_apps()
        if not apps or time.time() - t0 > a.wait_h * 3600:
            break
        time.sleep(60)
    import torch
    import transformers
    out = {"design": __doc__, "script_sha256": sha(os.path.abspath(__file__)), "waited_min": round((time.time() - t0) / 60, 1),
           "gpu_apps_at_start": apps, "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
           "args": vars(a), "results": {}, "stderr_tail": {}, "gpu_apps_after": {}}
    for cfg in a.configs:
        cmd = [sys.executable, os.path.abspath(__file__), "--config", cfg, "--n_utts", str(a.n_utts), "--iters", str(a.iters)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        out["results"][cfg] = json.loads(line[-1][7:]) if line else {"error": p.returncode}
        out["stderr_tail"][cfg] = p.stderr[-1500:]
        out["gpu_apps_after"][cfg] = gpu_apps()
        r = out["results"][cfg]
        print(cfg, {k: v for k, v in r.items() if k != "per_utt"}, flush=True)
    R = out["results"]
    if "error" not in R.get("dual_online_iemocap", {"error": 1}) and "error" not in R.get("e2_iemocap", {"error": 1}):
        t = {r["key"]: r["predict_ms"] for r in R["dual_online_iemocap"]["per_utt"]}
        e = {r["key"]: r["predict_ms"] for r in R["e2_iemocap"]["per_utt"]}
        assert set(t) == set(e)
        out["derived_iemocap"] = {"online+e2_sequential_ms_mean": round(float(np.mean([t[k] + e[k] for k in t])), 2),
                                  "online+e2_parallel_ms_mean": round(float(np.mean([max(t[k], e[k]) for k in t])), 2)}
        print(out["derived_iemocap"], flush=True)
    json.dump(out, open(os.path.join(a.out_dir, "efficiency.json"), "w"), indent=1)
    sys.exit(0 if all("error" not in r for r in R.values()) else 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--out_dir")
    ap.add_argument("--n_utts", type=int, default=20)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--wait_h", type=float, default=6)
    a = ap.parse_args()
    if a.config:
        child(a)
    else:
        assert a.out_dir, "--out_dir needed in parent mode"
        parent(a)
