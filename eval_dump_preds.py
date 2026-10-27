"""R0-1 (2026-10-14): per-sample candidate scores under clean / drop / zero.

Adds per-sample output without touching any existing evaluation path.
Loading mirrors eval_pilot3_test.py; candidate construction and scoring are
copied line for line from run_pilot3_dialogue_visual.score_candidates.

Conditions
  clean : original modality features
  drop  : projected modality tokens set to zero -- the exact state produced by
          training-time modality dropout (compute_visual_tokens returns zeros)
  zero  : raw modality features set to zero (the paper's all-zero protocol)
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

import backbone
import config as C
from dataset_spec import get_spec
from infer import load_model_and_tokenizer, compute_metrics, _build_candidate_token_ids
from run_v1 import seed_all, setup_lora
from run_pilot3_dialogue_visual import (
    DialogueVisualAdapter, load_modality_feats, install_hooks, get_dialogue_feats,
    N_FRAMES_VISUAL, N_FRAMES_AUDIO,
)

CONDITIONS = ("clean", "drop", "zero")
FAST = False  # --fast_eval; see fast_scoring.py

# --- "drop" condition: reproduce training-time modality dropout at eval -------
_orig_compute = DialogueVisualAdapter.compute_visual_tokens


def _compute_with_drop(self):
    if getattr(self, "_force_drop", False) and self._raw_turn_feats is not None:
        B = self._raw_turn_feats[0].shape[0]
        n_total_frames = sum(f.shape[1] for f in self._raw_turn_feats)
        dim = self.proj.out_features
        return torch.zeros(B, n_total_frames, dim,
                           device=self._raw_turn_feats[0].device)
    return _orig_compute(self)


DialogueVisualAdapter.compute_visual_tokens = _compute_with_drop


@torch.no_grad()
def candidate_logps(model, adapter, tokenizer, messages, turn_feats_list, device, spec):
    """Copied from run_pilot3_dialogue_visual.score_candidates; returns all scores."""
    prompt_ids, cands = _build_candidate_token_ids(tokenizer, messages, spec=spec)

    vis_list = [torch.tensor(f[None], device=device, dtype=torch.float32)
                for f in turn_feats_list]
    adapter.set_visual(vis_list)

    if FAST:
        from fast_scoring import candidate_avg_logps
        return candidate_avg_logps(
            model, prompt_ids, cands, device,
            on_batch=lambda n: adapter.set_visual([v.expand(n, -1, -1) for v in vis_list]))

    avg_logps = []
    for lab, ans in cands:
        seq = prompt_ids + ans
        input_ids = torch.tensor([seq], device=device)
        attn_mask = torch.ones_like(input_ids)
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits[0]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        total = sum(logprobs[len(prompt_ids) + j - 1, tok_id].item()
                    for j, tok_id in enumerate(ans))
        avg_logps.append(total / len(ans))
    return avg_logps


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True, choices=["meld", "iemocap"])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--n_blocks", type=int, default=2)
    ap.add_argument("--n_prev_turns", type=int, required=True)
    ap.add_argument("--no_frame_attn", action="store_true")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N samples")
    ap.add_argument("--fast_eval", action="store_true")
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--meld_audio", action="store_true")
    ap.add_argument("--cue_file", default=None)
    args = ap.parse_args()
    global FAST
    FAST = args.fast_eval

    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise SystemExit(f"out_dir exists and is not empty: {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    seed_all(C.SEED)
    spec = get_spec(args.dataset)
    D = spec.get_data_module()
    modality = "audio" if (args.dataset == "iemocap" or args.meld_audio) else "visual"
    n_frames = N_FRAMES_AUDIO if modality == "audio" else N_FRAMES_VISUAL
    insert_layers = {2: [9, 18], 4: [4, 9, 14, 18]}[args.n_blocks]

    model, tokenizer, _ = load_model_and_tokenizer(for_training=True)
    model = setup_lora(model)
    model.load_adapter(args.checkpoint, adapter_name="default")
    device = model.device
    adapter = DialogueVisualAdapter(
        n_blocks=len(insert_layers), dim=backbone.hidden_size(model),
        n_prev_turns=args.n_prev_turns,
        use_frame_attn=not args.no_frame_attn,
    ).to(device).float()
    adapter.load_state_dict(torch.load(os.path.join(args.checkpoint, "visual_adapter.pt"),
                                       weights_only=False))
    handles = install_hooks(model, adapter, insert_layers)
    model.eval(); adapter.eval()

    if args.cue_file:
        import cue_text  # H2 (2026-10-18): verbalised acoustic cue appended to the target utterance
        D.load_split = cue_text.wrap_load_split(D.load_split, args.cue_file)
    data = D.load_split(args.split, context_window=args.ctx)
    if args.limit:
        data = data[:args.limit]
    feat_map = load_modality_feats(spec, args.split, modality)

    preds = {c: [] for c in args.conditions}
    golds = []
    t0 = time.time()
    with open(os.path.join(args.out_dir, f"preds_{args.split}.jsonl"), "w") as fo:
        for s in tqdm(data, desc=f"{args.dataset}/{args.split}"):
            feats = get_dialogue_feats(s, feat_map, args.n_prev_turns, n_frames)
            rec = {k: s.get(k) for k in ("dialogue_id", "utt_id", "clip_id", "speaker")}
            rec["gold"] = int(s["label_id"])
            rec["scores"], rec["pred"] = {}, {}
            for cond in args.conditions:
                adapter._force_drop = (cond == "drop")
                fl = [np.zeros_like(f) for f in feats] if cond == "zero" else feats
                sc = candidate_logps(model, adapter, tokenizer, s["messages"], fl, device, spec)
                pred = spec.label2id[spec.label_list[int(np.argmax(sc))]]
                rec["scores"][cond] = sc
                rec["pred"][cond] = pred
                preds[cond].append(pred)
            adapter._force_drop = False
            golds.append(rec["gold"])
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elapsed = time.time() - t0

    metrics = {c: compute_metrics(preds[c], golds, spec=spec) for c in args.conditions}
    with open(os.path.join(args.out_dir, f"metrics_{args.split}.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    info = {
        "args": vars(args), "n": len(golds), "elapsed_sec": round(elapsed, 1),
        "sec_per_1k_samples": round(1000 * elapsed / max(1, len(golds)), 1),
        "label_list": list(spec.label_list),
        "sha256": {
            "eval_dump_preds.py": sha256_of(__file__),
            "run_pilot3_dialogue_visual.py": sha256_of("run_pilot3_dialogue_visual.py"),
            "adapter_model.safetensors": sha256_of(os.path.join(args.checkpoint, "adapter_model.safetensors")),
            "visual_adapter.pt": sha256_of(os.path.join(args.checkpoint, "visual_adapter.pt")),
        },
    }
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    for c in args.conditions:
        print(f"[{c}] wF1={metrics[c]['weighted_f1']*100:.2f} acc={metrics[c]['accuracy']*100:.2f} n={metrics[c]['n']}")
    print(f"elapsed {elapsed:.0f}s")
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
