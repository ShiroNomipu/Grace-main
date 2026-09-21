"""
Robustness Matrix Evaluation: test modality robustness.

Conditions:
  - clean: normal features (should reproduce reported test wF1)
  - noise_0.5: Gaussian noise σ=0.5 added to features
  - noise_1.0: Gaussian noise σ=1.0 added to features
  - zero: all features zeroed out (simulates modality missing)

Supports both MELD (visual) and IEMOCAP (audio).
Compares models trained WITH vs WITHOUT modality dropout to demonstrate
graceful degradation — the core contribution of "Modality-Robust Training".
"""
import argparse
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config as C
from dataset_spec import get_spec
from infer import (load_model_and_tokenizer, _build_candidate_token_ids,
                   compute_metrics, print_metrics)
from run_v1 import seed_all, setup_lora

import backbone

# Defaults for Qwen2-VL-2B; both are re-read from the backbone config in main()
# so a checkpoint trained on another backbone evaluates without an edit here.
HIDDEN_DIM = 1536
INSERT_LAYERS = [9, 18]
FEAT_DIM = 768
N_FRAMES_VISUAL = 8
N_FRAMES_AUDIO = 64
GATE_INIT = 0.1
MAX_TURNS = 4

# Block-level "true missing" operationalization, read by the forward hook.
#   None      : normal injection (feature-level conditions: clean / noise_* / zero)
#   "bypass"  : adapter block skipped entirely (residual stream untouched -> true absent)
#   "attnmask": literal-zero KV so cross-attn output == 0 (attn branch severed,
#               bypassing the projection bias that "zero" would still inject); FF branch active
_MISSING = {"mode": None}
_BLOCK_LEVEL_CONDS = ("bypass", "attnmask")


# ─── Model architectures (must match training) ───

class GatedCrossAttnBlock(nn.Module):
    def __init__(self, dim=HIDDEN_DIM, n_heads=12, ff_mult=2, gate_init=GATE_INIT,
                 no_gating=False, direction_decomp=False, pure_orthogonal=False,
                 emotion_anchor=False, emo_k=0, gate_mode="tanh"):
        super().__init__()
        self.no_gating = no_gating
        # Mirrors run_pilot3_dialogue_visual.GatedCrossAttnBlock; the amplitude-matched
        # controls (tab:gateform) were trained under "linear" and "fixed".
        self.gate_mode = gate_mode
        self.direction_decomp = direction_decomp
        self.pure_orthogonal = pure_orthogonal
        self.emotion_anchor = emotion_anchor
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.gate_attn = nn.Parameter(torch.full((1,), gate_init))
        if direction_decomp:
            self.gate_par = nn.Parameter(torch.full((1,), gate_init))
        if emotion_anchor:
            self.register_buffer("emo_Q", torch.zeros(dim, emo_k))
        self.norm2 = nn.LayerNorm(dim)
        ff_dim = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))
        self.gate_ff = nn.Parameter(torch.full((1,), gate_init))

    def _gate(self, raw):
        if self.no_gating:
            return 1.0
        if self.gate_mode == "tanh":
            return torch.tanh(raw)
        return raw  # "linear" (learnable, unbounded) or "fixed" (constant)

    def forward(self, text_hidden, visual_kv):
        normed = self.norm1(text_hidden.float())
        attn_out, _ = self.cross_attn(normed, visual_kv, visual_kv)
        if self.direction_decomp:
            h = text_hidden.float()
            f = attn_out
            h_sq = (h * h).sum(-1, keepdim=True).clamp(min=1e-8)
            f_par = ((f * h).sum(-1, keepdim=True) / h_sq) * h
            f_perp = f - f_par
            if self.emotion_anchor:
                Q = self.emo_Q.to(f_perp.dtype)
                f_anchor = (f_perp @ Q) @ Q.t()
                fa_par = ((f_anchor * h).sum(-1, keepdim=True) / h_sq) * h
                f_perp = f_anchor - fa_par
            inj = torch.tanh(self.gate_attn) * f_perp
            if not self.pure_orthogonal:
                inj = inj + torch.tanh(self.gate_par) * f_par
            text_hidden = text_hidden + inj.to(text_hidden.dtype)
        else:
            g_attn = self._gate(self.gate_attn)
            text_hidden = text_hidden + g_attn * attn_out.to(text_hidden.dtype)
        normed2 = self.norm2(text_hidden.float())
        ff_out = self.ff(normed2)
        g_ff = self._gate(self.gate_ff)
        text_hidden = text_hidden + g_ff * ff_out.to(text_hidden.dtype)
        return text_hidden


class VisualAdapter(nn.Module):
    """Pilot-1 architecture: simple projection + cross-attn blocks."""

    def __init__(self, n_blocks=2, dim=HIDDEN_DIM, feat_dim=FEAT_DIM):
        super().__init__()
        self.proj = nn.Linear(feat_dim, dim)
        self.proj_ln = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([GatedCrossAttnBlock(dim) for _ in range(n_blocks)])
        self._raw_feats = None

    def set_visual(self, raw_feats):
        self._raw_feats = raw_feats.detach()

    def compute_visual_tokens(self):
        if self._raw_feats is None:
            return None
        projected = self.proj(self._raw_feats.float())
        return self.proj_ln(projected)

    def get_block(self, idx):
        return self.blocks[idx]


class FrameAttention(nn.Module):
    def __init__(self, dim=FEAT_DIM, hidden_div=12, dropout=0.3, temperature=2.0,
                 frame_shuffle_prob=0.3):
        super().__init__()
        hidden = max(dim // hidden_div, 32)
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
        self.temperature = temperature
        self.frame_shuffle_prob = frame_shuffle_prob

    def forward(self, frames):
        if self.training and self.frame_shuffle_prob > 0:
            if random.random() < self.frame_shuffle_prob:
                perm = torch.randperm(frames.shape[1], device=frames.device)
                frames = frames[:, perm, :]
        scores = self.scorer(frames)
        weights = torch.softmax(scores / self.temperature, dim=1)
        n_frames = frames.shape[1]
        return frames * weights * n_frames


class DialogueVisualAdapter(nn.Module):
    """Step3 architecture: same as Pilot3 with mdrop support."""

    def __init__(self, n_blocks=2, dim=HIDDEN_DIM, feat_dim=FEAT_DIM,
                 max_turns=MAX_TURNS, n_prev_turns=0,
                 use_frame_attn=False, dialogue_dropout=0.0,
                 modality_dropout=0.0, no_gating=False, gate_init=GATE_INIT,
                 direction_decomp=False, pure_orthogonal=False,
                 emotion_anchor=False, emo_k=0, gate_mode="tanh"):
        super().__init__()
        self.n_prev_turns = n_prev_turns
        self.max_turns = max_turns
        self.dialogue_dropout = dialogue_dropout
        self.modality_dropout = modality_dropout
        self.frame_attn = FrameAttention(feat_dim) if use_frame_attn else None
        self.proj = nn.Linear(feat_dim, dim)
        self.proj_ln = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([GatedCrossAttnBlock(dim, gate_init=gate_init,
                                                         no_gating=no_gating,
                                                         direction_decomp=direction_decomp,
                                                         pure_orthogonal=pure_orthogonal,
                                                         emotion_anchor=emotion_anchor,
                                                         emo_k=emo_k,
                                                         gate_mode=gate_mode)
                                     for _ in range(n_blocks)])
        self.turn_pos_embed = nn.Embedding(max_turns, dim)
        self._raw_turn_feats = None

    def set_visual(self, turn_feats_list):
        self._raw_turn_feats = [f.detach() for f in turn_feats_list]

    def compute_visual_tokens(self):
        if self._raw_turn_feats is None:
            return None
        all_tokens = []
        n_turns = len(self._raw_turn_feats)
        for turn_idx, feats in enumerate(self._raw_turn_feats):
            f = feats.float()
            if self.frame_attn is not None:
                f = self.frame_attn(f)
            projected = self.proj_ln(self.proj(f))
            pos_offset = self.max_turns - n_turns + turn_idx
            pos_emb = self.turn_pos_embed.weight[pos_offset]
            projected = projected + pos_emb
            all_tokens.append(projected)
        return torch.cat(all_tokens, dim=1)

    def get_block(self, idx):
        return self.blocks[idx]


# ─── Feature loading ───

def load_modality_feats(spec, split, modality="visual"):
    """Load frame-level features for the given modality."""
    if modality == "visual":
        path = os.path.join(spec.video_frame_dir,
                            spec.video_frame_pt_tmpl.format(split=split))
        expected_frames = N_FRAMES_VISUAL
    else:
        path = os.path.join(spec.audio_frame_dir,
                            spec.audio_frame_pt_tmpl.format(split=split))
        expected_frames = N_FRAMES_AUDIO
    data = torch.load(path, weights_only=False)
    feats, keys = data["feats"], data["order_keys"]
    assert feats.shape[1] == expected_frames and feats.shape[2] == FEAT_DIM, \
        f"Expected ({expected_frames}, {FEAT_DIM}), got {feats.shape[1:]}"
    return {k: feats[i] for i, k in enumerate(keys)}


# ─── Perturbation functions ───

def perturb_feats(feat, condition):
    """Apply perturbation to a single sample's modality features.

    Args:
        feat: numpy array (N_FRAMES, FEAT_DIM)
        condition: one of "clean", "noise_0.5", "noise_1.0", "zero"
    Returns:
        perturbed numpy array (same shape)
    """
    if condition == "clean" or condition in _BLOCK_LEVEL_CONDS:
        # bypass / attnmask are applied at the forward hook, not on features.
        return feat
    elif condition == "zero":
        return np.zeros_like(feat)
    elif condition.startswith("noise_"):
        sigma = float(condition.split("_")[1])
        noise = np.random.randn(*feat.shape).astype(np.float32) * sigma
        return feat + noise
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ─── Hook management ───

def install_hooks(model, visual_adapter, insert_layers):
    layers = backbone.decoder_layers(model)
    handles = []
    for block_idx, layer_idx in enumerate(insert_layers):
        block = visual_adapter.get_block(block_idx)

        def make_hook(blk):
            def hook_fn(module, input, output):
                # Qwen2VLDecoderLayer returns a tuple; LlamaDecoderLayer in
                # transformers >= 4.54 returns the hidden states bare. Hand back
                # whichever shape was given.
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                # True-missing: bypass the whole adapter block (residual untouched).
                if _MISSING["mode"] == "bypass":
                    return output
                v_tokens = visual_adapter.compute_visual_tokens()
                if v_tokens is None:
                    return output
                # True-missing: sever the cross-attn pathway with literal-zero KV
                # (attn_out == 0), avoiding the projection-bias signal that the
                # feature-level "zero" condition would still inject.
                if _MISSING["mode"] == "attnmask":
                    v_tokens = torch.zeros_like(v_tokens)
                modified = blk(hidden, v_tokens).to(hidden.dtype)
                return (modified,) + output[1:] if is_tuple else modified
            return hook_fn

        h = layers[layer_idx].register_forward_hook(make_hook(block))
        handles.append(h)
    return handles


# ─── Evaluation ───

@torch.no_grad()
def score_candidates(model, visual_adapter, tokenizer, messages, turn_feats_list,
                     device, spec, adapter_type="pilot1"):
    """Score each candidate label. turn_feats_list is a list of np arrays for multi-turn."""
    from infer import _build_candidate_token_ids
    prompt_ids, cands = _build_candidate_token_ids(tokenizer, messages, spec=spec)

    if adapter_type == "pilot1":
        v = torch.tensor(turn_feats_list[-1][None], device=device, dtype=torch.float32)
        visual_adapter.set_visual(v)
    else:
        vis_list = [torch.tensor(f[None], device=device, dtype=torch.float32)
                    for f in turn_feats_list]
        visual_adapter.set_visual(vis_list)

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

    pred_idx = int(np.argmax(avg_logps))
    return spec.label2id[spec.label_list[pred_idx]]


# Mirrors the training flag of the same name: a checkpoint trained with only the
# target utterance's features injected has to be scored the same way, or the
# context slot it never saw filled arrives full of real signal at test time.
MODALITY_TARGET_ONLY = False


def get_dialogue_feats(sample, feat_map, n_prev=1, n_frames=N_FRAMES_VISUAL):
    """Get modality features for target + N previous turns (matches training)."""
    feat_keys = sample.get("context_feat_keys") or sample.get("context_clip_ids")
    if feat_keys is None:
        key = sample.get("clip_id") or f"{sample['dialogue_id']}|{sample['utt_id']}"
        feat_keys = [key]

    if MODALITY_TARGET_ONLY:
        feat_keys = feat_keys[-1:]

    target_idx = len(feat_keys) - 1
    start_idx = max(0, target_idx - n_prev)

    result = []
    for i in range(start_idx, target_idx + 1):
        key = feat_keys[i]
        if key in feat_map:
            f = feat_map[key]
            result.append(f.numpy() if isinstance(f, torch.Tensor) else f)
        else:
            result.append(np.zeros((n_frames, FEAT_DIM), dtype=np.float32))

    while len(result) < n_prev + 1:
        result.insert(0, np.zeros((n_frames, FEAT_DIM), dtype=np.float32))

    return result


def evaluate_condition(model, adapter, tokenizer, samples, feat_map,
                       spec, condition, adapter_type="pilot1", n_frames=N_FRAMES_VISUAL):
    """Evaluate under a specific modality perturbation condition."""
    device = model.device
    n_prev = adapter.n_prev_turns if hasattr(adapter, 'n_prev_turns') else 0
    model.eval()
    adapter.eval()
    _MISSING["mode"] = condition if condition in _BLOCK_LEVEL_CONDS else None
    preds, golds = [], []

    for s in tqdm(samples, desc=f"eval[{condition}]"):
        if adapter_type == "pilot1":
            key = spec.feat_key(s)
            if key in feat_map:
                feat = feat_map[key].numpy() if isinstance(feat_map[key], torch.Tensor) else feat_map[key]
            else:
                feat = np.zeros((n_frames, FEAT_DIM), dtype=np.float32)
            turn_feats = [perturb_feats(feat, condition)]
        else:
            feats_list = get_dialogue_feats(s, feat_map, n_prev, n_frames)
            turn_feats = [perturb_feats(f, condition) for f in feats_list]

        preds.append(score_candidates(
            model, adapter, tokenizer, s["messages"],
            turn_feats, device, spec, adapter_type))
        golds.append(s["label_id"])

    return compute_metrics(preds, golds, spec=spec)


def main():
    parser = argparse.ArgumentParser(description="Robustness Matrix Evaluation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best checkpoint directory")
    parser.add_argument("--adapter_type", choices=["pilot1", "step3"], required=True,
                        help="Which adapter architecture to use")
    parser.add_argument("--dataset", default="meld", choices=["iemocap", "meld"])
    parser.add_argument("--conditions", nargs="+",
                        default=["clean", "noise_0.5", "noise_1.0", "zero"],
                        help="Modality perturbation conditions to evaluate")
    parser.add_argument("--n_blocks", type=int, default=2,
                        help="Number of cross-attn blocks (2 or 4)")
    parser.add_argument("--n_prev_turns", type=int, default=None,
                        help="Dialogue context length. Default: read from the "
                             "'dial{n}t' field of the checkpoint name.")
    parser.add_argument("--no_gating", action="store_true",
                        help="Disable tanh gating (gate=1.0)")
    parser.add_argument("--gate_mode", default="tanh", choices=["tanh", "linear", "fixed"],
                        help="Gate form used at training time: tanh (GRACE) / "
                             "linear (learnable, no tanh) / fixed (constant gate_init). "
                             "Must match the checkpoint, or the injection is mis-scaled.")
    parser.add_argument("--direction_decomp", action="store_true",
                        help="DDG: orthogonal/parallel decomposition with dual gates")
    parser.add_argument("--pure_orthogonal", action="store_true",
                        help="DDG ablation: inject only orthogonal component")
    parser.add_argument("--emotion_anchor", action="store_true",
                        help="EAS: restrict orthogonal injection to emotion-label subspace (implies --direction_decomp)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: auto-named)")
    parser.add_argument("--modality_target_only", action="store_true",
                        help="score with the target utterance's features only, matching a checkpoint trained the same way")
    args = parser.parse_args()
    if args.emotion_anchor:
        args.direction_decomp = True

    global MODALITY_TARGET_ONLY
    MODALITY_TARGET_ONLY = args.modality_target_only

    bb = backbone.describe(C.MODEL_PATH)
    insert_layers = backbone.insert_layers_for(args.n_blocks, bb["num_hidden_layers"])
    hidden_dim = bb["hidden_size"]

    seed_all(C.SEED)
    spec = get_spec(args.dataset)
    D = spec.get_data_module()

    modality = "audio" if args.dataset == "iemocap" else "visual"
    n_frames = N_FRAMES_AUDIO if modality == "audio" else N_FRAMES_VISUAL

    print(f"=== Robustness Matrix: {args.adapter_type} on {args.dataset} ({modality}) ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Conditions: {args.conditions}")
    print(f"  Modality: {modality}, n_frames: {n_frames}")
    print(f"  backbone: {bb['model_type']} | {bb['num_hidden_layers']} layers | width {hidden_dim}")
    print(f"  n_blocks: {args.n_blocks}, insert_layers: {insert_layers}")
    print(f"  no_gating: {args.no_gating}, gate_mode: {args.gate_mode}")

    # Guard against silently evaluating an amplitude-matched control under the wrong
    # gate form: tanh(0.1) and a constant 0.1 differ by 0.3%, so a mismatch would not
    # show up in the numbers, and an unbounded alpha would be squashed without warning.
    _ckpt_tag = os.path.basename(os.path.normpath(args.checkpoint))
    _implied = ("fixed" if "gatefixed" in _ckpt_tag
                else "linear" if "gatelinear" in _ckpt_tag else "tanh")
    if _implied != args.gate_mode and not args.no_gating:
        raise SystemExit(
            f"[gate_mode] checkpoint name implies '{_implied}' but --gate_mode is "
            f"'{args.gate_mode}'. Pass --gate_mode {_implied} (or rename the run).")

    model, tokenizer, _ = load_model_and_tokenizer(for_training=True)
    model = setup_lora(model)
    device = model.device

    model.load_adapter(args.checkpoint, adapter_name="default")
    print("[LoRA] Loaded adapter weights")

    # The trainer writes the dialogue context length into the run name as
    # "dial{n}t", and an adapter evaluated with a different n_prev_turns than it was
    # trained with injects the wrong number of turn tokens. Read it off the name
    # rather than hardcoding, and allow an explicit override for reproducing an
    # older run.
    _m = re.search(r"dial(\d+)t", _ckpt_tag)
    n_prev_turns = args.n_prev_turns if args.n_prev_turns is not None else (
        int(_m.group(1)) if _m else 1)
    print(f"  n_prev_turns: {n_prev_turns}"
          + ("" if _m is None else f" (checkpoint name implies {_m.group(1)})"))

    if args.adapter_type == "pilot1":
        adapter = VisualAdapter(n_blocks=args.n_blocks, dim=hidden_dim).to(device).float()
    else:
        adapter = DialogueVisualAdapter(
            n_blocks=args.n_blocks, dim=hidden_dim,
            n_prev_turns=n_prev_turns, use_frame_attn=False,
            dialogue_dropout=0.0, modality_dropout=0.0,
            no_gating=args.no_gating,
            direction_decomp=args.direction_decomp,
            pure_orthogonal=args.pure_orthogonal,
            emotion_anchor=args.emotion_anchor,
            emo_k=spec.num_classes if args.emotion_anchor else 0,
            gate_mode=args.gate_mode,
        ).to(device).float()

    va_state = torch.load(os.path.join(args.checkpoint, "visual_adapter.pt"),
                          weights_only=False)
    adapter.load_state_dict(va_state)
    print(f"[Adapter] Loaded, params={sum(p.numel() for p in adapter.parameters()):,}")

    _gates = []
    for _bi, _blk in enumerate(adapter.blocks):
        _ga = float(_blk._gate(_blk.gate_attn)) if not args.no_gating else 1.0
        _gf = float(_blk._gate(_blk.gate_ff)) if not args.no_gating else 1.0
        _gates.append({"block": _bi, "gate_attn": _ga, "gate_ff": _gf})
        print(f"  Block {_bi}: gate_attn={_ga:.4f}, gate_ff={_gf:.4f}")

    handles = install_hooks(model, adapter, insert_layers)

    test_data = D.load_split("test")
    feat_te = load_modality_feats(spec, "test", modality)
    print(f"  Test samples: {len(test_data)}, feat keys: {len(feat_te)}")

    results = {}
    for cond in args.conditions:
        print(f"\n--- Condition: {cond} ---")
        metrics = evaluate_condition(
            model, adapter, tokenizer, test_data, feat_te,
            spec, cond, args.adapter_type, n_frames)
        print_metrics(metrics, spec=spec)
        results[cond] = {
            "weighted_f1": metrics["weighted_f1"],
            "accuracy": metrics["accuracy"],
        }

    print("\n" + "=" * 60)
    print(f"{'Condition':<15} {'wF1':>8} {'Acc':>8} {'Δ wF1 vs clean':>15}")
    print("-" * 60)
    clean_f1 = results.get("clean", {}).get("weighted_f1", 0)
    for cond in args.conditions:
        f1 = results[cond]["weighted_f1"]
        acc = results[cond]["accuracy"]
        delta = f1 - clean_f1
        print(f"{cond:<15} {f1:>8.2f} {acc:>8.2f} {delta:>+15.2f}")
    print("=" * 60)

    if args.output is None:
        # Gate-form controls come in four checkpoints per dataset, so keep them apart.
        suffix = "" if args.gate_mode == "tanh" else f"_{args.gate_mode}_{_ckpt_tag}"
        out_path = os.path.join(
            C.OUTPUT_DIR, f"robustness_{args.adapter_type}_{args.dataset}{suffix}.json")
    else:
        out_path = args.output

    output_data = {
        "adapter_type": args.adapter_type,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "modality": modality,
        "gate_mode": "none" if args.no_gating else args.gate_mode,
        "realised_gates": _gates,
        "conditions": results,
    }
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
