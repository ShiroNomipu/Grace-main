
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import backbone
import config as C
from dataset_spec import get_spec
from infer import (load_model_and_tokenizer, _build_candidate_token_ids,
                   compute_metrics, print_metrics)
from run_v1 import seed_all, build_train_example, collate_batch, setup_lora

# ─── Config ───
# Default is Qwen2-VL-2B's language tower (28 layers, width 1536). Both are
# re-read from the loaded model in main(), so pointing C.MODEL_PATH at another
# backbone needs no edit here; these remain the defaults for callers that
# construct the adapter without a model in hand.
HIDDEN_DIM = 1536
INSERT_LAYERS = [4, 9, 14, 18]  # 4 blocks, uniformly spread across 28 layers
FEAT_DIM = 768          # both POSTER V2 and emotion2vec produce 768-d
N_FRAMES_VISUAL = 8    # MELD video: 8 frames
N_FRAMES_AUDIO = 64    # IEMOCAP audio: 64 frames
GATE_INIT = 0.1
N_PREV_TURNS = 1    # number of previous turns to include (dialogue context)
MAX_TURNS = 4       # max supported turns (for pos embedding table)


# ─── Frame Attention (Temporal Emotion-Peak) ───
class FrameAttention(nn.Module):
    """Content-based per-frame importance scoring with regularization.

    Learns which frames are emotionally salient before cross-attention.
    Regularization to prevent overfitting dev-specific frame patterns:
      - Reduced capacity (dim//12 hidden, ~64 for 768-d input)
      - Dropout in scorer MLP
      - Temperature-smoothed softmax (tau=2.0 prevents peaky weights)
      - Optional frame shuffle augmentation during training
    """

    def __init__(self, dim=FEAT_DIM, hidden_div=12, dropout=0.3, temperature=2.0,
                 frame_shuffle_prob=0.3):
        super().__init__()
        hidden = max(dim // hidden_div, 32)
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.temperature = temperature
        self.frame_shuffle_prob = frame_shuffle_prob

    def forward(self, frames):
        """frames: (B, N_frames, D) -> (B, N_frames, D) reweighted."""
        if self.training and self.frame_shuffle_prob > 0:
            if random.random() < self.frame_shuffle_prob:
                perm = torch.randperm(frames.shape[1], device=frames.device)
                frames = frames[:, perm, :]

        scores = self.scorer(frames)              # (B, N, 1)
        weights = torch.softmax(scores / self.temperature, dim=1)  # smoothed
        n_frames = frames.shape[1]
        return frames * weights * n_frames        # preserve magnitude


# ─── Gated Cross-Attention Block (same as Pilot-1) ───
class GatedCrossAttnBlock(nn.Module):
    """Flamingo-style gated cross-attention for visual injection."""

    def __init__(self, dim=HIDDEN_DIM, n_heads=12, ff_mult=2, gate_init=GATE_INIT, no_gating=False,
                 direction_decomp=False, pure_orthogonal=False,
                 emotion_anchor=False, emo_k=0, gate_mode="tanh"):
        super().__init__()
        self.no_gating = no_gating
        # gate_mode (non-DDG path only): fair-baseline injection scaling
        #   "tanh"  : g = tanh(alpha)               (standard GRACE, bounded learnable)
        #   "linear": g = alpha (learnable, no tanh) (isolates boundedness of tanh)
        #   "fixed" : g = gate_init constant, non-learnable (isolates small amplitude)
        self.gate_mode = gate_mode
        self.direction_decomp = direction_decomp
        self.pure_orthogonal = pure_orthogonal
        self.emotion_anchor = emotion_anchor
        self.check_ortho = False  # toggled True during sanity to run assertion
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=True)
        # gate_attn doubles as the orthogonal-component gate (perp) under DDG
        self.gate_attn = nn.Parameter(torch.full((1,), gate_init))
        # parallel-component gate: registered ONLY under DDG (keeps non-DDG ckpts loadable)
        if direction_decomp:
            self.gate_par = nn.Parameter(torch.full((1,), gate_init))
            if pure_orthogonal:
                self.gate_par.requires_grad_(False)
        # Emotion-Anchored Subspace (EAS): frozen orthonormal basis (D, k) spanned by
        # emotion-label token embeddings. Registered as a buffer so it is saved in the
        # checkpoint and restored at eval without needing the tokenizer/LLM. Filled by
        # DialogueVisualAdapter.set_emotion_basis() after the frozen LLM is available.
        if emotion_anchor:
            assert direction_decomp, "emotion_anchor requires direction_decomp (operates on f_perp)"
            assert emo_k > 0, "emotion_anchor needs emo_k = number of emotion labels"
            self.register_buffer("emo_Q", torch.zeros(dim, emo_k))

        self.norm2 = nn.LayerNorm(dim)
        ff_dim = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))
        self.gate_ff = nn.Parameter(torch.full((1,), gate_init))
        if gate_mode == "fixed":
            # constant, non-learnable gate at gate_init (small-amplitude, no learning)
            self.gate_attn.requires_grad_(False)
            self.gate_ff.requires_grad_(False)

    def _gate(self, raw):
        if self.no_gating:
            return 1.0
        if self.gate_mode == "tanh":
            return torch.tanh(raw)
        return raw  # "linear" (learnable, unbounded) or "fixed" (constant)

    def forward(self, text_hidden, visual_kv):
        """text_hidden: (B, T, D), visual_kv: (B, Nv, D)"""
        normed = self.norm1(text_hidden.float())
        attn_out, _ = self.cross_attn(normed, visual_kv, visual_kv)

        if self.direction_decomp:
            # Decompose the cross-attn output f relative to the residual stream h,
            # then gate the orthogonal (novel) and parallel (echo/cancel) parts separately.
            h = text_hidden.float()                       # (B, T, D)
            f = attn_out                                  # (B, T, D)
            h_sq = (h * h).sum(-1, keepdim=True).clamp(min=1e-8)
            f_par = ((f * h).sum(-1, keepdim=True) / h_sq) * h
            f_perp = f - f_par
            if self.emotion_anchor:
                # Restrict the orthogonal (novel) injection to the emotion-relevant
                # subspace, then re-orthogonalize to h so the non-destructive property
                # (zero projection onto the residual-stream direction) is preserved.
                Q = self.emo_Q.to(f_perp.dtype)              # (D, k)
                f_anchor = (f_perp @ Q) @ Q.t()              # (B, T, D)
                fa_par = ((f_anchor * h).sum(-1, keepdim=True) / h_sq) * h
                f_perp = f_anchor - fa_par
            if self.check_ortho:
                resid = (f_perp * h).sum(-1).abs()
                scale = f_perp.norm(dim=-1) * h.norm(dim=-1) + 1e-8
                max_cos = (resid / scale).max().item()
                assert max_cos < 1e-3, f"DDG orthogonality check failed: max|cos|={max_cos:.2e}"
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


# ─── Dialogue Visual Adapter ───
class DialogueVisualAdapter(nn.Module):
    """Visual adapter with frame attention + multi-turn dialogue KV support.

    Extends Pilot-1's VisualAdapter with:
      1. FrameAttention applied to raw features before projection
      2. Learnable turn-position embeddings added to projected tokens
      3. Support for concatenating multiple turns' visual tokens as KV
      4. Dialogue dropout: stochastic zero-out of non-target turns during training
    """

    def __init__(self, n_blocks=4, dim=HIDDEN_DIM, feat_dim=FEAT_DIM,
                 max_turns=MAX_TURNS, n_prev_turns=N_PREV_TURNS,
                 use_frame_attn=True, dialogue_dropout=0.0,
                 modality_dropout=0.0, feat_noise=0.0, gate_init=GATE_INIT,
                 no_gating=False, direction_decomp=False, pure_orthogonal=False,
                 emotion_anchor=False, emo_k=0, gate_mode="tanh"):
        super().__init__()
        self.n_prev_turns = n_prev_turns
        self.max_turns = max_turns
        self.dialogue_dropout = dialogue_dropout
        self.modality_dropout = modality_dropout
        self.feat_noise = feat_noise
        self.emotion_anchor = emotion_anchor
        self.frame_attn = FrameAttention(feat_dim) if use_frame_attn else None
        self.proj = nn.Linear(feat_dim, dim)
        self.proj_ln = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            GatedCrossAttnBlock(dim, gate_init=gate_init, no_gating=no_gating,
                                direction_decomp=direction_decomp,
                                pure_orthogonal=pure_orthogonal,
                                emotion_anchor=emotion_anchor, emo_k=emo_k,
                                gate_mode=gate_mode)
            for _ in range(n_blocks)])
        self.turn_pos_embed = nn.Embedding(max_turns, dim)
        self._raw_turn_feats = None  # list of detached tensors (no graph)

    def set_visual(self, turn_feats_list):
        """Store raw (detached) turn features; projection recomputed in hook.

        Args:
            turn_feats_list: list of tensors [(B, N_FRAMES, VISUAL_DIM), ...],
                ordered oldest-turn-first, target-turn-last.
        """
        self._raw_turn_feats = [f.detach() for f in turn_feats_list]

    def compute_visual_tokens(self):
        """Recompute projection from raw feats (gradient checkpoint safe).

        Called from hook on every forward pass; builds fresh graph each time.
        Applies modality dropout (zero all visual) or dialogue dropout (zero prev turns).
        """
        if self._raw_turn_feats is None:
            return None

        # Modality dropout: zero-out ALL visual with probability p
        if (self.training and self.modality_dropout > 0
                and random.random() < self.modality_dropout):
            B = self._raw_turn_feats[0].shape[0]
            n_total_frames = sum(f.shape[1] for f in self._raw_turn_feats)
            dim = self.proj.out_features
            return torch.zeros(B, n_total_frames, dim,
                               device=self._raw_turn_feats[0].device)

        all_tokens = []
        n_turns = len(self._raw_turn_feats)
        for turn_idx, feats in enumerate(self._raw_turn_feats):
            is_target_turn = (turn_idx == n_turns - 1)
            f = feats.float()

            # Feature noise augmentation during training
            if self.training and self.feat_noise > 0:
                f = f + torch.randn_like(f) * self.feat_noise

            # Dialogue dropout: zero-out non-target turns with probability p
            if (self.training and not is_target_turn
                    and self.dialogue_dropout > 0
                    and random.random() < self.dialogue_dropout):
                f = torch.zeros_like(f)

            if self.frame_attn is not None:
                f = self.frame_attn(f)
            projected = self.proj_ln(self.proj(f))   # (B, N_FRAMES, dim)
            pos_offset = self.max_turns - n_turns + turn_idx
            pos_emb = self.turn_pos_embed.weight[pos_offset]  # (dim,)
            projected = projected + pos_emb
            all_tokens.append(projected)

        return torch.cat(all_tokens, dim=1)  # (B, n_turns*N_FRAMES, dim)

    def get_block(self, idx):
        return self.blocks[idx]

    def set_emotion_basis(self, Q):
        """Copy the frozen emotion-anchored orthonormal basis (D, k) into every block."""
        for blk in self.blocks:
            assert hasattr(blk, "emo_Q"), "block was not built with emotion_anchor=True"
            with torch.no_grad():
                blk.emo_Q.copy_(Q.to(blk.emo_Q.device, blk.emo_Q.dtype))


def build_emotion_basis(model, tokenizer, spec, device):
    """Build an orthonormal basis (D, k) spanning the emotion-label token embeddings.

    For each emotion label word, we look up its (possibly multi-token) input embeddings
    from the frozen LLM and average them into a single direction. The k label directions
    are then orthonormalized via QR. The residual stream at the injection layers is an
    additive accumulation that still contains these token-embedding directions, so this
    basis defines the 'emotion-relevant' subspace used to filter the modality injection.
    """
    emb = model.get_input_embeddings().weight.detach().float()  # (V, D)
    vecs = []
    for lab in spec.label_list:
        word = spec.label2word[lab]
        # include the leading-space BPE variant (how the word appears mid-sentence/as an answer)
        ids = set()
        for variant in (word, " " + word):
            ids.update(tokenizer.encode(variant, add_special_tokens=False))
        ids = [i for i in ids if 0 <= i < emb.shape[0]]
        assert ids, f"no token ids for emotion word '{word}'"
        vecs.append(emb[ids].mean(0))
    M = torch.stack(vecs, dim=0)              # (k, D)
    Q, _ = torch.linalg.qr(M.t())             # Q: (D, k) orthonormal columns
    ortho_err = torch.dist(Q.t() @ Q, torch.eye(Q.shape[1], device=Q.device, dtype=Q.dtype))
    print(f"[EAS] emotion basis built: {len(spec.label_list)} labels -> "
          f"basis {tuple(Q.shape)} (rank check ||QtQ-I||={ortho_err:.2e})")
    return Q.to(device).float()


# ─── Modality features loading ───
def load_modality_feats(spec, split, modality="visual"):
    """Load frame-level features for either visual, audio, or dual modality.

    Returns: {feat_key: np.float32(N_FRAMES, 768)}
    """
    if modality == "dual":
        return load_dual_modality_feats(spec, split)
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


def load_dual_modality_feats(spec, split):
    """Load and concatenate audio + visual features for dual-modality mode.

    Returns: {feat_key: np.float32(N_FRAMES_AUDIO + N_FRAMES_VISUAL, 768)}
    """
    v_path = os.path.join(spec.video_frame_dir,
                          spec.video_frame_pt_tmpl.format(split=split))
    a_path = os.path.join(spec.audio_frame_dir,
                          spec.audio_frame_pt_tmpl.format(split=split))
    v_data = torch.load(v_path, weights_only=False)
    a_data = torch.load(a_path, weights_only=False)
    v_feats, v_keys = v_data["feats"], v_data["order_keys"]
    a_feats, a_keys = a_data["feats"], a_data["order_keys"]
    v_map = {k: v_feats[i] for i, k in enumerate(v_keys)}
    a_map = {k: a_feats[i] for i, k in enumerate(a_keys)}
    all_keys = set(v_keys) | set(a_keys)
    result = {}
    for k in all_keys:
        v = v_map.get(k, np.zeros((N_FRAMES_VISUAL, FEAT_DIM), dtype=np.float32))
        a = a_map.get(k, np.zeros((N_FRAMES_AUDIO, FEAT_DIM), dtype=np.float32))
        if isinstance(v, torch.Tensor):
            v = v.numpy()
        if isinstance(a, torch.Tensor):
            a = a.numpy()
        result[k] = np.concatenate([a, v], axis=0)  # (72, 768)
    print(f"  [dual] audio({N_FRAMES_AUDIO}) + visual({N_FRAMES_VISUAL}) = {N_FRAMES_AUDIO + N_FRAMES_VISUAL} frames")
    return result


# IEMOCAP gained context_feat_keys on 2026-06-16; before that its samples had
# no key list, so this function fell through to the target utterance alone and
# zero-filled the context slot. Runs from that period are not reproducible
# otherwise, and the amount of modality injected turns out to dominate how much
# a model loses when the modality is removed, so the old behaviour is worth
# being able to ask for explicitly rather than recovering it by deleting a field.
MODALITY_TARGET_ONLY = False


def get_dialogue_feats(sample, feat_map, n_prev=N_PREV_TURNS, n_frames=N_FRAMES_VISUAL):
    """Get modality features for target + N previous turns.

    Works with both MELD (context_clip_ids) and IEMOCAP (context_feat_keys).
    Returns list of np arrays [(n_frames, 768), ...], oldest first, target last.
    Missing turns get zero-filled.
    """
    # Unified key list: MELD uses context_clip_ids, IEMOCAP uses context_feat_keys
    feat_keys = sample.get("context_feat_keys") or sample.get("context_clip_ids")
    if feat_keys is None:
        # Fallback: single-turn (target only)
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
            result.append(feat_map[key])
        else:
            result.append(np.zeros((n_frames, FEAT_DIM), dtype=np.float32))

    while len(result) < n_prev + 1:
        result.insert(0, np.zeros((n_frames, FEAT_DIM), dtype=np.float32))

    return result


# ─── Hook management ───
def install_hooks(model, visual_adapter, insert_layers):
    """Install forward hooks on specified layers."""
    layers = backbone.decoder_layers(model)
    handles = []

    for block_idx, layer_idx in enumerate(insert_layers):
        block = visual_adapter.get_block(block_idx)

        def make_hook(blk):
            def hook_fn(module, input, output):
                # Qwen2VLDecoderLayer returns a tuple; LlamaDecoderLayer in
                # transformers >= 4.54 returns the hidden states bare. The hook
                # has to hand back whichever shape it was given, or the next
                # layer's norm receives a tuple.
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                v_tokens = visual_adapter.compute_visual_tokens()
                if v_tokens is None:
                    return output
                modified = blk(hidden, v_tokens).to(hidden.dtype)
                return (modified,) + output[1:] if is_tuple else modified
            return hook_fn

        h = layers[layer_idx].register_forward_hook(make_hook(block))
        handles.append(h)

    return handles


# ─── Sanity check ───
def tiny_overfit_sanity(model, visual_adapter, tokenizer, samples, feat_map, spec,
                        n_frames, n=8, steps=60):
    print(f"\n[SANITY] tiny-overfit on {n} samples ({n_frames}-frame adapter) for {steps} steps ...")
    device = model.device
    subset = samples[:n]
    examples = [build_train_example(tokenizer, s) for s in subset]
    bs = min(n, C.BATCH_SIZE)
    batch = collate_batch(tokenizer, examples[:bs])

    turn_feats_batch = _collate_dialogue_feats(
        subset[:bs], feat_map, device, visual_adapter.n_prev_turns, n_frames)

    params = ([p for p in model.parameters() if p.requires_grad] +
              list(visual_adapter.parameters()))
    opt = torch.optim.AdamW(params, lr=2e-4)

    # Enable DDG orthogonality self-check during sanity only (avoids per-step GPU sync in full training)
    for blk in visual_adapter.blocks:
        if getattr(blk, "direction_decomp", False):
            blk.check_ortho = True

    model.train(); visual_adapter.train()
    first_loss = last_loss = None
    for step in range(steps):
        visual_adapter.set_visual(turn_feats_batch)
        input_ids, attn, labels = (t.to(device) for t in batch)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        loss.backward(); opt.step(); opt.zero_grad()
        first_loss = first_loss if first_loss is not None else loss.item()
        last_loss = loss.item()
        if step % 10 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={loss.item():.4f}")

    print(f"[SANITY] first={first_loss:.4f} -> last={last_loss:.4f}")
    assert last_loss < first_loss * 0.5, (
        f"overfit failed ({first_loss:.3f}->{last_loss:.3f})")
    # Disable ortho self-check after sanity (keep full-training loop free of GPU sync)
    for blk in visual_adapter.blocks:
        if getattr(blk, "direction_decomp", False):
            blk.check_ortho = False
            print(f"[SANITY] DDG orthogonality assertion passed.")
            break
    print("[SANITY] PASS\n")


def _collate_dialogue_feats(samples, feat_map, device, n_prev, n_frames):
    """Collate multi-turn modality features for a batch.

    Returns: list of (B, n_frames, FEAT_DIM) tensors, one per turn position.
    """
    n_turns = n_prev + 1
    batch_turns = [[] for _ in range(n_turns)]

    for s in samples:
        feats_list = get_dialogue_feats(s, feat_map, n_prev, n_frames)
        for t_idx in range(n_turns):
            batch_turns[t_idx].append(feats_list[t_idx])

    result = []
    for t_idx in range(n_turns):
        stacked = np.stack(batch_turns[t_idx])
        result.append(torch.tensor(stacked, device=device, dtype=torch.float32))
    return result


# ─── Evaluation ───
@torch.no_grad()
def score_candidates(model, adapter, tokenizer, messages, turn_feats_list,
                     device, spec):
    """Candidate scoring with modality adapter active."""
    prompt_ids, cands = _build_candidate_token_ids(tokenizer, messages, spec=spec)

    vis_list = [torch.tensor(f[None], device=device, dtype=torch.float32)
                for f in turn_feats_list]
    adapter.set_visual(vis_list)

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


def evaluate(model, adapter, tokenizer, samples, feat_map, spec, n_frames, desc="eval"):
    device = model.device
    n_prev = adapter.n_prev_turns
    model.eval(); adapter.eval()
    preds, golds = [], []
    for s in tqdm(samples, desc=desc):
        feats_list = get_dialogue_feats(s, feat_map, n_prev, n_frames)
        preds.append(score_candidates(
            model, adapter, tokenizer, s["messages"], feats_list, device, spec))
        golds.append(s["label_id"])
    return compute_metrics(preds, golds, spec=spec)


# ─── Training ───
def train_full(model, adapter, tokenizer, D, spec, tag="pilot3",
               epochs_override=None, adapter_lr=None, adapter_wd=0.0,
               use_cosine=False, modality="visual", n_frames=N_FRAMES_VISUAL,
               focal_gamma=0.0, gate_lr=None):
    from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
    device = model.device
    n_epochs = epochs_override if epochs_override is not None else C.EPOCHS
    train_data = D.load_split("train")
    dev_data = D.load_split("dev")
    feat_tr = load_modality_feats(spec, "train", modality)
    feat_dv = load_modality_feats(spec, "dev", modality)
    print(f"train={len(train_data)} dev={len(dev_data)}")
    print(f"feat_tr keys: {len(feat_tr)}, feat_dv keys: {len(feat_dv)}")

    examples = [build_train_example(tokenizer, s) for s in train_data]

    n_batches = len(examples) // C.BATCH_SIZE
    total_steps = (n_batches // C.GRAD_ACCUM) * n_epochs
    warmup = int(total_steps * C.WARMUP_RATIO)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    # The gate scalars sit in the adapter group by default, so at adapter_lr an
    # AdamW step moves them by at most ~lr and they stay near their initial
    # value. --gate_lr gives them their own rate so that mobility can be tested
    # separately from the rest of the adapter.
    gate_params = [p for n, p in adapter.named_parameters() if "gate" in n]
    gate_ids = {id(p) for p in gate_params}
    va_params = [p for p in adapter.parameters() if id(p) not in gate_ids] \
        if gate_lr is not None else list(adapter.parameters())
    all_params = lora_params + list(adapter.parameters())
    if adapter_lr is not None:
        param_groups = [
            {"params": lora_params, "lr": C.LR, "weight_decay": 0.0},
            {"params": va_params, "lr": adapter_lr, "weight_decay": adapter_wd},
        ]
    else:
        param_groups = [
            {"params": lora_params + va_params, "lr": C.LR, "weight_decay": 0.0},
        ]
    if gate_lr is not None:
        param_groups.append({"params": gate_params, "lr": gate_lr, "weight_decay": 0.0})
        print(f"[gate_lr] {len(gate_params)} gate scalars in their own group at lr={gate_lr:g} "
              f"(adapter group lr={adapter_lr if adapter_lr is not None else C.LR:g})")
    opt = torch.optim.AdamW(param_groups)
    if use_cosine:
        sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    else:
        sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    best_dev_f1 = -1.0
    best_dir = os.path.join(C.OUTPUT_DIR, f"{tag}_best")
    use_focal = focal_gamma > 0

    indices = list(range(len(examples)))
    for epoch in range(n_epochs):
        random.shuffle(indices)
        model.train(); adapter.train()
        epoch_loss, n_steps = 0.0, 0
        opt.zero_grad()

        pbar = tqdm(range(0, len(indices) - C.BATCH_SIZE + 1, C.BATCH_SIZE),
                    desc=f"Epoch {epoch}")
        for i in pbar:
            bi = indices[i:i + C.BATCH_SIZE]
            batch = collate_batch(tokenizer, [examples[j] for j in bi])

            batch_samples = [train_data[j] for j in bi]
            turn_feats_batch = _collate_dialogue_feats(
                batch_samples, feat_tr, device, adapter.n_prev_turns, n_frames)
            adapter.set_visual(turn_feats_batch)

            input_ids, attn, labels = (t.to(device) for t in batch)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)

            if use_focal:
                logits = out.logits[:, :-1, :].contiguous().float()
                tgt = labels[:, 1:].contiguous()
                ce = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), tgt.view(-1),
                    ignore_index=-100, reduction='none')
                pt = torch.exp(-ce.detach()).clamp(min=1e-7, max=1.0)
                focal_weight = (1.0 - pt) ** focal_gamma
                n_valid = (tgt != -100).sum().clamp(min=1)
                raw_loss = (focal_weight * ce).sum() / n_valid
            else:
                raw_loss = out.loss

            loss = raw_loss / C.GRAD_ACCUM
            loss.backward()
            epoch_loss += raw_loss.item()
            n_steps += 1

            if n_steps % C.GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(all_params, C.MAX_GRAD_NORM)
                opt.step(); sched.step(); opt.zero_grad()

            if n_steps % 50 == 0:
                pbar.set_postfix(loss=f"{epoch_loss/n_steps:.4f}")

        avg_loss = epoch_loss / max(n_steps, 1)
        print(f"\n[Epoch {epoch}] avg_loss={avg_loss:.4f}")

        for bi_idx, blk in enumerate(adapter.blocks):
            ga = torch.tanh(blk.gate_attn).item()
            gf = torch.tanh(blk.gate_ff).item()
            if getattr(blk, "direction_decomp", False):
                gp = torch.tanh(blk.gate_par).item()
                print(f"  Block {bi_idx}: gate_perp={ga:.4f}, gate_par={gp:.4f}, gate_ff={gf:.4f}")
            else:
                print(f"  Block {bi_idx}: gate_attn={ga:.4f}, gate_ff={gf:.4f}")

        metrics = evaluate(model, adapter, tokenizer, dev_data,
                           feat_dv, spec, n_frames, desc=f"dev-ep{epoch}")
        print_metrics(metrics, spec=spec)
        dev_f1 = metrics["weighted_f1"]

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            torch.save(adapter.state_dict(),
                       os.path.join(best_dir, "visual_adapter.pt"))
            print(f"  >> new best dev wf1={dev_f1:.4f}, saved to {best_dir}")

    print(f"\nTraining done. Best dev wf1 = {best_dev_f1:.4f}")
    return best_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="meld", choices=["iemocap", "meld"])
    parser.add_argument("--small", type=int, default=0,
                        help="Subset size for quick test (0=full)")
    parser.add_argument("--n_prev_turns", type=int, default=N_PREV_TURNS,
                        help="Number of previous turns for dialogue context")
    parser.add_argument("--no_frame_attn", action="store_true",
                        help="Disable frame attention (for ablation)")
    parser.add_argument("--n_blocks", type=int, default=len(INSERT_LAYERS),
                        help="Number of cross-attn blocks")
    parser.add_argument("--skip_sanity", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override C.EPOCHS (for quick experiments)")
    parser.add_argument("--dialogue_dropout", type=float, default=0.0,
                        help="Prob of zeroing prev turn features during training")
    parser.add_argument("--modality_dropout", type=float, default=0.0,
                        help="Prob of zeroing ALL modality features during training")
    parser.add_argument("--feat_noise", type=float, default=0.0,
                        help="Gaussian noise std added to features during training")
    parser.add_argument("--adapter_lr", type=float, default=None,
                        help="Separate lr for adapter (differential lr)")
    parser.add_argument("--adapter_wd", type=float, default=0.0,
                        help="Weight decay for adapter params")
    parser.add_argument("--cosine", action="store_true",
                        help="Use cosine LR schedule instead of linear")
    parser.add_argument("--focal_loss", action="store_true",
                        help="Use focal loss instead of CE (helps minority classes)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma (higher = more focus on hard examples)")
    parser.add_argument("--persona", action="store_true",
                        help="Use persona-enhanced prompt (MELD only)")
    parser.add_argument("--dual_modality", action="store_true",
                        help="Use audio+visual dual modality (MELD only)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override C.SEED for multi-seed experiments")
    parser.add_argument("--gate_init", type=float, default=None,
                        help="Override GATE_INIT (default 0.1; set 0 for Flamingo-style)")
    parser.add_argument("--gate_lr", type=float, default=None,
                        help="Separate learning rate for the gate scalars; at the adapter rate "
                             "they can only move ~lr*steps and stay near gate_init")
    parser.add_argument("--gate_mode", default="tanh", choices=["tanh", "linear", "fixed"],
                        help="Fair-baseline gate: tanh (GRACE) / linear (learnable no tanh) / fixed (constant gate_init)")
    parser.add_argument("--no_gating", action="store_true",
                        help="Disable gating (gate≡1.0, direct addition baseline)")
    parser.add_argument("--direction_decomp", action="store_true",
                        help="DDG: decompose cross-attn output into orthogonal/parallel parts with dual gates")
    parser.add_argument("--pure_orthogonal", action="store_true",
                        help="DDG ablation: inject ONLY the orthogonal component (gate_par frozen at 0)")
    parser.add_argument("--emotion_anchor", action="store_true",
                        help="EAS: restrict the orthogonal injection to the emotion-label embedding subspace (implies --direction_decomp)")
    parser.add_argument("--modality_target_only", action="store_true",
                        help="inject the target utterance's features only and leave the context slot zero, as IEMOCAP runs did before context_feat_keys existed")
    args = parser.parse_args()
    if args.emotion_anchor:
        args.direction_decomp = True

    global MODALITY_TARGET_ONLY
    MODALITY_TARGET_ONLY = args.modality_target_only

    n_prev_turns = args.n_prev_turns

    # Auto-select modality based on dataset
    if args.dual_modality and args.dataset == "meld":
        modality = "dual"
        n_frames = N_FRAMES_AUDIO + N_FRAMES_VISUAL  # 64 + 8 = 72
    elif args.dataset == "iemocap":
        modality = "audio"
        n_frames = N_FRAMES_AUDIO
    else:
        modality = "visual"
        n_frames = N_FRAMES_VISUAL

    # Insertion depths and residual width come from the backbone's own config,
    # so a cross-backbone run of the same depth keeps the main-line layers 9/18
    # and only the family and width change.
    bb = backbone.describe(C.MODEL_PATH)
    insert_layers = backbone.insert_layers_for(args.n_blocks, bb["num_hidden_layers"])
    hidden_dim = bb["hidden_size"]

    run_seed = args.seed if args.seed is not None else C.SEED
    seed_all(run_seed)
    spec = get_spec(args.dataset)
    D = spec.get_data_module()
    tag = f"pilot3_dial{n_prev_turns}t_{args.n_blocks}blk_{args.dataset}"
    if args.modality_target_only:
        tag += "_tgtonly"
    if args.no_frame_attn:
        tag += "_nofa"
    if args.dialogue_dropout > 0:
        tag += f"_ddrop{int(args.dialogue_dropout*100)}"
    if args.modality_dropout > 0:
        tag += f"_mdrop{int(args.modality_dropout*100)}"
    if args.feat_noise > 0:
        tag += f"_fnoise{args.feat_noise}"
    if args.gate_init is not None:
        tag += f"_ginit{args.gate_init}"
    if args.no_gating:
        tag += "_nogating"
    if args.gate_mode != "tanh":
        tag += f"_gate{args.gate_mode}"
    if args.gate_lr is not None:
        tag += f"_glr{args.gate_lr}"
    if args.direction_decomp:
        if args.emotion_anchor:
            tag += "_easortho" if args.pure_orthogonal else "_eas"
        else:
            tag += "_ddgortho" if args.pure_orthogonal else "_ddg"
    if args.adapter_lr is not None:
        tag += f"_alr{args.adapter_lr}"
    if args.cosine:
        tag += "_cos"
    if args.dual_modality:
        tag += "_dual"
    if args.focal_loss:
        tag += f"_focal{args.focal_gamma}"
    if args.persona:
        tag += "_persona"
    if args.seed is not None:
        tag += f"_s{args.seed}"
    print(f"=== Pilot-3: Dialogue Modality Adapter [{args.dataset}] ===")
    print(f"  backbone: {bb['model_type']} | {bb['num_hidden_layers']} layers "
          f"| width {hidden_dim} | {os.path.basename(os.path.normpath(C.MODEL_PATH))}")
    print(f"  modality={modality}, n_frames={n_frames}")
    print(f"  blocks={args.n_blocks} at layers {insert_layers}")
    print(f"  dialogue turns: target + {n_prev_turns} previous")
    print(f"  frame_attention: {not args.no_frame_attn}")
    print(f"  dialogue_dropout: {args.dialogue_dropout}")
    print(f"  modality_dropout: {args.modality_dropout}")
    print(f"  feat_noise: {args.feat_noise}")
    print(f"  gate_init: {args.gate_init if args.gate_init is not None else GATE_INIT}")
    print(f"  gate_mode: {args.gate_mode}")
    print(f"  adapter_lr: {args.adapter_lr}, adapter_wd: {args.adapter_wd}")
    print(f"  focal_loss: {args.focal_loss} (gamma={args.focal_gamma})")
    print(f"  persona: {args.persona}")
    print(f"  seed: {run_seed}")
    print(f"  tag: {tag}")

    # Load model
    model, tokenizer, _ = load_model_and_tokenizer(for_training=True)
    model = setup_lora(model)
    device = model.device

    # Build adapter (architecture identical for both modalities)
    assert backbone.hidden_size(model) == hidden_dim, "config/model width mismatch"
    assert len(backbone.decoder_layers(model)) == bb["num_hidden_layers"], \
        "config/model depth mismatch"

    adapter = DialogueVisualAdapter(
        n_blocks=len(insert_layers),
        dim=hidden_dim,
        n_prev_turns=n_prev_turns,
        use_frame_attn=not args.no_frame_attn,
        dialogue_dropout=args.dialogue_dropout,
        modality_dropout=args.modality_dropout,
        feat_noise=args.feat_noise,
        gate_init=args.gate_init if args.gate_init is not None else GATE_INIT,
        no_gating=args.no_gating,
        direction_decomp=args.direction_decomp,
        pure_orthogonal=args.pure_orthogonal,
        emotion_anchor=args.emotion_anchor,
        emo_k=spec.num_classes if args.emotion_anchor else 0,
        gate_mode=args.gate_mode,
    ).to(device).float()
    if args.emotion_anchor:
        emo_Q = build_emotion_basis(model, tokenizer, spec, device)
        adapter.set_emotion_basis(emo_Q)
    va_params = sum(p.numel() for p in adapter.parameters())
    print(f"[DialogueAdapter] params: {va_params:,} | modality={modality}")

    # Install hooks
    handles = install_hooks(model, adapter, insert_layers)
    print(f"[Hooks] Installed at layers {insert_layers}")

    # Wrap data module to pass persona flag (only affects MELD)
    use_persona = args.persona and args.dataset == "meld"
    _orig_load = D.load_split
    if use_persona:
        import functools
        D.load_split = functools.partial(_orig_load, use_persona=True)

    # Load modality features for sanity check
    train_data = D.load_split("train")
    feat_tr = load_modality_feats(spec, "train", modality)

    # Sanity check
    if not args.skip_sanity:
        tiny_overfit_sanity(model, adapter, tokenizer, train_data, feat_tr, spec,
                            n_frames)

    # Train
    if args.small:
        class SmallD:
            def load_split(self, s):
                return D.load_split(s)[:args.small]
        best_dir = train_full(model, adapter, tokenizer, SmallD(), spec,
                              tag=tag, epochs_override=args.epochs,
                              adapter_lr=args.adapter_lr, adapter_wd=args.adapter_wd,
                              use_cosine=args.cosine,
                              modality=modality, n_frames=n_frames,
                              focal_gamma=args.focal_gamma if args.focal_loss else 0.0,
                              gate_lr=args.gate_lr)
    else:
        best_dir = train_full(model, adapter, tokenizer, D, spec,
                              tag=tag, epochs_override=args.epochs,
                              adapter_lr=args.adapter_lr, adapter_wd=args.adapter_wd,
                              use_cosine=args.cosine,
                              modality=modality, n_frames=n_frames,
                              focal_gamma=args.focal_gamma if args.focal_loss else 0.0,
                              gate_lr=args.gate_lr)

    # Final test
    if not args.skip_test:
        print("\n=== Test evaluation ===")
        test_data = D.load_split("test")
        feat_te = load_modality_feats(spec, "test", modality)

        from peft import PeftModel
        model.load_adapter(best_dir, adapter_name="default")
        va_state = torch.load(os.path.join(best_dir, "visual_adapter.pt"),
                              weights_only=False)
        adapter.load_state_dict(va_state)

        metrics = evaluate(model, adapter, tokenizer, test_data,
                           feat_te, spec, n_frames, desc="test")
        print_metrics(metrics, spec=spec)

        out_json = os.path.join(C.OUTPUT_DIR, f"{tag}_test_metrics.json")
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {out_json}")

    # Cleanup hooks
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
