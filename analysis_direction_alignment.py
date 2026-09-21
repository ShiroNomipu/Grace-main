"""
DDG Phase-0 zero-training pre-verification (Go/No-Go gate for L12).

Measures the per-token cosine alignment cos(h, f) between:
  - h: the REAL LLM hidden state at the injection layer (residual stream, pre-injection)
  - f: the cross-attention output attn_out (the signal about to be injected), BEFORE gating

This tells us whether the modality signal learned by the *existing* checkpoint is:
  - near-orthogonal (|mean cos| < 0.05)  -> effective injection lives in orthogonal subspace  => DDG Go
  - negatively parallel (mean cos < -0.1) -> modality is cancelling text                       => DDG Go (stronger)
  - positively parallel (mean cos > 0.1)  -> modality mostly echoes text                        => DDG No-Go (link to ESMC redundancy)

Reuses the real model + hook flow from eval_robustness_matrix.py.
No training. Single forward per sample (prompt only).
"""
import argparse
import json
import os

import numpy as np
import torch

import config as C
from dataset_spec import get_spec
from infer import load_model_and_tokenizer, _build_candidate_token_ids
from run_v1 import seed_all, setup_lora
from eval_robustness_matrix import (
    DialogueVisualAdapter, VisualAdapter,
    load_modality_feats, get_dialogue_feats, perturb_feats,
    N_FRAMES_VISUAL, N_FRAMES_AUDIO,
)

OUTPUT_DIR = "outputs/direction_alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _cos_rows(a, b):
    """Per-token cosine between two (B, T, D) tensors -> flat numpy array."""
    num = (a * b).sum(-1)
    denom = a.norm(dim=-1) * b.norm(dim=-1) + 1e-8
    return (num / denom).flatten().detach().cpu().numpy()


def summ(arr):
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "frac_near_orth": float((np.abs(arr) < 0.05).mean()),
            "n_tokens": int(arr.size)}


def install_alignment_hooks(model, adapter, insert_layers, store):
    """Install hooks that (1) capture cos(h, attn_out) per token, then
    (2) faithfully replicate the block forward so the model output is unchanged.
    store: dict {layer_idx: {"cos": [...], "labels": [...]}} filled in-place.
    A module-level `_current_label` is read for per-class bucketing.
    """
    layers = model.model.language_model.layers
    handles = []
    for block_idx, layer_idx in enumerate(insert_layers):
        block = adapter.get_block(block_idx)

        def make_hook(blk, lidx):
            def hook_fn(module, inp, output):
                if isinstance(output, tuple):
                    hidden, rest = output[0], output[1:]
                else:
                    hidden, rest = output, ()
                v = adapter.compute_visual_tokens()
                if v is None:
                    return output

                h = hidden.float()                        # (B, T, D)
                normed = blk.norm1(h)
                attn_out, _ = blk.cross_attn(normed, v, v)  # (B, T, D) = f

                # per-token cosine alignment cos(h, f)
                store[lidx]["cos"].append(_cos_rows(h, attn_out))
                # null controls: an isotropic random direction, and the same
                # cross-attention driven by ANOTHER sample's modality features.
                store[lidx]["cos_rand"].append(
                    _cos_rows(h, torch.randn_like(attn_out)))
                v_shuf = _STATE.get("v_shuf")
                if v_shuf is not None:
                    attn_shuf, _ = blk.cross_attn(normed, v_shuf, v_shuf)
                    store[lidx]["cos_shuf"].append(_cos_rows(h, attn_shuf))
                    store[lidx]["fnorm_shuf"].append(
                        attn_shuf.norm(dim=-1).flatten().detach().cpu().numpy())
                store[lidx]["fnorm"].append(
                    attn_out.norm(dim=-1).flatten().detach().cpu().numpy())
                store[lidx]["hnorm"].append(
                    h.norm(dim=-1).flatten().detach().cpu().numpy())
                store[lidx]["labels"].append(_STATE["label"])
                store[lidx]["dim"] = int(attn_out.shape[-1])

                # faithfully replicate real block forward (so downstream is correct)
                g_attn = torch.tanh(blk.gate_attn)
                th = hidden + (g_attn * attn_out).to(hidden.dtype)
                normed2 = blk.norm2(th.float())
                ff_out = blk.ff(normed2)
                g_ff = torch.tanh(blk.gate_ff)
                th = th + (g_ff * ff_out).to(hidden.dtype)
                return (th,) + rest
            return hook_fn

        h = layers[layer_idx].register_forward_hook(make_hook(block, layer_idx))
        handles.append(h)
    return handles


_STATE = {"label": -1, "v_shuf": None}


@torch.no_grad()
def run(dataset, checkpoint, n_blocks, max_samples):
    if n_blocks == 4:
        insert_layers = [4, 9, 14, 18]
    else:
        insert_layers = [9, 18]

    seed_all(C.SEED)
    spec = get_spec(dataset)
    D = spec.get_data_module()
    modality = "audio" if dataset == "iemocap" else "visual"
    n_frames = N_FRAMES_AUDIO if modality == "audio" else N_FRAMES_VISUAL

    print(f"=== DDG Phase-0 Direction Alignment: {dataset} ({modality}) ===")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  insert_layers: {insert_layers}, max_samples: {max_samples}")

    model, tokenizer, _ = load_model_and_tokenizer(for_training=True)
    model = setup_lora(model)
    device = model.device
    model.load_adapter(checkpoint, adapter_name="default")

    adapter = DialogueVisualAdapter(
        n_blocks=n_blocks, n_prev_turns=1, use_frame_attn=False,
        dialogue_dropout=0.0, modality_dropout=0.0, no_gating=False,
    ).to(device).float()
    va_state = torch.load(os.path.join(checkpoint, "visual_adapter.pt"),
                          weights_only=False)
    adapter.load_state_dict(va_state)
    adapter.eval()

    store = {li: {"cos": [], "cos_rand": [], "cos_shuf": [],
                  "fnorm": [], "fnorm_shuf": [], "hnorm": [], "labels": []}
             for li in insert_layers}
    handles = install_alignment_hooks(model, adapter, insert_layers, store)

    test_data = D.load_split("test")
    feat_te = load_modality_feats(spec, "test", modality)

    def build_vis(sample):
        feats_list = get_dialogue_feats(sample, feat_te, 1, n_frames)
        turn_feats = [perturb_feats(f, "clean") for f in feats_list]
        return [torch.tensor(f[None], device=device, dtype=torch.float32)
                for f in turn_feats]

    subset = test_data[:max_samples]
    n_sub = len(subset)
    rng = np.random.RandomState(C.SEED)
    # partner index for the shuffled-modality control (never the sample itself)
    partner = [(i + 1 + int(rng.randint(0, max(n_sub - 1, 1)))) % n_sub
               for i in range(n_sub)]

    n_proc = 0
    for i, s in enumerate(subset):
        vis_list = build_vis(s)

        # precompute the partner's modality tokens, then restore the real ones
        adapter.set_visual(build_vis(subset[partner[i]]))
        v_shuf = adapter.compute_visual_tokens()
        _STATE["v_shuf"] = None if v_shuf is None else v_shuf.detach()
        adapter.set_visual(vis_list)

        prompt_ids, _ = _build_candidate_token_ids(tokenizer, s["messages"], spec=spec)
        input_ids = torch.tensor([prompt_ids], device=device)
        attn_mask = torch.ones_like(input_ids)
        _STATE["label"] = int(s["label_id"])
        model(input_ids=input_ids, attention_mask=attn_mask)
        n_proc += 1

    for h in handles:
        h.remove()

    # ─── Aggregate ───
    report = {"dataset": dataset, "checkpoint": checkpoint,
              "n_samples": n_proc, "layers": {}}
    print(f"\n  Processed {n_proc} samples\n")
    print(f"  {'Layer':<8}{'meanCos':>10}{'medCos':>10}{'|cos|<.05':>12}"
          f"{'cos<-.1':>10}{'cos>.1':>10}{'meanFnorm':>12}{'meanHnorm':>12}")
    print("  " + "-" * 84)

    overall_flags = []
    for li in insert_layers:
        cos = np.concatenate(store[li]["cos"]) if store[li]["cos"] else np.array([])
        fnorm = np.concatenate(store[li]["fnorm"])
        hnorm = np.concatenate(store[li]["hnorm"])
        mean_cos = float(cos.mean())
        med_cos = float(np.median(cos))
        frac_orth = float((np.abs(cos) < 0.05).mean())
        frac_neg = float((cos < -0.1).mean())
        frac_pos = float((cos > 0.1).mean())
        report["layers"][str(li)] = {
            "mean_cos": mean_cos, "median_cos": med_cos,
            "frac_near_orth": frac_orth, "frac_neg_parallel": frac_neg,
            "frac_pos_parallel": frac_pos,
            "mean_fnorm": float(fnorm.mean()), "mean_hnorm": float(hnorm.mean()),
            "n_tokens": int(cos.size),
        }

        # ─── null controls: random direction + shuffled modality ───
        dim = int(store[li].get("dim", 0))
        cos_rand = np.concatenate(store[li]["cos_rand"])
        ctrl = {"hidden_dim": dim,
                "theoretical_random_std": float(1.0 / np.sqrt(dim)) if dim else None,
                "real": summ(cos),
                "random_direction": summ(cos_rand)}
        if store[li]["cos_shuf"]:
            cos_shuf = np.concatenate(store[li]["cos_shuf"])
            fnorm_shuf = np.concatenate(store[li]["fnorm_shuf"])
            ctrl["shuffled_modality"] = summ(cos_shuf)
            ctrl["shuffled_modality"]["mean_fnorm"] = float(fnorm_shuf.mean())
        ctrl["fnorm_over_hnorm"] = float(fnorm.mean() / hnorm.mean())
        report["layers"][str(li)]["null_controls"] = ctrl

        # self-check: the random control must reproduce the isotropic null
        assert abs(ctrl["random_direction"]["mean"]) < 5e-3, \
            f"L{li}: random-direction mean {ctrl['random_direction']['mean']:.5f} " \
            "should be ~0 (isotropic null violated)"
        assert abs(ctrl["random_direction"]["std"] - ctrl["theoretical_random_std"]) < 5e-3, \
            f"L{li}: random-direction std {ctrl['random_direction']['std']:.5f} " \
            f"deviates from 1/sqrt(D)={ctrl['theoretical_random_std']:.5f}"

        overall_flags.append(mean_cos)
        print(f"  L{li:<7}{mean_cos:>10.4f}{med_cos:>10.4f}{frac_orth:>12.3f}"
              f"{frac_neg:>10.3f}{frac_pos:>10.3f}"
              f"{fnorm.mean():>12.2f}{hnorm.mean():>12.2f}")
        print(f"      real: sd={ctrl['real']['std']:.4f}  "
              f"|f|/|h|={ctrl['fnorm_over_hnorm']:.3f}")
        print(f"      rand: mean={ctrl['random_direction']['mean']:+.4f}  "
              f"sd={ctrl['random_direction']['std']:.4f} "
              f"(theory 1/sqrt(D)={ctrl['theoretical_random_std']:.4f})  "
              f"orth={ctrl['random_direction']['frac_near_orth']:.3f}")
        if "shuffled_modality" in ctrl:
            sm = ctrl["shuffled_modality"]
            print(f"      shuf: mean={sm['mean']:+.4f}  sd={sm['std']:.4f}  "
                  f"orth={sm['frac_near_orth']:.3f}  |f|={sm['mean_fnorm']:.2f}")

    # ─── Verdict ───
    grand_mean = float(np.mean(overall_flags))
    if abs(grand_mean) < 0.05:
        verdict = "GO (near-orthogonal: effective injection lives in orthogonal subspace)"
    elif grand_mean < -0.1:
        verdict = "GO-STRONG (negative parallel: modality is cancelling text -> explicit target)"
    elif grand_mean > 0.1:
        verdict = "NO-GO (positive parallel: modality echoes text -> link to ESMC redundancy)"
    else:
        verdict = "WEAK (mixed: DDG plausible but effect may be small; inspect per-layer)"
    report["grand_mean_cos"] = grand_mean
    report["verdict"] = verdict
    print(f"\n  GRAND MEAN cos = {grand_mean:+.4f}")
    print(f"  VERDICT: {verdict}")

    out = os.path.join(OUTPUT_DIR, f"align_{dataset}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved -> {out}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["meld", "iemocap"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_blocks", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=400)
    a = p.parse_args()
    run(a.dataset, a.checkpoint, a.n_blocks, a.max_samples)
