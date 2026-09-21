"""
B2 — Gradient-conflict analysis between the modality-present and modality-absent
working points (zero-retraining; reuses the real model + best checkpoints).

Motivation (connects to DDG / Contribution 1 & 2):
  Modality-robust training (modality dropout) jointly optimizes two objectives on
  the SAME shared parameters theta = {LoRA on the frozen LLM} + {adapter}:
    - L_full : predict emotion WITH the auxiliary modality injected
    - L_zero : predict emotion with the modality ABSENT (mdrop -> zero injection)
  If the two per-parameter gradients g_full = dL_full/dtheta and g_zero = dL_zero/dtheta
  are non-conflicting (cos(g_full, g_zero) >= ~0) the robust-training objective is
  essentially "free" — no tug-of-war on shared weights. A negative cosine would mean
  the two working points fight over the same directions in parameter space.

  We compare a BASELINE checkpoint (standard gated injection) against a DDG checkpoint
  (orthogonal/parallel decomposed injection). The DDG hypothesis predicts the orthogonal
  injection keeps gradient interference bounded (cosine not more negative than baseline),
  giving an "amplitude -> direction -> optimization-geometry" three-layer story.

No training. Per batch: one full-modal fwd/bwd + one zero-modal fwd/bwd, collect grads,
compute cosine over the concatenated parameter vector and per group (LoRA / adapter).
"""
import argparse
import json
import os

import numpy as np
import torch

import config as C
from dataset_spec import get_spec
from infer import load_model_and_tokenizer
from run_v1 import seed_all, setup_lora, build_train_example, collate_batch
from run_pilot3_dialogue_visual import (
    DialogueVisualAdapter, install_hooks, _collate_dialogue_feats,
    load_modality_feats, N_FRAMES_VISUAL, N_FRAMES_AUDIO,
)

OUTPUT_DIR = "outputs/gradient_conflict"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _flat_grads(params):
    """Concatenate grads of the given params into one 1-D tensor (missing grads -> zeros)."""
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), device=p.device))
        else:
            chunks.append(p.grad.detach().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(1)


def _cosine(a, b):
    na, nb = a.norm(), b.norm()
    if na.item() < 1e-12 or nb.item() < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


@torch.enable_grad()
def _grad_for_condition(model, adapter, batch, turn_feats_batch, device, zero_modal):
    """Run one fwd/bwd and return grads for (lora_params, adapter_params).

    zero_modal=True forces the mdrop path (zero injection) exactly as during training.
    Model stays in eval() (deterministic, LoRA dropout off); only the adapter mode is
    toggled so compute_visual_tokens() returns zeros under mdrop.
    """
    model.zero_grad(set_to_none=True)
    adapter.zero_grad(set_to_none=True)

    if zero_modal:
        adapter.train()
        saved_p = adapter.modality_dropout
        adapter.modality_dropout = 1.0   # always drop -> zero injection
    else:
        adapter.eval()

    adapter.set_visual(turn_feats_batch)
    input_ids, attn, labels = (t.to(device) for t in batch)
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    out.loss.backward()

    lora_params = [p for p in model.parameters() if p.requires_grad]
    ad_params = list(adapter.parameters())
    g_lora = _flat_grads(lora_params)
    g_ad = _flat_grads(ad_params)

    if zero_modal:
        adapter.modality_dropout = saved_p
        adapter.eval()

    loss_val = float(out.loss.detach())
    model.zero_grad(set_to_none=True)
    adapter.zero_grad(set_to_none=True)
    return g_lora, g_ad, loss_val


@torch.no_grad()
def _load(dataset, checkpoint, n_blocks, direction_decomp):
    if n_blocks == 4:
        insert_layers = [4, 9, 14, 18]
    else:
        insert_layers = [9, 18]
    seed_all(C.SEED)
    spec = get_spec(dataset)
    model, tokenizer, _ = load_model_and_tokenizer(for_training=True)
    model = setup_lora(model)
    device = model.device
    model.load_adapter(checkpoint, adapter_name="default")

    adapter = DialogueVisualAdapter(
        n_blocks=n_blocks, n_prev_turns=1, use_frame_attn=False,
        dialogue_dropout=0.0, modality_dropout=0.0, feat_noise=0.0,
        direction_decomp=direction_decomp,
    ).to(device).float()
    va_state = torch.load(os.path.join(checkpoint, "visual_adapter.pt"),
                          weights_only=False)
    adapter.load_state_dict(va_state)
    adapter.eval()
    install_hooks(model, adapter, insert_layers)
    model.eval()
    return model, tokenizer, adapter, spec, device


def run(dataset, checkpoint, n_blocks, direction_decomp, max_batches, batch_size, tag):
    modality = "audio" if dataset == "iemocap" else "visual"
    n_frames = N_FRAMES_AUDIO if modality == "audio" else N_FRAMES_VISUAL

    print(f"=== B2 Gradient Conflict: {dataset} ({'DDG' if direction_decomp else 'baseline'}) ===")
    print(f"  ckpt: {checkpoint}")
    print(f"  batches: {max_batches} x bs{batch_size}")

    model, tokenizer, adapter, spec, device = _load(
        dataset, checkpoint, n_blocks, direction_decomp)
    D = spec.get_data_module()
    train_data = D.load_split("train")
    feat_tr = load_modality_feats(spec, "train", modality)

    cos_all, cos_lora, cos_ad = [], [], []
    lf_list, lz_list = [], []
    i = 0
    used = 0
    while used < max_batches and i + batch_size <= len(train_data):
        samples = train_data[i:i + batch_size]
        i += batch_size
        try:
            examples = [build_train_example(tokenizer, s) for s in samples]
        except Exception:
            continue
        batch = collate_batch(tokenizer, examples)
        turn_feats_batch = _collate_dialogue_feats(
            samples, feat_tr, device, adapter.n_prev_turns, n_frames)

        g_lora_f, g_ad_f, lf = _grad_for_condition(
            model, adapter, batch, turn_feats_batch, device, zero_modal=False)
        g_lora_z, g_ad_z, lz = _grad_for_condition(
            model, adapter, batch, turn_feats_batch, device, zero_modal=True)

        g_full = torch.cat([g_lora_f, g_ad_f])
        g_zero = torch.cat([g_lora_z, g_ad_z])
        cos_all.append(_cosine(g_full, g_zero))
        cos_lora.append(_cosine(g_lora_f, g_lora_z))
        cos_ad.append(_cosine(g_ad_f, g_ad_z))
        lf_list.append(lf); lz_list.append(lz)
        used += 1
        if used % 20 == 0:
            print(f"  [{used}/{max_batches}] cos_all={np.nanmean(cos_all):+.4f} "
                  f"cos_lora={np.nanmean(cos_lora):+.4f} cos_ad={np.nanmean(cos_ad):+.4f}")

    def _stat(x):
        arr = np.array(x, dtype=float)
        arr = arr[~np.isnan(arr)]
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "median": float(np.median(arr)), "n": int(arr.size),
                "frac_neg": float((arr < 0).mean())}

    report = {
        "dataset": dataset, "checkpoint": checkpoint,
        "variant": "ddg" if direction_decomp else "baseline",
        "n_batches": used, "batch_size": batch_size,
        "cos_all": _stat(cos_all),
        "cos_lora": _stat(cos_lora),
        "cos_adapter": _stat(cos_ad),
        "loss_full_mean": float(np.mean(lf_list)),
        "loss_zero_mean": float(np.mean(lz_list)),
    }
    print("\n  ---- RESULT ----")
    for k in ("cos_all", "cos_lora", "cos_adapter"):
        s = report[k]
        print(f"  {k:<12} mean={s['mean']:+.4f} ± {s['std']:.4f}  "
              f"median={s['median']:+.4f}  frac_neg={s['frac_neg']:.3f}  (n={s['n']})")
    print(f"  loss_full={report['loss_full_mean']:.4f}  loss_zero={report['loss_zero_mean']:.4f}")

    out = os.path.join(OUTPUT_DIR, f"gradconf_{tag}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  saved -> {out}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["meld", "iemocap"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_blocks", type=int, default=2)
    p.add_argument("--direction_decomp", action="store_true")
    p.add_argument("--max_batches", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--tag", required=True, help="short id for output json, e.g. meld_baseline")
    a = p.parse_args()
    run(a.dataset, a.checkpoint, a.n_blocks, a.direction_decomp,
        a.max_batches, a.batch_size, a.tag)
