"""Evaluate Pilot-3 best checkpoint on test set (MELD or IEMOCAP)."""
import argparse
import json
import os

import torch

import config as C
from dataset_spec import get_spec
from infer import load_model_and_tokenizer, compute_metrics, print_metrics
from run_v1 import seed_all, setup_lora
from run_pilot3_dialogue_visual import (
    DialogueVisualAdapter, load_modality_feats, install_hooks, evaluate,
    N_FRAMES_VISUAL, N_FRAMES_AUDIO, FEAT_DIM, HIDDEN_DIM,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(
        C.OUTPUT_DIR, "pilot3_dial1t_2blk_meld_best"))
    parser.add_argument("--dataset", default="meld", choices=["meld", "iemocap"])
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--n_prev_turns", type=int, default=1)
    parser.add_argument("--no_frame_attn", action="store_true")
    args = parser.parse_args()

    seed_all(C.SEED)
    spec = get_spec(args.dataset)
    D = spec.get_data_module()

    modality = "audio" if args.dataset == "iemocap" else "visual"
    n_frames = N_FRAMES_AUDIO if modality == "audio" else N_FRAMES_VISUAL

    if args.n_blocks == 2:
        insert_layers = [9, 18]
    elif args.n_blocks == 4:
        insert_layers = [4, 9, 14, 18]
    else:
        step = 28 // (args.n_blocks + 1)
        insert_layers = [step * (i + 1) for i in range(args.n_blocks)]

    print(f"=== Pilot-3 Test Eval [{args.dataset}] ({modality}) ===")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  blocks={args.n_blocks} at layers {insert_layers}")
    print(f"  n_prev_turns={args.n_prev_turns}, frame_attn={not args.no_frame_attn}")
    print(f"  modality={modality}, n_frames={n_frames}")

    model, tokenizer, _ = load_model_and_tokenizer(for_training=True)
    model = setup_lora(model)
    model.load_adapter(args.checkpoint, adapter_name="default")
    device = model.device

    adapter = DialogueVisualAdapter(
        n_blocks=len(insert_layers),
        n_prev_turns=args.n_prev_turns,
        use_frame_attn=not args.no_frame_attn,
    ).to(device).float()

    va_path = os.path.join(args.checkpoint, "visual_adapter.pt")
    va_state = torch.load(va_path, weights_only=False)
    adapter.load_state_dict(va_state)
    print(f"  Loaded adapter from {va_path}")

    handles = install_hooks(model, adapter, insert_layers)

    test_data = D.load_split("test")
    feat_te = load_modality_feats(spec, "test", modality)
    print(f"  test samples: {len(test_data)}, feat keys: {len(feat_te)}")

    metrics = evaluate(model, adapter, tokenizer, test_data,
                       feat_te, spec, n_frames, desc="test")
    print_metrics(metrics, spec=spec)

    tag = f"pilot3_dial{args.n_prev_turns}t_{args.n_blocks}blk_{args.dataset}"
    out_json = os.path.join(C.OUTPUT_DIR, f"{tag}_test_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_json}")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
