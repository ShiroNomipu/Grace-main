"""Parameter counts for the paper (2026-10-27, CPU only, reads tensor shapes only). Sources: the lock / B1 artifacts the
paper numbers come from, seed 123. Backbone: safetensors headers of models/Qwen2.5-7B-Instruct. LoRA: adapter of the
IEMOCAP dual run R20261021-102 (_lora_last). E2: expert_bf16.pt of R20261021-21 (WavLM Base+ encoder + head).
GRACE: visual_adapter.pt of R20261022-306 (_last; the LoRA of GRACE has the same shape as ours and is counted apart).
Usage (repo root): python <this> --out <json>"""
import argparse, glob, json, struct
import numpy as np
import torch
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
def st_count(paths):
    n = 0
    for p in paths:
        with open(p, "rb") as f:
            h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        n += sum(int(np.prod(v["shape"])) for k, v in h.items() if k != "__metadata__")
    return n
def pt_count(p): return sum(v.numel() for v in torch.load(p, map_location="cpu", weights_only=True).values() if torch.is_tensor(v))
g = glob.glob("outputs/runs/R20261022-306/pilot3_*_last")[0]
out = {"design": __doc__,
       "backbone_qwen2.5_7b": st_count(sorted(glob.glob("models/Qwen2.5-7B-Instruct/*.safetensors"))),
       "lora_ours_R20261021-102": st_count(["outputs/runs/R20261021-102/v1f_dual_c16_s123_lora_last/adapter_model.safetensors"]),
       "e2_R20261021-21": pt_count("outputs/runs/R20261021-21/expert_bf16.pt"),
       "grace_injection_R20261022-306": sum(v.numel() for v in torch.load(f"{g}/visual_adapter.pt", map_location="cpu", weights_only=False).values() if torch.is_tensor(v)),
       "grace_lora_R20261022-306": st_count([f"{g}/adapter_model.safetensors"])}
sd = torch.load("outputs/runs/R20261021-21/expert_bf16.pt", map_location="cpu", weights_only=True)
out["e2_frozen_cnn_feature_encoder"] = sum(v.numel() for k, v in sd.items() if k.startswith("enc.feature_extractor"))
out["e2_head"] = sum(v.numel() for k, v in sd.items() if k.startswith("head"))
out["trainable_ours"] = out["lora_ours_R20261021-102"] + out["e2_R20261021-21"] - out["e2_frozen_cnn_feature_encoder"]
out["trainable_grace"] = out["grace_injection_R20261022-306"] + out["grace_lora_R20261022-306"]
out["trainable_grace+e2"] = out["trainable_grace"] + out["e2_R20261021-21"] - out["e2_frozen_cnn_feature_encoder"]
json.dump(out, open(a.out, "w"), indent=1); print({k: v for k, v in out.items() if k != "design"})
