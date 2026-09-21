"""Summarise the IEMOCAP amplitude sweep and gate=1.0 learning-rate scan
(run_amplitude_sweep.ps1) against same-batch references.

All numbers are test weighted-F1 read from outputs/*_test_metrics.json. The
2026-07-25 batch (tanh / linear / fixed-0.1, seed 123) is the same-batch
reference for the constant-gate rows; the paper's Text-only 67.73 is quoted as
the modality-free floor. Absolute values of this batch are not on the main-table
basis (see tab:gateform caption) -- read the *shape* across amplitudes.
"""
import json
import os

OUT = "outputs"
TEXT_ONLY_PAPER = 67.73  # tab:ablation, IEMOCAP Text-only (LoRA)
BASE = "pilot3_dial1t_2blk_iemocap_nofa_mdrop30"


def wf1(tag):
    p = os.path.join(OUT, f"{tag}_test_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return 100.0 * json.load(f)["weighted_f1"]


def fmt(x):
    return "   --  " if x is None else f"{x:7.2f}"


print("=== Constant-gate amplitude sweep (IEMOCAP, seed 123, adapter lr 5e-5) ===")
rows = [
    ("tanh gate (learnable, converges ~0.10)", f"{BASE}_alr5e-05_s123"),
    ("constant 0.10", f"{BASE}_gatefixed_alr5e-05_s123"),
    ("constant 0.25", f"{BASE}_ginit0.25_gatefixed_alr5e-05_s123"),
    ("constant 0.50", f"{BASE}_ginit0.5_gatefixed_alr5e-05_s123"),
    ("constant 0.75", f"{BASE}_ginit0.75_gatefixed_alr5e-05_s123"),
    ("constant 1.00 (= no gating)", f"{BASE}_nogating_alr5e-05_s123"),
]
ref = wf1(f"{BASE}_gatefixed_alr5e-05_s123")
print(f"{'configuration':42s} {'wF1':>7s} {'vs const 0.10':>14s} {'vs Text-only 67.73':>19s}")
for name, tag in rows:
    v = wf1(tag)
    d_ref = None if (v is None or ref is None) else v - ref
    d_txt = None if v is None else v - TEXT_ONLY_PAPER
    print(f"{name:42s} {fmt(v)} {fmt(d_ref):>14s} {fmt(d_txt):>19s}")

print("\n=== gate=1.0 learning-rate scan (IEMOCAP, seed 123) ===")
for lr in ["5e-05", "1.5e-05", "5e-06"]:
    v = wf1(f"{BASE}_nogating_alr{lr}_s123")
    d_txt = None if v is None else v - TEXT_ONLY_PAPER
    print(f"adapter lr {lr:8s} {fmt(v)}   vs Text-only {fmt(d_txt)}")

p = os.path.join(OUT, "robust_backup_iemocap_s123.json")
if os.path.exists(p):
    with open(p) as f:
        c = json.load(f)["conditions"]
    clean, zero = 100 * c["clean"]["weighted_f1"], 100 * c["zero"]["weighted_f1"]
    print("\n=== Original reported checkpoint (_5ep_backup), re-evaluated ===")
    print(f"clean {clean:.2f}   zero {zero:.2f}   delta {zero - clean:+.2f}"
          f"   (paper: 72.78 -> 70.08, -2.70; local 2026-07-24: 71.51 -> 70.37, -1.14)")
