# -*- coding: utf-8 -*-
"""
Generate gate evidence figure for the paper (2-panel, publication-ready):
- Panel (a): Converged gate magnitudes per block/type (MELD vs IEMOCAP)
- Panel (b): Multi-seed gate stability (3 MELD seeds → all converge to same range)

Output: elsarticle/figures/fig_gate_evidence.png & .pdf (300 dpi)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

OUT_DIR = Path("elsarticle/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path("outputs")

GATE_POSITIONS = ['B0-Attn', 'B0-FF', 'B1-Attn', 'B1-FF']


def extract_gates_ordered(ckpt_path):
    """Extract |tanh(gate)| values in block order."""
    adapter_path = ckpt_path / "visual_adapter.pt"
    if not adapter_path.exists():
        return None
    state = torch.load(str(adapter_path), map_location="cpu", weights_only=True)
    ordered = []
    for k in sorted(k for k in state.keys() if "gate_attn" in k or "gate_ff" in k):
        ordered.append(abs(torch.tanh(state[k]).item()))
    return ordered


def panel_a_converged_gates(ax):
    """Bar chart: converged gate magnitudes for MELD best and IEMOCAP best."""
    meld_best = CKPT_DIR / "pilot3_dial1t_2blk_meld_nofa_mdrop30_alr8e-05_cos_focal0.5_s123_best"
    iem_best = CKPT_DIR / "pilot3_dial1t_2blk_iemocap_nofa_mdrop30_alr5e-05_s123_best"

    meld_vals = extract_gates_ordered(meld_best)
    iem_vals = extract_gates_ordered(iem_best)

    x = np.arange(len(GATE_POSITIONS))
    width = 0.35

    bars1 = ax.bar(x - width / 2, meld_vals, width, label='MELD (Visual)',
                   color='#4C78A8', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, iem_vals, width, label='IEMOCAP (Audio)',
                   color='#E45756', edgecolor='white', linewidth=0.5)

    ax.axhspan(0.08, 0.12, alpha=0.10, color='#2ca02c', zorder=0)
    ax.axhline(y=0.08, color='#2ca02c', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.axhline(y=0.12, color='#2ca02c', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.text(3.6, 0.122, 'effective\nrange', fontsize=7, color='#2ca02c',
            ha='right', va='bottom', alpha=0.8)

    ax.set_xlabel('Gate Position')
    ax.set_ylabel(r'$|\tanh(\alpha)|$')
    ax.set_title('(a) Cross-Modal Gate Convergence')
    ax.set_xticks(x)
    ax.set_xticklabels(GATE_POSITIONS)
    ax.set_ylim(0, 0.16)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f'{h:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5)


def panel_b_multiseed_stability(ax):
    """Show gates from 3 different seeds all converge to same narrow range."""
    seed_configs = {
        'Seed 123\n(69.51)': CKPT_DIR / "pilot3_dial1t_2blk_meld_nofa_mdrop30_alr8e-05_cos_focal0.5_s123_best",
        'Seed 42\n(69.23)': CKPT_DIR / "pilot3_dial1t_2blk_meld_nofa_mdrop30_alr8e-05_cos_focal0.5_s42_best",
        'Seed 3047\n(69.40)': CKPT_DIR / "pilot3_dial1t_2blk_meld_nofa_mdrop30_alr5e-05_cos_s3047_best",
    }

    colors = ['#4C78A8', '#F58518', '#72B7B2']
    x = np.arange(len(GATE_POSITIONS))
    n_seeds = len(seed_configs)
    total_width = 0.75
    bar_width = total_width / n_seeds

    all_vals = []
    for i, (label, path) in enumerate(seed_configs.items()):
        vals = extract_gates_ordered(path)
        if vals is None:
            continue
        all_vals.append(vals)
        offset = (i - (n_seeds - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=label,
                      color=colors[i], edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=6.5)

    ax.axhspan(0.08, 0.12, alpha=0.10, color='#2ca02c', zorder=0)
    ax.axhline(y=0.08, color='#2ca02c', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.axhline(y=0.12, color='#2ca02c', linestyle='--', linewidth=0.7, alpha=0.6)

    if all_vals:
        arr = np.array(all_vals)
        mean_per_pos = arr.mean(axis=0)
        std_per_pos = arr.std(axis=0)
        ax.text(0.02, 0.95, f'Mean: {arr.mean():.4f} ± {arr.std():.4f}',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Gate Position (MELD)')
    ax.set_ylabel(r'$|\tanh(\alpha)|$')
    ax.set_title('(b) Multi-Seed Gate Stability')
    ax.set_xticks(x)
    ax.set_xticklabels(GATE_POSITIONS)
    ax.set_ylim(0, 0.16)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    panel_a_converged_gates(ax1)
    panel_b_multiseed_stability(ax2)

    plt.tight_layout(pad=2.0)

    for ext in ['png', 'pdf']:
        out_path = OUT_DIR / f"fig_gate_evidence.{ext}"
        plt.savefig(out_path)
        print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    plt.close()


if __name__ == "__main__":
    main()
