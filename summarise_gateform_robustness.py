"""Tabulates outputs/gateform_robustness/*.json into the comparison the review asks for.

Reports, per dataset and gate form, the clean wF1 and the drop from clean under each
perturbation. The decisive column is Zero (complete modality absence): if the constant
gate degrades further than the tanh gate there, gating is doing work on the robustness
axis that the clean-only Table 10 cannot see.
"""
import glob
import json
import os
from collections import defaultdict

CONDS = ["clean", "noise_0.5", "noise_1.0", "zero"]
MODES = ["tanh", "linear", "fixed"]

rows = defaultdict(dict)
for path in sorted(glob.glob(os.path.join("outputs", "gateform_robustness", "*.json"))):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    tag = os.path.splitext(os.path.basename(path))[0]
    dataset, mode, seed = tag.rsplit("_", 2)
    # eval_robustness_matrix stores wF1 as a fraction; the paper reports points.
    rows[(dataset, seed)][mode] = {c: 100.0 * d["conditions"][c]["weighted_f1"]
                                   for c in CONDS if c in d["conditions"]}

if not rows:
    raise SystemExit("no results yet -- run run_gateform_robustness.ps1 first")

for (dataset, seed) in sorted(rows):
    got = rows[(dataset, seed)]
    print(f"\n=== {dataset}  seed {seed[1:] if seed.startswith('s') else seed} ===")
    head = f"{'gate form':<10}" + "".join(f"{c:>20}" for c in CONDS)
    print(head)
    print("-" * len(head))
    for mode in MODES:
        if mode not in got:
            continue
        vals = got[mode]
        clean = vals.get("clean")
        cells = ""
        for c in CONDS:
            if c not in vals:
                cells += f"{'--':>20}"
            elif c == "clean":
                cells += f"{vals[c]:>20.2f}"
            else:
                cells += f"{vals[c]:>13.2f} ({vals[c] - clean:+.2f})"
        print(f"{mode:<10}{cells}")

    # The comparison the framing decision turns on.
    if "tanh" in got and "fixed" in got and "zero" in got["tanh"] and "zero" in got["fixed"]:
        d_tanh = got["tanh"]["zero"] - got["tanh"]["clean"]
        d_fixed = got["fixed"]["zero"] - got["fixed"]["clean"]
        gap = d_tanh - d_fixed
        verdict = ("gate helps under absence" if gap > 0.3 else
                   "constant gate is no worse" if abs(gap) <= 0.3 else
                   "constant gate degrades LESS -- gate framing not defensible")
        print(f"\n  degradation under Zero: tanh {d_tanh:+.2f}pp, "
              f"constant {d_fixed:+.2f}pp, difference {gap:+.2f}pp -> {verdict}")
