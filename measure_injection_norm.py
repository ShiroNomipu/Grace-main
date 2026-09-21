"""Is the constant-gate sweep really a sweep over injection amplitude?

Section 3 reports that the cross-attention output has norm 0.47-0.93 relative
to the residual stream, which is what licenses reading the gate coefficient as
the amplitude the backbone sees. That was measured on the standard-gated model.
If a model trained with a large fixed gate compensated by shrinking its output
projection, its effective displacement would not match its nominal gate and the
sweep would not be manipulating what it claims to.

This measures the displacement each block actually writes into the residual
stream, ||blk(h) - h|| / ||h||, at both injection layers. It runs the validated
robustness evaluator unchanged and only wraps the adapter blocks on the way
past, so the forward path, checkpoint loading and gate-form guard are the same
ones used for every other number in the paper. The clean wF1 it prints should
reproduce the corresponding table entry; if it does not, the measurement is not
being taken on the configuration it claims.

Usage mirrors eval_robustness_matrix.py, plus --norm_output:

  python measure_injection_norm.py --checkpoint <dir> --adapter_type step3 \
      --dataset iemocap --gate_mode fixed --conditions clean \
      --norm_output outputs/injnorm_gate0.25.json
"""
import json
import statistics
import sys

import torch

import eval_robustness_matrix as E

STATS = {}


def _wrap_blocks(adapter):
    """Record ||blk(h) - h|| / ||h|| for every call of every injection block."""
    for i, blk in enumerate(adapter.blocks):
        if getattr(blk, "_norm_wrapped", False):
            continue
        original = blk.forward

        def forward(h, *a, _i=i, _f=original, **kw):
            out = _f(h, *a, **kw)
            with torch.no_grad():
                hf = h.float()
                d = (out.float() - hf).norm(dim=-1)
                n = hf.norm(dim=-1).clamp_min(1e-6)
                STATS.setdefault(f"block{_i}", []).append((d / n).mean().item())
            return out

        blk.forward = forward
        blk._norm_wrapped = True


_original_install = E.install_hooks


def install_hooks(model, visual_adapter, insert_layers):
    _wrap_blocks(visual_adapter)
    return _original_install(model, visual_adapter, insert_layers)


E.install_hooks = install_hooks

if __name__ == "__main__":
    argv, out_path = [], None
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--norm_output":
            out_path = next(it)
        else:
            argv.append(a)
    sys.argv = [sys.argv[0]] + argv

    E.main()

    print("\n=== effective displacement, ||blk(h) - h|| / ||h|| ===")
    summary = {}
    for k, v in sorted(STATS.items()):
        summary[k] = {"mean": statistics.fmean(v),
                      "median": statistics.median(v),
                      "n_calls": len(v)}
        print(f"  {k}: mean {summary[k]['mean']:.4f}  "
              f"median {summary[k]['median']:.4f}  over {len(v)} calls")
    if out_path:
        json.dump({"argv": argv, "displacement": summary}, open(out_path, "w"), indent=2)
        print(f"written to {out_path}")
