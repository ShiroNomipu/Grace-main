"""Self-distillation for the one-model-two-modes training (2026-10-21; KD gate, card runs/C15与KD闸门_运行卡_2026-10-21.md).
Off unless run_v1.py gets --future_kd LAMBDA (requires --future_mode dual). In-place distillation, as in dual-mode ASR:
in a dual-mode training batch, every sample drawn in the online mode (future dropped) also gets
    L_kd = KL(q || p)
    q = softmax over the K label words of the same model's logits on the FULL prompt of the same sample, computed in the
        same train-mode forward under no_grad (stop-gradient teacher);
    p = softmax over the K label words of the online CE forward's logits (shared with the CE loss);
both at the position that predicts the first answer token (as soft_label.py), temperature 1.
L = CE (unchanged) + LAMBDA * mean over the batch's online samples of L_kd (0 if the batch has none).
Samples drawn in the full mode get the CE loss only. Inference is unchanged.
"""
import atexit
import json

import torch

LAMBDA = 0.0   # set by the trainer (--future_kd); the card fixes 1.0
_S = None


class _State:
    pass


def setup(tokenizer, spec, train, build_example, collate, max_len, log_path=None):
    global _S
    import soft_label
    _S = _State()
    _S.first = soft_label.label_first_ids(tokenizer, spec, train[0]["messages_online"])
    assert _S.first == soft_label.label_first_ids(tokenizer, spec, train[0]["messages_full"]), "label first ids differ by mode"
    _S.tok, _S.train, _S.build, _S.collate, _S.max_len, _S.spec = tokenizer, train, build_example, collate, max_len, spec
    _S.log = []
    if log_path:
        atexit.register(_dump, log_path)
    print(f"[future_kd] lambda={LAMBDA:g}, first ids {_S.first} ({[tokenizer.decode([i]) for i in _S.first]})", flush=True)


def _first_pos(labels_row):
    """Position whose logits predict the first answer token (labels are -100 on the prompt, right padding)."""
    return int((labels_row != -100).nonzero()[0]) - 1


def loss(model, logits, input_ids, labels, batch_idx, modes, device):
    """LAMBDA * mean KL(teacher full || student online) over the online samples of the batch."""
    on = [b for b, j in enumerate(batch_idx) if modes[j] == "online"]
    if not on:
        return logits.new_zeros(())
    first = torch.tensor(_S.first, device=device)
    ps = [_first_pos(labels[b]) for b in on]
    for b, p in zip(on, ps):   # the student position predicts the gold label's first token
        assert int(input_ids[b, p + 1]) == _S.first[_S.spec.label2id[_S.train[batch_idx[b]]["label"]]]
    zs = torch.stack([logits[b, p][first] for b, p in zip(on, ps)]).float()
    t_ex = [_S.build(_S.tok, dict(_S.train[batch_idx[b]], messages=_S.train[batch_idx[b]]["messages_full"]), max_len=_S.max_len) for b in on]
    t_ids, t_attn, t_lab = (t.to(device) for t in _S.collate(_S.tok, t_ex))
    with torch.no_grad():
        t_logits = model(input_ids=t_ids, attention_mask=t_attn).logits
        pt = [_first_pos(t_lab[i]) for i in range(len(on))]
        for i, p in enumerate(pt):
            assert int(t_ids[i, p + 1]) == int(input_ids[on[i], ps[i] + 1]), "teacher and student answer tokens differ"
        zt = torch.stack([t_logits[i, p][first] for i, p in enumerate(pt)]).float()
    del t_logits
    q = torch.softmax(zt, -1)
    kl = (q * (torch.log(q.clamp_min(1e-12)) - torch.log_softmax(zs, -1))).sum(-1).mean()
    _S.log.append(round(kl.item(), 5))
    if len(_S.log) % 200 == 0:
        print(f"[future_kd] batches with online samples {len(_S.log)}: mean KL (last 200) = {sum(_S.log[-200:]) / 200:.4f}", flush=True)
    return LAMBDA * kl


def _dump(path):
    if _S is not None and _S.log:
        with open(path, "w") as f:
            json.dump({"lambda": LAMBDA, "per_batch_kl": _S.log}, f)
