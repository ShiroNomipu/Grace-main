"""Candidate A (2026-10-19): annotation-distribution-aware training, soft-label auxiliary loss.

Off unless a trainer gets --soft_file (card: runs/候选A_分歧感知_提案卡_2026-10-18.md, approved 10-18).
  L      = CE(majority label word, all answer tokens)          # the trainers' existing loss, unchanged
         + LAMBDA * L_soft
  L_soft = - sum_c q_c * log softmax_c( z[first token of label word c] )
           z = logits at the position that predicts the FIRST answer token; softmax over the K label words only;
           q = external-evaluator vote distribution of the utterance (train split only; see the builder).
Inference is unchanged (candidate scoring).

Soft file: JSON {"rule": ..., "classes": [label_list order], "q": {"<dialogue_id>|<utt_id>": [K floats]}}.
Every train sample must have an entry; the builder already falls back to the one-hot of the dataset label.
run_v1.forward_batch only returns the loss, so for that trainer the logits are taken from a forward hook
(stored only while the model is in train mode); run_pilot3 passes out.logits directly.
"""
import atexit
import json
import math

import torch

LAMBDA = 1.0   # fixed by the card; deliberately not a CLI option
PATH = None    # set by the trainer when --soft_file is given
CURRICULUM = False  # round 2 (2026-10-19): set by --curriculum; easy-to-hard order within each epoch


class _State:
    pass


_S = None


def label_first_ids(tokenizer, spec, messages):
    """First answer token of each label word, built exactly like run_v1.build_train_example (chat template
    with the assistant answer appended), so the tokenisation context is the one seen in training."""
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    ids = []
    for lab in spec.label_list:
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": spec.label2word[lab]}],
            add_generation_prompt=False, tokenize=True)
        assert full[:len(prompt)] == prompt, "chat template: prompt is not a prefix of prompt+answer"
        ids.append(full[len(prompt)])
    assert len(set(ids)) == len(ids), f"label words share a first token: {ids}"
    return ids


def setup(tokenizer, spec, samples, examples, model=None, log_path=None):
    """samples[j] <-> examples[j] = (input_ids, labels) as built by build_train_example."""
    global _S
    blob = json.load(open(PATH))
    assert blob["classes"] == list(spec.label_list), (blob["classes"], spec.label_list)
    qmap = blob["q"]
    assert len(samples) == len(examples)
    first = label_first_ids(tokenizer, spec, samples[0]["messages"])
    pos, q, n_onehot = [], [], 0
    for s, (ids, labels) in zip(samples, examples):
        k = f"{s['dialogue_id']}|{s['utt_id']}"
        assert k in qmap, f"no soft target for train sample {k}"
        a = next(i for i, l in enumerate(labels) if l != -100)        # index of the first answer token
        assert ids[a] == first[spec.label2id[s["label"]]], f"{k}: first answer token mismatch"
        pos.append(a - 1)                                              # logits here predict ids[a]
        v = qmap[k]
        assert len(v) == len(first) and min(v) >= 0 and abs(sum(v) - 1) < 1e-6, k
        n_onehot += max(v) == 1.0
        q.append(v)
    _S = _State()
    _S.first, _S.pos, _S.q = first, pos, torch.tensor(q, dtype=torch.float32)
    _S.logits, _S.log = None, []
    _S.ent = [round(-sum(p * math.log(p) for p in v if p > 0), 6) for v in q]   # curriculum key
    if model is not None:
        def _hook(mod, inp, out):
            if mod.training:
                _S.logits = out.logits
        _S.handle = model.register_forward_hook(_hook)
    if log_path:
        atexit.register(_dump, log_path)
    lab_mass = sum(_S.q[j, spec.label2id[s["label"]]].item() for j, s in enumerate(samples)) / len(samples)
    print(f"[soft_label] {PATH}: {len(samples)} train samples, lambda={LAMBDA:g}, first ids {first} "
          f"({[tokenizer.decode([i]) for i in first]}), one-hot q {n_onehot}/{len(samples)}, "
          f"mean q[label]={lab_mass:.3f}, logits from {'forward hook' if model is not None else 'caller'}",
          flush=True)


def loss(idx, logits=None):
    """LAMBDA * mean_b L_soft for the train examples idx (positions in the samples list given to setup)."""
    if logits is None:
        logits, _S.logits = _S.logits, None
        assert logits is not None, "soft_label: no captured logits (hook not installed or not in train mode)"
    b = torch.arange(len(idx), device=logits.device)
    p = torch.tensor([_S.pos[j] for j in idx], device=logits.device)
    z = logits[b, p][:, _S.first].float()                              # [B, K]
    q = _S.q[list(idx)].to(logits.device)                              # [B, K]
    ls = -(q * torch.log_softmax(z, dim=-1)).sum(-1).mean()
    _S.log.append(round(ls.item(), 5))
    if len(_S.log) % 200 == 0:
        print(f"[soft_label] batches {len(_S.log)}: mean L_soft (last 200) = "
              f"{sum(_S.log[-200:]) / 200:.4f}", flush=True)
    return LAMBDA * ls


def _dump(path):
    if _S is not None and _S.log:
        with open(path, "w") as f:
            json.dump({"lambda": LAMBDA, "soft_file": PATH, "per_batch_L_soft": _S.log}, f)


def curriculum_sort(indices):
    """Round 2 (2026-10-19): in place, stable sort of this epoch's shuffled indices by the entropy of q, ascending.
    Unanimous utterances (entropy 0) come first; ties keep the shuffled order."""
    indices.sort(key=lambda j: _S.ent[j])
    e = [_S.ent[j] for j in indices]
    assert all(x <= y for x, y in zip(e, e[1:])), "curriculum order is not non-decreasing"
    h = len(e) // 2
    print(f"[curriculum] {len(e)} samples: entropy first/last {e[0]:.3f}/{e[-1]:.3f}, mean first half "
          f"{sum(e[:h]) / h:.3f}, second half {sum(e[h:]) / (len(e) - h):.3f}, zero-entropy {sum(x == 0 for x in e)}",
          flush=True)
