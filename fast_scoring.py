"""Prefix-cached candidate scoring (evaluation speed-up for the 7B line, 2026-10-15).

Computes the same quantity as infer.score_candidates and
run_pilot3_dialogue_visual.score_candidates -- the mean log-probability of each
candidate's answer tokens after the prompt -- but runs the shared prompt once
with a KV cache, then scores every candidate's answer tokens in one batched
forward on top of it. log_softmax is taken only at positions that predict
answer tokens.

The modality adapter is per-token (cross-attention from each text position to
the modality tokens, then a position-wise FF), so a cached prefix sees exactly
what it would in a full forward. Matmul shapes differ from the per-candidate
path, so scores agree to rounding rather than bit for bit. Callers reach this
only through an explicit --fast_eval flag, admitted on a regression check
against the per-candidate path (runs/registry.jsonl, R20261015 engineering).
"""
import torch


@torch.no_grad()
def candidate_avg_logps(model, prompt_ids, cands, device, on_batch=None):
    """prompt_ids: list[int]; cands: [(label, answer_ids)] from
    infer._build_candidate_token_ids. on_batch(n) runs before the batched
    continuation so the caller can broadcast per-sample side inputs (the
    modality adapter's features) to n rows. Returns one float per candidate."""
    answers = [ans for _, ans in cands]
    n, P = len(answers), len(prompt_ids)
    ids = torch.tensor([prompt_ids], device=device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=True, logits_to_keep=1)
    first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    totals = [first[a[0]].item() for a in answers]

    L = max(len(a) for a in answers) - 1
    if L > 0:
        # Right-padded continuations; padding sits after every real token, so
        # causality already hides it, and its logits are never read.
        cont = torch.full((n, L), answers[0][0], dtype=torch.long, device=device)
        mask = torch.zeros((n, P + L), dtype=torch.long, device=device)
        mask[:, :P] = 1
        for i, a in enumerate(answers):
            k = len(a) - 1
            if k:
                cont[i, :k] = torch.tensor(a[:-1], device=device)
                mask[i, P:P + k] = 1
        past = out.past_key_values
        past.batch_repeat_interleave(n)
        if on_batch is not None:
            on_batch(n)
        logits = model(input_ids=cont, attention_mask=mask,
                       past_key_values=past, use_cache=True).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        for i, a in enumerate(answers):
            for j in range(1, len(a)):
                totals[i] += logp[i, j - 1, a[j]].item()
    return [t / len(a) for t, a in zip(totals, answers)]
