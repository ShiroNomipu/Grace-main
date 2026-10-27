"""Future context and the one-model-two-modes training (2026-10-21; F / DM gates, runs/CPU分析_20261021 §5-§6).
Off unless a trainer is given --future_mode. Wraps a data module's load_split like cue_text.py does.

Prompt (one template for every mode, so the modes differ only in the future section):
    system : ... identify the emotion of the TARGET utterance ...
    user   : Conversation before the target: <past turns or (none)>
             TARGET utterance: Speaker_X: <text>
             Conversation after the target: <next k turns | (not available) | (none: the conversation ends here)>
             Question: What is the emotion of the TARGET utterance spoken by Speaker_X? ...
The past turns are those of the original window (context_window counts the target, so 8 -> 7 past turns).
Future turns are the next k samples of the same dialogue in utterance order.
Modes:
    online : future section "(not available)"            -> past-only model with the new template (F control)
    full   : next k turns                                   -> F treatment
    dual   : both variants kept on every sample; training draws online with probability P_DROP per sample and
             epoch (own RNG, global random untouched); evaluation reports both modes (DM gate).
"""
import random

MODE = None      # set by the trainer
K = 2
P_DROP = 0.5
SEED = None      # set by the trainer (run seed); used by epoch_messages
NOT_AVAILABLE = "(not available)"
ENDS_HERE = "(none: the conversation ends here)"


def system_prompt(spec):
    return ("You are an expert in emotion recognition in conversations. "
            "Read the conversation and identify the emotion of the TARGET utterance. "
            "Respond with exactly one word chosen from: "
            + ", ".join(spec.label2word[lab] for lab in spec.label_list) + ".")


def build(spec, past, target, future):
    """past / future: list of (speaker, text); future None = not available."""
    spk, txt = target
    p = "\n".join(f"Speaker_{s}: {t}" for s, t in past) if past else "(none)"
    if future is None:
        f = NOT_AVAILABLE
    else:
        f = "\n".join(f"Speaker_{s}: {t}" for s, t in future) if future else ENDS_HERE
    user = (f"Conversation before the target:\n{p}\n\n"
            f"TARGET utterance:\nSpeaker_{spk}: {txt}\n\n"
            f"Conversation after the target:\n{f}\n\n"
            f"Question: What is the emotion of the TARGET utterance spoken by Speaker_{spk}? Answer with one word.")
    return [{"role": "system", "content": system_prompt(spec)}, {"role": "user", "content": user}]


def apply(samples, spec, mode, k):
    by_dlg = {}
    for s in samples:
        by_dlg.setdefault(s["dialogue_id"], []).append(s)
    for lst in by_dlg.values():
        lst.sort(key=lambda s: int(s["utt_id"]))
        for i, s in enumerate(lst):
            past = list(s["context_turns"][:-1])
            target = (s["speaker"], s["text"])
            assert s["context_turns"][-1][1] == s["text"], (s["dialogue_id"], s["utt_id"])
            fut = [(x["speaker"], x["text"]) for x in lst[i + 1:i + 1 + k]]
            s["messages_online"] = build(spec, past, target, None)
            s["messages_full"] = build(spec, past, target, fut)
            s["n_future"] = len(fut)
            s["messages"] = s["messages_online"] if mode == "online" else s["messages_full"]
    return samples


def wrap_load_split(orig, spec, mode, k):
    assert mode in ("online", "full", "dual"), mode

    def wrapped(split, *a, **kw):
        return apply(orig(split, *a, **kw), spec, mode, k)
    return wrapped


def epoch_messages(samples, epoch, seed):
    """dual mode: which variant each training sample uses in this epoch (list of 'online' / 'full')."""
    rng = random.Random(f"{seed}-{epoch}-future")
    return ["online" if rng.random() < P_DROP else "full" for _ in samples]


def with_mode(samples, mode):
    return [dict(s, messages=s[f"messages_{mode}"]) for s in samples]
