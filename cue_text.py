"""H2 cue text (2026-10-18): append a verbalised acoustic cue to the target utterance in the prompt.

Off unless a script gets --cue_file. It wraps a data module's load_split; the data modules are unchanged.
Cue file: JSON {"rule": ..., "cues": {"<dialogue_id>|<utt_id>": "[voice: ...]"}}. A sample without an
entry (or with an empty string) keeps its prompt unchanged. The cue goes at the end of the target
utterance's line, i.e. right before the "Question:" block that build_messages appends.
"""
import json

MARK = "\n\nQuestion:"


def load_cues(path):
    return json.load(open(path))["cues"]


def apply(samples, cues, split="", path=""):
    n = 0
    for s in samples:
        c = cues.get(f"{s['dialogue_id']}|{s['utt_id']}")
        if not c:
            continue
        u = s["messages"][-1]
        assert u["role"] == "user" and u["content"].count(MARK) == 1, "unexpected prompt layout"
        u["content"] = u["content"].replace(MARK, f" {c}{MARK}")
        n += 1
    print(f"[cue_text] {split}: cue added to {n}/{len(samples)} samples ({path})", flush=True)
    return samples


def wrap_load_split(orig, path):
    cues = load_cues(path)

    def wrapped(split, *a, **k):
        return apply(orig(split, *a, **k), cues, split, path)

    return wrapped
