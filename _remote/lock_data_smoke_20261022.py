"""Lock batch CPU smoke (2026-10-22): for iemocap and meld at ctx 16 and future k 2, future_ctx.apply on train / dev /
test (it asserts that the target is the last context turn); counts, n_future histogram, label counts, and token lengths
of the online and full prompts plus the answer under the Qwen2.5-7B-Instruct chat template. Training examples longer
than MAX_SEQ_LEN are left-truncated by run_v1.build_train_example, so their number is reported. No model weights are
loaded. Usage (repo root, env_grace.sh sourced, GRACE_MODEL_PATH = the 7B): python <this>"""
import collections, json, os, sys
sys.path.insert(0, os.getcwd())
import config as C
import future_ctx
from dataset_spec import get_spec
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(C.MODEL_PATH)
out = {"model_path": C.MODEL_PATH, "max_seq_len": C.MAX_SEQ_LEN}
for ds in ("iemocap", "meld"):
    spec = get_spec(ds); D = spec.get_data_module()
    for split in ("train", "dev", "test"):
        S = future_ctx.apply(D.load_split(split, context_window=16), spec, "dual", 2)
        o = {"n": len(S), "n_dialogues": len({s["dialogue_id"] for s in S}),
             "n_future": dict(sorted(collections.Counter(s["n_future"] for s in S).items())),
             "labels": dict(sorted(collections.Counter(s["label_id"] for s in S).items()))}
        for m in ("online", "full"):
            v = sorted(len(tok.apply_chat_template(s[f"messages_{m}"] + [{"role": "assistant", "content": s["label_word"]}], tokenize=True)) for s in S)
            o[f"len_{m}"] = {"median": v[len(v) // 2], "p99": v[int(0.99 * len(v))], "max": v[-1], "over_max_seq_len": sum(x > C.MAX_SEQ_LEN for x in v)}
        out[f"{ds}_{split}"] = o
        print(ds, split, json.dumps(o), flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w"), indent=1)
