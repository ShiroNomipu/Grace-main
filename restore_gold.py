"""Put the IEMOCAP gold labels back into the released prediction files.

The public repository leaves the "gold" field out of every IEMOCAP prediction file, because the labels belong to
the licensed corpus. With a licensed copy of IEMOCAP in the InstructERC format that data_iemocap.py reads, this
script restores the field in place, at its original position, so that the judge scripts reproduce the paper.
Usage (repository root): GRACE_DATA_ROOT=<directory holding instructerc_iemocap/> python restore_gold.py
"""
import glob
import json

import data_iemocap as D

gold = {}
for split in ("dev", "test"):
    for s in D.load_split(split):
        gold[f"{s['dialogue_id']}|{s['utt_id']}"] = int(s["label_id"])

n_files = n_rows = 0
for path in sorted(glob.glob("outputs/runs/**/preds_*.jsonl", recursive=True)):
    rows = [json.loads(line) for line in open(path, encoding="utf-8").read().splitlines()]
    if not rows or not str(rows[0]["dialogue_id"]).startswith("Ses") or "gold" in rows[0]:
        continue
    out = []
    for r in rows:
        after = "speaker" if "speaker" in r else "utt_id"
        new = {}
        for key, value in r.items():
            new[key] = value
            if key == after:
                new["gold"] = gold[f"{r['dialogue_id']}|{r['utt_id']}"]
        out.append(json.dumps(new))
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    n_files += 1
    n_rows += len(out)
print(f"restored the gold label in {n_files} files, {n_rows} rows")
