# LiWa: Listen first, Wait when unsure

Code, run records, judge outputs and MELD per-utterance predictions for the paper

> **Speech as a Substitute for Future Context in Real-Time Emotion Recognition in Conversation with Large Language Models**
> F. Xiao, Y. Chen, P. Liu, D. Dai, Z. Yang, L. He, X. Mei. Submitted to *Speech Communication*.

LiWa trains one LLM (Qwen2.5-7B-Instruct with LoRA) to recognise emotion with and without the next two turns of a
dialogue, fuses a fine-tuned WavLM Base+ speech expert by late fusion, and waits for the future only on the
utterances where it is unsure. Every number in the paper can be recomputed with the commands below from the files in
this repository and the IEMOCAP predictions, which holders of an IEMOCAP licence obtain from the authors. The name *GRACE* in file and run names refers to the gated cross-attention injection baseline
of the paper.

## What is here

| Path | Content |
|---|---|
| `run_v1.py`, `future_ctx.py`, `infer.py`, `fast_scoring.py`, `backbone.py`, `config.py`, `dataset_spec.py`, `data_iemocap.py`, `data_meld.py`, ... | Text model: dual-mode LoRA training, prompts, label-word scoring |
| `_remote/smoke_textdump_future_split_20261022.py` | Per-utterance score dumps of the text models (online and offline prompts) |
| `_remote/smoke_g1expert_20261021.py`, `_remote/smoke_expert2_20261022.py` | Speech experts (frozen emotion2vec head; WavLM Base+ and WavLM Large, fine-tuned) |
| `run_pilot3_dialogue_visual.py` | Gated cross-attention injection baseline |
| `extract_meld_audio.py`, `extract_audio_emotion2vec.py`, `extract_audio_frames.py` | Data preparation: MELD audio from the official clips, emotion2vec features |
| `_remote/lock_judge_20261022.py`, `_remote/b1_grace_judge_20261022.py`, `_remote/*_2026102*.py`, `runs/lock_20261022/*.py` | Judges and analyses that turn predictions into every reported number |
| `runs/lock_20261022/lock_runs.json`, `b1_runs.json`, `z*.jobs*` | Run lists and the exact command of every run; `_remote/qd.sh` is the job queue that ran them |
| `outputs/runs/<run>/` | Per run: arguments, configuration, metrics, judge outputs and, for MELD, `preds_dev.jsonl` / `preds_test.jsonl` |
| `restore_gold.py` | Puts the IEMOCAP gold labels back into the predictions (needs a licensed copy of IEMOCAP) |
| `MANIFEST.sha256` | SHA-256 of every released file |

Five seeds (123, 42, 3047, 7, 2026) are used for every system. The judge outputs behind the paper are
`outputs/runs/R20261022-211/lock_full.json` (main results, criteria, experts, waiting curve), `outputs/runs/R20261022-340/b1_grace.json`
(injection baseline), `outputs/runs/R20261025-11/meld_bestdev.json`, `outputs/runs/R20261027-01/eval/` (latency and
memory), `outputs/runs/R20261027-03/dev_checks.json`, `outputs/runs/R20261027-04/dev_threshold.json`,
`outputs/runs/R20261027-05/expert_layer_weights.json` and the JSON files in `runs/lock_20261022/`.

## What is not here

- IEMOCAP audio, transcripts, features, gold labels and per-utterance predictions. The IEMOCAP licence forbids
  passing on the data or derivatives of it. Holders of an IEMOCAP licence can request the 82 prediction files from
  the corresponding authors; they come without the `gold` field, which `restore_gold.py` puts back from the
  licensed corpus. The utterance texts are removed from `runs/lock_20261022/lock_cpu_analyses.json`.
- MELD data; download it from the official source.
- Model weights and training logs. Checkpoints of the WavLM Large experts and of the MELD WavLM Base+ experts were
  not kept; their MELD predictions are included here and their IEMOCAP predictions are among the files available
  on request. Other checkpoints are available from the corresponding authors.

## Recomputing the paper's numbers

Requirements: Python 3.12 with NumPy (the judges need nothing else). Training used PyTorch 2.9.0,
Transformers 4.56.1 and PEFT 0.18.0 on one NVIDIA RTX 6000D.

1. **IEMOCAP predictions and gold labels.** Copy the IEMOCAP prediction files obtained from the authors into
   `outputs/runs/`, then, with a licensed IEMOCAP in the InstructERC format (`iemocap.pkl`):
   `GRACE_DATA_ROOT=<dir containing instructerc_iemocap/> python restore_gold.py`
2. **Timing and repeated lines.** `_remote/iemocap_align_20261018.py` builds `runs/iemocap_align_20261018.json`
   from the raw IEMOCAP release. Set `ROOT` in that script, and `RAW` in `_remote/lock_judge_20261022.py` and in
   `runs/lock_20261022/lock_cpu_analyses_20261025.py`, to your copy.
3. **MELD timestamps.** Put the official `train_sent_emo.csv`, `dev_sent_emo.csv` and `test_sent_emo.csv` into
   `datasets/meld_csv/`.
4. **Run the judges** from the repository root with `PYTHONPATH=.`:

```bash
python _remote/lock_judge_20261022.py --stage full --out out/lock_full.json
python _remote/b1_grace_judge_20261022.py --out out/b1_grace.json
python _remote/lock_meld_bestdev_20261025.py --out out/meld_bestdev.json
python runs/lock_20261022/lock_cpu_analyses_20261025.py --out out/lock_cpu_analyses.json
python runs/lock_20261022/lock_ensemble_row_20261025.py --out out/lock_ensemble_row.json
python runs/lock_20261022/lock_meld_perclass_20261025.py --out out/lock_meld_perclass.json
python _remote/b1_dev_check_20261027.py --out out/b1_dev_check.json
python runs/lock_20261022/dev_checks_20261027b.py --out out/dev_checks.json
python runs/lock_20261022/dev_threshold_20261027.py --out out/dev_threshold.json
```

Before release, these nine outputs were recomputed in a fresh copy of this repository, with the IEMOCAP prediction
files added and steps 1 to 3 done, and compared byte for byte with the files behind the paper: all nine are
identical. Latency, memory, parameter counts, data statistics and the
expert layer weights need the models and the corpora and are reported as run.

## Training

Each run's command is in its job file and in `outputs/runs/<run>/args.txt` and `script.txt`. Paths default to the
authors' machines in `config.py`; set `GRACE_MODEL_PATH`, `GRACE_DATA_ROOT` and `GRACE_OUTPUT_DIR` to override them.
The LLM and the WavLM encoders come from their official Hugging Face repositories (`Qwen/Qwen2.5-7B-Instruct`,
`microsoft/wavlm-base-plus`, `microsoft/wavlm-large`); emotion2vec features are extracted with
`extract_audio_emotion2vec.py` and `extract_audio_frames.py`.

## License

The code, run records and judge outputs in this repository are released under the MIT License (`LICENSE`).
MELD and IEMOCAP, and anything derived from them, remain under the terms of their own licences.

## Citation

Please cite the paper above once it is published; the reference will be added here.
