"""
emotion2vec_base 音频特征提取（V2 用）—— 数据源 = HF AbstractTTS/IEMOCAP（含 script，audio 内嵌）。

对齐策略（L10/续8 修订，彻底脱离 MAGTKD 本地 wav）：
  - HF parquet 行带 file(wav名)/transcription/audio.bytes。
  - 用 (dialogue_id, 规范化文本) 为键建 deque（按 file 名排序，保证可复现），
    遍历 data_iemocap.load_split 时按对话内时序顺序消费 —— 已验证对 7433 句 100% 命中
    (唯一 7146 + 歧义 287；歧义为同对话同文本短句，音频高度相似，顺序消费影响可忽略)。
  - 解码 audio.bytes → 16k 单声道 ndarray → emotion2vec_base(纯 SSL) 提 768-d。
  - 按 (dialogue_id, utt_id) 键存储，顺序与 load_split 严格一致；断言逐条对齐 + 数量。

产出：C.AUDIO_FEAT_DIR / IEMOCAP_{split}_emotion2vec.pt
  { "split", "feat_dim":768, "order_keys":[f"{did}|{utt_id}"], "feats":(N,768) f32, "labels":(N,) i64 }
V2 用法：建 key->idx，遍历文本 sample 用 f"{did}|{utt_id}" 取 feat 并断言 label 一致。
"""
import glob
import io
import os
import re
from collections import defaultdict, deque

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

import config as C
import data_iemocap as D

OUTPUT_DIR = C.AUDIO_FEAT_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_GLOB = r"datasets/iemocap_hf/data/*.parquet"
SMALL_SUBSET = os.environ.get("E2V_SMALL", "0") == "1"   # 索引仍建全量


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).strip().lower())


def build_hf_index():
    """读全量 HF parquet，建 (did, norm_text) -> deque[audio_bytes]（按 file 排序）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = sorted(glob.glob(HF_GLOB))
    assert files, f"未找到 HF parquet: {HF_GLOB}"
    print(f"读取 {len(files)} 个 HF parquet ...")
    tbl = pa.concat_tables(
        [pq.read_table(f, columns=["file", "transcription", "audio"]) for f in files])
    file_arr = tbl.column("file").to_pylist()
    txt_arr = tbl.column("transcription").to_pylist()
    audio_arr = tbl.column("audio").to_pylist()
    print(f"  HF 总行数: {len(file_arr)}")

    order = sorted(range(len(file_arr)), key=lambda i: file_arr[i])
    idx = defaultdict(deque)
    for i in order:
        f = file_arr[i]
        stem = f[:-4] if f.lower().endswith(".wav") else f
        did = stem.rsplit("_", 1)[0]
        idx[(did, norm(txt_arr[i]))].append(audio_arr[i]["bytes"])
    return idx


def load_emotion2vec():
    from funasr import AutoModel
    print(f"Loading emotion2vec_base from {C.EMOTION2VEC_PATH} (device={DEVICE}) ...")
    model = AutoModel(model=C.EMOTION2VEC_PATH, disable_update=True, device=DEVICE)
    print("  emotion2vec_base loaded OK")
    return model


def decode_wav(audio_bytes: bytes) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(audio_bytes))
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav


def extract_one(model, wav: np.ndarray) -> np.ndarray:
    res = model.generate(wav, granularity="utterance", extract_embedding=True)
    arr = np.array(res[0]["feats"], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    return arr


def process_split(model, hf_idx, split):
    samples = D.load_split(split)
    if SMALL_SUBSET:
        samples = samples[:30]
        print(f"  [SMALL_SUBSET] first 30 samples")

    feats, keys, labels = [], [], []
    for s in tqdm(samples, desc=f"audio {split}"):
        key = (s["dialogue_id"], norm(s["text"]))
        dq = hf_idx.get(key)
        assert dq, f"{split}: HF 无音频命中 {key}（对齐应 100%，请检查）"
        audio_bytes = dq.popleft()
        feats.append(extract_one(model, decode_wav(audio_bytes)))
        keys.append(f"{s['dialogue_id']}|{s['utt_id']}")
        labels.append(s["label_id"])

    return samples, {
        "split": split, "feat_dim": 768,
        "order_keys": keys,
        "feats": np.stack(feats).astype(np.float32),
        "labels": np.array(labels, dtype=np.int64),
    }


def assert_aligned(samples, data, split):
    assert len(data["order_keys"]) == len(samples), (
        f"{split}: feat {len(data['order_keys'])} != samples {len(samples)}")
    feats = data["feats"]
    assert feats.shape == (len(samples), 768), f"{split}: bad shape {feats.shape}"
    assert not np.isnan(feats).any() and not np.isinf(feats).any(), f"{split}: NaN/Inf"
    for i, s in enumerate(samples):
        exp_key = f"{s['dialogue_id']}|{s['utt_id']}"
        assert data["order_keys"][i] == exp_key, (
            f"{split}[{i}]: key {data['order_keys'][i]} != {exp_key}")
        assert data["labels"][i] == s["label_id"], (
            f"{split}[{i}]: label {data['labels'][i]} != {s['label_id']}")
    print(f"  [OK] {split}: {len(samples)} feats aligned (key+label verified)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hf_idx = build_hf_index()
    model = load_emotion2vec()

    all_keys = {}
    for split in ["train", "dev", "test"]:
        print(f"\n── {split} ──")
        samples, data = process_split(model, hf_idx, split)
        assert_aligned(samples, data, split)
        all_keys[split] = set(data["order_keys"])
        out = os.path.join(OUTPUT_DIR, f"IEMOCAP_{split}_emotion2vec.pt")
        torch.save(data, out)
        print(f"  saved -> {out}  feats={data['feats'].shape}")

    tr, dv, te = all_keys["train"], all_keys["dev"], all_keys["test"]
    assert tr.isdisjoint(te) and dv.isdisjoint(te) and tr.isdisjoint(dv), "split key overlap!"
    print("\n[OK] train/dev/test keys disjoint. Done.")


if __name__ == "__main__":
    main()
