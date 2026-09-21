"""
emotion2vec_base 帧级序列特征提取（V3.5 用）。

与 extract_audio_emotion2vec.py 的唯一区别：
  - granularity="utterance"（单 768-d 向量）→ granularity="frame"（T 帧 × 768-d 序列）。
  - 每句帧序列用 adaptive avg pool 沿时间轴下采样到固定 T=AUDIO_FRAME_LEN（定长 batch、控显存）。
对齐策略 / HF 索引 / key 顺序 与句级脚本完全一致（复用同一对齐逻辑，保证 V2/V3.5 可比）。

产出：C.AUDIO_FRAME_DIR / IEMOCAP_{split}_emotion2vec_frame.pt
  { "split", "feat_dim":768, "n_frames":T, "order_keys":[f"{did}|{utt_id}"],
    "feats":(N,T,768) float16, "labels":(N,) i64 }

用法（V3.5）：建 key->idx，遍历文本 sample 用 f"{did}|{utt_id}" 取 (T,768) 序列。
"""
import argparse
import glob
import io
import os
import re
from collections import defaultdict, deque

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as C
import data_iemocap as D

OUTPUT_DIR = C.AUDIO_FRAME_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_GLOB = r"datasets/iemocap_hf/data/*.parquet"
T_FIX = C.AUDIO_FRAME_LEN


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


def pool_to_fixed(arr: np.ndarray, t_fix: int) -> np.ndarray:
    """(T,768) -> (t_fix,768)，沿时间轴 adaptive avg pool；T<t_fix 时上采样填充。"""
    if arr.ndim == 1:
        arr = arr[None, :]
    x = torch.from_numpy(arr.astype(np.float32)).T.unsqueeze(0)   # (1,768,T)
    x = F.adaptive_avg_pool1d(x, t_fix)                           # (1,768,t_fix)
    return x.squeeze(0).T.contiguous().numpy()                    # (t_fix,768)


def extract_frames(model, wav: np.ndarray) -> np.ndarray:
    """返回 (T,768) 帧级序列。"""
    res = model.generate(wav, granularity="frame", extract_embedding=True)
    arr = np.array(res[0]["feats"], dtype=np.float32)
    if arr.ndim == 1:                 # 兜底：万一退化成单向量
        arr = arr[None, :]
    assert arr.shape[-1] == 768, f"bad frame feat dim {arr.shape}"
    return arr


def process_split(model, hf_idx, split, small=0):
    samples = D.load_split(split)
    if small:
        samples = samples[:small]
        print(f"  [SMALL] first {small} samples")

    feats, keys, labels = [], [], []
    raw_lens = []
    for s in tqdm(samples, desc=f"audio-frame {split}"):
        key = (s["dialogue_id"], norm(s["text"]))
        dq = hf_idx.get(key)
        assert dq, f"{split}: HF 无音频命中 {key}（对齐应 100%）"
        audio_bytes = dq.popleft()
        seq = extract_frames(model, decode_wav(audio_bytes))     # (T,768)
        raw_lens.append(seq.shape[0])
        feats.append(pool_to_fixed(seq, T_FIX).astype(np.float16))
        keys.append(f"{s['dialogue_id']}|{s['utt_id']}")
        labels.append(s["label_id"])

    rl = np.array(raw_lens)
    print(f"  raw frame len: min={rl.min()} median={int(np.median(rl))} "
          f"max={rl.max()} -> pooled to {T_FIX}")
    return samples, {
        "split": split, "feat_dim": 768, "n_frames": T_FIX,
        "order_keys": keys,
        "feats": np.stack(feats).astype(np.float16),
        "labels": np.array(labels, dtype=np.int64),
    }


def assert_aligned(samples, data, split):
    assert len(data["order_keys"]) == len(samples)
    feats = data["feats"]
    assert feats.shape == (len(samples), T_FIX, 768), f"{split}: bad shape {feats.shape}"
    assert not np.isnan(feats.astype(np.float32)).any(), f"{split}: NaN"
    for i, s in enumerate(samples):
        exp_key = f"{s['dialogue_id']}|{s['utt_id']}"
        assert data["order_keys"][i] == exp_key
        assert data["labels"][i] == s["label_id"]
    print(f"  [OK] {split}: {len(samples)} frame-feats aligned (key+label verified)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只跑每 split 前 3 句，验证 shape")
    ap.add_argument("--small", type=int, default=0, help="每 split 前 N 句（验证用）")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hf_idx = build_hf_index()
    model = load_emotion2vec()

    if args.probe:
        samples = D.load_split("test")[:3]
        for s in samples:
            key = (s["dialogue_id"], norm(s["text"]))
            seq = extract_frames(model, decode_wav(hf_idx[key].popleft()))
            pooled = pool_to_fixed(seq, T_FIX)
            print(f"[probe] {s['dialogue_id']}|{s['utt_id']} raw={seq.shape} "
                  f"pooled={pooled.shape} mean={pooled.mean():.3f} std={pooled.std():.3f}")
        print("[probe] done. emotion2vec frame granularity OK.")
        return

    small = args.small
    all_keys = {}
    for split in ["train", "dev", "test"]:
        print(f"\n── {split} ──")
        samples, data = process_split(model, hf_idx, split, small=small)
        assert_aligned(samples, data, split)
        all_keys[split] = set(data["order_keys"])
        suffix = f"_small{small}" if small else ""
        out = os.path.join(OUTPUT_DIR, f"IEMOCAP_{split}_emotion2vec_frame{suffix}.pt")
        torch.save(data, out)
        sz = data["feats"].nbytes / 1e6
        print(f"  saved -> {out}  feats={data['feats'].shape} ({sz:.0f} MB)")

    if not small:
        tr, dv, te = all_keys["train"], all_keys["dev"], all_keys["test"]
        assert tr.isdisjoint(te) and dv.isdisjoint(te) and tr.isdisjoint(dv), "split key overlap!"
        print("\n[OK] train/dev/test keys disjoint. Done.")


if __name__ == "__main__":
    main()
