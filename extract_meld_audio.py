"""
MELD 音频特征提取（emotion2vec_base，与 IEMOCAP 同协议）。

数据源：MELD 每句 mp4 切片（dia{D}_utt{U}.mp4），librosa 抽 16k 单声道音轨 → emotion2vec_base
  （纯 SSL，未碰情感标签，R5 干净）。一次抽轨同时提两份：
    - 句级（granularity=utterance）768-d  → V2 用
    - 帧级（granularity=frame）(T,768) pool 到 AUDIO_FRAME_LEN → V3.5 用
对齐：按 data_meld.load_split 顺序遍历，key=clip_id（dia{D}_utt{U}），逐条断言 label 一致。

容错：个别 mp4 缺失/无音轨/解码失败 → 用零特征占位并记录（保证与文本样本严格对齐），
  失败数报告；MELD 已知极少数损坏文件，零占位等价"音频缺失"，不影响主流程。

产出：
  C.MELD_AUDIO_FEAT_DIR  / MELD_{split}_emotion2vec.pt        {order_keys, feats:(N,768) f32, labels}
  C.MELD_AUDIO_FRAME_DIR / MELD_{split}_emotion2vec_frame.pt  {order_keys, feats:(N,T,768) f16, labels}
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as C
import data_meld as D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_FIX = C.AUDIO_FRAME_LEN


def load_emotion2vec():
    from funasr import AutoModel
    print(f"Loading emotion2vec_base from {C.EMOTION2VEC_PATH} (device={DEVICE}) ...")
    model = AutoModel(model=C.EMOTION2VEC_PATH, disable_update=True, device=DEVICE)
    print("  emotion2vec_base loaded OK")
    return model


def load_wav(path):
    import librosa
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wav, _ = librosa.load(path, sr=16000, mono=True)
    return np.asarray(wav, dtype=np.float32)


def pool_to_fixed(arr, t_fix):
    if arr.ndim == 1:
        arr = arr[None, :]
    x = torch.from_numpy(arr.astype(np.float32)).T.unsqueeze(0)
    x = F.adaptive_avg_pool1d(x, t_fix)
    return x.squeeze(0).T.contiguous().numpy()


def extract_both(model, wav):
    """返回 (utt_768, frame_T_768)。"""
    utt = model.generate(wav, granularity="utterance", extract_embedding=True)
    u = np.array(utt[0]["feats"], dtype=np.float32)
    if u.ndim == 2:
        u = u[0]
    frm = model.generate(wav, granularity="frame", extract_embedding=True)
    fr = np.array(frm[0]["feats"], dtype=np.float32)
    if fr.ndim == 1:
        fr = fr[None, :]
    return u, fr


def mp4_path(split, clip_id):
    return os.path.join(C.MELD_MP4_DIR[split], clip_id + ".mp4")


def process_split(model, split, small=0):
    samples = D.load_split(split)
    if small:
        samples = samples[:small]
        print(f"  [SMALL] first {small}")

    utt_feats, frm_feats, keys, labels = [], [], [], []
    raw_lens, missing = [], []
    for s in tqdm(samples, desc=f"meld-audio {split}"):
        cid = s["clip_id"]
        path = mp4_path(split, cid)
        try:
            wav = load_wav(path)
            if wav.size < 160:           # < 0.01s 视为无效
                raise ValueError("empty/too-short audio")
            u, fr = extract_both(model, wav)
            raw_lens.append(fr.shape[0])
        except Exception as e:           # 缺失/损坏 → 零占位（等价音频缺失）
            missing.append((cid, str(e)[:60]))
            u = np.zeros(768, dtype=np.float32)
            fr = np.zeros((T_FIX, 768), dtype=np.float32)
        utt_feats.append(u.astype(np.float32))
        frm_feats.append(pool_to_fixed(fr, T_FIX).astype(np.float16))
        keys.append(cid)
        labels.append(s["label_id"])

    if raw_lens:
        rl = np.array(raw_lens)
        print(f"  raw frame len: min={rl.min()} median={int(np.median(rl))} max={rl.max()} -> {T_FIX}")
    if missing:
        print(f"  [WARN] {len(missing)} clips failed (zero-filled). e.g. {missing[:3]}")
    labels = np.array(labels, dtype=np.int64)
    return samples, (
        {"split": split, "feat_dim": 768, "order_keys": keys,
         "feats": np.stack(utt_feats).astype(np.float32), "labels": labels},
        {"split": split, "feat_dim": 768, "n_frames": T_FIX, "order_keys": keys,
         "feats": np.stack(frm_feats).astype(np.float16), "labels": labels},
    ), missing


def assert_aligned(samples, utt_data, frm_data, split):
    assert len(utt_data["order_keys"]) == len(samples)
    assert utt_data["feats"].shape == (len(samples), 768)
    assert frm_data["feats"].shape == (len(samples), T_FIX, 768)
    for i, s in enumerate(samples):
        assert utt_data["order_keys"][i] == s["clip_id"]
        assert utt_data["labels"][i] == s["label_id"]
    print(f"  [OK] {split}: {len(samples)} aligned (clip_id+label verified)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(C.MELD_AUDIO_FEAT_DIR, exist_ok=True)
    os.makedirs(C.MELD_AUDIO_FRAME_DIR, exist_ok=True)
    model = load_emotion2vec()

    total_missing = 0
    for split in ["train", "dev", "test"]:
        print(f"\n── {split} ──")
        samples, (utt_data, frm_data), missing = process_split(model, split, small=args.small)
        assert_aligned(samples, utt_data, frm_data, split)
        total_missing += len(missing)
        if not args.small:
            torch.save(utt_data, os.path.join(C.MELD_AUDIO_FEAT_DIR, f"MELD_{split}_emotion2vec.pt"))
            torch.save(frm_data, os.path.join(C.MELD_AUDIO_FRAME_DIR, f"MELD_{split}_emotion2vec_frame.pt"))
            print(f"  saved utt{utt_data['feats'].shape} + frame{frm_data['feats'].shape}")
    print(f"\n[DONE] total zero-filled (missing/broken) clips = {total_missing}")


if __name__ == "__main__":
    main()
