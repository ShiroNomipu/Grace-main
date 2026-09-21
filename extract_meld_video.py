"""
MELD 视频特征提取（POSTER V2 + AffectNet-7 预训练，与音频同协议、严格对齐）。

数据源：MAGTKD 已切好的 8 帧对齐人脸张量 (8,3,224,224) f32（仅人脸检测/裁剪/归一化，
  未碰 MELD 情感标签）。编码器 = POSTER V2，权重 = AffectNet-7（通用 7 类表情库，
  非 MELD 情感 → R5 纯度干净）。我们只取分类前的 768-d 表征（hook VIT.head 输入），
  不用其 7 类 logits。

每句 → 8 帧 × 768-d：
  - 帧级 (8,768) f16  → V3.5 帧级路由（与音频 64 帧对称）
  - 句级 768-d  f32   → V2 前缀注入（8 帧均值）
对齐：按 data_meld.load_split 顺序遍历，key=clip_id（dia{D}_utt{U}），逐条断言 label 一致。
  npy 文件名 = {Dialogue_ID}_{Utterance_ID}.npy。

容错：个别 npy 缺失/损坏 → 零特征占位并记录（保证与文本样本严格对齐）。

产出（镜像音频）：
  C.MELD_VIDEO_FEAT_DIR  / MELD_{split}_poster.pt        {order_keys, feats:(N,768) f32, labels}
  C.MELD_VIDEO_FRAME_DIR / MELD_{split}_poster_frame.pt  {order_keys, feats:(N,8,768) f16, labels}
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

import config as C
import data_meld as D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_FIX = C.VIDEO_FRAME_LEN  # 8


# ── 绕过 AffectNet ckpt pickle 引用的训练期日志类（只取 state_dict）──
class RecorderMeter(object):
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)


class RecorderMeter1(RecorderMeter):
    pass


def load_poster():
    import __main__
    __main__.RecorderMeter = RecorderMeter
    __main__.RecorderMeter1 = RecorderMeter1
    sys.path.insert(0, C.POSTER_V2_DIR)
    sys.path.insert(0, os.path.join(C.POSTER_V2_DIR, "models"))
    from models.PosterV2_7cls import pyramid_trans_expr2

    print(f"Loading POSTER V2 (AffectNet-7) from {C.POSTER_AFFECTNET_CKPT} ...")
    model = pyramid_trans_expr2(num_classes=7)
    ck = torch.load(C.POSTER_AFFECTNET_CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert len(miss) == 0 and len(unexp) == 0, f"weight mismatch: miss={miss[:5]} unexp={unexp[:5]}"
    model.eval().to(DEVICE)

    # hook：抓分类头输入 = se_block 输出 = 768-d pre-head 表征
    feat_buf = {}
    model.VIT.head.register_forward_hook(
        lambda mod, inp, out: feat_buf.__setitem__("f", inp[0].detach())
    )
    print("  POSTER V2 loaded OK (768-d pre-head feature via hook)")
    return model, feat_buf


def npy_path(split, sample):
    # clip_id = dia{D}_utt{U} → 文件名 {D}_{U}.npy
    return os.path.join(C.MELD_VIDEO_NPY_DIR[split],
                        f"{sample['dialogue_id']}_{sample['utt_id']}.npy")


@torch.no_grad()
def encode_frames(model, feat_buf, faces):
    """faces: (8,3,224,224) → (8,768) f32。"""
    x = torch.from_numpy(faces).float().to(DEVICE)
    _ = model(x)
    return feat_buf["f"].float().cpu().numpy()  # (8,768)


def process_split(model, feat_buf, split, small=0):
    samples = D.load_split(split)
    if small:
        samples = samples[:small]
        print(f"  [SMALL] first {small}")

    utt_feats, frm_feats, keys, labels = [], [], [], []
    missing = []
    for s in tqdm(samples, desc=f"meld-video {split}"):
        cid = s["clip_id"]
        path = npy_path(split, s)
        try:
            faces = np.load(path)
            if faces.shape != (T_FIX, 3, 224, 224):
                raise ValueError(f"bad shape {faces.shape}")
            fr = encode_frames(model, feat_buf, faces)  # (8,768)
        except Exception as e:                          # 缺失/损坏 → 零占位
            missing.append((cid, str(e)[:60]))
            fr = np.zeros((T_FIX, 768), dtype=np.float32)
        utt_feats.append(fr.mean(axis=0).astype(np.float32))
        frm_feats.append(fr.astype(np.float16))
        keys.append(cid)
        labels.append(s["label_id"])

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
    ap.add_argument("--probe", action="store_true", help="只跑 test 前 4 句验证形状")
    args = ap.parse_args()

    os.makedirs(C.MELD_VIDEO_FEAT_DIR, exist_ok=True)
    os.makedirs(C.MELD_VIDEO_FRAME_DIR, exist_ok=True)
    model, feat_buf = load_poster()

    if args.probe:
        samples, (utt_data, frm_data), missing = process_split(model, feat_buf, "test", small=4)
        assert_aligned(samples, utt_data, frm_data, "test")
        print(f"  probe utt{utt_data['feats'].shape} frame{frm_data['feats'].shape} "
              f"miss={len(missing)} | feat[0] mean={utt_data['feats'][0].mean():.4f}")
        return

    total_missing = 0
    for split in ["train", "dev", "test"]:
        print(f"\n── {split} ──")
        samples, (utt_data, frm_data), missing = process_split(model, feat_buf, split, small=args.small)
        assert_aligned(samples, utt_data, frm_data, split)
        total_missing += len(missing)
        if not args.small:
            torch.save(utt_data, os.path.join(C.MELD_VIDEO_FEAT_DIR, f"MELD_{split}_poster.pt"))
            torch.save(frm_data, os.path.join(C.MELD_VIDEO_FRAME_DIR, f"MELD_{split}_poster_frame.pt"))
            print(f"  saved utt{utt_data['feats'].shape} + frame{frm_data['feats'].shape}")
    print(f"\n[DONE] total zero-filled (missing/broken) clips = {total_missing}")


if __name__ == "__main__":
    main()
