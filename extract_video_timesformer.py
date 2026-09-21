"""
TimesFormer-base-finetuned-k400 视频特征提取脚本（IEMOCAP）

产出：每个 split 一个 .pt 文件，结构与 emotion2vec 完全对齐：
  {
    "vids":    List[str],
    "dia2utt": Dict[str, List[str]],
    "video":   Dict[str, List[np.ndarray]],   # utterance 级 768-d float32
    "labels":  Dict[str, List[int]],
    "speakers": Dict[str, List[int]],
  }

纯度说明：使用 timesformer-base-finetuned-k400（k400 预训练），
  完全冻结，从未在 IEMOCAP 情感标签上微调，满足 R5 纯度要求。
  模型输出取 CLS token（last_hidden_state[:,0,:]，768-d）。
"""

import csv
import os
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from transformers import AutoImageProcessor, TimesformerModel

# ─────────────────────── 配置区 ──────────────────────────────────────────────

import config as _C
TIMESFORMER_PATH = _C.TIMESFORMER_PATH   # 项目内本地路径 D:\LLM-ERC\models\...
IEMOCAP_ROOT     = r"E:\复现库\（可复现ERC）MAGTKD-main\datasets\datasets\IEMOCAP"
OUTPUT_DIR       = r"D:\LLM-ERC\datasets\video_timesformer"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES       = 8

IEMOCAP_EMO2ID = {"neu": 0, "sur": 1, "fru": 2, "ang": 3, "sad": 4,
                  "hap": 5, "exc": 5, "dis": 2}

SMALL_SUBSET = False   # True = 只处理前 50 条做 sanity check

# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading TimesFormer from {TIMESFORMER_PATH} ...")
    processor = AutoImageProcessor.from_pretrained(TIMESFORMER_PATH)
    model = TimesformerModel.from_pretrained(TIMESFORMER_PATH)
    model.eval()
    model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  TimesFormer loaded OK | hidden_size={model.config.hidden_size} | device={DEVICE}")
    return processor, model


def open_csv(path):
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            f = open(path, "r", encoding=enc, newline="")
            rows = list(csv.DictReader(f))
            f.close()
            return rows
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode CSV: {path}")


def fix_path(raw):
    return str(Path(raw.replace("/", os.sep).replace("\\", os.sep)))


def build_avi_path(iemocap_root, dialogue_id):
    """根据 Dialogue_ID 重建 avi 路径。"""
    session_num = int(dialogue_id[3:5])
    return os.path.join(
        iemocap_root,
        f"Session{session_num}",
        "dialog", "avi", "DivX",
        f"{dialogue_id}.avi"
    )


def sample_frames(avi_path, start_time, end_time, n_frames=8):
    """从 avi 截取 [start, end] 区间，均匀采 n_frames 帧，返回 BGR list。"""
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    t0 = max(float(start_time) - 0.5, 0)
    t1 = float(end_time) + 0.5
    f0 = int(t0 * fps)
    f1 = int(t1 * fps)
    total_clip = max(f1 - f0, 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    for _ in range(total_clip):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return None

    # 均匀采 n_frames 帧
    indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    sampled = [frames[i] for i in indices]

    # 不足则用最后一帧补齐
    while len(sampled) < n_frames:
        sampled.append(sampled[-1].copy())

    return sampled[:n_frames]


def extract_one(processor, model, frames):
    """frames: list of BGR numpy arrays -> 768-d float32 numpy array."""
    # cv2 BGR -> RGB
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    inputs = processor(images=rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)   # (1, T, C, H, W)

    with torch.no_grad():
        out = model(pixel_values=pixel_values)
        feat = out.last_hidden_state[0, 0, :].cpu().numpy().astype(np.float32)  # CLS
    return feat


def process_split(processor, model, csv_path, split):
    rows = open_csv(csv_path)
    if SMALL_SUBSET:
        rows = rows[:50]
        print(f"  [SMALL_SUBSET] first 50 rows only")

    vids, dia2utt, video, labels, speakers = [], {}, {}, {}, {}
    skipped = 0

    # 缓存已打开的 avi 路径（每个 dialogue 只找一次）
    avi_cache = {}

    for row in tqdm(rows, desc=f"IEMOCAP {split}"):
        vid    = row["Dialogue_ID"]
        utt_id = row["Utterance_ID"]
        emo    = row["Emotion"].strip().strip("'\"").lower()
        spk    = 0 if row["Speaker"].strip().lower() in ("female", "f") else 1
        start  = row["Start_Time"]
        end    = row["End_Time"]

        if emo not in IEMOCAP_EMO2ID:
            skipped += 1
            continue
        label = IEMOCAP_EMO2ID[emo]

        # 解析 avi 路径（优先 CSV，fallback 规则重建）
        if vid not in avi_cache:
            avi_primary = fix_path(row["Video_Path"])
            if os.path.exists(avi_primary):
                avi_cache[vid] = avi_primary
            else:
                avi_fallback = build_avi_path(IEMOCAP_ROOT, vid)
                if os.path.exists(avi_fallback):
                    avi_cache[vid] = avi_fallback
                else:
                    avi_cache[vid] = None

        avi_path = avi_cache[vid]
        if avi_path is None:
            print(f"  [WARN] avi not found, skip: {vid}")
            skipped += 1
            continue

        # script session 跳过（同音频）
        if "script" in vid:
            skipped += 1
            continue

        frames = sample_frames(avi_path, start, end)
        if frames is None:
            print(f"  [WARN] frame extraction failed, skip: {vid}_{utt_id}")
            skipped += 1
            continue

        feat = extract_one(processor, model, frames)

        if vid not in vids:
            vids.append(vid)
            dia2utt[vid] = []
            video[vid]   = []
            labels[vid]  = []
            speakers[vid] = []

        dia2utt[vid].append(utt_id)
        video[vid].append(feat)
        labels[vid].append(label)
        speakers[vid].append(spk)

    total = sum(len(v) for v in video.values())
    print(f"  {split}: {total} utterances collected, {skipped} skipped")
    return {"vids": vids, "dia2utt": dia2utt, "video": video,
            "labels": labels, "speakers": speakers}


# ───────────────────── 自检断言 ───────────────────────────────────────────────

def assert_integrity(data, split, expected_dim=768):
    vids = data["vids"]
    assert len(vids) > 0, f"{split}: empty!"
    total = 0
    for vid in vids:
        n = len(data["dia2utt"][vid])
        assert n == len(data["video"][vid]) == len(data["labels"][vid]) == len(data["speakers"][vid]), \
            f"{split} {vid}: misaligned lengths"
        for i, feat in enumerate(data["video"][vid]):
            arr = np.array(feat)
            assert arr.ndim == 1 and arr.shape[0] == expected_dim, \
                f"{split} {vid}[{i}]: bad shape {arr.shape}"
            assert not np.isnan(arr).any() and not np.isinf(arr).any(), \
                f"{split} {vid}[{i}]: NaN/Inf in feature"
        total += n
    print(f"  [OK] {split}: {len(vids)} dialogs, {total} utts, feat_dim={expected_dim}")


def assert_no_overlap(train, dev, test):
    tv, dv, ev = set(train["vids"]), set(dev["vids"]), set(test["vids"])
    assert tv.isdisjoint(dv),  f"train/dev overlap: {tv & dv}"
    assert tv.isdisjoint(ev),  f"train/test overlap: {tv & ev}"
    assert dv.isdisjoint(ev),  f"dev/test overlap: {dv & ev}"
    print("  [OK] train/dev/test vids disjoint (no data leakage)")


def assert_audio_video_aligned():
    """断言音视频 utterance 数量对齐（同一个 vid 的 utt 顺序一致）。"""
    for split in ["train", "dev", "test"]:
        audio = torch.load(
            f"D:/LLM-ERC/datasets/audio_emotion2vec/IEMOCAP_{split}_emotion2vec.pt",
            weights_only=False)
        video = torch.load(
            f"D:/LLM-ERC/datasets/video_timesformer/IEMOCAP_{split}_timesformer.pt",
            weights_only=False)
        # 只检查两者都有的 vid
        common = set(audio["vids"]) & set(video["vids"])
        for vid in common:
            na = len(audio["audio"][vid])
            nv = len(video["video"][vid])
            assert na == nv, f"{split} {vid}: audio={na} != video={nv}"
            # 标签必须一致
            assert audio["labels"][vid] == video["labels"][vid], \
                f"{split} {vid}: label mismatch between audio and video features"
        print(f"  [OK] {split}: {len(common)} shared vids, audio/video aligned")


# ────────────────────────── 主流程 ──────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor, model = load_model()

    splits = {
        "train": os.path.join(IEMOCAP_ROOT, "IEMOCAP_train.csv"),
        "dev":   os.path.join(IEMOCAP_ROOT, "IEMOCAP_dev.csv"),
        "test":  os.path.join(IEMOCAP_ROOT, "IEMOCAP_test.csv"),
    }

    results = {}
    for split, csv_path in splits.items():
        print(f"\n── {split} ──")
        data = process_split(processor, model, csv_path, split)
        assert_integrity(data, split)
        results[split] = data

    assert_no_overlap(results["train"], results["dev"], results["test"])

    for split, data in results.items():
        out = os.path.join(OUTPUT_DIR, f"IEMOCAP_{split}_timesformer.pt")
        torch.save(data, out)
        print(f"  saved -> {out}")

    # 与音频特征对齐检查（若音频文件存在）
    audio_dir = "D:/LLM-ERC/datasets/audio_emotion2vec"
    if all(os.path.exists(f"{audio_dir}/IEMOCAP_{s}_emotion2vec.pt")
           for s in ["train", "dev", "test"]):
        print("\n── audio/video alignment check ──")
        assert_audio_video_aligned()

    print("\nDone.")


if __name__ == "__main__":
    main()
