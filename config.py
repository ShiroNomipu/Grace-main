"""
LLM-PEFT-ERC 中央配置（单一事实源）。
所有脚本从这里读取路径、标签方案、超参数、随机种子。
改配置只动这里，不动其他代码。
"""
import os

def _env(name, default):
    """Environment override for a path.

    Windows defaults below are the single source of truth for local runs. The
    cross-backbone control runs on a Linux GPU box, where the project root,
    backbone and output directory differ; those three are the only things that
    move, so they are the only ones made overridable.
    """
    return os.environ.get(name, default)


# ─── 强制所有模型/缓存留在项目盘，绝不污染 C 盘（必须在 import funasr/transformers 前生效）───
_CACHE_ROOT = _env("GRACE_CACHE_ROOT", r"D:\LLM-ERC\models\.cache")
os.environ["MODELSCOPE_CACHE"]    = os.path.join(_CACHE_ROOT, "modelscope")
os.environ["HF_HOME"]             = os.path.join(_CACHE_ROOT, "huggingface")
os.environ["HF_HUB_CACHE"]        = os.path.join(_CACHE_ROOT, "huggingface", "hub")
os.environ["TRANSFORMERS_CACHE"]  = os.path.join(_CACHE_ROOT, "huggingface")
os.environ["TORCH_HOME"]          = os.path.join(_CACHE_ROOT, "torch")
for _d in os.environ.values():
    pass  # 目录由各库自动创建

# ───────────────────────── 路径 ─────────────────────────
MODEL_PATH       = _env("GRACE_MODEL_PATH", r"D:\LLM-ERC\models\Qwen2-VL-2B-Instruct")
# 纯 SSL 预训练（fine-tuning data="/"，从未碰情感标签，满足 R5 纯度）。
# 不可用 emotion2vec_base_finetuned（在含 IEMOCAP 的 SER 数据微调过 → test 泄露）。
EMOTION2VEC_PATH = r"D:\LLM-ERC\models\emotion2vec_base"
TIMESFORMER_PATH = r"D:\LLM-ERC\models\timesformer-base-finetuned-k400" # 收进项目（见下）

# ── 文本协议数据源：InstructERC 公开 IEMOCAP（COSMIC 预处理，与 LLM-ERC SOTA 同款）──
# 标准 6 类 + 含 script + Ses5 做 test。这是对外可直接对比的事实标准（见 L10 修订）。
_DATA_ROOT = _env("GRACE_DATA_ROOT", r"D:\LLM-ERC\datasets")
INSTRUCTERC_DIR = os.path.join(_DATA_ROOT, "instructerc_iemocap")
INSTRUCTERC_PKL = os.path.join(INSTRUCTERC_DIR, "iemocap.pkl")

# ── MAGTKD 仅作可选利用项（V2 音频 wav 自提时可能指向其原始 wav），文本已不再用其 CSV ──
CSV_DIR      = r"E:\复现库\（可复现ERC）MAGTKD-main\datasets\datasets\IEMOCAP"
CSV_PATHS    = {
    "train": os.path.join(CSV_DIR, "IEMOCAP_train.csv"),
    "dev":   os.path.join(CSV_DIR, "IEMOCAP_dev.csv"),
    "test":  os.path.join(CSV_DIR, "IEMOCAP_test.csv"),
}
OUTPUT_DIR   = _env("GRACE_OUTPUT_DIR", r"D:\LLM-ERC\outputs")   # checkpoint / 指标落盘
AUDIO_FEAT_DIR = os.path.join(_DATA_ROOT, "audio_emotion2vec")         # V2 用(句级 768-d 单向量)
AUDIO_FRAME_DIR = os.path.join(_DATA_ROOT, "audio_emotion2vec_frame")  # V3.5 用(帧级序列)
VIDEO_FEAT_DIR = r"D:\LLM-ERC\datasets\video_timesformer"   # V2 用

# ───────────────────── 标签方案（与 MAGTKD 一致）─────────────────────
# 顺序固定 = label id 0..5，绝不可改，否则与已训模型/历史结果错位
LABEL_LIST = ["ang", "exc", "fru", "hap", "neu", "sad"]
LABEL2ID   = {lab: i for i, lab in enumerate(LABEL_LIST)}
ID2LABEL   = {i: lab for i, lab in enumerate(LABEL_LIST)}
# prompt 里给 LLM 看的自然词（缩写不利于 LLM 理解，用全称）
LABEL2WORD = {
    "ang": "angry", "exc": "excited", "fru": "frustrated",
    "hap": "happy", "neu": "neutral", "sad": "sad",
}
WORD2LABEL = {w: lab for lab, w in LABEL2WORD.items()}   # 解析生成结果用
NUM_CLASSES = len(LABEL_LIST)

# InstructERC/COSMIC pkl 内的情感 id → 我们的情感字符串（用 test=Ses5 分布对照锁定，绝不可改）。
# 注意：这是 pkl 内部 id，与本项目 LABEL2ID 顺序无关；读 pkl 时转字符串后再走 LABEL2ID。
COSMIC_ID2EMO = {0: "hap", 1: "sad", 2: "neu", 3: "ang", 4: "exc", 5: "fru"}

# ── 数据协议（L10 修订：全面向 LLM-ERC 看齐）──
# 采用 InstructERC/COSMIC 标准 IEMOCAP：6 类全量（含 script）、带 speaker、Ses5 做 test。
# split 由 pkl 自带（train/valid/test 对话 id 列表），dev = pkl 的 valid。
# 对话顺序即 pkl 内顺序（COSMIC 已按时序组织），无需再按时间排序。
# 预期数量（自检锚点）：与 InstructERC 论文一致 train 5163 / dev(valid) 647 / test 1623。
EXPECTED_COUNTS = {"train": 5163, "dev": 647, "test": 1623}

# ───────── MELD（第二数据集，7类，双集标配；源=MAGTKD 原始 csv+mp4，特征我们自提保纯度）─────────
MELD_DIR = _env("GRACE_MELD_DIR", r"E:\复现库\（可复现ERC）MAGTKD-main\datasets\datasets\MELD")
MELD_CSV = {
    "train": os.path.join(MELD_DIR, "train_sent_emo.csv"),
    "dev":   os.path.join(MELD_DIR, "dev_sent_emo.csv"),
    "test":  os.path.join(MELD_DIR, "test_sent_emo.csv"),
}
MELD_MP4_DIR = {           # 每句 dia{Dialogue_ID}_utt{Utterance_ID}.mp4
    "train": os.path.join(MELD_DIR, "train_splits"),
    "dev":   os.path.join(MELD_DIR, "dev_splits_complete"),
    "test":  os.path.join(MELD_DIR, "output_repeated_splits_test"),
}
# 标准 MELD 7 类，顺序固定 = label id 0..6（绝不可改，否则与已训模型/结果错位）
MELD_LABEL_LIST = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
MELD_LABEL2ID   = {lab: i for i, lab in enumerate(MELD_LABEL_LIST)}
MELD_ID2LABEL   = {i: lab for i, lab in enumerate(MELD_LABEL_LIST)}
MELD_LABEL2WORD = {lab: lab for lab in MELD_LABEL_LIST}   # MELD 类名本身即自然词
MELD_NUM_CLASSES = len(MELD_LABEL_LIST)
# 官方数量锚点（自检用）：train 9989 / dev 1109 / test 2610
MELD_EXPECTED_COUNTS = {"train": 9989, "dev": 1109, "test": 2610}
MELD_AUDIO_FEAT_DIR  = os.path.join(_DATA_ROOT, "meld_audio_emotion2vec")        # 句级 768-d
MELD_AUDIO_FRAME_DIR = os.path.join(_DATA_ROOT, "meld_audio_emotion2vec_frame")  # 帧级序列

# ── MELD 视频（POSTER V2，AffectNet-7 预训练=通用表情库，未碰 MELD 标签 → R5 干净）──
# 源：MAGTKD 已切好的 8 帧对齐人脸张量 (8,3,224,224) f32（仅裁剪/归一化，无情感微调）。
# 我们用纯 AffectNet-7 POSTER V2 提 pre-head 768-d 特征（hook VIT.head 输入），8 帧序列 + 句级均值。
POSTER_V2_DIR        = r"E:\复现库\（可复现ERC）MAGTKD-main\MAGTKD-main\POSTER_V2"
POSTER_AFFECTNET_CKPT = os.path.join(POSTER_V2_DIR, "checkpoint", "affectnet-7-model_best.pth")
_MELD_VPOSTER_ROOT = r"E:\复现库\（可复现ERC）MAGTKD-main\MAGTKD-main\MELD\feature\video_poster"
MELD_VIDEO_NPY_DIR = {           # 每句 {Dialogue_ID}_{Utterance_ID}.npy → (8,3,224,224) f32
    "train": os.path.join(_MELD_VPOSTER_ROOT, "train"),
    "dev":   os.path.join(_MELD_VPOSTER_ROOT, "dev"),
    "test":  os.path.join(_MELD_VPOSTER_ROOT, "test"),
}
MELD_VIDEO_FEAT_DIR  = os.path.join(_DATA_ROOT, "meld_video_poster")        # 句级 768-d（8帧均值）
MELD_VIDEO_FRAME_DIR = os.path.join(_DATA_ROOT, "meld_video_poster_frame")  # 帧级序列 (N,8,768)
VIDEO_FRAME_LEN      = 8       # POSTER 输入即 8 帧人脸 → 8 帧视频序列（与音频帧级对称）

# ───────────────────────── 数据/Prompt ─────────────────────────
CONTEXT_WINDOW = 8        # 上下文取前 N 句（含目标句之前），控制序列长度
SEED = 42

# ───────────────────────── 训练超参（V1）─────────────────────────
LORA_R         = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
LR             = 1e-4
EPOCHS         = 5      # 全量 5163 句，LoRA 文本 2-4 epoch 见顶，6 留余量；按 dev 选 best
BATCH_SIZE     = 2      # V3 起降到 2（治 12GB 显存 spill），有效 batch 仍 = 16
GRAD_ACCUM     = 8      # 有效 batch = BATCH_SIZE * GRAD_ACCUM = 16（与 V1/V2 一致，可比）
MAX_GRAD_NORM  = 1.0
WARMUP_RATIO   = 0.05
MAX_SEQ_LEN    = 1024
USE_4BIT       = True

# ───────────────────────── V1R（检索增强 + 辅助任务，仿 InstructERC）─────────────────────────
RETRIEVAL_METHOD  = "auto"   # auto→优先 MiniLM 语义检索，失败回退 TF-IDF
EI_ALPHA          = 0.5      # emotion-impact 辅助任务 loss 权重（L = L_main + α·L_ei）
SPK_WARMUP_EPOCHS = 1        # stage1：speaker-ID 预热轮数（预热对话角色建模）
V1R_EPOCHS        = 6        # stage2：ERC 主任务 + emotion-impact 联合训练轮数

# ───────────────────────── V3.5（帧级序列 + 文本引导 cross-attention 路由）─────────────────────────
# 创新①真形态：音频从句级单向量→帧级序列，文本 token 作 query 路由 attend 音频帧。
# 不再有标量门控 ḡ（已 5 次证伪塌常数），不为 CP 服务。沿用 V2 前缀注入口径（K 前缀）可比。
AUDIO_FRAME_LEN   = 64       # 每句音频帧固定下采样到 T 帧（adaptive avg pool），控显存+定长 batch
ROUTE_QUERIES     = 4        # 文本引导路由输出的前缀 token 数 K（与 V2 前缀数一致，可比）
ROUTE_HEADS       = 4        # cross-attention 头数
