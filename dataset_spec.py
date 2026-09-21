"""
跨数据集统一规格（LabelSpec）+ 注册中心。

目的：让 infer/run_v1/run_v2/run_v3p5 不再硬编码 IEMOCAP 标签与候选集，
  通过 `get_spec(name)` 拿到该数据集的标签方案 + 数据模块 + 多模态特征目录，
  从而 V1/V2/V3.5 一套代码同时跑 IEMOCAP / MELD。

向后兼容：infer 各函数 `spec=None` 时回退到 config 的 IEMOCAP 全局（C.LABEL_LIST…），
  保证既有 V0/V1/V2/V3.5 的 IEMOCAP 行为与历史结果完全不变。

新增数据集只需在 _REGISTRY 里加一项。
"""
from dataclasses import dataclass
from typing import Optional

import config as C


@dataclass(frozen=True)
class LabelSpec:
    name: str
    label_list: list           # 固定顺序 = label id
    label2id: dict
    id2label: dict
    label2word: dict           # 给 LLM 看的自然词（候选打分用）
    num_classes: int
    data_module_name: str      # "data_iemocap" / "data_meld"
    # 多模态特征目录（V2 句级 / V3.5 帧级），文本数据集可为 None
    audio_feat_dir: Optional[str] = None
    audio_frame_dir: Optional[str] = None
    video_feat_dir: Optional[str] = None
    video_frame_dir: Optional[str] = None
    audio_pt_tmpl: Optional[str] = None    # 句级 .pt 文件名模板（{split} 占位）
    audio_frame_pt_tmpl: Optional[str] = None
    video_pt_tmpl: Optional[str] = None
    video_frame_pt_tmpl: Optional[str] = None
    # 特征 .pt 的 order_keys 与 sample 的对齐 key 形态：
    #   "pipe"   -> f"{dialogue_id}|{utt_id}"（IEMOCAP 既有口径）
    #   "clip_id"-> sample["clip_id"]（MELD：dia{D}_utt{U}）
    feat_key_mode: str = "pipe"

    def get_data_module(self):
        import importlib
        return importlib.import_module(self.data_module_name)

    def feat_key(self, sample) -> str:
        """返回该 sample 对应的多模态特征对齐 key（与提取脚本 order_keys 一致）。"""
        if self.feat_key_mode == "clip_id":
            return sample["clip_id"]
        return f"{sample['dialogue_id']}|{sample['utt_id']}"


def _iemocap_spec() -> LabelSpec:
    return LabelSpec(
        name="iemocap",
        label_list=list(C.LABEL_LIST),
        label2id=dict(C.LABEL2ID),
        id2label=dict(C.ID2LABEL),
        label2word=dict(C.LABEL2WORD),
        num_classes=C.NUM_CLASSES,
        data_module_name="data_iemocap",
        audio_feat_dir=C.AUDIO_FEAT_DIR,
        audio_frame_dir=C.AUDIO_FRAME_DIR,
        video_feat_dir=getattr(C, "VIDEO_FEAT_DIR", None),
        video_frame_dir=None,
        # IEMOCAP 既有命名（保持与历史产物一致；如与实际不符在用到时再核对）
        audio_pt_tmpl="IEMOCAP_{split}_emotion2vec.pt",
        audio_frame_pt_tmpl="IEMOCAP_{split}_emotion2vec_frame.pt",
        video_pt_tmpl="{split}_timesformer.pt",
        video_frame_pt_tmpl=None,
        feat_key_mode="pipe",
    )


def _meld_spec() -> LabelSpec:
    return LabelSpec(
        name="meld",
        label_list=list(C.MELD_LABEL_LIST),
        label2id=dict(C.MELD_LABEL2ID),
        id2label=dict(C.MELD_ID2LABEL),
        label2word=dict(C.MELD_LABEL2WORD),
        num_classes=C.MELD_NUM_CLASSES,
        data_module_name="data_meld",
        audio_feat_dir=C.MELD_AUDIO_FEAT_DIR,
        audio_frame_dir=C.MELD_AUDIO_FRAME_DIR,
        video_feat_dir=C.MELD_VIDEO_FEAT_DIR,
        video_frame_dir=C.MELD_VIDEO_FRAME_DIR,
        audio_pt_tmpl="MELD_{split}_emotion2vec.pt",
        audio_frame_pt_tmpl="MELD_{split}_emotion2vec_frame.pt",
        video_pt_tmpl="MELD_{split}_poster.pt",
        video_frame_pt_tmpl="MELD_{split}_poster_frame.pt",
        feat_key_mode="clip_id",
    )


_REGISTRY = {
    "iemocap": _iemocap_spec,
    "meld": _meld_spec,
}


def get_spec(name: str = "iemocap") -> LabelSpec:
    name = (name or "iemocap").lower()
    if name not in _REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; choices={list(_REGISTRY)}")
    return _REGISTRY[name]()


def default_spec() -> LabelSpec:
    """infer 内部回退：等价于历史 IEMOCAP 全局口径。"""
    return _iemocap_spec()
