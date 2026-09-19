from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch
from torch.utils.data import Dataset


# ============================================================
# Constants
# 常量
# ============================================================
DEFAULT_PHONEME_PAD_ID = 0
DEFAULT_SEMANTIC_PAD_ID = 1024


# ============================================================
# IO helpers
# IO 辅助函数
# ============================================================
def _load_torch(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """
    EN:
    Load a torch object with compatibility across PyTorch versions.

    ZH:
    兼容不同 PyTorch 版本的 torch.load 包装。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_idx}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Each JSONL row must be an object, got {type(obj)} at {path}:{line_idx}")
            rows.append(obj)
    return rows


def _resolve_cache_path(raw_path: str | Path, root_dir: str | Path) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw
    return (Path(root_dir) / raw).resolve()


def _as_1d_long(x: Any, name: str) -> torch.LongTensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.detach().cpu().long()
    if x.ndim == 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D or (1, T), got shape={tuple(x.shape)}")
    return x


def _as_bert_2d(x: Any, name: str = "bert_feature") -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.detach().cpu().float()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D or (1, C, T), got shape={tuple(x.shape)}")
    return x.contiguous()


# ============================================================
# Data structures
# 数据结构
# ============================================================
@dataclass(frozen=True)
class Stage1FewshotManifestItem:
    """
    EN:
    One row in a v6.7.0 stage-1 few-shot manifest.

    ZH:
    v6.7.0 stage1 few-shot manifest 中的一条记录。
    """

    item_id: str
    wav_path: str
    text: str
    language: str
    speaker_id: str
    frontend_path: str
    target_semantic_path: str
    duration_sec: float | None = None
    phoneme_len: int | None = None
    target_semantic_len: int | None = None
    split: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Stage1FewshotManifestItem":
        required = [
            "item_id",
            "wav_path",
            "text",
            "language",
            "speaker_id",
            "frontend_path",
            "target_semantic_path",
        ]
        missing = [k for k in required if k not in row or row[k] is None]
        if missing:
            raise KeyError(f"Manifest row missing required keys: {missing}. row={row}")
        return cls(
            item_id=str(row["item_id"]),
            wav_path=str(row["wav_path"]),
            text=str(row["text"]),
            language=str(row["language"]),
            speaker_id=str(row["speaker_id"]),
            frontend_path=str(row["frontend_path"]),
            target_semantic_path=str(row["target_semantic_path"]),
            duration_sec=None if row.get("duration_sec") is None else float(row.get("duration_sec")),
            phoneme_len=None if row.get("phoneme_len") is None else int(row.get("phoneme_len")),
            target_semantic_len=None if row.get("target_semantic_len") is None else int(row.get("target_semantic_len")),
            split=None if row.get("split") is None else str(row.get("split")),
        )


# ============================================================
# Dataset
# 数据集
# ============================================================
class Stage1FewshotDataset(Dataset):
    """
    EN:
    Dataset for VoiceLab v6.7.0 Original-style Stage1 Few-shot.

    It mirrors the original GPT-SoVITS stage-1 training contract:
        phoneme_ids + bert_feature + full target semantic_ids
    are returned for teacher-forcing semantic CE training.

    No fixed reference prompt and no manually sliced prefix are used in the
    dataset. Prompt semantic is only an inference-time concept in v6.7.0.

    ZH:
    VoiceLab v6.7.0 Original-style Stage1 Few-shot 数据集。

    它对齐原 GPT-SoVITS stage1 训练契约：
        phoneme_ids + bert_feature + full target semantic_ids
    用于 teacher-forcing semantic CE 训练。

    数据集阶段不使用 fixed reference prompt，也不人为切 same-sample prefix。
    在 v6.7.0 中，prompt semantic 只属于推理阶段概念。
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path | None = None,
        map_location: str | torch.device = "cpu",
        max_items: int | None = None,
        validate_on_load: bool = False,
        strict: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if root_dir is None:
            root_dir = self.manifest_path.parent
        self.root_dir = Path(root_dir).resolve()
        self.map_location = map_location
        self.strict = bool(strict)

        rows = _read_jsonl(self.manifest_path)
        if max_items is not None:
            rows = rows[: int(max_items)]
        self.items = [Stage1FewshotManifestItem.from_dict(row) for row in rows]

        if len(self.items) == 0:
            raise ValueError(f"No items found in manifest: {self.manifest_path}")

        if validate_on_load:
            self.validate_all()

    def __len__(self) -> int:
        return len(self.items)

    def _load_item_cache(self, item: Stage1FewshotManifestItem) -> tuple[dict[str, Any], dict[str, Any]]:
        frontend_path = _resolve_cache_path(item.frontend_path, self.root_dir)
        semantic_path = _resolve_cache_path(item.target_semantic_path, self.root_dir)

        if not frontend_path.exists():
            raise FileNotFoundError(f"Frontend cache not found for item={item.item_id}: {frontend_path}")
        if not semantic_path.exists():
            raise FileNotFoundError(f"Semantic cache not found for item={item.item_id}: {semantic_path}")

        frontend = _load_torch(frontend_path, map_location=self.map_location)
        semantic = _load_torch(semantic_path, map_location=self.map_location)

        if not isinstance(frontend, dict):
            raise TypeError(f"Frontend cache must be dict, got {type(frontend)}: {frontend_path}")
        if not isinstance(semantic, dict):
            raise TypeError(f"Semantic cache must be dict, got {type(semantic)}: {semantic_path}")

        return frontend, semantic

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        frontend, semantic = self._load_item_cache(item)

        phoneme_ids = _as_1d_long(frontend.get("phoneme_ids"), "phoneme_ids")
        bert_feature = _as_bert_2d(frontend.get("bert_feature"), "bert_feature")
        semantic_ids = _as_1d_long(semantic.get("target_semantic_ids"), "target_semantic_ids")

        phoneme_len = int(phoneme_ids.numel())
        semantic_len = int(semantic_ids.numel())

        if bert_feature.shape[-1] != phoneme_len:
            raise ValueError(
                f"BERT/phoneme length mismatch for item={item.item_id}: "
                f"bert T={bert_feature.shape[-1]}, phoneme_len={phoneme_len}"
            )

        if self.strict:
            if semantic_len <= 0:
                raise ValueError(f"Empty target semantic for item={item.item_id}")
            if int(semantic_ids.min().item()) < 0:
                raise ValueError(f"Negative semantic token found for item={item.item_id}")
            if int(semantic_ids.max().item()) > DEFAULT_SEMANTIC_PAD_ID:
                raise ValueError(
                    f"Semantic token above pad/eos id {DEFAULT_SEMANTIC_PAD_ID} for item={item.item_id}: "
                    f"max={int(semantic_ids.max().item())}"
                )

        return {
            "item_id": item.item_id,
            "wav_path": item.wav_path,
            "text": item.text,
            "language": item.language,
            "speaker_id": item.speaker_id,
            "split": item.split,
            "phoneme_ids": phoneme_ids,
            "phoneme_ids_len": phoneme_len,
            "bert_feature": bert_feature,
            "semantic_ids": semantic_ids,
            "semantic_ids_len": semantic_len,
            "manifest": item,
        }

    def validate_all(self) -> dict[str, Any]:
        """
        EN:
        Load every item once and return simple validation stats.

        ZH:
        逐条加载一次所有样本，并返回基础校验统计。
        """
        phoneme_lens: list[int] = []
        semantic_lens: list[int] = []
        for i in range(len(self)):
            sample = self[i]
            phoneme_lens.append(int(sample["phoneme_ids_len"]))
            semantic_lens.append(int(sample["semantic_ids_len"]))
        return {
            "num_items": len(self),
            "phoneme_len_min": min(phoneme_lens),
            "phoneme_len_max": max(phoneme_lens),
            "semantic_len_min": min(semantic_lens),
            "semantic_len_max": max(semantic_lens),
        }


# ============================================================
# Collator
# Collator
# ============================================================
class Stage1FewshotCollator:
    """
    EN:
    Collator for original-style stage-1 teacher-forcing training.

    Output keys intentionally align with GPT-SoVITS stage-1 names:
        phoneme_ids, phoneme_ids_len, semantic_ids, semantic_ids_len, bert_feature

    ZH:
    原 GPT-SoVITS 风格 stage1 teacher-forcing 训练用 collator。

    输出 key 尽量对齐 GPT-SoVITS stage1：
        phoneme_ids, phoneme_ids_len, semantic_ids, semantic_ids_len, bert_feature
    """

    def __init__(
        self,
        phoneme_pad_id: int = DEFAULT_PHONEME_PAD_ID,
        semantic_pad_id: int = DEFAULT_SEMANTIC_PAD_ID,
        bert_pad_value: float = 0.0,
    ) -> None:
        self.phoneme_pad_id = int(phoneme_pad_id)
        self.semantic_pad_id = int(semantic_pad_id)
        self.bert_pad_value = float(bert_pad_value)

    def __call__(self, samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
        samples = list(samples)
        if len(samples) == 0:
            raise ValueError("Stage1FewshotCollator got an empty batch.")

        batch_size = len(samples)
        max_phoneme_len = max(int(s["phoneme_ids_len"]) for s in samples)
        max_semantic_len = max(int(s["semantic_ids_len"]) for s in samples)
        bert_dim = int(samples[0]["bert_feature"].shape[0])

        phoneme_ids = torch.full(
            (batch_size, max_phoneme_len),
            fill_value=self.phoneme_pad_id,
            dtype=torch.long,
        )
        semantic_ids = torch.full(
            (batch_size, max_semantic_len),
            fill_value=self.semantic_pad_id,
            dtype=torch.long,
        )
        bert_feature = torch.full(
            (batch_size, bert_dim, max_phoneme_len),
            fill_value=self.bert_pad_value,
            dtype=torch.float32,
        )

        phoneme_lens = torch.zeros(batch_size, dtype=torch.long)
        semantic_lens = torch.zeros(batch_size, dtype=torch.long)

        item_ids: list[str] = []
        wav_paths: list[str] = []
        texts: list[str] = []
        languages: list[str] = []
        speaker_ids: list[str] = []

        for i, sample in enumerate(samples):
            p = sample["phoneme_ids"]
            y = sample["semantic_ids"]
            b = sample["bert_feature"]
            p_len = int(sample["phoneme_ids_len"])
            y_len = int(sample["semantic_ids_len"])

            if b.shape[0] != bert_dim:
                raise ValueError(
                    f"BERT dim mismatch in batch: expected {bert_dim}, got {b.shape[0]} for item={sample['item_id']}"
                )
            if b.shape[-1] != p_len:
                raise ValueError(
                    f"BERT/phoneme length mismatch in collator for item={sample['item_id']}: "
                    f"bert T={b.shape[-1]}, phoneme_len={p_len}"
                )

            phoneme_ids[i, :p_len] = p
            semantic_ids[i, :y_len] = y
            bert_feature[i, :, :p_len] = b
            phoneme_lens[i] = p_len
            semantic_lens[i] = y_len

            item_ids.append(str(sample["item_id"]))
            wav_paths.append(str(sample["wav_path"]))
            texts.append(str(sample["text"]))
            languages.append(str(sample["language"]))
            speaker_ids.append(str(sample["speaker_id"]))

        return {
            "item_ids": item_ids,
            "wav_paths": wav_paths,
            "texts": texts,
            "languages": languages,
            "speaker_ids": speaker_ids,
            "phoneme_ids": phoneme_ids,
            "phoneme_ids_len": phoneme_lens,
            "bert_feature": bert_feature,
            "semantic_ids": semantic_ids,
            "semantic_ids_len": semantic_lens,
        }


__all__ = [
    "DEFAULT_PHONEME_PAD_ID",
    "DEFAULT_SEMANTIC_PAD_ID",
    "Stage1FewshotManifestItem",
    "Stage1FewshotDataset",
    "Stage1FewshotCollator",
]
