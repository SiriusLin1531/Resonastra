from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import csv
import json
import random
import re
import secrets
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import librosa
import torch

# ============================================================
# Make project root importable
# 让项目根目录可导入
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.adapters.gsv_prompt_tokenizer import GSVPromptTokenizer
from src.adapters.gsv_text_frontend import GSVTextFrontend
from src.data.stage1_fewshot_dataset import Stage1FewshotCollator, Stage1FewshotDataset


# ============================================================
# Constants
# 常量
# ============================================================
SEMANTIC_TOKEN_MIN = 0
SEMANTIC_TOKEN_MAX = 1023
SEMANTIC_PAD_ID = 1024

DEFAULT_TRAIN_RATIO = 0.9
SPLIT_REPORT_SCHEMA_VERSION = "voicelab_stage1_split_v2"


# ============================================================
# Data structures
# 数据结构
# ============================================================
@dataclass
class RawStage1Sample:
    item_id: str
    wav_path: str
    text: str
    language: str
    speaker_id: str
    split: str | None = None
    source_line: int | None = None


@dataclass(frozen=True)
class Stage1SplitSettings:
    """Normalized Stage1 split settings used by the builder."""

    strategy: str
    requested_train_ratio: float | None
    requested_val_ratio: float | None
    effective_train_ratio: float | None
    effective_val_ratio: float | None
    val_count: int | None
    requested_seed: int | None
    effective_seed: int | None
    respect_existing_split: bool
    existing_split_used: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuildStats:
    num_input: int = 0
    num_seen: int = 0
    num_kept: int = 0
    num_train: int = 0
    num_val: int = 0
    num_skipped_missing_wav: int = 0
    num_skipped_empty_text: int = 0
    num_skipped_frontend_error: int = 0
    num_skipped_tokenizer_error: int = 0
    num_skipped_duration: int = 0
    num_skipped_short_phoneme: int = 0
    num_skipped_short_semantic: int = 0
    num_skipped_bad_bert_shape: int = 0
    num_skipped_bad_semantic_range: int = 0
    num_overwritten: int = 0
    num_reused_existing: int = 0


# ============================================================
# Generic helpers
# 通用工具函数
# ============================================================
def _safe_item_id(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r'[\\/:*?"<>|\s]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False)


def _write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(_json_dumps(row) + "\n")


def _load_torch(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_torch(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def _path_rel_to(path: str | Path, root: str | Path) -> str:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _get_first(
    row: dict[str, Any],
    keys: list[str],
    default: Any = None,
) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _resolve_wav_path(
    raw_path: str,
    manifest_dir: Path,
    wav_root: str | None = None,
) -> Path:
    raw_path = str(raw_path).strip().strip('"')
    path = Path(raw_path)
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        if wav_root:
            root = Path(wav_root)
            candidates.append(root / path)
            candidates.append(root / path.name)
        candidates.append(manifest_dir / path)
        candidates.append(PROJECT_ROOT / path)

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    if candidates:
        return candidates[0].resolve()
    return path.resolve()


def _duration_sec(wav_path: str | Path) -> float:
    return float(librosa.get_duration(path=str(wav_path)))


def _validate_ratio(name: str, value: float) -> float:
    ratio = float(value)
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {ratio}.")
    return ratio


# ============================================================
# Manifest parsing
# Manifest 解析
# ============================================================
def _sample_from_json_row(
    row: dict[str, Any],
    line_idx: int,
    default_language: str,
    default_speaker_id: str,
) -> RawStage1Sample:
    wav_path = _get_first(
        row,
        ["wav_path", "audio_path", "audio", "path", "wav", "file"],
    )
    text = _get_first(
        row,
        ["text", "transcript", "normalized_text", "raw_text", "content"],
    )
    if wav_path is None or text is None:
        raise ValueError(
            "JSON manifest row must contain wav_path/audio/path and "
            f"text/transcript fields: {row}"
        )

    language = str(_get_first(row, ["language", "lang"], default_language))
    speaker_id = str(
        _get_first(
            row,
            ["speaker_id", "speaker", "spk", "spk_name"],
            default_speaker_id,
        )
    )
    item_id = _get_first(row, ["item_id", "id", "name", "utt_id", "wav_name"])
    if item_id is None:
        item_id = Path(str(wav_path)).stem or f"{line_idx:06d}"

    split = _get_first(row, ["split", "subset"], None)
    return RawStage1Sample(
        item_id=_safe_item_id(str(item_id)),
        wav_path=str(wav_path),
        text=str(text),
        language=language,
        speaker_id=speaker_id,
        split=None if split is None else str(split),
        source_line=line_idx,
    )


def _sample_from_delimited_fields(
    fields: list[str],
    line_idx: int,
    default_language: str,
    default_speaker_id: str,
) -> RawStage1Sample:
    fields = [item.strip() for item in fields]
    if len(fields) >= 4:
        wav_path, speaker_id, language = fields[0], fields[1], fields[2]
        text = "|".join(fields[3:])
    elif len(fields) == 3:
        wav_path, language, text = fields
        speaker_id = default_speaker_id
    elif len(fields) == 2:
        wav_path, text = fields
        language = default_language
        speaker_id = default_speaker_id
    else:
        raise ValueError(
            f"Unsupported manifest row with {len(fields)} fields: {fields}"
        )

    item_id = Path(wav_path).stem or f"{line_idx:06d}"
    return RawStage1Sample(
        item_id=_safe_item_id(item_id),
        wav_path=wav_path,
        text=text,
        language=language or default_language,
        speaker_id=speaker_id or default_speaker_id,
        split=None,
        source_line=line_idx,
    )


def load_raw_manifest(
    input_manifest: str | Path,
    default_language: str = "zh",
    default_speaker_id: str = "speaker",
) -> list[RawStage1Sample]:
    path = Path(input_manifest).resolve()
    suffix = path.suffix.lower()
    samples: list[RawStage1Sample] = []

    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as file:
            for line_idx, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"JSONL row must be object at line {line_idx}: {row}"
                    )
                samples.append(
                    _sample_from_json_row(
                        row,
                        line_idx,
                        default_language,
                        default_speaker_id,
                    )
                )
        return samples

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            if reader.fieldnames:
                for line_idx, row in enumerate(reader, start=2):
                    samples.append(
                        _sample_from_json_row(
                            dict(row),
                            line_idx,
                            default_language,
                            default_speaker_id,
                        )
                    )
                return samples

    with path.open("r", encoding="utf-8") as file:
        for line_idx, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                fields = line.split("|")
            elif "\t" in line:
                fields = line.split("\t")
            else:
                fields = next(csv.reader([line]))
            samples.append(
                _sample_from_delimited_fields(
                    fields,
                    line_idx,
                    default_language,
                    default_speaker_id,
                )
            )
    return samples


# ============================================================
# Split helpers
# 切分辅助函数
# ============================================================
def resolve_split_settings(
    *,
    samples: list[RawStage1Sample],
    train_ratio: float | None,
    val_ratio: float | None,
    val_count: int | None,
    requested_seed: int | None,
    respect_existing_split: bool,
) -> Stage1SplitSettings:
    """Normalize user/developer split arguments into one traceable contract."""

    if train_ratio is not None and val_ratio is not None:
        raise ValueError("train_ratio and val_ratio are mutually exclusive.")
    if val_count is not None and (
        train_ratio is not None or val_ratio is not None
    ):
        raise ValueError(
            "val_count cannot be combined with train_ratio or val_ratio."
        )

    existing_split_used = bool(
        respect_existing_split and any(sample.split for sample in samples)
    )

    requested_train_ratio = (
        None if train_ratio is None else _validate_ratio("train_ratio", train_ratio)
    )
    requested_val_ratio = (
        None if val_ratio is None else _validate_ratio("val_ratio", val_ratio)
    )

    if val_count is not None:
        resolved_val_count = int(val_count)
        if resolved_val_count <= 0:
            raise ValueError(f"val_count must be > 0, got {resolved_val_count}.")
        if len(samples) > 0 and resolved_val_count >= len(samples):
            raise ValueError(
                "val_count must be smaller than the number of input samples, "
                f"got val_count={resolved_val_count}, samples={len(samples)}."
            )
        effective_train_ratio = None
        effective_val_ratio = None
        strategy = "val_count"
    else:
        resolved_val_count = None
        if requested_train_ratio is not None:
            effective_train_ratio = requested_train_ratio
            effective_val_ratio = 1.0 - requested_train_ratio
        elif requested_val_ratio is not None:
            effective_val_ratio = requested_val_ratio
            effective_train_ratio = 1.0 - requested_val_ratio
        else:
            effective_train_ratio = DEFAULT_TRAIN_RATIO
            effective_val_ratio = 1.0 - DEFAULT_TRAIN_RATIO
        strategy = "train_ratio"

    if existing_split_used:
        strategy = "existing_manifest"
        effective_seed = None
    else:
        effective_seed = (
            int(requested_seed)
            if requested_seed is not None
            else int(secrets.randbits(32))
        )

    return Stage1SplitSettings(
        strategy=strategy,
        requested_train_ratio=requested_train_ratio,
        requested_val_ratio=requested_val_ratio,
        effective_train_ratio=effective_train_ratio,
        effective_val_ratio=effective_val_ratio,
        val_count=resolved_val_count,
        requested_seed=None if requested_seed is None else int(requested_seed),
        effective_seed=effective_seed,
        respect_existing_split=bool(respect_existing_split),
        existing_split_used=existing_split_used,
    )


def assign_splits(
    samples: list[RawStage1Sample],
    *,
    val_ratio: float | None,
    val_count: int | None,
    seed: int | None,
    respect_existing_split: bool,
) -> dict[str, str]:
    if respect_existing_split and any(sample.split for sample in samples):
        output: dict[str, str] = {}
        for sample in samples:
            split = str(sample.split or "train").lower()
            if split in {"valid", "validation", "dev"}:
                split = "val"
            if split not in {"train", "val"}:
                split = "train"
            output[sample.item_id] = split
        return output

    if seed is None:
        raise ValueError("A resolved effective seed is required for random split.")

    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)

    if val_count is None:
        if val_ratio is None:
            raise ValueError("val_ratio is required when val_count is not set.")
        n_val = int(round(len(samples) * float(val_ratio)))
    else:
        n_val = int(val_count)

    if len(samples) > 1:
        n_val = max(1, min(n_val, len(samples) - 1))
    else:
        n_val = 0

    val_set = set(indices[:n_val])
    return {
        sample.item_id: ("val" if index in val_set else "train")
        for index, sample in enumerate(samples)
    }


# ============================================================
# Build helpers
# 构建辅助函数
# ============================================================
def build_frontend_cache(
    frontend: GSVTextFrontend,
    sample: RawStage1Sample,
    frontend_path: Path,
) -> dict[str, Any]:
    output = frontend.prepare_inputs(
        text=sample.text,
        language=sample.language,
    )
    phoneme_ids = torch.LongTensor(output.phoneme_ids)
    phoneme_lens = torch.tensor([phoneme_ids.numel()], dtype=torch.long)
    bert_feature = output.bert_feature.detach().cpu().float()

    cache = {
        "item_id": sample.item_id,
        "text": sample.text,
        "language": sample.language,
        "speaker_id": sample.speaker_id,
        "norm_text": output.norm_text,
        "phones": output.phones,
        "phoneme_ids": phoneme_ids,
        "phoneme_lens": phoneme_lens,
        "bert_feature": bert_feature,
        "word2ph": output.word2ph,
        "frontend_version": "gsv_v2",
    }
    _save_torch(cache, frontend_path)
    return cache


def build_semantic_cache(
    tokenizer: GSVPromptTokenizer,
    sample: RawStage1Sample,
    wav_path: Path,
    semantic_path: Path,
    sovits_ckpt: str | None,
) -> dict[str, Any]:
    tokenizer_output = tokenizer.extract_prompt_tokens_from_wav(
        wav_path=wav_path,
        sovits_checkpoint_path=sovits_ckpt,
    )
    semantic_ids = tokenizer_output.prompt_semantic_1d.detach().cpu().long()
    cache = {
        "item_id": sample.item_id,
        "target_semantic_ids": semantic_ids,
        "target_semantic_len": int(semantic_ids.numel()),
        "semantic_frame_rate": "25hz",
        "source_wav_path": str(wav_path),
        "source_tokenizer": "gsv_prompt_tokenizer_v2",
        "source_sovits_ckpt": sovits_ckpt,
        "token_min": (
            int(semantic_ids.min().item())
            if semantic_ids.numel() > 0
            else None
        ),
        "token_max": (
            int(semantic_ids.max().item())
            if semantic_ids.numel() > 0
            else None
        ),
    }
    _save_torch(cache, semantic_path)
    return cache


def validate_built_cache(
    frontend_cache: dict[str, Any],
    semantic_cache: dict[str, Any],
    min_phoneme_len: int,
    min_semantic_len: int,
) -> tuple[bool, str]:
    phoneme_ids = frontend_cache.get("phoneme_ids")
    bert_feature = frontend_cache.get("bert_feature")
    semantic_ids = semantic_cache.get("target_semantic_ids")

    if not isinstance(phoneme_ids, torch.Tensor):
        return False, "phoneme_ids_not_tensor"
    if not isinstance(bert_feature, torch.Tensor):
        return False, "bert_feature_not_tensor"
    if not isinstance(semantic_ids, torch.Tensor):
        return False, "semantic_ids_not_tensor"

    if phoneme_ids.ndim != 1:
        return False, "phoneme_ids_not_1d"
    if bert_feature.ndim != 2:
        return False, "bert_feature_not_2d"
    if semantic_ids.ndim != 1:
        return False, "semantic_ids_not_1d"

    if int(phoneme_ids.numel()) < int(min_phoneme_len):
        return False, "short_phoneme"
    if int(semantic_ids.numel()) < int(min_semantic_len):
        return False, "short_semantic"
    if int(bert_feature.shape[-1]) != int(phoneme_ids.numel()):
        return False, "bad_bert_shape"
    if (
        int(semantic_ids.min().item()) < SEMANTIC_TOKEN_MIN
        or int(semantic_ids.max().item()) > SEMANTIC_TOKEN_MAX
    ):
        return False, "bad_semantic_range"

    return True, "ok"


# ============================================================
# CLI
# 命令行
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Stage1 v6.7.0 Original-style Few-shot dataset."
        )
    )
    parser.add_argument(
        "--input_manifest",
        type=str,
        required=True,
        help="Raw wav/text manifest path.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output dataset root.",
    )
    parser.add_argument(
        "--wav_root",
        type=str,
        default=None,
        help="Optional root for resolving relative wav paths.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Default language if manifest has none.",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default="speaker",
        help="Default speaker id if manifest has none.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    parser.add_argument("--use_half", action="store_true", help="Use fp16 on CUDA.")
    parser.add_argument(
        "--sovits_ckpt",
        type=str,
        default=None,
        help="Optional tokenizer SoVITS ckpt override.",
    )
    parser.add_argument(
        "--target_tail_silence_sec",
        type=float,
        default=0.0,
        help=(
            "Tail silence for target semantic extraction. Default 0.0 for "
            "original-style training target."
        ),
    )
    parser.add_argument(
        "--enforce_ref_duration_check",
        action="store_true",
        help=(
            "Enable 3~10s reference duration check. Default is disabled for "
            "training-target extraction."
        ),
    )

    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help=(
            "Training-set ratio. User-facing default is 0.9 when no split "
            "ratio/count argument is supplied."
        ),
    )
    split_group.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help=(
            "Legacy validation-set ratio. Kept for backward compatibility; "
            "new callers should use --train_ratio."
        ),
    )
    split_group.add_argument(
        "--val_count",
        type=int,
        default=None,
        help="Optional explicit validation item count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional reproducible split seed. When omitted, a random effective "
            "seed is generated and recorded in metadata/split.json."
        ),
    )
    parser.add_argument(
        "--respect_existing_split",
        action="store_true",
        help="Use split fields from the manifest when present.",
    )

    # Developer-only compatibility switch. The user WebUI intentionally does
    # not expose or pass this option; None means process all input samples.
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--min_phoneme_len",
        type=int,
        default=3,
        help="Minimum phoneme length.",
    )
    parser.add_argument(
        "--min_semantic_len",
        type=int,
        default=8,
        help="Minimum semantic length.",
    )
    parser.add_argument(
        "--max_duration_sec",
        type=float,
        default=None,
        help="Optional max wav duration filter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse and split manifest only; do not build caches.",
    )
    parser.add_argument(
        "--validate_dataset",
        action="store_true",
        help="Load built train/val dataset and one batch for validation.",
    )
    return parser


# ============================================================
# Main
# 主流程
# ============================================================
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    start_time = time.time()
    input_manifest = Path(args.input_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    manifest_dir = input_manifest.parent

    metadata_dir = output_root / "metadata"
    frontend_root = output_root / "frontend_cache"
    semantic_root = output_root / "semantic_cache"
    train_manifest_path = output_root / "train_manifest.jsonl"
    val_manifest_path = output_root / "val_manifest.jsonl"
    split_report_path = metadata_dir / "split.json"
    build_report_path = metadata_dir / "build_report.json"

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    raw_samples = load_raw_manifest(
        input_manifest=input_manifest,
        default_language=args.language,
        default_speaker_id=args.speaker_id,
    )

    if args.max_items is not None:
        max_items = int(args.max_items)
        if max_items <= 0:
            raise ValueError(
                f"Developer option --max_items must be > 0, got {max_items}."
            )
        raw_samples = raw_samples[:max_items]

    split_settings = resolve_split_settings(
        samples=raw_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        val_count=args.val_count,
        requested_seed=args.seed,
        respect_existing_split=args.respect_existing_split,
    )
    split_map = assign_splits(
        raw_samples,
        val_ratio=split_settings.effective_val_ratio,
        val_count=split_settings.val_count,
        seed=split_settings.effective_seed,
        respect_existing_split=split_settings.existing_split_used,
    )
    split_counts = {
        "train": sum(1 for split in split_map.values() if split == "train"),
        "val": sum(1 for split in split_map.values() if split == "val"),
    }

    print("=" * 60)
    print("Stage1 v6.7.0 few-shot dataset builder")
    print("=" * 60)
    print("input_manifest :", input_manifest)
    print("output_root    :", output_root)
    print("device         :", args.device)
    print("use_half       :", args.use_half)
    print("dry_run        :", args.dry_run)
    print("split_strategy :", split_settings.strategy)
    print("train_ratio    :", split_settings.effective_train_ratio)
    print("val_ratio      :", split_settings.effective_val_ratio)
    print("val_count      :", split_settings.val_count)
    print("requested_seed :", split_settings.requested_seed)
    print("effective_seed :", split_settings.effective_seed)
    print("=" * 60)

    split_payload = {
        "schema_version": SPLIT_REPORT_SCHEMA_VERSION,
        **split_settings.to_dict(),
        "num_input": len(raw_samples),
        "split_counts": split_counts,
        "items": split_map,
    }
    _write_json(split_report_path, split_payload)

    args_payload = vars(args).copy()
    args_payload["max_items_is_developer_only"] = True

    if args.dry_run:
        preview = [
            asdict(sample)
            | {"assigned_split": split_map.get(sample.item_id, "train")}
            for sample in raw_samples[:10]
        ]
        report = {
            "status": "DRY_RUN_OK",
            "task": (
                "Stage1 v6.7.0 Original-style Few-shot dataset "
                "split dry-run"
            ),
            "args": args_payload,
            "split": split_payload,
            "num_input": len(raw_samples),
            "preview": preview,
            "split_counts": split_counts,
            "elapsed_sec": round(time.time() - start_time, 4),
        }
        _write_json(build_report_path, report)
        print("[DRY_RUN] Parsed manifest and wrote report:", build_report_path)
        return

    stats = BuildStats(num_input=len(raw_samples))

    print("[1/3] Initializing frontend and tokenizer...")
    frontend = GSVTextFrontend(
        version="v2",
        device=args.device,
        use_half=args.use_half,
    )
    tokenizer = GSVPromptTokenizer(
        version="v2",
        device=args.device,
        use_half=args.use_half,
        tail_silence_sec=args.target_tail_silence_sec,
        enforce_ref_seconds=args.enforce_ref_duration_check,
    )

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    print("[2/3] Building cache files...")
    for index, sample in enumerate(raw_samples, start=1):
        stats.num_seen += 1
        split = split_map.get(sample.item_id, "train")
        wav_path = _resolve_wav_path(
            sample.wav_path,
            manifest_dir=manifest_dir,
            wav_root=args.wav_root,
        )

        print(f"[{index}/{len(raw_samples)}] {sample.item_id} -> {split}")

        if not wav_path.exists():
            stats.num_skipped_missing_wav += 1
            skipped.append(
                {
                    "item_id": sample.item_id,
                    "reason": "missing_wav",
                    "wav_path": str(wav_path),
                }
            )
            print("  [SKIP] missing wav:", wav_path)
            continue

        if not sample.text.strip():
            stats.num_skipped_empty_text += 1
            skipped.append(
                {"item_id": sample.item_id, "reason": "empty_text"}
            )
            print("  [SKIP] empty text")
            continue

        try:
            duration = _duration_sec(wav_path)
        except Exception:
            duration = None

        if (
            args.max_duration_sec is not None
            and duration is not None
            and duration > float(args.max_duration_sec)
        ):
            stats.num_skipped_duration += 1
            skipped.append(
                {
                    "item_id": sample.item_id,
                    "reason": "duration",
                    "duration_sec": duration,
                }
            )
            print(
                f"  [SKIP] duration {duration:.3f}s "
                f"> {args.max_duration_sec}s"
            )
            continue

        frontend_path = frontend_root / split / f"{sample.item_id}.pt"
        semantic_path = semantic_root / split / f"{sample.item_id}.pt"

        if (
            frontend_path.exists()
            and semantic_path.exists()
            and not args.overwrite
        ):
            try:
                frontend_cache = _load_torch(frontend_path)
                semantic_cache = _load_torch(semantic_path)
                stats.num_reused_existing += 1
            except Exception:
                frontend_cache = None
                semantic_cache = None
        else:
            if frontend_path.exists() or semantic_path.exists():
                stats.num_overwritten += 1

            try:
                frontend_cache = build_frontend_cache(
                    frontend,
                    sample,
                    frontend_path,
                )
            except Exception as exc:
                stats.num_skipped_frontend_error += 1
                skipped.append(
                    {
                        "item_id": sample.item_id,
                        "reason": "frontend_error",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print("  [SKIP] frontend error:", exc)
                continue

            try:
                semantic_cache = build_semantic_cache(
                    tokenizer,
                    sample,
                    wav_path,
                    semantic_path,
                    args.sovits_ckpt,
                )
            except Exception as exc:
                stats.num_skipped_tokenizer_error += 1
                skipped.append(
                    {
                        "item_id": sample.item_id,
                        "reason": "tokenizer_error",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print("  [SKIP] tokenizer error:", exc)
                continue

        if frontend_cache is None or semantic_cache is None:
            stats.num_skipped_frontend_error += 1
            skipped.append(
                {
                    "item_id": sample.item_id,
                    "reason": "existing_cache_load_error",
                }
            )
            print("  [SKIP] existing cache could not be loaded")
            continue

        valid, reason = validate_built_cache(
            frontend_cache,
            semantic_cache,
            min_phoneme_len=args.min_phoneme_len,
            min_semantic_len=args.min_semantic_len,
        )
        if not valid:
            if reason == "short_phoneme":
                stats.num_skipped_short_phoneme += 1
            elif reason == "short_semantic":
                stats.num_skipped_short_semantic += 1
            elif reason == "bad_bert_shape":
                stats.num_skipped_bad_bert_shape += 1
            elif reason == "bad_semantic_range":
                stats.num_skipped_bad_semantic_range += 1
            else:
                stats.num_skipped_frontend_error += 1
            skipped.append({"item_id": sample.item_id, "reason": reason})
            print("  [SKIP] validation failed:", reason)
            continue

        phoneme_len = int(frontend_cache["phoneme_ids"].numel())
        semantic_len = int(
            semantic_cache["target_semantic_ids"].numel()
        )
        row = {
            "item_id": sample.item_id,
            "wav_path": str(wav_path),
            "text": sample.text,
            "language": sample.language,
            "speaker_id": sample.speaker_id,
            "frontend_path": _path_rel_to(frontend_path, output_root),
            "target_semantic_path": _path_rel_to(
                semantic_path,
                output_root,
            ),
            "duration_sec": duration,
            "phoneme_len": phoneme_len,
            "target_semantic_len": semantic_len,
            "semantic_frame_rate": "25hz",
            "source_stage2_tokenizer_ckpt": args.sovits_ckpt,
            "frontend_version": "gsv_v2",
            "split": split,
        }
        if split == "val":
            val_rows.append(row)
        else:
            train_rows.append(row)
        stats.num_kept += 1

    stats.num_train = len(train_rows)
    stats.num_val = len(val_rows)

    print("[3/3] Writing manifests and report...")
    _write_jsonl(train_manifest_path, train_rows)
    _write_jsonl(val_manifest_path, val_rows)

    report = {
        "status": (
            "OK"
            if stats.num_kept > 0 and stats.num_train > 0
            else "FAILED"
        ),
        "task": (
            "Stage1 v6.7.0 Original-style Few-shot dataset build"
        ),
        "args": args_payload,
        "split": split_payload,
        "paths": {
            "input_manifest": str(input_manifest),
            "output_root": str(output_root),
            "train_manifest": str(train_manifest_path),
            "val_manifest": str(val_manifest_path),
            "frontend_root": str(frontend_root),
            "semantic_root": str(semantic_root),
            "split_report": str(split_report_path),
        },
        "stats": asdict(stats),
        "skipped_preview": skipped[:50],
        "num_skipped_total": len(skipped),
        "elapsed_sec": round(time.time() - start_time, 4),
    }

    if args.validate_dataset and report["status"] == "OK":
        validation: dict[str, Any] = {}
        try:
            if train_rows:
                train_dataset = Stage1FewshotDataset(
                    train_manifest_path,
                    root_dir=output_root,
                    validate_on_load=False,
                )
                validation["train_dataset_len"] = len(train_dataset)
                batch = Stage1FewshotCollator()([train_dataset[0]])
                validation["single_train_batch"] = {
                    "phoneme_ids_shape": list(
                        batch["phoneme_ids"].shape
                    ),
                    "bert_feature_shape": list(
                        batch["bert_feature"].shape
                    ),
                    "semantic_ids_shape": list(
                        batch["semantic_ids"].shape
                    ),
                }
            if val_rows:
                val_dataset = Stage1FewshotDataset(
                    val_manifest_path,
                    root_dir=output_root,
                    validate_on_load=False,
                )
                validation["val_dataset_len"] = len(val_dataset)
        except Exception as exc:
            validation["status"] = "FAILED"
            validation["error"] = str(exc)
            validation["traceback"] = traceback.format_exc()
        else:
            validation["status"] = "OK"
        report["dataset_validation"] = validation

    _write_json(build_report_path, report)

    print("=" * 60)
    print("Stage1 few-shot dataset build summary")
    print("=" * 60)
    print("status         :", report["status"])
    print("num_input      :", stats.num_input)
    print("num_kept       :", stats.num_kept)
    print("num_train      :", stats.num_train)
    print("num_val        :", stats.num_val)
    print("num_skipped    :", len(skipped))
    print("split_strategy :", split_settings.strategy)
    print("effective_seed :", split_settings.effective_seed)
    print("train_manifest :", train_manifest_path)
    print("val_manifest   :", val_manifest_path)
    print("split_report   :", split_report_path)
    print("build_report   :", build_report_path)
    print("=" * 60)

    if report["status"] != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
