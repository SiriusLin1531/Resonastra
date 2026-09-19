from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ============================================================
# Data structures
# 数据结构
# ============================================================
@dataclass
class SplitConfig:
    """
    EN:
    Configuration for splitting one directory of .pt samples into
    train / validation subsets.

    ZH:
    将一个 .pt 样本目录切分为 train / validation 子集时使用的配置。
    """

    data_root: Path
    train_out: Path
    val_out: Path

    train_ratio: float | None
    val_count: int | None

    shuffle: bool
    seed: int

    mode: str
    overwrite: bool
    report_path: Path | None


# ============================================================
# Utility functions
# 工具函数
# ============================================================
def str2bool(x: str) -> bool:
    """
    EN:
    Convert common string forms to bool.

    ZH:
    将常见字符串形式转换为布尔值。
    """
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def natural_sort_key(text: str) -> list[Any]:
    """
    EN:
    Natural sort key for strings containing digits.

    Example:
        000001      -> [1]
        000001_00   -> [1, "_", 0]
        000010      -> [10]

    ZH:
    针对带数字字符串的自然排序键。

    例如：
        000001      -> [1]
        000001_00   -> [1, "_", 0]
        000010      -> [10]
    """
    text = str(text)
    parts = re.split(r"(\d+)", text)
    key: list[Any] = []
    for part in parts:
        if part == "":
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def scan_pt_files(data_root: Path) -> list[Path]:
    """
    EN:
    Scan all .pt files directly under data_root.

    Note:
    First version only scans the top level of data_root,
    not recursive subdirectories.

    ZH:
    扫描 data_root 目录下一层的所有 .pt 文件。

    注意：
    第一版只扫描 data_root 顶层，不递归子目录。
    """
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"data_root is not a directory: {data_root}")

    pt_files = [p.resolve() for p in data_root.glob("*.pt") if p.is_file()]
    pt_files = sorted(pt_files, key=lambda p: natural_sort_key(p.stem))
    return pt_files


def ensure_clean_dir(path: Path, overwrite: bool) -> Path:
    """
    EN:
    Ensure one output directory is ready.

    Behavior:
    - If directory does not exist: create it
    - If exists and empty: keep it
    - If exists and non-empty:
        - overwrite=True  -> remove and recreate
        - overwrite=False -> raise error

    ZH:
    确保输出目录可用。

    行为：
    - 若目录不存在：创建
    - 若目录存在且为空：保留
    - 若目录存在且非空：
        - overwrite=True  -> 删除后重建
        - overwrite=False -> 报错
    """
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {path}")

        has_any = any(path.iterdir())
        if has_any:
            if overwrite:
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(
                    f"Output directory already exists and is not empty: {path}\n"
                    f"Please use --overwrite true if you want to replace it."
                )
    else:
        path.mkdir(parents=True, exist_ok=True)

    return path.resolve()


def copy_or_move_one_file(src: Path, dst: Path, mode: str) -> None:
    """
    EN:
    Transfer one .pt file from src to dst according to mode.

    Supported modes:
    - copy
    - move
    - hardlink

    ZH:
    根据 mode 将单个 .pt 文件从 src 处理到 dst。

    支持模式：
    - copy
    - move
    - hardlink
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "move":
        shutil.move(str(src), str(dst))
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create hardlink from\n  {src}\n-> {dst}\n"
                f"Hardlink often requires same filesystem / enough permissions."
            ) from e
        return

    raise ValueError(f"Unsupported mode: {mode}")


def resolve_report_path(config: SplitConfig) -> Path:
    """
    EN:
    Resolve report output path.

    If user did not provide report_path, default to:
        parent(train_out)/split_report.json

    ZH:
    解析 split report 的输出路径。

    若用户没有提供 report_path，则默认输出到：
        parent(train_out)/split_report.json
    """
    if config.report_path is not None:
        return config.report_path.resolve()

    return (config.train_out.parent / "split_report.json").resolve()


def compute_split(
    pt_files: list[Path],
    train_ratio: float | None,
    val_count: int | None,
    shuffle: bool,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """
    EN:
    Compute train / validation split from a list of .pt files.

    Supported split strategies:
    1. train_ratio
    2. val_count

    Rules:
    - Exactly one of train_ratio / val_count must be provided
    - Always keep both train and val non-empty

    ZH:
    根据 .pt 文件列表计算 train / validation 划分。

    支持两种方式：
    1. train_ratio
    2. val_count

    规则：
    - train_ratio / val_count 必须二选一
    - train 和 val 都必须非空
    """
    n = len(pt_files)
    if n == 0:
        raise ValueError("No .pt files were found to split.")

    files = list(pt_files)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(files)

    if (train_ratio is None) == (val_count is None):
        raise ValueError("Exactly one of --train_ratio or --val_count must be provided.")

    if train_ratio is not None:
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

        train_n = int(round(n * train_ratio))
        train_n = max(1, min(train_n, n - 1))
        val_n = n - train_n

    else:
        assert val_count is not None
        if val_count <= 0:
            raise ValueError(f"val_count must be > 0, got {val_count}")
        if val_count >= n:
            raise ValueError(
                f"val_count must be smaller than total number of files ({n}), got {val_count}"
            )

        val_n = val_count
        train_n = n - val_n

    train_files = files[:train_n]
    val_files = files[train_n:]

    if len(train_files) == 0 or len(val_files) == 0:
        raise ValueError(
            f"Invalid split result: train={len(train_files)}, val={len(val_files)}"
        )

    return train_files, val_files


def save_json(obj: dict[str, Any], path: Path) -> None:
    """
    EN:
    Save JSON file.

    ZH:
    保存 JSON 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# CLI parsing
# 命令行参数解析
# ============================================================
def parse_args() -> SplitConfig:
    """
    EN:
    Parse CLI arguments into SplitConfig.

    ZH:
    将命令行参数解析成 SplitConfig。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)

    # Exactly one of these should be provided
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_count", type=int, default=None)

    parser.add_argument("--shuffle", type=str, default="false")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "move", "hardlink"])
    parser.add_argument("--overwrite", type=str, default="false")
    parser.add_argument("--report_path", type=str, default=None)

    args = parser.parse_args()

    return SplitConfig(
        data_root=Path(args.data_root).resolve(),
        train_out=Path(args.train_out).resolve(),
        val_out=Path(args.val_out).resolve(),
        train_ratio=args.train_ratio,
        val_count=args.val_count,
        shuffle=str2bool(args.shuffle),
        seed=int(args.seed),
        mode=str(args.mode).strip().lower(),
        overwrite=str2bool(args.overwrite),
        report_path=Path(args.report_path).resolve() if args.report_path else None,
    )


# ============================================================
# Main logic
# 主逻辑
# ============================================================
def main() -> None:
    config = parse_args()

    # --------------------------------------------------------
    # Basic path validation
    # 基础路径检查
    # --------------------------------------------------------
    if config.data_root == config.train_out:
        raise ValueError("data_root and train_out cannot be the same path.")
    if config.data_root == config.val_out:
        raise ValueError("data_root and val_out cannot be the same path.")
    if config.train_out == config.val_out:
        raise ValueError("train_out and val_out cannot be the same path.")

    pt_files = scan_pt_files(config.data_root)
    if len(pt_files) == 0:
        raise ValueError(f"No .pt files found under data_root: {config.data_root}")

    train_files, val_files = compute_split(
        pt_files=pt_files,
        train_ratio=config.train_ratio,
        val_count=config.val_count,
        shuffle=config.shuffle,
        seed=config.seed,
    )

    train_out = ensure_clean_dir(config.train_out, overwrite=config.overwrite)
    val_out = ensure_clean_dir(config.val_out, overwrite=config.overwrite)

    print("====================================================")
    print("Stage2 PT dataset split started")
    print(f"data_root    : {config.data_root}")
    print(f"train_out    : {train_out}")
    print(f"val_out      : {val_out}")
    print(f"mode         : {config.mode}")
    print(f"shuffle      : {config.shuffle}")
    print(f"seed         : {config.seed}")
    print(f"total_files  : {len(pt_files)}")
    print(f"train_files  : {len(train_files)}")
    print(f"val_files    : {len(val_files)}")
    print("====================================================")

    # --------------------------------------------------------
    # Execute split
    # 执行切分
    # --------------------------------------------------------
    for src in train_files:
        dst = train_out / src.name
        copy_or_move_one_file(src, dst, config.mode)

    for src in val_files:
        dst = val_out / src.name
        copy_or_move_one_file(src, dst, config.mode)

    # --------------------------------------------------------
    # Save split report
    # 保存切分报告
    # --------------------------------------------------------
    report_path = resolve_report_path(config)
    report = {
        "data_root": str(config.data_root),
        "train_out": str(train_out),
        "val_out": str(val_out),
        "mode": config.mode,
        "shuffle": config.shuffle,
        "seed": config.seed,
        "train_ratio": config.train_ratio,
        "val_count": config.val_count,
        "total_files": len(pt_files),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "train_file_names": [p.name for p in train_files],
        "val_file_names": [p.name for p in val_files],
    }
    save_json(report, report_path)

    print(f"split_report : {report_path}")
    print("====================================================")
    print("[DONE] Stage2 PT dataset split finished.")
    print("====================================================")


if __name__ == "__main__":
    main()