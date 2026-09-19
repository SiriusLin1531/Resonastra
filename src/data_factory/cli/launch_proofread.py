from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# Make project root importable
# 把项目根目录加入导入路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """
    EN:
    Parse arguments for the proofreading launcher.

    ZH:
    解析人工校对启动器的参数。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_list", type=str, required=True)
    parser.add_argument("--webui_port_subfix", type=int, default=9871)
    parser.add_argument("--is_share", type=str, default="False")
    parser.add_argument("--g_batch", type=int, default=10)

    # Backup behavior
    parser.add_argument("--backup_suffix", type=str, default="before_proofread")
    parser.add_argument("--overwrite_backup", type=str, default="false")

    return parser.parse_args()


def str2bool(x: str) -> bool:
    """
    EN:
    Convert common string forms to bool.

    ZH:
    将常见字符串形式转成布尔值。
    """
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_exists(path: str | Path, name: str) -> Path:
    """
    EN:
    Ensure file exists.

    ZH:
    确保文件存在。
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")
    if not p.is_file():
        raise ValueError(f"{name} is not a file: {p}")
    return p


def build_backup_path(file_path: Path, backup_suffix: str) -> Path:
    """
    EN:
    Convert:
        dataset.list -> dataset.before_proofread.list
        manifest.jsonl -> manifest.before_proofread.jsonl

    ZH:
    将：
        dataset.list -> dataset.before_proofread.list
        manifest.jsonl -> manifest.before_proofread.jsonl
    """
    suffixes = "".join(file_path.suffixes)
    if not suffixes:
        return file_path.with_name(f"{file_path.name}.{backup_suffix}")
    stem = file_path.name[: -len(suffixes)]
    return file_path.with_name(f"{stem}.{backup_suffix}{suffixes}")


def backup_file_if_needed(
    src_path: Path,
    backup_suffix: str,
    overwrite: bool,
) -> Path:
    """
    EN:
    Create backup file if needed.

    ZH:
    如有需要，创建备份文件。
    """
    backup_path = build_backup_path(src_path, backup_suffix)

    if backup_path.exists() and not overwrite:
        return backup_path

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, backup_path)
    return backup_path


def save_launch_report(report: dict, path: Path) -> None:
    """
    EN:
    Save launch report to JSON.

    ZH:
    将本次启动与备份信息保存为 JSON。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    dataset_list = ensure_exists(args.dataset_list, "dataset_list")
    export_dir = dataset_list.parent

    overwrite_backup = str2bool(args.overwrite_backup)

    # --------------------------------------------------------
    # Find sibling files that should be backed up automatically
    # 找到需要自动备份的同级文件
    # --------------------------------------------------------
    manifest_path = export_dir / "manifest.jsonl"
    stage2_manifest_path = export_dir / "stage2_manifest.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected manifest.jsonl not found beside dataset.list: {manifest_path}")

    if not stage2_manifest_path.exists():
        raise FileNotFoundError(
            f"Expected stage2_manifest.jsonl not found beside dataset.list: {stage2_manifest_path}"
        )

    # --------------------------------------------------------
    # Create backups
    # 创建备份
    # --------------------------------------------------------
    backup_dataset_list = backup_file_if_needed(
        dataset_list,
        backup_suffix=args.backup_suffix,
        overwrite=overwrite_backup,
    )
    backup_manifest = backup_file_if_needed(
        manifest_path,
        backup_suffix=args.backup_suffix,
        overwrite=overwrite_backup,
    )
    backup_stage2_manifest = backup_file_if_needed(
        stage2_manifest_path,
        backup_suffix=args.backup_suffix,
        overwrite=overwrite_backup,
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_list": str(dataset_list),
        "manifest_jsonl": str(manifest_path),
        "stage2_manifest_jsonl": str(stage2_manifest_path),
        "backup_dataset_list": str(backup_dataset_list),
        "backup_manifest_jsonl": str(backup_manifest),
        "backup_stage2_manifest_jsonl": str(backup_stage2_manifest),
        "webui_port_subfix": args.webui_port_subfix,
        "is_share": args.is_share,
        "g_batch": args.g_batch,
    }

    report_path = export_dir / "proofread_launch_report.json"
    save_launch_report(report, report_path)

    print("====================================================")
    print("Proofread launcher started")
    print(f"dataset_list            : {dataset_list}")
    print(f"backup_dataset_list     : {backup_dataset_list}")
    print(f"backup_manifest_jsonl   : {backup_manifest}")
    print(f"backup_stage2_manifest  : {backup_stage2_manifest}")
    print(f"launch_report           : {report_path}")
    print("====================================================")

    # --------------------------------------------------------
    # IMPORTANT:
    # use module mode to avoid `ModuleNotFoundError: No module named 'tools'`
    # 重要：
    # 用模块方式启动，避免 `No module named 'tools'`
    # --------------------------------------------------------
    cmd = [
        sys.executable,
        "-m",
        "tools.subfix_webui",
        "--load_list",
        str(dataset_list),
        "--webui_port_subfix",
        str(args.webui_port_subfix),
        "--is_share",
        str(args.is_share),
        "--g_batch",
        str(args.g_batch),
    ]

    print("Launching subfix_webui with command:")
    print(" ".join(cmd))
    print("====================================================")

    subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    print("====================================================")
    print("[DONE] Proofread launcher finished.")
    print("====================================================")


if __name__ == "__main__":
    main()