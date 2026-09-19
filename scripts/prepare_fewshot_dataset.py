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
from typing import Any


# ============================================================
# Make project root importable
# 把项目根目录加入导入路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Basic helpers
# 基础工具函数
# ============================================================
def str2bool(x: str | bool) -> bool:
    """
    EN:
    Convert common string values to bool.

    ZH:
    将常见字符串形式转换为 bool。
    """
    if isinstance(x, bool):
        return x

    s = str(x).strip().lower()

    if s in {"1", "true", "yes", "y", "on"}:
        return True

    if s in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {x}")


def bool_to_cli_str(x: bool) -> str:
    """
    EN:
    Convert bool to the string style expected by current data-factory CLIs.

    ZH:
    将 bool 转换为当前数据工厂 CLI 使用的字符串形式。
    """
    return "true" if bool(x) else "false"


def validate_stage1_ownership_flags(args: argparse.Namespace) -> None:
    """Reject legacy Stage1-cache ownership requests at the Prepare boundary.

    REL-HARDEN-2 keeps the old argument names parseable so existing launchers
    that explicitly pass ``false`` remain compatible.  ``true`` is rejected
    before any build starts because Stage1 validation/pruning is owned only by
    ``build_stage1_fewshot_dataset_hardened.py``.
    """

    enable = bool(getattr(args, "enable_stage1_cache_contract", False))
    prune = bool(getattr(args, "prune_stage1_orphans", False))
    if not enable and not prune:
        return

    requested: list[str] = []
    if enable:
        requested.append("enable_stage1_cache_contract=true")
    if prune:
        requested.append("prune_stage1_orphans=true")
    raise ValueError(
        "Prepare does not own Stage1 cache validation or orphan prune. "
        f"Rejected legacy option(s): {', '.join(requested)}. "
        "Run scripts/build_stage1_fewshot_dataset_hardened.py for Stage1 "
        "cache validation/prune semantics."
    )


def stage1_ownership_report() -> dict[str, Any]:
    """Describe the Stage1-cache ownership boundary in Prepare reports."""

    return {
        "status": "not_owned_by_prepare",
        "owner": "scripts/build_stage1_fewshot_dataset_hardened.py",
        "validation_run": False,
        "prune_run": False,
    }


def parse_args() -> argparse.Namespace:
    """
    EN:
    Parse CLI arguments.

    ZH:
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Few-shot Data Factory v1 entrypoint. "
            "It wraps `python -m src.data_factory.cli.build_dataset` "
            "to build initial clips / ASR labels / manifests from raw audio."
        )
    )

    # --------------------------------------------------------
    # Required input / output
    # 必需输入输出
    # --------------------------------------------------------
    parser.add_argument(
        "--raw_input_dir",
        type=str,
        required=True,
        help=(
            "Raw user audio directory or a single raw audio file. "
            "This will be passed to src.data_factory.cli.build_dataset."
        ),
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help=(
            "Few-shot data factory working directory. "
            "Outputs will be written under this directory."
        ),
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        required=True,
        help=(
            "User speaker name / character name. "
            "Few-shot v1 assumes mostly single-speaker data."
        ),
    )

    # --------------------------------------------------------
    # Language / ASR
    # 语言 / ASR
    # --------------------------------------------------------
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Dataset language. Default: zh.",
    )
    parser.add_argument(
        "--asr_backend",
        type=str,
        default="auto",
        help="ASR backend passed to build_dataset.py. Default: auto.",
    )
    parser.add_argument(
        "--asr_model_size",
        type=str,
        default="large-v3",
        help="Faster-Whisper model size when faster_whisper is used.",
    )
    parser.add_argument(
        "--asr_precision",
        type=str,
        default="float32",
        help="ASR precision passed to build_dataset.py.",
    )

    # --------------------------------------------------------
    # Optional preprocess
    # 可选预处理
    # --------------------------------------------------------
    parser.add_argument(
        "--enable_uvr",
        type=str2bool,
        default=False,
        help=(
            "Whether to enable UVR in the underlying data factory. "
            "Current build_dataset.py still treats this as TODO and may raise "
            "NotImplementedError if set true."
        ),
    )
    parser.add_argument(
        "--enable_denoise",
        type=str2bool,
        default=False,
        help=(
            "Whether to enable denoise in the underlying data factory. "
            "Current build_dataset.py still treats this as TODO and may raise "
            "NotImplementedError if set true."
        ),
    )

    # --------------------------------------------------------
    # Slice parameters
    # 切分参数
    # --------------------------------------------------------
    parser.add_argument(
        "--threshold",
        type=int,
        default=-34,
        help="Silence slicing threshold.",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=4000,
        help="Minimum slice length in ms-like slicer setting, aligned with old data factory.",
    )
    parser.add_argument(
        "--min_interval",
        type=int,
        default=300,
        help="Minimum silence interval.",
    )
    parser.add_argument(
        "--hop_size",
        type=int,
        default=10,
        help="Slicer hop size.",
    )
    parser.add_argument(
        "--max_sil_kept",
        type=int,
        default=500,
        help="Maximum silence kept during slicing.",
    )
    parser.add_argument(
        "--normalize_max",
        type=float,
        default=0.9,
        help="Slice normalization target max.",
    )
    parser.add_argument(
        "--alpha_mix",
        type=float,
        default=0.25,
        help="Slice normalization mix ratio.",
    )

    # --------------------------------------------------------
    # Export switches
    # 导出开关
    # --------------------------------------------------------
    parser.add_argument(
        "--export_list",
        type=str2bool,
        default=True,
        help="Whether to export dataset.list. Required for proofread.",
    )
    parser.add_argument(
        "--export_stage2_manifest",
        type=str2bool,
        default=True,
        help=(
            "Whether to export initial stage2_manifest.jsonl. "
            "Few-shot final training should later use stage2_manifest.fewshot.jsonl."
        ),
    )

    # --------------------------------------------------------
    # Work dir behavior
    # 工作目录行为
    # --------------------------------------------------------
    parser.add_argument(
        "--overwrite_work_dir",
        type=str2bool,
        default=False,
        help=(
            "If true and work_dir exists, remove it before running. "
            "Use carefully."
        ),
    )
    parser.add_argument(
        "--dry_run",
        type=str2bool,
        default=False,
        help="Print commands and planned paths without running build_dataset.",
    )

    # --------------------------------------------------------
    # Legacy RH-1B flags retained only for CLI compatibility
    # --------------------------------------------------------
    parser.add_argument(
        "--enable_stage1_cache_contract",
        type=str2bool,
        default=False,
        help=(
            "Deprecated compatibility flag. Prepare does not own Stage1 cache "
            "validation; true is rejected. Use the hardened Stage1 builder."
        ),
    )
    parser.add_argument(
        "--prune_stage1_orphans",
        type=str2bool,
        default=False,
        help=(
            "Deprecated compatibility flag. Prepare does not prune Stage1 "
            "orphans; true is rejected. Use the hardened Stage1 builder."
        ),
    )

    # --------------------------------------------------------
    # Report / next steps
    # 报告 / 下一步命令
    # --------------------------------------------------------
    parser.add_argument(
        "--report_path",
        type=str,
        default=None,
        help=(
            "Output report path. If omitted, save to "
            "<work_dir>/fewshot_prepare_report.json."
        ),
    )
    parser.add_argument(
        "--print_next_commands",
        type=str2bool,
        default=True,
        help="Print proofread / rebuild / export-fewshot next-step commands.",
    )

    # --------------------------------------------------------
    # Optional proofread launcher
    # 可选自动启动校对
    # --------------------------------------------------------
    parser.add_argument(
        "--launch_proofread_after_prepare",
        type=str2bool,
        default=False,
        help=(
            "If true, automatically launch src.data_factory.cli.launch_proofread "
            "after build_dataset succeeds."
        ),
    )
    parser.add_argument(
        "--webui_port_subfix",
        type=int,
        default=9871,
        help="Port for subfix proofread WebUI if launched.",
    )
    parser.add_argument(
        "--is_share",
        type=str,
        default="False",
        help="Gradio share flag passed to launch_proofread.",
    )
    parser.add_argument(
        "--g_batch",
        type=int,
        default=10,
        help="Batch size parameter passed to subfix WebUI.",
    )

    return parser.parse_args()


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    """
    EN:
    Save JSON object.

    ZH:
    保存 JSON 对象。
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def quote_win_path(path: str | Path) -> str:
    """
    EN:
    Quote a path for printed Windows command.

    ZH:
    为打印 Windows 命令包装路径。
    """
    return f'"{Path(path).resolve()}"'


def build_dataset_command(args: argparse.Namespace) -> list[str]:
    """
    EN:
    Build the command that invokes the old data factory.

    ZH:
    构造调用旧数据工厂的命令。
    """
    cmd = [
        sys.executable,
        "-m",
        "src.data_factory.cli.build_dataset",
        "--raw_input_dir",
        str(Path(args.raw_input_dir).resolve()),
        "--work_dir",
        str(Path(args.work_dir).resolve()),
        "--speaker_name",
        str(args.speaker_name),
        "--language",
        str(args.language),
        "--enable_uvr",
        bool_to_cli_str(bool(args.enable_uvr)),
        "--enable_denoise",
        bool_to_cli_str(bool(args.enable_denoise)),
        "--threshold",
        str(int(args.threshold)),
        "--min_length",
        str(int(args.min_length)),
        "--min_interval",
        str(int(args.min_interval)),
        "--hop_size",
        str(int(args.hop_size)),
        "--max_sil_kept",
        str(int(args.max_sil_kept)),
        "--normalize_max",
        str(float(args.normalize_max)),
        "--alpha_mix",
        str(float(args.alpha_mix)),
        "--asr_backend",
        str(args.asr_backend),
        "--asr_model_size",
        str(args.asr_model_size),
        "--asr_precision",
        str(args.asr_precision),
        "--export_list",
        bool_to_cli_str(bool(args.export_list)),
        "--export_stage2_manifest",
        bool_to_cli_str(bool(args.export_stage2_manifest)),
    ]

    return cmd


def build_proofread_command(
    *,
    dataset_list_path: str | Path,
    webui_port_subfix: int,
    is_share: str,
    g_batch: int,
) -> list[str]:
    """
    EN:
    Build proofread launcher command.

    ZH:
    构造人工校对启动命令。
    """
    return [
        sys.executable,
        "-m",
        "src.data_factory.cli.launch_proofread",
        "--dataset_list",
        str(Path(dataset_list_path).resolve()),
        "--webui_port_subfix",
        str(int(webui_port_subfix)),
        "--is_share",
        str(is_share),
        "--g_batch",
        str(int(g_batch)),
    ]


def command_to_printable(cmd: list[str]) -> str:
    """
    EN:
    Convert command list to a readable Windows-style command string.

    ZH:
    将命令列表转换为更适合 Windows 复制的字符串。
    """
    parts: list[str] = []

    for x in cmd:
        s = str(x)
        if " " in s or "\\" in s or ":" in s:
            parts.append(f'"{s}"')
        else:
            parts.append(s)

    return " ".join(parts)


def expected_output_paths(work_dir: str | Path) -> dict[str, str]:
    """
    EN:
    Return key output paths generated by build_dataset.py.

    ZH:
    返回 build_dataset.py 预期生成的关键路径。
    """
    root = Path(work_dir).resolve()
    export_dir = root / "06_export"

    return {
        "work_dir": str(root),
        "raw_audio_index": str(root / "00_raw_index" / "raw_audio_index.jsonl"),
        "manifest_jsonl": str(export_dir / "manifest.jsonl"),
        "dataset_list": str(export_dir / "dataset.list"),
        "stage2_manifest_jsonl": str(export_dir / "stage2_manifest.jsonl"),
        "export_dir": str(export_dir),
        "export_clips_dir": str(export_dir / "clips"),
        "manifest_before_proofread": str(export_dir / "manifest.before_proofread.jsonl"),
        "dataset_before_proofread": str(export_dir / "dataset.before_proofread.list"),
        "stage2_manifest_before_proofread": str(export_dir / "stage2_manifest.before_proofread.jsonl"),
        "manifest_corrected": str(export_dir / "manifest.corrected.jsonl"),
        "stage2_manifest_corrected": str(export_dir / "stage2_manifest.corrected.jsonl"),
        "correction_report": str(export_dir / "correction_report.json"),
        "stage2_manifest_fewshot": str(export_dir / "stage2_manifest.fewshot.jsonl"),
        "prompt_selection_report": str(export_dir / "prompt_selection_report.json"),
        "fewshot_manifest_conversion_report": str(export_dir / "fewshot_manifest_conversion_report.json"),
    }


def check_expected_outputs(
    paths: dict[str, str],
    *,
    export_list: bool,
    export_stage2_manifest: bool,
) -> dict[str, Any]:
    """
    EN:
    Check key expected files after build_dataset.

    ZH:
    检查 build_dataset 后的关键输出文件。
    """
    required_keys = [
        "raw_audio_index",
        "manifest_jsonl",
    ]

    if bool(export_list):
        required_keys.append("dataset_list")

    if bool(export_stage2_manifest):
        required_keys.append("stage2_manifest_jsonl")

    checks: dict[str, Any] = {}

    missing: list[str] = []

    for key in required_keys:
        p = Path(paths[key]).resolve()
        exists = p.exists()
        is_file = p.is_file()

        checks[key] = {
            "path": str(p),
            "exists": bool(exists),
            "is_file": bool(is_file),
        }

        if not exists or not is_file:
            missing.append(key)

    return {
        "required_keys": required_keys,
        "checks": checks,
        "missing_required_outputs": missing,
        "ok": len(missing) == 0,
    }


def build_next_step_commands(
    *,
    paths: dict[str, str],
    webui_port_subfix: int,
    is_share: str,
    g_batch: int,
) -> dict[str, str]:
    """
    EN:
    Build suggested next-step commands.

    ZH:
    构造建议的下一步命令。
    """
    proofread_cmd = [
        "python",
        "-m",
        "src.data_factory.cli.launch_proofread",
        "--dataset_list",
        quote_win_path(paths["dataset_list"]),
        "--webui_port_subfix",
        str(int(webui_port_subfix)),
        "--is_share",
        str(is_share),
        "--g_batch",
        str(int(g_batch)),
    ]

    rebuild_cmd = [
        "python",
        "-m",
        "src.data_factory.importers.rebuild_corrected_manifest",
        "--original_manifest",
        quote_win_path(paths["manifest_before_proofread"]),
        "--original_list",
        quote_win_path(paths["dataset_before_proofread"]),
        "--corrected_list",
        quote_win_path(paths["dataset_list"]),
        "--output_manifest",
        quote_win_path(paths["manifest_corrected"]),
        "--output_stage2_manifest",
        quote_win_path(paths["stage2_manifest_corrected"]),
        "--output_report",
        quote_win_path(paths["correction_report"]),
    ]

    export_fewshot_cmd = [
        "python",
        "scripts\\export_fewshot_stage2_manifest.py",
        "--manifest_jsonl",
        quote_win_path(paths["manifest_corrected"]),
        "--output_stage2_manifest",
        quote_win_path(paths["stage2_manifest_fewshot"]),
        "--prompt_mode",
        "speaker_pool",
        "--prompt_selection_report_path",
        quote_win_path(paths["prompt_selection_report"]),
        "--conversion_report_path",
        quote_win_path(paths["fewshot_manifest_conversion_report"]),
    ]

    health_check_cmd = [
        "python",
        "scripts\\check_fewshot_dataset_health.py",
        "--stage2_manifest_jsonl",
        quote_win_path(paths["stage2_manifest_fewshot"]),
        "--report_path",
        quote_win_path(Path(paths["export_dir"]) / "fewshot_health_stage2_manifest_report.json"),
    ]

    return {
        "proofread": " ".join(proofread_cmd),
        "rebuild_corrected_manifest": " ".join(rebuild_cmd),
        "export_fewshot_stage2_manifest": " ".join(export_fewshot_cmd),
        "health_check_fewshot_stage2_manifest": " ".join(health_check_cmd),
    }


def maybe_overwrite_work_dir(work_dir: str | Path, overwrite: bool) -> None:
    """
    EN:
    Optionally remove existing work_dir.

    ZH:
    可选删除已有 work_dir。
    """
    p = Path(work_dir).resolve()

    if not p.exists():
        return

    if not bool(overwrite):
        return

    if not p.is_dir():
        raise ValueError(f"work_dir exists but is not a directory: {p}")

    print(f"[WARN] overwrite_work_dir=true, removing: {p}")
    shutil.rmtree(p)


def print_next_commands(commands: dict[str, str]) -> None:
    """
    EN:
    Print next step commands.

    ZH:
    打印后续命令。
    """
    print("====================================================")
    print("Next suggested commands")
    print("====================================================")
    print("[1] Proofread dataset.list")
    print(commands["proofread"])
    print("----------------------------------------------------")
    print("[2] Rebuild corrected manifest after proofreading")
    print(commands["rebuild_corrected_manifest"])
    print("----------------------------------------------------")
    print("[3] Export few-shot Stage2 manifest with speaker_pool prompt")
    print(commands["export_fewshot_stage2_manifest"])
    print("----------------------------------------------------")
    print("[4] Health-check few-shot Stage2 manifest")
    print(commands["health_check_fewshot_stage2_manifest"])
    print("====================================================")


def print_summary(report: dict[str, Any]) -> None:
    """
    EN:
    Print preparation summary.

    ZH:
    打印准备结果摘要。
    """
    print("====================================================")
    print("Few-shot Data Factory v1 prepare finished")
    print("====================================================")
    print(f"status        : {report['status']}")
    print(f"raw_input_dir : {report['inputs']['raw_input_dir']}")
    print(f"work_dir      : {report['inputs']['work_dir']}")
    print(f"speaker_name  : {report['inputs']['speaker_name']}")
    print(f"language      : {report['inputs']['language']}")
    print(f"report_path   : {report['report_path']}")
    print("----------------------------------------------------")
    print("Key outputs:")
    for key in [
        "raw_audio_index",
        "manifest_jsonl",
        "dataset_list",
        "stage2_manifest_jsonl",
    ]:
        check = report["output_check"]["checks"].get(key)
        if check is None:
            continue
        mark = "OK" if check["exists"] and check["is_file"] else "MISSING"
        print(f"  {key:24s}: {mark} | {check['path']}")
    print("====================================================")


# ============================================================
# Main
# 主入口
# ============================================================
def main() -> None:
    args = parse_args()
    validate_stage1_ownership_flags(args)

    raw_input_dir = str(Path(args.raw_input_dir).resolve())
    work_dir = str(Path(args.work_dir).resolve())

    report_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else Path(work_dir).resolve() / "fewshot_prepare_report.json"
    )

    paths = expected_output_paths(work_dir)

    build_cmd = build_dataset_command(args)
    proofread_cmd = build_proofread_command(
        dataset_list_path=paths["dataset_list"],
        webui_port_subfix=int(args.webui_port_subfix),
        is_share=str(args.is_share),
        g_batch=int(args.g_batch),
    )

    next_step_commands = build_next_step_commands(
        paths=paths,
        webui_port_subfix=int(args.webui_port_subfix),
        is_share=str(args.is_share),
        g_batch=int(args.g_batch),
    )

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "PENDING",
        "project_root": str(PROJECT_ROOT),
        "report_path": str(report_path),
        "inputs": {
            "raw_input_dir": raw_input_dir,
            "work_dir": work_dir,
            "speaker_name": str(args.speaker_name),
            "language": str(args.language),
        },
        "config": {
            "asr_backend": str(args.asr_backend),
            "asr_model_size": str(args.asr_model_size),
            "asr_precision": str(args.asr_precision),
            "enable_uvr": bool(args.enable_uvr),
            "enable_denoise": bool(args.enable_denoise),
            "threshold": int(args.threshold),
            "min_length": int(args.min_length),
            "min_interval": int(args.min_interval),
            "hop_size": int(args.hop_size),
            "max_sil_kept": int(args.max_sil_kept),
            "normalize_max": float(args.normalize_max),
            "alpha_mix": float(args.alpha_mix),
            "export_list": bool(args.export_list),
            "export_stage2_manifest": bool(args.export_stage2_manifest),
            "overwrite_work_dir": bool(args.overwrite_work_dir),
            "dry_run": bool(args.dry_run),
            "launch_proofread_after_prepare": bool(args.launch_proofread_after_prepare),
            "enable_stage1_cache_contract": bool(args.enable_stage1_cache_contract),
            "prune_stage1_orphans": bool(args.prune_stage1_orphans),
        },
        "paths": paths,
        "commands": {
            "build_dataset": command_to_printable(build_cmd),
            "proofread": command_to_printable(proofread_cmd),
            "next_steps": next_step_commands,
        },
        "output_check": None,
        "stage1_cache_contract": stage1_ownership_report(),
        "return_codes": {},
    }

    print("====================================================")
    print("Few-shot Data Factory v1 prepare")
    print("====================================================")
    print(f"raw_input_dir : {raw_input_dir}")
    print(f"work_dir      : {work_dir}")
    print(f"speaker_name  : {args.speaker_name}")
    print(f"language      : {args.language}")
    print("----------------------------------------------------")
    print("Underlying command:")
    print(command_to_printable(build_cmd))
    print("====================================================")

    if bool(args.dry_run):
        report["status"] = "DRY_RUN"
        report["output_check"] = {
            "ok": None,
            "message": "dry_run=true, build_dataset was not executed.",
        }

        save_json(report, report_path)
        print_summary(report)

        if bool(args.print_next_commands):
            print_next_commands(next_step_commands)

        return

    maybe_overwrite_work_dir(
        work_dir=work_dir,
        overwrite=bool(args.overwrite_work_dir),
    )

    # ------------------------------------------------------------
    # Run old data factory
    # 调用旧数据工厂
    # ------------------------------------------------------------
    try:
        completed = subprocess.run(
            build_cmd,
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        report["return_codes"]["build_dataset"] = int(completed.returncode)
    except subprocess.CalledProcessError as e:
        report["status"] = "FAILED_BUILD_DATASET"
        report["return_codes"]["build_dataset"] = int(e.returncode)
        report["error"] = {
            "type": "CalledProcessError",
            "message": str(e),
        }
        save_json(report, report_path)
        raise

    # ------------------------------------------------------------
    # Check outputs
    # 检查输出文件
    # ------------------------------------------------------------
    output_check = check_expected_outputs(
        paths,
        export_list=bool(args.export_list),
        export_stage2_manifest=bool(args.export_stage2_manifest),
    )
    report["output_check"] = output_check

    if not output_check["ok"]:
        report["status"] = "FAILED_OUTPUT_CHECK"
        save_json(report, report_path)
        print_summary(report)
        raise FileNotFoundError(
            "Few-shot prepare output check failed. Missing: "
            f"{output_check['missing_required_outputs']}"
        )

    # ------------------------------------------------------------
    # Optional proofread launcher
    # 可选启动人工校对
    # ------------------------------------------------------------
    if bool(args.launch_proofread_after_prepare):
        print("====================================================")
        print("Launching proofread WebUI")
        print("====================================================")
        print(command_to_printable(proofread_cmd))
        print("====================================================")

        try:
            completed = subprocess.run(
                proofread_cmd,
                cwd=str(PROJECT_ROOT),
                check=True,
            )
            report["return_codes"]["proofread"] = int(completed.returncode)
        except subprocess.CalledProcessError as e:
            report["status"] = "FAILED_LAUNCH_PROOFREAD"
            report["return_codes"]["proofread"] = int(e.returncode)
            report["error"] = {
                "type": "CalledProcessError",
                "message": str(e),
            }
            save_json(report, report_path)
            raise

    report["status"] = "OK"
    save_json(report, report_path)

    print_summary(report)

    if bool(args.print_next_commands):
        print_next_commands(next_step_commands)


if __name__ == "__main__":
    main()
