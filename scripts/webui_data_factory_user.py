from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ============================================================
# Project bootstrap
# 项目路径引导
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Third-party imports
# 第三方导入
# ============================================================
import gradio as gr

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.voicelab_user.data_factory import (
    DATA_FACTORY_STATUS_FAILED,
    DATA_FACTORY_STATUS_SUCCEEDED,
    DataFactoryRequest,
    create_data_factory_adapter,
    create_data_factory_user_pipeline_service,
    user_pipeline_result_to_dict,
)
from src.voicelab_user.data_factory.resume_pipeline import (
    run_resumable_postprocess_pipeline,
)


# ============================================================
# Constants
# 常量
# ============================================================
AUDIO_FILE_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
}
AUDIO_FILE_TYPES = sorted(AUDIO_FILE_SUFFIXES)

DATA_FACTORY_PROCESS_SCRIPT_NAMES = {
    "prepare_fewshot_dataset.py",
    "check_fewshot_dataset_health.py",
    "export_fewshot_stage2_manifest.py",
    "rebuild_corrected_manifest.py",
    "preprocess_stage2_fm_dataset.py",
    "export_stage2_continuous_semantic_cache.py",
    "export_fewshot_style_cache.py",
    "split_stage2_pt_dataset.py",
    "filter_v662_bad_style_samples.py",
    "launch_proofread.py",
}

WEBUI_SCRIPT_NAMES = {
    "webui_data_factory_user.py",
    "webui_data_factory.py",
}

RUNTIME_STATE_FILENAME = "data_factory_runtime_state.json"
STOP_REPORT_FILENAME = "data_factory_stop_report.json"


# ============================================================
# Generic helpers
# 通用工具
# ============================================================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _blank_to_none(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _as_int(value: Any, default: int) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _as_timeout(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None

    v = float(value)
    if v <= 0:
        return None
    return v


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path).resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


# ============================================================
# Upload helpers
# 上传辅助函数
# ============================================================
def _uploaded_item_to_path(item: Any) -> Optional[Path]:
    """Convert one Gradio uploaded item to a Path."""

    if item is None:
        return None

    if isinstance(item, (str, Path)):
        return Path(item)

    name = getattr(item, "name", None)
    if name:
        return Path(name)

    return None


def _iter_uploaded_paths(uploaded_audio_files: Any) -> list[Path]:
    """Normalize Gradio uploaded files into a list of existing file paths."""

    if uploaded_audio_files is None:
        return []

    if isinstance(uploaded_audio_files, (list, tuple)):
        items = uploaded_audio_files
    else:
        items = [uploaded_audio_files]

    paths: list[Path] = []
    for item in items:
        path = _uploaded_item_to_path(item)
        if path is None:
            continue

        path = path.expanduser().resolve(strict=False)
        if path.exists() and path.is_file():
            paths.append(path)

    return paths


def _is_audio_file(path: Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_FILE_SUFFIXES


def _unique_destination_path(dst_dir: Path, filename: str) -> Path:
    candidate = dst_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix

    for index in range(1, 10000):
        next_candidate = dst_dir / f"{stem}_{index:04d}{suffix}"
        if not next_candidate.exists():
            return next_candidate

    raise RuntimeError(f"Too many duplicate uploaded filenames for: {filename}")


def _resolve_work_dir_for_upload(
    *,
    work_dir: str,
    speaker_name: str,
) -> Path:
    adapter = create_data_factory_adapter()

    resolved_work_dir_text = _blank_to_none(work_dir)
    if resolved_work_dir_text is not None:
        return adapter.resolve_path(resolved_work_dir_text)

    return adapter.suggest_work_dir(str(speaker_name).strip())


def _copy_uploaded_audio_to_work_dir(
    *,
    uploaded_audio_files: Any,
    work_dir: str,
    speaker_name: str,
    overwrite_uploaded_audio_cache: bool,
    upload_mode_label: str,
) -> tuple[str, dict[str, Any]]:
    """Copy uploaded audio files into work_dir/00_uploaded_raw_audio."""

    uploaded_paths = _iter_uploaded_paths(uploaded_audio_files)
    audio_paths = [path for path in uploaded_paths if _is_audio_file(path)]

    if not audio_paths:
        raise ValueError(
            "未检测到已上传的音频文件。请上传 wav/mp3/flac/ogg/m4a/aac/wma/opus 文件。"
        )

    resolved_work_dir = _resolve_work_dir_for_upload(
        work_dir=work_dir,
        speaker_name=speaker_name,
    )
    upload_root = resolved_work_dir / "00_uploaded_raw_audio"

    if upload_root.exists() and overwrite_uploaded_audio_cache:
        shutil.rmtree(upload_root)

    upload_root.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    skipped_files: list[str] = []

    for src in audio_paths:
        try:
            dst = _unique_destination_path(upload_root, src.name)
            shutil.copy2(str(src), str(dst))
            copied_files.append(str(dst))
        except Exception as exc:
            skipped_files.append(f"{src} :: {type(exc).__name__}: {exc}")

    if not copied_files:
        raise RuntimeError(
            "上传音频复制失败，没有任何音频文件被复制到工作目录。"
        )

    info = {
        "upload_mode": str(upload_mode_label),
        "upload_root": str(upload_root),
        "num_uploaded_items": len(uploaded_paths),
        "num_audio_files": len(audio_paths),
        "num_copied_files": len(copied_files),
        "num_skipped_files": len(skipped_files),
        "copied_files_preview": copied_files[:20],
        "skipped_files_preview": skipped_files[:20],
        "overwrite_uploaded_audio_cache": bool(overwrite_uploaded_audio_cache),
    }

    return str(upload_root), info


def _resolve_raw_input_dir_for_prepare(
    *,
    audio_source_mode: str,
    raw_input_dir: str,
    uploaded_audio_files: Any,
    uploaded_audio_folder: Any,
    work_dir: str,
    speaker_name: str,
    overwrite_uploaded_audio_cache: bool,
) -> tuple[str, Optional[dict[str, Any]]]:
    mode = str(audio_source_mode or "").strip()

    if mode == "上传音频文件":
        upload_root, upload_info = _copy_uploaded_audio_to_work_dir(
            uploaded_audio_files=uploaded_audio_files,
            work_dir=work_dir,
            speaker_name=speaker_name,
            overwrite_uploaded_audio_cache=overwrite_uploaded_audio_cache,
            upload_mode_label=mode,
        )
        return upload_root, upload_info

    if mode == "上传音频文件夹":
        upload_root, upload_info = _copy_uploaded_audio_to_work_dir(
            uploaded_audio_files=uploaded_audio_folder,
            work_dir=work_dir,
            speaker_name=speaker_name,
            overwrite_uploaded_audio_cache=overwrite_uploaded_audio_cache,
            upload_mode_label=mode,
        )
        return upload_root, upload_info

    local_dir = _blank_to_none(raw_input_dir)
    if local_dir is None:
        raise ValueError("请选择本地音频目录，或切换为上传音频文件/文件夹模式。")

    return local_dir, None


def _format_upload_info(upload_info: Optional[dict[str, Any]]) -> str:
    if not upload_info:
        return ""

    return "\n".join(
        [
            "",
            "### 上传音频缓存",
            f"- 上传模式：`{upload_info.get('upload_mode')}`",
            f"- 上传缓存目录：`{upload_info.get('upload_root')}`",
            f"- 检测到音频文件数：`{upload_info.get('num_audio_files')}`",
            f"- 已复制文件数：`{upload_info.get('num_copied_files')}`",
            f"- 跳过文件数：`{upload_info.get('num_skipped_files')}`",
        ]
    )


def audio_source_mode_change(audio_source_mode: str):
    mode = str(audio_source_mode or "").strip()
    use_local = mode == "本地目录路径"
    use_upload_files = mode == "上传音频文件"
    use_upload_folder = mode == "上传音频文件夹"
    use_upload = use_upload_files or use_upload_folder

    return (
        gr.update(visible=use_local),
        gr.update(visible=use_upload_files),
        gr.update(visible=use_upload_folder),
        gr.update(visible=use_upload),
    )


# ============================================================
# Work-dir scanning helpers
# 工作目录扫描工具
# ============================================================
def _resolve_active_work_dir(
    *,
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
) -> Path:
    """Resolve active work_dir from current output, editable textbox or speaker default."""

    adapter = create_data_factory_adapter()

    output_text = _blank_to_none(work_dir_output)
    if output_text is not None:
        return adapter.resolve_path(output_text)

    work_dir_text = _blank_to_none(work_dir)
    if work_dir_text is not None:
        return adapter.resolve_path(work_dir_text)

    return adapter.suggest_work_dir(str(speaker_name).strip())


def _path_exists(path: Optional[Path]) -> bool:
    return path is not None and Path(path).exists()


def _count_jsonl_rows(path: Optional[Path]) -> Optional[int]:
    if path is None or not Path(path).exists() or not Path(path).is_file():
        return None

    count = 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_pt_files(path: Optional[Path]) -> Optional[int]:
    if path is None or not Path(path).exists() or not Path(path).is_dir():
        return None

    return len(list(Path(path).glob("*.pt")))


def _format_exists(value: bool) -> str:
    return "已存在" if value else "缺失"


def _status_icon(value: bool) -> str:
    return "✅" if value else "⬜"


def _build_scan_payload(active_work_dir: Path) -> tuple[dict[str, Any], str, str, str]:
    """Build work-dir scan payload and markdown blocks."""

    adapter = create_data_factory_adapter()
    artifacts = adapter.expected_artifacts(active_work_dir)

    manifest_rows = _count_jsonl_rows(artifacts.manifest_jsonl_path)
    corrected_rows = _count_jsonl_rows(artifacts.manifest_corrected_path)
    fewshot_rows = _count_jsonl_rows(artifacts.stage2_manifest_fewshot_path)

    stage2_pt_count = _count_pt_files(artifacts.stage2_pt_root)
    continuous_count = _count_pt_files(artifacts.continuous_semantic_root)
    style_count = _count_pt_files(artifacts.style_cache_root)
    train_count = _count_pt_files(artifacts.train_pt_root)
    val_count = _count_pt_files(artifacts.val_pt_root)

    prepare_done = bool(
        _path_exists(artifacts.manifest_jsonl_path)
        and _path_exists(artifacts.dataset_list_path)
    )
    corrected_done = bool(_path_exists(artifacts.manifest_corrected_path))
    fewshot_done = bool(_path_exists(artifacts.stage2_manifest_fewshot_path))
    stage2_done = bool(stage2_pt_count and stage2_pt_count > 0)
    continuous_done = bool(continuous_count and continuous_count > 0)
    style_done = bool(style_count and style_count > 0)
    split_done = bool(
        _path_exists(artifacts.split_report_path)
        and _path_exists(artifacts.train_pt_root)
        and _path_exists(artifacts.val_pt_root)
    )
    filter_done = bool(_path_exists(artifacts.filter_report_json_path))

    output_summary = {
        "work_dir": str(artifacts.work_dir),
        "manifest_jsonl": str(artifacts.manifest_jsonl_path),
        "dataset_list": str(artifacts.dataset_list_path),
        "manifest_corrected": str(artifacts.manifest_corrected_path),
        "stage2_manifest_fewshot": str(artifacts.stage2_manifest_fewshot_path),
        "stage2_pt_root": str(artifacts.stage2_pt_root),
        "continuous_semantic_root": str(artifacts.continuous_semantic_root),
        "style_cache_root": str(artifacts.style_cache_root),
        "train_pt_root": str(artifacts.train_pt_root),
        "val_pt_root": str(artifacts.val_pt_root),
        "split_report": str(artifacts.split_report_path),
        "filter_report_json": str(artifacts.filter_report_json_path),
        "filter_report_csv": str(artifacts.filter_report_csv_path),
    }

    scan_steps = [
        {
            "name": "prepare_pipeline",
            "display_name": "执行数据工厂",
            "status": "succeeded" if prepare_done else "prepared",
            "message": "检测到 manifest.jsonl 和 dataset.list。" if prepare_done else "尚未检测到完整数据工厂产物。",
        },
        {
            "name": "proofread_or_skip",
            "display_name": "人工校对 / 跳过校对",
            "status": "succeeded" if corrected_done else "prepared",
            "message": "检测到 manifest.corrected.jsonl。" if corrected_done else "尚未检测到 corrected manifest。",
        },
        {
            "name": "export_fewshot_manifest",
            "display_name": "导出 few-shot Stage2 manifest",
            "status": "succeeded" if fewshot_done else "prepared",
            "message": "检测到 stage2_manifest.fewshot.jsonl。" if fewshot_done else "尚未检测到 few-shot Stage2 manifest。",
        },
        {
            "name": "preprocess_stage2_pt_dataset",
            "display_name": "Stage2 .pt 预处理",
            "status": "succeeded" if stage2_done else "prepared",
            "message": f"检测到 {stage2_pt_count} 个 .pt 文件。" if stage2_done else "尚未检测到 Stage2 .pt 文件。",
        },
        {
            "name": "export_continuous_semantic_cache",
            "display_name": "导出 continuous semantic cache",
            "status": "succeeded" if continuous_done else "prepared",
            "message": f"检测到 {continuous_count} 个 .pt 文件。" if continuous_done else "尚未检测到 continuous cache。",
        },
        {
            "name": "export_fewshot_style_cache",
            "display_name": "导出 Style / F0 / Speaker cache",
            "status": "succeeded" if style_done else "prepared",
            "message": f"检测到 {style_count} 个 .pt 文件。" if style_done else "尚未检测到 style cache。",
        },
        {
            "name": "split_stage2_pt_dataset",
            "display_name": "切分 train / val",
            "status": "succeeded" if split_done else "prepared",
            "message": "检测到 train/val 和 split_report.json。" if split_done else "尚未检测到完整 train/val 切分产物。",
        },
        {
            "name": "filter_v662_bad_style_samples",
            "display_name": "过滤异常样本",
            "status": "succeeded" if filter_done else "prepared",
            "message": "检测到异常样本过滤报告。" if filter_done else "尚未检测到异常样本过滤报告。",
        },
    ]

    payload = {
        "status": "succeeded",
        "operation": "inspect_work_dir",
        "message": "工作目录扫描完成。",
        "work_dir": str(active_work_dir),
        "output_summary": output_summary,
        "counts": {
            "manifest_rows": manifest_rows,
            "manifest_corrected_rows": corrected_rows,
            "stage2_manifest_fewshot_rows": fewshot_rows,
            "stage2_pt_count": stage2_pt_count,
            "continuous_count": continuous_count,
            "style_count": style_count,
            "train_count": train_count,
            "val_count": val_count,
        },
        "steps": scan_steps,
        "artifacts_exist": {
            "prepare_done": prepare_done,
            "corrected_done": corrected_done,
            "fewshot_done": fewshot_done,
            "stage2_done": stage2_done,
            "continuous_done": continuous_done,
            "style_done": style_done,
            "split_done": split_done,
            "filter_done": filter_done,
        },
    }

    status_lines = [
        "## 当前状态：已载入工作目录",
        "",
        f"**工作目录：** `{active_work_dir}`",
        "",
        "### 关键产物",
        f"- 数据工厂基础产物：{_format_exists(prepare_done)}",
        f"- corrected manifest：{_format_exists(corrected_done)}",
        f"- few-shot Stage2 manifest：{_format_exists(fewshot_done)}",
        f"- Stage2 .pt：{_format_exists(stage2_done)}"
        + (f"，数量 `{stage2_pt_count}`" if stage2_pt_count is not None else ""),
        f"- continuous cache：{_format_exists(continuous_done)}"
        + (f"，数量 `{continuous_count}`" if continuous_count is not None else ""),
        f"- Style / F0 / Speaker cache：{_format_exists(style_done)}"
        + (f"，数量 `{style_count}`" if style_count is not None else ""),
        f"- train / val：{_format_exists(split_done)}"
        + (
            f"，train `{train_count}`，val `{val_count}`"
            if train_count is not None or val_count is not None
            else ""
        ),
        f"- 异常样本报告：{_format_exists(filter_done)}",
        "",
        "可以基于该工作目录继续启动人工校对、跳过校对、重建 corrected manifest 或继续生成训练数据。",
    ]

    output_md = _format_output_summary(output_summary)
    step_md = _format_step_summary(payload)
    return payload, "\n".join(status_lines), output_md, step_md


def _format_work_dir_status_panel(scan_payload: dict[str, Any]) -> str:
    """Format a compact visual status panel from inspect_work_dir payload."""

    if not scan_payload:
        return "暂无工作目录状态。"

    work_dir = scan_payload.get("work_dir") or ""
    counts = scan_payload.get("counts") or {}
    exists = scan_payload.get("artifacts_exist") or {}

    lines = [
        "### 工作目录状态面板",
        "",
        f"**工作目录：** `{work_dir}`" if work_dir else "**工作目录：** 未载入",
        "",
        "| 阶段 | 状态 | 数量/说明 |",
        "|---|---:|---|",
        f"| 执行数据工厂 | {_status_icon(bool(exists.get('prepare_done')))} | manifest 行数：`{counts.get('manifest_rows')}` |",
        f"| 人工校对 / 跳过校对 | {_status_icon(bool(exists.get('corrected_done')))} | corrected 行数：`{counts.get('manifest_corrected_rows')}` |",
        f"| few-shot Stage2 manifest | {_status_icon(bool(exists.get('fewshot_done')))} | 行数：`{counts.get('stage2_manifest_fewshot_rows')}` |",
        f"| Stage2 .pt | {_status_icon(bool(exists.get('stage2_done')))} | .pt 数：`{counts.get('stage2_pt_count')}` |",
        f"| continuous cache | {_status_icon(bool(exists.get('continuous_done')))} | .pt 数：`{counts.get('continuous_count')}` |",
        f"| Style/F0/Speaker cache | {_status_icon(bool(exists.get('style_done')))} | .pt 数：`{counts.get('style_count')}` |",
        f"| train / val | {_status_icon(bool(exists.get('split_done')))} | train：`{counts.get('train_count')}`，val：`{counts.get('val_count')}` |",
        f"| 异常样本报告 | {_status_icon(bool(exists.get('filter_done')))} | filter report |",
        "",
        "提示：如果某一步状态为空但已有部分文件，点击 **继续生成训练数据** 会尝试从缺失或不完整步骤继续。",
    ]
    return "\n".join(lines)


# ============================================================
# Stop helpers
# 停止任务工具
# ============================================================
def _path_patterns_for_process_match(active_work_dir: Path) -> list[str]:
    """Build path patterns used to identify DataFactory child processes."""

    adapter = create_data_factory_adapter()
    artifacts = adapter.expected_artifacts(active_work_dir)

    candidate_paths: list[Optional[Path]] = [
        active_work_dir,
        artifacts.manifest_jsonl_path,
        artifacts.dataset_list_path,
        artifacts.stage2_manifest_fewshot_path,
        artifacts.stage2_pt_root,
        artifacts.continuous_semantic_root,
        artifacts.style_cache_root,
        artifacts.train_pt_root,
        artifacts.val_pt_root,
        None if artifacts.train_pt_root is None else artifacts.train_pt_root.parent,
        artifacts.rejected_pt_root,
        artifacts.filter_report_json_path,
    ]

    patterns: list[str] = []
    for path in candidate_paths:
        if path is None:
            continue
        resolved = Path(path).resolve(strict=False)
        variants = {
            str(resolved).lower(),
            resolved.as_posix().lower(),
        }
        for value in variants:
            if value and value not in patterns:
                patterns.append(value)

    return patterns


def _query_python_processes() -> list[dict[str, Any]]:
    """Query running python/pythonw processes on Windows via PowerShell."""

    ps_script = """
$items = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe'
} | Select-Object ProcessId,Name,CommandLine
$items | ConvertTo-Json -Compress -Depth 4
""".strip()

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_script,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if completed.returncode != 0 or not completed.stdout.strip():
        return []

    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def _is_matching_data_factory_process(
    process_info: dict[str, Any],
    *,
    path_patterns: list[str],
) -> bool:
    command_line = str(process_info.get("CommandLine") or "")
    lower = command_line.lower()

    if not lower:
        return False

    # Never kill the current WebUI server process.
    if any(name in lower for name in WEBUI_SCRIPT_NAMES):
        return False

    if not any(name.lower() in lower for name in DATA_FACTORY_PROCESS_SCRIPT_NAMES):
        return False

    return any(pattern in lower for pattern in path_patterns)


def _find_running_data_factory_processes(active_work_dir: Path) -> list[dict[str, Any]]:
    patterns = _path_patterns_for_process_match(active_work_dir)
    matches: list[dict[str, Any]] = []
    for proc in _query_python_processes():
        if _is_matching_data_factory_process(proc, path_patterns=patterns):
            matches.append(proc)
    return matches


def _kill_process_tree(pid: int) -> dict[str, Any]:
    completed = subprocess.run(
        ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "pid": int(pid),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "succeeded": completed.returncode == 0,
    }


def _write_stop_state(
    *,
    active_work_dir: Path,
    report: dict[str, Any],
) -> tuple[Path, Path]:
    runtime_state_path = active_work_dir / RUNTIME_STATE_FILENAME
    stop_report_path = active_work_dir / STOP_REPORT_FILENAME

    runtime_state = {
        "schema_version": "voicelab_data_factory_runtime_state_v1",
        "status": report.get("status"),
        "operation": "stop_current_task",
        "work_dir": str(active_work_dir),
        "updated_at": now_iso(),
        "stop_report_path": str(stop_report_path),
        "matched_processes": report.get("matched_processes", []),
        "kill_results": report.get("kill_results", []),
    }

    _write_json(runtime_state_path, runtime_state)
    _write_json(stop_report_path, report)
    return runtime_state_path, stop_report_path


# ============================================================
# Port helpers
# 端口工具
# ============================================================
def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, int(port))) != 0


def _find_available_port(
    host: str,
    start_port: int,
    *,
    max_tries: int = 20,
) -> int:
    start_port = int(start_port)
    max_tries = int(max_tries)

    for port in range(start_port, start_port + max_tries):
        if _is_port_available(host, port):
            return port

    raise OSError(
        f"No empty port found in range {start_port}-{start_port + max_tries - 1}."
    )


# ============================================================
# Result formatting
# 结果格式化
# ============================================================
def _status_zh(status: Optional[str]) -> str:
    mapping = {
        "prepared": "已准备",
        "running": "运行中",
        "succeeded": "成功",
        "failed": "失败",
        "partial": "部分完成",
        "cancelled": "已中断",
        "not_found": "未找到运行任务",
    }
    return mapping.get(str(status), str(status))


def _safe_json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return {"value": str(value)}


def _format_output_summary(summary: dict[str, Any]) -> str:
    if not summary:
        return "暂无输出。"

    lines = []
    for key, value in summary.items():
        if value is None:
            continue
        lines.append(f"- **{key}**：`{value}`")

    return "\n".join(lines) if lines else "暂无输出。"


def _format_step_summary(result_dict: dict[str, Any]) -> str:
    steps = result_dict.get("steps") or []
    if not steps:
        return "暂无步骤记录。"

    lines = []
    for index, step in enumerate(steps, start=1):
        name = step.get("display_name") or step.get("name") or f"Step {index}"
        status = _status_zh(step.get("status"))
        seconds = step.get("total_seconds")
        message = step.get("message") or ""
        skipped = bool((step.get("extra") or {}).get("auto_resume_skipped"))
        skipped_text = "，自动跳过" if skipped else ""
        suffix = f"，{message}" if message else ""
        if seconds is None:
            lines.append(f"{index}. **{name}**：{status}{skipped_text}{suffix}")
        else:
            lines.append(f"{index}. **{name}**：{status}，耗时 {seconds:.2f} 秒{skipped_text}{suffix}")

    return "\n".join(lines)


def _format_user_result(result: Any) -> tuple[str, dict[str, Any], str, str]:
    result_dict = user_pipeline_result_to_dict(result)
    status = result_dict.get("status")
    status_text = _status_zh(status)

    message = result_dict.get("message") or ""
    current_stage = result_dict.get("current_stage") or ""
    current_step = result_dict.get("current_step") or ""
    failed_step = result_dict.get("failed_step") or ""
    failure_reason = result_dict.get("failure_reason") or ""
    failure_hint = result_dict.get("failure_hint") or ""
    total_seconds = result_dict.get("total_seconds")
    extra = result_dict.get("extra") or {}

    lines = [
        f"## 当前状态：{status_text}",
        "",
        f"**消息：** {message}",
    ]

    if current_stage:
        lines.append(f"**当前阶段：** {current_stage}")

    if current_step:
        lines.append(f"**当前步骤：** `{current_step}`")

    if total_seconds is not None:
        lines.append(f"**总耗时：** {float(total_seconds):.2f} 秒")

    if extra.get("auto_resume_enabled"):
        lines.extend(
            [
                "",
                "### 自动续跑摘要",
                f"- 自动跳过步骤数：`{extra.get('num_skipped_steps', 0)}`",
                f"- 实际执行步骤数：`{extra.get('num_executed_steps', 0)}`",
            ]
        )

    if status == DATA_FACTORY_STATUS_FAILED:
        lines.extend(
            [
                "",
                "### 失败信息",
                f"**失败步骤：** `{failed_step}`" if failed_step else "**失败步骤：** 未知",
                f"**失败原因：** {failure_reason}" if failure_reason else "**失败原因：** 未提供",
            ]
        )
        if failure_hint:
            lines.append(f"**建议：** {failure_hint}")

    if status == DATA_FACTORY_STATUS_SUCCEEDED:
        lines.append("")
        lines.append("处理完成。")

    output_summary = result_dict.get("output_summary") or {}

    return (
        "\n".join(lines),
        result_dict,
        _format_output_summary(output_summary),
        _format_step_summary(result_dict),
    )


def _format_exception(exc: Exception, *, title: str) -> tuple[str, dict[str, Any], str, str]:
    payload = {
        "status": "failed",
        "error_type": type(exc).__name__,
        "message": str(exc),
    }

    markdown = "\n".join(
        [
            f"## {title}",
            "",
            f"**错误类型：** `{type(exc).__name__}`",
            f"**错误消息：** {exc}",
        ]
    )

    return markdown, payload, "暂无输出。", "暂无步骤记录。"


# ============================================================
# Request builder
# 请求构造
# ============================================================
def _make_request(
    *,
    raw_input_dir: str,
    speaker_name: str,
    language: str,
    work_dir: str,
    asr_backend: str,
    asr_model_size: str,
    asr_precision: str,
    threshold: Any,
    min_length: Any,
    min_interval: Any,
    hop_size: Any,
    max_sil_kept: Any,
    normalize_max: Any,
    alpha_mix: Any,
    prompt_mode: str,
    min_prompt_sec: Any,
    max_prompt_sec: Any,
    prefer_prompt_sec: Any,
    allow_self_prompt: bool,
    overwrite_work_dir: bool,
    timeout_seconds: Any,
) -> DataFactoryRequest:
    return DataFactoryRequest(
        raw_input_dir=str(raw_input_dir).strip(),
        speaker_name=str(speaker_name).strip(),
        language=str(language).strip() or "zh",
        work_dir=_blank_to_none(work_dir),
        asr_backend=str(asr_backend).strip() or "auto",
        asr_model_size=str(asr_model_size).strip() or "large-v3",
        asr_precision=str(asr_precision).strip() or "float32",
        threshold=_as_int(threshold, -34),
        min_length=_as_int(min_length, 4000),
        min_interval=_as_int(min_interval, 300),
        hop_size=_as_int(hop_size, 10),
        max_sil_kept=_as_int(max_sil_kept, 500),
        normalize_max=_as_float(normalize_max, 0.9),
        alpha_mix=_as_float(alpha_mix, 0.25),
        export_list=True,
        export_stage2_manifest=True,
        prompt_mode=str(prompt_mode).strip() or "speaker_pool",
        min_prompt_sec=_as_float(min_prompt_sec, 3.0),
        max_prompt_sec=_as_float(max_prompt_sec, 10.0),
        prefer_prompt_sec=_as_float(prefer_prompt_sec, 6.0),
        allow_self_prompt=bool(allow_self_prompt),
        overwrite_work_dir=bool(overwrite_work_dir),
        dry_run=False,
        timeout_seconds=_as_timeout(timeout_seconds) or 7200,
    )


# ============================================================
# Click handlers
# 按钮事件
# ============================================================
def inspect_work_dir_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    try:
        active_work_dir = _resolve_active_work_dir(
            work_dir_output=work_dir_output,
            work_dir=work_dir,
            speaker_name=speaker_name,
        )
        payload, status_md, output_md, step_md = _build_scan_payload(active_work_dir)
        panel_md = _format_work_dir_status_panel(payload)
        return status_md, payload, output_md, step_md, str(active_work_dir), panel_md

    except Exception as exc:
        status_md, dev_json, output_md, step_md = _format_exception(
            exc,
            title="载入/扫描工作目录失败",
        )
        return status_md, dev_json, output_md, step_md, work_dir_output or work_dir or "", "暂无工作目录状态。"


def stop_current_task_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    try:
        active_work_dir = _resolve_active_work_dir(
            work_dir_output=work_dir_output,
            work_dir=work_dir,
            speaker_name=speaker_name,
        )
        active_work_dir.mkdir(parents=True, exist_ok=True)

        matched = _find_running_data_factory_processes(active_work_dir)
        kill_results = []
        for proc in matched:
            pid = proc.get("ProcessId")
            if pid is None:
                continue
            kill_results.append(_kill_process_tree(int(pid)))

        stopped_count = sum(1 for item in kill_results if item.get("succeeded"))
        status = "cancelled" if stopped_count > 0 else "not_found"
        message = (
            f"已停止 {stopped_count} 个 DataFactory 子进程。"
            if stopped_count > 0
            else "未找到与当前工作目录匹配的运行中 DataFactory 子进程。"
        )

        scan_payload, _, output_md, step_md = _build_scan_payload(active_work_dir)
        panel_md = _format_work_dir_status_panel(scan_payload)
        report = {
            "schema_version": "voicelab_data_factory_stop_report_v1",
            "status": status,
            "message": message,
            "work_dir": str(active_work_dir),
            "created_at": now_iso(),
            "matched_processes": matched,
            "kill_results": kill_results,
            "scan_after_stop": scan_payload,
        }
        runtime_state_path, stop_report_path = _write_stop_state(
            active_work_dir=active_work_dir,
            report=report,
        )
        report["runtime_state_path"] = str(runtime_state_path)
        report["stop_report_path"] = str(stop_report_path)

        status_lines = [
            f"## 当前状态：{_status_zh(status)}",
            "",
            f"**消息：** {message}",
            f"**工作目录：** `{active_work_dir}`",
            f"**停止报告：** `{stop_report_path}`",
            f"**运行状态文件：** `{runtime_state_path}`",
            "",
            "### 停止结果",
            f"- 匹配到进程数：`{len(matched)}`",
            f"- 成功停止进程数：`{stopped_count}`",
            "",
            "### 中断后的产物扫描",
            "已刷新工作目录状态面板。当前正在运行的步骤可能留下部分产物；后续继续前建议先点击 **载入/扫描工作目录**。",
        ]

        return "\n".join(status_lines), report, output_md, step_md, str(active_work_dir), panel_md

    except Exception as exc:
        status_md, dev_json, output_md, step_md = _format_exception(
            exc,
            title="停止当前任务失败",
        )
        return status_md, dev_json, output_md, step_md, work_dir_output or work_dir or "", "暂无工作目录状态。"


def run_prepare_click(
    audio_source_mode: str,
    raw_input_dir: str,
    uploaded_audio_files: Any,
    uploaded_audio_folder: Any,
    overwrite_uploaded_audio_cache: bool,
    speaker_name: str,
    language: str,
    work_dir: str,
    asr_backend: str,
    asr_model_size: str,
    asr_precision: str,
    threshold: Any,
    min_length: Any,
    min_interval: Any,
    hop_size: Any,
    max_sil_kept: Any,
    normalize_max: Any,
    alpha_mix: Any,
    prompt_mode: str,
    min_prompt_sec: Any,
    max_prompt_sec: Any,
    prefer_prompt_sec: Any,
    allow_self_prompt: bool,
    overwrite_work_dir: bool,
    timeout_seconds: Any,
    show_terminal_progress: bool,
) -> tuple[str, dict[str, Any], str, str, str]:
    try:
        effective_raw_input_dir, upload_info = _resolve_raw_input_dir_for_prepare(
            audio_source_mode=audio_source_mode,
            raw_input_dir=raw_input_dir,
            uploaded_audio_files=uploaded_audio_files,
            uploaded_audio_folder=uploaded_audio_folder,
            work_dir=work_dir,
            speaker_name=speaker_name,
            overwrite_uploaded_audio_cache=overwrite_uploaded_audio_cache,
        )
        request = _make_request(
            raw_input_dir=effective_raw_input_dir,
            speaker_name=speaker_name,
            language=language,
            work_dir=work_dir,
            asr_backend=asr_backend,
            asr_model_size=asr_model_size,
            asr_precision=asr_precision,
            threshold=threshold,
            min_length=min_length,
            min_interval=min_interval,
            hop_size=hop_size,
            max_sil_kept=max_sil_kept,
            normalize_max=normalize_max,
            alpha_mix=alpha_mix,
            prompt_mode=prompt_mode,
            min_prompt_sec=min_prompt_sec,
            max_prompt_sec=max_prompt_sec,
            prefer_prompt_sec=prefer_prompt_sec,
            allow_self_prompt=allow_self_prompt,
            overwrite_work_dir=overwrite_work_dir,
            timeout_seconds=timeout_seconds,
        )

        pipeline = create_data_factory_user_pipeline_service(
            show_terminal_progress=bool(show_terminal_progress),
        )
        result = pipeline.run_prepare_pipeline(request)

        status_md, dev_json, output_md, step_md = _format_user_result(result)

        upload_info_md = _format_upload_info(upload_info)
        if upload_info_md:
            status_md = f"{status_md}\n{upload_info_md}"

        if upload_info:
            dev_json["upload_info"] = upload_info

        resolved_work_dir = str(result.work_dir)

        return status_md, dev_json, output_md, step_md, resolved_work_dir

    except Exception as exc:
        status_md, dev_json, output_md, step_md = _format_exception(
            exc,
            title="执行数据工厂失败",
        )
        return status_md, dev_json, output_md, step_md, work_dir or ""


def launch_proofread_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
    proofread_port: Any,
    proofread_g_batch: Any,
    proofread_overwrite_backup: bool,
) -> tuple[str, dict[str, Any], str]:
    try:
        active_work_dir = _resolve_active_work_dir(
            work_dir_output=work_dir_output,
            work_dir=work_dir,
            speaker_name=speaker_name,
        )
        pipeline = create_data_factory_user_pipeline_service()
        payload = pipeline.launch_proofread(
            work_dir=active_work_dir,
            webui_port_subfix=_as_int(proofread_port, 9871),
            g_batch=_as_int(proofread_g_batch, 10),
            overwrite_backup=bool(proofread_overwrite_backup),
        )

        markdown = "\n".join(
            [
                "## 人工校对器已启动",
                "",
                f"**工作目录：** `{active_work_dir}`",
                f"**校对器地址：** `{payload.get('proofread_url')}`",
                f"**进程 PID：** `{payload.get('pid')}`",
                f"**dataset.list：** `{payload.get('dataset_list')}`",
                "",
                "请在旧版校对器中完成文本校对。完成后回到本页面，点击 **我已完成校对**。",
            ]
        )

        payload["active_work_dir"] = str(active_work_dir)
        return markdown, payload, str(active_work_dir)

    except Exception as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        markdown = "\n".join(
            [
                "## 人工校对器启动失败",
                "",
                f"**错误类型：** `{type(exc).__name__}`",
                f"**错误消息：** {exc}",
            ]
        )
        return markdown, payload, work_dir_output or work_dir or ""


def rebuild_after_proofread_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
    duration_tolerance_sec: Any,
    rebuild_timeout_seconds: Any,
) -> tuple[str, dict[str, Any], str, str, str]:
    try:
        active_work_dir = _resolve_active_work_dir(
            work_dir_output=work_dir_output,
            work_dir=work_dir,
            speaker_name=speaker_name,
        )
        pipeline = create_data_factory_user_pipeline_service()
        result = pipeline.rebuild_after_proofread(
            work_dir=active_work_dir,
            duration_tolerance_sec=_as_float(duration_tolerance_sec, 0.05),
            prompt_mode="self",
            timeout_seconds=_as_timeout(rebuild_timeout_seconds) or 600,
        )
        status_md, dev_json, output_md, step_md = _format_user_result(result)
        return status_md, dev_json, output_md, step_md, str(active_work_dir)

    except Exception as exc:
        status_md, dev_json, output_md, step_md = _format_exception(
            exc,
            title="重建校对后 manifest 失败",
        )
        return status_md, dev_json, output_md, step_md, work_dir_output or work_dir or ""


def skip_proofread_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
) -> tuple[str, dict[str, Any], str, str, str]:
    try:
        active_work_dir = _resolve_active_work_dir(
            work_dir_output=work_dir_output,
            work_dir=work_dir,
            speaker_name=speaker_name,
        )
        pipeline = create_data_factory_user_pipeline_service()
        result = pipeline.skip_proofread(
            work_dir=active_work_dir,
            overwrite=True,
        )
        status_md, dev_json, output_md, step_md = _format_user_result(result)
        return status_md, dev_json, output_md, step_md, str(active_work_dir)

    except Exception as exc:
        status_md, dev_json, output_md, step_md = _format_exception(
            exc,
            title="跳过人工校对失败",
        )
        return status_md, dev_json, output_md, step_md, work_dir_output or work_dir or ""


def run_postprocess_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
    stage2_device: str,
    stage2_use_half: bool,
    stage2_overwrite: bool,
    continuous_dtype: str,
    continuous_overwrite: bool,
    style_overwrite: bool,
    enable_speaker_embedding: bool,
    auto_reject_bad_samples: bool,
    train_ratio: Any,
    split_overwrite: bool,
    timeout_seconds: Any,
    prompt_mode: str,
    min_prompt_sec: Any,
    max_prompt_sec: Any,
    prefer_prompt_sec: Any,
    allow_self_prompt: bool,
    target_sr: Any,
    n_fft: Any,
    hop_length: Any,
    win_length: Any,
    n_mels: Any,
    fmin: Any,
    fmax: Any,
    f0_min_hz: Any,
    f0_max_hz: Any,
    show_terminal_progress: bool,
) -> tuple[str, dict[str, Any], str, str, str]:
    try:
        active_work_dir = _resolve_active_work_dir(
            work_dir_output=work_dir_output,
            work_dir=work_dir,
            speaker_name=speaker_name,
        )
        pipeline = create_data_factory_user_pipeline_service(
            show_terminal_progress=bool(show_terminal_progress),
        )

        result = pipeline.run_postprocess_pipeline(
            work_dir=active_work_dir,
            prompt_mode=str(prompt_mode).strip() or "speaker_pool",
            min_prompt_sec=_as_float(min_prompt_sec, 3.0),
            max_prompt_sec=_as_float(max_prompt_sec, 10.0),
            prefer_prompt_sec=_as_float(prefer_prompt_sec, 6.0),
            allow_self_prompt=bool(allow_self_prompt),
            stage2_device=str(stage2_device).strip() or "cuda",
            stage2_use_half=bool(stage2_use_half),
            stage2_overwrite=bool(stage2_overwrite),
            target_sr=_as_int(target_sr, 22050),
            n_fft=_as_int(n_fft, 1024),
            hop_length=_as_int(hop_length, 256),
            win_length=_as_int(win_length, 1024),
            n_mels=_as_int(n_mels, 80),
            fmin=_as_float(fmin, 0.0),
            fmax=_as_float(fmax, 8000.0),
            continuous_device=str(stage2_device).strip() or "cuda",
            continuous_dtype=str(continuous_dtype).strip() or "float32",
            continuous_overwrite=bool(continuous_overwrite),
            style_device=str(stage2_device).strip() or "cuda",
            style_overwrite=bool(style_overwrite),
            enable_speaker_embedding=bool(enable_speaker_embedding),
            f0_min_hz=_as_float(f0_min_hz, 50.0),
            f0_max_hz=_as_float(f0_max_hz, 1100.0),
            split_train_ratio=_as_float(train_ratio, 0.9),
            split_val_count=None,
            split_overwrite=bool(split_overwrite),
            auto_reject_bad_samples=bool(auto_reject_bad_samples),
            timeout_seconds=_as_timeout(timeout_seconds),
        )

        status_md, dev_json, output_md, step_md = _format_user_result(result)
        return status_md, dev_json, output_md, step_md, str(active_work_dir)

    except Exception as exc:
        status_md, dev_json, output_md, step_md = _format_exception(
            exc,
            title="一键生成训练数据失败",
        )
        return status_md, dev_json, output_md, step_md, work_dir_output or work_dir or ""


def run_resume_postprocess_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
    stage2_device: str,
    stage2_use_half: bool,
    stage2_overwrite: bool,
    continuous_dtype: str,
    continuous_overwrite: bool,
    style_overwrite: bool,
    enable_speaker_embedding: bool,
    auto_reject_bad_samples: bool,
    train_ratio: Any,
    split_overwrite: bool,
    timeout_seconds: Any,
    prompt_mode: str,
    min_prompt_sec: Any,
    max_prompt_sec: Any,
    prefer_prompt_sec: Any,
    allow_self_prompt: bool,
    target_sr: Any,
    n_fft: Any,
    hop_length: Any,
    win_length: Any,
    n_mels: Any,
    fmin: Any,
    fmax: Any,
    f0_min_hz: Any,
    f0_max_hz: Any,
    show_terminal_progress: bool,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    try:
        active_work_dir = _resolve_active_work_dir(
            work_dir_output=work_dir_output,
            work_dir=work_dir,
            speaker_name=speaker_name,
        )

        result = run_resumable_postprocess_pipeline(
            work_dir=active_work_dir,
            show_terminal_progress=bool(show_terminal_progress),
            prompt_mode=str(prompt_mode).strip() or "speaker_pool",
            min_prompt_sec=_as_float(min_prompt_sec, 3.0),
            max_prompt_sec=_as_float(max_prompt_sec, 10.0),
            prefer_prompt_sec=_as_float(prefer_prompt_sec, 6.0),
            allow_self_prompt=bool(allow_self_prompt),
            stage2_device=str(stage2_device).strip() or "cuda",
            stage2_use_half=bool(stage2_use_half),
            stage2_overwrite=bool(stage2_overwrite),
            target_sr=_as_int(target_sr, 22050),
            n_fft=_as_int(n_fft, 1024),
            hop_length=_as_int(hop_length, 256),
            win_length=_as_int(win_length, 1024),
            n_mels=_as_int(n_mels, 80),
            fmin=_as_float(fmin, 0.0),
            fmax=_as_float(fmax, 8000.0),
            continuous_device=str(stage2_device).strip() or "cuda",
            continuous_dtype=str(continuous_dtype).strip() or "float32",
            continuous_overwrite=bool(continuous_overwrite),
            style_device=str(stage2_device).strip() or "cuda",
            style_overwrite=bool(style_overwrite),
            enable_speaker_embedding=bool(enable_speaker_embedding),
            f0_min_hz=_as_float(f0_min_hz, 50.0),
            f0_max_hz=_as_float(f0_max_hz, 1100.0),
            split_train_ratio=_as_float(train_ratio, 0.9),
            split_val_count=None,
            split_overwrite=bool(split_overwrite),
            auto_reject_bad_samples=bool(auto_reject_bad_samples),
            timeout_seconds=_as_timeout(timeout_seconds),
        )

        status_md, dev_json, output_md, step_md = _format_user_result(result)
        scan_payload, _, _, _ = _build_scan_payload(active_work_dir)
        dev_json["scan_after_resume"] = scan_payload
        panel_md = _format_work_dir_status_panel(scan_payload)
        return status_md, dev_json, output_md, step_md, str(active_work_dir), panel_md

    except Exception as exc:
        status_md, dev_json, output_md, step_md = _format_exception(
            exc,
            title="继续生成训练数据失败",
        )
        return status_md, dev_json, output_md, step_md, work_dir_output or work_dir or "", "暂无工作目录状态。"


# ============================================================
# Demo
# UI 定义
# ============================================================
def create_demo() -> gr.Blocks:
    with gr.Blocks(title="VoiceLab 数据工厂 - 用户版") as demo:
        gr.Markdown(
            """
# VoiceLab 数据工厂 - 用户版

这个页面用于把原始音频转换为 VoiceLab Stage2 训练数据。

推荐流程：

1. 填写原始音频目录和说话人信息；
2. 点击 **执行数据工厂**；
3. 选择 **启动人工校对器** 或 **跳过人工校对**；
4. 首次生成点击 **一键生成训练数据**；已有工作目录或中断后恢复点击 **继续生成训练数据**。

如果页面刷新或已经存在工作目录，可以填写工作目录后点击 **载入/扫描工作目录**，再继续后续步骤。
长任务运行中如需中断，可以点击 **停止当前任务**。中断后建议再次点击 **载入/扫描工作目录** 查看已保留产物。
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. 基础输入")

                audio_source_mode = gr.Radio(
                    label="原始音频来源",
                    choices=["本地目录路径", "上传音频文件", "上传音频文件夹"],
                    value="本地目录路径",
                    info="大数据集建议使用本地目录路径；小样本或测试数据可以使用上传入口。",
                )

                raw_input_dir = gr.Textbox(
                    label="本地原始音频目录",
                    value="",
                    info="填写已存在的本地音频目录。适合十几分钟、数小时或更大的数据集。",
                    visible=True,
                )

                uploaded_audio_files = gr.File(
                    label="上传音频文件",
                    file_count="multiple",
                    file_types=AUDIO_FILE_TYPES,
                    type="filepath",
                    visible=False,
                )

                uploaded_audio_folder = gr.File(
                    label="上传音频文件夹",
                    file_count="directory",
                    type="filepath",
                    visible=False,
                )

                upload_hint = gr.Markdown(
                    """
                上传模式会先把音频复制到：

                `work_dir/00_uploaded_raw_audio/`

                然后再把该目录作为原始音频目录传给数据工厂。大数据集不建议使用浏览器上传。
                """,
                    visible=False,
                )

                speaker_name = gr.Textbox(
                    label="说话人 / 角色名",
                    value="",
                )

                language = gr.Dropdown(
                    label="语言",
                    choices=["zh", "en", "ja"],
                    value="zh",
                )

                work_dir = gr.Textbox(
                    label="工作目录（可选）",
                    value="",
                    info="留空时自动使用 user_data/{speaker_name}_factory。页面刷新后可在这里填入旧工作目录再点击载入/扫描。",
                )

                with gr.Row():
                    load_work_dir_button = gr.Button(
                        "载入/扫描工作目录",
                        variant="secondary",
                    )
                    stop_current_task_button = gr.Button(
                        "停止当前任务",
                        variant="stop",
                    )

                with gr.Accordion("常用设置", open=True):
                    overwrite_work_dir = gr.Checkbox(
                        label="覆盖已有数据工厂工作目录",
                        value=False,
                    )
                    overwrite_uploaded_audio_cache = gr.Checkbox(
                        label="覆盖已上传音频缓存",
                        value=True,
                        info="仅清理 work_dir/00_uploaded_raw_audio，不会删除原始本地音频目录。",
                    )
                    timeout_seconds = gr.Number(
                        label="单步超时秒数，0 表示不限制",
                        value=0,
                        precision=0,
                    )
                    train_ratio = gr.Number(
                        label="训练集比例 train_ratio",
                        value=0.9,
                    )
                    enable_speaker_embedding = gr.State(True)
                    auto_reject_bad_samples = gr.Checkbox(
                        label="自动隔离异常样本",
                        value=False,
                        info="关闭时只生成异常报告，不移动样本；开启后会把异常样本移动到 rejected 目录。",
                    )
                    show_terminal_progress = gr.Checkbox(
                        label="显示终端进度窗口",
                        value=True,
                        info="长任务运行时打开终端窗口实时查看日志；任务结束后窗口会保留，可手动关闭。",
                    )

                with gr.Accordion("高级设置", open=False):
                    gr.Markdown("### ASR 设置")
                    asr_backend = gr.Dropdown(
                        label="ASR 后端",
                        choices=["auto", "faster_whisper", "whisper", "none"],
                        value="auto",
                    )
                    asr_model_size = gr.Textbox(
                        label="ASR 模型大小",
                        value="large-v3",
                    )
                    asr_precision = gr.Textbox(
                        label="ASR 精度",
                        value="float32",
                    )

                    gr.Markdown("### 音频切分参数")
                    with gr.Row():
                        threshold = gr.Number(label="threshold", value=-34, precision=0)
                        min_length = gr.Number(label="min_length", value=4000, precision=0)
                        min_interval = gr.Number(label="min_interval", value=300, precision=0)
                    with gr.Row():
                        hop_size = gr.Number(label="hop_size", value=10, precision=0)
                        max_sil_kept = gr.Number(label="max_sil_kept", value=500, precision=0)
                    with gr.Row():
                        normalize_max = gr.Number(label="normalize_max", value=0.9)
                        alpha_mix = gr.Number(label="alpha_mix", value=0.25)

                    gr.Markdown("### Prompt 设置")
                    with gr.Row():
                        prompt_mode = gr.Dropdown(
                            label="prompt_mode",
                            choices=["self", "speaker_pool", "fixed_reference"],
                            value="speaker_pool",
                        )
                        allow_self_prompt = gr.Checkbox(
                            label="allow_self_prompt",
                            value=True,
                        )
                    with gr.Row():
                        min_prompt_sec = gr.Number(label="min_prompt_sec", value=3.0)
                        max_prompt_sec = gr.Number(label="max_prompt_sec", value=10.0)
                        prefer_prompt_sec = gr.Number(label="prefer_prompt_sec", value=6.0)

                    gr.Markdown("### Stage2 / Cache 参数")
                    with gr.Row():
                        stage2_device = gr.Dropdown(
                            label="device",
                            choices=["cuda", "cpu"],
                            value="cuda",
                        )
                        stage2_use_half = gr.Checkbox(
                            label="Stage2 use_half",
                            value=False,
                        )
                    with gr.Row():
                        stage2_overwrite = gr.Checkbox(
                            label="覆盖已有 Stage2 .pt",
                            value=False,
                        )
                        continuous_overwrite = gr.Checkbox(
                            label="覆盖已有 continuous cache",
                            value=False,
                        )
                        style_overwrite = gr.Checkbox(
                            label="覆盖已有 style cache",
                            value=False,
                        )
                        split_overwrite = gr.Checkbox(
                            label="覆盖已有 train/val",
                            value=False,
                        )

                    continuous_dtype = gr.Dropdown(
                        label="continuous dtype",
                        choices=["float16", "float32"],
                        value="float32",
                    )

                    gr.Markdown("### Mel / F0 参数")
                    with gr.Row():
                        target_sr = gr.Number(label="target_sr", value=22050, precision=0)
                        n_fft = gr.Number(label="n_fft", value=1024, precision=0)
                        hop_length = gr.Number(label="hop_length", value=256, precision=0)
                        win_length = gr.Number(label="win_length", value=1024, precision=0)
                    with gr.Row():
                        n_mels = gr.Number(label="n_mels", value=80, precision=0)
                        fmin = gr.Number(label="fmin", value=0.0)
                        fmax = gr.Number(label="fmax", value=8000.0)
                    with gr.Row():
                        f0_min_hz = gr.Number(label="f0_min_hz", value=50.0)
                        f0_max_hz = gr.Number(label="f0_max_hz", value=1100.0)

                gr.Markdown("## 2. 执行流程")

                run_prepare_button = gr.Button(
                    "执行数据工厂",
                    variant="primary",
                )

                with gr.Row():
                    launch_proofread_button = gr.Button(
                        "启动人工校对器",
                        variant="secondary",
                    )
                    rebuild_after_proofread_button = gr.Button(
                        "我已完成校对",
                        variant="secondary",
                    )
                    skip_proofread_button = gr.Button(
                        "跳过人工校对",
                        variant="secondary",
                    )

                with gr.Row():
                    run_postprocess_button = gr.Button(
                        "一键生成训练数据",
                        variant="primary",
                    )
                    resume_postprocess_button = gr.Button(
                        "继续生成训练数据",
                        variant="secondary",
                    )

                with gr.Accordion("人工校对设置", open=False):
                    proofread_port = gr.Number(
                        label="校对器端口",
                        value=9871,
                        precision=0,
                    )
                    proofread_g_batch = gr.Number(
                        label="校对批大小 g_batch",
                        value=10,
                        precision=0,
                    )
                    proofread_overwrite_backup = gr.Checkbox(
                        label="覆盖已有校对备份",
                        value=False,
                    )
                    duration_tolerance_sec = gr.Number(
                        label="重建 corrected manifest 时长容差",
                        value=0.05,
                    )
                    rebuild_timeout_seconds = gr.Number(
                        label="重建超时秒数",
                        value=600,
                        precision=0,
                    )

            with gr.Column(scale=1):
                gr.Markdown("## 状态")

                status_markdown = gr.Markdown("等待开始。")

                gr.Markdown("## 工作目录状态")
                work_dir_status_markdown = gr.Markdown("暂无工作目录状态。")

                gr.Markdown("## 输出摘要")
                output_summary_markdown = gr.Markdown("暂无输出。")

                gr.Markdown("## 步骤摘要")
                step_summary_markdown = gr.Markdown("暂无步骤记录。")

                work_dir_output = gr.Textbox(
                    label="当前工作目录",
                    value="",
                    interactive=False,
                )

                proofread_status = gr.Markdown("人工校对器尚未启动。")
                proofread_json = gr.JSON(label="人工校对启动信息", visible=False)

                with gr.Accordion("开发者详情", open=False):
                    developer_json = gr.JSON(label="用户级 result JSON")

        # ----------------------------------------------------
        # Bind events
        # 绑定事件
        # ----------------------------------------------------
        audio_source_mode.change(
            fn=audio_source_mode_change,
            inputs=[audio_source_mode],
            outputs=[
                raw_input_dir,
                uploaded_audio_files,
                uploaded_audio_folder,
                upload_hint,
            ],
        )

        load_work_dir_button.click(
            fn=inspect_work_dir_click,
            inputs=[
                work_dir_output,
                work_dir,
                speaker_name,
            ],
            outputs=[
                status_markdown,
                developer_json,
                output_summary_markdown,
                step_summary_markdown,
                work_dir_output,
                work_dir_status_markdown,
            ],
        )

        stop_current_task_button.click(
            fn=stop_current_task_click,
            inputs=[
                work_dir_output,
                work_dir,
                speaker_name,
            ],
            outputs=[
                status_markdown,
                developer_json,
                output_summary_markdown,
                step_summary_markdown,
                work_dir_output,
                work_dir_status_markdown,
            ],
        )

        run_prepare_button.click(
            fn=run_prepare_click,
            inputs=[
                audio_source_mode,
                raw_input_dir,
                uploaded_audio_files,
                uploaded_audio_folder,
                overwrite_uploaded_audio_cache,
                speaker_name,
                language,
                work_dir,
                asr_backend,
                asr_model_size,
                asr_precision,
                threshold,
                min_length,
                min_interval,
                hop_size,
                max_sil_kept,
                normalize_max,
                alpha_mix,
                prompt_mode,
                min_prompt_sec,
                max_prompt_sec,
                prefer_prompt_sec,
                allow_self_prompt,
                overwrite_work_dir,
                timeout_seconds,
                show_terminal_progress,
            ],
            outputs=[
                status_markdown,
                developer_json,
                output_summary_markdown,
                step_summary_markdown,
                work_dir_output,
            ],
        )

        launch_proofread_button.click(
            fn=launch_proofread_click,
            inputs=[
                work_dir_output,
                work_dir,
                speaker_name,
                proofread_port,
                proofread_g_batch,
                proofread_overwrite_backup,
            ],
            outputs=[
                proofread_status,
                proofread_json,
                work_dir_output,
            ],
        )

        rebuild_after_proofread_button.click(
            fn=rebuild_after_proofread_click,
            inputs=[
                work_dir_output,
                work_dir,
                speaker_name,
                duration_tolerance_sec,
                rebuild_timeout_seconds,
            ],
            outputs=[
                status_markdown,
                developer_json,
                output_summary_markdown,
                step_summary_markdown,
                work_dir_output,
            ],
        )

        skip_proofread_button.click(
            fn=skip_proofread_click,
            inputs=[
                work_dir_output,
                work_dir,
                speaker_name,
            ],
            outputs=[
                status_markdown,
                developer_json,
                output_summary_markdown,
                step_summary_markdown,
                work_dir_output,
            ],
        )

        postprocess_inputs = [
            work_dir_output,
            work_dir,
            speaker_name,
            stage2_device,
            stage2_use_half,
            stage2_overwrite,
            continuous_dtype,
            continuous_overwrite,
            style_overwrite,
            enable_speaker_embedding,
            auto_reject_bad_samples,
            train_ratio,
            split_overwrite,
            timeout_seconds,
            prompt_mode,
            min_prompt_sec,
            max_prompt_sec,
            prefer_prompt_sec,
            allow_self_prompt,
            target_sr,
            n_fft,
            hop_length,
            win_length,
            n_mels,
            fmin,
            fmax,
            f0_min_hz,
            f0_max_hz,
            show_terminal_progress,
        ]

        run_postprocess_button.click(
            fn=run_postprocess_click,
            inputs=postprocess_inputs,
            outputs=[
                status_markdown,
                developer_json,
                output_summary_markdown,
                step_summary_markdown,
                work_dir_output,
            ],
        )

        resume_postprocess_button.click(
            fn=run_resume_postprocess_click,
            inputs=postprocess_inputs,
            outputs=[
                status_markdown,
                developer_json,
                output_summary_markdown,
                step_summary_markdown,
                work_dir_output,
                work_dir_status_markdown,
            ],
        )

    return demo


# ============================================================
# CLI
# 命令行入口
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoiceLab 数据工厂用户版 WebUI")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="If the requested port is occupied, automatically try later ports.",
    )
    parser.add_argument(
        "--auto-port-tries",
        type=int,
        default=20,
        help="How many consecutive ports to try when --auto-port is enabled.",
    )
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    port = int(args.port)
    if bool(args.auto_port):
        selected_port = _find_available_port(
            args.host,
            port,
            max_tries=int(args.auto_port_tries),
        )
        if selected_port != port:
            print(
                f"[INFO] Requested port {port} is occupied. "
                f"Using available port {selected_port} instead."
            )
        port = selected_port

    demo = create_demo()
    try:
        demo.queue(default_concurrency_limit=8)
    except TypeError:
        demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=port,
        inbrowser=bool(args.inbrowser),
    )


if __name__ == "__main__":
    main()
