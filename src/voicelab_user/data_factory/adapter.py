from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import DataFactoryConfig, build_data_factory_config
from .request import DataFactoryRequest
from .result import DataFactoryArtifactPaths


PathLike = str | Path

DEFAULT_PREPARE_FEWSHOT_SCRIPT = "scripts/prepare_fewshot_dataset.py"
DEFAULT_HEALTH_CHECK_SCRIPT = "scripts/check_fewshot_dataset_health.py"
DEFAULT_EXPORT_FEWSHOT_STAGE2_SCRIPT = "scripts/export_fewshot_stage2_manifest.py"
DEFAULT_PREPROCESS_STAGE2_FM_SCRIPT = "scripts/preprocess_stage2_fm_dataset.py"
DEFAULT_CONTINUOUS_SEMANTIC_CACHE_SCRIPT = "scripts/export_stage2_continuous_semantic_cache.py"
DEFAULT_FEWSHOT_STYLE_CACHE_SCRIPT = "scripts/export_fewshot_style_cache.py"
DEFAULT_SPLIT_STAGE2_PT_SCRIPT = "scripts/split_stage2_pt_dataset.py"
DEFAULT_FILTER_V662_BAD_STYLE_SAMPLES_SCRIPT = "scripts/filter_v662_bad_style_samples.py"

DEFAULT_LAUNCH_PROOFREAD_SCRIPT = "src/data_factory/cli/launch_proofread.py"
DEFAULT_REBUILD_CORRECTED_MANIFEST_SCRIPT = (
    "src/data_factory/importers/rebuild_corrected_manifest.py"
)


@dataclass(frozen=True)
class DataFactoryCommand:
    """One prepared data factory command.

    中文说明：
        一条已经构造好的数据工厂命令。

        P10-2 只构造命令，不执行 heavy job。
        P10-3 / P10-4 后续 service 再决定是否运行。
    """

    name: str
    argv: list[str]
    cwd: Path
    expected_report_path: Optional[Path] = None
    expected_output_paths: dict[str, Path] = field(default_factory=dict)

    def to_printable(self) -> str:
        """Return a readable Windows-friendly command string."""

        parts: list[str] = []
        for value in self.argv:
            text = str(value)
            if " " in text or "\\" in text or ":" in text:
                parts.append(f'"{text}"')
            else:
                parts.append(text)
        return " ".join(parts)


@dataclass(frozen=True)
class DataFactoryCommandPlan:
    """Dry-run command plan for one data factory request.

    中文说明：
        一次数据工厂请求对应的 dry-run 命令计划。

        这里包含：
            - resolved request
            - expected artifacts
            - command list

        但不会执行任何命令。
    """

    request: DataFactoryRequest
    artifacts: DataFactoryArtifactPaths
    commands: list[DataFactoryCommand]
    dry_run_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert plan to a JSON-friendly dictionary."""

        return {
            "dry_run_only": bool(self.dry_run_only),
            "request": {
                "schema_version": self.request.schema_version,
                "raw_input_dir": str(self.request.raw_input_dir),
                "speaker_name": str(self.request.speaker_name),
                "language": str(self.request.language),
                "work_dir": None if self.request.work_dir is None else str(self.request.work_dir),
                "asr_backend": str(self.request.asr_backend),
                "asr_model_size": str(self.request.asr_model_size),
                "asr_precision": str(self.request.asr_precision),
                "enable_uvr": bool(self.request.enable_uvr),
                "enable_denoise": bool(self.request.enable_denoise),
                "threshold": int(self.request.threshold),
                "min_length": int(self.request.min_length),
                "min_interval": int(self.request.min_interval),
                "hop_size": int(self.request.hop_size),
                "max_sil_kept": int(self.request.max_sil_kept),
                "normalize_max": float(self.request.normalize_max),
                "alpha_mix": float(self.request.alpha_mix),
                "export_list": bool(self.request.export_list),
                "export_stage2_manifest": bool(self.request.export_stage2_manifest),
                "prompt_mode": str(self.request.prompt_mode),
                "min_prompt_sec": float(self.request.min_prompt_sec),
                "max_prompt_sec": float(self.request.max_prompt_sec),
                "prefer_prompt_sec": float(self.request.prefer_prompt_sec),
                "allow_self_prompt": bool(self.request.allow_self_prompt),
                "overwrite_work_dir": bool(self.request.overwrite_work_dir),
                "dry_run": bool(self.request.dry_run),
                "timeout_seconds": self.request.timeout_seconds,
            },
            "artifacts": {
                "work_dir": str(self.artifacts.work_dir),
                "request_json_path": (
                    None
                    if self.artifacts.request_json_path is None
                    else str(self.artifacts.request_json_path)
                ),
                "result_json_path": (
                    None
                    if self.artifacts.result_json_path is None
                    else str(self.artifacts.result_json_path)
                ),
                "raw_audio_index_path": (
                    None
                    if self.artifacts.raw_audio_index_path is None
                    else str(self.artifacts.raw_audio_index_path)
                ),
                "prepare_report_path": (
                    None
                    if self.artifacts.prepare_report_path is None
                    else str(self.artifacts.prepare_report_path)
                ),
                "manifest_jsonl_path": (
                    None
                    if self.artifacts.manifest_jsonl_path is None
                    else str(self.artifacts.manifest_jsonl_path)
                ),
                "dataset_list_path": (
                    None
                    if self.artifacts.dataset_list_path is None
                    else str(self.artifacts.dataset_list_path)
                ),
                "stage2_manifest_jsonl_path": (
                    None
                    if self.artifacts.stage2_manifest_jsonl_path is None
                    else str(self.artifacts.stage2_manifest_jsonl_path)
                ),
                "stage2_manifest_fewshot_path": (
                    None
                    if self.artifacts.stage2_manifest_fewshot_path is None
                    else str(self.artifacts.stage2_manifest_fewshot_path)
                ),
                "manifest_health_report_path": (
                    None
                    if self.artifacts.manifest_health_report_path is None
                    else str(self.artifacts.manifest_health_report_path)
                ),
                "stage2_manifest_health_report_path": (
                    None
                    if self.artifacts.stage2_manifest_health_report_path is None
                    else str(self.artifacts.stage2_manifest_health_report_path)
                ),
            },
            "commands": [
                {
                    "name": cmd.name,
                    "argv": [str(x) for x in cmd.argv],
                    "cwd": str(cmd.cwd),
                    "printable": cmd.to_printable(),
                    "expected_report_path": (
                        None if cmd.expected_report_path is None else str(cmd.expected_report_path)
                    ),
                    "expected_output_paths": {
                        str(k): str(v) for k, v in cmd.expected_output_paths.items()
                    },
                }
                for cmd in self.commands
            ],
        }


class DataFactoryAdapter:
    """Build command plans for user-edition data factory.

    中文说明：
        用户版数据工厂 adapter。

        P10-2 只做 command build / dry-run plan。
        不执行 subprocess，不启动 ASR，不启动切分，不写大文件。

        后续 P10-3 service 可以基于这里的 command plan：
            - 保存 data_factory_request.json
            - 保存 data_factory_command_plan.json
            - 再逐步执行 command
    """

    def __init__(
        self,
        *,
        project_root: Optional[PathLike] = None,
        python_executable: Optional[PathLike] = None,
        config: Optional[DataFactoryConfig] = None,
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[3]
        )

        self.python_executable = (
            str(Path(python_executable).resolve())
            if python_executable is not None
            else sys.executable
        )

        self.config = config if config is not None else build_data_factory_config()

    # ------------------------------------------------------------------
    # Path helpers
    # 路径工具
    # ------------------------------------------------------------------
    def resolve_path(self, value: PathLike) -> Path:
        """Resolve user path against project root when relative."""

        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve(strict=False)
        return (self.project_root / path).resolve(strict=False)

    def suggest_work_dir(self, speaker_name: str) -> Path:
        """Suggest default work_dir from config.

        默认：
            user_data/{speaker_name}_factory
        """

        safe_speaker_name = str(speaker_name).strip()
        if not safe_speaker_name:
            raise ValueError("speaker_name must not be empty when suggesting work_dir.")

        dirname = self.config.paths.work_dir_pattern.format(
            speaker_name=safe_speaker_name
        )
        return (self.config.paths.default_user_data_root / dirname).resolve(strict=False)

    def resolve_work_dir(self, request: DataFactoryRequest) -> Path:
        """Resolve request work_dir or suggest one."""

        if request.work_dir is None or str(request.work_dir).strip() == "":
            return self.suggest_work_dir(request.speaker_name)

        return self.resolve_path(request.work_dir)

    def expected_artifacts(self, work_dir: Path) -> DataFactoryArtifactPaths:
        """Return expected artifact paths for a data factory run."""

        work_dir = Path(work_dir).resolve(strict=False)
        export_dir = work_dir / "06_export"
        stage2_pt_root = _suggest_stage2_pt_root_from_work_dir(work_dir)
        continuous_semantic_root = _suggest_continuous_semantic_root_from_stage2_pt_root(
            stage2_pt_root
        )
        style_cache_root = _suggest_style_cache_root_from_stage2_pt_root(
            stage2_pt_root
        )
        train_pt_root, val_pt_root, split_report_path = _suggest_split_roots_from_style_cache_root(
            style_cache_root
        )
        rejected_pt_root, filter_report_json_path, filter_report_csv_path = (
            _suggest_filter_outputs_from_split_parent(train_pt_root.parent)
        )

        return DataFactoryArtifactPaths(
            work_dir=work_dir,
            request_json_path=work_dir / "data_factory_request.json",
            result_json_path=work_dir / "data_factory_result.json",

            raw_audio_index_path=work_dir / "00_raw_index" / "raw_audio_index.jsonl",
            prepare_report_path=work_dir / "fewshot_prepare_report.json",

            manifest_jsonl_path=export_dir / "manifest.jsonl",
            dataset_list_path=export_dir / "dataset.list",
            stage2_manifest_jsonl_path=export_dir / "stage2_manifest.jsonl",

            export_clips_dir=export_dir / "clips",

            manifest_before_proofread_path=export_dir / "manifest.before_proofread.jsonl",
            dataset_before_proofread_path=export_dir / "dataset.before_proofread.list",
            stage2_manifest_before_proofread_path=export_dir / "stage2_manifest.before_proofread.jsonl",

            manifest_corrected_path=export_dir / "manifest.corrected.jsonl",
            stage2_manifest_corrected_path=export_dir / "stage2_manifest.corrected.jsonl",
            correction_report_path=export_dir / "correction_report.json",

            stage2_manifest_fewshot_path=export_dir / "stage2_manifest.fewshot.jsonl",
            prompt_selection_report_path=export_dir / "prompt_selection_report.json",
            fewshot_manifest_conversion_report_path=export_dir / "fewshot_manifest_conversion_report.json",

            manifest_health_report_path=export_dir / "fewshot_health_manifest_report.json",
            stage2_manifest_health_report_path=export_dir / "fewshot_health_stage2_manifest_report.json",

            stage2_pt_root=stage2_pt_root,
            stage2_pt_health_report_path=stage2_pt_root / "fewshot_health_pt_report.json",

            continuous_semantic_root=continuous_semantic_root,
            style_cache_root=style_cache_root,

            train_pt_root=train_pt_root,
            val_pt_root=val_pt_root,
            split_report_path=split_report_path,

            rejected_pt_root=rejected_pt_root,
            filter_report_json_path=filter_report_json_path,
            filter_report_csv_path=filter_report_csv_path,
        )

    # ------------------------------------------------------------------
    # Validation
    # 校验
    # ------------------------------------------------------------------
    def validate_request(self, request: DataFactoryRequest) -> None:
        """Validate request before building commands."""

        if request.schema_version != "voicelab_data_factory_request_v1":
            raise ValueError(f"Unsupported request schema_version: {request.schema_version}")

        if not str(request.raw_input_dir).strip():
            raise ValueError("raw_input_dir must not be empty.")

        if not str(request.speaker_name).strip():
            raise ValueError("speaker_name must not be empty.")

        if not str(request.language).strip():
            raise ValueError("language must not be empty.")

        if request.enable_uvr:
            raise ValueError(
                "enable_uvr=true is not supported yet. "
                "The underlying build_dataset currently raises NotImplementedError."
            )

        if request.enable_denoise:
            raise ValueError(
                "enable_denoise=true is not supported yet. "
                "The underlying build_dataset currently raises NotImplementedError."
            )

        if request.prompt_mode not in {"self", "speaker_pool", "fixed_reference"}:
            raise ValueError("prompt_mode must be self, speaker_pool, or fixed_reference.")

        if request.min_prompt_sec <= 0:
            raise ValueError("min_prompt_sec must be > 0.")

        if request.max_prompt_sec < request.min_prompt_sec:
            raise ValueError("max_prompt_sec must be >= min_prompt_sec.")

    # ------------------------------------------------------------------
    # Command builders
    # 命令构造
    # ------------------------------------------------------------------
    def build_prepare_command(
        self,
        request: DataFactoryRequest,
        artifacts: DataFactoryArtifactPaths,
    ) -> DataFactoryCommand:
        """Build scripts/prepare_fewshot_dataset.py command."""

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_PREPARE_FEWSHOT_SCRIPT),
            "--raw_input_dir",
            str(self.resolve_path(request.raw_input_dir)),
            "--work_dir",
            str(artifacts.work_dir),
            "--speaker_name",
            str(request.speaker_name),
            "--language",
            str(request.language),
            "--asr_backend",
            str(request.asr_backend),
            "--asr_model_size",
            str(request.asr_model_size),
            "--asr_precision",
            str(request.asr_precision),
            "--enable_uvr",
            _bool_text(request.enable_uvr),
            "--enable_denoise",
            _bool_text(request.enable_denoise),
            "--threshold",
            str(int(request.threshold)),
            "--min_length",
            str(int(request.min_length)),
            "--min_interval",
            str(int(request.min_interval)),
            "--hop_size",
            str(int(request.hop_size)),
            "--max_sil_kept",
            str(int(request.max_sil_kept)),
            "--normalize_max",
            str(float(request.normalize_max)),
            "--alpha_mix",
            str(float(request.alpha_mix)),
            "--export_list",
            _bool_text(request.export_list),
            "--export_stage2_manifest",
            _bool_text(request.export_stage2_manifest),
            "--overwrite_work_dir",
            _bool_text(request.overwrite_work_dir),
            "--dry_run",
            _bool_text(request.dry_run),
            "--report_path",
            str(artifacts.prepare_report_path),
            "--print_next_commands",
            "true",
        ]

        return DataFactoryCommand(
            name="prepare",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=artifacts.prepare_report_path,
            expected_output_paths={
                "raw_audio_index": artifacts.raw_audio_index_path,
                "manifest_jsonl": artifacts.manifest_jsonl_path,
                "dataset_list": artifacts.dataset_list_path,
                "stage2_manifest_jsonl": artifacts.stage2_manifest_jsonl_path,
            },
        )

    def build_manifest_health_check_command(
        self,
        request: DataFactoryRequest,
        artifacts: DataFactoryArtifactPaths,
    ) -> DataFactoryCommand:
        """Build manifest health-check command."""

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_HEALTH_CHECK_SCRIPT),
            "--manifest_jsonl",
            str(artifacts.manifest_jsonl_path),
            "--report_path",
            str(artifacts.manifest_health_report_path),
            "--min_samples_pass",
            str(int(self.config.health.min_samples_pass)),
            "--min_samples_warn",
            str(int(self.config.health.min_samples_warn)),
            "--min_total_duration_sec_pass",
            str(float(self.config.health.min_total_duration_sec_pass)),
            "--min_total_duration_sec_warn",
            str(float(self.config.health.min_total_duration_sec_warn)),
            "--min_clip_sec_fail",
            str(float(self.config.health.min_clip_sec_fail)),
            "--min_clip_sec_warn",
            str(float(self.config.health.min_clip_sec_warn)),
            "--max_clip_sec_warn",
            str(float(self.config.health.max_clip_sec_warn)),
            "--max_clip_sec_fail",
            str(float(self.config.health.max_clip_sec_fail)),
            "--print_samples",
            "true",
            "--exit_nonzero_on_fail",
            "false",
        ]

        return DataFactoryCommand(
            name="manifest_health_check",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=artifacts.manifest_health_report_path,
            expected_output_paths={
                "manifest_health_report": artifacts.manifest_health_report_path,
            },
        )

    def build_export_fewshot_stage2_manifest_command(
        self,
        request: DataFactoryRequest,
        artifacts: DataFactoryArtifactPaths,
    ) -> DataFactoryCommand:
        """Build export_fewshot_stage2_manifest.py command."""

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_EXPORT_FEWSHOT_STAGE2_SCRIPT),
            "--manifest_jsonl",
            str(artifacts.manifest_jsonl_path),
            "--output_stage2_manifest",
            str(artifacts.stage2_manifest_fewshot_path),
            "--prompt_mode",
            str(request.prompt_mode),
            "--min_prompt_sec",
            str(float(request.min_prompt_sec)),
            "--max_prompt_sec",
            str(float(request.max_prompt_sec)),
            "--prefer_prompt_sec",
            str(float(request.prefer_prompt_sec)),
            "--allow_self_prompt",
            _bool_text(request.allow_self_prompt),
            "--prompt_selection_report_path",
            str(artifacts.prompt_selection_report_path),
            "--conversion_report_path",
            str(artifacts.fewshot_manifest_conversion_report_path),
        ]

        return DataFactoryCommand(
            name="export_fewshot_stage2_manifest",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=artifacts.fewshot_manifest_conversion_report_path,
            expected_output_paths={
                "stage2_manifest_fewshot": artifacts.stage2_manifest_fewshot_path,
                "prompt_selection_report": artifacts.prompt_selection_report_path,
                "fewshot_manifest_conversion_report": artifacts.fewshot_manifest_conversion_report_path,
            },
        )

    def build_stage2_manifest_health_check_command(
        self,
        request: DataFactoryRequest,
        artifacts: DataFactoryArtifactPaths,
    ) -> DataFactoryCommand:
        """Build stage2 manifest health-check command."""

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_HEALTH_CHECK_SCRIPT),
            "--stage2_manifest_jsonl",
            str(artifacts.stage2_manifest_fewshot_path),
            "--report_path",
            str(artifacts.stage2_manifest_health_report_path),
            "--min_samples_pass",
            str(int(self.config.health.min_samples_pass)),
            "--min_samples_warn",
            str(int(self.config.health.min_samples_warn)),
            "--min_total_duration_sec_pass",
            str(float(self.config.health.min_total_duration_sec_pass)),
            "--min_total_duration_sec_warn",
            str(float(self.config.health.min_total_duration_sec_warn)),
            "--min_clip_sec_fail",
            str(float(self.config.health.min_clip_sec_fail)),
            "--min_clip_sec_warn",
            str(float(self.config.health.min_clip_sec_warn)),
            "--max_clip_sec_warn",
            str(float(self.config.health.max_clip_sec_warn)),
            "--max_clip_sec_fail",
            str(float(self.config.health.max_clip_sec_fail)),
            "--min_prompt_sec",
            str(float(self.config.health.min_prompt_sec)),
            "--max_prompt_sec",
            str(float(self.config.health.max_prompt_sec)),
            "--print_samples",
            "true",
            "--exit_nonzero_on_fail",
            "false",
        ]

        return DataFactoryCommand(
            name="stage2_manifest_health_check",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=artifacts.stage2_manifest_health_report_path,
            expected_output_paths={
                "stage2_manifest_health_report": artifacts.stage2_manifest_health_report_path,
            },
        )

    def build_command_plan(
        self,
        request: DataFactoryRequest,
    ) -> DataFactoryCommandPlan:
        """Build dry-run command plan for one request.

        中文说明：
            根据 DataFactoryRequest 构造命令计划。

            P10-2 阶段只返回命令，不执行。
        """

        self.validate_request(request)

        work_dir = self.resolve_work_dir(request)
        artifacts = self.expected_artifacts(work_dir)

        commands: list[DataFactoryCommand] = []

        if request.steps.run_prepare:
            commands.append(self.build_prepare_command(request, artifacts))

        if request.steps.run_manifest_health_check:
            commands.append(self.build_manifest_health_check_command(request, artifacts))

        if request.steps.run_export_fewshot_stage2_manifest:
            commands.append(self.build_export_fewshot_stage2_manifest_command(request, artifacts))

        if request.steps.run_stage2_manifest_health_check:
            commands.append(self.build_stage2_manifest_health_check_command(request, artifacts))

        return DataFactoryCommandPlan(
            request=request,
            artifacts=artifacts,
            commands=commands,
            dry_run_only=True,
        )
    def build_launch_proofread_command(
        self,
        artifacts: DataFactoryArtifactPaths,
        *,
        webui_port_subfix: int = 9871,
        is_share: str = "False",
        g_batch: int = 10,
        backup_suffix: str = "before_proofread",
        overwrite_backup: bool = False,
    ) -> DataFactoryCommand:
        """Build command for launching the legacy proofread WebUI.

        中文说明：
            构造启动旧版人工校对器的命令。

            注意：
                这个命令后续应由 service 使用 subprocess.Popen 启动，
                避免阻塞主数据工厂 UI。
        """

        if artifacts.dataset_list_path is None:
            raise ValueError("dataset_list_path is missing in artifacts.")

        expected_report_path = artifacts.work_dir / "06_export" / "proofread_launch_report.json"

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_LAUNCH_PROOFREAD_SCRIPT),
            "--dataset_list",
            str(artifacts.dataset_list_path),
            "--webui_port_subfix",
            str(int(webui_port_subfix)),
            "--is_share",
            str(is_share),
            "--g_batch",
            str(int(g_batch)),
            "--backup_suffix",
            str(backup_suffix),
            "--overwrite_backup",
            _bool_text(overwrite_backup),
        ]

        return DataFactoryCommand(
            name="launch_proofread",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=expected_report_path,
            expected_output_paths={
                "proofread_launch_report": expected_report_path,
                "dataset_list": artifacts.dataset_list_path,
                "manifest_before_proofread": artifacts.manifest_before_proofread_path,
                "dataset_before_proofread": artifacts.dataset_before_proofread_path,
                "stage2_manifest_before_proofread": artifacts.stage2_manifest_before_proofread_path,
            },
        )

    def build_rebuild_corrected_manifest_command(
        self,
        artifacts: DataFactoryArtifactPaths,
        *,
        duration_tolerance_sec: float = 0.05,
        prompt_mode: str = "self",
    ) -> DataFactoryCommand:
        """Build command for rebuilding corrected manifest files.

        中文说明：
            构造“从人工校对后的 dataset.list 反向重建 corrected manifest”的命令。
        """

        export_dir = artifacts.work_dir / "06_export"

        original_manifest = artifacts.manifest_jsonl_path
        corrected_list = artifacts.dataset_list_path
        original_list = artifacts.dataset_before_proofread_path
        output_manifest = artifacts.manifest_corrected_path
        output_stage2_manifest = artifacts.stage2_manifest_corrected_path
        output_report = artifacts.correction_report_path or (export_dir / "correction_report.json")

        required = {
            "original_manifest": original_manifest,
            "corrected_list": corrected_list,
            "output_manifest": output_manifest,
            "output_stage2_manifest": output_stage2_manifest,
        }
        missing_keys = [k for k, v in required.items() if v is None]
        if missing_keys:
            raise ValueError(f"Missing artifact paths: {missing_keys}")

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_REBUILD_CORRECTED_MANIFEST_SCRIPT),
            "--original_manifest",
            str(original_manifest),
            "--corrected_list",
            str(corrected_list),
            "--output_manifest",
            str(output_manifest),
            "--output_stage2_manifest",
            str(output_stage2_manifest),
            "--output_report",
            str(output_report),
            "--duration_tolerance_sec",
            str(float(duration_tolerance_sec)),
            "--prompt_mode",
            str(prompt_mode),
        ]

        if original_list is not None:
            argv.extend(
                [
                    "--original_list",
                    str(original_list),
                ]
            )

        return DataFactoryCommand(
            name="rebuild_corrected_manifest",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=output_report,
            expected_output_paths={
                "manifest_corrected": output_manifest,
                "stage2_manifest_corrected": output_stage2_manifest,
                "correction_report": output_report,
            },
        )

    def build_export_corrected_fewshot_stage2_manifest_command(
        self,
        artifacts: DataFactoryArtifactPaths,
        *,
        prompt_mode: str = "speaker_pool",
        min_prompt_sec: float = 3.0,
        max_prompt_sec: float = 10.0,
        prefer_prompt_sec: float = 6.0,
        allow_self_prompt: bool = True,
        fixed_prompt_wav_path: Optional[PathLike] = None,
        strict_prompt_duration: bool = True,
        skip_invalid_rows: bool = False,
        allow_empty_text: bool = False,
    ) -> DataFactoryCommand:
        """Build command for exporting few-shot Stage2 manifest from corrected manifest.

        中文说明：
            从 manifest.corrected.jsonl 导出最终用于 Stage2 预处理的：

                stage2_manifest.fewshot.jsonl

            这是 P10-7 的核心命令。
        """

        input_manifest = artifacts.manifest_corrected_path
        output_stage2_manifest = artifacts.stage2_manifest_fewshot_path

        export_dir = artifacts.work_dir / "06_export"
        prompt_selection_report = artifacts.prompt_selection_report_path or (
            export_dir / "prompt_selection_report.json"
        )
        conversion_report = artifacts.fewshot_manifest_conversion_report_path or (
            export_dir / "fewshot_manifest_conversion_report.json"
        )

        required = {
            "manifest_corrected": input_manifest,
            "stage2_manifest_fewshot": output_stage2_manifest,
        }
        missing_keys = [k for k, v in required.items() if v is None]
        if missing_keys:
            raise ValueError(f"Missing artifact paths: {missing_keys}")

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_EXPORT_FEWSHOT_STAGE2_SCRIPT),
            "--manifest_jsonl",
            str(input_manifest),
            "--output_stage2_manifest",
            str(output_stage2_manifest),
            "--prompt_mode",
            str(prompt_mode),
            "--min_prompt_sec",
            str(float(min_prompt_sec)),
            "--max_prompt_sec",
            str(float(max_prompt_sec)),
            "--prefer_prompt_sec",
            str(float(prefer_prompt_sec)),
            "--allow_self_prompt",
            _bool_text(allow_self_prompt),
            "--strict_prompt_duration",
            _bool_text(strict_prompt_duration),
            "--skip_invalid_rows",
            _bool_text(skip_invalid_rows),
            "--allow_empty_text",
            _bool_text(allow_empty_text),
            "--prompt_selection_report_path",
            str(prompt_selection_report),
            "--conversion_report_path",
            str(conversion_report),
        ]

        if fixed_prompt_wav_path is not None and str(fixed_prompt_wav_path).strip():
            argv.extend(
                [
                    "--fixed_prompt_wav_path",
                    str(self.resolve_path(fixed_prompt_wav_path)),
                ]
            )

        return DataFactoryCommand(
            name="export_corrected_fewshot_stage2_manifest",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=conversion_report,
            expected_output_paths={
                "manifest_corrected": input_manifest,
                "stage2_manifest_fewshot": output_stage2_manifest,
                "prompt_selection_report": prompt_selection_report,
                "fewshot_manifest_conversion_report": conversion_report,
            },
        )

    def build_preprocess_stage2_pt_command(
        self,
        artifacts: DataFactoryArtifactPaths,
        *,
        output_dir: Optional[PathLike] = None,
        device: str = "cuda",
        use_half: bool = False,
        overwrite: bool = False,
        target_sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
    ) -> DataFactoryCommand:
        """Build command for Stage2 .pt offline preprocessing.

        中文说明：
            构造 Stage2 Flow-Matching `.pt` 离线预处理命令。

            输入：
                stage2_manifest.fewshot.jsonl

            输出：
                stage2_pt_root/*.pt
        """

        manifest = artifacts.stage2_manifest_fewshot_path
        resolved_output_dir = (
            self.resolve_path(output_dir)
            if output_dir is not None and str(output_dir).strip()
            else artifacts.stage2_pt_root
        )

        if manifest is None:
            raise ValueError("stage2_manifest_fewshot_path is missing in artifacts.")

        if resolved_output_dir is None:
            raise ValueError("stage2_pt_root/output_dir is missing in artifacts.")

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_PREPROCESS_STAGE2_FM_SCRIPT),
            "--manifest",
            str(manifest),
            "--output_dir",
            str(resolved_output_dir),
            "--device",
            str(device),
            "--target_sr",
            str(int(target_sr)),
            "--n_fft",
            str(int(n_fft)),
            "--hop_length",
            str(int(hop_length)),
            "--win_length",
            str(int(win_length)),
            "--n_mels",
            str(int(n_mels)),
            "--fmin",
            str(float(fmin)),
            "--fmax",
            str(float(fmax)),
        ]

        if use_half:
            argv.append("--use_half")

        if overwrite:
            argv.append("--overwrite")

        return DataFactoryCommand(
            name="preprocess_stage2_pt",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=None,
            expected_output_paths={
                "stage2_manifest_fewshot": manifest,
                "stage2_pt_root": resolved_output_dir,
            },
        )

    def build_export_continuous_semantic_cache_command(
        self,
        artifacts: DataFactoryArtifactPaths,
        *,
        input_root: Optional[PathLike] = None,
        output_root: Optional[PathLike] = None,
        sovits_checkpoint_path: Optional[PathLike] = None,
        device: str = "cuda",
        use_half: bool = False,
        continuous_dtype: str = "float32",
        require_predicted: bool = False,
        overwrite: bool = False,
        max_files: Optional[int] = None,
        copy_non_pt_files: bool = False,
    ) -> DataFactoryCommand:
        """Build command for continuous semantic cache export."""

        resolved_input_root = (
            self.resolve_path(input_root)
            if input_root is not None and str(input_root).strip()
            else artifacts.stage2_pt_root
        )
        resolved_output_root = (
            self.resolve_path(output_root)
            if output_root is not None and str(output_root).strip()
            else artifacts.continuous_semantic_root
        )

        if resolved_input_root is None:
            raise ValueError("stage2_pt_root/input_root is missing in artifacts.")
        if resolved_output_root is None:
            raise ValueError("continuous_semantic_root/output_root is missing in artifacts.")

        report_json = resolved_output_root / "continuous_semantic_export_report.json"

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_CONTINUOUS_SEMANTIC_CACHE_SCRIPT),
            "--input_root",
            str(resolved_input_root),
            "--output_root",
            str(resolved_output_root),
            "--device",
            str(device),
            "--continuous_dtype",
            str(continuous_dtype),
            "--report_json",
            str(report_json),
        ]

        if sovits_checkpoint_path is not None and str(sovits_checkpoint_path).strip():
            argv.extend(
                [
                    "--sovits_checkpoint_path",
                    str(self.resolve_path(sovits_checkpoint_path)),
                ]
            )

        if use_half:
            argv.append("--use_half")

        if require_predicted:
            argv.append("--require_predicted")

        if overwrite:
            argv.append("--overwrite")

        if max_files is not None:
            argv.extend(["--max_files", str(int(max_files))])

        if copy_non_pt_files:
            argv.append("--copy_non_pt_files")

        return DataFactoryCommand(
            name="export_continuous_semantic_cache",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=report_json,
            expected_output_paths={
                "stage2_pt_root": resolved_input_root,
                "continuous_semantic_root": resolved_output_root,
                "continuous_semantic_report": report_json,
            },
        )

    def build_export_fewshot_style_cache_command(
        self,
        artifacts: DataFactoryArtifactPaths,
        *,
        input_root: Optional[PathLike] = None,
        output_root: Optional[PathLike] = None,
        device: str = "cuda",
        overwrite: bool = False,
        copy_non_pt_files: bool = False,
        recursive: bool = False,
        max_files: Optional[int] = None,
        extract_prompt_acoustic: bool = True,
        extract_energy: bool = True,
        extract_f0: bool = True,
        extract_speaker_embedding: bool = False,
        prefer_sample_acoustic_meta: bool = True,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        f0_min_hz: float = 50.0,
        f0_max_hz: float = 1100.0,
        speaker_backend: str = "speechbrain_ecapa",
        speaker_model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        speaker_savedir: PathLike = "pretrained_models/speechbrain_spkrec_ecapa_voxceleb",
        speaker_device: Optional[str] = None,
        speaker_sample_rate: int = 16000,
        speaker_normalize: bool = True,
        fail_on_error: bool = True,
    ) -> DataFactoryCommand:
        """Build command for style/F0/speaker cache export."""

        resolved_input_root = (
            self.resolve_path(input_root)
            if input_root is not None and str(input_root).strip()
            else artifacts.continuous_semantic_root or artifacts.stage2_pt_root
        )
        resolved_output_root = (
            self.resolve_path(output_root)
            if output_root is not None and str(output_root).strip()
            else artifacts.style_cache_root
        )

        if resolved_input_root is None:
            raise ValueError("style cache input_root is missing.")
        if resolved_output_root is None:
            raise ValueError("style_cache_root/output_root is missing.")

        report_json = resolved_output_root / "style_cache_report.json"

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_FEWSHOT_STYLE_CACHE_SCRIPT),
            "--input_root",
            str(resolved_input_root),
            "--output_root",
            str(resolved_output_root),
            "--report_json",
            str(report_json),
            "--device",
            str(device),
            "--overwrite",
            _bool_text(overwrite),
            "--copy_non_pt_files",
            _bool_text(copy_non_pt_files),
            "--recursive",
            _bool_text(recursive),
            "--extract_prompt_acoustic",
            _bool_text(extract_prompt_acoustic),
            "--extract_energy",
            _bool_text(extract_energy),
            "--extract_f0",
            _bool_text(extract_f0),
            "--extract_speaker_embedding",
            _bool_text(extract_speaker_embedding),
            "--prefer_sample_acoustic_meta",
            _bool_text(prefer_sample_acoustic_meta),
            "--sample_rate",
            str(int(sample_rate)),
            "--n_fft",
            str(int(n_fft)),
            "--hop_length",
            str(int(hop_length)),
            "--win_length",
            str(int(win_length)),
            "--n_mels",
            str(int(n_mels)),
            "--fmin",
            str(float(fmin)),
            "--fmax",
            str(float(fmax)),
            "--f0_min_hz",
            str(float(f0_min_hz)),
            "--f0_max_hz",
            str(float(f0_max_hz)),
            "--speaker_backend",
            str(speaker_backend),
            "--speaker_model_source",
            str(speaker_model_source),
            "--speaker_savedir",
            str(self.resolve_path(speaker_savedir)),
            "--speaker_sample_rate",
            str(int(speaker_sample_rate)),
            "--speaker_normalize",
            _bool_text(speaker_normalize),
            "--fail_on_error",
            _bool_text(fail_on_error),
        ]

        if speaker_device is not None and str(speaker_device).strip():
            argv.extend(["--speaker_device", str(speaker_device)])

        if max_files is not None:
            argv.extend(["--max_files", str(int(max_files))])

        return DataFactoryCommand(
            name="export_fewshot_style_cache",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=report_json,
            expected_output_paths={
                "style_cache_input_root": resolved_input_root,
                "style_cache_root": resolved_output_root,
                "style_cache_report": report_json,
            },
        )

    def build_split_stage2_pt_dataset_command(
        self,
        artifacts: DataFactoryArtifactPaths,
        *,
        data_root: Optional[PathLike] = None,
        train_out: Optional[PathLike] = None,
        val_out: Optional[PathLike] = None,
        train_ratio: Optional[float] = 0.9,
        val_count: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 2026,
        mode: str = "copy",
        overwrite: bool = False,
        report_path: Optional[PathLike] = None,
    ) -> DataFactoryCommand:
        """Build command for train/val split of final Stage2 .pt dataset."""

        resolved_data_root = (
            self.resolve_path(data_root)
            if data_root is not None and str(data_root).strip()
            else artifacts.style_cache_root
        )
        resolved_train_out = (
            self.resolve_path(train_out)
            if train_out is not None and str(train_out).strip()
            else artifacts.train_pt_root
        )
        resolved_val_out = (
            self.resolve_path(val_out)
            if val_out is not None and str(val_out).strip()
            else artifacts.val_pt_root
        )
        resolved_report_path = (
            self.resolve_path(report_path)
            if report_path is not None and str(report_path).strip()
            else artifacts.split_report_path
        )

        if resolved_data_root is None:
            raise ValueError("split data_root is missing.")
        if resolved_train_out is None:
            raise ValueError("split train_out is missing.")
        if resolved_val_out is None:
            raise ValueError("split val_out is missing.")
        if resolved_report_path is None:
            raise ValueError("split report_path is missing.")

        if mode not in {"copy", "move", "hardlink"}:
            raise ValueError("split mode must be copy, move, or hardlink.")

        if (train_ratio is None) == (val_count is None):
            raise ValueError("Exactly one of train_ratio or val_count must be provided.")

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_SPLIT_STAGE2_PT_SCRIPT),
            "--data_root",
            str(resolved_data_root),
            "--train_out",
            str(resolved_train_out),
            "--val_out",
            str(resolved_val_out),
            "--shuffle",
            _bool_text(shuffle),
            "--seed",
            str(int(seed)),
            "--mode",
            str(mode),
            "--overwrite",
            _bool_text(overwrite),
            "--report_path",
            str(resolved_report_path),
        ]

        if train_ratio is not None:
            argv.extend(["--train_ratio", str(float(train_ratio))])

        if val_count is not None:
            argv.extend(["--val_count", str(int(val_count))])

        return DataFactoryCommand(
            name="split_stage2_pt_dataset",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=resolved_report_path,
            expected_output_paths={
                "data_root": resolved_data_root,
                "train_pt_root": resolved_train_out,
                "val_pt_root": resolved_val_out,
                "split_report": resolved_report_path,
            },
        )

    def build_filter_v662_bad_style_samples_command(
        self,
        artifacts: DataFactoryArtifactPaths,
        *,
        roots: Optional[list[PathLike]] = None,
        rejected_dir: Optional[PathLike] = None,
        recursive: bool = False,
        mode: str = "move",
        require_target_f0: bool = True,
        require_target_voiced_mask: bool = True,
        reject_all_unvoiced_target_f0: bool = True,
        require_prompt_f0: bool = False,
        check_speaker_fields: bool = True,
        speaker_dim: int = 192,
        report_json: Optional[PathLike] = None,
        report_csv: Optional[PathLike] = None,
    ) -> DataFactoryCommand:
        """Build command for filtering bad v6.6.2 style-cache samples."""

        if roots:
            resolved_roots = [self.resolve_path(x) for x in roots if str(x).strip()]
        else:
            resolved_roots = [
                p for p in [artifacts.train_pt_root, artifacts.val_pt_root] if p is not None
            ]

        if not resolved_roots:
            raise ValueError("filter roots are missing.")

        resolved_rejected_dir = (
            self.resolve_path(rejected_dir)
            if rejected_dir is not None and str(rejected_dir).strip()
            else artifacts.rejected_pt_root
        )
        resolved_report_json = (
            self.resolve_path(report_json)
            if report_json is not None and str(report_json).strip()
            else artifacts.filter_report_json_path
        )
        resolved_report_csv = (
            self.resolve_path(report_csv)
            if report_csv is not None and str(report_csv).strip()
            else artifacts.filter_report_csv_path
        )

        if resolved_rejected_dir is None:
            raise ValueError("filter rejected_dir is missing.")
        if resolved_report_json is None:
            raise ValueError("filter report_json is missing.")
        if resolved_report_csv is None:
            raise ValueError("filter report_csv is missing.")

        if mode not in {"move", "copy", "delete", "report_only"}:
            raise ValueError("filter mode must be move, copy, delete, or report_only.")

        argv = [
            self.python_executable,
            str(self.project_root / DEFAULT_FILTER_V662_BAD_STYLE_SAMPLES_SCRIPT),
            "--roots",
        ]
        argv.extend(str(p) for p in resolved_roots)

        argv.extend(
            [
                "--rejected_dir",
                str(resolved_rejected_dir),
                "--recursive",
                _bool_text(recursive),
                "--mode",
                str(mode),
                "--require_target_f0",
                _bool_text(require_target_f0),
                "--require_target_voiced_mask",
                _bool_text(require_target_voiced_mask),
                "--reject_all_unvoiced_target_f0",
                _bool_text(reject_all_unvoiced_target_f0),
                "--require_prompt_f0",
                _bool_text(require_prompt_f0),
                "--check_speaker_fields",
                _bool_text(check_speaker_fields),
                "--speaker_dim",
                str(int(speaker_dim)),
                "--report_json",
                str(resolved_report_json),
                "--report_csv",
                str(resolved_report_csv),
            ]
        )

        return DataFactoryCommand(
            name="filter_v662_bad_style_samples",
            argv=argv,
            cwd=self.project_root,
            expected_report_path=resolved_report_json,
            expected_output_paths={
                "roots": resolved_roots[0],
                "roots_list": resolved_roots,
                "rejected_dir": resolved_rejected_dir,
                "filter_report_json": resolved_report_json,
                "filter_report_csv": resolved_report_csv,
            },
        )


def _suggest_stage2_pt_root_from_work_dir(work_dir: Path) -> Path:
    """Suggest default Stage2 .pt output root from data factory work_dir.

    中文说明：
        根据数据工厂 work_dir 推断默认 Stage2 .pt 输出目录。

        例如：
            user_data/{speaker_name}_factory
        ->  user_data/{speaker_name}_stage2_pt_all
    """

    work_dir = Path(work_dir).resolve(strict=False)
    name = work_dir.name

    if name.endswith("_factory"):
        speaker_prefix = name[: -len("_factory")]
        if speaker_prefix:
            return work_dir.parent / f"{speaker_prefix}_stage2_pt_all"

    return work_dir / "07_stage2_pt"


def _suggest_continuous_semantic_root_from_stage2_pt_root(stage2_pt_root: Path) -> Path:
    """Suggest continuous semantic cache root from Stage2 .pt root.

    示例：
        user_data/{speaker_name}_stage2_pt_all
    ->  user_data/{speaker_name}_stage2_pt_all_continuous
    """

    stage2_pt_root = Path(stage2_pt_root).resolve(strict=False)
    return stage2_pt_root.parent / f"{stage2_pt_root.name}_continuous"


def _suggest_style_cache_root_from_stage2_pt_root(stage2_pt_root: Path) -> Path:
    """Suggest style/F0/speaker cache root from Stage2 .pt root.

    示例：
        user_data/{speaker_name}_stage2_pt_all
    ->  user_data/{speaker_name}_stage2_pt_all_v662_style_f0_spk
    """

    stage2_pt_root = Path(stage2_pt_root).resolve(strict=False)
    return stage2_pt_root.parent / f"{stage2_pt_root.name}_v662_style_f0_spk"


def _suggest_split_roots_from_style_cache_root(style_cache_root: Path) -> tuple[Path, Path, Path]:
    """Suggest train/val split roots from style cache root.

    示例：
        user_data/{speaker_name}_stage2_pt_all_v662_style_f0_spk
    ->  user_data/{speaker_name}_stage2_pt_v662_style_f0_spk/train
        user_data/{speaker_name}_stage2_pt_v662_style_f0_spk/val
        user_data/{speaker_name}_stage2_pt_v662_style_f0_spk/split_report.json
    """

    style_cache_root = Path(style_cache_root).resolve(strict=False)
    name = style_cache_root.name

    split_name = name.replace("_stage2_pt_all_", "_stage2_pt_")
    if split_name == name and name.endswith("_all"):
        split_name = name[: -len("_all")]

    split_root = style_cache_root.parent / split_name
    train_root = split_root / "train"
    val_root = split_root / "val"
    split_report = split_root / "split_report.json"
    return train_root, val_root, split_report


def _suggest_filter_outputs_from_split_parent(split_parent: Path) -> tuple[Path, Path, Path]:
    """Suggest rejected dir and filter reports from split output parent."""

    split_parent = Path(split_parent).resolve(strict=False)
    rejected_root = split_parent.parent / f"{split_parent.name}_rejected"
    report_json = split_parent.parent / f"filter_{split_parent.name}_bad_style_samples_report.json"
    report_csv = split_parent.parent / f"filter_{split_parent.name}_bad_style_samples_bad.csv"
    return rejected_root, report_json, report_csv


def _bool_text(value: bool) -> str:
    """Return lowercase CLI-friendly bool string."""

    return "true" if bool(value) else "false"


def create_data_factory_adapter(
    *,
    project_root: Optional[PathLike] = None,
    python_executable: Optional[PathLike] = None,
    config: Optional[DataFactoryConfig] = None,
) -> DataFactoryAdapter:
    """Factory helper for future service / WebUI."""

    return DataFactoryAdapter(
        project_root=project_root,
        python_executable=python_executable,
        config=config,
    )


__all__ = [
    "DEFAULT_PREPARE_FEWSHOT_SCRIPT",
    "DEFAULT_HEALTH_CHECK_SCRIPT",
    "DEFAULT_EXPORT_FEWSHOT_STAGE2_SCRIPT",
    "DataFactoryCommand",
    "DataFactoryCommandPlan",
    "DataFactoryAdapter",
    "create_data_factory_adapter",
    "DEFAULT_LAUNCH_PROOFREAD_SCRIPT",
    "DEFAULT_REBUILD_CORRECTED_MANIFEST_SCRIPT",
    "DEFAULT_PREPROCESS_STAGE2_FM_SCRIPT",
    "DEFAULT_CONTINUOUS_SEMANTIC_CACHE_SCRIPT",
    "DEFAULT_FEWSHOT_STYLE_CACHE_SCRIPT",
    "DEFAULT_SPLIT_STAGE2_PT_SCRIPT",
    "DEFAULT_FILTER_V662_BAD_STYLE_SAMPLES_SCRIPT",
]