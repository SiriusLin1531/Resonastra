from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DATA_FACTORY_REQUEST_SCHEMA_VERSION = "voicelab_data_factory_request_v1"


@dataclass(frozen=True)
class DataFactoryStepSelection:
    """Which data-factory steps should be executed.

    中文说明：
        控制本次数据工厂要执行哪些步骤。

        P10-1 只定义字段，不执行。
        P10-2 / P10-3 中 service 和 adapter 会真正使用这些开关。
    """

    run_prepare: bool = True
    run_manifest_health_check: bool = True
    run_export_fewshot_stage2_manifest: bool = True
    run_stage2_manifest_health_check: bool = True

    # Heavy steps. Keep disabled in the first WebUI version.
    # 重型步骤，第一版 UI 默认不启用。
    run_stage2_pt_preprocess: bool = False
    run_continuous_semantic_cache: bool = False
    run_style_cache: bool = False
    run_split_train_val: bool = False
    run_filter_bad_samples: bool = False


@dataclass(frozen=True)
class DataFactoryRequest:
    """User-edition data factory request.

    中文说明：
        用户版数据工厂请求。

        它对应未来 Data Factory WebUI 的输入，也对应当前命令链中的：
            scripts/prepare_fewshot_dataset.py
            scripts/check_fewshot_dataset_health.py
            scripts/export_fewshot_stage2_manifest.py

        注意：
            这里不负责执行，只描述“用户想做什么”。
    """

    schema_version: str = DATA_FACTORY_REQUEST_SCHEMA_VERSION

    # ------------------------------------------------------------------
    # Required user inputs
    # 用户必填输入
    # ------------------------------------------------------------------
    raw_input_dir: Path | str = ""
    speaker_name: str = ""
    language: str = "zh"

    # If None / empty, service may suggest:
    #   user_data/{speaker_name}_factory
    #
    # 如果为空，service 后续可自动建议：
    #   user_data/{speaker_name}_factory
    work_dir: Optional[Path | str] = None

    # ------------------------------------------------------------------
    # ASR settings
    # ASR 设置
    # ------------------------------------------------------------------
    asr_backend: str = "auto"
    asr_model_size: str = "large-v3"
    asr_precision: str = "float32"

    # ------------------------------------------------------------------
    # Optional preprocess flags
    # 可选预处理开关
    # ------------------------------------------------------------------
    enable_uvr: bool = False
    enable_denoise: bool = False

    # ------------------------------------------------------------------
    # Slicing settings
    # 切分参数
    # ------------------------------------------------------------------
    threshold: int = -34
    min_length: int = 4000
    min_interval: int = 300
    hop_size: int = 10
    max_sil_kept: int = 500
    normalize_max: float = 0.9
    alpha_mix: float = 0.25

    # ------------------------------------------------------------------
    # Export settings
    # 导出设置
    # ------------------------------------------------------------------
    export_list: bool = True
    export_stage2_manifest: bool = True

    # Used by export_fewshot_stage2_manifest.py.
    # 用于 few-shot stage2 manifest 导出。
    prompt_mode: str = "speaker_pool"
    min_prompt_sec: float = 3.0
    max_prompt_sec: float = 10.0
    prefer_prompt_sec: float = 6.0
    allow_self_prompt: bool = True

    # ------------------------------------------------------------------
    # Runtime behavior
    # 运行行为
    # ------------------------------------------------------------------
    overwrite_work_dir: bool = False
    dry_run: bool = False
    timeout_seconds: Optional[float] = 7200

    steps: DataFactoryStepSelection = field(default_factory=DataFactoryStepSelection)