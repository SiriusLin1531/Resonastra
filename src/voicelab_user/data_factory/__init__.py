"""
User-edition data factory interfaces for VoiceLab.

中文说明：
    这里是 VoiceLab 用户版数据工厂的轻量接口层。
"""

from .artifact_layout import (
    COMPACT_ARTIFACT_LAYOUT_VERSION,
    LEGACY_MIGRATION_REPORT_FILENAME,
    artifact_layout_debug_summary,
    compact_artifacts_from_work_dir,
    install_compact_artifact_layout,
    legacy_artifacts_from_work_dir,
    legacy_sibling_artifacts_from_work_dir,
    migrate_legacy_artifacts,
    scan_legacy_artifacts,
)

# P10-11-9:
# New user runs use compact work_dir-contained artifact layout.
install_compact_artifact_layout()

from .request import (
    DataFactoryRequest,
    DataFactoryStepSelection,
)

from .result import (
    DATA_FACTORY_RESULT_SCHEMA_VERSION,
    DATA_FACTORY_STATUS_FAILED,
    DATA_FACTORY_STATUS_PARTIAL,
    DATA_FACTORY_STATUS_PREPARED,
    DATA_FACTORY_STATUS_RUNNING,
    DATA_FACTORY_STATUS_SUCCEEDED,
    DataFactoryArtifactPaths,
    DataFactoryErrorInfo,
    DataFactoryResult,
    DataFactoryStepResult,
    DataFactoryTiming,
    make_prepared_data_factory_result,
    result_to_dict,
    write_data_factory_result_json,
)

from .config import (
    DATA_FACTORY_CONFIG_SCHEMA_VERSION,
    DEFAULT_DATA_FACTORY_CONFIG_PATH,
    DataFactoryASRConfig,
    DataFactoryConfig,
    DataFactoryConfigError,
    DataFactoryDatasetConfig,
    DataFactoryExportConfig,
    DataFactoryHealthConfig,
    DataFactoryPathConfig,
    DataFactoryPreprocessConfig,
    DataFactoryRuntimeConfig,
    DataFactorySliceConfig,
    build_data_factory_config,
    load_data_factory_config,
)

from .adapter import (
    DEFAULT_EXPORT_FEWSHOT_STAGE2_SCRIPT,
    DEFAULT_HEALTH_CHECK_SCRIPT,
    DEFAULT_PREPARE_FEWSHOT_SCRIPT,
    DataFactoryAdapter,
    DataFactoryCommand,
    DataFactoryCommandPlan,
    create_data_factory_adapter,
)

from .service import (
    DataFactoryService,
    DataFactoryExecutionOptions,
    create_data_factory_service,
    request_to_dict,
)

from .user_pipeline import (
    USER_PIPELINE_SCHEMA_VERSION,
    DataFactoryUserPipelineService,
    UserPipelineResult,
    UserPipelineStepResult,
    create_data_factory_user_pipeline_service,
    user_pipeline_result_to_dict,
    write_user_pipeline_result_json,
)

from .resume_pipeline import (
    run_resumable_postprocess_pipeline,
)

__all__ = [
    "COMPACT_ARTIFACT_LAYOUT_VERSION",
    "LEGACY_MIGRATION_REPORT_FILENAME",
    "artifact_layout_debug_summary",
    "compact_artifacts_from_work_dir",
    "install_compact_artifact_layout",
    "legacy_artifacts_from_work_dir",
    "legacy_sibling_artifacts_from_work_dir",
    "scan_legacy_artifacts",
    "migrate_legacy_artifacts",
    "DataFactoryRequest",
    "DataFactoryStepSelection",
    "DATA_FACTORY_RESULT_SCHEMA_VERSION",
    "DATA_FACTORY_STATUS_FAILED",
    "DATA_FACTORY_STATUS_PARTIAL",
    "DATA_FACTORY_STATUS_PREPARED",
    "DATA_FACTORY_STATUS_RUNNING",
    "DATA_FACTORY_STATUS_SUCCEEDED",
    "DataFactoryArtifactPaths",
    "DataFactoryErrorInfo",
    "DataFactoryResult",
    "DataFactoryStepResult",
    "DataFactoryTiming",
    "make_prepared_data_factory_result",
    "result_to_dict",
    "write_data_factory_result_json",
    "DATA_FACTORY_CONFIG_SCHEMA_VERSION",
    "DEFAULT_DATA_FACTORY_CONFIG_PATH",
    "DataFactoryASRConfig",
    "DataFactoryConfig",
    "DataFactoryConfigError",
    "DataFactoryDatasetConfig",
    "DataFactoryExportConfig",
    "DataFactoryHealthConfig",
    "DataFactoryPathConfig",
    "DataFactoryPreprocessConfig",
    "DataFactoryRuntimeConfig",
    "DataFactorySliceConfig",
    "build_data_factory_config",
    "load_data_factory_config",
    "DEFAULT_EXPORT_FEWSHOT_STAGE2_SCRIPT",
    "DEFAULT_HEALTH_CHECK_SCRIPT",
    "DEFAULT_PREPARE_FEWSHOT_SCRIPT",
    "DataFactoryAdapter",
    "DataFactoryCommand",
    "DataFactoryCommandPlan",
    "create_data_factory_adapter",
    "DataFactoryService",
    "DataFactoryExecutionOptions",
    "create_data_factory_service",
    "request_to_dict",
    "USER_PIPELINE_SCHEMA_VERSION",
    "DataFactoryUserPipelineService",
    "UserPipelineResult",
    "UserPipelineStepResult",
    "create_data_factory_user_pipeline_service",
    "user_pipeline_result_to_dict",
    "write_user_pipeline_result_json",
    "run_resumable_postprocess_pipeline",
]
