from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import os
import sys
from pathlib import Path
from dataclasses import dataclass


# ---------------------------------------------------------------------
# Project-wide constants
# 项目级常量
# ---------------------------------------------------------------------

# EN:
# Compatibility layer phase 1 is hard-locked to GPT-SoVITS v2.
#
# ZH:
# 兼容层第一期被“强锁定”为 GPT-SoVITS v2。
SUPPORTED_GSV_VERSION = "v2"

# EN:
# Default asset locations under the new GameVoiceLab project.
#
# ZH:
# 在新的 GameVoiceLab 项目中约定的默认资源路径。
DEFAULT_COMPAT_RELPATH = "third_party/gsv_compat"
DEFAULT_GSV_ASSET_RELPATH = "GPT_SoVITS"
DEFAULT_BERT_RELPATH = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
DEFAULT_CNHUBERT_RELPATH = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
DEFAULT_G2PW_RELPATH = "GPT_SoVITS/text/G2PWModel"

# EN:
# We explicitly lock the default stage-1 and prompt-tokenizer checkpoints to v2.
#
# ZH:
# 我们明确把默认的第一阶段权重和 prompt tokenizer 权重都锁定为 v2。
DEFAULT_V2_STAGE1_CKPT_RELPATH = (
    "GPT_SoVITS/pretrained_models/"
    "gsv-v2final-pretrained/"
    "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
)

DEFAULT_V2_SOVITS_CKPT_RELPATH = (
    "GPT_SoVITS/pretrained_models/"
    "gsv-v2final-pretrained/"
    "s2G2333k.pth"
)

DEFAULT_FAST_LANGDETECT_RELPATH = "GPT_SoVITS/pretrained_models/fast_langdetect"
DEFAULT_SPLIT_LANG_RELPATH = "third_party/gsv_compat/split_lang"


# ---------------------------------------------------------------------
# GSVEnvInfo
#
# EN:
# A structured container holding all resolved compatibility-layer paths.
#
# ZH:
# 一个结构化的数据容器，用于保存兼容层相关的所有解析后路径。
# ---------------------------------------------------------------------
@dataclass
class GSVEnvInfo:
    # Root path of the GameVoiceLab project
    # GameVoiceLab 项目根目录
    project_root: Path

    # Root path of copied GPT-SoVITS compatibility code
    # 复制过来的 GPT-SoVITS 兼容层代码根目录
    compat_root: Path

    # Root path of GPT_SoVITS-style assets
    # GPT_SoVITS 风格资产根目录
    gsv_asset_root: Path

    # Pretrained model directory
    # 预训练模型目录
    pretrained_root: Path

    # G2PW model directory
    # G2PW 模型目录
    g2pw_root: Path

    # Locked compatibility version
    # 被锁定的兼容层版本
    version: str

    # Resolved BERT path
    # 解析后的 BERT 路径
    bert_path: str

    # Resolved cnhubert path
    # 解析后的 cnhubert 路径
    cnhubert_base_path: str

    # Default v2 stage-1 checkpoint
    # 默认 v2 第一阶段 checkpoint
    default_stage1_ckpt: str

    # Default v2 SoVITS checkpoint used by prompt tokenizer
    # prompt tokenizer 默认使用的 v2 SoVITS checkpoint
    default_sovits_ckpt: str

    # fast_langdetect cache path
    # fast_langdetect 缓存目录
    fast_langdetect_path: str

    # split_lang path
    # split_lang 目录路径
    split_lang_path: str


def _append_sys_path(path: Path) -> None:
    """
    EN:
    Add a path to sys.path if it is not already there.

    ZH:
    如果某个路径还没有加入 sys.path，就把它加入进去。
    """
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def get_project_root() -> Path:
    """
    EN:
    Infer the project root from the current file path.

    Expected layout:
        GameVoiceLab/src/adapters/gsv_env.py

    Then:
        parents[2] -> GameVoiceLab/

    ZH:
    根据当前文件路径自动推断项目根目录。

    默认假设当前文件位于：
        GameVoiceLab/src/adapters/gsv_env.py

    因此：
        parents[2] -> GameVoiceLab/
    """
    return Path(__file__).resolve().parents[2]


def _validate_version(version: str) -> None:
    """
    EN:
    Compatibility layer phase 1 is strictly v2-only.

    ZH:
    兼容层第一期严格只支持 v2。
    """
    if version != SUPPORTED_GSV_VERSION:
        raise ValueError(
            f"Compatibility layer is locked to {SUPPORTED_GSV_VERSION!r} in phase 1, "
            f"but got version={version!r}."
        )


def setup_gsv_env(
    version: str = SUPPORTED_GSV_VERSION,
    project_root: str | Path | None = None,
    compat_relpath: str = DEFAULT_COMPAT_RELPATH,
    gsv_asset_relpath: str = DEFAULT_GSV_ASSET_RELPATH,
    bert_relpath: str = DEFAULT_BERT_RELPATH,
    cnhubert_relpath: str = DEFAULT_CNHUBERT_RELPATH,
    g2pw_relpath: str = DEFAULT_G2PW_RELPATH,
    stage1_ckpt_relpath: str = DEFAULT_V2_STAGE1_CKPT_RELPATH,
    sovits_ckpt_relpath: str = DEFAULT_V2_SOVITS_CKPT_RELPATH,
    fast_langdetect_relpath: str = DEFAULT_FAST_LANGDETECT_RELPATH,
    split_lang_relpath: str = DEFAULT_SPLIT_LANG_RELPATH,
) -> GSVEnvInfo:
    """
    EN:
    Prepare the runtime environment for the GPT-SoVITS compatibility layer.

    What this function does:
    1. Hard-locks compatibility layer to v2.
    2. Adds `third_party/gsv_compat` to sys.path so original imports like
       `from text...` and `from AR...` still work.
    3. Force-sets environment variables used by the original GPT-SoVITS code.
    4. Returns a structured object containing all resolved paths.

    ZH:
    为 GPT-SoVITS 兼容层准备运行环境。

    这个函数会做四件事：
    1. 强制把兼容层锁定到 v2。
    2. 把 `third_party/gsv_compat` 加入 sys.path，
       这样原项目里的 `from text...`、`from AR...` 还能直接工作。
    3. 强制设置原 GPT-SoVITS 代码需要用到的环境变量。
    4. 返回一个结构化对象，包含所有解析后的路径。
    """
    _validate_version(version)

    root = Path(project_root).resolve() if project_root is not None else get_project_root()
    compat_root = (root / compat_relpath).resolve()
    gsv_asset_root = (root / gsv_asset_relpath).resolve()
    pretrained_root = (gsv_asset_root / "pretrained_models").resolve()
    g2pw_root = (root / g2pw_relpath).resolve()

    bert_path = str((root / bert_relpath).resolve())
    cnhubert_base_path = str((root / cnhubert_relpath).resolve())
    default_stage1_ckpt = str((root / stage1_ckpt_relpath).resolve())
    default_sovits_ckpt = str((root / sovits_ckpt_relpath).resolve())
    fast_langdetect_path = str((root / fast_langdetect_relpath).resolve())
    split_lang_path = str((root / split_lang_relpath).resolve())

    if not compat_root.exists():
        raise FileNotFoundError(
            f"Compatibility root not found: {compat_root}\n"
            f"Expected copied GPT-SoVITS compatibility files under third_party/gsv_compat/."
        )

    # EN:
    # Keep original GPT-SoVITS importers style working:
    #   from text...
    #   from AR...
    #   from feature_extractor...
    #   from split_lang...
    #
    # ZH:
    # 保持原 GPT-SoVITS 风格的导入方式可用：
    #   from text...
    #   from AR...
    #   from feature_extractor...
    #   from split_lang...
    _append_sys_path(compat_root)

    # EN:
    # IMPORTANT:
    # Here we use direct assignment, not os.environ.setdefault(...).
    # Because phase-1 compatibility is hard-locked to v2, we do NOT want stale
    # external environment variables to silently override our paths.
    #
    # ZH:
    # 这里很重要：
    # 我们使用“直接赋值”，而不是 os.environ.setdefault(...)。
    # 因为第一期兼容层已经强锁定为 v2，我们不希望系统环境里残留的旧变量
    # 悄悄覆盖掉当前项目路径。
    os.environ["version"] = version
    os.environ["bert_path"] = bert_path
    os.environ["cnhubert_base_path"] = cnhubert_base_path
    os.environ["gsv_project_root"] = str(root)

    env_info = GSVEnvInfo(
        project_root=root,
        compat_root=compat_root,
        gsv_asset_root=gsv_asset_root,
        pretrained_root=pretrained_root,
        g2pw_root=g2pw_root,
        version=version,
        bert_path=bert_path,
        cnhubert_base_path=cnhubert_base_path,
        default_stage1_ckpt=default_stage1_ckpt,
        default_sovits_ckpt=default_sovits_ckpt,
        fast_langdetect_path=fast_langdetect_path,
        split_lang_path=split_lang_path,
    )
    return env_info


def ensure_gsv_assets_exist(
    env: GSVEnvInfo,
    require_stage1: bool = True,
    require_prompt_tokenizer: bool = True,
    require_split_lang: bool = False,
    require_fast_langdetect: bool = False,
) -> None:
    """
    EN:
    Perform sanity checks on required GPT-SoVITS compatibility assets.

    ZH:
    对 GPT-SoVITS 兼容层所需的关键资产做存在性检查。

    Notes / 说明:
    - `require_stage1=True` means we also require the v2 stage-1 checkpoint.
    - `require_prompt_tokenizer=True` means we also require cnhubert and v2 SoVITS checkpoint.
    - `require_split_lang` and `require_fast_langdetect` are optional because some forks
      may generate/download them lazily, but for a fully sealed compatibility layer
      they are recommended to exist.
    """
    required_paths = [
        env.pretrained_root,
        Path(env.bert_path),
        env.g2pw_root,
    ]

    if require_stage1:
        required_paths.append(Path(env.default_stage1_ckpt))

    if require_prompt_tokenizer:
        required_paths.append(Path(env.cnhubert_base_path))
        required_paths.append(Path(env.default_sovits_ckpt))

    if require_split_lang:
        required_paths.append(Path(env.split_lang_path))

    if require_fast_langdetect:
        required_paths.append(Path(env.fast_langdetect_path))

    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required GPT-SoVITS compatibility assets:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def debug_print_env(env: GSVEnvInfo) -> None:
    """
    EN:
    Pretty-print current compatibility-layer environment information.

    ZH:
    以较清晰的形式打印当前兼容层环境信息，方便调试。
    """
    print("=== GSV ENV INFO (V2-LOCKED) ===")
    print("project_root       :", env.project_root)
    print("compat_root        :", env.compat_root)
    print("gsv_asset_root     :", env.gsv_asset_root)
    print("pretrained_root    :", env.pretrained_root)
    print("g2pw_root          :", env.g2pw_root)
    print("version            :", env.version)
    print("bert_path          :", env.bert_path)
    print("cnhubert_base_path :", env.cnhubert_base_path)
    print("default_stage1_ckpt:", env.default_stage1_ckpt)
    print("default_sovits_ckpt:", env.default_sovits_ckpt)
    print("fast_langdetect    :", env.fast_langdetect_path)
    print("split_lang_path    :", env.split_lang_path)
    print("================================")