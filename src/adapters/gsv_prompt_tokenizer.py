from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import librosa
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Local project imports
# 本项目内部导入
# ============================================================
from src.adapters.gsv_env import (
    setup_gsv_env,
    ensure_gsv_assets_exist,
    GSVEnvInfo,
    SUPPORTED_GSV_VERSION,
)


# ---------------------------------------------------------------------
# DictToAttrRecursive
#
# EN:
# A small helper that converts nested dictionaries into attribute-style objects.
#
# ZH:
# 一个小工具，把嵌套字典转换成支持“点号访问”的对象。
# ---------------------------------------------------------------------
class DictToAttrRecursive(dict):
    def __init__(self, input_dict: dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(f"Attribute {item} not found") from e

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError as e:
            raise AttributeError(f"Attribute {item} not found") from e


# ---------------------------------------------------------------------
# PromptTokenizerOutput
#
# EN:
# Structured return type for prompt-token extraction.
#
# ZH:
# 用于统一封装 prompt token 提取结果的数据结构。
# ---------------------------------------------------------------------
@dataclass
class PromptTokenizerOutput:
    # Original audio path
    # 原始参考音频路径
    wav_path: str

    # Loaded mono 16k waveform BEFORE tail silence append
    # 加载后的单声道 16k 波形（尚未拼接尾部静音）
    wav16k: torch.Tensor  # (T,)

    # Waveform AFTER tail silence append
    # 拼接尾部静音后的波形
    wav16k_with_tail_silence: torch.Tensor  # (T_aug,)

    # SSL feature from cnhubert
    # 由 cnhubert 提取的 SSL 特征
    ssl_content: torch.Tensor  # (1, C, T_ssl)

    # Full discrete codes returned by tokenizer
    # tokenizer 返回的完整离散 code
    full_codes: torch.Tensor  # typically (n_q, B, T)

    # 1D prompt semantic sequence
    # 一维 prompt semantic token 序列
    prompt_semantic_1d: torch.Tensor  # (T,)

    # Stage-1 ready prompt tensor
    # 可直接送进 stage-1 的二维 prompt 张量
    prompt_tokens_2d: torch.Tensor  # (1, T)


# ---------------------------------------------------------------------
# PromptTokenizerModuleV2
#
# EN:
# Minimal v2-only tokenizer module extracted from the old SoVITS family.
#
# The goal is to keep only the truly necessary tokenizer path:
#   ssl_content -> ssl_proj -> quantizer -> codes
#
# We intentionally DO NOT keep the old second-stage generator chain.
#
# ZH:
# 这是从旧 v2 SoVITS 家族中裁出来的“最小 tokenizer 模块”。
#
# 目标是只保留真正必要的 tokenizer 路径：
#   ssl_content -> ssl_proj -> quantizer -> codes
#
# 我们刻意不再保留旧第二阶段的声学生成主链。
# ---------------------------------------------------------------------
class PromptTokenizerModuleV2(nn.Module):
    def __init__(
        self,
        ssl_in_channels: int = 768,
        ssl_out_channels: int = 768,
        ssl_kernel_size: int = 2,
        ssl_stride: int = 2,
        codebook_bins: int = 1024,
        n_q: int = 1,
    ) -> None:
        super().__init__()

        # EN:
        # This matches the v2 tokenizer-style projection:
        # - semantic_frame_rate == "25hz" -> kernel=2, stride=2
        # - semantic_frame_rate == "50hz" -> kernel=1, stride=1
        #
        # Here we are v2-locked and default to the common 25hz setup.
        #
        # ZH:
        # 这里对齐 v2 tokenizer 的 SSL 投影逻辑：
        # - semantic_frame_rate == "25hz" -> kernel=2, stride=2
        # - semantic_frame_rate == "50hz" -> kernel=1, stride=1
        #
        # 当前兼容层锁定 v2，默认走常见的 25hz 配置。
        self.ssl_proj = nn.Conv1d(
            ssl_in_channels,
            ssl_out_channels,
            kernel_size=ssl_kernel_size,
            stride=ssl_stride,
        )

        # EN:
        # Import quantizer lazily from copied GPT-SoVITS compatibility code.
        #
        # ZH:
        # 从复制过来的 GPT-SoVITS 兼容层中懒加载 quantizer。
        from module.quantize import ResidualVectorQuantizer

        # EN:
        # The original v2 tokenizer family commonly uses:
        # - dimension = 768
        # - n_q = 1
        # - bins = 1024
        #
        # ZH:
        # 原 v2 tokenizer 家族常见配置为：
        # - dimension = 768
        # - n_q = 1
        # - bins = 1024
        self.quantizer = ResidualVectorQuantizer(
            dimension=ssl_out_channels,
            n_q=n_q,
            bins=codebook_bins,
        )

    @torch.inference_mode()
    def extract_latent(self, ssl_content: torch.Tensor) -> torch.Tensor:
        """
        EN:
        Match original GPT-SoVITS tokenizer behavior:
            ssl = ssl_proj(ssl_content)
            quantized, codes, commit_loss, quantized_list = quantizer(ssl)
            return codes.transpose(0, 1)

        ZH:
        对齐原 GPT-SoVITS tokenizer 行为：
            ssl = ssl_proj(ssl_content)
            quantized, codes, commit_loss, quantized_list = quantizer(ssl)
            return codes.transpose(0, 1)
        """
        ssl = self.ssl_proj(ssl_content)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)


class GSVPromptTokenizer:
    """
    EN:
    v2-only compatibility wrapper for:
        reference audio -> prompt semantic tokens

    Phase-1 policy:
    - hard-locked to v2
    - use cnhubert + minimal tokenizer module
    - keep behavior close to original GPT-SoVITS v2 prompt extraction
    - do NOT depend on the full old second-stage generator anymore

    ZH:
    这是一个 v2-only 的兼容包装器，用于实现：
        参考音频 -> prompt semantic tokens

    第一阶段兼容策略：
    - 强锁定为 v2
    - 使用 cnhubert + 最小 tokenizer 模块
    - 尽量贴近原 GPT-SoVITS v2 的 prompt 提取行为
    - 不再依赖完整旧第二阶段生成器
    """

    # EN:
    # Original prompt extraction path expects reference audio to be between 3s and 10s.
    #
    # ZH:
    # 原 prompt 提取路径要求参考音频长度在 3 秒到 10 秒之间。
    MIN_REF_SECONDS = 3.0
    MAX_REF_SECONDS = 10.0

    # EN:
    # Original inference appends a short silence tail before SSL extraction.
    # We make it configurable but default to 0.3s.
    #
    # ZH:
    # 原推理会在送入 SSL 之前给参考音频尾部拼一小段静音。
    # 这里做成可配置，默认 0.3 秒。
    DEFAULT_TAIL_SILENCE_SEC = 0.3

    def __init__(
        self,
        version: str = SUPPORTED_GSV_VERSION,
        device: str | torch.device = "cpu",
        use_half: bool = False,
        project_root: str | None = None,
        tail_silence_sec: float = DEFAULT_TAIL_SILENCE_SEC,
        enforce_ref_seconds: bool = True,
    ) -> None:
        if version != SUPPORTED_GSV_VERSION:
            raise ValueError(
                f"GSVPromptTokenizer is locked to {SUPPORTED_GSV_VERSION!r} in phase 1, "
                f"but got version={version!r}."
            )

        self.env: GSVEnvInfo = setup_gsv_env(version=version, project_root=project_root)

        # EN:
        # Prompt tokenizer requires:
        # - bert/G2PW are not strictly needed here, but we reuse the same env checks
        # - cnhubert
        # - v2 SoVITS checkpoint
        #
        # ZH:
        # prompt tokenizer 需要：
        # - 虽然这里不直接依赖 bert/G2PW，但沿用统一环境检查
        # - cnhubert
        # - v2 SoVITS checkpoint
        ensure_gsv_assets_exist(
            self.env,
            require_stage1=False,
            require_prompt_tokenizer=True,
            require_split_lang=False,
            require_fast_langdetect=False,
        )

        self.version = version
        self.device = torch.device(device)
        self.use_half = bool(use_half) and str(self.device).startswith("cuda")
        self.tail_silence_sec = float(tail_silence_sec)
        self.enforce_ref_seconds = bool(enforce_ref_seconds)

        # EN:
        # Lazy importers after environment setup.
        #
        # ZH:
        # 兼容环境准备完成后再导入原 cnhubert 模块。
        from feature_extractor import cnhubert

        self._cnhubert = cnhubert

        self.ssl_model = None
        self.tokenizer_model: PromptTokenizerModuleV2 | None = None
        self.sovits_ckpt_path: str | None = None
        self.hps: DictToAttrRecursive | None = None

    # ============================================================
    # Internal helpers / 内部辅助函数
    # ============================================================
    def _resolve_sovits_ckpt(self, sovits_checkpoint_path: str | Path | None) -> str:
        """
        EN:
        Resolve final SoVITS checkpoint path.
        If caller does not provide one, use the v2 default SoVITS checkpoint.

        ZH:
        解析最终使用的 SoVITS checkpoint 路径。
        如果调用方没有显式提供，就使用 v2 默认 SoVITS checkpoint。

        EN:
        IMPORTANT:
        We only inspect the checkpoint file name and its direct parent directory
        when deciding whether it "looks non-v2". We intentionally do NOT inspect
        the whole absolute path, otherwise user folder names like "VoiceProjects"
        may accidentally trigger false positives because they contain "pro".

        ZH:
        重要说明：
        在判断 checkpoint 是否“看起来不像 v2”时，我们只检查：
        - checkpoint 文件名
        - 直接父目录名

        我们故意不再检查整条绝对路径，
        否则像 "VoiceProjects" 这样的目录名中含有 "pro"，
        会被误判成 "v2Pro" 相关路径。
        """
        if sovits_checkpoint_path is None:
            sovits_checkpoint_path = self.env.default_sovits_ckpt

        checkpoint_path = str(Path(sovits_checkpoint_path).resolve())

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"SoVITS checkpoint not found: {checkpoint_path}")

        ckpt_path_obj = Path(checkpoint_path)
        ckpt_file = ckpt_path_obj.name.lower()
        ckpt_parent = ckpt_path_obj.parent.name.lower()

        # EN:
        # Explicitly allow the known default v2 checkpoint.
        #
        # ZH:
        # 明确放行当前已知的 v2 默认权重。
        if ckpt_file == "s2g2333k.pth":
            return checkpoint_path

        # EN:
        # Only inspect parent directory + file name, not the whole absolute path.
        #
        # ZH:
        # 只检查父目录名 + 文件名，不检查整条绝对路径。
        joined_name = f"{ckpt_parent}/{ckpt_file}"

        # EN:
        # Reject only clearly non-v2 checkpoints.
        #
        # ZH:
        # 只拒绝那些“明显不像 v2”的 checkpoint。
        suspicious_tokens = ["s2gv3", "s2gv4", "v2pro", "proplus", "sovitsv3", "sovitsv4"]
        if any(tok in joined_name for tok in suspicious_tokens):
            raise ValueError(
                f"Phase-1 compatibility is v2-only, but checkpoint looks non-v2: {checkpoint_path}"
            )

        return checkpoint_path

    def _load_checkpoint_dict(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """
        EN:
        Load raw SoVITS checkpoint dictionary.

        ZH:
        加载原始 SoVITS checkpoint 字典。
        """
        checkpoint_path = str(Path(checkpoint_path).resolve())
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "config" not in ckpt:
            raise KeyError(f"SoVITS checkpoint missing 'config': {checkpoint_path}")
        if "weight" not in ckpt:
            raise KeyError(f"SoVITS checkpoint missing 'weight': {checkpoint_path}")

        return ckpt

    def _prepare_hps(self, ckpt: dict[str, Any]) -> DictToAttrRecursive:
        """
        EN:
        Convert checkpoint config dict into attribute-style object.

        ZH:
        把 checkpoint config 字典转换成属性访问对象。
        """
        hps = DictToAttrRecursive(ckpt["config"])

        # EN:
        # Force v2 compatibility assumptions.
        #
        # ZH:
        # 强制使用 v2 兼容假设。
        if not hasattr(hps, "model"):
            raise KeyError("SoVITS checkpoint config missing 'model' section.")
        if not hasattr(hps, "data"):
            raise KeyError("SoVITS checkpoint config missing 'data' section.")

        if not hasattr(hps.model, "semantic_frame_rate") or hps.model.semantic_frame_rate is None:
            hps.model.semantic_frame_rate = "25hz"

        hps.model.version = "v2"
        return hps

    def _infer_tokenizer_hparams_from_ckpt(
        self,
        ckpt: dict[str, Any],
        hps: DictToAttrRecursive,
    ) -> dict[str, Any]:
        """
        EN:
        Infer minimal tokenizer hyperparameters from checkpoint weights/config.

        We deliberately avoid importing the full old `module.models`,
        and only reconstruct the tokenizer path:
            ssl_proj + quantizer

        ZH:
        从 checkpoint 权重/配置中推断最小 tokenizer 所需超参数。

        我们刻意不再导入完整旧 `module.models`，
        而是只重建 tokenizer 路径：
            ssl_proj + quantizer
        """
        weight = ckpt["weight"]

        if "ssl_proj.weight" not in weight:
            raise KeyError("Checkpoint weight missing 'ssl_proj.weight'.")
        if "ssl_proj.bias" not in weight:
            raise KeyError("Checkpoint weight missing 'ssl_proj.bias'.")

        ssl_proj_w = weight["ssl_proj.weight"]

        # Conv1d weight shape: (out_channels, in_channels, kernel_size)
        # Conv1d 权重形状：(out_channels, in_channels, kernel_size)
        ssl_out_channels = int(ssl_proj_w.shape[0])
        ssl_in_channels = int(ssl_proj_w.shape[1])
        ssl_kernel_size = int(ssl_proj_w.shape[2])

        # EN:
        # stride is not stored as weight shape, so we infer from semantic_frame_rate.
        #
        # ZH:
        # stride 不体现在权重 shape 里，因此根据 semantic_frame_rate 推断。
        semantic_frame_rate = str(hps.model.semantic_frame_rate).lower()
        if semantic_frame_rate == "25hz":
            ssl_stride = 2
        elif semantic_frame_rate == "50hz":
            ssl_stride = 1
        else:
            raise ValueError(
                f"Unsupported semantic_frame_rate for v2 tokenizer: {semantic_frame_rate}"
            )

        # EN:
        # V2 tokenizer family commonly uses n_q=1 and bins=1024.
        # We keep them fixed here for phase-1 v2 locking.
        #
        # ZH:
        # v2 tokenizer 家族常见地使用 n_q=1 和 bins=1024。
        # 在 v2 锁定策略下，这里直接固定。
        n_q = 1
        codebook_bins = 1024

        return {
            "ssl_in_channels": ssl_in_channels,
            "ssl_out_channels": ssl_out_channels,
            "ssl_kernel_size": ssl_kernel_size,
            "ssl_stride": ssl_stride,
            "n_q": n_q,
            "codebook_bins": codebook_bins,
        }

    def _extract_tokenizer_state_dict(self, ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        EN:
        Extract only tokenizer-related weights from full SoVITS checkpoint.

        ZH:
        从完整 SoVITS checkpoint 中筛出 tokenizer 相关权重。
        """
        weight = ckpt["weight"]
        tokenizer_state = {}

        for k, v in weight.items():
            if k.startswith("ssl_proj.") or k.startswith("quantizer."):
                tokenizer_state[k] = v

        if not any(k.startswith("ssl_proj.") for k in tokenizer_state):
            raise KeyError("Tokenizer state extraction failed: no 'ssl_proj.*' weights found.")
        if not any(k.startswith("quantizer.") for k in tokenizer_state):
            raise KeyError("Tokenizer state extraction failed: no 'quantizer.*' weights found.")

        return tokenizer_state

    def _build_tokenizer_model_from_ckpt(self, ckpt: dict[str, Any]) -> PromptTokenizerModuleV2:
        """
        EN:
        Build a minimal v2 tokenizer module and load tokenizer-related weights.

        ZH:
        构建最小化的 v2 tokenizer 模块，并加载 tokenizer 相关权重。
        """
        hps = self._prepare_hps(ckpt)
        hp = self._infer_tokenizer_hparams_from_ckpt(ckpt, hps)

        tokenizer_model = PromptTokenizerModuleV2(
            ssl_in_channels=hp["ssl_in_channels"],
            ssl_out_channels=hp["ssl_out_channels"],
            ssl_kernel_size=hp["ssl_kernel_size"],
            ssl_stride=hp["ssl_stride"],
            codebook_bins=hp["codebook_bins"],
            n_q=hp["n_q"],
        )

        tokenizer_state = self._extract_tokenizer_state_dict(ckpt)
        load_msg = tokenizer_model.load_state_dict(tokenizer_state, strict=False)
        print(f"[GSVPromptTokenizer] tokenizer load_state_dict: {load_msg}")

        tokenizer_model.eval()
        if self.use_half:
            tokenizer_model = tokenizer_model.half().to(self.device)
        else:
            tokenizer_model = tokenizer_model.to(self.device)

        self.hps = hps
        return tokenizer_model

    def _ensure_ssl_loaded(self) -> None:
        """
        EN:
        Lazy-load the original cnhubert SSL model.

        ZH:
        懒加载原始 cnhubert SSL 模型。
        """
        if self.ssl_model is not None:
            return

        # EN:
        # Follow original GPT-SoVITS behavior:
        # set global cnhubert base path before get_model().
        #
        # ZH:
        # 对齐原 GPT-SoVITS 的行为：
        # 在调用 get_model() 之前先设置全局 cnhubert 路径。
        self._cnhubert.cnhubert_base_path = self.env.cnhubert_base_path
        ssl_model = self._cnhubert.get_model()

        if self.use_half:
            ssl_model = ssl_model.half().to(self.device)
        else:
            ssl_model = ssl_model.to(self.device)

        ssl_model.eval()
        self.ssl_model = ssl_model

    def ensure_tokenizer_loaded(self, sovits_checkpoint_path: str | Path | None = None) -> None:
        """
        EN:
        Ensure both SSL model and v2 tokenizer backend are loaded.

        ZH:
        确保 SSL 模型和 v2 tokenizer 后端都已经加载。
        """
        self._ensure_ssl_loaded()

        if self.tokenizer_model is not None:
            return

        checkpoint_path = self._resolve_sovits_ckpt(sovits_checkpoint_path)
        ckpt = self._load_checkpoint_dict(checkpoint_path)
        self.tokenizer_model = self._build_tokenizer_model_from_ckpt(ckpt)
        self.sovits_ckpt_path = checkpoint_path

    # ============================================================
    # Audio preprocessing / 音频预处理
    # ============================================================
    def load_wav16k(self, wav_path: str | Path) -> torch.Tensor:
        """
        EN:
        Load audio as mono 16kHz waveform.

        ZH:
        读取音频，并转换为单声道 16kHz 波形。

        Returns:
        - 1D float tensor with shape (T,)

        返回：
        - 形状为 (T,) 的一维 float tensor
        """
        wav_path = str(Path(wav_path).resolve())
        wav, _ = librosa.load(wav_path, sr=16000, mono=True)
        wav = np.asarray(wav, dtype=np.float32)
        return torch.from_numpy(wav)

    def _validate_reference_duration(self, wav16k: torch.Tensor) -> None:
        """
        EN:
        Match original GPT-SoVITS v2 reference-audio length constraints:
        reference audio should be roughly 3s to 10s.

        ZH:
        对齐原 GPT-SoVITS v2 的参考音频长度约束：
        参考音频时长应大致在 3 秒到 10 秒之间。
        """
        if not self.enforce_ref_seconds:
            return

        num_samples = int(wav16k.shape[0])
        min_samples = int(self.MIN_REF_SECONDS * 16000)
        max_samples = int(self.MAX_REF_SECONDS * 16000)

        if num_samples < min_samples or num_samples > max_samples:
            raise ValueError(
                f"Reference audio length must be between {self.MIN_REF_SECONDS:.1f}s and "
                f"{self.MAX_REF_SECONDS:.1f}s for the formal v2 prompt path, "
                f"but got {num_samples / 16000:.2f}s."
            )

    def append_tail_silence(self, wav16k: torch.Tensor) -> torch.Tensor:
        """
        EN:
        Append a short silence tail before SSL extraction,
        following the original GPT-SoVITS prompt extraction style.

        ZH:
        在做 SSL 提取前给参考音频拼接一小段尾部静音，
        以贴近原 GPT-SoVITS 的 prompt 提取行为。
        """
        silence_len = max(int(self.tail_silence_sec * 16000), 0)

        if silence_len == 0:
            return wav16k

        zero_wav = torch.zeros(silence_len, dtype=wav16k.dtype)
        return torch.cat([wav16k, zero_wav], dim=0)

    # ============================================================
    # SSL extraction / SSL 特征提取
    # ============================================================
    @torch.inference_mode()
    def extract_ssl_from_wav(self, wav_path: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        EN:
        Extract SSL features from a reference waveform using cnhubert.

        Returns:
        - wav16k: original loaded waveform, shape (T,)
        - wav16k_with_tail_silence: waveform after appending silence
        - ssl_content: SSL feature, shape (1, C, T_ssl)

        ZH:
        使用 cnhubert 从参考音频中提取 SSL 特征。

        返回：
        - wav16k: 原始加载的波形，形状 (T,)
        - wav16k_with_tail_silence: 拼接静音后的波形
        - ssl_content: SSL 特征，形状 (1, C, T_ssl)
        """
        self._ensure_ssl_loaded()
        assert self.ssl_model is not None

        wav16k = self.load_wav16k(wav_path)
        self._validate_reference_duration(wav16k)

        wav16k_aug = self.append_tail_silence(wav16k)

        wav_input = wav16k_aug.unsqueeze(0).to(self.device)
        if self.use_half:
            wav_input = wav_input.half()

        # EN:
        # Follow original GPT-SoVITS extraction logic:
        # ssl_model.model(wav_input)["last_hidden_state"].transpose(1, 2)
        #
        # ZH:
        # 对齐原 GPT-SoVITS 的 SSL 提取逻辑：
        # ssl_model.model(wav_input)["last_hidden_state"].transpose(1, 2)
        ssl_content = self.ssl_model.model(wav_input)["last_hidden_state"].transpose(1, 2)

        return wav16k, wav16k_aug, ssl_content

    # ============================================================
    # Prompt-token extraction / Prompt token 提取
    # ============================================================
    @torch.inference_mode()
    def extract_prompt_tokens_from_ssl(
        self,
        ssl_content: torch.Tensor,
        sovits_checkpoint_path: str | Path | None = None,
    ) -> torch.Tensor:
        """
        EN:
        Convert SSL feature into discrete prompt tokens using the v2 tokenizer backend.

        Input shape:
        - ssl_content: (1, C, T_ssl)

        Output shape:
        - full_codes: typically (n_q, B, T)

        ZH:
        使用 v2 tokenizer 后端，把 SSL 特征转换成离散 prompt token。

        输入形状：
        - ssl_content: (1, C, T_ssl)

        输出形状：
        - full_codes: 通常为 (n_q, B, T)
        """
        self.ensure_tokenizer_loaded(sovits_checkpoint_path=sovits_checkpoint_path)
        assert self.tokenizer_model is not None

        ssl_content = ssl_content.to(self.device)
        if self.use_half:
            ssl_content = ssl_content.half()

        full_codes = self.tokenizer_model.extract_latent(ssl_content)
        return full_codes

    @torch.inference_mode()
    def extract_prompt_tokens_from_wav(
        self,
        wav_path: str | Path,
        sovits_checkpoint_path: str | Path | None = None,
    ) -> PromptTokenizerOutput:
        """
        EN:
        End-to-end formal prompt extraction path:
            reference wav
            -> wav16k
            -> append tail silence
            -> cnhubert SSL
            -> v2 tokenizer extract_latent
            -> prompt_semantic = codes[0, 0]
            -> prompt_tokens_2d = prompt_semantic.unsqueeze(0)

        ZH:
        正式的端到端 prompt 提取路径：
            参考音频
            -> wav16k
            -> 拼接尾部静音
            -> cnhubert SSL
            -> v2 tokenizer extract_latent
            -> prompt_semantic = codes[0, 0]
            -> prompt_tokens_2d = prompt_semantic.unsqueeze(0)
        """
        wav16k, wav16k_aug, ssl_content = self.extract_ssl_from_wav(wav_path)

        full_codes = self.extract_prompt_tokens_from_ssl(
            ssl_content=ssl_content,
            sovits_checkpoint_path=sovits_checkpoint_path,
        )

        # EN:
        # Match original GPT-SoVITS v2 inference behavior:
        #   prompt_semantic = codes[0, 0]
        #
        # ZH:
        # 对齐原 GPT-SoVITS v2 的推理行为：
        #   prompt_semantic = codes[0, 0]
        prompt_semantic_1d = full_codes[0, 0].detach().long()

        # EN:
        # Stage-1 prompt interface expects shape (1, T).
        #
        # ZH:
        # 第一阶段 prompt 接口通常期望形状为 (1, T)。
        prompt_tokens_2d = prompt_semantic_1d.unsqueeze(0).to(self.device)

        return PromptTokenizerOutput(
            wav_path=str(Path(wav_path).resolve()),
            wav16k=wav16k,
            wav16k_with_tail_silence=wav16k_aug,
            ssl_content=ssl_content,
            full_codes=full_codes,
            prompt_semantic_1d=prompt_semantic_1d,
            prompt_tokens_2d=prompt_tokens_2d,
        )

    @torch.inference_mode()
    def extract_prompt_semantic_from_wav(
        self,
        wav_path: str | Path,
        sovits_checkpoint_path: str | Path | None = None,
    ) -> torch.Tensor:
        """
        EN:
        Convenience API:
        return only stage-1-ready prompt tokens with shape (1, T).

        ZH:
        便捷接口：
        只返回第一阶段可直接使用的 prompt token，形状为 (1, T)。
        """
        out = self.extract_prompt_tokens_from_wav(
            wav_path=wav_path,
            sovits_checkpoint_path=sovits_checkpoint_path,
        )
        return out.prompt_tokens_2d