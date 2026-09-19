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
import torch

# ============================================================
# Local imports
# 本项目导入
# ============================================================
from src.adapters.gsv_prompt_tokenizer import GSVPromptTokenizer


@dataclass
class GSVSemanticCodecOutput:
    """
    EN:
    Structured output for semantic-code decoding.

    ZH:
    semantic code 解码结果。
    """

    input_codes: torch.Tensor
    normalized_codes: torch.Tensor
    continuous: torch.Tensor
    continuous_lengths: torch.Tensor
    decode_input_shape: list[int]
    continuous_shape: list[int]
    meta: dict[str, Any]


class GSVSemanticCodecV2:
    """
    EN:
    v2-only semantic codec wrapper.

    This class reuses the already implemented GSVPromptTokenizer backend,
    especially the minimal SoVITS tokenizer module:

        ssl_proj + ResidualVectorQuantizer

    The new capability added here is:

        semantic code ids -> quantizer.decode(...) -> continuous semantic vectors

    It intentionally does NOT run the old SoVITS generator.

    ZH:
    v2-only 的 semantic codec 包装器。

    该类复用已有的 GSVPromptTokenizer 后端，尤其是其中的最小 SoVITS
    tokenizer 模块：

        ssl_proj + ResidualVectorQuantizer

    这里新增的能力是：

        semantic token id -> quantizer.decode(...) -> 连续 semantic 向量

    注意：这里不会运行旧 SoVITS 声学生成器。
    """

    def __init__(
            self,
            version: str = "v2",
            device: str | torch.device = "cpu",
            use_half: bool = False,
            project_root: str | None = None,
            expected_continuous_dim: int = 768,
    ) -> None:
        self.device = torch.device(device)
        self.use_half = bool(use_half) and self.device.type == "cuda"
        self.expected_continuous_dim = int(expected_continuous_dim)
        if self.expected_continuous_dim <= 0:
            raise ValueError(
                f"expected_continuous_dim must be positive, got {expected_continuous_dim}"
            )

        # Reuse existing tokenizer wrapper.
        # 复用已有 prompt tokenizer wrapper。
        self.prompt_tokenizer = GSVPromptTokenizer(
            version=version,
            device=self.device,
            use_half=self.use_half,
            project_root=project_root,
            # For codec usage, we may decode arbitrary semantic tokens.
            # 这里只是 codec，不强制参考音频时长。
            enforce_ref_seconds=False,
        )

    # ============================================================
    # Loading
    # 加载
    # ============================================================
    def ensure_loaded(
        self,
        sovits_checkpoint_path: str | Path | None = None,
    ) -> None:
        """
        EN:
        Ensure the minimal SoVITS tokenizer backend is loaded.

        ZH:
        确保最小 SoVITS tokenizer 后端已经加载。
        """
        self.prompt_tokenizer.ensure_tokenizer_loaded(
            sovits_checkpoint_path=sovits_checkpoint_path,
        )

    # ============================================================
    # Code shape normalization
    # code 形状规范化
    # ============================================================
    def normalize_codes_for_decode(
        self,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """
        EN:
        Normalize semantic code ids into shape:

            (B, n_q, T)

        In v2, n_q is normally 1.

        Accepted input shapes:
            (T,)
            (B, T)
            (B, 1, T)
            (1, B, T)  # will be treated carefully if possible

        ZH:
        将 semantic code ids 统一整理成：

            (B, n_q, T)

        在 v2 中，n_q 通常是 1。

        支持输入：
            (T,)
            (B, T)
            (B, 1, T)
            (1, B, T)
        """
        if not torch.is_tensor(codes):
            raise TypeError(f"codes must be torch.Tensor, got {type(codes)}")

        x = codes.detach().long().cpu()

        if x.ndim == 1:
            # (T,) -> (1, 1, T)
            x = x.view(1, 1, -1)

        elif x.ndim == 2:
            # (B, T) -> (B, 1, T)
            x = x.unsqueeze(1)

        elif x.ndim == 3:
            # Usually already (B, n_q, T)
            # If user passes (n_q, B, T), and n_q=1 while second dim > 1,
            # we cannot always know. For this project's normal path,
            # Stage1 / dataset tokens should be (B, T) or (B, 1, T).
            if x.shape[1] == 1:
                pass
            elif x.shape[0] == 1:
                # Interpret (1, B, T) as (B, 1, T)
                x = x.transpose(0, 1).contiguous()
            else:
                raise ValueError(
                    "3D codes must have shape (B, 1, T) or (1, B, T) for v2, "
                    f"got {tuple(x.shape)}"
                )

        else:
            raise ValueError(
                f"codes must be 1D/2D/3D tensor, got shape={tuple(x.shape)}"
            )

        if x.shape[1] != 1:
            raise ValueError(
                f"v2 semantic codec expects n_q=1, got normalized shape={tuple(x.shape)}"
            )

        return x.contiguous().long()

    def _normalize_quantizer_output_to_btc(
            self,
            quantized: torch.Tensor,
    ) -> torch.Tensor:
        """
        EN:
        Normalize quantizer.decode output to shape:

            (B, T, C)

        For GPT-SoVITS v2 semantic codec, C is expected to be 768.

        Common raw shape:
            (B, C, T)

        Already-normalized shape:
            (B, T, C)

        ZH:
        将 quantizer.decode 的输出统一为：

            (B, T, C)

        对 GPT-SoVITS v2 semantic codec 来说，C 预期为 768。
        """
        if not torch.is_tensor(quantized):
            raise TypeError(f"quantized must be torch.Tensor, got {type(quantized)}")

        q = quantized.detach()
        c = int(self.expected_continuous_dim)

        if q.ndim == 2:
            # Most likely (C, T)
            if q.shape[0] == c:
                return q.unsqueeze(0).transpose(1, 2).contiguous()

            # Less likely (T, C)
            if q.shape[1] == c:
                return q.unsqueeze(0).contiguous()

            raise ValueError(
                f"2D quantizer output must be (C,T) or (T,C) with C={c}, "
                f"got shape={tuple(q.shape)}"
            )

        if q.ndim == 3:
            # Raw GPT-SoVITS-style output: (B, C, T)
            if q.shape[1] == c:
                return q.transpose(1, 2).contiguous()

            # Already normalized: (B, T, C)
            if q.shape[2] == c:
                return q.contiguous()

            raise ValueError(
                f"3D quantizer output must be (B,C,T) or (B,T,C) with C={c}, "
                f"got shape={tuple(q.shape)}"
            )

        raise ValueError(
            f"Unsupported quantizer output shape: {tuple(q.shape)}"
        )

    # ============================================================
    # Core decode API
    # 核心 decode 接口
    # ============================================================
    @torch.inference_mode()
    def decode_codes_to_continuous(
        self,
        codes: torch.Tensor,
        *,
        sovits_checkpoint_path: str | Path | None = None,
        return_cpu: bool = True,
        dtype: torch.dtype | None = torch.float32,
    ) -> GSVSemanticCodecOutput:
        """
        EN:
        Decode semantic code ids into GPT-SoVITS quantizer continuous vectors.

        Input:
            codes:
                (T,), (B, T), (B, 1, T), or (1, B, T)

        Output:
            continuous:
                (B, T, C), normally C=768

        ZH:
        将 semantic token id 解码成 GPT-SoVITS quantizer 的连续向量。

        输入：
            codes:
                (T,), (B, T), (B, 1, T), 或 (1, B, T)

        输出：
            continuous:
                (B, T, C)，通常 C=768
        """
        self.ensure_loaded(sovits_checkpoint_path=sovits_checkpoint_path)

        tokenizer_model = self.prompt_tokenizer.tokenizer_model
        if tokenizer_model is None:
            raise RuntimeError("Tokenizer model is not loaded.")

        normalized = self.normalize_codes_for_decode(codes)
        decode_input = normalized.to(self.device)

        # Keep code ids as long.
        # code id 必须保持 long。
        decode_input = decode_input.long()

        # In current project, PromptTokenizerModuleV2.extract_latent returns
        # codes in (B, n_q, T). Original SoVITS decode path also consumes this
        # family of shape.
        quantized = tokenizer_model.quantizer.decode(decode_input)

        continuous = self._normalize_quantizer_output_to_btc(quantized)

        if dtype is not None:
            continuous = continuous.to(dtype=dtype)

        continuous_lengths = torch.full(
            size=(continuous.shape[0],),
            fill_value=int(continuous.shape[1]),
            dtype=torch.long,
            device=continuous.device,
        )

        if return_cpu:
            continuous = continuous.detach().cpu()
            continuous_lengths = continuous_lengths.detach().cpu()
            normalized_out = normalized.detach().cpu()
        else:
            normalized_out = normalized.to(self.device)

        return GSVSemanticCodecOutput(
            input_codes=codes.detach().cpu(),
            normalized_codes=normalized_out,
            continuous=continuous,
            continuous_lengths=continuous_lengths,
            decode_input_shape=list(decode_input.shape),
            continuous_shape=list(continuous.shape),
            meta={
                "codec": "GSVSemanticCodecV2",
                "expected_continuous_dim": int(self.expected_continuous_dim),
                "sovits_checkpoint_path": (
                    str(sovits_checkpoint_path)
                    if sovits_checkpoint_path is not None
                    else self.prompt_tokenizer.sovits_ckpt_path
                ),
                "device": str(self.device),
                "use_half": bool(self.use_half),
            },
        )

    # ============================================================
    # Convenience APIs
    # 便捷接口
    # ============================================================
    @torch.inference_mode()
    def decode_1d_tokens_to_continuous(
        self,
        tokens: torch.Tensor,
        *,
        sovits_checkpoint_path: str | Path | None = None,
        return_cpu: bool = True,
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        """
        EN:
        Convenience API for one 1D token sequence.

        Return:
            (T, C)

        ZH:
        单条 1D token 序列的便捷接口。

        返回：
            (T, C)
        """
        out = self.decode_codes_to_continuous(
            tokens,
            sovits_checkpoint_path=sovits_checkpoint_path,
            return_cpu=return_cpu,
            dtype=dtype,
        )
        return out.continuous[0]

    @torch.inference_mode()
    def extract_prompt_continuous_from_wav(
        self,
        wav_path: str | Path,
        *,
        sovits_checkpoint_path: str | Path | None = None,
        return_cpu: bool = True,
        dtype: torch.dtype | None = torch.float32,
    ) -> GSVSemanticCodecOutput:
        """
        EN:
        End-to-end reference wav -> prompt semantic tokens -> continuous vectors.

        ZH:
        端到端：
            参考 wav -> prompt semantic tokens -> continuous semantic vectors
        """
        prompt_out = self.prompt_tokenizer.extract_prompt_tokens_from_wav(
            wav_path=wav_path,
            sovits_checkpoint_path=sovits_checkpoint_path,
        )

        return self.decode_codes_to_continuous(
            prompt_out.prompt_tokens_2d,
            sovits_checkpoint_path=sovits_checkpoint_path,
            return_cpu=return_cpu,
            dtype=dtype,
        )