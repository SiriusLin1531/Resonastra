from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import inspect
import tempfile
import warnings
from pathlib import Path
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.adapters.gsv_env import (
    setup_gsv_env,
    GSVEnvInfo,
    SUPPORTED_GSV_VERSION,
)


class GSVT2S:
    """
    EN:
    GPT-SoVITS stage-1 Text2Semantic wrapper.

    Phase-1 policy:
    - hard-locked to v2
    - default checkpoint is the official v2 stage-1 checkpoint
    - prompt-based inference is the formal path
    - prompt-free inference is kept only as a debugging / fallback mode

    ZH:
    这是对 GPT-SoVITS 第一阶段 Text2Semantic 的包装器。

    第一阶段兼容策略：
    - 强锁定为 v2
    - 默认 checkpoint 使用官方 v2 stage-1 权重
    - 正式使用路径是“带 prompt 的推理”
    - prompt-free 只保留为调试/兜底路径
    """

    def __init__(
        self,
        version: str = SUPPORTED_GSV_VERSION,
        device: str | torch.device = "cpu",
        use_half: bool = False,
        project_root: str | None = None,
        allow_prompt_free_debug: bool = False,
    ) -> None:
        if version != SUPPORTED_GSV_VERSION:
            raise ValueError(
                f"GSVT2S is locked to {SUPPORTED_GSV_VERSION!r} in phase 1, "
                f"but got version={version!r}."
            )

        self.env: GSVEnvInfo = setup_gsv_env(version=version, project_root=project_root)
        self.version = version
        self.device = torch.device(device)
        self.use_half = bool(use_half) and str(self.device).startswith("cuda")

        # EN:
        # Formal pipeline should use prompt tokens extracted from reference audio.
        # We still keep a debugging fallback to allow prompt-free experiments.
        #
        # ZH:
        # 正式流程应使用参考音频提取出的 prompt token。
        # 这里仍保留一个调试开关，允许在必要时做 prompt-free 实验。
        self.allow_prompt_free_debug = allow_prompt_free_debug

        # Lazy importers after env/path setup
        # 必须在兼容环境配置好之后再导入
        from AR.models.t2s_lightning_module import Text2SemanticLightningModule

        self._Text2SemanticLightningModule = Text2SemanticLightningModule

        self.ckpt_path: str | None = None
        self.model_cfg: dict[str, Any] | None = None
        self.lightning_module = None
        self.model = None

    # ============================================================
    # Internal helpers / 内部辅助函数
    # ============================================================
    def _looks_like_non_v2_checkpoint(self, checkpoint_path: str | Path) -> bool:
        """
        EN:
        A conservative heuristic for rejecting clearly non-v2 checkpoints.

        IMPORTANT:
        Only inspect the checkpoint file name and its direct parent directory.
        Do NOT inspect the whole absolute path, otherwise folder names like
        "VoiceProjects" may falsely trigger "pro" detection.

        ZH:
        一个保守的启发式判断，用于拒绝明显不像 v2 的 checkpoint。

        重要说明：
        只检查 checkpoint 文件名和直接父目录名，
        不检查整条绝对路径，否则像 "VoiceProjects" 这样的目录名
        会误触发 "pro" 检测。
        """
        ckpt_path_obj = Path(checkpoint_path)

        ckpt_file = ckpt_path_obj.name.lower()
        ckpt_parent = ckpt_path_obj.parent.name.lower()

        # EN:
        # Explicitly allow the known default v2 checkpoint.
        #
        # ZH:
        # 明确放行当前已知的 v2 默认权重。
        if ckpt_file == "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt":
            return False

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
        suspicious_tokens = ["s1v3", "s1v4", "v2pro", "proplus", "gptv3", "gptv4"]
        return any(tok in joined_name for tok in suspicious_tokens)

    def _resolve_stage1_ckpt(self, checkpoint_path: str | Path | None) -> str:
        """
        EN:
        Resolve the final stage-1 checkpoint path.
        If caller does not provide one, use the v2 default checkpoint.

        ZH:
        解析最终要使用的第一阶段 checkpoint 路径。
        如果调用方没有显式提供，就使用 v2 默认 checkpoint。
        """
        if checkpoint_path is None:
            checkpoint_path = self.env.default_stage1_ckpt

        checkpoint_path = str(Path(checkpoint_path).resolve())

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {checkpoint_path}")

        if self._looks_like_non_v2_checkpoint(checkpoint_path):
            raise ValueError(
                f"Phase-1 compatibility is v2-only, but checkpoint looks non-v2: {checkpoint_path}"
            )

        return checkpoint_path

    # ============================================================
    # Loading / 模型加载
    # ============================================================
    def load_from_checkpoint(self, checkpoint_path: str | Path | None = None) -> None:
        """
        EN:
        Load stage-1 checkpoint and instantiate the original GPT-SoVITS
        Text2SemanticLightningModule in inference mode.

        ZH:
        加载第一阶段 checkpoint，并以推理模式实例化原始 GPT-SoVITS
        的 Text2SemanticLightningModule。
        """
        checkpoint_path = self._resolve_stage1_ckpt(checkpoint_path)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "config" not in ckpt:
            raise KeyError(f"Checkpoint missing 'config': {checkpoint_path}")
        if "weight" not in ckpt:
            raise KeyError(f"Checkpoint missing 'weight': {checkpoint_path}")

        cfg = ckpt["config"]

        # EN:
        # Align with the original GPT-SoVITS loading style:
        # instantiate with is_train=False.
        #
        # ZH:
        # 对齐原 GPT-SoVITS 的加载方式：
        # 以 is_train=False 实例化。
        output_dir = Path(tempfile.mkdtemp(prefix="gsv_t2s_"))
        lightning_module = self._Text2SemanticLightningModule(
            config=cfg,
            output_dir=output_dir,
            is_train=False,
        )

        load_msg = lightning_module.load_state_dict(ckpt["weight"], strict=False)
        print(f"[GSVT2S] load_state_dict: {load_msg}")

        lightning_module.eval()
        if self.use_half:
            lightning_module = lightning_module.half().to(self.device)
        else:
            lightning_module = lightning_module.to(self.device)

        self.ckpt_path = checkpoint_path
        self.model_cfg = cfg
        self.lightning_module = lightning_module
        self.model = lightning_module.model

    def ensure_loaded(self, checkpoint_path: str | Path | None = None) -> None:
        """
        EN:
        Ensure the stage-1 model is loaded before inference.

        ZH:
        在推理前确保第一阶段模型已经加载。
        """
        if self.model is not None:
            return
        self.load_from_checkpoint(checkpoint_path)

    # ============================================================
    # Prompt helpers / Prompt 辅助函数
    # ============================================================
    def normalize_prompt_tokens(
        self,
        prompt_tokens: torch.Tensor | list[int] | None,
    ) -> torch.Tensor | None:
        """
        EN:
        Accept prompt semantic tokens in one of these forms:
        - None
        - list[int]
        - (T,)
        - (1, T)

        Returns:
        - None
        - (1, T) LongTensor on target device

        ZH:
        接受多种形式的 prompt token：
        - None
        - list[int]
        - 一维张量 (T,)
        - 二维张量 (1, T)

        返回：
        - None
        - 形状规范为 (1, T) 的 LongTensor，并放到目标设备上
        """
        if prompt_tokens is None:
            return None

        if isinstance(prompt_tokens, list):
            prompt_tokens = torch.LongTensor(prompt_tokens)

        if not isinstance(prompt_tokens, torch.Tensor):
            raise TypeError(f"Unsupported prompt_tokens type: {type(prompt_tokens)}")

        prompt_tokens = prompt_tokens.long()

        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        elif prompt_tokens.ndim != 2:
            raise ValueError(
                f"prompt_tokens must be 1D or 2D tensor, got shape={tuple(prompt_tokens.shape)}"
            )

        return prompt_tokens.to(self.device)

    def _validate_prompt_policy(self, prompt: torch.Tensor | None) -> None:
        """
        EN:
        Formal path in phase 1 is prompt-based inference.
        Prompt-free mode is only allowed if explicitly enabled for debugging.

        ZH:
        第一阶段正式路径是带 prompt 的推理。
        如果要走 prompt-free，必须显式打开调试模式。
        """
        if prompt is None:
            if not self.allow_prompt_free_debug:
                raise ValueError(
                    "prompt_tokens is None, but phase-1 formal inference path requires prompt tokens.\n"
                    "If you intentionally want prompt-free debug mode, initialize GSVT2S with "
                    "allow_prompt_free_debug=True."
                )
            warnings.warn(
                "Running stage-1 in prompt-free debug mode. "
                "This is not the formal compatibility-layer path.",
                stacklevel=2,
            )

    # ============================================================
    # Semantic generation / 语义生成
    # ============================================================
    @torch.inference_mode()
    def generate_semantic(
        self,
        phoneme_ids: torch.LongTensor,
        phoneme_lens: torch.LongTensor,
        bert_feature: torch.Tensor,
        prompt_tokens: torch.Tensor | list[int] | None = None,
        checkpoint_path: str | Path | None = None,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        early_stop_num: int = -1,
        repetition_penalty: float = 1.35,
    ) -> tuple[torch.Tensor, Any]:
        """
        EN:
        Stage-1 semantic generation.

        Inputs:
            phoneme_ids:  (1, T_text)
            phoneme_lens: (1,)
            bert_feature: (1, C, T_text)
            prompt_tokens:
                - formal path: (1, T_prompt)
                - debug path : None only if allow_prompt_free_debug=True

        Returns:
            pred_semantic, aux

        ZH:
        第一阶段 semantic 生成接口。

        输入：
            phoneme_ids:  (1, T_text)
            phoneme_lens: (1,)
            bert_feature: (1, C, T_text)
            prompt_tokens:
                - 正式路径：形状为 (1, T_prompt)
                - 调试路径：仅当 allow_prompt_free_debug=True 时可为 None

        返回：
            pred_semantic, aux
        """
        self.ensure_loaded(checkpoint_path=checkpoint_path)
        assert self.model is not None

        phoneme_ids = phoneme_ids.to(self.device)
        phoneme_lens = phoneme_lens.to(self.device)
        bert_feature = bert_feature.to(self.device)
        prompt = self.normalize_prompt_tokens(prompt_tokens)

        self._validate_prompt_policy(prompt)

        # EN:
        # Prefer infer_panel(...) because that is the main GPT-SoVITS-style interface.
        #
        # ZH:
        # 优先使用 infer_panel(...)，因为这是 GPT-SoVITS 更常见的主接口。
        if hasattr(self.model, "infer_panel"):
            infer_fn = getattr(self.model, "infer_panel")
            kwargs = dict(
                top_k=top_k,
                top_p=top_p,
                early_stop_num=early_stop_num,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            sig = inspect.signature(infer_fn)
            call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

            return infer_fn(
                phoneme_ids,
                phoneme_lens,
                prompt,
                bert_feature,
                **call_kwargs,
            )

        # EN:
        # Fallback for branches that expose infer(...) instead of infer_panel(...).
        #
        # ZH:
        # 兜底处理：某些分支可能暴露的是 infer(...) 而不是 infer_panel(...)。
        if hasattr(self.model, "infer"):
            infer_fn = getattr(self.model, "infer")
            sig = inspect.signature(infer_fn)
            kwargs = dict(
                top_k=top_k,
                early_stop_num=early_stop_num,
                temperature=temperature,
            )
            call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

            return infer_fn(
                phoneme_ids,
                phoneme_lens,
                prompt,
                bert_feature,
                **call_kwargs,
            )

        raise AttributeError("Loaded stage-1 model has neither 'infer_panel' nor 'infer' method.")

    @torch.inference_mode()
    def generate_semantic_from_text(
        self,
        text: str,
        language: str,
        frontend,
        prompt_tokens: torch.Tensor | list[int] | None = None,
        checkpoint_path: str | Path | None = None,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        early_stop_num: int = -1,
        repetition_penalty: float = 1.35,
    ) -> dict[str, Any]:
        """
        EN:
        Convenience wrapper:
            text
            -> frontend.prepare_ids_and_bert(...)
            -> stage-1 semantic generation

        ZH:
        便捷包装接口：
            文本
            -> frontend.prepare_ids_and_bert(...)
            -> 第一阶段 semantic 生成
        """
        phoneme_ids, phoneme_lens, bert_feature, norm_text = frontend.prepare_ids_and_bert(
            text=text,
            language=language,
        )

        pred_semantic, aux = self.generate_semantic(
            phoneme_ids=phoneme_ids,
            phoneme_lens=phoneme_lens,
            bert_feature=bert_feature,
            prompt_tokens=prompt_tokens,
            checkpoint_path=checkpoint_path,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stop_num=early_stop_num,
            repetition_penalty=repetition_penalty,
        )

        return {
            "norm_text": norm_text,
            "phoneme_ids": phoneme_ids,
            "phoneme_lens": phoneme_lens,
            "bert_feature": bert_feature,
            "pred_semantic": pred_semantic,
            "aux": aux,
            "prompt_used": prompt_tokens is not None,
        }