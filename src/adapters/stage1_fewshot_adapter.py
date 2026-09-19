from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch
import torch.nn as nn

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.adapters.gsv_env import (
    setup_gsv_env,
    ensure_gsv_assets_exist,
    GSVEnvInfo,
    SUPPORTED_GSV_VERSION,
)


Stage1TrainableScope = Literal[
    "frozen_debug",
    "head_only",
    "bert_proj_plus_head",
    "head_and_last_n",
    "full_stage1_debug",
]


@dataclass
class Stage1ParamReport:
    """
    EN:
    Parameter accounting report after applying a trainable scope.

    ZH:
    应用 trainable scope 后的参数统计报告。
    """

    scope: str
    last_n_layers: int
    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_ratio: float
    num_trainable_tensors: int
    num_frozen_tensors: int
    trainable_names_preview: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage1ForwardOutput:
    """
    EN:
    Structured output for one teacher-forcing forward_old call.

    ZH:
    一次 teacher-forcing forward_old 调用的结构化输出。
    """

    loss: torch.Tensor
    loss_value: float
    top_k_acc: float
    top_k: int
    batch_size: int
    phoneme_len_max: int
    semantic_len_max: int

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "loss_value": self.loss_value,
            "top_k_acc": self.top_k_acc,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "phoneme_len_max": self.phoneme_len_max,
            "semantic_len_max": self.semantic_len_max,
        }


class Stage1FewshotAdapter(nn.Module):
    """
    EN:
    Thin training adapter for VoiceLab v6.7.0 Original-style Stage1 Few-shot.

    This wrapper intentionally keeps the original GPT-SoVITS stage-1 training
    contract:
        model.forward_old(phoneme_ids, phoneme_ids_len, semantic_ids,
                          semantic_ids_len, bert_feature)

    The first v6.7.0 version does NOT add LoRA and does NOT introduce a custom
    prompt strategy. It only provides:
    - checkpoint loading
    - trainable scope control
    - one teacher-forcing forward loss call
    - parameter accounting

    ZH:
    VoiceLab v6.7.0 Original-style Stage1 Few-shot 的轻量训练适配器。

    这个包装器刻意保留原 GPT-SoVITS stage1 训练契约：
        model.forward_old(phoneme_ids, phoneme_ids_len, semantic_ids,
                          semantic_ids_len, bert_feature)

    v6.7.0 第一版不加入 LoRA，也不引入自定义 prompt 策略。它只提供：
    - checkpoint 加载
    - trainable scope 控制
    - 一次 teacher-forcing forward loss 调用
    - 参数统计
    """

    SUPPORTED_SCOPES = {
        "frozen_debug",
        "head_only",
        "bert_proj_plus_head",
        "head_and_last_n",
        "full_stage1_debug",
    }

    def __init__(
        self,
        stage1_ckpt_path: str | Path | None = None,
        version: str = SUPPORTED_GSV_VERSION,
        device: str | torch.device = "cpu",
        use_half: bool = False,
        project_root: str | Path | None = None,
        trainable_scope: Stage1TrainableScope = "frozen_debug",
        last_n_layers: int = 4,
        load_strict: bool = False,
    ) -> None:
        super().__init__()

        if version != SUPPORTED_GSV_VERSION:
            raise ValueError(
                f"Stage1FewshotAdapter is locked to {SUPPORTED_GSV_VERSION!r}, got version={version!r}."
            )

        self.env: GSVEnvInfo = setup_gsv_env(version=version, project_root=project_root)
        ensure_gsv_assets_exist(self.env, require_stage1=True, require_prompt_tokenizer=False)

        self.version = version
        self.device = torch.device(device)
        self.use_half = bool(use_half) and str(self.device).startswith("cuda")
        self.load_strict = bool(load_strict)
        self.stage1_ckpt_path = self._resolve_stage1_ckpt(stage1_ckpt_path)
        self.ckpt_config: dict[str, Any] | None = None
        self.load_msg: Any = None
        self.param_report: Stage1ParamReport | None = None

        self.lightning_module, self.model = self._load_lightning_and_model()
        self.apply_trainable_scope(trainable_scope, last_n_layers=last_n_layers)

    # ============================================================
    # Loading helpers
    # 加载辅助函数
    # ============================================================
    def _resolve_stage1_ckpt(self, stage1_ckpt_path: str | Path | None) -> str:
        if stage1_ckpt_path is None:
            stage1_ckpt_path = self.env.default_stage1_ckpt
        ckpt_path = str(Path(stage1_ckpt_path).resolve())
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt_path}")
        return ckpt_path

    def _load_ckpt(self) -> dict[str, Any]:
        ckpt = torch.load(self.stage1_ckpt_path, map_location="cpu", weights_only=False)
        if "config" not in ckpt:
            raise KeyError(f"Stage-1 checkpoint missing 'config': {self.stage1_ckpt_path}")
        if "weight" not in ckpt:
            raise KeyError(f"Stage-1 checkpoint missing 'weight': {self.stage1_ckpt_path}")
        return ckpt

    def _load_lightning_and_model(self) -> tuple[nn.Module, nn.Module]:
        from AR.models.t2s_lightning_module import Text2SemanticLightningModule

        ckpt = self._load_ckpt()
        cfg = ckpt["config"]
        self.ckpt_config = cfg

        output_dir = Path(tempfile.mkdtemp(prefix="stage1_v670_"))
        lightning_module = Text2SemanticLightningModule(
            config=cfg,
            output_dir=output_dir,
            is_train=False,
        )
        self.load_msg = lightning_module.load_state_dict(ckpt["weight"], strict=self.load_strict)
        print(f"[Stage1FewshotAdapter] load_state_dict: {self.load_msg}")

        if self.use_half:
            lightning_module = lightning_module.half().to(self.device)
        else:
            lightning_module = lightning_module.to(self.device)

        model = lightning_module.model
        return lightning_module, model

    # ============================================================
    # Trainable scope
    # 可训练范围
    # ============================================================
    def _get_num_layers(self) -> int:
        model = self.model
        if hasattr(model, "num_layers"):
            return int(getattr(model, "num_layers"))
        if hasattr(model, "h") and hasattr(model.h, "layers"):
            return len(model.h.layers)
        raise AttributeError("Cannot infer number of stage-1 transformer layers from model.")

    def _is_head_param(self, name: str) -> bool:
        return name.startswith("ar_predict_layer.")

    def _is_bert_proj_param(self, name: str) -> bool:
        return name.startswith("bert_proj.")

    def _is_last_n_layer_param(self, name: str, last_n_layers: int) -> bool:
        if last_n_layers <= 0:
            return False
        num_layers = self._get_num_layers()
        start_idx = max(num_layers - int(last_n_layers), 0)
        for idx in range(start_idx, num_layers):
            if name.startswith(f"h.layers.{idx}."):
                return True
        return False

    def _scope_match(self, name: str, scope: str, last_n_layers: int) -> bool:
        if scope == "frozen_debug":
            return False
        if scope == "full_stage1_debug":
            return True
        if scope == "head_only":
            return self._is_head_param(name)
        if scope == "bert_proj_plus_head":
            return self._is_bert_proj_param(name) or self._is_head_param(name)
        if scope == "head_and_last_n":
            return self._is_head_param(name) or self._is_last_n_layer_param(name, last_n_layers)
        raise ValueError(f"Unsupported trainable scope: {scope!r}")

    def apply_trainable_scope(
        self,
        scope: Stage1TrainableScope,
        last_n_layers: int = 4,
    ) -> Stage1ParamReport:
        scope = str(scope)
        if scope not in self.SUPPORTED_SCOPES:
            raise ValueError(f"Unsupported trainable_scope={scope!r}. Supported: {sorted(self.SUPPORTED_SCOPES)}")
        last_n_layers = int(last_n_layers)

        for name, param in self.model.named_parameters():
            param.requires_grad = bool(self._scope_match(name, scope, last_n_layers))

        self.param_report = self.build_param_report(scope=scope, last_n_layers=last_n_layers)
        return self.param_report

    def build_param_report(self, scope: str, last_n_layers: int) -> Stage1ParamReport:
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        num_trainable_tensors = 0
        num_frozen_tensors = 0
        trainable_names: list[str] = []

        for name, param in self.model.named_parameters():
            n = int(param.numel())
            total_params += n
            if param.requires_grad:
                trainable_params += n
                num_trainable_tensors += 1
                if len(trainable_names) < 50:
                    trainable_names.append(name)
            else:
                frozen_params += n
                num_frozen_tensors += 1

        ratio = float(trainable_params / total_params) if total_params > 0 else 0.0
        return Stage1ParamReport(
            scope=scope,
            last_n_layers=int(last_n_layers),
            total_params=total_params,
            trainable_params=trainable_params,
            frozen_params=frozen_params,
            trainable_ratio=ratio,
            num_trainable_tensors=num_trainable_tensors,
            num_frozen_tensors=num_frozen_tensors,
            trainable_names_preview=trainable_names,
        )

    def get_param_report(self) -> dict[str, Any]:
        if self.param_report is None:
            self.param_report = self.build_param_report(scope="unknown", last_n_layers=0)
        return self.param_report.to_dict()

    def iter_trainable_parameters(self):
        for param in self.model.parameters():
            if param.requires_grad:
                yield param

    # ============================================================
    # Batch helpers
    # Batch 辅助函数
    # ============================================================
    def move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
                if key == "bert_feature" and self.use_half:
                    value = value.half()
                elif key == "bert_feature":
                    value = value.float()
            out[key] = value
        return out

    def forward_loss(self, batch: dict[str, Any]) -> Stage1ForwardOutput:
        """
        EN:
        Run one original GPT-SoVITS-style teacher-forcing loss call.

        ZH:
        执行一次原 GPT-SoVITS 风格的 teacher-forcing loss 计算。
        """
        batch = self.move_batch_to_device(batch)
        self.model.train()

        loss, acc = self.model.forward_old(
            batch["phoneme_ids"],
            batch["phoneme_ids_len"],
            batch["semantic_ids"],
            batch["semantic_ids_len"],
            batch["bert_feature"],
        )

        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"forward_old loss must be Tensor, got {type(loss)}")

        return Stage1ForwardOutput(
            loss=loss,
            loss_value=float(loss.detach().float().cpu().item()),
            top_k_acc=float(acc),
            top_k=int(getattr(self.lightning_module, "top_k", 3)),
            batch_size=int(batch["phoneme_ids"].shape[0]),
            phoneme_len_max=int(batch["phoneme_ids"].shape[1]),
            semantic_len_max=int(batch["semantic_ids"].shape[1]),
        )

    # ============================================================
    # State helpers
    # 状态辅助函数
    # ============================================================
    def save_stage1_checkpoint(
        self,
        path: str | Path,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        """
        EN:
        Save a GPT-SoVITS-compatible stage-1 checkpoint layout.

        ZH:
        保存为 GPT-SoVITS 兼容的 stage1 checkpoint 结构。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": "v6.7.0",
            "base_stage1_ckpt": self.stage1_ckpt_path,
            "param_report": self.get_param_report(),
        }
        if extra_meta:
            meta.update(extra_meta)
        torch.save(
            {
                "weight": self.lightning_module.state_dict(),
                "config": self.ckpt_config,
                "info": "VoiceLab-v6.7.0-stage1-fewshot-adapter",
                "voicelab_meta": meta,
            },
            path,
        )


__all__ = [
    "Stage1TrainableScope",
    "Stage1ParamReport",
    "Stage1ForwardOutput",
    "Stage1FewshotAdapter",
]
