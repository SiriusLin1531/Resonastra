from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Stage2Inputs:
    """
    EN:
    Unified input structure for the new stage-2 acoustic model.

    v6.4 additions:
    - semantic_source_ids:
        Explicitly tells the model / trainer where the selected semantic
        condition comes from.
    - semantic_reliability:
        A soft reliability prior for the selected semantic condition.
    - oracle_semantic_tokens / pred_semantic_tokens:
        Optional paired semantic streams reserved for source-aware training,
        diagnostics, future paired losses, and teacher-target generation.
    - semantic_continuous:
        Optional GPT-SoVITS quantizer.decode continuous semantic vectors,
        reserved for v6.4.1.

    ZH:
    新第二阶段声学生成模型的统一输入结构。

    v6.4 新增：
    - semantic_source_ids:
        显式告诉模型 / 训练器当前选中的 semantic 条件来自哪里。
    - semantic_reliability:
        当前 semantic 条件的软可信度先验。
    - oracle_semantic_tokens / pred_semantic_tokens:
        预留的成对 semantic 流，用于 source-aware training、诊断、
        后续 paired loss 和 teacher target 生成。
    - semantic_continuous:
        预留给 v6.4.1 的 GPT-SoVITS quantizer.decode 连续 semantic 表示。
    """

    # ------------------------------------------------------------
    # Required selected semantic input
    # 当前实际送入 Stage2 的 semantic 条件
    # ------------------------------------------------------------
    semantic_tokens: torch.Tensor  # (B, T_sem), int64

    # Optional selected semantic length info
    # 当前实际 semantic 的长度
    semantic_lengths: torch.Tensor | None = None  # (B,)

    # ------------------------------------------------------------
    # v6.4 source-aware semantic metadata
    # v6.4 semantic 来源感知信息
    # ------------------------------------------------------------
    semantic_source_ids: torch.Tensor | None = None
    # (B,)
    # Suggested convention:
    #   0 = oracle
    #   1 = predicted
    #   2 = noised_oracle
    #   3 = aligned_predicted

    semantic_reliability: torch.Tensor | None = None
    # (B,), float in [0, 1]
    # A soft prior describing how trustworthy the selected semantic condition is.

    # ------------------------------------------------------------
    # v6.4 paired semantic streams
    # v6.4 成对 semantic 流
    # ------------------------------------------------------------
    oracle_semantic_tokens: torch.Tensor | None = None
    oracle_semantic_lengths: torch.Tensor | None = None

    pred_semantic_tokens: torch.Tensor | None = None
    pred_semantic_lengths: torch.Tensor | None = None

    # ------------------------------------------------------------
    # v6.4.1 future hook: continuous semantic representation
    # v6.4.1 预留：连续 semantic 表示
    # ------------------------------------------------------------
    semantic_continuous: torch.Tensor | None = None
    semantic_continuous_lengths: torch.Tensor | None = None
    # Suggested future shape:
    #   semantic_continuous: (B, T_sem, C_sem_cont)
    #   semantic_continuous_lengths: (B,)

    oracle_semantic_continuous: torch.Tensor | None = None
    pred_semantic_continuous: torch.Tensor | None = None

    oracle_semantic_continuous_lengths: torch.Tensor | None = None
    pred_semantic_continuous_lengths: torch.Tensor | None = None

    # ------------------------------------------------------------
    # Optional text-side conditions
    # 文本侧可选条件
    # ------------------------------------------------------------
    phoneme_ids: torch.Tensor | None = None       # (B, T_text)
    phoneme_lens: torch.Tensor | None = None      # (B,)
    bert_feature: torch.Tensor | None = None      # (B, 1024, T_text)

    # ------------------------------------------------------------
    # Optional prompt/reference-side conditions
    # prompt / 参考音频侧可选条件
    # ------------------------------------------------------------
    prompt_tokens: torch.Tensor | None = None     # (B, T_prompt)
    prompt_lengths: torch.Tensor | None = None    # (B,)

    # ------------------------------------------------------------
    # Future extension hooks
    # 未来扩展预留
    # ------------------------------------------------------------
    speaker_embedding: torch.Tensor | None = None
    style_embedding: torch.Tensor | None = None

    # ------------------------------------------------------------
    # Training targets
    # 训练目标
    # ------------------------------------------------------------
    target_acoustic: torch.Tensor | None = None   # (B, T_ac, C)
    target_lengths: torch.Tensor | None = None    # (B,)

    # ------------------------------------------------------------
    # Metadata
    # 辅助元信息
    # ------------------------------------------------------------
    norm_text: str | None = None
    raw_text: str | None = None
    language: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage2Outputs:
    """
    EN:
    Unified output structure for the new stage-2 acoustic model.

    ZH:
    新第二阶段声学生成模型的统一输出结构。
    """

    acoustic: torch.Tensor
    acoustic_type: str
    lengths: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)