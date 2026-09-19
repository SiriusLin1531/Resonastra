from __future__ import annotations

# ============================================================
# Standard / third-party imports
# 标准库 / 第三方库导入
# ============================================================
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

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
# SegmentFrontendOutput
#
# EN:
# Per-segment intermediate result. This is useful for debugging multilingual
# or mixed-language text processing.
#
# ZH:
# 单个分段的中间处理结果，方便以后调试多语言/混合语言文本前端。
# ---------------------------------------------------------------------
@dataclass
class SegmentFrontendOutput:
    seg_text: str
    seg_lang: str
    phones: list[str]
    phoneme_ids: list[int]
    word2ph: list[int] | None
    norm_text: str
    bert_feature: torch.Tensor  # (C, T_phone)


# ---------------------------------------------------------------------
# FrontendOutput
#
# EN:
# Final merged frontend result.
#
# ZH:
# 最终拼接后的文本前端输出结果。
# ---------------------------------------------------------------------
@dataclass
class FrontendOutput:
    raw_text: str
    language: str
    version: str
    phones: list[str]
    phoneme_ids: list[int]
    word2ph: list[int] | None
    norm_text: str
    bert_feature: torch.Tensor  # (C, T_phone)

    # EN:
    # Keep all segment-level intermediate results for future debugging.
    #
    # ZH:
    # 保留每个分段的中间结果，方便以后调试。
    segment_infos: list[SegmentFrontendOutput] = field(default_factory=list)


class GSVTextFrontend:
    """
    EN:
    Thin wrapper around GPT-SoVITS text frontend.

    Phase-1 policy:
    - hard-locked to v2
    - Chinese-first support
    - multilingual / mixed-language behavior is best-effort under v2 frontend assumptions

    This class preserves the original GPT-SoVITS frontend behavior:
    - clean_text(...)
    - cleaned_text_to_sequence(...)
    - LangSegmenter.getTexts(...)
    - Chinese BERT feature construction

    ZH:
    这是对 GPT-SoVITS 文本前端的轻量包装。

    第一阶段兼容策略：
    - 强锁定为 v2
    - 以中文优先为正式支持目标
    - 多语言/混合语言能力按 v2 前端逻辑做 best-effort 支持

    这个类尽量保留原 GPT-SoVITS 的文本前端行为：
    - clean_text(...)
    - cleaned_text_to_sequence(...)
    - LangSegmenter.getTexts(...)
    - 中文路径下的 BERT 特征构造
    """

    # EN:
    # Officially supported language modes for phase-1 compatibility.
    #
    # ZH:
    # 第一阶段兼容层正式支持/允许的 language 参数。
    SUPPORTED_LANGUAGE_MODES = {
        "zh",
        "en",
        "ja",
        "ko",
        "yue",
        "auto",
        "auto_yue",
        "all_zh",
        "all_yue",
        "all_ja",
        "all_ko",
    }

    def __init__(
        self,
        version: str = SUPPORTED_GSV_VERSION,
        device: str | torch.device = "cpu",
        use_half: bool = False,
        project_root: str | None = None,
    ) -> None:
        # EN:
        # Compatibility layer phase 1 is strictly v2-only.
        #
        # ZH:
        # 兼容层第一期严格只支持 v2。
        if version != SUPPORTED_GSV_VERSION:
            raise ValueError(
                f"GSVTextFrontend is locked to {SUPPORTED_GSV_VERSION!r} in phase 1, "
                f"but got version={version!r}."
            )

        self.env: GSVEnvInfo = setup_gsv_env(version=version, project_root=project_root)

        # EN:
        # Text frontend requires at least:
        # - bert_path
        # - G2PWModel
        #
        # ZH:
        # 文本前端至少要求：
        # - bert_path
        # - G2PWModel
        ensure_gsv_assets_exist(
            self.env,
            require_stage1=False,
            require_prompt_tokenizer=False,
            require_split_lang=False,
            require_fast_langdetect=False,
        )

        self.version = version
        self.device = torch.device(device)
        self.use_half = bool(use_half) and str(self.device).startswith("cuda")

        # EN:
        # Lazy importers after compatibility env is ready, so original imports
        # like `from text.cleaner importers clean_text` keep working.
        #
        # ZH:
        # 必须等兼容环境准备好以后再导入，这样原项目里的
        # `from text.cleaner importers clean_text` 才能正常工作。
        from text.cleaner import clean_text
        from text import cleaned_text_to_sequence
        from text.LangSegmenter import LangSegmenter

        self._clean_text = clean_text
        self._cleaned_text_to_sequence = cleaned_text_to_sequence
        self._LangSegmenter = LangSegmenter

        # EN:
        # BERT tokenizer/model are loaded lazily on first real use.
        #
        # ZH:
        # BERT tokenizer / model 采用懒加载策略，第一次真正用到时才初始化。
        self._tokenizer = None
        self._bert_model = None

        # EN:
        # The shipped Chinese RoBERTa usually uses hidden size 1024.
        # We still update it from config after actual model load.
        #
        # ZH:
        # 配套中文 RoBERTa 一般 hidden size 是 1024。
        # 实际加载后仍会从模型配置中重新更新。
        self._bert_hidden_size = 1024

    # ============================================================
    # BERT helpers / BERT 辅助函数
    # ============================================================
    def _lazy_load_bert(self) -> None:
        """
        EN:
        Load tokenizer/model only when needed.

        ZH:
        只有在真正需要时才加载 tokenizer 和 BERT 模型。
        """
        if self._tokenizer is not None and self._bert_model is not None:
            return

        bert_path = self.env.bert_path
        self._tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self._bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

        if self.use_half:
            self._bert_model = self._bert_model.half().to(self.device)
        else:
            self._bert_model = self._bert_model.to(self.device)

        hidden_size = getattr(self._bert_model.config, "hidden_size", None)
        if isinstance(hidden_size, int):
            self._bert_hidden_size = hidden_size

        self._bert_model.eval()

    def _get_bert_feature(self, text: str, word2ph: list[int]) -> torch.Tensor:
        """
        EN:
        Build phone-level BERT feature for Chinese text.

        This follows GPT-SoVITS style:
        1. tokenize normalized text
        2. run BERT
        3. collect hidden state
        4. expand token-level feature to phone-level feature using word2ph

        Output shape:
            (C, T_phone)

        ZH:
        为中文文本构造音素级 BERT 特征。

        这个过程对齐 GPT-SoVITS 原逻辑：
        1. 对规范化文本做 tokenizer
        2. 运行 BERT
        3. 取中间隐藏层
        4. 按 word2ph 把“字/子词级特征”扩展到“音素级特征”

        输出形状：
            (C, T_phone)
        """
        self._lazy_load_bert()

        assert self._tokenizer is not None
        assert self._bert_model is not None

        # EN:
        # Original Chinese frontend assumes word2ph length equals normalized text length.
        #
        # ZH:
        # 原始中文前端默认假设 word2ph 长度与规范化文本长度一致。
        assert len(word2ph) == len(text), (
            f"word2ph length ({len(word2ph)}) must equal normalized text length ({len(text)})."
        )

        with torch.no_grad():
            inputs = self._tokenizer(text, return_tensors="pt")
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)

            outputs = self._bert_model(**inputs, output_hidden_states=True)

            # EN:
            # Follow GPT-SoVITS hidden-state extraction style.
            #
            # ZH:
            # 对齐 GPT-SoVITS 的隐藏层提取方式。
            hidden = torch.cat(outputs["hidden_states"][-3:-2], dim=-1)[0].detach().cpu()[1:-1]

        phone_level_feature = []
        for i, repeat_n in enumerate(word2ph):
            repeat_feature = hidden[i].repeat(repeat_n, 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)  # (T_phone, C)
        return phone_level_feature.T.contiguous()  # (C, T_phone)

    def _get_bert_inf(
        self,
        phones: list[int],
        word2ph: list[int] | None,
        norm_text: str,
        language: str,
    ) -> torch.Tensor:
        """
        EN:
        Match GPT-SoVITS inference convention:
        - zh -> real BERT feature
        - others -> zero feature

        ZH:
        对齐 GPT-SoVITS 推理约定：
        - 中文 -> 使用真实 BERT 特征
        - 其他语言 -> 返回零特征
        """
        language = language.replace("all_", "")

        if language == "zh":
            if word2ph is None:
                raise ValueError("word2ph is required for zh BERT feature extraction.")
            feat = self._get_bert_feature(norm_text, word2ph)
        else:
            feat = torch.zeros((self._bert_hidden_size, len(phones)), dtype=torch.float32)

        if self.use_half:
            feat = feat.half()
        return feat.to(self.device)

    # ============================================================
    # Text segmentation and cleanup / 文本切分与清洗
    # ============================================================
    def _clean_text_inf(self, text: str, language: str) -> tuple[list[int], list[int] | None, str, list[str]]:
        """
        EN:
        Run:
            phones, word2ph, norm_text = clean_text(...)
            phoneme_ids = cleaned_text_to_sequence(...)

        ZH:
        执行：
            phones, word2ph, norm_text = clean_text(...)
            phoneme_ids = cleaned_text_to_sequence(...)
        """
        language = language.replace("all_", "")
        phones, word2ph, norm_text = self._clean_text(text, language, self.version)
        phoneme_ids = self._cleaned_text_to_sequence(phones, self.version)
        return phoneme_ids, word2ph, norm_text, phones

    def _segment_text(self, text: str, language: str) -> list[dict[str, str]]:
        """
        EN:
        Segment text according to GPT-SoVITS-style language policies.

        ZH:
        按照 GPT-SoVITS 风格的语言策略对文本做切分。
        """
        if language not in self.SUPPORTED_LANGUAGE_MODES:
            raise ValueError(
                f"Unsupported language mode: {language!r}. "
                f"Supported modes: {sorted(self.SUPPORTED_LANGUAGE_MODES)}"
            )

        text = re.sub(r" {2,}", " ", text)

        if language == "all_zh":
            return self._LangSegmenter.getTexts(text, "zh")

        if language == "all_yue":
            segments = self._LangSegmenter.getTexts(text, "zh")
            for seg in segments:
                if seg["lang"] == "zh":
                    seg["lang"] = "yue"
            return segments

        if language == "all_ja":
            return self._LangSegmenter.getTexts(text, "ja")

        if language == "all_ko":
            return self._LangSegmenter.getTexts(text, "ko")

        if language == "en":
            return [{"lang": "en", "text": text}]

        if language == "auto":
            return self._LangSegmenter.getTexts(text)

        if language == "auto_yue":
            segments = self._LangSegmenter.getTexts(text)
            for seg in segments:
                if seg["lang"] == "zh":
                    seg["lang"] = "yue"
            return segments

        # EN:
        # Mixed mode for zh / ja / ko / yue:
        # keep English as "en", force all other non-English segments into target language.
        #
        # ZH:
        # zh / ja / ko / yue 混合模式：
        # 英文保留为 en，其余非英文段统一改成目标语言。
        segments = []
        for seg in self._LangSegmenter.getTexts(text):
            if segments:
                prev = segments[-1]
                if (seg["lang"] == "en" and prev["lang"] == "en") or (
                    seg["lang"] != "en" and prev["lang"] != "en"
                ):
                    prev["text"] += seg["text"]
                    continue

            if seg["lang"] == "en":
                segments.append(seg)
            else:
                segments.append({"lang": language, "text": seg["text"]})

        return segments

    # ============================================================
    # Public APIs / 对外接口
    # ============================================================
    def prepare_inputs(self, text: str, language: str = "zh") -> FrontendOutput:
        """
        EN:
        Full frontend path:
            raw text
            -> segmentation
            -> clean_text
            -> phoneme ids
            -> BERT feature
            -> merge all segments

        ZH:
        完整文本前端路径：
            原始文本
            -> 切分
            -> clean_text
            -> phoneme ids
            -> BERT 特征
            -> 拼接所有分段结果
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input text must be a non-empty string.")

        segments = self._segment_text(text, language)

        all_phones_symbols: list[str] = []
        all_phone_ids: list[int] = []
        all_norm_text: list[str] = []
        bert_list: list[torch.Tensor] = []
        segment_infos: list[SegmentFrontendOutput] = []

        for seg in segments:
            seg_lang = seg["lang"]
            seg_text = seg["text"]

            phone_ids, word2ph, norm_text, phone_symbols = self._clean_text_inf(seg_text, seg_lang)
            bert_feat = self._get_bert_inf(phone_ids, word2ph, norm_text, seg_lang)

            all_phones_symbols.extend(phone_symbols)
            all_phone_ids.extend(phone_ids)
            all_norm_text.append(norm_text)
            bert_list.append(bert_feat)

            segment_infos.append(
                SegmentFrontendOutput(
                    seg_text=seg_text,
                    seg_lang=seg_lang,
                    phones=phone_symbols,
                    phoneme_ids=phone_ids,
                    word2ph=word2ph,
                    norm_text=norm_text,
                    bert_feature=bert_feat,
                )
            )

        if len(bert_list) == 0:
            raise ValueError("No valid text segments were produced by the frontend.")

        bert = torch.cat(bert_list, dim=1)
        norm_text = "".join(all_norm_text)

        # EN:
        # Keep the original safeguard:
        # if phone sequence is too short, prepend a dot once and retry.
        #
        # ZH:
        # 保留原风格保护逻辑：
        # 如果音素序列太短，则在前面补一个点号后重试一次。
        if len(all_phone_ids) < 6 and not text.startswith("."):
            return self.prepare_inputs("." + text, language=language)

        return FrontendOutput(
            raw_text=text,
            language=language,
            version=self.version,
            phones=all_phones_symbols,
            phoneme_ids=all_phone_ids,
            word2ph=None,  # EN: not aggregated globally / ZH: 全局聚合后不再直接返回
            norm_text=norm_text,
            bert_feature=bert,
            segment_infos=segment_infos,
        )

    def prepare_ids_and_bert(
        self,
        text: str,
        language: str = "zh",
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, str]:
        """
        EN:
        Convenience wrapper for stage-1 inference.

        Returns:
            phoneme_ids:  (1, T)
            phoneme_lens: (1,)
            bert_feature: (1, C, T)
            norm_text: str

        ZH:
        给第一阶段推理使用的便捷包装接口。

        返回：
            phoneme_ids:  (1, T)
            phoneme_lens: (1,)
            bert_feature: (1, C, T)
            norm_text: str
        """
        out = self.prepare_inputs(text, language=language)

        phoneme_ids = torch.LongTensor(out.phoneme_ids).unsqueeze(0).to(self.device)
        phoneme_lens = torch.tensor([phoneme_ids.shape[-1]], dtype=torch.long, device=self.device)
        bert_feature = out.bert_feature.unsqueeze(0).to(self.device)

        return phoneme_ids, phoneme_lens, bert_feature, out.norm_text