from __future__ import annotations

import re
from dataclasses import dataclass

from .emilia_item import EmiliaRawItem


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class EmiliaFilterConfig:
    """
    EN:
    Filtering rules for Emilia raw items.

    ZH:
    Emilia 原始样本过滤规则。
    """

    language: str = "zh"

    min_duration: float = 3.0
    max_duration: float = 15.0

    min_dnsmos: float | None = 3.2
    min_chinese_ratio: float = 0.7

    require_speaker: bool = True
    require_text: bool = True


def normalize_language(x: str | None) -> str:
    """
    EN:
    Normalize language string.

    ZH:
    规范化语言字段。
    """
    if x is None:
        return ""
    return str(x).strip().lower()


def chinese_char_ratio(text: str) -> float:
    """
    EN:
    Compute the ratio of Chinese characters among non-space visible characters.

    ZH:
    计算中文字符在非空白可见字符中的比例。
    """
    if not text:
        return 0.0

    visible_chars = [c for c in text if not c.isspace()]
    if not visible_chars:
        return 0.0

    num_cjk = sum(1 for c in visible_chars if _CJK_RE.match(c))
    return num_cjk / max(len(visible_chars), 1)


def filter_emilia_item(
    item: EmiliaRawItem,
    config: EmiliaFilterConfig,
) -> tuple[bool, str]:
    """
    EN:
    Return:
        keep: bool
        reason: "accepted" or reject reason

    ZH:
    返回：
        keep: 是否保留
        reason: "accepted" 或拒绝原因
    """
    expected_lang = normalize_language(config.language)
    item_lang = normalize_language(item.language)

    if expected_lang and item_lang != expected_lang:
        return False, "language_mismatch"

    if config.require_text and not str(item.text or "").strip():
        return False, "missing_text"

    if config.require_speaker and not str(item.speaker or "").strip():
        return False, "missing_speaker"

    duration = float(item.duration_sec or 0.0)

    if duration < config.min_duration:
        return False, "duration_too_short"

    if duration > config.max_duration:
        return False, "duration_too_long"

    if config.min_dnsmos is not None:
        if item.dnsmos is None:
            return False, "missing_dnsmos"

        if float(item.dnsmos) < float(config.min_dnsmos):
            return False, "dnsmos_too_low"

    ratio = chinese_char_ratio(item.text)

    if ratio < config.min_chinese_ratio:
        return False, "chinese_ratio_too_low"

    return True, "accepted"