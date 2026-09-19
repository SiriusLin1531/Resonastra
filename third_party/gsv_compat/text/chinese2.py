import os
import re
from pathlib import Path

import cn2an
from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from text.symbols import punctuation
from text.tone_sandhi import ToneSandhi
from text.zh_normalization.text_normlization import TextNormalizer

# ============================================================
# Basic normalization helpers
# 基础规范化辅助函数
# ============================================================

# EN:
# Convert Arabic numerals to Chinese numerals when needed.
#
# ZH:
# 在需要时把阿拉伯数字转换成中文数字。
normalizer = lambda x: cn2an.transform(x, "an2cn")

current_file_path = os.path.dirname(__file__)

# EN:
# Load pinyin-to-symbol mapping table used by GPT-SoVITS Chinese frontend.
#
# ZH:
# 加载 GPT-SoVITS 中文前端所使用的拼音到音素符号映射表。
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(
        os.path.join(current_file_path, "opencpop-strict.txt"),
        encoding="utf-8",
    ).readlines()
}

# ============================================================
# jieba / jieba_fast compatibility
# jieba / jieba_fast 兼容层
# ============================================================

# EN:
# Original GPT-SoVITS code often hard-depends on jieba_fast for speed.
# However, on Windows it is common that jieba_fast fails to build.
#
# To make the compatibility layer robust:
# 1. Prefer jieba_fast if available
# 2. Fall back to regular jieba if jieba_fast is unavailable
#
# ZH:
# 原 GPT-SoVITS 代码常常默认依赖 jieba_fast 来提升速度。
# 但在 Windows 环境中，jieba_fast 经常会因为本地编译失败而无法安装。
#
# 为了让兼容层更稳：
# 1. 优先使用 jieba_fast
# 2. 如果没有 jieba_fast，则自动回退到普通 jieba
import logging

try:
    import jieba_fast as jieba
    import jieba_fast.posseg as psg
except ImportError:
    import jieba
    import jieba.posseg as psg

jieba.setLogLevel(logging.CRITICAL)

# ============================================================
# Path resolution helpers for G2PW / BERT
# G2PW / BERT 路径解析辅助函数
# ============================================================


def _get_project_root() -> Path:
    """
    EN:
    Resolve project root in a way that matches the current VoiceLab layout.

    Priority:
    1. Environment variable `gsv_project_root`
    2. Infer from current file path:
       .../third_party/gsv_compat/text/chinese2.py
       parents[3] -> project root

    ZH:
    按照当前 VoiceLab 的目录结构解析项目根目录。

    优先级：
    1. 环境变量 `gsv_project_root`
    2. 根据当前文件路径反推：
       .../third_party/gsv_compat/text/chinese2.py
       parents[3] -> 项目根目录
    """
    env_root = os.environ.get("gsv_project_root")
    if env_root:
        return Path(env_root).resolve()

    current_file = Path(__file__).resolve()
    try:
        return current_file.parents[3]
    except IndexError:
        return Path.cwd().resolve()


def _resolve_g2pw_model_dir() -> str:
    """
    EN:
    Resolve G2PW model directory.

    Preferred target in current project layout:
        <project_root>/GPT_SoVITS/text/G2PWModel

    ZH:
    解析 G2PW 模型目录。

    当前项目结构下的优先目标路径为：
        <project_root>/GPT_SoVITS/text/G2PWModel
    """
    project_root = _get_project_root()
    candidates = [
        project_root / "GPT_SoVITS" / "text" / "G2PWModel",
        Path("GPT_SoVITS") / "text" / "G2PWModel",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    # EN:
    # If nothing exists yet, still return the most likely modern target path
    # so that future error messages are meaningful.
    #
    # ZH:
    # 如果当前都不存在，仍返回最可能的现代项目路径，
    # 这样后续报错信息也更有意义。
    return str(project_root / "GPT_SoVITS" / "text" / "G2PWModel")


def _resolve_bert_model_source() -> str:
    """
    EN:
    Resolve BERT model directory for G2PW.

    Priority:
    1. Environment variable `bert_path`
    2. <project_root>/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
    3. Relative fallback under current working directory

    ZH:
    解析 G2PW 所需的 BERT 模型目录。

    优先级：
    1. 环境变量 `bert_path`
    2. <project_root>/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
    3. 当前工作目录下的相对回退路径
    """
    env_bert = os.environ.get("bert_path")
    if env_bert:
        return env_bert

    project_root = _get_project_root()
    candidates = [
        project_root / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large",
        Path("GPT_SoVITS") / "pretrained_models" / "chinese-roberta-wwm-ext-large",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return str(project_root / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large")


# ============================================================
# G2PW setup
# G2PW 初始化
# ============================================================

# is_g2pw_str = os.environ.get("is_g2pw", "True")  # 默认开启
# is_g2pw = False  # True if is_g2pw_str.lower() == 'true' else False
is_g2pw = True  # True if is_g2pw_str.lower() == 'true' else False

if is_g2pw:
    # EN:
    # Use G2PW for Chinese polyphone disambiguation.
    #
    # ZH:
    # 使用 G2PW 做中文多音字消歧。
    from text.g2pw import G2PWPinyin, correct_pronunciation

    g2pw = G2PWPinyin(
        model_dir=_resolve_g2pw_model_dir(),
        model_source=_resolve_bert_model_source(),
        v_to_u=False,
        neutral_tone_with_five=True,
    )

# ============================================================
# Punctuation mapping
# 标点映射
# ============================================================

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "/": ",",
    "—": "-",
    "~": "…",
    "～": "…",
}

tone_modifier = ToneSandhi()


def replace_punctuation(text):
    """
    EN:
    Normalize punctuation and keep only Chinese characters + supported punctuation.

    ZH:
    规范化标点，并只保留中文字符与允许的标点。
    """
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    replaced_text = re.sub(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text)

    return replaced_text


def g2p(text):
    """
    EN:
    Convert normalized Chinese text into phone sequence and word2ph mapping.

    ZH:
    把规范化后的中文文本转换成音素序列和 word2ph 映射。
    """
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, word2ph = _g2p(sentences)
    return phones, word2ph


def _get_initials_finals(word):
    """
    EN:
    Get initials and finals from pypinyin baseline.

    ZH:
    使用 pypinyin 基线获取声母和韵母。
    """
    initials = []
    finals = []

    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)

    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


must_erhua = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"}
not_erhua = {
    "虐儿",
    "为儿",
    "护儿",
    "瞒儿",
    "救儿",
    "替儿",
    "有儿",
    "一儿",
    "我儿",
    "俺儿",
    "妻儿",
    "拐儿",
    "聋儿",
    "乞儿",
    "患儿",
    "幼儿",
    "孤儿",
    "婴儿",
    "婴幼儿",
    "连体儿",
    "脑瘫儿",
    "流浪儿",
    "体弱儿",
    "混血儿",
    "蜜雪儿",
    "舫儿",
    "祖儿",
    "美儿",
    "应采儿",
    "可儿",
    "侄儿",
    "孙儿",
    "侄孙儿",
    "女儿",
    "男儿",
    "红孩儿",
    "花儿",
    "虫儿",
    "马儿",
    "鸟儿",
    "猪儿",
    "猫儿",
    "狗儿",
    "少儿",
}


def _merge_erhua(initials: list[str], finals: list[str], word: str, pos: str) -> list[list[str]]:
    """
    EN:
    Apply erhua (儿化) handling.

    ZH:
    处理儿化音。
    """
    # fix er1
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "儿" and phn == "er1":
            finals[i] = "er2"

    # 发音
    if word not in must_erhua and (word in not_erhua or pos in {"a", "j", "nr"}):
        return initials, finals

    # "……" 等情况直接返回
    if len(finals) != len(word):
        return initials, finals

    assert len(finals) == len(word)

    # 与前一个字发同音
    new_initials = []
    new_finals = []
    for i, phn in enumerate(finals):
        if (
            i == len(finals) - 1
            and word[i] == "儿"
            and phn in {"er2", "er5"}
            and word[-2:] not in not_erhua
            and new_finals
        ):
            phn = "er" + new_finals[-1][-1]

        new_initials.append(initials[i])
        new_finals.append(phn)

    return new_initials, new_finals


def _g2p(segments):
    """
    EN:
    Core Chinese G2P path.

    ZH:
    中文 G2P 的核心流程。
    """
    phones_list = []
    word2ph = []

    for seg in segments:
        pinyins = []

        # EN:
        # Remove English words in Chinese G2P stage.
        #
        # ZH:
        # 中文 G2P 阶段先去掉英文词。
        seg = re.sub("[a-zA-Z]+", "", seg)

        seg_cut = psg.lcut(seg)
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)

        initials = []
        finals = []

        if not is_g2pw:
            for word, pos in seg_cut:
                if pos == "eng":
                    continue

                sub_initials, sub_finals = _get_initials_finals(word)
                sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)

                # 儿化
                sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)

                initials.append(sub_initials)
                finals.append(sub_finals)

            initials = sum(initials, [])
            finals = sum(finals, [])
            # print("pypinyin结果", initials, finals)

        else:
            # EN:
            # G2PW runs on the whole sentence for polyphone disambiguation.
            #
            # ZH:
            # G2PW 采用整句推理进行多音字消歧。
            pinyins = g2pw.lazy_pinyin(seg, neutral_tone_with_five=True, style=Style.TONE3)

            pre_word_length = 0
            for word, pos in seg_cut:
                sub_initials = []
                sub_finals = []
                now_word_length = pre_word_length + len(word)

                if pos == "eng":
                    pre_word_length = now_word_length
                    continue

                word_pinyins = pinyins[pre_word_length:now_word_length]

                # 多音字消歧
                word_pinyins = correct_pronunciation(word, word_pinyins)

                for pinyin in word_pinyins:
                    if pinyin and pinyin[0].isalpha():
                        sub_initials.append(to_initials(pinyin))
                        sub_finals.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
                    else:
                        sub_initials.append(pinyin)
                        sub_finals.append(pinyin)

                pre_word_length = now_word_length

                sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)

                # 儿化
                sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)

                initials.append(sub_initials)
                finals.append(sub_finals)

            initials = sum(initials, [])
            finals = sum(finals, [])
            # print("g2pw结果", initials, finals)

        for c, v in zip(initials, finals):
            raw_pinyin = c + v

            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:
                assert c in punctuation
                phone = [c]
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
                new_v = new_v + tone
                phone = [new_c, new_v]
                word2ph.append(len(phone))

            phones_list += phone

    return phones_list, word2ph


def replace_punctuation_with_en(text):
    """
    EN:
    Normalize punctuation while keeping English letters.

    ZH:
    规范化标点，同时保留英文字符。
    """
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    replaced_text = re.sub(r"[^\u4e00-\u9fa5A-Za-z" + "".join(punctuation) + r"]+", "", replaced_text)

    return replaced_text


def replace_consecutive_punctuation(text):
    """
    EN:
    Collapse repeated punctuation to reduce leakage / instability.

    ZH:
    折叠连续重复标点，减少参考泄露和不稳定性。
    """
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def text_normalize(text):
    """
    EN:
    Chinese text normalization entry.

    Based on PaddleSpeech zh_normalization pipeline.

    ZH:
    中文文本规范化入口。

    基于 PaddleSpeech 的 zh_normalization 逻辑。
    """
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)

    # 避免重复标点引起的参考泄露
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


if __name__ == "__main__":
    text = "啊——但是《原神》是由,米哈\游自主，研发的一款全.新开放世界.冒险游戏"
    text = "呣呣呣～就是…大人的鼹鼠党吧？"
    text = "你好"
    text = text_normalize(text)
    print(g2p(text))

    # 示例用法
    # text = "这是一个示例文本：,你好！这是一个测试..."
    # print(g2p(text))
