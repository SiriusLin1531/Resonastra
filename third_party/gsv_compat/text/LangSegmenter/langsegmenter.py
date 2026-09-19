import logging
import os
import re
from pathlib import Path

# jieba静音
import jieba
jieba.setLogLevel(logging.CRITICAL)

import fast_langdetect
from split_lang import LangSplitter


def _resolve_fast_langdetect_cache_dir() -> Path:
    """
    EN:
    Resolve the fast-langdetect cache/model directory in a way that matches
    the current GameVoiceLab project layout.

    Search priority:
    1. Environment variable `gsv_project_root` -> GPT_SoVITS/pretrained_models/fast_langdetect
    2. Infer project root from current file path -> GPT_SoVITS/pretrained_models/fast_langdetect
    3. Legacy compatibility-root style path -> gsv_compat/pretrained_models/fast_langdetect

    ZH:
    按照当前 GameVoiceLab 项目的目录结构，解析 fast-langdetect 的缓存/模型目录。

    搜索优先级：
    1. 环境变量 `gsv_project_root` -> GPT_SoVITS/pretrained_models/fast_langdetect
    2. 根据当前文件路径反推项目根目录 -> GPT_SoVITS/pretrained_models/fast_langdetect
    3. 兼容旧路径风格 -> gsv_compat/pretrained_models/fast_langdetect
    """
    candidates: list[Path] = []

    # 1) Prefer explicit project root from environment
    env_root = os.environ.get("gsv_project_root")
    if env_root:
        candidates.append(Path(env_root).resolve() / "GPT_SoVITS" / "pretrained_models" / "fast_langdetect")

    # 2) Infer current project root from file location
    # Current file is expected to be:
    #   .../third_party/gsv_compat/text/LangSegmenter/langsegmenter.py
    # parents[4] -> project root
    current_file = Path(__file__).resolve()
    try:
        project_root = current_file.parents[4]
        candidates.append(project_root / "GPT_SoVITS" / "pretrained_models" / "fast_langdetect")
    except IndexError:
        pass

    # 3) Legacy style path under gsv_compat/pretrained_models/fast_langdetect
    # parents[2] -> gsv_compat
    try:
        compat_root = current_file.parents[2]
        candidates.append(compat_root / "pretrained_models" / "fast_langdetect")
    except IndexError:
        pass

    for path in candidates:
        if path.exists():
            return path

    # If none exists, return the most likely modern target path.
    if env_root:
        return Path(env_root).resolve() / "GPT_SoVITS" / "pretrained_models" / "fast_langdetect"

    try:
        return current_file.parents[4] / "GPT_SoVITS" / "pretrained_models" / "fast_langdetect"
    except IndexError:
        return Path("GPT_SoVITS") / "pretrained_models" / "fast_langdetect"


def _configure_fast_langdetect() -> None:
    """
    EN:
    Configure fast-langdetect default detector in a version-compatible way.

    We try the modern top-level API first:
        fast_langdetect.LangDetector
        fast_langdetect.LangDetectConfig

    If unavailable, fall back to the older style:
        fast_langdetect.infer.LangDetector
        fast_langdetect.infer.LangDetectConfig

    ZH:
    以兼容不同版本的方式配置 fast-langdetect 的默认 detector。

    先尝试当前更常见的顶层 API：
        fast_langdetect.LangDetector
        fast_langdetect.LangDetectConfig

    如果不可用，再回退到旧写法：
        fast_langdetect.infer.LangDetector
        fast_langdetect.infer.LangDetectConfig
    """
    cache_dir = _resolve_fast_langdetect_cache_dir()

    # Modern / top-level API
    if hasattr(fast_langdetect, "LangDetector") and hasattr(fast_langdetect, "LangDetectConfig"):
        detector = fast_langdetect.LangDetector(
            fast_langdetect.LangDetectConfig(cache_dir=cache_dir)
        )
        # Create/overwrite module-level default detector
        fast_langdetect._default_detector = detector
        return

    # Older API under `.infer`
    if hasattr(fast_langdetect, "infer"):
        infer_mod = fast_langdetect.infer
        if hasattr(infer_mod, "LangDetector") and hasattr(infer_mod, "LangDetectConfig"):
            infer_mod._default_detector = infer_mod.LangDetector(
                infer_mod.LangDetectConfig(cache_dir=cache_dir)
            )
            return

    raise AttributeError(
        "Unsupported fast_langdetect API. "
        "Neither top-level LangDetector/LangDetectConfig nor infer.LangDetector/LangDetectConfig was found."
    )


# Initialize fast-langdetect default detector once at importers time
_configure_fast_langdetect()


def full_en(text):
    pattern = r'^(?=.*[A-Za-z])[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$'
    return bool(re.match(pattern, text))


def full_cjk(text):
    # 来自wiki
    cjk_ranges = [
        (0x4E00, 0x9FFF),        # CJK Unified Ideographs
        (0x3400, 0x4DB5),        # CJK Extension A
        (0x20000, 0x2A6DD),      # CJK Extension B
        (0x2A700, 0x2B73F),      # CJK Extension C
        (0x2B740, 0x2B81F),      # CJK Extension D
        (0x2B820, 0x2CEAF),      # CJK Extension E
        (0x2CEB0, 0x2EBEF),      # CJK Extension F
        (0x30000, 0x3134A),      # CJK Extension G
        (0x31350, 0x323AF),      # CJK Extension H
        (0x2EBF0, 0x2EE5D),      # CJK Extension H
    ]

    pattern = r'[0-9、-〜。！？.!?… /]+$'

    cjk_text = ""
    for char in text:
        code_point = ord(char)
        in_cjk = any(start <= code_point <= end for start, end in cjk_ranges)
        if in_cjk or re.match(pattern, char):
            cjk_text += char
    return cjk_text


def split_jako(tag_lang, item):
    if tag_lang == "ja":
        pattern = r"([\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]+(?:[0-9、-〜。！？.!?… ]+[\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]*)*)"
    else:
        pattern = r"([\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+(?:[0-9、-〜。！？.!?… ]+[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]*)*)"

    lang_list: list[dict] = []
    tag = 0
    for match in re.finditer(pattern, item['text']):
        if match.start() > tag:
            lang_list.append({'lang': item['lang'], 'text': item['text'][tag:match.start()]})

        tag = match.end()
        lang_list.append({'lang': tag_lang, 'text': item['text'][match.start():match.end()]})

    if tag < len(item['text']):
        lang_list.append({'lang': item['lang'], 'text': item['text'][tag:len(item['text'])]})

    return lang_list


def merge_lang(lang_list, item):
    if lang_list and item['lang'] == lang_list[-1]['lang']:
        lang_list[-1]['text'] += item['text']
    else:
        lang_list.append(item)
    return lang_list


class LangSegmenter():
    # 默认过滤器, 基于gsv目前四种语言
    DEFAULT_LANG_MAP = {
        "zh": "zh",
        "yue": "zh",  # 粤语
        "wuu": "zh",  # 吴语
        "zh-cn": "zh",
        "zh-tw": "x",  # 繁体设置为x
        "ko": "ko",
        "ja": "ja",
        "en": "en",
    }

    def getTexts(text, default_lang=""):
        lang_splitter = LangSplitter(lang_map=LangSegmenter.DEFAULT_LANG_MAP)
        lang_splitter.merge_across_digit = False
        substr = lang_splitter.split_by_lang(text=text)

        lang_list: list[dict] = []

        have_num = False

        for _, item in enumerate(substr):
            dict_item = {'lang': item.lang, 'text': item.text}

            if dict_item['lang'] == 'digit':
                if default_lang != "":
                    dict_item['lang'] = default_lang
                else:
                    have_num = True
                lang_list = merge_lang(lang_list, dict_item)
                continue

            # 处理短英文被识别为其他语言的问题
            if full_en(dict_item['text']):
                dict_item['lang'] = 'en'
                lang_list = merge_lang(lang_list, dict_item)
                continue

            if default_lang != "":
                dict_item['lang'] = default_lang
                lang_list = merge_lang(lang_list, dict_item)
                continue
            else:
                # 处理非日语夹日文的问题(不包含CJK)
                ja_list: list[dict] = []
                if dict_item['lang'] != 'ja':
                    ja_list = split_jako('ja', dict_item)

                if not ja_list:
                    ja_list.append(dict_item)

                # 处理非韩语夹韩语的问题(不包含CJK)
                ko_list: list[dict] = []
                temp_list: list[dict] = []
                for _, ko_item in enumerate(ja_list):
                    if ko_item["lang"] != 'ko':
                        ko_list = split_jako('ko', ko_item)

                    if ko_list:
                        temp_list.extend(ko_list)
                    else:
                        temp_list.append(ko_item)

                # 未存在非日韩文夹日韩文
                if len(temp_list) == 1:
                    # 未知语言检查是否为CJK
                    if dict_item['lang'] == 'x':
                        cjk_text = full_cjk(dict_item['text'])
                        if cjk_text:
                            dict_item = {'lang': 'zh', 'text': cjk_text}
                            lang_list = merge_lang(lang_list, dict_item)
                        else:
                            lang_list = merge_lang(lang_list, dict_item)
                        continue
                    else:
                        lang_list = merge_lang(lang_list, dict_item)
                        continue

                # 存在非日韩文夹日韩文
                for _, temp_item in enumerate(temp_list):
                    # 未知语言检查是否为CJK
                    if temp_item['lang'] == 'x':
                        cjk_text = full_cjk(temp_item['text'])
                        if cjk_text:
                            lang_list = merge_lang(lang_list, {'lang': 'zh', 'text': cjk_text})
                        else:
                            lang_list = merge_lang(lang_list, temp_item)
                    else:
                        lang_list = merge_lang(lang_list, temp_item)

        # 有数字
        if have_num:
            temp_list = lang_list
            lang_list = []
            for i, temp_item in enumerate(temp_list):
                if temp_item['lang'] == 'digit':
                    if default_lang:
                        temp_item['lang'] = default_lang
                    elif lang_list and i == len(temp_list) - 1:
                        temp_item['lang'] = lang_list[-1]['lang']
                    elif not lang_list and i < len(temp_list) - 1:
                        temp_item['lang'] = temp_list[1]['lang']
                    elif lang_list and i < len(temp_list) - 1:
                        if lang_list[-1]['lang'] == temp_list[i + 1]['lang']:
                            temp_item['lang'] = lang_list[-1]['lang']
                        elif lang_list[-1]['text'][-1] in [",", ".", "!", "?", "，", "。", "！", "？"]:
                            temp_item['lang'] = temp_list[i + 1]['lang']
                        elif temp_list[i + 1]['text'][0] in [",", ".", "!", "?", "，", "。", "！", "？"]:
                            temp_item['lang'] = lang_list[-1]['lang']
                        elif temp_item['text'][-1] in ["。", "."]:
                            temp_item['lang'] = lang_list[-1]['lang']
                        elif len(lang_list[-1]['text']) >= len(temp_list[i + 1]['text']):
                            temp_item['lang'] = lang_list[-1]['lang']
                        else:
                            temp_item['lang'] = temp_list[i + 1]['lang']
                    else:
                        temp_item['lang'] = 'zh'

                lang_list = merge_lang(lang_list, temp_item)

        # 筛X
        temp_list = lang_list
        lang_list = []
        for _, temp_item in enumerate(temp_list):
            if temp_item['lang'] == 'x':
                if lang_list:
                    temp_item['lang'] = lang_list[-1]['lang']
                elif len(temp_list) > 1:
                    temp_item['lang'] = temp_list[1]['lang']
                else:
                    temp_item['lang'] = 'zh'

            lang_list = merge_lang(lang_list, temp_item)

        return lang_list


if __name__ == "__main__":
    text = "MyGO?,你也喜欢まいご吗？"
    print(LangSegmenter.getTexts(text))

    text = "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。君の瞳が星空みたいにキラキラしてるからさ。"
    print(LangSegmenter.getTexts(text))

    text = "当时ThinkPad T60刚刚发布，一同推出的还有一款名为Advanced Dock的扩展坞配件。这款扩展坞通过连接T60底部的插槽，扩展出包括PCIe在内的一大堆接口，并且自带电源，让T60可以安装桌面显卡来提升性能。"
    print(LangSegmenter.getTexts(text, "zh"))
    print(LangSegmenter.getTexts(text))