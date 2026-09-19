# -*- coding: utf-8 -*-
"""Resonastra v1.0.0 canonical offline FunASR adapter.

Derived from the long-tested GPT-SoVITS-compatible ``tools/asr/funasr_asr.py``
used by the project. The v1.0.0 release contract changes only model acquisition semantics:

- release execution is strictly offline;
- the three accepted Chinese FunASR model trees must already exist locally;
- no ModelScope/Hugging Face download API is imported or called;
- FunASR startup update checks are disabled;
- the CLI/output contract remains compatible with the DataFactory router.

This tracked canonical release source is mapped to ``tools/asr/funasr_asr.py``
in both the public source tree and the packaged v1.0.0 release.
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from funasr import AutoModel
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "tools" / "asr" / "models"

funasr_models = {}


_MODEL_LAYOUT = {
    "asr": {
        "dir": "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "required": ["model.pt", "config.yaml", "configuration.json", "tokens.json", "seg_dict"],
    },
    "vad": {
        "dir": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "required": ["model.pt", "config.yaml", "configuration.json"],
    },
    "punc": {
        "dir": "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "required": ["model.pt", "config.yaml", "configuration.json", "tokens.json"],
    },
}


def _require_local_model(kind: str) -> Path:
    spec = _MODEL_LAYOUT[kind]
    root = MODELS_ROOT / str(spec["dir"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"Required offline FunASR model directory is missing: {root}. "
            "Resonastra does not download models at runtime."
        )

    missing = [name for name in spec["required"] if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required offline FunASR model files are missing under {root}: {missing}. "
            "Resonastra does not download models at runtime."
        )
    return root.resolve()


def only_asr(input_file, language):
    try:
        model = create_model(language)
        text = model.generate(input=input_file)[0]["text"]
    except Exception:
        text = ""
        print(traceback.format_exc())
    return text


def create_model(language="zh"):
    if str(language).strip().lower() != "zh":
        raise ValueError(
            "Resonastra v1.0.0 DataFactory supports Chinese (zh) local FunASR only."
        )

    if "zh" in funasr_models:
        return funasr_models["zh"]

    path_asr = _require_local_model("asr")
    path_vad = _require_local_model("vad")
    path_punc = _require_local_model("punc")

    model = AutoModel(
        model=str(path_asr),
        model_revision="v2.0.4",
        vad_model=str(path_vad),
        vad_model_revision="v2.0.4",
        punc_model=str(path_punc),
        punc_model_revision="v2.0.4",
        disable_update=True,
    )
    print("FunASR local offline model loaded: ZH")
    funasr_models["zh"] = model
    return model


def execute_asr(input_folder, output_folder, model_size, language):
    del model_size
    input_file_names = os.listdir(input_folder)
    input_file_names.sort()

    output = []
    output_file_name = os.path.basename(input_folder)
    model = create_model(language)

    for file_name in tqdm(input_file_names):
        try:
            print("\n" + file_name)
            file_path = os.path.join(input_folder, file_name)
            text = model.generate(input=file_path)[0]["text"]
            output.append(f"{file_path}|{output_file_name}|ZH|{text}")
        except Exception:
            print(traceback.format_exc())

    output_folder = output_folder or "output/asr_opt"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.abspath(f"{output_folder}/{output_file_name}.list")

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        print(f"ASR task complete -> label file: {output_file_path}\n")
    return output_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument("-s", "--model_size", type=str, default="large")
    parser.add_argument("-l", "--language", type=str, default="zh", choices=["zh"])
    parser.add_argument("-p", "--precision", type=str, default="float32", choices=["float32"])
    cmd = parser.parse_args()
    execute_asr(
        input_folder=cmd.input_folder,
        output_folder=cmd.output_folder,
        model_size=cmd.model_size,
        language=cmd.language,
    )
