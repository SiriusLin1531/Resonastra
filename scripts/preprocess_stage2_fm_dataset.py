from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import sys
from pathlib import Path

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import librosa
import numpy as np
import torch

# ============================================================
# Make project root importable
# 把项目根目录加入导入路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.adapters.gsv_text_frontend import GSVTextFrontend
from src.adapters.gsv_prompt_tokenizer import GSVPromptTokenizer
from src.adapters.gsv_t2s import GSVT2S


DEFAULT_HIFIGAN_SR = 22050
DEFAULT_HIFIGAN_HOP = 256
DEFAULT_HIFIGAN_ACOUSTIC_RATE_HZ = DEFAULT_HIFIGAN_SR / DEFAULT_HIFIGAN_HOP


def compute_acoustic_rate_hz(target_sr: int, hop_length: int) -> float:
    """
    EN:
    Compute mel frame rate in Hz.

    ZH:
    计算 mel 帧率（Hz）。
    """
    return float(target_sr) / float(hop_length)


def parse_args() -> argparse.Namespace:
    """
    EN:
    Parse command-line arguments for stage-2 offline preprocessing.

    ZH:
    解析第二阶段离线预处理脚本的命令行参数。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_half", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    # --------------------------------------------------------
    # Mel extraction settings
    # Mel 提取参数
    # --------------------------------------------------------
    parser.add_argument("--target_sr", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=8000.0)

    return parser.parse_args()


def load_manifest(manifest_path: str | Path) -> list[dict]:
    """
    EN:
    Load a JSONL manifest file.

    Each line should be a JSON dict with at least:
        sample_id, raw_text, language, prompt_wav_path, target_wav_path

    ZH:
    读取 JSONL 格式的 manifest 文件。

    每一行至少应包含：
        sample_id, raw_text, language, prompt_wav_path, target_wav_path
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    items: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            required_keys = ["sample_id", "raw_text", "language", "prompt_wav_path", "target_wav_path"]
            for key in required_keys:
                if key not in item:
                    raise KeyError(f"Manifest line {line_idx} missing required key: {key}")

            items.append(item)

    return items


def extract_mel_80(
    wav_path: str | Path,
    target_sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    """
    EN:
    Extract 80-dim mel spectrogram from target waveform.

    Output shape:
        (T_ac, 80)

    ZH:
    从目标音频中提取 80 维 mel 频谱。

    输出形状：
        (T_ac, 80)
    """
    wav_path = str(Path(wav_path).resolve())
    wav, _ = librosa.load(wav_path, sr=target_sr, mono=True)

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    # EN:
    # Convert power mel to log-mel.
    #
    # ZH:
    # 将功率 mel 转换成 log-mel。
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

    # librosa output: (n_mels, T)
    # Convert to: (T, n_mels)
    mel = torch.from_numpy(mel).float().transpose(0, 1).contiguous()
    return mel


def build_single_sample(
    item: dict,
    frontend: GSVTextFrontend,
    prompt_tokenizer: GSVPromptTokenizer,
    stage1: GSVT2S,
    args: argparse.Namespace,
) -> dict:
    """
    EN:
    Build one finalized `.pt` training sample from raw text/audio inputs.

    ZH:
    从原始文本/音频输入构建一条最终 `.pt` 训练样本。
    """
    sample_id = str(item["sample_id"])
    raw_text = str(item["raw_text"])
    language = str(item["language"])
    prompt_wav_path = str(Path(item["prompt_wav_path"]).resolve())
    target_wav_path = str(Path(item["target_wav_path"]).resolve())

    # --------------------------------------------------------
    # 1) Text frontend
    # 文本前端
    # --------------------------------------------------------
    phoneme_ids, phoneme_lens, bert_feature, norm_text = frontend.prepare_ids_and_bert(
        text=raw_text,
        language=language,
    )

    # --------------------------------------------------------
    # 2) Prompt tokenizer
    # Prompt tokenizer
    # --------------------------------------------------------
    prompt_tokens = prompt_tokenizer.extract_prompt_semantic_from_wav(
        wav_path=prompt_wav_path
    )

    # --------------------------------------------------------
    # 3) Stage-1 semantic generation
    # 第一阶段 semantic 生成
    # --------------------------------------------------------
    pred_semantic, aux = stage1.generate_semantic(
        phoneme_ids=phoneme_ids,
        phoneme_lens=phoneme_lens,
        bert_feature=bert_feature,
        prompt_tokens=prompt_tokens,
    )

    # --------------------------------------------------------
    # 4) Target acoustic = 80-dim mel
    # 目标声学表示 = 80维 mel
    # --------------------------------------------------------
    target_acoustic = extract_mel_80(
        wav_path=target_wav_path,
        target_sr=args.target_sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    acoustic_rate_hz = compute_acoustic_rate_hz(
        target_sr=args.target_sr,
        hop_length=args.hop_length,
    )

    # --------------------------------------------------------
    # 5) Format into finalized sample dict
    # 整理成最终样本字典
    # --------------------------------------------------------
    sample = {
        # ---------- 核心输入 ----------
        "semantic_tokens": pred_semantic.squeeze(0).detach().cpu().long(),    # (T_sem,)
        "bert_feature": bert_feature.squeeze(0).detach().cpu().float(),        # (1024, T_text)
        "prompt_tokens": prompt_tokens.squeeze(0).detach().cpu().long(),       # (T_prompt,)
        "target_acoustic": target_acoustic.detach().cpu().float(),             # (T_ac, 80)

        # ---------- 辅助输入 ----------
        "phoneme_ids": phoneme_ids.squeeze(0).detach().cpu().long(),           # (T_text,)
        "phoneme_lens": phoneme_lens.squeeze(0).detach().cpu().long(),         # scalar-like (1,)

        # ---------- 元信息 ----------
        "raw_text": raw_text,
        "norm_text": norm_text,
        "language": language,

        # ---------- 可选扩展 ----------
        "sample_id": sample_id,
        "prompt_wav_path": prompt_wav_path,
        "target_wav_path": target_wav_path,

        # ---------- acoustic target metadata ----------
        "acoustic_meta": {
            "target_sr": int(args.target_sr),
            "n_fft": int(args.n_fft),
            "hop_length": int(args.hop_length),
            "win_length": int(args.win_length),
            "n_mels": int(args.n_mels),
            "fmin": float(args.fmin),
            "fmax": float(args.fmax),
            "acoustic_rate_hz": float(acoustic_rate_hz),
            "mel_backend": "librosa_power_to_log",
        },
    }

    return sample


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_manifest(args.manifest)
    acoustic_rate_hz = compute_acoustic_rate_hz(
        target_sr=args.target_sr,
        hop_length=args.hop_length,
    )

    print("==== Stage2 FM offline preprocessing started ====")
    print(f"manifest    : {Path(args.manifest).resolve()}")
    print(f"output_dir  : {output_dir}")
    print(f"num_items   : {len(items)}")
    print(f"device      : {args.device}")
    print("================================================")
    print(f"target_sr   : {args.target_sr}")
    print(f"n_fft       : {args.n_fft}")
    print(f"hop_length  : {args.hop_length}")
    print(f"win_length  : {args.win_length}")
    print(f"n_mels      : {args.n_mels}")
    print(f"fmin        : {args.fmin}")
    print(f"fmax        : {args.fmax}")
    print(f"acoustic_hz : {acoustic_rate_hz:.6f}")
    print("================================================")

    # --------------------------------------------------------
    # Initialize upstream compatible modules
    # 初始化上游兼容模块
    # --------------------------------------------------------
    frontend = GSVTextFrontend(
        version="v2",
        device=args.device,
        use_half=args.use_half,
    )
    prompt_tokenizer = GSVPromptTokenizer(
        version="v2",
        device=args.device,
        use_half=args.use_half,
    )
    stage1 = GSVT2S(
        version="v2",
        device=args.device,
        use_half=args.use_half,
    )

    success = 0
    failed = 0

    for idx, item in enumerate(items, start=1):
        sample_id = str(item["sample_id"])
        out_path = output_dir / f"{sample_id}.pt"

        if out_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(items)}] skip existing -> {out_path.name}")
            continue

        try:
            sample = build_single_sample(
                item=item,
                frontend=frontend,
                prompt_tokenizer=prompt_tokenizer,
                stage1=stage1,
                args=args,
            )
            torch.save(sample, out_path)
            print(f"[{idx}/{len(items)}] saved -> {out_path.name}")
            success += 1

        except Exception as e:
            print(f"[{idx}/{len(items)}] FAILED sample_id={sample_id}: {e}")
            failed += 1

    print("================================================")
    print(f"preprocessing finished. success={success}, failed={failed}")
    print("================================================")


if __name__ == "__main__":
    main()