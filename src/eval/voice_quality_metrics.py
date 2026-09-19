from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


# ============================================================
# JSON helpers
# ============================================================

def to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return obj.item()
        return obj.tolist()

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n")


# ============================================================
# Audio helpers
# ============================================================

def load_audio_mono(
    wav_path: str | Path,
    *,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    wav_path = Path(wav_path)

    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = np.asarray(wav, dtype=np.float32)

    if target_sr is not None and int(sr) != int(target_sr):
        import librosa

        wav = librosa.resample(
            wav,
            orig_sr=int(sr),
            target_sr=int(target_sr),
        ).astype(np.float32)
        sr = int(target_sr)

    return wav, int(sr)


def audio_duration_sec(wav_path: str | Path) -> float:
    info = sf.info(str(wav_path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def safe_mean(xs: list[float]) -> float | None:
    ys = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not ys:
        return None
    return float(np.mean(ys))


def safe_std(xs: list[float]) -> float | None:
    ys = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(ys) <= 1:
        return 0.0 if ys else None
    return float(np.std(ys))


# ============================================================
# Text normalization / WER / CER
# ============================================================

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(str(text)))


def normalize_text_for_eval(text: str) -> str:
    """
    Light normalization for ASR/WER/CER.

    ZH:
    - 全角转半角
    - 小写
    - 去常见标点
    - 压缩空白

    EN:
    - NFKC
    - lowercase
    - remove punctuation
    - collapse spaces
    """
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()

    # Keep Chinese chars, ascii letters/numbers and whitespace.
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_for_cer(text: str) -> list[str]:
    text = normalize_text_for_eval(text)
    text = text.replace(" ", "")
    return list(text)


def tokenize_for_wer(text: str) -> list[str]:
    text = normalize_text_for_eval(text)

    if contains_cjk(text):
        # For Chinese, word segmentation is ambiguous.
        # We still return char tokens and report it as "wer_char_proxy".
        return list(text.replace(" ", ""))

    return text.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    n = len(a)
    m = len(b)

    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i

        for j in range(1, m + 1):
            old = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost,
            )
            prev = old

    return dp[m]


def compute_error_rate(
    ref_text: str,
    hyp_text: str,
    *,
    mode: str = "auto",
) -> dict[str, Any]:
    """
    Returns WER-like and CER metrics.

    For Chinese:
        "wer" is computed using char tokens and marked as char_proxy.
        "cer" is the main recommended metric.

    For English:
        "wer" uses whitespace tokens.
        "cer" uses characters.
    """
    ref_norm = normalize_text_for_eval(ref_text)
    hyp_norm = normalize_text_for_eval(hyp_text)

    is_cjk = contains_cjk(ref_norm)

    ref_cer_tokens = tokenize_for_cer(ref_norm)
    hyp_cer_tokens = tokenize_for_cer(hyp_norm)

    cer_dist = edit_distance(ref_cer_tokens, hyp_cer_tokens)
    cer_den = max(len(ref_cer_tokens), 1)
    cer = float(cer_dist) / float(cer_den)

    ref_wer_tokens = tokenize_for_wer(ref_norm)
    hyp_wer_tokens = tokenize_for_wer(hyp_norm)

    wer_dist = edit_distance(ref_wer_tokens, hyp_wer_tokens)
    wer_den = max(len(ref_wer_tokens), 1)
    wer = float(wer_dist) / float(wer_den)

    return {
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "contains_cjk": bool(is_cjk),
        "wer": float(wer),
        "wer_is_char_proxy": bool(is_cjk),
        "cer": float(cer),
        "edit_distance_word_or_char": int(wer_dist),
        "num_ref_word_or_char_tokens": int(len(ref_wer_tokens)),
        "edit_distance_char": int(cer_dist),
        "num_ref_chars": int(len(ref_cer_tokens)),
    }


# ============================================================
# ASR backends
# ============================================================

@dataclass
class ASRConfig:
    backend: str = "faster_whisper"
    model_name: str = "medium"
    language: str = "zh"
    device: str = "cuda"
    compute_type: str = "float16"


class ASRTranscriber:
    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg
        self.backend = str(cfg.backend).strip().lower()
        self.model = None

    def _lazy_load(self) -> None:
        if self.model is not None:
            return

        if self.backend == "none":
            return

        if self.backend == "faster_whisper":
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                self.cfg.model_name,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
            )
            return

        if self.backend == "whisper":
            import whisper

            self.model = whisper.load_model(
                self.cfg.model_name,
                device=self.cfg.device,
            )
            return

        raise ValueError(f"Unsupported ASR backend: {self.cfg.backend}")

    def transcribe(self, wav_path: str | Path) -> dict[str, Any]:
        self._lazy_load()

        if self.backend == "none":
            return {
                "backend": "none",
                "text": "",
                "segments": [],
            }

        if self.backend == "faster_whisper":
            segments, info = self.model.transcribe(
                str(wav_path),
                language=self.cfg.language,
                beam_size=5,
                vad_filter=False,
            )

            segs = []
            texts = []

            for seg in segments:
                item = {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": str(seg.text),
                }
                segs.append(item)
                texts.append(str(seg.text))

            return {
                "backend": "faster_whisper",
                "model_name": self.cfg.model_name,
                "language": self.cfg.language,
                "text": "".join(texts).strip(),
                "segments": segs,
                "language_probability": float(getattr(info, "language_probability", 0.0)),
            }

        if self.backend == "whisper":
            out = self.model.transcribe(
                str(wav_path),
                language=self.cfg.language,
                fp16=(self.cfg.device == "cuda"),
            )

            return {
                "backend": "whisper",
                "model_name": self.cfg.model_name,
                "language": self.cfg.language,
                "text": str(out.get("text", "")).strip(),
                "segments": out.get("segments", []),
            }

        raise ValueError(f"Unsupported ASR backend: {self.cfg.backend}")


# ============================================================
# DNSMOS
# ============================================================

@dataclass
class DNSMOSConfig:
    onnx_path: str | None = None
    sample_rate: int = 16000
    chunk_sec: float = 9.01
    hop_sec: float = 9.01
    use_gpu: bool = False


class DNSMOSScorer:
    """
    Generic ONNX DNSMOS scorer.

    Expected:
        P.835-like model returning SIG / BAK / OVRL, or one output tensor
        whose last dimension has >= 3 values.

    Because DNSMOS ONNX variants differ slightly, this class uses
    input/output introspection and a robust output parser.
    """

    def __init__(self, cfg: DNSMOSConfig) -> None:
        self.cfg = cfg
        self.session = None
        self.input_name = None
        self.output_names = None

        if cfg.onnx_path:
            self._load_session(Path(cfg.onnx_path))

    def available(self) -> bool:
        return self.session is not None

    def _load_session(self, onnx_path: Path) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(f"DNSMOS ONNX not found: {onnx_path}")

        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if self.cfg.use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=providers,
        )

        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError(f"DNSMOS model has no inputs: {onnx_path}")

        self.input_name = inputs[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _iter_chunks(self, wav: np.ndarray) -> list[np.ndarray]:
        sr = int(self.cfg.sample_rate)
        chunk_len = int(round(float(self.cfg.chunk_sec) * sr))
        hop_len = int(round(float(self.cfg.hop_sec) * sr))

        if chunk_len <= 0:
            raise ValueError("chunk_len must be positive.")

        if wav.size <= 0:
            wav = np.zeros(chunk_len, dtype=np.float32)

        chunks = []

        if wav.size <= chunk_len:
            y = np.zeros(chunk_len, dtype=np.float32)
            y[: wav.size] = wav
            chunks.append(y)
            return chunks

        start = 0
        while start < wav.size:
            end = start + chunk_len
            y = wav[start:end]

            if y.size < chunk_len:
                pad = np.zeros(chunk_len, dtype=np.float32)
                pad[: y.size] = y
                y = pad

            chunks.append(y.astype(np.float32))

            if end >= wav.size:
                break

            start += hop_len

        return chunks

    def _prepare_input(self, chunk: np.ndarray) -> np.ndarray:
        x = chunk.astype(np.float32)

        # Most DNSMOS ONNX models accept [B, T].
        # Some accept [B, T, 1]. We try [1, T] first.
        return x.reshape(1, -1).astype(np.float32)

    def _parse_outputs(self, outs: list[np.ndarray]) -> dict[str, float]:
        flat_arrays = [np.asarray(o).reshape(-1).astype(np.float32) for o in outs]

        merged = np.concatenate(flat_arrays, axis=0)

        # Common P.835 order is SIG, BAK, OVRL.
        # If the ONNX exports separate named outputs, try to infer by name.
        by_name: dict[str, float] = {}

        if self.output_names and len(outs) == len(self.output_names):
            for name, out in zip(self.output_names, outs):
                n = str(name).lower()
                arr = np.asarray(out).reshape(-1).astype(np.float32)
                if arr.size <= 0:
                    continue
                value = float(arr[0])

                if "sig" in n:
                    by_name["sig"] = value
                elif "bak" in n:
                    by_name["bak"] = value
                elif "ovr" in n or "overall" in n:
                    by_name["ovrl"] = value

        if {"sig", "bak", "ovrl"}.issubset(by_name.keys()):
            return by_name

        if merged.size >= 3:
            return {
                "sig": float(merged[0]),
                "bak": float(merged[1]),
                "ovrl": float(merged[2]),
            }

        if merged.size == 1:
            # P.808-like single MOS fallback.
            return {
                "sig": None,
                "bak": None,
                "ovrl": float(merged[0]),
            }

        return {
            "sig": None,
            "bak": None,
            "ovrl": None,
        }

    def score_file(self, wav_path: str | Path) -> dict[str, Any]:
        if self.session is None:
            return {
                "enabled": False,
                "error": "dnsmos_onnx_path_not_provided",
                "sig": None,
                "bak": None,
                "ovrl": None,
            }

        wav, _ = load_audio_mono(wav_path, target_sr=int(self.cfg.sample_rate))
        chunks = self._iter_chunks(wav)

        per_chunk = []

        for chunk in chunks:
            x = self._prepare_input(chunk)

            try:
                outs = self.session.run(None, {self.input_name: x})
            except Exception:
                # Try [B, T, 1] fallback.
                x3 = x.reshape(1, -1, 1).astype(np.float32)
                outs = self.session.run(None, {self.input_name: x3})

            parsed = self._parse_outputs(outs)
            per_chunk.append(parsed)

        sigs = [x.get("sig") for x in per_chunk if x.get("sig") is not None]
        baks = [x.get("bak") for x in per_chunk if x.get("bak") is not None]
        ovrls = [x.get("ovrl") for x in per_chunk if x.get("ovrl") is not None]

        return {
            "enabled": True,
            "sample_rate": int(self.cfg.sample_rate),
            "num_chunks": int(len(per_chunk)),
            "sig": safe_mean(sigs),
            "bak": safe_mean(baks),
            "ovrl": safe_mean(ovrls),
            "per_chunk": per_chunk,
        }


# ============================================================
# Speaker similarity
# ============================================================

@dataclass
class SpeakerSimConfig:
    backend: str = "speechbrain_ecapa"
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: str = "pretrained_models/speechbrain_spkrec_ecapa_voxceleb"
    device: str = "cuda"


class SpeakerSimilarityScorer:
    def __init__(self, cfg: SpeakerSimConfig) -> None:
        self.cfg = cfg
        self.backend = str(cfg.backend).strip().lower()
        self.model = None

    def _lazy_load(self) -> None:
        if self.model is not None:
            return

        if self.backend == "none":
            return

        if self.backend == "speechbrain_ecapa":
            import torch
            from speechbrain.inference.speaker import EncoderClassifier

            run_opts = {"device": self.cfg.device}

            self.model = EncoderClassifier.from_hparams(
                source=self.cfg.model_source,
                savedir=self.cfg.savedir,
                run_opts=run_opts,
            )
            return

        raise ValueError(f"Unsupported speaker sim backend: {self.cfg.backend}")

    def score_files(
        self,
        reference_wav_path: str | Path,
        generated_wav_path: str | Path,
    ) -> dict[str, Any]:
        self._lazy_load()

        if self.backend == "none":
            return {
                "enabled": False,
                "cosine": None,
            }

        if self.backend == "speechbrain_ecapa":
            import torch
            import torchaudio
            import torch.nn.functional as F

            def load_for_model(path: str | Path):
                wav, sr = torchaudio.load(str(path))
                if wav.ndim == 2 and wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                return wav, sr

            wav_ref, sr_ref = load_for_model(reference_wav_path)
            wav_gen, sr_gen = load_for_model(generated_wav_path)

            with torch.no_grad():
                emb_ref = self.model.encode_batch(wav_ref, wav_lens=None)
                emb_gen = self.model.encode_batch(wav_gen, wav_lens=None)

                emb_ref = emb_ref.squeeze()
                emb_gen = emb_gen.squeeze()

                cosine = F.cosine_similarity(
                    emb_ref.reshape(1, -1),
                    emb_gen.reshape(1, -1),
                    dim=-1,
                )[0].item()

            return {
                "enabled": True,
                "backend": self.backend,
                "model_source": self.cfg.model_source,
                "cosine": float(cosine),
                "reference_sr": int(sr_ref),
                "generated_sr": int(sr_gen),
            }

        raise ValueError(f"Unsupported speaker sim backend: {self.cfg.backend}")


# ============================================================
# Full evaluator
# ============================================================

@dataclass
class VoiceMetricConfig:
    compute_dnsmos: bool = True
    compute_asr_wer: bool = True
    compute_speaker_sim: bool = True

    dnsmos: DNSMOSConfig = field(default_factory=DNSMOSConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    speaker: SpeakerSimConfig = field(default_factory=SpeakerSimConfig)


class VoiceQualityEvaluator:
    def __init__(self, cfg: VoiceMetricConfig) -> None:
        self.cfg = cfg

        self.dnsmos_scorer = (
            DNSMOSScorer(cfg.dnsmos)
            if cfg.compute_dnsmos
            else None
        )

        self.asr_transcriber = (
            ASRTranscriber(cfg.asr)
            if cfg.compute_asr_wer
            else None
        )

        self.speaker_scorer = (
            SpeakerSimilarityScorer(cfg.speaker)
            if cfg.compute_speaker_sim
            else None
        )

    def evaluate_file(
        self,
        *,
        generated_wav_path: str | Path,
        target_text: str | None = None,
        reference_wav_path: str | Path | None = None,

        # Backward-compatible name.
        # 兼容旧字段：这里的 inference_seconds 现在按 generation_seconds 理解。
        inference_seconds: float | None = None,

        # New explicit runtime fields.
        # 新增显式运行时间字段。
        generation_seconds: float | None = None,
        metrics_seconds: float | None = None,
        total_seconds: float | None = None,
    ) -> dict[str, Any]:
        generated_wav_path = Path(generated_wav_path)

        duration = audio_duration_sec(generated_wav_path)

        # Prefer explicit generation_seconds.
        # 优先使用显式 generation_seconds；如果没有，则兼容旧的 inference_seconds。
        if generation_seconds is None:
            generation_seconds = inference_seconds

        def _safe_float(x):
            if x is None:
                return None
            try:
                v = float(x)
            except Exception:
                return None
            if not math.isfinite(v):
                return None
            return v

        generation_seconds = _safe_float(generation_seconds)
        metrics_seconds = _safe_float(metrics_seconds)
        total_seconds = _safe_float(total_seconds)

        generation_rtf = (
            float(generation_seconds) / float(duration)
            if generation_seconds is not None and duration > 0
            else None
        )

        metrics_rtf = (
            float(metrics_seconds) / float(duration)
            if metrics_seconds is not None and duration > 0
            else None
        )

        total_rtf = (
            float(total_seconds) / float(duration)
            if total_seconds is not None and duration > 0
            else None
        )

        out: dict[str, Any] = {
            "generated_wav_path": str(generated_wav_path),
            "duration_sec": float(duration),

            # Backward-compatible fields.
            # 兼容旧字段：rtf 默认指 generation_rtf。
            "inference_seconds": generation_seconds,
            "rtf": generation_rtf,

            # New explicit fields.
            # 新增显式字段。
            "generation_seconds": generation_seconds,
            "generation_rtf": generation_rtf,
            "metrics_seconds": metrics_seconds,
            "metrics_rtf": metrics_rtf,
            "total_seconds": total_seconds,
            "total_rtf": total_rtf,
        }

        if out["inference_seconds"] is not None and duration > 0:
            out["rtf"] = float(out["inference_seconds"]) / float(duration)

        if self.dnsmos_scorer is not None:
            try:
                out["dnsmos"] = self.dnsmos_scorer.score_file(generated_wav_path)
            except Exception as e:
                out["dnsmos"] = {
                    "enabled": True,
                    "error": str(e),
                    "sig": None,
                    "bak": None,
                    "ovrl": None,
                }

        if self.asr_transcriber is not None and target_text is not None:
            try:
                asr = self.asr_transcriber.transcribe(generated_wav_path)
                err = compute_error_rate(target_text, asr.get("text", ""))
                out["asr"] = asr
                out["wer"] = err
            except Exception as e:
                out["asr"] = {"error": str(e)}
                out["wer"] = None

        if self.speaker_scorer is not None and reference_wav_path is not None:
            try:
                out["speaker_sim"] = self.speaker_scorer.score_files(
                    reference_wav_path=reference_wav_path,
                    generated_wav_path=generated_wav_path,
                )
            except Exception as e:
                out["speaker_sim"] = {
                    "enabled": True,
                    "error": str(e),
                    "cosine": None,
                }

        return out


def summarize_metric_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def pick(path: list[str]) -> list[float]:
        vals = []

        for r in records:
            cur: Any = r
            ok = True

            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    ok = False
                    break
                cur = cur[key]

            if ok and cur is not None:
                try:
                    v = float(cur)
                    if math.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass

        return vals

    metrics = {
        "dnsmos_ovrl": pick(["dnsmos", "ovrl"]),
        "dnsmos_sig": pick(["dnsmos", "sig"]),
        "dnsmos_bak": pick(["dnsmos", "bak"]),
        "wer": pick(["wer", "wer"]),
        "cer": pick(["wer", "cer"]),
        "speaker_sim_cosine": pick(["speaker_sim", "cosine"]),

        # Backward-compatible RTF.
        # 兼容旧字段：rtf 默认等价于 generation_rtf。
        "rtf": pick(["rtf"]),

        # Explicit runtime metrics.
        # 显式运行时间指标。
        "generation_rtf": pick(["generation_rtf"]),
        "metrics_rtf": pick(["metrics_rtf"]),
        "total_rtf": pick(["total_rtf"]),
        "duration_sec": pick(["duration_sec"]),
        "generation_seconds": pick(["generation_seconds"]),
        "metrics_seconds": pick(["metrics_seconds"]),
        "total_seconds": pick(["total_seconds"]),
    }

    summary: dict[str, Any] = {
        "num_records": int(len(records)),
        "metrics": {},
    }

    for name, vals in metrics.items():
        summary["metrics"][name] = {
            "count": int(len(vals)),
            "mean": safe_mean(vals),
            "std": safe_std(vals),
            "min": float(np.min(vals)) if vals else None,
            "max": float(np.max(vals)) if vals else None,
            "p50": float(np.percentile(vals, 50)) if vals else None,
            "p90": float(np.percentile(vals, 90)) if vals else None,
        }

    return summary