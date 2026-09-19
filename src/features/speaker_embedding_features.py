from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch
import torch.nn.functional as F


@dataclass
class SpeakerEmbeddingConfig:
    backend: str = "speechbrain_ecapa"
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: str = "pretrained_models/speechbrain_spkrec_ecapa_voxceleb"
    device: str = "cpu"
    sample_rate: int = 16000
    normalize: bool = True
    local_strategy: str = "COPY"


class SpeakerEmbeddingExtractor:
    """
    EN:
    Lazy speaker embedding extractor for v6.6.2-D style-cache export.

    ZH:
    用于 v6.6.2-D style cache 导出的懒加载说话人嵌入提取器。
    """

    def __init__(self, cfg: SpeakerEmbeddingConfig) -> None:
        self.cfg = cfg
        self.backend = str(cfg.backend).strip().lower()
        self.model = None

    def _resolve_speechbrain_local_strategy(self) -> Any:
        """Return a SpeechBrain LocalStrategy enum when available.

        中文说明：
            SpeechBrain 的 local_strategy 参数在当前 runtime 中不是普通字符串，
            而是 speechbrain.utils.fetching.LocalStrategy 枚举。

            直接传入 "copy" 会导致：
                ValueError: Illegal local strategy copy passed for linking

            因此这里把用户友好的字符串映射成真正的枚举值。
        """

        raw = str(self.cfg.local_strategy or "COPY").strip().upper().replace("-", "_")
        aliases = {
            "COPY": "COPY",
            "COPY_SKIP_CACHE": "COPY_SKIP_CACHE",
            "COPY_SKIP": "COPY_SKIP_CACHE",
            "NO_LINK": "NO_LINK",
            "NOLINK": "NO_LINK",
            "SYMLINK": "SYMLINK",
        }
        enum_name = aliases.get(raw, "COPY")

        try:
            from speechbrain.utils.fetching import LocalStrategy
        except Exception:
            # Older SpeechBrain versions may accept strings. Keep a best-effort
            # fallback, but current VoiceLab runtime should use the enum branch.
            return enum_name

        if hasattr(LocalStrategy, enum_name):
            return getattr(LocalStrategy, enum_name)

        # Extremely defensive fallback for SpeechBrain variants that may not expose
        # COPY. NO_LINK avoids symlink creation, but may return cache paths instead
        # of materializing files under savedir.
        if hasattr(LocalStrategy, "NO_LINK"):
            return getattr(LocalStrategy, "NO_LINK")

        return getattr(LocalStrategy, "SYMLINK")

    def _lazy_load(self) -> None:
        if self.model is not None:
            return

        if self.backend == "none":
            return

        if self.backend == "speechbrain_ecapa":
            from speechbrain.inference.speaker import EncoderClassifier

            # SpeechBrain defaults to a symlink-based local strategy for some
            # versions. On Windows, creating symlinks usually requires elevated
            # privileges or Developer Mode, which causes:
            #   [WinError 1314] A required privilege is not held by the client.
            #
            # User-edition VoiceLab should work from a normal Windows terminal,
            # so we force the safer LocalStrategy.COPY strategy. This avoids
            # privilege issues when materializing files from HuggingFace cache
            # into savedir.
            self.model = EncoderClassifier.from_hparams(
                source=str(self.cfg.model_source),
                savedir=str(self.cfg.savedir),
                run_opts={"device": str(self.cfg.device)},
                local_strategy=self._resolve_speechbrain_local_strategy(),
            )
            return

        raise ValueError(f"Unsupported speaker embedding backend: {self.cfg.backend}")

    def _load_wav(self, wav_path: str | Path) -> torch.Tensor:
        import torchaudio

        path = Path(wav_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"speaker embedding wav not found: {path}")

        wav, sr = torchaudio.load(str(path))
        wav = wav.float()

        if wav.ndim != 2:
            raise ValueError(f"Expected torchaudio waveform [C,T], got {tuple(wav.shape)}")

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        target_sr = int(self.cfg.sample_rate)
        if int(sr) != target_sr:
            wav = torchaudio.functional.resample(
                wav,
                orig_freq=int(sr),
                new_freq=target_sr,
            )

        if wav.numel() <= 0:
            raise ValueError(f"speaker embedding wav is empty: {path}")

        return wav.contiguous()

    @torch.inference_mode()
    def extract_from_wav(self, wav_path: str | Path) -> torch.Tensor:
        self._lazy_load()

        if self.backend == "none":
            raise RuntimeError("Speaker embedding backend is 'none'.")

        if self.model is None:
            raise RuntimeError("Speaker embedding model was not loaded.")

        wav = self._load_wav(wav_path)
        device = torch.device(str(self.cfg.device))
        wav = wav.to(device)

        emb = self.model.encode_batch(wav, wav_lens=None)
        emb = emb.detach().float().squeeze()

        if emb.ndim != 1:
            emb = emb.reshape(-1)

        if bool(self.cfg.normalize):
            emb = F.normalize(emb.view(1, -1), dim=-1, eps=1e-6).view(-1)

        if emb.numel() <= 0:
            raise ValueError(f"speaker embedding is empty for: {wav_path}")

        if not torch.isfinite(emb).all():
            raise ValueError(f"speaker embedding contains NaN/Inf for: {wav_path}")

        return emb.detach().cpu().contiguous()
