from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from dataclasses import dataclass
from functools import lru_cache

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import librosa
import numpy as np
import torch


@dataclass
class MelExtractConfig:
    """
    EN:
    Log-mel extraction config.

    Default config matches the current VoiceLab stage2 acoustic rate:
        sample_rate = 22050
        hop_length = 256
        acoustic_rate_hz = 22050 / 256 = 86.1328125

    ZH:
    log-mel 提取配置。

    默认配置对应当前 VoiceLab stage2 acoustic rate：
        sample_rate = 22050
        hop_length = 256
        acoustic_rate_hz = 22050 / 256 = 86.1328125
    """

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0

    center: bool = True
    power: float = 2.0

    log_base: str = "natural"
    log_clip: float = 1e-5

    # If true, return shape (T, n_mels), matching current Stage2 target_acoustic.
    transpose_to_time_first: bool = True


@lru_cache(maxsize=32)
def _cached_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    """
    EN:
    Build and cache mel filterbank.

    ZH:
    构建并缓存 mel filterbank。
    """
    mel = librosa.filters.mel(
        sr=int(sample_rate),
        n_fft=int(n_fft),
        n_mels=int(n_mels),
        fmin=float(fmin),
        fmax=float(fmax),
        htk=False,
        norm="slaney",
    )

    mel_np = np.asarray(mel, dtype=np.float32)
    return torch.from_numpy(mel_np)


class LogMelExtractor:
    """
    EN:
    Torch STFT + librosa mel filterbank based log-mel extractor.

    ZH:
    基于 Torch STFT + librosa mel filterbank 的 log-mel 提取器。
    """

    def __init__(
        self,
        config: MelExtractConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if config is None:
            config = MelExtractConfig()

        self.config = config
        self.device = torch.device(device)

        self.window = torch.hann_window(
            int(config.win_length),
            periodic=True,
            dtype=torch.float32,
        ).to(self.device)

        self.mel_fb = _cached_mel_filterbank(
            int(config.sample_rate),
            int(config.n_fft),
            int(config.n_mels),
            float(config.fmin),
            float(config.fmax),
        ).to(self.device)

    @property
    def acoustic_rate_hz(self) -> float:
        return float(self.config.sample_rate) / float(self.config.hop_length)

    def __call__(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        EN:
        Extract log-mel acoustic target.

        Args:
            waveform:
                shape (T,), (1, T), or (B, T). For preview/export usage,
                B should usually be 1.

        Return:
            If transpose_to_time_first=True:
                shape (T_frames, n_mels)
            else:
                shape (n_mels, T_frames)

        ZH:
        提取 log-mel 声学目标。

        参数：
            waveform:
                形状可以是 (T,), (1, T), 或 (B, T)。预览/导出阶段通常 B=1。

        返回：
            如果 transpose_to_time_first=True：
                形状 (T_frames, n_mels)
            否则：
                形状 (n_mels, T_frames)
        """
        cfg = self.config

        wav = waveform.detach().to(self.device).float()

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        if wav.ndim != 2:
            raise ValueError(f"waveform must have shape (T,) or (B, T), got {tuple(wav.shape)}")

        if wav.shape[0] != 1:
            # For this first preview layer, only single utterance waveform is supported.
            raise ValueError(
                f"LogMelExtractor preview mode expects batch size 1, got {wav.shape[0]}"
            )

        stft = torch.stft(
            wav.squeeze(0),
            n_fft=int(cfg.n_fft),
            hop_length=int(cfg.hop_length),
            win_length=int(cfg.win_length),
            window=self.window,
            center=bool(cfg.center),
            return_complex=True,
        )

        # stft: (freq, frames)
        magnitude = stft.abs()

        if float(cfg.power) == 1.0:
            spec = magnitude
        elif float(cfg.power) == 2.0:
            spec = magnitude.pow(2.0)
        else:
            spec = magnitude.pow(float(cfg.power))

        mel = torch.matmul(self.mel_fb, spec)
        mel = torch.clamp(mel, min=float(cfg.log_clip))

        if cfg.log_base == "natural":
            log_mel = torch.log(mel)
        elif cfg.log_base == "10":
            log_mel = torch.log10(mel)
        else:
            raise ValueError(f"Unsupported log_base: {cfg.log_base}")

        if cfg.transpose_to_time_first:
            log_mel = log_mel.transpose(0, 1).contiguous()

        return log_mel.detach().cpu()

    def expected_num_frames(
        self,
        num_samples: int,
    ) -> int:
        """
        EN:
        Estimate number of STFT frames for a waveform length.

        ZH:
        根据 waveform 采样点数估算 STFT 帧数。
        """
        cfg = self.config

        if cfg.center:
            # torch.stft pads n_fft // 2 on both sides.
            effective = int(num_samples) + int(cfg.n_fft)
        else:
            effective = int(num_samples)

        if effective < int(cfg.n_fft):
            return 1

        return 1 + (effective - int(cfg.n_fft)) // int(cfg.hop_length)