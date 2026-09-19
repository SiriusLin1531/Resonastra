from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


class _AttrDict(dict):
    """
    EN:
    Minimal AttrDict replacement for HiFi-GAN config.

    ZH:
    用于 HiFi-GAN config 的最小 AttrDict 替代实现。
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(item) from e
        if isinstance(value, dict) and not isinstance(value, _AttrDict):
            value = _AttrDict(value)
            self[item] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Stage2VocoderAdapter:
    """
    EN:
    Unified vocoder adapter for stage-2 acoustic outputs.

    Current first implementation:
    - backend-agnostic interface
    - HiFi-GAN backend implemented
    - official pretrained profile switching supported

    ZH:
    用于第二阶段声学输出的统一 vocoder 适配器。

    当前第一版实现：
    - 对外保持 backend 无关接口
    - 先实现 HiFi-GAN backend
    - 支持切换官方预训练 profile
    """

    OFFICIAL_HIFIGAN_PROFILES = {
        "universal_v1": "UNIVERSAL_V1",
        "vctk_v1": "VCTK_V1",
        "vctk_v2": "VCTK_V2",
        "vctk_v3": "VCTK_V3",
        "lj_v1": "LJ_V1",
        "lj_v2": "LJ_V2",
        "lj_v3": "LJ_V3",
        "lj_ft_t2_v1": "LJ_FT_T2_V1",
        "lj_ft_t2_v2": "LJ_FT_T2_V2",
        "lj_ft_t2_v3": "LJ_FT_T2_V3",
    }

    def __init__(
        self,
        vocoder_type: str = "hifigan",
        device: str = "cpu",
        profile: str = "universal_v1",
        hifigan_root: str | Path | None = None,
    ) -> None:
        self.vocoder_type = str(vocoder_type).strip().lower()
        self.device = torch.device(device)
        self.profile = str(profile).strip().lower()
        self.hifigan_root = Path(hifigan_root).resolve() if hifigan_root is not None else None

        self.model = None
        self.config: _AttrDict | None = None
        self.checkpoint_path: Path | None = None
        self.config_path: Path | None = None
        self.sample_rate: int | None = None
        self.n_mels: int | None = None
        self.hop_size: int | None = None
        self.is_loaded: bool = False
        self.backend_meta: dict[str, Any] = {}

    # --------------------------------------------------------
    # Path resolution / 路径解析
    # --------------------------------------------------------
    def resolve_profile_paths(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> tuple[Path, Path]:
        """
        EN:
        Resolve checkpoint/config paths.

        Priority:
        1. explicit checkpoint_path / config_path
        2. profile under hifigan_root/pretrained_models/<PROFILE_DIR>

        ZH:
        解析 checkpoint/config 路径。

        优先级：
        1. 显式传入 checkpoint_path / config_path
        2. 根据 hifigan_root/pretrained_models/<PROFILE_DIR> 自动解析
        """
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path).resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Vocoder checkpoint not found: {ckpt_path}")
        else:
            if self.vocoder_type != "hifigan":
                raise NotImplementedError(f"Unsupported vocoder_type: {self.vocoder_type}")
            if self.hifigan_root is None:
                raise ValueError("hifigan_root is required when checkpoint_path is not explicitly provided.")
            profile_dir_name = self.OFFICIAL_HIFIGAN_PROFILES.get(self.profile)
            if profile_dir_name is None:
                raise ValueError(
                    f"Unknown HiFi-GAN profile: {self.profile}. "
                    f"Available profiles: {sorted(self.OFFICIAL_HIFIGAN_PROFILES.keys())}"
                )
            profile_dir = self.hifigan_root / "pretrained_models" / profile_dir_name
            if not profile_dir.exists():
                raise FileNotFoundError(f"HiFi-GAN profile directory not found: {profile_dir}")
            candidates = sorted(profile_dir.glob("g_*"))
            if len(candidates) == 0:
                raise FileNotFoundError(f"No generator checkpoint matching g_* found under: {profile_dir}")
            ckpt_path = candidates[-1].resolve()

        if config_path is not None:
            cfg_path = Path(config_path).resolve()
            if not cfg_path.exists():
                raise FileNotFoundError(f"Vocoder config not found: {cfg_path}")
        else:
            cfg_candidate = ckpt_path.parent / "config.json"
            if not cfg_candidate.exists():
                raise FileNotFoundError(
                    f"Could not infer config.json beside checkpoint: {cfg_candidate}"
                )
            cfg_path = cfg_candidate.resolve()

        return ckpt_path, cfg_path

    # --------------------------------------------------------
    # Dynamic module loading / 动态模块加载
    # --------------------------------------------------------
    def _load_module_from_path(self, module_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for: {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _load_hifigan_backend_modules(self) -> tuple[Any, Any]:
        if self.hifigan_root is None:
            raise ValueError("hifigan_root is required for HiFi-GAN backend.")

        utils_path = self.hifigan_root / "utils.py"
        models_path = self.hifigan_root / "models.py"

        if not utils_path.exists():
            raise FileNotFoundError(f"HiFi-GAN utils.py not found: {utils_path}")
        if not models_path.exists():
            raise FileNotFoundError(f"HiFi-GAN models.py not found: {models_path}")

        old_utils = sys.modules.get("utils")
        try:
            self._load_module_from_path("utils", utils_path)
            models_module = self._load_module_from_path("_voice_lab_hifigan_models", models_path)
        finally:
            if old_utils is not None:
                sys.modules["utils"] = old_utils
            else:
                sys.modules.pop("utils", None)

        return None, models_module

    # --------------------------------------------------------
    # Load / 加载
    # --------------------------------------------------------
    def load(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        if self.vocoder_type != "hifigan":
            raise NotImplementedError(
                f"Vocoder type '{self.vocoder_type}' is not implemented yet. "
                f"Future extension hook is reserved for BigVGAN and others."
            )

        ckpt_path, cfg_path = self.resolve_profile_paths(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
        _, models_module = self._load_hifigan_backend_modules()

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = _AttrDict(cfg_dict)

        generator = models_module.Generator(cfg).to(self.device)
        state_dict = torch.load(str(ckpt_path), map_location=self.device)
        generator_state = state_dict.get("generator", state_dict)
        generator.load_state_dict(generator_state, strict=True)
        generator.eval()
        if hasattr(generator, "remove_weight_norm"):
            generator.remove_weight_norm()

        self.model = generator
        self.config = cfg
        self.checkpoint_path = ckpt_path
        self.config_path = cfg_path
        self.sample_rate = int(cfg.sampling_rate)
        self.n_mels = int(cfg.num_mels)
        self.hop_size = int(cfg.hop_size)
        self.is_loaded = True
        self.backend_meta = {
            "profile": self.profile,
            "checkpoint_path": str(ckpt_path),
            "config_path": str(cfg_path),
            "sampling_rate": self.sample_rate,
            "num_mels": self.n_mels,
            "hop_size": self.hop_size,
        }

    def ensure_loaded(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        if not self.is_loaded:
            self.load(checkpoint_path=checkpoint_path, config_path=config_path)

    # --------------------------------------------------------
    # Decode / 解码
    # --------------------------------------------------------
    def normalize_mel_shape(self, mel: torch.Tensor) -> torch.Tensor:
        """
        EN:
        Normalize mel shape to HiFi-GAN input layout: (B, 80, T).

        External convention in this project prefers:
            (B, T, 80)

        ZH:
        将 mel 统一转成 HiFi-GAN 需要的输入布局： (B, 80, T)

        本项目外部更偏向使用：
            (B, T, 80)
        """
        if self.n_mels is None:
            raise RuntimeError("Vocoder config is not loaded; n_mels is unknown.")

        if mel.dim() == 2:
            if mel.shape[0] == self.n_mels:
                mel = mel.unsqueeze(0)  # (1, 80, T)
            elif mel.shape[1] == self.n_mels:
                mel = mel.unsqueeze(0).transpose(1, 2)  # (1, 80, T)
            else:
                raise ValueError(
                    f"Unsupported 2D mel shape: {tuple(mel.shape)}. Expected (80, T) or (T, 80)."
                )
            return mel

        if mel.dim() != 3:
            raise ValueError(f"mel must be 2D or 3D, got shape={tuple(mel.shape)}")

        if mel.shape[1] == self.n_mels:
            return mel
        if mel.shape[2] == self.n_mels:
            return mel.transpose(1, 2)

        raise ValueError(
            f"Unsupported 3D mel shape: {tuple(mel.shape)}. "
            f"Expected (B, 80, T) or (B, T, 80)."
        )

    @torch.inference_mode()
    def decode(
        self,
        mel: torch.Tensor,
        lengths: torch.Tensor | None = None,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> torch.Tensor:
        self.ensure_loaded(checkpoint_path=checkpoint_path, config_path=config_path)
        assert self.model is not None
        assert self.hop_size is not None

        mel_for_vocoder = self.normalize_mel_shape(mel).to(self.device)
        waveform = self.model(mel_for_vocoder).squeeze(1)  # (B, N)
        waveform = waveform.float().cpu()

        if lengths is not None:
            trimmed = []
            for i in range(waveform.shape[0]):
                n_samples = int(lengths[i].item()) * int(self.hop_size)
                trimmed.append(waveform[i, :n_samples])
            max_len = max(x.shape[0] for x in trimmed)
            padded = []
            for x in trimmed:
                if x.shape[0] < max_len:
                    pad = torch.zeros(max_len - x.shape[0], dtype=x.dtype)
                    x = torch.cat([x, pad], dim=0)
                padded.append(x)
            waveform = torch.stack(padded, dim=0)

        waveform = waveform.clamp(min=-1.0, max=1.0)
        return waveform

    def save_wav(
        self,
        waveform: torch.Tensor,
        output_path: str | Path,
        sample_rate: int | None = None,
    ) -> Path:
        if sf is None:
            raise RuntimeError("soundfile is required for save_wav, but it is not available.")
        if waveform.dim() == 2:
            if waveform.shape[0] != 1:
                raise ValueError(
                    f"save_wav currently expects a single waveform, got shape={tuple(waveform.shape)}"
                )
            waveform = waveform[0]
        if waveform.dim() != 1:
            raise ValueError(f"waveform must be 1D or (1, N), got shape={tuple(waveform.shape)}")

        sr = int(sample_rate if sample_rate is not None else (self.sample_rate or 22050))
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), waveform.detach().cpu().numpy(), sr)
        return out_path