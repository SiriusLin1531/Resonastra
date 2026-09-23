<div align="center">

# Resonastra

**Local Voice Training & Text-to-Speech**

Fully offline. No cloud required.

**Train Your Voice. Keep It Local.**

**English** | [**中文简体**](./docs/cn/README.md)

[![Release](https://img.shields.io/badge/release-v1.0.0-blue?style=flat-square)](https://github.com/SiriusLin1531/Resonastra/releases/tag/v1.0.0)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D4?style=flat-square&logo=windows)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/SiriusLin1531/Resonastra/blob/main/LICENSE)
![Offline](https://img.shields.io/badge/runtime-offline-success?style=flat-square)

</div>

---

## Overview

Resonastra is a local voice training and text-to-speech toolkit for Windows, covering the complete workflow of **data preparation, model training, and speech generation**. Core training and inference run locally without relying on cloud TTS services.

## Features

- **Local Voice Training** — Train a voice model using your own speech data.
- **Integrated DataFactory** — Prepare data, run ASR, proofread transcripts, and organize training datasets.
- **Complete Training Workflow** — Run Stage1 / Stage2 training through the WebUI.
- **Local Speech Generation** — Generate speech with a trained Voice Profile.
- **Quality Evaluation** — Optional DNSMOS, Speaker Similarity, and WER/CER metrics.
- **Offline Workflow** — Includes the local runtime and major dependencies, with no cloud service required for the core workflow.

## Download

### Latest Stable Release · Resonastra v1.0.0

- **Platform:** Windows 10 / Windows 11 x64
- **GitHub Release:** <https://github.com/SiriusLin1531/Resonastra/releases/tag/v1.0.0>
- **Baidu Netdisk:** <https://pan.baidu.com/s/1_344UD7sJci6IuBgtbmF6A?pwd=8848>
- **Extraction Code:** `8848`
- **Archive:** `Resonastra_v1.0.0.zip`
- **Size:** `13,866,120,773` bytes
- **SHA256:** `420772a840fe496e175501085c20191c83b1433d366ea5bb060af1c5db093e41`

> GitHub-generated **Source code (zip / tar.gz)** files are source snapshots only. They are not the complete Windows package.

For complete release metadata and version records, see the corresponding GitHub Release.

See GitHub Releases for release history and previous versions.

## Quick Start

1. Download and extract `Resonastra_v1.0.0.zip`.
2. Run `launch_check_env.bat` to check the environment.
3. Run `launch_data_factory_ui.bat` to prepare training data.
4. Run `launch_training_ui.bat` to train Stage1 / Stage2.
5. Run `launch_infer_ui.bat` to generate speech.

For first-time use, run the environment check before proceeding through DataFactory, Training, and Inference in order.

A complete walkthrough will be provided in the standalone Quick Start guide.

## Workflow

The main Resonastra workflow consists of DataFactory, Training, and Inference:

```text
Raw Audio
    ↓
DataFactory
Data Preparation · ASR · Proofreading
    ↓
Training Dataset
    ↓
Training
Stage1 · Stage2
    ↓
Voice Profile
    ↓
Inference
Text-to-Speech
    ↓
Generated Audio
```

## System Requirements

- Windows 10 / Windows 11 x64
- NVIDIA GPU recommended for training and inference
- Microsoft Visual C++ v14 x64 Redistributable 14.44.35211 or later
- Sufficient local disk space

Resonastra v1.0.0 includes a bundled local runtime and does not require a separate Python installation.

**Runtime:** Python 3.10.20 · PyTorch 2.5.1 · CUDA Runtime 11.8

## Compatibility and Known Limitations

Resonastra v1.0.0 includes the following runtime:

**Python 3.10.20 · PyTorch 2.5.1 · CUDA Runtime 11.8**

### RTX 50 Series

GeForce RTX 50 Series GPUs use the NVIDIA Blackwell architecture with Compute Capability `12.0` (`sm_120`).

CUDA Toolkit added `sm_120` compiler support starting with CUDA 12.8, while PyTorch introduced official Blackwell support and CUDA 12.8 builds starting with PyTorch 2.7. The **PyTorch 2.5.1 + CUDA 11.8 runtime bundled with Resonastra v1.0.0 is therefore not part of the supported configuration for RTX 50 Series GPUs**.

This does not necessarily mean that Resonastra cannot start on an RTX 50 Series system:

- Windows, Python, and the WebUI may still start normally.
- The NVIDIA driver or PyTorch may still detect the GPU.
- CUDA kernels required for training or inference may produce architecture compatibility warnings or fail at runtime.
- The current environment checker validates the runtime, dependencies, configuration, and model assets, but does not execute a CUDA kernel to verify `sm_120` compatibility.

For this reason, **RTX 50 Series GPU training and inference are not supported configurations for Resonastra v1.0.0**.

Directly replacing the bundled PyTorch or CUDA components is not recommended. Official RTX 50 Series support would require upgrading the runtime and re-validating the complete Stage1, Stage2, Vocoder, DataFactory, ASR, and quality evaluation workflows.

Compatibility with other new GPU architectures likewise depends on the CUDA and PyTorch support provided by the actual runtime environment.

## Documentation

The README provides the main entry points. Full documentation will be organized by function:

- **Quick Start** — From download to first speech generation
- **DataFactory** — Data preparation, ASR, and transcript proofreading
- **Training** — Stage1 / Stage2 training workflow
- **Inference** — Voice Profiles and speech generation
- **Compatibility** — GPU, CUDA, and runtime environment
- **Troubleshooting** — Common startup and runtime issues
- **Release Verification** — SHA256 and official package verification

Detailed documentation will be added progressively.

## Local Processing and Privacy

Resonastra's core training and inference workflows run locally. Training audio, datasets, voice models, and generated outputs can remain on your local device without being uploaded to a cloud TTS service.

## Technical Overview

Resonastra combines text processing, semantic modeling, and acoustic generation into a complete local TTS pipeline:

```text
Text
 ↓
Text Frontend
 ↓
Stage1
Semantic Representation
 ↓
Stage2
Acoustic Representation
 ↓
Vocoder
 ↓
Speech
```

The current version retains the GPT-SoVITS v2 text frontend and Stage1, while Resonastra Stage2 handles downstream acoustic modeling for speech generation.

More detailed model and system architecture documentation will be provided separately.

## Acknowledgements

Resonastra builds on the work of several excellent open-source projects and research efforts. We thank their authors and contributors.

### Core Speech Technology

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — Text frontend, Stage1, and related compatibility components
- [HiFi-GAN](https://github.com/jik876/hifi-gan) — Neural vocoder

### Data Processing and Speech Recognition

- [FunASR](https://github.com/modelscope/FunASR) — Local ASR, VAD, and text processing
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Efficient Whisper inference and speech recognition
- [SubFix](https://github.com/cronrpc/SubFix) — WebUI for manual correction of ASR transcripts and subtitles

### Speech Quality Evaluation

- [DNSMOS / DNS Challenge](https://github.com/microsoft/DNS-Challenge) — Non-intrusive speech quality evaluation
- [SpeechBrain](https://github.com/speechbrain/speechbrain) — Speaker representation and similarity evaluation

### Interface and Multimedia

- [Gradio](https://github.com/gradio-app/gradio) — WebUI
- [FFmpeg](https://github.com/FFmpeg/FFmpeg) — Audio and media processing

We also thank all related open-source projects, model authors, and community contributors.

For complete third-party software, model, license, and copyright information, see `THIRD_PARTY_NOTICES.md`.

## License

Resonastra's original source code is licensed under the **MIT License**.

Third-party code, models, runtime components, and other assets remain subject to their respective license terms. See `THIRD_PARTY_NOTICES.md` for details.
