<div align="center">

# Resonastra

**本地语音训练与文本转语音**

完全离线 · 无需云端服务

**Train Your Voice. Keep It Local.**

[**English**](../../README.md) | **中文简体**

[![Release](https://img.shields.io/badge/release-v1.0.0-blue?style=flat-square)](https://github.com/SiriusLin1531/Resonastra/releases/tag/v1.0.0)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D4?style=flat-square&logo=windows)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/SiriusLin1531/Resonastra/blob/main/LICENSE)
![Offline](https://img.shields.io/badge/runtime-offline-success?style=flat-square)

</div>

---

## 项目简介

Resonastra 是一个面向 Windows 的本地语音训练与文本转语音工具，覆盖**数据准备、模型训练与语音生成**完整流程。核心训练与推理可在本地完成，无需依赖云端 TTS 服务。

## 核心特性

- **本地声音训练** — 使用自己的语音数据训练声音模型。
- **一体化 DataFactory** — 完成数据准备、ASR、校对与训练集整理。
- **完整训练流程** — 通过 WebUI 完成 Stage1 / Stage2 训练。
- **本地语音生成** — 使用训练生成的声音配置（Voice Profile）进行文本转语音。
- **质量评估** — 支持 DNSMOS、Speaker Similarity 与 WER/CER 等可选指标。
- **离线工作流** — 整合本地运行环境与主要依赖，核心流程无需云端服务。

## 下载

### 最新稳定版本 · Resonastra v1.0.0

- **平台：** Windows 10 / Windows 11 x64
- **GitHub Release：** <https://github.com/SiriusLin1531/Resonastra/releases/tag/v1.0.0>
- **百度网盘：** <https://pan.baidu.com/s/1_344UD7sJci6IuBgtbmF6A?pwd=8848>
- **提取码：** `8848`
- **发布包：** `Resonastra_v1.0.0.zip`
- **大小：** `13,866,120,773` bytes
- **SHA256：** `420772a840fe496e175501085c20191c83b1433d366ea5bb060af1c5db093e41`

> GitHub 自动生成的 **Source code (zip / tar.gz)** 仅为源码快照，不是完整 Windows 整合包。

更完整的发布身份与版本记录请参阅对应 GitHub Release。

历史版本与更新内容请参阅 GitHub Releases。

## 快速开始

1. 下载并解压 `Resonastra_v1.0.0.zip`。
2. 运行 `launch_check_env.bat` 完成环境检查。
3. 运行 `launch_data_factory_ui.bat` 准备训练数据。
4. 运行 `launch_training_ui.bat` 完成 Stage1 / Stage2 训练。
5. 运行 `launch_infer_ui.bat` 生成语音。

首次使用建议先完成环境检查，再依次进入 DataFactory、Training 和 Inference。

完整操作将在独立 Quick Start 文档中说明。

## 工作流程

Resonastra 的主要工作流由 DataFactory、Training 和 Inference 三个阶段组成：

```text
原始语音
    ↓
DataFactory
数据准备 · ASR · 校对
    ↓
训练数据集
    ↓
Training
Stage1 · Stage2
    ↓
声音配置（Voice Profile）
    ↓
Inference
文本转语音
    ↓
生成语音
```

## 系统要求

- Windows 10 / Windows 11 x64
- NVIDIA GPU 推荐用于训练与推理
- Microsoft Visual C++ v14 x64 Redistributable 14.44.35211 或更高版本
- 足够的本地磁盘空间

Resonastra v1.0.0 已集成本地运行环境，无需单独安装 Python。

**运行环境：** Python 3.10.20 · PyTorch 2.5.1 · CUDA Runtime 11.8

## 兼容性与已知限制

Resonastra v1.0.0 集成的运行环境为：

**Python 3.10.20 · PyTorch 2.5.1 · CUDA Runtime 11.8**

### RTX 50 系列

GeForce RTX 50 系列采用 NVIDIA Blackwell 架构，Compute Capability 为 `12.0`（`sm_120`）。

CUDA Toolkit 从 12.8 开始加入 `sm_120` 编译支持，PyTorch 从 2.7 开始正式提供面向 Blackwell 的 CUDA 12.8 构建。因此，Resonastra v1.0.0 当前集成的 **PyTorch 2.5.1 + CUDA 11.8 不属于 RTX 50 系列的受支持 GPU 环境**。

这并不意味着 Resonastra 在 RTX 50 系列设备上完全无法启动：

- Windows、Python 与 WebUI 本身仍可能正常启动。
- NVIDIA 驱动或 PyTorch 可能能够识别显卡。
- 但真正执行训练或推理所需的 CUDA kernel 时，可能出现架构不兼容警告或运行失败。
- 当前环境检查工具主要验证 runtime、依赖、配置与模型资产，并不会实际验证 `sm_120` CUDA kernel 是否能够执行。

因此，**Resonastra v1.0.0 不声明 RTX 50 系列 GPU 训练与推理支持**。

不建议直接替换整合包中的 PyTorch 或 CUDA 组件。正式支持 RTX 50 系列需要升级运行环境，并重新验证 Stage1、Stage2、Vocoder、DataFactory、ASR 与质量评估等完整工作流。

其他新架构 GPU 的兼容性同样以实际运行环境和对应 CUDA / PyTorch 支持情况为准。

## 使用文档

README 仅提供快速入口，完整文档将按功能拆分：

- **快速开始** — 从下载到第一次生成语音
- **DataFactory** — 数据准备、ASR 与文本校对
- **Training** — Stage1 / Stage2 训练流程
- **Inference** — 声音配置与语音生成
- **兼容性说明** — GPU、CUDA 与运行环境
- **故障排查** — 常见启动与运行问题
- **版本校验** — SHA256 与正式发布包验证

详细文档将逐步补充。

## 本地运行与隐私

Resonastra 的核心训练与推理流程均在本地运行。训练语音、数据集、声音模型与生成结果可以保留在本地设备中，无需上传至云端 TTS 服务。

## 技术概览

Resonastra 将文本处理、语义建模与声学生成组合为完整的本地 TTS 流程：

```text
文本
 ↓
Text Frontend
 ↓
Stage1
语义表示
 ↓
Stage2
声学表示
 ↓
Vocoder
 ↓
语音
```

当前版本保留 GPT-SoVITS v2 文本前端与 Stage1，并使用 Resonastra Stage2 声学模型完成后续语音生成。

更完整的模型与系统架构将在技术文档中说明。

## 致谢

Resonastra 的开发建立在多个优秀的开源项目与研究工作的基础之上，感谢这些项目及其贡献者。

### 核心语音技术

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 文本前端、Stage1 及相关兼容组件
- [HiFi-GAN](https://github.com/jik876/hifi-gan) — 神经声码器

### 数据处理与语音识别

- [FunASR](https://github.com/modelscope/FunASR) — 本地 ASR、VAD 与文本处理能力
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper 高效推理与语音识别
- [SubFix](https://github.com/cronrpc/SubFix) — ASR 标注结果的人工校对与字幕编辑 WebUI

### 语音质量评估

- [DNSMOS / DNS Challenge](https://github.com/microsoft/DNS-Challenge) — 非侵入式语音质量评估
- [SpeechBrain](https://github.com/speechbrain/speechbrain) — 说话人表征与相似度评估

### 界面与多媒体

- [Gradio](https://github.com/gradio-app/gradio) — WebUI
- [FFmpeg](https://github.com/FFmpeg/FFmpeg) — 音频处理与媒体工具

同时感谢所有相关开源项目、模型作者与社区贡献者。

完整的第三方软件、模型、许可与版权信息请参阅 `THIRD_PARTY_NOTICES.md`。

## 开源许可

Resonastra 项目自有源代码采用 **MIT License**。

第三方代码、模型、运行时组件及其他资产遵循各自的许可条款。详细信息请参阅 `THIRD_PARTY_NOTICES.md`。
