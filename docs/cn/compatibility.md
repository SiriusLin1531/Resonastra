# Resonastra 兼容性指南

[项目 README](./README.md) | [快速开始](./quick-start.md) | [数据准备](./data-factory.md) | [模型训练](./training.md) | [语音生成](./inference.md) | **兼容性** | [版本校验](./release-verification.md) | [English](../en/compatibility.md) | **中文简体**

适用版本：**Resonastra v1.0.0**

> [!NOTE]
> 本指南说明 **Resonastra v1.0.0** 的运行环境、GPU / CUDA 支持边界和环境检查方式。DataFactory、Training、Inference 的具体操作与组件内故障处理，请查看对应使用指南。

---

## 1. 本指南解决什么问题

本指南主要回答：

- v1.0.0 支持什么 Windows / Python / PyTorch / CUDA 环境；
- 内置运行环境与系统 Python 有什么区别；
- NVIDIA 驱动、CUDA Runtime 与 GPU 架构之间是什么关系；
- RTX 50 / `sm_120` 为什么不属于 v1.0.0 的正式支持范围；
- `launch_check_env.bat` 能检查什么、不能证明什么；
- 启动器、端口和离线模型资产有哪些发行边界。

如果问题已经明确发生在 DataFactory、Training 或 Inference 内部，请直接查看对应指南，不需要先在本文重复排查。

---

## 2. v1.0.0 支持环境速查

| 项目 | v1.0.0 边界 |
| --- | --- |
| 操作系统 | Windows 10 / Windows 11 x64 |
| Python | 内置 CPython 3.10.20 |
| PyTorch | 2.5.1 |
| CUDA Runtime | 11.8 |
| VC++ Runtime | Microsoft Visual C++ v14 x64 Redistributable 14.44.35211 或更高 |
| GPU | 兼容 NVIDIA GPU 推荐用于训练 / 推理 |
| RTX 50 / `sm_120` | v1.0.0 不声明支持 |
| 网络依赖 | 核心用户工作流按离线发行边界设计 |

“兼容 NVIDIA GPU”不等于“所有 NVIDIA GPU 都保证支持”。实际 CUDA 执行还取决于 GPU 架构、PyTorch 构建、CUDA Runtime 和所需 CUDA 内核是否匹配。

---

## 3. 内置运行环境与依赖边界

### 3.1 内置运行环境

正式用户路径优先使用：

```text
runtime/env/python.exe
```

发行包内的启动器和环境检查围绕这套内置环境设计。

如果 `runtime/env/` 缺失，部分底层启动路径可能退回系统 Python，但这不是正式用户包的推荐修复方式。对于已经下载并完整解压的发行包，内置运行环境缺失应优先视为：

- 发行包不完整；
- 解压失败；
- 文件被误删；
- 安全软件隔离了文件。

不建议为了修复普通启动问题直接替换发行包中的 Python、PyTorch 或 CUDA 组件。这样会使环境偏离 v1.0.0 已验证组合。

### 3.2 VC++ Runtime

`launch_check_env.bat` 会检查 x64 Microsoft Visual C++ v14 Redistributable。

v1.0.0 的发行检查要求：

```text
Microsoft Visual C++ v14 x64 Redistributable
>= 14.44.35211
```

如果检查失败，应安装或更新官方 x64 Redistributable，再重新运行环境检查。

不要通过手工复制单个 DLL 到项目目录来替代正常安装。

### 3.3 离线模型与本地资产

Resonastra v1.0.0 的核心用户流程按离线发行边界设计。

例如：

- Inference 使用发行包内的 FastText 语言识别（language-ID）模型，并启用离线模式；
- DataFactory 正式 ASR 路径使用本地中文 FunASR 资产；
- 核心模型不会以“运行时自动联网下载”作为正常用户路径。

如果出现本地模型或核心资产缺失，应优先检查发行包完整性、解压结果和文件是否被删除 / 隔离。

---

## 4. GPU 与 CUDA 兼容性

### 4.1 NVIDIA 驱动、CUDA Runtime 与 GPU 架构不是同一层

排查 GPU 兼容性时，需要区分：

1. **NVIDIA 驱动**：系统与 GPU 的驱动层；
2. **驱动可支持的 CUDA 能力**：例如 `nvidia-smi` 显示的信息；
3. **Resonastra 内置 CUDA Runtime**：v1.0.0 为 CUDA Runtime 11.8；
4. **PyTorch 构建**：v1.0.0 为 PyTorch 2.5.1；
5. **GPU 架构 / 计算能力（compute capability）**：决定所需 CUDA 内核是否支持当前 GPU。

因此，`nvidia-smi` 显示较新的 CUDA 版本，不代表 Resonastra 正在使用那个 CUDA Runtime。更新 NVIDIA 驱动也不会把发行包中的：

```text
PyTorch 2.5.1 + CUDA Runtime 11.8
```

变成 CUDA 12.8 或更高版本的软件栈。

### 4.2 RTX 50 / Blackwell / `sm_120`

Resonastra v1.0.0 **不声明 RTX 50 系列 / `sm_120` 原生支持**。

这类环境可能出现：

```text
Windows 正常
Python 正常
GPU 可以被识别
WebUI 可以启动
        ↓
真正执行 CUDA 工作负载时失败
```

这并不与环境检查通过相矛盾，因为环境检查不会实际执行 CUDA 内核。

不建议把“手工升级发行包内的 PyTorch / CUDA”作为普通用户修复方法。更安全的做法是使用 v1.0.0 已兼容的硬件环境，或等待经过重新验证、明确支持新架构的后续发行版。

### 4.3 CPU 回退的边界

Inference 默认请求：

```text
device = cuda
```

如果 CUDA 被请求但 `torch.cuda.is_available()` 为 False，正式推理脚本会给出警告并回退到 CPU。

这意味着：

- Inference 存在明确的 CPU 回退路径；
- CPU 推理通常会明显更慢；
- CPU 路径成功不代表 CUDA 环境已经正常。

Training 默认同样面向 CUDA，但用户界面不提供通用设备切换入口。底层个别路径可能存在 CPU 回退，并不等于 v1.0.0 正式支持使用 CPU 完成 Training。

### 4.4 显存不足不是架构不兼容

`CUDA out of memory` 表示显存资源不足，不等同于 GPU 架构不受支持。

如果 OOM 发生在 Training，应按照 [模型训练指南（Training）](./training.md) 中的显存与 `batch_size` 建议处理；不要把 OOM 当成 RTX 50 / CUDA 架构兼容性问题。

---

## 5. 环境检查与报告

正式入口：

```text
launch_check_env.bat
```

结构化报告：

```text
logs/check_user_env_report.json
```

### 5.1 环境检查会检查什么

环境检查主要覆盖：

- Microsoft Visual C++ v14 x64 Redistributable；
- 内置 `runtime/env/` 与 Python；
- 关键 Python 软件包；
- `configs/user_inference_default.yaml`；
- `user_profiles/default_zh/profile_config.json`；
- GPT-SoVITS 兼容层；
- Chinese BERT、CNHuBERT、G2PW；
- 默认 Stage1 / SoVITS / Resonastra Stage2 模型；
- HiFi-GAN 目录。

结构化状态包括：

```text
ok
warn
fail
```

整体规则：

- 任一必需项失败 → `fail`；
- 没有 `fail` 但存在警告 → `warn`；
- 全部通过 → `ok`。

发行版启动器不会把普通警告当成失败退出条件。即使启动器最终打印 `[OK] Environment check completed successfully.`，结构化报告中仍可能存在 `warn` / `[WARN]`，因此应以具体检查项和 JSON 报告为准。

### 5.2 环境检查不会证明什么

环境检查不会：

- 执行 CUDA 内核；
- 验证 GPU 计算能力（compute capability）；
- 实际加载全部 Stage1 / Stage2 模型；
- 证明 Training 能完成；
- 证明 Inference 能完成；
- 测试所有 WebUI 端口；
- 检查所有可选质量检测后端。

> [!IMPORTANT]
> “环境检查通过”只说明环境检查覆盖到的项目通过，不等于 GPU 架构或 CUDA 工作负载已经完成实机验证。

---

## 6. 启动器、端口与发行行为差异

默认地址：

| 界面 / 组件 | 默认地址 | 端口行为 |
| --- | --- | --- |
| DataFactory | `127.0.0.1:7861` | 占用时自动寻找其它可用端口 |
| Training | `127.0.0.1:7863` | 占用时自动寻找其它可用端口 |
| Inference | `127.0.0.1:7860` | 不提供相同的自动换端口行为 |
| DataFactory 校对器 | `127.0.0.1:9871` | 默认端口，可在 DataFactory 设置中调整 |

DataFactory / Training 发生端口占用时，应以启动终端最终打印的 URL 为准。

Inference 如果 `7860` 被其它程序占用，应先处理端口冲突后再启动。

如果服务已经启动并在终端打印 URL，只是浏览器没有自动打开，可以直接复制该地址到浏览器。

---

## 7. 遇到问题时去哪里查

为避免同一问题在多份文档中重复维护，组件内故障由对应使用指南承担。

| 问题类型 | 建议查看 |
| --- | --- |
| 下载、解压、第一次启动、基础路线选择 | [快速开始](./quick-start.md) |
| 原始音频、ASR、人工校对、Stage1 / Stage2 数据生成与续跑 | [数据准备指南（DataFactory）](./data-factory.md) |
| Training 启动、失败、Stop、Trainer Best / Active Best、历史模型检查点 | [模型训练指南（Training）](./training.md) |
| Voice Profile、参考音频、推理失败、超时、质量检测与生成结果 | [语音生成指南（Inference）](./inference.md) |
| Windows、VC++、内置运行环境、GPU / CUDA、RTX 50 / `sm_120` | **本指南** |
| 下载文件完整性与正式发行身份 | [版本校验指南](./release-verification.md) |

如果问题跨越多个组件，优先从**第一个明确失败点**开始。例如 DataFactory 已成功而 Training 启动失败，应先看 Training，而不是重新排查已经成功的数据准备流程。

---

## 8. 更多问题与反馈

当前文档覆盖的是 v1.0.0 已确认的主要兼容性边界。更多硬件组合、兼容性案例和常见问题仍在持续收集中。

如果你遇到本文和现有使用指南尚未覆盖的问题，欢迎通过项目 [GitHub Issues](https://github.com/SiriusLin1531/Resonastra/issues) 反馈。

为了便于定位，建议附上：

- Resonastra 版本；
- Windows 版本；
- GPU 型号；
- 问题发生的组件；
- 第一个明确错误；
- 是否使用默认配置；
- 能否稳定复现；
- 经过隐私检查后的相关日志或诊断信息。

常用诊断入口：

| 层级 | 优先查看 |
| --- | --- |
| 环境检查 | `logs/check_user_env_report.json` |
| DataFactory | DataFactory Result、失败步骤和对应日志 |
| Training | `训练失败摘要`、`training_stderr.log`、`training_stdout.log`、`training_command.json` |
| Inference | `诊断信息`、错误类型 / 阶段、标准输出 / 标准错误 |

> [!WARNING]
> 日志和诊断内容可能包含本机用户名、绝对路径、运行命令和错误文本。公开分享前，请先删除不希望公开的信息。
