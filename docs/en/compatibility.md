# Resonastra Compatibility Guide

[Project README](https://github.com/SiriusLin1531/Resonastra/blob/main/README.md) | [Quick Start](./quick-start.md) | [DataFactory](./data-factory.md) | [Training](./training.md) | [Inference](./inference.md) | **English** | [中文简体](../cn/compatibility.md)

Applies to: **Resonastra v1.0.0**

> [!NOTE]
> This guide documents the runtime environment, GPU / CUDA support boundaries, and environment-check behavior for **Resonastra v1.0.0**. For DataFactory, Training, or Inference procedures and component-specific troubleshooting, use the corresponding guide.

---

## 1. What This Guide Covers

This guide answers the following questions:

- Which Windows / Python / PyTorch / CUDA environment does v1.0.0 target?
- How does the bundled runtime differ from system Python?
- How do the NVIDIA driver, CUDA Runtime, and GPU architecture relate to one another?
- Why are RTX 50 Series / `sm_120` GPUs outside the formally supported v1.0.0 boundary?
- What does `launch_check_env.bat` check, and what does it not prove?
- What release-level launcher, port, and offline-asset boundaries should users know about?

If a problem is already clearly inside DataFactory, Training, or Inference, go directly to that component's guide instead of repeating its troubleshooting here.

---

## 2. v1.0.0 Compatibility at a Glance

| Item | v1.0.0 Boundary |
| --- | --- |
| Operating system | Windows 10 / Windows 11 x64 |
| Python | Bundled CPython 3.10.20 |
| PyTorch | 2.5.1 |
| CUDA Runtime | 11.8 |
| VC++ Runtime | Microsoft Visual C++ v14 x64 Redistributable 14.44.35211 or later |
| GPU | A compatible NVIDIA GPU is recommended for training / inference |
| RTX 50 / `sm_120` | Not claimed as supported by v1.0.0 |
| Network dependency | Core user workflows are designed around the offline release boundary |

"A compatible NVIDIA GPU is recommended" does not mean that every NVIDIA GPU architecture is guaranteed to work. Actual CUDA execution still depends on the GPU architecture, PyTorch build, CUDA Runtime, and whether the required CUDA kernels support that GPU.

---

## 3. Bundled Runtime and Dependency Boundaries

### 3.1 Bundled Runtime

The normal user path prefers:

```text
runtime/env/python.exe
```

The packaged launchers and environment checker are designed around this bundled environment.

If `runtime/env/` is missing, some lower-level launch paths may fall back to system Python, but this is not the recommended repair path for the packaged user release. For a downloaded and fully extracted release package, a missing bundled runtime should first be treated as one of the following:

- an incomplete release package;
- an incomplete extraction;
- files deleted by mistake;
- files quarantined by security software.

Do not replace the packaged Python, PyTorch, or CUDA components as a first-line fix for an ordinary startup problem. Doing so moves the environment away from the v1.0.0 combination that was actually validated.

### 3.2 VC++ Runtime

`launch_check_env.bat` checks for the x64 Microsoft Visual C++ v14 Redistributable.

The v1.0.0 release check requires:

```text
Microsoft Visual C++ v14 x64 Redistributable
>= 14.44.35211
```

If the check fails, install or update the official x64 Redistributable, then run the environment check again.

Do not replace a proper VC++ Runtime installation by manually copying individual DLL files into the project directory.

### 3.3 Offline Models and Local Assets

The core Resonastra v1.0.0 user workflow is designed around the offline release boundary.

For example:

- Inference uses the packaged FastText language-identification model and enables offline mode;
- the supported DataFactory ASR path uses local Chinese FunASR assets;
- downloading core models at runtime is not the normal user path.

If a local model or core asset is missing, first check the release package, extraction result, and whether the file was deleted or quarantined.

---

## 4. GPU and CUDA Compatibility

### 4.1 NVIDIA Driver, CUDA Runtime, and GPU Architecture Are Different Layers

When evaluating GPU compatibility, distinguish these layers:

1. **NVIDIA driver** — the system driver layer for the GPU;
2. **CUDA capability exposed by the driver** — for example, information shown by `nvidia-smi`;
3. **Resonastra's bundled CUDA Runtime** — CUDA Runtime 11.8 in v1.0.0;
4. **PyTorch build** — PyTorch 2.5.1 in v1.0.0;
5. **GPU architecture / compute capability** — determines whether the required CUDA kernels support the GPU.

A newer CUDA version shown by `nvidia-smi` does not mean that Resonastra is using that CUDA Runtime. Updating the NVIDIA driver also does not turn the packaged:

```text
PyTorch 2.5.1 + CUDA Runtime 11.8
```

into a CUDA 12.8-or-later software stack.

### 4.2 RTX 50 / Blackwell / `sm_120`

Resonastra v1.0.0 **does not claim native support for RTX 50 Series / `sm_120` GPUs**.

On such a system, it is possible to see:

```text
Windows works
Python works
GPU is detected
WebUI starts
        ↓
A real CUDA workload fails
```

This does not contradict a successful environment check because the environment checker does not execute CUDA kernels.

Do not treat an in-place upgrade of the packaged PyTorch / CUDA stack as a normal user fix. The safer options are to use hardware compatible with the validated v1.0.0 environment or a later release that has been explicitly revalidated for the newer architecture.

### 4.3 CPU Fallback Boundary

Inference requests:

```text
device = cuda
```

by default.

If CUDA is requested but `torch.cuda.is_available()` is False, the official inference script prints a warning and falls back to CPU.

This means:

- Inference has an explicit CPU fallback path;
- CPU inference is usually much slower;
- successful CPU execution does not mean the CUDA environment is healthy.

Training is also CUDA-oriented by default, and the user interface does not expose a general device switch. Some lower-level paths may have CPU fallback behavior, but that does not mean Resonastra v1.0.0 formally supports completing Training on CPU.

### 4.4 Out-of-Memory Is Not Architecture Incompatibility

`CUDA out of memory` indicates insufficient VRAM. It does not mean that the GPU architecture itself is unsupported.

If OOM occurs during Training, follow the VRAM and `batch_size` guidance in the [Training Guide](./training.md). Do not treat OOM as an RTX 50 / CUDA-architecture compatibility problem.

---

## 5. Environment Check and Report

Official entry point:

```text
launch_check_env.bat
```

Structured report:

```text
logs/check_user_env_report.json
```

### 5.1 What the Environment Check Covers

The environment checker primarily verifies:

- Microsoft Visual C++ v14 x64 Redistributable;
- the bundled `runtime/env/` and Python runtime;
- key Python packages;
- `configs/user_inference_default.yaml`;
- `user_profiles/default_zh/profile_config.json`;
- the GPT-SoVITS compatibility layer;
- Chinese BERT, CNHuBERT, and G2PW;
- the default Stage1 / SoVITS / Resonastra Stage2 models;
- the HiFi-GAN directory.

Structured statuses are:

```text
ok
warn
fail
```

Overall status is determined as follows:

- any required failure → `fail`;
- no `fail`, but at least one warning → `warn`;
- everything passes → `ok`.

The packaged launcher does not treat an ordinary warning as a failed process exit. Even if the launcher ends with:

```text
[OK] Environment check completed successfully.
```

the structured report can still contain `warn` / `[WARN]` items. Read the individual checks and the JSON report when warnings are present.

### 5.2 What the Environment Check Does Not Prove

The environment checker does not:

- execute CUDA kernels;
- validate GPU compute-capability compatibility;
- load the complete Stage1 / Stage2 model stack in a real workload;
- prove that Training can complete;
- prove that Inference can complete;
- test every WebUI port;
- check every optional quality-evaluation backend.

> [!IMPORTANT]
> A successful environment check means that the items covered by the checker passed. It does not mean that the GPU architecture or CUDA workload has been validated by actual execution.

---

## 6. Launcher, Port, and Release-Behavior Differences

Default addresses:

| UI / Component | Default Address | Port Behavior |
| --- | --- | --- |
| DataFactory | `127.0.0.1:7861` | Automatically searches for another available port if occupied |
| Training | `127.0.0.1:7863` | Automatically searches for another available port if occupied |
| Inference | `127.0.0.1:7860` | Does not provide the same automatic port-switching behavior |
| DataFactory proofreader | `127.0.0.1:9871` | Default port; configurable in DataFactory settings |

For DataFactory and Training, use the final URL shown in the launcher terminal if the default port was already occupied.

If Inference port `7860` is occupied, resolve the port conflict before starting Inference again.

If the service has started and the terminal already shows a URL but the browser did not open automatically, copy that URL into your browser manually.

---

## 7. Where to Look When Something Goes Wrong

To avoid maintaining the same issue in multiple documents, component-specific failures remain owned by the corresponding component guide.

| Issue Type | Recommended Guide |
| --- | --- |
| Download, extraction, first startup, basic route selection | [Quick Start](./quick-start.md) |
| Raw audio, ASR, proofreading, Stage1 / Stage2 data generation and Resume | [DataFactory Guide](./data-factory.md) |
| Training launch/failure, Stop, Trainer Best / Active Best, historical checkpoints | [Training Guide](./training.md) |
| Voice Profile, reference audio, generation failure, timeout, quality evaluation, output issues | [Inference Guide](./inference.md) |
| Windows, VC++, bundled runtime, GPU / CUDA, RTX 50 / `sm_120` | **This guide** |
| Download integrity and official-release identity | Release Verification Guide (not yet published) |

If an issue crosses multiple components, start from the **first clear failure point**. For example, if DataFactory completed successfully but Training does not start, begin with the Training Guide rather than rechecking a data-preparation flow that already succeeded.

---

## 8. More Compatibility Cases and Feedback

This guide covers the primary v1.0.0 compatibility boundaries that are currently verified. Additional hardware combinations, compatibility cases, and common issues are still being collected.

If you encounter an issue not covered by this guide or the existing component guides, please report it through the project's [GitHub Issues](https://github.com/SiriusLin1531/Resonastra/issues).

To make the report easier to diagnose, include when relevant:

- Resonastra version;
- Windows version;
- GPU model;
- affected component;
- the first clear error;
- whether you are using default settings;
- whether the issue is reproducible;
- relevant logs or diagnostics after reviewing them for sensitive information.

Useful diagnostic entry points:

| Layer | Check First |
| --- | --- |
| Environment check | `logs/check_user_env_report.json` |
| DataFactory | DataFactory Result, the failed step, and its logs |
| Training | `训练失败摘要` ("Training Failure Summary"), `training_stderr.log`, `training_stdout.log`, `training_command.json` |
| Inference | `诊断信息` ("Diagnostics"), error type / stage, stdout / stderr |

> [!WARNING]
> Logs and diagnostic output may contain local usernames, absolute paths, commands, or error text. Remove information you do not want to make public before sharing them.
