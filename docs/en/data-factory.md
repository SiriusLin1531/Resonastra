# Resonastra DataFactory Guide

[Project README](https://github.com/SiriusLin1531/Resonastra/blob/main/README.md) | [Quick Start](./quick-start.md) | **English** | [中文简体](../cn/data-factory.md)

Applies to: **Resonastra v1.0.0**

> [!NOTE]
> This guide documents the official **Resonastra v1.0.0** DataFactory behavior. If a later release changes the DataFactory UI, ASR routing, data layout, or training-data contract, refer to the documentation for that release.

---

## 1. What This Guide Covers

Resonastra DataFactory prepares user-provided raw speech so it can be used for Stage1 / Stage2 Training.

If you only use the bundled Zero-shot models, you do not need DataFactory. DataFactory is required only for the Stage1-only, Stage2-only, and Full Few-shot routes.

After completing the steps required by your selected route, you should be able to:

- choose and import supported raw audio;
- create or restore a DataFactory work directory;
- complete audio normalization, slicing, and local offline FunASR transcription;
- confirm the training transcript;
- generate Stage1 or Stage2 training data independently;
- recover correctly after an interrupted task;
- determine whether the data required by your route is ready for Training.

This guide does not cover Training parameters, checkpoint / Active Best management, Voice Profiles, Inference parameters, or general GPU / CUDA compatibility policy. DataFactory-specific recovery and diagnostics are covered in the relevant sections of this guide.

---

## 2. DataFactory Workflow and Before You Start

DataFactory is the data-preparation layer before Few-shot Training:

```text
Raw Audio
  ↓
Normalize / Slice
  ↓
Local Offline FunASR
  ↓
Transcript Confirmation
  ↓
Corrected Data Output
  ├─→ Stage1 Training Data
  └─→ Stage2 Training Data
          ↓
       Training
```

DataFactory is complete when **the data required by your selected route is actually ready**, not merely when the last button you clicked finishes running.

| Route | DataFactory Required? | Completion Condition |
| --- | --- | --- |
| Zero-shot | No | DataFactory is not required |
| Stage1-only Few-shot | Yes | Transcript confirmed + Stage1 ready |
| Stage2-only Few-shot | Yes | Transcript confirmed + Stage2 ready |
| Full Few-shot | Yes | Stage1 ready + Stage2 ready |

> [!IMPORTANT]
> Stage1 and Stage2 data can be generated independently. Stage1-only and Stage2-only are both valid completion states; not every Few-shot route requires both stages to be complete.

### 2.1 Before You Start

Recommended prerequisites:

1. fully extract Resonastra v1.0.0;
2. run `launch_check_env.bat` and confirm that the basic environment check passes;
3. prepare Chinese speech data;
4. choose a **Speaker / Character Name** to identify this project.

The official v1.0.0 DataFactory user path is:

```text
language = zh
ASR = local offline FunASR
```

For more stable Few-shot data, prefer a single speaker, clear pronunciation, normal speaking speed, and little or no obvious background noise. Avoid damaged files, abnormal silence, or repeated lossy transcoding when possible. These are quality recommendations, not a complete list of hard input constraints.

---

## 3. Start DataFactory

From the Resonastra root directory, run:

```text
launch_data_factory_ui.bat
```

The default port is `7861`. If the port is already in use, the launcher automatically selects another available port. Use the address shown in the launcher terminal.

> [!IMPORTANT]
> The main launcher terminal keeps the DataFactory WebUI running. Do not close it while using DataFactory.

The page mainly contains:

- **`基础输入` ("Basic Input")** — raw-audio source, speaker, language, and work directory;
- **`执行流程` ("Execution Flow")** — data preparation, proofreading, Stage1, and Stage2;
- **`设置` ("Settings")**;
- recommended actions / data status;
- **`当前任务` ("Current Task")**;
- proofreader status;
- **`诊断信息` ("Diagnostics")**.

For your first run, keep the default settings and complete the standard workflow before changing advanced options.

### 3.1 Main Launcher Terminal and Progress Terminal

When **`显示终端进度窗口` ("Show Terminal Progress Window")** is enabled, long-running tasks may open an additional log terminal.

| Window | Purpose | Can You Close It? |
| --- | --- | --- |
| Main WebUI launcher terminal | Keeps the DataFactory WebUI running | Do not close it while using DataFactory |
| Progress / log terminal | Read-only task logs | Yes. Closing it does not stop the real task |

---

## 4. Raw Audio and Work Directory

### 4.1 Three Raw-Audio Input Methods

**`本地目录路径` ("Local Directory Path")** is recommended for normal or larger datasets. DataFactory recursively scans the directory and its subdirectories for supported audio.

**`上传音频文件` ("Upload Audio Files")** is useful for a small number of files or quick tests.

**`上传音频文件夹` ("Upload Audio Folder")** lets you choose a complete local folder.

For upload modes, accepted audio is first copied to:

```text
{work_dir}/00_uploaded_raw_audio/
```

DataFactory then uses this cache directory as the raw-audio input. The common setting `覆盖已上传音频缓存` ("Overwrite Uploaded Audio Cache") is enabled by default. It affects only the upload cache and does not delete the original local source directory. If disabled, uploading another file with the same name creates a non-conflicting destination name instead of directly overwriting the cached file.

> [!NOTE]
> Upload modes are convenience features. For larger Few-shot datasets, `本地目录路径` ("Local Directory Path") remains the preferred input method.

### 4.2 Supported Formats and Quality Boundary

v1.0.0 officially recognizes:

```text
.wav  .mp3  .flac  .ogg
.m4a  .aac  .wma   .opus
```

Extension matching is case-insensitive, and one directory may contain multiple supported formats.

> [!IMPORTANT]
> A supported extension means DataFactory will attempt to process the file. It does not guarantee that every codec, container variant, or damaged file can be decoded successfully.

WEBM, MP4, AIFF/AIF, CAF, and AMR are not official v1.0.0 DataFactory input formats.

When possible, prefer good-quality WAV / FLAC. Lossy formats are supported, but converting MP3 / AAC or another lossy source to WAV does not restore information that was already lost.

### 4.3 Automatic Normalization and Input-Failure Boundary

You do not need to manually convert every source file to mono, 32 kHz, or PCM16. The current slicing path performs:

```text
Supported Raw Audio
  ↓
Decode
  ↓
Mono
  ↓
32 kHz
  ↓
Silence Slicing
  ↓
PCM 16-bit WAV Clips
```

Data preparation may fail if the source directory does not exist, no supported audio is found, an upload cannot be copied, a file is damaged, or the internal encoding cannot be decoded. The current slicing path does not guarantee that it will skip every bad file and continue automatically; locate the problematic source file first when decoding fails.

### 4.4 Speaker Name and Work Directory

If you do not specify a work directory manually, DataFactory uses:

```text
user_data/{speaker_name}_factory
```

For example:

```text
说话人 / 角色名：Character_A
工作目录（work_dir）：user_data/Character_A_factory
```

If you want a custom location or need to restore an existing project, enter the path under **`已有项目 / 工作目录` ("Existing Project / Work Directory")** → **`工作目录（可选）` ("Work Directory — Optional")**.

### 4.5 Load / Scan an Existing Project

Click:

```text
载入/扫描工作目录
```

("Load / Scan Work Directory")

DataFactory rescans actual artifacts and refreshes:

- data preparation;
- transcript confirmation;
- Stage1;
- Stage2 sub-steps;
- missing or partially completed states.

> [!IMPORTANT]
> DataFactory recovery is centered on the work directory (`work_dir`). Losing page state, refreshing the browser, or restarting the WebUI does not mean that the existing artifacts in the work directory have disappeared.

---

## 5. Data Preparation and Transcript Confirmation

### 5.1 Start Data Preparation

After completing the basic inputs, click:

```text
开始准备数据
```

("Start Data Preparation")

The default pipeline is approximately:

```text
Scan Raw Audio
  ↓
Build Index
  ↓
Decode / Mono / 32 kHz
  ↓
Silence Slicing
  ↓
Local Offline FunASR
  ↓
Generate manifest.jsonl / dataset.list
  ↓
Generate Initial stage2_manifest.jsonl
  ↓
Basic Health Checks
```

The initial `stage2_manifest.jsonl` is only an intermediate output of data preparation. It **does not mean Stage2 training data is ready**.

### 5.2 Local Offline FunASR and Success Condition

The official release path uses:

```text
language = zh
backend = auto → local FunASR
```

FunASR uses the locally bundled ASR, VAD, and punctuation models. At runtime, these models are not downloaded from ModelScope or Hugging Face, and update checks are disabled. If model assets are missing, inspect the release package or environment rather than waiting for an online download.

After data preparation succeeds, the page should report:

```text
音频处理与基础识别产物已就绪。
```

("Audio processing and basic recognition outputs are ready.")

A work-directory scan should detect at least:

```text
06_export/manifest.jsonl
06_export/dataset.list
```

If the task fails or is interrupted, inspect Current Task / Diagnostics and rescan the work directory before deciding to rebuild the project.

### 5.3 Manual Proofreader

Click:

```text
启动 / 打开人工校对器
```

("Launch / Open Manual Proofreader")

The proofreader reads the current `dataset.list`. Review the ASR text against the corresponding audio, correct obvious wrong characters, omissions, or recognition errors, and save the changes using the proofreader's own save function. If a proofreader for the same `work_dir` is already running, the release prevents another duplicate instance from being launched.

DataFactory also provides a separate:

```text
关闭人工校对器
```

("Close Manual Proofreader")

This is different from:

```text
停止当前任务
```

("Stop Current Task")

The former manages the proofreader process; the latter manages DataFactory / Stage1 / Stage2 build tasks.

### 5.4 Complete Transcript Confirmation

After saving proofreading changes, return to DataFactory and click:

```text
已保存并完成校对
```

("Saved and Finished Proofreading")

If you reviewed the ASR result and confirmed that no edits are needed, click:

```text
无需校对，继续
```

("No Proofreading Needed, Continue")

This accepts the current transcript; it does not skip transcript confirmation.

After success, the page should report:

```text
文本确认结果已就绪。
```

("Transcript confirmation output is ready.")

The key artifact is:

```text
06_export/manifest.corrected.jsonl
```

Stage1 and Stage2 both require transcript confirmation first, but they consume the confirmed data differently:

- **Stage1** mainly uses `06_export/dataset.list` and `06_export/clips/`; the proofreader directly updates `dataset.list`;
- **Stage2** formal postprocessing requires `06_export/manifest.corrected.jsonl`.

---

## 6. Generate Stage1 / Stage2 Training Data

### 6.1 Generate Stage1 Data

Stage1-only and Full Few-shot routes require:

```text
生成 Stage1 训练数据
```

("Generate Stage1 Training Data")

The current user path mainly uses:

```text
06_export/dataset.list
06_export/clips/
```

The output root is:

```text
07_stage1_ft/
```

The default training-set ratio is:

```text
train_ratio = 0.90
```

The validation ratio is automatically `1 - train_ratio`. The split seed is generated by the build process and recorded in:

```text
07_stage1_ft/metadata/split.json
```

On success, the page should report:

```text
Stage1 训练数据已完成。
```

("Stage1 training data is complete.")

Stage1 readiness is not determined by directory existence alone. The release also checks:

```text
07_stage1_ft/metadata/build_report.json
07_stage1_ft/train_manifest.jsonl
07_stage1_ft/val_manifest.jsonl
07_stage1_ft/frontend_cache/
07_stage1_ft/semantic_cache/
```

### 6.2 Stage1 Overwrite Protection and Recovery Boundary

If `07_stage1_ft/` already contains any files and overwrite has not been explicitly authorized, the release blocks a new Stage1 build.

> [!WARNING]
> `覆盖已有 Stage1 cache` ("Overwrite Existing Stage1 Cache") is an explicit rebuild operation. Enable it only when you have decided to rebuild the current Stage1 data.

Stage1 **does not support** Stage2-style artifact-based Resume:

- already fully ready → keep the existing result;
- partially complete or failed → inspect the work directory and logs first;
- confirmed rebuild required → explicitly enable `覆盖已有 Stage1 cache`.

After a successful overwrite rebuild, the release removes stale cache entries that are no longer referenced by the new manifest and runs strict cache-contract validation.

### 6.3 Generate Stage2 Data

Stage2-only and Full Few-shot routes require:

```text
生成 Stage2 训练数据
```

("Generate Stage2 Training Data")

Stage2 is a multi-step chain:

```text
manifest.corrected.jsonl
  ↓
stage2_manifest.fewshot.jsonl
  ↓
Stage2 .pt
  ↓
continuous semantic cache
  ↓
Style / F0 / Speaker cache
  ↓
Train / Validation Split
  ↓
Bad-Sample Filtering
```

Default Prompt-selection rules:

| Parameter | Default |
| --- | ---: |
| `prompt_mode` | `speaker_pool` |
| `min_prompt_sec` | `3.0` |
| `max_prompt_sec` | `10.0` |
| `prefer_prompt_sec` | `6.0` |
| `allow_self_prompt` | `True` |

The 3–10 second range here is the **reference-audio Prompt-selection rule used during Stage2 data construction**. Do not confuse it with the 3–10 second Prompt WAV that the Inference UI requires from the user.

Main Stage2 artifacts are stored in:

```text
07_stage2_pt_all/
08_continuous_semantic/
09_style_f0_spk/
10_train_val/
11_filter/
```

The v1.0.0 User Edition always enables automatic bad-sample quarantine. The final filter uses `move` semantics: rejected samples are moved into:

```text
11_filter/rejected/
```

and JSON / CSV filter reports are generated. Therefore, having fewer trainable samples after filtering than before filtering can be normal.

Stage2 is considered ready only when the Few-shot manifest, Stage2 `.pt` files, continuous / style caches, train / validation split, and valid filter results consistent with the current split and file counts are all complete. On success, the user summary should report:

```text
Stage2 训练数据已完成。
```

("Stage2 training data is complete.")

### 6.4 Stage2 Artifact-Based Resume

After interruption, restart, page refresh, or fixing a failed step, use:

```text
继续生成 Stage2 训练数据
```

("Continue Generating Stage2 Training Data")

Recommended recovery flow:

```text
Restart DataFactory
  ↓
Enter the Original Speaker / Character Name and Work Directory
  ↓
Load / Scan Work Directory
  ↓
Confirm Actual Artifact State
  ↓
Continue Generating Stage2 Training Data
```

Resume uses the actual artifacts to determine what can be reused:

- complete and still-valid key artifacts may be skipped;
- existing but incomplete `.pt` / caches are completed when possible;
- explicitly enabling overwrite forces the corresponding step to run again;
- expired or filesystem-inconsistent split / filter reports are not treated as valid artifacts;
- if a real step fails, the downstream chain stops instead of presenting the failure as success.

> [!IMPORTANT]
> After a Stage2 interruption, prefer **scan state + Resume**. Do not turn on every overwrite option and rebuild everything by default.

### 6.5 Completion Conditions by Route

| Route | DataFactory Must Confirm |
| --- | --- |
| Stage1-only | Transcript confirmed + Stage1 ready |
| Stage2-only | Transcript confirmed + Stage2 ready |
| Full Few-shot | Stage1 ready + Stage2 ready |

For Training handoff, rely on the independent Stage1 / Stage2 readiness states rather than only checking whether a directory exists or a button has been executed.

---

## 7. Stop Tasks, Status, and Diagnostics

### 7.1 Stop the Current Task

Under **`已有项目 / 工作目录` ("Existing Project / Work Directory")**, click:

```text
停止当前任务
```

("Stop Current Task")

The system attempts to stop DataFactory build subprocesses associated with the current work directory. Rescan the work directory afterward, because already-completed steps may have left partial artifacts.

Recommended sequence:

```text
Stop Current Task
  ↓
Load / Scan Work Directory
  ↓
Inspect Actual State
  ↓
Choose Resume / Fix / Explicit Overwrite
```

`停止当前任务` does not close the proofreader.

### 7.2 Live Status

Under **`当前任务` ("Current Task")**, you can use **`立即刷新` ("Refresh Now")**. Long-running tasks also refresh automatically.

User-facing states include:

| Status | Meaning |
| --- | --- |
| `尚未开始` ("Not Started") | No valid progress |
| `等待前置步骤` ("Waiting for Prerequisites") | An upstream step is incomplete |
| `等待确认` ("Waiting for Confirmation") | Data preparation is complete and transcript confirmation is pending |
| `部分完成` ("Partially Complete") | Valid artifacts exist but the stage is incomplete |
| `已完成` ("Complete") | The current stage passes its status checks |

Diagnostics may also expose the internal enum `missing / blocked / partial / done`. For Training handoff, the important signal is independent **Stage1 / Stage2 readiness**, not the existence of any single directory.

### 7.3 Diagnostics and Logs

Normal use usually does not require expanding **`诊断信息` ("Diagnostics")**. For troubleshooting, inspect recent operations, work-directory state, step summaries, proofreader status, and DataFactory Result / Live Monitor diagnostic JSON.

Main runtime logs are stored in:

```text
{work_dir}/logs/
```

Stage1 also records separate standard output (`stdout`), standard error (`stderr`), and execution-command information. Identify the failing step first, then inspect the corresponding logs.

---

## 8. Settings and Advanced Parameters

For your first successful DataFactory run, keep the defaults. Change advanced settings only when you know which problem you are trying to solve.

### 8.1 Common Settings and Timeout Boundary

| Setting | Default | Purpose |
| --- | ---: | --- |
| `覆盖已有数据工厂工作目录` | Off | Allow overwrite-style operations during data preparation |
| `覆盖已上传音频缓存` | On | Clear `00_uploaded_raw_audio/` before upload |
| `单步超时秒数，0 表示不限制` | `0` | Control per-step timeout |
| Stage1 training ratio | `0.90` | Stage1 train / validation split |
| Stage2 training ratio | `0.90` | Stage2 train / validation split |
| `显示终端进度窗口` | On | Show an additional read-only log terminal |

> [!NOTE]
> In v1.0.0, timeout value `0` is handled differently by different execution paths: Stage1 / Stage2 normalize `0` to "no explicit timeout", while Data Preparation falls back to the internal default of `7200` seconds. If Data Preparation may take more than two hours, enter a larger positive value instead of relying on `0` for unlimited runtime.

> [!WARNING]
> To restore an existing project, first **Load / Scan Work Directory**. Prefer Resume for Stage2; handle partial Stage1 output according to the overwrite-protection rules in §6.2. Do not enable every overwrite option and rebuild from scratch as the default recovery strategy.

### 8.2 ASR Settings

The official v1.0.0 user interface locks **ASR Backend / ASR Model Size / ASR Precision / Language** to:

```text
ASR backend = auto
ASR model size = large
ASR precision = float32
language = zh
```

`auto` enters the local FunASR path. Backend, model size, and precision are not editable in the final release UI, and language is limited to `zh`.

### 8.3 Advanced Settings Reference

These settings remain available in the release UI, but normal Few-shot users usually do not need to change them.

| Category | Setting | Default / State | User Boundary |
| --- | --- | --- | --- |
| Audio slicing | `threshold` | `-34` | Change only if the default slices are obviously too long / too fragmented |
| Audio slicing | `min_length` | `4000` | Same |
| Audio slicing | `min_interval` | `300` | Same |
| Audio slicing | `hop_size` | `10` | Same |
| Audio slicing | `max_sil_kept` | `500` | Same |
| Audio slicing | `normalize_max` | `0.9` | Same |
| Audio slicing | `alpha_mix` | `0.25` | Same |
| Stage1 | device | `cuda` | Device used for Stage1 data construction |
| Stage1 | Stage1 use_half | On | Allow half precision in the corresponding model path |
| Stage1 | `覆盖已有 Stage1 cache` | Off | Explicit rebuild |
| Stage1 | `构建后验证 dataset` | On | Validate the dataset contract after building |
| Stage2 | device | `cuda` | Device used for Stage2 data processing |
| Stage2 | Stage2 use_half | Off | Precision for Stage2 `.pt` processing |
| Stage2 | `覆盖已有 .pt` | Off | Force rebuild of the corresponding artifacts |
| Stage2 | `覆盖 continuous cache` | Off | Force rebuild of the corresponding cache |
| Stage2 | `覆盖 style cache` | Off | Force rebuild of the corresponding cache |
| Stage2 | `覆盖 train/val` | Off | Force a new train / validation split |
| Stage2 | continuous dtype | `float32` | Data type of the continuous semantic cache |
| Proofreader | port | `9871` | Usually change only for a port conflict |
| Proofreader | `g_batch` | `10` | Proofreader batch setting |
| Proofreader | `覆盖已有校对备份` | Off | Control whether an existing backup is overwritten |
| Proofreader | `重建 corrected manifest 时长容差` | `0.05` | Matching tolerance used during rebuild |
| Proofreader | `重建超时秒数` | `600` | Timeout used during post-proofreading rebuild |

Stage2 Prompt-selection defaults are already documented in §6.3 and are not repeated here.

Mel / F0 feature defaults:

```text
target_sr = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 80
fmin = 0.0
fmax = 8000.0
f0_min_hz = 50.0
f0_max_hz = 1100.0
```

> [!IMPORTANT]
> Mel / F0 settings directly participate in the Stage2 data contract. Normal users should not change them casually just to "try different results."

---

## 9. Work Directory, Key Artifacts, and Training Handoff

Normal users do not need to memorize every internal file. The most important things are the current work directory (`work_dir`), transcript-confirmation result, and whether Stage1 / Stage2 is ready.

### 9.1 Main Directories Users Should Know

```text
{speaker_name}_factory/
├── 00_uploaded_raw_audio/       # upload modes only
├── 06_export/                   # clips / manifests
├── 07_stage1_ft/                # Stage1 data
├── 07_stage2_pt_all/            # Stage2 .pt
├── 08_continuous_semantic/
├── 09_style_f0_spk/
├── 10_train_val/
├── 11_filter/
└── logs/
```

`01_uvr_*` and `02_denoise/` may also exist in the base directory structure, but the official v1.0.0 user workflow does not expose UVR / denoise as enabled features. Empty directories here do not indicate failure.

### 9.2 Key Artifacts

| Artifact | User-Level Meaning |
| --- | --- |
| `06_export/clips/` | normalized and sliced training audio |
| `06_export/manifest.jsonl` | base manifest from data preparation |
| `06_export/dataset.list` | important list for Stage1 / proofreading |
| `06_export/manifest.corrected.jsonl` | primary manifest after transcript confirmation |
| `06_export/stage2_manifest.fewshot.jsonl` | formal Stage2 Few-shot manifest |
| `07_stage1_ft/` | data and caches required for Stage1 readiness |
| `07_stage2_pt_all/` through `11_filter/` | Stage2 multi-step data chain |
| `11_filter/rejected/` | quarantined bad Stage2 samples |
| `logs/` | troubleshooting logs |

Data preparation may also create an initial `stage2_manifest.jsonl` and other intermediate files. Those files alone do not prove that Stage2 is ready.

### 9.3 Training Uses the Same Project Identity

When you move to Training, continue using the same **Speaker / Character Name** and work directory (`work_dir`):

- default rule: `user_data/{speaker_name}_factory`;
- if DataFactory used a custom work directory, enter the same directory in Training.

DataFactory only prepares the route-specific data until it is trainable. Epochs, batch size, checkpoints, and Active Best are covered by the [Training Guide](./training.md).

---

## 10. Common Boundaries and Safe Operations

| Common Misunderstanding | Correct Interpretation |
| --- | --- |
| Supported extension = guaranteed decode | No. Codec, container, and file integrity still affect decoding |
| MP3 / AAC → WAV restores quality | No. Changing containers does not restore information lost by lossy encoding |
| Partial Stage1 can Resume directly | No. Stage1 does not support Stage2-style artifact-based Resume; explicitly authorize overwrite to rebuild |
| Interrupted Stage2 should restart from zero | Not recommended. Scan first, then use `继续生成 Stage2 训练数据` |
| Closing the progress terminal stops the task | No. The read-only progress terminal can be closed; use the WebUI stop control to stop the actual task |
| `停止当前任务` also closes the proofreader | No. The proofreader has its own close control |
| Few-shot always requires both Stage1 and Stage2 | No. Stage1-only and Stage2-only are both officially supported routes |
| Current `zh + local FunASR` is a permanent product capability | No. It is the Resonastra v1.0.0 release boundary |

---

## 11. Next Steps and Related Documentation

After DataFactory, the next stage is Training:

```text
Quick Start
  ↓
DataFactory
  ↓
Training
  ↓
Inference
```

Related documentation:

- [Quick Start](./quick-start.md)
- [Training Guide](./training.md)
- [Inference Guide](./inference.md)
- [Compatibility Guide](./compatibility.md)
