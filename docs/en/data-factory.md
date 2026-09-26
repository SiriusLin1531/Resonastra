# Resonastra DataFactory Guide

[Project README](https://github.com/SiriusLin1531/Resonastra/blob/main/README.md) | [Quick Start](./quick-start.md) | **English** | [中文简体](../cn/data-factory.md)

Applies to: **Resonastra v1.0.0**

> [!NOTE]
> This guide documents the official **Resonastra v1.0.0** DataFactory behavior. If a later release changes the DataFactory UI, ASR routing, data layout, or training-data contract, refer to the documentation for that release.

---

## 1. What This Guide Covers

Resonastra DataFactory prepares user-provided raw speech so it can be used by Stage1 / Stage2 Training.

If you only use the bundled Zero-shot models, you do not need DataFactory. This guide is required when preparing Stage1-only, Stage2-only, or Full Few-shot training data.

After completing the sections for your selected route, you should be able to:

- choose an appropriate raw-audio input method;
- determine whether an audio format is part of the official v1.0.0 input contract;
- create or restore a DataFactory work directory;
- complete audio slicing and local offline FunASR transcription;
- review and confirm the ASR transcript;
- generate Stage1 or Stage2 training data independently;
- determine whether the data required by your selected route is truly ready;
- continue after an interrupted task, page refresh, or WebUI restart;
- hand the same work directory to Training.

This guide does not cover Stage1 / Stage2 training parameters, checkpoint / Active Best management, Voice Profiles, Inference parameters, GPU / CUDA compatibility details, or comprehensive troubleshooting. Those topics belong to the Training, Inference, Compatibility, and Troubleshooting guides.

---

## 2. Where DataFactory Fits in Resonastra

DataFactory is the data-preparation layer before Few-shot training:

```text
Raw Audio
  ↓
Decode / Normalize
  ↓
Slicing
  ↓
Local Offline FunASR
  ↓
Transcript Confirmation
  ↓
Canonical Corrected Manifest
  ├─→ Stage1 Training Data
  └─→ Stage2 Training Data
          ↓
       Training
```

DataFactory does not train the models themselves. DataFactory is complete when **the data required by your selected route is ready**, not merely when a particular button finishes running.

| Route | DataFactory Required? | What DataFactory Must Complete |
| --- | --- | --- |
| Zero-shot | No | None |
| Stage1-only Few-shot | Yes | Prepare + transcript confirmation + Stage1 data |
| Stage2-only Few-shot | Yes | Prepare + transcript confirmation + Stage2 data |
| Full Few-shot | Yes | Prepare + transcript confirmation + Stage1 + Stage2 data |

> [!IMPORTANT]
> Stage1 and Stage2 data can be generated independently. Do not treat "both Stage1 and Stage2 complete" as a universal completion requirement for every Few-shot route.

---

## 3. Before You Start

Before opening DataFactory:

1. fully extract Resonastra v1.0.0;
2. run `launch_check_env.bat` and confirm that the environment check passes;
3. prepare your Chinese speech data;
4. choose a **Speaker / Character Name** for this dataset.

### 3.1 v1.0.0 Language and ASR Boundary

The official v1.0.0 DataFactory release exposes only:

```text
language = zh
```

The corresponding recognition path uses local offline FunASR.

### 3.2 Data-Quality Recommendations

For more stable Few-shot data, the source recordings should preferably have:

- a single speaker;
- clear pronunciation;
- normal speaking speed;
- little or no obvious background noise;
- few abnormal edits, unusual silence patterns, or clearly damaged files;
- high-quality source audio with as little repeated lossy transcoding as possible.

These are **training-quality recommendations**, not a complete list of mandatory DataFactory input constraints.

---

## 4. Start DataFactory

From the Resonastra root directory, run:

```text
launch_data_factory_ui.bat
```

The default port is:

```text
7861
```

If `7861` is already in use, the launcher automatically selects another available port. Use the WebUI address shown in the launcher terminal.

> [!IMPORTANT]
> The terminal opened by `launch_data_factory_ui.bat` is the **main WebUI launcher terminal**. Keep it open while using DataFactory; closing it ends the corresponding WebUI process.

### 4.1 Main Page Areas

The v1.0.0 DataFactory page is organized into:

- **`基础输入`** ("Basic Input") — raw-audio source, speaker / character name, language, and work directory;
- **`执行流程`** ("Execution Flow") — Prepare, proofreading, Stage1, and Stage2;
- **`设置`** ("Settings") — common and advanced settings;
- **recommended actions / data status** — what the current project should do next;
- **`当前任务`** ("Current Task") — long-running tasks and Stage2 sub-step status;
- **proofreader status** — whether the proofreading WebUI is running;
- **`诊断信息`** ("Diagnostics") — detailed status, paths, steps, and JSON.

For a first run, you do not need to expand every advanced setting. Use the defaults first.

### 4.2 Main Launcher Terminal vs. Progress Terminal

The **`显示终端进度窗口`** ("Show Terminal Progress Window") option is enabled by default. Long-running Prepare, Stage1, or Stage2 tasks may open an additional log terminal.

| Window | Purpose | Can You Close It? |
| --- | --- | --- |
| Main WebUI launcher terminal | Keeps DataFactory WebUI running | Do not close it while using DataFactory |
| Progress / log terminal | Read-only live task logs | Yes. Closing it does not stop the real task |

---

## 5. Prepare the Raw Speech Data

### 5.1 Three Raw-Audio Input Methods

Under **`原始音频来源`** ("Raw Audio Source"), DataFactory provides three input methods.

#### Method A: `本地目录路径` ("Local Directory Path")

This is the recommended method for larger datasets and normal training workflows.

Enter your source directory in:

```text
本地原始音频目录
```

("Local Raw Audio Directory")

DataFactory recursively scans the directory and its subdirectories for supported audio files.

Because this method does not first copy the full dataset into the browser-upload cache, it is better suited to datasets ranging from tens of minutes to multiple hours or more.

#### Method B: `上传音频文件` ("Upload Audio Files")

This mode provides a multi-file selector for choosing local audio files.

Supported uploaded files are first copied into:

```text
{work_dir}/00_uploaded_raw_audio/
```

DataFactory then uses that cache directory as the raw-audio input directory.

This mode is better suited to small samples, quick tests, or small additions to a dataset.

#### Method C: `上传音频文件夹` ("Upload Audio Folder")

This mode lets you select a complete local folder.

Supported audio files from the selected folder are also copied into:

```text
{work_dir}/00_uploaded_raw_audio/
```

and then enter the same data-preparation pipeline.

> [!NOTE]
> Upload modes are convenience features. Large Few-shot datasets do not need to pass through browser upload. For larger datasets, `本地目录路径` remains the preferred input method.

### 5.2 Audio Formats Officially Recognized by v1.0.0

The official extension allowlist is:

```text
.wav
.mp3
.flac
.ogg
.m4a
.aac
.wma
.opus
```

Directory scanning compares lowercase extensions, so uppercase forms such as `.WAV` and `.FLAC` are also recognized.

A single raw-audio directory may contain multiple supported formats, such as WAV, FLAC, and MP3 together. DataFactory normalizes them to its internal downstream format.

> [!IMPORTANT]
> An extension being on the allowlist means DataFactory will attempt to process the file. It does not guarantee that every internal codec variant, container variation, or damaged file can be decoded successfully.

Formats outside the allowlist are not official v1.0.0 Raw Audio inputs. For example, WEBM, MP4, AIFF/AIF, CAF, and AMR are not part of the official input contract.

### 5.3 WAV / FLAC Are Preferred for Training Quality

From a compatibility perspective, all eight formats above can enter the official pipeline.

For training-data quality, good-quality source files in the following formats are preferred when available:

```text
WAV
FLAC
```

Lossy sources such as MP3, AAC, M4A, OGG, OPUS, and WMA are also supported, but information already lost during lossy encoding is not restored when DataFactory later writes the audio as WAV.

Therefore:

- an existing high-quality MP3/AAC dataset does not need to be manually converted to WAV merely for format compliance;
- if you also have the original lossless WAV / FLAC source, prefer the original lossless version.

### 5.4 Automatic Audio Normalization

You do not need to manually convert every source file to mono, 32 kHz, or PCM16 before using DataFactory.

The slicing path reads and normalizes supported audio as follows:

```text
supported raw audio
  ↓
decode
  ↓
mono
  ↓
32 kHz
  ↓
silence slicing
  ↓
PCM 16-bit WAV clips
```

The resulting standard short clips are stored as `.wav` files for downstream ASR and training-data preparation.

### 5.5 Upload-Cache Overwrite Behavior

The common setting:

```text
覆盖已上传音频缓存
```

("Overwrite Uploaded Audio Cache")

is enabled by default.

It only controls:

```text
{work_dir}/00_uploaded_raw_audio/
```

and does not delete your original local audio directory.

If you disable this option, uploading a file whose name already exists causes DataFactory to create a non-conflicting destination name instead of directly overwriting the existing cached file.

### 5.6 Common Raw-Input Failure Boundaries

Before or during Prepare, failures can occur when:

- the local directory does not exist;
- the directory contains no audio with a supported extension;
- the upload selection contains no supported audio;
- an uploaded file cannot be copied;
- the extension is supported but the internal audio cannot be decoded;
- an individual source file is damaged.

If an error occurs during slicing, do not assume DataFactory will always skip every bad file and continue through the rest of the dataset. The current slicing chain does not define per-file failure-and-continue behavior as its default fault-tolerance contract. Locate the problematic source file first.

---

## 6. Speaker Name and Work Directory

### 6.1 `说话人 / 角色名` ("Speaker / Character Name")

Enter a name for the dataset, for example:

```text
Character_A
```

If you do not manually specify a work directory, DataFactory automatically uses:

```text
user_data/Character_A_factory
```

The general rule is:

```text
user_data/{speaker_name}_factory
```

When you move to Training, continue using the same character name and the same work directory.

### 6.2 Custom Work Directory

Expand:

```text
已有项目 / 工作目录
```

("Existing Project / Work Directory")

and fill in:

```text
工作目录（可选）
```

("Work Directory — Optional")

Use a custom work directory when:

- you want to store the current DataFactory project in a custom location;
- you want to restore a previous project after a page refresh;
- you restarted Resonastra and want to continue an existing project;
- you already know the path of an existing project and want to scan its status.

If the field is empty, DataFactory continues to use `user_data/{speaker_name}_factory`.

### 6.3 Load / Scan an Existing Work Directory

After entering an existing work directory or speaker name, click:

```text
载入/扫描工作目录
```

("Load / Scan Work Directory")

DataFactory rescans existing artifacts and refreshes:

- whether Prepare is complete;
- whether transcript confirmation is complete;
- whether Stage1 data is complete;
- whether each Stage2 sub-step is complete;
- which steps are missing or partial.

This is one of the most important controls for restoring an existing project.

> [!IMPORTANT]
> DataFactory recovery is centered on `work_dir`. Losing page state, refreshing the browser, or restarting the WebUI does not mean the real artifacts in the work directory have disappeared.

---

## 7. Work-Directory Layout and Key Artifacts

Normal users do not need to memorize every internal file, but understanding the main directories makes status checks and troubleshooting easier.

### 7.1 Base Processing Directories

A typical work directory contains:

```text
{speaker_name}_factory/
├── 00_uploaded_raw_audio/   # upload modes only
├── 00_raw_index/
├── 01_uvr_vocal/
├── 01_uvr_other/
├── 02_denoise/
├── 03_clips_raw/
├── 04_asr/
├── 05_label/
├── 06_export/
└── logs/
```

`01_uvr_*` and `02_denoise/` are retained directory structures. The official v1.0.0 user workflow does not expose UVR / denoise as enabled DataFactory features, so empty directories here do not by themselves indicate failure.

### 7.2 `06_export/` Is the Base Data Export

Important Stage1 / Stage2 files appear here:

| File / Directory | User-Level Meaning |
| --- | --- |
| `clips/` | normalized and sliced training audio |
| `manifest.jsonl` | base manifest after Prepare |
| `dataset.list` | list used by Stage1 data-building and related steps |
| `stage2_manifest.jsonl` | base Stage2 manifest |
| `manifest.corrected.jsonl` | primary corrected manifest after transcript confirmation |
| `stage2_manifest.corrected.jsonl` | corrected Stage2 counterpart |
| `stage2_manifest.fewshot.jsonl` | manifest used by formal Stage2 Few-shot postprocessing |

Health, prompt-selection, conversion, and other reports also appear in this area.

> [!NOTE]
> Prepare already generates the initial `stage2_manifest.jsonl` and runs its corresponding health check, but this **does not mean Stage2 training data is complete**. After transcript confirmation, formal Stage2 Few-shot postprocessing creates `stage2_manifest.fewshot.jsonl` from `manifest.corrected.jsonl`, then builds Stage2 `.pt`, caches, train / val data, and filter reports.

### 7.3 Stage1 Data Directory

Final Stage1 data is stored in:

```text
07_stage1_ft/
```

Typical contents:

```text
07_stage1_ft/
├── frontend_cache/
├── semantic_cache/
├── train_manifest.jsonl
├── val_manifest.jsonl
└── metadata/
    ├── build_report.json
    └── split.json
```

### 7.4 Stage2 Compact Layout

In v1.0.0, heavy Stage2 artifacts are consolidated under the same `{speaker_name}_factory` work directory:

```text
07_stage2_pt_all/
08_continuous_semantic/
09_style_f0_spk/
10_train_val/
├── train/
├── val/
└── split_report.json
11_filter/
├── rejected/
├── filter_bad_style_samples_report.json
└── filter_bad_style_samples_bad.csv
```

This keeps stopping, scanning, resuming, and Training handoff centered on one work directory.

### 7.5 Two Levels Normal Users Should Remember

**Must know:**

- the current `work_dir`;
- whether `manifest.corrected.jsonl` has been created;
- whether Stage1 / Stage2 has reached the ready state required by your route.

**Useful for troubleshooting:**

- `logs/`;
- build / health / split / filter reports;
- train / val and cache counts.

---

## 8. Step 1: Start Data Preparation

After entering the basic inputs, click:

```text
开始准备数据
```

("Start Data Preparation")

### 8.1 Recommended First-Run Inputs

For a first run, use:

```text
原始音频来源：本地目录路径
本地原始音频目录：your raw-audio directory
说话人 / 角色名：for example Character_A
语言：zh
工作目录：leave empty
```

This causes DataFactory to use:

```text
user_data/Character_A_factory
```

### 8.2 What Prepare Actually Does

A single **`开始准备数据`** operation runs a sequence of steps:

```text
scan raw audio
  ↓
build raw index
  ↓
decode / mono / 32 kHz
  ↓
silence slicing
  ↓
local offline FunASR
  ↓
generate manifest.jsonl / dataset.list
  ↓
generate initial stage2_manifest.jsonl
  ↓
run base manifest / Stage2 manifest health checks
```

These initial Stage2 artifacts support later proofreading and Few-shot processing. They are not equivalent to final Stage2 training-data readiness.

### 8.3 Local Offline FunASR

The official Resonastra v1.0.0 DataFactory route uses:

```text
language = zh
backend = auto → local FunASR
```

FunASR uses locally bundled:

- ASR model assets;
- VAD model assets;
- punctuation model assets.

At runtime, these models are not downloaded from ModelScope or Hugging Face, and FunASR update checks are disabled.

If the local model assets are missing, inspect the release package or environment rather than waiting for the program to download them.

### 8.4 Prepare Success Condition

After Prepare finishes, the user-facing Audio Processing stage should report:

```text
音频处理与基础识别产物已就绪。
```

("Audio processing and basic recognition outputs are ready.")

A work-directory scan should at minimum detect:

```text
06_export/manifest.jsonl
06_export/dataset.list
```

You can then move to transcript confirmation.

### 8.5 If Prepare Fails, Check the Existing State First

If Prepare fails or is interrupted, do not immediately delete the entire work directory.

Use this order:

1. inspect **`当前任务`** ("Current Task") and the progress terminal;
2. expand **`诊断信息`** ("Diagnostics") and inspect the latest operation;
3. click **`载入/扫描工作目录`** ("Load / Scan Work Directory");
4. check which artifacts already exist;
5. decide whether to continue, fix the input, or explicitly overwrite.

---

## 9. Step 2: Confirm the ASR Transcript

ASR output becomes part of the downstream training data. Significant mismatches between the audio and transcript can reduce data quality, so transcript confirmation should normally be completed before Stage1 / Stage2 data generation.

### 9.1 Launch the Manual Proofreader

Click:

```text
启动 / 打开人工校对器
```

("Launch / Open Manual Proofreader")

DataFactory starts the proofreader as a separate WebUI and displays its status and access address.

If a proofreader for the same work directory is already running, the release prevents another duplicate instance from being launched.

### 9.2 What to Do in the Proofreader

The proofreader loads the current work directory's `dataset.list`. Review the ASR text against the corresponding audio entries, correct obvious wrong characters, omissions, or recognition errors, and save the changes using the proofreader's own save function.

The goal is not to rewrite what was said, but to make the training text match the spoken audio as closely as possible.

### 9.3 The Proofreader Has an Independent Lifecycle

DataFactory provides a dedicated control:

```text
关闭人工校对器
```

("Close Manual Proofreader")

This is different from:

```text
停止当前任务
```

("Stop Current Task")

- `停止当前任务` manages DataFactory / Stage1 / Stage2 build subprocesses.
- `关闭人工校对器` manages the proofreader process.

When DataFactory needs to rebuild corrected manifests from saved proofreading results, it ensures the proofreader can be safely closed so that the proofreader and rebuild operation are not modifying the same data at the same time.

### 9.4 Finish After Saving Proofreading Changes

After saving changes in the proofreader, return to DataFactory and click:

```text
已保存并完成校对
```

("Saved and Finished Proofreading")

DataFactory rebuilds the corrected manifests from the saved proofreading result.

### 9.5 If the ASR Result Needs No Changes

If you have reviewed the automatic transcript and confirmed that it can be used as-is, click:

```text
无需校对，继续
```

("No Proofreading Needed, Continue")

This does not mean "skip transcript confirmation." It means **accept the current recognition result as the downstream training transcript** and create the corrected manifests needed by later steps.

### 9.6 Transcript-Confirmation Success Condition

The user-facing Text Confirmation stage should report:

```text
文本确认结果已就绪。
```

("Text confirmation result is ready.")

The key artifact is:

```text
06_export/manifest.corrected.jsonl
```

In the normal user workflow, both Stage1 and Stage2 should follow transcript confirmation, but the two data-building paths do not read exactly the same file:

- the **Stage1** builder reads `06_export/dataset.list`; the proofreader edits this list directly, while `无需校对，继续` means accepting its current contents;
- formal **Stage2** postprocessing explicitly requires `06_export/manifest.corrected.jsonl` and will not start without it.

---

## 10. Step 3A: Generate Stage1 Training Data

The following routes require Stage1 data:

- Stage1-only Few-shot;
- Full Few-shot.

Click:

```text
生成 Stage1 训练数据
```

("Generate Stage1 Training Data")

### 10.1 Stage1 Input Data

The current user path primarily uses:

```text
06_export/dataset.list
06_export/clips/
```

It builds Stage1 frontend / semantic caches and train / val manifests under the current work directory.

### 10.2 Default Stage1 Split

The default **Stage1 training-set ratio** is:

```text
train_ratio = 0.90
```

The validation ratio automatically uses:

```text
1 - train_ratio
```

Normal users do not enter the Stage1 split seed manually. The builder generates an effective seed and records it in:

```text
07_stage1_ft/metadata/split.json
```

By default, all input samples are processed.

### 10.3 Stage1 Overwrite Protection

If the current work directory already contains Stage1 training data or cache files and you have not explicitly authorized overwrite, the release prevents an accidental replacement build.

> [!WARNING]
> **`覆盖已有 Stage1 cache`** ("Overwrite Existing Stage1 Cache") is an explicit rebuild operation. Enable it only when you have decided to rebuild the existing Stage1 data.

Stage1 and Stage2 use different recovery models. **Stage1 does not provide an artifact-based Resume button.** If any file already exists under `07_stage1_ft/`—including partial output from a failed or interrupted build—clicking **Generate Stage1 Training Data** again without overwrite authorization is blocked before the builder starts.

Therefore:

- if Stage1 is already fully ready, keep the existing result;
- if Stage1 is partial / failed and you decide to rebuild it, first check the work directory and logs, then explicitly enable `覆盖已有 Stage1 cache` and generate Stage1 data again.

After a successful overwrite build, release hardening removes stale cache files no longer referenced by the new manifests and runs a strict cache-contract check so the train / val manifests remain consistent with the frontend / semantic caches.

### 10.4 Stage1 Success Condition

The Stage1 user-facing stage should report:

```text
Stage1 训练数据已完成。
```

("Stage1 training data is complete.")

Stage1 readiness does not depend only on whether `07_stage1_ft/` exists. The status check also validates:

- the build status in `metadata/build_report.json`;
- `train_manifest.jsonl`;
- `val_manifest.jsonl`;
- `frontend_cache/`;
- `semantic_cache/`.

The main output directory is:

```text
{work_dir}/07_stage1_ft/
```

If your route is Stage1-only, DataFactory is complete for that route and you can proceed to Training.

---

## 11. Step 3B: Generate Stage2 Training Data

The following routes require Stage2 data:

- Stage2-only Few-shot;
- Full Few-shot.

Click:

```text
生成 Stage2 训练数据
```

("Generate Stage2 Training Data")

### 11.1 Stage2 Generation Is a Multi-Step Chain

One button operation runs several consecutive stages:

```text
manifest.corrected.jsonl
  ↓
Stage2 few-shot manifest
  ↓
Stage2 .pt preprocess
  ↓
continuous semantic cache
  ↓
Style / F0 / Speaker cache
  ↓
train / val split
  ↓
bad-sample filter report
```

Stage2 data preparation is therefore substantially heavier than Stage1 and benefits more from the Live Monitor and resume workflow.

### 11.2 Stage2 Few-shot Manifest

The first Stage2 postprocessing step exports:

```text
06_export/manifest.corrected.jsonl
```

to:

```text
06_export/stage2_manifest.fewshot.jsonl
```

This step also performs the prompt selection required by the Stage2 dataset.

### 11.3 Default Prompt-Selection Rules

| Parameter | Default | User-Level Meaning |
| --- | ---: | --- |
| `prompt_mode` | `speaker_pool` | select prompts from the speaker's sample pool |
| `min_prompt_sec` | `3.0` | minimum prompt duration |
| `max_prompt_sec` | `10.0` | maximum prompt duration |
| `prefer_prompt_sec` | `6.0` | preferred prompt duration target |
| `allow_self_prompt` | `True` | allow a sample to use itself as prompt when needed |

> [!NOTE]
> The 3–10 second range here belongs to **Stage2 dataset prompt selection**. It is a different operational context from the 3–10 second Prompt WAV uploaded on the Inference page, even though the current release uses the same duration boundary.

### 11.4 Stage2 `.pt` and Downstream Caches

Stage2 then generates:

```text
07_stage2_pt_all/
08_continuous_semantic/
09_style_f0_spk/
10_train_val/
11_filter/
```

The Live Monitor shows multiple sub-steps at the same time. A `partial` state during processing does not necessarily mean failure; it may simply indicate that some samples are complete while the remaining samples are still being processed.

The v1.0.0 user release also enables automatic Stage2 bad-sample quarantine. The final filtering step uses `move` mode, moving detected rejected samples out of train / val into:

```text
11_filter/rejected/
```

It also produces JSON / CSV filter reports. After filtering, the number of train / val files may therefore be lower than the pre-filter source count without indicating a failed Stage2 build.

### 11.5 Stage2 Success Condition

The user-facing summary should report:

```text
Stage2 训练数据已完成。
```

("Stage2 training data is complete.")

At the detailed status level, complete Stage2 readiness normally requires all of the following:

- Stage2 Few-shot manifest;
- Stage2 `.pt`;
- continuous semantic cache;
- Style / F0 / Speaker cache;
- train / val split;
- a valid bad-sample filter report consistent with the current split and file counts.

If your route is Stage2-only, you can proceed to Training at this point; Stage1 data is not required.

---

## 12. Resume Stage2 Data Generation

Because Stage2 is a long multi-step chain, Resonastra provides:

```text
继续生成 Stage2 训练数据
```

("Continue Generating Stage2 Training Data")

### 12.1 When to Use It

Use Stage2 Resume when:

- Stage2 was manually stopped partway through;
- the WebUI was closed and restarted;
- the browser page was refreshed;
- a failed step has been fixed;
- the work directory already contains partial `.pt` or cache output;
- you want to continue Stage2 from an existing project.

### 12.2 Artifact-Based Auto Resume

Resume decides which steps still need to run by inspecting the real artifacts on disk.

Core behavior:

- a step with complete key artifacts can be skipped automatically;
- a partially completed `.pt` / cache directory can continue filling missing samples;
- explicitly enabled overwrite options still cause the corresponding step to rerun;
- stale or count-inconsistent split / filter reports are not treated as ready;
- a real step failure still stops the chain rather than being presented as success.

For example, if `stage2_manifest.fewshot.jsonl` already exists and is not older than the corrected manifest, Resume can skip re-exporting it. If the number of Stage2 `.pt` files already meets the expected sample count and overwrite was not requested, that step can also be skipped.

### 12.3 Recommended Recovery Flow

```text
Restart DataFactory
  ↓
enter the original speaker_name or work_dir
  ↓
click "载入/扫描工作目录"
  ↓
inspect existing artifacts and partial states
  ↓
click "继续生成 Stage2 训练数据"
```

> [!IMPORTANT]
> After an interruption, prefer **scan state + resume**. Do not use "rerun everything from the beginning with all overwrite options enabled" as the default recovery method.

---

## 13. Stop Tasks, Refresh Status, and Use the Live Monitor

### 13.1 Stop the Current Task

Under:

```text
已有项目 / 工作目录
```

("Existing Project / Work Directory")

you can click:

```text
停止当前任务
```

("Stop Current Task")

DataFactory searches for build subprocesses associated with the current work directory and attempts to stop them.

After stopping, DataFactory rescans the work directory because the interrupted step may already have produced partial artifacts.

The normal follow-up flow is:

```text
停止当前任务
  ↓
载入/扫描工作目录
  ↓
inspect the actual remaining state
  ↓
choose Resume / fix / explicit overwrite
```

`停止当前任务` does not close the proofreader. Use **`关闭人工校对器`** for the proofreader process.

### 13.2 Current Task and Live Monitor

The **`当前任务`** ("Current Task") area on the right side shows the current long-running operation. You can manually refresh it with:

```text
立即刷新
```

("Refresh Now")

During long-running tasks, the page also refreshes task state automatically at a fixed interval.

While Stage2 is running, the Live Monitor shows sub-step status without requiring you to wait for the entire callback to finish.

### 13.3 How to Read Status

The four user-facing flow cards use these states:

| User-Facing State | Meaning |
| --- | --- |
| `尚未开始` ("Not Started") | no valid progress yet |
| `等待前置步骤` ("Waiting for Prerequisite") | an upstream step is incomplete |
| `等待确认` ("Waiting for Confirmation") | audio processing is complete and transcript confirmation is required |
| `部分完成` ("Partially Complete") | some valid artifacts exist but the stage is not complete |
| `已完成` ("Complete") | the stage passes its corresponding status checks |

Inside **`诊断信息`** ("Diagnostics"), the internal unified status may show `missing / blocked / partial / done`.

For Training handoff, the most important user-facing signal is the independent **Stage1 / Stage2 readiness**. Treat the target stage as complete for your route when it reports:

```text
可进入训练
```

("Ready for Training")

Do not determine success from the existence of a directory alone.

### 13.4 Diagnostics

Normal users generally do not need to expand **`诊断信息`** during a successful workflow.

When troubleshooting, it can show:

- latest operation;
- work-directory status;
- output summary;
- step summary;
- current work directory;
- proofreader diagnostics;
- DataFactory Result JSON;
- DataFactory Live Monitor JSON.

Use these fields to identify the failed step and confirm the actual paths.

---

## 14. Complete DataFactory for Your Training Route

### 14.1 Stage1-only Few-shot

```text
Prepare
  ↓
Transcript Confirmation
  ↓
Stage1 Data
  ↓
Training
```

Checklist:

- [ ] Prepare is complete.
- [ ] `manifest.corrected.jsonl` is ready.
- [ ] Stage1 training data is complete.
- [ ] the `07_stage1_ft/` build report / manifests / caches pass the status scan.

Stage2 data is not required.

### 14.2 Stage2-only Few-shot

```text
Prepare
  ↓
Transcript Confirmation
  ↓
Stage2 Data
  ↓
Training
```

Checklist:

- [ ] Prepare is complete.
- [ ] `manifest.corrected.jsonl` is ready.
- [ ] Stage2 Few-shot manifest is complete.
- [ ] Stage2 `.pt` is complete.
- [ ] continuous / style caches are complete.
- [ ] train / val is complete.
- [ ] filter report is complete.

Stage1 data is not required.

### 14.3 Full Few-shot

```text
Prepare
  ↓
Transcript Confirmation
  ├─→ Stage1 Data
  └─→ Stage2 Data
          ↓
       Training
```

The final recommendation summary should report:

```text
Stage1 与 Stage2 数据均已完成。
```

("Both Stage1 and Stage2 data are complete.")

Checklist:

- [ ] Prepare is complete.
- [ ] transcript confirmation is complete.
- [ ] Stage1 data is ready.
- [ ] Stage2 data is ready.

---

## 15. Settings Reference

For your first successful DataFactory run, use the defaults whenever possible. Change advanced settings only when you know what problem you are trying to solve.

### 15.1 Common Settings

| UI Setting | Default | Purpose |
| --- | ---: | --- |
| `覆盖已有数据工厂工作目录` ("Overwrite Existing DataFactory Work Directory") | Off | allows Prepare to perform overwrite behavior on an existing work directory |
| `覆盖已上传音频缓存` ("Overwrite Uploaded Audio Cache") | On | clears `00_uploaded_raw_audio/` before a new upload |
| `单步超时秒数，0 表示不限制` ("Per-Step Timeout in Seconds, 0 Means No Limit") | `0` | v1.0.0 does not normalize `0` identically across every execution path; see below |
| Stage1 `train_ratio` | `0.90` | Stage1 train / val split ratio |
| Stage2 `train_ratio` | `0.90` | Stage2 train / val split ratio |
| `显示终端进度窗口` ("Show Terminal Progress Window") | On | opens an additional read-only log terminal for long-running tasks |

> [!NOTE]
> The v1.0.0 UI label says that `0` means "no limit," but the actual execution paths contain a release-specific difference. Stage1 / Stage2 normalize `0` to "no explicit timeout," while Prepare falls back to an internal `7200`-second (2-hour) default. If you expect Prepare to take longer than 2 hours, explicitly enter a larger positive number of seconds instead of relying on `0` for unlimited runtime.

> [!WARNING]
> Do not enable multiple overwrite options without understanding the existing work directory. For normal project recovery, first use `载入/扫描工作目录`; use Stage2 Resume for Stage2, while partial Stage1 output follows the overwrite-protection behavior in Section 10.3.

### 15.2 ASR Settings

The release UI retains an ASR settings area, but the official v1.0.0 user route is locked to:

```text
ASR backend: auto
ASR model size: large
ASR precision: float32
language: zh
```

`auto` resolves to local FunASR.

In the final v1.0.0 release shell, **`ASR 后端`** ("ASR Backend"), **`ASR 模型大小`** ("ASR Model Size"), and **`ASR 精度`** ("ASR Precision") are locked and non-editable. **`语言`** ("Language") exposes only `zh`. These are not normal user-selectable alternatives.

### 15.3 Audio-Slicing Parameters

Default values:

| Parameter | Default | Effect |
| --- | ---: | --- |
| `threshold` | `-34` | silence-detection threshold |
| `min_length` | `4000` | influences the minimum target slice length |
| `min_interval` | `300` | controls the minimum silence interval used for splitting |
| `hop_size` | `10` | slicer analysis step size |
| `max_sil_kept` | `500` | maximum silence retained around slices |
| `normalize_max` | `0.9` | target peak used during slice normalization |
| `alpha_mix` | `0.25` | blend between the original waveform and normalized result |

These values follow the GPT-SoVITS-style silence slicer. Do not change them for a first run. Adjust them only when the default slicing is clearly too long, too fragmented, or retains silence poorly for your dataset.

### 15.4 Prompt Settings

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `prompt_mode` | `speaker_pool` | Stage2 prompt-selection mode |
| `allow_self_prompt` | On | allows a sample to use itself as prompt |
| `min_prompt_sec` | `3.0` | minimum prompt duration |
| `max_prompt_sec` | `10.0` | maximum prompt duration |
| `prefer_prompt_sec` | `6.0` | preferred prompt duration |

Unless you are specifically experimenting with Stage2 dataset design, keep these defaults.

### 15.5 Stage1 Settings

| Parameter / UI Setting | Default | Meaning |
| --- | ---: | --- |
| Stage1 device | `cuda` | use GPU for Stage1 data building |
| Stage1 use_half | On | allow the corresponding model path to use half precision |
| `覆盖已有 Stage1 cache` ("Overwrite Existing Stage1 Cache") | Off | explicitly rebuild existing Stage1 cache |
| `构建后验证 Stage1 dataset` ("Validate Stage1 Dataset After Build") | On | validate the dataset contract after building |

If your GPU or environment cannot run the current GPU path, refer to Compatibility / Troubleshooting instead of changing unrelated settings at random.

### 15.6 Stage2 / Cache Settings

| Parameter / UI Setting | Default | Meaning |
| --- | ---: | --- |
| device | `cuda` | device for Stage2 data processing |
| Stage2 use_half | Off | precision choice for Stage2 `.pt` preprocessing |
| `覆盖已有 Stage2 .pt` ("Overwrite Existing Stage2 .pt") | Off | force a Stage2 `.pt` rebuild |
| `覆盖已有 continuous cache` ("Overwrite Existing Continuous Cache") | Off | force a continuous-cache rebuild |
| `覆盖已有 style cache` ("Overwrite Existing Style Cache") | Off | force a style-cache rebuild |
| `覆盖已有 train/val` ("Overwrite Existing Train/Val") | Off | force a new Stage2 train / val split |
| continuous dtype | `float32` | continuous semantic cache data type |

Resume is designed to reuse already-complete artifacts whenever possible, so these overwrite settings should normally remain off.

### 15.7 Mel / F0 Settings

Defaults:

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

These values directly affect Stage2 `.pt`, mel, F0, and related feature construction and should be treated as advanced settings.

> [!IMPORTANT]
> Normal Few-shot users should not change these feature settings merely to "try different values." They are part of the downstream Stage2 training-data contract.

### 15.8 Proofreader Settings

| UI Setting | Default | Meaning |
| --- | ---: | --- |
| `校对器端口` ("Proofreader Port") | `9871` | manual-proofreading WebUI port |
| `g_batch` | `10` | proofreader batch setting |
| `覆盖已有校对备份` ("Overwrite Existing Proofreading Backup") | Off | whether to replace an existing proofreading backup |
| corrected-manifest duration tolerance | `0.05` | duration-match tolerance while rebuilding corrected manifests |
| `重建超时秒数` ("Rebuild Timeout in Seconds") | `600` | timeout for the post-proofreading rebuild step |

Most users only need to change the proofreader port if the default port is already occupied.

---

## 16. Outputs, Logs, and Training Handoff

### 16.1 Confirm That DataFactory Is Truly Complete

Do not use "the last button did not report an error" as the only success criterion.

Return to your selected route:

- Stage1-only: Stage1 readiness is sufficient;
- Stage2-only: Stage2 readiness is sufficient;
- Full Few-shot: both Stage1 and Stage2 must be ready.

If you are unsure, click:

```text
载入/扫描工作目录
```

("Load / Scan Work Directory")

and review the unified status and recommended next action.

### 16.2 Training Uses the Same Project Identity

When you move to Training, continue using the same:

```text
说话人 / 角色名
```

("Speaker / Character Name")

If DataFactory used the default work-directory rule, Training can locate:

```text
user_data/{speaker_name}_factory
```

from that name.

If DataFactory used a custom `work_dir`, enter the same work directory in Training.

DataFactory's responsibility ends when the data required by the selected route is trainable. Epochs, batch size, checkpoints, and Active Best belong to the Training Guide.

### 16.3 Logs

The main DataFactory runtime logs are stored under:

```text
{work_dir}/logs/
```

Stage1 also records separate build stdout / stderr and command information. When a failure occurs, first use the page diagnostics to identify the failed step, then inspect the relevant log instead of reviewing every file in the project at once.

---

## 17. Common Boundaries and Safe Operation

### 17.1 Supported Extension Does Not Guarantee Successful Decoding

The extension allowlist defines accepted entry formats. The internal codec, file integrity, and decoding support available in the release package still affect whether the audio can actually be read.

### 17.2 Converting Lossy Audio to WAV Does Not Restore Quality

Converting MP3 / AAC or another lossy source to WAV stores the already-compressed result in a lossless container. It does not recover information that was previously lost.

### 17.3 Do Not Use Overwrite as the Default Fix for Every Problem

The work-directory model supports state scanning and recovery. After an interruption, inspect existing artifacts first:

- for Stage2, prefer artifact-based Resume with `继续生成 Stage2 训练数据`;
- Stage1 has no equivalent Resume path; rebuilding partial Stage1 output requires explicit authorization with `覆盖已有 Stage1 cache`.

### 17.4 Do Not Close the Main WebUI Launcher Terminal

The main launcher terminal owns the WebUI process. The read-only progress terminal can be closed when you no longer need it.

### 17.5 Use the Dedicated Proofreader Close Control

`停止当前任务` ("Stop Current Task") and `关闭人工校对器` ("Close Manual Proofreader") manage different process lifecycles.

### 17.6 Partial Few-shot Is a Valid Route

Stage1-only and Stage2-only are both officially supported combinations. DataFactory does not require both Stage1 and Stage2 to be complete for every Few-shot route.

### 17.7 `zh + local FunASR` Is a v1.0.0 Boundary

Do not treat the current Chinese-only DataFactory constraint as a permanent capability boundary for all future Resonastra releases.

---

## 18. Next Step and Related Documentation

After DataFactory is complete for your route, proceed to Training.

Recommended reading order:

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
- Training Guide (not yet published)
- Compatibility Guide (not yet published)
- Troubleshooting Guide (not yet published)
