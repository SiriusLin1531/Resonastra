# Resonastra Training Guide

[Project README](../../README.md) | [Quick Start](./quick-start.md) | [DataFactory](./data-factory.md) | **Training** | [Inference](./inference.md) | [Compatibility](./compatibility.md) | [Release Verification](./release-verification.md) | **English** | [中文简体](../cn/training.md)

Applies to: **Resonastra v1.0.0**

> [!NOTE]
> This guide documents the official **Resonastra v1.0.0** Training behavior. If a later release changes the Training UI, training parameters, checkpoint management, or Voice Profile contract, refer to the documentation for that release.

---

## 1. What This Guide Covers

Resonastra Training turns the Stage1 / Stage2 training data prepared by DataFactory into Few-shot model artifacts that can be used by Voice Profiles.

If you only use the bundled Zero-shot models, you do not need Training. Stage1-only, Stage2-only, and Full Few-shot routes require Training.

After completing the steps required by your selected route, you should be able to:

- scan training inputs from the correct DataFactory work directory;
- determine whether the Stage1 or Stage2 training entry can start;
- adjust the training parameters exposed by the v1.0.0 user edition;
- complete Stage1 or Stage2 Training independently;
- understand the difference between a training run, Trainer Best, and Active Best;
- use Training Live Monitor to inspect progress, GPU / VRAM, and failure summaries;
- stop training and correctly understand the v1.0.0 no-Training-Resume boundary;
- manage historical checkpoints and Active Best;
- generate the Voice Profile required for Stage1-only, Stage2-only, or Full Few-shot;
- hand the Voice Profile to Inference.

This guide does not cover raw audio, ASR, dataset construction, general GPU / CUDA compatibility, Stage1 / Stage2 training-algorithm theory, or research-oriented hyperparameter tuning. Training-specific failures and recovery are covered in the relevant sections of this guide.

---

## 2. Training Workflow and Before You Start

Training sits between DataFactory and Voice Profile / Inference:

```text
DataFactory Training Data
  ↓
Scan / Validate Training Entry
  ↓
Stage1 / Stage2 Training
  ↓
Each Launch Creates an Independent Run
  ↓
Trainer Best
  ↓
Stage1 / Stage2 Active Best
  ↓
Voice Profile
  ↓
Inference
```

### 2.1 Four Routes

| Route | Training Required | Model Sources in Voice Profile |
| --- | --- | --- |
| Zero-shot | None | Stage1 default model + Stage2 default model |
| Stage1-only Few-shot | Stage1 | Stage1 Active Best + Stage2 default model |
| Stage2-only Few-shot | Stage2 | Stage1 default model + Stage2 Active Best |
| Full Few-shot | Stage1 + Stage2 | Stage1 Active Best + Stage2 Active Best |

Stage1 and Stage2 training entries are independent. Partial Few-shot routes do not require both stages to be complete.

### 2.2 Three Core Concepts

The workflow is easiest to understand in three layers:

1. **Data readiness** — whether the current Stage has valid training input;
2. **Training run** — one independent training execution and the logs / checkpoints it produces;
3. **Active Best** — the Few-shot checkpoint currently selected for Voice Profile generation.

> [!IMPORTANT]
> "Latest training run," "Trainer Best from this run," and "current Active Best" are not the same thing. After Active Best has been manually locked, a newer successful run can produce a new Trainer Best without automatically replacing the locked Active Best.

### 2.3 Continue Using the Same DataFactory Work Directory

Before starting, it is recommended to:

1. fully extract Resonastra v1.0.0;
2. run `launch_check_env.bat` and confirm the basic environment check passes;
3. complete the training data required by your route in DataFactory;
4. keep track of the **Speaker / Character Name** and work directory used by DataFactory.

Default work directory:

```text
user_data/{speaker_name}_factory
```

If DataFactory used the default directory rule, enter the same **`说话人 / 角色名`** ("Speaker / Character Name") in Training and leave **`DataFactory 工作目录`** ("DataFactory Work Directory") empty.

If DataFactory used a custom work directory, enter that same directory in Training.

> [!IMPORTANT]
> Training does not copy DataFactory data into a second project directory. The DataFactory work directory (`work_dir`) remains the project identity for Few-shot data, training history, and checkpoints.

---

## 3. Start Training and Scan Data

From the Resonastra root directory, run:

```text
launch_training_ui.bat
```

The default port is:

```text
7863
```

If the port is already in use, the launcher automatically selects another available port. Use the address shown in the launcher terminal.

The main launcher terminal keeps the Training WebUI running. Do not close it during normal use. It is different from the optional separate training-log terminal described later; closing the log viewer does not close the WebUI and does not stop training.

The main page contains:

- **`输入`** ("Input")
- **`训练操作`** ("Training Actions")
- **Checkpoint / Active Best**
- **`生成 Voice Profile v2`** ("Generate Voice Profile v2")
- **`状态`** ("Status")
- **Training Live Monitor**
- **`诊断信息`** ("Diagnostics")

### 3.1 Scan Training Data

The left-side **`输入`** ("Input") section contains:

- **`说话人 / 角色名`** ("Speaker / Character Name")
- **`DataFactory 工作目录`** ("DataFactory Work Directory")
- **`扫描时刷新 DataFactory 状态`** ("Refresh DataFactory Status During Scan")

If you use the default work directory:

1. enter the same speaker / character name used in DataFactory;
2. leave **`DataFactory 工作目录`** empty;
3. keep **`扫描时刷新 DataFactory 状态`** enabled.

If you use a custom work directory, enter it directly. The work directory and speaker name cannot both be empty.

Click:

```text
扫描训练数据
```

("Scan Training Data")

Training rereads the current DataFactory state and creates / refreshes:

```text
{work_dir}/training_interface.json
```

This file is the DataFactory → Training input-contract snapshot.

### 3.2 Focus on Stage1 / Stage2 Training Entry Status

After scanning, the page separately reports:

```text
Stage1 Training：✅ 可以启动
Stage2 Training：✅ 可以启动
```

or:

```text
Stage1 Training：❌ 暂不可启动
Stage2 Training：❌ 暂不可启动
```

Only the Stage required by your selected route needs to pass its entry check.

Before Stage1 can start, Training verifies:

- Stage1 readiness;
- train / validation manifests;
- frontend cache;
- semantic cache;
- the Stage1 output root.

Before Stage2 can start, Training verifies:

- Stage2 readiness;
- Stage2 `.pt` data;
- train / validation split;
- filter / quarantine contract;
- trainable `.pt` files in the train / validation directories;
- the Stage2 base checkpoint required by the official release package.

If required input is missing, Training rejects that Stage launch instead of passing incomplete input into the trainer.

Training Interface, Training Defaults, Live Monitor, Checkpoint Scan, and other diagnostic JSON views are mainly for troubleshooting. During normal use, focus on training-entry status, Training Live Monitor, and Checkpoint / Active Best.

---

## 4. User Training Parameters

Under **`本次训练参数（仅当前 run）`** ("Training Parameters for This Run Only"), the v1.0.0 user edition exposes only four training parameters:

| Stage | Parameter | Default |
| --- | --- | ---: |
| Stage1 | **Stage1 epochs** | `10` |
| Stage1 | **Stage1 batch_size** | `1` |
| Stage2 | **Stage2 epochs** | `20` |
| Stage2 | **Stage2 batch_size** | `1` |

Values must be positive integers.

### 4.1 Parameters Affect Only the Current Run

The scope of these four inputs is `this_run_only`:

- they affect only the current training run;
- they do not write back to `configs/training_default.yaml`;
- reopening the page loads defaults from the official default configuration again.

For a first run, keep:

```text
Stage1: epochs=10, batch_size=1
Stage2: epochs=20, batch_size=1
```

First confirm that training, checkpoint management, and Voice Profile handoff all complete successfully.

> [!NOTE]
> Keeping the defaults is a practical first-run recommendation. It does not mean these values are optimal for every dataset.

### 4.2 batch_size and VRAM Guidance

Based on current project development and testing experience, training with the default settings and `batch_size=1` typically uses about **2–3 GB of GPU VRAM**. Actual usage varies with the Stage, sample length, GPU / CUDA environment, and other programs using the GPU. The values below are conservative starting recommendations, not hard limits.

| GPU VRAM | Recommended Starting `batch_size` | Range to Try | Guidance |
| ---: | ---: | ---: | --- |
| 4 GB | `1` | `1` | Keep the default and prioritize stability |
| 6 GB | `1` | `1–2` | Try `2` only when sufficient VRAM remains |
| 8 GB | `2` | `1–2` | Start at `2`; return to `1` if OOM occurs |
| 12 GB | `2` | `2–4` | Increase gradually while watching peak VRAM |
| 16 GB | `4` | `2–6` | Start at `4` and increase only after confirming stability |
| 24 GB or more | `4` | `4–8+` | Increase further based on actual speed and peak VRAM |

> [!NOTE]
> VRAM usage is not guaranteed to scale linearly with `batch_size`, and it is not recommended to keep VRAM near 100% for long periods. As a conservative practice, leave at least about **1–2 GB** free; leave more when other GPU workloads are active.

If you encounter `CUDA out of memory`, training-launch failure, or sustained near-capacity VRAM usage, reduce `batch_size` first instead of modifying other release-managed training parameters.

### 4.3 User Settings and Release-Managed Boundary

Normal users can edit:

- Stage1 / Stage2 epochs;
- Stage1 / Stage2 batch_size;
- whether to open the separate training log terminal.

Users can also manage:

- the DataFactory work directory;
- whether DataFactory status is refreshed during scanning;
- Stage1 / Stage2 Active Best;
- historical checkpoints;
- whether Stage1 / Stage2 is forced to use Zero-shot;
- whether an existing Profile can be overwritten.

Training also contains internal parameters managed by the release. Normal users do not need to edit them for standard training.

---

## 5. Stage1 / Stage2 Training

Stage1 and Stage2 follow the same basic training-run lifecycle:

```text
Entry Check Passes
  ↓
Start Training
  ↓
Create New run_id / Independent Run Directory
  ↓
pending → running
  ↓
Training Ends
  ↓
succeeded / failed / stopped / stop_failed
  ↓
Discover Trainer Best on Success
  ↓
Update Active Best Depending on Selection Mode
```

Each launch creates a new training run instead of overwriting the previous one. If the same Stage already has an active `pending / running / stopping` run, a new launch for that Stage is rejected. Stage1 and Stage2 active-run protection are managed independently; normal users are still advised to train sequentially to avoid GPU contention.

### 5.1 Stage1 Training

Stage1-only and Full Few-shot routes require Stage1.

Confirm:

```text
Stage1 Training：✅ 可以启动
```

then click:

```text
启动 Stage1 Training
```

("Start Stage1 Training")

Each launch creates:

```text
{work_dir}/12_training/stage1/runs/<run_id>/
```

A typical checkpoints directory contains:

```text
best_model.ckpt
last_model.ckpt
history.json
summary.json
stage1_fewshot_train_report.json
```

`best_model.ckpt` is the Trainer Best for this run, while `last_model.ckpt` is the model from the final epoch. The normal user path does not require saving a checkpoint for every epoch.

When validation data is available, the main Stage1 Trainer Best metric is:

```text
Val Loss / Token
```

with the underlying field:

```text
val_loss_per_token
```

If the low-level training execution has no validation loader, Stage1 falls back to train loss/token for best-model selection. In that case, `Val Loss / Token` metadata in **Checkpoint / Active Best** may show `N/A`. The normal DataFactory → Training path should still retain valid train / validation data.

### 5.2 Stage2 Training

Stage2-only and Full Few-shot routes require Stage2.

Confirm:

```text
Stage2 Training：✅ 可以启动
```

then click:

```text
启动 Stage2 Training
```

("Start Stage2 Training")

Each launch creates:

```text
{work_dir}/12_training/stage2/runs/<run_id>/
```

Stage2 Few-shot automatically starts from the default Stage2 checkpoint selected by the official release configuration. The user edition:

- does not provide a base-checkpoint picker;
- does not require you to enter a base-checkpoint path manually;
- should not be worked around by editing internal configuration for normal use.

The main Stage2 Trainer Best validation metric shown in the UI is:

```text
Val Loss
```

### 5.3 Training Success and Active Best Handoff Are Separate

When a Stage1 / Stage2 run itself ends normally, it should report:

```text
status = succeeded
```

The training worker then attempts to discover this run's `best_model.*` and register Trainer Best / Active Best.

> [!IMPORTANT]
> `succeeded` means the training process exited successfully, but it does not guarantee that Active Best was updated. If the Trainer Best file is missing or automatic registration fails, the run may remain `succeeded` while recording `auto_best_error`. After training, still check **Checkpoint / Active Best**.

If the current Stage is in Automatic Best mode and Trainer Best registration succeeds:

```text
Training Succeeds
  ↓
Trainer Best
  ↓
Active Best Updates Automatically
```

If the current Stage has been manually locked, a new Trainer Best remains in its own run but does not automatically replace the current Active Best.

> [!NOTE]
> Stage1 and Stage2 active-run protection are independent. Normal users should still train them sequentially rather than run both on the same GPU at once, which reduces VRAM contention and simplifies resource and log interpretation.

---

## 6. Monitoring, Stop, and Training-Run State

### 6.1 Training Live Monitor

**Training Live Monitor** is the main status area during normal training.

While the state is:

- `pending`
- `running`
- `stopping`

the live monitor refreshes at approximately 5-second intervals. High-frequency polling pauses when no training run is active. You can also click **`立即刷新监控`** ("Refresh Monitor Now") at any time to force an immediate refresh.

**`最近一次训练`** ("Latest Training") shows:

- Stage;
- status;
- `run_id`;
- epochs / batch_size;
- start time;
- completion or update time.

It is the chronologically latest training run; it does not mean the current Active Best came from that run.

**Epoch / Step Details** show the current epoch, step, and whether training is still advancing.

While training is active, the page also shows **GPU / VRAM**. Automatic GPU monitoring pauses when no training run is active.

### 6.2 Optional Separate Training Log Terminal

Option:

```text
打开独立训练日志终端（可选）
```

("Open Separate Training Log Terminal — Optional")

It is disabled by default.

When enabled, an additional read-only log viewer opens. It:

- only reads the current training run's logs;
- does not own the training process;
- can be closed without stopping training;
- is not required for training to complete.

To actually stop training, use:

```text
停止 Stage1 Training
停止 Stage2 Training
```

### 6.3 Stop and Training States

When stopping a Stage, the system writes a stop request, attempts to stop the training subprocess / worker, updates state, and preserves the run directory, logs, metadata, and checkpoints that have already been produced.

Stop does not delete the training run.

Common states:

| State | Meaning |
| --- | --- |
| `pending` | training worker is starting |
| `running` | training process is running |
| `stopping` | a stop has been requested and is being processed |
| `succeeded` | training process exited successfully; Trainer Best / Active Best handoff still needs confirmation |
| `failed` | training or worker failed |
| `stopped` | stopped by user request |
| `stop_failed` | the stop operation itself did not complete normally |

A run already in a terminal state does not need another Stop action.

`stopped` / `failed` do not mean `succeeded`. Automatic Trainer Best → Active Best registration occurs only for successfully completed runs. Historical checkpoints left by failed or stopped runs may still appear in the list, but their presence does not mean the run passed the success condition.

### 6.4 v1.0.0 Has No User-Level Training Resume

The v1.0.0 Training UI does not support resuming an interrupted run in place from optimizer state.

This differs from DataFactory Stage2 artifact-based Resume.

If training is interrupted and you need to train again, start a new training run.

> [!IMPORTANT]
> "Selecting a historical checkpoint / Active Best" is model selection. "Training Resume" is training-state continuation. Do not confuse the two.

### 6.5 Training Failure Summary and Logs

If the latest training run fails, the UI attempts to extract useful error information from standard error (`stderr`) / standard output (`stdout`), including Python traceback, `CUDA out of memory`, RuntimeError / ValueError, and similar key errors.

Check:

```text
训练失败摘要
```

("Training Failure Summary")

first.

For deeper troubleshooting, inspect the run's:

```text
training_stderr.log
training_stdout.log
training_command.json
```

If you need to confirm the epochs / batch_size actually used by this run, inspect `training_command.json`.

---

## 7. Run Directories, Trainer Best, and Active Best

Training root:

```text
{work_dir}/12_training/
```

Stage1 and Stage2 are stored separately:

```text
12_training/
├── stage1/
└── stage2/
```

### 7.1 Every Training Run Has Its Own Directory

Typical structure:

```text
12_training/<stage>/
└── runs/
    └── <run_id>/
        ├── checkpoints/
        │   ├── best_model.*
        │   ├── last_model.*
        │   ├── history.json
        │   ├── summary.json
        │   └── ...
        ├── logs/
        │   ├── training_stdout.log
        │   └── training_stderr.log
        ├── training_command.json
        └── training_run_state.json
```

Different runs do not overwrite one another.

### 7.2 Stable State at the Stage Root

```text
12_training/<stage>/
```

also contains:

| File | User-Level Meaning |
| --- | --- |
| `training_run_state.json` | Stage-level state copy for the current / latest Stage run |
| `latest_run.json` | most recently launched run |
| `best_model.pt` | stable working copy of the current Active Best |
| `active_best.json` | source of the current Active Best |
| `best_checkpoint.json` | Active Best metadata |

`training_command.json` records the official defaults, this run's UI overrides, effective parameters, training command, log paths, log-terminal policy, Stage2 base checkpoint, and other runtime context.

### 7.3 Trainer Best vs. Active Best

**Trainer Best**:

> the best checkpoint selected within one training run according to that run's validation metric.

It belongs to:

```text
runs/<run_id>/
```

Different runs can each have their own Trainer Best.

**Active Best**:

> the Few-shot checkpoint currently used by this Stage when generating a Voice Profile.

The system copies the current Active Best to:

```text
12_training/<stage>/best_model.pt
```

and records its source run, source checkpoint, epoch, validation metric, and selection mode.

### 7.4 Automatic Best and Manual Lock

When no manual lock is active, the Stage normally operates in Automatic Best mode:

```text
Run Succeeds
  ↓
Trainer Best
  ↓
Copy to Stage-level best_model.pt
  ↓
Update Active Best
```

To keep using a specific historical checkpoint, select it from the list and click:

```text
手动锁定 Stage1 Active Best
```

or:

```text
手动锁定 Stage2 Active Best
```

After manual lock:

- the selected checkpoint becomes Active Best;
- selection mode becomes Manual;
- later training runs can continue normally and produce new Trainer Best checkpoints;
- new Trainer Best does not automatically replace the locked Active Best.

To restore automatic selection, click:

```text
恢复 Stage1 Automatic Best
恢复 Stage2 Automatic Best
```

Restore Automatic scans the historical trainer-produced `best_model*` candidates for that Stage, selects the latest one, copies it as the new Active Best, and restores Automatic mode. If no trainer-produced `best_model*` exists, restore fails.

---

## 8. Historical Checkpoints and Multiple Runs

### 8.1 Refresh and Select Historical Checkpoints

The **Checkpoint / Active Best** area provides:

```text
刷新 checkpoint 列表
```

("Refresh Checkpoint List")

The historical list shows the run, checkpoint name, epoch, validation metric, file size, and protection label.

Each new training execution receives a new `run_id` and does not overwrite old runs, so multiple training results can be retained and compared.

The effective epochs / batch_size for each run are preserved in that run's `training_command.json`.

A common workflow is:

```text
Current Model You Like
  ↓
Manually Lock Active Best
  ↓
Create More Training Runs
  ↓
Compare Results
  ↓
Switch Active Best When Satisfied
```

### 8.2 Permanent Deletion and Protection Rules

Deletion handles one historical checkpoint at a time. You must first enable the corresponding permanent-delete confirmation.

v1.0.0 blocks deletion of:

1. the stable Stage-level Active Best working copy;
2. the source checkpoint of the current Active Best;
3. a checkpoint belonging to a currently active training run;
4. files outside the manageable historical-run scope.

If a checkpoint is the current Active Best source, switch to another Active Best and refresh the list first.

A historical Trainer Best can be deleted after it is no longer the Active Best source and no longer belongs to an active run, but doing so is higher risk because that run loses its best model file.

> [!WARNING]
> Checkpoint deletion permanently removes the model file. It is not a hide action, a list cleanup, or a cache cleanup. The deleted model cannot be reconstructed automatically from history / summary metadata.

Deleting a checkpoint does not delete the rest of the training-run record. History, summary, run-state data, stdout / stderr logs, and other metadata remain.

Deletion is also recorded in:

```text
12_training/checkpoint_deletion_history.jsonl
```

including deletion time, Stage, run, checkpoint name, size, and relevant validation information.

---

## 9. Generate Voice Profile v2

After completing Training for the selected route and confirming Active Best, use the **`生成 Voice Profile v2`** ("Generate Voice Profile v2") area to create the Voice Profile used by Inference.

Voice Profile is the formal Training → Inference handoff.

### 9.1 profile_name and Output Directory

Field:

```text
profile_name
```

Use a recognizable name when possible.

If left empty:

1. the system first uses the current **`说话人 / 角色名`** ("Speaker / Character Name");
2. if the speaker is also empty while a custom work directory is used, it falls back to `default_user`.

The final directory name is sanitized: letters, digits, underscores, and hyphens are preserved; other characters become underscores.

Output location:

```text
user_profiles/<profile_name>/
```

The directory stores the Profile configuration, Profile manifest, and copies of the Stage1 / Stage2 Few-shot checkpoints required by that Profile.

### 9.2 Default Model Sources and Force Zero-shot

Normally:

- Stage1 has Active Best → use Stage1 Few-shot;
- Stage1 has no Active Best → fall back to the Stage1 Zero-shot default;
- Stage2 has Active Best → use Stage2 Few-shot;
- Stage2 has no Active Best → fall back to the Stage2 Zero-shot default.

The UI also provides:

```text
Stage1 强制使用 zero-shot
Stage2 强制使用 zero-shot
```

to explicitly create Partial Few-shot Profiles.

| Profile Type | Stage1 | Stage2 | Force Zero-shot Setting |
| --- | --- | --- | --- |
| `base_zeroshot` | default model | default model | both Stages use defaults |
| `few_shot_stage1_only` | Active Best | default model | Stage2 enabled |
| `few_shot_stage2_only` | default model | Active Best | Stage1 enabled |
| `few_shot_dual` | Active Best | Active Best | both disabled |

Normal Zero-shot users usually do not need to enter Training just to use the bundled defaults.

### 9.3 Overwrite an Existing Profile

Option:

```text
覆盖已有 profile
```

("Overwrite Existing Profile")

is enabled by default.

When enabled, the system can write into an existing Profile directory, rewrite the Profile configuration / manifest, and copy the Few-shot checkpoints required by this generation.

Overwriting does not automatically create a versioned backup of the old Profile.

> [!WARNING]
> If you want to keep an older Profile as a rollback option, use a new `profile_name` instead of directly overwriting the old Profile.

### 9.4 Profile Success Condition

After success, the page shows:

- profile name;
- profile type;
- Stage1 source;
- Stage2 source;
- whether the Profile is ready for inference.

In a normal release environment, you should see:

```text
可用于推理：True
```

("Ready for Inference: True")

The final completion condition can be summarized as:

```text
Target Stage Training succeeded
  ↓
Target Active Best is Correct
  ↓
Voice Profile Generated Successfully
  ↓
Ready for Inference: True
```

> [!IMPORTANT]
> "The training button ran" is not the final Training completion condition. Before moving to Inference, confirm that the target Stage succeeded, Active Best is correct, and the Voice Profile was generated successfully and is ready for inference.

---

## 10. Common Boundaries and Safe Operations

| Common Misunderstanding | Correct Interpretation |
| --- | --- |
| Latest training run = current Active Best source | Not necessarily, especially after manual lock |
| Trainer Best = Active Best | No. Trainer Best belongs to one run; Active Best is the Stage's current selection |
| `succeeded` = Active Best definitely updated | Not necessarily. Check Checkpoint / Active Best and any `auto_best_error` |
| Stop deletes the training run | No. Existing directories, logs, metadata, and checkpoints remain |
| Closing the separate log terminal stops training | No. Use the Stage1 / Stage2 Stop buttons to actually stop training |
| v1.0.0 supports Training Resume | No. Retraining creates a new run |
| Checkpoint deletion only hides the checkpoint | No. Deletion permanently removes the model file |
| Active Best source can be deleted directly | No. Switch Active Best first |
| Stage1-only / Stage2-only must wait for the other Stage | No. Both are officially supported Partial Few-shot routes |
| Training directory can be used directly as the Inference entry | Not recommended. Voice Profile is the formal Training → Inference handoff |

---

## 11. Failure and Recovery Guidance

### 11.1 Training Fails to Start

If Stage1 / Stage2 fails immediately after launch, first check:

1. whether the correct DataFactory work directory was scanned;
2. whether the target Stage reports **Can Start**;
3. whether epochs / batch_size are positive integers;
4. whether the same Stage already has an active training run;
5. whether the Stage2 base checkpoint is complete.

### 11.2 Training Fails While Running

If a training run enters `failed` after launch:

1. check **`训练失败摘要`** ("Training Failure Summary");
2. inspect `training_stderr.log`;
3. then inspect `training_stdout.log`;
4. inspect `training_command.json` if you need to confirm the effective parameters.

If the error contains CUDA unavailable, `CUDA out of memory`, or similar GPU failures, reduce `batch_size` first. GPU / CUDA support belongs to the [Compatibility Guide](./compatibility.md); do not work around the error by editing release-managed internal model configuration.

### 11.3 Active Best Does Not Change

If training succeeds but Active Best does not change, first check the current selection mode.

If the Stage is manually locked, this is expected. To resume automatic selection, use:

```text
恢复 Stage1 Automatic Best
恢复 Stage2 Automatic Best
```

### 11.4 Checkpoint Cannot Be Deleted

Common reasons:

- the checkpoint is the current Active Best source;
- the checkpoint belongs to an active training run;
- the list is stale and needs **`刷新 checkpoint 列表`**;
- the permanent-delete confirmation is not enabled;
- the target file is outside the supported historical-checkpoint scope.

---

## 12. Next Steps and Related Documentation

After Training, use the generated Voice Profile in Inference:

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
- [DataFactory Guide](./data-factory.md)
- [Inference Guide](./inference.md)
- [Compatibility Guide](./compatibility.md)
