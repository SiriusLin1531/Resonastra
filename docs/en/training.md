# Resonastra Training Guide

[Project README](https://github.com/SiriusLin1531/Resonastra/blob/main/README.md) | [Quick Start](./quick-start.md) | [DataFactory](./data-factory.md) | **English** | [中文简体](../cn/training.md)

Applies to: **Resonastra v1.0.0**

> [!NOTE]
> This guide documents the official **Resonastra v1.0.0** Training behavior. If a later release changes the Training UI, training parameters, checkpoint management, or Voice Profile contract, refer to the documentation for that release.

---

## 1. What This Guide Covers

Resonastra Training turns the Stage1 / Stage2 training data prepared by DataFactory into Few-shot model artifacts that can be used by Voice Profiles.

If you only use the bundled Zero-shot models, you do not need Training. This guide is required when preparing Stage1-only, Stage2-only, or Full Few-shot models.

After completing the sections for your selected route, you should be able to:

- start Training from the correct DataFactory work directory;
- determine whether the Stage1 or Stage2 training entry is ready;
- understand which training parameters the v1.0.0 user edition lets you change;
- complete Stage1 or Stage2 Training independently;
- determine whether a training run actually succeeded;
- use Training Live Monitor to view progress, GPU / VRAM usage, and failure summaries;
- understand the separate directory, logs, and checkpoints for each training run;
- distinguish **Trainer Best**, historical checkpoints, and **Active Best**;
- manually lock Active Best or restore Automatic Best;
- safely delete a single historical checkpoint after understanding the risk;
- generate the correct Voice Profile for Stage1-only, Stage2-only, or Full Few-shot;
- hand the generated Voice Profile to Inference.

This guide does not cover:

- raw audio, ASR, manual proofreading, or training-data construction;
- general GPU / CUDA compatibility policy;
- Stage1 / Stage2 model architecture or training-algorithm theory;
- systematic research-oriented hyperparameter tuning;
- Inference generation parameters or quality evaluation;
- comprehensive symptom-oriented troubleshooting.

Those topics belong to the DataFactory, Compatibility, Inference, Troubleshooting, or future advanced documentation.

---

## 2. Where Training Fits in the Resonastra Workflow

Training sits between DataFactory and Voice Profile / Inference:

```text
DataFactory Training Data
  ↓
Training Input Validation
  ├─→ Stage1 Training
  │      ↓
  │   Trainer Best
  │      ↓
  │   Stage1 Active Best
  │
  └─→ Stage2 Training
         ↓
      Trainer Best
         ↓
      Stage2 Active Best

Stage1 / Stage2 Active Best
  ↓
Voice Profile
  ↓
Inference
```

### 2.1 Relationship to the Four Model Routes

| Route | Training Required | Model Sources in Voice Profile |
| --- | --- | --- |
| Zero-shot | None | Stage1 default + Stage2 default |
| Stage1-only Few-shot | Stage1 | Stage1 Active Best + Stage2 default |
| Stage2-only Few-shot | Stage2 | Stage1 default + Stage2 Active Best |
| Full Few-shot | Stage1 + Stage2 | Stage1 Active Best + Stage2 Active Best |

Stage1 and Stage2 have independent training-entry checks. Partial Few-shot routes do not require both stages to be complete.

### 2.2 Three Core Training Concepts

The workflow is easiest to understand in three layers:

1. **Data readiness** — whether the current Stage has valid training input;
2. **Training run** — one independent training execution and the logs / checkpoints it produces;
3. **Active Best** — the Few-shot checkpoint currently selected for Voice Profile generation.

> [!IMPORTANT]
> "Latest training run," "Trainer Best from this run," and "current Active Best" are not the same thing. After Active Best has been manually locked, a newer successful run can produce a new Trainer Best without automatically replacing the locked Active Best.

---

## 3. Before You Start

Before opening Training, you should have:

1. fully extracted Resonastra v1.0.0;
2. run `launch_check_env.bat` and confirmed the basic environment check passes;
3. completed the training data required by your route in DataFactory;
4. kept track of the **`说话人 / 角色名`** ("Speaker / Character Name") and work directory used by DataFactory.

If you have not prepared training data yet, read the [DataFactory Guide](./data-factory.md) first.

### 3.1 Continue Using the Same DataFactory Work Directory

If DataFactory used the default directory rule:

```text
user_data/{speaker_name}_factory
```

enter the same **`说话人 / 角色名`** ("Speaker / Character Name") in Training and leave **`DataFactory 工作目录`** ("DataFactory Work Directory") empty.

If DataFactory used a custom work directory, enter that same directory in Training.

> [!IMPORTANT]
> Training does not copy DataFactory data into a separate user project. The DataFactory `work_dir` remains the project identity for this Few-shot dataset, training history, and checkpoints.

### 3.2 Data Requirements by Route

| Route | Required Before Training |
| --- | --- |
| Stage1-only Few-shot | Stage1 data ready |
| Stage2-only Few-shot | Stage2 data ready |
| Full Few-shot | Stage1 + Stage2 data ready |
| Zero-shot | Training not required |

Therefore, "all training data" does not need to be complete before every route can start.

For example:

- Stage1-only only needs Stage1 Training to report that it can start;
- Stage2-only only needs Stage2 Training to report that it can start.

---

## 4. Start Training

From the Resonastra root directory, run:

```text
launch_training_ui.bat
```

The default port is:

```text
7863
```

If the port is already in use, the launcher automatically selects another available port. Use the address shown in the launcher terminal.

### 4.1 Do Not Close the Main WebUI Launcher Terminal

Running `launch_training_ui.bat` opens a main terminal that keeps the Training WebUI process alive.

Do not close this terminal while using Training.

It is different from the optional:

```text
打开独立训练日志终端（可选）
```

("Open Separate Training Log Terminal — Optional")

described later.

### 4.2 Main Page Areas

The Resonastra · Training page is organized into:

- **`输入`** ("Input")
- **`训练操作`** ("Training Actions")
- **Checkpoint / Active Best**
- **`生成 Voice Profile v2`** ("Generate Voice Profile v2")
- **`状态`** ("Status")
- **Training Live Monitor**
- **`诊断信息`** ("Diagnostics")

For a first successful training workflow, the main sequence is:

```text
Input
  ↓
Scan Training Data
  ↓
Stage1 / Stage2 Training
  ↓
Training Live Monitor
  ↓
Checkpoint / Active Best
  ↓
Generate Voice Profile v2
```

---

## 5. Scan DataFactory Training Data

### 5.1 Fill In the Inputs

The left-side **`输入`** ("Input") section contains:

- **`说话人 / 角色名`** ("Speaker / Character Name")
- **`DataFactory 工作目录`** ("DataFactory Work Directory")
- **`扫描时刷新 DataFactory 状态`** ("Refresh DataFactory Status During Scan")

If you use the default work-directory rule:

1. enter the same **`说话人 / 角色名`** used in DataFactory;
2. leave **`DataFactory 工作目录`** empty;
3. keep **`扫描时刷新 DataFactory 状态`** enabled.

If you use a custom work directory, enter it directly.

The work directory and speaker name cannot both be empty.

### 5.2 Scan Training Data

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

### 5.3 Focus on the Training Entry Status

After scanning, the page separately reports:

```text
Stage1 Training：✅ 可以启动
Stage2 Training：✅ 可以启动
```

("Can Start")

or:

```text
Stage1 Training：❌ 暂不可启动
Stage2 Training：❌ 暂不可启动
```

("Cannot Start Yet")

Only the Stage required by your selected route needs to pass its entry check.

### 5.4 What Stage1 Checks

Before Stage1 Training can start, Training verifies that DataFactory has produced and retained the required Stage1 artifacts, including:

- Stage1 readiness;
- training manifest;
- validation manifest;
- frontend cache;
- semantic cache;
- Stage1 output root.

If these inputs are incomplete, Training rejects the Stage1 launch rather than passing missing input into the trainer.

### 5.5 What Stage2 Checks

Before Stage2 Training can start, Training verifies:

- Stage2 readiness;
- completed Stage2 `.pt` data;
- completed train / val split;
- completed filter / quarantine contract;
- train / val directories containing trainable `.pt` files;
- the Stage2 base checkpoint required by the official release package.

If DataFactory has not completed the Stage2 split / filter flow, Training reports the Stage2 entry as not ready.

### 5.6 Diagnostic JSON Is Not Required for Normal Use

The **`诊断信息`** ("Diagnostics") area provides Training Interface, Training Defaults, Live Monitor, Checkpoint Scan, and other JSON views.

These are mainly for troubleshooting.

For a normal training workflow, focus on:

- data-preparation status;
- whether Stage1 / Stage2 Training can start;
- the training-run state in Training Live Monitor;
- Checkpoint / Active Best status.

---

## 6. User-Editable Training Parameters

Under **`本次训练参数（仅当前 run）`** ("Training Parameters for This Run Only"), the v1.0.0 user edition exposes only:

### Stage1

- **Stage1 epochs**
- **Stage1 batch_size**

### Stage2

- **Stage2 epochs**
- **Stage2 batch_size**

The current defaults are:

| Stage | epochs | batch_size |
| --- | ---: | ---: |
| Stage1 | `10` | `1` |
| Stage2 | `20` | `1` |

### 6.1 Changes Apply Only to the Current Run

All four inputs affect only the current training run. Internally their scope is `this_run_only`.

Changes:

- affect only this training run;
- do not write back to `configs/training_default.yaml`;
- do not change the defaults loaded the next time the page is opened.

Values must be positive integers.

### 6.2 Keep the Defaults for Your First Run

For a first run, use:

- Stage1: `epochs=10, batch_size=1`;
- Stage2: `epochs=20, batch_size=1`;
- first confirm that training, checkpoint management, and Voice Profile handoff all complete successfully.

> [!NOTE]
> Keeping the defaults is a practical first-run recommendation. It does not mean these values are optimal for every dataset. Research-oriented tuning is outside the scope of this basic guide.

### 6.3 batch_size and VRAM Guidance

Based on current development and testing experience, training with the default settings and `batch_size=1` typically uses about **2–3 GB of GPU VRAM**. Actual usage still varies with the Stage, sample length, GPU / CUDA environment, and other programs using the GPU. The values below are therefore **conservative starting recommendations**, not hard limits.

| GPU VRAM | Recommended Starting `batch_size` | Range to Try | Guidance |
| ---: | ---: | ---: | --- |
| 4 GB | `1` | `1` | Keep the default and prioritize stability |
| 6 GB | `1` | `1–2` | Try `2` only when sufficient VRAM remains |
| 8 GB | `2` | `1–2` | Start at `2`; return to `1` if OOM occurs |
| 12 GB | `2` | `2–4` | Increase gradually while watching peak VRAM |
| 16 GB | `4` | `2–6` | Start at `4` and increase only after confirming stability |
| 24 GB or more | `4` | `4–8+` | Increase further based on actual speed and peak VRAM |

> [!NOTE]
> VRAM usage is not guaranteed to scale linearly with `batch_size`. Even on a large GPU, avoid keeping VRAM usage near 100%. Leave headroom for desktop display, CUDA context, and other processes. As a conservative practice, leave at least about **1–2 GB** free, and leave more when other GPU workloads are active.

If increasing `batch_size` causes `CUDA out of memory`, training-launch failure, or sustained near-capacity VRAM usage, reduce `batch_size` first rather than editing internal training parameters.

---

## 7. Optional Separate Training Log Terminal

Below the training parameters is:

```text
打开独立训练日志终端（可选）
```

("Open Separate Training Log Terminal — Optional")

It is disabled by default.

When enabled, starting Stage1 or Stage2 opens an additional log-viewer window.

This window:

- only reads the current run's logs;
- does not own the training process;
- can be closed without stopping training;
- is not required to complete training.

For normal progress monitoring, **Training Live Monitor** is usually sufficient.

> [!IMPORTANT]
> Closing the separate log terminal does not stop training. To actually stop training, use **`停止 Stage1 Training`** ("Stop Stage1 Training") or **`停止 Stage2 Training`** ("Stop Stage2 Training").

---

## 8. Stage1 Training

Stage1-only Few-shot and Full Few-shot require Stage1.

### 8.1 Start Stage1

Confirm:

```text
Stage1 Training：✅ 可以启动
```

then click:

```text
启动 Stage1 Training
```

("Start Stage1 Training")

After launch, the page reports the current run's:

- `run_id`;
- effective epochs;
- effective batch_size;
- separate-log-terminal setting.

### 8.2 Each Launch Creates a New Training Run

Stage1 does not overwrite the previous run directory.

Each launch creates:

```text
{work_dir}/12_training/stage1/runs/<run_id>/
```

Older runs remain available.

If a Stage1 run is already active, the system rejects another Stage1 launch until the current run ends or is stopped.

### 8.3 Main Stage1 Run Outputs

A typical Stage1 run's `checkpoints/` directory contains:

```text
best_model.ckpt
last_model.ckpt
history.json
summary.json
stage1_fewshot_train_report.json
```

where:

- `best_model.ckpt` is the Trainer Best selected within this run;
- `last_model.ckpt` is the model from the last epoch;
- `history.json` records per-epoch training / validation history;
- `summary.json` records the run summary;
- `stage1_fewshot_train_report.json` records the training-task report.

The normal user path does not require saving a checkpoint for every epoch.

### 8.4 Stage1 Trainer Best Metric

When validation data is available, the primary Stage1 best-model metric is:

```text
Val Loss / Token
```

corresponding to:

```text
val_loss_per_token
```

Within a run, an improved validation metric updates that run's Trainer Best.

If a low-level training execution has no validation loader, the Stage1 trainer falls back to train loss/token for best-model selection. In that case, `Val Loss / Token` metadata in **Checkpoint / Active Best** may show `N/A`. The normal DataFactory → Training route should still retain valid train / val data.

### 8.5 Stage1 Success Condition

Do not treat "training started" as success.

A training run that finishes normally should report:

```text
status = succeeded
```

The training worker then attempts to discover this run's `best_model.*` and register Trainer Best / Active Best.

> [!IMPORTANT]
> `succeeded` means the training process exited successfully, but it does not guarantee that Active Best was updated. If the Trainer Best file is missing or automatic registration fails, the run can remain `succeeded` while recording `auto_best_error` in the run state. After Stage1 finishes, also check **Checkpoint / Active Best**.

If Stage1 is in Automatic Best mode and Trainer Best registration succeeds:

```text
Stage1 Training Run Succeeds
  ↓
Trainer Best
  ↓
Stage1 Active Best Updates Automatically
```

If Stage1 Active Best is manually locked, the new Trainer Best remains in its own run directory but does not automatically replace the locked Active Best.

---

## 9. Stage2 Training

Stage2-only Few-shot and Full Few-shot require Stage2.

### 9.1 Start Stage2

Confirm:

```text
Stage2 Training：✅ 可以启动
```

then click:

```text
启动 Stage2 Training
```

("Start Stage2 Training")

### 9.2 The Stage2 Base Checkpoint Does Not Need Manual Selection

Stage2 Few-shot automatically initializes from the default Stage2 checkpoint selected by the official release configuration.

The user edition:

- does not provide a base-checkpoint picker;
- does not require you to enter a base-checkpoint path;
- should not be worked around by editing internal configuration for normal use.

### 9.3 Stage2 Also Uses Independent Training Runs

Each launch creates:

```text
{work_dir}/12_training/stage2/runs/<run_id>/
```

Previous Stage2 runs are not overwritten.

If a Stage2 run is already active, another Stage2 launch is rejected.

### 9.4 Stage2 Trainer Best

The primary Stage2 validation metric shown in the UI is:

```text
Val Loss
```

A successful run produces a Trainer Best in its own run directory.

### 9.5 Stage2 Success Condition

A training run that finishes normally should report:

```text
status = succeeded
```

The training worker then attempts to discover this run's `best_model.*` and register Trainer Best / Active Best.

Therefore, check "run success" and "Active Best handoff success" separately. After training, confirm in **Checkpoint / Active Best** that the intended model is in the expected state.

If Stage2 is in Automatic Best mode and Trainer Best registration succeeds:

```text
Stage2 Training Run Succeeds
  ↓
Trainer Best
  ↓
Stage2 Active Best Updates Automatically
```

If Stage2 is manually locked, the new Trainer Best does not replace the current Active Best.

> [!NOTE]
> Stage1 and Stage2 active-run protection are managed independently. For normal users, it is still recommended to train them sequentially rather than run both on the same GPU at once. This reduces VRAM contention and makes each run's resource usage and logs easier to interpret.

---

## 10. Use Training Live Monitor

The **Training Live Monitor** on the right side of the Training page is the primary status area during normal training.

### 10.1 Automatic Refresh

While training is in:

- `pending`
- `running`
- `stopping`

the live monitor refreshes at approximately 5-second intervals.

When no training run is active, high-frequency polling pauses.

### 10.2 Latest Training Run

**`最近一次训练`** ("Latest Training") shows:

- Stage;
- status;
- `run_id`;
- epochs / batch_size;
- start time;
- completion or update time.

This is the chronologically latest Stage1 / Stage2 run. It does not necessarily mean the current Active Best came from that run.

### 10.3 Epoch / Step Details

While training is active, use:

```text
Epoch / Step 详情
```

("Epoch / Step Details")

to view the current progress snapshot.

It helps answer:

- which epoch is currently running;
- current step progress;
- whether training is still advancing.

### 10.4 GPU / VRAM

While training is active, the UI queries:

```text
GPU / 显存
```

("GPU / VRAM")

GPU monitoring pauses automatically when no training run is active.

This guide only explains the monitor; GPU / CUDA support policy belongs to the Compatibility documentation.

### 10.5 Training Failure Summary

If the latest run fails, the UI attempts to extract useful error information from stderr / stdout, including:

- Python traceback;
- `CUDA out of memory`;
- `RuntimeError` / `ValueError` and similar exceptions;
- other key error lines.

Check:

```text
训练失败摘要
```

("Training Failure Summary")

first, then open the run's full stdout / stderr logs if needed.

### 10.6 Manual Refresh

Click:

```text
立即刷新监控
```

("Refresh Monitor Now")

to force an immediate refresh.

---

## 11. Stop, Interrupt, and Restart Training

Training provides two independent buttons:

```text
停止 Stage1 Training
停止 Stage2 Training
```

("Stop Stage1 Training" / "Stop Stage2 Training")

### 11.1 What Stop Does

Stopping a Stage causes the system to:

1. record a stop request;
2. attempt to stop the current training child / worker process;
3. update run state;
4. preserve the run directory, logs, metadata, and any checkpoints already produced.

Stop does not delete the training run.

### 11.2 Finished Runs Do Not Need Another Stop

If a run is already:

- `succeeded`
- `failed`
- `stopped`
- `stop_failed`

another Stop is effectively a no-op for an already-finished run.

### 11.3 stopped Is Not succeeded

A user-stopped run enters a stopped state, not a succeeded state.

Automatic Trainer Best → Active Best registration occurs only for runs that complete successfully.

If a `stopped` / `failed` run has already left historical checkpoints behind, those files may still appear in the checkpoint list. Manually selecting one later is an explicit model-selection action; it does not mean that run met the success condition.

### 11.4 v1.0.0 Has No User-Level Training Resume

This is different from DataFactory Stage2 Resume.

DataFactory Resume means:

> reuse completed data artifacts and skip safely reusable steps.

The v1.0.0 Training UI does not provide:

> in-place continuation of an interrupted run from optimizer state.

If training is interrupted and you want to train again, start a new run.

> [!IMPORTANT]
> Selecting a historical checkpoint / Active Best is model selection. Training Resume is continuation of training state. Do not confuse the two.

---

## 12. Training-Run Directories and Logs

The Training root is:

```text
{work_dir}/12_training/
```

Stage1 and Stage2 are stored separately:

```text
12_training/
├── stage1/
└── stage2/
```

### 12.1 Each Independent Training Run

Typical layout:

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

Runs do not overwrite each other.

### 12.2 Stable Stage-Level State

Under:

```text
12_training/<stage>/
```

you also see stable files that do not belong only to one historical run:

```text
training_run_state.json
latest_run.json
best_model.pt
active_best.json
best_checkpoint.json
```

Their roles differ:

- `training_run_state.json` — Stage-level copy of current / latest run state;
- `latest_run.json` — points to the most recently launched run;
- `best_model.pt` — stable working copy of the current Active Best;
- `active_best.json` — records which run / checkpoint the current Active Best came from;
- `best_checkpoint.json` — records Active Best-related metadata.

### 12.3 training_command.json

Each run's:

```text
training_command.json
```

records:

- official defaults;
- UI overrides for this run;
- effective parameters for this run;
- actual training command;
- stdout / stderr log paths;
- log-terminal policy;
- runtime context such as the Stage2 base checkpoint.

If you need to confirm which epochs / batch_size were actually used, check this file rather than relying only on memory of the UI values.

---

## 13. Understand Trainer Best and Active Best

This is the most important distinction in Training.

### 13.1 Trainer Best

**Trainer Best** is:

> the best checkpoint selected within one independent training run according to that run's validation metric.

It belongs under:

```text
runs/<run_id>/
```

Different runs can each have their own Trainer Best.

### 13.2 Active Best

**Active Best** is:

> the Few-shot checkpoint that the current Stage will actually use when generating a Voice Profile.

The system copies the current Active Best to the stable path:

```text
12_training/<stage>/best_model.pt
```

and records:

- source run;
- source checkpoint;
- checkpoint epoch;
- validation metric;
- selection mode.

### 13.3 Automatic Best

On a normal first training run, the Stage is usually in Automatic Best mode:

```text
Training Run Succeeds
  ↓
Trainer Best Produced
  ↓
Copied to Stage-Level best_model.pt
  ↓
Active Best Updated
```

As long as there is no manual lock, later successful runs can continue updating Active Best.

### 13.4 Manually Lock Active Best

If you decide that a historical checkpoint works better for your use case, select it from the historical list and click:

```text
手动锁定 Stage1 Active Best
```

("Manually Lock Stage1 Active Best")

or:

```text
手动锁定 Stage2 Active Best
```

("Manually Lock Stage2 Active Best")

After manual locking:

- the selected checkpoint becomes Active Best;
- selection mode becomes Manual;
- new training runs can still be launched;
- new runs can still produce new Trainer Best checkpoints;
- they do not automatically replace the manually selected Active Best.

> [!NOTE]
> The historical list may contain Trainer Best, Last checkpoint, or other historical checkpoints. Manual locking overrides the automatic choice, so use validation metrics and actual inference listening when making this decision.

### 13.5 Restore Automatic Best

To let the system follow Trainer Best again, click:

```text
恢复 Stage1 Automatic Best
```

("Restore Stage1 Automatic Best")

or:

```text
恢复 Stage2 Automatic Best
```

("Restore Stage2 Automatic Best")

Restore scans the current Stage's historical `best_model*` candidates, selects the newest scanned item, copies it as the new Active Best, and returns selection mode to Automatic. If the Stage has no trainer-produced `best_model*`, restore fails.

After restore succeeds, future successful training runs can update Active Best automatically again.

---

## 14. Historical Checkpoints and Permanent Deletion

The Checkpoint / Active Best area provides:

```text
刷新 checkpoint 列表
```

("Refresh Checkpoint List")

and:

- **`Stage1 历史 checkpoint`** ("Stage1 Historical Checkpoint")
- **`Stage2 历史 checkpoint`** ("Stage2 Historical Checkpoint")

The historical list shows the run, checkpoint name, epoch, validation metric, file size, and protection labels.

### 14.1 Delete One Historical Checkpoint at a Time

After selecting a checkpoint, expand:

```text
删除 Stage1 checkpoint
```

("Delete Stage1 Checkpoint")

or:

```text
删除 Stage2 checkpoint
```

("Delete Stage2 Checkpoint")

You must enable the corresponding permanent-delete confirmation before deletion can proceed.

### 14.2 Checkpoints That Cannot Be Deleted

v1.0.0 blocks deletion of:

1. the stable Active Best working copy in the Stage root;
2. the current Active Best source checkpoint;
3. a checkpoint belonging to the currently active training run;
4. files outside the manageable historical-run scope.

If a checkpoint is the current Active Best source and you truly want to delete it, first switch Active Best to another checkpoint, then refresh the list.

### 14.3 Historical Trainer Best Can Be Deleted, but It Is Higher Risk

A Trainer Best from a historical run is not permanently undeletable.

If it is no longer the Active Best source and does not belong to an active run, it can be deleted after confirmation.

The UI marks this as higher risk because:

- it is that run's best model file;
- the model file cannot be recovered after deletion;
- that run will lose its own Trainer Best file.

> [!WARNING]
> Deleting a historical checkpoint permanently deletes the model file. Do not treat this as "hide," "remove from list," or "clear cache."

### 14.4 Deleting a Checkpoint Does Not Delete the Full Run Record

After the model file is deleted, the system keeps:

- history;
- summary;
- run_state;
- stdout / stderr logs;
- other run metadata.

The training history remains auditable, but the deleted model file cannot be reconstructed from this metadata.

### 14.5 Deletion Audit Log

Deletion is recorded in:

```text
12_training/checkpoint_deletion_history.jsonl
```

including deletion time, Stage, run, checkpoint name, file size, and related validation information.

---

## 15. Multiple Training Runs and Model Re-Selection

### 15.1 New Training Does Not Overwrite Old Runs

Each launch gets a new `run_id`.

You can therefore keep:

```text
run A
run B
run C
...
```

and compare the results later.

### 15.2 epochs / batch_size Affect Only the Current Run

For example, you may first run:

```text
epochs = 10
batch_size = 1
```

and later change epochs to 15 for a new run.

The previous run's effective parameters remain recorded in its own `training_command.json`.

### 15.3 Keep Training While Temporarily Keeping a Manually Locked Active Best

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

This allows new experiments without automatically changing the model currently selected for Voice Profile generation.

### 15.4 This Guide Does Not Define One "Best epochs" Value

Training quality depends on dataset size, dataset quality, speaker characteristics, and the specific goal.

The basic Training Guide is designed to help you manage the **training run / checkpoint / Active Best** lifecycle correctly, not to provide research-level tuning conclusions.

---

## 16. Generate Voice Profile v2

After training the selected route and confirming Active Best, use:

```text
生成 Voice Profile v2
```

("Generate Voice Profile v2")

to create the Voice Profile used by Inference.

Voice Profile is the formal Training → Inference handoff.

### 16.1 profile_name

UI field:

```text
profile_name
```

Using an explicit, recognizable name is recommended.

If left empty:

1. the system first uses the current **`说话人 / 角色名`** ("Speaker / Character Name");
2. if that is also empty while a custom `work_dir` is used, it falls back to `default_user`.

The final profile directory name is also sanitized: letters, digits, underscores, and hyphens are preserved; other characters become underscores.

### 16.2 Current Active Best Is Used by Default

Normally:

- Stage1 has Active Best → Profile uses Stage1 Few-shot;
- Stage1 has no Active Best → falls back to the Stage1 Zero-shot default;
- Stage2 has Active Best → Profile uses Stage2 Few-shot;
- Stage2 has no Active Best → falls back to the Stage2 Zero-shot default.

### 16.3 Force Zero-shot

The UI provides:

```text
Stage1 强制使用 zero-shot
Stage2 强制使用 zero-shot
```

("Force Stage1 to Use Zero-shot" / "Force Stage2 to Use Zero-shot")

These controls are used to explicitly create Partial Few-shot profiles.

### 16.4 Stage1-only Few-shot

Set:

- **`Stage1 强制使用 zero-shot`:** Off
- **`Stage2 强制使用 zero-shot`:** On

Then click:

```text
生成 Voice Profile v2
```

Expected profile type:

```text
few_shot_stage1_only
```

Model sources:

```text
Stage1 → Few-shot Active Best
Stage2 → Zero-shot default
```

### 16.5 Stage2-only Few-shot

Set:

- **`Stage1 强制使用 zero-shot`:** On
- **`Stage2 强制使用 zero-shot`:** Off

Expected profile type:

```text
few_shot_stage2_only
```

Model sources:

```text
Stage1 → Zero-shot default
Stage2 → Few-shot Active Best
```

### 16.6 Full Few-shot

Set:

- **`Stage1 强制使用 zero-shot`:** Off
- **`Stage2 强制使用 zero-shot`:** Off

and confirm that both Stages have the intended Active Best.

Expected profile type:

```text
few_shot_dual
```

### 16.7 base_zeroshot

If both Stages use the default models, the profile type is:

```text
base_zeroshot
```

Normal Zero-shot users usually do not need to enter Training merely to use the defaults.

### 16.8 Overwrite an Existing Profile

Option:

```text
覆盖已有 profile
```

("Overwrite Existing Profile")

is enabled by default.

When enabled, the system can write into an existing profile directory, rewrite the current Profile configuration / manifest, and copy the Few-shot checkpoints actually needed by this generation.

Overwriting does not automatically create a versioned backup of the old Profile. The newly generated `profile_config.json` / `profile_manifest.json` define the current inference model sources.

> [!WARNING]
> If you want to keep an old Profile as a rollback option, use a new `profile_name` instead of directly overwriting it.

### 16.9 Profile Output Location

Generated Profiles are stored under:

```text
user_profiles/<profile_name>/
```

The directory contains:

- profile config;
- profile manifest;
- copies of the Stage1 / Stage2 Few-shot checkpoints required by that Profile.

After success, the page shows:

- profile name;
- profile type;
- Stage1 source;
- Stage2 source;
- whether it is ready for inference.

In a normal release environment, you should see:

```text
可用于推理：True
```

("Ready for Inference: True")

---

## 17. Complete Training by Route

### 17.1 Stage1-only Few-shot

Flow:

```text
Stage1 Data Ready
  ↓
Stage1 Training Succeeds
  ↓
Stage1 Active Best Ready
  ↓
Voice Profile:
Stage1 Few-shot + Stage2 Default
```

Checklist:

- [ ] Stage1 Training reports it can start;
- [ ] Stage1 run ends as `succeeded`;
- [ ] Stage1 Active Best exists;
- [ ] `Stage2 强制使用 zero-shot` is enabled;
- [ ] Profile type is `few_shot_stage1_only`;
- [ ] Profile reports that it is ready for inference.

### 17.2 Stage2-only Few-shot

Flow:

```text
Stage2 Data Ready
  ↓
Stage2 Training Succeeds
  ↓
Stage2 Active Best Ready
  ↓
Voice Profile:
Stage1 Default + Stage2 Few-shot
```

Checklist:

- [ ] Stage2 Training reports it can start;
- [ ] Stage2 run ends as `succeeded`;
- [ ] Stage2 Active Best exists;
- [ ] `Stage1 强制使用 zero-shot` is enabled;
- [ ] Profile type is `few_shot_stage2_only`;
- [ ] Profile reports that it is ready for inference.

### 17.3 Full Few-shot

Flow:

```text
Stage1 + Stage2 Data Ready
  ↓
Stage1 Training
  ↓
Stage2 Training
  ↓
Stage1 + Stage2 Active Best Ready
  ↓
few_shot_dual Voice Profile
```

Checklist:

- [ ] Stage1 Training succeeds;
- [ ] Stage2 Training succeeds;
- [ ] Stage1 Active Best is correct;
- [ ] Stage2 Active Best is correct;
- [ ] both **Force Zero-shot** options are disabled;
- [ ] Profile type is `few_shot_dual`;
- [ ] Profile reports that it is ready for inference.

> [!IMPORTANT]
> "The training button ran" is not the final completion condition. Before moving to Inference, confirm **target Stage succeeded → Active Best is correct → Voice Profile generated successfully**.

---

## 18. Training States, Failures, and Log Troubleshooting

### 18.1 Common Training-Run States

| State | Meaning |
| --- | --- |
| `pending` | training worker is starting |
| `running` | training process is running |
| `stopping` | a stop has been requested |
| `succeeded` | training process exited successfully; Trainer Best / Active Best handoff still needs confirmation |
| `failed` | training or worker failed |
| `stopped` | stopped by user request |
| `stop_failed` | the stop operation itself did not complete normally |

The target run state for normal training completion is:

```text
succeeded
```

### 18.2 Training Fails to Start

If Stage1 / Stage2 fails immediately after launch, first check:

1. whether the correct DataFactory work directory (`work_dir`) was scanned;
2. whether the target Stage reports that it can start;
3. whether epochs / batch_size are positive integers;
4. whether the same Stage already has an active run;
5. whether the required Stage2 base checkpoint exists.

### 18.3 Training Fails After Starting

If an active run enters `failed`:

1. check **`训练失败摘要`** ("Training Failure Summary");
2. open:
   ```text
   training_stderr.log
   ```
3. then inspect:
   ```text
   training_stdout.log
   ```
4. if you need to confirm the actual parameters used, inspect:
   ```text
   training_command.json
   ```

### 18.4 CUDA / OOM

If the failure summary contains CUDA unavailable, `CUDA out of memory`, or similar GPU errors:

- batch_size is one of the user-editable parameters;
- GPU / CUDA compatibility and supported hardware belong to Compatibility / Troubleshooting;
- do not try to work around the error by editing internal release-managed model settings.

### 18.5 Training Succeeds but Active Best Does Not Change

First check the current selection mode.

If it is manually locked, this is expected.

A new Trainer Best does not replace a manually locked Active Best.

To restore automatic updates, use:

```text
恢复 Stage1 Automatic Best
```

or:

```text
恢复 Stage2 Automatic Best
```

### 18.6 Delete Button Is Unavailable or Deletion Is Rejected

Common reasons include:

- the checkpoint is the current Active Best source;
- the checkpoint belongs to an active training run;
- the selection is stale and the checkpoint list needs refreshing;
- the permanent-delete confirmation is not enabled;
- the target file is outside the supported historical-checkpoint scope.

---

## 19. Settings and Management Boundaries

### 19.1 Normal User-Editable Settings

Training parameters:

- Stage1 epochs;
- Stage1 batch_size;
- Stage2 epochs;
- Stage2 batch_size.

Runtime experience:

- whether to open the separate training log terminal.

### 19.2 User-Managed Items That Are Not Training Hyperparameters

- DataFactory `work_dir`;
- whether to refresh DataFactory status during scan;
- Stage1 / Stage2 Active Best;
- checkpoint history;
- forcing Stage1 / Stage2 to use Zero-shot;
- whether an existing Profile can be overwritten.

### 19.3 Release-Managed Internal Parameters

Training also contains internal parameters managed by the release in addition to the settings explicitly exposed in the page.

These internal parameters are not exposed in the normal user UI and do not need to be manually edited for normal training.

---

## 20. Common Boundaries and Safe Operation

### 20.1 Trainer Best Is Not Active Best

Trainer Best belongs to a specific training run.

Active Best is the current user selection state for a Stage.

### 20.2 The Latest Run Is Not Necessarily the Current Active Best Source

This is especially important after manual locking.

### 20.3 Stop Does Not Delete a Training Run

Stop terminates the training process while preserving existing artifacts and logs.

### 20.4 Closing the Log Viewer Does Not Stop Training

Use the Stop buttons to actually stop training.

### 20.5 v1.0.0 Has No User-Level Training Resume

Starting training again creates a new run.

### 20.6 Checkpoint Deletion Is Permanent

Deleted model files cannot be reconstructed from history / summary files.

### 20.7 The Active Best Source Is Protected

To delete it, switch Active Best first.

### 20.8 Stage1-only / Stage2-only Are Valid Completion States

Do not wait for the other Stage merely because you are using a Partial Few-shot route.

### 20.9 Voice Profile Is the Formal Handoff to Inference

Do not treat a training-run directory itself as the normal user entry point for Inference.

Confirm Active Best, then generate a Voice Profile.

---

## 21. Next Step and Related Documentation

After Training, use the generated Voice Profile in Inference.

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
- [DataFactory Guide](./data-factory.md)
- Inference Guide (not yet published)
- Compatibility Guide (not yet published)
- Troubleshooting Guide (not yet published)
