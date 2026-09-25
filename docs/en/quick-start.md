# Resonastra Quick Start

[Project README](https://github.com/SiriusLin1531/Resonastra/blob/main/README.md) | **English** | [中文简体](../cn/quick-start.md)

Applies to: **Resonastra v1.0.0**

This guide is for first-time Resonastra users. Resonastra supports two main ways to get started:

- **Zero-shot** — use the bundled base models directly, with no user training.
- **Few-shot** — adapt the models with your own speech dataset before inference.

Few-shot can be further split into Stage1-only, Stage2-only, or full Stage1 + Stage2 training.

You do not need to finish the full training workflow before trying Resonastra. The fastest path is to use the bundled Zero-shot models first. If you want the system to adapt to your own speech data, continue with one of the Few-shot routes.

```text
Download and Extract
  ↓
Environment Check
  ↓
Choose a Route
  ├─ Zero-shot ─────────────────────────────┐
  │                                        │
  └─ Few-shot                              │
       ↓                                   │
     DataFactory                           │
       ↓                                   │
     Train Stage1 / Stage2 as Needed       │
       ↓                                   │
     Voice Profile                         │
       └───────────────────────────────────┤
                                           ↓
                                      Inference
                                           ↓
                                      Generated Audio
```

This Quick Start follows the standard user workflow and default settings. Dataset quality, training parameters, checkpoint management, GPU / CUDA compatibility, and detailed troubleshooting are covered in separate documentation.

---

## 1. Before You Start

Resonastra v1.0.0 targets:

- Windows 10 / Windows 11 x64
- local execution
- NVIDIA GPUs recommended for training and inference

The DataFactory workflow in v1.0.0 currently exposes Chinese `zh` only.

Regardless of the route you choose, Inference requires:

- a **3–10 second** reference audio clip / Prompt WAV
- a matching transcript for that reference audio / Prompt Text
- the text you want to synthesize / Target Text

The 3–10 second reference-audio range is a hard requirement in the current v1.0.0 inference path. Reference audio shorter than 3 seconds or longer than 10 seconds is rejected.

If you choose Few-shot, you also need your own source speech recordings for DataFactory and training.

For a first Few-shot run, a dataset on the scale of **a few minutes to a few dozen minutes** is a practical starting point when balancing adaptation quality with preparation and training cost. This is not a hard dataset-size limit; larger datasets can also be used.

For more stable training results, your dataset should preferably contain:

- **a single speaker**
- **clear pronunciation**
- **normal speaking speed**
- **no noticeable background noise**

These are dataset-quality recommendations, not hard DataFactory input restrictions.

> [!IMPORTANT]
> Resonastra v1.0.0 bundles Python 3.10.20, PyTorch 2.5.1, and CUDA Runtime 11.8. Passing the environment check does not mean every newer GPU architecture has been validated by actually running CUDA kernels. If you use an RTX 50 Series GPU or another new architecture, read the compatibility notes in the [project README](https://github.com/SiriusLin1531/Resonastra/blob/main/README.md) first.

---

## 2. Download and Extract Resonastra

### 2.1 Download the Official Release Package

Resonastra v1.0.0:

- **GitHub Release:** <https://github.com/SiriusLin1531/Resonastra/releases/tag/v1.0.0>
- **Baidu Netdisk:** <https://pan.baidu.com/s/1_344UD7sJci6IuBgtbmF6A?pwd=8848>
- **Extraction Code:** `8848`
- **Archive:** `Resonastra_v1.0.0.zip`

> [!IMPORTANT]
> GitHub-generated **Source code (zip / tar.gz)** files are source snapshots only. They are not the complete Windows package. For first-time use, download the official `Resonastra_v1.0.0.zip` package.

### 2.2 Extract the Entire Package

Extract `Resonastra_v1.0.0.zip` completely to a local drive.

Run all launchers from the extracted Resonastra root directory, for example:

```text
Resonastra/
├── launch_check_env.bat
├── launch_data_factory_ui.bat
├── launch_training_ui.bat
├── launch_infer_ui.bat
├── runtime/
├── scripts/
├── src/
└── ...
```

Do not copy individual `.bat` files elsewhere and run them separately.

> [!IMPORTANT]
> `launch_data_factory_ui.bat`, `launch_training_ui.bat`, and `launch_infer_ui.bat` each keep a **WebUI launcher terminal** open while the corresponding WebUI is running. Do not close this main terminal while using that WebUI; closing it stops the corresponding WebUI process.
>
> The separate progress / log terminals mentioned later are read-only viewer windows and are different from the main WebUI launcher terminal.

---

## 3. Run the Environment Check

Before using DataFactory, Training, or Inference, run:

```text
launch_check_env.bat
```

You can double-click the file in File Explorer.

The environment checker validates the main local components required by v1.0.0, including:

- Microsoft Visual C++ v14 x64 Redistributable
- the bundled Resonastra runtime
- required Python packages
- default configuration
- default model assets

The minimum Microsoft Visual C++ Redistributable version checked by Resonastra is:

```text
14.44.35211
```

### Successful Environment Check

When the check completes, you can continue if the terminal reports:

```text
[OK] Environment check completed successfully.
```

The checker also writes:

```text
logs/check_user_env_report.json
```

which records the environment-check result.

> [!NOTE]
> The current environment checker validates the runtime, dependencies, configuration, and model assets. It does not run a CUDA kernel to verify GPU compute capability.

If the terminal reports `[FAIL]`, resolve the reported problem before continuing.

---

## 4. Choose Your Route

Resonastra v1.0.0 supports four model combinations:

| Route | Stage1 | Stage2 | Training Cost | Adaptation Scope | Guidance |
| --- | --- | --- | --- | --- | --- |
| **Zero-shot** | Default model | Default model | No training | No user-data training | **Recommended for first-time use** |
| **Stage1-only Few-shot** | Few-shot | Default model | Lower | Stage1 only | Good for lightweight Few-shot adaptation |
| **Stage2-only Few-shot** | Default model | Few-shot | Higher | Stage2 only | Supported, but usually not recommended on its own |
| **Full Few-shot** | Few-shot | Few-shot | Highest, but only a limited increase over Stage2-only | Stage1 + Stage2 | **Recommended complete training route** |

### Which Route Should You Choose?

If you only want to confirm that Resonastra runs correctly and generate your first sample as quickly as possible:

**Choose Zero-shot.**

If you want a lighter personalization path with lower data-preparation and training cost:

**Choose Stage1-only Few-shot.**

If you already plan to train Stage2:

**Full Few-shot is usually the better choice.**

Based on current project development and training experience, the total time required for Stage1 data preparation and training is significantly lower than Stage2. In practice, Stage2-only already accounts for most of the Few-shot time cost, while adding Stage1 introduces a comparatively small additional time cost.

Therefore, when considering both time cost and adaptation scope:

> **Stage2-only is generally not recommended purely as a time-saving option. If you already intend to train Stage2, completing both Stage1 and Stage2 is usually the better route.**

Full Few-shot adapts both stages with user data and is the current recommended complete training route for the best overall result.

> [!NOTE]
> Stage2-only remains a supported configuration. It can be useful for controlled experiments, comparisons, or cases where you specifically want to replace Stage2 only. The recommendation above is intended for normal first-time and general personalization workflows.

The corresponding Voice Profile types are:

```text
Zero-shot              → base_zeroshot
Stage1-only Few-shot   → few_shot_stage1_only
Stage2-only Few-shot   → few_shot_stage2_only
Full Few-shot          → few_shot_dual
```

If you choose **Zero-shot**, skip Sections 5–7 and go directly to **Section 8: Generate Speech with a Voice Profile**.

If you choose any Few-shot route, continue below.

---

## 5. Few-shot: Prepare Training Data with DataFactory

Run:

```text
launch_data_factory_ui.bat
```

The launcher starts Resonastra DataFactory and attempts to open it in your browser.

The default port is:

```text
7861
```

If that port is already in use, the DataFactory launcher automatically selects another available port. Use the address shown in the launcher terminal.

> [!NOTE]
> DataFactory's **`显示终端进度窗口`** ("Show terminal progress window") option is enabled by default. Long operations such as data preparation, Stage1 dataset generation, or Stage2 processing may open an additional terminal that shows live logs. This terminal is only a progress viewer; closing it does not stop the running DataFactory task.
>
> Do not close the main WebUI launcher terminal opened by `launch_data_factory_ui.bat`.

### 5.1 Enter the Basic Inputs

In **Resonastra · Data Factory**, fill in the **`基础输入`** ("Basic Input") section.

For a first run, use the simplest local-directory workflow:

- **`原始音频来源`** ("Raw Audio Source"): choose `本地目录路径`
- **`本地原始音频目录`** ("Local Raw Audio Directory"): enter the directory containing your source recordings
- **`说话人 / 角色名`** ("Speaker / Character Name"): enter a name for this voice
- **`语言`** ("Language"): keep `zh`

For example, if you enter:

```text
Character_A
```

as the speaker name and leave the work directory empty, DataFactory uses:

```text
user_data/Character_A_factory
```

as the default work directory.

For a first run, you can leave the work-directory field under **`已有项目 / 工作目录`** ("Existing Project / Work Directory") empty.

### 5.2 Start Data Preparation

After confirming the basic inputs, click:

```text
开始准备数据
```

("Start Data Preparation")

DataFactory processes the source audio and runs the initial recognition step.

#### Success Condition

When complete, the Audio Processing stage should report:

```text
音频处理与基础识别产物已就绪。
```

("Audio processing and basic recognition outputs are ready.")

---

### 5.3 Review and Confirm the ASR Transcript

To avoid sending obvious recognition errors directly into training data, this Quick Start uses the manual review path.

Click:

```text
启动 / 打开人工校对器
```

("Launch / Open Manual Proofreader")

Review the ASR transcript and make corrections if needed.

When finished:

1. Save your changes in the proofreading tool.
2. Return to DataFactory.
3. Click:

```text
已保存并完成校对
```

("Saved and Finished Proofreading")

If you have already confirmed that the automatic transcript does not need changes, you can instead choose:

```text
无需校对，继续
```

("No Proofreading Needed, Continue")

#### Success Condition

The Text Confirmation stage should report:

```text
文本确认结果已就绪。
```

("Text confirmation result is ready.")

---

### 5.4 Generate Training Data for Your Route

After text confirmation, Stage1 and Stage2 training data can be generated independently.

#### Stage1-only Few-shot

Click only:

```text
生成 Stage1 训练数据
```

("Generate Stage1 Training Data")

When complete, DataFactory should report:

```text
Stage1 训练数据已完成。
```

("Stage1 training data is complete.")

You do not need to generate Stage2 training data.

#### Stage2-only Few-shot

Click only:

```text
生成 Stage2 训练数据
```

("Generate Stage2 Training Data")

When complete, DataFactory should report:

```text
Stage2 训练数据已完成。
```

("Stage2 training data is complete.")

You do not need to generate Stage1 training data.

#### Full Few-shot

Complete both:

```text
生成 Stage1 训练数据
生成 Stage2 训练数据
```

("Generate Stage1 Training Data" and "Generate Stage2 Training Data")

The final recommendation summary should report:

```text
Stage1 与 Stage2 数据均已完成。
```

("Both Stage1 and Stage2 data are complete.")

### End of This Section

Remember the **Speaker / Character Name** you used. If you kept the default work-directory rule, Training will continue from:

```text
user_data/{speaker_name}_factory
```

> [!NOTE]
> DataFactory also supports browser uploads, loading an existing work directory, resuming Stage2 processing, and additional advanced settings. Those features are outside this Quick Start.

---

## 6. Few-shot: Train the Required Stages

Run:

```text
launch_training_ui.bat
```

The default port is:

```text
7863
```

If that port is already in use, the Training launcher automatically selects an available port.

> [!NOTE]
> Training's **`打开独立训练日志终端（可选）`** ("Open Independent Training Log Terminal — Optional") option is disabled by default. The Training Live Monitor already shows training status. If you want more detailed live logs, you can enable the independent log terminal manually. It is a read-only viewer; closing it does not stop training.
>
> Do not close the main WebUI launcher terminal opened by `launch_training_ui.bat`.

### 6.1 Scan the DataFactory Data

In **Resonastra · Training**:

1. Enter the same name used in DataFactory under **`说话人 / 角色名`** ("Speaker / Character Name").
2. If you used the default work-directory rule, leave **`DataFactory 工作目录`** ("DataFactory Work Directory") empty.
3. Click:

```text
扫描训练数据
```

("Scan Training Data")

If you used a custom DataFactory work directory, enter that same directory here.

Check the data and training entry points required by your route:

| Route | Data That Must Be Ready | Training That Must Be Available |
| --- | --- | --- |
| Stage1-only | Stage1 data | Stage1 Training |
| Stage2-only | Stage2 data | Stage2 Training |
| Full Few-shot | Stage1 + Stage2 data | Stage1 + Stage2 Training |

Partial Few-shot does not require the overall "complete training data" state to be ready. Only the stage used by your selected route needs to pass its entry checks.

---

### 6.2 Stage1 Training

The following routes require Stage1:

- Stage1-only Few-shot
- Full Few-shot

Keep the default `epochs` and `batch_size` values already loaded in the WebUI, then click:

```text
启动 Stage1 Training
```

("Start Stage1 Training")

Wait for training to complete.

#### Success Condition

The training state should eventually become:

```text
succeeded
```

and may report something similar to:

```text
stage1 训练完成：run_id=...
```

("stage1 training completed: run_id=...")

After a normal successful run produces a best checkpoint, the Training worker automatically registers the Trainer Best. In a first-run workflow without a manual Active Best lock, it updates the Stage1 Active Best automatically.

---

### 6.3 Stage2 Training

The following routes require Stage2:

- Stage2-only Few-shot
- Full Few-shot

Click:

```text
启动 Stage2 Training
```

("Start Stage2 Training")

Wait for training to complete.

#### Success Condition

The training state should eventually become:

```text
succeeded
```

and may report something similar to:

```text
stage2 训练完成：run_id=...
```

("stage2 training completed: run_id=...")

After a normal successful run produces a best checkpoint, the Trainer Best updates the Stage2 Active Best automatically in a first-run workflow without a manual Active Best lock.

> [!IMPORTANT]
> Stage2 data preparation and training account for most of the Few-shot time cost. Because Stage1 is significantly faster overall, if you are already planning to train Stage2, Full Few-shot is usually more worthwhile than choosing Stage2-only purely to save time.

> [!NOTE]
> The Training UI also provides manual Active Best selection, Automatic Best management, checkpoint deletion, and other training-management features. They are not required for the first successful workflow.

---

## 7. Few-shot: Generate a Voice Profile

After completing the stages required by your route, use **`生成 Voice Profile v2`** ("Generate Voice Profile v2") to create the model combination used by Inference.

You can leave `profile_name` empty. In that case, the system prefers the current **Speaker / Character Name**.

### Stage1-only Few-shot

Set:

- **`Stage1 强制使用 zero-shot`** ("Force Stage1 zero-shot"): Off
- **`Stage2 强制使用 zero-shot`** ("Force Stage2 zero-shot"): On

Then click:

```text
生成 Voice Profile v2
```

Expected result:

```text
类型：few_shot_stage1_only
Stage1 来源：few-shot active best
Stage2 来源：zero-shot default
可用于推理：True
```

Meaning:

```text
Type: few_shot_stage1_only
Stage1 source: few-shot active best
Stage2 source: zero-shot default
Ready for inference: True
```

### Stage2-only Few-shot

Set:

- **`Stage1 强制使用 zero-shot`** ("Force Stage1 zero-shot"): On
- **`Stage2 强制使用 zero-shot`** ("Force Stage2 zero-shot"): Off

Then click:

```text
生成 Voice Profile v2
```

Expected result:

```text
类型：few_shot_stage2_only
Stage1 来源：zero-shot default
Stage2 来源：few-shot active best
可用于推理：True
```

Meaning:

```text
Type: few_shot_stage2_only
Stage1 source: zero-shot default
Stage2 source: few-shot active best
Ready for inference: True
```

### Full Few-shot

Set:

- **`Stage1 强制使用 zero-shot`** ("Force Stage1 zero-shot"): Off
- **`Stage2 强制使用 zero-shot`** ("Force Stage2 zero-shot"): Off

Click:

```text
生成 Voice Profile v2
```

Expected result:

```text
类型：few_shot_dual
Stage1 来源：few-shot active best
Stage2 来源：few-shot active best
可用于推理：True
```

Meaning:

```text
Type: few_shot_dual
Stage1 source: few-shot active best
Stage2 source: few-shot active best
Ready for inference: True
```

The generated Voice Profile is stored under:

```text
user_profiles/<profile_name>/
```

and includes:

```text
profile_config.json
profile_manifest.json
```

A Voice Profile is Resonastra's inference-time model-combination configuration. It determines whether Stage1 and Stage2 use the bundled default models or your trained Few-shot models.

---

## 8. Generate Speech with a Voice Profile

Run:

```text
launch_infer_ui.bat
```

Inference uses:

```text
http://127.0.0.1:7860
```

The launcher attempts to open your browser automatically.

If the browser does not open automatically but the launcher terminal does not report a startup failure, open the address above manually.

> [!IMPORTANT]
> Keep the WebUI launcher terminal opened by `launch_infer_ui.bat` running while using Inference. Do not close that window.

### 8.1 Select a Voice Profile

Under **`1. 选择声音角色`** ("1. Select Voice Profile"), choose the appropriate Profile.

#### Zero-shot

Choose the bundled Chinese base profile:

```text
default_zh
```

Its Profile type is:

```text
base_zeroshot
```

This Profile uses the bundled default Stage1 and Stage2 models and does not require DataFactory or Training.

#### Few-shot

Choose the Voice Profile generated in Section 7.

#### Success Condition

The Profile summary should report:

```text
可用于生成
```

("Ready for generation")

and both Stage1 and Stage2 model states should be ready.

---

### 8.2 Add Reference Audio

Under **`2. 添加参考音频`** ("2. Add Reference Audio"), provide:

- **`参考音频 / Prompt WAV`**
- **`参考音频文本 / Prompt Text`**

The reference audio must be:

```text
3–10 seconds
```

The formal v1.0.0 prompt-extraction path strictly checks this duration. Audio shorter than 3 seconds or longer than 10 seconds is rejected.

The Prompt Text should match what is actually spoken in the reference audio as closely as possible.

For example, if the reference audio says:

```text
欢迎使用 Resonastra。
```

then the Prompt Text should contain that same spoken content.

---

### 8.3 Enter the Target Text

Enter the text to generate in:

```text
目标文本 / Target Text
```

For example:

```text
这是使用 Resonastra 生成的第一段语音。
```

> [!NOTE]
> Based on current development testing, a single Chinese generation request is best kept to roughly **80–100 Chinese characters or fewer**. This is not a hard character limit in the code. The practical stable length varies with the text content, model state, reference transcript, and other factors.
>
> Very long text may increase the chance of unclear pronunciation, repeated speech, or disordered content. For longer material, split the text into shorter segments and generate them separately.

---

### 8.4 Keep the Default Inference Settings

For a first run, you do not need to change **Advanced Options**.

The following quality checks are disabled by default:

- DNSMOS speech-quality evaluation
- Speaker Similarity
- WER / CER text-accuracy evaluation

This Quick Start does not require them.

---

### 8.5 Generate Speech

Confirm that:

- a usable Voice Profile is selected
- Prompt WAV is provided
- Prompt Text is filled in
- Target Text is filled in

Then click:

```text
生成语音
```

("Generate Speech")

> [!NOTE]
> **The first inference run may take noticeably longer because of warm-up.** The first generation usually needs to load models and initialize the runtime before actual synthesis begins, so it can be much slower than later runs. Please wait patiently; do not repeatedly click "Generate Speech" or close the WebUI launcher terminal just because no result appears immediately.

#### Success Condition

The result area should report:

```text
生成完成
```

("Generation complete")

and:

```text
语音已生成，可以直接在上方播放器试听。
```

("Speech has been generated and can be played directly in the player above.")

The generated result appears in:

```text
生成音频 / Generated Audio
```

You can play the generated speech directly in the WebUI.

By default, inference results are saved under:

```text
outputs/inference_runs/
```

If you do not specify a custom output directory, each inference request creates a separate run directory.

---

## 9. Completion Checklist by Route

Only check the items that apply to your selected route.

### Zero-shot

- [ ] `launch_check_env.bat` passes
- [ ] `default_zh` is selected in Inference
- [ ] the Profile is ready for generation
- [ ] Inference reports `生成完成`
- [ ] the generated audio plays successfully

### Stage1-only Few-shot

- [ ] DataFactory text confirmation is complete
- [ ] Stage1 training data is complete
- [ ] Stage1 Training status is `succeeded`
- [ ] Voice Profile type is `few_shot_stage1_only`
- [ ] Stage1 source is `few-shot active best`
- [ ] Stage2 source is `zero-shot default`
- [ ] Voice Profile reports `可用于推理：True`
- [ ] Inference completes successfully

### Stage2-only Few-shot

- [ ] DataFactory text confirmation is complete
- [ ] Stage2 training data is complete
- [ ] Stage2 Training status is `succeeded`
- [ ] Voice Profile type is `few_shot_stage2_only`
- [ ] Stage1 source is `zero-shot default`
- [ ] Stage2 source is `few-shot active best`
- [ ] Voice Profile reports `可用于推理：True`
- [ ] Inference completes successfully

### Full Few-shot

- [ ] DataFactory text confirmation is complete
- [ ] Stage1 training data is complete
- [ ] Stage2 training data is complete
- [ ] Stage1 Training status is `succeeded`
- [ ] Stage2 Training status is `succeeded`
- [ ] Voice Profile type is `few_shot_dual`
- [ ] Stage1 source is `few-shot active best`
- [ ] Stage2 source is `few-shot active best`
- [ ] Voice Profile reports `可用于推理：True`
- [ ] Inference completes successfully

---

## 10. Next Steps

If you completed only the Zero-shot route, you can continue with Few-shot later to adapt Resonastra using your own speech data.

For Few-shot:

- **Stage1-only** is the lighter training route with lower time cost
- **Stage2-only** is supported, but is usually not recommended purely from a time-cost perspective
- **Full Few-shot** is the current recommended complete personalization route

More detailed documentation will cover:

- **DataFactory** — data preparation, ASR, manual proofreading, and Stage1 / Stage2 dataset generation
- **Training** — training parameters, Live Monitor, checkpoints, and Active Best
- **Inference** — Voice Profiles, inference controls, and quality evaluation
- **Compatibility** — GPU, CUDA, and runtime environment
- **Troubleshooting** — startup, CUDA, ASR, training, and inference issues
- **Release Verification** — ZIP, SHA256, and official package verification
