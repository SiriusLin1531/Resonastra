# Resonastra Inference Guide

[Project README](https://github.com/SiriusLin1531/Resonastra/blob/main/README.md) | [Quick Start](./quick-start.md) | [DataFactory](./data-factory.md) | [Training](./training.md) | **English** | [中文简体](../cn/inference.md)

Applies to: **Resonastra v1.0.0**

> [!NOTE]
> This guide documents the official **Resonastra v1.0.0** Inference behavior. If a later release changes the Inference UI, Voice Profile contract, reference-audio rules, generation parameters, quality checks, or output structure, refer to the documentation for that release.

---

## 1. What This Guide Covers

This guide is for users who already have a Voice Profile ready and want to generate speech with Resonastra. You will learn how to:

- start the Inference WebUI;
- select the system default configuration or a ready Voice Profile;
- prepare reference audio, reference text, and target text;
- complete a normal speech-generation request and check the result;
- adjust advanced parameters or enable quality checks when needed;
- inspect diagnostics after a failure and return to a reproducible baseline configuration.

This guide does not cover DataFactory data preparation, Stage1 / Stage2 training, the Voice Profile training workflow, the full GPU / CUDA compatibility policy, or model-architecture theory. Inference-specific input, generation, and quality-evaluation issues are covered in the diagnostic sections of this guide.

---

## 2. Inference Workflow and Preflight

Inference is the normal user-facing entry point after Training:

```text
Voice Profile
  ↓
Reference Audio + Reference Text
  ↓
Target Text
  ↓
Inference
  ↓
Generated Audio
  ↓
Optional Quality Checks
```

### 2.1 Zero-shot and Few-shot Both Enter Inference Through a Voice Profile

Inference does not require you to manually reassemble Training artifacts. The difference between routes is already recorded by the Voice Profile:

| Route | Stage1 Source | Stage2 Source |
| --- | --- | --- |
| Zero-shot | Default Stage1 | Default Stage2 |
| Stage1-only Few-shot | Custom Stage1 | Default Stage2 |
| Stage2-only Few-shot | Default Stage1 | Custom Stage2 |
| Full Few-shot | Custom Stage1 | Custom Stage2 |

If you completed Training, the formal handoff is:

```text
Training
  ↓
Confirm Active Best
  ↓
Generate Voice Profile
  ↓
Inference
```

### 2.2 Before Your First Run

Recommended prerequisites:

1. fully extract Resonastra v1.0.0;
2. run `launch_check_env.bat` and confirm that the basic environment check passes;
3. have a usable Voice Profile, or use the system default configuration;
4. prepare a 3–10 second reference-audio clip;
5. prepare reference text that matches the spoken content of the reference audio;
6. prepare the Chinese target text you want to synthesize.

For your first request, keep the number of changing variables as small as possible:

| Item | Recommendation |
| --- | --- |
| Voice Profile | Use the system default configuration, or a Profile clearly shown as ready for generation |
| Reference audio | 3–10 seconds, single speaker, clear content |
| Reference text | Match what is actually spoken in the reference audio |
| Target text | Start with a short Chinese passage |
| Advanced Options | Keep the defaults |
| Quality checks | Keep all disabled |

Confirm that the basic generation path works first. Then add a custom Profile, quality checks, or sampling changes one at a time. This usually makes problems easier to isolate.

---

## 3. Start Inference and Read the Main UI

From the Resonastra root directory, run:

```text
launch_infer_ui.bat
```

The default address is:

```text
http://127.0.0.1:7860
```

The Inference launcher prefers the bundled `runtime/env/python.exe`, forces `HF_HUB_OFFLINE=1`, and uses the FastText language-ID model bundled with the release package. If that model is missing, startup fails directly rather than downloading it at runtime.

> [!IMPORTANT]
> The main launcher terminal provides the Inference WebUI service. Closing that terminal closes the WebUI. Unlike DataFactory / Training, the current Inference launcher does not automatically switch to another port. If `7860` is already in use, resolve the port conflict first.

### 3.1 Main UI Flow

The normal workflow is:

1. choose a voice;
2. add reference audio;
3. enter reference text and target text;
4. click `生成语音` ("Generate Speech");
5. listen to the result and review performance or quality metrics in the result area.

Other areas include `管理声音角色` ("Manage Voice Profiles"), `高级选项 / Advanced Options`, `质量检测` ("Quality Checks"), and `诊断信息` ("Diagnostics"). Advanced Options and Diagnostics are collapsed by default, and all three quality-check toggles are disabled by default.

You can think of the UI as three layers:

| Layer | Contents | Needed for the First Generation? |
| --- | --- | --- |
| Required inputs | Voice, reference audio, reference text, target text | Yes |
| Optional evaluation | DNSMOS, Speaker Similarity, WER / CER | No |
| Advanced controls | checkpoint override, device, seed, sampling parameters, output directory, timeout | Usually no |

---

## 4. Select and Manage a Voice Profile

### 4.1 System Default Configuration and `default_zh`

The voice dropdown always includes `使用默认配置 / Use default config`. For the current request, this uses the default Profile specified by `configs/user_inference_default.yaml`, which is `user_profiles/default_zh` in v1.0.0.

The formal Profile type of `default_zh` is `base_zeroshot`. At the user level, you can understand it as:

```text
Stage1 → GPT-SoVITS v2 default text-to-semantic model
Stage2 → Resonastra default Chinese acoustic generation model
```

A normal Zero-shot user can use it directly without running DataFactory or Training first.

### 4.2 How to Tell Whether a Custom Profile Is Usable

The UI summarizes Profile state into user-facing results:

| Status | Meaning | What to Do |
| --- | --- | --- |
| `可用于生成` ("Ready for Generation") | Stage1 / Stage2 sources resolve correctly | Continue with inference |
| `尚未就绪` ("Not Ready") | At least one required model source is missing or cannot be resolved | Return to Training / Voice Profile and fix it |
| `角色信息不可用` ("Voice Information Unavailable") | The current selection can no longer be found in the Registry | Refresh the voice list and select again |

Internally, the key condition is `ready_for_inference=True`. If a Profile is shown as "Not Ready", an Advanced checkpoint override cannot force it into a usable state. The Profile itself must become ready first.

### 4.3 Refreshing Voices, Temporary Selection, and Setting the Default Voice

The `管理声音角色` ("Manage Voice Profiles") area provides `刷新角色列表` ("Refresh Voice List") and `设为默认角色` ("Set as Default Voice").

- `刷新角色列表`: rescans `user_profiles/`;
- temporarily selecting a Profile affects only the current page / current request;
- `设为默认角色`: writes the current ready Profile to `user_profiles/active_profile.json`, so it is preferred the next time the UI opens;
- `使用默认配置 / Use default config`: uses the default configuration only for the current request and does not clear the persisted active Profile.

After startup or refresh, selection priority is:

1. a valid active Profile;
2. the newest Profile that is ready for inference;
3. `使用默认配置 / Use default config`.

> [!NOTE]
> `default_zh` is a built-in base Profile and should not normally be edited directly. Create a new Voice Profile if you want to save your own voice configuration.

---

## 5. Prepare Inference Inputs

All three inputs are formally required:

| Input | Required? | Key Rule |
| --- | --- | --- |
| `参考音频 / Prompt WAV` | Yes | Must be 3–10 seconds |
| `参考音频文本 / Prompt Text` | Yes | Must match the actual content of the reference audio |
| `目标文本 / Target Text` | Yes | The current User Edition follows the Chinese generation path |

### 5.1 Reference Audio: 3–10 Seconds Is a Hard Limit

The formal reference-audio extraction path requires:

```text
3.0 s ≤ reference-audio duration ≤ 10.0 s
```

Audio shorter than 3 seconds or longer than 10 seconds is rejected. The 3.0-second and 10.0-second boundary values themselves are accepted.

> [!IMPORTANT]
> The 3–10 second range is a hard runtime validation rule, not only a quality recommendation.

For more stable results, use reference audio with a single speaker, clear pronunciation, low noise, limited reverberation, and no excessively long silence. Even when using a Voice Profile that has completed Few-shot Training, normal inference still requires reference audio and reference text. They are the prompt for the current request, not a replacement for training data.

### 5.2 Reference Text

For more stable generation, the reference text should match what is actually spoken in the reference audio as accurately as possible. More accurate text generally provides more reliable alignment between the reference semantics and the text frontend. Empty reference text is rejected before the backend starts.

### 5.3 Target Text

Empty target text is rejected before the backend starts. Chinese generation is the formally supported scope of the current release.

The current UI does not define an explicit hard character-count limit for target text. Based on project development testing, longer passages are more likely to produce omissions, repetitions, abnormal speaking speed, or pronunciation issues. As a practical guideline, keep a single generation request to **about 80–100 Chinese characters or fewer** when possible, and split longer content into multiple segments. This is usage guidance, not a hard limit.

---

## 6. Generate Speech and Review the Result

For your first generation, keep Advanced Options at their defaults and leave all three quality-check toggles Off, then click `生成语音` ("Generate Speech").

After you click it, the button temporarily changes to `生成中...` ("Generating..."), and the result area shows `正在生成` ("Generating"). The current backend uses a synchronous subprocess, so the UI does not invent Stage1 / Stage2 percentage progress. The normal state transition is simply:

```text
Generating
  ↓
Final Result
```

The v1.0.0 Inference UI does not provide a dedicated Stop / Cancel button. Normally, wait for the synchronous inference request to return or for the configured timeout to trigger.

### 6.1 Successful Result

A successful request shows `生成完成` ("Generation Complete") and the message:

```text
语音已生成，可以直接在上方播放器试听。
```

("Speech has been generated. You can listen to it directly in the player above.")

The result area includes the `生成音频 / Generated Audio` player, the current voice, total time, RTF, and any enabled quality metrics.

A basic acceptance check can follow this order:

1. confirm that the status is `生成完成`;
2. confirm that the player has a playable result;
3. listen for clipping, truncation, abnormal silence, or obvious content errors;
4. then review total time and RTF;
5. enable quality checks only when you need quantitative comparison.

Success is not determined by subprocess exit code alone. The formal adapter also requires the standardized `inference_output.wav` to exist. If the process returns exit code 0 but no output WAV is found, the result is still recorded as failed.

### 6.2 RTF and Total Time

RTF can be understood as:

```text
elapsed time / generated audio duration
```

In general, `RTF < 1` means faster than real time, while `RTF > 1` means slower than real time. Under the same RTF definition, for example, `RTF = 0.5` means that generating 10 seconds of audio takes about 5 seconds.

The RTF displayed in the UI normally uses **full-request elapsed time**. If DNSMOS, Speaker Similarity, or WER / CER is enabled, metric evaluation time is also included in total time and the RTF calculation.

> [!NOTE]
> Keep the quality-check toggles consistent when comparing "pure generation speed"; otherwise, two UI RTF values are not measured under the same conditions. The first inference request may also be slower because of model initialization and cache preparation. This is a project runtime observation, not a fixed code guarantee.

---

## 7. Output Directory and Result Files

If `输出目录 / Output Directory` is left empty, the default root is:

```text
outputs/inference_runs/
```

The system creates a separate directory for each request. The directory name uses a microsecond-resolution timestamp plus `profile_name`.

Primary user outputs:

| File | Purpose |
| --- | --- |
| `inference_output.wav` | Main generated audio |
| `inference_output_peaknorm.wav` | Peak-normalized version, when available |
| `inference_summary.json` | Summary of the generation request |
| `inference_metrics.json` | Quality-metric results when metrics are enabled and produced |

The WebUI player prefers `inference_output.wav` and falls back to the peak-normalized output only when the primary output is unavailable.

A request directory may also contain `prompt_reference*` reference-audio copies and request / diagnostic files. These files are mainly for recording and troubleshooting; their exact count and suffixes are not part of the stable user interface.

> [!WARNING]
> If you manually specify a custom output directory, the system does not automatically add the default "timestamp + profile_name" isolation layer. Multiple requests written to the same directory may overwrite fixed filenames or mix diagnostic artifacts. Unless you specifically need custom directory management, leave the output directory empty.

---

## 8. Advanced Options

For first-time use, `高级选项 / Advanced Options` normally does not need to be changed:

| Control | Default / Initial Value | Main Purpose |
| --- | --- | --- |
| Output Directory | Empty | Specify the result directory |
| Stage1 checkpoint override | Empty | Temporarily compare another Stage1 checkpoint |
| Stage2 checkpoint override | Empty | Temporarily compare another Stage2 checkpoint |
| `设备 / Device` | `cuda` | Select the requested device for the main TTS request |
| `固定随机种子 / Fix seed` | Off | Make parameter comparisons more controlled |
| Stage1 sampling | Defaults | Adjust Stage1 sampling |
| Stage2 `length_scale` | `1.0` | Adjust Stage2 length / duration-related behavior |
| `推理超时秒数 / Timeout Seconds` | `600` | Control the subprocess timeout |

### 8.1 Checkpoint Override

Precedence is:

```text
Manual checkpoint override
  >
Currently selected Voice Profile
  >
User Edition default configuration
```

However, a custom Profile must first resolve normally and satisfy `ready_for_inference=True`. An override cannot force a `尚未就绪` ("Not Ready") Profile into a usable state.

An override affects only the current request. It does not modify `profile_config.json`, `profile_manifest.json`, Active Best, or the Voice Profile itself. It is mainly intended for temporary model comparisons; for normal daily generation, keeping the Profile's own model sources complete is recommended.

### 8.2 Device, Seed, and Timeout

`设备 / Device` supports `cuda` and `cpu`, with `cuda` requested by default. The current models are primarily designed around GPU use, so explicitly selecting CPU should normally be expected to increase inference time substantially. If CUDA is requested but `torch.cuda.is_available()` is False at runtime, the formal inference script warns and falls back to CPU. The UI Device field therefore represents the **requested device**; confirm the actual execution device from the launcher terminal or Diagnostics when needed. Quality metrics use their own backend devices, and faster-whisper for WER / CER always runs on CPU / int8.

`固定随机种子 / Fix seed` is Off by default, so the seed is `None`. When enabled, the UI uses `Seed value`; an invalid or negative value is normalized to 0. Fixing the seed helps reduce randomness during parameter comparison, but does not guarantee bit-exact output across every environment.

`推理超时秒数 / Timeout Seconds` defaults to 600 seconds. A value less than or equal to 0 means no explicit subprocess timeout is set. If a timeout occurs, the adapter terminates the subprocess and formally records:

```text
status = failed
error_type = DeveloperInferenceTimeoutError
```

Any stdout / stderr tail that can be captured is retained for diagnostics.

### 8.3 Stage1 Sampling

The formal UI exposes:

| Parameter | Default | UI Range | User-Level Interpretation |
| --- | ---: | --- | --- |
| `temperature` | `1.0` | `0.1–2.0` | Controls sampling randomness |
| `top_p` | `1.0` | `0.1–1.0` | Restricts candidates by cumulative probability |
| `top_k` | `15` | `1–100` | Restricts candidates by candidate count |

These parameters interact with one another, and there is no single fixed combination that is "best" for every voice. When comparing settings, keep the Voice Profile, reference audio / text, target text, and seed fixed, and change only one main variable at a time.

### 8.4 Stage2 `length_scale`

The default is `1.0`, with a UI range of `0.5–2.0`. It can be understood as a Stage2 target-length / duration-related control; it is not a volume or pitch parameter. Keep it at 1.0 for first-time use. If you encounter obvious rhythm or duration problems, return it to the default before continuing troubleshooting.

### 8.5 User Settings vs. Internal Parameters

The controls shown in the formal UI are the settings exposed to users in v1.0.0. Other internal parameters are managed by the release and do not need to be modified manually for normal use.

---

## 9. Optional Quality Checks

The formal UI provides:

- `DNSMOS 语音质量评价` ("DNSMOS Speech Quality Evaluation");
- `Speaker Similarity 声音相似度` ("Speaker Similarity");
- `WER / CER 文本准确度` ("WER / CER Text Accuracy").

All three toggles are Off by default. Quality checks are not required to generate speech. When first confirming that the main generation path works, keep them disabled and enable them later only as needed.

| Metric | Main Question | Typical Direction |
| --- | --- | --- |
| DNSMOS OVRL | Overall naturalness / overall quality | Higher is usually better |
| DNSMOS SIG | Quality of the speech signal itself | Higher is usually better |
| Speaker Sim | How close the voice is to the reference speaker | Higher usually means closer |
| WER / CER | Whether the generated content can be recognized correctly | Lower is better |

These metrics evaluate different dimensions. No single metric should replace actual listening or the other metrics.

### 9.1 DNSMOS and Speaker Similarity

DNSMOS results can display `DNSMOS OVRL` and `DNSMOS SIG`. This guide does not define one fixed pass threshold that applies across every dataset and voice; DNSMOS is more useful for relative comparison under the same test conditions.

Speaker Similarity is shown as `Speaker Sim`. A higher value usually means the embedding representation of the generated speech is closer to that of the reference speaker, but it does not imply that pronunciation accuracy, rhythm, emotion, or noise quality is also better.

### 9.2 WER / CER

The current User Edition uses local faster-whisper:

```text
model = pretrained_models/faster_whisper_medium
device = cpu
compute_type = int8
```

This ASR path is independent of the Device selected for the main TTS request. CPU / int8 is the formal configuration for the current Windows User Edition so that this evaluation path does not introduce an additional CUDA-runtime dependency. For Chinese, CER is a character-level error rate, while the current WER implementation includes character-proxy logic and should not be interpreted simply as English-style word-level WER.

If WER / CER is high while the generated audio sounds mostly correct, also inspect the ASR transcript. The final error value is affected by both the generated content and the evaluation ASR.

### 9.3 Individual Metric Failure vs. Main Task Failure

The quality evaluator catches most individual backend errors separately, so you may see:

```text
Main audio succeeded
+ one metric shows "not computed" or "failed"
```

However, this does not mean every metrics-stage error is always isolated from the main task. If evaluator initialization or another uncaught error causes the entire inference subprocess to exit non-zero, the request is still recorded as `生成失败` ("Generation Failed").

When troubleshooting, first distinguish between "main audio succeeded, one metric failed" and "the entire inference task failed during the metrics stage."

---

## 10. Diagnostics and Common Problems

`诊断信息` ("Diagnostics") is collapsed by default and is mainly intended for troubleshooting. It may contain Profile Registry JSON, Inference Diagnostics JSON, resolved paths, the developer command, stdout / stderr, and backend errors.

The normal result panel intentionally hides absolute paths, the developer command, raw tracebacks, and raw backend errors. Expand Diagnostics only when you need to investigate a problem.

> [!WARNING]
> Diagnostic content may include your local username, absolute paths, runtime commands, or error text. Before posting logs to a public issue, forum, or chat, review them and remove anything you do not want to disclose.

### 10.1 Common Input and Profile Errors

| Message / Status | Common Cause | What to Do |
| --- | --- | --- |
| `缺少参考音频` ("Missing Reference Audio") | No Prompt WAV was provided | Add a valid 3–10 second reference-audio clip |
| `缺少参考文本` ("Missing Reference Text") | Prompt Text is empty | Enter text that matches the reference audio |
| `缺少生成文本` ("Missing Target Text") | Target Text is empty | Enter target text |
| `尚未就绪` ("Not Ready") | A Profile model source is missing or cannot be resolved | Return to Training / Voice Profile and fix it |
| `角色信息不可用` ("Voice Information Unavailable") | The current selection is no longer in the Registry | Refresh the voice list and select again |

The first three input errors are blocked before the backend starts, so you do not need to investigate CUDA or checkpoints first.

### 10.2 Generation Failure or Frequent Timeout

If the final result shows `生成失败` ("Generation Failed"), troubleshoot in this order:

1. return to a valid 3–10 second reference-audio clip, accurate reference text, and shorter target text;
2. use the system default configuration or a Voice Profile clearly shown as ready for generation;
3. clear checkpoint overrides and restore Advanced Options to their defaults;
4. disable all three quality checks;
5. test the main generation path again;
6. if it still fails, expand `诊断信息` ("Diagnostics") and inspect stderr, the developer command, and records such as `result.json` in the request directory.

If basic generation finishes within 600 seconds but timeout occurs only after enabling a quality check, enable one metric at a time to distinguish generation time from evaluation time.

For additional Inference troubleshooting, continue with this section and the related sections of this guide. For general runtime, GPU, or CUDA issues, use the [Compatibility Guide](./compatibility.md).

---

## 11. Next Steps and Related Documentation

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
- [Training Guide](./training.md)
- [Compatibility Guide](./compatibility.md)
