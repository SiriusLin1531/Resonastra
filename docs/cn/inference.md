# Resonastra Inference 使用指南

[项目 README](./README.md) | [快速开始](./quick-start.md) | [DataFactory](./data-factory.md) | [Training](./training.md) | **中文简体**

适用版本：**Resonastra v1.0.0**

> [!NOTE]
> 本指南针对 **Resonastra v1.0.0** 的正式发行行为编写。后续版本如果调整 Inference 界面、Voice Profile 契约、参考音频规则、生成参数、质量检测或输出结构，应以对应版本文档为准。

---

## 1. 本指南解决什么问题

本指南面向已经准备好 Voice Profile、希望使用 Resonastra 生成语音的用户。你将完成以下任务：

- 启动 Inference WebUI；
- 选择系统默认配置或已经就绪的 Voice Profile；
- 准备参考音频、参考文本和目标文本；
- 完成一次正常语音生成并检查结果；
- 在需要时调整高级参数或开启质量检测；
- 在失败时查看诊断信息并回到可复现的基准配置。

本指南不展开 DataFactory 数据准备、Stage1 / Stage2 训练、Voice Profile 训练过程、完整 GPU / CUDA 兼容性政策或模型结构理论。Inference 自身的输入、生成失败与质量检测问题在本文诊断章节说明。

---

## 2. Inference 工作流与开始之前

Inference 是 Training 之后的正式使用入口：

```text
Voice Profile
  ↓
参考音频 + 参考文本
  ↓
目标文本
  ↓
Inference
  ↓
生成音频
  ↓
按需做质量检测
```

### 2.1 Zero-shot 与 Few-shot 都通过 Voice Profile 进入 Inference

Inference 不要求用户重新拼接 Training 产物。不同路线的差别已经由 Voice Profile 记录：

| 路线 | Stage1 来源 | Stage2 来源 |
| --- | --- | --- |
| Zero-shot | 默认 Stage1 | 默认 Stage2 |
| Stage1-only Few-shot | 自定义 Stage1 | 默认 Stage2 |
| Stage2-only Few-shot | 默认 Stage1 | 自定义 Stage2 |
| Full Few-shot | 自定义 Stage1 | 自定义 Stage2 |

如果走过 Training，正式交接路径是：

```text
Training
  ↓
确认 Active Best
  ↓
生成 Voice Profile
  ↓
Inference
```

### 2.2 第一次运行前准备

建议先满足：

1. 已完整解压 Resonastra v1.0.0；
2. 已运行 `launch_check_env.bat` 并确认基础环境检查通过；
3. 有一个可用 Voice Profile，或直接使用系统默认配置；
4. 准备 3–10 秒参考音频；
5. 准备与参考音频内容一致的参考文本；
6. 准备需要生成的中文目标文本。

第一次使用时尽量减少变量：

| 项目 | 建议 |
| --- | --- |
| Voice Profile | 使用系统默认配置，或一个明确显示“可用于生成”的 Profile |
| 参考音频 | 3–10 秒、单说话人、内容清晰 |
| 参考文本 | 与参考音频实际内容一致 |
| 目标文本 | 先用较短中文文本 |
| 高级选项 | 保持默认 |
| 质量检测 | 全部关闭 |

先确认基础生成链能成功，再逐步加入自定义 Profile、质量检测和采样参数，通常更容易排查问题。

---

## 3. 启动 Inference 与主界面

从 Resonastra 根目录运行 `launch_infer_ui.bat`，默认打开：

```text
http://127.0.0.1:7860
```

Inference 启动器会优先使用内置 `runtime/env/python.exe`，强制 `HF_HUB_OFFLINE=1`，并使用发布包内置的 FastText language-ID 模型；缺少该模型时会直接失败，不会在运行时联网下载。

> [!IMPORTANT]
> 主启动终端提供 Inference WebUI 服务，关闭终端等价于关闭 WebUI。当前启动器也不会像 DataFactory / Training 一样自动换端口；如果 `7860` 被占用，需要先处理端口冲突。

### 3.1 主界面怎么读

正式主流程是：

1. 选择声音角色；
2. 添加参考音频；
3. 输入参考文本和目标文本；
4. 点击 `生成语音`；
5. 在结果区试听并查看性能或质量指标。

其它区域包括 `管理声音角色`、`高级选项 / Advanced Options`、`质量检测` 和 `诊断信息`。高级选项与诊断信息默认收起，三个质量检测开关默认关闭。

可以把界面分成三层：

| 层级 | 内容 | 第一次生成是否需要 |
| --- | --- | --- |
| 必要输入 | 声音角色、参考音频、参考文本、目标文本 | 是 |
| 可选验收 | DNSMOS、Speaker Similarity、WER / CER | 否 |
| 高级控制 | checkpoint override、设备、seed、采样参数、输出目录、timeout | 通常否 |

---

## 4. 选择和管理 Voice Profile

### 4.1 系统默认配置与 `default_zh`

`声音角色` 下拉框始终包含 `使用默认配置 / Use default config`。它表示本次请求使用 `configs/user_inference_default.yaml` 中指定的默认 Profile，v1.0.0 对应 `user_profiles/default_zh`。

`default_zh` 的正式类型是 `base_zeroshot`，用户层可以理解为：

```text
Stage1 → GPT-SoVITS v2 默认文本转语义模型
Stage2 → Resonastra 默认中文声学生成模型
```

普通 Zero-shot 用户可以直接使用它，不需要先运行 DataFactory 或 Training。

### 4.2 如何判断自定义 Profile 是否可用

界面会把 Profile 状态整理成用户级结果：

| 状态 | 含义 | 建议 |
| --- | --- | --- |
| `可用于生成` | Stage1 / Stage2 来源可以正常解析 | 可以继续推理 |
| `尚未就绪` | 至少一个必要模型来源缺失或解析失败 | 返回 Training / Voice Profile 检查 |
| `角色信息不可用` | 当前选择已无法在 Registry 中找到 | 刷新角色列表并重新选择 |

内部对应的核心条件是 `ready_for_inference=True`。如果 Profile 显示“尚未就绪”，Advanced checkpoint override 也不能把它强行变成可用 Profile；必须先解决 Profile 本身的就绪问题。

### 4.3 刷新角色、临时选择与“设为默认角色”

`管理声音角色` 中提供 `刷新角色列表` 和 `设为默认角色`。

- `刷新角色列表`：重新扫描 `user_profiles/`；
- 临时选择某个 Profile：只影响当前页面 / 当前请求；
- `设为默认角色`：把当前已就绪 Profile 写入 `user_profiles/active_profile.json`，下次打开 UI 时优先选择；
- `使用默认配置 / Use default config`：只让当前请求走默认配置，不会清除已经持久化的 active profile。

启动或刷新后的选择顺序是：

1. 有效的 active Profile；
2. 最新的可用于推理的 Profile；
3. `使用默认配置 / Use default config`。

> [!NOTE]
> `default_zh` 是系统内置基础 Profile，不建议直接修改。需要保存自己的角色时，应创建新的 Voice Profile。

---

## 5. 准备推理输入

三个输入都属于正式必填项：

| 输入 | 是否必填 | 关键规则 |
| --- | --- | --- |
| `参考音频 / Prompt WAV` | 是 | 必须为 3–10 秒 |
| `参考音频文本 / Prompt Text` | 是 | 与参考音频实际内容一致 |
| `目标文本 / Target Text` | 是 | 当前用户版按中文路径生成 |

### 5.1 参考音频：3–10 秒是硬限制

正式参考音频提取路径要求：

```text
3.0 秒 ≤ 参考音频时长 ≤ 10.0 秒
```

小于 3 秒或大于 10 秒都会被拒绝，3.0 秒和 10.0 秒边界本身允许。

> [!IMPORTANT]
> 3–10 秒是运行时硬校验，不只是质量建议。

为了获得更稳定的结果，建议使用单说话人、发音清晰、低噪声、不过度混响且没有过长静音的参考音频。即使使用已经完成 Few-shot Training 的 Voice Profile，正常推理仍然需要参考音频和参考文本；它们属于本次请求的 prompt，不是训练数据的替代物。

### 5.2 参考文本

为了获得更稳定的生成效果，参考文本应尽量准确对应参考音频实际说出的内容。文本越准确，参考语义与文本前端的对齐通常越可靠；空文本会在启动后端前直接被拒绝。

### 5.3 目标文本

空目标文本会在启动后端前被拒绝。当前版本以中文生成为正式支持范围。

当前 UI 没有设置明确的目标文本字符数硬上限。根据项目开发测试经验，较长文本更容易出现漏字、重复、语速异常或发音混乱等问题，因此建议单次生成尽量控制在**约 80–100 个汉字以内**；更长内容可分段生成。这是使用建议，不是硬限制。

---

## 6. 生成语音与查看结果

第一次生成建议保持 Advanced Options 默认，并让三个质量检测开关保持 Off，然后点击 `生成语音`。

点击后，按钮会暂时变为 `生成中...`，结果区显示 `正在生成`。当前后端是同步 subprocess，因此不会伪造 Stage1 / Stage2 百分比进度；正常状态就是：

```text
正在生成
  ↓
最终结果
```

v1.0.0 Inference UI 没有专用 Stop / Cancel 按钮，正常情况下需要等待本次同步推理返回结果或触发 timeout。

### 6.1 成功结果

成功状态显示 `生成完成`，并提示：

```text
语音已生成，可以直接在上方播放器试听。
```

结果区包括 `生成音频 / Generated Audio` 播放器、当前声音角色、总耗时、RTF，以及已开启的质量指标。

一次基础验收可以按以下顺序进行：

1. 状态是否为 `生成完成`；
2. 播放器是否有可播放结果；
3. 实际试听是否存在爆音、截断、异常静音或明显内容错误；
4. 再查看总耗时和 RTF；
5. 需要量化比较时再开启质量检测。

成功并不只看 subprocess exit code。正式 adapter 还要求标准化后的 `inference_output.wav` 实际存在；若进程返回 0 但没有找到输出 WAV，结果仍会记为 failed。

### 6.2 RTF 与总耗时

RTF 可以理解为：

```text
耗时 / 生成音频时长
```

通常 `RTF < 1` 表示快于实时，`RTF > 1` 表示慢于实时。例如在同一种 RTF 定义下，`RTF = 0.5` 可理解为生成 10 秒音频约需要 5 秒。

UI 中展示的 RTF 通常按**完整请求耗时**计算。开启 DNSMOS、Speaker Similarity 或 WER / CER 后，指标耗时也会计入总耗时与 RTF 的计算。

> [!NOTE]
> 比较“纯生成速度”时应保持质量检测开关一致，否则两次 UI RTF 不属于同一计算条件。首次推理也可能因模型初始化和缓存准备而更慢，这属于项目运行经验，不是固定代码保证。

---

## 7. 输出目录与结果文件

如果 `输出目录 / Output Directory` 留空，默认写入：

```text
outputs/inference_runs/
```

系统会为每次请求创建独立目录，目录名使用微秒级时间戳 + `profile_name`。

主要用户输出：

| 文件 | 用途 |
| --- | --- |
| `inference_output.wav` | 主要生成音频 |
| `inference_output_peaknorm.wav` | 峰值归一化版本（存在时额外保存） |
| `inference_summary.json` | 本次生成摘要 |
| `inference_metrics.json` | 启用并产生质量指标时的结果 |

WebUI 播放器优先使用 `inference_output.wav`，主输出不可用时才回退到峰值归一化输出。

任务目录中还可能包含 prompt_reference* 参考音频副本和请求 / 诊断文件。这些文件主要用于记录与排查，具体数量和后缀不属于稳定用户接口。

> [!WARNING]
> 如果手工指定自定义输出目录，系统不会自动再附加默认的“时间戳 + profile_name”隔离层。多个任务写进同一目录可能覆盖固定文件名或混合诊断产物。没有明确目录管理需求时，建议保持输出目录为空。

---

## 8. 高级选项

`高级选项 / Advanced Options` 第一次使用通常无需修改：

| 控件 | 默认 / 初始值 | 主要用途 |
| --- | --- | --- |
| 输出目录 | 留空 | 指定结果目录 |
| Stage1 checkpoint override | 留空 | 临时比较其它 Stage1 checkpoint |
| Stage2 checkpoint override | 留空 | 临时比较其它 Stage2 checkpoint |
| `设备 / Device` | `cuda` | 选择主 TTS 请求设备 |
| `固定随机种子 / Fix seed` | Off | 做更可控的参数比较 |
| Stage1 sampling | 默认值 | 调整 Stage1 采样 |
| Stage2 `length_scale` | `1.0` | 调整 Stage2 长度 / 时长相关行为 |
| `推理超时秒数 / Timeout Seconds` | `600` | 控制 subprocess timeout |

### 8.1 checkpoint override

优先级是：

```text
手动 checkpoint override
  >
当前选择的 Voice Profile
  >
用户版默认配置
```

但自定义 Profile 必须先正常解析并满足 `ready_for_inference=True`，override 不能把 `尚未就绪` 的 Profile 强行变成可用角色。

override 只影响当前请求，不会修改 `profile_config.json`、`profile_manifest.json`、Active Best 或 Voice Profile 本身。它主要适合临时模型对比；日常生成更推荐保持 Profile 自己的模型来源完整。

### 8.2 Device、Seed 与 Timeout

`设备 / Device` 支持 `cuda` 和 `cpu`，默认请求 `cuda`。当前模型主要按 GPU 使用场景设计，主动选择 CPU 时通常应预期明显更高的推理耗时。如果请求 CUDA 但实际运行时 `torch.cuda.is_available()` 为 False，正式推理脚本会发出警告并回退到 CPU，因此 UI Device 是**请求设备**，实际执行设备应结合终端或诊断信息确认。质量指标使用各自后端设备，其中 WER / CER 的 faster-whisper 固定走 CPU / int8。

`固定随机种子 / Fix seed` 默认关闭，此时 seed 为 `None`；开启后使用 `Seed value`，无效或负数会归一化为 0。固定 seed 有助于减少参数比较中的随机变量，但不保证所有环境下 bit-exact 输出。

`推理超时秒数 / Timeout Seconds` 默认 600 秒；小于等于 0 表示不设置显式 subprocess timeout。发生 timeout 时，adapter 会终止 subprocess，并正式记录：

```text
status = failed
error_type = DeveloperInferenceTimeoutError
```

同时保留能够捕获到的 stdout / stderr 尾部供诊断。

### 8.3 Stage1 sampling

正式 UI 提供：

| 参数 | 默认值 | UI 范围 | 用户层理解 |
| --- | ---: | --- | --- |
| `temperature` | `1.0` | `0.1–2.0` | 控制采样随机性强弱 |
| `top_p` | `1.0` | `0.1–1.0` | 按累计概率限制候选范围 |
| `top_k` | `15` | `1–100` | 按候选数量限制采样范围 |

这些参数存在联合作用，不存在一个对所有声音都“最佳”的固定组合。做参数比较时，建议固定 Voice Profile、参考音频 / 文本、目标文本和 seed，每次只改一个主要变量。

### 8.4 Stage2 `length_scale`

默认值为 `1.0`，UI 范围为 `0.5–2.0`。它可以理解为 Stage2 目标长度 / 时长相关控制，不是音量或音高参数。第一次使用保持 1.0 即可；出现明显节奏或时长异常时，应先恢复默认值再继续排查。

### 8.5 用户设置与内部参数边界

正式 UI 中显示的控件，就是 v1.0.0 面向用户开放的可调设置。其它内部参数由发行版管理，正常使用无需手工修改。

---

## 9. 可选质量检测

正式 UI 提供：

- `DNSMOS 语音质量评价`；
- `Speaker Similarity 声音相似度`；
- `WER / CER 文本准确度`。

三个开关默认全部 Off。质量检测不是生成语音的前提；第一次确认主链是否正常时，建议保持关闭，之后再按需要开启。

| 指标 | 主要回答的问题 | 常见方向 |
| --- | --- | --- |
| DNSMOS OVRL | 整体自然度 / 综合质量 | 通常越高越好 |
| DNSMOS SIG | 语音主体信号质量 | 通常越高越好 |
| Speaker Sim | 与参考说话人的音色是否接近 | 通常越高越接近 |
| WER / CER | 生成内容是否能被正确识别 | 越低越好 |

这些指标评价不同维度，不能用单一指标替代实际试听或其它指标。

### 9.1 DNSMOS 与 Speaker Similarity

DNSMOS 结果可以显示 `DNSMOS OVRL` 和 `DNSMOS SIG`。本指南不定义跨语料、跨声音都通用的固定合格线；它更适合同一测试条件下的相对比较。

Speaker Similarity 结果显示 `Speaker Sim`，数值越高通常表示生成语音与参考说话人的嵌入表示更接近，但不代表发音准确度、节奏、情绪或噪声也一定更好。

### 9.2 WER / CER

当前用户版使用本地 faster-whisper：

```text
model = pretrained_models/faster_whisper_medium
device = cpu
compute_type = int8
```

这条 ASR 路径独立于主 TTS 的 Device 选择。CPU / int8 是当前 Windows User Edition 的正式配置，用于避免该验收路径额外依赖另一套 CUDA 运行库。中文场景中 CER 是字符级错误率，而当前 WER 实现包含字符代理逻辑，不应简单按英文单词级 WER 理解。

如果 WER / CER 很高但实际试听基本正确，应同时检查 ASR 识别文本，因为最终误差同时受到“生成内容”和“验收 ASR”影响。

### 9.3 单项指标失败与主任务失败

质量评估器会分别捕获多数单项后端错误，因此可能出现：

```text
主音频已经成功
+ 某个指标显示“未计算”或“计算失败”
```

但这不代表所有 metrics 异常都一定与主任务隔离。如果初始化阶段或其它未捕获异常导致整个推理 subprocess 非零退出，本次结果仍会记为 `生成失败`。

因此排查时应先判断是“主音频成功、单项指标失败”，还是“整个推理任务在 metrics 阶段失败”。

---

## 10. 诊断与常见问题

`诊断信息` 默认折叠，主要用于故障排查。它可能包含 Profile Registry JSON、Inference Diagnostics JSON、已解析路径、developer command、stdout / stderr 和后端错误。

正常结果面板会刻意隐藏绝对路径、developer command、原始 traceback 和原始后端错误；需要排查时再展开诊断信息。

> [!WARNING]
> 诊断内容可能包含本机用户名、绝对路径、运行命令或错误文本。把日志发布到公开 issue、论坛或聊天前，请先检查并删除不希望公开的信息。

### 10.1 常见输入与 Profile 错误

| 提示 / 状态 | 常见原因 | 处理 |
| --- | --- | --- |
| `缺少参考音频` | 未提供 Prompt WAV | 添加有效的 3–10 秒参考音频 |
| `缺少参考文本` | Prompt Text 为空 | 填写与参考音频对应的文本 |
| `缺少生成文本` | Target Text 为空 | 填写目标文本 |
| `尚未就绪` | Profile 模型来源缺失或解析失败 | 返回 Training / Voice Profile 检查 |
| `角色信息不可用` | 当前选择已不在 Registry | 刷新角色列表并重新选择 |

前三类输入错误会在启动后端前被拦截，因此无需先检查 CUDA 或 checkpoint。

### 10.2 生成失败或频繁 timeout

如果最终显示 `生成失败`，建议按以下顺序排查：

1. 回到合法 3–10 秒参考音频、准确参考文本和较短目标文本；
2. 使用系统默认配置或一个明确“可用于生成”的 Voice Profile；
3. 清空 checkpoint override，并恢复 Advanced Options 默认值；
4. 关闭三个质量检测；
5. 再测试主生成链；
6. 仍失败时展开 `诊断信息`，检查 stderr、developer command 和本次任务目录中的 `result.json` 等记录。

如果基础生成能在 600 秒内完成，但只在开启某个质量检测后超时，可以单独开启一个指标复测，以区分生成耗时和验收耗时。

更完整的 Inference 排查仍以本节和本文相关章节为准；通用环境、GPU / CUDA 问题请查看 [Compatibility Guide](./compatibility.md)。

---

## 11. 下一步与相关文档

推荐阅读顺序：

```text
Quick Start
  ↓
DataFactory
  ↓
Training
  ↓
Inference
```

相关文档：

- [快速开始](./quick-start.md)
- [DataFactory 使用指南](./data-factory.md)
- [Training 使用指南](./training.md)
- [Compatibility Guide](./compatibility.md)
