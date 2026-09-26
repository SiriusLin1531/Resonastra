# Resonastra DataFactory 使用指南

[项目 README](./README.md) | [快速开始](./quick-start.md) | **中文简体**

适用版本：**Resonastra v1.0.0**

> [!NOTE]
> 本指南针对 **Resonastra v1.0.0** 的正式发行行为编写。后续版本如果调整 DataFactory 界面、ASR 路由、数据目录或训练数据契约，应以对应版本文档为准。

---

## 1. 本指南解决什么问题

Resonastra 的 DataFactory 负责把用户提供的原始语音整理成可以交给 Stage1 / Stage2 Training 使用的数据。

如果你只是使用内置 Zero-shot 模型，不需要经过 DataFactory。只有在准备 Stage1-only、Stage2-only 或 Full Few-shot 时，才需要使用本指南。

完成本指南中与你所选路线对应的步骤后，你应该能够：

- 选择合适的原始音频输入方式；
- 判断音频格式是否属于 v1.0.0 正式输入范围；
- 创建或恢复一个 DataFactory 工作目录；
- 完成音频切分和本地离线 FunASR 识别；
- 检查并确认 ASR 文本；
- 独立生成 Stage1 或 Stage2 训练数据；
- 判断目标路线的数据是否真正就绪；
- 在任务中断、页面刷新或重新启动后继续处理；
- 将同一个工作目录交给 Training 使用。

本指南不展开 Stage1 / Stage2 的训练参数、checkpoint / Active Best 管理、Voice Profile、Inference 参数、GPU / CUDA 兼容性原理或完整故障排查。这些内容会分别进入 Training、Inference、Compatibility 与 Troubleshooting 文档。

---

## 2. DataFactory 在 Resonastra 工作流中的位置

DataFactory 可以理解为 Few-shot 训练之前的数据准备层：

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

它并不负责模型训练本身。DataFactory 的结束条件不是“某个按钮运行完毕”，而是**你当前路线所需要的数据状态已经就绪**。

| 路线 | 是否需要 DataFactory | DataFactory 需要完成的内容 |
| --- | --- | --- |
| Zero-shot | 否 | 无 |
| Stage1-only Few-shot | 是 | Prepare + 文本确认 + Stage1 数据 |
| Stage2-only Few-shot | 是 | Prepare + 文本确认 + Stage2 数据 |
| Full Few-shot | 是 | Prepare + 文本确认 + Stage1 + Stage2 数据 |

> [!IMPORTANT]
> Stage1 与 Stage2 数据可以独立生成。不要把“Stage1 + Stage2 全部完成”误认为所有 Few-shot 路线都必须满足的统一结束条件。

---

## 3. 开始之前

开始 DataFactory 前，请先完成以下基础准备：

1. 完整解压 Resonastra v1.0.0；
2. 运行 `launch_check_env.bat` 并确认环境检查通过；
3. 准备自己的中文语音数据；
4. 确定一个用于区分本次数据的 **说话人 / 角色名**。

### 3.1 v1.0.0 的语言与 ASR 边界

当前正式发行版 DataFactory 只开放：

```text
language = zh
```

对应的识别路径固定使用本地离线 FunASR。

### 3.2 数据质量建议

为了获得更稳定的 Few-shot 数据，建议原始数据尽量满足：

- 单说话人；
- 发音清晰；
- 语速正常；
- 无明显背景噪声；
- 避免大量剪辑异常、静音异常或明显损坏文件；
- 尽量使用高质量源文件，避免反复有损转码。

这些属于**训练质量建议**，不是 DataFactory 的全部强制输入条件。

---

## 4. 启动 DataFactory

从 Resonastra 根目录运行：

```text
launch_data_factory_ui.bat
```

默认端口为：

```text
7861
```

如果 `7861` 已被占用，启动器会自动选择其他可用端口。实际 WebUI 地址以启动终端显示的信息为准。

> [!IMPORTANT]
> `launch_data_factory_ui.bat` 打开的终端是 **WebUI 主启动终端**。使用 DataFactory 期间不要关闭它；关闭后对应 WebUI 会结束。

### 4.1 页面主要区域

当前发行版页面主要分为：

- **基础输入**：原始音频来源、角色名、语言和工作目录；
- **执行流程**：Prepare、人工校对、Stage1、Stage2；
- **设置**：常用设置和高级设置；
- **推荐操作 / 数据状态**：告诉你当前项目下一步更适合做什么；
- **当前任务**：查看正在运行的长任务与 Stage2 子步骤；
- **人工校对状态**：显示校对器是否正在运行；
- **诊断信息**：查看详细状态、路径、步骤与 JSON。

第一次使用时，不需要先展开所有高级设置。建议先使用默认值跑通正常路径。

### 4.2 主启动终端与进度终端

DataFactory 的 **显示终端进度窗口** 默认开启。执行 Prepare、Stage1 或 Stage2 等长任务时，系统可能额外打开一个日志终端。

两类终端的作用不同：

| 窗口 | 用途 | 可以关闭吗 |
| --- | --- | --- |
| WebUI 主启动终端 | 维持 DataFactory WebUI | 使用期间不要关闭 |
| 进度 / 日志终端 | 只读显示长任务日志 | 可以关闭，不会因此停止真实任务 |

---

## 5. 准备原始语音数据

### 5.1 三种原始音频来源

在 **原始音频来源** 中可以选择三种方式。

#### 方式 A：`本地目录路径`

这是大数据集和常规训练最推荐的方式。

填写：

```text
本地原始音频目录
```

DataFactory 会递归扫描该目录及其子目录中的受支持音频文件。

这种方式不会先把你的整套数据复制进浏览器上传缓存，因此更适合十几分钟、数小时或更大的数据集。

#### 方式 B：`上传音频文件`

选择后会出现多文件选择入口。你可以选择一组本地音频。

符合格式要求的上传文件会先复制到当前工作目录：

```text
{work_dir}/00_uploaded_raw_audio/
```

然后 DataFactory 再把这个缓存目录作为原始音频目录继续处理。

该方式更适合小样本、快速测试或少量补充数据。

#### 方式 C：`上传音频文件夹`

选择后可以通过文件夹入口选择一整个本地目录。

其中被识别为受支持音频的文件同样会复制到：

```text
{work_dir}/00_uploaded_raw_audio/
```

之后进入相同的数据准备链。

> [!NOTE]
> 上传模式主要提供使用便利，并不代表大数据集必须经过浏览器上传。对于较大的 Few-shot 数据集，仍优先推荐 `本地目录路径`。

### 5.2 v1.0.0 正式识别的音频格式

当前正式白名单为：

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

目录扫描会按扩展名的小写形式匹配，因此 `.WAV`、`.FLAC` 等大小写形式也可以被识别。

同一个原始音频目录中可以混合多种受支持格式，例如同时放入 WAV、FLAC 和 MP3。进入 DataFactory 后，后续处理会把它们统一到内部标准格式。

> [!IMPORTANT]
> “扩展名在白名单中”只表示 DataFactory 会尝试处理该文件，并不保证任意内部 codec、任意封装变体或损坏文件都一定能够成功解码。

白名单之外的文件不会成为正式 Raw Audio 输入。例如 WEBM、MP4、AIFF/AIF、CAF、AMR 等不属于当前 v1.0.0 的正式入口格式。

### 5.3 推荐使用 WAV / FLAC

从兼容性角度，上面的 8 种格式都可以进入正式流程。

从训练数据质量角度，如果条件允许，更推荐使用质量良好的：

```text
WAV
FLAC
```

MP3、AAC、M4A、OGG、OPUS、WMA 等有损来源也可以使用，但编码时已经丢失的信息不会因为 DataFactory 后面重新写成 WAV 而恢复。

因此：

- 已有的高质量 MP3/AAC 数据不需要为了“格式合规”而先手工转 WAV；
- 如果你拥有原始无损版本，则优先使用原始 WAV / FLAC 更合适。

### 5.4 DataFactory 会自动完成的标准化

你不需要事先把每个原始文件手工统一为单声道、32 kHz 或 PCM16。

当前数据链会在切分阶段把受支持音频读取并标准化：

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

切分后的标准短音频会以 `.wav` 形式进入后续 ASR 和训练数据准备。

### 5.5 上传缓存的覆盖行为

常用设置中的：

```text
覆盖已上传音频缓存
```

默认开启。

它只针对：

```text
{work_dir}/00_uploaded_raw_audio/
```

上传缓存，不会删除你原来的本地音频目录。

如果关闭这个选项，再次上传同名文件时系统会为目标文件生成不冲突的新名称，而不是直接覆盖已有文件。

### 5.6 原始输入常见失败边界

Prepare 前或 Prepare 过程中可能遇到以下情况：

- 本地目录不存在；
- 本地目录中没有任何受支持扩展名的音频；
- 上传选择中没有检测到受支持音频；
- 上传文件复制失败；
- 文件后缀受支持，但内部音频无法解码；
- 个别源文件损坏。

如果错误发生在切分阶段，不要假设系统一定会自动跳过所有坏文件。当前切分链并不是以“每个文件失败后继续全部剩余文件”为默认容错契约；出现解码错误时应先定位异常源文件。

---

## 6. 说话人名称与工作目录

### 6.1 说话人 / 角色名

在 **说话人 / 角色名** 中填写这套数据的名称，例如：

```text
Character_A
```

如果不手动指定工作目录，DataFactory 会根据这个名称自动使用：

```text
user_data/Character_A_factory
```

通用规则是：

```text
user_data/{speaker_name}_factory
```

后续 Training 建议继续使用同一个角色名和同一个工作目录。

### 6.2 自定义工作目录

展开：

```text
已有项目 / 工作目录
```

可以填写：

```text
工作目录（可选）
```

适合以下情况：

- 希望把当前 DataFactory 项目放在自定义位置；
- 页面刷新后恢复旧项目；
- 重新启动 Resonastra 后继续旧项目；
- 明确知道已有工作目录，希望直接扫描状态。

如果留空，DataFactory 会继续使用默认的 `user_data/{speaker_name}_factory`。

### 6.3 载入 / 扫描已有工作目录

填写旧工作目录或角色名后，点击：

```text
载入/扫描工作目录
```

DataFactory 会重新检查已有产物，并刷新：

- Prepare 是否完成；
- 文本确认是否完成；
- Stage1 数据是否完成；
- Stage2 各子步骤是否完成；
- 哪些步骤仍然缺失或处于 partial 状态。

这个按钮是恢复旧项目时最重要的入口之一。

> [!IMPORTANT]
> DataFactory 的恢复模型以 `work_dir` 为中心。页面状态丢失、浏览器刷新或 WebUI 重启，并不等于工作目录里的真实产物消失。

---

## 7. 工作目录结构与关键产物

普通用户不需要记住每个内部文件，但理解主要目录有助于判断状态和排查问题。

### 7.1 基础数据处理目录

一个典型工作目录中会包含：

```text
{speaker_name}_factory/
├── 00_uploaded_raw_audio/   # 仅上传模式使用
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

其中 `01_uvr_*` 与 `02_denoise/` 属于保留的数据目录结构。v1.0.0 当前正式用户流程并未开放 UVR / denoise 作为可启用的 DataFactory 功能，因此不要仅因为这些目录为空就判断任务失败。

### 7.2 `06_export/` 是基础数据出口

这里会出现后续 Stage1 / Stage2 使用的重要文件，例如：

| 文件 / 目录 | 用户层含义 |
| --- | --- |
| `clips/` | 标准化和切分后的训练音频 |
| `manifest.jsonl` | Prepare 后的基础 manifest |
| `dataset.list` | Stage1 数据构建等流程使用的列表 |
| `stage2_manifest.jsonl` | 基础 Stage2 manifest |
| `manifest.corrected.jsonl` | 文本确认后的主 corrected manifest |
| `stage2_manifest.corrected.jsonl` | 校对后的 Stage2 对应 manifest |
| `stage2_manifest.fewshot.jsonl` | Stage2 Few-shot 后处理正式使用的 manifest |

此外还会有 health、prompt selection、conversion 等报告文件，用于检查数据是否符合后续步骤要求。

> [!NOTE]
> Prepare 阶段就会生成初始的 `stage2_manifest.jsonl` 并执行相应 health check，但这**不代表 Stage2 训练数据已经完成**。真正的 Stage2 Few-shot 后处理会在文本确认后从 `manifest.corrected.jsonl` 生成 `stage2_manifest.fewshot.jsonl`，再继续构建 Stage2 `.pt`、cache、train / val 与过滤报告。

### 7.3 Stage1 数据目录

Stage1 的最终数据单独存放在：

```text
07_stage1_ft/
```

典型内容包括：

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

### 7.4 Stage2 compact layout

v1.0.0 的正式工作目录已经把 Stage2 的重产物收拢在同一个 `{speaker_name}_factory` 内：

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

这意味着停止、扫描、续跑和 Training handoff 都可以围绕同一个工作目录进行。

### 7.5 普通用户需要记住的两个层级

**必须知道：**

- 当前 `work_dir`；
- `manifest.corrected.jsonl` 是否已产生；
- Stage1 / Stage2 是否达到当前路线所需 ready 状态。

**排查时有用：**

- `logs/`；
- build / health / split / filter report；
- train / val 与 cache 数量。

---

## 8. 第一步：开始准备数据

完成基础输入后，点击：

```text
开始准备数据
```

### 8.1 第一次使用的推荐输入

如果你还不熟悉 DataFactory，可以先使用：

```text
原始音频来源：本地目录路径
本地原始音频目录：你的原始音频目录
说话人 / 角色名：例如 Character_A
语言：zh
工作目录：留空
```

这样系统会自动使用：

```text
user_data/Character_A_factory
```

### 8.2 Prepare 实际会做什么

用户点击一次 **开始准备数据** 后，背后会完成一组连续步骤：

```text
扫描原始音频
  ↓
建立 raw index
  ↓
解码 / 单声道化 / 32 kHz
  ↓
静音切分
  ↓
本地离线 FunASR
  ↓
生成 manifest.jsonl / dataset.list
  ↓
生成初始 stage2_manifest.jsonl
  ↓
执行基础 manifest / Stage2 manifest health check
```

这些初始 Stage2 产物用于后续校对与 Few-shot 数据链衔接，不等同于最终 Stage2 training data ready。

### 8.3 本地离线 FunASR

Resonastra v1.0.0 的正式 DataFactory 发行路径使用：

```text
language = zh
backend = auto → local FunASR
```

FunASR 使用发布包中预置的本地：

- ASR 模型；
- VAD 模型；
- Punctuation 模型。

运行时不会从 ModelScope 或 Hugging Face 下载这些模型，FunASR 启动更新检查也处于关闭状态。

因此，如果这些本地模型资产缺失，正确处理方式是检查发布包 / 环境，而不是等待程序联网下载。

### 8.4 Prepare 成功条件

完成后，用户状态区应显示：

```text
音频处理与基础识别产物已就绪。
```

工作目录扫描至少应检测到：

```text
06_export/manifest.jsonl
06_export/dataset.list
```

这时可以进入文本确认阶段。

### 8.5 Prepare 失败时先检查状态

如果任务失败或被中止，不建议第一反应就是删除整个工作目录。

先按以下顺序处理：

1. 查看 **当前任务** 和进度终端；
2. 展开 **诊断信息** 查看最近一次操作；
3. 点击 **载入/扫描工作目录**；
4. 确认哪些产物已存在；
5. 根据错误原因决定继续、修复输入或明确覆盖。

---

## 9. 第二步：确认 ASR 文本

ASR 结果会参与后续训练数据构建。音频内容与训练文本明显不一致时，会降低数据质量，因此建议在进入 Stage1 / Stage2 数据生成前完成文本确认。

### 9.1 启动人工校对器

点击：

```text
启动 / 打开人工校对器
```

DataFactory 会启动独立的人工校对器，并在页面中显示运行状态和访问地址。

如果同一个工作目录的校对器已经在运行，发行版会阻止重复启动新的同类实例。

### 9.2 校对时做什么

人工校对器会载入当前工作目录的 `dataset.list`。请在校对器中根据对应音频条目检查 ASR 文本，修正明显错字、漏字或识别错误，并使用校对器自身的保存功能保存修改。

这里的目标不是改写说话内容，而是尽量让训练文本准确对应音频中实际说出的内容。

### 9.3 校对器有独立生命周期

DataFactory 页面中会提供专用：

```text
关闭人工校对器
```

它与：

```text
停止当前任务
```

不是同一个功能。

- `停止当前任务` 负责 DataFactory / Stage1 / Stage2 的构建子进程；
- `关闭人工校对器` 专门负责人工校对器进程。

在需要从保存后的校对结果重建 corrected manifest 时，系统会确保校对器能够安全关闭，以避免仍在运行的校对器和重建操作同时修改同一组数据。

### 9.4 保存后完成校对

确认校对器中已经保存修改后，返回 DataFactory，点击：

```text
已保存并完成校对
```

DataFactory 会根据保存后的校对结果重建 corrected manifest。

### 9.5 如果 ASR 结果无需修改

如果你已经人工检查并确认自动识别文本可以直接使用，可以点击：

```text
无需校对，继续
```

这个操作不是“跳过文本确认阶段”，而是**接受当前识别结果作为后续训练文本**，并生成后续需要的 corrected manifest。

### 9.6 文本确认成功条件

页面应显示：

```text
文本确认结果已就绪。
```

关键产物为：

```text
06_export/manifest.corrected.jsonl
```

正常用户流程中，Stage1 与 Stage2 都应先完成文本确认，但两条数据链读取的文件并不完全相同：

- **Stage1** builder 读取 `06_export/dataset.list`；人工校对器直接编辑这份列表，而“无需校对，继续”表示接受当前列表内容。
- **Stage2** 后处理明确要求 `06_export/manifest.corrected.jsonl`，没有 corrected manifest 时不会启动正式 Stage2 后处理。

---

## 10. 第三步 A：生成 Stage1 训练数据

如果你的路线是：

- Stage1-only Few-shot；
- Full Few-shot；

需要生成 Stage1 数据。

点击：

```text
生成 Stage1 训练数据
```

### 10.1 Stage1 数据从哪里来

当前用户路径主要使用：

```text
06_export/dataset.list
06_export/clips/
```

并在当前工作目录中构建 Stage1 frontend / semantic cache 与 train / val manifest。

### 10.2 默认 Stage1 切分

常用设置中的 Stage1 训练集比例默认：

```text
train_ratio = 0.90
```

验证集比例自动使用：

```text
1 - train_ratio
```

Stage1 split seed 不要求普通用户手工填写；builder 会自动生成并记录在：

```text
07_stage1_ft/metadata/split.json
```

默认处理全部输入样本。

### 10.3 Stage1 覆盖保护

如果当前工作目录已经存在 Stage1 训练数据或 cache，而你没有显式要求覆盖，发行版会阻止意外启动新的覆盖式 Stage1 构建。

> [!WARNING]
> `覆盖已有 Stage1 cache` 是显式重建操作。只有在你确定希望重新构建当前 Stage1 数据时才启用。

Stage1 与 Stage2 的恢复方式不同：**Stage1 没有独立的 artifact-based Resume 按钮**。只要 `07_stage1_ft/` 下已经存在任何文件——包括失败或中断留下的 partial output——再次点击“生成 Stage1 训练数据”而未授权覆盖时，发行版会阻止 builder 启动。

因此，如果 Stage1 已经完整 ready，只需保留现有结果；如果 Stage1 处于 partial / failed 状态且你决定重建，则先确认工作目录和日志，再显式勾选 `覆盖已有 Stage1 cache` 后重新生成。

启用覆盖后，发行版会在成功构建之后清理新 manifest 不再引用的旧 cache，并执行 strict cache contract 检查，避免 train / val manifest 与 frontend / semantic cache 不一致。

### 10.4 Stage1 成功条件

页面应显示：

```text
Stage1 训练数据已完成。
```

Stage1 ready 状态并不只看 `07_stage1_ft/` 文件夹是否存在，而会综合检查：

- `metadata/build_report.json` 的构建状态；
- `train_manifest.jsonl`；
- `val_manifest.jsonl`；
- `frontend_cache/`；
- `semantic_cache/`。

主要输出目录：

```text
{work_dir}/07_stage1_ft/
```

如果你的路线是 Stage1-only，到这里 DataFactory 对 Stage1 的任务已经完成，可以进入 Training。

---

## 11. 第三步 B：生成 Stage2 训练数据

如果你的路线是：

- Stage2-only Few-shot；
- Full Few-shot；

需要生成 Stage2 数据。

点击：

```text
生成 Stage2 训练数据
```

### 11.1 Stage2 数据生成不是单一步骤

一次按钮操作内部包含多个连续阶段：

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

因此 Stage2 数据准备通常明显比 Stage1 更重，也更需要 Live Monitor 和断点恢复。

### 11.2 Stage2 Few-shot manifest

第一阶段会从：

```text
06_export/manifest.corrected.jsonl
```

导出：

```text
06_export/stage2_manifest.fewshot.jsonl
```

这个步骤同时会处理 Stage2 所需的 prompt selection。

### 11.3 Prompt selection 默认规则

当前默认值为：

| 参数 | 默认值 | 用户含义 |
| --- | ---: | --- |
| `prompt_mode` | `speaker_pool` | 从说话人数据池中选择 prompt |
| `min_prompt_sec` | `3.0` | prompt 最短时长 |
| `max_prompt_sec` | `10.0` | prompt 最长时长 |
| `prefer_prompt_sec` | `6.0` | 优先选择附近时长的候选 |
| `allow_self_prompt` | `True` | 允许在需要时使用样本自身作为 prompt |

> [!NOTE]
> 这里的 3–10 秒属于 **Stage2 数据构建时的 prompt-selection 规则**。它与 Inference 页面要求用户上传 3–10 秒 Prompt WAV 是两个不同的操作场景，只是当前版本使用了相同的时长边界。

### 11.4 Stage2 .pt 与后续 cache

Stage2 后续会依次生成：

```text
07_stage2_pt_all/
08_continuous_semantic/
09_style_f0_spk/
10_train_val/
11_filter/
```

Live Monitor 会把多个子步骤同时展示出来。处理过程中出现 partial 状态并不一定表示失败，它可能只是说明当前已有部分样本完成、后续样本仍在继续。

当前 v1.0.0 用户发行 UI 还固定启用了 Stage2 异常样本自动隔离。最终过滤步骤会以 `move` 模式处理命中的异常样本，将其从 train / val 目录移入：

```text
11_filter/rejected/
```

并同时生成 JSON / CSV 过滤报告。因此过滤完成后，train / val 中实际可训练样本数少于过滤前的 source count 可能是正常现象。

### 11.5 Stage2 成功条件

Quick Start 中对普通用户显示的最终摘要是：

```text
Stage2 训练数据已完成。
```

详细状态层面，完整 Stage2 ready 通常需要以下部分全部就绪：

- Stage2 few-shot manifest；
- Stage2 `.pt`；
- continuous semantic cache；
- Style / F0 / Speaker cache；
- train / val split；
- 有效且与当前 split / 文件数量一致的 bad-sample filter report。

如果你的路线是 Stage2-only，到这里可以进入 Training；不需要额外生成 Stage1 数据。

---

## 12. 继续生成 Stage2 数据与断点恢复

Stage2 是长链任务，因此 Resonastra 提供：

```text
继续生成 Stage2 训练数据
```

### 12.1 什么时候使用

适合以下情况：

- Stage2 中途手动停止；
- WebUI 被关闭后重新启动；
- 浏览器页面刷新；
- 某一步失败后已经修复问题；
- 工作目录中已经存在部分 `.pt` 或 cache；
- 希望从旧项目继续完成剩余 Stage2 产物。

### 12.2 artifact-based auto resume

继续模式会根据真实文件产物判断是否需要运行某一步。

核心规则是：

- 如果某一步关键产物已经完整存在，会自动跳过；
- 如果 `.pt` / cache 只完成了一部分，会继续尝试补齐缺失样本；
- 如果你显式启用了对应 overwrite，该步骤仍然会重跑；
- split / filter 报告如果过期、数量不一致或与当前文件系统状态不一致，不会被视为 ready；
- 如果某一步真正失败，后续链会停止，不会把失败伪装成成功。

例如，如果 `stage2_manifest.fewshot.jsonl` 已经存在且不早于 corrected manifest，Resume 可以直接跳过重新导出 manifest；如果 Stage2 `.pt` 数量已经达到预期样本数并且没有请求 overwrite，也会跳过该步骤。

### 12.3 推荐的恢复步骤

```text
重新启动 DataFactory
  ↓
填写原 speaker_name 或原 work_dir
  ↓
点击“载入/扫描工作目录”
  ↓
确认已有产物和 partial 状态
  ↓
点击“继续生成 Stage2 训练数据”
```

> [!IMPORTANT]
> 中断后优先 **扫描状态 + 继续生成**，不要把“从头重跑并开启所有 overwrite”当成默认恢复方式。

---

## 13. 停止任务、刷新状态与 Live Monitor

### 13.1 停止当前任务

在：

```text
已有项目 / 工作目录
```

区域可以点击：

```text
停止当前任务
```

系统会查找与当前工作目录匹配的 DataFactory 构建子进程并尝试停止。

停止后会重新扫描工作目录，因为正在执行的步骤可能已经留下部分产物。

因此正确的后续动作通常是：

```text
停止当前任务
  ↓
载入/扫描工作目录
  ↓
查看真实剩余状态
  ↓
决定 Resume / 修复 / 显式覆盖
```

`停止当前任务` 不负责关闭人工校对器；人工校对器使用自己的 **关闭人工校对器**。

### 13.2 当前任务与 Live Monitor

页面右侧 **当前任务** 会显示当前长任务信息，并可通过：

```text
立即刷新
```

主动刷新。

长任务期间系统还会按固定间隔自动刷新状态。

Stage2 运行时可以看到各个子步骤的实时状态，而不需要等整个 callback 结束后才知道最终结果。

### 13.3 如何理解状态

顶部四阶段卡片使用的是用户级状态：

| 用户看到的状态 | 含义 |
| --- | --- |
| 尚未开始 | 当前阶段还没有有效进度 |
| 等待前置步骤 | 上游步骤尚未完成 |
| 等待确认 | 音频处理已经完成，正在等待文本确认 |
| 部分完成 | 已有部分有效产物，但尚未完整 |
| 已完成 | 当前阶段通过对应状态检查 |

展开诊断信息时，内部统一状态可能显示 `missing / blocked / partial / done`。Training handoff 最重要的是顶部独立的 **Stage1 / Stage2 readiness**：目标阶段显示 **可进入训练** 时，才应把它视为该路线的数据准备完成。

不要只根据一个文件夹是否存在判断成功。

### 13.4 诊断信息

普通用户正常流程通常不需要展开 **诊断信息**。遇到异常时，它可以提供：

- 最近一次操作；
- 工作目录状态；
- 输出摘要；
- 步骤摘要；
- 当前工作目录；
- 人工校对器诊断；
- DataFactory Result JSON；
- DataFactory Live Monitor JSON。

这些内容主要用于定位失败步骤和确认真实路径。

---

## 14. 按训练路线完成 DataFactory

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

完成检查：

- [ ] Prepare 已完成；
- [ ] `manifest.corrected.jsonl` 已就绪；
- [ ] Stage1 训练数据已完成；
- [ ] `07_stage1_ft/` 的 build report / manifests / caches 通过状态扫描。

不需要生成 Stage2 数据。

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

完成检查：

- [ ] Prepare 已完成；
- [ ] `manifest.corrected.jsonl` 已就绪；
- [ ] Stage2 few-shot manifest 已完成；
- [ ] Stage2 `.pt` 已完成；
- [ ] continuous / style cache 已完成；
- [ ] train / val 已完成；
- [ ] filter report 已完成。

不需要生成 Stage1 数据。

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

最终推荐摘要应显示：

```text
Stage1 与 Stage2 数据均已完成。
```

完成检查：

- [ ] Prepare 已完成；
- [ ] 文本确认已完成；
- [ ] Stage1 数据 ready；
- [ ] Stage2 数据 ready。

---

## 15. 设置说明

第一次跑通 DataFactory 时，建议优先使用默认值。只有在明确知道要解决什么问题时，才调整高级设置。

### 15.1 常用设置

| 设置 | 默认值 | 作用 |
| --- | ---: | --- |
| 覆盖已有数据工厂工作目录 | 关闭 | 允许 Prepare 对已有工作目录执行覆盖式操作 |
| 覆盖已上传音频缓存 | 开启 | 重新上传时先清理 `00_uploaded_raw_audio/` |
| 单步超时秒数，0 表示不限制 | `0` | v1.0.0 各执行路径对 `0` 的内部处理并不完全一致，见下方说明 |
| Stage1 训练集比例 | `0.90` | Stage1 train / val 切分比例 |
| Stage2 训练集比例 | `0.90` | Stage2 train / val 切分比例 |
| 显示终端进度窗口 | 开启 | 长任务时额外显示只读日志终端 |

> [!NOTE]
> v1.0.0 的界面标签写作“0 表示不限制”，但实际执行路径存在一个 release-specific 差异：Stage1 / Stage2 会把 `0` 归一化为“未显式设置超时”，而 Prepare 会回落到内部 `7200` 秒（2 小时）默认值。如果你的 Prepare 预计可能超过 2 小时，请显式填写更大的正数秒数，而不要依赖 `0` 获得无限时长。

> [!WARNING]
> 不要在不了解已有工作目录内容时同时开启多个 overwrite 选项。正常恢复旧项目首先应使用 `载入/扫描工作目录`；Stage2 使用 Resume，Stage1 partial output 则按第 10.3 节的覆盖保护处理。

### 15.2 ASR 设置

发行版界面保留 ASR 参数区域，但 v1.0.0 正式用户路径实际锁定为：

```text
ASR backend: auto
ASR model size: large
ASR precision: float32
language: zh
```

`auto` 会进入本地 FunASR。

其中 `ASR 后端`、`ASR 模型大小` 与 `ASR 精度` 在最终 v1.0.0 release shell 中被锁定为不可编辑；`语言` 只有 `zh` 一个可选值。它们不是普通用户的自由切换入口。

### 15.3 音频切分参数

默认值：

| 参数 | 默认值 | 影响 |
| --- | ---: | --- |
| `threshold` | `-34` | 静音切分判定阈值 |
| `min_length` | `4000` | 控制切分后片段的最小长度倾向 |
| `min_interval` | `300` | 控制可用于切分的静音间隔 |
| `hop_size` | `10` | 切分检测步长 |
| `max_sil_kept` | `500` | 切分时保留静音的上限设置 |
| `normalize_max` | `0.9` | 切片归一化目标幅度 |
| `alpha_mix` | `0.25` | 原始波形与归一化结果的混合程度 |

这组参数沿用 GPT-SoVITS 风格的静音切分逻辑。第一次使用不建议修改；只有当默认切片明显过长、过碎或静音保留不符合你的数据特征时，再考虑调整。

### 15.4 Prompt 设置

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `prompt_mode` | `speaker_pool` | Stage2 prompt 选择方式 |
| `allow_self_prompt` | 开启 | 允许使用样本自身作为 prompt |
| `min_prompt_sec` | `3.0` | prompt 最短时长 |
| `max_prompt_sec` | `10.0` | prompt 最长时长 |
| `prefer_prompt_sec` | `6.0` | 候选 prompt 的优选时长 |

除非你正在针对 Stage2 数据设计做实验，否则保持默认即可。

### 15.5 Stage1 参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| Stage1 device | `cuda` | 使用 GPU 构建 Stage1 数据 |
| Stage1 use_half | 开启 | 允许对应模型流程使用半精度 |
| 覆盖已有 Stage1 cache | 关闭 | 显式重新构建已有 Stage1 cache |
| 构建后验证 Stage1 dataset | 开启 | 构建完成后检查数据契约 |

如果显卡或环境无法运行当前 GPU 路径，应优先参考 Compatibility / Troubleshooting，而不是随意修改其他参数组合。

### 15.6 Stage2 / Cache 参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| device | `cuda` | Stage2 数据处理设备 |
| Stage2 use_half | 关闭 | Stage2 `.pt` 预处理精度选择 |
| 覆盖已有 Stage2 `.pt` | 关闭 | 强制重建 Stage2 `.pt` |
| 覆盖已有 continuous cache | 关闭 | 强制重建 continuous cache |
| 覆盖已有 style cache | 关闭 | 强制重建 style cache |
| 覆盖已有 train/val | 关闭 | 强制重新切分 Stage2 train / val |
| continuous dtype | `float32` | continuous semantic cache 数据类型 |

Resume 的默认设计是尽量复用已经完成的产物，因此这些 overwrite 选项正常情况下保持关闭。

### 15.7 Mel / F0 参数

默认值：

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

这些参数直接影响 Stage2 `.pt`、mel / F0 等特征构建，应视为高级设置。

> [!IMPORTANT]
> 普通 Few-shot 用户不建议为了“试试效果”随意修改这组特征参数。它们与后续 Stage2 训练数据契约有关。

### 15.8 人工校对设置

| 设置 | 默认值 | 说明 |
| --- | ---: | --- |
| 校对器端口 | `9871` | 人工校对 WebUI 端口 |
| `g_batch` | `10` | 校对器批处理设置 |
| 覆盖已有校对备份 | 关闭 | 是否覆盖既有校对备份 |
| corrected manifest 时长容差 | `0.05` | 重建 corrected manifest 时的时长匹配容差 |
| 重建超时秒数 | `600` | 校对后重建步骤超时设置 |

一般用户只需要在默认校对器端口发生占用时考虑修改端口。

---

## 16. 输出、日志与 Training handoff

### 16.1 如何确认 DataFactory 真正完成

不要把“最后一个按钮没有报错”作为唯一判断。

应该回到你的路线：

- Stage1-only：Stage1 ready 即可；
- Stage2-only：Stage2 ready 即可；
- Full Few-shot：Stage1 与 Stage2 都 ready。

如果不确定，点击：

```text
载入/扫描工作目录
```

查看统一状态与推荐下一步。

### 16.2 Training 使用同一个项目身份

进入 Training 时继续使用同一个：

```text
说话人 / 角色名
```

如果 DataFactory 使用默认工作目录规则，Training 也可以按该角色名找到：

```text
user_data/{speaker_name}_factory
```

如果 DataFactory 使用了自定义 `work_dir`，Training 应填写同一个目录。

DataFactory 只负责把路线需要的数据准备到可训练状态；epochs、batch size、checkpoint、Active Best 等留给 Training Guide。

### 16.3 日志

DataFactory 的主要运行日志集中在：

```text
{work_dir}/logs/
```

Stage1 还会记录独立的构建 stdout / stderr 与 command 信息。出现失败时，优先从页面诊断信息定位失败步骤，再查看对应日志，而不是一次性检查整个项目所有文件。

---

## 17. 常见边界与安全操作

### 17.1 支持格式不等于所有文件都能解码

扩展名白名单只代表入口接受范围。内部 codec、文件完整性和发布包内解码能力仍然会影响实际读取。

### 17.2 不要把有损文件转 WAV 当作“恢复质量”

MP3 / AAC 等文件转换成 WAV 后只是换成无损容器保存当前结果，并不会找回此前压缩丢失的音频信息。

### 17.3 不要默认用 overwrite 解决所有问题

工作目录的设计支持扫描和恢复。发生中断时先判断已有产物，再决定下一步：

- Stage2 优先使用 `继续生成 Stage2 训练数据` 的 artifact-based Resume；
- Stage1 没有同等的 Resume；partial Stage1 需要确认后显式授权 `覆盖已有 Stage1 cache` 才能重建。

### 17.4 不要关闭 WebUI 主启动终端

主启动终端负责 WebUI 进程；只读进度终端则可以按需关闭。

### 17.5 人工校对器使用专用关闭按钮

`停止当前任务` 与 `关闭人工校对器` 有不同进程职责。

### 17.6 Partial Few-shot 是合法路线

Stage1-only 和 Stage2-only 都是当前正式支持的组合。DataFactory 不要求一定同时把 Stage1 与 Stage2 全部做完。

### 17.7 `zh + local FunASR` 是 v1.0.0 限制

不要把当前发行版的中文-only DataFactory 约束理解为 Resonastra 未来版本的永久能力边界。

---

## 18. 下一步与相关文档

完成 DataFactory 后，下一步是进入 Training。

当前推荐阅读顺序：

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
- Training Guide（尚未发布）
- Compatibility Guide（尚未发布）
- Troubleshooting Guide（尚未发布）
