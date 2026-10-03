# Resonastra 数据准备指南（DataFactory）

[项目 README](./README.md) | [快速开始](./quick-start.md) | **数据准备** | [模型训练](./training.md) | [语音生成](./inference.md) | [兼容性](./compatibility.md) | [版本校验](./release-verification.md) | [English](../en/data-factory.md) | **中文简体**

适用版本：**Resonastra v1.0.0**

> [!NOTE]
> 本指南针对 **Resonastra v1.0.0** 的正式发行行为编写。后续版本如果调整 DataFactory 界面、ASR 路由、数据目录或训练数据契约，应以对应版本文档为准。

---

## 1. 本指南解决什么问题

Resonastra 的 DataFactory 负责把用户提供的原始语音整理成可以交给 Stage1 / Stage2 训练使用的数据。

如果你只使用内置 Zero-shot 模型，不需要经过 DataFactory。只有 Stage1-only、Stage2-only 或 Full Few-shot 路线需要进入这里。

完成与你所选路线对应的步骤后，你应该能够：

- 选择并导入受支持的原始音频；
- 建立或恢复一个 DataFactory 工作目录；
- 完成音频标准化、切分和本地离线 FunASR；
- 确认训练文本；
- 独立生成 Stage1 或 Stage2 训练数据；
- 在中断后按正确方式恢复；
- 判断当前路线是否已经可以进入 Training。

本指南不展开 Training 参数、模型检查点 / Active Best、Voice Profile、Inference 参数或通用 GPU / CUDA 兼容性政策。DataFactory 自身的恢复与诊断规则在本文对应章节说明。

---

## 2. DataFactory 工作流与开始之前

DataFactory 是 Few-shot 训练之前的数据准备层：

```text
原始音频
  ↓
标准化 / 切分
  ↓
本地离线 FunASR
  ↓
文本确认
  ↓
校对后的数据出口
  ├─→ Stage1 训练数据
  └─→ Stage2 训练数据
          ↓
       Training
```

DataFactory 的完成条件不是“最后一个按钮执行过”，而是**当前路线真正需要的数据已经就绪**。

| 路线 | 是否需要 DataFactory | 结束条件 |
| --- | --- | --- |
| Zero-shot | 否 | 不需要 DataFactory |
| Stage1-only Few-shot | 是 | 文本确认 + Stage1 已就绪 |
| Stage2-only Few-shot | 是 | 文本确认 + Stage2 已就绪 |
| Full Few-shot | 是 | Stage1 已就绪 + Stage2 已就绪 |

> [!IMPORTANT]
> Stage1 与 Stage2 数据可以独立生成。Stage1-only / Stage2-only 都是合法完成状态，不要求所有 Few-shot 路线都同时完成两个 Stage。

### 2.1 开始前准备

建议先完成：

1. 完整解压 Resonastra v1.0.0；
2. 运行 `launch_check_env.bat` 并确认基础环境检查通过；
3. 准备中文语音数据；
4. 确定一个用于区分本项目的 **说话人 / 角色名**。

当前 v1.0.0 DataFactory 正式用户路径为：

```text
language = zh
ASR = 本地离线 FunASR
```

为了获得更稳定的 Few-shot 数据，建议使用单说话人、发音清晰、语速正常、无明显背景噪声的数据，并尽量避免损坏文件、异常静音或反复有损转码。这些属于质量建议，不是所有输入的硬性校验条件。

---

## 3. 启动 DataFactory

从 Resonastra 根目录运行：

```text
launch_data_factory_ui.bat
```

默认端口为 `7861`。如果端口已占用，启动器会自动选择其他可用端口，实际地址以启动终端显示为准。

> [!IMPORTANT]
> 主启动终端负责维持 DataFactory WebUI，使用期间不要关闭。

页面主要包括：

- **基础输入**：原始音频来源、说话人、语言、工作目录；
- **执行流程**：数据准备、人工校对、Stage1、Stage2；
- **设置**；
- **推荐操作 / 数据状态**；
- **当前任务**；
- **人工校对状态**；
- **诊断信息**。

第一次使用建议保持默认设置，先跑通标准流程。

### 3.1 主启动终端与进度终端

执行长任务时，如果 **显示终端进度窗口** 开启，系统可能额外打开日志终端。

| 窗口 | 用途 | 是否可以关闭 |
| --- | --- | --- |
| WebUI 主启动终端 | 维持 DataFactory WebUI | 使用期间不要关闭 |
| 进度 / 日志终端 | 只读显示长任务日志 | 可以关闭，不会因此停止真实任务 |

---

## 4. 原始音频与工作目录

### 4.1 三种原始音频来源

**本地目录路径**适合常规或较大的数据集。DataFactory 会递归扫描目录和子目录中的受支持音频。

**上传音频文件**适合少量文件或快速测试。

**上传音频文件夹**适合从文件夹入口选择一组本地音频。

上传模式中，被接受的音频会先复制到：

```text
{work_dir}/00_uploaded_raw_audio/
```

再以该缓存目录作为原始音频输入。常用设置中的 `覆盖已上传音频缓存` 默认开启；它只影响这个上传缓存，不会删除原始本地目录。关闭时，再次上传同名文件会生成不冲突的新名称，而不是直接覆盖已有缓存文件。

> [!NOTE]
> 上传模式主要提供便利。较大的 Few-shot 数据集仍优先推荐 **本地目录路径**。

### 4.2 支持格式与质量边界

v1.0.0 正式识别：

```text
.wav  .mp3  .flac  .ogg
.m4a  .aac  .wma   .opus
```

扩展名匹配不区分大小写，同一目录可以混合多种受支持格式。

> [!IMPORTANT]
> “扩展名受支持”只表示 DataFactory 会尝试处理，并不保证任意编码格式（codec）、封装变体或损坏文件都一定能够成功解码。

WEBM、MP4、AIFF/AIF、CAF、AMR 等不属于当前正式入口格式。

如果有条件，更推荐质量良好的 WAV / FLAC。有损音频也可以使用，但把 MP3 / AAC 等转成 WAV 不会恢复已经丢失的信息。

### 4.3 自动标准化与输入失败边界

你不需要先手工把原始音频统一为单声道、32 kHz 或 PCM16。当前切分链会自动完成：

```text
受支持原始音频
  ↓
解码
  ↓
单声道化
  ↓
32 kHz
  ↓
静音切分
  ↓
PCM 16-bit WAV 片段
```

如果源目录不存在、没有受支持音频、上传复制失败、音频损坏或内部编码无法解码，数据准备可能失败。当前切分链不承诺“单个坏文件失败后一定继续处理所有剩余文件”，遇到解码错误时应先定位异常源文件。

### 4.4 说话人名称与工作目录

如果不手工指定工作目录，DataFactory 按说话人名称使用：

```text
user_data/{speaker_name}_factory
```

例如：

```text
说话人 / 角色名：Character_A
工作目录（work_dir）：user_data/Character_A_factory
```

如果需要自定义位置或恢复旧项目，可在 **已有项目 / 工作目录** 中填写 **工作目录（可选）**。

### 4.5 载入 / 扫描已有项目

点击：

```text
载入/扫描工作目录
```

系统会重新检查真实文件产物并刷新：

- 数据准备；
- 文本确认；
- Stage1；
- Stage2 各子步骤；
- 缺失或部分完成状态。

> [!IMPORTANT]
> DataFactory 的恢复机制以工作目录（`work_dir`）为中心。浏览器刷新、页面状态丢失或 WebUI 重启，并不代表已有数据产物消失。

---

## 5. 数据准备与文本确认

### 5.1 开始准备数据

完成基础输入后点击：

```text
开始准备数据
```

默认流程大致为：

```text
扫描原始音频
  ↓
建立索引
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
基础健康检查
```

初始 `stage2_manifest.jsonl` 只是数据准备阶段的中间出口，**不代表 Stage2 训练数据已经就绪**。

### 5.2 本地离线 FunASR 与成功条件

正式发行路径使用：

```text
language = zh
backend = auto → 本地 FunASR
```

FunASR 使用发布包中的本地 ASR / VAD / 标点模型；运行时不会从 ModelScope 或 Hugging Face 下载这些模型，更新检查也处于关闭状态。缺少模型资产时应检查发行包或环境，而不是等待联网下载。

数据准备完成后，页面应显示：

```text
音频处理与基础识别产物已就绪。
```

工作目录扫描至少应检测到：

```text
06_export/manifest.jsonl
06_export/dataset.list
```

如果任务失败或中止，优先查看当前任务 / 诊断信息并重新扫描工作目录，不要默认删除整个项目重来。

### 5.3 人工校对器

点击：

```text
启动 / 打开人工校对器
```

校对器会读取当前 `dataset.list`。根据对应音频检查 ASR 文本，修正错字、漏字或明显识别错误，并使用校对器自身的保存功能保存。同一 `work_dir` 的校对器已经运行时，发行版会阻止重复启动新的同类实例。

DataFactory 还提供独立的：

```text
关闭人工校对器
```

它与：

```text
停止当前任务
```

不是同一个功能。前者管理校对器进程，后者管理 DataFactory / Stage1 / Stage2 构建任务。

### 5.4 完成文本确认

保存校对结果后，返回 DataFactory 点击：

```text
已保存并完成校对
```

如果人工检查后确认 ASR 结果可以直接使用，则点击：

```text
无需校对，继续
```

这表示接受当前识别文本，不是跳过文本确认阶段。

成功后页面应显示：

```text
文本确认结果已就绪。
```

关键产物：

```text
06_export/manifest.corrected.jsonl
```

Stage1 与 Stage2 都应先完成文本确认，但两条链读取方式不同：

- **Stage1** 主要使用 `06_export/dataset.list` 和 `06_export/clips/`；校对器直接修改 `dataset.list`；
- **Stage2** 正式后处理要求 `06_export/manifest.corrected.jsonl`。

---

## 6. 生成 Stage1 / Stage2 训练数据

### 6.1 Stage1 数据生成

Stage1-only 与 Full Few-shot 需要点击：

```text
生成 Stage1 训练数据
```

当前用户路径主要使用：

```text
06_export/dataset.list
06_export/clips/
```

输出根目录：

```text
07_stage1_ft/
```

默认训练集比例：

```text
train_ratio = 0.90
```

验证集比例自动为 `1 - train_ratio`；划分随机种子由构建流程自动生成并记录到：

```text
07_stage1_ft/metadata/split.json
```

成功时页面应显示：

```text
Stage1 训练数据已完成。
```

Stage1 是否已就绪，不能只看目录是否存在，还会检查：

```text
07_stage1_ft/metadata/build_report.json
07_stage1_ft/train_manifest.jsonl
07_stage1_ft/val_manifest.jsonl
07_stage1_ft/frontend_cache/
07_stage1_ft/semantic_cache/
```

### 6.2 Stage1 覆盖保护与恢复边界

如果 `07_stage1_ft/` 下已经存在任何文件，而没有显式授权覆盖，发行版会阻止新的 Stage1 构建。

> [!WARNING]
> `覆盖已有 Stage1 cache` 是显式重建操作。只有确定要重建当前 Stage1 数据时才启用。

Stage1 **不支持** Stage2 这种基于已有产物的续跑（Resume）：

- 已完整就绪 → 保留现有结果即可；
- 部分完成或失败 → 先检查工作目录和日志；
- 确认需要重建后 → 显式启用 `覆盖已有 Stage1 cache`。

成功覆盖后，发行版会清理新 manifest 不再引用的旧缓存，并执行严格缓存契约检查。

### 6.3 Stage2 数据生成

Stage2-only 与 Full Few-shot 需要点击：

```text
生成 Stage2 训练数据
```

Stage2 是多阶段链：

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
训练 / 验证集划分
  ↓
异常样本过滤
```

参考音频（Prompt）选择默认规则：

| 参数 | 默认值 |
| --- | ---: |
| `prompt_mode` | `speaker_pool` |
| `min_prompt_sec` | `3.0` |
| `max_prompt_sec` | `10.0` |
| `prefer_prompt_sec` | `6.0` |
| `allow_self_prompt` | `True` |

这里的 3–10 秒是 **Stage2 数据构建时的参考音频（Prompt）选择规则**，不要与 Inference 页面要求用户提供 3–10 秒 Prompt WAV 混为同一个操作。

主要 Stage2 产物位于：

```text
07_stage2_pt_all/
08_continuous_semantic/
09_style_f0_spk/
10_train_val/
11_filter/
```

v1.0.0 用户发行路径固定启用异常样本自动隔离，最终过滤步骤使用 `move` 语义：命中的异常样本会移入：

```text
11_filter/rejected/
```

并生成 JSON / CSV 格式的过滤报告。因此过滤后可训练样本数少于过滤前是可能的正常结果。

Stage2 只有在 Few-shot manifest、Stage2 `.pt`、continuous / style 缓存、训练 / 验证集划分，以及与当前划分和文件数量一致的有效过滤结果全部完成后，才视为已就绪。成功时用户摘要应显示：

```text
Stage2 训练数据已完成。
```

### 6.4 Stage2 基于已有产物的续跑（Resume）

中断、重启、刷新或修复某一步问题后，可以：

```text
继续生成 Stage2 训练数据
```

推荐恢复路径：

```text
重新启动 DataFactory
  ↓
填写原来的说话人 / 角色名和工作目录
  ↓
载入/扫描工作目录
  ↓
确认真实产物状态
  ↓
继续生成 Stage2 训练数据
```

续跑会根据真实产物决定哪些步骤可以复用：

- 已完整且仍有效的关键产物可以跳过；
- 已存在但尚未完整的 `.pt` / 缓存会尝试继续补齐；
- 显式启用覆盖后，会强制对应步骤重新执行；
- 已过期，或与当前文件系统状态不一致的划分 / 过滤报告，不会被视为有效产物；
- 任一步真正失败时，后续链停止，不会把失败伪装成成功。

> [!IMPORTANT]
> Stage2 中断后优先 **扫描状态 + 续跑**，不要默认打开所有覆盖选项从头重跑。

### 6.5 各路线的完成条件

| 路线 | DataFactory 最终应确认 |
| --- | --- |
| Stage1-only | 文本确认完成 + Stage1 已就绪 |
| Stage2-only | 文本确认完成 + Stage2 已就绪 |
| Full Few-shot | Stage1 已就绪 + Stage2 已就绪 |

顶部状态或 Training 交接应以 Stage1 / Stage2 的独立就绪状态为准，不要只根据目录是否存在或按钮是否执行过判断成功。

---

## 7. 停止任务、状态与诊断

### 7.1 停止当前任务

在 **已有项目 / 工作目录** 区域点击：

```text
停止当前任务
```

系统会尝试停止与当前工作目录匹配的 DataFactory 构建子进程。停止后应重新扫描工作目录，因为已执行步骤可能留下部分产物。

推荐：

```text
停止当前任务
  ↓
载入/扫描工作目录
  ↓
查看真实状态
  ↓
决定续跑 / 修复 / 显式覆盖
```

`停止当前任务` 不负责关闭人工校对器。

### 7.2 实时状态

**当前任务** 可以主动 **立即刷新**，长任务期间也会自动刷新。

用户级状态包括：

| 状态 | 含义 |
| --- | --- |
| 尚未开始 | 没有有效进度 |
| 等待前置步骤 | 上游尚未完成 |
| 等待确认 | 数据准备完成，等待文本确认 |
| 部分完成 | 已有有效产物但尚未完整 |
| 已完成 | 当前阶段通过状态检查 |

诊断层可能显示 `missing / blocked / partial / done`。Training 交接最重要的是独立的 **Stage1 / Stage2 就绪状态**，而不是某个文件夹是否存在。

### 7.3 诊断信息与日志

正常流程通常不需要展开 **诊断信息**。排查时可查看最近操作、工作目录状态、步骤摘要、人工校对器状态，以及 DataFactory Result / Live Monitor 等诊断 JSON。

主要运行日志位于：

```text
{work_dir}/logs/
```

Stage1 还会记录独立的标准输出（stdout）、标准错误（stderr）和执行命令信息。优先先定位失败步骤，再查看对应日志。

---

## 8. 设置与高级参数

第一次跑通 DataFactory 时建议优先使用默认值。只有明确知道要解决什么问题时，再调整高级设置。

### 8.1 常用设置与超时边界

| 设置 | 默认值 | 作用 |
| --- | ---: | --- |
| 覆盖已有数据工厂工作目录 | 关闭 | 允许数据准备执行覆盖式操作 |
| 覆盖已上传音频缓存 | 开启 | 上传时清理 `00_uploaded_raw_audio/` |
| 单步超时秒数，0 表示不限制 | `0` | 控制单步超时 |
| Stage1 训练集比例 | `0.90` | Stage1 训练 / 验证集划分 |
| Stage2 训练集比例 | `0.90` | Stage2 训练 / 验证集划分 |
| 显示终端进度窗口 | 开启 | 额外显示只读日志终端 |

> [!NOTE]
> v1.0.0 中，超时值为 `0` 在不同执行路径中的实际处理并不完全一致：Stage1 / Stage2 会把 `0` 归一化为“未显式设置超时”，而数据准备流程会回落到内部 `7200` 秒默认值。如果数据准备预计可能超过 2 小时，请填写更大的正数，不要依赖 `0` 获得无限时长。

> [!WARNING]
> 正常恢复旧项目首先应 **载入/扫描工作目录**。Stage2 优先续跑；Stage1 的部分产物按 §6.2 的覆盖保护规则处理。不要把“同时开启所有覆盖选项并从头重跑”当作默认恢复方式。

### 8.2 ASR 设置

v1.0.0 正式用户发行界面的 **ASR 后端 / ASR 模型大小 / ASR 精度 / 语言** 锁定为：

```text
ASR backend = auto
ASR model size = large
ASR precision = float32
language = zh
```

`auto` 进入本地 FunASR。ASR 后端、模型大小和精度在最终发行界面中不可编辑，语言只有 `zh`。

### 8.3 高级设置参考

这些设置全部保留在发行 UI 中，但普通 Few-shot 用户通常不需要修改。

| 类别 | 参数 | 默认值 / 状态 | 使用边界 |
| --- | --- | --- | --- |
| 音频切分 | `threshold` | `-34` | 默认切片明显过长 / 过碎时再考虑调整 |
| 音频切分 | `min_length` | `4000` | 同上 |
| 音频切分 | `min_interval` | `300` | 同上 |
| 音频切分 | `hop_size` | `10` | 同上 |
| 音频切分 | `max_sil_kept` | `500` | 同上 |
| 音频切分 | `normalize_max` | `0.9` | 同上 |
| 音频切分 | `alpha_mix` | `0.25` | 同上 |
| Stage1 | device | `cuda` | Stage1 数据构建使用的设备 |
| Stage1 | Stage1 use_half | 开启 | 对应模型流程允许使用半精度 |
| Stage1 | 覆盖已有 Stage1 cache | 关闭 | 显式重建 |
| Stage1 | 构建后验证 dataset | 开启 | 构建后检查数据集契约 |
| Stage2 | device | `cuda` | Stage2 数据处理使用的设备 |
| Stage2 | Stage2 use_half | 关闭 | Stage2 `.pt` 处理精度 |
| Stage2 | 覆盖已有 `.pt` | 关闭 | 强制重建对应产物 |
| Stage2 | 覆盖 continuous cache | 关闭 | 强制重建对应产物 |
| Stage2 | 覆盖 style cache | 关闭 | 强制重建对应产物 |
| Stage2 | 覆盖 train/val | 关闭 | 强制重新划分 |
| Stage2 | continuous dtype | `float32` | continuous semantic 缓存的数据类型 |
| 校对器 | 端口 | `9871` | 端口冲突时才通常需要改 |
| 校对器 | `g_batch` | `10` | 校对器批处理设置 |
| 校对器 | 覆盖已有校对备份 | 关闭 | 控制是否覆盖已有备份 |
| 校对器 | 重建 corrected manifest 时长容差 | `0.05` | 重建匹配容差 |
| 校对器 | 重建超时秒数 | `600` | 校对后重建的超时设置 |

Stage2 参考音频选择的默认值已经在 §6.3 说明，这里不再重复。

Mel / F0 特征默认值：

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
> Mel / F0 等特征参数直接参与 Stage2 数据契约。普通用户不建议为了“试效果”随意修改。

---

## 9. 工作目录、关键产物与 Training 交接

普通用户不需要记住所有内部文件。最重要的是当前工作目录（`work_dir`）、文本确认结果，以及 Stage1 / Stage2 是否已就绪。

### 9.1 用户需要知道的主要目录

```text
{speaker_name}_factory/
├── 00_uploaded_raw_audio/       # 仅上传模式使用
├── 06_export/                   # clips / manifests
├── 07_stage1_ft/                # Stage1 数据
├── 07_stage2_pt_all/            # Stage2 .pt
├── 08_continuous_semantic/
├── 09_style_f0_spk/
├── 10_train_val/
├── 11_filter/
└── logs/
```

`01_uvr_*` 与 `02_denoise/` 也可能存在于基础目录结构中，但 v1.0.0 正式用户流程没有开放 UVR / denoise 功能；这些目录为空不代表失败。

### 9.2 关键产物

| 产物 | 用户层含义 |
| --- | --- |
| `06_export/clips/` | 标准化切分后的训练音频 |
| `06_export/manifest.jsonl` | 数据准备阶段的基础 manifest（数据清单） |
| `06_export/dataset.list` | Stage1 / 校对流程的重要列表 |
| `06_export/manifest.corrected.jsonl` | 文本确认后的主 manifest（数据清单） |
| `06_export/stage2_manifest.fewshot.jsonl` | Stage2 Few-shot 正式 manifest（数据清单） |
| `07_stage1_ft/` | Stage1 就绪所需的数据与缓存 |
| `07_stage2_pt_all/` ～ `11_filter/` | Stage2 多阶段数据链 |
| `11_filter/rejected/` | 被自动隔离的异常 Stage2 样本 |
| `logs/` | 排查日志 |

数据准备阶段还会产生初始 `stage2_manifest.jsonl` 等文件；它们不能单独证明 Stage2 已就绪。

### 9.3 Training 使用同一个项目身份

进入 Training 时继续使用同一个 **说话人 / 角色名** 和工作目录（`work_dir`）：

- 默认规则：`user_data/{speaker_name}_factory`；
- 如果 DataFactory 使用了自定义工作目录，Training 中填写同一个目录。

DataFactory 只负责把路线需要的数据准备到可训练状态；epochs、batch_size、模型检查点、Active Best 等属于 [模型训练指南（Training）](./training.md)。

---

## 10. 常见边界与安全操作

| 容易误解的行为 | 正确理解 |
| --- | --- |
| 支持扩展名 = 一定能解码 | 错。编码格式、封装和文件完整性仍会影响读取 |
| MP3 / AAC 转 WAV = 恢复质量 | 错。转容器不会恢复有损编码丢失的信息 |
| Stage1 部分完成后可以直接续跑 | 错。Stage1 不支持 Stage2 式的基于已有产物续跑，需要显式启用覆盖后重建 |
| Stage2 中断后默认从头重跑 | 不建议。先扫描，再使用 `继续生成 Stage2 训练数据` |
| 关闭进度终端 = 停止任务 | 错。只读进度终端可以关闭；真正的停止操作需要通过 WebUI 中的停止功能执行 |
| `停止当前任务` 会关闭校对器 | 错。人工校对器有独立关闭按钮 |
| Few-shot 必须 Stage1 + Stage2 都完成 | 错。Stage1-only / Stage2-only 都是正式支持路线 |
| 当前 `zh + 本地 FunASR` = 永久产品能力 | 错。这是 Resonastra v1.0.0 的发行边界 |

---

## 11. 下一步与相关文档

完成 DataFactory 后，下一步是进入 Training：

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
- [模型训练指南（Training）](./training.md)
- [语音生成指南（Inference）](./inference.md)
- [兼容性指南](./compatibility.md)
