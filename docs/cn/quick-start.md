# Resonastra 快速开始

[项目 README](https://github.com/SiriusLin1531/Resonastra/blob/main/docs/cn/README.md) | [English](../en/quick-start.md) | **中文简体**

适用版本：**Resonastra v1.0.0**

本指南面向第一次使用 Resonastra 的用户。Resonastra 同时提供 **Zero-shot（无需训练，直接使用基础模型推理）** 与 **Few-shot（使用自己的语音数据微调后推理）** 两类使用方式；Few-shot 又可以只训练 Stage1、只训练 Stage2，或同时训练 Stage1 + Stage2。

因此，你不需要先完成全部训练才能开始使用 Resonastra。最快的体验路径是直接使用内置 Zero-shot 基础模型；如果希望让模型进一步适配自己的语音数据，再进入 Few-shot 流程。

```text
下载并解压
  ↓
环境检查
  ↓
选择路线
  ├─ Zero-shot ─────────────────────────────┐
  │                                        │
  └─ Few-shot                              │
       ↓                                   │
     DataFactory                           │
       ↓                                   │
     Stage1 / Stage2 按需训练              │
       ↓                                   │
     Voice Profile                         │
       └───────────────────────────────────┤
                                           ↓
                                      Inference
                                           ↓
                                      Generated Audio
```

本指南使用正常用户流程和默认设置。数据质量、训练参数、checkpoint 管理、GPU / CUDA 兼容性和详细故障排查将在独立文档中说明。

---

## 1. 开始之前

Resonastra v1.0.0 面向：

- Windows 10 / Windows 11 x64
- 本地运行
- NVIDIA GPU 推荐用于训练与推理

v1.0.0 的 DataFactory 当前只开放中文 `zh` 数据准备流程。

无论选择哪条路线，Inference 都需要：

- 一段 **3–10 秒**的参考音频 / Prompt WAV
- 与参考音频实际内容一致的参考文本 / Prompt Text
- 希望生成的目标文本 / Target Text

参考音频的 3–10 秒范围是当前 v1.0.0 推理链路的硬性要求；低于 3 秒或高于 10 秒都会被拒绝。

如果选择 Few-shot，还需要另外准备一组自己的原始语音文件，用于 DataFactory 与后续训练。

为了兼顾训练效果与数据准备 / 训练效率，首次 Few-shot 建议从**几分钟到几十分钟级别**的语音数据开始。这不是硬性的数据量限制，更大的数据集也可以继续使用。

为了获得更稳定的训练效果，语音数据集建议尽量满足：

- **单说话人**
- **发音清晰**
- **语速正常**
- **无明显背景噪声**

这些属于训练数据质量建议，而不是 DataFactory 的强制输入限制。

> [!IMPORTANT]
> Resonastra v1.0.0 集成的运行环境为 Python 3.10.20、PyTorch 2.5.1 和 CUDA Runtime 11.8。环境检查通过并不代表所有新架构 GPU 都已经通过实际 CUDA kernel 验证。如果你使用 RTX 50 系列等新架构 GPU，请先阅读[项目 README](https://github.com/SiriusLin1531/Resonastra/blob/main/docs/cn/README.md)中的兼容性说明。

---

## 2. 下载并解压 Resonastra

### 2.1 下载正式发布包

Resonastra v1.0.0：

- **GitHub Release：** <https://github.com/SiriusLin1531/Resonastra/releases/tag/v1.0.0>
- **百度网盘：** <https://pan.baidu.com/s/1_344UD7sJci6IuBgtbmF6A?pwd=8848>
- **提取码：** `8848`
- **发布包：** `Resonastra_v1.0.0.zip`

> [!IMPORTANT]
> GitHub 自动生成的 **Source code (zip / tar.gz)** 只是源码快照，不是完整的 Windows 整合包。首次使用请下载正式的 `Resonastra_v1.0.0.zip`。

### 2.2 完整解压

将 `Resonastra_v1.0.0.zip` 完整解压到本地磁盘。

后续所有启动器都从解压后的 Resonastra 根目录运行，例如：

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

不要只把单个 `.bat` 文件复制到其他位置运行。

> [!IMPORTANT]
> `launch_data_factory_ui.bat`、`launch_training_ui.bat` 和 `launch_infer_ui.bat` 启动后都会保留一个 **WebUI 启动终端**。在使用对应 WebUI 期间不要关闭这个主终端窗口；关闭它会结束对应的 WebUI 进程。
>
> 后文提到的“进度终端 / 日志终端”属于额外的只读查看窗口，与这个主启动终端不同。

---

## 3. 运行环境检查

开始前先运行：

```text
launch_check_env.bat
```

你可以在资源管理器中双击该文件。

环境检查会验证 v1.0.0 正常运行所需的主要本地组件，包括：

- Microsoft Visual C++ v14 x64 Redistributable
- Resonastra bundled runtime
- 必要 Python packages
- 默认配置
- 默认模型资产

其中 Microsoft Visual C++ Redistributable 的最低检查版本为：

```text
14.44.35211
```

### 环境检查成功

检查完成后，如果终端显示：

```text
[OK] Environment check completed successfully.
```

即可继续。

环境检查还会生成：

```text
logs/check_user_env_report.json
```

用于记录本次检查结果。

> [!NOTE]
> 当前环境检查主要验证 runtime、依赖、配置和模型资产，不会实际执行 CUDA kernel 来验证 GPU compute capability。

如果环境检查显示 `[FAIL]`，先根据终端中的具体提示处理问题，再继续后续步骤。

---

## 4. 选择你的使用路线

Resonastra v1.0.0 支持四种模型组合：

| 路线 | Stage1 | Stage2 | 训练成本 | 适配范围 | 建议 |
| --- | --- | --- | --- | --- | --- |
| **Zero-shot** | 默认模型 | 默认模型 | 无训练 | 不进行用户数据训练 | **首次体验推荐** |
| **Stage1-only Few-shot** | Few-shot | 默认模型 | 较低 | 仅适配 Stage1 | 适合轻量 Few-shot |
| **Stage2-only Few-shot** | 默认模型 | Few-shot | 较高 | 仅适配 Stage2 | 支持，但通常不建议单独使用 |
| **Full Few-shot** | Few-shot | Few-shot | 最高，但相较 Stage2-only 增量有限 | 同时适配 Stage1 + Stage2 | **完整训练推荐** |

### 如何选择

如果你只是想确认 Resonastra 能否正常运行并尽快生成第一段语音：

**选择 Zero-shot。**

如果你希望以较低的数据准备和训练成本先进行轻量个性化适配：

**选择 Stage1-only Few-shot。**

如果你已经计划训练 Stage2：

**通常建议直接选择 Full Few-shot。**

根据当前项目开发和训练经验，Stage1 的数据准备与训练总耗时显著短于 Stage2。也就是说，Stage2-only 已经承担了 Few-shot 流程中的主要时间成本，而在此基础上再完成 Stage1 所增加的额外时间相对有限。

因此，从时间成本与完整适配范围综合考虑：

> **不推荐普通用户为了节省时间而只训练 Stage2；如果已经决定训练 Stage2，更推荐同时完成 Stage1 + Stage2。**

Full Few-shot 会让两个阶段都使用用户数据完成适配，也是当前推荐用于获得最佳整体效果的完整训练路线。

> [!NOTE]
> Stage2-only 仍然是正式支持的配置，适合特定实验、对照或只希望替换 Stage2 的场景。这里的路线建议针对普通用户的首次使用和常规个性化训练。

对应的 Voice Profile 类型为：

```text
Zero-shot              → base_zeroshot
Stage1-only Few-shot   → few_shot_stage1_only
Stage2-only Few-shot   → few_shot_stage2_only
Full Few-shot          → few_shot_dual
```

如果你选择 **Zero-shot**，可以跳过第 5～7 节，直接进入 **第 8 节：使用 Voice Profile 生成语音**。

如果你选择任一 Few-shot 路线，请继续下一节。

---

## 5. Few-shot：使用 DataFactory 准备训练数据

运行：

```text
launch_data_factory_ui.bat
```

启动器会启动 Resonastra DataFactory，并尝试自动打开浏览器。

默认端口为：

```text
7861
```

如果该端口已被占用，DataFactory 启动器会自动选择其他可用端口。实际地址以启动终端显示的信息为准。

> [!NOTE]
> DataFactory 的 **显示终端进度窗口** 默认开启。执行数据准备、Stage1 数据生成或 Stage2 数据处理等较长任务时，可能会额外打开一个进度终端实时显示日志。这个窗口只是进度查看器，关闭它不会停止正在运行的 DataFactory 任务。
>
> 但不要关闭通过 `launch_data_factory_ui.bat` 打开的 WebUI 主启动终端。

### 5.1 填写基础输入

在 **Resonastra · Data Factory** 页面中，先填写左侧的 **基础输入**。

对于第一次使用，建议采用最简单的本地目录流程：

- **原始音频来源：** `本地目录路径`
- **本地原始音频目录：** 填写你的原始语音文件所在目录
- **说话人 / 角色名：** 为这套声音数据填写一个名称
- **语言：** 保持 `zh`

> [!NOTE]
> **原始音频来源** 还提供 `上传音频文件` 与 `上传音频文件夹` 两种入口。选择后可以通过文件选择器选择本地音频文件或整个文件夹。对于较大的数据集，仍建议使用 `本地目录路径`；上传入口更适合小样本或测试数据。
>
> DataFactory v1.0.0 正式识别以下原始音频扩展名：`WAV / MP3 / FLAC / OGG / M4A / AAC / WMA / OPUS`。无需提前把这些文件统一转换为 WAV；DataFactory 会在处理流程中完成解码与标准化，再进入后续数据准备步骤。
>
> 如果目标是准备高质量训练数据，优先推荐使用质量良好的 `WAV` 或 `FLAC` 原始音频。MP3、AAC 等有损格式也可以使用，但已有的有损压缩信息不会因为后续转换为 WAV 而恢复。扩展名受支持也不代表任意编码或损坏文件都一定能够成功解码；更详细的输入边界将在 DataFactory 文档中说明。

例如，如果角色名填写：

```text
Character_A
```

并且不手动填写工作目录，DataFactory 默认会使用：

```text
user_data/Character_A_factory
```

作为本次数据准备的工作目录。

首次使用时，可以保持 **已有项目 / 工作目录** 中的工作目录为空。

### 5.2 开始准备数据

确认基础输入后，点击：

```text
开始准备数据
```

DataFactory 会进行原始音频处理和基础识别。

#### 成功条件

完成后，“音频处理”阶段应显示：

```text
音频处理与基础识别产物已就绪。
```

---

### 5.3 检查并确认 ASR 文本

为了避免识别文本中的明显错误直接进入训练数据，本指南采用人工确认流程。

点击：

```text
启动 / 打开人工校对器
```

在人工校对器中检查 ASR 文本并根据需要修改。

完成后：

1. 在人工校对器中保存结果。
2. 返回 DataFactory。
3. 点击：

```text
已保存并完成校对
```

如果你已经确认自动识别结果无需修改，也可以选择：

```text
无需校对，继续
```

#### 成功条件

“文本确认”阶段应显示：

```text
文本确认结果已就绪。
```

---

### 5.4 根据路线生成训练数据

文本确认完成后，Stage1 与 Stage2 训练数据可以按需要独立生成。

#### Stage1-only Few-shot

只点击：

```text
生成 Stage1 训练数据
```

成功后应显示：

```text
Stage1 训练数据已完成。
```

不需要生成 Stage2 训练数据。

#### Stage2-only Few-shot

只点击：

```text
生成 Stage2 训练数据
```

成功后应显示：

```text
Stage2 训练数据已完成。
```

不需要生成 Stage1 训练数据。

#### Full Few-shot

依次完成：

```text
生成 Stage1 训练数据
生成 Stage2 训练数据
```

最终页面推荐摘要应显示：

```text
Stage1 与 Stage2 数据均已完成。
```

### 本节结束状态

记住当前使用的 **说话人 / 角色名**。如果使用默认工作目录规则，Training 会继续使用：

```text
user_data/{speaker_name}_factory
```

> [!NOTE]
> DataFactory 还支持载入已有工作目录、继续生成 Stage2 数据以及更多高级参数。Quick Start 不展开这些功能。

---

## 6. Few-shot：训练所需阶段

运行：

```text
launch_training_ui.bat
```

默认端口为：

```text
7863
```

如果该端口已被占用，Training 启动器会自动选择可用端口。

> [!NOTE]
> Training 的 **打开独立训练日志终端（可选）** 默认关闭。Training Live Monitor 已经可以查看训练状态；如果希望观察更详细的实时日志，可以手动开启独立训练日志终端。该窗口只是只读日志查看器，关闭它不会停止训练。
>
> 同样，不要关闭通过 `launch_training_ui.bat` 打开的 WebUI 主启动终端。

### 6.1 扫描 DataFactory 数据

在 **Resonastra · Training** 页面中：

1. 在 **说话人 / 角色名** 中填写与 DataFactory 相同的角色名。
2. 如果之前使用默认工作目录规则，可以保持 **DataFactory 工作目录** 为空。
3. 点击：

```text
扫描训练数据
```

如果 DataFactory 使用了自定义工作目录，则在这里填写同一个目录。

检查与你所选路线对应的数据和训练入口：

| 路线 | 需要就绪的数据 | 需要可启动的训练 |
| --- | --- | --- |
| Stage1-only | Stage1 数据 | Stage1 Training |
| Stage2-only | Stage2 数据 | Stage2 Training |
| Full Few-shot | Stage1 + Stage2 数据 | Stage1 + Stage2 Training |

Partial Few-shot 不要求“完整训练数据”整体状态为已就绪；只需要本路线实际使用的阶段通过入口检查。

---

### 6.2 Stage1 Training

以下路线需要 Stage1：

- Stage1-only Few-shot
- Full Few-shot

使用 WebUI 已加载的默认训练参数，不修改 `epochs` 和 `batch_size`，点击：

```text
启动 Stage1 Training
```

等待训练完成。

#### 成功条件

训练状态最终应进入：

```text
succeeded
```

并可能显示：

```text
stage1 训练完成：run_id=...
```

正常训练成功并产生最佳 checkpoint 后，Training worker 会自动注册本轮 Trainer Best；在未进行手动 Active Best 锁定的首次使用流程中，它会更新 Stage1 Active Best。

---

### 6.3 Stage2 Training

以下路线需要 Stage2：

- Stage2-only Few-shot
- Full Few-shot

点击：

```text
启动 Stage2 Training
```

等待训练完成。

#### 成功条件

训练状态最终应进入：

```text
succeeded
```

并可能显示：

```text
stage2 训练完成：run_id=...
```

正常训练成功并产生最佳 checkpoint 后，本轮 Trainer Best 会在未进行手动 Active Best 锁定的首次使用流程中更新 Stage2 Active Best。

> [!IMPORTANT]
> Stage2 的数据准备与训练是 Few-shot 路线中主要的时间成本。由于 Stage1 的整体耗时显著更短，如果你已经完成或准备进行 Stage2 Training，通常更推荐同时完成 Stage1，形成 Full Few-shot，而不是仅为了节省时间选择 Stage2-only。

> [!NOTE]
> Training UI 还提供手动 Active Best、Automatic Best、checkpoint 删除等训练管理功能。第一次跑通流程不需要使用这些功能。

---

## 7. Few-shot：生成 Voice Profile

完成所选训练路线后，在 **生成 Voice Profile v2** 区域创建用于推理的声音配置。

`profile_name` 可以留空，此时系统会优先使用当前 **说话人 / 角色名**。

### Stage1-only Few-shot

设置：

- **Stage1 强制使用 zero-shot：** 关闭
- **Stage2 强制使用 zero-shot：** 开启

然后点击：

```text
生成 Voice Profile v2
```

预期结果：

```text
类型：few_shot_stage1_only
Stage1 来源：few-shot active best
Stage2 来源：zero-shot default
可用于推理：True
```

### Stage2-only Few-shot

设置：

- **Stage1 强制使用 zero-shot：** 开启
- **Stage2 强制使用 zero-shot：** 关闭

然后点击：

```text
生成 Voice Profile v2
```

预期结果：

```text
类型：few_shot_stage2_only
Stage1 来源：zero-shot default
Stage2 来源：few-shot active best
可用于推理：True
```

### Full Few-shot

设置：

- **Stage1 强制使用 zero-shot：** 关闭
- **Stage2 强制使用 zero-shot：** 关闭

点击：

```text
生成 Voice Profile v2
```

预期结果：

```text
类型：few_shot_dual
Stage1 来源：few-shot active best
Stage2 来源：few-shot active best
可用于推理：True
```

生成的 Voice Profile 位于：

```text
user_profiles/<profile_name>/
```

其中包含：

```text
profile_config.json
profile_manifest.json
```

Voice Profile 可以理解为 Resonastra 在推理时使用的模型组合配置：它决定 Stage1 和 Stage2 分别使用默认模型还是训练得到的 Few-shot 模型。

---

## 8. 使用 Voice Profile 生成语音

运行：

```text
launch_infer_ui.bat
```

Inference 默认地址为：

```text
http://127.0.0.1:7860
```

启动器会尝试自动打开浏览器。

如果浏览器没有自动打开，但终端中没有显示启动失败，可以手动访问上面的地址。

> [!IMPORTANT]
> 推理期间请保持 `launch_infer_ui.bat` 打开的 WebUI 启动终端运行，不要关闭该窗口。

### 8.1 选择声音角色

在 **1. 选择声音角色** 中选择对应的 Profile。

#### Zero-shot

选择发布包内置的中文基础模型：

```text
default_zh
```

其 Profile 类型为：

```text
base_zeroshot
```

该配置使用 Resonastra 自带的默认 Stage1 / Stage2 模型，不需要 DataFactory 或 Training。

#### Few-shot

选择第 7 节刚刚生成的 Voice Profile。

#### 成功条件

角色摘要应显示：

```text
可用于生成
```

并显示 Stage1 / Stage2 模型均已就绪。

---

### 8.2 添加参考音频

在 **2. 添加参考音频** 中提供：

- **参考音频 / Prompt WAV**
- **参考音频文本 / Prompt Text**

参考音频必须控制在：

```text
3–10 秒
```

当前 v1.0.0 的正式 prompt 提取路径会严格检查参考音频长度；低于 3 秒或高于 10 秒会直接拒绝推理。

参考文本应尽量与参考音频中实际说出的文字一致。

例如，参考音频中说的是：

```text
欢迎使用 Resonastra。
```

那么 Prompt Text 也应填写对应内容。

---

### 8.3 输入目标文本

在：

```text
目标文本 / Target Text
```

中输入希望生成的内容。

例如：

```text
这是使用 Resonastra 生成的第一段语音。
```

> [!NOTE]
> 根据当前版本的开发测试经验，建议单次生成文本控制在约 **80–100 个汉字以内**。这不是代码中的硬性字符限制，实际稳定长度会受到文本内容、模型状态和参考文本等因素影响。
>
> 文本过长时，可能出现发音不清、语音重复或内容混乱。较长内容建议拆分为多个较短段落分别生成。

---

### 8.4 保持默认推理设置

第一次使用时，不需要修改 **高级选项 / Advanced Options**。

以下质量检测默认均为关闭：

- DNSMOS 语音质量评价
- Speaker Similarity 声音相似度
- WER / CER 文本准确度

Quick Start 不要求开启这些指标。

---

### 8.5 生成语音

确认：

- 已选择可用于生成的 Voice Profile
- 已添加 Prompt WAV
- 已填写 Prompt Text
- 已填写 Target Text

点击：

```text
生成语音
```

> [!NOTE]
> **首次推理可能需要较长时间进行暖机。** 第一次生成时通常需要完成模型加载、运行环境初始化等准备工作，因此可能明显慢于后续使用。请耐心等待，不要因为短时间没有结果而重复点击“生成语音”或关闭 WebUI 启动终端。

#### 成功条件

结果区域应显示：

```text
生成完成
```

并提示：

```text
语音已生成，可以直接在上方播放器试听。
```

生成结果会出现在：

```text
生成音频 / Generated Audio
```

播放器中。

默认推理结果保存到：

```text
outputs/inference_runs/
```

如果没有指定自定义输出目录，每次推理会创建独立 run 目录。

---

## 9. 按路线完成检查

只检查与你选择的路线对应的项目。

### Zero-shot

- [ ] `launch_check_env.bat` 环境检查通过
- [ ] Inference 中选择 `default_zh`
- [ ] Profile 可用于生成
- [ ] Inference 显示 `生成完成`
- [ ] 已试听 Generated Audio

### Stage1-only Few-shot

- [ ] DataFactory 文本确认完成
- [ ] Stage1 训练数据完成
- [ ] Stage1 Training 状态为 `succeeded`
- [ ] Voice Profile 类型为 `few_shot_stage1_only`
- [ ] Stage1 来源为 `few-shot active best`
- [ ] Stage2 来源为 `zero-shot default`
- [ ] Voice Profile 显示 `可用于推理：True`
- [ ] Inference 生成完成

### Stage2-only Few-shot

- [ ] DataFactory 文本确认完成
- [ ] Stage2 训练数据完成
- [ ] Stage2 Training 状态为 `succeeded`
- [ ] Voice Profile 类型为 `few_shot_stage2_only`
- [ ] Stage1 来源为 `zero-shot default`
- [ ] Stage2 来源为 `few-shot active best`
- [ ] Voice Profile 显示 `可用于推理：True`
- [ ] Inference 生成完成

### Full Few-shot

- [ ] DataFactory 文本确认完成
- [ ] Stage1 训练数据完成
- [ ] Stage2 训练数据完成
- [ ] Stage1 Training 状态为 `succeeded`
- [ ] Stage2 Training 状态为 `succeeded`
- [ ] Voice Profile 类型为 `few_shot_dual`
- [ ] Stage1 来源为 `few-shot active best`
- [ ] Stage2 来源为 `few-shot active best`
- [ ] Voice Profile 显示 `可用于推理：True`
- [ ] Inference 生成完成

---

## 10. 下一步

如果你只是完成了 Zero-shot 路线，可以继续尝试 Few-shot，让 Resonastra 使用自己的语音数据进行模型适配。

如果你准备进行 Few-shot：

- **Stage1-only** 适合作为较低时间成本的轻量训练路线
- **Stage2-only** 技术上受支持，但从时间成本角度通常不建议单独选择
- **Full Few-shot** 是当前推荐的完整个性化训练路线

后续文档将进一步说明：

- **DataFactory** — 数据准备、ASR、人工校对与 Stage1 / Stage2 数据处理
- **Training** — 训练参数、Live Monitor、checkpoint 与 Active Best
- **Inference** — Voice Profile、推理参数与质量评估
- **Compatibility** — GPU、CUDA 与运行环境
- **Troubleshooting** — 启动、CUDA、ASR、训练与推理问题
- **Release Verification** — ZIP、SHA256 与正式发布包校验
