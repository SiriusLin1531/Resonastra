# Resonastra 模型训练指南（Training）

[项目 README](./README.md) | [快速开始](./quick-start.md) | [数据准备](./data-factory.md) | **模型训练** | [语音生成](./inference.md) | [兼容性](./compatibility.md) | [版本校验](./release-verification.md) | [English](../en/training.md) | **中文简体**

适用版本：**Resonastra v1.0.0**

> [!NOTE]
> 本指南针对 **Resonastra v1.0.0** 的正式发行行为编写。后续版本如果调整 Training 界面、训练参数、模型检查点（checkpoint）管理或 Voice Profile 契约，应以对应版本文档为准。

---

## 1. 本指南解决什么问题

Resonastra 的 Training 负责把 DataFactory 已经准备好的 Stage1 / Stage2 训练数据转换成可供 Voice Profile 使用的 Few-shot 模型产物。

如果你只使用内置 Zero-shot 模型，不需要进入 Training。Stage1-only、Stage2-only 和 Full Few-shot 路线才需要 Training。

完成与你所选路线对应的步骤后，你应该能够：

- 从正确的 DataFactory 工作目录扫描训练输入；
- 判断 Stage1 / Stage2 哪一条训练入口已经可以启动；
- 调整 v1.0.0 用户版开放的训练参数；
- 独立完成 Stage1 或 Stage2 Training；
- 理解每一次训练任务、Trainer Best 和 Active Best 的区别；
- 查看 Training Live Monitor、GPU / 显存和失败摘要；
- 停止训练并正确理解 v1.0.0 不支持训练续跑（Training Resume）的边界；
- 管理历史模型检查点与 Active Best；
- 生成与 Stage1-only、Stage2-only 或 Full Few-shot 对应的 Voice Profile；
- 将 Voice Profile 交给 Inference。

本指南不展开原始音频、ASR、数据构建、GPU / CUDA 通用兼容性、模型训练算法理论或研究型超参数调优。Training 自身的失败与恢复规则在本文对应章节说明。

---

## 2. Training 工作流与开始之前

Training 位于 DataFactory 与 Voice Profile / Inference 之间：

```text
DataFactory 训练数据
  ↓
扫描 / 验证训练入口
  ↓
Stage1 / Stage2 Training
  ↓
每次训练产生独立训练任务
  ↓
Trainer Best
  ↓
Stage1 / Stage2 Active Best
  ↓
Voice Profile
  ↓
Inference
```

### 2.1 四种路线

| 路线 | Training 需要完成的阶段 | Voice Profile 中的模型来源 |
| --- | --- | --- |
| Zero-shot | 无 | Stage1 默认模型 + Stage2 默认模型 |
| Stage1-only Few-shot | Stage1 | Stage1 Active Best + Stage2 默认模型 |
| Stage2-only Few-shot | Stage2 | Stage1 默认模型 + Stage2 Active Best |
| Full Few-shot | Stage1 + Stage2 | Stage1 Active Best + Stage2 Active Best |

Stage1 与 Stage2 的训练入口彼此独立。部分 Few-shot 路线不要求两个 Stage 都完成。

### 2.2 三个核心概念

后续操作可以用三个层次理解：

1. **数据就绪状态**：当前 Stage 是否具备可训练输入；
2. **训练任务（run）**：一次独立训练执行，以及这次执行产生的日志和模型检查点；
3. **Active Best**：当前生成 Voice Profile 时真正采用的 Few-shot 模型检查点。

> [!IMPORTANT]
> “最近一次训练”“本次 Trainer Best”和“当前 Active Best”不是同一个概念。手动锁定 Active Best 后，新的成功训练仍能产生 Trainer Best，但不会自动替换当前 Active Best。

### 2.3 继续使用同一个 DataFactory 工作目录

开始前建议先：

1. 完整解压 Resonastra v1.0.0；
2. 运行 `launch_check_env.bat` 并确认基础环境检查通过；
3. 使用 DataFactory 完成当前路线需要的训练数据；
4. 记住 DataFactory 使用的 **说话人 / 角色名** 和工作目录。

默认工作目录：

```text
user_data/{speaker_name}_factory
```

如果 DataFactory 使用默认目录规则，Training 中填写相同的 **说话人 / 角色名**，并保持 **DataFactory 工作目录** 为空即可。

如果 DataFactory 使用了自定义工作目录，则在 Training 中填写同一个目录。

> [!IMPORTANT]
> Training 不会把 DataFactory 数据复制到另一套项目目录。DataFactory 的工作目录（`work_dir`）继续作为 Few-shot 数据、训练记录和模型检查点的项目身份。

---

## 3. 启动 Training 与扫描数据

从 Resonastra 根目录运行：

```text
launch_training_ui.bat
```

默认端口：

```text
7863
```

如果端口已占用，启动器会自动选择其它可用端口，实际地址以启动终端显示为准。

主启动终端负责维持 Training WebUI，正常使用期间不要关闭。它与后面可选的独立训练日志终端不是同一个窗口；关闭日志查看器不会关闭 WebUI，也不会停止训练。

页面主要区域包括：

- **输入**
- **训练操作**
- **Checkpoint / Active Best**
- **生成 Voice Profile v2**
- **状态**
- **Training Live Monitor**
- **诊断信息**

### 3.1 扫描训练数据

左侧 **输入** 区域包含：

- **说话人 / 角色名**
- **DataFactory 工作目录**
- **扫描时刷新 DataFactory 状态**

如果使用默认工作目录：

1. 填写与 DataFactory 相同的说话人 / 角色名；
2. **DataFactory 工作目录** 保持为空；
3. 保持 **扫描时刷新 DataFactory 状态** 开启。

如果使用自定义工作目录，则直接填写同一个目录。工作目录和说话人不能同时为空。

点击：

```text
扫描训练数据
```

Training 会重新读取 DataFactory 状态并建立 / 刷新：

```text
{work_dir}/training_interface.json
```

它是 DataFactory → Training 的输入契约快照。

### 3.2 重点看 Stage1 / Stage2 训练入口

扫描后会分别看到：

```text
Stage1 Training：✅ 可以启动
Stage2 Training：✅ 可以启动
```

或：

```text
Stage1 Training：❌ 暂不可启动
Stage2 Training：❌ 暂不可启动
```

只需要让当前路线真正使用的 Stage 通过入口检查。

Stage1 启动前会确认：

- Stage1 就绪；
- 训练 / 验证 manifest（数据清单）；
- 文本前端缓存（frontend cache）；
- 语义缓存（semantic cache）；
- Stage1 输出根目录。

Stage2 启动前会确认：

- Stage2 就绪；
- Stage2 `.pt` 数据；
- 训练 / 验证集划分；
- 过滤 / 隔离契约；
- 训练 / 验证目录中存在可训练 `.pt`；
- 正式发行包所需的 Stage2 基础模型检查点（checkpoint）存在。

缺少必要输入时，Training 会拒绝启动对应 Stage，而不是带着缺失输入进入训练脚本。

页面底部的 Training Interface、Training Defaults、Live Monitor、Checkpoint Scan 等诊断 JSON 主要用于排错，普通流程优先看训练入口、Live Monitor 和 Checkpoint / Active Best 状态。

---

## 4. 用户训练参数

在 **本次训练参数（仅当前 run）** 中，v1.0.0 用户版只开放四个训练参数：

| Stage | 参数 | 默认值 |
| --- | --- | ---: |
| Stage1 | **Stage1 epochs** | `10` |
| Stage1 | **Stage1 batch_size** | `1` |
| Stage2 | **Stage2 epochs** | `20` |
| Stage2 | **Stage2 batch_size** | `1` |

输入必须是大于 0 的整数。

### 4.1 参数只影响当前训练任务

这四个输入的作用范围为 `this_run_only`：

- 只影响当前一次训练任务；
- 不会写回 `configs/training_default.yaml`；
- 下次打开页面仍从正式默认配置读取默认值。

第一次训练建议先保持：

```text
Stage1: epochs=10, batch_size=1
Stage2: epochs=20, batch_size=1
```

先确认训练、模型检查点管理和 Voice Profile 交接能够完整跑通。

> [!NOTE]
> “保持默认”是首次跑通流程的实践建议，不代表这些参数对所有数据集都必然最优。

### 4.2 batch_size 与显存建议

根据当前开发和测试经验，默认设置下 `batch_size=1` 时训练通常占用约 **2–3 GB GPU 显存**。实际占用仍会受到 Stage、样本长度、GPU / CUDA 环境和其它程序影响，因此下面仅作为保守起点，不是硬性限制。

| GPU 显存 | 建议起始 `batch_size` | 可尝试范围 | 建议 |
| ---: | ---: | ---: | --- |
| 4 GB | `1` | `1` | 保持默认，优先确保稳定 |
| 6 GB | `1` | `1–2` | 显存余量充足时可尝试 `2` |
| 8 GB | `2` | `1–2` | 可从 `2` 开始，OOM 时退回 `1` |
| 12 GB | `2` | `2–4` | 可逐步提高并观察峰值显存 |
| 16 GB | `4` | `2–6` | 可先用 `4`，稳定后再提高 |
| 24 GB 及以上 | `4` | `4–8+` | 可根据训练速度和显存峰值继续增加 |

> [!NOTE]
> 显存占用不保证随 `batch_size` 完全线性变化，也不建议长期把可用显存压到接近 100%。作为保守做法，建议至少预留约 **1–2 GB**；同时运行其它 GPU 程序时应保留更多。

出现 CUDA out of memory、训练启动失败或显存持续接近上限时，优先降低 `batch_size`，而不是修改其它发行版内部训练参数。

### 4.3 用户设置与发行版管理边界

普通用户可编辑：

- Stage1 / Stage2 epochs；
- Stage1 / Stage2 batch_size；
- 是否打开独立训练日志终端。

用户也可以管理：

- DataFactory 工作目录；
- 扫描时是否刷新 DataFactory 状态；
- Stage1 / Stage2 Active Best；
- 历史模型检查点；
- Stage1 / Stage2 是否强制使用 zero-shot；
- Profile 是否覆盖已有目录。

除此之外，Training 还包含由发行版统一管理的内部训练参数，普通用户不需要为了正常训练手工修改。

---

## 5. Stage1 / Stage2 Training

Stage1 和 Stage2 都遵循同一个基本训练任务生命周期：

```text
入口检查通过
  ↓
启动 Training
  ↓
创建新的 run_id / 独立训练任务目录
  ↓
pending → running
  ↓
训练结束
  ↓
succeeded / failed / stopped / stop_failed
  ↓
成功时发现 Trainer Best
  ↓
根据 Active Best 选择模式决定是否自动更新
```

每次启动都会创建新的训练任务，不覆盖旧训练任务。同一 Stage 已存在 `pending / running / stopping` 的活动任务时，新的同 Stage 训练会被拒绝。Stage1 与 Stage2 的活动训练任务保护分别管理；普通用户仍建议按顺序训练，避免同时争用同一块 GPU。

### 5.1 Stage1 Training

Stage1-only 和 Full Few-shot 需要 Stage1。

确认：

```text
Stage1 Training：✅ 可以启动
```

然后点击：

```text
启动 Stage1 Training
```

每次启动创建：

```text
{work_dir}/12_training/stage1/runs/<run_id>/
```

典型 checkpoints 目录包含：

```text
best_model.ckpt
last_model.ckpt
history.json
summary.json
stage1_fewshot_train_report.json
```

其中 `best_model.ckpt` 是本次训练任务的 Trainer Best，`last_model.ckpt` 是最后一个 epoch 对应的模型。正常用户路径不要求每个 epoch 都保存模型检查点。

Stage1 正常有验证数据时，Trainer Best 的主要指标是：

```text
Val Loss / Token
```

底层字段：

```text
val_loss_per_token
```

如果底层训练没有可用验证数据加载器（validation loader），Stage1 会退回使用训练损失 / token（loss/token）选择最佳模型；此时 **Checkpoint / Active Best** 中的 `Val Loss / Token` 元数据可能显示 `N/A`。正常 DataFactory → Training 路线仍建议保留有效的训练 / 验证数据。

### 5.2 Stage2 Training

Stage2-only 和 Full Few-shot 需要 Stage2。

确认：

```text
Stage2 Training：✅ 可以启动
```

然后点击：

```text
启动 Stage2 Training
```

每次启动创建：

```text
{work_dir}/12_training/stage2/runs/<run_id>/
```

Stage2 Few-shot 会自动从正式发行配置指定的默认 Stage2 模型检查点开始训练。用户版：

- 不提供基础模型检查点选择器；
- 不需要手工输入基础模型检查点路径；
- 不建议通过修改内部配置绕开这一发行契约。

Stage2 Trainer Best 的主要验证指标在 UI 中显示为：

```text
Val Loss
```

### 5.3 训练成功与 Active Best 交接是两件事

Stage1 / Stage2 的训练任务本身正常结束时，应看到：

```text
status = succeeded
```

随后训练工作进程会尝试发现本次训练任务的 `best_model.*` 并注册 Trainer Best / Active Best。

> [!IMPORTANT]
> `succeeded` 只表示训练进程正常结束，不保证 Active Best 一定已经更新。如果 Trainer Best 文件缺失或自动注册异常，训练任务仍可能保持 `succeeded`，同时记录 `auto_best_error`。训练结束后仍应检查 **Checkpoint / Active Best**。

如果当前 Stage 处于 Automatic Best 且 Trainer Best 注册成功：

```text
训练成功
  ↓
Trainer Best
  ↓
自动更新 Active Best
```

如果当前 Stage 已被手动锁定，新 Trainer Best 仍保留在自己的训练任务中，但不会自动替换当前 Active Best。

> [!NOTE]
> Stage1 与 Stage2 的活动训练任务保护彼此独立。普通用户仍建议按顺序完成训练，而不是同时占用同一块 GPU 运行两个训练任务，以减少显存争用并简化日志与资源判断。

---

## 6. 监控、停止与训练任务状态

### 6.1 Training Live Monitor

**Training Live Monitor** 是正常训练时最重要的状态区。

当状态为：

- `pending`
- `running`
- `stopping`

时，实时监控区域约每 5 秒自动刷新；没有活动训练时，高频轮询会暂停。也可以随时点击 **立即刷新监控** 强制刷新当前状态。

**最近一次训练** 会显示：

- Stage；
- 状态；
- `run_id`；
- epochs / batch_size；
- 开始时间；
- 完成或更新时间。

它表示时间上最近的训练任务，不代表当前 Active Best 一定来自这个训练任务。

**Epoch / Step 详情**用于确认当前 epoch、step 和训练是否仍在推进。

训练运行时还会显示 **GPU / 显存**；没有活动训练时 GPU 自动监控暂停。

### 6.2 可选的独立训练日志终端

选项：

```text
打开独立训练日志终端（可选）
```

默认关闭。

开启后会额外打开只读日志窗口。它：

- 只读取当前训练任务日志；
- 不拥有训练进程；
- 关闭不会停止训练；
- 不需要打开才能完成训练。

真正停止训练必须使用：

```text
停止 Stage1 Training
停止 Stage2 Training
```

### 6.3 Stop 与训练状态

停止对应 Stage 时，系统会写入停止请求（stop request）、尝试停止训练子进程 / 工作进程、更新状态，并保留已经产生的训练任务目录、日志、元数据和模型检查点。

Stop 不会删除整个训练任务。

常见状态：

| 状态 | 含义 |
| --- | --- |
| `pending` | 训练工作进程正在启动 |
| `running` | 训练进程正在运行 |
| `stopping` | 已请求停止，正在处理 |
| `succeeded` | 训练进程成功结束；仍需确认 Trainer Best / Active Best 交接 |
| `failed` | 训练或工作进程失败 |
| `stopped` | 按用户请求停止 |
| `stop_failed` | 停止流程本身未正常完成 |

已经处于终止状态的训练任务不需要再次执行 Stop。

`stopped` / `failed` 不等于 `succeeded`。自动 Trainer Best → Active Best 注册只发生在成功完成的训练任务上。失败或停止的训练任务如果留下历史模型检查点，它们仍可能显示在列表中，但这不表示该训练任务已通过成功条件。

### 6.4 v1.0.0 没有用户级训练续跑（Training Resume）

Training v1.0.0 用户 UI 不支持从中断训练任务的优化器状态（optimizer state）原地继续训练。

这与 DataFactory Stage2 基于已有产物的续跑（Resume）不同。

如果一次训练中断，需要重新训练时，应启动一个新的训练任务。

> [!IMPORTANT]
> “切换历史模型检查点 / Active Best”是模型选择；“Training Resume”是训练状态续跑。两者不要混淆。

### 6.5 训练失败摘要与日志

如果最近的训练任务失败，UI 会尝试从标准错误（stderr）/ 标准输出（stdout）中提取异常堆栈（traceback）、`CUDA out of memory`、RuntimeError / ValueError 等关键错误。

优先查看：

```text
训练失败摘要
```

需要进一步排查时再查看对应训练任务的：

```text
training_stderr.log
training_stdout.log
training_command.json
```

如果怀疑本次实际用了什么 epochs / batch_size，优先查看 `training_command.json`。

---

## 7. 训练任务目录、Trainer Best 与 Active Best

Training 根目录：

```text
{work_dir}/12_training/
```

Stage1 / Stage2 分开保存：

```text
12_training/
├── stage1/
└── stage2/
```

### 7.1 每个训练任务都有独立目录

典型结构：

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

不同训练任务之间互不覆盖。

### 7.2 Stage 根目录的稳定状态

```text
12_training/<stage>/
```

还会保存：

| 文件 | 用户层含义 |
| --- | --- |
| `training_run_state.json` | 当前 / 最近 Stage 训练任务的 Stage 级状态副本 |
| `latest_run.json` | 最近启动的训练任务 |
| `best_model.pt` | 当前 Active Best 的稳定工作副本 |
| `active_best.json` | 当前 Active Best 来源 |
| `best_checkpoint.json` | Active Best 相关元数据 |

`training_command.json` 记录正式默认参数、本次 UI 覆盖值、实际生效参数、训练命令、日志路径、日志终端策略以及 Stage2 基础模型检查点等运行上下文。

### 7.3 Trainer Best 与 Active Best

**Trainer Best**：

> 某个独立训练任务根据自身验证指标选出的最佳模型检查点。

它属于：

```text
runs/<run_id>/
```

不同训练任务可以各自拥有 Trainer Best。

**Active Best**：

> 当前 Stage 生成 Voice Profile 时实际采用的 Few-shot 模型检查点。

系统会把当前 Active Best 复制为：

```text
12_training/<stage>/best_model.pt
```

并记录来源训练任务、来源模型检查点、epoch、验证指标和选择模式。

### 7.4 Automatic Best 与手动锁定

未人工锁定时，通常处于 Automatic Best：

```text
训练任务成功
  ↓
Trainer Best
  ↓
复制到 Stage 级 `best_model.pt`
  ↓
更新 Active Best
```

如果你希望固定使用某个历史模型检查点，可从列表中选择后点击：

```text
手动锁定 Stage1 Active Best
```

或：

```text
手动锁定 Stage2 Active Best
```

手动锁定后：

- 当前选择成为 Active Best；
- 选择模式变为手动（Manual）；
- 后续训练仍可以正常进行并产生新的 Trainer Best；
- 新 Trainer Best 不会自动覆盖当前 Active Best。

如果希望恢复自动选择，点击：

```text
恢复 Stage1 Automatic Best
恢复 Stage2 Automatic Best
```

恢复操作会从当前 Stage 扫描到的历史 `best_model*` 中选择最新项，复制为新的 Active Best，并恢复自动（Automatic）模式。如果当前 Stage 找不到训练器生成的 `best_model*`，恢复操作会失败。

---

## 8. 历史模型检查点与多次训练

### 8.1 刷新与选择历史模型检查点

**Checkpoint / Active Best** 区域提供：

```text
刷新 checkpoint 列表
```

历史列表会显示训练任务、模型检查点名称、epoch、验证指标、文件大小和保护标签。

每次新的训练都会获得新的 `run_id`，不会覆盖旧训练任务，因此可以保留多次训练结果再比较。

不同训练任务实际使用的 epochs / batch_size 也会保存在各自的 `training_command.json` 中。

一种常见用法是：

```text
当前满意模型
  ↓
手动锁定 Active Best
  ↓
继续创建新的训练任务
  ↓
比较结果
  ↓
满意后再切换 Active Best
```

### 8.2 永久删除与保护规则

删除操作一次只处理一个历史模型检查点。必须先勾选对应的永久删除确认框。

v1.0.0 会阻止删除：

1. Stage 根目录的稳定 Active Best 工作副本；
2. 当前 Active Best 的来源模型检查点；
3. 当前正在运行的训练任务中的模型检查点；
4. 不属于可管理历史训练任务范围的文件。

如果某个模型检查点是当前 Active Best 的来源，应先切换到其它 Active Best，再刷新列表。

历史 Trainer Best 在已经不是 Active Best 来源、也不属于活动训练任务时可以删除，但风险更高，因为删除后该训练任务将失去自己的最佳模型文件。

> [!WARNING]
> 模型检查点删除是永久文件删除，不是隐藏、移出列表或清理缓存。模型文件不能从 history / summary 自动恢复。

删除模型检查点不会删除整个训练任务记录，history、summary、run_state、标准输出 / 标准错误日志和其它元数据会继续保留。

删除操作还会写入：

```text
12_training/checkpoint_deletion_history.jsonl
```

记录删除时间、Stage、训练任务、模型检查点名称、大小和相关验证信息。

---

## 9. 生成 Voice Profile v2

完成所选路线的 Training 并确认 Active Best 后，在 **生成 Voice Profile v2** 区域创建 Inference 使用的 Voice Profile。

Voice Profile 是 Training → Inference 的正式交接。

### 9.1 profile_name 与输出目录

字段：

```text
profile_name
```

建议填写容易识别的名称。

如果留空：

1. 优先使用当前 **说话人 / 角色名**；
2. 如果说话人为空、但使用自定义工作目录，则使用 `default_user`。

最终目录名会安全化：字母、数字、下划线和连字符保留，其它字符替换为下划线。

输出位置：

```text
user_profiles/<profile_name>/
```

目录会保存 Profile 配置、Profile manifest，以及当前需要的 Stage1 / Stage2 Few-shot 模型检查点副本。

### 9.2 默认模型来源与强制 zero-shot

正常情况下：

- Stage1 有 Active Best → 使用 Stage1 Few-shot；
- Stage1 无 Active Best → 回退 Stage1 Zero-shot 默认模型；
- Stage2 有 Active Best → 使用 Stage2 Few-shot；
- Stage2 无 Active Best → 回退 Stage2 Zero-shot 默认模型。

UI 还提供：

```text
Stage1 强制使用 zero-shot
Stage2 强制使用 zero-shot
```

用于明确构建 Partial Few-shot。

| Profile 类型 | Stage1 | Stage2 | 强制 zero-shot 设置 |
| --- | --- | --- | --- |
| `base_zeroshot` | 默认模型 | 默认模型 | 两个 Stage 都使用默认模型 |
| `few_shot_stage1_only` | Active Best | 默认模型 | Stage2 开启 |
| `few_shot_stage2_only` | 默认模型 | Active Best | Stage1 开启 |
| `few_shot_dual` | Active Best | Active Best | 两个 Stage 都关闭 |

普通 Zero-shot 用户通常不需要为了默认模型专门进入 Training。

### 9.3 覆盖已有 Profile

选项：

```text
覆盖已有 profile
```

默认开启。

开启时允许写入已有 Profile 目录，并重写 Profile 配置 / manifest，同时复制本次实际需要的 Few-shot 模型检查点。

覆盖已有目录不会自动创建旧 Profile 版本备份。

> [!WARNING]
> 如果希望保留旧 Profile 作为可回退版本，建议使用新的 `profile_name`，而不是直接覆盖旧 Profile。

### 9.4 Profile 成功条件

生成成功后页面会显示：

- profile 名称；
- profile 类型；
- Stage1 来源；
- Stage2 来源；
- 是否可用于推理。

正常发行环境下应看到：

```text
可用于推理：True
```

最终完成条件可以归纳为：

```text
目标 Stage Training succeeded
  ↓
目标 Active Best 正确
  ↓
Voice Profile 生成成功
  ↓
可用于推理：True
```

> [!IMPORTANT]
> “训练按钮执行过”不是 Training 的最终完成条件。进入 Inference 前，应确认目标 Stage 成功、Active Best 正确、Voice Profile 已生成并可用于推理。

---

## 10. 常见问题与安全边界

| 容易误解的行为 | 正确理解 |
| --- | --- |
| 最新训练任务 = 当前 Active Best 来源 | 不一定。手动锁定后尤其不是 |
| Trainer Best = Active Best | 不等同。Trainer Best 属于单个训练任务；Active Best 属于 Stage 当前选择 |
| `succeeded` = Active Best 一定已更新 | 不一定。还应检查 Checkpoint / Active Best 和可能的 `auto_best_error` |
| Stop 会删除训练任务 | 不会。已有目录、日志、元数据和模型检查点会保留 |
| 关闭独立日志终端 = 停止训练 | 错。真正停止必须使用 Stage1 / Stage2 Stop 按钮 |
| v1.0.0 支持 Training Resume | 不支持。重新训练会创建新的训练任务 |
| 模型检查点删除只是隐藏 | 错。删除是永久文件删除 |
| Active Best source 可以直接删除 | 不可以，必须先切换 Active Best |
| Stage1-only / Stage2-only 必须等另一个 Stage | 不需要，两个路线都是正式支持的 Partial Few-shot |
| 训练目录可以直接作为 Inference 用户入口 | 不建议。Voice Profile 才是正式 Training → Inference 交接 |

---

## 11. 故障与恢复提示

### 11.1 启动失败

如果点击 Stage1 / Stage2 后立即失败，优先检查：

1. 是否扫描了正确的 DataFactory 工作目录；
2. 对应 Stage 是否显示 **可以启动**；
3. epochs / batch_size 是否为正整数；
4. 同一 Stage 是否已有活动训练任务；
5. Stage2 基础模型检查点是否完整。

### 11.2 运行中失败

如果训练任务已启动后进入 `failed`：

1. 查看 **训练失败摘要**；
2. 查看 `training_stderr.log`；
3. 再查看 `training_stdout.log`；
4. 如需确认实际参数，查看 `training_command.json`。

如果出现 CUDA unavailable、`CUDA out of memory` 或类似 GPU 错误，可以先降低 `batch_size`。GPU / CUDA 支持范围应参考 [兼容性指南](./compatibility.md)，不建议为了绕过错误直接修改发行版内部模型配置。

### 11.3 Active Best 没变化

如果训练成功但 Active Best 没变化，先检查选择模式。

如果处于手动锁定，这是预期行为。需要恢复自动更新时，使用：

```text
恢复 Stage1 Automatic Best
恢复 Stage2 Automatic Best
```

### 11.4 模型检查点无法删除

常见原因：

- 当前模型检查点是 Active Best 来源；
- 模型检查点属于活动训练任务；
- 列表状态已过期，需要先 **刷新 checkpoint 列表**；
- 没有勾选永久删除确认；
- 目标文件不属于受支持的历史模型检查点范围。

---

## 12. 下一步与相关文档

完成 Training 后，下一步是使用 Voice Profile 进入 Inference：

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
- [数据准备指南（DataFactory）](./data-factory.md)
- [语音生成指南（Inference）](./inference.md)
- [兼容性指南](./compatibility.md)
