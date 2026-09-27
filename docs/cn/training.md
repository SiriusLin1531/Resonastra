# Resonastra Training 使用指南

[项目 README](./README.md) | [快速开始](./quick-start.md) | [DataFactory](./data-factory.md) | **中文简体**

适用版本：**Resonastra v1.0.0**

> [!NOTE]
> 本指南针对 **Resonastra v1.0.0** 的正式发行行为编写。后续版本如果调整 Training 界面、训练参数、模型检查点（checkpoint）管理或 Voice Profile 契约，应以对应版本文档为准。

---

## 1. 本指南解决什么问题

Resonastra 的 Training 负责把 DataFactory 已经准备好的 Stage1 / Stage2 训练数据转换成可供 Voice Profile 使用的 Few-shot 模型产物。

如果你只使用内置 Zero-shot 模型，不需要进入 Training。只有在准备 Stage1-only、Stage2-only 或 Full Few-shot 时，才需要使用本指南。

完成与你所选路线对应的步骤后，你应该能够：

- 从正确的 DataFactory 工作目录启动 Training；
- 判断 Stage1 / Stage2 哪一条训练入口已经可用；
- 理解 v1.0.0 用户版可以修改哪些训练参数；
- 独立完成 Stage1 或 Stage2 Training；
- 判断一次训练任务是否真正成功；
- 使用 Training Live Monitor 查看训练进度、GPU / 显存和失败摘要；
- 理解每一次训练任务的独立目录、日志与模型检查点；
- 区分 **Trainer Best**、历史 checkpoint 与 **Active Best**；
- 手动锁定 Active Best，或恢复 Automatic Best；
- 在明确风险后安全删除单个历史模型检查点；
- 根据 Stage1-only、Stage2-only 或 Full Few-shot 生成正确的 Voice Profile；
- 将生成的 Voice Profile 交给 Inference。

本指南不展开：

- 原始音频、ASR、人工校对和训练数据构建细节；
- GPU / CUDA 的通用兼容性政策；
- Stage1 / Stage2 模型结构和训练算法理论；
- 面向研究实验的系统性超参数调优；
- Inference 的生成参数与质量评估细节；
- 完整的按症状组织的故障排查。

这些内容分别由 DataFactory、Compatibility、Inference、Troubleshooting 或后续高级文档承担。

---

## 2. Training 在 Resonastra 工作流中的位置

Training 位于 DataFactory 与 Voice Profile / Inference 之间：

```text
DataFactory 训练数据
  ↓
训练输入检查
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

### 2.1 与四种路线的关系

| 路线 | Training 需要完成的阶段 | Voice Profile 中的模型来源 |
| --- | --- | --- |
| Zero-shot | 无 | Stage1 默认模型 + Stage2 默认模型 |
| Stage1-only Few-shot | Stage1 | Stage1 Active Best + Stage2 默认模型 |
| Stage2-only Few-shot | Stage2 | Stage1 默认模型 + Stage2 Active Best |
| Full Few-shot | Stage1 + Stage2 | Stage1 Active Best + Stage2 Active Best |

Stage1 与 Stage2 的训练入口是独立的。部分 Few-shot 路线不要求两个阶段都完成。

### 2.2 理解 Training 的三个核心概念

后续所有操作都可以用三个层次理解：

1. **数据就绪状态**：当前 Stage 是否已经具备可训练输入；
2. **训练任务（run）**：一次独立的训练执行，以及这次执行产生的日志和模型检查点；
3. **Active Best**：当前真正会被复制进 Voice Profile 的 Few-shot 模型检查点。

> [!IMPORTANT]
> “最近一次训练”“本次 Trainer Best”和“当前 Active Best”不是同一个概念。尤其在手动锁定 Active Best 后，最新成功的训练任务可以产生新的 Trainer Best，但不会自动替换你已经锁定的 Active Best。

---

## 3. 开始之前

开始 Training 前，建议已经完成：

1. 完整解压 Resonastra v1.0.0；
2. 运行 `launch_check_env.bat` 并确认基础环境检查通过；
3. 使用 DataFactory 完成当前路线需要的训练数据；
4. 记住 DataFactory 使用的 **说话人 / 角色名** 和工作目录。

如果你还没有准备训练数据，请先阅读 [DataFactory 使用指南](./data-factory.md)。

### 3.1 继续使用同一个 DataFactory 工作目录

如果 DataFactory 使用默认目录规则：

```text
user_data/{speaker_name}_factory
```

那么 Training 中填写同一个 **说话人 / 角色名**，并保持 **DataFactory 工作目录** 为空即可。

如果 DataFactory 使用了自定义工作目录，则在 Training 中填写同一个目录。

> [!IMPORTANT]
> Training 不会把 DataFactory 数据复制到另一套项目目录。DataFactory 的 `work_dir` 继续作为这一套 Few-shot 数据、训练记录和模型检查点的项目身份。

### 3.2 不同路线需要的数据

| 路线 | Training 前需要的数据 |
| --- | --- |
| Stage1-only Few-shot | Stage1 数据已就绪 |
| Stage2-only Few-shot | Stage2 数据已就绪 |
| Full Few-shot | Stage1 + Stage2 数据都已就绪 |
| Zero-shot | 不需要 Training |

因此，看到“完整训练数据”尚未全部就绪并不一定代表不能开始训练。

例如：

- Stage1-only：只要 Stage1 Training 显示可以启动即可；
- Stage2-only：只要 Stage2 Training 显示可以启动即可。

---

## 4. 启动 Training

从 Resonastra 根目录运行：

```text
launch_training_ui.bat
```

默认地址使用端口：

```text
7863
```

如果端口已被占用，启动器会自动寻找可用端口。实际地址以启动终端显示的信息为准。

### 4.1 不要关闭 WebUI 主启动终端

运行 `launch_training_ui.bat` 后会出现一个主终端窗口，它负责维持 Training WebUI。

正常使用期间不要关闭这个终端。

它和后面可选的：

```text
打开独立训练日志终端（可选）
```

不是同一个东西。

### 4.2 页面主要区域

Resonastra · Training 页面主要分为：

- **输入**
- **训练操作**
- **Checkpoint / Active Best**
- **生成 Voice Profile v2**
- **状态**
- **Training Live Monitor**
- **诊断信息**

第一次完整跑通训练时，主流程主要使用：

```text
输入
  ↓
扫描训练数据
  ↓
Stage1 / Stage2 Training
  ↓
Training Live Monitor
  ↓
Checkpoint / Active Best
  ↓
生成 Voice Profile v2
```

---

## 5. 扫描 DataFactory 数据

### 5.1 填写输入

左侧 **输入** 区域包含：

- **说话人 / 角色名**
- **DataFactory 工作目录**
- **扫描时刷新 DataFactory 状态**

如果你使用默认工作目录规则：

1. 填写与 DataFactory 相同的 **说话人 / 角色名**；
2. **DataFactory 工作目录** 保持为空；
3. 保持 **扫描时刷新 DataFactory 状态** 开启。

如果使用自定义工作目录，则直接填写该目录。

工作目录和说话人不能同时为空。

### 5.2 点击扫描训练数据

点击：

```text
扫描训练数据
```

Training 会重新读取 DataFactory 的当前状态，并建立 / 刷新：

```text
{work_dir}/training_interface.json
```

这个文件是 DataFactory → Training 的输入契约快照。

### 5.3 重点看“训练入口”

扫描后，页面会分别显示：

```text
Stage1 Training：✅ 可以启动
Stage2 Training：✅ 可以启动
```

或：

```text
Stage1 Training：❌ 暂不可启动
Stage2 Training：❌ 暂不可启动
```

只需要让当前路线实际使用的 Stage 通过入口检查。

### 5.4 Stage1 会检查什么

Stage1 Training 启动前会确认 DataFactory 已经产生并保留必要的 Stage1 训练产物，包括：

- Stage1 就绪状态；
- 训练 manifest；
- 验证 manifest；
- 文本前端缓存（frontend cache）；
- 语义缓存（semantic cache）；
- Stage1 输出根目录。

如果这些数据不完整，Training 会拒绝启动 Stage1，而不是带着缺失输入进入训练脚本。

### 5.5 Stage2 会检查什么

Stage2 Training 启动前会确认：

- Stage2 就绪状态；
- Stage2 `.pt` 数据已经完成；
- 训练 / 验证划分（train / val split）已完成；
- 过滤 / 隔离契约（filter / quarantine contract）已完成；
- train / val 目录中存在可训练 `.pt`；
- 正式发行包所需的 Stage2 基础 checkpoint 存在。

如果 DataFactory Stage2 流程尚未完成 split / filter，Training 会把它显示为入口未就绪。

### 5.6 诊断 JSON 不是普通用户必读内容

页面底部的 **诊断信息** 会提供 Training Interface、Training Defaults、Live Monitor、Checkpoint Scan 等 JSON。

这些内容主要用于排错。

普通训练流程只需要优先确认：

- 数据准备状态；
- Stage1 / Stage2 Training 是否可以启动；
- Training Live Monitor 中的训练任务状态；
- Checkpoint / Active Best 状态。

---

## 6. 用户版训练参数

在 **本次训练参数（仅当前 run）** 中，v1.0.0 用户版只开放：

### Stage1

- **Stage1 epochs**
- **Stage1 batch_size**

### Stage2

- **Stage2 epochs**
- **Stage2 batch_size**

当前默认值为：

| Stage | epochs | batch_size |
| --- | ---: | ---: |
| Stage1 | `10` | `1` |
| Stage2 | `20` | `1` |

### 6.1 修改只影响当前 run

这四个输入都只对当前一次训练任务生效，内部作用范围为 `this_run_only`。修改后：

- 只影响这一次训练任务；
- 不会写回 `configs/training_default.yaml`；
- 下一次打开页面时仍会从正式默认配置读取默认值。

输入必须是大于 0 的整数。

### 6.2 第一次训练建议保持默认

第一次使用时建议：

- Stage1 保持 `epochs=10, batch_size=1`；
- Stage2 保持 `epochs=20, batch_size=1`；
- 先确认完整训练、模型检查点管理和 Voice Profile 交接都能正常完成。

> [!NOTE]
> “保持默认”是首次跑通流程的实践建议，不代表这些参数对所有数据集都必然最优。研究型调参不属于本基础用户指南的范围。

### 6.3 batch_size 与显存建议

根据当前开发和测试经验，在默认设置下，`batch_size=1` 时训练通常会占用大约 **2–3 GB GPU 显存**。实际占用仍会受到 Stage、样本长度、GPU / CUDA 环境以及同时运行的其它程序影响，因此下面的数值只适合作为**保守的起始建议**，不是硬性限制。

| GPU 显存 | 建议起始 `batch_size` | 可尝试范围 | 建议 |
| ---: | ---: | ---: | --- |
| 4 GB | `1` | `1` | 保持默认，优先确保训练稳定 |
| 6 GB | `1` | `1–2` | 可在显存余量充足时尝试 `2` |
| 8 GB | `2` | `1–2` | 建议从 `2` 开始，出现 OOM 时退回 `1` |
| 12 GB | `2` | `2–4` | 可逐步提高并观察峰值显存 |
| 16 GB | `4` | `2–6` | 建议先用 `4`，稳定后再尝试提高 |
| 24 GB 及以上 | `4` | `4–8+` | 可根据实际训练速度和显存峰值继续增加 |

> [!NOTE]
> 显存占用不会保证随 `batch_size` 完全线性变化。即使显卡总显存较大，也不建议把可用显存长期压到接近 100%。实际设置时应为桌面显示、CUDA 上下文和其它进程留出一定余量；作为保守做法，建议至少预留约 **1–2 GB**，如果系统还同时运行其它 GPU 程序，则应保留更多。

如果提高 `batch_size` 后出现 CUDA out of memory、训练启动失败或显存持续接近上限，请优先降低 `batch_size`，而不是修改其它内部训练参数。

---

## 7. 可选的独立训练日志终端

训练参数下方提供：

```text
打开独立训练日志终端（可选）
```

默认关闭。

如果开启，在启动 Stage1 或 Stage2 后会额外打开一个日志查看窗口。

这个窗口：

- 只读取当前训练任务的日志；
- 不拥有训练进程；
- 关闭它不会停止训练；
- 不需要打开它才能完成训练。

如果你只想观察正常进度，页面右侧的 **Training Live Monitor** 通常已经足够。

> [!IMPORTANT]
> 关闭独立日志终端不等于停止训练。真正停止训练必须使用 **停止 Stage1 Training** 或 **停止 Stage2 Training**。

---

## 8. Stage1 Training

Stage1-only Few-shot 和 Full Few-shot 需要完成 Stage1。

### 8.1 启动 Stage1

确认：

```text
Stage1 Training：✅ 可以启动
```

然后点击：

```text
启动 Stage1 Training
```

启动成功后，页面会显示本次训练任务的：

- `run_id`；
- 本次实际生效的 epochs；
- 本次实际生效的 batch_size；
- 是否开启独立日志终端。

### 8.2 每次启动都会创建新的训练任务

Stage1 不会覆盖上一轮训练目录。

每次启动都会创建：

```text
{work_dir}/12_training/stage1/runs/<run_id>/
```

旧训练任务会继续保留。

如果当前已经有一个正在运行的 Stage1 训练任务，系统会拒绝再次启动 Stage1，直到当前 run 结束或被停止。

### 8.3 Stage1 训练任务会产生什么

典型 Stage1 训练任务的 `checkpoints/` 目录会保存：

```text
best_model.ckpt
last_model.ckpt
history.json
summary.json
stage1_fewshot_train_report.json
```

其中：

- `best_model.ckpt`：本次训练任务根据最佳指标保存的 Trainer Best；
- `last_model.ckpt`：最后一个 epoch 对应的模型；
- `history.json`：逐 epoch 训练 / 验证记录；
- `summary.json`：本次训练总结；
- `stage1_fewshot_train_report.json`：训练任务报告。

正常用户路径不要求每个 epoch 都保存 checkpoint。

### 8.4 Stage1 Trainer Best 看什么指标

正常存在验证数据时，Stage1 的主要最佳模型指标为：

```text
Val Loss / Token
```

底层对应：

```text
val_loss_per_token
```

同一个训练任务内，验证指标改善时会更新该任务的 Trainer Best。

如果某次底层训练没有可用验证数据加载器（validation loader），Stage1 训练器会退回使用训练 loss/token 选择最佳模型；这时 **Checkpoint / Active Best** 面板中的 `Val Loss / Token` 元数据可能显示为 `N/A`。正常 DataFactory → Training 路线仍建议保留有效 train / val 数据。

### 8.5 Stage1 成功条件

不要只根据“训练开始过”判断成功。

训练任务本身正常完成时应看到：

```text
status = succeeded
```

随后训练工作进程会尝试发现本次训练任务的 `best_model.*` 并注册 Trainer Best / Active Best。

> [!IMPORTANT]
> `succeeded` 表示训练进程正常结束，但它本身不等于“Active Best 一定已经更新”。如果 Trainer Best 文件缺失，或自动注册发生异常，训练任务仍可能保持 `succeeded`，同时在训练任务状态中记录 `auto_best_error`。因此完成 Stage1 后还应检查 **Checkpoint / Active Best**。

如果 Stage1 当前处于 Automatic Best，并且 Trainer Best 注册成功：

```text
Stage1 训练任务成功
  ↓
Trainer Best
  ↓
自动更新 Stage1 Active Best
```

如果 Stage1 Active Best 已经被手动锁定，本次 Trainer Best 仍会保留在自己的训练任务目录中，但不会自动替换当前 Active Best。

---

## 9. Stage2 Training

Stage2-only Few-shot 和 Full Few-shot 需要完成 Stage2。

### 9.1 启动 Stage2

确认：

```text
Stage2 Training：✅ 可以启动
```

然后点击：

```text
启动 Stage2 Training
```

### 9.2 Stage2 基础 checkpoint 不需要手工选择

Stage2 Few-shot 会自动从正式发行配置指定的默认 Stage2 checkpoint 初始化。

用户版：

- 不提供 基础 checkpoint 选择器；
- 不需要手工输入 base checkpoint 路径；
- 不建议通过修改内部配置绕开这一发行契约。

### 9.3 Stage2 同样使用独立训练任务

每次启动都会创建：

```text
{work_dir}/12_training/stage2/runs/<run_id>/
```

不会覆盖之前的 Stage2 训练任务。

同一时间已经有活动中的 Stage2 训练任务 时，新的 Stage2 启动会被拒绝。

### 9.4 Stage2 Trainer Best

Stage2 的主要验证指标在 UI 中显示为：

```text
Val Loss
```

成功训练会在自己的训练任务目录中产生 Trainer Best。

### 9.5 Stage2 成功条件

run 本身正常完成时应确认：

```text
status = succeeded
```

随后训练工作进程会尝试发现本次训练任务的 `best_model.*` 并注册 Trainer Best / Active Best。

因此 Stage2 也应把“训练任务成功”和“Active Best 交接成功”分开检查。完成训练后应在 **Checkpoint / Active Best** 中确认目标模型已经处于预期状态。

如果 Stage2 当前处于 Automatic Best，并且 Trainer Best 注册成功：

```text
Stage2 训练任务成功
  ↓
Trainer Best
  ↓
自动更新 Stage2 Active Best
```

如果 Stage2 当前已被手动锁定，则新 Trainer Best 不会覆盖当前 Active Best。

> [!NOTE]
> Stage1 与 Stage2 的 活动训练任务保护是分别管理的。对普通用户，仍建议按顺序完成训练，而不是同时占用同一块 GPU 运行两个训练任务，这样更容易避免显存争用，也更容易判断每个训练任务的资源与日志状态。

---

## 10. 使用 Training Live Monitor

Training 页面右侧的 **Training Live Monitor** 是正常训练时最重要的状态区。

### 10.1 自动刷新

训练处于：

- `pending`
- `running`
- `stopping`

时，训练实时监控区域会以约 5 秒的间隔自动刷新。

当前没有活动训练时，高频轮询会暂停。

### 10.2 最近一次训练

**最近一次训练** 会显示：

- Stage；
- 状态；
- `run_id`；
- epochs / batch_size；
- 开始时间；
- 完成或更新时间。

这里显示的是 Stage1 / Stage2 中时间上最近的训练任务，不等同于“当前 Active Best 来自这个 run”。

### 10.3 Epoch / Step 详情

训练运行时可以在：

```text
Epoch / Step 详情
```

查看当前进度快照。

它用于回答：

- 当前训练到哪个 epoch；
- 当前 step 进度；
- 训练是否仍在推进。

### 10.4 GPU / 显存

训练运行时 UI 会主动查询：

```text
GPU / 显存
```

当前没有训练任务时 GPU 自动监控会暂停。

Training Guide 只解释这个监控面板；具体 GPU / CUDA 支持范围由 Compatibility 文档负责。

### 10.5 训练失败摘要

如果最近的训练任务进入失败状态，UI 会尝试从标准错误（stderr）/ 标准输出（stdout）中提取：

- Python 异常堆栈（traceback）；
- CUDA 显存不足（`CUDA out of memory`）；
- RuntimeError / ValueError 等异常；
- 其它关键错误行。

优先查看：

```text
训练失败摘要
```

再根据需要打开该训练任务的完整标准输出 / 标准错误日志。

### 10.6 手动刷新

可以随时点击：

```text
立即刷新监控
```

强制刷新当前状态。

---

## 11. 停止训练、中断与重新训练

Training 提供两个独立按钮：

```text
停止 Stage1 Training
停止 Stage2 Training
```

### 11.1 Stop 会做什么

停止对应 Stage 时，系统会：

1. 写入 stop request；
2. 尝试停止当前训练的子进程 / 工作进程；
3. 更新训练任务状态；
4. 保留该训练任务已经产生的目录、日志、元数据和已有 checkpoint 文件。

Stop 不会删除整个训练任务。

### 11.2 已经结束的训练任务不需要再次 Stop

如果训练任务已经处于：

- `succeeded`
- `failed`
- `stopped`
- `stop_failed`

再次执行 Stop 不会重新停止一个已经结束的任务。

### 11.3 stopped 不等于 succeeded

被用户停止的训练任务会进入停止状态，而不是成功状态。

自动 Trainer Best → Active Best 注册只发生在成功完成的 run 上。

如果 状态为 `stopped` / `failed` 的训练任务已经留下某些历史 checkpoint，它们可能仍然出现在 checkpoint 列表中；是否手动使用其中某个 checkpoint 是后续明确的人工选择，不等于这次 run 已经通过成功条件。

### 11.4 v1.0.0 没有用户级 Training Resume

这一点和 DataFactory Stage2 Resume 不同。

DataFactory 的 Resume 是：

> 根据已完成数据产物跳过可安全复用的步骤。

Training v1.0.0 用户 UI 没有：

> 从某个中断训练任务的优化器状态（optimizer state）原地继续训练。

因此如果一次训练被中断，希望重新训练时，应重新启动一个新的训练任务。

> [!IMPORTANT]
> “切换历史 checkpoint / Active Best”是模型选择行为；“Training Resume”是训练状态续跑行为。两者不要混淆。

---

## 12. 训练任务目录与日志

Training 根目录位于：

```text
{work_dir}/12_training/
```

Stage1 与 Stage2 分开保存：

```text
12_training/
├── stage1/
└── stage2/
```

### 12.1 每个独立训练任务

典型目录：

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

### 12.2 Stage 根目录的稳定状态

在：

```text
12_training/<stage>/
```

还会看到一组不属于单个历史 run 的稳定文件，例如：

```text
training_run_state.json
latest_run.json
best_model.pt
active_best.json
best_checkpoint.json
```

它们的职责不同：

- `training_run_state.json`：当前 / 最近 Stage 训练任务的 Stage 级状态副本；
- `latest_run.json`：指向最近启动的训练任务；
- `best_model.pt`：当前 Active Best 的稳定工作副本；
- `active_best.json`：记录当前 Active Best 来自哪个 run / checkpoint；
- `best_checkpoint.json`：保存 Active Best 相关元数据。

### 12.3 training_command.json

每个训练任务的：

```text
training_command.json
```

会记录：

- 正式默认参数；
- 本次 UI 覆盖值；
- 本次实际生效参数；
- 实际训练命令；
- 标准输出 / 标准错误日志路径；
- 日志终端策略；
- Stage2 基础 checkpoint 等运行上下文。

如果你怀疑“这次到底用了什么 epochs / batch_size”，优先查看这里，而不是只回忆 UI 中曾填写过什么。

---

## 13. 理解 Trainer Best 与 Active Best

这是 Training 中最容易混淆、也最重要的概念。

### 13.1 Trainer Best

**Trainer Best** 是：

> 某一次独立训练任务根据该任务的验证指标选出的本轮最佳模型检查点。

它属于：

```text
runs/<run_id>/
```

因此不同训练任务可以各自拥有自己的 Trainer Best。

### 13.2 Active Best

**Active Best** 是：

> 当前 Stage 在生成 Voice Profile 时实际采用的 Few-shot checkpoint。

系统会把当前 Active Best 复制为稳定路径：

```text
12_training/<stage>/best_model.pt
```

并记录：

- 来源训练任务；
- 来源模型检查点；
- checkpoint 所在 epoch；
- 验证指标；
- 选择模式。

### 13.3 Automatic Best

首次正常训练时，通常处于 Automatic Best 自动选择模式：

```text
训练任务成功
  ↓
产生 Trainer Best
  ↓
复制到 Stage-level best_model.pt
  ↓
更新 Active Best
```

只要没有人工锁定，后续新的成功 run 也可以继续更新 Active Best。

### 13.4 手动锁定 Active Best

如果你认为某个历史 checkpoint 的实际效果更适合当前用途，可以从历史列表中选择它，然后点击：

```text
手动锁定 Stage1 Active Best
```

或：

```text
手动锁定 Stage2 Active Best
```

手动锁定后：

- 当前选择会成为 Active Best；
- 选择模式变为 Manual；
- 后续新的训练任务仍然可以正常训练；
- 新的训练任务仍然可以产生新的 Trainer Best；
- 但不会自动覆盖当前手动选择。

> [!NOTE]
> 历史列表可能包含 Trainer Best、Last checkpoint 或其它历史 checkpoint。手动锁定意味着你主动覆盖自动选择，因此应结合验证指标和实际推理试听做决定。

### 13.5 恢复 Automatic Best

如果希望重新让系统跟随 Trainer Best，点击：

```text
恢复 Stage1 Automatic Best
```

或：

```text
恢复 Stage2 Automatic Best
```

恢复操作会从当前 Stage 扫描到的历史 `best_model*` 中选取最新项，复制为新的 Active Best，并把选择模式恢复为 Automatic。如果当前 Stage 找不到 训练器生成的 `best_model*`，恢复操作会失败。

恢复成功后，后续成功训练产生的新 Trainer Best 可以继续自动更新 Active Best。

---

## 14. 历史 checkpoint 与永久删除

Checkpoint / Active Best 区域提供：

```text
刷新 checkpoint 列表
```

以及：

- **Stage1 历史 checkpoint**
- **Stage2 历史 checkpoint**

历史列表会显示训练任务、checkpoint 名称、epoch、验证指标、文件大小以及保护标签。

### 14.1 删除一次只处理一个历史 checkpoint

选择 checkpoint 后，可以展开：

```text
删除 Stage1 checkpoint
```

或：

```text
删除 Stage2 checkpoint
```

必须先勾选对应的永久删除确认框，再执行删除。

### 14.2 哪些 checkpoint 不能删除

v1.0.0 会阻止删除：

1. Stage 根目录的稳定 Active Best 工作副本；
2. 当前 Active Best 的 source checkpoint；
3. 当前正在训练任务中的 checkpoint；
4. 不属于可管理历史训练任务范围的文件。

如果某个 checkpoint 是当前 Active Best source，而你确实想删除它，应先切换到其它 Active Best，再重新刷新列表。

### 14.3 历史 Trainer Best 可以删除，但风险更高

历史训练任务的 Trainer Best 并不是永远禁止删除。

如果它已经不是 Active Best source，也不属于正在运行的训练任务，则可以在确认后删除。

但是 UI 会把它标记为高风险，因为：

- 这是该训练任务的最佳模型文件；
- 删除后模型文件本身不可恢复；
- 该训练任务将失去自己的 Trainer Best 文件。

> [!WARNING]
> 删除历史 checkpoint 是永久文件删除。不要把它当作“隐藏”“移出列表”或“清理缓存”。

### 14.4 删除 checkpoint 不会删除整个训练任务记录

删除模型文件后，系统会保留：

- history；
- summary；
- 训练任务状态（run_state）；
- 标准输出 / 标准错误日志；
- 其它训练任务元数据。

因此训练历史仍可审计，但被删除的模型文件不能通过这些 metadata 自动恢复。

### 14.5 删除审计记录

删除操作会写入：

```text
12_training/checkpoint_deletion_history.jsonl
```

用于记录删除时间、Stage、训练任务、checkpoint 名称、大小和相关验证信息。

---

## 15. 多次训练与重新选择模型

### 15.1 新训练不会覆盖旧训练任务

每次启动都会获得新的 `run_id`。

这意味着你可以保留：

```text
run A
run B
run C
...
```

之后再比较不同 run 的结果。

### 15.2 epochs / batch_size 只影响当前一次训练

例如你先运行：

```text
epochs = 10
batch_size = 1
```

下一次把 epochs 改为 15，只会影响新的训练任务。

旧训练任务的实际参数会保存在自己的 `training_command.json` 中。

### 15.3 手动锁定 Active Best 后继续训练而暂不切换

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

这样新的训练实验不会自动改变当前用于 Voice Profile 的模型选择。

### 15.4 本指南不提供“最佳 epochs”统一结论

训练质量取决于数据规模、数据质量、说话人特征和具体目标。

基础 Training 指南 的目标是让用户正确控制 **训练任务 / checkpoint / Active Best** 生命周期，而不是提供研究级调参结论。

---

## 16. 生成 Voice Profile v2

完成所选路线的训练并确认 Active Best 后，在：

```text
生成 Voice Profile v2
```

区域创建用于 Inference 的 Voice Profile。

Voice Profile 是 Training → Inference 的正式交接。

### 16.1 profile_name

UI 字段：

```text
profile_name
```

建议显式填写一个容易识别的名称。

如果留空：

1. 系统优先使用当前 **说话人 / 角色名**；
2. 如果说话人也为空、但你通过自定义 work_dir 工作，则会使用 `default_user`。

最终 profile 目录名还会进行安全化处理：字母、数字、下划线和连字符会保留，其它字符会替换为下划线。

### 16.2 默认使用当前 Active Best

正常情况下：

- Stage1 有 Active Best → Profile 使用 Stage1 Few-shot；
- Stage1 没有 Active Best → 回退 Stage1 Zero-shot 默认模型；
- Stage2 有 Active Best → Profile 使用 Stage2 Few-shot；
- Stage2 没有 Active Best → 回退 Stage2 Zero-shot 默认模型。

### 16.3 强制使用 zero-shot

UI 提供：

```text
Stage1 强制使用 zero-shot
Stage2 强制使用 zero-shot
```

它们用于明确构建 Partial Few-shot。

### 16.4 Stage1-only Few-shot

设置：

- **Stage1 强制使用 zero-shot：** 关闭
- **Stage2 强制使用 zero-shot：** 开启

然后点击：

```text
生成 Voice Profile v2
```

预期 profile 类型：

```text
few_shot_stage1_only
```

模型来源：

```text
Stage1 → few-shot Active Best
Stage2 → zero-shot default
```

### 16.5 Stage2-only Few-shot

设置：

- **Stage1 强制使用 zero-shot：** 开启
- **Stage2 强制使用 zero-shot：** 关闭

预期：

```text
few_shot_stage2_only
```

模型来源：

```text
Stage1 → zero-shot default
Stage2 → few-shot Active Best
```

### 16.6 Full Few-shot

设置：

- **Stage1 强制使用 zero-shot：** 关闭
- **Stage2 强制使用 zero-shot：** 关闭

并确认两个 Stage 都已经存在正确的 Active Best。

预期：

```text
few_shot_dual
```

### 16.7 base_zeroshot

如果两个 Stage 都使用默认模型，Profile 类型为：

```text
base_zeroshot
```

但普通 Zero-shot 用户通常不需要为了使用默认模型专门进入 Training。

### 16.8 覆盖已有 profile

选项：

```text
覆盖已有 profile
```

默认开启。

开启时允许写入已经存在的 profile 目录，并重写当前 Profile 配置 / manifest，同时复制本次实际需要的 Few-shot checkpoint。

覆盖已有目录并不等于给旧 Profile 自动创建版本备份；最终推理模型来源以新生成的 `profile_config.json` / `profile_manifest.json` 为准。

> [!WARNING]
> 如果希望保留旧 Profile 作为可回退版本，建议使用新的 `profile_name`，而不是直接覆盖旧 Profile。

### 16.9 Profile 输出位置

生成结果位于：

```text
user_profiles/<profile_name>/
```

其中会保存：

- profile config；
- profile manifest；
- 当前需要的 Stage1 / Stage2 Few-shot checkpoint 副本。

成功后页面会显示：

- profile 名称；
- profile 类型；
- Stage1 来源；
- Stage2 来源；
- 是否可用于推理。

正常发行环境下应看到：

```text
可用于推理：True
```

---

## 17. 按路线完成 Training

### 17.1 Stage1-only Few-shot

流程：

```text
Stage1 数据已就绪
  ↓
Stage1 Training 成功
  ↓
Stage1 Active Best 已就绪
  ↓
Voice Profile:
Stage1 Few-shot + Stage2 默认模型
```

完成检查：

- [ ] Stage1 Training 显示可以启动；
- [ ] Stage1 run 最终为 `succeeded`；
- [ ] Stage1 Active Best 存在；
- [ ] Stage2 强制 zero-shot 开启；
- [ ] Profile 类型为 `few_shot_stage1_only`；
- [ ] Profile 显示可用于推理。

### 17.2 Stage2-only Few-shot

流程：

```text
Stage2 数据已就绪
  ↓
Stage2 Training 成功
  ↓
Stage2 Active Best 已就绪
  ↓
Voice Profile:
Stage1 默认模型 + Stage2 Few-shot
```

完成检查：

- [ ] Stage2 Training 显示可以启动；
- [ ] Stage2 run 最终为 `succeeded`；
- [ ] Stage2 Active Best 存在；
- [ ] Stage1 强制 zero-shot 开启；
- [ ] Profile 类型为 `few_shot_stage2_only`；
- [ ] Profile 显示可用于推理。

### 17.3 Full Few-shot

流程：

```text
Stage1 + Stage2 数据已就绪
  ↓
Stage1 Training
  ↓
Stage2 Training
  ↓
Stage1 + Stage2 Active Best 已就绪
  ↓
few_shot_dual Voice Profile
```

完成检查：

- [ ] Stage1 Training 成功；
- [ ] Stage2 Training 成功；
- [ ] Stage1 Active Best 正确；
- [ ] Stage2 Active Best 正确；
- [ ] 两个 **强制使用 zero-shot** 均关闭；
- [ ] Profile 类型为 `few_shot_dual`；
- [ ] Profile 显示可用于推理。

> [!IMPORTANT]
> “训练按钮执行过”不是 Training 流程的最终完成条件。真正进入 Inference 前，应确认 **目标 Stage 成功 → Active Best 正确 → Voice Profile 生成成功**。

---

## 18. 训练状态、失败与日志排查

### 18.1 常见训练任务状态

| 状态 | 含义 |
| --- | --- |
| `pending` | 训练工作进程正在启动 |
| `running` | 训练进程正在运行 |
| `stopping` | 已请求停止，正在处理 |
| `succeeded` | 训练进程以成功状态结束；仍需确认 Trainer Best / Active Best 交接 |
| `failed` | 训练或工作进程失败 |
| `stopped` | 按用户请求停止 |
| `stop_failed` | 停止流程本身未正常完成 |

正常完成训练任务的目标状态是：

```text
succeeded
```

### 18.2 训练启动失败

如果点击 Stage1 / Stage2 启动后立即失败，优先检查：

1. 是否扫描了正确的 DataFactory 工作目录（work_dir）；
2. 对应 Stage 是否显示 **可以启动**；
3. epochs / batch_size 是否为正整数；
4. 同一 Stage 是否已有活动训练任务；
5. Stage2 所需基础 checkpoint 是否完整。

### 18.3 训练运行中失败

如果训练任务已经启动后进入 `failed`：

1. 查看 **训练失败摘要**；
2. 查看：
   ```text
   training_stderr.log
   ```
3. 再查看：
   ```text
   training_stdout.log
   ```
4. 如需确认本次实际参数，查看：
   ```text
   training_command.json
   ```

### 18.4 CUDA / OOM

如果失败摘要中出现 CUDA 不可用（CUDA unavailable）、CUDA 显存不足（`CUDA out of memory`） 或类似 GPU 错误：

- batch_size 是用户版可以调整的参数之一；
- 但 GPU / CUDA 兼容性和显卡支持范围应参考 Compatibility / Troubleshooting；
- 不建议为了绕过错误直接修改 由发行版统一管理的模型配置。

### 18.5 训练成功但 Active Best 没变化

首先检查当前选择模式。

如果是：

```text
手动锁定
```

则这是预期行为。

新 Trainer Best 不会覆盖人工锁定的 Active Best。

如果希望恢复自动更新，使用：

```text
恢复 Stage1 Automatic Best
```

或：

```text
恢复 Stage2 Automatic Best
```

### 18.6 删除按钮不可用或删除被拒绝

常见原因：

- 当前 checkpoint 是 Active Best 来源；
- checkpoint 属于正在训练的任务；
- 选择信息已经过期，需要先 **刷新 checkpoint 列表**；
- 没有勾选永久删除确认框；
- 目标文件不属于受支持的历史 checkpoint 范围。

---

## 19. 设置与管理边界

### 19.1 普通用户可以编辑

训练参数：

- Stage1 epochs；
- Stage1 batch_size；
- Stage2 epochs；
- Stage2 batch_size。

运行体验：

- 是否打开独立训练日志终端。

### 19.2 用户可以管理，但它们不是训练超参数

- DataFactory work_dir；
- 是否在扫描时刷新 DataFactory 状态；
- Stage1 / Stage2 Active Best；
- checkpoint 历史；
- 强制 Stage1 / Stage2 使用 zero-shot；
- Profile 是否覆盖已有目录。

### 19.3 由发行版统一管理的参数

除页面明确开放给用户的设置外，Training 还包含一部分由发行版内部统一管理的训练参数。

这些内部参数不在普通用户界面中开放，也不需要为了正常训练手工修改。

---

## 20. 常见边界与安全操作

### 20.1 Trainer Best 不等于 Active Best

Trainer Best 属于某个训练任务。

Active Best 属于当前 Stage 的用户选择状态。

### 20.2 最新训练任务不等于当前 Active Best 来源

手动锁定后尤其如此。

### 20.3 Stop 不会删除训练任务

Stop 只是停止训练进程，并保留已有产物和日志。

### 20.4 关闭日志查看器 不会停止训练

真正停止必须使用 Stop 按钮。

### 20.5 v1.0.0 没有用户级训练续跑（Training Resume）

重新训练会创建新的 run。

### 20.6 checkpoint 删除是永久操作

删除后模型文件不能从 history / summary 文件 自动恢复。

### 20.7 Active Best 来源受保护

想删除它，必须先切换 Active Best。

### 20.8 Stage1-only / Stage2-only 都是合法完成状态

不要为了部分 Few-shot 路线强行等待另一个 Stage 完成。

### 20.9 Voice Profile 才是进入 Inference 的正式交接

训练完成后不要直接把某个训练任务目录当作 Inference 的用户入口。

应先确认 Active Best，再生成 Voice Profile。

---

## 21. 下一步与相关文档

完成 Training 后，下一步是使用 Voice Profile 进入 Inference。

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
- Inference Guide（尚未发布）
- Compatibility Guide（尚未发布）
- Troubleshooting Guide（尚未发布）
