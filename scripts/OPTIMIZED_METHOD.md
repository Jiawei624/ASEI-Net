**ASEI-Net 统一方案与服务器运行说明**

重点比较 `f_score`、`VUS_ROC`、`VUS_PR`。F1 统一以 alpha=0.01 对照历史，并额外报告九档阈值；VUS 每次评分只计算一次，其他指标继续完整输出。训练及验证保留原损失公式和权重，检查点保留原选模规则。四个数据集先用 seed=42 运行当前配置。候选配置依据 014 结果制定，正式效果尚待服务器训练；本地合成检查仅证明实现路径可运行。

完整方法定义见根目录 `In1_2_4.md`。根目录文档默认被 Git 忽略，因此本文件也提供可随代码同步的实现与运行入口。

四套主配置统一继承 `configs/method.yaml`。完整方法的模块、任务定义、类型识别、校准和评价流程一致；数据集配置只覆盖编码器深宽、检测头宽度、原型容量及训练超参数。损失系数可调，但完整方法的五类损失全部启用。去除模块只作为单独的消融实验，在四个数据集上使用相同消融定义。

**方法定义。**

| 部分 | 当前配置与作用 |
|---|---|
| 正常参照 | 全部可用非目标通道、0.25 正常状态先验。目标自身实际轨迹始终排除，外部支持不足时使用正常状态的边际回退 |
| 查询执行 | 沿时间顺序更新正常先验；冻结路由和特征按原机制缓存，不能打乱因果状态 |
| 参照支持 | 将拟合段支持尺度归一化后的外部代价及软可信量加入证据，并缩放短参照差；不将可信量解释为异常概率 |
| 证据时间尺度 | 双轨迹外统一维护 4/16 两档证据均值，读出输入均为 160 维。证据均值不接受教师坐标损失 |
| 证据编码 | signed_log1p、8 维通道身份、32 维证据；MLP 层数及连接方式一致，隐藏宽度可调整 |
| 任务强度 | 数值干预强度 0.5/1.0，四个任务族及布局保持均衡。真实正常成员不变；离散状态不插值；最后重新检查可观测性 |
| 正常尺度 | 全部启用原型局部尺度，伪计数 64、裁剪 10、局部/全局比值限制 0.5—3；估计公式相同 |
| 训练与验证损失 | 原加权 BCE、正常/恢复状态锚定、正常尾部约束、通道监督和同背景配对全部启用；保留两遍尾部训练。各项权重属于训练超参数，完整方法不使用零权重关闭损失 |
| 选模 | 原 `normal_constrained`：动态正常 FPR 分档优先，档内比较合成 AP 与覆盖率；参考不足时退回验证损失。选择段内部使用原经验尾秩，三个主指标只在最终评估计算 |
| 排名 | 正常参考 95% 位置以上用固定超额尺度连续延伸，避免所有超出正常最大值的分数并列；用于 VUS 排序 |
| 报警 | 经验尾秩、即时/动态预算 0.05/0.95，alpha=0.0005/0.001/0.005/0.01/0.02/0.05/0.1/0.2/0.3。单值 alpha=0.01 用于原选模和默认部署；连续排名分数不是报警 p 值 |
| 类型识别 | 同一规则只读拟合段：2—16 个已见值判为有限状态，其余保留数值；常量不足以确认有限状态词表。主配置无数据集专属覆盖，词表和输入通道数自然随数据变化 |

[CATCH，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b25c39788e5cf11d3541de433ebf4c0-Abstract-Conference.html) 说明通道上下文的重要性；当前主方法固定使用目标排除的外部上下文，不复现其通道选择器。因果多尺度证据与 [CCM-TAD，ICML 2026](https://proceedings.mlr.press/v306/murad26a.html) 的时序连续性思想相关，本实现不回填事件边界。[CAROTS，ICML 2025](https://proceedings.mlr.press/v267/kim25aa.html) 对正常变化和异常干预的区分说明任务语义的重要性；当前实现保留真实正常背景，不声称复现其因果增强或对比预训练。这些来源不证明本组合的新颖性或收益。

| 数据集 | 正常预测器宽度×层数 | evidence/readout 隐藏宽度 | 检测 epoch 上限 / patience | batch_pairs | lr | 训练 / 选择任务 |
|---|---|---|---|---:|---:|---|
| PSM | 48×1 | 64/32 | 20/4 | 32 | 5e-5 | 4096/1024 |
| PUMP | 64×2 | 48/24 | 12/4 | 64 | 5e-5 | 4096/1024 |
| SMD | 48×1 | 32/16 | 10/3 | 64 | 2e-5 | 4096/1024 |
| SWAT | 64×2 | 96/32 | 32/5 | 32 | 5e-5 | 4096/1024 |

四套窗口均为 128。边际/联合原型容量为 128/256、192/384、256/512、512/768；教师上限为 36/32/20/20，每批 32 条正常片段，均按原正常验证损失早停。PUMP/SMD 限制后期检测更新；SWAT 扩大正常记忆与背景覆盖。依据及修改前 23 份配置/源码备份位于本地 `result/014/tuning_notes.md` 与 `configs_before_tuning`；该目录默认不随 Git 同步。较大的记忆会增加检索成本，实际耗时需服务器测量。

**主要代码位置。**

| 文件 | 实现 |
|---|---|
| `asei/normal.py` | 冻结邻居表、目标排除代价、批量外部查询、CPU 并行原型聚类 |
| `asei/model.py`、`state.py` | 显式证据布局、参照支持、多尺度证据状态、因果更新和梯度 |
| `asei/tasks.py`、`training.py` | 多强度任务、跨任务预测批量化、顺序异步缓存写入 |
| `asei/detection_training.py`、`selection.py` | 原损失与选模、正常审计冻结特征缓存及分阶段计时 |
| `asei/calibration.py` | 经验报警与连续排名两种输出，正常数据拟合与保存加载 |
| `asei/pipeline.py`、`results.py` | 新状态部署、阶段计时、全部指标的具名 CSV、汇总去重 |
| `asei/evaluation/vus_metrics.py` | 独立容差窗口并行，保持原阈值、并列和积分规则 |

**工程执行。**

既有日志中，012 的训练/选择任务缓存准备合计约 20–26 分钟；SMD 的教师结束到训练缓存完成还包含额外正常尺度回放。因此 GPU 低利用率不能只通过增加训练 batch 解释。

当前执行方式为：CPU 独立通道聚类最多四线程；任务生成预取；合并多条任务的预测位置填满 GPU 批次；一条有界后台线程顺序写缓存；外部查询在支持充分、先验权重为零时按时间批量化；正常审计复用最多 512 MiB 的冻结特征；VUS 的独立容差窗口最多四线程。CUDA 模型计算仍由主线程负责，因果状态沿时间顺序推进。缺测、无邻居或配置要求状态先验时使用顺序回退。

新日志给出教师 train/validation、stage2 train/synthetic validation/normal audit、任务 forecast/routing/write，以及最终 evaluation 的耗时。异步写入耗时与前向可能重叠，不能把子项机械相加当作总耗时。512 MiB 缓存按进程分配；多 GPU 并行任务应同时核算主机内存和 CPU 预算。

这些修改减少重复计算与小批调用，实际吞吐、峰值显存及 GPU 利用率需要服务器测量。多尺度状态也增加激活内存，不能承诺所有阶段均达到训练阶段的 GPU 利用率。

**服务器命令。**

在项目根目录、现有 Python 环境中执行。配置文件仍为 `configs/psm.yaml`、`pump.yaml`、`smd.yaml`、`swat.yaml`，分别包含完整方法。

先检查四套主配置与共同定义一致：

```bash
python scripts/check_method_consistency.py --output result/015/method_consistency.json
```

检查会拒绝数据集单独关闭 EMA、局部尺度、通道身份、损失项或切换参照、类型与校准流程。正常词表只由拟合数据生成，不要求不同数据集的词表或参数量相同。

准备服务器自己的数据清单；数据目录不同时给所有命令增加相同的 `--data-root /实际路径`：

```bash
python -m asei inspect --config configs/psm.yaml \
  --datasets PSM PUMP SMD SWAT --prepare
```

可先做一次小规模 GPU 合成检查，它不属于正式实验：

```bash
python scripts/check_pipeline.py --config configs/psm.yaml --gpu 0 --with-vus
```

直接训练四套完整配置；有多张 GPU 时可分配不同编号：

```bash
python -u -m asei run --config configs/psm.yaml --seed 42 --gpu 0 --log
python -u -m asei run --config configs/pump.yaml --seed 42 --gpu 0 --log
python -u -m asei run --config configs/smd.yaml --seed 42 --gpu 0 --log
python -u -m asei run --config configs/swat.yaml --seed 42 --gpu 0 --log
```

本轮使用新 Run ID。正常参照、类型及核心源码已属于当前方法身份，不能用历史 Run ID 绕过重训。中断后仅续跑本轮同一配置和源码对应的 Run ID。

**结果文件与反馈。**

默认日志为 `logs/<配置日志名>_<Run ID>.log`，额外全指标文件为 `outputs/results/<相同文件名>.csv`。例如日志 `psm_20261112_120000_123456.log` 对应 `outputs/results/psm_20261112_120000_123456.csv`。

额外 CSV 与 Run 内的 `metrics.csv` 逐字节一致，包含该 Run 当前全部已完成实体、种子和变体。每个 alpha 一行标签指标，另有一行分数指标；九档 alpha 对应九行标签指标和一行分数指标。重复评估后再次复制整个 Run CSV，汇总按实验身份去重。读取主指标时，将所需 alpha 的标签行与同实验的分数行对应。

每个检测器目录还保留 `resolved_config.json`、`result.json`、`calibration.json`、`bundle.pt` 和 `predictions.npz`。默认保留逐点预测；大量完整任务缓存仍可按原设置在完成后清理。

如需逐事件及分支诊断，对四个实际目录分别执行：

```bash
python scripts/validation_review.py diagnose \
  --detector-dir outputs/runs/实际Run_ID/PSM/PSM/seed_42/detectors/full
```

将四个 Run ID 替换进下面命令，仅收集本轮，避免打包服务器的全部历史运行：

```bash
python scripts/validation_review.py collect \
  --root outputs --output result/015 \
  --run-ids PSM实际Run_ID PUMP实际Run_ID SMD实际Run_ID SWAT实际Run_ID \
  --include-predictions
```

返回 `result/015/feedback.zip`，其中含具名 CSV、正常 Run CSV、配置、日志、逐点预测及已生成的诊断。没有 checkpoint 或原始数据。日后回放需要的 bundle 保留在服务器。

014 的 seed=42、alpha=0.01 探索结果作为本轮直接参照：

| 数据集 | f_score | VUS_ROC | VUS_PR |
|---|---:|---:|---:|
| PSM | 0.024471 | 0.672827 | 0.463776 |
| PUMP | 0.436754 | 0.813724 | 0.251241 |
| SMD | 0.088827 | 0.651173 | 0.079368 |
| SWAT | 0.287201 | 0.761963 | 0.271658 |

三项主指标逐项比较，保留实际 FPR、点召回及排名映射，区分工作点变化与排名变化。014 中 PSM/SMD 到 alpha=0.1 时 F1 仍上升，而 SWAT 在最小值 0.001 时最高，因此统一补测 0.0005/0.2/0.3。默认 alpha 和内部选模保持 0.01。PSM/PUMP 的短期分支在 0.0005 下正常尾秩分辨率不足，检查 CSV 中的支持标记；不把该工作点作为统一主结果。固定评分下扩大 alpha 不能提高 VUS；各方法在同一 alpha 比较，测试集最优阈值仅用于探索诊断。

本地 135 项相关 CPU 检查通过，包含主方法一致性、单数据集模块切换拒绝、统一类型识别、继承配置冻结及 CSV 去重。四套实际尺寸的合成前向、反向与状态连续性通过；共同方法的单 epoch 完整合成流程覆盖全部 22 指标、九档 alpha、原损失分项、CSV 镜像及流式部署。记录在本地 `result/014/validation_summary.json` 和 `profile_validation.json`。检查入口为 `scripts/check_pipeline.py --with-vus`。未执行正式数据训练或 GPU 测速，检测收益尚待测量。没有新增依赖。
