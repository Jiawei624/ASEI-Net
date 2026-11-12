**ASEI-Net 统一方案与服务器运行说明**

`f_score`、`VUS_ROC`、`VUS_PR` 用于比较每次修改的实际效果。训练及验证保留原损失，检查点保留原选模规则。四个数据集直接运行当前完整配置，先用 seed=42。其余指标继续计算，不要求先完成逐轮组件对照。正式结果尚待服务器训练，本地合成检查仅证明实现路径可运行。

完整方法定义见根目录 `In1_2_4.md`。根目录文档默认被 Git 忽略，因此本文件也提供可随代码同步的实现与运行入口。

**方法定义。**

| 部分 | 当前配置与作用 |
|---|---|
| 正常参照 | 冻结正常预测关系，为每个目标选择最多四个非对角有效邻居。目标自身实际轨迹始终排除；外部支持不足时使用正常状态的边际回退 |
| 查询先验 | 有外部支持时 `reference_prior=0`，查询由外部上下文决定；可批量检索，不让正常教师先验主导已覆盖工况的选择 |
| 参照支持 | 将拟合段支持尺度归一化后的外部代价及软可信量加入证据，并缩放短参照差；不将可信量解释为异常概率 |
| 多尺度证据 | 在原双轨迹之外对可学习证据维护 4/16/64 三种指数均值；与当前证据、状态差、正常状态共同读出。三个均值不接受教师坐标损失 |
| 任务强度 | 数值干预包含 0.25/0.5/1.0 三种强度，族与布局保持均衡。真实正常成员不变；离散状态不插值；最后重新检查可观测性 |
| 训练与验证损失 | 加权 BCE、正常/恢复状态锚定、正常尾部约束、可选通道监督和同背景配对。保留各数据集原有权重及两遍尾部训练，不新增跨背景排序损失 |
| 选模 | 原 `normal_constrained`：动态正常 FPR 分档优先，档内比较合成 AP 与覆盖率；参考不足时退回验证损失。选择段内部使用原经验尾秩，不使用三个主评价指标或新的组合代理 |
| 排名 | 正常参考 95% 位置以上用固定超额尺度连续延伸，避免所有超出正常最大值的分数并列；用于 VUS 排序 |
| 报警 | 保留经验尾秩与固定分支预算，主 F1 取 alpha=0.01；连续排名分数不是 p 值，不用于误报保证 |
| SMD 类型 | 配置显式按连续数值处理，逐通道展开进 resolved config；避免拟合常量的所有新值落入同一个 unknown 符号。属于表示假设，不能单独归功于双轨迹 |

通道选择借鉴 [CATCH，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b25c39788e5cf11d3541de433ebf4c0-Abstract-Conference.html) 对无关通道混合的分析；因果多尺度证据与 [CCM-TAD，ICML 2026](https://proceedings.mlr.press/v306/murad26a.html) 的时序连续性思想相关，本实现不回填事件边界。[CAROTS，ICML 2025](https://proceedings.mlr.press/v267/kim25aa.html) 对正常变化和异常干预的区分说明任务语义的重要性；当前实现保留真实正常背景并覆盖干预强度，不声称复现其因果增强或对比预训练。这些来源不证明本组合的新颖性或收益。

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

额外 CSV 的每个 alpha 一行包含全部 22 项指标，三个主指标可在同一行读取；VUS 在各 alpha 行重复展示，不重复计算。递归汇总会将该镜像还原成原有的标签/分数记录并去重。原 Run 内的 `metrics.csv` 继续保留。`ranking_mapping` 标明排名方式，`export_role=combined_view` 标明镜像。

每个检测器目录还保留 `resolved_config.json`、`result.json`、`calibration.json`、`bundle.pt` 和 `predictions.npz`。默认保留逐点预测；大量完整任务缓存仍可按原设置在完成后清理。

如需逐事件及分支诊断，对四个实际目录分别执行：

```bash
python scripts/validation_review.py diagnose \
  --detector-dir outputs/runs/实际Run_ID/PSM/PSM/seed_42/detectors/full
```

将四个 Run ID 替换进下面命令，仅收集本轮，避免打包服务器的全部历史运行：

```bash
python scripts/validation_review.py collect \
  --root outputs --output result/014_integrated \
  --run-ids PSM实际Run_ID PUMP实际Run_ID SMD实际Run_ID SWAT实际Run_ID \
  --include-predictions
```

返回 `result/014_integrated/feedback.zip`，其中含具名全指标 CSV、正常 Run CSV、配置、日志、逐点预测及已生成的诊断。没有 checkpoint 或原始数据。日后回放需要的 bundle 保留在服务器。

历史 012 的 seed=42、alpha=0.01 探索结果仅作参照：

| 数据集 | f_score | VUS_ROC | VUS_PR |
|---|---:|---:|---:|
| PSM | 0.3569 | 0.6589 | 0.4246 |
| PUMP | 0.3386 | 0.7615 | 0.3040 |
| SMD | 0.0851 | 0.6238 | 0.0807 |
| SWAT | 0.4824 | 0.7909 | 0.3297 |

三个主指标逐项比较。保留实际 FPR、点召回及排名映射，区分分类功效、工作点变化与排名变化；当前整合实验不能识别每个组件的独立贡献。方法方向成立后再决定是否精简和做必要归因。

本地验证记录：115 项相关检查通过，包含四套配置的损失系数及选模设置与修改前快照的一致性检查，以及最终排名选择不影响内部选模的检查。四套配置在实际通道数、模型宽度及完整原型容量下完成合成尺度拟合和反向检查；PSM 混合类型与 SMD 连续类型的单 epoch 完整流程通过，均覆盖全部 22 指标，并核对训练、验证总损失等于原各分项加权和。额外 CSV、递归去重及指定 Run 打包通过，旧 `/4` bundle 的流式结果与原保存预测一致。没有运行正式数据训练，没有测量服务器 CUDA 吞吐或三个主指标的正式收益。没有新增依赖。
