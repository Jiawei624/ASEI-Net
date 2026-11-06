# ASEI-Net

根据 [In1_2_2.md](In1_2_2.md) 实现正常原型参照与共享参数双轨迹检测。候选数据集采用 [dataset.txt](dataset.txt) 中的 PSM、SMD、SWAT、PUMP。数据格式转换位于 [asei/data_process.py](asei/data_process.py)，基础指标实现位于 [asei/evaluation](asei/evaluation/__init__.py)，ADD、阈值和评价结果组织位于 [asei/metrics.py](asei/metrics.py)。

正常参照一致性与共享收缩状态的研究设计见 [In1_2_3.md](In1_2_3.md)。该文档中的教师、辅助目标及新时序单元属于设计方案，当前检测器保持共享 GRU 结构。

固定路径和训练默认值由 [configs/default.yaml](configs/default.yaml) 管理；GPU、数据集、实体、种子等通过运行命令选择。四套数据集配置位于 `configs/`，服务器操作步骤见下文。已提供的单种子实验位于 `result/001`、`result/002`，多种子效果由服务器实验验证。正式训练种子为 41–45，验证任务生成种子为 1001、1002。

合成训练保留真实事件边界，正样本损失从事件首次出现已观测差异的位置开始；此前与干净成员相同的事件前缀不承担正损失，事件之后的持续位置保留正监督。监督协议为 `causal_event_prefix_v2`，任务缓存按该协议校验，训练使用新的实验目录。

## 环境

优先复用已有 TAB 环境，支持 Python 3.8 及以上。只读检查依赖并保存环境报告：

```bash
python scripts/check_environment.py --check-imports
```

服务器 GPU 检查可使用 `python scripts/check_environment.py --check-imports --gpu 0 --seed 42`，在物理 GPU 0 上用合成输入检查两阶段前向与反向计算，并将结果写入环境报告。完整的 seed 42、GPU 0 检查与训练命令见 [服务器操作说明](docs/server.md)。

检查显示 `Environment check: PASSED` 后即可继续。核心要求与 TAB 对齐：`torch>=1.11.0`、`statsmodels>=0.14.0`、`tqdm>=4.64.0`；pandas、SciPy、scikit-learn 沿用已安装版本。NumPy 默认要求 `>=1.21.0`，兼容 TAB 的固定值及现有较新版本。YAML 配置额外需要 `PyYAML>=5.1`，仅缺该包时可补装：

```bash
python -m pip install --no-deps "PyYAML>=5.1"
```

已有环境无需全量重装。`requirements-strict.txt` 为 Python 3.8/3.9 提供 TAB 的 `numpy==1.21.0` 精确约束；Python 3.10+ 应保留其可用版本。gdown 和 pytest 位于可选的 `requirements-tools.txt`，不参与使用本地数据的训练。详细范围见 [依赖兼容说明](docs/dependency_compatibility.md)。

本地指定环境检查：

```powershell
& 'E:\Miniconda3\Scripts\conda.exe' run -n codex python scripts/check_environment.py --check-imports
```

数据处理与评价模块随项目一起分发，业务代码直接使用项目内导入。使用已有本地数据时，运行无需访问外部源码仓库。

## 固定配置

默认配置指定数据根目录 `dataset`、结果目录 `outputs`、日志目录 `logs`，并选择物理 GPU 1。相对路径解析规则：

| 配置项 | 相对路径的起点 | 默认结果 |
|---|---|---|
| `data.root` | 项目根目录 | `/ASEI-Net/dataset` |
| `data.manifest` | 项目根目录 | 空，使用 TAB 元数据或生成的合格清单 |
| `data.archive` | `data.root` | `dataset/TAB_dataset.zip` |
| `runtime.output` | 项目根目录 | `/ASEI-Net/outputs` |
| `runtime.log_dir` | 项目根目录 | `/ASEI-Net/logs` |
| `runtime.run_subdir` | `runtime.output` | `outputs/runs/<时间戳>/` |
| `runtime.log_file` | `runtime.log_dir` | 启用 `--log` 后为 `logs/main_<时间戳>.log` |
| `reports.*` | `runtime.output` | 检查报告、合格清单、汇总及环境报告 |

路径不依赖启动目录。在服务器项目根目录为 `/ASEI-Net` 时，默认数据位置就是 `/ASEI-Net/dataset`；项目放在其他位置时无需修改代码。

`--gpu 0`、`--gpu 1` 直接选择物理 GPU，优先于 YAML 中的卡号；所选卡在进程内映射为逻辑 `cuda:0`，日志同时显示物理卡号。`--gpus` 是同义参数，每个进程使用一张卡。省略时读取 `runtime.cuda_visible_devices`。CPU 运行使用 `--device cpu`；`--device` 和 `--gpu` 互斥。

共享服务器默认每进程限制为 8 个逻辑 CPU、4 个计算线程、1 个 interop 线程，数据集配置最多使用 4 个 DataLoader worker。Linux 下限制当前线程、现有线程及继承的子进程的 CPU 亲和性；GPU 0、1 在有足够 CPU 时使用不同的 CPU 集合。整机 CPU 忙碌率超过 80% 时，在批次和计算块边界协作等待。固定设置位于各 YAML 的 `runtime.cpu_cores`、`threads`、`interop_threads`、`cpu_max_percent`、`cpu_check_interval_seconds`。80% 是协作阈值，无法强制限制其他用户的任务，长时间的单次操作也可能暂时超过阈值。实现仅使用标准库，无需安装监控依赖。

## 数据

当前数据布局可直接使用：

```text
ASEI-Net/
  dataset/
    anomaly_detect/
      DETECT_META.csv
      data/
        <TAB 原始文件名>.csv
```

配置保持 `data.root: dataset`，自动识别 `anomaly_detect` 层。也支持元数据位于 `dataset/`、CSV 位于 `dataset/data/` 或与元数据相邻的布局。文件名与 `train_lens` 保持 TAB 原样。

`data.multivariate_only: true` 沿用 TAB 多变量实验口径，排除元数据中 `if_univariate=true` 的条目，并在准备阶段核验有效通道数。元数据可能同时列出未下载的单变量切片，不能将它们混入本次多变量实体清单。官方文件名中的 `@` 编号会原样保留。

采用 TAB 的[预处理数据入口](https://drive.google.com/file/d/1V5BAHWBKU8uih3hE1R7WdF6_crZlIbQT/view)。已有本地数据时跳过下载；使用下载脚本时，需要可选的 gdown：

```bash
python scripts/download_data.py
```

该脚本读取配置中的数据根目录和压缩包名。代码调用项目内集成的 TAB 长表读取函数，不重复执行数据集特有的清洗或重采样，仅执行训练段标准化和因果前向填充。

检查并准备候选实体：

```bash
python -m asei inspect --prepare --datasets PSM SMD SWAT PUMP
```

检查报告与合格清单自动保存到配置的输出目录。官方训练前缀必须全正常，按时间将前 80% 用于拟合、后 20% 用于验证；含异常的官方测试段用于最终评价。训练前缀含异常的实体会被拒绝，原因保留在检查报告中。标准化仅由正常拟合段估计，阈值仅由正常验证分数校准，测试标签只用于评价。仅查看文件和元数据时可以省略 `--prepare`，或用 `--read-data` 检查而不更新清单。

当 `data.use_prepared_manifest: true` 且清单存在时，训练自动复用它。程序核验原始元数据、数据根目录、划分及检查范围，防止使用过期清单。只检查了 SMD 后运行 PSM，需要先检查 PSM。没有生成清单时，读取原始 TAB 元数据；建议正式运行前执行 `--prepare`。

手工清单可写入 `data.manifest`，优先于自动清单，UTF-8 CSV 字段为：

```text
dataset,entity,file_name,path,train_lens,test_continuous,validation_path
```

路径可为绝对路径或相对于清单目录的路径。独立正常验证记录通过 `validation_path` 指定。测试记录默认独立，只有来源确认其紧接官方训练记录时才将 `test_continuous` 设为 `true`。自动生成的清单应与 `.meta.json` 来源记录一起保存；需要手工调整时使用独立清单并设置 `data.manifest`。

## 训练和评价

先从检查表复制实际实体名，运行一个实体与 seed 41：

```bash
python -u -m asei run --config configs/smd.yaml --datasets SMD --entities "实际实体名" --gpu 1 --seed 41
```

`--seed 41` 选择单个种子，`--seeds 41 42 43 44 45` 选择多个种子；两种参数名同义。省略时使用配置中的 41、42、43、44、45。一个进程依次执行所选种子。按数据集配置运行：

```bash
python -u -m asei run --config configs/smd.yaml --datasets SMD --gpu 1 --seeds 41 42 43 44 45
```

四个数据集采用各自的下一轮候选配置，面向每张 16GB RTX 4080 SUPER 的单进程实验。以下设置依据 `result/001`、`result/002` 和所提供参考结果制定，尚未获得这一轮的真实训练结果。分析见 [当前调参说明](ccfa-workfiles/asei-review/tuning_002.md)。

| 配置 | 历史窗口 | 隐藏维度 | 阶段一 lr | 阶段二 lr | 阶段二最大轮数 | 训练任务对数 |
|---|---:|---:|---:|---:|---:|---:|
| `configs/psm.yaml` | 128 | 64 | 0.0002 | 0.0001 | 60 | 2048 |
| `configs/pump.yaml` | 256 | 96 | 0.0001 | 0.0003 | 80 | 512 |
| `configs/smd.yaml` | 128 | 64 | 0.0003 | 0.0003 | 60 | 2048 |
| `configs/swat.yaml` | 128 | 96 | 0.0002 | 0.0003 | 80 | 2048 |

| 配置 | 阶段一 batch | 阶段二任务对/batch | 特征提取 batch |
|---|---:|---:|---:|
| `configs/psm.yaml` | 128 | 32 | 256 |
| `configs/pump.yaml` | 64 | 16 | 128 |
| `configs/smd.yaml` | 192 | 32 | 384 |
| `configs/swat.yaml` | 128 | 24 | 256 |

`configs/reference/psm_002.yaml`、`configs/reference/pump_002.yaml` 保留 `result/002` 的参数用于对照；常规新实验使用上述四份根目录配置。

显存峰值和实际吞吐量以服务器测量为准，比较完整 epoch 耗时与验证损失。阶段二每个任务对有两条序列；增大批量会减少优化步数，也可能改变最终指标。批量调整后开启新的时间戳实验；正在运行的进程沿用启动时读取的配置。VUS 等评价计算的耗时由指标实现决定，不随训练 batch 调整。

`configs/swat.yaml` 默认选择物理 GPU 0，命令行可覆盖。后台运行 SWAT 的五个种子：

```bash
mkdir -p logs
nohup uv run python -u -m asei run --config configs/swat.yaml --datasets SWAT --gpu 0 --seeds 41 42 43 44 45 --log > "logs/swat_launch_$(date +%Y%m%d_%H%M%S_%N).log" 2>&1 < /dev/null &
```

调整阶段一训练批量时修改 `stage1.batch_size`；阶段二修改 `stage2.batch_pairs`，每对包含正常和扰动两条序列；冻结特征提取修改 `runtime.feature_batch_size`。按显存余量逐步增大。改变训练批量会改变优化步数和检查点身份。

与 TAB 对应的命令参数别名为 `--config-path`、`--data-set-name`、`--data-name-list` 和 `--gpus`。实体选择可使用实体名或 CSV 文件名。常规实验不需要反复指定数据、结果、日志和清单路径；`--data-root`、`--output`、`--manifest`、`--device` 仍可用于临时覆盖。

两阶段训练默认先学习历史预测与 VQ 原型，再冻结正常模型并准备任务缓存，最后训练双轨迹检测器。官方测试边界固定，选定的连续正常训练记录按时间分成前 80% 拟合和后 20% 验证；独立正常验证记录优先。测试标签仅参与最终评价。

可分别运行阶段：

```bash
python -m asei run --datasets SMD --seeds 41 --stage representation
python -m asei run --datasets SMD --seeds 41 --resume RUN_ID --stage cache
python -m asei run --datasets SMD --seeds 41 --resume RUN_ID --stage detector
python -m asei run --datasets SMD --seeds 41 --resume RUN_ID --stage calibrate
python -m asei run --datasets SMD --seeds 41 --resume RUN_ID --stage evaluate
```

每次新训练自动创建 `outputs/runs/<时间戳>/`，时间戳采用配置的 UTC+8，并包含微秒。重跑同一命令会创建新实验。`RUN_ID` 使用启动时打印的时间戳；`--resume RUN_ID` 复用同一实验，省略 `--config` 时自动读取该次实验保存的配置。时间戳功能启用前的目录通过 `--resume legacy` 继续。未完成阶段保留缓存供续建，训练阶段从头训练；改变训练参数应新开实验。

任务缓存是训练输入，默认 packed 格式，按需内存映射读取。生成索引每 16 对任务提交一次，中断后可继续构建。每阶段只保存验证损失最小的权重；完整评价成功后，将两阶段最优权重、标准化统计、阈值、配置和训练历史合并保存至 `bundle.pt`，随后清理任务缓存、重复权重和中间文件。`--keep-intermediates` 或 `runtime.keep_intermediates: true` 可保留检查材料；`--diagnostics` 同时启用保留。

后台运行并使用配置中的日志文件：

```bash
nohup uv run python -u -m asei run --config configs/psm.yaml --datasets PSM --gpu 1 --seed 41 --log > "logs/psm_launch_$(date +%Y%m%d_%H%M%S_%N).log" 2>&1 < /dev/null &
```

使用 `--log` 时，配置的日志文件名自动加入实验时间戳，例如 `logs/psm_<时间戳>.log`；续跑同一实验时追加。shell 重定向的日志也应使用唯一文件名。每块 GPU 同时运行一个训练进程，按数据集选择对应的 `configs/psm.yaml`、`smd.yaml`、`swat.yaml` 或 `pump.yaml`。

训练每个 epoch 输出一条摘要，包含训练损失、验证损失和本轮耗时。训练、验证、缓存及特征计算的中间进度由 `runtime.log_interval_seconds` 控制，默认 `300` 秒；设置为 `0` 时关闭中间进度，只保留 epoch 摘要与阶段完成信息。日志使用普通文本行，后台运行时不会逐 batch 刷新进度条。调整该配置不影响检查点和缓存复用。

## 指标

最终阈值由正常验证分数按 `alpha=0.01` 的次序统计量确定，严格大于阈值报警。测试标签只参与最终评价。

实验统一计算 [metrics.txt](metrics.txt) 的 16 种标签指标和 6 种连续分数指标，共 22 个；不计算回归指标、额外运行点或独立事件诊断。即使续跑旧配置，也采用该指标集合。

仅四个 `adjust_*` 使用点调整，其他基于报警的指标和 ADD 使用原始报警。ADD 对真实事件等权平均首次实际报警延迟，漏检记为事件长度，单位为采样点。`auc_pr` 为 Average Precision。`precision_at_k` 以真实异常点数作为 K，用连续异常分数排序；边界同分按剩余名额比例分摊命中数。评价协议记录为 `detection22_score_topk_v2`。

完整时间轴保留预热与缺测点，报警记为 0；无效分数排列在所有有效分数以下。无定义的指标在 CSV 中留空并标记。旧结果缺少的指标不补造；旧二值输入的 `precision_at_k` 不作为当前协议下的有效值导出。

已有模型可直接补算全部指标，使用原配置和原 seed，并指定已有 Run ID：

```bash
uv run python -m asei run --datasets PSM \
  --seed 42 --gpu 0 --resume RUN_ID --stage evaluate --all-metrics
```

该命令复用已有最优权重，不重新训练；精简保存后也可直接从 `bundle.pt` 重新评价。旧参数别名 `--all-metrics`、`--all-label-metrics` 仍可用，默认即启用完整 22 指标。

若原始 `predictions.npz` 仍在，可仅补算 16 个标签指标，并直接复用已有 6 个分数指标，避免重复计算 VUS：

```bash
uv run python -m asei export --root outputs/runs/RUN_ID \
  --output outputs/metrics_recomputed.csv --recompute-labels
```

VUS 对长事件可能耗时较长，初步排查可加 `--skip-vus`，正式结果通过 `--stage evaluate` 补算。

## 输出与汇总

```text
outputs/
  environment.json
  data_inventory.csv
  eligible_manifest.csv
  eligible_manifest.meta.json
  runs/<时间戳>/
    run.json
    metrics.csv
    <dataset>/<entity>/seed_<seed>/
      detectors/full/
        bundle.pt
  summary/<时间戳>/
    dataset_metrics.csv
```

```bash
python -m asei aggregate
```

同一运行命令的所有数据集、实体和种子写入同一个 `metrics.csv`，每行由 dataset/entity/seed/variant 标识，保留 22 个指标及少量来源字段。模型文件同时保存配置、数据来源和阈值，`run.json` 保存运行身份及启动配置。

`aggregate` 为每组 dataset/entity/seed/variant 选择最新已完成运行，再对实体等权平均，只生成一个 `dataset_metrics.csv`；保留种子行，并按评价协议分组。限定一次实验可用 `--root outputs/runs/RUN_ID`。已有 JSON 或运行 CSV 可通过 `python -m asei export --root result/001 --output result/001/metrics_22.csv` 导出为统一列结构。

汇总默认只选取配置中的训练种子 41–45。查看旧试运行时可指定 `python -m asei aggregate --seeds 0`，避免将试运行混入正式种子统计。

## 消融和推断

`--diagnostics` 保存恢复门、状态差、原型差异和证据通道轨迹，并在 `result.json` 的 `metrics.event` 中输出事件覆盖、后半段覆盖、实际正常误报率和恢复拖尾。常规 CSV 保持 22 指标。该选项也适用于已完成精简实验的诊断评价，复用最优网络和阈值。检测头支持 `mlp`、`single_gru`、`no_alignment`、`zero_alignment`、`fixed_gate`、`independent_gru`，例如：

```bash
python -m asei run --datasets SMD --seeds 41 --resume RUN_ID --variant no_alignment
```

来源可通过 `--sources` 选择。通过 `--resume RUN_ID` 指定实验并保持其他配置，可复用冻结表征与任务缓存；严格参数匹配和外部完整基线仍需按正式预算组织。

```python
from asei.pipeline import load_bundle

stream, channel_names = load_bundle("outputs/.../bundle.pt", device="cpu")
result = stream.push(current_sample)
later_results = stream.process(next_block)
stream.reset()  # 仅用于新实体或独立记录
```

```bash
python -m asei predict --bundle outputs/.../bundle.pt --input-csv observations.csv
```

命令行推断沿用配置中的 GPU，结果自动保存到配置的预测目录。模型实例跨块保留历史与两条状态，不读取标签或在线更新原型。

本地验证：`python -m pytest -q tests`。小模型与合成数据只用于实现检查，不作为正式效果证据。
