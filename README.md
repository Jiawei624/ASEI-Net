# ASEI-Net

根据 [In1_2_2.md](In1_2_2.md) 实现正常原型参照与共享参数双轨迹检测。候选数据集采用 [dataset.txt](dataset.txt) 中的 **PSM、SMD、SWAT、GECCO、PUMP**。数据读取与指标调用固定版本的 [TAB](https://github.com/decisionintelligence/TAB)，另外实现 ADD 和事件恢复指标。

目前完成代码与 CPU 小规模关键路径验证，尚未下载完整数据或进行正式实验。真实数据适用性需先运行 `inspect --read-data` 核验；GECCO、PUMP 是否具有可用的正常训练边界，以实际 TAB 元数据和训练段为准。

## 环境

使用 Python 3.10 或更新版本，在项目根目录执行：

```bash
python -m pip install -r requirements.txt
python scripts/bootstrap_tab.py
```

本地指定环境为：

```powershell
& 'E:\Miniconda3\Scripts\conda.exe' run -n codex python -m pytest -q tests
```

服务器上使用对应的 Python 环境。GPU 训练需要该环境的 PyTorch 能识别 CUDA。默认 `device: auto`，也可传入 `--device cuda:0`。代码不依赖 TAB 的整个基线依赖列表。

TAB 固定提交为 `5a7e6f80c405eed7c4dd1390f69fbba2fd8ddcf2`。`bootstrap_tab.py` 下载该版本；本项目不修改其源码。

## 数据

采用 TAB 提供的[预处理数据压缩包](https://drive.google.com/file/d/1V5BAHWBKU8uih3hE1R7WdF6_crZlIbQT/view)。可使用下载脚本：

```bash
python scripts/download_tab.py --output dataset/TAB_dataset.zip
```

解压后，将 `--data-root` 指向包含以下结构的目录：

```text
dataset/anomaly_detect/
  DETECT_META.csv
  data/
    <TAB 原始文件名>.csv
```

直接调用 TAB 的读取函数处理其 `data,cols` 或 `date,data,cols` 长表。保留变量顺序，不重复执行重采样或数据集特有的清洗。读取后仅执行方案要求的训练段标准化和因果前向填充。

```bash
python -m asei inspect --data-root dataset/anomaly_detect --read-data --output outputs/data_inventory.csv
```

该命令检查文件、官方 `train_lens` 边界及正常训练条件。训练段含异常时直接标记不适用，不删除异常点后拼接记录，也不自动从测试标签重建训练集。缺少训练边界的实体需要先提供有来源依据的划分。

如果文件名不能识别数据集，或需要明确记录连续性、独立验证记录，提供 UTF-8 CSV 清单：

```csv
dataset,entity,file_name,path,train_lens,test_continuous,validation_path
SMD,machine-1-1,SMD_machine-1-1.csv,/data/TAB/data/SMD_machine-1-1.csv,28479,false,
```

上面的文件名和长度仅示范清单格式，应替换为实际元数据。`path` 相对清单目录解析；省略 `path` 时使用 `file_name`。`validation_path` 为独立正常验证记录，同样必须包含二值 `label` 列。宽表也支持，列为数值通道及 `label`；`date`/`timestamp`/`time` 不作为通道。`test_continuous=true` 仅用于已确认测试紧接官方训练记录的实体，不能仅凭 TAB 将二者放在同一个 CSV 中认定连续。

## 训练和评价

先选一个经检查适用的实体运行：

```bash
python -m asei run --data-root dataset/anomaly_detect --datasets SMD --entities <实际实体名> --seeds 0 --device cuda:0
```

正式运行五个种子：

```bash
python -m asei run --data-root dataset/anomaly_detect --datasets PSM SMD SWAT GECCO PUMP --seeds 0 1 2 3 4 --device cuda:0
```

仅选择 `inspect` 确认适用的数据集和实体。使用清单时加 `--manifest /path/manifest.csv`。

默认配置见 [configs/default.yaml](configs/default.yaml)，完整定义与代码对应见 [docs/implementation.md](docs/implementation.md)。数据集官方测试边界固定；没有独立验证集时，官方正常训练记录按时间分成 80%/20%。标准化和正常表征只使用前段。阶段一根据正常验证损失选择，阶段二根据正常记录生成的固定验证任务损失选择。

默认先训练历史预测与 VQ 原型，再冻结这些模块并准备任务缓存，最后训练检测器。缓存只保存标准化轨迹、观测掩码、两路预测和原型编号；完整描述及差异在块内恢复。大型实体的缓存准备可能较慢，磁盘占用会记录在 `task_cache/manifest.json`。

可按阶段执行，后续阶段读取已有检查点：

```bash
python -m asei run --datasets SMD --entities <实际实体名> --seeds 0 --stage representation
python -m asei run --datasets SMD --entities <实际实体名> --seeds 0 --stage cache
python -m asei run --datasets SMD --entities <实际实体名> --seeds 0 --stage detector
python -m asei run --datasets SMD --entities <实际实体名> --seeds 0 --stage calibrate
python -m asei run --datasets SMD --entities <实际实体名> --seeds 0 --stage evaluate
```

使用相同配置、数据路径和种子。已完成阶段可复用；不兼容的配置或缓存会报错，需要换一个 `--output`。中断的任务缓存可继续生成；训练中断后会从该阶段重新训练，已完成阶段不会重训。

## 指标

最终阈值只使用正常验证轨迹，默认 `alpha=0.01`，按方案附录的次序统计量确定，报警条件为严格 `score > threshold`。`0.005`、`0.001` 的敏感性结果复用同一正常验证分数。测试标签仅进入最终指标计算。

默认调用 TAB 的点级 Precision、Recall、F1、Accuracy，Range 指标，Affiliation 指标，以及六种连续分数指标。`auc_pr` 是 `average_precision_score`，即 AP。新增 ADD 对每个真实事件等权平均首次报警延迟，漏检记为事件长度，单位为采样点。事件覆盖、正常误报率和恢复拖尾另行报告。

主指标不做点调整。需要 `metrics.txt` 中其余标签指标时加 `--all-tab-label-metrics`，结果单列在 `supplementary`。回归指标提供 `asei.metrics.regression_metrics` 接口用于预测诊断，不与异常检测主指标混用。TAB 中 `best_*` 的测试标签阈值扫描不接入。

不可评分点保留在完整时间轴上，报警为 0；完整时间轴排序指标将其分数置于全部有效分数以下，并另给只含可评分点的 AP/AUC。缺省不压缩时间轴计算区间指标。无定义的指标写为 JSON `null`，记录原因；汇总不补值。

TAB 原版 VUS 对长事件可能计算较慢。`--skip-vus` 可在初步排查时跳过，正式结果应按一致配置计算；不能把跳过的字段当作已有结果。

## 输出与消融

```text
outputs/<dataset>/<entity>/seed_<seed>/
  data_provenance.json       # 文件哈希、边界、通道、标准化参数
  resolved_config.json
  representation.pt         # 冻结正常模型、原型缓存及使用率
  stage1_history.json
  task_cache/                # 可在检测头消融之间共享
  detectors/full/
    detector.pt
    calibration.json
    normal_validation_scores.npz
    bundle.pt               # 推断所需模型、统计量、通道顺序与阈值
    predictions.npz         # 完整测试时间轴
    result.json
```

`--diagnostics` 额外保存恢复门、状态差和原型差异轨迹。

检测头支持 `mlp`、`single_gru`、`no_alignment`、`zero_alignment`、`fixed_gate`、`independent_gru`。例如：

```bash
python -m asei run --datasets SMD --entities <实际实体名> --seeds 0 --variant no_alignment
```

保持其他配置即可复用正常模型及任务缓存。`--sources marginal joint` 关闭关系证据；`--sources relation marginal` 关闭联合证据。这里属于同一冻结表征下的来源消融。不同结构的参数匹配，以及官方外部基线的复现，仍需按正式实验预算另行组织。

## 流式推断与汇总

```python
from asei.pipeline import load_bundle

stream, channel_names = load_bundle("outputs/.../detectors/full/bundle.pt", device="cuda:0")
result = stream.push(current_sample)  # 顺序与 channel_names 一致；缺测填 NaN
later_results = stream.process(next_block)
stream.reset()  # 仅在新实体或独立记录开始时调用
```

同一个实例跨计算块保留历史和两条状态，不读标签，不在线更新原型或模型。

```bash
python -m asei predict --bundle outputs/.../bundle.pt --input-csv observations.csv --output alarms.csv
python -m asei aggregate --root outputs --output outputs/summary
```

汇总顺序为每个种子先在数据集内按实体宏平均，再跨数据集等权平均，最后计算种子均值与样本标准差，并保存有效数量。配对比较可加 `--baseline-root`、`--variant`、`--baseline-variant`、`--metric score.auc_pr`，按配对种子在实体内求均值后分层重采样实体。

本地验证：`python -m pytest -q tests`。测试中的小模型和合成数据仅用于检查实现，不是正式实验配置或检测效果证据。
