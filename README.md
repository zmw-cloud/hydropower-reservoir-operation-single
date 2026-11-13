# Single reservoir paper benchmark

论文月尺度水电调度复现代码，包含 DDP70、SDP175、ISO-RF700、DQN650。正式数据划分为 TRAIN / 独立 EVAL。

仓库保留源码、README、复现说明、实验配置、必要输入数据、依赖声明、许可证和测试源码。训练模型、教师轨迹、验证扫描、评价结果、日志、图件及整合过程备份均由本地运行生成，不随源码提交。

## 安装与输入检查

在仓库根目录使用 Python 3.10+：

```bash
python -m pip install -e ".[all]"
python -B scripts/audit_repository.py
python -B scripts/run_workflow.py inspect
python -B -m unittest discover -s tests -v
```

输入检查只验证正式径流文件及其 SHA-256，不加载模型、不训练、不评价。

## 复现

先生成策略，再评价；运行结果保存在已被 Git 忽略的 `outputs/`。以下以 SDP 为例：

```bash
python -B scripts/run_workflow.py run --algorithm sdp --stage train
python -B scripts/run_workflow.py run --algorithm sdp --stage evaluate
```

完整阶段顺序、其他算法命令和绘图输入配置见 [复现说明](docs/REPRODUCTION.md)。命令附加 `--show-commands` 可只查看具体入口。初次克隆不包含预训练模型，不能跳过训练直接评价。

## 仓库结构

- `src/`：算法与公共物理模型。
- `scripts/`：运行、输入检查入口。
- `data/`：正式径流、工程曲线及必要配置。
- `configs/`：论文参数、扫描设置与结果路径模板。
- `tests/`：测试源码，不含测试产物。
- `environments/`：依赖要求及已知历史版本说明。
- `docs/REPRODUCTION.md`：复现步骤与协议边界。


## 代码与数据来源

正式源码采用 single-comparison-iso-rf 工作副本；正式 ISO 为逐月随机森林。保留的 ISO-SURF 是历史实现。 代码许可证见 [LICENSE](LICENSE)。

### 工程参数与曲线

- 正式输入为 [工程参数](data/reservoir_settings.txt)、[水位—库容曲线](data/level_storage_curve.csv)和[总下泄流量—尾水位原始点](data/tailwater_discharge_curve.csv)。
- 曲线与研究留存的 `4-水位-库容曲线.xlsx`、`7-尾水位泄流量曲线.xlsx` 中 `1小湾` 工作表对应。库容 CSV 包含运行范围内 75 点，单位为 10⁹ m³；原工程表的万 m³ 数值除以 100000 后转换为该单位。尾水位 CSV 包含 41 个原始点，流量单位为 m³/s，水位单位为 m。这些控制点也可与上游平台随包提供的小湾曲线对应。
- 小湾尾水位拟合另使用配置中的两个锚点 `(15666 m³/s, 1012.71 m)`、`(20683 m³/s, 1016.64 m)`；它们不属于上述 41 个原始点。现有归档尚未完整记录工程资料的原发布单位、报告编号及两个锚点的上游工程出处。

### 历史与合成径流

- 生成所依据的历史期为 1978—2008 年。合成数据由 `Target-Synthetic-Streamflow-Generator-monthly-dual-v1.0` 生成；本单库采用其中 Reservoir X 自然入流分量，生成器源码及历史输入随配套梯级平台的 `hydrology/generator/` 提供。
- 本仓库实际读取 [TRAIN](data/SynMonthlyInflow_train.xlsx) 和 [EVAL](data/SynMonthlyInflow_evaluation.xlsx)，每组 400 年、4800 个月，年份标签为 2009—2408，入流单位为 m³/s。TRAIN 随机种子为 `2026090301`，EVAL 为 `2026090302`。这些文件是合成序列，不是 400 年实测历史记录。
- 正式实验仅使用 TRAIN/EVAL；不采用上游历史版本的 dry/wet 情景文件。原始历史径流的具体提供单位、原始资料名称及部分上游数据的单独使用条款尚未在现有归档中完整记录。


## 复现边界

完整历史训练环境锁文件尚不齐全，已记录版本见 `environments/historical_versions.json`。重新训练所得结果可能受软件版本、随机性和硬件影响，不保证逐位一致。

单库 ISO 使用 320 年拟合、80 年验证；开发锚点 λ=1200，经验证扫描选定 λ=700，再使用全部 400 年 TRAIN 拟合最终策略。DQN 使用种子 1、42、100，最终预算为每种子 2300 回合。

## 结果管理

结果与模型保留在 `outputs/`；`.gitignore` 同时排除模型权重、缓存、日志及旧结果归档目录。输入数据和实验配置需要纳入版本控制，因此不按 CSV、Excel 或 JSON 扩展名一律忽略。
