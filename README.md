# Single reservoir paper benchmark

本仓库固定论文单库正式实验：DDP70、SDP175、ISO-RF700、DQN650。只有 TRAIN 与独立 EVAL，无 dry/wet 正式情景，也暂不包含单库绘图脚本。

源码采用已审查的 single-comparison-iso-rf 工作副本；配置、模型及选择记录恢复自实际结果包。原目录与结果包不变。核心算法源码保持原字节，运行隔离由额外入口实现。

## Code and Data Sources（代码与数据来源）

### 代码与既有实验结果

- 正式算法源码采用 `Platform-for-monthly-hydropower-reservoir-operation-single-comparison-iso-rf` 工作副本；正式方法 ISO 使用逐月随机森林，代表点为教师惩罚系数 700。历史 ISO-SURF 模块仅保留作参考。
- 配置、冻结模型和评价结果来自 `Platform-for-monthly-hydropower-reservoir-operation-single-comparison` 中实际保存的结果包；不将结果归档目录视为另一套正式算法源码。
- 原文件路径及 SHA-256 见 [来源清单](provenance/source_manifest.json)，整合时的适配见 [整合变更记录](provenance/integration_changes.json)。代码许可证见 [LICENSE](LICENSE)。

### 工程参数与曲线

- 正式输入为 [工程参数](data/reservoir_settings.txt)、[水位—库容曲线](data/level_storage_curve.csv)和[总下泄流量—尾水位原始点](data/tailwater_discharge_curve.csv)。
- 曲线与研究留存的 `4-水位-库容曲线.xlsx`、`7-尾水位泄流量曲线.xlsx` 中 `1小湾` 工作表对应。库容 CSV 包含运行范围内 75 点，单位为 10⁹ m³；原工程表的万 m³ 数值除以 100000 后转换为该单位。尾水位 CSV 包含 41 个原始点，流量单位为 m³/s，水位单位为 m。这些控制点也可与上游平台随包提供的小湾曲线对应。
- 小湾尾水位拟合另使用配置中的两个锚点 `(15666 m³/s, 1012.71 m)`、`(20683 m³/s, 1016.64 m)`；它们不属于上述 41 个原始点。现有归档尚未完整记录工程资料的原发布单位、报告编号及两个锚点的上游工程出处。

### 历史与合成径流

- 生成所依据的历史期为 1978—2008 年。合成数据由 `Target-Synthetic-Streamflow-Generator-monthly-dual-v1.0` 生成；本单库采用其中 Reservoir X 自然入流分量，生成器源码及历史输入随配套梯级平台的 `hydrology/generator/` 提供。
- 本仓库实际读取 [TRAIN](data/SynMonthlyInflow_train.xlsx) 和 [EVAL](data/SynMonthlyInflow_evaluation.xlsx)，每组 400 年、4800 个月，年份标签为 2009—2408，入流单位为 m³/s。TRAIN 随机种子为 `2026090301`，EVAL 为 `2026090302`。这些文件是合成序列，不是 400 年实测历史记录。
- 正式实验仅使用 TRAIN/EVAL；不采用上游历史版本的 dry/wet 情景文件。原始历史径流的具体提供单位、原始资料名称及部分上游数据的单独使用条款尚未在现有归档中完整记录。

详细复现步骤见 [REPRODUCTION.md](docs/REPRODUCTION.md)。当前文件核验使用 [发布清单](provenance/release_manifest.json)；本次 README 更新前的文档和清单原件保存在 [文档变更归档](provenance/documentation_updates/readme_sources_20261025/change.json)。

## 先核验既有结果

在本目录中使用 Python 3.10+：

~~~text
python -m pip install -r environments/requirements-audit.txt
python -B scripts/audit_repository.py
python -B scripts/run_workflow.py inspect
python -B -m unittest discover -s tests -p test_integration.py -v
~~~

核验入口只读取保存的轨迹、JSON和哈希，不加载策略、不重新评价或训练。

| 方法 | λ | 年均发电量 TWh/yr | 保证率 % |
|---|---:|---:|---:|
| DDP | 70 | 21.661920 | 79.770833 |
| SDP | 175 | 20.805226 | 80.062500 |
| ISO-RF | 700 | 20.413238 | 81.979167 |
| DQN | 650 | 19.716678 | 79.930556 |

DQN 为种子1、42、100的均值。ISO 的旧79.98%属于ISO-SURF1100，与当前RF700不匹配。

## 仓库内容

- src：原始物理、DDP、SDP、ISO-RF及DQN；保留的ISO-SURF是历史源码，不是正式ISO。
- data：正式TRAIN/EVAL、原字节工程曲线和物理参数、有效冻结配置。
- configs/paper：正式代表参数、结果清单及工作流。
- configs/sweeps：实际扫描设置。
- artifacts：固定SDP策略、RF700及DQN三种子权重，选择证据。
- reference_results：保存的正式评价轨迹和JSON，供独立核验。
- provenance：来源、原始配置、源码快照、适配记录和发布文件哈希。
- scripts/run_workflow.py：准备独立工作目录，再显式执行选定阶段。
- docs：复现命令、SI依据及实验局限。
- outputs：新操作唯一允许的输出位置，不是历史结果来源。

## 独立评价及训练

~~~text
python -B scripts/run_workflow.py prepare --algorithm dqn --stage evaluate
python -B scripts/run_workflow.py execute outputs/runs/上一步生成的目录名
~~~

第一条只准备，第二条才执行。SDP与ISO使用相同的evaluate阶段。DDP的solve阶段包含完整有限时域求解，不能当作读取结果的快速命令。

正式算法需要另行安装项目及其依赖：python -m pip install -e ".[all]"。加载冻结RF前须匹配scikit-learn1.7.2，入口会核对版本。正式训练、开发及求解阶段的说明见 docs/REPRODUCTION.md。本次整合没有执行这些实验。

## 复现边界

RF原模型使用sklearn1.7.2。完整原训练环境锁文件并未全部保存在历史结果中；已知版本见 environments/historical_versions.json。加载旧pickle时应使用对应版本，不自动重存模型。默认安装范围是可安装环境说明，不冒充原实验的完整锁文件。

将本仓库整体移动后，数据核验与新准备工作仍使用相对路径；已准备工作目录需要重新prepare。历史JSON中的原服务器路径作为溯源文本保留。
