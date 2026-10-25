# Single reservoir paper benchmark

本仓库固定论文单库正式实验：DDP70、SDP175、ISO-RF700、DQN650。只有 TRAIN 与独立 EVAL，无 dry/wet 正式情景，也暂不包含单库绘图脚本。

源码采用已审查的 single-comparison-iso-rf 工作副本；配置、模型及选择记录恢复自实际结果包。原目录与结果包不变。核心算法源码保持原字节，运行隔离由额外入口实现。

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
