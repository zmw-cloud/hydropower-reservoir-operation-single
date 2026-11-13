# 从源码复现

所有命令均在当前仓库根目录运行。输入数据与论文配置随仓库保留；模型、教师、选择结果、评价轨迹和图件由运行生成，统一保存在 `outputs/`。各阶段共享这个目录，以便后续阶段读取前一阶段的产物。首次运行必须按顺序生成依赖。

```bash
python -m pip install -e ".[all]"
python -B scripts/audit_repository.py
python -B scripts/run_workflow.py inspect
```

`run_workflow.py run` 执行相应原算法入口，不复制旧模型。附加 `--show-commands` 只打印命令，不执行。大规模求解、训练与扫描可能耗时较长。

## DDP 与 SDP

DDP 在 EVAL 上直接求解，作为完美预见参考；SDP 先在 TRAIN 上生成策略，再评价：

```bash
python -B scripts/run_workflow.py run --algorithm ddp --stage solve
python -B scripts/run_workflow.py run --algorithm sdp --stage train
python -B scripts/run_workflow.py run --algorithm sdp --stage evaluate
```

## DQN

论文代表配置已保留。开发阶段用于验证，最终训练使用既定参数与预算，最后才读取独立 EVAL：

```bash
python -B scripts/run_workflow.py run --algorithm dqn --stage development
python -B scripts/run_workflow.py run --algorithm dqn --stage final
python -B scripts/run_workflow.py run --algorithm dqn --stage evaluate
```

种子为 1、42、100。具体网络、训练预算及惩罚系数见 `data/experiment_settings.yaml`（单库）或 `configs/paper/dqn5000*.yaml`（双库）。需要更改实验参数时先另存配置，避免将新实验误称为原论文实验。

## ISO-RF

单库使用 320 年拟合、80 年验证，再用全部 400 年 TRAIN 生成最终策略：

```bash
python -B scripts/run_workflow.py run --algorithm iso --stage development
python -B scripts/run_workflow.py run --algorithm iso --stage lambda-scan
python -B scripts/run_workflow.py run --algorithm iso --stage select-lambda
python -B scripts/run_workflow.py run --algorithm iso --stage final-train
python -B scripts/run_workflow.py run --algorithm iso --stage evaluate
```

扫描与选择在当前环境重新执行，选择结果以实际输出为准。论文代表 λ=700；固定实验与扫描参数见 `data/iso_rf_settings.yaml`、`configs/paper/` 和 `configs/sweeps/`。历史 ISO-SURF 源码保留供参考，不是论文正式 ISO-RF。

本单库仓库暂不包含单库绘图入口。

## 校验与版本

```bash
python -B -m unittest discover -s tests -v
```

默认 `audit_repository.py` 检查两份正式径流与 `data/datasets_manifest.json` 的 SHA-256，不需要训练产物。

如需校验本地评价结果，先将 `configs/paper/reference_results.json` 的轨迹和汇总路径指向自己生成的文件，再运行：

```bash
python -B scripts/audit_repository.py --results
```

完整历史训练环境锁文件未全部留存；已知版本记录在 `environments/historical_versions.json`，不作为完整环境锁定文件。新训练的 RF 策略应在兼容的软件环境中评价，重训结果可能受随机性、库版本与硬件影响。

极端入流超过下泄能力且水库已满时，历史物理实现可能报告未解决库容超限和非零水量平衡残差；保留相应测试，不在本次仓库清理中改变论文算法行为。

## 仓库约定

`outputs/`、旧 `artifacts/`、`reference_results/`、`provenance/` 目录，以及模型权重、缓存、日志均已忽略。不要使用 `git add -f` 强行提交运行产物。输入径流、工程曲线、实验配置与测试源码属于复现必需内容，应保留在 Git 中。
