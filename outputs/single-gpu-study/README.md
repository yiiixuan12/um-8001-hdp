# 当前动机实验代码

[总导航](../../README.md) · [报告](../../reports/motivation/动机实验报告.md) · [运行说明](../../docs/REPRODUCE.md)

| 文件 | 用途 |
| --- | --- |
| `grasp_pose_experiment.py` | 四组协议、单例 worker 与完整实验收集 |
| `grasp_pose_suffix.py` | 原实验的平滑末段 DLS 和反馈伺服 |
| `analyse_grasp_pose.py` | 80 次配对记录的核验、统计及绘图 |
| `analyse_grasp_error_origin.py` | 20 个规划终点与四组实际状态的离线追溯 |
| `geometry.py` | 原已验证的 FK、雅可比、位姿变换和过滤索引 |
| `experiment_common.py` | 原子记录、配对统计、前缀对比、名义 Reflexxes 网格 |
| `simulation_support.py` | 40 秒预算和实际物理记录器，无恢复逻辑 |
| `project_paths.py` | 仓库相对路径与历史位置兼容；科学配置仍严格校验 |
| `study.py` | 单 GPU 数据采集、训练、模型及观测基础设施 |
| `simulator_kinematics.py` | 已导出的本机可微运动学适配 |
| `experiment.yaml` | 原训练/诊断数值配置；机器位置在运行时解析 |
| `test_grasp_pose_suffix.py` | 5 项干预控制检查 |
| `test_grasp_pose_analysis.py` | 拒绝不完整收集 |
| `test_grasp_error_origin.py` | 3 项误差和数据保护检查 |
| `test_project_paths.py` | 3 项路径迁移/配置一致性检查 |

历史的恢复、门进度和其他探索代码已删除，当前模块不再导入它们。共享代码从旧模块中原样抽取，控制与统计参数不变。主实验使用的冻结源计划和原始模型预测已独立整理为 `results/grasp-source-plans/`、`results/grasp-source-predictions/`。

从仓库根目录执行：

```bash
python scripts/analyze_saved.py
```

在已安装仿真环境的 WSL 中：

```bash
bash scripts/run.sh test
bash scripts/run.sh experiment --workers 1 --output "$PWD/local/new-experiment.json"
```

新实验须用新路径，不能覆盖已有结果。`study.py` 的早期通用诊断接口不等于当前主实验；当前主线没有目标扰动或失败重试。
