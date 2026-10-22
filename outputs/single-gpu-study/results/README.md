# 当前动机实验的结果证据

## 正式结果

- `grasp-pose-protocol.json`：冻结四组协议 v2。
- `grasp-pose-val.json`：80 条正式记录；同前缀 `-summary.json`、`-audit.json`、`-episodes.csv` 和两张 PNG 为统计和核验。
- `grasp-pose-val-cases/`：80 例的控制、物理轨迹、预测、观测和终态图。
- `grasp-error-origin-val*`：离线误差来源报告、JSON、逐场景 CSV、核验和图。

## 当前不可缺少的来源

- `grasp-source-plans/`、`grasp-source-manifest.json`：同一批 20 个冻结抓取样本与专家目标。
- `grasp-source-predictions/`：原始和精炼关节预测，用于误差定位和来源核对。
- `simulator-kinematics.json`、`simulator-kinematics-agreement.json`：几何链及验证。
- `stage-expert-limit-diagnostic.json`：真实物理关节限位。
- `open_microwave-stage-substitution-val.json`、对应 summary 和 `stage-substitution-episodes.csv`：报告中早期阶段替换的辅助证据。
- `open_microwave-seed7-u20000-training.json`：当前检查点的训练来源。

## 工程检查

`grasp-pose-smoke*`、`grasp-pose-repro*`、`grasp-pose-protocol-v1-engineering.json` 保存与主实验直接相关的协议演进和工程检查，不混入正式 80 例。

已删除旧恢复、调速和其他非主线结果。保留在这里的 NPZ/JSON 是复核当前报告所需证据，随代码提交；完整 RLBench 示范与权重在 Git 忽略的 `local/state/`，不在本目录。

记录路径已转换成仓库相对形式，读取时自动解析；原始科学数值及数组不变，迁移前主要 JSON 在本地 `local/original-main-records.zip` 保留。

[完整报告](../../../reports/motivation/动机实验报告.md) · [复现说明](../../../docs/REPRODUCE.md)
