# 已生成的 RLBench 数据

本组已经在 RLBench 中生成了 `open_microwave` variation 0 的专家示范，并非只下载了空框架。

| 划分 | 示范数 | 场景种子 | 当前用途 |
| --- | ---: | --- | --- |
| train | 100 | 100000–100099 | 本地 RK-Diffuser 训练 |
| val | 20 | 200000–200019 | 当前正式动机实验；这些场景已用于开发分析 |
| test | 20 | 300000–300019 | 原先生成的其他评估数据；不是本报告的独立测试结果 |
| fresh_feedback_v2 | 20 | 400000–400019 | 旧反馈实验采集；不作为当前主结果 |

总计 160 条示范。前 3 个基本划分共 140 条；另 20 条旧实验数据仍保留在本地，因为重新生成有成本，但不进入普通 Git 提交。

完整数据已迁到本机 `D:\Coding\HDP-Motivation\local\state\data\open_microwave`，WSL 对应 `/mnt/d/Coding/HDP-Motivation/local/state/data/open_microwave`。每个 episode 包含：

```text
episode_000000/
├── arrays.npz     # 观测、关节、末端位姿等数组
├── demo.pkl       # RLBench 示范及可用于重置的场景信息
└── preview.png
```

仓库中的 [dataset-index.json](dataset-index.json) 和 [manifests/](manifests/) 只保存划分、种子、关键帧和生成来源。原始图像/点云示范不随普通 Git 上传。

组员只复核现有动机结果时不需要完整数据；重新执行四组实验时需要 val20，它与推理权重已打包为 `runtime-assets.zip`，配套 `runtime-assets.json` 提供 SHA-256。附件通过同一仓库的 [motivation-v1 Release](https://github.com/yiiixuan12/um-8001-hdp/releases/tag/motivation-v1) 分发；下载、校验和解压步骤见 [README](../README.md#路径二下载数据和权重重新执行仿真)。本机两个附件位于 `local/share/`。训练需要 train100，当前分享包不包含它，可另行提供本地数据，或在相同仿真依赖下重新采集。

重新采集命令（已安装完整仿真环境后，从根目录运行）：

```bash
bash scripts/run.sh collect --splits train val test
```

已有数据会校验配置并复用完成记录，不应把新生成的不同场景覆盖旧实验数据。`fresh_feedback_v2` 是历史扩展划分，通用采集入口默认不生成它。
