# 第三方来源

本仓库的动机实验建立在 [Dyson AI HDP](https://github.com/dyson-ai/hdp) 上。官方 HDP 通过子模块固定在提交 `383d5323b8817e1ce9acd18b7d4ee3860e80184f`，保留其原 MIT 许可及版权声明，见 `work/hdp/LICENSE`。

PyRep、RLBench、YARR 和 CLIP 是运行依赖，通过安装脚本获取固定提交；其源码、许可和资产以各自项目为准，不将它们的本机完整安装目录直接加入本项目 Git 提交。固定提交清单见 [source-revisions.json](work/hdp-runtime/source-revisions.json)。

CoppeliaSim 二进制不随本仓库或运行资产包分发。运行资产包只包含本组本地训练的推理检查点和本机采集的 val20 示范，供组内复现实验使用；数据与外部仿真资产的进一步分发应遵循原项目条件。
