# 依赖安装目录

`source-revisions.json` 保留 PyRep、RLBench、YARR、CLIP 和 HDP 的来源提交。`sources/` 是本机已安装依赖源码，Git 忽略，不应直接当成项目代码上传。

新环境通过 [setup_runtime.sh](../../scripts/setup_runtime.sh) 获取固定提交并安装。旧部署和报告生成脚本已清理。仿真二进制和 Python 虚拟环境仍由 WSL 管理。

[复现说明](../../docs/REPRODUCE.md)
