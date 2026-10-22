# 组员复核与重新执行指南

## 1. 最快方式：直接重算已有实验

使用 Python 3.10，进入仓库根目录：

```bash
python -m pip install -r requirements-analysis.txt
python scripts/check_project.py
python scripts/analyze_saved.py
```

这条路径可以在 Windows 或 Linux 上运行，不需要 CUDA、PyTorch、RLBench、模型权重或示范数据。`results/` 中已经包含正式 80 例的记录和 160 份物理轨迹，分析会重算四组成功率、配对统计和规划误差链，并与既有结果比较。

输出自动写入 `local/reanalysis/时间戳/`。也可指定尚不存在的目录：

```bash
python scripts/analyze_saved.py --output local/reanalysis/my-check
```

主要预期结果：ORIGINAL/HOLD/XYZ 为 9/20，SE3 为 19/20；精炼后规划朝向误差均值约 3.1343°。科学整数结果精确比较，跨平台连续指标允许 1e-8 绝对、1e-7 相对偏差；配对统计仍精确比较；不允许因此修改正式成功判据。

## 2. 重新执行仿真实验

这是另一条路径，需要 Ubuntu/WSL、CUDA GPU、Python 3.10、CoppeliaSim Edu 4.1、HDP 及其仿真依赖。它不属于“只查看结果”所需的安装。

### 官方源码与环境

初始化官方 HDP 固定提交：

```bash
git submodule update --init --recursive
```

参考本机已验证组合安装系统库：Ubuntu 20.04、编译工具、Python 3.10 venv、`libffi-dev`、OpenGL/Mesa、X11/Qt 依赖。CoppeliaSim Edu 4.1 需要自行从官方渠道取得并解压；仿真二进制不随仓库分发。设置解压目录后运行安装入口：

```bash
export HDP_COPPELIASIM_ROOT=/path/to/CoppeliaSim
bash scripts/setup_runtime.sh
export HDP_PYTHON="$PWD/local/venv/bin/python"
```

安装脚本会固定 Torch 2.7.1/cu128、依赖版本和 PyRep/RLBench/YARR/CLIP 的提交。本机软件 OpenGL 渲染设置为 `LIBGL_ALWAYS_SOFTWARE=1`，仅影响渲染，扩散模型仍使用 CUDA。WSLg 提供显示环境；没有显示服务的 Linux 机器需另行配置显示环境，例如 Xvfb。

安装脚本已按本机依赖编写并完成 shell 语法检查；本次迁移验证复用了已安装环境，没有在另一台全新系统上重装验证，不能把它描述为所有系统的一键保证。

### 数据和冻结权重

数据与权重通过同一仓库的 [motivation-v1 Release](https://github.com/yiiixuan12/um-8001-hdp/releases/tag/motivation-v1) 分发，普通 Git 提交不包含大文件。附件为 `runtime-assets.zip`（约 674 MiB）和配套 `runtime-assets.json`；Windows 下载、校验与防覆盖解压步骤见 [README](../README.md#2-windows-下载校验并解压)。两个文件也已在本机 `local/share/` 准备好，当前本机不需要重新下载。包内有：

```text
state/
├── evaluations/open_microwave-seed7-u20000.pt
└── data/open_microwave/val/           # manifest + 20 条完整示范
```

将它解压到仓库的 `local/`，使实际路径为 `local/state/...`。旁边的 `runtime-assets.json` 给出文件大小和用于传输完整性检查的 SHA-256；不会因整理而重新训练模型。分享包的权重参数与原模型相同，只省略优化器等训练状态，适合推理，不能用作完整训练恢复检查点。本机另保留原始完整检查点。

包中不含 train100、其他数据划分、PerAct 权重或仿真器安装文件；当前四组实验不需要这些额外数据或 PerAct 权重。val20 已参与开发分析，不是新的未见测试集。

#### Linux 下载与解压

以下为原生 Linux 或希望在 WSL 内下载的组员准备，在仓库根目录运行。命令使用与 `motivation-v1` 代码配套的固定附件；也可手动下载至 `local/share/`，再执行校验和解压。

```bash
(
set -euo pipefail
repo_slug='yiiixuan12/um-8001-hdp'
release_tag='motivation-v1'
asset_base_url="https://github.com/$repo_slug/releases/download/$release_tag"
mkdir -p local/share
curl --fail --location "$asset_base_url/runtime-assets.json" --output local/share/runtime-assets.json
curl --fail --location "$asset_base_url/runtime-assets.zip" --output local/share/runtime-assets.zip
)
```

校验并解压（需要 Python 3.10；只在传输后校验一次即可）：

```bash
(
set -euo pipefail
expected_sha256="$(python3.10 -c 'import json; print(json.load(open("local/share/runtime-assets.json"))["sha256"])')"
printf '%s  %s\n' "$expected_sha256" local/share/runtime-assets.zip | sha256sum --check -
if [[ -e local/state/evaluations/open_microwave-seed7-u20000.pt || -e local/state/data/open_microwave/val ]]; then
    printf '%s\n' '目标权重或 val 数据已存在，请保留并核对，或改用干净的代码目录。' >&2
    exit 1
fi
python3.10 -m zipfile -e local/share/runtime-assets.zip local
)
```

下载或校验有错误时先解决错误，不要继续后续安装和运行。解压成功后回到仓库根目录继续下面的检查与实验。

准备完毕后：

```bash
bash scripts/run.sh verify --runtime
bash scripts/run.sh test
bash scripts/run.sh experiment --workers 1 --output "$PWD/local/rerun-val.json"
```

实验使用新的输出路径，保留已发表的 80 例证据。`--workers 1` 是单卡保守设置；本机原正式实验使用 2 个 worker。这里没有训练 PerAct，仍使用协议中的专家目标。

如仅验证一个场景，可直接调用 worker；必须使用新的结果文件：

```bash
source scripts/activate.sh
"$HDP_PYTHON" outputs/single-gpu-study/grasp_pose_experiment.py worker --episode 0 --condition SE3 --output "$PWD/local/single-SE3.json"
```

## 3. 本机现有环境

课题目录已从 C 盘迁到 `D:\Coding\HDP-Motivation`。数据、有效冻结权重及主线证据位于新目录；Python 和 CoppeliaSim 仍在 WSL `/opt/hdp`。原 C 盘位置只保留迁移指引。

本机直接运行：

```bash
cd /mnt/d/Coding/HDP-Motivation
export HDP_COPPELIASIM_ROOT=/opt/hdp/CoppeliaSim
export HDP_PYTHON=/opt/hdp/venv/bin/python
bash scripts/run.sh analyze
```

需要重新执行仿真时，在同一终端将最后一行换为 `bash scripts/run.sh verify --runtime`，通过后按上面的单场景或 80 次实验命令运行。本机已配置环境和资产，不要再次解压覆盖，也无需重复运行安装脚本。

`scripts/activate.sh` 从文件自身定位仓库，不依赖用户账号或原 C 盘工作区。仅 `hdp_root` 和 `state_root` 允许在检查点配置比较时随机器改变；训练、采样和数据配置仍须完全一致。

## 4. 继续思考方法时看哪里

- `work/hdp/rk_diffuser/robot.py`：原推理精炼计算旋转误差，但实际使用平移损失。
- `work/hdp/rk_diffuser/models/multi_level_diffusion.py`：双分支采样、全轨迹及额外终点精炼。
- `grasp_pose_suffix.py`：当前已验证的传统控制强基线，不等同于新的联合优化算法。
- `grasp_pose_experiment.py`：四组控制、固定预算、真实后续观测条件化和失败分母。
- `analyse_grasp_error_origin.py`：规划残差与执行差异的分离方式。

后续提出的方法应与现有 SE3 强基线比较，不能只对比原来的 9/20。新场景和不同训练种子应作为后续评估，不能将当前已分析 val20 称为未见测试集。
