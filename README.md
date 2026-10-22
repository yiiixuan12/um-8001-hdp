# HDP 抓取完整位姿残差：动机实验

本项目整理本组针对 HDP / RK-Diffuser 的问题诊断、误差来源分析和四组配对动机实验。目标正确不代表生成的关节终点完整到位：本地冻结模型精炼后仍有约 3.13° 朝向残差；完整位姿修正后，20 个已分析验证场景的后续拉门成功数由 9 增至 19。

这是单任务、单训练种子、专家目标的探索性证据，不是完整 HDP 复现，也不把收益归因于旋转这一唯一因素。

## 先读什么

- [完整动机报告与三张图](reports/motivation/动机实验报告.md)
- [组员如何直接复核、重新跑仿真](docs/REPRODUCE.md)
- [数据集说明](data/README.md)
- [当前实验代码导航](outputs/single-gpu-study/README.md)
- [结果证据索引](outputs/single-gpu-study/results/README.md)
- [第三方源码与许可](THIRD_PARTY_NOTICES.md)

## 获取代码

代码仓库：[yiiixuan12/um-8001-hdp](https://github.com/yiiixuan12/um-8001-hdp)。当前动机实验的固定发布版本为 [`motivation-v1`](https://github.com/yiiixuan12/um-8001-hdp/releases/tag/motivation-v1)。

```bash
git clone --recurse-submodules https://github.com/yiiixuan12/um-8001-hdp.git HDP-Motivation
cd HDP-Motivation
git checkout motivation-v1
git submodule update --init --recursive
```

已有代码目录只需在根目录运行 `git submodule update --init --recursive`。从 GitHub 下载的源码 ZIP 不包含子模块内容，重新运行仿真建议使用上述 Git 克隆方式。

## 路径一：不装仿真，直接分析已有结果

使用 Python 3.10，在仓库根目录运行（Windows 和 Linux 均可）：

```bash
python -m pip install -r requirements-analysis.txt
python scripts/check_project.py
python scripts/analyze_saved.py
```

结果写入新的 `local/reanalysis/` 子目录，原始证据不被覆盖。不需要 GPU、模型权重或 RLBench，重新核对完整 80 次记录、160 份物理轨迹和规划误差链。不同平台的浮点结果按 1e-8 绝对、1e-7 相对容差比较。

## 路径二：下载数据和权重，重新执行仿真

### 1. Release 附件与包含内容

数据与权重通过同一 GitHub 仓库的 Release 附件提供，大文件不进入普通 Git 历史：

| 附件 | 用途 |
| --- | --- |
| [`runtime-assets.zip`](https://github.com/yiiixuan12/um-8001-hdp/releases/download/motivation-v1/runtime-assets.zip) | 约 674 MiB；冻结 RK-Diffuser 推理权重和 `open_microwave` val20 完整示范 |
| [`runtime-assets.json`](https://github.com/yiiixuan12/um-8001-hdp/releases/download/motivation-v1/runtime-assets.json) | 包大小、SHA-256、训练步数和解压位置说明 |

下载入口：[motivation-v1 Release](https://github.com/yiiixuan12/um-8001-hdp/releases/tag/motivation-v1)。建议代码和附件均使用该固定版本；本机附件副本位于 `local/share/`。

包内权重与本地原检查点的模型参数相同，省略了优化器等训练状态；可重新执行当前四组实验，不能用于完整恢复训练。包内不含 train100、其他划分、PerAct 权重或 CoppeliaSim 安装文件。只分析已有结果时无需下载。

### 2. Windows 下载、校验并解压

在仓库根目录打开 PowerShell，执行以下下载命令。也可从 Release 页面手动下载这两个附件到 `local/share/`，然后从校验步骤继续。

```powershell
# 下载与当前代码版本配套的附件。
$repoSlug = 'yiiixuan12/um-8001-hdp'
$releaseTag = 'motivation-v1'
$assetBaseUrl = "https://github.com/$repoSlug/releases/download/$releaseTag"
New-Item -ItemType Directory -Path .\local\share -Force | Out-Null
Invoke-WebRequest -Uri "$assetBaseUrl/runtime-assets.json" -OutFile .\local\share\runtime-assets.json -ErrorAction Stop
Invoke-WebRequest -Uri "$assetBaseUrl/runtime-assets.zip" -OutFile .\local\share\runtime-assets.zip -ErrorAction Stop
```

下载完成后，或当前本机直接使用已有分享包，执行：

```powershell
# 校验并解压：发生校验失败或目标已存在时停止，不覆盖现有数据/权重。
$runtimeAssetInfo = Get-Content -LiteralPath .\local\share\runtime-assets.json -Raw | ConvertFrom-Json
$actualHash = (Get-FileHash -LiteralPath .\local\share\runtime-assets.zip -Algorithm SHA256).Hash
if ($actualHash -ne $runtimeAssetInfo.sha256) { throw '附件 SHA-256 不一致，请重新下载。' }
if ((Test-Path -LiteralPath .\local\state\evaluations\open_microwave-seed7-u20000.pt) -or
    (Test-Path -LiteralPath .\local\state\data\open_microwave\val)) {
    throw '目标权重或 val 数据已存在；请保留现有文件，核对后直接运行，或换一个干净的代码目录。'
}
Expand-Archive -LiteralPath .\local\share\runtime-assets.zip -DestinationPath .\local
```

解压完成后的路径应为：

```text
local/state/
├── evaluations/open_microwave-seed7-u20000.pt
└── data/open_microwave/val/
    ├── manifest.json
    └── episode_000000/ … episode_000019/
```

不要解压成 `local/runtime-assets/state/`。Linux 下载和解压命令见[复现指南](docs/REPRODUCE.md#linux-下载与解压)。

### 3. 在 Ubuntu / WSL 中准备仿真环境

仿真实验使用 Ubuntu/WSL、CUDA GPU、Python 3.10 和 CoppeliaSim Edu 4.1。在 Windows 中解压后，WSL 可直接访问同一份文件。以下以本机目录为例，组员应替换为自己的克隆目录：

```bash
cd /mnt/d/Coding/HDP-Motivation
git submodule update --init --recursive
# 先安装 Python 3.10、系统库，并自行取得和解压 CoppeliaSim Edu 4.1。
export HDP_COPPELIASIM_ROOT=/path/to/CoppeliaSim
bash scripts/setup_runtime.sh
export HDP_PYTHON="$PWD/local/venv/bin/python"
```

`/path/to/CoppeliaSim` 需要替换为真实安装目录。系统依赖和显示环境要求见[复现指南](docs/REPRODUCE.md)。安装脚本已做语法检查，但尚未在另一台全新系统上完成从零安装验证。

本机已有 `/opt/hdp` 环境，无需重复安装；每次打开新终端可按[本机运行说明](docs/REPRODUCE.md#3-本机现有环境)启用。

### 4. 检查环境并运行

在配置好环境的同一个 Ubuntu/WSL 终端中，从仓库根目录执行：

```bash
bash scripts/run.sh verify --runtime
bash scripts/run.sh test
```

建议先运行一个场景确认仿真可用：

```bash
source scripts/activate.sh
"$HDP_PYTHON" outputs/single-gpu-study/grasp_pose_experiment.py worker \
  --episode 0 --condition SE3 --output "$PWD/local/single-SE3.json"
```

再运行正式四组配对实验（20 个场景 × 4 组，共 80 次）：

```bash
bash scripts/run.sh experiment --workers 1 --output "$PWD/local/rerun-val.json"
```

输出使用新的文件名；若已存在，请换一个名称。单卡先使用 `--workers 1`。这些命令使用冻结模型和专家目标，不需要重新训练 PerAct 或 RK-Diffuser；原始证据中 ORIGINAL/HOLD/XYZ 各为 9/20，SE3 为 19/20，跨机器仿真不保证逐轨迹完全相同。

## 目录树与 GitHub 范围

```text
HDP-Motivation/
├── README.md
├── requirements-analysis.txt          # CPU 离线分析依赖
├── requirements-runtime.txt           # 仿真/训练依赖（Torch 单独安装）
├── scripts/                           # 复核、仿真入口和环境准备
├── docs/REPRODUCE.md
├── data/                              # 数据说明与四个划分清单，无大数据本体
├── reports/motivation/                # 报告与三张插图
├── outputs/single-gpu-study/
│   ├── grasp_pose_*.py                # 当前动机干预
│   ├── analyse_grasp_*.py             # 统计与误差追溯
│   ├── geometry.py                    # FK、雅可比、姿态及过滤工具
│   ├── experiment_common.py           # 配对统计、边界与名义时间网格
│   ├── simulation_support.py          # 当前使用的物理记录器
│   ├── project_paths.py               # 可迁移路径和检查点配置兼容
│   ├── study.py / simulator_kinematics.py
│   ├── test_*.py                      # 当前主线的 12 项定向检查
│   └── results/                       # 正式证据、源预测、几何与工程记录
├── work/hdp/                          # 固定提交的官方源码子模块
├── work/hdp-runtime/                  # 依赖版本清单；sources 为本地安装目录
└── local/                             # Git 忽略：数据、权重、分享包与新运行
    ├── state/data/open_microwave/
    ├── state/evaluations/
    ├── share/runtime-assets.zip       # 通过 Release 分发
    ├── share/runtime-assets.json
    └── reanalysis/
```

普通 Git 提交包含 `local/` 和本地依赖源码之外的当前代码、报告、结果证据、数据清单与官方源码版本引用。历史恢复/调速等方案及旧 TXT 已删除，共享功能已经抽成当前专用工具，避免依赖已删除的历史脚本。运行所需的 val20 数据和推理权重通过 Release 分发；train100、其他划分和完整训练状态仍只保留在本地。

## 本机路径

- Windows：`D:\Coding\HDP-Motivation`
- WSL：`/mnt/d/Coding/HDP-Motivation`
- 数据：`local/state/data/open_microwave`（100 train、20 val、20 test，另有旧实验采集的 20 fresh_feedback_v2）
- 冻结权重：`local/state/evaluations/open_microwave-seed7-u20000.pt`
- WSL Python/仿真器仍安装在 `/opt/hdp`，这是运行环境，不是另一个课题副本。

路径在运行时相对仓库定位，可用 `HDP_STATE_ROOT`、`HDP_UPSTREAM_ROOT`、`HDP_PYTHON` 覆盖。记录中只转换路径元数据，指标、控制参数、原始数组和模型参数不变；迁移前的主要 JSON 原件保存在本地忽略的 `local/original-main-records.zip`。
