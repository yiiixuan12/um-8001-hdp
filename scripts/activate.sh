#!/usr/bin/env bash
# Source from any checkout location. This file does not change the caller's cwd.
HDP_MOTIVATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HDP_MOTIVATION_ROOT
export HDP_UPSTREAM_ROOT="${HDP_UPSTREAM_ROOT:-$HDP_MOTIVATION_ROOT/work/hdp}"
export HDP_STATE_ROOT="${HDP_STATE_ROOT:-$HDP_MOTIVATION_ROOT/local/state}"
export COPPELIASIM_ROOT="${HDP_COPPELIASIM_ROOT:-${COPPELIASIM_ROOT:-/opt/hdp/CoppeliaSim}}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
if [[ -n "${HDP_PYTHON:-}" ]]; then
    export HDP_PYTHON
elif [[ -x /opt/hdp/venv/bin/python ]]; then
    export HDP_PYTHON=/opt/hdp/venv/bin/python
else
    export HDP_PYTHON=python3
fi
