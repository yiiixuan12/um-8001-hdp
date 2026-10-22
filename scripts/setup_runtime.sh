#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate.sh"
if [[ ! -f "$COPPELIASIM_ROOT/libcoppeliaSim.so" ]]; then
    printf '%s\n' 'Install/extract CoppeliaSim Edu 4.1 first, then set HDP_COPPELIASIM_ROOT.' >&2
    exit 1
fi
if [[ ! -f "$HDP_UPSTREAM_ROOT/setup.py" ]]; then
    printf '%s\n' 'Initialize the pinned HDP submodule: git submodule update --init --recursive' >&2
    exit 1
fi
export PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV="${HDP_VENV:-$HDP_MOTIVATION_ROOT/local/venv}"
if [[ ! -x "$VENV/bin/python" ]]; then "$PYTHON_BIN" -m venv "$VENV"; fi
PYTHON="$VENV/bin/python"
"$PYTHON" -m pip install --upgrade pip 'setuptools<81' wheel
"$PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$PYTHON" -m pip install -r "$HDP_MOTIVATION_ROOT/requirements-runtime.txt"
DEPS="$HDP_MOTIVATION_ROOT/work/hdp-runtime/sources"
mkdir -p "$DEPS"
checkout() {
    local name="$1" url="$2" revision="$3"
    if [[ ! -d "$DEPS/$name" ]]; then
        git clone "$url" "$DEPS/$name"
        git -C "$DEPS/$name" checkout --detach "$revision"
    elif [[ -d "$DEPS/$name/.git" ]]; then
        if [[ "$(git -C "$DEPS/$name" rev-parse HEAD)" != "$revision" ]]; then
            printf 'Existing %s differs from the pinned revision; preserve it and use a fresh dependency directory.\n' "$name" >&2
            exit 1
        fi
    else
        printf 'Using existing local source snapshot: %s\n' "$DEPS/$name"
    fi
    "$PYTHON" -m pip install --no-deps --no-build-isolation -e "$DEPS/$name"
}
checkout PyRep https://github.com/stepjam/PyRep.git 8f420be8064b1970aae18a9cfbc978dfb15747ef
checkout RLBench https://github.com/stepjam/RLBench.git 02720bba4c73fe02eb75df946b8791b806028a9d
checkout YARR https://github.com/Yusufma03/YARR.git 0d769c133f31c7049ec8acfb57b0a71eedb01bc6
checkout CLIP https://github.com/openai/CLIP.git d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
"$PYTHON" -m pip install --no-deps --no-build-isolation -e "$HDP_UPSTREAM_ROOT"
printf 'Runtime installed. Use: export HDP_PYTHON=%q\n' "$PYTHON"
