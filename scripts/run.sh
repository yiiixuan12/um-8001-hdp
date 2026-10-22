#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate.sh"
MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  analyze) exec "$HDP_PYTHON" "$HDP_MOTIVATION_ROOT/scripts/analyze_saved.py" "$@";;
  verify) exec "$HDP_PYTHON" "$HDP_MOTIVATION_ROOT/scripts/check_project.py" "$@";;
  test)
    cd "$HDP_MOTIVATION_ROOT/outputs/single-gpu-study"
    exec "$HDP_PYTHON" -m unittest -v test_grasp_pose_suffix test_grasp_pose_analysis test_grasp_error_origin test_project_paths "$@";;
  experiment)
    cd "$HDP_MOTIVATION_ROOT/outputs/single-gpu-study"
    exec "$HDP_PYTHON" grasp_pose_experiment.py run "$@";;
  collect|train)
    cd "$HDP_MOTIVATION_ROOT/outputs/single-gpu-study"
    exec "$HDP_PYTHON" study.py "$MODE" "$@";;
  *) printf '%s\n' 'Usage: bash scripts/run.sh analyze|verify|test|experiment|collect|train [options]';;
esac
