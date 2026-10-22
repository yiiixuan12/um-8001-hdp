"""Recompute the published motivation evidence without GPU or RLBench."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs/single-gpu-study"
sys.path.insert(0, str(STUDY))
from project_paths import load_json
from analyse_grasp_pose import run as analyze_pose
from analyse_grasp_error_origin import run as analyze_origin


def assert_agreement(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert_agreement(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_agreement(a, b)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "local/reanalysis" / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = STUDY / "results/grasp-pose-val.json"
    copied = output / source.name
    shutil.copy2(source, copied)
    pose, _ = analyze_pose(copied, plots=True)
    origin, audit = analyze_origin(source, output / "grasp-error-origin-val")
    expected = load_json(STUDY / "results/grasp-pose-val-summary.json")
    expected_origin = load_json(STUDY / "results/grasp-error-origin-val.json")
    assert pose["paired_comparisons"] == expected["paired_comparisons"]
    for key in ("conditions", "SE3_XYZ_preclose_actual_position_difference_mm"):
        assert_agreement(pose[key], expected[key])
    for key in ("planning", "conditions", "original_diagnostics"):
        assert_agreement(origin[key], expected_origin[key])
    (output / "agreement.json").write_text(json.dumps({
        "conditions_and_paired_statistics_equal": True, "endpoint_diagnostics_equal": True,
        "cases": audit["paired_cases"], "new_simulator_episodes": 0,
        "source_evidence_untouched": True,
        "cross_platform_float_atol": 1e-8,
        "cross_platform_float_rtol": 1e-7,
    }, indent=2))
    print(f"Saved-data reproduction passed. Output: {output}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
