"""Validate the portable evidence/code layout without changing results."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs/single-gpu-study"
sys.path.insert(0, str(STUDY))
from project_paths import load_json, ROOT as PATH_ROOT, STATE_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    assert ROOT == PATH_ROOT
    result = load_json(STUDY / "results/grasp-pose-val.json")
    assert result["collection_complete"] and len(result["records"]) == 80
    pairs = {(r["episode"], r["condition"]) for r in result["records"]}
    assert pairs == {(ep, co) for ep in range(20) for co in ("ORIGINAL", "HOLD", "XYZ", "SE3")}
    references = 0
    for row in result["records"]:
        assert row["scientific_record_complete"]
        for path in (row["grasp_controller_trace"], row["grasp_plan_file"], row["attempt_file"]):
            assert Path(path).is_file(), path
            references += 1
        for action in row["actions"]:
            assert Path(action["trace"]["trace_file"]).is_file()
            references += 1
    report = ROOT / "reports/motivation/动机实验报告.md"
    docs = [ROOT / "README.md", report, ROOT / "docs/REPRODUCE.md", ROOT / "data/README.md",
            ROOT / "THIRD_PARTY_NOTICES.md", STUDY / "README.md", STUDY / "results/README.md",
            ROOT / "work/hdp-runtime/README.md"]
    links = 0
    optional_upstream = 0
    for path in docs:
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if link.startswith(("http://", "https://", "#")):
                continue
            target = (path.parent / unquote(link.split("#", 1)[0].strip("<>"))).resolve()
            if not target.exists() and ROOT / "work/hdp" in target.parents:
                optional_upstream += 1
                continue
            assert target.exists(), (path, link)
            links += 1
    obsolete = {"feedback_recovery", "stage_substitution", "correction_control", "progress_experiment",
                "analyse_progress_oracle", "analyse_correction_control", "orientation_restore", "door_constraint_analysis"}
    modules = {}
    for path in STUDY.glob("*.py"):
        imports = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                imports.update(x.name.split(".")[0] for x in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        assert not imports & obsolete, (path, imports & obsolete)
        modules[path.name] = sorted(imports)
    assert not (STUDY / "archive/legacy-notes").exists()
    protocol = load_json(STUDY / "results/grasp-pose-protocol.json")
    assets_present = Path(protocol["checkpoint"]).is_file() and all(Path(x["demo_file"]).is_file() for x in protocol["items"])
    runtime = None
    if args.runtime:
        assert assets_present, "Provide the runtime-assets package in local/ first"
        import torch
        import study
        import simulation_support
        assert torch.cuda.is_available(), "CUDA is required for the current frozen-policy runner"
        runtime = {"cuda": torch.cuda.get_device_name(0), "data_root": str(STATE_ROOT),
                   "upstream_imported": True, "recorder_imported": True}
    print(json.dumps({"formal_cases": 80, "source_file_references": references,
                      "local_links": links, "optional_upstream_links_not_initialized": optional_upstream,
                      "current_python_modules": len(modules), "historical_imports_remaining": 0,
                      "runtime_assets_present": assets_present, "runtime": runtime}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
