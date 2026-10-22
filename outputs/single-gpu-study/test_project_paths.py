import copy
import unittest
from pathlib import Path
import tempfile

from project_paths import ROOT, STATE_ROOT, resolve_path, load_json, portable_record, same_experiment_config


class PathTests(unittest.TestCase):
    def test_relative_and_legacy_paths(self):
        suffix = "outputs/single-gpu-study/results/grasp-pose-val.json"
        self.assertEqual(resolve_path(suffix), ROOT / suffix)
        self.assertEqual(resolve_path("/mnt/c/Users/User/Documents/Codex/2026-10-20/https-github-com-dyson-ai-hdp/" + suffix), ROOT / suffix)
        self.assertEqual(resolve_path("/opt/hdp/experiments/nbp-pilot/data/open_microwave"), STATE_ROOT / "data/open_microwave")

    def test_portability_roundtrip_preserves_scientific_values(self):
        import json
        data = {"path": str(ROOT / "outputs/single-gpu-study/results/grasp-pose-val.json"),
                "successes": 19, "rotation": .003273145097527379, "goal": [0., 1., .5]}
        normalized = portable_record(data)
        self.assertEqual(normalized["successes"], data["successes"])
        self.assertEqual(normalized["rotation"], data["rotation"])
        self.assertEqual(normalized["goal"], data["goal"])
        self.assertEqual(normalized["path"], "outputs/single-gpu-study/results/grasp-pose-val.json")
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "record.json"
            p.write_text(json.dumps(normalized))
            self.assertEqual(load_json(p), data)

    def test_only_machine_locations_ignored_in_checkpoint_comparison(self):
        first = {"hdp_root": "/old", "state_root": "/old-state", "training": {"updates": 20000}, "seed": 7}
        second = copy.deepcopy(first)
        second.update(hdp_root="/new", state_root="/new-state")
        self.assertTrue(same_experiment_config(first, second))
        second["training"]["updates"] = 60
        self.assertFalse(same_experiment_config(first, second))
        self.assertEqual(first["training"]["updates"], 20000)


if __name__ == "__main__":
    unittest.main()
