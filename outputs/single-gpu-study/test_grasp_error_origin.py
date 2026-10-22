import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import analyse_grasp_error_origin as a
import geometry as g


class ErrorOriginTests(unittest.TestCase):
    def test_planning_and_tracking_errors_are_separate(self):
        goal = np.eye(4)
        planned = np.eye(4)
        planned[:3, :3] = Rotation.from_euler("z", 3, degrees=True).as_matrix()
        actual = planned.copy()
        actual[:3, 3] = [.002, 0, 0]
        self.assertAlmostEqual(a.error(planned, goal)["rotation_deg"], 3)
        self.assertAlmostEqual(a.error(actual, goal)["rotation_deg"], 3)
        self.assertAlmostEqual(a.error(actual, planned)["rotation_deg"], 0)
        self.assertAlmostEqual(a.error(actual, planned)["position_mm"], 2)

    def test_quaternion_sign_and_world_frame_preserve_errors(self):
        p = np.array([.2, -.1, .7, .1, -.2, .3, .9])
        reference = g.pose_matrix([.3, .2, .6, -.2, .1, .2, .9])
        frame = g.pose_matrix([.8, -.5, .3, -.4, .2, .1, .8])
        one = a.error(g.pose_matrix(p), reference)
        two = a.error(g.pose_matrix(np.r_[p[:3], -p[3:]]), reference)
        three = a.error(frame @ g.pose_matrix(p), frame @ reference)
        for key in one:
            self.assertAlmostEqual(one[key], two[key], places=10)
            self.assertAlmostEqual(one[key], three[key], places=10)

    def test_reject_incomplete_or_duplicate_scientific_records(self):
        data = {"collection_complete": True, "protocol": {"scenes": 1},
                "records": [{"episode": 0, "condition": co, "scientific_record_complete": True}
                            for co in a.CONDITIONS]}
        self.assertEqual(len(a.collect_groups(data)), 4)
        for mode in ("missing", "duplicate", "incomplete"):
            invalid = copy.deepcopy(data)
            if mode == "missing":
                invalid["records"].pop()
            elif mode == "duplicate":
                invalid["records"].append(invalid["records"][0])
            else:
                invalid["records"][0]["scientific_record_complete"] = False
            with self.assertRaises(ValueError):
                a.collect_groups(invalid)


if __name__ == "__main__":
    unittest.main()
