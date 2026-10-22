import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import geometry as g
import grasp_pose_suffix as m


class SuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot=json.loads((g.RESULTS/"simulator-kinematics.json").read_text())
        cls.limits=np.asarray(json.loads((g.RESULTS/"stage-expert-limit-diagnostic.json").read_text())["physical_limits"])
        q=np.asarray(cls.robot["reference_joints"])
        cls.commands=np.linspace(q,q+np.array([.03,.01,0,0,0,-.02,0]),40).astype(np.float32)
        final=g.forward_kinematics(cls.commands,cls.robot)[-1]
        final[:3,3]+=[.001,-.001,0]
        final[:3,:3]=Rotation.from_rotvec([0,.015,0]).as_matrix() @ final[:3,:3]
        cls.goal=np.r_[final[:3,3],Rotation.from_matrix(final[:3,:3]).as_quat()]

    def test_full_pose_retarget_preserves_prefix_and_reaches_goal(self):
        result,diag=m.solve_suffix(self.commands,self.goal,self.robot,self.limits,"SE3")
        np.testing.assert_array_equal(result[:diag["boundary"]+1],self.commands[:diag["boundary"]+1])
        self.assertLess(diag["planned_end_position_error_mm"],.03)
        self.assertLess(diag["planned_end_rotation_error_deg"],.02)
        self.assertLessEqual(diag["max_joint_change_rad"],.350001)

    def test_position_only_retains_original_rotation(self):
        p,r,b=m.targets(self.commands,self.goal,self.robot,"XYZ")
        original=g.forward_kinematics(self.commands,self.robot)
        np.testing.assert_allclose(r,original[:,:3,:3],atol=1e-12)
        np.testing.assert_allclose(p[-1],self.goal[:3],atol=1e-12)
        np.testing.assert_allclose(p[:b+1],original[:b+1,:3,3],atol=1e-12)

    def test_hold_and_original_commands_identical(self):
        for condition in ("HOLD","ORIGINAL"):
            result,_=m.solve_suffix(self.commands,self.goal,self.robot,self.limits,condition)
            np.testing.assert_array_equal(result,self.commands)

    def test_live_servo_reduces_measured_pose_error(self):
        q=self.commands[-1].astype(float)
        start=g.forward_kinematics(q[None],self.robot)[0]
        pose=np.r_[start[:3,3],Rotation.from_matrix(start[:3,:3]).as_quat()]
        command=m.servo_command(q,pose,self.goal,q,self.robot,self.limits)
        frame=g.forward_kinematics(command[None],self.robot)[0]
        self.assertLess(np.linalg.norm(frame[:3,3]-self.goal[:3]),np.linalg.norm(start[:3,3]-self.goal[:3]))
        target=Rotation.from_quat(self.goal[3:]).as_matrix()
        self.assertLess(Rotation.from_matrix(target @ frame[:3,:3].T).magnitude(),Rotation.from_matrix(target @ start[:3,:3].T).magnitude())
        self.assertLessEqual(np.max(np.abs(command-q)),.020001)

    def test_servo_accumulates_stationary_load_residual(self):
        q=self.commands[-1].astype(float)
        start=g.forward_kinematics(q[None],self.robot)[0]
        pose=np.r_[start[:3,3],Rotation.from_matrix(start[:3,:3]).as_quat()]
        first=m.servo_command(q,pose,self.goal,q,self.robot,self.limits,previous_target=q)
        second=m.servo_command(q,pose,self.goal,q,self.robot,self.limits,previous_target=first)
        np.testing.assert_allclose(second-q,2*(first-q),atol=1e-12)


if __name__=="__main__":unittest.main()
