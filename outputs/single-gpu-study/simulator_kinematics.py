"""Differentiable FK from the robot geometry actually loaded by RLBench.

The exported chain uses the captured posture as joint zero. Subtracting those
reference angles recovers the simulator's joint coordinates. The inherited IK
routine is unchanged; FK uses a quaternion converter with safe zero gradients.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
import pytorch_kinematics as pk
from rk_diffuser.robot import DiffRobot
from rk_diffuser.utils import matrix_to_quaternion, proc_quaternion


def chain_urdf(joint_frames, tip_frame):
    robot = ET.Element("robot", name="rlbench_simulator_geometry")
    ET.SubElement(robot, "link", name="world")
    previous = np.eye(4)
    parent = "world"
    for index, frame in enumerate([*joint_frames, tip_frame]):
        last = index == len(joint_frames)
        child = "sim_tip" if last else f"joint_link_{index}"
        ET.SubElement(robot, "link", name=child)
        joint = ET.SubElement(robot, "joint", name=f"joint_{index}", type="fixed" if last else "continuous")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        relative = np.linalg.solve(previous, np.asarray(frame, dtype=np.float64))
        rpy = Rotation.from_matrix(relative[:3, :3]).as_euler("xyz")
        fmt = lambda values: " ".join(f"{value:.17g}" for value in values)
        ET.SubElement(joint, "origin", xyz=fmt(relative[:3, 3]), rpy=fmt(rpy))
        if not last:
            ET.SubElement(joint, "axis", xyz="0 0 1")
        previous = np.asarray(frame)
        parent = child
    return ET.tostring(robot, encoding="unicode")


def load_calibration(path):
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    if calibration.get("schema_version") != 1 or calibration.get("mode") != "simulator_geometry":
        raise ValueError("Unsupported simulator kinematics calibration")
    frames = np.asarray(calibration["joint_frames_world"], dtype=np.float64)
    tip = np.asarray(calibration["tip_frame_world"], dtype=np.float64)
    reference = np.asarray(calibration["reference_joints"], dtype=np.float64)
    if frames.shape != (7, 4, 4) or tip.shape != (4, 4) or reference.shape != (7,):
        raise ValueError("Calibration must describe seven revolute joints and one tip")
    for matrix in [*frames, tip]:
        if (not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-5)
                or np.linalg.det(matrix[:3, :3]) <= 0):
            raise ValueError("Invalid rigid transform in simulator calibration")
    if not np.isfinite(reference).all():
        raise ValueError("Nonfinite reference joint angles")
    expected = chain_urdf(frames, tip)
    if calibration["urdf"] != expected:
        raise ValueError("Exported URDF does not match captured simulator geometry")
    return calibration


class SimulatorDiffRobot(DiffRobot):
    def __init__(self, calibration):
        nn.Module.__init__(self)
        self._robot = pk.build_serial_chain_from_urdf(calibration["urdf"], "sim_tip")
        self.register_buffer("_offset_mat", torch.eye(4)[None])
        self.register_buffer("_reference_joints", torch.tensor(calibration["reference_joints"], dtype=torch.float32))

    def forward_kinematics_batch(self, joints):
        matrix = self._robot.forward_kinematics(joints - self._reference_joints).get_matrix()
        # The installed pk converter differentiates sqrt(0), producing NaN
        # even for a position-only loss at some orientations. The upstream
        # repository also supplies this zero-subgradient-safe converter.
        quaternion = matrix_to_quaternion(matrix[:, :3, :3])
        quaternion_xyzw = torch.cat([quaternion[:, 1:], quaternion[:, :1]], dim=-1)
        return proc_quaternion(torch.cat([matrix[:, :3, 3], quaternion_xyzw], dim=-1))
