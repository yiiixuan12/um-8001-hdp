"""Shared geometry from the validated simulator chain; no simulator imports."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
def pose_matrix(pose):
    pose = np.asarray(pose, float)
    result = np.broadcast_to(np.eye(4), pose.shape[:-1] + (4, 4)).copy()
    result[..., :3, :3] = Rotation.from_quat(pose.reshape(-1, 7)[:, 3:]).as_matrix().reshape(pose.shape[:-1] + (3, 3))
    result[..., :3, 3] = pose[..., :3]
    return result

def inverse(matrix):
    result = np.broadcast_to(np.eye(4), matrix.shape).copy()
    result[..., :3, :3] = matrix[..., :3, :3].swapaxes(-1, -2)
    result[..., :3, 3] = -np.einsum("...ij,...j->...i", result[..., :3, :3], matrix[..., :3, 3])
    return result

def forward_kinematics(joints, calibration):
    """Float64 evaluation of the already exported chain, no refitting."""
    joints = np.asarray(joints, float)
    flat = joints.reshape(-1, 7)
    reference = np.asarray(calibration["reference_joints"])
    frames = [*np.asarray(calibration["joint_frames_world"]), np.asarray(calibration["tip_frame_world"])]
    result = np.broadcast_to(np.eye(4), (len(flat), 4, 4)).copy()
    previous = np.eye(4)
    for i, frame in enumerate(frames):
        result = result @ np.linalg.solve(previous, frame)
        if i < 7:
            turn = np.broadcast_to(np.eye(4), result.shape).copy()
            angle = flat[:, i] - reference[i]
            turn[:, 0, 0] = turn[:, 1, 1] = np.cos(angle)
            turn[:, 1, 0], turn[:, 0, 1] = np.sin(angle), -np.sin(angle)
            result = result @ turn
        previous = frame
    return result.reshape(joints.shape[:-1] + (4, 4))
def fk_jacobian(joints, calibration):
    flat = np.asarray(joints, float).reshape(-1, 7)
    reference = np.asarray(calibration["reference_joints"])
    frames = [*np.asarray(calibration["joint_frames_world"]), np.asarray(calibration["tip_frame_world"])]
    result = np.broadcast_to(np.eye(4), (len(flat), 4, 4)).copy()
    previous = np.eye(4)
    points, axes = [], []
    for i, frame in enumerate(frames):
        result = result @ np.linalg.solve(previous, frame)
        if i < 7:
            points.append(result[:, :3, 3].copy())
            axes.append(result[:, :3, 2].copy())
            rotation = np.broadcast_to(np.eye(4), result.shape).copy()
            angle = flat[:, i]-reference[i]
            rotation[:, 0, 0] = rotation[:, 1, 1] = np.cos(angle)
            rotation[:, 1, 0], rotation[:, 0, 1] = np.sin(angle), -np.sin(angle)
            result = result @ rotation
        previous = frame
    axes, points = np.stack(axes, axis=1), np.stack(points, axis=1)
    jacobian = np.concatenate([np.cross(axes, result[:, None, :3, 3]-points).swapaxes(1, 2), axes.swapaxes(1, 2)], axis=1)
    return result, jacobian

def filter_indices(joints):
    indices = [0]
    for i in range(1, len(joints)-2):
        if np.max(np.abs(joints[indices[-1]]-joints[i])) > .05:
            indices.append(i)
    return np.array([*indices, len(joints)-1], int)
