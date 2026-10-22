"""Current experiment bookkeeping, execution grid and paired statistics."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import geometry as g
from project_paths import portable_record, resolve_path
HERE, RESULTS = g.HERE, g.RESULTS
MANIFEST = RESULTS / "grasp-source-manifest.json"
BOUNDS = {"tip_position_mm": .1, "tip_rotation_deg": .05, "max_joint_delta_deg": .05,
    "door_angle_delta_deg": .01, "gripper_joint_delta_mm": .05,
    "tip_in_door_position_mm": .1, "door_position_mm": .1, "door_rotation_deg": .02,
    "arm_velocity_delta_rad_s": .01, "door_velocity_delta_rad_s": .01,
    "gripper_velocity_delta_m_s": .001, "contact_boolean_mismatch": 0.}
def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(portable_record(value), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)

def matrix_rotation_error(a, b):
    return np.degrees(Rotation.from_matrix(a[..., :3, :3].swapaxes(-1, -2) @ b[..., :3, :3]).magnitude())

def comparison(current, reference):
    def pose_error(a, b):
        a, b = g.pose_matrix(a), g.pose_matrix(b)
        return float(np.linalg.norm(a[:3, 3]-b[:3, 3])*1000), float(matrix_rotation_error(a, b))
    tip_p, tip_r = pose_error(current["tip_pose"], reference["tip_pose"])
    door_p, door_r = pose_error(current["door_pose"], reference["door_pose"])
    relative_p, _ = pose_error(current["tip_in_door"], reference["tip_in_door"])
    max_delta = lambda key, scale=1.: float(np.max(np.abs(np.asarray(current[key])-reference[key]))*scale)
    contacts = lambda x: [c["points"] > 0 for c in x["contacts"]]
    return {"tip_position_mm": tip_p, "tip_rotation_deg": tip_r,
        "max_joint_delta_deg": max_delta("joints", 180/np.pi),
        "door_angle_delta_deg": max_delta("door_angle_rad", 180/np.pi),
        "gripper_joint_delta_mm": max_delta("gripper_joints", 1000),
        "tip_in_door_position_mm": relative_p, "door_position_mm": door_p, "door_rotation_deg": door_r,
        "arm_velocity_delta_rad_s": max_delta("arm_joint_velocities"),
        "door_velocity_delta_rad_s": max_delta("door_velocity"),
        "gripper_velocity_delta_m_s": max_delta("gripper_joint_velocities"),
        "contact_boolean_mismatch": float(contacts(current) != contacts(reference))}
def exact_p(positive, negative):
    n=positive+negative
    return min(1.,2*sum(math.comb(n,k) for k in range(min(positive,negative)+1))/2**n) if n else 1.

def pair(group, first, second):
    ids=sorted(group[first])
    if ids != sorted(group[second]): raise ValueError("Unpaired scenes")
    difference=np.array([int(group[first][i]["task_success"])-int(group[second][i]["task_success"]) for i in ids])
    rng=np.random.default_rng(91028)
    samples=difference[rng.integers(0,len(ids),(4000,len(ids)))].mean(axis=1)
    gains=[i for i,d in zip(ids,difference) if d>0]
    losses=[i for i,d in zip(ids,difference) if d<0]
    return {"n":len(ids),"difference":float(difference.mean()),"paired_scene_bootstrap_95ci":np.quantile(samples,[.025,.975]).tolist(),
        "success_gains":gains,"success_losses":losses,"two_sided_exact_p":exact_p(len(gains),len(losses)),
        "note":"Observed paired contrast; percentile intervals can be degenerate with few discordances"}
def load_arrays(path):
    with np.load(resolve_path(path),allow_pickle=False) as values:
        result={k:values[k].copy() for k in values.files}
    if any(not np.isfinite(v).all() for v in result.values()): raise ValueError(f"Nonfinite arrays: {path}")
    return result

def equal(a,b):
    return set(a)==set(b) and all(np.array_equal(a[k],b[k]) for k in a)
def ensure_new_output(path, check_log=False):
    """Refuse reuse of an incomplete attempt's artifacts or driver log."""
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    artifact=path.parent/(path.stem+"-artifacts")
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(f"Preserve partial attempt at {artifact}; use a fresh output path")
    if check_log and path.with_suffix(".log").exists():
        raise FileExistsError(f"Preserve previous log at {path.with_suffix('.log')}; use a fresh output path")

def nominal_commands(arm,path):
    """Dry-run ONLY the RML clock, with no physics or target-position writes."""
    from pyrep.backend import sim
    from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
    # Preserve upstream float32 action arithmetic. NumPy's appended scalar
    # gripper command does NOT promote this array to float64.
    path=np.asarray(path)
    trajectory=ArmConfigurationPath(arm,path.reshape(-1))
    lengths=trajectory._get_path_point_lengths()
    handle=trajectory._get_rml_handle()
    dt=sim.simGetSimulationTimeStep()
    commands=[path[0].copy()]
    try:
        state=0
        while state==0:
            state,pva=sim.simRMLStep(handle,dt,1)
            if state<0: raise RuntimeError("RML generation failed")
            pos=pva[0]
            for i in range(len(lengths)-1):
                if lengths[i]<=pos<=lengths[i+1] or i==len(lengths)-2:
                    t=(pos-lengths[i])/(lengths[i+1]-lengths[i])
                    commands.append(path[i]+(path[i+1]-path[i])*t)
                    break
            if len(commands)>10000: raise RuntimeError("Invalid RML duration")
    finally:
        sim.simRMLRemove(handle)
    return np.asarray(commands)
