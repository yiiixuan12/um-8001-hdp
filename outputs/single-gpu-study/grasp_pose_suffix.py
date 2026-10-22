"""Bounded grasp suffix SE(3) retargeting; no policy or simulator imports."""
import numpy as np
from scipy.spatial.transform import Rotation
import geometry as g
from geometry import fk_jacobian

CONDITIONS=["ORIGINAL","HOLD","XYZ","SE3"]
CONFIG={"suffix_start_fraction":.75,"preclose_seconds":1.,"iterations":100,
    "damping":.0001,"orientation_length_scale_m":.1,"max_ik_step_rad":.03,
    "joint_trust_region_rad":.35,"servo_gain":.8,"servo_max_step_rad":.02,
    "servo_target_base":"previous_command_integrated_measured_residual",
    "position_tolerance_m":1e-5,"rotation_tolerance_rad":1e-4,
    "preclose_position_qualification_mm":.5,"preclose_rotation_qualification_deg":.2,
    "pull_budget_seconds":15.}


def targets(commands,goal,calibration,condition):
    commands=np.asarray(commands)
    frames=g.forward_kinematics(commands,calibration)
    boundary=int(np.floor((len(commands)-1)*CONFIG["suffix_start_fraction"]))
    u=np.clip((np.arange(len(commands))-boundary)/max(1,len(commands)-1-boundary),0,1)
    weights=u**3*(10+u*(-15+6*u))
    position=frames[:,:3,3].copy()
    rotation=frames[:,:3,:3].copy()
    target=g.pose_matrix(goal)
    if condition in ("XYZ","SE3"):
        position+=weights[:,None]*(target[:3,3]-position[-1])
    if condition=="SE3":
        correction=Rotation.from_matrix(target[:3,:3] @ rotation[-1].T).as_rotvec()
        rotation=Rotation.from_rotvec(weights[:,None]*correction).as_matrix() @ rotation
    return position,rotation,boundary


def solve_suffix(commands,goal,calibration,limits,condition):
    commands=np.asarray(commands)
    position,rotation,boundary=targets(commands,goal,calibration,condition)
    if condition not in ("XYZ","SE3"):
        return commands.copy(),{"boundary":boundary,"iterations":0,"max_joint_change_rad":0.}
    nominal=np.asarray(commands,float)
    q=nominal.copy()
    lower=np.minimum(nominal,np.maximum(limits[0],nominal-CONFIG["joint_trust_region_rad"]))
    upper=np.maximum(nominal,np.minimum(limits[1],nominal+CONFIG["joint_trust_region_rad"]))
    scale=CONFIG["orientation_length_scale_m"]
    for iteration in range(CONFIG["iterations"]):
        frame,jac=fk_jacobian(q,calibration)
        ep=position-frame[:,:3,3]
        er=Rotation.from_matrix(rotation @ frame[:,:3,:3].swapaxes(-1,-2)).as_rotvec()
        active=(np.linalg.norm(ep,axis=1)>CONFIG["position_tolerance_m"]) | (np.linalg.norm(er,axis=1)>CONFIG["rotation_tolerance_rad"])
        active[:boundary+1]=False
        if not active.any():break
        error=np.c_[ep,scale*er]
        jac[:,3:]*=scale
        step=np.einsum("nij,nj->ni",jac.swapaxes(-1,-2),
            np.linalg.solve(jac @ jac.swapaxes(-1,-2)+CONFIG["damping"]**2*np.eye(6),error[...,None])[...,0])
        step*=np.minimum(1.,CONFIG["max_ik_step_rad"]/np.maximum(np.max(np.abs(step),axis=1),1e-12))[:,None]
        step[~active]=0
        before=np.square(error).sum(axis=1)
        accepted=~active
        for divisor in (1,2,4,8,16):
            proposal=np.clip(q+step/divisor,lower,upper)
            f=g.forward_kinematics(proposal,calibration)
            pe=position-f[:,:3,3]
            re=Rotation.from_matrix(rotation @ f[:,:3,:3].swapaxes(-1,-2)).as_rotvec()
            cost=np.square(pe).sum(axis=1)+scale**2*np.square(re).sum(axis=1)
            take=~accepted & (cost<before)
            q[take]=proposal[take];accepted|=take
        q[:boundary+1]=nominal[:boundary+1]
    result=q.astype(commands.dtype)
    result[:boundary+1]=commands[:boundary+1]
    final=g.forward_kinematics(result,calibration)
    return result,{"boundary":boundary,"iterations":iteration+1,
        "max_joint_change_rad":float(np.max(np.abs(result-nominal))),
        "planned_end_position_error_mm":float(np.linalg.norm(final[-1,:3,3]-position[-1])*1000),
        "planned_end_rotation_error_deg":float(np.degrees(Rotation.from_matrix(rotation[-1] @ final[-1,:3,:3].T).magnitude())),
        "common_prefix_commands_identical":bool(np.array_equal(result[:boundary+1],commands[:boundary+1]))}


def servo_command(measured_q,measured_pose,target_pose,anchor,calibration,limits,previous_target=None):
    """Use measured tip pose to cancel the residual calibrated FK bias.

    Writes a bounded target only; never resets actual joint or door state.
    A fixed duration is used in the experiment, independent of success labels.
    """
    q=np.asarray(measured_q,float)
    observed=g.pose_matrix(measured_pose)
    target=g.pose_matrix(target_pose)
    _,jac=fk_jacobian(q[None],calibration)
    ep=target[:3,3]-observed[:3,3]
    er=Rotation.from_matrix(target[:3,:3] @ observed[:3,:3].T).as_rotvec()
    scale=CONFIG["orientation_length_scale_m"]
    j=jac[0].copy();j[3:]*=scale
    step=j.T @ np.linalg.solve(j @ j.T+CONFIG["damping"]**2*np.eye(6),np.r_[ep,scale*er])
    step*=CONFIG["servo_gain"]
    step*=min(1.,CONFIG["servo_max_step_rad"]/max(np.max(np.abs(step)),1e-12))
    anchor=np.asarray(anchor,float)
    lower=np.minimum(anchor,np.maximum(limits[0],anchor-CONFIG["joint_trust_region_rad"]))
    upper=np.maximum(anchor,np.minimum(limits[1],anchor+CONFIG["joint_trust_region_rad"]))
    # Accumulate measured residual on the previous motor target. Re-anchoring
    # to measured q every step left a gravity/load-dependent steady offset.
    base=q if previous_target is None else np.asarray(previous_target,float)
    return np.clip(base+step,lower,upper)
