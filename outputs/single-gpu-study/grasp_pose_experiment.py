"""Frozen-policy grasp suffix intervention before normal gripper closure."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from project_paths import load_json, portable_record, STATE_ROOT, same_experiment_config
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
import experiment_common as c
import geometry as g
import grasp_pose_suffix as m
from experiment_common import nominal_commands,ensure_new_output
from geometry import filter_indices

RESULTS=g.RESULTS
PROTOCOL=RESULTS/"grasp-pose-protocol.json"
CHECKPOINT=str(STATE_ROOT / "evaluations/open_microwave-seed7-u20000.pt")


def prepare():
    if PROTOCOL.exists():raise FileExistsError("Reuse frozen protocol")
    source=load_json(c.MANIFEST)
    items=[i for i in source["items"] if i["prefix"]=="MODEL"]
    if len(items)!=20:raise ValueError("Need all20 fixed model grasp samples")
    protocol={"version":1,"conditions":m.CONDITIONS,"scenes":20,"expected_cases":80,
        "items":items,"config":m.CONFIG,"checkpoint":CHECKPOINT,
        "kinematics_calibration":str(RESULTS/"simulator-kinematics.json"),
        "policy_seed":"80000+episode*100+1 for the one subsequent RK-Diffuser pull sample",
        "role":"Exploratory mechanism intervention on previously inspected val20, one seed, expert goals",
        "main_comparisons":[["SE3","HOLD"],["SE3","XYZ"]],
        "secondary_comparisons":[["SE3","ORIGINAL"],["XYZ","HOLD"],["HOLD","ORIGINAL"]],
        "scope":"Frozen original grasp sample; only last25% nominal grasp time can change before closure; no regrasp or recovery",
        "groups":{"ORIGINAL":"Unmodified original RML commands; original closure time",
            "HOLD":"Same original commands, hold original final joint target for1s before closure",
            "XYZ":"Smooth suffix XYZ retarget to expert position, preserve original planned orientation;1s measured-tip servo",
            "SE3":"Smooth suffix position+orientation retarget to expert pose;1s measured-tip SE3 servo"},
        "subsequent_policy":"Same frozen RK-Diffuser, expert pull target, seed and original controller; condition on each actual post-grasp observation. Do not replay an obsolete pull start posture.",
        "measurement":"True robot joint/EE state for bounded preclose servo; no visual module or object contact feedback used for correction",
        "denominator":"All20 per condition, including infeasible/insufficient pose correction, invalid plans and task failures",
        "smoke":"Episode0 all4 conditions; engineering only, excluded from formal80; no outcome-based parameter tuning",
        "common_prefix_bounds":c.BOUNDS,"no_retraining":True,"new_diffusion_samples_per_case":1,
        "common_physics_budget_seconds":40.,"pull_budget_seconds":15.,
        "infrastructure":"Fresh native simulator/model process per case; completed JSON saved before shutdown; preserve all attempts in separate paths, never overwrite partial artifacts"}
    c.atomic(PROTOCOL,protocol)
    print("Frozen",PROTOCOL,flush=True)


def worker(args):
    ensure_new_output(args.output)
    import study as s
    import simulation_support as f
    from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
    protocol=load_json(PROTOCOL)
    if protocol["config"]!=m.CONFIG:raise ValueError("Frozen configuration changed")
    item=next(i for i in protocol["items"] if i["episode"]==args.episode)
    output=Path(args.output)
    artifact=output.parent/(output.stem+"-artifacts");artifact.mkdir(exist_ok=False)
    with np.load(item["plan_file"],allow_pickle=False) as source:
        grasp=source["grasp"].copy()
        grasp_goal=source["grasp_goal"].copy()
        pull_goal=source["pull_goal"].copy()
    calibration=load_json(RESULTS/"simulator-kinematics.json")
    limits=np.asarray(load_json(RESULTS/"stage-expert-limit-diagnostic.json")["physical_limits"])
    began=time.perf_counter()
    row={"episode":args.episode,"condition":args.condition,"scene_seed":item["scene_seed"],
        "checkpoint":protocol["checkpoint"],"grasp_plan_file":item["plan_file"],
        "grasp_goal":grasp_goal.tolist(),"pull_goal":pull_goal.tolist(),
        "gripper_commands":item["gripper_commands"],"worker_pid":os.getpid(),
        "task_success":False,"execution_failure":None,"actions":[],"new_diffusion_calls":0,
        "scientific_record_complete":False,"worker_status":"loading_model","pull_timeout":False,
        "diffusion_seed":80000+args.episode*100+1}
    c.atomic(output,row)
    s.torch.set_num_threads(4)
    payload=s.torch.load(protocol["checkpoint"],map_location="cpu",weights_only=True)
    if payload["step"]!=20000 or payload["pilot_only"] or payload["task"]!="open_microwave":raise ValueError("Wrong checkpoint")
    if not same_experiment_config(payload["config"],s.OmegaConf.to_container(s.CFG,resolve=True)):raise ValueError("Checkpoint config mismatch")
    kin=s.inference_kinematics(SimpleNamespace(kinematics_calibration=Path(protocol["kinematics_calibration"])))
    robot,model=s.create_model(kin)
    model.load_state_dict(payload["model"]);model.eval();del payload

    class Mode(s.JointsTrajectoryActionMode):
        stage="grasp"
        def action(self,scene,action):
            if self.stage=="pull":
                if not np.isfinite(action).all() or np.any(action.reshape(-1,7)<limits[0]-1e-3) or np.any(action.reshape(-1,7)>limits[1]+1e-3):
                    raise s.InvalidTrajectoryError("Invalid predicted pull")
                path=ArmConfigurationPath(scene.robot.arm,self._pre_proc_traj(action))
                try:
                    for step in range(int(round(m.CONFIG["pull_budget_seconds"]/recorder.dt))):
                        done=path.step();scene.step()
                        success,terminate=scene.task.success()
                        if done or success or terminate:
                            row["pull_stop_reason"]="success" if success else ("terminate" if terminate else "path_end")
                            break
                    else:row["pull_timeout"]=True;row["pull_stop_reason"]="15s_budget"
                finally:
                    if path._rml_handle is not None and not path._path_done:
                        from pyrep.backend import sim
                        sim.simRMLRemove(path._rml_handle)
                return
            original=nominal_commands(scene.robot.arm,grasp[filter_indices(grasp)])
            corrected,solver=m.solve_suffix(original,grasp_goal,calibration,limits,args.condition)
            row["suffix_solver"]=solver
            row["nominal_grasp_seconds"]=(len(original)-1)*recorder.dt
            row["nominal_grasp_steps"]=len(original)-1
            np.savez_compressed(artifact/"grasp-command-plans.npz",nominal=original,corrected=corrected)
            vmax=np.asarray(scene.robot.arm.get_joint_upper_velocity_limits())
            previous=original[0].copy()
            rows=[]
            def command_step(target,phase,index):
                nonlocal previous
                bounded=np.minimum(np.maximum(target,previous-vmax*recorder.dt),previous+vmax*recorder.dt)
                if not np.isfinite(bounded).all() or np.any(bounded<limits[0]-1e-3) or np.any(bounded>limits[1]+1e-3):
                    raise s.InvalidTrajectoryError("Invalid suffix target")
                scene.robot.arm.set_joint_target_positions(bounded)
                scene.step()
                state=snapshot()
                rows.append({"time_s":len(rows)*recorder.dt,"phase":phase,"reference_index":index,
                    "command":bounded.copy(),"joints":np.asarray(state["joints"]),"tip_pose":np.asarray(state["tip_pose"]),
                    "joint_velocities":np.asarray(state["arm_joint_velocities"]),
                    "velocity_limited":bool(np.max(np.abs(bounded-target))>1e-10)})
                previous=bounded.copy()
            for index in range(1,len(original)):
                if index==solver["boundary"]+1:
                    row["common_prefix_checkpoint"]=snapshot()
                    row["common_prefix_steps"]=index-1
                    c.atomic(output,{**row,"worker_status":"executing_suffix"})
                command_step(corrected[index],0,index)
            row["before_extra_second"]=snapshot()
            # Preserve the nominal predicted orientation in XYZ, even while
            # measured-tip feedback removes the calibrated FK translation bias.
            nominal_final=g.forward_kinematics(original[-1:],calibration)[0]
            xyz_goal=np.r_[grasp_goal[:3],Rotation.from_matrix(nominal_final[:3,:3]).as_quat()]
            if args.condition!="ORIGINAL":
                for _ in range(int(round(m.CONFIG["preclose_seconds"]/recorder.dt))):
                    if args.condition=="HOLD":target=original[-1]
                    else:
                        state=snapshot()
                        target=m.servo_command(state["joints"],state["tip_pose"],grasp_goal if args.condition=="SE3" else xyz_goal,
                            original[-1],calibration,limits,previous_target=previous)
                    command_step(target,1,len(original)-1)
            row["preclose_state"]=snapshot()
            row["preclose_pose_error"]=pose_error(row["preclose_state"],grasp_goal)
            np.savez_compressed(artifact/"grasp-controller.npz",**{k:np.asarray([r[k] for r in rows]) for k in rows[0]},
                joint_velocity_limits=vmax,initial_command=original[0])
            row["grasp_controller_trace"]=str(artifact/"grasp-controller.npz")
            row["grasp_controller_steps"]=len(rows)
            c.atomic(output,{**row,"worker_status":"preclose_complete"})

    arm_mode=Mode(64)
    env=s.Environment(s.MoveArmThenGripper(arm_mode,s.Discrete()),obs_config=s._create_obs_config(["front"],[128,128]),headless=True)
    recorder=None
    env.launch()
    try:
        task=env.get_task(s.task_file_to_task_class("open_microwave"))
        door=s.Joint("microwave_door_joint")
        with Path(item["demo_file"]).open("rb") as handle:demo=s.pickle.load(handle)
        s.seed_all(170000+args.episode*10)
        obs=task.reset_to_demo(demo)[1]
        recorder=f.BudgetRecorder(env,door)
        def snapshot():
            return {**recorder.snapshot(),"door_pose":recorder.door_shape.get_pose().tolist(),
                "arm_joint_velocities":[j.get_joint_velocity() for j in env._robot.arm.joints],
                "gripper_joint_velocities":[j.get_joint_velocity() for j in env._robot.gripper.joints],
                "door_velocity":door.get_joint_velocity()}
        def pose_error(state,goal):
            cm,deg=s.pose_errors(state["tip_pose"],goal)
            return {"position_mm":cm*10,"rotation_deg":deg,
                "max_joint_velocity_rad_s":float(np.max(np.abs(state["arm_joint_velocities"]))),
                "full_pose_qualified":bool(cm*10<=m.CONFIG["preclose_position_qualification_mm"] and deg<=m.CONFIG["preclose_rotation_qualification_deg"])}
        row["initial_state"]=snapshot()
        recorder.total_steps,recorder.enforce_budget=0,True
        def execute(path,command,name):
            nonlocal obs
            arm_mode.stage=name
            entry={"name":name,"start_state":snapshot()}
            recorder.start()
            try:
                obs,reward,terminate=task.step(np.r_[path.reshape(-1),command])
                row["task_success"],row["terminated"]=bool(reward>=1),bool(terminate)
            finally:
                entry["trace"]=recorder.finish(artifact/f"{name}-trace.npz")
                entry["end_state"]=snapshot()
                row["actions"].append(entry)
                c.atomic(output,{**row,"worker_status":name+"_recorded"})
        execute(grasp,item["gripper_commands"][0],"grasp")
        row["handoff"]=snapshot()
        row["postclose_pose_error"]=pose_error(row["handoff"],grasp_goal)
        if not row["task_success"]:
            s.seed_all(row["diffusion_seed"])
            joints=np.repeat(obs.joint_positions[None],2,axis=0)
            poses=np.stack([obs.gripper_pose,pull_goal]).astype(np.float32)
            pcd=s.cv2.resize(obs.front_point_cloud,(64,64),interpolation=s.cv2.INTER_NEAREST)
            rgb=s.cv2.resize(obs.front_rgb,(64,64))
            proprio=obs.get_low_dim_data()
            np.savez_compressed(artifact/"pull-input.npz",joint_positions=joints,gripper_poses=poses,pcd=pcd,rgb=rgb,proprio=proprio)
            batch=s.model_input(joints,poses,pcd,rgb,proprio,robot)
            row["worker_status"]="sampling_pull";c.atomic(output,row)
            prediction=model.conditional_sample(**batch,verbose=False,return_diffusion=False)
            pull=prediction["multi"]["joint_positions"][0].detach().cpu().numpy()
            row["new_diffusion_calls"]+=1
            np.savez_compressed(artifact/"pull-prediction.npz",refined_joints=pull,
                raw_joints=prediction["joint_positions"]["joint_positions"][0].detach().cpu().numpy(),
                pose_trajectory=prediction["gripper_poses"]["traj"][0].detach().cpu().numpy())
            execute(pull,item["gripper_commands"][1],"pull")
        row["final_state"]=snapshot()
        row["final_door_angle_deg"]=float(np.degrees(door.get_joint_position()))
        row["physics_seconds"]=recorder.total_steps*recorder.dt
        row["physics_dt"]=recorder.dt
        row["peak_cuda_allocated_gib"]=float(s.torch.cuda.max_memory_allocated()/2**30)
        s.Image.fromarray(obs.front_rgb).save(artifact/"final.png")
        row["scientific_record_complete"]=True
    except (s.IKError,s.ConfigurationPathError,s.InvalidActionError,s.InvalidTrajectoryError) as exc:
        row["execution_failure"]=f"{type(exc).__name__}: {exc}"
        row["scientific_record_complete"]=True
        if recorder is not None:
            row["final_state"]=snapshot();row["final_door_angle_deg"]=float(np.degrees(door.get_joint_position()))
            row["physics_seconds"]=recorder.total_steps*recorder.dt
            row["physics_dt"]=recorder.dt
    finally:
        row["wall_seconds"]=time.perf_counter()-began
        if row["scientific_record_complete"]:row["worker_status"]="completed_before_shutdown"
        c.atomic(output,row)
        if recorder is not None:recorder.restore()
        env.shutdown()


def run(args):
    protocol=load_json(PROTOCOL)
    n=1 if args.smoke else protocol["scenes"]
    output=Path(args.output) if args.output else RESULTS/("grasp-pose-smoke.json" if args.smoke else "grasp-pose-val.json")
    run_protocol={**protocol,"scenes":n,"expected_cases":n*4,"smoke":args.smoke}
    state={"protocol":run_protocol,"collection_complete":False,"records":[],"infrastructure_errors":[]}
    if output.exists():
        state=load_json(output)
        if state["protocol"]!=run_protocol:raise ValueError("Frozen run protocol changed")
    root=output.parent/(output.stem+"-cases");root.mkdir(exist_ok=True)
    completed={(r["episode"],r["condition"]) for r in state["records"]}
    c.atomic(output,state)
    def execute(key):
        episode,condition=key
        case_root=root/f"episode-{episode:02d}-{condition}"
        case_root.mkdir(exist_ok=True)
        previous=sorted(case_root.glob("attempt-*.json"))
        attempts=[]
        for path in previous:
            row=load_json(path)
            if (row["episode"],row["condition"])!=key:raise ValueError("Wrong saved case identity")
            if row.get("scientific_record_complete"):
                return row
            attempts.append({"path":str(path),"status":row.get("worker_status"),"retained_incomplete_attempt":True})
        # The cap is global across resumes, not two extra attempts per resume.
        for attempt in range(len(previous)+1,3):
            path=case_root/f"attempt-{attempt:02d}.json"
            ensure_new_output(path,check_log=True)
            command=[sys.executable,str(Path(__file__).resolve()),"worker","--episode",str(episode),"--condition",condition,"--output",str(path)]
            with path.with_suffix(".log").open("w") as handle:
                result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,timeout=480)
            if not path.exists():raise RuntimeError(f"No attempt JSON, exit={result.returncode}: {path.with_suffix('.log')}")
            row=load_json(path)
            attempts.append({"path":str(path),"exit_code":result.returncode,"scientific_record_complete":row.get("scientific_record_complete",False)})
            if row.get("scientific_record_complete"):
                row["attempt_file"]=str(path)
                row["attempt_history"]=attempts
                if result.returncode:row["native_exit_after_saved_result"]=result.returncode
                c.atomic(path,row)
                return row
            if result.returncode>=0:
                raise RuntimeError(f"Python/incomplete scientific error, exit={result.returncode}: {path.with_suffix('.log')}")
            # Only an incomplete native infrastructure crash gets one fresh
            # capture attempt, with all original files retained in attempt01.
            if attempt>=2:break
        raise RuntimeError(f"Repeated native failure, preserved attempts: {case_root}")
    keys=[]
    for episode in range(n):
        order=m.CONDITIONS[episode%4:]+m.CONDITIONS[:episode%4]
        keys.extend((episode,condition) for condition in order if (episode,condition) not in completed)
    errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(execute,key):key for key in keys}
        for future in as_completed(futures):
            try:row=future.result()
            except Exception as exc:
                error={"case":futures[future],"error":f"{type(exc).__name__}: {exc}"}
                errors.append(error);state["infrastructure_errors"].append(error)
                c.atomic(output,state);print("Infrastructure",error,flush=True);continue
            key=(row["episode"],row["condition"])
            if key in completed:raise ValueError("Duplicate scientific case")
            completed.add(key);state["records"].append(row)
            state["collection_complete"]=len(completed)==n*4
            c.atomic(output,state)
            e=row.get("preclose_pose_error",{})
            print(f"Grasp pose {len(completed)}/{n*4} ep={row['episode']:02d} {row['condition']}: S={row['task_success']} preclose={e.get('position_mm',float('nan')):.3f}mm/{e.get('rotation_deg',float('nan')):.3f}deg error={row['execution_failure']}",flush=True)
    if errors:raise RuntimeError(f"{len(errors)} incomplete infrastructure cases; completed executions retained")
    print("Complete",output,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","worker","run"])
    parser.add_argument("--episode",type=int)
    parser.add_argument("--condition",choices=m.CONDITIONS)
    parser.add_argument("--output")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if args.mode=="prepare":prepare()
    elif args.mode=="worker":worker(args)
    else:run(args)
