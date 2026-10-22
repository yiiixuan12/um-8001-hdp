"""Check grasp intervention validity before interpreting downstream success."""
import argparse
import csv
import json
from project_paths import load_json, portable_record, STATE_ROOT, same_experiment_config
from pathlib import Path
import numpy as np
import experiment_common as c
import geometry as g
import grasp_pose_suffix as m
from experiment_common import pair
from experiment_common import load_arrays,equal


def run(path,plots=False):
    source=load_json(path)
    protocol=source["protocol"];n=protocol["scenes"]
    if not source["collection_complete"] or len(source["records"])!=n*4:raise ValueError("Incomplete experiment")
    groups={condition:{} for condition in m.CONDITIONS}
    audit={"cases":n*4,"physical_trace_files":0,"controller_files":0,"all_records_complete":True,
        "common_prefix_state_groups":n,"identical_common_prefix_state_groups":0,
        "identical_common_prefix_physical_groups":0,"identical_common_prefix_contact_trace_groups":0,"original_hold_identical_pre_hold_groups":0,
        "max_common_prefix_state_difference":{k:0. for k in c.BOUNDS},
        "velocity_bound_pass":True,"max_command_velocity_ratio":0.,"suffix_prefix_unchanged_plans":0,
        "matched_expert_goal_groups":0,"same_complete_nominal_grasp_input_groups":0,"fixed_subsequent_policy_groups":0,
        "new_pull_diffusion_calls":0,"native_exits_after_complete_record":[],"incomplete_native_attempts":[]}
    controllers={};plans_by_case={};traces={};table=[]
    for row in source["records"]:
        ep,condition=row["episode"],row["condition"]
        if ep in groups[condition] or not row["scientific_record_complete"]:raise ValueError("Duplicate/incomplete record")
        groups[condition][ep]=row
        if row.get("native_exit_after_saved_result"):
            audit["native_exits_after_complete_record"].append({"episode":ep,"condition":condition,"code":row["native_exit_after_saved_result"]})
        for attempt in row.get("attempt_history",[]):
            if not attempt.get("scientific_record_complete",False):audit["incomplete_native_attempts"].append({"episode":ep,"condition":condition,**attempt})
        audit["new_pull_diffusion_calls"]+=row["new_diffusion_calls"]
        record={"episode":ep,"condition":condition,"success":row["task_success"],"execution_failure":row["execution_failure"],
            "pull_timeout":row["pull_timeout"],"final_door_angle_deg":row.get("final_door_angle_deg"),
            "physics_seconds":row.get("physics_seconds"),"wall_seconds":row["wall_seconds"],
            "preclose_position_mm":row.get("preclose_pose_error",{}).get("position_mm"),
            "preclose_rotation_deg":row.get("preclose_pose_error",{}).get("rotation_deg"),
            "preclose_full_pose_qualified":row.get("preclose_pose_error",{}).get("full_pose_qualified",False),
            "preclose_max_joint_velocity_rad_s":row.get("preclose_pose_error",{}).get("max_joint_velocity_rad_s"),
            "postclose_position_mm":row.get("postclose_pose_error",{}).get("position_mm"),
            "postclose_rotation_deg":row.get("postclose_pose_error",{}).get("rotation_deg"),
            "postclose_full_pose_qualified":row.get("postclose_pose_error",{}).get("full_pose_qualified",False),
            "postclose_max_joint_velocity_rad_s":row.get("postclose_pose_error",{}).get("max_joint_velocity_rad_s")}
        for entry in row["actions"]:
            arrays=load_arrays(entry["trace"]["trace_file"])
            if len(arrays["time_s"])!=entry["trace"]["samples"] or np.any(np.diff(arrays["time_s"])<0):raise ValueError("Invalid trace length/time")
            audit["physical_trace_files"]+=1
            traces[(ep,condition,entry["name"])]=arrays
            if entry["name"]=="grasp":record["grasp_total_seconds"]=entry["trace"]["physics_steps"]*row["physics_dt"]
            if entry["name"]=="pull":
                keep=arrays["time_s"]<=1.00001
                relative=arrays["tip_in_door"][keep,:3]
                record.update(pull_physics_seconds=entry["trace"]["physics_steps"]*row["physics_dt"],
                    pull_contact_fraction=entry["trace"]["bilateral_finger_door_contact_fraction"],
                    first_second_relative_drift_mm=float(np.max(np.linalg.norm(relative-relative[0],axis=1))*1000),
                    first_second_door_progress_deg=float(np.degrees(arrays["door_angle_rad"][keep][-1]-arrays["door_angle_rad"][0])))
        if row.get("grasp_controller_trace"):
            ctrl=load_arrays(row["grasp_controller_trace"])
            controllers[(ep,condition)]=ctrl;audit["controller_files"]+=1
            if len(ctrl["time_s"])!=row["grasp_controller_steps"]:raise ValueError("Controller samples")
            previous=np.vstack([ctrl["initial_command"],ctrl["command"][:-1]])
            ratio=np.max(np.abs(ctrl["command"]-previous)/row["physics_dt"]/ctrl["joint_velocity_limits"])
            if ratio>1.000001:raise ValueError("Command velocity exceeded")
            audit["max_command_velocity_ratio"]=max(audit["max_command_velocity_ratio"],float(ratio))
            folder=Path(row["grasp_controller_trace"]).parent
            plans=load_arrays(folder/"grasp-command-plans.npz")
            plans_by_case[(ep,condition)]=plans
            boundary=row["suffix_solver"]["boundary"]
            if not np.array_equal(plans["nominal"][:boundary+1],plans["corrected"][:boundary+1]):raise ValueError("Modified pre-intervention command")
            audit["suffix_prefix_unchanged_plans"]+=1
            expected_extra=0 if condition=="ORIGINAL" else round(m.CONFIG["preclose_seconds"]/row["physics_dt"])
            if int(np.sum(ctrl["phase"]==1))!=expected_extra:raise ValueError("Wrong fixed preclose time")
            record["velocity_limited_steps"]=int(ctrl["velocity_limited"].sum())
            record["grasp_physics_seconds"]=(len(ctrl["time_s"]))*row["physics_dt"]
            record["gripper_closure_seconds"]=record["grasp_total_seconds"]-record["grasp_physics_seconds"]
            if row["new_diffusion_calls"]:
                inputs=load_arrays(folder/"pull-input.npz")
                np.testing.assert_array_equal(inputs["gripper_poses"][-1],np.asarray(row["pull_goal"],np.float32))
                np.testing.assert_allclose(inputs["joint_positions"][0],row["handoff"]["joints"],atol=1e-7,rtol=0)
                np.testing.assert_allclose(inputs["gripper_poses"][0],row["handoff"]["tip_pose"],atol=1e-7,rtol=0)
        table.append(record)
    if any(set(group)!=set(range(n)) for group in groups.values()):raise ValueError("Unpaired scenes")
    for ep in range(n):
        rows=[groups[co][ep] for co in m.CONDITIONS]
        if not all(np.array_equal(rows[0]["grasp_goal"],r["grasp_goal"]) and np.array_equal(rows[0]["pull_goal"],r["pull_goal"]) and rows[0]["gripper_commands"]==r["gripper_commands"] for r in rows[1:]):raise ValueError("Expert target/closure differs")
        audit["matched_expert_goal_groups"]+=1
        reference=plans_by_case[(ep,"ORIGINAL")]["nominal"]
        if not all(np.array_equal(reference,plans_by_case[(ep,co)]["nominal"]) for co in m.CONDITIONS):raise ValueError("Frozen complete grasp input differs")
        audit["same_complete_nominal_grasp_input_groups"]+=1
        if not all(r["checkpoint"]==protocol["checkpoint"] and r["diffusion_seed"]==80000+ep*100+1 for r in rows):raise ValueError("Subsequent policy changed")
        audit["fixed_subsequent_policy_groups"]+=1
        if any("common_prefix_checkpoint" not in r for r in rows):raise ValueError("Missing prefix validity check")
        for row in rows[1:]:
            delta=c.comparison(row["common_prefix_checkpoint"],rows[0]["common_prefix_checkpoint"])
            for key,value in delta.items():audit["max_common_prefix_state_difference"][key]=max(audit["max_common_prefix_state_difference"][key],value)
            if any(delta[k]>bound for k,bound in c.BOUNDS.items()):raise ValueError("Physical prefix mismatch")
        audit["identical_common_prefix_state_groups"]+=int(all(all(v==0 for v in c.comparison(r["common_prefix_checkpoint"],rows[0]["common_prefix_checkpoint"]).values()) for r in rows[1:]))
        count=rows[0]["common_prefix_steps"]
        cs=[controllers[(ep,co)] for co in m.CONDITIONS]
        keys=("time_s","phase","reference_index","command","joints","tip_pose","joint_velocities")
        same_prefix=all(all(np.array_equal(cs[0][k][:count],ctrl[k][:count]) for k in keys) for ctrl in cs[1:])
        if not same_prefix:raise ValueError("Actual prefix trajectory differs")
        audit["identical_common_prefix_physical_groups"]+=1
        cutoff=count*rows[0]["physics_dt"]
        base=traces[(ep,"ORIGINAL","grasp")]
        base_keep=base["time_s"]<=cutoff+1e-6
        for condition in m.CONDITIONS[1:]:
            other=traces[(ep,condition,"grasp")]
            keep=other["time_s"]<=cutoff+1e-6
            if not all(np.array_equal(base[k][base_keep],other[k][keep]) for k in base):raise ValueError("Pre-intervention contact/door/gripper trace differs")
        audit["identical_common_prefix_contact_trace_groups"]+=1
        natural=rows[0]["nominal_grasp_steps"]
        same_original_hold=all(np.array_equal(cs[0][k][:natural],cs[1][k][:natural]) for k in keys)
        if not same_original_hold:raise ValueError("HOLD altered trajectory before extra second")
        audit["original_hold_identical_pre_hold_groups"]+=1
    preclose_position_match=[np.linalg.norm(np.asarray(groups["SE3"][ep]["preclose_state"]["tip_pose"][:3])-groups["XYZ"][ep]["preclose_state"]["tip_pose"][:3])*1000 for ep in range(n)]
    summary={"source":str(path),"role":protocol["role"],"scenes":n,"conditions":{},"paired_comparisons":{},"audit":audit,
        "SE3_XYZ_preclose_actual_position_difference_mm":{"mean":float(np.mean(preclose_position_match)),"max":float(np.max(preclose_position_match)),"per_episode":preclose_position_match}}
    numeric=["preclose_position_mm","preclose_rotation_deg","preclose_max_joint_velocity_rad_s",
        "postclose_position_mm","postclose_rotation_deg","postclose_max_joint_velocity_rad_s",
        "grasp_physics_seconds","gripper_closure_seconds","grasp_total_seconds","pull_physics_seconds","physics_seconds","pull_contact_fraction",
        "first_second_relative_drift_mm","first_second_door_progress_deg","velocity_limited_steps"]
    for condition in m.CONDITIONS:
        selected=[r for r in table if r["condition"]==condition]
        def mean(key):
            values=[r[key] for r in selected if r.get(key) is not None]
            return float(np.mean(values)) if values else None
        summary["conditions"][condition]={"successes":sum(r["success"] for r in selected),"n":n,
            "execution_failures":sum(r["execution_failure"] is not None for r in selected),
            "pull_timeouts":sum(r["pull_timeout"] for r in selected),
            "preclose_full_pose_qualified":sum(r["preclose_full_pose_qualified"] for r in selected),
            "postclose_full_pose_qualified":sum(r["postclose_full_pose_qualified"] for r in selected),
            **{key+"_mean":mean(key) for key in numeric}}
    for first,second in protocol["main_comparisons"]+protocol["secondary_comparisons"]:
        summary["paired_comparisons"][f"{first}_minus_{second}"]=pair(groups,first,second)
    main=[summary["paired_comparisons"][f"{a}_minus_{b}"] for a,b in protocol["main_comparisons"]]
    prev=0.
    for rank,index in enumerate(np.argsort([r["two_sided_exact_p"] for r in main])):
        prev=max(prev,min(1.,(len(main)-rank)*main[index]["two_sided_exact_p"]))
        main[index]["holm_two_primary_comparisons_p"]=prev
    summary["limitations"]=["One task/seed and previously inspected val20; expert goals, no PerAct",
        "SE3 changes suffix posture/contact/history; not pure rotation isolation",
        "Same downstream policy/seed but actual observations and resulting trajectories vary, as intended",
        "Correction failures remain in denominator; pose validity must precede success interpretation",
        "Small-sample percentile bootstrap can be degenerate; zero observed effect is not equivalence"]
    c.atomic(path.with_name(path.stem+"-summary.json"),summary)
    c.atomic(path.with_name(path.stem+"-audit.json"),audit)
    with path.with_name(path.stem+"-episodes.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        fields=list(dict.fromkeys(k for r in table for k in r));writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(table)
    if plots:plot(path,summary,table)
    print(json.dumps(summary,indent=2))
    return summary,table


def plot(path,summary,table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels=["Original","Wait 1s","XYZ + 1s","SE(3) + 1s"]
    colors=["#7a8794","#b7bdc5","#d6a049","#3b8069"]
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout="constrained")
    for ax,key,title,ylabel in ((axes[0],"successes","Downstream task success","Successful episodes"),
        (axes[1],"preclose_position_mm_mean","Measured position BEFORE closure","Position error (mm)"),
        (axes[2],"preclose_rotation_deg_mean","Measured rotation BEFORE closure","Orientation error (deg)")):
        y=[summary["conditions"][co][key] for co in m.CONDITIONS]
        bars=ax.bar(range(4),y,color=colors)
        for bar,value in zip(bars,y):
            text=f"{int(value)}/{summary['scenes']}" if key=="successes" else f"{value:.3f}"
            ax.annotate(text,(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),textcoords="offset points",ha="center",fontsize=9)
        ax.set(xticks=range(4),xticklabels=labels,title=title,ylabel=ylabel)
        ax.tick_params(axis="x",labelrotation=15)
        if key=="successes":ax.set_ylim(0,summary["scenes"]+2)
        else:ax.set_ylim(0,max(y)*1.2+.01)
    fig.suptitle("Grasp suffix pose correction | frozen RK-Diffuser, expert goals, inspected val20",fontsize=12)
    fig.savefig(path.with_name(path.stem+"-results.png"),dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,8),layout="constrained")
    data=np.array([[next(r for r in table if r["episode"]==ep and r["condition"]==co)["success"] for co in m.CONDITIONS] for ep in range(summary["scenes"])])
    ax.imshow(data,cmap="RdYlGn",vmin=0,vmax=1,aspect="auto")
    for (i,j),v in np.ndenumerate(data):ax.text(j,i,"S" if v else "F",ha="center",va="center",fontsize=9)
    ax.set(xticks=range(4),xticklabels=labels,yticks=range(len(data)),ylabel="Episode",title="All paired outcomes; no case excluded")
    fig.savefig(path.with_name(path.stem+"-episodes.png"),dpi=160);plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=g.RESULTS/"grasp-pose-val.json")
    parser.add_argument("--plot",action="store_true")
    args=parser.parse_args();run(args.input,args.plot)
