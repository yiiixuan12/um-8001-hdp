"""Trace saved grasp endpoint errors without loading a policy or simulator.

The twenty original samples are the statistical unit. The eighty intervention
records only provide paired diagnostic traces; they are not eighty new scenes.
"""
import argparse
import csv
import json
from project_paths import load_json, portable_record, STATE_ROOT, same_experiment_config
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import geometry as g
from geometry import filter_indices


CONDITIONS = ("ORIGINAL", "HOLD", "XYZ", "SE3")
PLAN_STAGES = ("pose_diffusion", "raw_joint", "refined_joint", "filtered_joint", "timed_command")
CASE_STAGES = ("offline_suffix", "before_extra_second", "last_motor_target", "preclose", "postclose")


def error(actual, reference):
    """Position distance and SO(3) geodesic angle, not quaternion subtraction."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != (4, 4) or reference.shape != (4, 4):
        raise ValueError("Expected two single rigid transforms")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite endpoint")
    return {
        "position_mm": float(np.linalg.norm(actual[:3, 3] - reference[:3, 3]) * 1000),
        "rotation_deg": float(np.degrees(Rotation.from_matrix(actual[:3, :3] @ reference[:3, :3].T).magnitude())),
    }


def collect_groups(data):
    if not data.get("collection_complete"):
        raise ValueError("Incomplete scientific collection")
    n = data["protocol"]["scenes"]
    groups = {co: {} for co in CONDITIONS}
    for row in data["records"]:
        co, ep = row["condition"], row["episode"]
        if co not in groups or ep in groups[co] or not row.get("scientific_record_complete"):
            raise ValueError("Unexpected, duplicate, or incomplete case")
        groups[co][ep] = row
    if any(set(rows) != set(range(n)) for rows in groups.values()):
        raise ValueError("Missing paired case")
    return groups


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k].copy() for k in z.files}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError(f"Nonfinite source {path}")
    return result


def flatten(row, label, actual, reference):
    row.update({f"{label}_{k}": v for k, v in error(actual, reference).items()})


def distribution(values):
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


def summarize(rows, labels):
    return {label: {unit: distribution([r[f"{label}_{unit}"] for r in rows])
                    for unit in ("position_mm", "rotation_deg")} for label in labels}


def run(input_path, output_prefix):
    data = load_json(input_path)
    groups = collect_groups(data)
    n = data["protocol"]["scenes"]
    root = input_path.parent
    calibration_path = Path(data["protocol"]["kinematics_calibration"])
    calibration = load_json(calibration_path)
    paths = [Path(str(output_prefix) + suffix) for suffix in
             (".json", "-audit.json", "-plans.csv", "-cases.csv", ".png", ".txt")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Choose a new prefix; saved analyses are never overwritten")
    plans, cases = [], []
    audit = {"scenes": n, "paired_cases": n * 4, "saved_predictions": 0,
             "source_grasps_exact": 0, "filtered_endpoint_exact": 0,
             "nominal_timed_endpoint_exact": 0, "nominal_timed_endpoint_within_1e_8rad": 0,
             "max_nominal_timed_endpoint_delta_rad": 0., "paired_nominal_commands_exact": 0,
             "controller_before_extra_snapshot_exact": 0, "controller_preclose_snapshot_exact": 0,
             "saved_preclose_errors_recomputed": 0, "saved_postclose_errors_recomputed": 0,
             "original_commands_equal_motor_targets": 0, "all_sources_read_only": True,
             "new_simulator_episodes": 0, "new_policy_samples": 0,
             "calibration": str(calibration_path),
             "fk_note": "Exported simulator chain reused unchanged. Measured-q FK versus actual tip discrepancies are reported separately."}
    for ep in range(n):
        original = groups["ORIGINAL"][ep]
        goal = g.pose_matrix(original["grasp_goal"])
        prediction_path = root / "grasp-source-predictions" / f"episode-{ep:02d}-MM-stage-0-prediction.npz"
        prediction = arrays(prediction_path)
        audit["saved_predictions"] += 1
        source = arrays(original["grasp_plan_file"])
        np.testing.assert_array_equal(source["grasp"], prediction["refined_joints"])
        np.testing.assert_array_equal(source["grasp_goal"], original["grasp_goal"])
        audit["source_grasps_exact"] += 1
        filtered = source["grasp"][filter_indices(source["grasp"])]
        np.testing.assert_array_equal(filtered[-1], prediction["refined_joints"][-1])
        audit["filtered_endpoint_exact"] += 1
        folder = Path(original["grasp_controller_trace"]).parent
        nominal = arrays(folder / "grasp-command-plans.npz")["nominal"]
        # The scalar-time path interpolation can round a float32 endpoint by
        # one ULP. Measure this instead of claiming bitwise identity.
        endpoint_delta = float(np.max(np.abs(nominal[-1].astype(float) - filtered[-1])))
        np.testing.assert_allclose(nominal[-1], filtered[-1], atol=1e-8, rtol=0)
        audit["nominal_timed_endpoint_exact"] += int(np.array_equal(nominal[-1], filtered[-1]))
        audit["nominal_timed_endpoint_within_1e_8rad"] += 1
        audit["max_nominal_timed_endpoint_delta_rad"] = max(audit["max_nominal_timed_endpoint_delta_rad"], endpoint_delta)
        matrices = {
            "pose_diffusion": g.pose_matrix(prediction["pose_trajectory"][-1]),
            "raw_joint": g.forward_kinematics(prediction["raw_joints"][-1:], calibration)[0],
            "refined_joint": g.forward_kinematics(prediction["refined_joints"][-1:], calibration)[0],
            "filtered_joint": g.forward_kinematics(filtered[-1:], calibration)[0],
            "timed_command": g.forward_kinematics(nominal[-1:], calibration)[0],
        }
        plan = {"episode": ep, "original_success": bool(original["task_success"]),
                "prediction_file": str(prediction_path)}
        for label, matrix in matrices.items():
            flatten(plan, label, matrix, goal)
        flatten(plan, "raw_to_refined_change", matrices["refined_joint"], matrices["raw_joint"])
        plans.append(plan)
        for co in CONDITIONS:
            row = groups[co][ep]
            np.testing.assert_array_equal(row["grasp_goal"], original["grasp_goal"])
            ctrl = arrays(row["grasp_controller_trace"])
            command_plans = arrays(Path(row["grasp_controller_trace"]).parent / "grasp-command-plans.npz")
            np.testing.assert_array_equal(command_plans["nominal"], nominal)
            audit["paired_nominal_commands_exact"] += 1
            phase0 = np.flatnonzero(ctrl["phase"] == 0)
            if len(phase0) != row["nominal_grasp_steps"]:
                raise ValueError("Wrong nominal-phase length")
            for saved, index, key in ((row["before_extra_second"], phase0[-1], "controller_before_extra_snapshot_exact"),
                                      (row["preclose_state"], len(ctrl["joints"]) - 1, "controller_preclose_snapshot_exact")):
                np.testing.assert_array_equal(ctrl["tip_pose"][index], saved["tip_pose"])
                np.testing.assert_array_equal(ctrl["joints"][index], saved["joints"])
                audit[key] += 1
            if co == "ORIGINAL":
                np.testing.assert_array_equal(ctrl["command"], nominal[1:].astype(float))
                audit["original_commands_equal_motor_targets"] += 1
            offline = g.forward_kinematics(command_plans["corrected"][-1:], calibration)[0]
            before = g.pose_matrix(row["before_extra_second"]["tip_pose"])
            before_command = g.forward_kinematics(ctrl["command"][phase0[-1]:phase0[-1] + 1], calibration)[0]
            last_command = g.forward_kinematics(ctrl["command"][-1:], calibration)[0]
            actual = g.pose_matrix(row["preclose_state"]["tip_pose"])
            measured_fk = g.forward_kinematics(ctrl["joints"][-1:], calibration)[0]
            post = g.pose_matrix(row["handoff"]["tip_pose"])
            case = {"episode": ep, "condition": co, "success": bool(row["task_success"]),
                    "attempt_file": row["attempt_file"],
                    "preclose_max_joint_velocity_rad_s": float(np.max(np.abs(ctrl["joint_velocities"][-1]))),
                    "preclose_max_joint_command_tracking_rad": float(np.max(np.abs(ctrl["joints"][-1] - ctrl["command"][-1])))}
            transforms = {"offline_suffix": offline, "before_extra_second": before,
                          "last_motor_target": last_command, "preclose": actual, "postclose": post}
            for label, matrix in transforms.items():
                flatten(case, label, matrix, goal)
            for label, first, second in (
                ("before_extra_vs_command", before, before_command),
                ("preclose_vs_command", actual, last_command),
                ("measured_q_vs_command", measured_fk, last_command),
                ("fk_vs_actual", measured_fk, actual),
                ("closure_change", post, actual),
            ):
                flatten(case, label, first, second)
            for label, saved in (("preclose", "preclose_pose_error"), ("postclose", "postclose_pose_error")):
                for unit in ("position_mm", "rotation_deg"):
                    np.testing.assert_allclose(case[f"{label}_{unit}"], row[saved][unit], atol=1e-8, rtol=1e-7)
                audit[f"saved_{label}_errors_recomputed"] += 1
            # Distances do not add. Verify the triangle bound for each endpoint.
            for unit in ("position_mm", "rotation_deg"):
                difference = abs(case[f"preclose_{unit}"] - case[f"last_motor_target_{unit}"])
                if difference > case[f"preclose_vs_command_{unit}"] + 1e-8:
                    raise ValueError("Invalid frame or endpoint distance")
            cases.append(case)
    comparisons = ("before_extra_vs_command", "preclose_vs_command", "measured_q_vs_command", "fk_vs_actual", "closure_change")
    condition_summaries = {}
    for co in CONDITIONS:
        subset = [r for r in cases if r["condition"] == co]
        condition_summaries[co] = {
            "successes": sum(r["success"] for r in subset),
            "stages_and_discrepancies": summarize(subset, CASE_STAGES + comparisons),
        }
    original_cases = [r for r in cases if r["condition"] == "ORIGINAL"]
    enriched = [{**p, **{k: v for k, v in r.items() if k not in p}} for p, r in zip(plans, original_cases)]
    rotation_increased = [r["episode"] for r in plans if r["refined_joint_rotation_deg"] > r["raw_joint_rotation_deg"] + 1e-8]
    summary = {
        "purpose": "Retrospective saved-trajectory grasp endpoint error origin audit",
        "source": str(input_path), "scenes": n, "cases": len(cases),
        "planning": summarize(plans, PLAN_STAGES + ("raw_to_refined_change",)),
        "conditions": condition_summaries,
        "original_diagnostics": {
            "planned_rotation_above_1deg": sum(r["refined_joint_rotation_deg"] > 1 for r in plans),
            "planned_rotation_above_qualification_0_2deg": sum(r["refined_joint_rotation_deg"] > .2 for r in plans),
            "planned_position_below_0_001mm": sum(r["refined_joint_position_mm"] < .001 for r in plans),
            "actual_rotation_above_1deg": sum(r["preclose_rotation_deg"] > 1 for r in original_cases),
            "actual_vs_command_rotation_below_0_5deg": sum(r["preclose_vs_command_rotation_deg"] < .5 for r in original_cases),
            "rotation_error_increased_by_refinement_episodes": rotation_increased,
            "rotation_error_decreased_by_refinement_episodes": [r["episode"] for r in plans if r["refined_joint_rotation_deg"] < r["raw_joint_rotation_deg"] - 1e-8],
            "original_success_and_failure_descriptive": {
                str(success): {"n": sum(r["original_success"] == success for r in enriched),
                               "metrics": summarize([r for r in enriched if r["original_success"] == success],
                                                    ("raw_joint", "refined_joint", "preclose", "preclose_vs_command"))}
                for success in (False, True)},
        },
        "scope": [
            "One locally trained frozen RK-Diffuser, one task, inspected validation scenes, expert goals; not the authors' checkpoint or full HDP reproduction.",
            "Known simulator chain reused without fitting. Planning error and actual-command discrepancy are different metrics.",
            "Pose error magnitudes are not additive; no percentage attribution from ratios or subtraction of norms.",
            "The official refinement loss is translation only. Association with this implementation is not a causal test of all failure mechanisms.",
            "XYZ/SE3 interventions combine offline suffix projection and measured-pose servo; saved intermediate endpoints are not an offline-only rollout ablation.",
            "No new policy inference, simulation, training, outcome exclusion, or changes to saved sources.",
        ],
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in ((paths[0], summary), (paths[1], audit)):
        path.write_text(json.dumps(portable_record(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    for path, rows in ((paths[2], plans), (paths[3], cases)):
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows([portable_record(r) for r in rows])
    plot(paths[4], plans, original_cases)
    paths[5].write_text(report(summary, audit), encoding="utf-8")
    print(report(summary, audit))
    return summary, audit


def plot(path, plans, originals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ("Pose branch", "Raw joint FK", "Refined joint FK", "Timed command FK", "Actual pre-close")
    keys = ("pose_diffusion", "raw_joint", "refined_joint", "timed_command", "preclose")
    combined = [{**p, **r} for p, r in zip(plans, originals)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, unit, title in zip(axes, ("position_mm", "rotation_deg"), ("Position error (mm)", "Orientation error (deg)")):
        values = np.array([[r[f"{key}_{unit}"] for key in keys] for r in combined])
        ax.plot(np.arange(5), values.T, color="#aab8c5", linewidth=.8, alpha=.5)
        means = values.mean(axis=0)
        ax.plot(np.arange(5), means, "o-", color="#126b72", linewidth=2.3, label="Mean; thin lines = scenes")
        for index, value in enumerate(means):
            ax.annotate(f"{value:.4f}" if value < .001 else f"{value:.3f}", (index, value),
                        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        ax.set(xticks=np.arange(5), xticklabels=labels, ylabel=title, ylim=(-.05 * values.max(), 1.15 * values.max()))
        ax.tick_params(axis="x", labelrotation=23)
        ax.grid(axis="y", alpha=.2)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Saved grasp endpoints | 20 original samples, expert goals, no new simulation")
    fig.savefig(path, dpi=170); plt.close(fig)


def report(s, audit):
    def mean(section, label, unit):
        return section[label][unit]["mean"]
    planning = s["planning"]
    lines = ["抓取误差来源：已保存轨迹离线核对", "日期：2026-10-21", "",
        "结论：约3度朝向误差在运动学精炼后的规划终点已经存在；执行不是其主要新增来源。位置则同时有生成误差与执行后的跟踪/模型残差。",
        "范围：20个已分析val场景、一个本地简化训练检查点、专家目标；核对80份干预记录，未新增仿真或模型采样。", "",
        "一、原始抓取误差链（20场景均值；位置mm / 朝向度）"]
    labels = {"pose_diffusion": "位姿扩散终点（inpainting）", "raw_joint": "原始关节预测的FK终点",
              "refined_joint": "运动学精炼后的FK终点", "filtered_joint": "轨迹过滤后的FK终点",
              "timed_command": "时间参数化后最终指令的FK终点"}
    for label in PLAN_STAGES:
        lines.append(f"{labels[label]}：{mean(planning,label,'position_mm'):.6f} / {mean(planning,label,'rotation_deg'):.6f}")
    original = s["conditions"]["ORIGINAL"]["stages_and_discrepancies"]
    for label, name in (("preclose", "实际闭合前"), ("postclose", "实际闭合后"),
                        ("preclose_vs_command", "实际闭合前与最终指令FK的差"),
                        ("measured_q_vs_command", "实测关节FK与最终指令FK的差"),
                        ("fk_vs_actual", "实测关节FK与实测末端的差")):
        lines.append(f"{name}：{mean(original,label,'position_mm'):.6f} / {mean(original,label,'rotation_deg'):.6f}")
    d = s["original_diagnostics"]
    lines += ["", "二、逐场景检查", f"规划位置误差<0.001mm：{d['planned_position_below_0_001mm']}/{s['scenes']}",
        f"规划朝向误差>1度：{d['planned_rotation_above_1deg']}/{s['scenes']}",
        f"实际朝向误差>1度：{d['actual_rotation_above_1deg']}/{s['scenes']}",
        f"实际与最终指令朝向差<0.5度：{d['actual_vs_command_rotation_below_0_5deg']}/{s['scenes']}",
        f"精炼使目标朝向误差增加的场景：{d['rotation_error_increased_by_refinement_episodes']}",
        f"精炼使目标朝向误差减少的场景：{d['rotation_error_decreased_by_refinement_episodes']}",
        "这些变化不是失败原因的单因素验证。误差角大小不可直接相加，不用均值比例分配责任。", "",
        "三、四组保存记录的中间状态（位置mm / 朝向度均值）"]
    for co in CONDITIONS:
        section = s["conditions"][co]
        lines.append(f"{co}，成功{section['successes']}/{s['scenes']}：")
        for label, name in (("offline_suffix", "离线末段规划终点"), ("before_extra_second", "名义轨迹执行完、额外1秒之前"),
                            ("last_motor_target", "实际闭合前最终电机目标的FK"), ("preclose", "实际闭合前")):
            a = section["stages_and_discrepancies"]
            lines.append(f"  {name}：{mean(a,label,'position_mm'):.6f} / {mean(a,label,'rotation_deg'):.6f}")
    lines += ["这些中间状态有助定位跟踪残差，但不是离线修正/在线伺服独立消融；后续成功率不能分配给某一个模块。", "",
        "四、原方法具体暴露的薄弱环节",
        "1. 位姿分支的硬目标约束没有完整传递到执行的关节终点：原始关节预测有误差，现有精炼把位置修到几乎精确，却留下约3度朝向偏差。",
        "2. 官方robot.py计算旋转误差，但loss=t_err，推理精炼只优化平移；multi_level_diffusion.py同时做全轨迹和额外终点精炼。作者已有优化，问题是当前优化没有显式处理旋转残差。",
        "3. 过滤保持20/20终点关节值；时间参数化只有可忽略的浮点舍入，完整记录最大关节差。这些处理不是这批约3度终点朝向偏差的来源；本检查不评价路径中间点的影响。",
        "4. 实际执行还产生毫米级位置偏差。实测关节跟踪偏差与FK/实测末端差异单列，不能将其全部称作电机误差。",
        "5. 当前完整SE3修正19/20表明干预有效，但仍不能证明残余旋转是所有失败的唯一原因，也不能宣称作者原检查点有相同成功率或同等误差。", "",
        "五、验证与保留范围",
        f"源抓取数组与原精炼预测逐项一致：{audit['source_grasps_exact']}/20；过滤终点逐项保持：{audit['filtered_endpoint_exact']}/20；时间参数化终点逐项相同：{audit['nominal_timed_endpoint_exact']}/20、1e-8rad内相同：{audit['nominal_timed_endpoint_within_1e_8rad']}/20，最大关节差{audit['max_nominal_timed_endpoint_delta_rad']:.12g}rad。",
        f"配对名义指令保持：{audit['paired_nominal_commands_exact']}/80；控制记录与额外等待前/闭合前快照一致：{audit['controller_before_extra_snapshot_exact']}/80、{audit['controller_preclose_snapshot_exact']}/80；原组实际发送指令未改变：{audit['original_commands_equal_motor_targets']}/20。",
        f"闭合前/后已存误差重新计算一致：{audit['saved_preclose_errors_recomputed']}/80、{audit['saved_postclose_errors_recomputed']}/80。",
        "使用SO(3)测地角，兼容四元数正负等价；复用已验证FK，不重新拟合几何。新增定向检查覆盖误差分离、坐标/四元数不变性、拒绝不完整或重复记录。",
        "没有重跑仿真、训练、全量测试、构建、lint/E2E、独立审查或哈希；既有FK/控制器验证复用。原始文件只读，新结果采用新前缀，拒绝覆盖。"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=g.RESULTS / "grasp-pose-val.json")
    parser.add_argument("--output-prefix", type=Path, default=g.RESULTS / "grasp-error-origin-val")
    args = parser.parse_args()
    run(args.input, args.output_prefix)
