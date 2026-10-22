"""Unchanged physical recorder extracted from earlier experiments."""
import study as s
from pyrep.const import ObjectType
from pyrep.objects.shape import Shape
PHYSICS_BUDGET_SECONDS = 40.
class BudgetExceeded(s.InvalidTrajectoryError):
    pass
def contact_summary(contacts, door_handle):
    relevant = [contact for contact in contacts if door_handle in contact["contact_handles"]]
    forces = [float(s.np.linalg.norm(contact["contact"][3:6])) for contact in relevant]
    return {"points": len(relevant), "positive_force_points": sum(force > 1e-6 for force in forces),
            "sum_force_magnitudes_N": float(sum(forces))}

class Recorder:
    def __init__(self, env, door):
        self.env, self.door = env, door
        self.door_shape = Shape("microwave_door_resp")
        shapes = env._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE)
        by_name = {shape.get_name(): shape for shape in shapes}
        self.fingers = [by_name[name] for name in ("Panda_leftfinger_respondable", "Panda_rightfinger_respondable")]
        self.arm, self.gripper = env._robot.arm, env._robot.gripper
        self.active = False
        self.rows = []
        self.steps = 0
        self.dt = env._pyrep.get_simulation_timestep()
        self.original_step = env._pyrep.step

        def step():
            self.original_step()
            if self.active:
                self.steps += 1
                if self.steps % 2 == 0:
                    self.rows.append(self.snapshot())
        # Captures gripper actuation too: its implementation bypasses Scene.step.
        env._pyrep.step = step

    def snapshot(self):
        contacts = [contact_summary(finger.get_contact(), self.door_shape.get_handle()) for finger in self.fingers]
        return {"time_s": self.steps * self.dt,
            "tip_pose": self.arm.get_tip().get_pose().tolist(),
            "tip_in_door": self.arm.get_tip().get_pose(relative_to=self.door_shape).tolist(),
            "joints": self.arm.get_joint_positions(), "gripper_joints": self.gripper.get_joint_positions(),
            "door_angle_rad": float(self.door.get_joint_position()), "contacts": contacts}

    def start(self):
        self.steps = 0
        self.rows = [self.snapshot()]
        self.active = True

    def finish(self, path):
        self.active = False
        self.rows.append(self.snapshot())
        rows = self.rows
        arrays = {name: s.np.asarray([row[name] for row in rows], dtype=s.np.float32)
                  for name in ("time_s", "tip_pose", "tip_in_door", "joints", "gripper_joints", "door_angle_rad")}
        for side, index in (("left", 0), ("right", 1)):
            for key in ("points", "positive_force_points", "sum_force_magnitudes_N"):
                arrays[f"{side}_{key}"] = s.np.asarray([row["contacts"][index][key] for row in rows])
        path.parent.mkdir(parents=True, exist_ok=True)
        s.np.savez_compressed(path, **arrays)
        bilateral = s.np.asarray([all(contact["points"] > 0 for contact in row["contacts"]) for row in rows])
        relative = arrays["tip_in_door"][:, :3]
        return {"trace_file": str(path), "samples": len(rows), "physics_steps": self.steps,
            "bilateral_finger_door_contact_fraction": float(bilateral.mean()),
            "max_tip_in_door_position_drift_mm": float(s.np.linalg.norm(relative - relative[0], axis=1).max() * 1000),
            "door_progress_deg": float(s.np.degrees(rows[-1]["door_angle_rad"] - rows[0]["door_angle_rad"]))}

    def restore(self):
        self.env._pyrep.step = self.original_step
class BudgetRecorder(Recorder):
    def __init__(self, env, door):
        super().__init__(env, door)
        self.total_steps = 0
        self.enforce_budget = False
        self.max_steps = int(round(PHYSICS_BUDGET_SECONDS / self.dt))
        recorded_step = env._pyrep.step

        def step():
            if self.enforce_budget and self.total_steps >= self.max_steps:
                raise BudgetExceeded("Common 40 s physics budget exhausted")
            recorded_step()
            if self.enforce_budget:
                self.total_steps += 1
        env._pyrep.step = step
