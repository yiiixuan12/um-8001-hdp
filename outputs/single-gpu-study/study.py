"""Single-GPU RK-Diffuser preparation, local training and offline goal diagnosis.

The upstream network, loss and trajectory resampling are reused. This runner
uses a bounded episode cache and local checkpoints instead of the upstream
multiprocess, all-in-memory dataset and online logger. Offline FK endpoint
errors are deliberately not described as execution or task-success metrics.
"""

import argparse
from collections import OrderedDict
import csv
import json
from project_paths import UPSTREAM_ROOT, STATE_ROOT, portable_record, same_experiment_config
import logging
import os
from pathlib import Path
import pickle
import random
import statistics
import sys
import time

from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
CFG = OmegaConf.load(HERE / "experiment.yaml")
CFG.hdp_root = str(UPSTREAM_ROOT)
CFG.state_root = str(STATE_ROOT)
if (CFG.data.camera != "front" or CFG.data.model_image_size != 64 or CFG.data.renderer != "opengl"
        or CFG.training.precision != "float32" or list(CFG.diagnostic.rotation_deg) != [0]
        or CFG.diagnostic.perturbation_frame != "world" or CFG.diagnostic.re_quantize):
    raise ValueError("This runner supports front/64px, OPENGL, float32 and world-frame translation-only perturbations")
sys.path.insert(0, str(CFG.hdp_root))

# Import RLBench before any module that imports OpenCV.
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from pyrep.objects.joint import Joint
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2

from rk_diffuser.dataset.rl_bench_dataset import RLBenchDataset, JointsTrajectoryActionMode, _create_obs_config
from rk_diffuser.utils import batch_to_device, keypoint_discovery, proc_quaternion
from train_low_level import _create_agent_fn

logging.getLogger("pytorch_kinematics").setLevel(logging.ERROR)


class InvalidTrajectoryError(ValueError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(portable_record(value), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_root(task, pilot):
    return Path(CFG.state_root) / ("pilot-data" if pilot else "data") / task


def collect(args):
    """Collect disjoint seed ranges; keep completed episodes when resuming."""
    task_name = args.task or CFG.task
    root = data_root(task_name, args.pilot)
    config = _create_obs_config([CFG.data.camera], [CFG.data.image_size] * 2)
    env = Environment(MoveArmThenGripper(JointPosition(), Discrete()),
                      obs_config=config, headless=True)
    env.launch()
    try:
        task = env.get_task(task_file_to_task_class(task_name))
        task.set_variation(int(CFG.variation))
        door = Joint("microwave_door_joint") if task_name == "open_microwave" else None
        for split in args.splits:
            count = int(CFG.data.splits[split].episodes)
            if args.pilot:
                count = {"train": 2, "val": 1, "test": 1}[split]
            split_root = root / split
            split_root.mkdir(parents=True, exist_ok=True)
            manifest_path = split_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
                "schema_version": 1, "task": task_name, "split": split,
                "pilot_only": bool(args.pilot), "variation": int(CFG.variation),
                "source": "RLBench live expert demonstrations; no learned policy",
                "stopping_delta": float(CFG.data.stopping_delta),
                "camera": str(CFG.data.camera), "render_mode": "OPENGL",
                "image_size": int(CFG.data.image_size), "model_image_size": int(CFG.data.model_image_size),
                "episodes": [],
            }
            complete = {item["episode"] for item in manifest["episodes"]}
            if (manifest["stopping_delta"] != float(CFG.data.stopping_delta)
                    or manifest["image_size"] != int(CFG.data.image_size)
                    or manifest["model_image_size"] != int(CFG.data.model_image_size)):
                raise ValueError("Existing data collection configuration differs; use a separate data root")
            for index in range(count):
                if index in complete:
                    episode_dir = split_root / f"episode_{index:06d}"
                    if not all((episode_dir / name).is_file() for name in ("arrays.npz", "demo.pkl", "preview.png")):
                        raise RuntimeError(f"Manifest references missing files: {episode_dir}")
                    continue
                seed = int(CFG.data.splits[split].seed_base) + index
                if args.pilot:
                    seed += 1000000
                seed_all(seed)
                started = time.perf_counter()
                def add_diagnostics(obs):
                    if door is not None:
                        obs.misc["study_door_angle_rad"] = float(door.get_joint_position())
                demo = task.get_demos(1, live_demos=True, max_attempts=3,
                                      callable_each_step=add_diagnostics)[0]
                if not task._task.success()[0]:
                    raise RuntimeError("Expert demo did not satisfy RLBench task success")
                frames = [0] + keypoint_discovery(demo, stopping_delta=float(CFG.data.stopping_delta))
                episode_dir = split_root / f"episode_{index:06d}"
                episode_dir.mkdir(exist_ok=True)
                if (episode_dir / "arrays.npz").exists() or (episode_dir / "demo.pkl").exists():
                    raise RuntimeError(f"Uncommitted episode files at {episode_dir}; inspect before resuming")
                Image.fromarray(demo[0].front_rgb).save(episode_dir / "preview.png")
                arrays = {
                    "joint_positions": np.stack([o.joint_positions for o in demo]).astype(np.float32),
                    "gripper_poses": proc_quaternion(np.stack([o.gripper_pose for o in demo]).astype(np.float32)),
                    "proprios": np.stack([o.get_low_dim_data() for o in demo]).astype(np.float32),
                    "rgbs": np.stack([cv2.resize(o.front_rgb, (64, 64)) for o in demo]).astype(np.uint8),
                    "pcds": np.stack([cv2.resize(o.front_point_cloud, (64, 64), interpolation=cv2.INTER_NEAREST) for o in demo]).astype(np.float32),
                    "door_angles": np.array([o.misc.get("study_door_angle_rad", np.nan) for o in demo], dtype=np.float32),
                }
                if arrays["proprios"].shape[1] != 24:
                    raise RuntimeError(f"Expected 24 proprio values, got {arrays['proprios'].shape}")
                for name in ("joint_positions", "gripper_poses", "proprios", "pcds"):
                    if not np.isfinite(arrays[name]).all():
                        raise RuntimeError(f"Nonfinite {name} in expert demo")
                np.savez_compressed(episode_dir / "arrays.npz", **arrays)
                # Preserve the low-dimensional expert prefix and exact reset RNG state.
                # Images/point clouds are in arrays.npz, avoiding duplicate storage.
                for obs in demo:
                    obs.front_rgb = obs.front_depth = obs.front_point_cloud = obs.front_mask = None
                with (episode_dir / "demo.pkl").open("wb") as stream:
                    pickle.dump(demo, stream, protocol=pickle.HIGHEST_PROTOCOL)
                record = {"episode": index, "requested_seed": seed,
                          "directory": episode_dir.name, "frames": len(demo),
                          "keyframes": frames, "expert_success": True,
                          "generation_seconds": time.perf_counter() - started}
                manifest["episodes"].append(record)
                atomic_json(manifest_path, manifest)
                print(f"Collected {task_name}/{split}/{index}: {len(demo)} frames, {len(frames)-1} segments, {record['generation_seconds']:.1f}s", flush=True)
    finally:
        env.shutdown()


class Column:
    def __init__(self, dataset, name):
        self.dataset, self.name = dataset, name

    def __getitem__(self, index):
        episode, start, end = self.dataset.segments[index]
        # Match the original training loader's [start:end] slice exactly.
        result = self.dataset.episode(episode)[self.name][start:end]
        return result.astype(np.float32, copy=False)


class EpisodeDataset(RLBenchDataset):
    """Reuse upstream __getitem__; bound each split's episode cache to 1.5 GiB."""
    def __init__(self, root, training, augment=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (self.manifest["stopping_delta"] != float(CFG.data.stopping_delta)
                or self.manifest["image_size"] != int(CFG.data.image_size)):
            raise ValueError("Dataset keyframe/image configuration mismatch")
        self.segments = []
        for episode, record in enumerate(self.manifest["episodes"]):
            self.segments.extend((episode, a, b) for a, b in zip(record["keyframes"][:-1], record["keyframes"][1:]) if b > a)
        if not self.segments:
            raise RuntimeError(f"No segments in {root}")
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit_bytes = 1536 * 2**20
        self._traj_len = int(CFG.training.horizon)
        self._frame_skips = 1
        self._rank_bins = 10
        self._diffusion_var = "multi"
        self._training = training
        self._demo_aug_ratio = float(CFG.training.demo_aug_ratio) if training and augment else 0
        self._demo_aug_min_len = int(CFG.training.demo_aug_min_len)
        self._dataset = {name: Column(self, name) for name in ("pcds", "rgbs", "gripper_poses", "proprios", "joint_positions")}

    def __len__(self):
        return len(self.segments)

    def episode(self, index):
        if index not in self._cache:
            record = self.manifest["episodes"][index]
            with np.load(self.root / record["directory"] / "arrays.npz", allow_pickle=False) as source:
                self._cache[index] = {key: source[key] for key in source.files}
            self._cache_bytes += sum(a.nbytes for a in self._cache[index].values())
            while len(self._cache) > 1 and self._cache_bytes > self._cache_limit_bytes:
                _, removed = self._cache.popitem(last=False)
                self._cache_bytes -= sum(a.nbytes for a in removed.values())
        self._cache.move_to_end(index)
        return self._cache[index]


def inference_kinematics(args):
    """Record the complete geometry so results cannot qualify a different FK."""
    path = getattr(args, "kinematics_calibration", None)
    if path is None:
        return {"mode": "upstream"}
    from simulator_kinematics import load_calibration
    calibration = load_calibration(path)
    return {"mode": "simulator_geometry", "calibration": calibration}


def create_model(kinematics=None):
    config_root = Path(CFG.hdp_root) / "cfgs"
    cfg = OmegaConf.merge(OmegaConf.load(config_root / "diffuser_config.yaml"),
                         {"env": OmegaConf.load(config_root / "env/sim.yaml"),
                          "method": OmegaConf.load(config_root / "method/rk_diffuser.yaml")})
    cfg.method.horizon = int(CFG.training.horizon)
    cfg.method.n_timesteps = int(CFG.training.diffusion_timesteps)
    robot, model = _create_agent_fn(cfg.method, "cuda", diffusion_var="multi", sim=True,
                                     diff_optim=True, diff_optim_steps=int(cfg.diff_optim_steps),
                                     diff_lr=float(cfg.diff_lr), pose_augment=False)
    if kinematics and kinematics["mode"] == "simulator_geometry":
        from simulator_kinematics import SimulatorDiffRobot
        robot = SimulatorDiffRobot(kinematics["calibration"])
        robot.to("cuda")
    return robot, model


def rng_state():
    state = np.random.get_state()
    return {"numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    a = state["numpy"]
    np.random.set_state((a[0], np.array(a[1], dtype=np.uint32), a[2], a[3], a[4]))
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, step, task, pilot, best, config, batch_size, selection):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({"schema_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "task": task, "pilot_only": pilot, "best_validation_loss": best,
                "config": config, "batch_size": batch_size, "data_selection": selection,
                "rng": rng_state()}, temporary)
    temporary.replace(path)


def validate_loss(model, robot, dataset, batch_size):
    # Repeat the same validation draws without advancing the training RNG stream.
    state = rng_state()
    seed_all(91001)
    model.eval()
    values = []
    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for index, batch in enumerate(loader):
                batch = batch_to_device(batch, "cuda")
                loss, _ = model.loss(**batch, robot=robot)
                values.append(float(loss))
                if index + 1 >= int(CFG.training.validation_batches):
                    break
    finally:
        model.train()
        restore_rng(state)
    if not values or not np.isfinite(values).all():
        raise RuntimeError("Invalid validation loss")
    return statistics.mean(values)


def train(args):
    task = args.task or CFG.task
    pilot = bool(args.pilot)
    seed_all(int(CFG.seed))
    torch.set_num_threads(int(CFG.training.cpu_threads))
    if not torch.cuda.is_available():
        raise RuntimeError("This training configuration requires CUDA")
    root = data_root(task, pilot)
    dataset = EpisodeDataset(root / "train", True)
    validation = EpisodeDataset(root / "val", True, augment=False)
    for split, selected in (("train", dataset), ("val", validation)):
        expected = {"train": 2, "val": 1}[split] if pilot else int(CFG.data.splits[split].episodes)
        if len(selected.manifest["episodes"]) != expected:
            raise ValueError(f"Incomplete {split} data: expected {expected} episodes; finish collect first")
    run_name = args.run or ("smoke" if pilot else "seed7")
    if Path(run_name).name != run_name:
        raise ValueError("--run must be a single directory name")
    run_root = Path(CFG.state_root) / "runs" / task / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint = run_root / "latest.pt"
    if checkpoint.exists() and not args.resume:
        raise RuntimeError(f"Checkpoint exists at {checkpoint}; use --resume or a new --run name")
    steps = int(args.updates or (40 if pilot else CFG.training.updates))
    batch_size = int(args.batch_size or CFG.training.batch_size)
    accumulate = int(CFG.training.gradient_accumulate_every)
    robot, model = create_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CFG.training.learning_rate))
    start_step, best = 0, float("inf")
    config = OmegaConf.to_container(CFG, resolve=True)
    selection = {"train": dataset.manifest, "val": validation.manifest}
    if args.resume:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["task"] != task or payload["pilot_only"] != pilot:
            raise RuntimeError("Checkpoint task/dataset scope mismatch")
        if not same_experiment_config(payload["config"], config):
            raise RuntimeError("Configuration changed; use a new run instead of silently resuming")
        if payload["batch_size"] != batch_size or payload["data_selection"] != selection:
            raise RuntimeError("Batch size or data selection changed; use a new run")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step, best = payload["step"], payload["best_validation_loss"]
        restore_rng(payload["rng"])
    if steps <= start_step:
        raise ValueError(f"--updates must exceed the saved update {start_step}")
    atomic_json(run_root / "experiment.json", {"configuration": config, "task": task,
                                               "batch_size": batch_size, "data_selection": selection})
    # Sampling with replacement makes resumed batch selection depend only on saved RNG.
    def next_batch():
        indices = np.random.randint(len(dataset), size=batch_size)
        return batch_to_device(torch.utils.data.default_collate([dataset[int(i)] for i in indices]), "cuda")
    torch.cuda.reset_peak_memory_stats()
    durations = []
    process_start = time.perf_counter()
    initial_val = validate_loss(model, robot, validation, batch_size)
    log_path = run_root / "metrics.csv"
    new_log = not log_path.exists()
    with log_path.open("a", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=["step", "train_loss", "validation_loss", "update_seconds", "peak_allocated_gib"])
        if new_log:
            writer.writeheader()
        model.train()
        for step in range(start_step + 1, steps + 1):
            torch.cuda.synchronize()
            before = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(accumulate):
                batch = next_batch()
                loss, _ = model.loss(**batch, robot=robot)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at update {step}")
                (loss / accumulate).backward()
                total_loss += float(loss.detach()) / accumulate
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.perf_counter() - before
            if step > start_step + int(CFG.training.warmup_updates):
                durations.append(duration)
            val = None
            if step % int(CFG.training.validate_every) == 0 or step == steps:
                val = validate_loss(model, robot, validation, batch_size)
                if val < best:
                    best = val
                    torch.save(model.state_dict(), run_root / "best-model.pt")
            peak = torch.cuda.max_memory_allocated() / 2**30
            writer.writerow({"step": step, "train_loss": total_loss, "validation_loss": val,
                             "update_seconds": duration, "peak_allocated_gib": peak})
            log.flush()
            if step % 10 == 0 or step == steps:
                print(f"Update {step}/{steps}: loss={total_loss:.4f}, {duration:.3f}s, peak={peak:.2f} GiB", flush=True)
            if step % int(CFG.training.save_every) == 0 or step == steps:
                save_checkpoint(checkpoint, model, optimizer, step, task, pilot, best, config, batch_size, selection)
    result = {"purpose": "Training infrastructure smoke/throughput check" if pilot else "Single-task exploratory training",
              "pilot_only": pilot, "task": task, "training_seed": int(CFG.seed),
              "start_update": start_step, "completed_updates": max(start_step, steps),
              "batch_size": batch_size, "gradient_accumulation": accumulate,
              "effective_batch_size": batch_size * accumulate, "precision": "float32",
              "model_parameters": sum(p.numel() for p in model.parameters()),
              "train_episodes": len(dataset.manifest["episodes"]), "train_segments": len(dataset),
              "validation_episodes": len(validation.manifest["episodes"]),
              "host_episode_cache_mib": (dataset._cache_bytes+validation._cache_bytes) / 2**20,
              "initial_validation_loss": initial_val, "best_validation_loss": best,
              "measured_updates_after_warmup": len(durations),
              "median_update_seconds": statistics.median(durations) if durations else None,
              "mean_update_seconds": statistics.mean(durations) if durations else None,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "elapsed_training_and_validation_seconds": time.perf_counter() - process_start,
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "checkpoint": str(checkpoint), "success_rate_measured": False,
              "notes": "Timing excludes model/data startup and checkpoints; full-data I/O and validation can add overhead. Validation loss is not task success. No PerAct; no EMA model used (upstream sampling also uses raw weights)."}
    atomic_json(run_root / "training_report.json", result)
    report_name = f"{task}-{run_name}-u{max(start_step, steps)}-training.json"
    atomic_json(HERE / "results" / report_name, result)
    print(json.dumps(result, indent=2), flush=True)


def model_input(joints, poses, pcd, rgb, proprio, robot):
    """Two endpoints, no future joint target; highest rank matches author eval."""
    def tensor(value):
        return torch.as_tensor(np.asarray(value, dtype=np.float32), device="cuda")
    return {"cond": {0: tensor(joints[0:1])},
            "joint_positions": tensor(joints[None]), "gripper_poses": tensor(poses[None]),
            "start": tensor(poses[0:1]), "end": tensor(poses[-1:]),
            "pcds": tensor(pcd[None]), "rgbs": tensor(rgb[None]), "proprios": tensor(proprio[None]),
            "rank": torch.nn.functional.one_hot(torch.tensor([9], device="cuda"), 10).float(),
            "robot": robot}


def pose_errors(endpoint, commanded):
    endpoint = np.asarray(endpoint, dtype=np.float64)
    commanded = np.asarray(commanded, dtype=np.float64)
    if not np.isfinite(endpoint).all():
        raise RuntimeError("Nonfinite endpoint")
    pos_cm = float(np.linalg.norm(endpoint[:3] - commanded[:3]) * 100)
    q = endpoint[3:] / np.linalg.norm(endpoint[3:])
    r = commanded[3:] / np.linalg.norm(commanded[3:])
    rot_deg = float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(q, r)), 0, 1))))
    return pos_cm, rot_deg


def require_qualified_validation(args, payload):
    if args.split != "test" or payload["pilot_only"]:
        return
    if not args.baseline_validation:
        raise ValueError("Formal test requires --baseline-validation pointing to a qualified validation summary")
    summary = json.loads(args.baseline_validation.read_text())
    validation = json.loads(Path(summary["source"]).read_text())
    if (not summary["baseline_validation_passed"] or not validation.get("collection_complete", True)
            or validation["split"] != "val" or validation["checkpoint_updates"] != payload["step"]
            or Path(validation["checkpoint"]).resolve() != args.checkpoint.resolve()
            or validation["task"] != payload["task"]
            or validation.get("inference_kinematics", {"mode": "upstream"}) != inference_kinematics(args)):
        raise ValueError("The nominal validation result does not qualify this checkpoint")


def offline(args):
    """Paired expert-goal perturbations: model FK endpoints, not simulation."""
    task = args.task or CFG.task
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["task"] != task:
        raise RuntimeError("Task mismatch")
    if not same_experiment_config(payload["config"], OmegaConf.to_container(CFG, resolve=True)):
        raise RuntimeError("Checkpoint configuration differs from experiment.yaml")
    require_qualified_validation(args, payload)
    pilot = bool(payload["pilot_only"])
    dataset = EpisodeDataset(data_root(task, pilot) / args.split, False, augment=False)
    if not pilot and len(dataset.manifest["episodes"]) != int(CFG.data.splits[args.split].episodes):
        raise ValueError("Formal evaluation data incomplete; finish collect first")
    kinematics = inference_kinematics(args)
    robot, model = create_model(kinematics)
    model.load_state_dict(payload["model"])
    model.eval()
    rng = np.random.default_rng(int(CFG.diagnostic.direction_seed))
    levels = list(CFG.diagnostic.translation_cm) if not args.quick else [0, 1]
    records = []
    for index, (episode_index, start, end) in enumerate(dataset.segments):
        if args.max_segments and index >= args.max_segments:
            break
        arrays = dataset.episode(episode_index)
        # Actual keyframe endpoint, as in the upstream online evaluator.
        poses = arrays["gripper_poses"][[start, end]].copy()
        joints = arrays["joint_positions"][[start, end]].copy()
        for repeat in range(int(CFG.diagnostic.direction_repeats)):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            for cm in levels:
                commanded = poses[-1].copy()
                commanded[:3] += float(cm) / 100 * direction
                input_poses = poses.copy()
                input_poses[-1] = commanded
                batch = model_input(joints, input_poses, arrays["pcds"][start], arrays["rgbs"][start], arrays["proprios"][start], robot)
                diffusion_seed = int(CFG.diagnostic.diffusion_seed_base) + index * 100 + repeat
                seed_all(diffusion_seed)
                try:
                    # The diffusion sampler disables grads internally; the FK correction
                    # requires gradients, so do not wrap this call in inference_mode.
                    prediction = model.conditional_sample(**batch, verbose=False, return_diffusion=False)
                    predicted_joints = prediction["multi"]["joint_positions"].detach()
                    endpoint = robot.forward_kinematics_batch(predicted_joints[:, -1])[0].detach().cpu().numpy()
                    pos_cm, rot_deg = pose_errors(endpoint, commanded)
                    error = None
                except (RuntimeError, ValueError) as exc:
                    pos_cm = rot_deg = None
                    error = str(exc)
                record = {"episode": episode_index, "segment": index, "start_frame": start, "end_frame": end,
                          "repeat": repeat, "translation_cm": float(cm), "direction": direction.tolist(),
                          "commanded_pose": commanded.tolist(),
                          "diffusion_seed": diffusion_seed, "endpoint_position_error_cm": pos_cm,
                          "endpoint_rotation_error_deg": rot_deg, "error": error}
                records.append(record)
                print(f"Offline segment {index}, noise {cm}cm: endpoint error {pos_cm}, error={error}", flush=True)
    result = {"purpose": "Offline paired expert-goal sensitivity; no simulation actions",
              "pilot_only": pilot, "checkpoint_updates": payload["step"], "split": args.split,
              "inference_kinematics": kinematics,
              "task_success_measured": False, "execution_success_measured": False,
              "all_conditions_retained": True, "records": records,
              "limitations": "Not PerAct predictions; not execution tracking; not a task-success robustness curve. A short pilot checkpoint is not evidence of an HDP defect."}
    target = Path(args.output) if args.output else HERE / "results" / f"{task}-offline-{'pilot' if pilot else args.split}.json"
    atomic_json(target, result)
    if any(record["error"] for record in records):
        raise RuntimeError(f"Some samples failed; all rows are retained in {target}")


def rollout(args):
    """Expert-keyframe guided full rollouts, paired by initial episode.

    This is an oracle high-level sequence, not a PerAct rollout or an arbitrary
    held-out intermediate-state reset. Future joint trajectories are never fed
    to the model. Failed trajectories remain in the denominator.
    """
    task_name = args.task or CFG.task
    if task_name != "open_microwave":
        raise ValueError("The articulated progress metric is configured for open_microwave only")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["task"] != task_name or not same_experiment_config(payload["config"], OmegaConf.to_container(CFG, resolve=True)):
        raise RuntimeError("Checkpoint task/configuration mismatch")
    require_qualified_validation(args, payload)
    pilot = bool(payload["pilot_only"])
    dataset = EpisodeDataset(data_root(task_name, pilot) / args.split, False, augment=False)
    if not pilot and len(dataset.manifest["episodes"]) != int(CFG.data.splits[args.split].episodes):
        raise ValueError("Formal evaluation data incomplete; finish collect first")
    if int(CFG.diagnostic.direction_repeats) != 1:
        raise ValueError("This rollout protocol uses one paired direction per decision/episode")
    kinematics = inference_kinematics(args)
    robot, model = create_model(kinematics)
    model.load_state_dict(payload["model"])
    model.eval()
    env = Environment(MoveArmThenGripper(JointsTrajectoryActionMode(int(CFG.training.horizon)), Discrete()),
                      obs_config=_create_obs_config(["front"], [128, 128]), headless=True)
    env.launch()
    levels = [0] if args.nominal_only else (list(CFG.diagnostic.translation_cm) if not args.quick else [0, 1])
    records = []
    target = Path(args.output) if args.output else HERE / "results" / f"{task_name}-rollout-{'pilot' if pilot else args.split}.json"
    try:
        task = env.get_task(task_file_to_task_class(task_name))
        door = Joint("microwave_door_joint")
        rng = np.random.default_rng(int(CFG.diagnostic.direction_seed))
        for episode, item in enumerate(dataset.manifest["episodes"]):
            if args.max_episodes and episode >= args.max_episodes:
                break
            with (dataset.root / item["directory"] / "demo.pkl").open("rb") as stream:
                demo = pickle.load(stream)
            arrays = dataset.episode(episode)
            directions = rng.normal(size=(len(item["keyframes"])-1, 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            reference_reset = None
            for cm in levels:
                _, obs = task.reset_to_demo(demo)
                initial_pose = np.asarray(obs.gripper_pose).copy()
                angle_start = float(door.get_joint_position())
                reset_state = {"tip": initial_pose, "joints": np.asarray(obs.joint_positions).copy(),
                               "base": np.asarray(task._task.get_base().get_pose()).copy(),
                               "door": np.asarray([angle_start, obs.gripper_open])}
                if reference_reset is None:
                    reference_reset = reset_state
                tip_cm, tip_deg = pose_errors(reset_state["tip"], reference_reset["tip"])
                base_cm, base_deg = pose_errors(reset_state["base"], reference_reset["base"])
                reset_deltas = {"tip_position_mm": tip_cm*10, "tip_rotation_deg": tip_deg,
                    "base_position_mm": base_cm*10, "base_rotation_deg": base_deg,
                    "max_joint_delta_deg": float(np.degrees(np.max(np.abs(reset_state["joints"]-reference_reset["joints"])))),
                    "door_delta_deg": float(np.degrees(abs(reset_state["door"][0]-reference_reset["door"][0]))),
                    "gripper_delta": float(abs(reset_state["door"][1]-reference_reset["door"][1]))}
                # Unit-aware bounds, including <=1/50 of the smallest 5 mm
                # perturbation. These are approximate paired physics resets.
                reset_bounds = {"tip_position_mm": 0.1, "tip_rotation_deg": 0.02,
                    "base_position_mm": 0.1, "base_rotation_deg": 0.02,
                    "max_joint_delta_deg": 0.05, "door_delta_deg": 0.01, "gripper_delta": 0.0}
                if not all(delta <= reset_bounds[name] for name, delta in reset_deltas.items()):
                    atomic_json(HERE / "results/reset-mismatch-debug.json", {
                        "purpose": "Infrastructure reset diagnostic; not a policy result",
                        "deltas": reset_deltas,
                        "reference": {k: v.tolist() for k, v in reference_reset.items()},
                        "current": {k: v.tolist() for k, v in reset_state.items()}})
                    raise RuntimeError(f"Reset states differ between paired conditions: {reset_deltas}; stop instead of reporting a policy comparison")
                row = {"episode": episode, "translation_cm": float(cm), "perturb_at": args.perturb_at,
                       "initial_pose": initial_pose.tolist(), "initial_door_angle_rad": angle_start,
                       "paired_reset_verified": True,
                       "initial_state_differences_from_nominal": reset_deltas,
                       "initial_state_tolerance_bounds": reset_bounds,
                       "task_success": False, "execution_failure": None, "stages": []}
                for stage, end in enumerate(item["keyframes"][1:]):
                    goal = arrays["gripper_poses"][end].copy()
                    if args.perturb_at == "every" or stage == 0:
                        goal[:3] += float(cm) / 100 * directions[stage]
                    joints = np.repeat(np.asarray(obs.joint_positions, dtype=np.float32)[None], 2, axis=0)
                    poses = np.stack([obs.gripper_pose, goal]).astype(np.float32)
                    batch = model_input(joints, poses,
                                        cv2.resize(obs.front_point_cloud, (64, 64), interpolation=cv2.INTER_NEAREST),
                                        cv2.resize(obs.front_rgb, (64, 64)), obs.get_low_dim_data(), robot)
                    diffusion_seed = int(CFG.diagnostic.diffusion_seed_base) + episode * 100 + stage
                    seed_all(diffusion_seed)
                    before_angle = float(door.get_joint_position())
                    try:
                        prediction = model.conditional_sample(**batch, verbose=False, return_diffusion=False)
                        trajectory = prediction["multi"]["joint_positions"][0].detach().cpu().numpy()
                        planned_endpoint = prediction["multi"]["traj"][0, -1].detach().cpu().numpy()
                        limits = np.asarray([[-2.89, -1.76, -2.89, -3.07, -2.89, -0.017, -2.89],
                                             [2.89, 1.76, 2.89, -0.069, 2.89, 3.75, 2.89]])
                        if not np.isfinite(trajectory).all():
                            raise InvalidTrajectoryError("Nonfinite joint trajectory")
                        if np.any(trajectory < limits[0]-1e-3) or np.any(trajectory > limits[1]+1e-3):
                            raise InvalidTrajectoryError("Joint trajectory outside configured limits")
                        if not np.isfinite(planned_endpoint).all():
                            raise InvalidTrajectoryError("Nonfinite planned endpoint")
                        planned_p_err, planned_r_err = pose_errors(planned_endpoint, goal)
                        gripper = float(demo[end].gripper_open)
                        obs, reward, terminate = task.step(np.concatenate([trajectory.reshape(-1), [gripper]]))
                        p_err, r_err = pose_errors(np.asarray(obs.gripper_pose), goal)
                        row["stages"].append({"stage": stage, "commanded_pose": goal.tolist(),
                            "direction": directions[stage].tolist(), "diffusion_seed": diffusion_seed,
                            "expert_gripper_command": gripper, "executed_position_error_cm": p_err,
                            "executed_rotation_error_deg": r_err,
                            "planned_endpoint_position_error_cm": planned_p_err,
                            "planned_endpoint_rotation_error_deg": planned_r_err,
                            "executed_endpoint_max_joint_error_deg": float(np.degrees(np.max(np.abs(obs.joint_positions-trajectory[-1])))),
                            "door_angle_change_deg": float(np.degrees(door.get_joint_position()-before_angle)),
                            "gripper_open_observed": float(obs.gripper_open)})
                        row["task_success"] = bool(reward >= 1)
                        if terminate or row["task_success"]:
                            break
                    except (IKError, ConfigurationPathError, InvalidActionError, InvalidTrajectoryError) as exc:
                        row["execution_failure"] = f"{type(exc).__name__}: {exc}"
                        break
                row["final_door_angle_rad"] = float(door.get_joint_position())
                records.append(row)
                atomic_json(target, {"purpose": "Expert-guided RK-Diffuser full rollouts; no PerAct",
                    "collection_complete": False, "planned_translation_cm": levels,
                    "pilot_only": pilot, "checkpoint_updates": payload["step"], "split": args.split,
                    "task": task_name, "checkpoint": str(args.checkpoint),
                    "inference_kinematics": kinematics,
                    "all_failures_retained": True, "baseline_qualification_required": True,
                    "records": records,
                    "limitations": "Short pilot results test execution infrastructure only. Physics resets are paired within recorded physical tolerances, not bitwise identical. Gripper closure is not a grasp-success measure. This does not implement intermediate-state reset diagnosis."})
                print(f"Rollout episode {episode}, noise {cm}cm: success={row['task_success']}, error={row['execution_failure']}", flush=True)
        complete_result = json.loads(target.read_text())
        complete_result["collection_complete"] = True
        atomic_json(target, complete_result)
    finally:
        env.shutdown()


def summarize(args):
    source = json.loads(args.input.read_text())
    if not source.get("collection_complete", True):
        raise ValueError("Rollout collection is incomplete; partial results are not a completed comparison")
    if "rollouts" not in source["purpose"]:
        raise ValueError("Summarize expects a rollout result, not offline endpoint errors")
    records = source["records"]
    by_level = {}
    for row in records:
        rows = by_level.setdefault(row["translation_cm"], {})
        if row["episode"] in rows:
            raise ValueError("Duplicate episode/condition; do not silently pool repeated trials")
        rows[row["episode"]] = row
    baseline = by_level[0.0]
    rng = np.random.default_rng(90001)
    result = {"source": str(args.input), "pilot_only": source["pilot_only"],
              "inference_kinematics_mode": source.get("inference_kinematics", {"mode": "upstream"})["mode"],
              "training_seed_count": 1, "bootstrap_unit": "paired initial episode",
              "baseline_validation_gate": "At least 20 validation episodes and >=70% nominal task success; pilot excluded",
              "conditions": []}
    for level, rows in sorted(by_level.items()):
        if set(rows) != set(baseline):
            raise ValueError("Incomplete paired conditions; finish collection before summarizing")
        ids = sorted(rows)
        success = np.array([rows[i]["task_success"] for i in ids], dtype=float)
        nominal = np.array([baseline[i]["task_success"] for i in ids], dtype=float)
        samples = rng.integers(0, len(ids), size=(int(CFG.diagnostic.bootstrap_repeats), len(ids)))
        means = success[samples].mean(axis=1)
        deltas = (success-nominal)[samples].mean(axis=1)
        result["conditions"].append({"translation_cm": level, "episodes": len(ids),
            "task_success_rate": float(success.mean()),
            "success_rate_bootstrap_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
            "paired_success_difference_from_nominal": float((success-nominal).mean()),
            "paired_difference_bootstrap_95ci": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "execution_failures": sum(rows[i]["execution_failure"] is not None for i in ids)})
    result["baseline_validation_passed"] = bool(not source["pilot_only"] and source["split"] == "val"
        and len(baseline) >= 20 and np.mean([r["task_success"] for r in baseline.values()]) >= 0.7)
    result["limitations"] = "Exploratory one-seed estimate; bootstrap does not capture training-seed variation. A failed baseline gate does not establish an HDP defect. Very small samples and all-equal outcomes give unreliable or degenerate percentile intervals."
    output = args.output or args.input.with_name(args.input.stem + "-summary.json")
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


def parser():
    def positive_int(text):
        value = int(text)
        if value <= 0:
            raise argparse.ArgumentTypeError("Expected a positive integer")
        return value
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    data = commands.add_parser("collect")
    data.add_argument("--task", choices=["reach_target", "open_microwave"])
    data.add_argument("--pilot", action="store_true")
    data.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    training = commands.add_parser("train")
    training.add_argument("--task", choices=["reach_target", "open_microwave"])
    training.add_argument("--pilot", action="store_true")
    training.add_argument("--updates", type=positive_int)
    training.add_argument("--batch-size", type=positive_int)
    training.add_argument("--run")
    training.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("offline")
    evaluation.add_argument("--task", choices=["reach_target", "open_microwave"])
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", choices=["val", "test"], default="val")
    evaluation.add_argument("--quick", action="store_true")
    evaluation.add_argument("--max-segments", type=positive_int)
    evaluation.add_argument("--output", type=Path)
    evaluation.add_argument("--baseline-validation", type=Path)
    evaluation.add_argument("--kinematics-calibration", type=Path,
                            help="Simulator geometry export; changes inference FK only")
    execution = commands.add_parser("rollout")
    execution.add_argument("--task", choices=["open_microwave"])
    execution.add_argument("--checkpoint", type=Path, required=True)
    execution.add_argument("--split", choices=["val", "test"], default="val")
    execution.add_argument("--quick", action="store_true")
    execution.add_argument("--nominal-only", action="store_true")
    execution.add_argument("--max-episodes", type=positive_int)
    execution.add_argument("--perturb-at", choices=["first", "every"], default="first")
    execution.add_argument("--output", type=Path)
    execution.add_argument("--baseline-validation", type=Path)
    execution.add_argument("--kinematics-calibration", type=Path,
                           help="Simulator geometry export; changes inference FK only")
    summary = commands.add_parser("summarize")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    return root


if __name__ == "__main__":
    options = parser().parse_args()
    {"collect": collect, "train": train, "offline": offline, "rollout": rollout, "summarize": summarize}[options.command](options)
