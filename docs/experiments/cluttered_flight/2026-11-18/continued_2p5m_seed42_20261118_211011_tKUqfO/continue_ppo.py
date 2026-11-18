import argparse
from contextlib import closing
import json
from pathlib import Path
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent
BASELINE = RUN.parent / "baseline_1m_seed42_20261118_203054_vHNu2a"
sys.path.insert(0, str(BASELINE))

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from train import REPO, ROLLOUT_STEPS, SEED, TRAIN_SCENE, make_env, write_json
from VisFly.utils.algorithms.PPO import PPO
from stable_ppo import stable_train

ORIGIN = 1_007_616
MILESTONES = (1_511_424, 2_002_944, 2_506_752)
CHECKPOINTS = {steps: RUN / f"PPO_{steps:07d}.zip" for steps in MILESTONES}
SOURCE = BASELINE / f"PPO_{ORIGIN:07d}.zip"


class CheckedPPO(PPO):
    def train(self):
        buffer = self.rollout_buffer
        inputs = dict(buffer.observations)
        for name in ("actions", "rewards", "returns", "advantages", "values", "log_probs"):
            inputs[name] = getattr(buffer, name)
        for name, array in inputs.items():
            if not np.isfinite(array).all():
                raise FloatingPointError(f"Nonfinite rollout buffer input: {name}, steps={self.num_timesteps}")

        try:
            stable_train(self)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            trace = error.__traceback__
            while trace is not None:
                state = trace.tb_frame.f_locals
                if "rollout_data" in state and "log_ratio" in state:
                    for name in ("log_ratio", "loss", "policy_loss", "value_loss"):
                        value = state.get(name)
                        if isinstance(value, torch.Tensor):
                            finite = value.detach()[torch.isfinite(value.detach())]
                            bounds = (finite.min().item(), finite.max().item()) if finite.numel() else None
                            print(f"DIAGNOSTIC {name}: finite {finite.numel()}/{value.numel()}, range {bounds}", flush=True)
                    break
                trace = trace.tb_next
            raise


def save_snapshot(model, path):
    partial = RUN / ".partial.zip"
    model.save(partial)
    partial.replace(path)


class RecoverySnapshots(BaseCallback):
    def __init__(self, manifest, manifest_path, elapsed, start_steps):
        super().__init__()
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.elapsed = elapsed
        self.last_save = start_steps

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        steps = self.model.num_timesteps
        if steps - self.last_save < 8 * ROLLOUT_STEPS:
            return
        assert self.model._n_updates == steps // ROLLOUT_STEPS * self.model.n_epochs
        path = RUN / f".recovery_{steps}.zip"
        save_snapshot(self.model, path)
        previous = self.manifest.get("progress_checkpoint")
        self.manifest["progress_checkpoint"] = {
            "steps": steps, "path": path.name, "elapsed_seconds": self.elapsed(),
        }
        write_json(self.manifest_path, self.manifest)
        if previous:
            (RUN / previous["path"]).unlink(missing_ok=True)
        self.last_save = steps
        print(f"RECOVERY_CHECKPOINT {steps}", flush=True)


def load_manifest():
    baseline = json.loads((BASELINE / "training.json").read_text())
    assert baseline["completed_checkpoints"][-1]["steps"] == ORIGIN
    assert (baseline["environment"]["num_agent_per_scene"], baseline["ppo"]["batch_size"]) == (48, 512)
    return {
        "source_checkpoint": str(SOURCE.relative_to(REPO)),
        "source_steps": ORIGIN,
        "seed": SEED,
        "repo_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "environment": baseline["environment"],
        "ppo": baseline["ppo"],
        "train_scene": baseline["train_scene"],
        "heldout_scene": baseline["heldout_scene"],
        "milestones": list(MILESTONES),
        "additional_training_steps": MILESTONES[-1] - ORIGIN,
        "completed_checkpoints": [],
    }


def train():
    manifest_path = RUN / "continuation.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else load_manifest()
    completed = manifest["completed_checkpoints"]
    if completed and completed[-1]["steps"] == MILESTONES[-1]:
        print("TRAINING_COMPLETE", flush=True)
        return
    latest = completed[-1] if completed else None
    progress = manifest.get("progress_checkpoint")
    if progress and (latest is None or progress["steps"] > latest["steps"]):
        latest = progress
    start_steps = latest["steps"] if latest else ORIGIN
    source = RUN / latest["path"] if latest else SOURCE
    elapsed_before = latest["elapsed_seconds"] if latest else 0.0
    manifest.setdefault("numerical_recovery", {
        "resume_steps": start_steps,
        "first_invalid_update_steps": 2_015_232,
        "diagnostic_update_steps": 2_113_536,
        "diagnostic_max_log_ratio": 468.8873596191406,
        "implementation": "equivalent PPO surrogate clipping in log space",
    })
    manifest["numerical_recovery"].update({
        "second_overflow_steps": 2_211_840,
        "second_overflow_max_log_ratio": 51.67120361328125,
        "implementation": "log-space PPO surrogate and mathematically equivalent scaled gradient clipping",
    })
    clock_start = time.monotonic()
    def elapsed():
        return round(elapsed_before + time.monotonic() - clock_start, 2)
    set_random_seed(SEED, using_cuda=True)
    logdir = RUN / "tensorboard"
    logdir.mkdir(exist_ok=True)
    write_json(manifest_path, manifest)

    with closing(make_env(manifest["environment"], TRAIN_SCENE)) as env:
        model = CheckedPPO.load(source, env=env, device="cuda")
        model.tensorboard_log = str(logdir)
        model.save_path = str(logdir)
        model.comment = "continued"
        assert model.num_timesteps == start_steps and model._n_updates == start_steps // ROLLOUT_STEPS * 10
        assert (model.n_envs, model.n_steps, model.batch_size, model.n_epochs) == (48, 256, 512, 10)
        assert len(model.policy.optimizer.state) > 0
        print(f"RESUMED {start_steps} with {len(model.policy.optimizer.state)} optimizer entries on {model.device}", flush=True)

        for milestone in MILESTONES:
            if milestone <= start_steps:
                continue
            callback = RecoverySnapshots(manifest, manifest_path, elapsed, model.num_timesteps)
            model.learn(total_timesteps=milestone - model.num_timesteps, reset_num_timesteps=False, callback=callback)
            torch.cuda.synchronize()
            if model.num_timesteps != milestone or model._n_updates != milestone // ROLLOUT_STEPS * 10:
                raise RuntimeError(f"Training stopped before a full update at {milestone}")
            if not all(torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()):
                raise RuntimeError(f"Non-finite policy parameters at {milestone}")
            checkpoint = CHECKPOINTS[milestone]
            model.policy_save_path = str(checkpoint)
            save_snapshot(model, checkpoint)
            completed.append({
                "steps": milestone,
                "path": checkpoint.name,
                "elapsed_seconds": elapsed(),
                "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            })
            progress = manifest.pop("progress_checkpoint", None)
            write_json(manifest_path, manifest)
            if progress:
                (RUN / progress["path"]).unlink(missing_ok=True)
            print(f"CHECKPOINT {milestone}: {checkpoint.name}", flush=True)
    print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train",))
    parser.parse_args()
    train()
