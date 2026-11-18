import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
from stable_baselines3.common.utils import set_random_seed

from VisFly.envs.NavigationEnv import NavigationEnv
from VisFly.exps.examples.run import load_yaml_config, resolve_config
from VisFly.utils.algorithms.PPO import PPO
from VisFly.utils.dataloader import SimpleDataLoader

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[4]
EXAMPLES = REPO / "exps/examples"
SCENE_DIR = REPO / "datasets/visfly-beta/configs/scenes/garage_simple_l_medium"
TRAIN_SCENE = SCENE_DIR / "garage_simple_l_medium_0.scene_instance.json"
HOLDOUT_SCENE = SCENE_DIR / "garage_simple_low_m_0.scene_instance.json"
SEED = 42
ROLLOUT_STEPS = 48 * 256
MILESTONES = (258_048, 503_808, 761_856, 1_007_616)
PREFLIGHT_OOM = 42


def make_env(config, scene):
    env = NavigationEnv(**resolve_config(config))
    env.envs.sceneManager._scene_loader = SimpleDataLoader(
        SimpleNamespace(paths=[str(scene)]), batch_size=1, shuffle=False,
    )
    return env


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def train(batch_size):
    config = load_yaml_config(EXAMPLES / "env_cfgs/cluttered_flight.yaml")["env"]
    config.update(num_scene=1, num_agent_per_scene=48, seed=SEED)
    ppo = load_yaml_config(EXAMPLES / "alg_cfgs/cluttered_flight/PPO.yaml")["algorithm"]
    ppo.update(n_steps=256, batch_size=batch_size, device="cuda")
    assert TRAIN_SCENE.is_file()
    assert ppo["n_epochs"] == 10
    assert ROLLOUT_STEPS % batch_size == 0
    manifest = {
        "seed": SEED,
        "repo_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "environment": config,
        "ppo": ppo,
        "train_scene": str(TRAIN_SCENE.relative_to(REPO)),
        "heldout_scene": str(HOLDOUT_SCENE.relative_to(REPO)),
        "requested_steps": 1_000_000,
        "milestones": list(MILESTONES),
        "completed_checkpoints": [],
    }
    env = None
    first_update_done = False
    started = time.monotonic()
    try:
        set_random_seed(SEED, using_cuda=True)
        torch.cuda.reset_peak_memory_stats()
        env = make_env(config, TRAIN_SCENE)
        log_dir = RUN / f"tensorboard_bs{batch_size}"
        log_dir.mkdir(exist_ok=True)
        model = PPO(env=env, seed=SEED, comment="baseline", save_path=str(log_dir), **resolve_config(ppo))
        model.learn(total_timesteps=ROLLOUT_STEPS, reset_num_timesteps=False)
        torch.cuda.synchronize()
        if model.num_timesteps != ROLLOUT_STEPS or model._n_updates != ppo["n_epochs"]:
            raise RuntimeError("The first rollout did not complete a full PPO update")
        if not all(torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()):
            raise RuntimeError("The first PPO update produced nonfinite parameters")
        first_update_done = True
        manifest["preflight"] = {
            "steps": model.num_timesteps,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            "gpu": torch.cuda.get_device_name(),
        }
        write_json(RUN / "training.json", manifest)
        print(f"PREFLIGHT {json.dumps(manifest['preflight'])}", flush=True)

        for milestone in MILESTONES:
            model.learn(total_timesteps=milestone - model.num_timesteps, reset_num_timesteps=False)
            torch.cuda.synchronize()
            if model.num_timesteps != milestone or model._n_updates != milestone // ROLLOUT_STEPS * ppo["n_epochs"]:
                raise RuntimeError(f"Training stopped before full update at {milestone}")
            checkpoint = RUN / f"PPO_{milestone:07d}.zip"
            partial = RUN / f".PPO_{milestone:07d}.partial.zip"
            model.save(partial)
            partial.replace(checkpoint)
            manifest["completed_checkpoints"].append({
                "steps": milestone, "path": checkpoint.name,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            })
            write_json(RUN / "training.json", manifest)
            print(f"CHECKPOINT {milestone}: {checkpoint.name}", flush=True)
    except torch.cuda.OutOfMemoryError:
        if first_update_done:
            raise
        print(f"PREFLIGHT_OUT_OF_MEMORY batch_size={batch_size}", file=sys.stderr, flush=True)
        return PREFLIGHT_OOM
    finally:
        if env is not None:
            env.close()
    print("TRAINING_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True, choices=(512, 256, 128))
    raise SystemExit(train(parser.parse_args().batch_size))
