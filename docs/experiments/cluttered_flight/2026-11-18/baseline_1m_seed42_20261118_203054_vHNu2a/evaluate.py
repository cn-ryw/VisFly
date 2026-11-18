import argparse
from collections import Counter
import csv
from itertools import product
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

from train import EXAMPLES, HOLDOUT_SCENE, MILESTONES, RUN, SEED, TRAIN_SCENE, make_env, write_json
from VisFly.exps.examples.run import load_yaml_config
from VisFly.utils.algorithms.PPO import PPO

LAYOUTS = {"train": TRAIN_SCENE, "heldout": HOLDOUT_SCENE}
STARTS = tuple(product((1.0,), (-1.8, -0.9, 0.0, 0.9, 1.8), (0.65, 1.5, 2.35)))
CHECKPOINTS = {32: RUN.parent / "PPO_quickstart_1.zip"}
CHECKPOINTS.update({steps: RUN / f"PPO_{steps:07d}.zip" for steps in MILESTONES})


def make_eval_env(layout, display=False):
    config = load_yaml_config(EXAMPLES / "env_cfgs/cluttered_flight.yaml")["eval_env"]
    config.update(num_scene=1, num_agent_per_scene=1, seed=SEED, visual=True)
    config["scene_kwargs"]["render_settings"] = (
        {"mode": "follow", "view": "near", "resolution": [540, 960], "trajectory": True, "line_width": 3.0}
        if display else None
    )
    return make_env(config, LAYOUTS[layout])


def reset_case(env, start):
    pos = torch.tensor([start], dtype=torch.float32)
    attitude = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    obs = env.reset_agent_by_id(state=(pos, attitude, torch.zeros(1, 3), torch.zeros(1, 3)))
    assert torch.allclose(env.position, pos) and torch.allclose(env.t, torch.zeros(1))
    assert torch.allclose(env.velocity, torch.zeros(1, 3))
    assert torch.allclose(env.angular_velocity, torch.zeros(1, 3))
    assert obs["depth"].shape == (1, 1, 64, 64)
    assert not bool(env.is_collision[0]) and not bool(env.is_out_bounds[0])
    return obs


def classify(env, info):
    if bool(env.is_collision[0]) or bool(info["episode"]["extra"]["collision"]):
        return "collision"
    if bool(env.is_out_bounds[0]):
        return "out_of_bounds"
    if bool(env.failure[0]):
        return "failure"
    if info["is_success"]:
        return "success"
    if info["TimeLimit.truncated"]:
        return "timeout"
    raise RuntimeError("Unexpected terminal condition")


def episode(env, model, start):
    obs = reset_case(env, start)
    minimum = env.collision_dis[0].item()
    for step in range(1, env.max_episode_steps + 1):
        previous_pos = env.position[0].clone()
        previous_surface = env.collision_point[0].clone()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action, is_test=True)
        minimum = min(minimum, env.collision_dis[0].item())
        if bool(np.asarray(done)[0]):
            break
    else:
        raise RuntimeError("Episode exceeded its step limit")
    result = classify(env, info[0])
    previous_delta = previous_surface - previous_pos
    approach = ""
    if result == "collision":
        vertical = float(previous_delta[2])
        lateral = float(previous_delta[:2].norm())
        approach = ("above" if vertical > 0 else "below") if abs(vertical) > lateral else "lateral"
    position = env.position[0]
    return {
        "outcome": result,
        "episode_steps": step,
        "sim_seconds": round(step * env.envs.dynamics.ctrl_dt, 3),
        "min_clearance_m": round(minimum, 5),
        "final_goal_distance_m": round((position - env.target[0]).norm().item(), 5),
        "terminal_x_m": round(position[0].item(), 5),
        "terminal_y_m": round(position[1].item(), 5),
        "terminal_z_m": round(position[2].item(), 5),
        "preterminal_obstacle_dx_m": round(previous_delta[0].item(), 5),
        "preterminal_obstacle_dy_m": round(previous_delta[1].item(), 5),
        "preterminal_obstacle_dz_m": round(previous_delta[2].item(), 5),
        "collision_approach": approach,
        "reward": round(float(info[0]["episode"]["r"]), 6),
    }


def summarise(rows):
    counts = Counter(row["outcome"] for row in rows)
    success = [row for row in rows if row["outcome"] == "success"]
    above = [row for row in rows if row["collision_approach"] == "above"]
    return {
        "outcomes": dict(counts),
        "success_median_time_s": round(statistics.median(row["sim_seconds"] for row in success), 3) if success else None,
        "success_median_min_clearance_m": round(statistics.median(row["min_clearance_m"] for row in success), 3) if success else None,
        "success_min_clearance_m": min((row["min_clearance_m"] for row in success), default=None),
        "median_final_goal_distance_m": round(statistics.median(row["final_goal_distance_m"] for row in rows), 3),
        "collision_approaches": dict(Counter(row["collision_approach"] for row in rows if row["outcome"] == "collision")),
        "overhead_contacts_above_5m": sum(row["terminal_z_m"] >= 5.0 for row in above),
    }


def choose_showcase(rows, summaries):
    best = max(
        MILESTONES,
        key=lambda steps: (
            summaries[str(steps)]["train"]["outcomes"].get("success", 0),
            -summaries[str(steps)]["train"]["outcomes"].get("collision", 0),
            steps,
        ),
    )
    matched = [row for row in rows if row["checkpoint_steps"] == best]
    selected = next((row for row in matched if row["outcome"] == "success"), None)
    if selected is None:
        train_rows = [row for row in matched if row["layout"] == "train"]
        common = Counter(row["outcome"] for row in train_rows).most_common(1)[0][0]
        selected = next(row for row in train_rows if row["outcome"] == common)
    return selected


def evaluate(smoke=False, wait=False):
    rows = []
    summaries = {}
    checkpoints = {32: CHECKPOINTS[32]} if smoke else CHECKPOINTS
    for steps, checkpoint in checkpoints.items():
        if wait and not checkpoint.is_file():
            print(f"WAITING_FOR_CHECKPOINT {steps}", flush=True)
            deadline = time.monotonic() + 3600
            while not checkpoint.is_file() and time.monotonic() < deadline:
                time.sleep(5)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        summaries[str(steps)] = {}
        for layout, scene in LAYOUTS.items():
            set_random_seed(SEED)
            env = make_eval_env(layout)
            try:
                model = PPO.load(checkpoint, env=env, device="cpu")
                assert model.num_timesteps == steps
                env.reset(is_test=True)
                episode_rows = []
                for index, start in enumerate(STARTS, 1):
                    if smoke and index > 2:
                        break
                    result = episode(env, model, start)
                    episode_rows.append({
                        "checkpoint_steps": steps,
                        "layout": layout,
                        "scene": scene.name,
                        "case_id": f"{layout}-{index:02d}",
                        "start_x_m": start[0],
                        "start_y_m": start[1],
                        "start_z_m": start[2],
                        **result,
                    })
                summaries[str(steps)][layout] = summarise(episode_rows)
                print(f"EVALUATED {steps} {layout} {summaries[str(steps)][layout]}", flush=True)
                rows.extend(episode_rows)
            finally:
                env.close()
        if not smoke:
            temporary = RUN / "episodes.tmp"
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(RUN / "episodes.csv")

    if smoke:
        assert len(rows) == 4
        return
    slots = {(row["checkpoint_steps"], row["case_id"]) for row in rows}
    assert len(rows) == len(slots) == 150
    write_json(RUN / "summary.json", {
        "models": summaries,
        "showcase": choose_showcase(rows, summaries),
    })
    print("EVALUATION_COMPLETE 150", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    evaluate(smoke=args.smoke, wait=args.wait)
