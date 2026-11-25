"""Evaluate frozen original PPO weights and record every actual integration position."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from paper_runtime import array_sha, load_example, make_env, reset_seed, sha, write_json


def positions(value):
    return value.detach().cpu().numpy().copy()


def episode(env, policy, seed):
    from utils.maths import Integrator

    obs = reset_seed(env, seed)
    initial = {key: array_sha(value) for key, value in obs.items()}
    control = [positions(env.position)]
    physics = [control[0].copy()]
    depth = [obs["depth"].copy()]
    states = [obs["state"].copy()]
    target = obs["target"].copy()
    actions, rewards, native_collision, native_oob, native_distance = [], [], [], [], []
    integrate = Integrator.integrate

    def record_integration(**kwargs):
        result = integrate(**kwargs)
        physics.append(positions(result[0].T))
        return result

    with patch.object(Integrator, "integrate", staticmethod(record_integration)), torch.no_grad():
        for step in range(1, env.max_episode_steps + 1):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action, is_test=True)
            actions.append(action.copy())
            rewards.append(reward.copy())
            control.append(positions(env.position))
            states.append(obs["state"].copy())
            depth.append(obs["depth"].copy())
            native_collision.append(positions(env.is_collision))
            native_oob.append(positions(env.envs._is_out_bounds))
            native_distance.append(positions(env.collision_dis))
            if np.asarray(done).any():
                if not np.asarray(done).all():
                    raise ValueError(f"Team terminal flags disagree for seed {seed}")
                break
        else:
            raise RuntimeError(f"Missing terminal flag on the time limit for seed {seed}")
    success = [bool(i["is_success"]) for i in info]
    timeout = [bool(i.get("TimeLimit.truncated")) for i in info]
    if len(set(success)) != 1 or len(set(timeout)) != 1:
        raise ValueError("Team flags disagree")
    if success[0]:
        native = "success"
    elif native_collision[-1].any():
        native = "collision"
    elif timeout[0]:
        native = "timeout"
    else:
        raise ValueError(f"Unclassified termination: {info[0].keys()}")
    data = dict(control_xyz=np.stack(control), substep_xyz=np.stack(physics), state=np.stack(states),
                depth=np.stack(depth), target=target, action=np.stack(actions), reward=np.stack(rewards),
                native_collision=np.stack(native_collision), out_of_bounds=np.stack(native_oob),
                native_distance=np.stack(native_distance), native_outcome=np.array(native),
                official_success=np.array(success), official_timeout=np.array(timeout), seed=np.array(seed))
    if len(physics) != step * 8 + 1 or not np.array_equal(data["substep_xyz"][::8], data["control_xyz"]):
        raise ValueError("Integration record and control poses are misaligned")
    if not all(np.isfinite(data[k]).all() for k in ("control_xyz","substep_xyz","depth","state","action","reward")):
        raise FloatingPointError(f"Non-finite observation or trajectory at seed {seed}")
    if not all(np.array_equal(info[i]["terminal_observation"]["state"], obs["state"][i]) for i in range(3)):
        raise ValueError("Returned terminal state differs from the official terminal observation")
    record = dict(seed=seed, native_outcome=native, steps=step, actual_physics_steps=len(physics)-1,
                  initial_observations_sha256=initial, trajectory_sha256=array_sha(data["substep_xyz"]),
                  depth_sha256=array_sha(data["depth"]), actions_sha256=array_sha(data["action"]),
                  final_official_success=success, final_official_collision=native_collision[-1].tolist(),
                  final_official_out_of_bounds=native_oob[-1].tolist(), final_official_timeout=timeout,
                  team_mean_return=float(data["reward"].sum(axis=0).mean()))
    return data, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("split", choices=["development", "held-out", "smoke"])
    parser.add_argument("--seeds", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    weight = args.checkpoint.resolve()
    protocol = run / "setup/protocol.json"
    meta = json.loads(weight.with_suffix(".json").read_text())
    if sha(weight) != meta["sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")
    if args.split == "smoke":
        first, last = args.seeds or (997, 998)
    else:
        spec = json.loads(protocol.read_text())
        first, last = spec["held_out" if args.split == "held-out" else "development"]["seeds_inclusive"]
        if args.seeds or meta["agent_steps"] not in spec["post_update_checkpoints"]:
            raise ValueError("Formal evaluations use the predefined complete cohort")
        if args.split == "held-out":
            if meta["agent_steps"] != spec["held_out"]["only_checkpoint"] or not (run/"train/complete.json").is_file():
                raise ValueError("Holdout needs the frozen final training checkpoint")
    dest = (args.output or run / "evaluation" / args.split / weight.stem).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    identity = dict(split=args.split, first=first, last=last, checkpoint=str(weight), checkpoint_sha256=meta["sha256"],
                    agent_steps=meta["agent_steps"], protocol_sha256=sha(protocol), policy_device="cpu",
                    initialization="fresh_official_scene_per_seed")
    manifest = dest / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError(f"Different evaluation already present in {dest}")
    write_json(manifest, identity)
    rl = load_example(run)
    verify_env = make_env(rl)
    try:
        model = rl.ppo.load(weight, env=verify_env, device="cpu")
    finally:
        verify_env.close()
    for seed in range(first, last + 1):
        trace = dest / f"seed_{seed}.npz"
        metadata = trace.with_suffix(".json")
        if trace.is_file() and metadata.is_file() and sha(trace) == json.loads(metadata.read_text())["file_sha256"]:
            continue
        # Original reset moves cameras but leaves the visual rigid bodies at their previous poses.
        # A fresh garage reproduces the author's first-episode behavior for every comparison.
        env = make_env(rl)
        try:
            data, record = episode(env, model, seed)
        finally:
            env.close()
        with trace.with_suffix(".partial").open("wb") as tmp:
            np.savez_compressed(tmp, **data)
        trace.with_suffix(".partial").replace(trace)
        write_json(metadata, dict(**record, file_sha256=sha(trace), checkpoint_sha256=meta["sha256"]))
        if (seed - first + 1) % 10 == 0 or seed == last:
            print(f"{args.split} {weight.stem} {seed-first+1}/{last-first+1}", flush=True)
    write_json(dest / "complete.json", dict(**identity, episodes=last-first+1))



if __name__ == "__main__":
    main()
