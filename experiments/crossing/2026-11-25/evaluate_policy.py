"""Evaluate one frozen policy in the common safe-start, peer-stopping garage."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

PREVIOUS = Path(__file__).resolve().parents[1] / "2026-11-23"
sys.path.insert(0, str(PREVIOUS))
from evaluate_paper import episode
from paper_runtime import load_example, make_env, sha, write_json
from peer_env import install, team_minimum


def read(path):
    return json.loads(path.read_text())


def weight_path(run, reference, arm, step):
    directory = reference / "train" if arm == "baseline" else run / "train" / arm
    return directory / f"PPO_{step}.zip"


def dataset_dir(run, split, arm, step):
    return run / "evaluation" / split / arm / f"steps_{step}"


def capture(run, reference, arm, step, split):
    protocol = run / "setup/protocol.json"
    spec = read(protocol)
    if arm == "baseline" and step != spec["checkpoints"][-1]:
        raise ValueError("The historical reference is the original frozen final policy")
    if step not in spec["checkpoints"]:
        raise ValueError("Checkpoint missing from the frozen protocol")
    weight = weight_path(run, reference, arm, step)
    card = read(weight.with_suffix(".json"))
    digest = sha(weight)
    if digest != card["sha256"] or card["agent_steps"] != step:
        raise ValueError("Unexpected checkpoint identity")
    if arm == "baseline" and digest != spec["evaluation"]["reference"]["sha256"]:
        raise ValueError("Different historical reference")
    if split == "holdout":
        fixed = read(run / "selection.json")
        if fixed["protocol_sha256"] != sha(protocol) or not any(row["arm"] == arm and row["step"] == step
                  and row["sha256"] == digest for row in fixed["holdout_weights"]):
            raise ValueError("Holdout is restricted to frozen model identities")
    first, last = spec["evaluation"][split]
    dest = dataset_dir(run, split, arm, step)
    dest.mkdir(parents=True, exist_ok=True)
    identity = dict(arm=arm, agent_steps=step, split=split, first=first, last=last,
                    checkpoint=str(weight), checkpoint_sha256=digest, protocol_sha256=sha(protocol),
                    common_peer_termination=True, initial_minimum_m=.20, inference="deterministic_cpu_fresh_scene")
    manifest = dest / "manifest.json"
    if manifest.exists() and read(manifest) != identity:
        raise ValueError(f"Existing cohort differs: {dest}")
    if (dest/"complete.json").exists() and read(dest/"complete.json") == dict(**identity, episodes=last-first+1):
        return
    write_json(manifest, identity)
    rl = load_example(run)
    install(rl, terminate=True)
    env = make_env(rl)
    try:
        model = rl.ppo.load(weight, env=env, device="cpu")
    finally:
        env.close()
    for seed in range(first, last+1):
        file = dest / f"seed_{seed}.npz"
        record = file.with_suffix(".json")
        if file.exists() and record.exists() and sha(file) == read(record)["file_sha256"]:
            continue
        env = make_env(rl)
        try:
            data, meta = episode(env, model, seed)
            start = float(team_minimum(data["control_xyz"][0]))
            assert start >= .20 and env.spawn_stats()["min_accepted_m"] >= .20
            peer = bool(env.last_peer_contact[0])
            if peer:
                assert meta["native_outcome"] == "collision" and not any(meta["final_official_timeout"])
            # Existing sampler calls this field native_outcome. In these files it is
            # always the common runtime label including the new physical peer stop.
            meta.update(initial_min_peer_m=start, recorded_peer_stop=peer,
                        peer_stop_substep=int(env.last_peer_substep[0]),
                        peer_stop_pair=env.last_peer_pair[0].tolist(), end_step_min_peer_m=float(env.last_peer_min_m[0]),
                        original_collision_at_terminal=env.envs.is_collision.detach().cpu().tolist(),
                        raw_all_past_goal=bool((data["control_xyz"][-1,:,0]>10.).all()))
        finally:
            env.close()
        with file.with_suffix(".partial").open("wb") as f:
            np.savez_compressed(f, **data)
        file.with_suffix(".partial").replace(file)
        write_json(record, dict(**meta, checkpoint_sha256=digest, file_sha256=sha(file)))
        if (seed-first+1) % 25 == 0:
            print(split, arm, step, f"{seed-first+1}/{last-first+1}", flush=True)
    write_json(dest/"complete.json", dict(**identity, episodes=last-first+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("arm", choices=["control", "peer", "baseline"])
    parser.add_argument("step", type=int)
    parser.add_argument("split", choices=["development", "holdout"])
    args = parser.parse_args()
    capture(args.run.resolve(), args.reference.resolve(), args.arm, args.step, args.split)
