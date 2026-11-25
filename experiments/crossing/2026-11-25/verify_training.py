"""Verify the two safe-start PPO runs from their archived weights and training logs."""
import argparse
import base64
from collections import Counter
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-11-23"))
from paper_runtime import array_sha, sha, write_json
from verify_run import read, training, weights


ARMS = ("control", "peer")
SCENES, AGENTS, SUBSTEPS = 24, 72, 8
ROLLOUTS, EPISODE_LIMIT = 543, 256
ROLLOUT_SIZE = AGENTS * EPISODE_LIMIT
MINIMUM_M = 0.20
MEMBERS = ("policy.pth", "policy.optimizer.pth")
OBS_SHAPES = {"state": (72, 13), "depth": (72, 1, 64, 64), "target": (72, 3), "swarm": (72, 2, 13)}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def file_records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def initial_state(directory):
    path = directory / "PPO_0.zip"
    metadata = read(path.with_suffix(".json"))
    effective = read(directory / "effective.json")
    condition = read(directory / "condition.json")
    digest = sha(path)
    check(digest == metadata["sha256"], f"{path}: archive hash mismatch")
    with ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
        check(data["num_timesteps"] == data["_n_updates"] == metadata["agent_steps"] == metadata["gradient_epochs"] == 0,
              f"{path}: checkpoint saved after training began")
        blobs = {key: archive.read(key) for key in MEMBERS}
        # These are the experiment's own SB3 checkpoints. _last_obs contains four NumPy arrays.
        obs = pickle.loads(base64.b64decode(data["_last_obs"][":serialized:"], validate=True))
    policy = torch.load(BytesIO(blobs["policy.pth"]), map_location="cpu", weights_only=True)
    optimizer = torch.load(BytesIO(blobs["policy.optimizer.pth"]), map_location="cpu", weights_only=True)
    check(not optimizer["state"] and effective["initial_optimizer_entries"] == 0, f"{path}: Adam already initialized")
    check(set(obs) == set(OBS_SHAPES) == set(effective["observations"]), f"{path}: unexpected observations")
    observations = {}
    for key, shape in OBS_SHAPES.items():
        value = obs[key]
        record = {"shape": list(value.shape), "sha256": array_sha(value)}
        check(value.shape == shape and value.dtype == np.float32 and record == effective["observations"][key],
              f"{path}: shape, dtype or raw observation hash differs for {key}")
        observations[key] = {**record, "dtype": str(value.dtype)}
    positions = obs["state"][:, :3].reshape(SCENES, 3, 3).astype(np.float64)
    i, j = np.triu_indices(3, k=1)
    minima = np.linalg.norm(positions[:, i] - positions[:, j], axis=-1).min(axis=1)
    check(np.all(minima >= MINIMUM_M) and np.array_equal(minima, condition["initial_team_minimum_m"]),
          f"{path}: archived starting coordinates do not match legal per-team minima")
    record = dict(archive_sha256=digest, members_sha256={k: sha256(v).hexdigest() for k, v in blobs.items()},
                  observations=observations, tensor_count=len(policy), optimizer_state_entries=len(optimizer["state"]),
                  min_initial_team_m=float(minima.min()))
    return record, policy, obs


def initialization(run):
    control, control_policy, control_obs = initial_state(run / "train/control")
    peer, peer_policy, peer_obs = initial_state(run / "train/peer")
    check(control_policy.keys() == peer_policy.keys() and all(
        torch.equal(value, peer_policy[key]) for key, value in control_policy.items()), "Initial policy tensors differ")
    check(control["members_sha256"] == peer["members_sha256"], "Initial policy or Adam member bytes differ")
    check(all(np.array_equal(control_obs[k], peer_obs[k]) for k in OBS_SHAPES), "Initial observations differ")
    check(read(run / "train/control/effective.json") == read(run / "train/peer/effective.json"),
          "Effective PPO configuration differs between arms")
    return dict(arms={"control": control, "peer": peer}, policy_tensors_identical=True,
                optimizer_member_identical=True, initial_observations_identical=True,
                effective_training_config_identical=True, source="Deserialized both archived PPO_0.zip files on CPU.")


def final_parity(directory, steps):
    files = [directory / f"PPO_{steps}.zip", directory / "official_final_1.zip"]
    records = {}
    for path in files:
        with ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
            check(archive.testzip() is None and data["num_timesteps"] == steps and data["_n_updates"] == ROLLOUTS * 10,
                  f"{path}: damaged or unexpected final checkpoint")
            members = {key: sha256(archive.read(key)).hexdigest() for key in MEMBERS}
        records[path.name] = dict(archive_sha256=sha(path), members_sha256=members)
    fixed, official = records.values()
    check(fixed["members_sha256"] == official["members_sha256"], f"{directory}: final policy or optimizer differs")
    return dict(files=records, policy_and_optimizer_identical=True,
                entire_archives_identical=fixed["archive_sha256"] == official["archive_sha256"])


def checkpoint(layout, steps):
    record = weights(layout, steps)
    path = layout / f"train/PPO_{steps}.zip"
    with ZipFile(path) as archive:
        optimizer = torch.load(BytesIO(archive.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
    groups = optimizer["param_groups"]
    parameter_ids = [parameter for group in groups for parameter in group["params"]]
    state = optimizer["state"]
    check(len(parameter_ids) == len(set(parameter_ids)) == 33 and
          set(state) == (set(parameter_ids) if steps else set()), f"{path}: missing or duplicate Adam parameter")
    for parameter, moments in state.items():
        first, second = moments["exp_avg"], moments["exp_avg_sq"]
        check(first.shape == second.shape and torch.isfinite(first).all() and torch.isfinite(second).all() and
              (second >= 0).all(), f"{path}: invalid Adam moments for parameter {parameter}")
    check(read(path.with_suffix(".json"))["gradient_epochs"] == record["gradient_epochs"],
          f"{path}: checkpoint metadata disagrees with optimizer progress")
    return {**record, "adam_parameters": len(parameter_ids), "finite_adam_moment_pairs": len(state)}


def safety_events(directory, curves, condition, safety):
    terminate = condition["terminate"]
    initial = condition["initial_reset_stats"]
    check(initial["accepted_teams"] == 2 * SCENES, f"{directory}: expected a constructor and an initial reset")
    check(MINIMUM_M <= initial["min_accepted_m"] <= min(condition["initial_team_minimum_m"]),
          f"{directory}: illegal initial reset statistics")
    starts, rejections, minimum = initial["accepted_teams"], initial["rejected_teams"], initial["min_accepted_m"]
    with (directory / "resets.csv").open() as stream:
        reset_rows = list(csv.DictReader(stream))
    check(len(reset_rows) == len(curves), f"{directory}: missing per-rollout reset statistics")

    contacts = {}
    contact_pairs = Counter()
    for contact in file_records(directory / "peer-contacts.jsonl"):
        key = (contact["agent_steps"], contact["scene"])
        pair = tuple(contact["pair"])
        check(terminate and key not in contacts and pair in ((1, 2), (1, 3), (2, 3)) and
              1 <= contact["substep"] <= SUBSTEPS and 0 <= contact["minimum_m"] < MINIMUM_M,
              f"{directory}: invalid or duplicate peer termination at {key}")
        contacts[key] = contact["control_step"]
        contact_pairs[pair] += 1
    expected_contacts = len(contacts)
    last_end = [0] * SCENES
    previous_key = (0, -1)
    count = peer_count = 0
    events = file_records(directory / "training-teams.jsonl")
    event = next(events, None)
    for rollout, (curve, reset) in enumerate(zip(curves, reset_rows), 1):
        end = rollout * ROLLOUT_SIZE
        check(end == int(curve["agent_steps"]) == int(reset["agent_steps"]) and int(curve["rollout"]) == rollout,
              f"{directory}: unexpected rollout boundary at {end}")
        while event is not None and event["agent_steps"] <= end:
            step, scene, length = event["agent_steps"], event["scene"], event["length"]
            key = (step, scene)
            check(key > previous_key and 0 <= scene < SCENES and step % AGENTS == 0 and 1 <= length <= EPISODE_LIMIT,
                  f"{directory}: invalid scene, length or ordering at {key}")
            check(step - last_end[scene] == length * AGENTS, f"{directory}: missing/duplicated team episode at {key}")
            contact_step = contacts.pop(key, None)
            if contact_step is not None:
                check(contact_step == length and event["team_success"] is False and event["time_limit"] is False,
                      f"{directory}: peer termination conflicts with the original team result at {key}")
                peer_count += 1
            check(event["time_limit"] == (length == EPISODE_LIMIT and contact_step is None),
                  f"{directory}: unexpected timeout flag at {key}")
            last_end[scene], previous_key = step, key
            count += 1
            event = next(events, None)
        accepted = int(reset["accepted_teams"])
        rejected = int(reset["rejected_teams"])
        new_minimum = float(reset["min_accepted_m"])
        check(count == int(curve["finished_teams"]) and accepted == starts + count and rejected >= rejections and
              MINIMUM_M <= new_minimum <= minimum and peer_count == int(reset["peer_terminals"]),
              f"{directory}: inconsistent team resets or peer totals at {end}")
        rejections, minimum = rejected, new_minimum
    check(event is None and not contacts and peer_count == expected_contacts and (peer_count > 0) == terminate,
          f"{directory}: peer events missing, unmatched, or recorded with a disabled detector")
    check(safety == dict(accepted_teams=starts + count, rejected_teams=rejections, min_accepted_m=minimum,
                         peer_terminals=peer_count, terminate=terminate), f"{directory}: inconsistent safety summary")
    return dict(**safety, initial_reset_teams=starts, matched_team_events=count,
                peer_pair_counts={f"{i}-{j}": n for (i, j), n in sorted(contact_pairs.items())},
                control_peer_events_meaning="Counts enabled peer terminations only; in-flight contacts require paired evaluation.")


def verify_arm(run, arm, protocol):
    directory = run / "train" / arm
    complete = read(directory / "complete.json")
    steps = protocol["checkpoints"]
    check(complete["rollouts"] == ROLLOUTS and complete["agent_steps"] == steps[-1] == ROLLOUTS * ROLLOUT_SIZE and
          complete["gradient_epochs"] == ROLLOUTS * 10, f"{arm}: final PPO budget or update count differs")
    effective = read(directory / "effective.json")
    check(effective["post_update_checkpoints"] == steps and effective["actual_target"] == steps[-1] and
          effective["budget"] == protocol["nominal_agent_interactions_per_arm"] and effective["seed"] == protocol["train_seed"],
          f"{arm}: effective training budget, seed or checkpoints differ from the protocol")
    condition = read(directory / "condition.json")
    check(condition["terminate"] is (arm == "peer") and condition["safe_start_minimum_m"] == MINIMUM_M and
          condition["protocol_sha256"] == sha(run / "setup/protocol.json"), f"{arm}: unexpected training condition")
    # The archived verifier expects a single arm at <run>/train. Its checks and input files are unchanged.
    with TemporaryDirectory() as temporary:
        layout = Path(temporary)
        (layout / "train").symlink_to(directory, target_is_directory=True)
        checked_weights = [checkpoint(layout, n) for n in steps]
        checked_curves = training(layout, complete)
    with (directory / "training.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    safety = safety_events(directory, curves, condition, read(directory / "safety-complete.json"))
    return dict(rollouts=ROLLOUTS, checkpoints=checked_weights, tensorboard_and_team_events=checked_curves,
                starts_and_peer_terminations=safety, final_parity=final_parity(directory, steps[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--initial-only", action="store_true", help="Verify both zero-step policies while training is running.")
    args = parser.parse_args()
    check(__debug__, "The archived weight and TensorBoard verifiers require Python assertions enabled.")
    torch.set_num_threads(1)
    run = args.run.resolve()
    initial = run / "setup/initialization-parity.json"
    write_json(initial, initialization(run))
    print(initial, flush=True)
    if not args.initial_only:
        protocol = read(run / "setup/protocol.json")
        check(protocol["checkpoints"] == [0, 1013760, 5013504, 10008576] and
              protocol["actual_agent_interactions_per_arm"] == ROLLOUTS * ROLLOUT_SIZE, "Unexpected protocol checkpoints")
        arms = {arm: verify_arm(run, arm, protocol) for arm in ARMS}
        result = run / "training-verification.json"
        write_json(result, dict(protocol_sha256=sha(run / "setup/protocol.json"), initialization_sha256=sha(initial), arms=arms))
        print(result, flush=True)


if __name__ == "__main__":
    main()
