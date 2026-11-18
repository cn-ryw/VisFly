"""Calibrate against seed 42 and replay the 100 paired development seeds with VisFly Test."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from replay_paths import REPO, archive_root, write_json

BUDGETS = (500736, 1000704)
SEEDS = range(1000, 1100)
BRANCHES = {"state", "depth", "target", "swarm"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_protocol(baseline, output):
    results = read(baseline / "evaluation/comparison.json")["budgets"]
    training = read(baseline / "train/training-config.json")
    weights = {}
    for budget in BUDGETS:
        row = next(item for item in results if item["agent_transitions"] == budget)
        checkpoint = Path("train") / f"ppo_{budget}.zip"
        assert sha(baseline / checkpoint) == row["checkpoint_sha256"], checkpoint
        weights[budget] = {"path": str(checkpoint), "sha256": row["checkpoint_sha256"]}

    sources = dict(training["active_local_source_sha256"])
    for previous, digest in training["yaml_sha256"].items():
        parts = Path(previous).parts
        relative = Path(*parts[parts.index("exps"):])
        sources[str(relative)] = digest
    for relative, digest in sources.items():
        assert sha(REPO / relative) == digest, relative
    local = Path(__file__).resolve().parent
    inputs = dict(weights=weights, budgets=BUDGETS, seeds=list(SEEDS), sources=sources,
                  initial_hash_fields=sorted(BRANCHES), deterministic=True, calibration_seed=42,
                  replay_files={name: sha(local / name) for name in ("run_pairs.py", "paired_collector.py", "replay_paths.py")})
    protocol = output / "protocol.json"
    if protocol.exists():
        assert read(protocol) == json.loads(json.dumps(inputs)), "Run settings differ from the saved protocol"
    else:
        write_json(protocol, inputs)
    return weights


def run_episode(base, baseline, budget, seed, weight):
    from paired_collector import collect_episode

    folder = base / f"steps_{budget}" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result, trace = folder / "episode.json", folder / "trajectory.npz"
    if result.exists():
        ep = read(result)
        if (ep["checkpoint_sha256"] == weight["sha256"] and ep["seed"] == seed
                and ep["agent_transitions"] == budget and trace.is_file()
                and sha(trace) == ep["trajectory_sha256"]):
            return ep
    start = time.perf_counter()
    ep = collect_episode(baseline / weight["path"], budget, seed, folder)
    assert set(ep["initial_observation_sha256"]) == BRANCHES
    ep.update(agent_transitions=budget, checkpoint_sha256=weight["sha256"], trajectory_sha256=sha(trace),
              trajectory=str(trace.relative_to(base)), runtime_seconds=round(time.perf_counter()-start, 6))
    write_json(result, ep)
    return ep


def calibration(baseline, output, weights):
    cases = []
    expectations = ("success", 51, [False]*3), ("collision", 41, [False, False, True])
    for budget, (end, length, flags) in zip(BUDGETS, expectations):
        base = output / "calibration"
        ep = run_episode(base, baseline, budget, 42, weights[budget])
        original = baseline / "evaluation" / f"steps_{budget}" / "seed_42"
        old = read(original / "episode.json")
        assert ep["end"] == old["end"] == end and ep["episode_length"] == old["episode_length"] == length
        assert ep["final_collision_flags"] == old["final_collision_flags"] == flags
        assert ep["initial_observation_sha256"] == old["initial_observation_sha256"]
        with np.load(base / ep["trajectory"]) as now, np.load(original / "trajectory.npz") as ref:
            assert now.files == ref.files
            for key in now.files:
                np.testing.assert_array_equal(now[key], ref[key])
            fields = sorted(now.files)
        cases.append(dict(agent_transitions=budget, seed=42, end=end, episode_length=length,
                          final_collision_flags=flags, exact_trace_fields=fields,
                          exact_initial_hashes=sorted(BRANCHES)))
    write_json(output / "calibration/verification.json", {"passed": True, "episodes": cases})
    print("Seed 42 calibration passed.")


def evaluate(baseline, output, calibration_only):
    import torch

    torch.set_num_threads(1)
    weights = freeze_protocol(baseline, output)
    calibration(baseline, output, weights)
    if calibration_only:
        return
    started = time.perf_counter()
    results = {b: [] for b in BUDGETS}
    for seed in SEEDS:
        for b in BUDGETS:
            results[b].append(run_episode(output, baseline, b, seed, weights[b]))
        a, b = (results[budget][-1] for budget in BUDGETS)
        assert a["initial_observation_sha256"] == b["initial_observation_sha256"], seed
        assert a["initial_positions"] == b["initial_positions"], seed
        print("PAIRED", seed, a["end"], b["end"], flush=True)
    summaries = []
    for b, episodes in results.items():
        assert [e["seed"] for e in episodes] == list(SEEDS)
        assert len({e["initial_observation_sha256"]["state"] for e in episodes}) == len(SEEDS)
        counts = {name: sum(e["end"] == name for e in episodes) for name in ("success", "collision", "timeout")}
        assert sum(counts.values()) == len(SEEDS)
        summary = dict(agent_transitions=b, checkpoint=weights[b]["path"], checkpoint_sha256=weights[b]["sha256"],
                       seeds=list(SEEDS), deterministic=True, evaluator="VisFly.exps.examples.test.navigation.test.Test",
                       counts=counts, mean_episode_length=float(np.mean([e["episode_length"] for e in episodes])))
        write_json(output / f"steps_{b}" / "summary.json", dict(summary, episodes=episodes))
        summaries.append(summary)
    write_json(output / "comparison.json",
               dict(seeds=list(SEEDS), paired_starts=len(SEEDS), initial_observations_identical_across_checkpoints=True,
                    initial_observation_fields=sorted(BRANCHES), budgets=summaries,
                    verified_calibration_episodes=2, formal_episodes=len(BUDGETS)*len(SEEDS),
                    pairing_and_evaluation_elapsed_s_this_invocation=round(time.perf_counter()-started, 6)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-run", required=True, type=Path, help="New run directory, or resume the same protocol")
    parser.add_argument("--calibration-only", action="store_true", help="Reproduce only the two historical seed-42 episodes")
    args = parser.parse_args()
    output = args.output_run.resolve() / "evaluation"
    baseline = archive_root() / "crossing-corrected-20261118"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate(baseline, output, args.calibration_only)


if __name__ == "__main__":
    main()
