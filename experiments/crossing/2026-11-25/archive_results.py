"""Export independently checked, compact measurements to the experiment's Git directory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_records(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def csv_write(path, records):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def export(run, dest):
    protocol = run / "setup/protocol.json"
    spec = read(protocol)
    final = spec["actual_agent_interactions_per_arm"]
    selection = run / "selection.json"
    locked = read(selection)
    done = read(run / "evaluation/complete.json")
    verification = read(run / "training-verification.json")
    assert digest(protocol) == locked["protocol_sha256"] == verification["protocol_sha256"]
    assert digest(selection) == done["selection_sha256"]
    dev = [("baseline", final), ("control", 0)]
    dev += [(arm, step) for step in spec["checkpoints"][1:] for arm in ("control", "peer")]
    holdout = [(w["arm"], w["step"]) for w in locked["holdout_weights"]]
    assert len(dev) == done["development_cohorts"] and len(holdout) == done["holdout_cohorts"]
    dest.mkdir(parents=True, exist_ok=True)
    episodes, cohorts = [], []
    for split, requested in (("development", dev), ("holdout", holdout)):
        for arm, step in requested:
            cohort = run / "evaluation" / split / arm / f"steps_{step}"
            data = read(cohort / "summary.json")
            entries = csv_records(cohort / "episodes.csv")
            assert len(entries) == data["episodes"] == 100 and data["initial_contact_teams"] == 0
            assert data["paired_initial_observations"] == 100 and data["minimum_start_m"] >= .20
            assert data["protocol_sha256"] == digest(protocol)
            if split == "holdout":
                assert selection.stat().st_mtime_ns <= (cohort / "manifest.json").stat().st_mtime_ns
            info = dict(split=split, arm=arm, agent_steps=step)
            for entry in entries:
                episodes.append(dict(**info, **entry))
            counts = data["counts"]
            low, high = data["strict_success_wilson95"]
            cohorts.append(dict(
                **info, episodes=data["episodes"], strict_successes=data["strict_successes"],
                peer_first=data["peer_first"], common_peer_stops=data["any_peer_stops"],
                collisions=counts.get("collision", 0), timeout=counts.get("timeout", 0),
                bypass=counts.get("bypass", 0), incomplete=counts.get("incomplete", 0),
                minimum_start_m=data["minimum_start_m"], median_minimum_m=data["min_peer_quantiles_m"]["0.5"],
                mean_control_steps=data["mean_control_steps"], success_ci_low=low, success_ci_high=high,
                checkpoint=data["checkpoint"], weight_sha256=data["checkpoint_sha256"],
                raw_cohort=str(cohort.relative_to(run))))
    for split in ("development", "holdout"):
        reference = [row for row in episodes if row["split"] == split and row["arm"] == "baseline"]
        assert len({row["first_observation_sha256"] for row in reference}) == 100
    dev_states = {r["first_observation_sha256"] for r in episodes if r["split"] == "development"}
    test_states = {r["first_observation_sha256"] for r in episodes if r["split"] == "holdout"}
    assert dev_states.isdisjoint(test_states)
    csv_write(dest / "cohorts.csv", cohorts)
    csv_write(dest / "episodes.csv", episodes)
    for name in ("training-verification.json", "selection.json", "setup/initialization-parity.json",
                 "paired/paired-metrics.csv", "paired/comparisons.json", "figures/paired-case.json"):
        shutil.copyfile(run / name, dest / Path(name).name)
    if (run / "setup/source-parity.json").is_file():
        shutil.copyfile(run / "setup/source-parity.json", dest / "source-parity.json")
    for arm in ("control", "peer"):
        for table in ("training", "resets"):
            csv_write(dest / f"{arm}-{table}.csv", csv_records(run / "train" / arm / f"{table}.csv"))
    proof = dict(protocol_sha256=digest(protocol), evaluation=done, total_episodes=len(episodes),
                 distinct_development_starts=len(dev_states), distinct_holdout_starts=len(test_states),
                 overlapping_development_holdout_starts=len(dev_states & test_states), cohorts=cohorts)
    (dest / "summary.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(cohorts)} cohorts, {len(episodes)} team episodes, {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.run.resolve(), args.output.resolve())
