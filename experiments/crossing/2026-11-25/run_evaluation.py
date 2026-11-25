"""Measure each immutable checkpoint and lock development selections before holdout."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
ARMS = ("control", "peer")


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path, record):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def directory(run, split, arm, step):
    return run / "evaluation" / split / arm / f"steps_{step}"


def measure(args, split, arm, step):
    dest = directory(args.run, split, arm, step)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet",
               PYTHONPATH=str(args.run / "setup/renderer-lib"))
    logs = args.run / "logs"
    tag = f"{split}-{arm}-{step}"
    if not (dest / "complete.json").exists():
        print(timestamp(), "capture", tag, flush=True)
        with (logs / f"{tag}-capture.log").open("a") as log:
            subprocess.run([args.visfly_python, "-B", HERE / "evaluate_policy.py", args.run,
                            args.reference, arm, str(step), split], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    if not (dest / "summary.json").exists():
        print(timestamp(), "geometry", tag, flush=True)
        with (logs / f"{tag}-audit.log").open("a") as log:
            subprocess.run([args.geometry_python, "-B", HERE / "audit_cohort.py", args.run, dest],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = read(dest / "summary.json")
    assert summary["episodes"] == 100 and summary["paired_initial_observations"] == 100
    assert summary["arm"] == arm and summary["agent_steps"] == step and summary["split"] == split
    assert summary["protocol_sha256"] == sha256(args.run / "setup/protocol.json")
    assert summary["checkpoint_sha256"] == sha256(Path(summary["checkpoint"]))
    print(timestamp(), "ready", tag, summary["strict_successes"], summary["peer_first"], flush=True)
    return summary


def await_training(run, arm, filename):
    train = run / "train" / arm
    while not (train / filename).exists():
        if (train / "failure.json").exists():
            raise RuntimeError(f"Training reported a failure in {train}")
        time.sleep(15)
    return read(train / filename)


def freeze(args, spec, development):
    final = spec["checkpoints"][-1]
    best = {}
    policy_rows = []
    for arm in ARMS:
        rows = [development[arm, step] for step in spec["checkpoints"] if step]
        winner = max(rows, key=lambda r: (r["strict_successes"], -r["peer_first"], r["agent_steps"]))
        best[arm] = winner["agent_steps"]
        for row in rows:
            policy_rows.append({k: row[k] for k in ("arm", "agent_steps", "strict_successes",
                                                    "peer_first", "checkpoint_sha256")})

    required = [("baseline", final)]
    for arm in ARMS:
        required.extend((arm, step) for step in sorted({final, best[arm]}))
    holdout = [dict(arm=arm, step=step, sha256=development[arm, step]["checkpoint_sha256"])
               for arm, step in required]
    expected = dict(protocol_sha256=sha256(args.run / "setup/protocol.json"), best_by_arm=best,
                    development_candidates=policy_rows, holdout_weights=holdout)
    path = args.run / "selection.json"
    if path.exists():
        locked = read(path)
        if any(locked[key] != value for key, value in expected.items()):
            raise ValueError("Development results differ from the already frozen selection")
    else:
        if any((args.run / "evaluation/holdout").glob("*/*/manifest.json")):
            raise RuntimeError("Found holdout measurements without their original selection")
        publish(path, dict(frozen_utc=timestamp(), **expected))
    return holdout


def run(args):
    spec = read(args.run / "setup/protocol.json")
    steps = spec["checkpoints"]
    final = steps[-1]
    dev = {("baseline", final): measure(args, "development", "baseline", final)}
    # The two zero policies and every initial observation are compared in verify_training.py.
    for step in steps:
        for arm in (("control",) if step == 0 else ARMS):
            await_training(args.run, arm, f"PPO_{step}.json")
            dev[arm, step] = measure(args, "development", arm, step)
    for arm in ARMS:
        await_training(args.run, arm, "safety-complete.json")
        complete = await_training(args.run, arm, "complete.json")
        assert complete["agent_steps"] == spec["actual_agent_interactions_per_arm"]
    weights = freeze(args, spec, dev)
    for entry in weights:
        measure(args, "holdout", entry["arm"], entry["step"])
    publish(args.run / "evaluation/complete.json",
            dict(finished_utc=timestamp(), development_cohorts=len(dev), holdout_cohorts=len(weights),
                 selection_sha256=sha256(args.run / "selection.json")))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("reference", type=Path)
    p.add_argument("--visfly-python", required=True, type=Path)
    p.add_argument("--geometry-python", default="/usr/bin/python3", type=Path)
    options = p.parse_args()
    options.run, options.reference = options.run.resolve(), options.reference.resolve()
    (options.run / "logs").mkdir(exist_ok=True)
    with (options.run / "logs/evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(options)
