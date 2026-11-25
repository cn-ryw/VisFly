"""Audit one completed cohort with the independently calibrated physical geometry."""
import argparse
from collections import Counter
import csv
import json
from math import sqrt
from pathlib import Path
import statistics

import numpy as np

from geometry_audit import audit_episode, build_geometry


def wilson(successes, n):
    z = 1.959963984540054
    center = (successes + z*z/2) / (n + z*z)
    half = z * sqrt(successes*(n-successes)/n + z*z/4) / (n + z*z)
    return [max(0., center-half), min(1., center+half)]


def contact_group(name):
    return name.split("_")[0] if name else None


def summarize(run, cohort):
    identity = json.loads((cohort/"complete.json").read_text())
    geometry = build_geometry(run/"source/datasets/spy_datasets")
    rows, details = [], []
    paired = 0
    for seed in range(identity["first"], identity["last"]+1):
        trace = cohort / f"seed_{seed}.npz"
        meta = json.loads(trace.with_suffix(".json").read_text())
        assert meta["checkpoint_sha256"] == identity["checkpoint_sha256"] and meta["seed"] == seed
        if identity["split"] == "development" and identity["agent_steps"] != 0:
            ref = run/"evaluation/development/PPO_0"/trace.with_suffix(".json").name
            same = json.loads(ref.read_text())["initial_observations_sha256"] == meta["initial_observations_sha256"]
            if not same:
                raise ValueError(f"Initial observation differs in paired development seed {seed}")
            paired += 1
        with np.load(trace, allow_pickle=False) as d:
            assert d["native_outcome"].item() == meta["native_outcome"]
            obs = audit_episode(geometry, d["control_xyz"], d["substep_xyz"], meta["native_outcome"])
            assert obs["control_steps"] == meta["steps"] and obs["measured_substeps"] == meta["actual_physics_steps"]
            first = obs["first_contact"] or {}
            at_first = d["substep_xyz"][first["sample"]].tolist() if first else None
        reasons = ",".join(first.get("objects", []))
        groups = "+".join(sorted({contact_group(x) for x in first.get("objects", [])}))
        rows.append(dict(seed=seed, control_steps=meta["steps"], physics_steps=meta["actual_physics_steps"],
                         native_outcome=meta["native_outcome"], legacy_outcome=obs["legacy_outcome"],
                         strict_outcome=obs["strict_outcome"], first_contact_objects=reasons, first_contact_group=groups,
                         first_contact_step=first.get("step"), first_contact_substep=first.get("substep"),
                         first_contact_drones=",".join(map(str, first.get("drones", []))),
                         min_peer_center_m=obs["min_peer_m"], min_gap_wall_surface_clearance_m=obs["min_wall_clearance_m"],
                         three_through_gap=obs["through_gap"], three_final_x_gt_10=obs["all_past_goal"],
                         native_terminal_collision_any=any(meta["final_official_collision"]),
                         native_terminal_out_of_bounds_any=any(meta["final_official_out_of_bounds"]),
                         native_terminal_timeout=all(meta["final_official_timeout"]), team_mean_return=meta["team_mean_return"],
                         checkpoint_sha256=identity["checkpoint_sha256"], trajectory_file=str(trace)))
        details.append(dict(seed=seed, first_contact_xyz=at_first, **obs))
    assert len(rows) == identity["episodes"]
    n = len(rows)
    counts = {which:dict(Counter(row[which+"_outcome"] for row in rows)) for which in ("native","legacy","strict")}
    agreements = Counter((r["native_outcome"],r["strict_outcome"]) for r in rows)
    first_objects = Counter(r["first_contact_objects"] for r in rows if r["first_contact_objects"])
    first_groups = Counter(r["first_contact_group"] for r in rows if r["first_contact_group"])
    false_wins = [r for r in rows if r["native_outcome"] == "success" and r["strict_outcome"] != "success"]
    result = dict(**identity, counts=counts,
                  strict_success_wilson95=wilson(counts["strict"].get("success",0),n),
                  mean_control_steps=statistics.mean(r["control_steps"] for r in rows),
                  mean_first_collision_step=statistics.mean(r["first_contact_step"] for r in rows if r["first_contact_objects"]) if first_objects else None,
                  native_strict_confusion={f"{a}/{b}":v for (a,b),v in sorted(agreements.items())},
                  first_contact_objects=dict(first_objects), first_contact_group=dict(first_groups),
                  native_success_strict_failure_first_group=dict(Counter(r["first_contact_group"] or r["strict_outcome"] for r in false_wins)),
                  initial_contact_seeds=[r["seed"] for r in rows if r["first_contact_step"]==0],
                  terminal_native_out_of_bounds=sum(r["native_terminal_out_of_bounds_any"] for r in rows),
                  legacy_strict_disagree=sum(r["legacy_outcome"]!=r["strict_outcome"] for r in rows),
                  paired_initial_observations_verified=paired,
                  best_recording_seed=next((r["seed"] for r in rows if r["strict_outcome"]=="success"),None),
                  failure_recording_seed=next((r["seed"] for r in rows if r["strict_outcome"]!="success"),None))
    with (cohort/"episodes.csv.partial").open("w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (cohort/"episodes.csv.partial").replace(cohort/"episodes.csv")
    for filename,value in (("geometry.json",details),("summary.json",result)):
        temp = cohort/(filename+".partial")
        temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        temp.replace(cohort/filename)
    print(identity["split"],identity["agent_steps"],counts,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("cohort", type=Path)
    args = parser.parse_args()
    summarize(args.run.resolve(), args.cohort.resolve())
