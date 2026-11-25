"""Independently audit actual trajectories and check every online peer stop."""
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

PREVIOUS = Path(__file__).resolve().parents[1] / "2026-11-23"
sys.path.insert(0, str(PREVIOUS))
from geometry_audit import audit_episode, build_geometry, first_peer
from summarize_evaluation import wilson


def read(path):
    return json.loads(path.read_text())


def put(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def audit(run, cohort):
    m = read(cohort/"complete.json")
    ref = run/"evaluation"/m["split"]/"baseline"/"steps_10008576"
    geom = build_geometry(run/"source/datasets/spy_datasets")
    count = Counter()
    rows, results = [], []
    for seed in range(m["first"], m["last"]+1):
        file = cohort/f"seed_{seed}.npz"
        meta = read(file.with_suffix(".json"))
        expected = read(ref/f"seed_{seed}.json")
        assert expected["initial_observations_sha256"] == meta["initial_observations_sha256"]
        assert meta["checkpoint_sha256"] == m["checkpoint_sha256"] and meta["seed"]==seed
        assert hashlib.sha256(file.read_bytes()).hexdigest() == meta["file_sha256"]
        with np.load(file, allow_pickle=False) as d:
            control, sub = d["control_xyz"], d["substep_xyz"]
            steps = meta["steps"]
            states, depth, actions = d["state"], d["depth"], d["action"]
            assert control.shape == (steps + 1, 3, 3) and sub.shape == (8 * steps + 1, 3, 3)
            assert np.array_equal(control, sub[::8]) and np.array_equal(states[:, :, :3], control)
            assert depth.shape == (steps + 1, 3, 1, 64, 64) and actions.shape == (steps, 3, 4)
            assert digest_array(sub) == meta["trajectory_sha256"]
            assert digest_array(depth) == meta["depth_sha256"] and digest_array(actions) == meta["actions_sha256"]
            swarm = np.asarray([states[0, [j for j in range(3) if j != i]] for i in range(3)])
            initial = dict(state=states[0], depth=depth[0], target=d["target"], swarm=swarm)
            assert {k: digest_array(v) for k, v in initial.items()} == meta["initial_observations_sha256"]
            native = d["native_outcome"].item()
            assert native == meta["native_outcome"]
            outcome = audit_episode(geom, control, sub, native)
            sub64 = sub.astype(np.float64)
            pairs = list(itertools.combinations(range(3),2))
            start_min = min(float(np.linalg.norm(sub64[0,i]-sub64[0,j])) for i,j in pairs)
            assert start_min>=.20 and abs(start_min-meta["initial_min_peer_m"])<1e-10
            first_pairs = [(first_peer(sub64,i,j,True)[0],i,j) for i,j in pairs]
            hits = [(t,i,j) for t,i,j in first_pairs if t is not None]
            assert bool(hits) == meta["recorded_peer_stop"], (file,hits,meta)
            if hits:
                earliest=min(t for t,_,_ in hits)
                assert (earliest-1)//8+1==meta["steps"] and (earliest-1)%8+1==meta["peer_stop_substep"]
                assert list(meta["peer_stop_pair"]) in [[a+1,b+1] for t,a,b in hits if t==earliest]
                assert native=="collision" and not any(meta["final_official_timeout"])
        first = outcome["first_contact"] or {}
        objects = first.get("objects",[])
        groups = sorted({x.split("_")[0] for x in objects})
        assert outcome["strict_outcome"]!="collision" or first["step"]>0
        category=outcome["strict_outcome"]
        count[category]+=1
        rows.append(dict(seed=seed, control_steps=meta["steps"], execution_outcome=native,
                         strict_outcome=category, first_objects="+".join(objects), first_groups="+".join(groups),
                         peer_first=int("drone" in groups), common_peer_stop=int(bool(hits)),
                         first_control_step=first.get("step"), first_substep=first.get("substep"),
                         initial_min_peer_m=start_min, min_peer_center_m=outcome["min_peer_m"],
                         min_wall_surface_clearance_m=outcome["min_wall_clearance_m"],
                         first_observation_sha256=meta["initial_observations_sha256"]["state"],
                         trajectory_sha256=meta["trajectory_sha256"]))
        results.append(dict(seed=seed,**outcome))
    total=len(rows)
    assert total==m["episodes"]==100
    area=Counter(row["first_groups"] for row in rows if row["first_groups"])
    nwin=count["success"]
    summary=dict(**m, counts=dict(count), execution_counts=dict(Counter(row["execution_outcome"] for row in rows)),
                 strict_successes=nwin, strict_success_wilson95=wilson(nwin,total),
                 peer_first=sum(row["peer_first"] for row in rows), any_peer_stops=sum(row["common_peer_stop"] for row in rows),
                 first_contact_groups=dict(area), minimum_start_m=min(row["initial_min_peer_m"] for row in rows),
                 minimum_wall_surface_m=min(row["min_wall_surface_clearance_m"] for row in rows),
                 mean_control_steps=float(np.mean([row["control_steps"] for row in rows])),
                 min_peer_quantiles_m={str(q):float(np.quantile([row["min_peer_center_m"] for row in rows],q)) for q in (.1,.5)},
                 paired_initial_observations=total, initial_contact_teams=0,
                 best_recording_seed=next((r["seed"] for r in rows if r["strict_outcome"]=="success"),None),
                 failure_recording_seed=next((r["seed"] for r in rows if r["strict_outcome"]!="success"),None))
    with (cohort/"episodes.csv.partial").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader();w.writerows(rows)
    (cohort/"episodes.csv.partial").replace(cohort/"episodes.csv")
    put(cohort/"geometry.json",results)
    put(cohort/"summary.json",summary)
    print(m["split"],m["arm"],m["agent_steps"],dict(count),"first peer",summary["peer_first"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("cohort",type=Path)
    args=parser.parse_args()
    audit(args.run.resolve(),args.cohort.resolve())
