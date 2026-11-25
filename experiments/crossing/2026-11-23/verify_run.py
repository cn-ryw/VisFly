"""Check archived weights, native event counts, paired observations and every evaluated trajectory."""
import argparse
from collections import Counter, deque
import csv
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from paper_runtime import array_sha, sha, write_json


def read(p):
    return json.loads(p.read_text())


def weights(run, steps):
    path=run/f"train/PPO_{steps}.zip"
    meta=read(path.with_suffix(".json"))
    assert sha(path)==meta["sha256"] and meta["agent_steps"]==steps
    with ZipFile(path) as z:
        assert z.testzip() is None
        info=json.loads(z.read("data"))
        opt=torch.load(BytesIO(z.read("policy.optimizer.pth")),map_location="cpu")["state"]
        params=torch.load(BytesIO(z.read("policy.pth")),map_location="cpu")
    assert all(torch.isfinite(t).all().item() for t in params.values())
    assert (info["n_steps"],info["n_envs"],info["batch_size"],info["n_epochs"],info["seed"])==(256,72,768,10,42)
    assert info["num_timesteps"]==steps and info["_n_updates"]==steps//18432*10
    assert {int(s["step"]) for s in opt.values()}==({steps//18432*240} if steps else set())
    return dict(agent_steps=steps,sha256=meta["sha256"],gradient_epochs=info["_n_updates"],
                adam_updates_per_parameter=steps//18432*240,optimizer_parameter_groups_with_state=len(opt),
                elapsed_seconds=meta["elapsed_s"])


def training(run, complete):
    curves=list(csv.DictReader((run/"train/training.csv").open()))
    events=EventAccumulator(str(run/"train/PPO_paper_crossing_1"),size_guidance={"scalars":0}).Reload()
    logger=events.Scalars("rollout/success_rate")
    returns=events.Scalars("rollout/ep_rew_mean")
    assert len(curves)==len(logger)==len(returns)==complete["rollouts"]
    with (run/"train/training-teams.jsonl").open() as fp:
        gen=(json.loads(line) for line in fp)
        current=next(gen,None)
        native=deque([False]*72,maxlen=100)
        teams=deque(maxlen=100)
        total=success=0
        for row,tb,tbr in zip(curves,logger,returns):
            end=int(row["agent_steps"])
            while current is not None and current["agent_steps"]<=end:
                win=current["team_success"]
                native.extend([win]*3)
                teams.append(win);total+=1;success+=win
                current=next(gen,None)
            assert end==tb.step==tbr.step
            assert abs(np.mean(native)-float(row["native_success_rate"]))<1e-8
            assert abs(np.mean(native)-tb.value)<1e-6 and abs(tbr.value-float(row["native_reward_mean"]))<2e-5
            assert abs(np.mean(teams)-float(row["teams_last100"]))<1e-8
            assert int(row["finished_teams"])==total and int(row["successful_teams"])==success
        assert current is None and total==complete["finished_teams"] and success==complete["successful_teams"]
    return dict(native_success_samples=len(logger),raw_reward_samples=len(returns),team_events=total,successful_teams=success)


def verify_cohort(run, cohort, protocol_hash):
    meta=read(cohort/"manifest.json")
    complete=read(cohort/"complete.json")
    summary=read(cohort/"summary.json")
    records=list(csv.DictReader((cohort/"episodes.csv").open()))
    detailed=read(cohort/"geometry.json")
    seeds=list(range(meta["first"],meta["last"]+1))
    assert [int(r["seed"]) for r in records]==[r["seed"] for r in detailed]==seeds
    assert len(records)==complete["episodes"]==100
    ck=Path(meta["checkpoint"])
    assert sha(ck)==meta["checkpoint_sha256"] and protocol_hash==meta["protocol_sha256"]
    assert meta["initialization"]=="fresh_official_scene_per_seed" and meta["policy_device"]=="cpu"
    initial=[]
    counts={k:Counter() for k in ("native","legacy","strict")}
    for row,geom in zip(records,detailed):
        seed=int(row["seed"]); n=int(row["control_steps"])
        path=cohort/f"seed_{seed}.npz"
        card=read(path.with_suffix(".json"))
        assert card["file_sha256"]==sha(path)
        assert row["checkpoint_sha256"]==meta["checkpoint_sha256"]==card["checkpoint_sha256"]
        with np.load(path,allow_pickle=False) as d:
            control=d["control_xyz"]; physics=d["substep_xyz"]; states=d["state"]; depth=d["depth"]
            assert control.shape==(n+1,3,3) and physics.shape==(8*n+1,3,3)
            assert depth.shape==(n+1,3,1,64,64) and np.array_equal(control,physics[::8])
            assert array_sha(physics)==card["trajectory_sha256"] and array_sha(depth)==card["depth_sha256"]
            assert array_sha(d["action"])==card["actions_sha256"] and d["action"].shape==(n,3,4)
            obs=dict(state=states[0], depth=depth[0], target=d["target"],
                     swarm=np.array([states[0,[j for j in range(3) if j!=i]] for i in range(3)]))
            assert {k:array_sha(v) for k,v in obs.items()}==card["initial_observations_sha256"]
            assert np.array_equal(states[:,:,:3],control)
            initial.append(card["initial_observations_sha256"])
            assert d["native_outcome"].item()==card["native_outcome"]==row["native_outcome"]
            assert geom["control_steps"]==n and geom["measured_substeps"]==8*n
        for k in counts:
            val=row[k+"_outcome"];counts[k][val]+=1
            assert val==geom[k+"_outcome"]
    assert {k:dict(v) for k,v in counts.items()}==summary["counts"]
    assert len({v["state"] for v in initial})==100
    return initial,dict(split=meta["split"],steps=meta["agent_steps"],episodes=len(records),strict=summary["counts"]["strict"],
                        weight_sha256=meta["checkpoint_sha256"],unique_starts=len(initial))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    run=p.parse_args().run.resolve()
    protocol=read(run/"setup/protocol.json")
    complete=read(run/"train/complete.json")
    steps=protocol["post_update_checkpoints"]
    assert complete["agent_steps"]==steps[-1]==10008576
    checked=[weights(run,n) for n in steps]
    events=training(run,complete)
    starts=results=None
    all_rows=[]
    for n in steps:
        initial,results=verify_cohort(run,run/f"evaluation/development/PPO_{n}",sha(run/"setup/protocol.json"))
        if starts is None:
            starts=initial
        else:
            assert initial==starts
        all_rows.append(results)
    hold=run/f"evaluation/held-out/PPO_{steps[-1]}"
    assert sorted(p.name for p in hold.parent.iterdir() if p.is_dir())==[hold.name]
    held,results=verify_cohort(run,hold,sha(run/"setup/protocol.json"))
    assert not {v["state"] for v in starts}.intersection(v["state"] for v in held)
    all_rows.append(results)
    curves=read(run/"figures/curve-windows.json")
    assert curves["rows"]==curves["tensorboard_rows_verified"]==complete["rollouts"]
    for w in curves["windows"]:
        assert w["meets_threshold"]==(w["actual"]>=w["required"])
    write_json(run/"verification.json",dict(archived_weights=checked,training=events,cohorts=all_rows,
               total_evaluated_teams=500,dev_paired_seeds=100,held_out_distinct_initial_states=100,
               data_integrity="All NPZ bytes, actions, depths, substeps, frame endpoints and four initial observations matched their records."))
    print("Verified four frozen weights, all original training logs, 400 development teams and 100 held-out teams.")


if __name__=="__main__":
    main()
