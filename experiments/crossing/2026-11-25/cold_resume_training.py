"""Resume one November 25 arm from its immutable shutdown snapshot in a new segment."""
import argparse
from collections import deque
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys
import traceback
from zipfile import ZipFile


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(run, arm, pause):
    spec = read(run / "setup/protocol.json")
    manifest = read(pause / "manifest.json")
    saved = manifest["arms"][arm]
    folder = pause / arm
    card = read(folder / "rolling.json")
    path = folder / "rolling.zip"
    protocol_hash = sha256((run / "setup/protocol.json").read_bytes()).hexdigest()
    if protocol_hash != manifest["accepted_protocol_sha256"] or read(folder/"condition.json")["terminate"] != (arm == "peer"):
        raise ValueError("Snapshot arm or protocol mismatch")
    if sha256(path.read_bytes()).hexdigest() != card["sha256"] or card["sha256"] != saved["sha256"]:
        raise ValueError("Cold checkpoint digest mismatch")
    with ZipFile(path) as archive:
        if archive.testzip() or any(sha256(archive.read(k)).hexdigest() != h for k,h in saved["members_sha256"].items()):
            raise ValueError("Corrupt policy or Adam checkpoint")
        data = json.loads(archive.read("data"))
    start = card["agent_steps"]
    size = data["n_envs"] * data["n_steps"]
    nominal, final = spec["nominal_agent_interactions_per_arm"], spec["actual_agent_interactions_per_arm"]
    if not (data["num_timesteps"] == saved["last_disk_steps"] == start and 0 < start < nominal and
            data["_n_updates"] == card["gradient_epochs"] == start//size * data["n_epochs"] and
            (data["n_envs"], data["n_steps"], data["batch_size"], data["n_epochs"], data["seed"]) == (72, 256, 768, 10, 42)
            and start % size == 0 and (nominal+size-1)//size*size == final):
        raise ValueError("Checkpoint counters or absolute rollout targets differ from the protocol")
    with (folder/"resets.csv").open() as stream:
        previous = next(r for r in csv.DictReader(stream) if int(r["agent_steps"]) == start)
    recent = deque(maxlen=100)
    teams = wins = 0
    with (folder/"training-teams.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["agent_steps"] > start:
                break
            recent.append(row["team_success"])
            teams += 1
            wins += row["team_success"]
    if (teams, wins) != (card["finished_teams"], card["successful_teams"]):
        raise ValueError("Prefix team events differ from checkpoint card")
    return dict(arm=arm, snapshot=str(path), checkpoint_sha256=card["sha256"],
                members_sha256=saved["members_sha256"], prefix_agent_steps=start,
                prefix_gradient_epochs=card["gradient_epochs"], prefix_elapsed_s=card["elapsed_s"],
                prefix_finished_teams=teams, prefix_successful_teams=wins, prefix_recent_teams=list(recent),
                prefix_resets=dict(accepted_teams=int(previous["accepted_teams"]),
                    rejected_teams=int(previous["rejected_teams"]), min_accepted_m=float(previous["min_accepted_m"]),
                    peer_terminals=int(previous["peer_terminals"])),
                abandoned_collected_steps_after_checkpoint=saved["frozen_collected"]-start,
                continuation_seed=spec["train_seed"], nominal_target=nominal, final_target=final,
                remaining_actual_interactions=final-start, remaining_absolute_checkpoints=[i for i in spec["checkpoints"] if i>start],
                protocol_sha256=protocol_hash, mode="cold_resume_fresh_episodes_and_rng",
                counters="Team counts and elapsed time use the checkpoint prefix. resets.csv and peer-contacts.jsonl are segment-local. Steps and gradient epochs are absolute.",
                cold_boundary="Simulator, partial team episodes and RNG streams restart. Adam moments, optimizer steps and both episode monitoring windows are restored.")


def publish(src, dst):
    """Create an exclusive link on the same filesystem; never replace previous evidence."""
    try:
        os.link(src, dst)
    except FileExistsError:
        if not src.samefile(dst):
            raise


def resume(run, arm, resume_info, segment):
    # --check-only stays in the standard library and never opens a renderer or CUDA.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-11-23"))
    sys.path.insert(0, str(run / "setup/renderer-lib"))
    import numpy as np
    import torch
    from paper_runtime import close_env, load_example, make_env, write_json
    from peer_env import install
    from run_training import SafetyRecord

    root = run / "train" / arm
    start = resume_info["prefix_agent_steps"]
    final = resume_info["final_target"]
    protocol_path = run / "setup/protocol.json"

    class ColdRecord(SafetyRecord):
        protocol = protocol_path

        def save_checkpoint(self, name, steps):
            if name == "PPO_0" and steps == 0:
                return  # The inherited initialization hook is also used on a cold start.
            super().save_checkpoint(name, steps)
            if name == f"PPO_{steps}" and steps in resume_info["remaining_absolute_checkpoints"]:
                publish(segment / (name+".zip"), root / (name+".zip"))
                publish(segment / (name+".json"), root / (name+".json"))

        def _on_training_start(self):
            self.teams = resume_info["prefix_finished_teams"]
            self.wins = resume_info["prefix_successful_teams"]
            self.recent.extend(resume_info["prefix_recent_teams"])
            self.start -= resume_info["prefix_elapsed_s"]
            super()._on_training_start()
            if self.final != final or self.model._total_timesteps != resume_info["nominal_target"]:
                raise ValueError("The learning schedule must retain its original absolute horizon")

        def _on_rollout_start(self):
            if self.model.num_timesteps == start:
                # The author's learn() seeds this existing deque with 72 artificial
                # failures each time it is called. Keep the actual checkpoint window.
                self.model.ep_success_buffer.clear()
                self.model.ep_success_buffer.extend(success_window)
                return
            super()._on_rollout_start()

    segment.mkdir(parents=True, exist_ok=False)
    write_json(segment / "cold-resume.json", dict(**resume_info, segment=str(segment)))
    env = None
    try:
        rl = load_example(run)
        install(rl, terminate=(arm == "peer"))
        env = make_env(rl, scenes=24)
        model = rl.ppo.load(resume_info["snapshot"], env=env, device="cuda", force_reset=True)
        success_window = tuple(model.ep_success_buffer)
        opt = model.policy.optimizer
        expected_adam = resume_info["prefix_gradient_epochs"] * model.n_envs * model.n_steps // model.batch_size
        if (model.num_timesteps != start or model._n_updates != resume_info["prefix_gradient_epochs"] or
                not isinstance(opt, torch.optim.Adam) or not opt.state or
                len(opt.state) != sum(len(group["params"]) for group in opt.param_groups) or
                any(int(v["step"].item()) != expected_adam for v in opt.state.values())):
            raise ValueError("Loaded PPO or Adam counters do not match the published checkpoint")
        model.tensorboard_log = str(segment)
        model.save_path = str(segment)
        model.policy_save_path = str(segment / "official_final")
        seed = resume_info["continuation_seed"]
        random.seed(seed)
        np.random.seed(seed)
        env.envs.set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if model._last_obs is not None:
            raise ValueError("The historical final observation cannot start a fresh simulated episode")
        with (segment/"training-teams.jsonl").open("x") as teams, (segment/"training.csv").open("x", newline="") as curve:
            cb = ColdRecord(segment, resume_info["nominal_target"], teams, curve)
            model.learn(resume_info["nominal_target"]-start, callback=cb, reset_num_timesteps=False)
        model.save()
        if model.num_timesteps != final:
            raise ValueError("The segment stopped before the absolute final budget")
        local = read(segment/"safety-complete.json")
        before = resume_info["prefix_resets"]
        totals = {k: before[k] + local[k] for k in ("accepted_teams", "rejected_teams", "peer_terminals")}
        totals.update(min_accepted_m=min(before["min_accepted_m"],local["min_accepted_m"]), terminate=(arm=="peer"))
        write_json(segment/"safety-cumulative.json", totals)
        publish(segment/"official_final_1.zip", root/"official_final_1.zip")
        publish(segment/"cold-resume.json", root/"cold-resume.json")
        publish(segment/"safety-cumulative.json", root/"safety-complete.json")
        publish(segment/"complete.json", root/"complete.json")
    except BaseException:
        write_json(segment/"failure.json", dict(traceback=traceback.format_exc(), segment=str(segment)))
        raise
    finally:
        if env is not None:
            close_env(env)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("arm", choices=("control", "peer"))
    p.add_argument("--snapshot", type=Path, help="snapshot directory containing manifest.json and both arms")
    p.add_argument("--check-only", action="store_true", help="verify frozen artifacts on CPU without starting a simulation")
    args = p.parse_args()
    run = args.run.resolve()
    info = snapshot(run, args.arm, (args.snapshot or run/"pause-20261125-2247").resolve())
    directory = run/"train"/args.arm/"segments"/f"after_{info['prefix_agent_steps']}"
    if args.check_only:
        print(json.dumps(dict(arm=args.arm, verified_sha256=info["checkpoint_sha256"], start=info["prefix_agent_steps"],
                              remaining=info["remaining_actual_interactions"], final=info["final_target"],
                              future_checkpoints=info["remaining_absolute_checkpoints"], segment=str(directory))))
    else:
        resume(run, args.arm, info, directory)
