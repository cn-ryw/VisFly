"""Run the original crossing example, adding evidence and post-update checkpoints."""
import argparse
from collections import deque
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback
from zipfile import ZipFile

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from paper_runtime import array_sha, close_env, load_example, sha, write_json


class TrainingRecord(BaseCallback):
    def __init__(self, directory, budget, team_file, curve_file):
        super().__init__()
        self.directory = directory
        self.budget = budget
        self.team_file = team_file
        self.curve_file = curve_file
        self.rows = csv.writer(curve_file)
        self.rows.writerow(["agent_steps", "rollout", "native_success_rate", "native_reward_mean",
                            "teams_last100", "finished_teams", "successful_teams", "elapsed_s",
                            "peak_rss_mib", "torch_peak_reserved_mib"])
        self.start = time.perf_counter()
        self.recent = deque(maxlen=100)
        self.teams = 0
        self.wins = 0
        self.saved = set()

    def _on_training_start(self):
        self.size = self.model.n_envs * self.model.n_steps
        self.final = ((self.budget + self.size - 1) // self.size) * self.size
        self.fixed = {0, self.final}
        for target in [1_000_000, 5_000_000]:
            if target < self.budget:
                self.fixed.add(((target + self.size - 1) // self.size) * self.size)
        env = self.model.env
        d = env.envs.dynamics
        assert self.model.n_steps == 256 and self.model.batch_size == 768 and self.model.n_epochs == 10
        assert env.num_agent_per_scene == 3 and env.max_episode_steps == 256 and self.model.seed == 42
        assert d.ctrl_dt == .02 and d.dt == .0025 and d._interval_steps == 8
        obs = self.model._last_obs
        write_json(self.directory / "effective.json", {
            "seed": self.model.seed, "scenes": env.num_scene, "agents_per_scene":3,
            "n_steps":256, "batch_size":768, "n_epochs":10, "learning_rate":self.model.lr_schedule(1.0),
            "physics_dt":d.dt, "control_dt":d.ctrl_dt, "substeps":d._interval_steps,
            "budget":self.budget, "actual_target":self.final, "post_update_checkpoints":sorted(self.fixed),
            "scene":str(Path("datasets/spy_datasets/configs/garage_crossing").resolve()),
            "observations":{key:dict(shape=list(value.shape),sha256=array_sha(value)) for key,value in obs.items()},
            "torch_version":torch.__version__, "device":str(self.model.device),
            "initial_optimizer_entries":len(self.model.policy.optimizer.state)})
        self.save_checkpoint("PPO_0", 0)

    def _on_step(self):
        dones = np.asarray(self.locals["dones"]).reshape(-1, 3)
        if not np.array_equal(dones[:, 0], dones.all(1)) or not np.array_equal(dones[:, 0], dones.any(1)):
            raise RuntimeError("Asynchronous termination inside one three-drone team")
        infos = self.locals["infos"]
        for idx in np.where(dones[:, 0])[0]:
            group = infos[3 * idx:3 * idx + 3]
            flags = [bool(v["is_success"]) for v in group]
            assert flags.count(flags[0]) == 3
            self.teams += 1
            self.wins += flags[0]
            self.recent.append(flags[0])
            row = {"agent_steps":self.model.num_timesteps, "scene":int(idx), "length":int(group[0]["episode"]["l"]),
                   "team_success":flags[0], "time_limit":bool(group[0].get("TimeLimit.truncated")),
                   "team_mean_return":float(np.mean([v["episode"]["r"] for v in group]))}
            self.team_file.write(json.dumps(row, allow_nan=False) + "\n")
        return True

    def _on_rollout_end(self):
        m = self.model
        steps = m.num_timesteps
        rate = float(np.mean(m.ep_success_buffer)) if m.ep_success_buffer else 0.0
        reward = float(np.mean([x["r"] for x in m.ep_info_buffer])) if m.ep_info_buffer else None
        recent = float(np.mean(self.recent)) if self.recent else None
        elapsed = time.perf_counter() - self.start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        gpu = torch.cuda.max_memory_reserved()/1024**2
        self.rows.writerow([steps, steps//self.size, rate, reward, recent, self.teams, self.wins,
                            round(elapsed,3),round(rss,1),round(gpu,1)])
        self.curve_file.flush()
        self.team_file.flush()
        write_json(self.directory / "progress.json", dict(agent_steps=steps,stage="rollout_collected",
                   rollouts=steps//self.size,completed_gradient_epochs=m._n_updates,native_success_rate=rate,
                   finished_teams=self.teams,successful_teams=self.wins,elapsed_s=round(elapsed,2),
                   peak_rss_mib=round(rss,1),torch_peak_reserved_mib=round(gpu,1)))

    def save_checkpoint(self, name, steps):
        m = self.model
        assert m._n_updates == steps // self.size * m.n_epochs
        if not all(torch.isfinite(p).all().item() for p in m.policy.parameters()):
            raise FloatingPointError(f"Non-finite policy parameters at {steps}")
        out = self.directory / (name + ".zip")
        temp = self.directory / (name + ".partial.zip")
        BaseAlgorithm.save(m, temp)
        with ZipFile(temp) as z:
            assert z.testzip() is None and {"policy.pth","policy.optimizer.pth","data"} <= set(z.namelist())
        temp.replace(out)
        write_json(out.with_suffix(".json"), dict(agent_steps=steps,gradient_epochs=m._n_updates,
                   finished_teams=self.teams, successful_teams=self.wins, elapsed_s=round(time.perf_counter()-self.start,3),
                   sha256=sha(out),checkpoint=str(out)))
        print("CHECKPOINT",name,steps,flush=True)

    def _on_rollout_start(self):
        m=self.model
        steps=m.num_timesteps
        if not steps:
            return
        if steps in self.fixed and steps not in self.saved:
            self.save_checkpoint(f"PPO_{steps}",steps)
            self.saved.add(steps)
        if (steps//self.size) % 25 == 0:
            self.save_checkpoint("rolling",steps)
        if steps//self.size in (1,2):
            gpu_total=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip()
            write_json(self.directory / f"capacity_{steps//self.size}.json",
                       dict(agent_steps=steps,gradient_epochs=m._n_updates,elapsed_s=time.perf_counter()-self.start,
                            process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                            board_used_mib=int(gpu_total),torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2))

    def _on_training_end(self):
        if self.model.num_timesteps != self.final:
            raise RuntimeError(f"Training interrupted at {self.model.num_timesteps}, target {self.final}")
        self.save_checkpoint(f"PPO_{self.final}",self.final)
        write_json(self.directory / "complete.json",dict(agent_steps=self.final,gradient_epochs=self.model._n_updates,
                   rollouts=self.final//self.size,finished_teams=self.teams,successful_teams=self.wins,
                   elapsed_s=time.perf_counter()-self.start))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--scenes",type=int,default=24)
    parser.add_argument("--budget",type=int,default=10_000_000)
    args=parser.parse_args()
    run,output=args.run.resolve(),args.output.resolve()
    output.mkdir(exist_ok=True,parents=True)
    rl=load_example(run)
    original=rl.ppo.learn
    if args.scenes!=24:
        make=rl.MultiNavigationEnv
        def resized(**kwargs):
            kwargs["num_scene"]=args.scenes
            return make(**kwargs)
        rl.MultiNavigationEnv=resized
    rl.training_params["learning_step"]=args.budget
    rl.save_folder=str(output)+os.sep
    def learn(model,total_timesteps):
        model.policy_save_path=str(output/"official_final")
        try:
            with (output/"training-teams.jsonl").open("x") as teams, (output/"training.csv").open("x",newline="") as curve:
                cb=TrainingRecord(output,int(total_timesteps),teams,curve)
                return original(model,total_timesteps,callback=cb)
        finally:
            close_env(model.env)
    rl.ppo.learn=learn
    try:
        rl.main()
    except BaseException:
        write_json(output/"failure.json",dict(traceback=traceback.format_exc()))
        raise


if __name__=="__main__":
    main()
