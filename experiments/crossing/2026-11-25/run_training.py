"""Launch one safe-start arm through the unchanged paper PPO runner."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PREVIOUS = Path(__file__).resolve().parents[1] / "2026-11-23"
sys.path.insert(0, str(PREVIOUS))
import train_paper
from paper_runtime import sha, write_json
from peer_env import install, team_minimum


class SafetyRecord(train_paper.TrainingRecord):
    def _on_training_start(self):
        super()._on_training_start()
        self.peer_ends = 0
        self.safety_file = (self.directory / "resets.csv").open("x", newline="")
        self.peer_file = (self.directory / "peer-contacts.jsonl").open("x")
        self.safety_log = csv.writer(self.safety_file, lineterminator="\n")
        self.safety_log.writerow(["agent_steps", "accepted_teams", "rejected_teams", "min_accepted_m", "peer_terminals"])
        env = self.model.env
        minima = team_minimum(env.position.detach().cpu().numpy().reshape(-1, 3, 3))
        assert np.all(minima >= .20)
        write_json(self.directory / "condition.json", dict(terminate=bool(env.peer_termination),
                   safe_start_minimum_m=.20, initial_team_minimum_m=minima.tolist(), initial_reset_stats=env.spawn_stats(),
                   policy_source="original examples/crossing/rl.py", protocol_sha256=sha(self.protocol)))

    def _on_step(self):
        proceed = super()._on_step()
        env = self.model.env
        if env.peer_termination:
            for scene in np.flatnonzero(env.last_peer_contact):
                first_agent = 3 * scene
                info = self.locals["infos"][first_agent]
                assert self.locals["dones"][first_agent] and info["peer_contact"]
                assert not info["is_success"] and not info["TimeLimit.truncated"]
                self.peer_ends += 1
                self.peer_file.write(json.dumps(dict(agent_steps=self.model.num_timesteps, scene=int(scene),
                     control_step=int(info["episode"]["l"]), substep=info["peer_substep"],
                     pair=info["peer_pair"], minimum_m=info["peer_min_m"]))+"\n")
        return proceed

    def _on_rollout_end(self):
        super()._on_rollout_end()
        stats = self.model.env.spawn_stats()
        assert stats["min_accepted_m"] >= .20
        self.safety_log.writerow([self.model.num_timesteps, stats["accepted_teams"], stats["rejected_teams"],
                                  stats["min_accepted_m"], self.peer_ends])
        self.safety_file.flush()
        self.peer_file.flush()

    def _on_training_end(self):
        try:
            super()._on_training_end()
            write_json(self.directory / "safety-complete.json", dict(**self.model.env.spawn_stats(),
                       peer_terminals=self.peer_ends, terminate=bool(self.model.env.peer_termination)))
        finally:
            self.safety_file.close()
            self.peer_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("arm", choices=["control", "peer"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenes", type=int, default=24)
    parser.add_argument("--budget", type=int, default=10_000_000)
    args = parser.parse_args()
    run = args.run.resolve()
    out = (args.output or run / "train" / args.arm).resolve()
    original_loader = train_paper.load_example

    def load_arm(directory):
        rl = original_loader(directory)
        install(rl, terminate=args.arm == "peer")
        return rl

    SafetyRecord.protocol = run / "setup/protocol.json"
    train_paper.load_example = load_arm
    train_paper.TrainingRecord = SafetyRecord
    sys.argv = [sys.argv[0], str(run), str(out), "--scenes", str(args.scenes), "--budget", str(args.budget)]
    train_paper.main()


if __name__ == "__main__":
    main()
