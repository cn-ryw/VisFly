"""Probe multi-scene partial resets, original terminal semantics, and repeatability."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_runtime import array_sha, close_env, load_example, make_env, reset_seed, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run.resolve()
    rl = load_example(run)
    env = make_env(rl, scenes=2)
    try:
        initial = reset_seed(env, 992)
        after = reset_seed(env, 992)
        hashes = {k: array_sha(v) for k, v in initial.items()}
        assert all(np.array_equal(initial[k], after[k]) for k in initial)
        assert np.array_equal(after["swarm"][0], after["state"][[1,2]]) and np.array_equal(after["swarm"][4], after["state"][[3,5]])
        env.step(np.zeros((6, 4), dtype=np.float32), is_test=True)
        before = {k:v.copy() for k,v in env.get_observation().items()}
        env.reset_by_id(torch.arange(3))
        reset = env.get_observation()
        isolated = {k:np.array_equal(v[3:], reset[k][3:]) for k,v in before.items()}
        assert all(isolated.values())
        assert not np.array_equal(reset["state"][:3], before["state"][:3])
        assert env._step_count.tolist() == [0,0,0,1,1,1]
        # Exercise the original auto-reset for just one team, without changing the training env.
        env._step_count[:3] = env.max_episode_steps - 1
        returned, _, done, infos = env.step(np.zeros((6,4), dtype=np.float32))
        actual = env.get_observation()
        assert done.tolist() == [True,True,True,False,False,False]
        assert all(x["TimeLimit.truncated"] for x in infos[:3])
        assert env._step_count.tolist() == [0,0,0,2,2,2]
        team0_is_terminal = np.array_equal(returned["state"][:3], np.stack([i["terminal_observation"]["state"] for i in infos[:3]]))
        team0_is_restart = np.array_equal(returned["state"][:3], actual["state"][:3])
        team1_is_current = np.array_equal(returned["state"][3:], actual["state"][3:])
        assert team0_is_terminal and not team0_is_restart and team1_is_current
        pairs = []
        for seed in (997,998):
            a,b = [json.loads((run/"smoke"/f"eval-{batch}"/f"seed_{seed}.json").read_text()) for batch in ("a","b")]
            keys = ["seed","checkpoint_sha256","steps","initial_observations_sha256","actions_sha256","depth_sha256","trajectory_sha256","native_outcome"]
            assert all(a[k] == b[k] for k in keys)
            pairs.append({k:a[k] for k in keys})
        write_json(run/"smoke/runtime-check.json",dict(repeated_initial_hashes=hashes, independent_scene_unchanged=isolated,
                   team0_returned_terminal_state=team0_is_terminal, team0_returned_new_initial_state=team0_is_restart,
                   other_team_returned_current_state=team1_is_current, split_done=done.tolist(),
                   paired_reloaded_trajectories=pairs))
    finally:
        close_env(env)


if __name__ == "__main__":
    main()
