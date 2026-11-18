"""Capture VisFly's deterministic Test outputs using the original paired trajectory schema."""
import hashlib
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from replay_paths import REPO

# The official demo uses the VisFly package name and resolves scene assets from its parent.
sys.path.insert(0, str(REPO.parent))
os.chdir(REPO.parent)
from VisFly.exps.examples.run import ALG_ALIAS, ENV_ALIAS, resolve_config
from VisFly.exps.examples.test.navigation.test import Test
from VisFly.utils.common import load_yaml_config, set_seed

CAMERA = dict(mode="fix", view="custom", resolution=[480, 800],
              position=[[4.5, -6., 4.], [4.5, 0., 3.]])


class WallTrace(list):
    def __init__(self, env):
        super().__init__()
        self.scene = env.envs.sceneManager

    def append(self, values):
        super().append(dict(values, swept_wall=self.scene.swept_wall_collision.copy()))


def collect_episode(checkpoint, budget, seed, folder):
    config = resolve_config(load_yaml_config(str(REPO / "exps/examples/env_cfgs/crossing.yaml")))["eval_env"]
    config.update(num_scene=1, seed=seed)
    config["scene_kwargs"]["render_settings"].update(CAMERA)
    env = ENV_ALIAS["crossing"](**config)
    try:
        model = ALG_ALIAS["PPO"].load(str(checkpoint), env=env, device="cuda")
        assert model.num_timesteps == budget
        set_seed(seed)
        env.envs.set_seed(seed)
        evaluator = Test(model=model, name=folder.name, save_path=str(folder.parent))
        evaluator.collision_all = WallTrace(env)
        _, _, mean_reward, mean_length = evaluator.test()
        observations = evaluator.obs_all
        infos = evaluator.info_all[1:]
        collisions = evaluator.collision_all
        state = np.stack([o["state"] for o in observations])
        depth = np.stack([o["depth"] for o in observations])
        action = np.stack(evaluator.action_all)
        distance = np.stack([c["col_dis"].cpu().numpy() for c in collisions])
        point = np.stack([c["col_pt"].cpu().numpy() for c in collisions])
        collided = np.stack([c["is_col"].cpu().numpy() for c in collisions])
        # Only a done transition has "episode" metadata in the official wrapper.
        done = np.asarray([["episode" in i for i in step] for step in infos])
        truncated = np.asarray([[i["TimeLimit.truncated"] for i in step] for step in infos])
        crossed = state[:, :, 0] > 10
        n = len(action)
        assert n == mean_length and all(int(i["episode"]["l"]) == n for i in infos[-1])
        assert state.shape == (n + 1, 3, 13) and action.shape == (n, 3, 4)
        assert depth.shape == (n + 1, 3, 1, 64, 64) and depth.dtype == np.float32
        assert all(np.isfinite(x).all() for x in (state, depth, action, distance, point))
        assert done[-1].all() and not done[:-1].any()
        np.testing.assert_array_equal((crossed & ~collided)[-1], env.get_success().cpu().numpy())
        np.testing.assert_array_equal([i["is_success"] for i in infos[-1]],
                                      np.repeat(crossed[-1].all() and not collided[-1].any(), 3))
        endpoint_contact = distance < env.envs.uav_radius
        swept_wall = np.stack([c["swept_wall"] for c in collisions])
        np.testing.assert_array_equal(collided, endpoint_contact | swept_wall)
        assert not collided[1:-1].any() and not truncated[:-1].any()
        np.testing.assert_allclose(np.linalg.norm(state[:, :, :3] - point, axis=2), distance,
                                   rtol=1e-5, atol=1e-6)
        assert all(np.shares_memory(a["depth"], b["depth"]) is False
                   for a, b in zip(observations, observations[1:]))
        success = bool(crossed[-1].all() and not collided.any())
        collision = bool(collided.any())
        timeout = bool(truncated[-1].all())
        assert sum([success, collision, timeout]) == 1, "unexpected or ambiguous terminal condition"
        hashes = {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                  for key, value in observations[0].items()}
        trajectory = folder / "trajectory.npz"
        np.savez_compressed(
            folder / "trajectory.partial.npz", state=state, depth=depth, action=action,
            reward=np.stack(evaluator.reward_all), per_agent_past_goal=crossed, per_agent_success=(crossed & ~collided), done=done,
            time_limit=truncated, collision_distance=distance, collision_point=point,
            is_collision=collided, swept_wall_contact=swept_wall, endpoint_contact=endpoint_contact, target=observations[0]["target"],
            ctrl_dt=env.envs.dynamics.ctrl_dt, action_delay_steps=env.envs.dynamics._comm_delay_steps,
        )
        (folder / "trajectory.partial.npz").replace(trajectory)
        return {
            "seed": seed, "episode_length": n, "mean_reward": float(mean_reward),
            "team_success": success, "any_collision": collision, "time_limit": timeout,
            "initial_collision_flags": collided[0].tolist(),
            "first_collision_step": int(np.flatnonzero(collided.any(axis=1))[0]) if collision else None,
            "final_endpoint_flags": endpoint_contact[-1].tolist(),
            "final_swept_wall_flags": swept_wall[-1].tolist(),
            "end": "success" if success else ("collision" if collision else "timeout"),
            "terminal_seconds": round(n * env.envs.dynamics.ctrl_dt, 6),
            "per_agent_success": (crossed & ~collided)[-1].tolist(), "final_positions": state[-1, :, :3].tolist(),
            "final_collision_flags": collided[-1].tolist(),
            "final_collision_distances_m": distance[-1].tolist(),
            "final_collision_points": point[-1].tolist(), "initial_observation_sha256": hashes,
            "initial_positions": state[0, :, :3].tolist(), "trajectory": str(trajectory),
        }
    finally:
        plt.close("all")
        env.close()
