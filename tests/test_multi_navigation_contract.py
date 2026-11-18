"""Real-renderer regression checks for the multi-agent SB3 environment.

Run from VisFly's parent directory:
python -m unittest discover -s VisFly/tests -p test_multi_navigation_contract.py -v
"""

from pathlib import Path
import unittest

import numpy as np
import torch as th

from VisFly.exps.examples.run import ENV_ALIAS, resolve_config
from VisFly.utils.common import load_yaml_config


class MultiNavigationContractTests(unittest.TestCase):
    def make_env(self, num_scene=1, max_steps=2, render=False):
        root = Path(__file__).resolve().parents[1]
        config = resolve_config(load_yaml_config(str(
            root / "exps/examples/env_cfgs/crossing.yaml"
        ))["eval_env" if render else "env"])
        config.update(num_scene=num_scene, max_episode_steps=max_steps, seed=42)
        config["random_kwargs"]["state_generator"]["kwargs"] = [
            {"position": {"mean": [1., y, 1.5], "half": [0., 0., 0.]}}
            for _ in range(num_scene) for y in (-1., 0., 1.)
        ]
        if render:
            config["scene_kwargs"]["render_settings"]["resolution"] = [96, 128]
        env = ENV_ALIAS["crossing"](**config)
        self.addCleanup(env.close)
        env.reset()
        return env

    @staticmethod
    def action(env):
        return np.tile(np.array([-1/3, .02, 0., 0.], np.float32), (env.num_envs, 1))

    def assert_current_observation(self, env, obs):
        np.testing.assert_array_equal(obs["state"], env.state.numpy())
        np.testing.assert_array_equal(obs["depth"], env.sensor_obs["depth"])
        for start in range(0, env.num_envs, 3):
            for i in range(start, start + 3):
                others = [j for j in range(start, start + 3) if j != i]
                np.testing.assert_array_equal(obs["swarm"][i], obs["state"][others])

    @staticmethod
    def set_positions(env, positions):
        env.envs.reset_agents(
            state=(th.tensor(positions, dtype=th.float32),
                   th.tensor([[1., 0., 0., 0.]]).repeat(env.num_envs, 1),
                   th.zeros((env.num_envs, 3)), th.zeros((env.num_envs, 3))),
            pos_reset_by_state=True,
        )

    def test_partial_reset_calls_the_existing_parent_method(self):
        env = self.make_env()
        env.step(self.action(env), is_test=True)
        env.reset_by_id(th.arange(3))
        np.testing.assert_array_equal(env._step_count.numpy(), [0, 0, 0])
        np.testing.assert_allclose(env.position.numpy(), [[1., -1., 1.5], [1., 0., 1.5], [1., 1., 1.5]])
        np.testing.assert_allclose(env.velocity.numpy(), 0)

    def test_timeout_returns_reset_observations_and_preserves_terminal_state(self):
        env = self.make_env()
        self.set_positions(env, [[2., -1., 2.5], [2., 0., 2.5], [2., 1., 2.5]])
        _, _, done, _ = env.step(self.action(env))
        self.assertFalse(done.any())
        obs, reward, done, infos = env.step(self.action(env))
        self.assertTrue(done.all())
        self.assertTrue(np.isfinite(reward).all())
        self.assert_current_observation(env, obs)
        np.testing.assert_array_equal(env._step_count.numpy(), [0, 0, 0])
        for i, info in enumerate(infos):
            self.assertTrue(info["TimeLimit.truncated"])
            self.assertEqual(info["episode"]["l"], 2)
            self.assertFalse(np.array_equal(
                info["terminal_observation"]["state"], obs["state"][i],
            ))
        env.step(self.action(env))
        np.testing.assert_array_equal(env._step_count.numpy(), [1, 1, 1])

    def test_terminal_depth_is_not_mutated_by_a_later_reset(self):
        env = self.make_env(max_steps=1)
        self.set_positions(env, [[2., -1., 2.5], [2., 0., 2.5], [2., 1., 2.5]])
        _, _, done, infos = env.step(self.action(env), is_test=True)
        self.assertTrue(done.all())
        depths = [info["terminal_observation"]["depth"].copy() for info in infos]
        env.reset_by_id(th.arange(3))
        self.assertTrue(any(not np.array_equal(depths[i], env.sensor_obs["depth"][i])
                            for i in range(3)))
        for info, depth in zip(infos, depths):
            np.testing.assert_array_equal(info["terminal_observation"]["depth"], depth)

    def test_collision_at_time_limit_resets_only_its_scene_without_bootstrap(self):
        env = self.make_env(num_scene=2)
        self.set_positions(env, [[1., 0., 1.5], [1., .05, 1.5], [1., 2., 1.5],
                                 [1., -1., 1.5], [1., 0., 1.5], [1., 1., 1.5]])
        self.assertTrue(env.is_collision[:2].all())
        env._step_count[:3] = 1
        obs, _, done, infos = env.step(self.action(env))
        np.testing.assert_array_equal(done, [True, True, True, False, False, False])
        self.assertTrue(all(not info["TimeLimit.truncated"] for info in infos[:3]))
        self.assertTrue(all("episode" not in info for info in infos[3:]))
        np.testing.assert_array_equal(env._step_count.numpy(), [0, 0, 0, 1, 1, 1])
        # Dynamics preserves its existing random wind-time phase on partial reset.
        np.testing.assert_allclose(env.t.numpy()[3:], [.03, .03, .03])
        self.assert_current_observation(env, obs)
        np.testing.assert_allclose(infos[0]["terminal_observation"]["swarm"][0, :3],
                                   infos[1]["terminal_observation"]["state"][:3])

    def test_success_at_time_limit_is_a_termination(self):
        env = self.make_env(max_steps=1)
        self.set_positions(env, [[11., -1., 1.5], [11., 0., 1.5], [11., 1., 1.5]])
        _, _, done, infos = env.step(self.action(env), is_test=True)
        self.assertTrue(done.all())
        self.assertTrue(all(info["is_success"] for info in infos))
        self.assertTrue(all(not info["TimeLimit.truncated"] for info in infos))
        np.testing.assert_array_equal(env._step_count.numpy(), [1, 1, 1])

    def test_reset_clears_a_previous_success(self):
        env = self.make_env(max_steps=1)
        self.set_positions(env, [[11., -1., 1.5], [11., 0., 1.5], [11., 1., 1.5]])
        env.step(self.action(env), is_test=True)
        self.assertTrue(env._success.all())
        env.reset()
        self.assertFalse(env._success.any())

    def test_public_render_accepts_the_existing_render_keywords(self):
        env = self.make_env(render=True)
        for frames in (env.render(), env.render(is_draw_axes=True)):
            self.assertEqual(np.asarray(frames).shape, (1, 96, 128, 4))
            self.assertGreater(np.ptp(frames[0][..., :3]), 0)

    def test_official_evaluation_reset_keyword_is_supported(self):
        env = self.make_env()
        self.assert_current_observation(env, env.reset(is_test=True))

    def test_official_evaluation_can_read_extended_state(self):
        env = self.make_env()
        np.testing.assert_array_equal(env.extend_state.numpy(), env.envs.extend_state.numpy())


if __name__ == "__main__":
    unittest.main()
