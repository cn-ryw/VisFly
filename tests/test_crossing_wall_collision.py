import unittest
from pathlib import Path

import numpy as np
import torch as th

from VisFly.exps.examples.run import ENV_ALIAS, resolve_config
from VisFly.utils.common import load_yaml_config
from VisFly.utils.crossing_walls import segment_hits_box


class CrossingWallCollisionTests(unittest.TestCase):
    def make_env(self):
        cfg_path = Path(__file__).resolve().parents[1] / 'exps/examples/env_cfgs/crossing.yaml'
        cfg = resolve_config(load_yaml_config(str(cfg_path))['env'])
        cfg['num_scene'] = 1
        env = ENV_ALIAS['crossing'](**cfg)
        self.addCleanup(env.close)
        env.reset()
        return env

    def place(self, env, positions, velocities):
        env.envs.reset_agents(
            state=(th.tensor(positions, dtype=th.float32),
                   th.tensor([[1., 0., 0., 0.]]).repeat(3, 1),
                   th.tensor(velocities, dtype=th.float32), th.zeros(3, 3)),
            pos_reset_by_state=True,
        )

    def test_recorded_inside_wall_ends_team_and_reset_starts_new_segment(self):
        env = self.make_env()
        # Recorded red center before/after the unflagged convex-wall contact at step 44.
        start = np.array([7.694959, -1.062, 3.816])
        end = np.array([8.093514, -.968253, 3.918615])
        self.place(env, [start.tolist(), [4,0,1], [3,0,3]],
                   [((end-start)/.03).tolist(), [0,0,0], [0,0,0]])
        _, _, done, infos = env.step(np.zeros((3,4), np.float32), is_test=True)
        self.assertTrue(done.all())
        self.assertTrue(all(not info['is_success'] for info in infos))
        np.testing.assert_array_equal(env.is_collision, [True,False,False])
        self.assertGreater(env.collision_dis[0].item(), .1)

        # Jump to the far side on reset. A reset must never sweep the old-to-new positions.
        self.place(env, [[9,-1,2],[9,0,3],[9,1,4]], [[0,0,0]]*3)
        self.assertFalse(env.is_collision.any())
        _, _, done, _ = env.step(np.zeros((3,4), np.float32), is_test=True)
        self.assertFalse(done.any())

    def test_safe_team_through_gap(self):
        env = self.make_env()
        self.place(env, [[7.6,-.3,2],[7.6,0,3],[7.6,.3,4]], [[18,0,0]]*3)
        for _ in range(10):
            _, _, done, infos = env.step(np.zeros((3,4),np.float32), is_test=True)
            self.assertFalse(env.is_collision.any())
            if done.all():
                break
        self.assertTrue(done.all())
        self.assertTrue(all(info['is_success'] for info in infos))

    def test_collision_on_goal_step_denies_success_bonus(self):
        env = self.make_env()
        self.place(env, [[10.2,0,2],[10.2,.04,2],[10.2,1,3]], [[0,0,0]]*3)
        _, reward, done, infos = env.step(np.zeros((3,4),np.float32), is_test=True)
        self.assertTrue(done.all())
        self.assertTrue(all(not info['is_success'] for info in infos))
        self.assertTrue((reward < 1.).all())

    def test_swept_edges_use_euclidean_clearance(self):
        half = np.ones(3)
        self.assertTrue(segment_hits_box(np.array([-2., 0, 0]), np.array([2., 0, 0]), half, .1))
        self.assertFalse(segment_hits_box(np.array([-2.,1.08,1.08]), np.array([2.,1.08,1.08]), half, .1))
        self.assertTrue(segment_hits_box(np.array([-2.,1.05,1.05]), np.array([2.,1.05,1.05]), half, .1))
