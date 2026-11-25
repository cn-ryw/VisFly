"""Geometry tests and opt-in tests against the November 23 pinned garage/renderer."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from peer_env import MIN_CENTER_M, PeerEpisode, SafeReset, install, swept_contacts, team_minimum


def state(positions):
    p = torch.as_tensor(positions, dtype=torch.float32)
    return p, torch.tensor([[1., 0., 0., 0.]]).repeat(len(p), 1), torch.zeros_like(p), torch.zeros_like(p)


class GeometryTests(unittest.TestCase):
    def test_oblique_tangency_at_team_termination(self):
        # Compare four independent scenes at equality, below, above, and inside
        # the historical audit's 1e-12 m numerical equality tolerance.
        offset_m = np.array([0., -2e-12, 2e-12, -0.5e-12])
        team = np.array([[.12, .16, 0.], [0., 0., 0.], [0., 2., 0.]])
        starts = np.repeat(team[None], len(offset_m), axis=0)
        starts[:, 0] *= (1. + offset_m / .20)[:, None]
        env = object.__new__(PeerEpisode)
        env.num_scene = len(starts)
        for motion in (0., .25):
            with self.subTest(motion=motion):
                starts[:, 0, 2] = -motion
                ends = starts.copy()
                ends[:, 0, 2] = motion
                env._set_contacts(swept_contacts([starts, ends]))
                self.assertEqual(env.last_peer_contact.tolist(), [False, True, False, False])
                self.assertEqual(env.last_peer_substep.tolist(), [0, 1, 0, 0])
                self.assertEqual(env.last_peer_pair[1].tolist(), [1, 2])
                np.testing.assert_allclose(env.last_peer_min_m, .20 + offset_m, atol=1e-16, rtol=0)

    def test_strict_threshold_and_distinct_scenes(self):
        team = [[0., 0., 0.], [.20, 0., 0.], [0., 2., 0.]]
        poses = np.array([team, team])
        d2 = swept_contacts(np.stack((poses, poses)))
        self.assertFalse((d2 < MIN_CENTER_M ** 2).any())
        poses[0, 1, 0] = .1999
        d2 = swept_contacts(np.stack((poses, poses)))
        self.assertEqual((d2 < MIN_CENTER_M ** 2).tolist(), [[[True, False, False], [False] * 3]])

    def test_swept_crossing_and_tangent(self):
        start = np.array([[[-.3, 0., 0.], [0., 0., 0.], [0., 2., 0.]]])
        end = start.copy()
        end[0, 0, 0] = .3
        self.assertGreater(team_minimum(start)[0], .20)
        self.assertGreater(team_minimum(end)[0], .20)
        self.assertEqual(swept_contacts([start, end])[0, 0, 0], 0.)
        start[0, 0, 1] = end[0, 0, 1] = .20
        self.assertGreaterEqual(swept_contacts([start, end])[0, 0, 0], MIN_CENTER_M ** 2)

    def test_joint_resampling_and_explicit_state(self):
        # The original generator supplies four state tensors for every requested aircraft.
        class Original:
            def __init__(self):
                self.dynamics = SimpleNamespace(num=6, position=torch.zeros(6, 3))
                self.calls = []

            def _generate_state(self, ids):
                self.calls.append(ids.tolist())
                ys = [0., .01, 1., 0., 1., 2.] if len(ids) == 6 else [.5, 1., 2.]
                return state([[0., y, 2.] for y in ys])

            def reset_agents(self, indices=None, state=None):
                target = slice(None) if indices is None else indices
                self.dynamics.position[target] = state[0]

        class Safe(SafeReset, Original):
            pass

        env = Safe()
        env.reset_agents()
        self.assertEqual(env.calls, [[0, 1, 2, 3, 4, 5], [0, 1, 2]])
        np.testing.assert_array_equal(env.dynamics.position[:, 1], [.5, 1., 2., 0., 1., 2.])
        self.assertEqual((env.accepted_teams, env.rejected_teams, env.min_accepted_m), (2, 1, .5))
        old = env.dynamics.position.clone()
        with self.assertRaisesRegex(ValueError, "overlaps"):
            env.reset_agents(indices=[1], state=state([[0., .51, 2.]]))
        np.testing.assert_array_equal(env.dynamics.position, old)
        env.reset_agents(indices=[1], state=state([[0., .75, 2.]]))
        np.testing.assert_array_equal(env.dynamics.position[[0, 2, 3, 4, 5]], old[[0, 2, 3, 4, 5]])
        self.assertEqual(env.min_accepted_m, .25)


@unittest.skipUnless(os.environ.get("VISFLY_PAPER_RUN"), "set VISFLY_PAPER_RUN for the pinned garage")
class GarageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from paper_runtime import load_example
        cls.rl = load_example(Path(os.environ["VISFLY_PAPER_RUN"]).resolve())

    def test_matched_starts_and_partial_resets(self):
        from paper_runtime import close_env, make_env, reset_seed
        records = []
        for terminate in (False, True):
            install(self.rl, terminate)
            env = make_env(self.rl, scenes=2)
            try:
                starts = []
                for seed in (1037, 1072, 1097, 1037):
                    reset_seed(env, seed)
                    p = env.position.detach().numpy().reshape(2, 3, 3).copy()
                    self.assertTrue((team_minimum(p) >= .20).all())
                    starts.append(p)
                np.testing.assert_array_equal(starts[0], starts[-1])
                env.reset_by_id(torch.arange(3))
                np.testing.assert_array_equal(env.position[3:], starts[-1][1])
                self.assertGreaterEqual(team_minimum(env.position[:3]), .20)
                stats = env.spawn_stats()
                self.assertEqual(stats["accepted_teams"], 11)
                self.assertGreater(stats["rejected_teams"], 0)
                self.assertGreaterEqual(stats["min_accepted_m"], .20)
                records.append((starts, stats))
            finally:
                close_env(env)
        np.testing.assert_array_equal(records[0][0], records[1][0])
        self.assertEqual(records[0][1], records[1][1])

    def run_crossing(self, terminate, initial_steps, auto_reset):
        from paper_runtime import close_env, make_env, reset_seed
        from utils.maths import Integrator
        install(self.rl, terminate)
        env = make_env(self.rl, scenes=2)
        try:
            reset_seed(env, 42)
            s = state([[10.5, 0., 2.], [10.5, .24, 2.], [10.5, 1.3, 2.],
                       [2., 0., 2.], [2., 1., 2.], [2., -1., 2.]])
            s[2][:2, 1] = torch.tensor([2., -2.])
            env.envs.reset_agents(state=s)
            env._step_count[:3] = initial_steps
            physics = [env.position.detach().numpy().copy()]
            integrate = Integrator.integrate
            original_observation = env.get_observation
            frames = []

            def record_integrator(**kwargs):
                result = integrate(**kwargs)
                physics.append(result[0].T.detach().numpy().copy())
                return result

            def record_observation():
                obs = original_observation()
                frames.append({key: value.copy() for key, value in obs.items()})
                return obs

            with patch.object(Integrator, "integrate", staticmethod(record_integrator)), \
                    patch.object(env, "get_observation", record_observation):
                obs, reward, done, infos = env.step(torch.zeros(6, 4), is_test=not auto_reset)
            self.assertEqual(len(physics), 9)
            np.testing.assert_array_equal(done, [True, True, True, False, False, False])
            for i in range(3):
                self.assertEqual(infos[i]["peer_contact"], terminate)
                self.assertEqual(infos[i]["is_success"], not terminate)
                self.assertEqual(infos[i]["TimeLimit.truncated"], initial_steps == 255 and not terminate)
                for key in frames[0]:
                    # The historical training path aliases the reset depth buffer. Preserve its
                    # timeout bootstrap in both arms. The evaluator uses is_test=True.
                    expected = obs[key][i] if auto_reset and key == "depth" else frames[0][key][i]
                    np.testing.assert_array_equal(infos[i]["terminal_observation"][key], expected)
            if auto_reset:
                self.assertFalse(np.array_equal(obs["depth"][:3], frames[0]["depth"][:3]))
                self.assertGreaterEqual(team_minimum(env.position[:3]), .20)
                self.assertEqual(env._step_count.tolist(), [0, 0, 0, 1, 1, 1])
            if terminate:
                np.testing.assert_array_equal(env.last_peer_contact, [True, False])
                distances = swept_contacts(np.array(physics).reshape(9, 2, 3, 3))
                expected = int(np.flatnonzero(distances[:, 0, 0] < .20 ** 2)[0] + 1)
                self.assertEqual(expected, 4)
                self.assertEqual(env.last_peer_substep.tolist(), [expected, 0])
                self.assertEqual(infos[0]["peer_substep"], expected)
                self.assertEqual(infos[0]["peer_pair"], [1, 2])
                self.assertAlmostEqual(infos[0]["peer_min_m"], .16, places=6)
                if not auto_reset:
                    self.assertEqual(env.is_collision.tolist(), [True, True, False, False, False, False])
            return reward, env.velocity.norm(dim=1).numpy().copy()
        finally:
            close_env(env)

    def test_priority_collision_over_success_and_timeout_with_auto_reset(self):
        self.run_crossing(True, initial_steps=255, auto_reset=True)
        self.run_crossing(False, initial_steps=255, auto_reset=True)

    def test_only_success_bonus_changes_in_evaluation_step(self):
        before, speed = self.run_crossing(False, initial_steps=5, auto_reset=False)
        after, _ = self.run_crossing(True, initial_steps=5, auto_reset=False)
        expected = (256 - 6) * .1 * (.5 + .5 / (1. + speed[:3]))
        np.testing.assert_allclose(before[:3] - after[:3], expected, rtol=1.e-6)
        np.testing.assert_array_equal(before[3:], after[3:])


if __name__ == "__main__":
    unittest.main()
