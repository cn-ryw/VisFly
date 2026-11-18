"""Check PPO logging with the episode fields produced by VisFly environments.

Run from VisFly's parent directory:
python -m unittest discover -s VisFly/tests -p test_ppo_episode_logging.py -v
"""

from collections import deque
import csv
from pathlib import Path
import tempfile
import time
import unittest

from stable_baselines3.common.logger import configure

from VisFly.utils.algorithms.PPO import PPO


class PPOEpisodeLoggingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = PPO.__new__(PPO)
        self.model._logger = configure(self.directory.name, ["csv"])
        self.addCleanup(self.model.logger.close)
        self.model.ep_success_buffer = deque([False, True])
        self.model.start_time = time.time_ns() - 1_000_000_000
        self.model.num_timesteps = 768
        self.model._num_timesteps_at_start = 0
        self.model._episode_num = 2

    def read_metrics(self):
        with (Path(self.directory.name) / "progress.csv").open() as stream:
            return next(csv.DictReader(stream))

    def test_logs_multi_navigation_episode_without_extra_metrics(self):
        self.model.ep_info_buffer = deque([
            {"r": 1., "l": 32, "t": .96},
            {"r": 3., "l": 64, "t": 1.92},
        ])

        self.model._dump_logs(1)

        metrics = self.read_metrics()
        self.assertEqual(float(metrics["rollout/ep_rew_mean"]), 2.)
        self.assertEqual(float(metrics["rollout/ep_len_mean"]), 48.)
        self.assertEqual(float(metrics["rollout/ep_success_rate"]), .5)
        self.assertEqual(int(metrics["time/total_timesteps"]), 768)

    def test_preserves_existing_episode_extra_metrics(self):
        self.model.ep_info_buffer = deque([
            {"r": 1., "l": 32, "extra": {"collision": 0., "past_gate": 2.}},
            {"r": 3., "l": 64, "extra": {"collision": 1., "past_gate": 4.}},
        ])

        self.model._dump_logs(1)

        metrics = self.read_metrics()
        self.assertEqual(float(metrics["rollout/ep_collision_mean"]), .5)
        self.assertEqual(float(metrics["rollout/ep_past_gate_mean"]), 3.)
        self.assertEqual(float(metrics["rollout/ep_rew_mean"]), 2.)


if __name__ == "__main__":
    unittest.main()
