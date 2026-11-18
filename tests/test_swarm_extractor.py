"""CPU regression checks for the official crossing policy feature extractor."""

from copy import deepcopy
from pathlib import Path
import unittest

from gymnasium import spaces
import numpy as np
import torch as th

from VisFly.utils.common import load_yaml_config
from VisFly.utils.policies.extractors import SwarmStateTargetImageExtractor


class SwarmExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_num_threads = th.get_num_threads()
        th.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        th.set_num_threads(cls.previous_num_threads)

    def setUp(self):
        th.manual_seed(42)
        root = Path(__file__).resolve().parents[1]
        config = load_yaml_config(str(root / "exps/examples/alg_cfgs/crossing/PPO.yaml"))
        self.net_arch = config["algorithm"]["policy_kwargs"]["features_extractor_kwargs"]["net_arch"]
        observation_space = spaces.Dict({
            "depth": spaces.Box(0., np.inf, shape=(1, 64, 64), dtype=np.float32),
            "state": spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32),
            "target": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "swarm": spaces.Box(-np.inf, np.inf, shape=(2, 13), dtype=np.float32),
        })
        self.extractor = SwarmStateTargetImageExtractor(
            observation_space, net_arch=deepcopy(self.net_arch),
        )

    @staticmethod
    def observations(batch_size):
        return {
            "depth": th.rand(batch_size, 1, 64, 64),
            "state": th.randn(batch_size, 13),
            "target": th.randn(batch_size, 3),
            "swarm": th.randn(batch_size, 2, 13),
        }

    def test_official_crossing_features_match_the_declared_dimension(self):
        self.assertEqual(self.extractor.features_dim, 384)
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                features = self.extractor(self.observations(batch_size))
                self.assertEqual(features.shape, (batch_size, self.extractor.features_dim))
                self.assertTrue(th.isfinite(features).all())

    def test_both_neighbors_keep_separate_ordered_feature_blocks(self):
        observations = self.observations(3)
        neighbor_width = self.net_arch["swarm"]["layer"][-1]
        baseline = self.extractor(observations)
        for neighbor in range(2):
            with self.subTest(neighbor=neighbor):
                changed = {key: value.clone() for key, value in observations.items()}
                changed["swarm"][:, neighbor] += 3.
                features = self.extractor(changed)
                offset = self.extractor.features_dim - 2 * neighbor_width
                own = slice(offset + neighbor * neighbor_width,
                            offset + (neighbor + 1) * neighbor_width)
                other = 1 - neighbor
                other_block = slice(offset + other * neighbor_width,
                                    offset + (other + 1) * neighbor_width)
                self.assertFalse(th.allclose(features[:, own], baseline[:, own]))
                th.testing.assert_close(features[:, other_block], baseline[:, other_block])
                th.testing.assert_close(features[:, :offset], baseline[:, :offset])

    def test_backward_reaches_both_neighbors(self):
        # Keep this CPU gradient check focused on the shared neighbor MLP.
        for parameter in self.extractor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.extractor.swarm_extractor.parameters():
            parameter.requires_grad_(True)
        observations = self.observations(3)
        observations["swarm"].requires_grad_()
        features = self.extractor(observations)
        features.square().sum().backward()

        gradient = observations["swarm"].grad
        self.assertEqual(gradient.shape, (3, 2, 13))
        self.assertTrue(th.isfinite(gradient).all())
        self.assertTrue((gradient.abs().sum(dim=-1) > 0).all())
        for parameter in self.extractor.swarm_extractor.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(th.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
