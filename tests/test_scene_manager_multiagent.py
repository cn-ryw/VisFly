"""Regression checks for multi-drone collisions and reset rendering poses.

Run from VisFly's parent directory with the installed rendering environment:
python -m unittest discover -s VisFly/tests -p test_scene_manager_multiagent.py -v
"""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import habitat_sim
import numpy as np
import torch as th

from VisFly.utils.SceneManager import SceneManager
from VisFly.utils.common import std_to_habitat


class SceneManagerMultiAgentTests(unittest.TestCase):
    def setUp(self):
        project = Path(__file__).resolve().parents[1]
        self.manager = SceneManager(
            path=str(project / "datasets/visfly-beta/configs/scenes/garage_crossing"),
            num_scene=1,
            num_agent_per_scene=3,
            multi_drone=True,
            sensor_settings=[{
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "uuid": "depth",
                "resolution": [16, 16],
            }],
            shuffle=False,
        )
        self.addCleanup(self.manager.close)
        self.manager.load_scenes()
        self.positions = th.tensor([[1., -2., 1.5], [1., 0., 1.5], [1., 2., 1.5]])
        self.orientations = th.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
        self.velocities = th.zeros((3, 3))
        self.manager.reset_agents(
            self.positions, self.orientations, std_velocities=self.velocities,
        )

    def assert_model_matches_agent(self, index):
        model = self.manager._drones[0][index].root_scene_node.transformation
        agent = self.manager.agents[0][index].scene_node.transformation
        np.testing.assert_allclose(np.array(model), np.array(agent), atol=1e-6)

    def test_close_pair_is_detected_with_a_distant_third_drone(self):
        positions = th.tensor([[1., 0., 1.5], [1., .05, 1.5], [1., 4., 1.5]])
        self.manager.reset_agents(
            positions, self.orientations, std_velocities=self.velocities,
        )

        collision_points = self.manager.get_collision_point().numpy()
        distances = np.linalg.norm(collision_points - positions.numpy(), axis=1)
        np.testing.assert_allclose(collision_points[:2], positions.numpy()[[1, 0]])
        np.testing.assert_allclose(distances[:2], [.05, .05], atol=1e-6)
        self.assertTrue(np.all(distances[:2] < self.manager.drone_radius))

    def test_full_reset_synchronizes_model_pose_before_the_first_step(self):
        for index in range(3):
            self.assert_model_matches_agent(index)
        observations = self.manager.get_observation()
        self.assertEqual(len(observations), 3)
        self.assertTrue(all(np.isfinite(obs["depth"]).all() for obs in observations))

    def test_partial_reset_updates_the_selected_model_and_preserves_other_models(self):
        self.manager.set_pose(self.positions, self.orientations, self.velocities)
        before = [np.array(drone.root_scene_node.transformation)
                  for drone in self.manager._drones[0]]
        position = th.tensor([[2., .5, 2.]])
        rotation = th.tensor([[np.sqrt(.5), 0., 0., np.sqrt(.5)]], dtype=th.float32)
        self.manager.reset_agents(
            position, rotation, indices=np.array([1]), std_velocities=th.zeros((1, 3)),
        )

        self.assert_model_matches_agent(1)
        expected_position, _ = std_to_habitat(position, None)
        np.testing.assert_allclose(
            np.array(self.manager._drones[0][1].root_scene_node.translation),
            expected_position[0], atol=1e-6,
        )
        for index in (0, 2):
            np.testing.assert_allclose(
                np.array(self.manager._drones[0][index].root_scene_node.transformation),
                before[index], atol=1e-6,
            )


class SceneManagerLifecycleTests(unittest.TestCase):
    def test_multiple_scenes_can_render_and_close(self):
        # A native context abort must remain contained in the child process.
        code = textwrap.dedent("""
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            import habitat_sim
            import numpy as np
            import torch as th
            from VisFly.utils.SceneManager import SceneManager

            manager = SceneManager(
                num_scene=2, num_agent_per_scene=3, multi_drone=True,
                sensor_settings=[{
                    "sensor_type": habitat_sim.SensorType.DEPTH,
                    "uuid": "depth", "resolution": [16, 16],
                }],
                path="VisFly/datasets/visfly-beta/configs/scenes/garage_crossing",
                shuffle=False,
            )
            manager.load_scenes()
            positions = th.tensor([[1., -2., 1.5], [1., 0., 1.5], [1., 2., 1.5]]).repeat(2, 1)
            rotations = th.tensor([[1., 0., 0., 0.]]).repeat(6, 1)
            manager.reset_agents(positions, rotations, std_velocities=th.zeros((6, 3)))
            observations = manager.get_observation()
            assert len(observations) == 6
            assert all(np.isfinite(obs["depth"]).all() for obs in observations)
            print("Rendered all scenes", flush=True)
            manager.close()
            manager.close()
            print("Closed all scenes", flush=True)
        """)
        result = subprocess.run(
            [sys.executable, "-u", "-X", "faulthandler", "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-6000:])
        self.assertIn("Rendered all scenes", result.stdout)
        self.assertIn("Closed all scenes", result.stdout)


if __name__ == "__main__":
    unittest.main()
