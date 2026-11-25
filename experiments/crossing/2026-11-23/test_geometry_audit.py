"""CPU regressions against the physical garage meshes and two trajectory conventions."""
import os
from pathlib import Path
import unittest

import numpy as np
import vtk

from geometry_audit import StageQuery, audit_episode, build_geometry


def crossing():
    path = np.empty((9, 3, 3), dtype=float)
    path[:, :, 0] = np.linspace(1., 11., 9)[:, None]
    path[:, :, 1] = [-.35, 0., .35]
    path[:, :, 2] = 1.5
    return path


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vtk.vtkObject.GlobalWarningDisplayOff()
        assets = Path(__file__).resolve().parents[3] / "datasets/visfly-beta"
        cls.geometry = build_geometry(Path(os.environ.get("VISFLY_AUDIT_ASSETS", assets)))

    def run_path(self, sub, outcome="success"):
        return audit_episode(self.geometry, sub[::8], sub, outcome)

    def test_safe_gap_and_final_goal(self):
        sub = crossing()
        result = self.run_path(sub)
        self.assertTrue(result["through_gap"])
        self.assertTrue(result["strict_success"])
        self.assertTrue(result["legacy_success"])
        self.assertIsNone(result["first_contact"])
        self.assertAlmostEqual(result["min_peer_m"], .35)
        self.assertAlmostEqual(result["min_wall_clearance_m"], .15115, places=4)
        sub[-1, 0, 0] = 10.
        result = self.run_path(sub, "timeout")
        self.assertTrue(result["through_gap"])
        self.assertFalse(result["all_past_goal"])
        self.assertEqual(result["strict_outcome"], "timeout")

    def test_ctrl_chord_tunnels_through_wall(self):
        sub = crossing()
        sub[:, 2, 1] = 1.1
        result = self.run_path(sub)
        self.assertEqual(result["native_outcome"], "success")
        self.assertEqual(result["legacy_outcome"], "collision")
        self.assertEqual(result["strict_outcome"], "collision")
        self.assertEqual(result["first_contact"]["objects"], ["wall_positive_y"])
        self.assertAlmostEqual(result["min_wall_clearance_m"], -.10)
        self.assertFalse(result["through_gap"])

    def test_wall_excursion_hidden_between_control_endpoints(self):
        sub = crossing()
        sub[5:7, 2, 1] = 1.0
        result = self.run_path(sub)
        self.assertTrue(result["legacy_success"])
        self.assertFalse(result["strict_success"])
        self.assertEqual(result["first_contact"]["objects"], ["wall_positive_y"])
        self.assertEqual(result["first_contact"]["substep"], 6)

    def test_two_finite_bodies_and_open_contact_threshold(self):
        sub = crossing()
        sub[:, :, 1] = [-.20, 0., .4]
        tangent = self.run_path(sub)
        self.assertAlmostEqual(tangent["min_peer_m"], .20)
        self.assertTrue(tangent["strict_success"])
        sub[:, 0, 1] = -.1999
        overlap = self.run_path(sub)
        self.assertTrue(overlap["legacy_success"])
        self.assertEqual(overlap["first_contact"]["objects"], ["drone_1_2"])
        self.assertEqual(overlap["first_contact"]["step"], 0)
        self.assertAlmostEqual(overlap["min_peer_m"], .1999)
        sub[:, 0, 1] = -.0999
        self.assertEqual(self.run_path(sub)["legacy_first_contact"]["objects"], ["drone_1_2"])

    def test_relative_sweep_finds_mid_substep_peer_collision(self):
        sub = crossing()
        sub[2, :, 1] = [-.26, .26, .55]
        sub[3, :, 1] = [.26, -.26, .55]
        initial_separation = np.linalg.norm(sub[:, 0] - sub[:, 1], axis=1)
        self.assertGreater(initial_separation.min(), .2)
        result = self.run_path(sub)
        self.assertTrue(result["legacy_success"])
        self.assertEqual(result["first_contact"]["objects"], ["drone_1_2"])
        self.assertEqual(result["first_contact"]["substep"], 3)
        self.assertAlmostEqual(result["min_peer_m"], 0.)

    def test_swept_floor_and_hangar_side(self):
        sub = crossing()
        sub[:, :, 0] = 3.
        sub[:, :, 1] = [0., 1., 2.]
        sub[:, 0, 2] = .5
        sub[4, 0, 2] = -.5
        ground = self.run_path(sub, "timeout")
        self.assertIsNone(ground["legacy_first_contact"])
        self.assertIn("stage_Plane", ground["first_contact"]["objects"])
        self.assertEqual(ground["first_contact"]["substep"], 4)
        sub[:, :, 2] = 2.2
        sub[:, 0, 1] = 6.5
        sub[4, 0, 1] = 12.
        side = self.run_path(sub, "timeout")
        self.assertIsNone(side["legacy_first_contact"])
        self.assertEqual(side["first_contact"]["objects"], ["stage_AirHangar"])
        self.assertEqual(side["first_contact"]["substep"], 4)

    def test_triangle_edge_graze_without_midpoint_or_endpoint_contact(self):
        query = StageQuery(next(mesh for mesh in self.geometry.stage if mesh.name == "stage_Plane"))
        # At the floor corner (ENU x=-1, y=9), skim past outside the finite rectangle.
        # The normal distance to this corner in the horizontal plane is sqrt(2)*.045.
        t = np.array([-1., .25]) / np.sqrt(2)
        u, v = .045 + t, .045 - t
        for z, expected in [(0.06, True), (.08, False)]:
            a, b = np.column_stack((-(9.+v), np.full(2, z), 1.+u))
            for p in (a, b, (a+b)*.5):
                self.assertGreater(query.within(p, 5.), .10)
            self.assertEqual(query.segment_hits(a, b), expected)

    def test_bypass_and_native_end_are_independent(self):
        sub = crossing()
        sub[:, :, 2] = 15.
        bypass = self.run_path(sub)
        self.assertEqual(bypass["strict_outcome"], "bypass")
        self.assertFalse(bypass["strict_success"])
        self.assertFalse(bypass["through_gap"])
        safe = self.run_path(crossing(), "collision")
        self.assertEqual(safe["native_outcome"], "collision")
        self.assertTrue(safe["strict_success"])

    def test_input_alignment_and_initial_frame(self):
        sub = crossing()
        with self.assertRaisesRegex(ValueError, "substeps"):
            self.run_path(sub[:-1])
        control = sub[::8].copy()
        control[-1, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, "endpoints"):
            audit_episode(self.geometry, control, sub, "success")
        initial = sub[:1]
        result = audit_episode(self.geometry, initial, initial, "timeout")
        self.assertEqual(result["measured_substeps"], 0)
        self.assertIsNone(result["first_contact"])


if __name__ == "__main__":
    unittest.main()
