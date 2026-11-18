"""Verify that evaluation videos pair RGB and depth from the same step.

Run from VisFly's parent directory:
python -m unittest discover -s VisFly/tests -p test_evaluation_video_alignment.py -v
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from VisFly.utils.evaluate import TestBase


class EvaluationVideoAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        env = SimpleNamespace(
            envs=SimpleNamespace(dynamics=SimpleNamespace(dt=.03, ctrl_dt=.03)),
        )
        self.evaluation = TestBase(
            env=env, name="alignment", save_path=self.directory.name,
        )
        self.evaluation._img_names = ["depth"]
        # test() stores the initial observation, then observations at steps 1 and 2.
        self.evaluation.obs_all = [
            {"depth": np.full((1, 1, 32, 32), depth, dtype=np.float32)}
            for depth in (0., 2., 8.)
        ]
        self.evaluation.t = [0., .03, .06]
        # The third-person RGB frames begin at step 1.
        self.evaluation.render_image_all = [
            np.full((32, 32, 3), value, dtype=np.uint8)
            for value in (80, 160)
        ]

    def read_video_means(self, name):
        path = Path(self.evaluation.save_path) / name
        video = cv2.VideoCapture(str(path))
        self.addCleanup(video.release)
        self.assertTrue(video.isOpened(), f"Cannot decode {path}")
        means = []
        while True:
            ok, frame = video.read()
            if not ok:
                break
            self.assertEqual(frame.shape, (32, 32, 3))
            means.append(float(frame.mean()))
        return means

    def test_exported_rgb_and_depth_include_the_same_two_steps(self):
        self.evaluation.save_video()

        rgb_means = self.read_video_means("video.mp4")
        depth_means = self.read_video_means("depth.mp4")
        self.assertEqual(len(rgb_means), 2)
        self.assertEqual(len(depth_means), 2)
        np.testing.assert_allclose(rgb_means, [80., 160.], atol=5.)
        # save_video maps depth [0, 10] m to grayscale [0, 255].
        np.testing.assert_allclose(depth_means, [51., 204.], atol=5.)

    def test_playback_displays_depth_from_each_rendered_step(self):
        displayed = {"video": [], "depth": []}

        def collect_frame(winname, mat):
            displayed[winname].append(float(mat.mean()))

        with patch("VisFly.utils.evaluate.cv2.imshow", side_effect=collect_frame):
            with patch("VisFly.utils.evaluate.cv2.waitKey", return_value=-1):
                self.evaluation.play()

        np.testing.assert_allclose(displayed["video"], [80., 160.])
        np.testing.assert_allclose(displayed["depth"], [2. / 20., 8. / 20.], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
