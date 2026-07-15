#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest

import numpy as np


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scene3_wrist_yolo_probe.py"
SPEC = importlib.util.spec_from_file_location("scene3_wrist_yolo_probe", MODULE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def candidate(xyz, confidence=0.8, bbox=(10, 20, 30, 40)):
    return {
        "target_base": np.asarray(xyz, dtype=float),
        "confidence": confidence,
        "bbox": bbox,
        "grasp_pixel": (20, 30),
    }


class WristYoloProbeTest(unittest.TestCase):
    def test_defaults_match_locked_senior_target(self):
        args = PROBE.build_parser().parse_args([])
        self.assertIn("locked_target_base", args.target_topic)
        self.assertIn("locked_target_base_xyz", args.target_param)

    def test_clamp_xyxy(self):
        self.assertEqual(
            PROBE.clamp_xyxy_to_xywh((100, 200, 3), (-5.2, 10.4, 220.0, 90.7)),
            (0, 10, 200, 81),
        )

    def test_confidence_selection_without_locked_target(self):
        selected = PROBE.choose_target_candidate([
            candidate((0.4, 0.0, 0.2), confidence=0.55),
            candidate((0.8, 0.0, 0.2), confidence=0.91),
        ])
        self.assertAlmostEqual(selected["confidence"], 0.91)

    def test_3d_matching_overrides_confidence(self):
        locked = np.array([0.60, -0.20, 0.25])
        selected = PROBE.choose_target_candidate([
            candidate((0.61, -0.20, 0.25), confidence=0.51),
            candidate((0.80, -0.20, 0.25), confidence=0.95),
        ], locked_target_base=locked)
        self.assertAlmostEqual(selected["target_base"][0], 0.61)
        self.assertLess(selected["match_distance_m"], 0.02)

    def test_distant_detection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nearest wrist tray"):
            PROBE.choose_target_candidate(
                [candidate((1.0, 0.0, 0.0))],
                locked_target_base=(0.0, 0.0, 0.0),
                maximum_match_distance_m=0.18,
            )

    def test_three_stable_frames(self):
        samples = [
            candidate((0.600, -0.200, 0.250)),
            candidate((0.603, -0.199, 0.251)),
            candidate((0.599, -0.201, 0.249)),
        ]
        stable = PROBE.stable_target_observation(samples, maximum_spread_m=0.008)
        self.assertIsNotNone(stable)
        np.testing.assert_allclose(
            stable["target_base"], [0.600, -0.200, 0.250], atol=1e-9
        )

    def test_unstable_frames_do_not_pass(self):
        samples = [
            candidate((0.600, -0.200, 0.250)),
            candidate((0.640, -0.200, 0.250)),
            candidate((0.580, -0.200, 0.250)),
        ]
        self.assertIsNone(
            PROBE.stable_target_observation(samples, maximum_spread_m=0.008)
        )


if __name__ == "__main__":
    unittest.main()
