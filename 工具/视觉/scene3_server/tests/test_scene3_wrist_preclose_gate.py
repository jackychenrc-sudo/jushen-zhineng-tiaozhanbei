#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(HERE)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import scene3_wrist_preclose_gate as gate


class WristPrecloseGeometryTest(unittest.TestCase):
    def setUp(self):
        self.k = [500.0, 0.0, 100.0, 0.0, 500.0, 80.0, 0.0, 0.0, 1.0]

    def test_projection_round_trip(self):
        point = np.array([0.024, -0.012, 0.60])
        pixel = gate.project_camera_point(point, self.k)
        restored = gate.deproject_pixel(pixel, 600.0, self.k)
        np.testing.assert_allclose(point, restored, atol=1e-10)

    def test_prompted_component_follows_shifted_target(self):
        depth = np.full((160, 220), 1800.0, dtype=np.float32)
        yy, xx = np.indices(depth.shape)
        depth[(xx - 142) ** 2 + (yy - 91) ** 2 <= 24 ** 2] = 640.0
        # Foreground shelf rail has a different depth and crosses the ROI.
        depth[72:76, 90:190] = 575.0
        result = gate.prompted_depth_component(
            depth,
            prompt_uv=(142, 91),
            expected_depth_mm=640.0,
            roi_radius_px=65,
            depth_band_mm=25.0,
        )
        self.assertGreater(result["surface_pixels"], 1200)
        self.assertLess(np.linalg.norm(np.asarray(result["target_pixel"]) - [142, 91]), 2)
        self.assertAlmostEqual(640.0, result["median_depth_mm"], delta=1.0)

    def test_gate_passes_when_tray_is_between_clear_fingers(self):
        depth = np.full((160, 220), 1800.0, dtype=np.float32)
        yy, xx = np.indices(depth.shape)
        tray_mask = (xx - 100) ** 2 + (yy - 80) ** 2 <= 15 ** 2
        depth[tray_mask] = 600.0
        result = gate.evaluate_preclose_gate(
            depth,
            tray_mask,
            target_xyz=[0.0, 0.0, 0.60],
            left_tip_xyz=[-0.030, 0.0, 0.60],
            right_tip_xyz=[0.030, 0.0, 0.60],
            camera_k=self.k,
        )
        self.assertTrue(result["passed"], result)
        self.assertLessEqual(result["obstacle_ratio"], 0.20)

    def test_gate_blocks_nearer_rack_in_closing_corridor(self):
        depth = np.full((160, 220), 1800.0, dtype=np.float32)
        yy, xx = np.indices(depth.shape)
        tray_mask = (xx - 100) ** 2 + (yy - 80) ** 2 <= 5 ** 2
        depth[tray_mask] = 600.0
        # A nearer rail occupies most of the two-finger closing segment.
        depth[70:91, 70:96] = 570.0
        result = gate.evaluate_preclose_gate(
            depth,
            tray_mask,
            target_xyz=[0.0, 0.0, 0.60],
            left_tip_xyz=[-0.030, 0.0, 0.60],
            right_tip_xyz=[0.030, 0.0, 0.60],
            camera_k=self.k,
            maximum_obstacle_ratio=0.20,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["obstacle_clearance"])

    def test_gate_blocks_target_outside_fingers(self):
        depth = np.full((160, 220), 1800.0, dtype=np.float32)
        tray_mask = np.zeros_like(depth, dtype=bool)
        target = np.array([0.055, 0.0, 0.60])
        target_uv = gate.project_camera_point(target, self.k)
        yy, xx = np.indices(depth.shape)
        tray_mask = (
            (xx - float(target_uv[0])) ** 2 + (yy - float(target_uv[1])) ** 2
            <= 8 ** 2
        )
        depth[tray_mask] = 600.0
        result = gate.evaluate_preclose_gate(
            depth,
            tray_mask,
            target_xyz=target,
            left_tip_xyz=[-0.030, 0.0, 0.60],
            right_tip_xyz=[0.030, 0.0, 0.60],
            camera_k=self.k,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["target_between_fingers"])

    def test_three_stable_passes_are_required(self):
        observations = []
        for offset in (0.0, 0.001, -0.001):
            observations.append(
                {
                    "passed": True,
                    "target_xyz": [0.50 + offset, -0.20, 0.30],
                    "tcp_xyz": [0.50, -0.20 + offset, 0.30],
                }
            )
        passed, details = gate.stable_preclose_gate(observations)
        self.assertTrue(passed, details)
        observations[-1]["passed"] = False
        passed, _ = gate.stable_preclose_gate(observations)
        self.assertFalse(passed)

    def test_parser_accepts_explicit_projection_frame(self):
        args = gate.build_parser().parse_args(
            ["--camera-frame", "right_wrist_camera_link"]
        )
        self.assertEqual("right_wrist_camera_link", args.camera_frame)


if __name__ == "__main__":
    unittest.main()

