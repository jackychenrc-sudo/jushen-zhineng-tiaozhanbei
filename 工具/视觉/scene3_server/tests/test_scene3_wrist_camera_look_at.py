#!/usr/bin/env python3

import importlib.util
import math
import pathlib
import unittest

import numpy as np


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scene3_wrist_camera_look_at.py"
)
SPEC = importlib.util.spec_from_file_location(
    "scene3_wrist_camera_look_at", MODULE_PATH
)
LOOK_AT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOOK_AT)


class WristCameraLookAtTest(unittest.TestCase):
    def test_quaternion_matrix_roundtrip(self):
        source = np.array([-0.08, -0.15, 0.86, 0.48], dtype=float)
        source /= np.linalg.norm(source)
        recovered = LOOK_AT.matrix_to_quaternion(
            LOOK_AT.quaternion_to_matrix(source)
        )
        self.assertAlmostEqual(abs(float(np.dot(source, recovered))), 1.0, places=6)

    def test_alignment_step_is_capped_at_eight_degrees(self):
        camera = np.eye(3)
        target = [math.sin(math.radians(30.0)), 0.0,
                  math.cos(math.radians(30.0))]
        delta, angle, step = LOOK_AT.optical_alignment_delta(camera, target)
        self.assertAlmostEqual(math.degrees(angle), 30.0, places=5)
        self.assertAlmostEqual(math.degrees(step), 8.0, places=5)
        new_forward = np.matmul(delta, camera)[:, 2]
        remaining = math.degrees(math.acos(np.clip(np.dot(
            new_forward, LOOK_AT.normalize_vector(target)
        ), -1.0, 1.0)))
        self.assertAlmostEqual(remaining, 22.0, places=5)

    def test_plan_applies_same_world_delta_to_eef(self):
        camera = np.eye(3)
        target = [0.2, 0.1, 1.0]
        quaternion, _, step = LOOK_AT.plan_eef_orientation(
            [0.0, 0.0, 0.0, 1.0], camera, target
        )
        expected_delta, _, _ = LOOK_AT.optical_alignment_delta(camera, target)
        np.testing.assert_allclose(
            LOOK_AT.quaternion_to_matrix(quaternion), expected_delta, atol=1e-7
        )
        self.assertLessEqual(math.degrees(step), 8.0)

    def test_stable_target_uses_median(self):
        target, spread = LOOK_AT.stable_target([
            [0.600, -0.200, 0.250],
            [0.602, -0.199, 0.251],
            [0.599, -0.201, 0.249],
        ])
        np.testing.assert_allclose(target, [0.600, -0.200, 0.250])
        self.assertLess(spread, 0.005)

    def test_unstable_target_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "unstable"):
            LOOK_AT.stable_target([
                [0.600, -0.200, 0.250],
                [0.650, -0.200, 0.250],
                [0.580, -0.200, 0.250],
            ])

    def test_improved_angle_with_small_translation_passes(self):
        ok, checks, before, after = LOOK_AT.validate_look_at(
            math.radians(28.0), math.radians(22.0), 0.010
        )
        self.assertTrue(ok)
        self.assertTrue(all(checks.values()))
        self.assertAlmostEqual(before, 28.0)
        self.assertAlmostEqual(after, 22.0)

    def test_wrong_angle_direction_is_blocked(self):
        ok, checks, _, _ = LOOK_AT.validate_look_at(
            math.radians(28.0), math.radians(30.0), 0.010
        )
        self.assertFalse(ok)
        self.assertFalse(checks["optical_error_reduced"])


if __name__ == "__main__":
    unittest.main()
