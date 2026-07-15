#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest

import numpy as np


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scene3_senior_bootstrap_approach.py"
)
SPEC = importlib.util.spec_from_file_location(
    "scene3_senior_bootstrap_approach", MODULE_PATH
)
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


class SeniorBootstrapApproachTest(unittest.TestCase):
    def test_forward_step_is_five_centimetres(self):
        target, step = BOOTSTRAP.plan_forward_step([0.46, -0.22, 0.27])
        np.testing.assert_allclose(target, [0.51, -0.22, 0.27])
        self.assertAlmostEqual(step, 0.05)

    def test_step_is_clamped(self):
        target, step = BOOTSTRAP.plan_forward_step(
            [0.46, -0.22, 0.27], step_m=0.20
        )
        self.assertAlmostEqual(target[0], 0.51)
        self.assertAlmostEqual(step, 0.05)

    def test_workspace_blocks_excess_forward_target(self):
        with self.assertRaisesRegex(ValueError, "forward workspace"):
            BOOTSTRAP.plan_forward_step([0.58, -0.22, 0.27])

    def test_good_observed_motion_passes(self):
        ok, checks, delta, motion, error = BOOTSTRAP.validate_forward_motion(
            [0.46, -0.22, 0.27],
            [0.505, -0.219, 0.271],
            [0.51, -0.22, 0.27],
        )
        self.assertTrue(ok)
        self.assertTrue(all(checks.values()))
        self.assertGreater(delta[0], 0.04)
        self.assertLess(motion, 0.05)
        self.assertLess(error, 0.01)

    def test_wrong_direction_is_blocked(self):
        ok, checks, _, _, _ = BOOTSTRAP.validate_forward_motion(
            [0.46, -0.22, 0.27],
            [0.44, -0.22, 0.27],
            [0.51, -0.22, 0.27],
        )
        self.assertFalse(ok)
        self.assertFalse(checks["forward_progress"])

    def test_large_side_motion_is_blocked(self):
        ok, checks, _, _, _ = BOOTSTRAP.validate_forward_motion(
            [0.46, -0.22, 0.27],
            [0.50, -0.16, 0.27],
            [0.51, -0.22, 0.27],
        )
        self.assertFalse(ok)
        self.assertFalse(checks["lateral_bounded"])


if __name__ == "__main__":
    unittest.main()
