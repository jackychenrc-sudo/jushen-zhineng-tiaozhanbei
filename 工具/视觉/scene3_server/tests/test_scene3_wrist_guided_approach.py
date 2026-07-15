#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(HERE)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import scene3_wrist_guided_approach as approach


class WristGuidedApproachTest(unittest.TestCase):
    def test_senior_segment_is_limited_to_six_centimetres(self):
        target, step, distance = approach.plan_bounded_senior_approach(
            [0.46, -0.22, 0.27],
            [0.57, -0.22, 0.27],
            maximum_step_m=0.06,
        )
        self.assertAlmostEqual(0.11, distance, places=9)
        self.assertAlmostEqual(0.06, step, places=9)
        np.testing.assert_allclose([0.52, -0.22, 0.27], target, atol=1e-9)

    def test_short_senior_segment_is_not_overshot(self):
        target, step, _ = approach.plan_bounded_senior_approach(
            [0.50, -0.22, 0.27],
            [0.53, -0.22, 0.27],
            maximum_step_m=0.06,
        )
        self.assertAlmostEqual(0.03, step, places=9)
        np.testing.assert_allclose([0.53, -0.22, 0.27], target, atol=1e-9)

    def test_observed_progress_requires_reduced_error_and_bounded_motion(self):
        passed, checks = approach.validate_observed_progress(0.151, 0.096, 0.058)
        self.assertTrue(passed, checks)
        passed, checks = approach.validate_observed_progress(0.151, 0.145, 0.058)
        self.assertFalse(passed)
        self.assertFalse(checks["error_reduced"])
        passed, checks = approach.validate_observed_progress(0.151, 0.090, 0.11)
        self.assertFalse(passed)
        self.assertFalse(checks["motion_bounded"])


if __name__ == "__main__":
    unittest.main()

