#!/usr/bin/env python3

import os
import sys
import unittest


SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from scene3_6d_pose_execute import execution_checks, quintic  # noqa: E402


class SixDPoseExecuteTests(unittest.TestCase):
    def test_quintic_has_fixed_endpoints(self):
        self.assertEqual(quintic(0.0), 0.0)
        self.assertEqual(quintic(1.0), 1.0)

    def test_nominal_retreat_passes(self):
        checks, progress, cross, motion = execution_checks(
            [-0.02, 0.0, 0.0],
            [-0.019, 0.001, -0.001],
            0.002,
            1.5,
            0.2,
            0.006,
            0.035,
        )
        self.assertTrue(all(checks.values()))
        self.assertGreater(progress, 0.018)
        self.assertLess(cross, 0.002)
        self.assertLess(motion, 0.021)

    def test_wrong_direction_is_blocked(self):
        checks, _, _, _ = execution_checks(
            [-0.02, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            0.002,
            1.0,
            0.1,
            0.006,
            0.035,
        )
        self.assertFalse(checks["forward_progress"])


if __name__ == "__main__":
    unittest.main()

