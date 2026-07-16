#!/usr/bin/env python3

import math
import os
import sys
import unittest


SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from scene3_6d_pose_dry_run import (  # noqa: E402
    apply_rotation_increment,
    normalize_quaternion,
    quaternion_error_degrees,
    quaternion_from_rpy_degrees,
    quaternion_multiply,
)


class SixDPoseDryRunTests(unittest.TestCase):
    def test_identity_increment_preserves_quaternion(self):
        current = normalize_quaternion([0.2, -0.3, 0.1, 0.9])
        actual = apply_rotation_increment(current, [0.0, 0.0, 0.0])
        self.assertLess(quaternion_error_degrees(current, actual), 1e-7)

    def test_quaternion_sign_is_equivalent(self):
        first = normalize_quaternion([0.2, -0.3, 0.1, 0.9])
        second = [-value for value in first]
        self.assertLess(quaternion_error_degrees(first, second), 1e-7)

    def test_ninety_degree_roll(self):
        roll = quaternion_from_rpy_degrees(90.0, 0.0, 0.0)
        expected = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
        self.assertLess(quaternion_error_degrees(expected, roll), 1e-7)

    def test_local_and_base_increments_differ_for_rotated_pose(self):
        current = quaternion_from_rpy_degrees(20.0, 30.0, 40.0)
        delta = quaternion_from_rpy_degrees(5.0, -3.0, 7.0)
        local = quaternion_multiply(current, delta)
        base = quaternion_multiply(delta, current)
        self.assertGreater(quaternion_error_degrees(local, base), 0.1)


if __name__ == "__main__":
    unittest.main()

