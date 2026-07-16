#!/usr/bin/env python3

import math
import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

from scene3_gripper_6d_align_execute import post_motion_checks
from scene3_gripper_6d_align_plan import (
    choose_target_gripper_rotation,
    gripper_geometry,
    matrix_from_rotation_vector,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_error_degrees,
    target_eef_pose,
)


class Gripper6DAlignTest(unittest.TestCase):

    def test_gripper_geometry_builds_right_handed_frame(self):
        result = gripper_geometry(
            [0.0, 0.0, 0.0],
            [1.0, -0.5, 0.0],
            [1.0, 0.5, 0.0],
            tcp_extension_m=0.2,
        )
        np.testing.assert_allclose(result["forward"], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(result["gap"], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(result["up"], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(result["tcp"], [1.2, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.det(result["rotation"]), 1.0)

    def test_desired_frame_uses_horizontal_tray_direction_and_up(self):
        rotation, up_sign, error = choose_target_gripper_rotation(
            np.eye(3),
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.5],
        )
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)
        self.assertEqual(up_sign, 1.0)
        self.assertAlmostEqual(error, 0.0)

    def test_bounded_eef_rotation_holds_physical_tcp(self):
        final = matrix_from_rotation_vector([math.pi / 2.0, 0.0, 0.0])
        target = target_eef_pose(
            [0.0, 0.0, 0.0],
            np.eye(3),
            [1.0, 0.0, 0.0],
            np.eye(3),
            final,
            maximum_step_degrees=10.0,
        )
        predicted_tcp = (
            target["eef_position"]
            + target["eef_rotation"].dot(target["eef_to_tcp"])
        )
        np.testing.assert_allclose(predicted_tcp, [1.0, 0.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(target["full_angle_deg"], 90.0, places=6)
        self.assertAlmostEqual(target["step_angle_deg"], 10.0, places=6)
        before = rotation_error_degrees(final, np.eye(3))
        after = rotation_error_degrees(final, target["physical_rotation"])
        self.assertAlmostEqual(before - after, 10.0, places=6)

    def test_quaternion_matrix_round_trip(self):
        rotation = matrix_from_rotation_vector([0.2, -0.4, 0.3])
        quaternion = matrix_to_quaternion(rotation)
        recovered = quaternion_to_matrix(quaternion)
        np.testing.assert_allclose(recovered, rotation, atol=1e-9)

    def test_post_motion_gate_accepts_improved_tcp_held_step(self):
        checks, reduction = post_motion_checks(
            before_error=40.0,
            after_error=35.3,
            tcp_shift=0.004,
            final_position_error=0.003,
            final_orientation_error=1.8,
            left_drift=0.1,
            right_tracking_error=0.5,
            tray_standoff=0.09,
        )
        self.assertAlmostEqual(reduction, 4.7)
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()

