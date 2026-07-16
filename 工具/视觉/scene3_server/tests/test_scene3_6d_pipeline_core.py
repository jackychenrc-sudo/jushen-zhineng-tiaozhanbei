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

from scene3_6d_pipeline_core import (  # noqa: E402
    cartesian_motion_metrics,
    command_reference_degrees,
    command_target_from_ik_delta,
    compute_grasp_targets,
    eef_target_for_physical_pose,
    insertion_line_step,
    interpolate_commands,
    physical_to_eef_calibration,
    position_only_eef_step,
    sensor_command_offsets_degrees,
)


class Scene36DPipelineCoreTest(unittest.TestCase):

    def test_command_uses_low_level_absolute_and_sensor_relative_delta(self):
        command = np.arange(14, dtype=float) + 20.0
        measured = np.deg2rad(np.arange(14, dtype=float) - 40.0)
        solved = measured.copy()
        solved[7:] += np.deg2rad([1, -2, 3, -4, 5, -6, 7])

        target, delta = command_target_from_ik_delta(
            command, measured, solved
        )

        np.testing.assert_allclose(target[:7], command[:7], atol=1e-12)
        np.testing.assert_allclose(
            target[7:], command[7:] + [1, -2, 3, -4, 5, -6, 7],
            atol=1e-12,
        )
        np.testing.assert_allclose(delta[:7], 0.0, atol=1e-12)

    def test_fixed_sensor_command_offset_does_not_change_zero_delta_command(self):
        command_rad = np.deg2rad(np.arange(14, dtype=float))
        measured = command_rad - np.deg2rad(37.0)
        reference = command_reference_degrees(command_rad)
        target, delta = command_target_from_ik_delta(
            reference, measured, measured
        )
        np.testing.assert_allclose(target, reference, atol=1e-12)
        np.testing.assert_allclose(delta, 0.0, atol=1e-12)
        np.testing.assert_allclose(
            sensor_command_offsets_degrees(measured, command_rad),
            37.0,
            atol=1e-12,
        )

    def test_regression_wrist_command_uses_delta_not_measured_absolute(self):
        # Real failure signature: command=49.423deg, measured=63.906deg.
        # Asking IK for +1deg must command 50.423deg, never 64.906deg.
        reference = np.zeros(14)
        reference[11] = 49.423
        measured = np.zeros(14)
        measured[11] = np.deg2rad(63.906)
        solved = measured.copy()
        solved[11] += np.deg2rad(1.0)
        target, delta = command_target_from_ik_delta(
            reference, measured, solved
        )
        self.assertAlmostEqual(delta[11], 1.0, places=9)
        self.assertAlmostEqual(target[11], 50.423, places=9)
        self.assertNotAlmostEqual(target[11], 64.906, places=3)

    def test_grasp_targets_share_one_straight_approach_line(self):
        result = compute_grasp_targets(
            [0.60, -0.20, 0.25],
            tray_center_to_grasp_m=0.105,
            preprocess_standoff_m=0.090,
            grasp_height_offset_m=0.015,
        )
        delta = result["final_tcp"] - result["preprocess_tcp"]
        np.testing.assert_allclose(
            delta, 0.090 * result["approach"], atol=1e-12
        )
        self.assertAlmostEqual(result["final_tcp"][2], 0.265)
        self.assertAlmostEqual(result["preprocess_tcp"][2], 0.265)

    def test_preprocess_step_does_not_construct_an_orientation_target(self):
        step = position_only_eef_step(
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            0.03,
        )
        np.testing.assert_allclose(
            step["target_eef_position"], [0.03, 0.0, 0.0]
        )
        self.assertNotIn("target_eef_rotation", step)

    def test_physical_pose_round_trip_holds_tcp_and_orientation(self):
        angle = math.radians(30.0)
        eef_rotation = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        physical_rotation = np.eye(3)
        eef_position = np.array([0.2, -0.3, 0.1])
        tcp = np.array([0.3, -0.25, 0.12])
        calibration = physical_to_eef_calibration(
            eef_position, eef_rotation, tcp, physical_rotation
        )
        desired_tcp = np.array([0.4, -0.1, 0.2])
        target = eef_target_for_physical_pose(
            desired_tcp,
            physical_rotation,
            calibration["eef_to_tcp"],
            calibration["eef_to_physical"],
        )
        recovered_tcp = (
            target["eef_position"]
            + target["eef_rotation"].dot(calibration["eef_to_tcp"])
        )
        recovered_physical = target["eef_rotation"].dot(
            calibration["eef_to_physical"]
        )
        np.testing.assert_allclose(recovered_tcp, desired_tcp, atol=1e-12)
        np.testing.assert_allclose(
            recovered_physical, physical_rotation, atol=1e-12
        )

    def test_insertion_step_returns_to_fixed_line_before_advancing(self):
        result = insertion_line_step(
            [0.01, 0.02, 0.0],
            [0.0, 0.0, 0.0],
            [0.09, 0.0, 0.0],
            0.02,
        )
        np.testing.assert_allclose(result["target_tcp"], [0.03, 0.0, 0.0])
        self.assertAlmostEqual(result["cross_track_m"], 0.02)
        self.assertAlmostEqual(result["step_m"], 0.02)

    def test_motion_metrics_separate_progress_and_cross_track(self):
        metrics = cartesian_motion_metrics(
            [0.0, 0.0, 0.0],
            [0.02, 0.003, 0.0],
            [0.02, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(metrics["progress_m"], 0.02)
        self.assertAlmostEqual(metrics["cross_track_m"], 0.003)
        self.assertAlmostEqual(metrics["target_error_m"], 0.003)

    def test_quintic_interpolation_keeps_exact_endpoints(self):
        points = interpolate_commands([0.0] * 14, [1.0] * 14, 20)
        np.testing.assert_allclose(points[0], 0.0)
        np.testing.assert_allclose(points[-1], 1.0)
        np.testing.assert_allclose(points[10], 0.5)


if __name__ == "__main__":
    unittest.main()
