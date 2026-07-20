#!/usr/bin/env python3

import os
import sys
import types
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

from scene3_arm_control_audit import (  # noqa: E402
    build_control_checks,
    compare_arm_coordinates,
    extract_mpc_state_values,
    maximum_sample_spread_deg,
)


class Scene3ArmControlAuditTest(unittest.TestCase):

    def test_extracts_wrapped_mpc_state(self):
        message = types.SimpleNamespace(
            state=types.SimpleNamespace(value=[1.0, 2.0, 3.0])
        )
        np.testing.assert_allclose(
            extract_mpc_state_values(message),
            [1.0, 2.0, 3.0],
        )

    def test_matching_mpc_and_sensor_arm_coordinates(self):
        arm = np.deg2rad(np.linspace(-20.0, 20.0, 14))
        mpc = np.zeros(39)
        sensor = np.zeros(29)
        mpc[25:39] = arm
        sensor[13:27] = arm

        report = compare_arm_coordinates(mpc, sensor, waist_dof=1)

        self.assertEqual(report["mpc_start"], 25)
        self.assertLess(report["maximum_difference_deg"], 1e-9)

    def test_sample_spread_is_reported_in_degrees(self):
        samples = np.zeros((3, 14))
        samples[2, 4] = np.deg2rad(0.08)
        self.assertAlmostEqual(maximum_sample_spread_deg(samples), 0.08)

    def test_audit_blocks_the_observed_85_degree_recovery_state(self):
        report = {"maximum_difference_deg": 0.001}
        measured = np.zeros(14)
        measured[2] = 85.19
        measured[9] = 85.18

        checks = build_control_checks(
            report,
            measured,
            measured_spread_deg=0.02,
            right_modes=[2] * 7,
            right_kp=[30.0] * 7,
            reported_mode=2,
            safe_topic_active=False,
            timed_topic_active=False,
        )

        self.assertFalse(checks["third_joints_in_recovery_range"])
        self.assertFalse(all(checks.values()))

    def test_clean_reset_state_passes(self):
        report = {"maximum_difference_deg": 0.001}
        checks = build_control_checks(
            report,
            measured_arm_deg=np.zeros(14),
            measured_spread_deg=0.02,
            right_modes=[2] * 7,
            right_kp=[30.0] * 7,
            reported_mode=2,
            safe_topic_active=False,
            timed_topic_active=False,
        )
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
