#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sys
import unittest


SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import scene3_senior_chain_dry_run as chain


class FakeSeniorTask(object):
    @staticmethod
    def build_coarse_recognition_arm_target(_self, message):
        return [0.30, max(-0.30, min(-0.10, message.point.y)), -0.12]

    @staticmethod
    def build_scene3_grasp_targets(_self, message):
        point = message.point
        return {
            "pregrasp": [point.x - 0.16, point.y, point.z + 0.02],
            "touch": [point.x - 0.05, point.y, point.z + 0.02],
            "grasp": [point.x - 0.01, point.y, point.z + 0.02],
            "lift": [point.x - 0.16, point.y, point.z + 0.10],
            "retreat": [point.x - 0.20, point.y, point.z + 0.10],
        }


class FakeSenior(object):
    Scene3Task = FakeSeniorTask


class SeniorChainDryRunTests(unittest.TestCase):
    def assertVectorAlmostEqual(self, expected, actual, places=8):
        self.assertEqual(len(expected), len(actual))
        for expected_value, actual_value in zip(expected, actual):
            self.assertAlmostEqual(expected_value, actual_value, places=places)

    def test_builds_original_stage_order_and_offsets(self):
        stages = chain.build_stage_targets(FakeSenior, [0.6073, -0.1841, 0.2547])
        self.assertEqual(
            ["coarse_standby", "pregrasp", "touch", "grasp", "lift", "retreat"],
            [name for name, _target in stages],
        )
        self.assertVectorAlmostEqual([0.30, -0.1841, -0.12], stages[0][1])
        self.assertVectorAlmostEqual([0.4473, -0.1841, 0.2747], stages[1][1])
        self.assertVectorAlmostEqual([0.5973, -0.1841, 0.2747], stages[3][1])

    def test_quaternion_error_is_sign_invariant(self):
        quaternion = [0.1, -0.2, 0.3, 0.9]
        negative = [-value for value in quaternion]
        self.assertAlmostEqual(
            0.0,
            chain.quaternion_angle_degrees(quaternion, negative),
            places=8,
        )

    def test_joint_delta_keeps_left_and_right_order(self):
        before = [0.0] * 14
        after = [math.radians(value) for value in range(14)]
        delta = chain.joint_delta_degrees(before, after)
        self.assertVectorAlmostEqual(
            [float(value) for value in range(14)], delta
        )

    def test_assessment_separates_position_from_orientation_failure(self):
        record = {
            "position_error_m": 0.001,
            "orientation_error_deg": 172.0,
            "maximum_right_delta_deg": 5.0,
            "maximum_left_delta_deg": 1.0,
        }
        ok, checks = chain.assess_records([record])
        self.assertFalse(ok)
        self.assertTrue(checks["all_positions_reached"])
        self.assertFalse(checks["all_orientations_reached"])
        self.assertTrue(checks["right_steps_bounded"])
        self.assertTrue(checks["left_arm_nearly_held"])


if __name__ == "__main__":
    unittest.main()
