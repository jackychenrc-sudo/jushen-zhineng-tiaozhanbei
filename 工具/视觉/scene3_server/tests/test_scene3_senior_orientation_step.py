#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest

import numpy as np


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scene3_senior_orientation_step.py"
)
SPEC = importlib.util.spec_from_file_location(
    "scene3_senior_orientation_step", MODULE_PATH
)
STEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP)


class SeniorOrientationStepTest(unittest.TestCase):
    def test_slerp_caps_step_at_four_degrees(self):
        current = [0.0, 0.0, 0.0, 1.0]
        goal = [0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)]
        desired, total, planned = STEP.plan_orientation_step(
            current, goal, maximum_step_degrees=4.0
        )
        self.assertAlmostEqual(90.0, total, places=5)
        self.assertAlmostEqual(4.0, planned, places=5)
        self.assertAlmostEqual(
            4.0,
            STEP.quaternion_angle_degrees(current, desired),
            places=5,
        )

    def test_slerp_uses_equivalent_short_quaternion_sign(self):
        current = [0.0, 0.0, 0.0, 1.0]
        desired, total, planned = STEP.plan_orientation_step(
            current, [0.0, 0.0, 0.0, -1.0]
        )
        self.assertAlmostEqual(0.0, total, places=5)
        self.assertAlmostEqual(0.0, planned, places=5)
        self.assertAlmostEqual(1.0, abs(float(desired[3])), places=5)

    def test_plan_gate_accepts_bounded_improving_solution(self):
        ok, checks, reduction = STEP.validate_orientation_plan(
            [0.1] * 7,
            [0.2, 0.0, 1.0, -0.5, 0.4, 1.4, 0.3],
            0.010,
            2.0,
            40.0,
            37.5,
        )
        self.assertTrue(ok)
        self.assertTrue(all(checks.values()))
        self.assertAlmostEqual(2.5, reduction)

    def test_plan_gate_blocks_worsening_orientation(self):
        ok, checks, _ = STEP.validate_orientation_plan(
            [0.1] * 7,
            [0.2] * 7,
            0.010,
            2.0,
            40.0,
            41.0,
        )
        self.assertFalse(ok)
        self.assertFalse(checks["senior_orientation_improved"])

    def test_measured_gate_requires_improvement_and_bounded_translation(self):
        ok, checks, reduction = STEP.validate_measured_step(
            30.0, 27.0, 0.012
        )
        self.assertTrue(ok)
        self.assertTrue(all(checks.values()))
        self.assertAlmostEqual(3.0, reduction)

        ok, checks, _ = STEP.validate_measured_step(30.0, 31.0, 0.012)
        self.assertFalse(ok)
        self.assertFalse(checks["senior_orientation_improved"])


if __name__ == "__main__":
    unittest.main()
