#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(HERE)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import scene3_gripper_axis_plan as axis_plan


class GripperAxisTargetTest(unittest.TestCase):
    def test_axis_standoff_is_positive_for_target_ahead(self):
        margin = axis_plan.target_axis_standoff(
            target=[0.55, -0.20, 0.26],
            midpoint=[0.50, -0.25, 0.24],
            axis=[1.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(0.05, margin)

    def test_axis_standoff_is_negative_for_target_behind(self):
        margin = axis_plan.target_axis_standoff(
            target=[0.48, -0.25, 0.24],
            midpoint=[0.50, -0.25, 0.24],
            axis=[1.0, 0.0, 0.0],
        )
        self.assertLess(margin, 0.0)

    def test_parser_accepts_fixed_wrist_target(self):
        args = axis_plan.build_parser().parse_args(["--target-source", "wrist"])
        self.assertEqual("wrist", args.target_source)
        self.assertIn("wrist_target_odom_xyz", args.wrist_target_odom_param)


if __name__ == "__main__":
    unittest.main()

