import math
import sys
import unittest
from argparse import Namespace
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import scene3_lateral_alignment as alignment


def settings(**overrides):
    values = {
        "target_odom_y": 0.0,
        "lateral_tolerance": 0.01,
        "max_lateral_pulse": 0.02,
        "lateral_speed": 0.02,
        "maximum_yaw_deg": 3.0,
        "odom_frame": "odom",
        "base_frame": "base_link",
        "execute": False,
    }
    values.update(overrides)
    return Namespace(**values)


class LateralAlignmentTests(unittest.TestCase):
    def test_execution_is_disabled_after_server_trial(self):
        with self.assertRaisesRegex(ValueError, "lateral execution is disabled"):
            alignment.validate_execution(settings(execute=True))

    def test_right_drift_plans_bounded_left_pulse(self):
        plan = alignment.build_plan(
            settings(), 0.3839, -0.1539, math.radians(-0.042)
        )

        self.assertEqual("left", plan["planned_direction"])
        self.assertAlmostEqual(0.02, plan["planned_base_lateral_pulse_m"])
        self.assertAlmostEqual(1.0, plan["planned_duration_s"])
        self.assertGreater(plan["expected_odom_y_after_m"], -0.1539)
        self.assertFalse(plan["safety"]["forward_motion_commanded"])

    def test_left_drift_plans_right_pulse(self):
        plan = alignment.build_plan(settings(), 0.2, 0.08, 0.0)

        self.assertEqual("right", plan["planned_direction"])
        self.assertAlmostEqual(-0.02, plan["planned_base_lateral_pulse_m"])

    def test_position_inside_tolerance_requires_no_motion(self):
        plan = alignment.build_plan(settings(), 0.2, -0.006, 0.0)

        self.assertEqual("none", plan["planned_direction"])
        self.assertEqual(0.0, plan["planned_duration_s"])

    def test_large_yaw_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lateral-only correction is unsafe"):
            alignment.build_plan(settings(), 0.2, -0.08, math.radians(4.0))

    def test_single_pulse_remains_bounded(self):
        plan = alignment.build_plan(
            settings(max_lateral_pulse=0.015), 0.2, -0.30, 0.0
        )

        self.assertAlmostEqual(0.015, plan["planned_base_lateral_pulse_m"])


if __name__ == "__main__":
    unittest.main()
