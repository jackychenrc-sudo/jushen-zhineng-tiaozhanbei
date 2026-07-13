import math
import sys
import unittest
from argparse import Namespace
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import scene3_closed_loop_velocity_motion as motion


def config():
    return Namespace(
        stop_margin=0.005,
        forward_kp=0.8,
        lateral_kp=0.5,
        yaw_kp=0.8,
        path_lookahead_m=0.20,
        lateral_deadband_m=0.005,
        yaw_deadband_deg=0.3,
        minimum_forward_speed=0.015,
        maximum_forward_speed=0.035,
        maximum_lateral_speed=0.02,
        maximum_yaw_rate_deg=2.0,
        maximum_reverse_m=0.02,
        maximum_overshoot_m=0.05,
        maximum_lateral_drift_m=0.035,
        maximum_yaw_drift_deg=4.0,
        minimum_valid_motion_m=0.015,
        post_forward_tolerance_m=0.015,
        post_lateral_tolerance_m=0.025,
        post_yaw_tolerance_deg=2.0,
    )


class ClosedLoopVelocityMotionTests(unittest.TestCase):
    def test_foot_midpoint_pose_averages_position_and_wrapped_yaw(self):
        midpoint = motion.midpoint_pose(
            [0.01, 0.07, math.radians(179.0)],
            [0.07, -0.17, math.radians(-179.0)],
        )

        self.assertAlmostEqual(0.04, midpoint[0], places=6)
        self.assertAlmostEqual(-0.05, midpoint[1], places=6)
        self.assertAlmostEqual(180.0, abs(math.degrees(midpoint[2])), places=6)

    def test_displacement_is_measured_in_start_frame(self):
        relative = motion.displacement_in_start_frame(
            [1.0, 2.0, math.pi / 2.0],
            [1.0, 2.05, math.pi / 2.0],
        )

        self.assertAlmostEqual(0.05, relative[0], places=6)
        self.assertAlmostEqual(0.0, relative[1], places=6)

    def test_controller_corrects_left_drift_and_positive_yaw(self):
        command, should_stop = motion.controller_command(
            [0.01, 0.02, math.radians(2.0)], 0.05, config()
        )

        self.assertFalse(should_stop)
        self.assertLess(command[1], 0.0)
        self.assertLess(command[2], 0.0)
        self.assertGreater(command[0], 0.0)

    def test_controller_corrects_right_drift_with_left_translation(self):
        command, should_stop = motion.controller_command(
            [0.01, -0.02, math.radians(-1.0)], 0.05, config()
        )

        self.assertFalse(should_stop)
        self.assertGreater(command[1], 0.0)
        self.assertGreater(command[2], 0.0)

    def test_lateral_correction_is_capped(self):
        command, should_stop = motion.controller_command(
            [0.01, -0.10, 0.0], 0.05, config()
        )

        self.assertFalse(should_stop)
        self.assertAlmostEqual(config().maximum_lateral_speed, command[1])

    def test_lateral_deadband_does_not_command_sideways_motion(self):
        command, should_stop = motion.controller_command(
            [0.01, 0.003, 0.0], 0.05, config()
        )

        self.assertFalse(should_stop)
        self.assertEqual(0.0, command[1])

    def test_controller_stops_at_early_stop_gate(self):
        command, should_stop = motion.controller_command(
            [0.046, 0.0, 0.0], 0.05, config()
        )

        self.assertTrue(should_stop)
        self.assertEqual([0.0, 0.0, 0.0], command)

    def test_safety_gate_detects_lateral_and_yaw_drift(self):
        self.assertIn(
            "lateral",
            motion.safety_violation([0.01, 0.04, 0.0], 0.05, config()),
        )
        self.assertIn(
            "yaw",
            motion.safety_violation(
                [0.01, 0.0, math.radians(5.0)], 0.05, config()
            ),
        )

    def test_command_distance_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "within"):
            motion.validate_forward_motion(0.02)
        with self.assertRaisesRegex(ValueError, "within"):
            motion.validate_forward_motion(0.06, maximum_distance=0.05)

    def test_safe_but_short_motion_is_not_reported_as_target_reached(self):
        verification = motion.post_motion_verification(
            [0.023, -0.005, math.radians(0.2)],
            0.05,
            "stance",
            config(),
        )

        self.assertTrue(verification["movement_observed"])
        self.assertTrue(verification["geometry_safe"])
        self.assertFalse(verification["target_reached"])
        self.assertTrue(verification["returned_to_stance"])

    def test_close_motion_passes_all_post_motion_checks(self):
        verification = motion.post_motion_verification(
            [0.043, 0.004, math.radians(0.3)],
            0.05,
            "stance",
            config(),
        )

        self.assertTrue(all(verification.values()))

    def test_safety_stop_cannot_be_relabelled_as_success(self):
        verification = {
            "movement_observed": True,
            "geometry_safe": True,
            "target_reached": True,
            "returned_to_stance": True,
        }

        self.assertFalse(
            motion.execution_succeeded("safety_stop", verification)
        )
        self.assertTrue(
            motion.execution_succeeded("target_gate_reached", verification)
        )

    def test_dry_run_exposes_feedback_and_stop_guards(self):
        args = Namespace(
            execute=False,
            cmd_vel_topic="/cmd_vel",
            rate_hz=20.0,
            maximum_forward_speed=0.035,
            maximum_lateral_speed=0.02,
            maximum_yaw_rate_deg=2.0,
            stop_margin=0.005,
            lateral_kp=0.5,
            path_lookahead_m=0.20,
            maximum_lateral_drift_m=0.035,
            maximum_yaw_drift_deg=4.0,
            maximum_overshoot_m=0.05,
            progress_watchdog=3.0,
            left_foot_frame="leg_l6_link",
            right_foot_frame="leg_r6_link",
        )

        payload = motion.command_plan(0.05, args)

        self.assertEqual(
            "foot_midpoint_holonomic_v4", payload["controller_version"]
        )
        self.assertEqual("cmd_vel_closed_loop", payload["motion_interface"])
        self.assertEqual(
            "TF odom -> midpoint(leg_l6_link, leg_r6_link)",
            payload["feedback"],
        )
        self.assertTrue(payload["safety"]["zero_velocity_after_motion"])
        self.assertEqual(
            "holonomic_short_step", payload["controller"]["model"]
        )
        self.assertTrue(
            payload["controller"]["direct_lateral_velocity_enabled"]
        )
        self.assertEqual(
            "CLOSED_LOOP_VELOCITY",
            payload["safety"]["explicit_confirmation"],
        )


if __name__ == "__main__":
    unittest.main()
