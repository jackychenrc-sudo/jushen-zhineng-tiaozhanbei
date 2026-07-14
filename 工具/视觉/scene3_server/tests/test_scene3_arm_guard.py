import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


DOWNLOADS = Path(__file__).resolve().parent
if str(DOWNLOADS) not in sys.path:
    sys.path.insert(0, str(DOWNLOADS))

import scene3_arm_guard as guard


def point_message(xyz, frame_id="base_link"):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        point=SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
    )


def stable_messages():
    return [
        point_message([0.650, -0.230, 0.270]),
        point_message([0.652, -0.229, 0.271]),
        point_message([0.649, -0.231, 0.269]),
        point_message([0.651, -0.230, 0.270]),
        point_message([0.650, -0.232, 0.271]),
    ]


class FakeRospy(object):
    @staticmethod
    def sleep(_duration):
        return None


class FakeTask(object):
    def __init__(self, messages=None, fk_offset=None):
        self.messages = list(messages or stable_messages())
        self.message_index = 0
        self.last_grasp_point_base_wall_time = 0.0
        self.rospy = FakeRospy()
        self.stop_calls = 0
        self.move_calls = []
        self.fk_offset = list(fk_offset or [0.0, 0.0, 0.0])
        self.current_joints = [0.0] * 14
        self.arm_mode_ok = True

    def stop_base(self):
        self.stop_calls += 1

    def get_recent_grasp_point_base(self, freshness=None):
        del freshness
        if self.message_index >= len(self.messages):
            return self.messages[-1] if self.messages else None
        message = self.messages[self.message_index]
        self.message_index += 1
        self.last_grasp_point_base_wall_time = float(self.message_index)
        return message

    def solve_right_hand_ik(self, position, quaternion):
        return list(position) + list(quaternion) + [0.0] * 7

    def call_fk(self, joints):
        position = [
            float(joints[index]) + self.fk_offset[index] for index in range(3)
        ]
        quaternion = list(joints[3:7])
        right_pose = SimpleNamespace(
            pos_xyz=position,
            quat_xyzw=quaternion,
        )
        return SimpleNamespace(right_pose=right_pose)

    def move_arm_degrees(self, target_degrees, duration=2.0):
        self.move_calls.append((list(target_degrees), duration))
        self.current_joints = [
            float(value) * math.pi / 180.0 for value in target_degrees
        ]

    def set_arm_mode(self, mode):
        return mode == 2 and self.arm_mode_ok

    def read_current_arm_joints(self, timeout=5.0):
        del timeout
        return list(self.current_joints)


class Scene3ArmGuardTests(unittest.TestCase):
    def test_stable_five_frame_target_uses_median(self):
        config = guard.ArmGuardConfig()
        samples = [
            guard.message_sample(message, index + 1.0)
            for index, message in enumerate(stable_messages())
        ]

        result = guard.aggregate_target_samples(samples, config)

        self.assertEqual("base_link", result["frame_id"])
        self.assertEqual(5, result["sample_count"])
        self.assertAlmostEqual(0.650, result["median_xyz_m"][0])
        self.assertLess(result["maximum_3d_spread_m"], 0.01)

    def test_target_jitter_over_one_centimeter_is_blocked(self):
        config = guard.ArmGuardConfig()
        messages = stable_messages()
        messages[-1] = point_message([0.680, -0.230, 0.270])
        samples = [
            guard.message_sample(message, index + 1.0)
            for index, message in enumerate(messages)
        ]

        with self.assertRaisesRegex(ValueError, "target spread"):
            guard.aggregate_target_samples(samples, config)

    def test_non_base_link_target_is_blocked(self):
        config = guard.ArmGuardConfig()
        messages = [
            point_message([0.65, -0.23, 0.27], frame_id="torso")
            for _ in range(5)
        ]
        samples = [
            guard.message_sample(message, index + 1.0)
            for index, message in enumerate(messages)
        ]

        with self.assertRaisesRegex(ValueError, "base_link"):
            guard.aggregate_target_samples(samples, config)

    def test_grasp_targets_preserve_staged_offsets(self):
        config = guard.ArmGuardConfig()
        guard.validate_config(config)

        targets = guard.build_grasp_targets([0.65, -0.23, 0.27], config)

        for actual, expected in zip(
            targets["pregrasp_xyz_m"], [0.49, -0.23, 0.29]
        ):
            self.assertAlmostEqual(expected, actual)
        for actual, expected in zip(
            targets["touch_xyz_m"], [0.60, -0.23, 0.29]
        ):
            self.assertAlmostEqual(expected, actual)
        for actual, expected in zip(
            targets["grasp_xyz_m"], [0.64, -0.23, 0.29]
        ):
            self.assertAlmostEqual(expected, actual)
        for actual, expected in zip(
            targets["initial_lift_xyz_m"], [0.64, -0.23, 0.31]
        ):
            self.assertAlmostEqual(expected, actual)

    def test_analysis_runs_21_ik_fk_checks_without_arm_command(self):
        task = FakeTask()

        report = guard.Scene3ArmGuard(task).run()

        self.assertEqual("single_pregrasp_ready", report["status"])
        self.assertEqual(21, report["ik_fk_robustness"]["check_count"])
        self.assertTrue(report["ik_fk_robustness"]["passed"])
        self.assertFalse(report["arm_command_sent"])
        self.assertEqual([], task.move_calls)

    def test_fk_position_error_blocks_pregrasp(self):
        task = FakeTask(fk_offset=[0.03, 0.0, 0.0])

        report = guard.Scene3ArmGuard(task).run()

        self.assertEqual("ik_fk_gate_blocked", report["status"])
        self.assertFalse(report["ready_for_single_pregrasp"])
        self.assertEqual([], task.move_calls)

    def test_pregrasp_requires_exact_confirmation(self):
        task = FakeTask()

        report = guard.Scene3ArmGuard(task).run(
            execute_pregrasp=True,
            confirmation="yes",
        )

        self.assertEqual("confirmation_blocked", report["status"])
        self.assertEqual([], task.move_calls)

    def test_external_arm_mode_failure_blocks_command(self):
        task = FakeTask()
        task.arm_mode_ok = False

        report = guard.Scene3ArmGuard(task).run(
            execute_pregrasp=True,
            confirmation=guard.PREGRASP_CONFIRMATION,
        )

        self.assertEqual("arm_mode_blocked", report["status"])
        self.assertEqual([], task.move_calls)

    def test_confirmed_pregrasp_executes_once_and_stops_before_claw(self):
        task = FakeTask()

        report = guard.Scene3ArmGuard(task).run(
            execute_pregrasp=True,
            confirmation=guard.PREGRASP_CONFIRMATION,
        )

        self.assertEqual("single_pregrasp_completed", report["status"])
        self.assertTrue(report["post_execution_fk"]["passed"])
        self.assertTrue(report["arm_command_sent"])
        self.assertFalse(report["claw_command_sent"])
        self.assertEqual(1, len(task.move_calls))

    def test_quaternion_sign_represents_the_same_orientation(self):
        quaternion = guard.normalize_quaternion(guard.RIGHT_GRIPPER_QUAT_XYZW)
        opposite = [-value for value in quaternion]

        self.assertAlmostEqual(
            0.0,
            guard.quaternion_angle_error_deg(quaternion, opposite),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
