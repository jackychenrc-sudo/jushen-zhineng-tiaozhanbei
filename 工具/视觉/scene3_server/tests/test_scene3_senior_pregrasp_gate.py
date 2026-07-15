#!/usr/bin/env python3

import os
import sys
import types
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(HERE)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import scene3_senior_pregrasp_gate as gate


class FakePoint(object):
    def __init__(self):
        self.x = 0.58
        self.y = -0.20
        self.z = 0.25


class FakeHeader(object):
    def __init__(self):
        self.frame_id = "base_link"
        self.stamp = None


class FakeTarget(object):
    def __init__(self):
        self.header = FakeHeader()
        self.point = FakePoint()


class FakePublisher(object):
    def __init__(self, ros, topic, message_type, queue_size, latch):
        self.ros = ros
        self.topic = topic
        self.message_type = message_type
        self.queue_size = queue_size
        self.latch = latch

    def publish(self, message):
        self.ros.published.append((self.topic, message, self.latch))


class FakeRos(object):
    def __init__(self):
        self.logs = []
        self.published = []
        self.params = {}

    def loginfo(self, message, *args):
        self.logs.append(message % args if args else message)

    def Publisher(self, topic, message_type, queue_size, latch):
        return FakePublisher(self, topic, message_type, queue_size, latch)

    def set_param(self, name, value):
        self.params[name] = list(value)


class FakeTask(object):
    target_freshness = 0.5

    def __init__(self, target=True, claw_ok=True):
        self.target = FakeTarget() if target else None
        self.claw_ok = claw_ok
        self.calls = []
        self.rospy = FakeRos()

    def wait_for_recent_base_target(self, timeout, freshness):
        self.calls.append(("wait", timeout, freshness))
        return self.target

    def build_scene3_grasp_targets(self, target):
        self.calls.append(("build", target))
        return {
            "pregrasp": [0.42, -0.20, 0.27],
            "touch": [0.53, -0.20, 0.27],
            "grasp": [0.57, -0.20, 0.27],
            "lift": [0.42, -0.20, 0.35],
            "retreat": [0.38, -0.20, 0.35],
        }

    def stop_base(self):
        self.calls.append(("stop_base",))

    def open_claw(self):
        self.calls.append(("open_claw",))
        return self.claw_ok

    def move_right_hand(self, target, duration):
        self.calls.append(("move_right_hand", list(target), duration))


class SeniorPregraspGateTest(unittest.TestCase):
    def test_runs_only_one_senior_pregrasp(self):
        task = FakeTask()
        result = gate.run_pregrasp_only(task, target_timeout=4.0, motion_seconds=3.0)
        self.assertEqual([0.42, -0.20, 0.27], result)
        moves = [call for call in task.calls if call[0] == "move_right_hand"]
        self.assertEqual([("move_right_hand", [0.42, -0.20, 0.27], 3.0)], moves)
        self.assertNotIn("close_claw", [call[0] for call in task.calls])
        self.assertEqual("stop_base", task.calls[-1][0])
        self.assertEqual(1, len(task.rospy.published))
        self.assertTrue(task.rospy.published[0][2])
        self.assertEqual(
            [0.58, -0.20, 0.25],
            task.rospy.params[gate.LOCKED_TARGET_PARAM],
        )
        self.assertEqual([0.58, -0.20, 0.25], task._scene3_locked_target_xyz)

    def test_missing_target_fails_before_arm_or_claw(self):
        task = FakeTask(target=False)
        with self.assertRaisesRegex(RuntimeError, "no fresh Scene3 target"):
            gate.run_pregrasp_only(task)
        self.assertFalse(any(call[0] == "move_right_hand" for call in task.calls))
        self.assertFalse(any(call[0] == "open_claw" for call in task.calls))

    def test_claw_open_failure_blocks_arm(self):
        task = FakeTask(claw_ok=False)
        with self.assertRaisesRegex(RuntimeError, "cannot confirm open claw"):
            gate.run_pregrasp_only(task)
        self.assertFalse(any(call[0] == "move_right_hand" for call in task.calls))

    def test_installed_gate_logs_ready_and_returns_false(self):
        module = types.SimpleNamespace(Scene3Task=types.SimpleNamespace())
        installed = gate.install_pregrasp_gate(module, motion_seconds=2.0)
        task = FakeTask()
        self.assertFalse(installed(task))
        self.assertIn("SENIOR_PREGRASP_READY", task.rospy.logs[-1])
        self.assertIn("locked_target", task.rospy.logs[-1])
        self.assertEqual(1, len([c for c in task.calls if c[0] == "move_right_hand"]))

    def test_non_base_target_is_blocked_before_arm(self):
        task = FakeTask()
        task.target.header.frame_id = "odom"
        with self.assertRaisesRegex(RuntimeError, "must already be in base_link"):
            gate.run_pregrasp_only(task)
        self.assertFalse(any(call[0] == "move_right_hand" for call in task.calls))


if __name__ == "__main__":
    unittest.main()
