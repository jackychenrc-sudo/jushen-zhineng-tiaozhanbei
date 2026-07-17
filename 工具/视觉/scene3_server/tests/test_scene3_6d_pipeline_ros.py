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

from scene3_6d_pipeline_ros import Scene36DRosController  # noqa: E402


class FakeRosException(Exception):
    pass


class FakeRospy(object):
    ROSException = FakeRosException

    def __init__(self, message=None, fail=False):
        self.message = message
        self.fail = bool(fail)
        self.calls = []

    def wait_for_message(self, topic, message_type, timeout):
        self.calls.append((topic, message_type, timeout))
        if self.fail:
            raise self.ROSException("timeout")
        return self.message


class Scene36DRosControllerTest(unittest.TestCase):

    @staticmethod
    def make_controller(rospy):
        controller = object.__new__(Scene36DRosController)
        controller.rospy = rospy
        controller.JointState = object
        controller.args = types.SimpleNamespace(arm_topic_quiet_seconds=0.4)
        return controller

    def test_arm_topic_is_quiet_after_observation_timeout(self):
        rospy = FakeRospy(fail=True)
        controller = self.make_controller(rospy)

        self.assertTrue(controller.arm_topic_is_quiet())
        self.assertEqual(
            rospy.calls,
            [("/kuavo_arm_traj", object, 0.4)],
        )

    def test_arm_topic_is_not_quiet_when_a_command_arrives(self):
        rospy = FakeRospy(message=object())
        controller = self.make_controller(rospy)

        self.assertFalse(controller.arm_topic_is_quiet())

    def test_active_arm_publisher_blocks_before_mode_switch(self):
        controller = object.__new__(Scene36DRosController)
        mode_switches = []
        controller.arm_topic_is_quiet = lambda: False
        controller.enable_external_arm_mode = lambda: mode_switches.append(True)

        state = {}
        plan = {
            "source_command_deg": np.zeros(14),
            "target_command_deg": np.ones(14),
        }
        ok, result = controller.execute_plan(
            state,
            plan,
            post_check=lambda *_args: True,
            label="TEST_ARM_CONFLICT",
        )

        self.assertFalse(ok)
        self.assertIsNone(result)
        self.assertEqual(mode_switches, [])


if __name__ == "__main__":
    unittest.main()
