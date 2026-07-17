#!/usr/bin/env python3

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

from scene3_6d_grasp_pipeline import (  # noqa: E402
    CONFIRMATION,
    LOCKED_BASE_PARAM,
    build_parser,
    desired_rotation_for_targets,
    grasp_targets,
    plan_align_segment,
    plan_insert_segment,
    plan_preprocess_segment,
    require_execution_confirmation,
    run_wrist_verification,
)
from scene3_gripper_6d_align_plan import (  # noqa: E402
    matrix_to_quaternion,
    quaternion_to_matrix,
)


class FakeController(object):
    def __init__(self):
        self.modes = []

    def solve_pose(self, state, target_position, target_quaternion,
                   constraint_mode):
        self.modes.append(int(constraint_mode))
        rotation = quaternion_to_matrix(target_quaternion)
        return {
            "success": True,
            "checks": {
                "ik_success": True,
                "position_residual": True,
                "orientation_residual": True,
                "right_joint_delta": True,
                "left_command_frozen": True,
                "values_finite": True,
            },
            "delta_deg": np.zeros(14),
            "source_command_deg": np.zeros(14),
            "target_command_deg": np.zeros(14),
            "predicted_position": np.asarray(target_position, dtype=float),
            "predicted_quaternion": list(target_quaternion),
            "predicted_rotation": rotation,
        }


class FakeRospy(object):
    def __init__(self):
        self.params = {}

    def set_param(self, name, value):
        self.params[name] = value


class FakeVerificationController(object):
    def __init__(self, state):
        self.state = state
        self.rospy = FakeRospy()

    def sample_state(self):
        return self.state

    def audit_checks(self, _state):
        return {"ready": True}


def make_state(tcp, physical_rotation=None, tray=None):
    rotation = np.eye(3) if physical_rotation is None else np.asarray(
        physical_rotation, dtype=float
    )
    tray = np.array([0.60, 0.0, 0.25]) if tray is None else np.asarray(
        tray, dtype=float
    )
    return {
        "tray_base": tray,
        "tray_odom": np.array([1.0, 2.0, 3.0]),
        "eef_position": np.asarray(tcp, dtype=float),
        "eef_rotation": rotation.copy(),
        "eef_quaternion": matrix_to_quaternion(rotation),
        "geometry": {
            "tcp": np.asarray(tcp, dtype=float),
            "rotation": rotation.copy(),
        },
    }


class Scene36DGraspPipelineTest(unittest.TestCase):

    def setUp(self):
        self.args = build_parser().parse_args([])

    def test_default_is_read_only_audit(self):
        self.assertEqual(self.args.stage, "audit")
        self.assertFalse(self.args.execute)
        self.assertFalse(self.args.close_claw)

    def test_execution_requires_exact_confirmation(self):
        args = build_parser().parse_args([
            "--stage", "preprocess", "--execute"
        ])
        with self.assertRaises(RuntimeError):
            require_execution_confirmation(args)
        args.confirmation = CONFIRMATION
        require_execution_confirmation(args)

    def test_preprocess_uses_position_hard_orientation_soft(self):
        controller = FakeController()
        state = make_state([0.10, 0.0, 0.10])
        plan = plan_preprocess_segment(controller, self.args, state)
        self.assertFalse(plan.get("blocked", False))
        self.assertEqual(controller.modes, [2])
        self.assertEqual(plan["stage"], "preprocess")

    def test_alignment_holds_safe_tcp_and_uses_hard_6d(self):
        controller = FakeController()
        targets = grasp_targets(
            self.args, make_state([0.0, 0.0, 0.0])
        )
        # Rotate the gripper 20 degrees around base Z while keeping it at safe.
        angle = np.deg2rad(20.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        state = make_state(targets["preprocess_tcp"], rotation)
        plan = plan_align_segment(controller, self.args, state)
        self.assertFalse(plan.get("blocked", False))
        self.assertEqual(controller.modes, [3])
        self.assertEqual(plan["stage"], "align")
        np.testing.assert_allclose(
            plan["predicted_physical_tcp"],
            targets["preprocess_tcp"],
            atol=1e-12,
        )

    def test_insert_keeps_hard_orientation_on_fixed_line(self):
        controller = FakeController()
        seed = make_state([0.0, 0.0, 0.0])
        targets = grasp_targets(self.args, seed)
        state = make_state(
            targets["preprocess_tcp"],
            np.eye(3),
        )
        plan = plan_insert_segment(controller, self.args, state)
        self.assertFalse(plan.get("blocked", False))
        self.assertEqual(controller.modes, [3])
        self.assertEqual(plan["stage"], "insert")
        delta = (
            plan["target_physical_tcp"] - targets["preprocess_tcp"]
        )
        np.testing.assert_allclose(
            delta,
            float(self.args.insert_step) * targets["approach"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            plan["predicted_physical_rotation"], np.eye(3), atol=1e-12
        )

    def test_wrist_gate_is_prompted_at_near_edge_not_tray_center(self):
        seed = make_state([0.0, 0.0, 0.0])
        targets = grasp_targets(self.args, seed)
        desired = desired_rotation_for_targets(targets)
        state = make_state(
            targets["final_tcp"],
            desired,
            tray=seed["tray_base"],
        )
        controller = FakeVerificationController(state)

        with patch(
            "scene3_6d_grasp_pipeline.subprocess.call",
            return_value=0,
        ):
            result = run_wrist_verification(controller, self.args)

        self.assertEqual(result, 0)
        np.testing.assert_allclose(
            controller.rospy.params[LOCKED_BASE_PARAM],
            targets["grasp_surface"],
            atol=1e-12,
        )
        self.assertGreater(
            np.linalg.norm(
                np.asarray(controller.rospy.params[LOCKED_BASE_PARAM])
                - state["tray_base"]
            ),
            0.10,
        )


if __name__ == "__main__":
    unittest.main()
