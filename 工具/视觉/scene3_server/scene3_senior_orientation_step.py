#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One bounded correction toward the senior Scene3 gripper orientation.

The senior implementation already defines ``RIGHT_GRIPPER_QUAT_XYZW`` for
the tray grasp.  Its normal IK path uses position-hard/orientation-soft mode,
so the arm can reach the Cartesian point while leaving the claw visibly
tilted.  This helper keeps the current hand position and takes at most one
small quaternion step toward that senior-authored orientation using explicit
position-hard/orientation-hard IK mode 3.

Default mode is calculation-only.  Execution requires an exact confirmation
token.  The base remains stopped and the claw remains open.  If the measured
orientation does not improve, the pre-command arm joint vector is restored.
"""

from __future__ import print_function

import argparse
import math
import os
import sys

import numpy as np

from scene3_wrist_camera_look_at import (
    quaternion_angle_degrees,
    solve_pose_hard_ik,
)


EXECUTION_CONFIRMATION = "SENIOR_ORIENTATION_4DEG"
DEFAULT_SENIOR_DIR = "/root/kuavo_ws/src/challenge_cup_task_template/scripts"
DEFAULT_TARGET_PARAM = "/challenge_cup_task_template/scene3/locked_target_base_xyz"


def normalize_quaternion(quaternion_xyzw):
    quaternion = np.asarray(quaternion_xyzw, dtype=float).copy()
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be finite xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be nonzero")
    return quaternion / norm


def plan_orientation_step(current_xyzw, goal_xyzw, maximum_step_degrees=4.0):
    """SLERP from current toward goal along the shortest quaternion arc."""
    current = normalize_quaternion(current_xyzw)
    goal = normalize_quaternion(goal_xyzw)
    dot = float(np.clip(np.dot(current, goal), -1.0, 1.0))
    if dot < 0.0:
        goal = -goal
        dot = -dot

    total_radians = 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))
    maximum_radians = math.radians(max(0.1, float(maximum_step_degrees)))
    if total_radians <= 1e-9:
        return current, 0.0, 0.0
    fraction = min(1.0, maximum_radians / total_radians)

    if dot > 0.9995:
        desired = normalize_quaternion(current + fraction * (goal - current))
    else:
        theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
        sin_theta = math.sin(theta)
        weight_current = math.sin((1.0 - fraction) * theta) / sin_theta
        weight_goal = math.sin(fraction * theta) / sin_theta
        desired = normalize_quaternion(
            weight_current * current + weight_goal * goal
        )
    return desired, math.degrees(total_radians), math.degrees(
        min(total_radians, maximum_radians)
    )


def validate_orientation_plan(
        left_delta_degrees,
        right_delta_degrees,
        fk_position_error_m,
        desired_fk_orientation_error_degrees,
        before_goal_error_degrees,
        predicted_goal_error_degrees,
        minimum_goal_reduction_degrees=0.5,
        minimum_right_joint_delta_degrees=0.1,
        maximum_right_joint_delta_degrees=15.0,
        maximum_left_joint_delta_degrees=2.0,
        maximum_fk_position_error_m=0.015,
        maximum_fk_orientation_error_degrees=3.0):
    left = np.asarray(left_delta_degrees, dtype=float)
    right = np.asarray(right_delta_degrees, dtype=float)
    values = np.concatenate((
        left,
        right,
        np.asarray([
            fk_position_error_m,
            desired_fk_orientation_error_degrees,
            before_goal_error_degrees,
            predicted_goal_error_degrees,
        ], dtype=float),
    ))
    reduction = float(before_goal_error_degrees - predicted_goal_error_degrees)
    checks = {
        "values_finite": bool(np.all(np.isfinite(values))),
        "right_joint_motion_nonzero": float(np.max(np.abs(right))) >= float(
            minimum_right_joint_delta_degrees
        ),
        "right_joint_delta_bounded": float(np.max(np.abs(right))) <= float(
            maximum_right_joint_delta_degrees
        ),
        "left_arm_held": float(np.max(np.abs(left))) <= float(
            maximum_left_joint_delta_degrees
        ),
        "hand_position_held": float(fk_position_error_m) <= float(
            maximum_fk_position_error_m
        ),
        "step_orientation_reached": float(
            desired_fk_orientation_error_degrees
        ) <= float(maximum_fk_orientation_error_degrees),
        "senior_orientation_improved": reduction >= float(
            minimum_goal_reduction_degrees
        ),
    }
    return bool(all(checks.values())), checks, reduction


def validate_measured_step(
        before_goal_error_degrees,
        after_goal_error_degrees,
        hand_translation_m,
        minimum_goal_reduction_degrees=0.5,
        maximum_hand_translation_m=0.035):
    reduction = float(before_goal_error_degrees - after_goal_error_degrees)
    values = np.asarray([
        before_goal_error_degrees,
        after_goal_error_degrees,
        hand_translation_m,
        reduction,
    ], dtype=float)
    checks = {
        "values_finite": bool(np.all(np.isfinite(values))),
        "senior_orientation_improved": reduction >= float(
            minimum_goal_reduction_degrees
        ),
        "translation_bounded": float(hand_translation_m) <= float(
            maximum_hand_translation_m
        ),
    }
    return bool(all(checks.values())), checks, reduction


def load_senior(senior_dir):
    senior_dir = os.path.abspath(senior_dir)
    source_file = os.path.join(senior_dir, "challenge_task_3.py")
    if not os.path.isfile(source_file):
        raise RuntimeError("senior challenge_task_3.py not found: {}".format(
            source_file
        ))
    if senior_dir in sys.path:
        sys.path.remove(senior_dir)
    sys.path.insert(0, senior_dir)
    from challenge_task_3 import (  # pylint: disable=import-error
        RIGHT_GRIPPER_QUAT_XYZW,
        Scene3Task,
        rad_to_deg,
    )
    return Scene3Task, RIGHT_GRIPPER_QUAT_XYZW, rad_to_deg


def restore_arm(task, rad_to_deg, joints, duration):
    print("Safety rollback: restoring pre-step arm joints")
    task.move_arm_degrees(rad_to_deg(joints), duration=float(duration))
    task.stop_base()


def run_ros(args):
    import rospy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import JointState

    rospy.init_node("scene3_senior_orientation_step", anonymous=True)
    if not rospy.has_param(args.target_param):
        raise RuntimeError(
            "no locked senior target; run scene3_senior_pregrasp_gate.py first"
        )
    locked_target = np.asarray(rospy.get_param(args.target_param), dtype=float)
    if locked_target.shape != (3,) or not np.all(np.isfinite(locked_target)):
        raise RuntimeError("locked senior target parameter is invalid")

    Scene3Task, senior_quaternion, rad_to_deg = load_senior(args.senior_dir)
    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    task = Scene3Task(cmd_vel_pub, arm_traj_pub)

    current_joints = task.read_current_arm_joints()
    current_poses = task.call_fk(current_joints)
    current_position = np.asarray(
        current_poses.right_pose.pos_xyz, dtype=float
    )
    current_quaternion = np.asarray(
        current_poses.right_pose.quat_xyzw, dtype=float
    )
    goal_quaternion = normalize_quaternion(senior_quaternion)
    desired_quaternion, before_goal_error, planned_step = plan_orientation_step(
        current_quaternion,
        goal_quaternion,
        maximum_step_degrees=args.maximum_angle_step,
    )
    if planned_step < 0.1:
        print("SENIOR_ORIENTATION_ALREADY_ALIGNED")
        return 0

    solution = solve_pose_hard_ik(
        task,
        current_joints,
        current_poses,
        current_position.tolist(),
        desired_quaternion.tolist(),
        position_tolerance_m=args.position_ik_tolerance,
        orientation_tolerance_rad=args.orientation_ik_tolerance,
    )
    if len(solution) != 14:
        raise RuntimeError("pose-hard IK did not return fourteen arm joints")

    joint_delta_degrees = np.degrees(
        np.asarray(solution, dtype=float) - np.asarray(current_joints, dtype=float)
    )
    left_delta = joint_delta_degrees[:7]
    right_delta = joint_delta_degrees[7:14]
    predicted_poses = task.call_fk(solution)
    predicted_position = np.asarray(
        predicted_poses.right_pose.pos_xyz, dtype=float
    )
    predicted_quaternion = np.asarray(
        predicted_poses.right_pose.quat_xyzw, dtype=float
    )
    fk_position_error = float(np.linalg.norm(
        predicted_position - current_position
    ))
    desired_fk_error = quaternion_angle_degrees(
        desired_quaternion, predicted_quaternion
    )
    predicted_goal_error = quaternion_angle_degrees(
        predicted_quaternion, goal_quaternion
    )
    plan_ok, plan_checks, predicted_reduction = validate_orientation_plan(
        left_delta,
        right_delta,
        fk_position_error,
        desired_fk_error,
        before_goal_error,
        predicted_goal_error,
        minimum_goal_reduction_degrees=args.minimum_orientation_reduction,
        minimum_right_joint_delta_degrees=args.minimum_joint_delta,
        maximum_right_joint_delta_degrees=args.maximum_joint_delta,
        maximum_left_joint_delta_degrees=args.maximum_left_joint_delta,
        maximum_fk_position_error_m=args.maximum_fk_position_error,
        maximum_fk_orientation_error_degrees=args.maximum_fk_orientation_error,
    )

    print("Locked senior target:", np.round(locked_target, 4).tolist())
    print("Senior gripper quaternion:", np.round(goal_quaternion, 6).tolist())
    print("Current hand position held at:", np.round(current_position, 4).tolist())
    print("Current senior-orientation error: {:.2f}deg".format(
        before_goal_error
    ))
    print("Planned orientation step: {:.2f}deg".format(planned_step))
    print("IK constraint mode: 3 (position hard + orientation hard)")
    print("Planned left-arm joint delta:", np.round(left_delta, 2).tolist())
    print("Planned right-arm joint delta:", np.round(right_delta, 2).tolist())
    print("Predicted hand translation: {:.4f}m".format(fk_position_error))
    print("Predicted step-orientation error: {:.2f}deg".format(
        desired_fk_error
    ))
    print("Predicted senior-orientation error: {:.2f}deg -> {:.2f}deg".format(
        before_goal_error, predicted_goal_error
    ))
    print("Predicted orientation reduction: {:.2f}deg".format(
        predicted_reduction
    ))
    print("Dry-run safety checks:", plan_checks)
    if not plan_ok:
        raise RuntimeError(
            "SENIOR_ORIENTATION_IK_BLOCKED: predicted solution failed safety gates"
        )
    print("SENIOR_ORIENTATION_IK_OK: verified calculation; no command sent yet")

    if not args.execute:
        print("SENIOR_ORIENTATION_DRY_RUN_OK: claw remains open")
        return 0
    if args.confirmation != EXECUTION_CONFIRMATION:
        raise RuntimeError(
            "execution blocked; pass --confirmation {}".format(
                EXECUTION_CONFIRMATION
            )
        )

    task.stop_base()
    task.wait_for_arm_subscriber(timeout=8.0)
    if not task.set_arm_mode(2):
        raise RuntimeError("cannot enable senior arm external-control mode")
    if not task.open_claw():
        raise RuntimeError("cannot confirm open claw before orientation step")
    print("Executing one bounded step toward senior gripper orientation")
    task.move_arm_degrees(rad_to_deg(solution), duration=args.motion_seconds)
    task.stop_base()
    rospy.sleep(args.settle_seconds)

    after_joints = task.read_current_arm_joints()
    after_poses = task.call_fk(after_joints)
    after_position = np.asarray(after_poses.right_pose.pos_xyz, dtype=float)
    after_quaternion = np.asarray(
        after_poses.right_pose.quat_xyzw, dtype=float
    )
    after_goal_error = quaternion_angle_degrees(
        after_quaternion, goal_quaternion
    )
    translation = float(np.linalg.norm(after_position - current_position))
    ok, checks, measured_reduction = validate_measured_step(
        before_goal_error,
        after_goal_error,
        translation,
        minimum_goal_reduction_degrees=args.minimum_orientation_reduction,
        maximum_hand_translation_m=args.maximum_hand_translation,
    )
    print("Actual hand position:", np.round(after_position, 4).tolist())
    print("Actual hand translation: {:.4f}m".format(translation))
    print("Measured senior-orientation error: {:.2f}deg -> {:.2f}deg".format(
        before_goal_error, after_goal_error
    ))
    print("Measured orientation reduction: {:.2f}deg".format(
        measured_reduction
    ))
    print("Post-motion safety checks:", checks)
    if not ok:
        try:
            restore_arm(task, rad_to_deg, current_joints, args.rollback_seconds)
            print("SENIOR_ORIENTATION_ROLLBACK_OK")
        except Exception as exc:
            print("SENIOR_ORIENTATION_ROLLBACK_FAILED: {}".format(exc))
        raise RuntimeError(
            "SENIOR_ORIENTATION_STEP_BLOCKED: measured response did not improve"
        )
    print("SENIOR_ORIENTATION_STEP_OK: claw remains open")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--senior-dir", default=DEFAULT_SENIOR_DIR)
    parser.add_argument("--target-param", default=DEFAULT_TARGET_PARAM)
    parser.add_argument("--maximum-angle-step", type=float, default=4.0)
    parser.add_argument("--minimum-orientation-reduction", type=float, default=0.5)
    parser.add_argument("--minimum-joint-delta", type=float, default=0.1)
    parser.add_argument("--maximum-joint-delta", type=float, default=15.0)
    parser.add_argument("--maximum-left-joint-delta", type=float, default=2.0)
    parser.add_argument("--position-ik-tolerance", type=float, default=0.004)
    parser.add_argument("--orientation-ik-tolerance", type=float, default=0.02)
    parser.add_argument("--maximum-fk-position-error", type=float, default=0.015)
    parser.add_argument("--maximum-fk-orientation-error", type=float, default=3.0)
    parser.add_argument("--maximum-hand-translation", type=float, default=0.035)
    parser.add_argument("--motion-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--rollback-seconds", type=float, default=2.5)
    return parser


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
