#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dry-run one bounded 6D right-arm IK target for Scene3.

The default target is exactly the current official right-arm end-effector
pose.  This zero-motion calculation verifies the joint mapping, quaternion
convention, IK frame and FK frame before any taught orientation is used.

Optional Cartesian and roll/pitch/yaw increments make the same program useful
for later 6D teaching.  This file deliberately creates no ROS publisher and no
claw service proxy, so it cannot move the base, arm or claw.
"""

from __future__ import print_function

import argparse
import math


REFERENCE_PARAM = (
    "/challenge_cup_task_template/scene3/arm_command_reference_deg"
)
PLAN_PARAM = "/challenge_cup_task_template/scene3/six_d_teach_plan"
IK_MODE_POSITION_AND_ORIENTATION_HARD = 3


def normalize_quaternion(quaternion):
    values = [float(value) for value in quaternion]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion must contain four finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-12:
        raise ValueError("quaternion has zero length")
    return [value / norm for value in values]


def quaternion_multiply(first, second):
    """Return ``first * second`` for quaternions in ROS xyzw order."""

    x1, y1, z1, w1 = normalize_quaternion(first)
    x2, y2, z2, w2 = normalize_quaternion(second)
    return normalize_quaternion([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quaternion_from_rpy_degrees(roll, pitch, yaw):
    """Build a ROS xyzw quaternion using the conventional Z-Y-X RPY order."""

    half_roll = math.radians(float(roll)) * 0.5
    half_pitch = math.radians(float(pitch)) * 0.5
    half_yaw = math.radians(float(yaw)) * 0.5
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return normalize_quaternion([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def apply_rotation_increment(current_xyzw, rpy_degrees, frame="local"):
    delta = quaternion_from_rpy_degrees(*rpy_degrees)
    if frame == "local":
        return quaternion_multiply(current_xyzw, delta)
    if frame == "base":
        return quaternion_multiply(delta, current_xyzw)
    raise ValueError("rotation frame must be 'local' or 'base'")


def quaternion_error_degrees(target_xyzw, actual_xyzw):
    target = normalize_quaternion(target_xyzw)
    actual = normalize_quaternion(actual_xyzw)
    dot = abs(sum(a * b for a, b in zip(target, actual)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def position_error(first, second):
    return math.sqrt(sum(
        (float(a) - float(b)) ** 2 for a, b in zip(first, second)
    ))


def extract_arm_joints(sensor_message):
    joint_q = list(sensor_message.joint_data.joint_q)
    if len(joint_q) >= 27:
        # V52: waist is index 12, left arm 13:20, right arm 20:27.
        return [float(value) for value in joint_q[13:27]], "joint_q[13:27]"
    if len(joint_q) >= 26:
        return [float(value) for value in joint_q[12:26]], "joint_q[12:26]"
    raise RuntimeError("joint_q is too short: {}".format(len(joint_q)))


def extract_ik_solution(response, current_arm):
    q_arm = list(getattr(response, "q_arm", []))
    if len(q_arm) >= 14:
        return [float(value) for value in q_arm[:14]]

    hand_poses = getattr(response, "hand_poses", None)
    if hand_poses is not None:
        left = list(hand_poses.left_pose.joint_angles)
        right = list(hand_poses.right_pose.joint_angles)
        if len(left) == 7 and len(right) == 7:
            return [float(value) for value in left + right]
        if len(right) == 7:
            return list(current_arm[:7]) + [float(value) for value in right]
    raise RuntimeError("IK response does not contain fourteen arm joints")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Scene3 official 6D IK dry run; no control publisher exists"
    )
    parser.add_argument("--dx", type=float, default=0.0,
                        help="base-frame X increment in metres")
    parser.add_argument("--dy", type=float, default=0.0,
                        help="base-frame Y increment in metres")
    parser.add_argument("--dz", type=float, default=0.0,
                        help="base-frame Z increment in metres")
    parser.add_argument("--droll", type=float, default=0.0,
                        help="roll increment in degrees")
    parser.add_argument("--dpitch", type=float, default=0.0,
                        help="pitch increment in degrees")
    parser.add_argument("--dyaw", type=float, default=0.0,
                        help="yaw increment in degrees")
    parser.add_argument("--rotation-frame", choices=("local", "base"),
                        default="local")
    parser.add_argument("--maximum-right-joint-delta", type=float, default=5.0)
    parser.add_argument("--maximum-position-error", type=float, default=0.005)
    parser.add_argument("--maximum-orientation-error", type=float, default=2.0)
    parser.add_argument("--save-plan", action="store_true",
                        help="save a calculation-only plan to the ROS parameter server")
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser


def run_ros(args):
    import rospy
    from kuavo_msgs.msg import ikSolveParam, sensorsData, twoArmHandPoseCmd
    from kuavo_msgs.srv import fkSrv, twoArmHandPoseCmdSrv

    rospy.init_node("scene3_6d_pose_dry_run", anonymous=True)

    sensor = rospy.wait_for_message(
        "/sensors_data_raw", sensorsData, timeout=float(args.timeout)
    )
    current_arm, mapping = extract_arm_joints(sensor)

    rospy.wait_for_service("/ik/fk_srv", timeout=float(args.timeout))
    fk_proxy = rospy.ServiceProxy("/ik/fk_srv", fkSrv)

    def call_fk(arm):
        response = fk_proxy([float(value) for value in arm])
        if not getattr(response, "success", False):
            raise RuntimeError("/ik/fk_srv failed")
        return response.hand_poses

    current_poses = call_fk(current_arm)
    current_right_position = [
        float(value) for value in current_poses.right_pose.pos_xyz
    ]
    current_right_quaternion = normalize_quaternion(
        current_poses.right_pose.quat_xyzw
    )

    translation_increment = [float(args.dx), float(args.dy), float(args.dz)]
    rotation_increment = [
        float(args.droll), float(args.dpitch), float(args.dyaw)
    ]
    target_position = [
        current_right_position[index] + translation_increment[index]
        for index in range(3)
    ]
    target_quaternion = apply_rotation_increment(
        current_right_quaternion,
        rotation_increment,
        frame=args.rotation_frame,
    )

    request = twoArmHandPoseCmd()
    request.hand_poses.header.frame_id = "base_link"
    request.use_custom_ik_param = True
    request.joint_angles_as_q0 = True

    ik_param = ikSolveParam()
    ik_param.major_optimality_tol = 1e-3
    ik_param.major_feasibility_tol = 1e-3
    ik_param.minor_feasibility_tol = 1e-3
    ik_param.major_iterations_limit = 500
    ik_param.oritation_constraint_tol = 1e-3
    ik_param.pos_constraint_tol = 1e-3
    ik_param.pos_cost_weight = 0.0
    ik_param.constraint_mode = IK_MODE_POSITION_AND_ORIENTATION_HARD
    request.ik_param = ik_param

    request.hand_poses.left_pose.pos_xyz = list(
        current_poses.left_pose.pos_xyz
    )
    request.hand_poses.left_pose.quat_xyzw = list(
        current_poses.left_pose.quat_xyzw
    )
    request.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.left_pose.joint_angles = list(current_arm[:7])

    request.hand_poses.right_pose.pos_xyz = target_position
    request.hand_poses.right_pose.quat_xyzw = target_quaternion
    request.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.right_pose.joint_angles = list(current_arm[7:])

    print("Joint mapping: {}".format(mapping))
    print("Current right EEF position: {}".format(
        [round(value, 6) for value in current_right_position]
    ))
    print("Current right EEF quaternion: {}".format(
        [round(value, 6) for value in current_right_quaternion]
    ))
    print("Requested translation increment: {}m".format(
        [round(value, 6) for value in translation_increment]
    ))
    print("Requested RPY increment: {}deg in {} frame".format(
        [round(value, 4) for value in rotation_increment],
        args.rotation_frame,
    ))
    print("Target right EEF position: {}".format(
        [round(value, 6) for value in target_position]
    ))
    print("Target right EEF quaternion: {}".format(
        [round(value, 6) for value in target_quaternion]
    ))
    print("Calling official hard-position/hard-orientation IK; no command publisher exists")

    rospy.wait_for_service(
        "/ik/two_arm_hand_pose_cmd_srv", timeout=float(args.timeout)
    )
    ik_proxy = rospy.ServiceProxy(
        "/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv
    )
    response = ik_proxy(request)
    if not getattr(response, "success", False):
        print("IK success: False")
        print("IK reason: {}".format(getattr(response, "error_reason", "")))
        print("SIX_D_TEACH_DRY_RUN_BLOCKED")
        print("No base, arm or claw command was sent")
        return 2

    raw_solution = extract_ik_solution(response, current_arm)
    raw_left_delta = [
        math.degrees(raw_solution[index] - current_arm[index])
        for index in range(7)
    ]

    # The final controller will freeze the left command explicitly.  Evaluate
    # the right solution using the measured current left arm.
    candidate_arm = list(current_arm)
    candidate_arm[7:14] = raw_solution[7:14]
    predicted_poses = call_fk(candidate_arm)
    predicted_position = [
        float(value) for value in predicted_poses.right_pose.pos_xyz
    ]
    predicted_quaternion = normalize_quaternion(
        predicted_poses.right_pose.quat_xyzw
    )

    right_delta = [
        math.degrees(candidate_arm[index] - current_arm[index])
        for index in range(7, 14)
    ]
    position_residual = position_error(target_position, predicted_position)
    orientation_residual = quaternion_error_degrees(
        target_quaternion, predicted_quaternion
    )
    maximum_right_delta = max(abs(value) for value in right_delta)
    maximum_raw_left_delta = max(abs(value) for value in raw_left_delta)
    values_finite = all(math.isfinite(value) for value in (
        candidate_arm + predicted_position + predicted_quaternion
    ))

    checks = {
        "ik_success": True,
        "position_hard": position_residual <= args.maximum_position_error,
        "orientation_hard": (
            orientation_residual <= args.maximum_orientation_error
        ),
        "right_joint_delta_bounded": (
            maximum_right_delta <= args.maximum_right_joint_delta
        ),
        "left_command_frozen": True,
        "values_finite": values_finite,
    }

    print("IK success: True")
    print("Raw IK left-arm maximum delta: {:.3f}deg (will be frozen)".format(
        maximum_raw_left_delta
    ))
    print("Right-arm joint delta: {}deg".format(
        [round(value, 3) for value in right_delta]
    ))
    print("Maximum right-arm delta: {:.3f}deg".format(maximum_right_delta))
    print("Predicted right EEF position: {}".format(
        [round(value, 6) for value in predicted_position]
    ))
    print("Predicted right EEF quaternion: {}".format(
        [round(value, 6) for value in predicted_quaternion]
    ))
    print("Position residual: {:.6f}m".format(position_residual))
    print("Orientation residual: {:.3f}deg".format(orientation_residual))
    print("Safety checks: {}".format(checks))

    if args.save_plan and all(checks.values()):
        reference = rospy.get_param(REFERENCE_PARAM, None)
        if not isinstance(reference, (list, tuple)) or len(reference) != 14:
            raise RuntimeError("arm command reference is unavailable")
        target_reference = [
            float(reference[index])
            + math.degrees(candidate_arm[index] - current_arm[index])
            for index in range(14)
        ]
        rospy.set_param(PLAN_PARAM, {
            "source_arm_rad": current_arm,
            "target_arm_rad": candidate_arm,
            "source_reference_deg": [float(value) for value in reference],
            "target_reference_deg": target_reference,
            "target_position": target_position,
            "target_quaternion_xyzw": target_quaternion,
            "translation_increment": translation_increment,
            "rpy_increment_deg": rotation_increment,
            "rotation_frame": args.rotation_frame,
            "position_residual_m": position_residual,
            "orientation_residual_deg": orientation_residual,
        })
        print("Calculation-only plan saved to {}".format(PLAN_PARAM))

    if all(checks.values()):
        if max(abs(value) for value in translation_increment + rotation_increment) < 1e-12:
            print("SIX_D_ZERO_POSE_IK_OK")
        else:
            print("SIX_D_TEACH_DRY_RUN_OK")
        print("No base, arm or claw command was sent")
        return 0

    print("SIX_D_TEACH_DRY_RUN_BLOCKED")
    print("No base, arm or claw command was sent")
    return 2


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

