#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate the original senior Scene3 arm chain without commanding hardware.

The calculation starts from the senior-authored ready joint pose and propagates
each IK result as the next stage's q0:

    ready -> coarse standby -> pregrasp -> touch -> grasp -> lift -> retreat

Only the official FK and IK services are called.  This program deliberately
does not create publishers for ``/cmd_vel``, ``/kuavo_arm_traj`` or the claw,
so it cannot move the base, arms or gripper.
"""

from __future__ import print_function

import argparse
import importlib.util
import math
import os
from types import SimpleNamespace


DEFAULT_TARGET_PARAM = \
    "/challenge_cup_task_template/scene3/locked_target_base_xyz"


def load_senior_module(source_path):
    source_path = os.path.abspath(source_path)
    if not os.path.isfile(source_path):
        raise RuntimeError("senior Scene3 source not found: {}".format(
            source_path
        ))
    spec = importlib.util.spec_from_file_location(
        "scene3_senior_chain_source", source_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "SCENE3_READY_ARM_POSE",
        "RIGHT_GRIPPER_QUAT_XYZW",
        "Scene3Task",
        "build_ik_param",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError("senior source is missing: {}".format(missing))
    return module


def quaternion_angle_degrees(first_xyzw, second_xyzw):
    first = [float(value) for value in first_xyzw]
    second = [float(value) for value in second_xyzw]
    if len(first) != 4 or len(second) != 4:
        raise ValueError("quaternions must contain four xyzw values")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        raise ValueError("quaternions must be nonzero")
    dot = sum(
        first[index] * second[index] for index in range(4)
    ) / (first_norm * second_norm)
    dot = min(1.0, max(-1.0, abs(dot)))
    return math.degrees(2.0 * math.acos(dot))


def vector_distance(first, second):
    if len(first) != 3 or len(second) != 3:
        raise ValueError("positions must contain three values")
    return math.sqrt(sum(
        (float(first[index]) - float(second[index])) ** 2
        for index in range(3)
    ))


def joint_delta_degrees(before, after):
    if len(before) != 14 or len(after) != 14:
        raise ValueError("arm joint vectors must contain fourteen values")
    return [
        math.degrees(float(after[index]) - float(before[index]))
        for index in range(14)
    ]


def build_stage_targets(senior, target_xyz):
    values = [float(value) for value in target_xyz]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("locked target must be a finite xyz vector")
    target_message = SimpleNamespace(
        point=SimpleNamespace(x=values[0], y=values[1], z=values[2])
    )
    coarse = senior.Scene3Task.build_coarse_recognition_arm_target(
        None, target_message
    )
    grasp_targets = senior.Scene3Task.build_scene3_grasp_targets(
        None, target_message
    )
    return [
        ("coarse_standby", list(coarse)),
        ("pregrasp", list(grasp_targets["pregrasp"])),
        ("touch", list(grasp_targets["touch"])),
        ("grasp", list(grasp_targets["grasp"])),
        ("lift", list(grasp_targets["lift"])),
        ("retreat", list(grasp_targets["retreat"])),
    ]


def response_arm_joints(response):
    values = list(getattr(response, "q_arm", []))
    if len(values) >= 14:
        return [float(value) for value in values[:14]]
    left = list(response.hand_poses.left_pose.joint_angles)
    right = list(response.hand_poses.right_pose.joint_angles)
    if len(left) == 7 and len(right) == 7:
        return [float(value) for value in left + right]
    raise RuntimeError("IK response does not contain fourteen arm joints")


def set_pose(pose_message, position, quaternion, joints):
    pose_message.pos_xyz = [float(value) for value in position]
    pose_message.quat_xyzw = [float(value) for value in quaternion]
    pose_message.elbow_pos_xyz = [0.0, 0.0, 0.0]
    pose_message.joint_angles = [float(value) for value in joints]


def solve_stage(senior, ik_proxy, fk_proxy, q0, target_xyz, target_quaternion):
    from kuavo_msgs.msg import twoArmHandPoseCmd

    current_fk = fk_proxy(list(q0))
    if not getattr(current_fk, "success", False):
        raise RuntimeError("FK failed for the stage q0")

    request = twoArmHandPoseCmd()
    # Preserve the original senior request convention.  In this competition
    # image frame=2 is the official Local Frame enum.
    if hasattr(request, "frame"):
        request.frame = 2
    request.use_custom_ik_param = True
    request.joint_angles_as_q0 = True
    request.ik_param = senior.build_ik_param()

    poses = current_fk.hand_poses
    set_pose(
        request.hand_poses.left_pose,
        poses.left_pose.pos_xyz,
        poses.left_pose.quat_xyzw,
        q0[:7],
    )
    set_pose(
        request.hand_poses.right_pose,
        target_xyz,
        target_quaternion,
        q0[7:],
    )

    response = ik_proxy(request)
    if not getattr(response, "success", False):
        raise RuntimeError("IK failed: {}".format(
            getattr(response, "error_reason", "")
        ))
    solved = response_arm_joints(response)
    predicted_fk = fk_proxy(solved)
    if not getattr(predicted_fk, "success", False):
        raise RuntimeError("FK failed for the solved stage")
    return solved, predicted_fk.hand_poses


def stage_record(name, target, quaternion, before, solved, poses):
    delta = joint_delta_degrees(before, solved)
    actual_position = [float(value) for value in poses.right_pose.pos_xyz]
    actual_quaternion = [float(value) for value in poses.right_pose.quat_xyzw]
    return {
        "name": str(name),
        "target": [float(value) for value in target],
        "actual_position": actual_position,
        "position_error_m": vector_distance(actual_position, target),
        "orientation_error_deg": quaternion_angle_degrees(
            actual_quaternion, quaternion
        ),
        "left_delta_deg": delta[:7],
        "right_delta_deg": delta[7:],
        "maximum_left_delta_deg": max(abs(value) for value in delta[:7]),
        "maximum_right_delta_deg": max(abs(value) for value in delta[7:]),
    }


def assess_records(records, maximum_position_error_m=0.005,
                   maximum_orientation_error_deg=10.0,
                   maximum_right_step_deg=50.0,
                   maximum_left_step_deg=10.0):
    checks = {
        "all_positions_reached": all(
            record["position_error_m"] <= maximum_position_error_m
            for record in records
        ),
        "all_orientations_reached": all(
            record["orientation_error_deg"] <= maximum_orientation_error_deg
            for record in records
        ),
        "right_steps_bounded": all(
            record["maximum_right_delta_deg"] <= maximum_right_step_deg
            for record in records
        ),
        "left_arm_nearly_held": all(
            record["maximum_left_delta_deg"] <= maximum_left_step_deg
            for record in records
        ),
    }
    return all(checks.values()), checks


def run(args):
    import rospy
    from kuavo_msgs.srv import fkSrv, twoArmHandPoseCmdSrv

    rospy.init_node("scene3_senior_chain_dry_run", anonymous=True)
    senior = load_senior_module(args.senior_file)

    if not rospy.has_param(args.target_param):
        raise RuntimeError("locked near target parameter is missing: {}".format(
            args.target_param
        ))
    target = [float(value) for value in rospy.get_param(args.target_param)]
    stages = build_stage_targets(senior, target)
    ready = [
        math.radians(float(value))
        for value in senior.SCENE3_READY_ARM_POSE
    ]
    if len(ready) != 14:
        raise RuntimeError("senior ready pose does not contain fourteen joints")
    quaternion = [
        float(value) for value in senior.RIGHT_GRIPPER_QUAT_XYZW
    ]

    rospy.wait_for_service(args.fk_service, timeout=args.timeout)
    rospy.wait_for_service(args.ik_service, timeout=args.timeout)
    fk_proxy = rospy.ServiceProxy(args.fk_service, fkSrv)
    ik_proxy = rospy.ServiceProxy(args.ik_service, twoArmHandPoseCmdSrv)

    print("Locked near tray:", [round(value, 4) for value in target])
    print("Starting from the original senior ready pose")
    print("IK mode: position hard + orientation soft (exact senior setting)")
    print("Calculation only: no base, arm or claw publisher exists")

    current = list(ready)
    records = []
    for name, stage_target in stages:
        try:
            solved, poses = solve_stage(
                senior,
                ik_proxy,
                fk_proxy,
                current,
                stage_target,
                quaternion,
            )
        except Exception as error:
            print("Stage {}: IK_BLOCKED {}".format(name, error))
            print("SENIOR_CHAIN_DRY_RUN_BLOCKED")
            print("No control command was sent")
            return 2
        record = stage_record(
            name,
            stage_target,
            quaternion,
            current,
            solved,
            poses,
        )
        records.append(record)
        current = list(solved)
        print(
            "Stage {}: target={} actual={} pos_error={:.4f}m "
            "ori_error={:.2f}deg left_step={:.2f}deg right_step={:.2f}deg".format(
                name,
                [round(value, 4) for value in record["target"]],
                [round(value, 4) for value in record["actual_position"]],
                record["position_error_m"],
                record["orientation_error_deg"],
                record["maximum_left_delta_deg"],
                record["maximum_right_delta_deg"],
            )
        )

    ok, checks = assess_records(
        records,
        maximum_position_error_m=args.maximum_position_error,
        maximum_orientation_error_deg=args.maximum_orientation_error,
        maximum_right_step_deg=args.maximum_right_step,
        maximum_left_step_deg=args.maximum_left_step,
    )
    print("Senior chain checks:", checks)
    print("SENIOR_CHAIN_DRY_RUN_COMPLETE")
    if ok:
        print("SENIOR_CHAIN_GEOMETRY_OK")
    else:
        print("SENIOR_CHAIN_NEEDS_IK_ORIENTATION_FIX")
    print("No control command was sent")
    return 0


def build_parser():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--senior-file",
        default=os.path.join(script_dir, "challenge_task_3.py"),
    )
    parser.add_argument("--target-param", default=DEFAULT_TARGET_PARAM)
    parser.add_argument("--fk-service", default="/ik/fk_srv")
    parser.add_argument(
        "--ik-service", default="/ik/two_arm_hand_pose_cmd_srv"
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--maximum-position-error", type=float, default=0.005)
    parser.add_argument("--maximum-orientation-error", type=float, default=10.0)
    parser.add_argument("--maximum-right-step", type=float, default=50.0)
    parser.add_argument("--maximum-left-step", type=float, default=10.0)
    return parser


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
