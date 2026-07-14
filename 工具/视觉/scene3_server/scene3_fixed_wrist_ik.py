#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scene3 fixed-wrist IK dry run.

The Scene3 simulator currently follows the first four joints of each arm but
does not follow the three wrist position targets.  This module therefore keeps
the measured right-wrist joints fixed and uses FK feedback to solve only the
right shoulder/elbow joints (right-arm joints 1..4).

The default entry point is analysis-only: it creates no command publishers and
does not control the base, arm, or gripper.
"""

from __future__ import print_function

import argparse
import math

import numpy as np


RIGHT_PROXIMAL_INDICES = (7, 8, 9, 10)
RIGHT_WRIST_INDICES = (11, 12, 13)

# Conservative limits for Kuavo 52 right shoulder/elbow joints, in radians.
DEFAULT_LOWER_BOUNDS = np.deg2rad(np.array([-180.0, -120.0, -90.0, -120.0]))
DEFAULT_UPPER_BOUNDS = np.deg2rad(np.array([40.0, 60.0, 90.0, 0.0]))


def _as_vector(values, length, name):
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size != length:
        raise ValueError("{} must contain {} values, got {}".format(name, length, result.size))
    if not np.all(np.isfinite(result)):
        raise ValueError("{} contains a non-finite value".format(name))
    return result


def solve_fixed_wrist_position(
    fk_position,
    current_arm_joints,
    target_xyz,
    tolerance_m=0.008,
    max_iterations=30,
    finite_difference_rad=0.003,
    damping=0.025,
    maximum_step_rad=math.radians(6.0),
    lower_bounds=None,
    upper_bounds=None,
):
    """Solve right-hand position while preserving both arms' other joints.

    ``fk_position`` accepts a 14-element arm vector in radians and returns the
    right end-effector XYZ in base_link.  The solver never mutates the input.
    """

    arm = _as_vector(current_arm_joints, 14, "current_arm_joints").copy()
    target = _as_vector(target_xyz, 3, "target_xyz")
    lower = _as_vector(
        DEFAULT_LOWER_BOUNDS if lower_bounds is None else lower_bounds,
        4,
        "lower_bounds",
    )
    upper = _as_vector(
        DEFAULT_UPPER_BOUNDS if upper_bounds is None else upper_bounds,
        4,
        "upper_bounds",
    )
    if np.any(lower >= upper):
        raise ValueError("every lower bound must be smaller than its upper bound")

    proximal = np.array(RIGHT_PROXIMAL_INDICES, dtype=int)
    original_wrist = arm[list(RIGHT_WRIST_INDICES)].copy()
    arm[proximal] = np.clip(arm[proximal], lower, upper)

    calls = 0

    def evaluate(candidate):
        nonlocal calls
        calls += 1
        return _as_vector(fk_position(candidate.copy()), 3, "fk_position result")

    position = evaluate(arm)
    initial_error = float(np.linalg.norm(target - position))
    best_arm = arm.copy()
    best_position = position.copy()
    best_error = initial_error
    current_damping = float(damping)
    stalled_iterations = 0
    history = [initial_error]

    for iteration in range(1, int(max_iterations) + 1):
        error_vector = target - position
        error_norm = float(np.linalg.norm(error_vector))
        if error_norm <= float(tolerance_m):
            break

        jacobian = np.zeros((3, 4), dtype=float)
        for column, arm_index in enumerate(proximal):
            candidate = arm.copy()
            available_positive = upper[column] - arm[arm_index]
            available_negative = arm[arm_index] - lower[column]
            if available_positive >= finite_difference_rad:
                delta = float(finite_difference_rad)
            elif available_negative >= finite_difference_rad:
                delta = -float(finite_difference_rad)
            else:
                continue
            candidate[arm_index] += delta
            jacobian[:, column] = (evaluate(candidate) - position) / delta

        # Damped least squares gives the minimum-norm four-joint correction.
        system = jacobian.dot(jacobian.T) + (current_damping ** 2) * np.eye(3)
        try:
            step = jacobian.T.dot(np.linalg.solve(system, error_vector))
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(jacobian).dot(error_vector)
        step = np.clip(step, -float(maximum_step_rad), float(maximum_step_rad))

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = arm.copy()
            candidate[proximal] = np.clip(arm[proximal] + scale * step, lower, upper)
            candidate_position = evaluate(candidate)
            candidate_error = float(np.linalg.norm(target - candidate_position))
            if candidate_error + 1e-7 < error_norm:
                arm = candidate
                position = candidate_position
                history.append(candidate_error)
                accepted = True
                stalled_iterations = 0
                current_damping = max(0.003, current_damping * 0.7)
                if candidate_error < best_error:
                    best_arm = candidate.copy()
                    best_position = candidate_position.copy()
                    best_error = candidate_error
                break

        if not accepted:
            stalled_iterations += 1
            current_damping = min(0.5, current_damping * 2.5)
            if stalled_iterations >= 5:
                break

    wrist_delta = best_arm[list(RIGHT_WRIST_INDICES)] - original_wrist
    if not np.array_equal(wrist_delta, np.zeros(3)):
        raise RuntimeError("fixed-wrist solver changed a wrist joint")

    return {
        "success": bool(best_error <= float(tolerance_m)),
        "arm_joints_rad": best_arm,
        "position_xyz": best_position,
        "target_xyz": target,
        "initial_error_m": initial_error,
        "final_error_m": best_error,
        "iterations": max(0, len(history) - 1),
        "fk_calls": calls,
        "error_history_m": history,
        "wrist_delta_rad": wrist_delta,
    }


def extract_arm_joints(sensor_message):
    """Extract the 14 arm joints from the Scene3 sensorsData layout."""

    joint_q = list(sensor_message.joint_data.joint_q)
    if len(joint_q) >= 27:
        return np.asarray(joint_q[13:27], dtype=float)
    if len(joint_q) >= 26:
        return np.asarray(joint_q[12:26], dtype=float)
    raise RuntimeError("joint_q length is too short: {}".format(len(joint_q)))


def _collect_stable_target(rospy, point_type, topic, frame_count, timeout_s):
    points = []
    for _ in range(int(frame_count)):
        message = rospy.wait_for_message(topic, point_type, timeout=float(timeout_s))
        points.append([message.point.x, message.point.y, message.point.z])
    points = np.asarray(points, dtype=float)
    median = np.median(points, axis=0)
    maximum_spread = float(np.max(np.linalg.norm(points - median, axis=1)))
    return median, maximum_spread


def ros_main(argv=None):
    parser = argparse.ArgumentParser(description="Scene3 fixed-wrist IK analysis only")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--message-timeout", type=float, default=5.0)
    parser.add_argument("--maximum-target-spread", type=float, default=0.01)
    parser.add_argument("--pregrasp-offset-x", type=float, default=-0.16)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument("--offset-z", type=float, default=0.02)
    parser.add_argument("--tolerance", type=float, default=0.008)
    parser.add_argument("--max-iterations", type=int, default=30)
    args = parser.parse_args(argv)

    import rospy
    from geometry_msgs.msg import PointStamped
    from kuavo_msgs.msg import sensorsData
    from kuavo_msgs.srv import fkSrv

    rospy.init_node("scene3_fixed_wrist_ik_analysis", anonymous=True)
    topic = "/challenge_cup_task_template/scene3/grasp_point_base"
    print("收集{}帧近距离料盘坐标；本脚本不会发布控制命令".format(args.frames))
    tray_xyz, spread = _collect_stable_target(
        rospy, PointStamped, topic, args.frames, args.message_timeout
    )
    if spread > args.maximum_target_spread:
        raise RuntimeError(
            "料盘坐标波动{:.4f}m，超过{:.4f}m，停止计算".format(
                spread, args.maximum_target_spread
            )
        )
    target = tray_xyz + np.array(
        [args.pregrasp_offset_x, args.offset_y, args.offset_z], dtype=float
    )

    sensor = rospy.wait_for_message(
        "/sensors_data_raw", sensorsData, timeout=float(args.message_timeout)
    )
    current_arm = extract_arm_joints(sensor)
    original_wrist = current_arm[list(RIGHT_WRIST_INDICES)].copy()

    rospy.wait_for_service("/ik/fk_srv", timeout=5.0)
    fk_proxy = rospy.ServiceProxy("/ik/fk_srv", fkSrv, persistent=True)

    def fk_position(arm_joints):
        response = fk_proxy([float(value) for value in arm_joints])
        if not getattr(response, "success", False):
            raise RuntimeError("/ik/fk_srv returned failure")
        return response.hand_poses.right_pose.pos_xyz

    result = solve_fixed_wrist_position(
        fk_position,
        current_arm,
        target,
        tolerance_m=args.tolerance,
        max_iterations=args.max_iterations,
    )

    right_before_deg = np.rad2deg(current_arm[7:14])
    right_after_deg = np.rad2deg(result["arm_joints_rad"][7:14])
    wrist_delta_deg = np.rad2deg(
        result["arm_joints_rad"][list(RIGHT_WRIST_INDICES)] - original_wrist
    )
    print("稳定料盘坐标:", np.round(tray_xyz, 4).tolist())
    print("五帧最大波动: {:.4f}m".format(spread))
    print("预抓取目标:", np.round(target, 4).tolist())
    print("当前右臂角度:", np.round(right_before_deg, 1).tolist())
    print("固定腕部解右臂角度:", np.round(right_after_deg, 1).tolist())
    print("腕部角增量:", np.round(wrist_delta_deg, 6).tolist())
    print(
        "FK位置: {}  误差: {:.4f}m  FK调用: {}".format(
            np.round(result["position_xyz"], 4).tolist(),
            result["final_error_m"],
            result["fk_calls"],
        )
    )
    if result["success"]:
        print("FIXED_WRIST_IK_OK：误差小于{:.1f}cm，仍未发布任何控制命令".format(args.tolerance * 100.0))
        return 0
    print("FIXED_WRIST_IK_FAILED：未达到误差门限，禁止执行机械臂")
    return 2


if __name__ == "__main__":
    raise SystemExit(ros_main())

