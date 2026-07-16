#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan one bounded fixed-wrist correction of the Scene3 gripper axis.

This program is analysis-only.  It reads the current arm state, tray target,
and gripper TF geometry, then uses the official FK service to find a small
right shoulder/elbow correction.  The left arm and all wrist joints are held.
No ROS publisher or claw service is created.
"""

from __future__ import print_function

import argparse
import math

import numpy as np


RIGHT_PROXIMAL_INDICES = (7, 8, 9, 10)
RIGHT_WRIST_INDICES = (11, 12, 13)
PLAN_PARAM = "/challenge_cup_task_template/scene3/gripper_axis_plan"
TARGET_PARAM = "/challenge_cup_task_template/scene3/locked_target_base_xyz"
WRIST_TARGET_ODOM_PARAM = (
    "/challenge_cup_task_template/scene3/wrist_target_odom_xyz"
)

GLOBAL_LOWER = np.deg2rad(np.array([-180.0, -120.0, -90.0, -120.0]))
GLOBAL_UPPER = np.deg2rad(np.array([40.0, 60.0, 90.0, 0.0]))


def normalize(vector, name="vector"):
    result = np.asarray(vector, dtype=float).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise ValueError("{} contains a non-finite value".format(name))
    norm = float(np.linalg.norm(result))
    if norm < 1e-9:
        raise ValueError("{} has zero length".format(name))
    return result / norm


def quaternion_to_matrix(quaternion_xyzw):
    x, y, z, w = normalize(quaternion_xyzw, "quaternion")
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def angle_degrees(first, second):
    dot = float(np.clip(np.dot(normalize(first), normalize(second)), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def target_axis_standoff(target, midpoint, axis):
    """Signed target distance along the physical gripper forward axis."""

    return float(
        np.dot(
            np.asarray(target, dtype=float) - np.asarray(midpoint, dtype=float),
            normalize(axis, "gripper_axis"),
        )
    )


def step_axis_toward(current_axis, goal_axis, maximum_step_degrees):
    current = normalize(current_axis, "current_axis")
    goal = normalize(goal_axis, "goal_axis")
    dot = float(np.clip(np.dot(current, goal), -1.0, 1.0))
    angle = math.acos(dot)
    step = min(angle, math.radians(float(maximum_step_degrees)))
    if angle < 1e-9:
        return current.copy(), 0.0, 0.0
    if math.pi - angle < 1e-6:
        raise ValueError("current and goal axes are opposite; bounded SLERP is ambiguous")
    fraction = step / angle
    sin_angle = math.sin(angle)
    stepped = (
        math.sin((1.0 - fraction) * angle) / sin_angle * current
        + math.sin(fraction * angle) / sin_angle * goal
    )
    return normalize(stepped, "stepped_axis"), math.degrees(angle), math.degrees(step)


def extract_arm_joints(sensor_message):
    joint_q = list(sensor_message.joint_data.joint_q)
    if len(joint_q) >= 27:
        return np.asarray(joint_q[13:27], dtype=float)
    if len(joint_q) >= 26:
        return np.asarray(joint_q[12:26], dtype=float)
    raise RuntimeError("joint_q length is too short: {}".format(len(joint_q)))


def solve_axis_step(
        fk_pose,
        current_arm,
        actual_midpoint,
        actual_axis,
        tray_xyz,
        maximum_axis_step_degrees=5.0,
        maximum_joint_delta_degrees=8.0,
        orientation_weight_m=0.25,
        finite_difference_rad=0.003,
        damping=0.02,
        max_iterations=40):
    """Return one analysis-only shoulder/elbow correction.

    ``fk_pose`` accepts 14 arm joints in radians and returns ``(xyz, xyzw)``.
    Current measured gripper TF calibrates the fixed transform between the
    official FK frame and the midpoint/axis used by the physical gripper.
    """

    arm0 = np.asarray(current_arm, dtype=float).reshape(-1)
    if arm0.size != 14:
        raise ValueError("current_arm must contain 14 values")
    midpoint0 = np.asarray(actual_midpoint, dtype=float).reshape(3)
    axis0 = normalize(actual_axis, "actual_axis")
    tray = np.asarray(tray_xyz, dtype=float).reshape(3)
    direction0 = normalize(tray - midpoint0, "midpoint_to_tray")
    target_axis, full_angle, planned_step = step_axis_toward(
        axis0, direction0, maximum_axis_step_degrees
    )

    fk_position0, fk_quaternion0 = fk_pose(arm0.copy())
    fk_position0 = np.asarray(fk_position0, dtype=float).reshape(3)
    rotation0 = quaternion_to_matrix(fk_quaternion0)
    midpoint_local = rotation0.T.dot(midpoint0 - fk_position0)
    axis_local = normalize(rotation0.T.dot(axis0), "axis_local")

    def predict(candidate):
        position, quaternion = fk_pose(candidate.copy())
        position = np.asarray(position, dtype=float).reshape(3)
        rotation = quaternion_to_matrix(quaternion)
        midpoint = position + rotation.dot(midpoint_local)
        axis = normalize(rotation.dot(axis_local), "predicted_axis")
        return midpoint, axis

    proximal = np.asarray(RIGHT_PROXIMAL_INDICES, dtype=int)
    joint_window = math.radians(float(maximum_joint_delta_degrees))
    lower = np.maximum(GLOBAL_LOWER, arm0[proximal] - joint_window)
    upper = np.minimum(GLOBAL_UPPER, arm0[proximal] + joint_window)

    def output(candidate):
        midpoint, axis = predict(candidate)
        return np.concatenate([midpoint, float(orientation_weight_m) * axis])

    desired_output = np.concatenate([
        midpoint0,
        float(orientation_weight_m) * target_axis,
    ])

    arm = arm0.copy()
    current_output = output(arm)
    best_arm = arm.copy()
    best_output = current_output.copy()
    best_cost = float(np.linalg.norm(desired_output - current_output))
    current_damping = float(damping)
    stalled = 0
    fk_calls = 1

    for _ in range(int(max_iterations)):
        residual = desired_output - current_output
        if float(np.linalg.norm(residual)) < 1e-4:
            break

        jacobian = np.zeros((6, 4), dtype=float)
        for column, arm_index in enumerate(proximal):
            positive = upper[column] - arm[arm_index]
            negative = arm[arm_index] - lower[column]
            if positive >= finite_difference_rad:
                delta = float(finite_difference_rad)
            elif negative >= finite_difference_rad:
                delta = -float(finite_difference_rad)
            else:
                continue
            candidate = arm.copy()
            candidate[arm_index] += delta
            jacobian[:, column] = (output(candidate) - current_output) / delta
            fk_calls += 1

        normal = jacobian.T.dot(jacobian) + (current_damping ** 2) * np.eye(4)
        try:
            step = np.linalg.solve(normal, jacobian.T.dot(residual))
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(jacobian).dot(residual)
        step = np.clip(step, -math.radians(2.0), math.radians(2.0))

        accepted = False
        current_cost = float(np.linalg.norm(residual))
        for scale in (1.0, 0.5, 0.25, 0.125):
            candidate = arm.copy()
            candidate[proximal] = np.clip(
                arm[proximal] + scale * step,
                lower,
                upper,
            )
            candidate_output = output(candidate)
            fk_calls += 1
            candidate_cost = float(np.linalg.norm(desired_output - candidate_output))
            if candidate_cost + 1e-8 < current_cost:
                arm = candidate
                current_output = candidate_output
                current_damping = max(0.003, current_damping * 0.75)
                stalled = 0
                accepted = True
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_arm = candidate.copy()
                    best_output = candidate_output.copy()
                break

        if not accepted:
            stalled += 1
            current_damping = min(0.5, current_damping * 2.0)
            if stalled >= 6:
                break

    predicted_midpoint = best_output[:3]
    predicted_axis = normalize(
        best_output[3:] / float(orientation_weight_m),
        "best_predicted_axis",
    )
    predicted_direction = normalize(tray - predicted_midpoint, "predicted_midpoint_to_tray")
    before_angle = angle_degrees(axis0, direction0)
    after_angle = angle_degrees(predicted_axis, predicted_direction)
    joint_delta = best_arm - arm0

    return {
        "arm_joints_rad": best_arm,
        "joint_delta_rad": joint_delta,
        "predicted_midpoint": predicted_midpoint,
        "predicted_axis": predicted_axis,
        "target_axis": target_axis,
        "midpoint_shift_m": float(np.linalg.norm(predicted_midpoint - midpoint0)),
        "before_axis_error_deg": before_angle,
        "after_axis_error_deg": after_angle,
        "axis_reduction_deg": before_angle - after_angle,
        "full_goal_angle_deg": full_angle,
        "planned_axis_step_deg": planned_step,
        "fk_calls": fk_calls,
        "cost": best_cost,
    }


def run_ros(args):
    import rospy
    import tf2_ros
    from geometry_msgs.msg import PointStamped
    from kuavo_msgs.msg import sensorsData
    from kuavo_msgs.srv import fkSrv

    rospy.init_node("scene3_gripper_axis_plan", anonymous=True)
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(2.0)

    def frame_xyz(frame_name):
        transform = tf_buffer.lookup_transform(
            "base_link", frame_name, rospy.Time(0), rospy.Duration(3.0)
        )
        translation = transform.transform.translation
        return np.array([translation.x, translation.y, translation.z], dtype=float)

    def point_in_base(xyz, source_frame):
        point = PointStamped()
        point.header.frame_id = str(source_frame)
        point.header.stamp = rospy.Time(0)
        point.point.x = float(xyz[0])
        point.point.y = float(xyz[1])
        point.point.z = float(xyz[2])
        transformed = tf_buffer.transform(
            point, "base_link", rospy.Duration(3.0)
        )
        return np.array(
            [
                transformed.point.x,
                transformed.point.y,
                transformed.point.z,
            ],
            dtype=float,
        )

    samples = []
    for _ in range(5):
        message = rospy.wait_for_message(
            "/sensors_data_raw", sensorsData, timeout=float(args.timeout)
        )
        samples.append(extract_arm_joints(message))
        rospy.sleep(0.05)
    current_arm = np.median(np.asarray(samples), axis=0)

    gripper_base = frame_xyz("right_gripper_base")
    left_finger = frame_xyz("right_gripper_left_inner_finger")
    right_finger = frame_xyz("right_gripper_right_inner_finger")
    midpoint = 0.5 * (left_finger + right_finger)
    actual_axis = normalize(midpoint - gripper_base, "gripper_axis")
    target_odom = None
    if args.target_source == "wrist":
        target_odom = np.asarray(
            rospy.get_param(args.wrist_target_odom_param), dtype=float
        )
        if target_odom.shape != (3,) or not np.all(np.isfinite(target_odom)):
            raise RuntimeError("saved wrist target odom coordinate is invalid")
        tray = point_in_base(target_odom, "odom")
        target_label = "fixed wrist target in odom"
    else:
        tray = np.asarray(rospy.get_param(TARGET_PARAM), dtype=float)
        target_label = "head-camera tray target in base_link"

    rospy.wait_for_service("/ik/fk_srv", timeout=float(args.timeout))
    fk_proxy = rospy.ServiceProxy("/ik/fk_srv", fkSrv, persistent=True)

    def fk_pose(arm):
        response = fk_proxy([float(value) for value in arm])
        if not getattr(response, "success", False):
            raise RuntimeError("/ik/fk_srv returned failure")
        pose = response.hand_poses.right_pose
        return (
            np.asarray(pose.pos_xyz, dtype=float),
            np.asarray(pose.quat_xyzw, dtype=float),
        )

    result = solve_axis_step(
        fk_pose,
        current_arm,
        midpoint,
        actual_axis,
        tray,
        maximum_axis_step_degrees=args.maximum_axis_step,
        maximum_joint_delta_degrees=args.maximum_joint_delta,
        orientation_weight_m=args.orientation_weight,
        max_iterations=args.maximum_iterations,
    )

    delta_deg = np.rad2deg(result["joint_delta_rad"])
    midpoint_shift = result["midpoint_shift_m"]
    remaining_x = float(tray[0] - result["predicted_midpoint"][0])
    predicted_axis_standoff = target_axis_standoff(
        tray,
        result["predicted_midpoint"],
        result["predicted_axis"],
    )
    if args.target_source == "wrist":
        target_ahead = predicted_axis_standoff >= args.minimum_axis_standoff
    else:
        target_ahead = remaining_x >= args.minimum_x_standoff
    checks = {
        "left_arm_frozen": float(np.max(np.abs(delta_deg[:7]))) < 1e-6,
        "wrist_frozen": float(np.max(np.abs(delta_deg[11:14]))) < 1e-6,
        "joint_delta_bounded": float(np.max(np.abs(delta_deg[7:11]))) <= args.maximum_joint_delta + 1e-6,
        "axis_improved": result["axis_reduction_deg"] >= args.minimum_axis_reduction,
        "midpoint_bounded": midpoint_shift <= args.maximum_midpoint_shift,
        "target_still_ahead": target_ahead,
        "values_finite": bool(np.all(np.isfinite(delta_deg))),
    }

    print("鐩爣鏉ユ簮:", target_label)
    if target_odom is not None:
        print("鍥哄畾鑵曢儴鐩爣odom:", np.round(target_odom, 4).tolist())
    print("鏂欑洏鎶撳彇鐐?", np.round(tray, 4).tolist())
    print("褰撳墠鍐呬晶鎵嬫寚涓偣:", np.round(midpoint, 4).tolist())
    print("褰撳墠澶圭埅杞?", np.round(actual_axis, 4).tolist())
    print("褰撳墠澶圭埅鏈濆悜璇樊: {:.2f}deg".format(result["before_axis_error_deg"]))
    print("鏈鍙鍒掔籂姝? {:.2f}deg".format(result["planned_axis_step_deg"]))
    print("鍙宠偐鑲樿鍒掑閲?", np.round(delta_deg[7:11], 3).tolist())
    print("鑵曢儴璁″垝澧為噺:", np.round(delta_deg[11:14], 6).tolist())
    print("宸﹁噦鏈€澶ц鍒掑閲? {:.6f}deg".format(float(np.max(np.abs(delta_deg[:7])))))
    print("棰勬祴鎵嬫寚涓偣:", np.round(result["predicted_midpoint"], 4).tolist())
    print("棰勬祴澶圭埅杞?", np.round(result["predicted_axis"], 4).tolist())
    print("棰勬祴涓偣绉诲姩: {:.1f}mm".format(midpoint_shift * 1000.0))
    print("棰勬祴澶圭埅鏈濆悜璇樊: {:.2f}deg -> {:.2f}deg".format(
        result["before_axis_error_deg"], result["after_axis_error_deg"]
    ))
    print("棰勬祴鏀瑰杽: {:.2f}deg".format(result["axis_reduction_deg"]))
    print("棰勬祴涓偣璺濇枡鐩榅鏂瑰悜浠嶆湁: {:.1f}mm".format(remaining_x * 1000.0))
    print("棰勬祴鐩爣娌垮す鐖酱浠嶅湪鍓嶆柟: {:.1f}mm".format(
        predicted_axis_standoff * 1000.0
    ))
    print("瀹夊叏妫€鏌?", checks)

    if all(checks.values()):
        plan = {
            "baseline_arm_rad": current_arm.tolist(),
            "joint_delta_deg": delta_deg.tolist(),
            "before_axis_error_deg": float(result["before_axis_error_deg"]),
            "predicted_axis_error_deg": float(result["after_axis_error_deg"]),
            "predicted_midpoint": result["predicted_midpoint"].tolist(),
            "tray_xyz": tray.tolist(),
            "target_source": str(args.target_source),
            "target_axis_standoff_m": float(predicted_axis_standoff),
        }
        if target_odom is not None:
            plan["target_odom_xyz"] = target_odom.tolist()
        rospy.set_param(PLAN_PARAM, plan)
        print("GRIPPER_AXIS_5DEG_PLAN_OK")
        print("Plan saved; no base, arm or claw command was sent")
        return 0

    if rospy.has_param(PLAN_PARAM):
        rospy.delete_param(PLAN_PARAM)
    print("GRIPPER_AXIS_PLAN_BLOCKED")
    print("No base, arm or claw command was sent")
    return 2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plan one fixed-wrist Scene3 gripper-axis correction; no commands"
    )
    parser.add_argument(
        "--target-source",
        choices=("head", "wrist"),
        default="head",
        help="align to the head target or the fixed wrist-depth odom point",
    )
    parser.add_argument(
        "--wrist-target-odom-param",
        default=WRIST_TARGET_ODOM_PARAM,
    )
    parser.add_argument("--maximum-axis-step", type=float, default=5.0)
    parser.add_argument("--maximum-joint-delta", type=float, default=8.0)
    parser.add_argument("--orientation-weight", type=float, default=0.25)
    parser.add_argument("--maximum-iterations", type=int, default=40)
    parser.add_argument("--minimum-axis-reduction", type=float, default=2.5)
    parser.add_argument("--maximum-midpoint-shift", type=float, default=0.020)
    parser.add_argument("--minimum-x-standoff", type=float, default=0.100)
    parser.add_argument("--minimum-axis-standoff", type=float, default=0.005)
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


