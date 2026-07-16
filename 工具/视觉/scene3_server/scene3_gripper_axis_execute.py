#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute one saved Scene3 gripper-axis correction with rollback gates.

The base and claw are never commanded.  A saved analysis plan is applied to
the retained arm-command reference with a quintic curve.  Actual gripper TF,
arm sensors, and same-tray vision are checked after motion.  A failed check
causes an automatic return to the previous command reference.
"""

from __future__ import print_function

import argparse
import math
import time

import numpy as np

from scene3_gripper_axis_plan import angle_degrees, extract_arm_joints, normalize


CONFIRMATION = "SCENE3_GRIPPER_AXIS_5DEG"
PLAN_PARAM = "/challenge_cup_task_template/scene3/gripper_axis_plan"
REFERENCE_PARAM = "/challenge_cup_task_template/scene3/arm_command_reference_deg"
LOCKED_ODOM_PARAM = "/challenge_cup_task_template/scene3/locked_target_odom_xyz"
BASE_TOPIC = "/challenge_cup_task_template/scene3/grasp_point_base"
ODOM_TOPIC = "/challenge_cup_task_template/scene3/grasp_point_odom"
ARM_NAMES = ["arm_joint_{}".format(index) for index in range(1, 15)]


def quintic(progress):
    value = min(1.0, max(0.0, float(progress)))
    return 10.0 * value ** 3 - 15.0 * value ** 4 + 6.0 * value ** 5


def maximum_pair_distance(points):
    values = np.asarray(points, dtype=float)
    return max(
        float(np.linalg.norm(first - second))
        for first in values for second in values
    )


def run_ros(args):
    import rospy
    import tf2_ros
    from geometry_msgs.msg import PointStamped
    from kuavo_msgs.msg import sensorsData
    from sensor_msgs.msg import JointState

    if not args.execute or args.confirmation != CONFIRMATION:
        raise RuntimeError(
            "execution blocked; pass --execute --confirmation {}".format(
                CONFIRMATION
            )
        )

    rospy.init_node("scene3_gripper_axis_execute", anonymous=True)
    publisher = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    deadline = time.time() + float(args.timeout)
    while publisher.get_num_connections() == 0:
        if time.time() >= deadline:
            raise RuntimeError("/kuavo_arm_traj has no subscriber")
        rospy.sleep(0.1)

    plan = rospy.get_param(PLAN_PARAM)
    previous_reference = np.asarray(rospy.get_param(REFERENCE_PARAM), dtype=float)
    plan_baseline = np.asarray(plan["baseline_arm_rad"], dtype=float)
    delta_deg = np.asarray(plan["joint_delta_deg"], dtype=float)
    locked_odom = np.asarray(rospy.get_param(LOCKED_ODOM_PARAM), dtype=float)
    if previous_reference.size != 14 or plan_baseline.size != 14 or delta_deg.size != 14:
        raise RuntimeError("saved plan or command reference has an invalid length")

    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(2.0)

    def sample_arm(count=5):
        values = []
        for _ in range(int(count)):
            message = rospy.wait_for_message(
                "/sensors_data_raw", sensorsData, timeout=float(args.timeout)
            )
            values.append(extract_arm_joints(message))
            rospy.sleep(0.05)
        return np.median(np.asarray(values), axis=0)

    def frame_xyz(frame_name):
        transform = tf_buffer.lookup_transform(
            "base_link", frame_name, rospy.Time(0), rospy.Duration(3.0)
        )
        value = transform.transform.translation
        return np.array([value.x, value.y, value.z], dtype=float)

    def sample_geometry(count=5):
        midpoints = []
        axes = []
        for _ in range(int(count)):
            gripper_base = frame_xyz("right_gripper_base")
            left = frame_xyz("right_gripper_left_inner_finger")
            right = frame_xyz("right_gripper_right_inner_finger")
            midpoint = 0.5 * (left + right)
            midpoints.append(midpoint)
            axes.append(normalize(midpoint - gripper_base, "gripper_axis"))
            rospy.sleep(0.08)
        midpoint = np.median(np.asarray(midpoints), axis=0)
        axis = normalize(np.median(np.asarray(axes), axis=0), "median_gripper_axis")
        return midpoint, axis

    def collect_points(topic, count=3):
        values = []
        for _ in range(int(count)):
            message = rospy.wait_for_message(
                topic, PointStamped, timeout=float(args.timeout)
            )
            values.append([message.point.x, message.point.y, message.point.z])
        return np.asarray(values, dtype=float)

    def observe_tray():
        base_values = collect_points(BASE_TOPIC)
        odom_values = collect_points(ODOM_TOPIC)
        base = np.median(base_values, axis=0)
        odom = np.median(odom_values, axis=0)
        result = {
            "base": base,
            "odom": odom,
            "base_spread": maximum_pair_distance(base_values),
            "odom_spread": maximum_pair_distance(odom_values),
            "identity_error": float(np.linalg.norm(odom - locked_odom)),
        }
        result["checks"] = {
            "base_stable": result["base_spread"] <= args.maximum_vision_spread,
            "odom_stable": result["odom_spread"] <= args.maximum_vision_spread,
            "same_tray": result["identity_error"] <= args.maximum_identity_error,
        }
        return result

    def publish_once(values):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = ARM_NAMES
        message.position = [float(value) for value in values]
        publisher.publish(message)

    def hold(values, seconds):
        rate = rospy.Rate(args.hz)
        end_time = time.time() + float(seconds)
        while not rospy.is_shutdown() and time.time() < end_time:
            publish_once(values)
            rate.sleep()

    def move(start, target, seconds):
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        steps = max(1, int(float(seconds) * args.hz))
        rate = rospy.Rate(args.hz)
        for index in range(steps + 1):
            alpha = quintic(float(index) / float(steps))
            command = start + alpha * (target - start)
            publish_once(command)
            if index % max(1, int(args.hz / 10.0)) == 0 or index == steps:
                rospy.set_param(REFERENCE_PARAM, command.tolist())
            rate.sleep()
        hold(target, args.hold_seconds)
        rospy.set_param(REFERENCE_PARAM, target.tolist())

    def rollback(target_reference, reason):
        print("ROLLBACK: {}".format(reason))
        move(target_reference, previous_reference, args.motion_seconds)
        rospy.set_param(REFERENCE_PARAM, previous_reference.tolist())
        rospy.sleep(args.vision_settle_seconds)
        restored = sample_arm()
        restored_error = float(np.max(np.abs(
            np.rad2deg(restored - plan_baseline)
        )))
        print("回退后相对方案基准最大误差: {:.3f}deg".format(restored_error))
        print("GRIPPER_AXIS_EXECUTION_ROLLED_BACK")
        print("底盘和夹爪未被控制")

    baseline = sample_arm()
    baseline_error = float(np.max(np.abs(np.rad2deg(baseline - plan_baseline))))
    before_vision = observe_tray()
    before_midpoint, before_axis = sample_geometry()
    before_direction = normalize(
        before_vision["base"] - before_midpoint,
        "before_midpoint_to_tray",
    )
    before_angle = angle_degrees(before_axis, before_direction)

    print("执行前方案基准误差: {:.3f}deg".format(baseline_error))
    print("执行前料盘:", np.round(before_vision["base"], 4).tolist())
    print("执行前内侧手指中点:", np.round(before_midpoint, 4).tolist())
    print("执行前夹爪轴:", np.round(before_axis, 4).tolist())
    print("执行前夹爪朝向误差: {:.2f}deg".format(before_angle))
    print("计划右肩肘增量:", np.round(delta_deg[7:11], 3).tolist())
    print("计划腕部增量:", np.round(delta_deg[11:14], 6).tolist())
    print("执行前视觉检查:", before_vision["checks"])

    before_checks = {
        "plan_fresh": baseline_error <= args.maximum_plan_baseline_error,
        "vision_ready": all(before_vision["checks"].values()),
        "left_delta_zero": float(np.max(np.abs(delta_deg[:7]))) < 1e-6,
        "wrist_delta_zero": float(np.max(np.abs(delta_deg[11:14]))) < 1e-6,
        "proximal_delta_bounded": float(np.max(np.abs(delta_deg[7:11]))) <= args.maximum_command_delta,
    }
    print("执行前安全检查:", before_checks)
    if not all(before_checks.values()):
        print("GRIPPER_AXIS_EXECUTION_BLOCKED_BEFORE_MOTION")
        print("未发送控制命令")
        return 2

    target_reference = previous_reference + delta_deg
    print("开始4秒五次曲线夹爪轴纠偏；底盘和夹爪不动")
    try:
        move(previous_reference, target_reference, args.motion_seconds)
        rospy.sleep(args.vision_settle_seconds)

        after_arm = sample_arm()
        after_vision = observe_tray()
        after_midpoint, after_axis = sample_geometry()
        after_direction = normalize(
            after_vision["base"] - after_midpoint,
            "after_midpoint_to_tray",
        )
        after_angle = angle_degrees(after_axis, after_direction)
        reduction = before_angle - after_angle
        midpoint_motion = float(np.linalg.norm(after_midpoint - before_midpoint))
        actual_delta_deg = np.rad2deg(after_arm - baseline)
        x_standoff = float(after_vision["base"][0] - after_midpoint[0])
    except Exception as exc:
        rollback(target_reference, "post-motion measurement failed: {}".format(exc))
        return 2

    checks = {
        "axis_improved": reduction >= args.minimum_actual_reduction,
        "midpoint_motion_bounded": midpoint_motion <= args.maximum_midpoint_motion,
        "right_proximal_bounded": float(np.max(np.abs(actual_delta_deg[7:11]))) <= args.maximum_actual_proximal_delta,
        "left_arm_bounded": float(np.max(np.abs(actual_delta_deg[:7]))) <= args.maximum_coupled_delta,
        "wrist_bounded": float(np.max(np.abs(actual_delta_deg[11:14]))) <= args.maximum_coupled_delta,
        "tray_still_ahead": x_standoff >= args.minimum_x_standoff,
        "vision_stable": all(after_vision["checks"].values()),
    }

    print("实际右肩肘增量:", np.round(actual_delta_deg[7:11], 3).tolist())
    print("实际腕部最大变化: {:.3f}deg".format(float(np.max(np.abs(actual_delta_deg[11:14])))))
    print("实际左臂最大变化: {:.3f}deg".format(float(np.max(np.abs(actual_delta_deg[:7])))))
    print("执行后料盘:", np.round(after_vision["base"], 4).tolist())
    print("执行后内侧手指中点:", np.round(after_midpoint, 4).tolist())
    print("执行后夹爪轴:", np.round(after_axis, 4).tolist())
    print("夹爪朝向误差: {:.2f}deg -> {:.2f}deg".format(before_angle, after_angle))
    print("实际改善: {:.2f}deg".format(reduction))
    print("手指中点移动: {:.1f}mm".format(midpoint_motion * 1000.0))
    print("中点距料盘X方向仍有: {:.1f}mm".format(x_standoff * 1000.0))
    print("运动后视觉: spread={:.4f}m identity_error={:.4f}m checks={}".format(
        after_vision["base_spread"],
        after_vision["identity_error"],
        after_vision["checks"],
    ))
    print("运动安全检查:", checks)

    if all(checks.values()):
        rospy.set_param(REFERENCE_PARAM, target_reference.tolist())
        if rospy.has_param(PLAN_PARAM):
            rospy.delete_param(PLAN_PARAM)
        print("GRIPPER_AXIS_5DEG_EXECUTION_OK")
        print("本次纠偏已提交；底盘和夹爪未被控制")
        return 0

    rollback(target_reference, "post-motion gripper/vision gate failed")
    return 2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Execute one saved Scene3 gripper-axis correction"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--motion-seconds", type=float, default=4.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--vision-settle-seconds", type=float, default=2.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--maximum-plan-baseline-error", type=float, default=1.0)
    parser.add_argument("--maximum-command-delta", type=float, default=3.0)
    parser.add_argument("--minimum-actual-reduction", type=float, default=1.5)
    parser.add_argument("--maximum-midpoint-motion", type=float, default=0.030)
    parser.add_argument("--maximum-actual-proximal-delta", type=float, default=5.0)
    parser.add_argument("--maximum-coupled-delta", type=float, default=1.5)
    parser.add_argument("--minimum-x-standoff", type=float, default=0.120)
    parser.add_argument("--maximum-vision-spread", type=float, default=0.012)
    parser.add_argument("--maximum-identity-error", type=float, default=0.030)
    return parser


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

