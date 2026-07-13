#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-odom-y", type=float, default=0.0)
    parser.add_argument("--lateral-tolerance", type=float, default=0.01)
    parser.add_argument("--max-lateral-pulse", type=float, default=0.02)
    parser.add_argument("--lateral-speed", type=float, default=0.02)
    parser.add_argument("--maximum-yaw-deg", type=float, default=3.0)
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def validate_args(args):
    if not math.isfinite(args.target_odom_y):
        raise ValueError("target odom y must be finite")
    if not 0.005 <= args.lateral_tolerance <= 0.03:
        raise ValueError("lateral tolerance must be within [0.005, 0.03] m")
    if not 0.005 <= args.max_lateral_pulse <= 0.03:
        raise ValueError("max lateral pulse must be within [0.005, 0.03] m")
    if not 0.01 <= args.lateral_speed <= 0.03:
        raise ValueError("lateral speed must be within [0.01, 0.03] m/s")
    if not 0.5 <= args.maximum_yaw_deg <= 5.0:
        raise ValueError("maximum yaw must be within [0.5, 5.0] degrees")


def build_plan(args, current_x, current_y, current_yaw_rad):
    values = (current_x, current_y, current_yaw_rad)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("odom pose contains a non-finite value")
    if abs(current_y) > 0.50:
        raise ValueError("current odom y is outside the supported recovery range")

    yaw_deg = math.degrees(current_yaw_rad)
    if abs(yaw_deg) > args.maximum_yaw_deg:
        raise ValueError(
            "yaw {:.3f} deg exceeds {:.3f} deg; lateral-only correction is unsafe".format(
                yaw_deg, args.maximum_yaw_deg
            )
        )

    odom_y_error = args.target_odom_y - current_y
    base_y_to_odom_y = math.cos(current_yaw_rad)
    if abs(base_y_to_odom_y) < 0.95:
        raise ValueError("base lateral axis is not sufficiently aligned with odom y")

    if abs(odom_y_error) <= args.lateral_tolerance:
        signed_pulse = 0.0
    else:
        required_base_y = odom_y_error / base_y_to_odom_y
        signed_pulse = math.copysign(
            min(abs(required_base_y), args.max_lateral_pulse), required_base_y
        )

    duration = (
        abs(signed_pulse) / args.lateral_speed
        if abs(signed_pulse) > 0.0
        else 0.0
    )
    expected_odom_y = current_y + signed_pulse * base_y_to_odom_y

    return {
        "mode": "execute" if args.execute else "dry_run",
        "source_frame": args.base_frame,
        "target_frame": args.odom_frame,
        "current_odom_x_m": float(current_x),
        "current_odom_y_m": float(current_y),
        "current_yaw_deg": float(yaw_deg),
        "target_odom_y_m": float(args.target_odom_y),
        "odom_y_error_m": float(odom_y_error),
        "lateral_tolerance_m": float(args.lateral_tolerance),
        "planned_base_lateral_pulse_m": float(signed_pulse),
        "planned_direction": (
            "left" if signed_pulse > 0.0 else "right" if signed_pulse < 0.0 else "none"
        ),
        "lateral_speed_mps": float(args.lateral_speed),
        "planned_duration_s": float(duration),
        "expected_odom_y_after_m": float(expected_odom_y),
        "safety": {
            "forward_motion_commanded": False,
            "angular_motion_commanded": False,
            "arm_commanded": False,
            "gripper_commanded": False,
            "maximum_single_lateral_pulse_m": float(args.max_lateral_pulse),
            "requires_post_motion_odom_and_vision_check": True,
        },
    }


def initialize_ros_and_read_pose(args):
    import rospy
    import tf2_ros
    from tf.transformations import euler_from_quaternion

    rospy.init_node("scene3_lateral_alignment", anonymous=True)
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
    time.sleep(1.0)
    transform = buffer.lookup_transform(
        args.odom_frame,
        args.base_frame,
        rospy.Time(0),
        rospy.Duration(3.0),
    )
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    _, _, yaw = euler_from_quaternion(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    )
    return rospy, float(translation.x), float(translation.y), float(yaw)


def execute_lateral_pulse(args, plan, rospy):
    if args.confirmation != "LATERAL_ONLY":
        raise ValueError("execution requires --confirmation LATERAL_ONLY")

    distance = float(plan["planned_base_lateral_pulse_m"])
    duration = float(plan["planned_duration_s"])
    if abs(distance) <= 0.0 or duration <= 0.0:
        plan["execution"] = "no_motion_needed"
        return
    if abs(distance) > args.max_lateral_pulse + 1e-9 or duration > 3.0:
        raise ValueError("planned lateral pulse exceeds safety limit")

    from geometry_msgs.msg import Twist

    publisher = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=10)
    deadline = time.monotonic() + 5.0
    while publisher.get_num_connections() == 0 and time.monotonic() < deadline:
        rospy.sleep(0.05)
    if publisher.get_num_connections() == 0:
        raise RuntimeError("no subscriber connected to {}".format(args.cmd_vel_topic))

    stop = Twist()
    lateral = Twist()
    lateral.linear.y = math.copysign(args.lateral_speed, distance)
    rate = rospy.Rate(20)
    try:
        for _ in range(5):
            publisher.publish(stop)
            rate.sleep()
        started = time.monotonic()
        while not rospy.is_shutdown() and time.monotonic() - started < duration:
            publisher.publish(lateral)
            rate.sleep()
    finally:
        for _ in range(12):
            publisher.publish(stop)
            rate.sleep()
    plan["execution"] = "lateral_pulse_completed"


def main():
    args = parse_args()
    validate_args(args)
    rospy, current_x, current_y, current_yaw = initialize_ros_and_read_pose(args)
    plan = build_plan(args, current_x, current_y, current_yaw)
    if args.execute:
        execute_lateral_pulse(args, plan, rospy)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
