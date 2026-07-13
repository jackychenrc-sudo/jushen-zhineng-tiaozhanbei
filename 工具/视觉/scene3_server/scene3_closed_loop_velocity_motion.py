#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
from pathlib import Path


def clamp(value, lower, upper):
    return max(float(lower), min(float(upper), float(value)))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def circular_mean_angle(first, second):
    sine = math.sin(first) + math.sin(second)
    cosine = math.cos(first) + math.cos(second)
    if abs(sine) < 1e-12 and abs(cosine) < 1e-12:
        return normalize_angle(first)
    return math.atan2(sine, cosine)


def midpoint_pose(left_pose, right_pose):
    return [
        0.5 * (float(left_pose[0]) + float(right_pose[0])),
        0.5 * (float(left_pose[1]) + float(right_pose[1])),
        circular_mean_angle(float(left_pose[2]), float(right_pose[2])),
    ]


def displacement_in_start_frame(start_pose, current_pose):
    start_x, start_y, start_yaw = [float(value) for value in start_pose]
    current_x, current_y, current_yaw = [float(value) for value in current_pose]
    delta_x = current_x - start_x
    delta_y = current_y - start_y
    cosine = math.cos(start_yaw)
    sine = math.sin(start_yaw)
    return [
        cosine * delta_x + sine * delta_y,
        -sine * delta_x + cosine * delta_y,
        normalize_angle(current_yaw - start_yaw),
    ]


def validate_forward_motion(distance, maximum_distance=0.05):
    if not math.isfinite(float(distance)) or not math.isfinite(
        float(maximum_distance)
    ):
        raise ValueError("closed-loop command contains a non-finite value")
    if not 0.03 <= maximum_distance <= 0.10:
        raise ValueError("maximum distance must be within [0.03, 0.10] m")
    if not 0.03 <= distance <= maximum_distance + 1e-9:
        raise ValueError(
            "forward distance must be within [0.03, maximum distance]"
        )


def controller_command(relative_pose, distance, config):
    forward, lateral, yaw = [float(value) for value in relative_pose]
    forward_error = float(distance) - forward
    if forward_error <= config.stop_margin:
        return [0.0, 0.0, 0.0], True

    forward_speed = clamp(
        config.forward_kp * forward_error,
        config.minimum_forward_speed,
        config.maximum_forward_speed,
    )

    lateral_error = 0.0 if abs(lateral) <= config.lateral_deadband_m else lateral
    lateral_speed = clamp(
        -config.lateral_kp * lateral_error,
        -config.maximum_lateral_speed,
        config.maximum_lateral_speed,
    )

    # Short humanoid steps need direct lateral correction. Steering by yaw alone
    # cannot cancel the side displacement before the current gait cycle settles.
    heading_error = normalize_angle(-yaw)
    if abs(math.degrees(heading_error)) <= config.yaw_deadband_deg:
        yaw_rate = 0.0
    else:
        yaw_rate = clamp(
            config.yaw_kp * heading_error,
            -math.radians(config.maximum_yaw_rate_deg),
            math.radians(config.maximum_yaw_rate_deg),
        )

    heading_scale = max(0.5, math.cos(abs(heading_error)))
    return [forward_speed * heading_scale, lateral_speed, yaw_rate], False


def safety_violation(relative_pose, distance, config):
    forward, lateral, yaw = [float(value) for value in relative_pose]
    if forward < -config.maximum_reverse_m:
        return "unexpected reverse motion"
    if forward > distance + config.maximum_overshoot_m:
        return "forward overshoot exceeded the safety limit"
    if abs(lateral) > config.maximum_lateral_drift_m:
        return "lateral drift exceeded the safety limit"
    if abs(math.degrees(yaw)) > config.maximum_yaw_drift_deg:
        return "yaw drift exceeded the safety limit"
    return None


def post_motion_verification(relative_pose, distance, gait_after, config):
    forward, lateral, yaw = [float(value) for value in relative_pose]
    movement_observed = forward >= config.minimum_valid_motion_m
    geometry_safe = (
        -config.maximum_reverse_m <= forward
        <= distance + config.maximum_overshoot_m
        and abs(lateral) <= config.post_lateral_tolerance_m
        and abs(math.degrees(yaw)) <= config.post_yaw_tolerance_deg
    )
    target_reached = (
        abs(forward - float(distance)) <= config.post_forward_tolerance_m
    )
    returned_to_stance = str(gait_after).strip().lower() == "stance"
    return {
        "movement_observed": movement_observed,
        "geometry_safe": geometry_safe,
        "target_reached": target_reached,
        "returned_to_stance": returned_to_stance,
    }


def execution_succeeded(status, verification):
    return status == "target_gate_reached" and all(verification.values())


def command_plan(distance, args=None):
    left_foot_frame = args.left_foot_frame if args else "leg_l6_link"
    right_foot_frame = args.right_foot_frame if args else "leg_r6_link"
    return {
        "mode": "execute" if args and args.execute else "dry_run",
        "controller_version": "foot_midpoint_holonomic_v4",
        "motion_interface": "cmd_vel_closed_loop",
        "command_topic": args.cmd_vel_topic if args else "/cmd_vel",
        "feedback": (
            "TF odom -> midpoint({}, {})".format(
                left_foot_frame, right_foot_frame
            )
        ),
        "relative_target": {
            "forward_m": float(distance),
            "lateral_m": 0.0,
            "yaw_deg": 0.0,
        },
        "controller": {
            "model": "holonomic_short_step",
            "rate_hz": float(args.rate_hz) if args else 20.0,
            "maximum_forward_speed_mps": (
                float(args.maximum_forward_speed) if args else 0.04
            ),
            "maximum_lateral_speed_mps": (
                float(args.maximum_lateral_speed) if args else 0.02
            ),
            "maximum_yaw_rate_deg_s": (
                float(args.maximum_yaw_rate_deg) if args else 2.0
            ),
            "stop_margin_m": float(args.stop_margin) if args else 0.005,
            "direct_lateral_velocity_enabled": True,
            "cross_track_gain": float(args.lateral_kp) if args else 0.5,
            "path_lookahead_m": (
                float(args.path_lookahead_m) if args else 0.20
            ),
        },
        "safety": {
            "requires_initial_stance": True,
            "maximum_lateral_drift_m": (
                float(args.maximum_lateral_drift_m) if args else 0.035
            ),
            "maximum_yaw_drift_deg": (
                float(args.maximum_yaw_drift_deg) if args else 4.0
            ),
            "maximum_overshoot_m": (
                float(args.maximum_overshoot_m) if args else 0.05
            ),
            "progress_watchdog_s": (
                float(args.progress_watchdog) if args else 3.0
            ),
            "zero_velocity_after_motion": True,
            "explicit_confirmation": "CLOSED_LOOP_VELOCITY",
            "arm_commanded": False,
            "gripper_commanded": False,
        },
        "verification": {
            "requires_post_motion_odometry_check": True,
            "requires_post_motion_vision_check": True,
            "base_link_is_diagnostic_only": True,
        },
    }


def _lookup_pose(buffer, rospy, odom_frame, base_frame, timeout_seconds=2.0):
    from tf.transformations import euler_from_quaternion

    transform = buffer.lookup_transform(
        odom_frame,
        base_frame,
        rospy.Time(0),
        rospy.Duration(timeout_seconds),
    )
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    _, _, yaw = euler_from_quaternion(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    )
    return [float(translation.x), float(translation.y), float(yaw)]


def _lookup_foot_midpoint_pose(buffer, rospy, args, timeout_seconds=2.0):
    left_pose = _lookup_pose(
        buffer,
        rospy,
        args.odom_frame,
        args.left_foot_frame,
        timeout_seconds=timeout_seconds,
    )
    right_pose = _lookup_pose(
        buffer,
        rospy,
        args.odom_frame,
        args.right_foot_frame,
        timeout_seconds=timeout_seconds,
    )
    return midpoint_pose(left_pose, right_pose), left_pose, right_pose


def _current_gait(service_name):
    import rospy
    import rosservice

    service_class = rosservice.get_service_class_by_name(service_name)
    if service_class is None:
        raise RuntimeError("unable to resolve gait service type")
    rospy.wait_for_service(service_name, timeout=3.0)
    response = rospy.ServiceProxy(service_name, service_class)()
    if not getattr(response, "success", False):
        raise RuntimeError("current gait query failed")
    return str(getattr(response, "gait_name", ""))


def _publish_zero(publisher, duration, rate_hz=20.0):
    from geometry_msgs.msg import Twist

    message = Twist()
    deadline = time.monotonic() + float(duration)
    interval = 1.0 / float(rate_hz)
    while time.monotonic() < deadline:
        try:
            publisher.publish(message)
        except BaseException:
            pass
        try:
            time.sleep(interval)
        except BaseException:
            pass


def execute_motion(distance, args):
    import rospy
    import tf2_ros
    from geometry_msgs.msg import Twist

    if args.confirmation != "CLOSED_LOOP_VELOCITY":
        raise ValueError(
            "execution requires --confirmation CLOSED_LOOP_VELOCITY"
        )
    if not rospy.core.is_initialized():
        rospy.init_node(
            "scene3_closed_loop_velocity_motion",
            anonymous=True,
            disable_signals=True,
        )

    gait_before = _current_gait(args.gait_service)
    if gait_before.strip().lower() != "stance":
        raise RuntimeError(
            "closed-loop motion requires stance, current gait is {}".format(
                gait_before
            )
        )

    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
    rospy.sleep(1.0)
    start_pose, start_left_pose, start_right_pose = _lookup_foot_midpoint_pose(
        buffer, rospy, args, timeout_seconds=3.0
    )
    start_base_pose = _lookup_pose(
        buffer, rospy, args.odom_frame, args.base_frame, timeout_seconds=3.0
    )

    publisher = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=1)
    deadline = time.monotonic() + 3.0
    while publisher.get_num_connections() == 0 and time.monotonic() < deadline:
        rospy.sleep(0.05)
    if publisher.get_num_connections() == 0:
        raise RuntimeError(
            "no subscriber connected to {}".format(args.cmd_vel_topic)
        )

    result = command_plan(distance, args)
    result["gait_before"] = gait_before
    result["start_feedback_pose"] = {
        "x_m": start_pose[0],
        "y_m": start_pose[1],
        "yaw_deg": math.degrees(start_pose[2]),
    }
    result["start_foot_poses"] = {
        args.left_foot_frame: start_left_pose,
        args.right_foot_frame: start_right_pose,
    }
    result["start_base_link_pose"] = {
        "x_m": start_base_pose[0],
        "y_m": start_base_pose[1],
        "yaw_deg": math.degrees(start_base_pose[2]),
    }

    started = time.monotonic()
    last_progress_time = started
    best_forward = 0.0
    sample_count = 0
    status = "timeout"
    stop_reason = "motion timed out"
    current_pose = list(start_pose)
    current_base_pose = list(start_base_pose)
    samples = []
    maximum_absolute_lateral = 0.0
    maximum_absolute_yaw = 0.0
    maximum_forward = 0.0
    maximum_base_absolute_lateral = 0.0
    maximum_base_absolute_yaw = 0.0
    rate = rospy.Rate(args.rate_hz)

    def shutdown_stop():
        _publish_zero(publisher, 0.3, args.rate_hz)

    rospy.on_shutdown(shutdown_stop)
    try:
        while not rospy.is_shutdown() and time.monotonic() - started < args.timeout:
            current_pose, _, _ = _lookup_foot_midpoint_pose(
                buffer, rospy, args, timeout_seconds=1.0
            )
            current_base_pose = _lookup_pose(
                buffer, rospy, args.odom_frame, args.base_frame, timeout_seconds=1.0
            )
            relative = displacement_in_start_frame(start_pose, current_pose)
            base_relative = displacement_in_start_frame(
                start_base_pose, current_base_pose
            )
            sample_count += 1
            maximum_absolute_lateral = max(
                maximum_absolute_lateral, abs(relative[1])
            )
            maximum_absolute_yaw = max(maximum_absolute_yaw, abs(relative[2]))
            maximum_forward = max(maximum_forward, relative[0])
            maximum_base_absolute_lateral = max(
                maximum_base_absolute_lateral, abs(base_relative[1])
            )
            maximum_base_absolute_yaw = max(
                maximum_base_absolute_yaw, abs(base_relative[2])
            )

            violation = safety_violation(relative, distance, args)
            if violation:
                samples.append(
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "relative_pose": [
                            relative[0],
                            relative[1],
                            math.degrees(relative[2]),
                        ],
                        "base_link_relative_pose": [
                            base_relative[0],
                            base_relative[1],
                            math.degrees(base_relative[2]),
                        ],
                        "command": [0.0, 0.0, 0.0],
                        "event": "safety_stop",
                    }
                )
                status = "safety_stop"
                stop_reason = violation
                break

            if relative[0] > best_forward + args.progress_epsilon:
                best_forward = relative[0]
                last_progress_time = time.monotonic()
            elif time.monotonic() - last_progress_time > args.progress_watchdog:
                status = "no_progress"
                stop_reason = "forward progress watchdog expired"
                break

            command_values, should_stop = controller_command(
                relative, distance, args
            )
            if should_stop:
                samples.append(
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "relative_pose": [
                            relative[0],
                            relative[1],
                            math.degrees(relative[2]),
                        ],
                        "base_link_relative_pose": [
                            base_relative[0],
                            base_relative[1],
                            math.degrees(base_relative[2]),
                        ],
                        "command": [0.0, 0.0, 0.0],
                        "event": "target_gate_reached",
                    }
                )
                status = "target_gate_reached"
                stop_reason = "forward stop gate reached"
                break

            samples.append(
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "relative_pose": [
                        relative[0],
                        relative[1],
                        math.degrees(relative[2]),
                    ],
                    "base_link_relative_pose": [
                        base_relative[0],
                        base_relative[1],
                        math.degrees(base_relative[2]),
                    ],
                    "command": [
                        command_values[0],
                        command_values[1],
                        math.degrees(command_values[2]),
                    ],
                    "event": "control",
                }
            )

            command = Twist()
            command.linear.x = command_values[0]
            command.linear.y = command_values[1]
            command.angular.z = command_values[2]
            publisher.publish(command)
            rate.sleep()
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "operator interrupted motion"
    finally:
        _publish_zero(publisher, args.zero_duration, args.rate_hz)

    rospy.sleep(0.4)
    current_pose, final_left_pose, final_right_pose = _lookup_foot_midpoint_pose(
        buffer, rospy, args, timeout_seconds=2.0
    )
    current_base_pose = _lookup_pose(
        buffer, rospy, args.odom_frame, args.base_frame, timeout_seconds=2.0
    )
    relative = displacement_in_start_frame(start_pose, current_pose)
    base_relative = displacement_in_start_frame(
        start_base_pose, current_base_pose
    )
    gait_after = _current_gait(args.gait_service)
    verification = post_motion_verification(
        relative, distance, gait_after, args
    )
    success = execution_succeeded(status, verification)
    if status == "target_gate_reached" and not success:
        status = "post_motion_verification_failed"
        stop_reason = "odometry or stance verification failed"
    elif success:
        status = "completed"
        stop_reason = "bounded motion completed and verified"

    result.update(
        {
            "status": status,
            "success": success,
            "stop_reason": stop_reason,
            "elapsed_seconds": time.monotonic() - started,
            "sample_count": sample_count,
            "gait_after": gait_after,
            "final_feedback_pose": {
                "x_m": current_pose[0],
                "y_m": current_pose[1],
                "yaw_deg": math.degrees(current_pose[2]),
            },
            "final_foot_poses": {
                args.left_foot_frame: final_left_pose,
                args.right_foot_frame: final_right_pose,
            },
            "final_base_link_pose": {
                "x_m": current_base_pose[0],
                "y_m": current_base_pose[1],
                "yaw_deg": math.degrees(current_base_pose[2]),
            },
            "actual_relative_motion": {
                "forward_m": relative[0],
                "lateral_m": relative[1],
                "yaw_deg": math.degrees(relative[2]),
            },
            "base_link_relative_motion": {
                "forward_m": base_relative[0],
                "lateral_m": base_relative[1],
                "yaw_deg": math.degrees(base_relative[2]),
            },
            "verification_results": verification,
            "observed_safety_metrics": {
                "maximum_forward_m": maximum_forward,
                "maximum_absolute_lateral_m": maximum_absolute_lateral,
                "maximum_absolute_yaw_deg": math.degrees(maximum_absolute_yaw),
                "maximum_base_link_absolute_lateral_m": (
                    maximum_base_absolute_lateral
                ),
                "maximum_base_link_absolute_yaw_deg": math.degrees(
                    maximum_base_absolute_yaw
                ),
            },
            "samples": samples,
        }
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", type=float, required=True)
    parser.add_argument("--maximum-distance", type=float, default=0.05)
    parser.add_argument("--forward-kp", type=float, default=0.8)
    parser.add_argument("--lateral-kp", type=float, default=0.5)
    parser.add_argument("--yaw-kp", type=float, default=0.8)
    parser.add_argument("--path-lookahead-m", type=float, default=0.20)
    parser.add_argument("--lateral-deadband-m", type=float, default=0.005)
    parser.add_argument("--yaw-deadband-deg", type=float, default=0.3)
    parser.add_argument("--minimum-forward-speed", type=float, default=0.015)
    parser.add_argument("--maximum-forward-speed", type=float, default=0.035)
    parser.add_argument("--maximum-lateral-speed", type=float, default=0.02)
    parser.add_argument("--maximum-yaw-rate-deg", type=float, default=2.0)
    parser.add_argument("--stop-margin", type=float, default=0.005)
    parser.add_argument("--maximum-reverse-m", type=float, default=0.02)
    parser.add_argument("--maximum-overshoot-m", type=float, default=0.05)
    parser.add_argument("--maximum-lateral-drift-m", type=float, default=0.035)
    parser.add_argument("--maximum-yaw-drift-deg", type=float, default=4.0)
    parser.add_argument("--progress-epsilon", type=float, default=0.002)
    parser.add_argument("--progress-watchdog", type=float, default=3.0)
    parser.add_argument("--minimum-valid-motion-m", type=float, default=0.015)
    parser.add_argument("--post-forward-tolerance-m", type=float, default=0.015)
    parser.add_argument("--post-lateral-tolerance-m", type=float, default=0.025)
    parser.add_argument("--post-yaw-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--zero-duration", type=float, default=1.5)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument(
        "--gait-service", default="/humanoid_get_current_gait_name"
    )
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--left-foot-frame", default="leg_l6_link")
    parser.add_argument("--right-foot-frame", default="leg_r6_link")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output")
    return parser.parse_args()


def validate_args(args):
    validate_forward_motion(args.forward, args.maximum_distance)
    if not 0.01 <= args.minimum_forward_speed <= 0.04:
        raise ValueError("minimum forward speed is outside the safety range")
    if not args.minimum_forward_speed <= args.maximum_forward_speed <= 0.06:
        raise ValueError("maximum forward speed is outside the safety range")
    if not 0.005 <= args.maximum_lateral_speed <= 0.03:
        raise ValueError("maximum lateral speed is outside the safety range")
    if not 1.0 <= args.maximum_yaw_rate_deg <= 6.0:
        raise ValueError("maximum yaw rate is outside the safety range")
    if not 0.1 <= args.lateral_kp <= 1.5:
        raise ValueError("cross-track gain is outside the safety range")
    if not 0.2 <= args.yaw_kp <= 2.0:
        raise ValueError("heading gain is outside the safety range")
    if not 0.10 <= args.path_lookahead_m <= 0.50:
        raise ValueError("path lookahead is outside the safety range")
    if not 0.0 <= args.lateral_deadband_m <= 0.015:
        raise ValueError("lateral deadband is outside the safety range")
    if not 0.0 <= args.yaw_deadband_deg <= 1.0:
        raise ValueError("yaw deadband is outside the safety range")
    if not 0.005 <= args.stop_margin <= 0.025:
        raise ValueError("stop margin is outside the safety range")
    if not 10.0 <= args.rate_hz <= 30.0:
        raise ValueError("controller rate must be within [10, 30] Hz")
    if not 5.0 <= args.timeout <= 20.0:
        raise ValueError("timeout must be within [5, 20] seconds")
    if args.left_foot_frame == args.right_foot_frame:
        raise ValueError("left and right foot frames must be different")


def write_output(payload, output_path):
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def main():
    args = parse_args()
    validate_args(args)
    if args.execute:
        payload = execute_motion(args.forward, args)
    else:
        payload = command_plan(args.forward, args)
        payload["status"] = "dry_run"
        payload["success"] = False
    write_output(payload, args.output)
    if args.execute and not payload.get("success"):
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
