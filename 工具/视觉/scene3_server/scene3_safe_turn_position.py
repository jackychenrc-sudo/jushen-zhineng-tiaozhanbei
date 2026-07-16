#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Move the physical right-gripper TCP to the Scene3 safe-turn point.

Only Cartesian position is planned during this transport stage.  No world
orientation target is imposed.  The three wrist joint commands are held at
their current values merely to keep the local solution continuous and avoid
an arbitrary IK wrist flip.  The gripper is aligned with a full 6D target only
after this script reports ``SAFE_TURN_POSITION_REACHED``.

The default mode is calculation-only.  Execution requires an explicit token,
uses short quintic segments, checks the physical TCP and the locked tray after
every segment, and rolls a failed segment back.  This file never creates a
base publisher or claw service proxy.
"""

from __future__ import print_function

import argparse
import math
import statistics
import time

import numpy as np

from scene3_fixed_wrist_ik import solve_fixed_wrist_position
from scene3_gripper_6d_align_plan import (
    GRIPPER_BASE_FRAME,
    LEFT_FINGER_FRAME,
    RIGHT_FINGER_FRAME,
    gripper_geometry,
    quaternion_to_matrix,
)


CONFIRMATION = "SCENE3_SAFE_TURN_POSITION"
REFERENCE_PARAM = "/challenge_cup_task_template/scene3/arm_command_reference_deg"
SAFE_TARGET_PARAM = (
    "/challenge_cup_task_template/scene3/safe_turn_tcp_base_xyz"
)
FINAL_GRASP_PARAM = (
    "/challenge_cup_task_template/scene3/final_grasp_tcp_base_xyz"
)
LOCKED_ODOM_PARAM = (
    "/challenge_cup_task_template/scene3/locked_target_odom_xyz"
)
BASE_TOPIC = "/challenge_cup_task_template/scene3/grasp_point_base"
ODOM_TOPIC = "/challenge_cup_task_template/scene3/grasp_point_odom"

ARM_NAMES = [
    "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
    "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
    "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
    "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
]


def vector_norm(values):
    return float(np.linalg.norm(np.asarray(values, dtype=float)))


def candidate_steps(maximum_step, minimum_step, remaining):
    step = min(float(maximum_step), float(remaining))
    values = []
    while step + 1e-9 >= float(minimum_step):
        values.append(step)
        step *= 0.5
    return values


def quintic(progress):
    value = max(0.0, min(1.0, float(progress)))
    return 10.0 * value ** 3 - 15.0 * value ** 4 + 6.0 * value ** 5


def tcp_step_target(current_eef, current_tcp, target_tcp, requested_step):
    eef = np.asarray(current_eef, dtype=float).reshape(3)
    tcp = np.asarray(current_tcp, dtype=float).reshape(3)
    target = np.asarray(target_tcp, dtype=float).reshape(3)
    error = target - tcp
    remaining = vector_norm(error)
    if remaining < 1e-9:
        return {
            "eef_target": eef.copy(),
            "direction": np.zeros(3, dtype=float),
            "step": 0.0,
            "remaining": 0.0,
        }
    step = min(float(requested_step), remaining)
    direction = error / remaining
    return {
        "eef_target": eef + step * direction,
        "direction": direction,
        "step": step,
        "remaining": remaining,
    }


def predict_physical_tcp(
        current_eef_position,
        current_eef_quaternion,
        current_tcp,
        predicted_eef_position,
        predicted_eef_quaternion):
    current_position = np.asarray(current_eef_position, dtype=float).reshape(3)
    current_rotation = quaternion_to_matrix(current_eef_quaternion)
    current_tcp = np.asarray(current_tcp, dtype=float).reshape(3)
    local_offset = current_rotation.T.dot(current_tcp - current_position)
    predicted_position = np.asarray(predicted_eef_position, dtype=float).reshape(3)
    predicted_rotation = quaternion_to_matrix(predicted_eef_quaternion)
    return predicted_position + predicted_rotation.dot(local_offset)


def cartesian_metrics(before, after, target, direction):
    before = np.asarray(before, dtype=float).reshape(3)
    after = np.asarray(after, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    movement = after - before
    progress = float(np.dot(movement, direction))
    cross_track = vector_norm(movement - progress * direction)
    return {
        "movement": movement,
        "progress": progress,
        "cross_track": cross_track,
        "motion": vector_norm(movement),
        "target_error": vector_norm(target - after),
    }


def segment_checks(planned_step, before_error, metrics):
    step = float(planned_step)
    return {
        "forward_progress": 0.30 * step <= metrics["progress"] <= 1.70 * step,
        "cross_track_bounded": metrics["cross_track"] <= max(0.008, 0.60 * step),
        "motion_bounded": metrics["motion"] <= 1.90 * step + 0.005,
        "target_error_reduced": (
            metrics["target_error"] <= float(before_error) - 0.15 * step
        ),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--max-segments", type=int, default=1)
    parser.add_argument("--maximum-cartesian-step", type=float, default=0.02)
    parser.add_argument("--minimum-cartesian-step", type=float, default=0.005)
    parser.add_argument("--arrival-tolerance", type=float, default=0.012)
    parser.add_argument("--maximum-joint-step-deg", type=float, default=6.0)
    parser.add_argument("--maximum-ik-error", type=float, default=0.004)
    parser.add_argument("--maximum-ik-iterations", type=int, default=35)
    parser.add_argument("--maximum-baseline-error-deg", type=float, default=1.5)
    parser.add_argument("--maximum-left-drift-deg", type=float, default=0.35)
    parser.add_argument("--maximum-wrist-drift-deg", type=float, default=0.35)
    parser.add_argument("--maximum-vision-spread", type=float, default=0.02)
    parser.add_argument("--maximum-identity-error", type=float, default=0.12)
    parser.add_argument("--motion-seconds", type=float, default=4.0)
    parser.add_argument("--rollback-seconds", type=float, default=4.0)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--vision-settle-seconds", type=float, default=1.5)
    parser.add_argument("--tcp-extension", type=float, default=0.045)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser


def run_ros(args):
    import rospy
    import tf2_ros
    from geometry_msgs.msg import PointStamped
    from kuavo_msgs.msg import sensorsData
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest, fkSrv
    from sensor_msgs.msg import JointState

    from scene3_6d_pose_dry_run import extract_arm_joints

    rospy.init_node("scene3_safe_turn_position", anonymous=True)
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    listener = tf2_ros.TransformListener(tf_buffer)  # noqa: F841
    rospy.sleep(1.2)

    safe_target = np.asarray(rospy.get_param(SAFE_TARGET_PARAM), dtype=float)
    locked_odom = np.asarray(rospy.get_param(LOCKED_ODOM_PARAM), dtype=float)
    if safe_target.shape != (3,) or not np.all(np.isfinite(safe_target)):
        raise RuntimeError("safe-turn TCP target is invalid")
    if locked_odom.shape != (3,) or not np.all(np.isfinite(locked_odom)):
        raise RuntimeError("world-locked tray identity is invalid")

    reference = rospy.get_param(REFERENCE_PARAM, None)
    if not isinstance(reference, (list, tuple)) or len(reference) != 14:
        raise RuntimeError("arm command reference is unavailable")
    reference = [float(value) for value in reference]

    rospy.wait_for_service("/ik/fk_srv", timeout=float(args.timeout))
    fk_proxy = rospy.ServiceProxy("/ik/fk_srv", fkSrv, persistent=True)

    def call_fk(arm):
        response = fk_proxy([float(value) for value in arm])
        if not getattr(response, "success", False):
            raise RuntimeError("/ik/fk_srv failed")
        return response.hand_poses

    def sample_arm(count=5):
        samples = []
        mapping = ""
        for _ in range(int(count)):
            message = rospy.wait_for_message(
                "/sensors_data_raw", sensorsData, timeout=float(args.timeout)
            )
            arm, mapping = extract_arm_joints(message)
            samples.append([float(value) for value in arm])
            rospy.sleep(0.06)
        median = [
            statistics.median(row[index] for row in samples)
            for index in range(14)
        ]
        return np.asarray(median, dtype=float), mapping

    def frame_xyz(frame_name):
        transform = tf_buffer.lookup_transform(
            "base_link", frame_name, rospy.Time(0), rospy.Duration(3.0)
        )
        value = transform.transform.translation
        return np.asarray([value.x, value.y, value.z], dtype=float)

    def physical_geometry():
        return gripper_geometry(
            frame_xyz(GRIPPER_BASE_FRAME),
            frame_xyz(LEFT_FINGER_FRAME),
            frame_xyz(RIGHT_FINGER_FRAME),
            tcp_extension_m=float(args.tcp_extension),
        )

    def collect_points(topic, count=3):
        points = []
        for _ in range(int(count)):
            message = rospy.wait_for_message(
                topic, PointStamped, timeout=float(args.timeout)
            )
            points.append([message.point.x, message.point.y, message.point.z])
        points = np.asarray(points, dtype=float)
        center = np.median(points, axis=0)
        spread = max(
            vector_norm(first - second)
            for first in points for second in points
        )
        return center, spread

    def observe_tray():
        base, base_spread = collect_points(BASE_TOPIC)
        odom, odom_spread = collect_points(ODOM_TOPIC)
        identity_error = vector_norm(odom - locked_odom)
        checks = {
            "base_stable": base_spread <= args.maximum_vision_spread,
            "odom_stable": odom_spread <= args.maximum_vision_spread,
            "same_tray": identity_error <= args.maximum_identity_error,
        }
        return {
            "base": base,
            "base_spread": base_spread,
            "odom_spread": odom_spread,
            "identity_error": identity_error,
            "checks": checks,
        }

    def plan_segment(current_arm, current_tcp, current_poses):
        current_eef = np.asarray(current_poses.right_pose.pos_xyz, dtype=float)
        current_quaternion = list(current_poses.right_pose.quat_xyzw)
        remaining = vector_norm(safe_target - current_tcp)
        if remaining <= float(args.arrival_tolerance):
            return None, remaining

        for requested_step in candidate_steps(
            args.maximum_cartesian_step,
            args.minimum_cartesian_step,
            max(0.0, remaining - 0.5 * args.arrival_tolerance),
        ):
            local = tcp_step_target(
                current_eef, current_tcp, safe_target, requested_step
            )

            def fk_position(candidate):
                return call_fk(candidate).right_pose.pos_xyz

            result = solve_fixed_wrist_position(
                fk_position,
                current_arm,
                local["eef_target"],
                tolerance_m=float(args.maximum_ik_error),
                max_iterations=int(args.maximum_ik_iterations),
            )
            if not result["success"]:
                continue
            solved = np.asarray(result["arm_joints_rad"], dtype=float)
            delta_deg = np.rad2deg(solved - current_arm)
            if np.max(np.abs(delta_deg[7:11])) > args.maximum_joint_step_deg:
                continue
            if np.max(np.abs(delta_deg[:7])) > 1e-7:
                continue
            if np.max(np.abs(delta_deg[11:14])) > 1e-7:
                continue

            predicted_poses = call_fk(solved)
            predicted_tcp = predict_physical_tcp(
                current_eef,
                current_quaternion,
                current_tcp,
                predicted_poses.right_pose.pos_xyz,
                predicted_poses.right_pose.quat_xyzw,
            )
            metrics = cartesian_metrics(
                current_tcp, predicted_tcp, safe_target, local["direction"]
            )
            checks = segment_checks(local["step"], remaining, metrics)
            if not all(checks.values()):
                continue

            target_reference = list(reference)
            for index in range(14):
                target_reference[index] += float(delta_deg[index])
            right_proximal = np.asarray(target_reference[7:11], dtype=float)
            lower = np.asarray([-170.0, -100.0, -80.0, -115.0])
            upper = np.asarray([30.0, 50.0, 80.0, -1.0])
            if np.any(right_proximal < lower) or np.any(right_proximal > upper):
                continue
            return {
                "step": local["step"],
                "direction": local["direction"],
                "target_reference": target_reference,
                "delta_deg": delta_deg,
                "ik_error": float(result["final_error_m"]),
                "predicted_tcp": predicted_tcp,
                "predicted_metrics": metrics,
                "predicted_checks": checks,
            }, remaining
        return False, remaining

    arm_publisher = None

    def publish_once(values):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = ARM_NAMES
        message.position = [float(value) for value in values]
        arm_publisher.publish(message)

    def move_reference(start, target, duration):
        steps = max(1, int(float(duration) * float(args.hz)))
        rate = rospy.Rate(float(args.hz))
        latest = list(start)
        for step_index in range(steps + 1):
            alpha = quintic(float(step_index) / float(steps))
            latest = [
                float(start[index])
                + (float(target[index]) - float(start[index])) * alpha
                for index in range(14)
            ]
            publish_once(latest)
            if step_index % max(1, int(args.hz / 10.0)) == 0:
                rospy.set_param(REFERENCE_PARAM, latest)
            rate.sleep()
        rospy.set_param(REFERENCE_PARAM, [float(value) for value in target])

    def hold_reference(values, duration):
        rate = rospy.Rate(float(args.hz))
        deadline = time.time() + float(duration)
        while not rospy.is_shutdown() and time.time() < deadline:
            publish_once(values)
            rate.sleep()

    def rollback(start, reason):
        print("ROLLBACK: {}".format(reason))
        current = rospy.get_param(REFERENCE_PARAM, start)
        move_reference(current, start, args.rollback_seconds)
        hold_reference(start, args.settle_seconds)
        rospy.set_param(REFERENCE_PARAM, list(start))

    execute = bool(args.execute)
    if execute and args.confirmation != CONFIRMATION:
        raise RuntimeError(
            "execution blocked; pass --execute --confirmation {}".format(
                CONFIRMATION
            )
        )
    if execute:
        arm_publisher = rospy.Publisher(
            "/kuavo_arm_traj", JointState, queue_size=10
        )
        deadline = time.time() + float(args.timeout)
        while arm_publisher.get_num_connections() == 0:
            if time.time() >= deadline:
                raise RuntimeError("/kuavo_arm_traj has no subscriber")
            rospy.sleep(0.05)

        mode_ok = False
        for service_name in (
            "/arm_traj_change_mode",
            "/humanoid_change_arm_ctrl_mode",
        ):
            try:
                rospy.wait_for_service(service_name, timeout=2.0)
                proxy = rospy.ServiceProxy(service_name, changeArmCtrlMode)
                request = changeArmCtrlModeRequest()
                request.control_mode = 2
                response = proxy(request)
                if getattr(response, "result", False):
                    mode_ok = True
                    print("Arm mode 2 enabled via {}".format(service_name))
                    break
            except Exception:
                pass
        if not mode_ok:
            raise RuntimeError("cannot enable arm mode 2")

    segment_limit = int(args.max_segments) if execute else 1
    for segment_index in range(1, segment_limit + 1):
        current_arm, mapping = sample_arm()
        current_poses = call_fk(current_arm)
        geometry = physical_geometry()
        current_tcp = np.asarray(geometry["tcp"], dtype=float)
        observation = observe_tray()
        baseline_error = max(
            abs(math.degrees(current_arm[index]) - reference[index])
            for index in range(14)
        )

        print("\n=== Safe-turn segment {} / {} ===".format(
            segment_index, segment_limit
        ))
        print("Joint mapping: {}".format(mapping))
        print("Safe-turn physical TCP target: {}".format(
            np.round(safe_target, 4).tolist()
        ))
        if rospy.has_param(FINAL_GRASP_PARAM):
            final_grasp = np.asarray(
                rospy.get_param(FINAL_GRASP_PARAM), dtype=float
            )
            print("Final-grasp physical TCP target: {}".format(
                np.round(final_grasp, 4).tolist()
            ))
        print("Physical TCP before: {}".format(
            np.round(current_tcp, 4).tolist()
        ))
        print("Remaining: {:.4f}m".format(vector_norm(safe_target - current_tcp)))
        print("Command-reference baseline error: {:.3f}deg".format(
            baseline_error
        ))
        print(
            "Vision: base={} spread={:.4f}m identity_error={:.4f}m checks={}".format(
                np.round(observation["base"], 4).tolist(),
                observation["base_spread"],
                observation["identity_error"],
                observation["checks"],
            )
        )
        if baseline_error > args.maximum_baseline_error_deg:
            print("SAFE_TURN_POSITION_BLOCKED: stale arm command reference")
            return 2
        if not all(observation["checks"].values()):
            print("SAFE_TURN_POSITION_BLOCKED: tray vision gate failed")
            return 2

        selected, remaining = plan_segment(
            current_arm, current_tcp, current_poses
        )
        if selected is None:
            print("SAFE_TURN_POSITION_REACHED")
            print("Direction has not been locked; claw remains open")
            return 0
        if selected is False:
            print("SAFE_TURN_POSITION_BLOCKED: no bounded local segment")
            return 2

        print("Planned physical TCP step: {:.1f}mm".format(
            1000.0 * selected["step"]
        ))
        print("Right shoulder/elbow delta: {}deg".format(
            np.round(selected["delta_deg"][7:11], 3).tolist()
        ))
        print("Wrist command delta: {}deg".format(
            np.round(selected["delta_deg"][11:14], 6).tolist()
        ))
        print("Predicted physical TCP: {}".format(
            np.round(selected["predicted_tcp"], 4).tolist()
        ))
        print("Predicted progress: {:.4f}m cross={:.4f}m remaining={:.4f}m".format(
            selected["predicted_metrics"]["progress"],
            selected["predicted_metrics"]["cross_track"],
            selected["predicted_metrics"]["target_error"],
        ))
        print("Predicted checks: {}".format(selected["predicted_checks"]))
        if not execute:
            print("SAFE_TURN_POSITION_PLAN_OK: calculation only")
            print("No base, arm or claw command was sent")
            return 0

        previous_reference = list(reference)
        before_arm = current_arm.copy()
        try:
            print("Executing one smooth position-only segment")
            print("World gripper direction is not constrained; base and claw stay untouched")
            move_reference(
                previous_reference,
                selected["target_reference"],
                args.motion_seconds,
            )
            hold_reference(selected["target_reference"], args.settle_seconds)
            after_arm, _ = sample_arm()
            rospy.sleep(0.25)
            after_tcp = np.asarray(physical_geometry()["tcp"], dtype=float)
            metrics = cartesian_metrics(
                current_tcp,
                after_tcp,
                safe_target,
                selected["direction"],
            )
            checks = segment_checks(selected["step"], remaining, metrics)
            left_drift = max(
                abs(math.degrees(after_arm[index] - before_arm[index]))
                for index in range(7)
            )
            wrist_drift = max(
                abs(math.degrees(after_arm[index] - before_arm[index]))
                for index in range(11, 14)
            )
            checks["left_arm_held"] = (
                left_drift <= args.maximum_left_drift_deg
            )
            checks["wrist_continuous"] = (
                wrist_drift <= args.maximum_wrist_drift_deg
            )
            rospy.sleep(args.vision_settle_seconds)
            after_observation = observe_tray()
            checks["same_tray"] = all(
                after_observation["checks"].values()
            )
            print("Physical TCP after: {}".format(
                np.round(after_tcp, 4).tolist()
            ))
            print("Observed delta: {}m".format(
                np.round(metrics["movement"], 4).tolist()
            ))
            print("Observed progress: {:.4f}m cross={:.4f}m remaining={:.4f}m".format(
                metrics["progress"],
                metrics["cross_track"],
                metrics["target_error"],
            ))
            print("Left drift: {:.3f}deg  wrist drift: {:.3f}deg".format(
                left_drift, wrist_drift
            ))
            print("Post-motion checks: {}".format(checks))
            if not all(checks.values()):
                rollback(previous_reference, "physical TCP or vision gate failed")
                print("SAFE_TURN_POSITION_EXECUTION_BLOCKED: rolled back")
                return 2
        except Exception as exc:
            rollback(previous_reference, "exception: {}".format(exc))
            print("SAFE_TURN_POSITION_EXECUTION_BLOCKED: rolled back")
            return 2

        reference = list(selected["target_reference"])
        rospy.set_param(REFERENCE_PARAM, reference)
        print("SAFE_TURN_SEGMENT_COMMITTED: index={} remaining={:.4f}m".format(
            segment_index, metrics["target_error"]
        ))

        if metrics["target_error"] <= args.arrival_tolerance:
            print("SAFE_TURN_POSITION_REACHED")
            print("Direction has not been locked; claw remains open")
            return 0

    print("SAFE_TURN_POSITION_PROGRESS_OK: segment limit reached safely")
    return 0


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

