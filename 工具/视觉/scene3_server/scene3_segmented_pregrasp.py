#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adaptive, vision-gated Scene3 fixed-wrist motion to the pregrasp point.

The script never commands the base or claw.  Execution requires an explicit
confirmation token.  Every accepted segment is followed by a same-tray vision
check; a failed segment is rolled back to its previous command reference.
"""

from __future__ import print_function

import argparse
import math

import numpy as np


CONFIRMATION = "SCENE3_SEGMENTED_PREGRASP"
REFERENCE_PARAM = "/challenge_cup_task_template/scene3/arm_command_reference_deg"
LOCKED_BASE_PARAM = "/challenge_cup_task_template/scene3/locked_target_base_xyz"
LOCKED_ODOM_PARAM = "/challenge_cup_task_template/scene3/locked_target_odom_xyz"
BASE_TOPIC = "/challenge_cup_task_template/scene3/grasp_point_base"
ODOM_TOPIC = "/challenge_cup_task_template/scene3/grasp_point_odom"


def _distance(first, second):
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


def _candidate_steps(maximum_step, minimum_step, remaining):
    step = min(float(maximum_step), float(remaining))
    values = []
    while step + 1e-9 >= float(minimum_step):
        values.append(step)
        step *= 0.5
    return values


def _motion_checks(planned_step, progress, cross_error, motion, target_error):
    step = float(planned_step)
    checks = {
        "forward_progress": 0.40 * step <= progress <= 1.60 * step,
        "cross_track_bounded": cross_error <= max(0.006, 0.50 * step),
        "motion_bounded": motion <= 1.80 * step + 0.005,
        "target_error_bounded": target_error <= max(0.008, 0.60 * step),
    }
    return checks


def run(args):
    import rospy
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import JointState

    from challenge_task_3 import Scene3Task
    from scene3_fixed_wrist_ik import solve_fixed_wrist_position

    if not args.execute or args.confirmation != CONFIRMATION:
        raise RuntimeError(
            "execution blocked; pass --execute --confirmation {}".format(CONFIRMATION)
        )

    rospy.init_node("scene3_segmented_pregrasp", anonymous=True)
    arm_publisher = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    task = Scene3Task(None, arm_publisher)
    task.wait_for_arm_subscriber(timeout=args.timeout)

    reference = [float(value) for value in rospy.get_param(REFERENCE_PARAM)]
    locked_odom = np.asarray(rospy.get_param(LOCKED_ODOM_PARAM), dtype=float)
    if len(reference) != 14 or locked_odom.size != 3:
        raise RuntimeError("saved command reference or tray identity is invalid")

    controlled_indices = list(range(4)) + list(range(7, 11))

    def sample_arm(count=5):
        samples = []
        for _ in range(int(count)):
            samples.append(np.asarray(task.read_current_arm_joints(), dtype=float))
            rospy.sleep(0.08)
        return np.median(np.asarray(samples), axis=0)

    def fk_position(arm):
        poses = task.call_fk([float(value) for value in arm])
        return np.asarray(poses.right_pose.pos_xyz, dtype=float)

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
            _distance(first, second) for first in points for second in points
        )
        return center, spread

    def observe_tray():
        base, base_spread = collect_points(BASE_TOPIC)
        odom, odom_spread = collect_points(ODOM_TOPIC)
        identity_error = _distance(odom, locked_odom)
        checks = {
            "base_stable": base_spread <= args.maximum_vision_spread,
            "odom_stable": odom_spread <= args.maximum_vision_spread,
            "same_tray": identity_error <= args.maximum_identity_error,
        }
        return {
            "base": base,
            "odom": odom,
            "base_spread": base_spread,
            "odom_spread": odom_spread,
            "identity_error": identity_error,
            "checks": checks,
        }

    def rollback(previous_reference, reason):
        print("ROLLBACK: {}".format(reason))
        task.move_arm_degrees(
            previous_reference, duration=args.motion_seconds, hz=args.hz
        )
        task.hold_arm_degrees(
            previous_reference, hold_time=args.settle_seconds, hz=args.hz
        )
        rospy.sleep(args.vision_settle_seconds)
        observation = observe_tray()
        print(
            "Rollback vision: base_spread={:.4f}m odom_spread={:.4f}m "
            "identity_error={:.4f}m checks={}".format(
                observation["base_spread"],
                observation["odom_spread"],
                observation["identity_error"],
                observation["checks"],
            )
        )

    for segment_index in range(1, int(args.max_segments) + 1):
        print("\n=== Segment {} / {} ===".format(segment_index, args.max_segments))
        task.hold_arm_degrees(reference, hold_time=1.0, hz=args.hz)
        baseline_first = sample_arm()
        task.hold_arm_degrees(reference, hold_time=0.8, hz=args.hz)
        baseline = sample_arm()
        baseline_drift = max(
            abs(math.degrees(baseline[index] - baseline_first[index]))
            for index in controlled_indices
        )
        if baseline_drift > args.maximum_baseline_drift_deg:
            print("SEGMENTED_PREGRASP_BLOCKED: arm baseline is drifting")
            return 2

        before_observation = observe_tray()
        print(
            "Before vision: base={} spread={:.4f}m identity_error={:.4f}m checks={}".format(
                np.round(before_observation["base"], 4).tolist(),
                before_observation["base_spread"],
                before_observation["identity_error"],
                before_observation["checks"],
            )
        )
        if not all(before_observation["checks"].values()):
            print("SEGMENTED_PREGRASP_BLOCKED: pre-motion vision gate failed")
            return 2

        tray = before_observation["base"]
        pregrasp = tray + np.asarray(
            [args.pregrasp_offset_x, args.offset_y, args.offset_z], dtype=float
        )
        before_xyz = fk_position(baseline)
        remaining_vector = pregrasp - before_xyz
        remaining_distance = float(np.linalg.norm(remaining_vector))
        print("Hand: {}".format(np.round(before_xyz, 4).tolist()))
        print("Pregrasp: {}".format(np.round(pregrasp, 4).tolist()))
        print("Remaining: {:.4f}m".format(remaining_distance))

        if remaining_distance <= args.arrival_tolerance:
            rospy.set_param(LOCKED_BASE_PARAM, tray.tolist())
            print("SEGMENTED_PREGRASP_REACHED")
            return 0

        direction = remaining_vector / remaining_distance
        selected = None
        for planned_step in _candidate_steps(
            args.maximum_cartesian_step,
            args.minimum_cartesian_step,
            max(0.0, remaining_distance - 0.5 * args.arrival_tolerance),
        ):
            local_target = before_xyz + planned_step * direction
            result = solve_fixed_wrist_position(
                fk_position,
                baseline,
                local_target,
                tolerance_m=args.maximum_ik_error,
                max_iterations=args.maximum_ik_iterations,
            )
            if not result["success"]:
                continue
            solved = np.asarray(result["arm_joints_rad"], dtype=float)
            delta_deg = np.rad2deg(solved - baseline)
            proximal_delta = delta_deg[7:11]
            wrist_delta = delta_deg[11:14]
            left_delta = delta_deg[:7]
            if float(np.max(np.abs(proximal_delta))) > args.maximum_joint_step_deg:
                continue
            if float(np.max(np.abs(wrist_delta))) > 1e-6:
                continue
            if float(np.max(np.abs(left_delta))) > 1e-6:
                continue
            target_reference = list(reference)
            for index in range(14):
                target_reference[index] += float(delta_deg[index])
            right_proximal = np.asarray(target_reference[7:11], dtype=float)
            lower = np.asarray([-170.0, -100.0, -80.0, -115.0])
            upper = np.asarray([30.0, 50.0, 80.0, -1.0])
            if np.any(right_proximal < lower) or np.any(right_proximal > upper):
                continue
            selected = {
                "step": planned_step,
                "target_xyz": local_target,
                "target_reference": target_reference,
                "proximal_delta": proximal_delta,
                "ik_error": float(result["final_error_m"]),
            }
            break

        if selected is None:
            print("SEGMENTED_PREGRASP_BLOCKED: no bounded fixed-wrist segment")
            return 2

        print("Planned Cartesian step: {:.4f}m".format(selected["step"]))
        print(
            "Right shoulder/elbow command delta: {}deg".format(
                np.round(selected["proximal_delta"], 3).tolist()
            )
        )
        print("IK error: {:.4f}m".format(selected["ik_error"]))

        previous_reference = list(reference)
        moved = False
        try:
            moved = True
            task.move_arm_degrees(
                selected["target_reference"],
                duration=args.motion_seconds,
                hz=args.hz,
            )
            task.hold_arm_degrees(
                selected["target_reference"],
                hold_time=args.settle_seconds,
                hz=args.hz,
            )
            after_arm = sample_arm()
            after_xyz = fk_position(after_arm)
            movement = after_xyz - before_xyz
            progress = float(np.dot(movement, direction))
            cross_error = float(np.linalg.norm(movement - progress * direction))
            motion = float(np.linalg.norm(movement))
            target_error = _distance(after_xyz, selected["target_xyz"])
            motion_checks = _motion_checks(
                selected["step"], progress, cross_error, motion, target_error
            )
            rospy.sleep(args.vision_settle_seconds)
            after_observation = observe_tray()
            print(
                "Observed: delta={} progress={:.4f}m cross={:.4f}m "
                "motion={:.4f}m target_error={:.4f}m".format(
                    np.round(movement, 4).tolist(),
                    progress,
                    cross_error,
                    motion,
                    target_error,
                )
            )
            print("Motion checks: {}".format(motion_checks))
            print(
                "After vision: base={} spread={:.4f}m identity_error={:.4f}m checks={}".format(
                    np.round(after_observation["base"], 4).tolist(),
                    after_observation["base_spread"],
                    after_observation["identity_error"],
                    after_observation["checks"],
                )
            )
            if not all(motion_checks.values()):
                rollback(previous_reference, "post-motion Cartesian gate failed")
                print("SEGMENTED_PREGRASP_BLOCKED")
                return 2
            if not all(after_observation["checks"].values()):
                rollback(previous_reference, "post-motion vision gate failed")
                print("SEGMENTED_PREGRASP_BLOCKED")
                return 2
        except Exception as exc:
            if moved:
                rollback(previous_reference, "exception: {}".format(exc))
            raise

        reference = list(selected["target_reference"])
        rospy.set_param(REFERENCE_PARAM, reference)
        rospy.set_param(LOCKED_BASE_PARAM, after_observation["base"].tolist())
        remaining_after = _distance(after_xyz, pregrasp)
        print(
            "SEGMENT_COMMITTED_OK: index={} remaining={:.4f}m".format(
                segment_index, remaining_after
            )
        )

    print("SEGMENTED_PREGRASP_PROGRESS_OK: max segment count reached safely")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--max-segments", type=int, default=1)
    parser.add_argument("--maximum-cartesian-step", type=float, default=0.02)
    parser.add_argument("--minimum-cartesian-step", type=float, default=0.005)
    parser.add_argument("--arrival-tolerance", type=float, default=0.015)
    parser.add_argument("--pregrasp-offset-x", type=float, default=-0.16)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument("--offset-z", type=float, default=0.02)
    parser.add_argument("--maximum-joint-step-deg", type=float, default=6.0)
    parser.add_argument("--maximum-ik-error", type=float, default=0.004)
    parser.add_argument("--maximum-ik-iterations", type=int, default=30)
    parser.add_argument("--maximum-baseline-drift-deg", type=float, default=0.15)
    parser.add_argument("--maximum-vision-spread", type=float, default=0.02)
    parser.add_argument("--maximum-identity-error", type=float, default=0.12)
    parser.add_argument("--motion-seconds", type=float, default=4.0)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--vision-settle-seconds", type=float, default=1.5)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

