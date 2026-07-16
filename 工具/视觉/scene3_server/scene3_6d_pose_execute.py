#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute one previously validated Scene3 6D teaching step.

The executor accepts only a small translation with nearly zero orientation
increment.  It interpolates from the saved command reference, verifies the
measured FK result and rolls back on any failed gate.  It never creates a base
publisher or a claw service proxy.
"""

from __future__ import print_function

import argparse
import math
import statistics
import time

from scene3_6d_pose_dry_run import (
    PLAN_PARAM,
    REFERENCE_PARAM,
    extract_arm_joints,
    position_error,
    quaternion_error_degrees,
)


CONFIRMATION = "SCENE3_6D_RETREAT_2CM"
ARM_NAMES = ["arm_joint_{}".format(index) for index in range(1, 15)]


def quintic(progress):
    value = max(0.0, min(1.0, float(progress)))
    return 10.0 * value ** 3 - 15.0 * value ** 4 + 6.0 * value ** 5


def vector_norm(values):
    return math.sqrt(sum(float(value) ** 2 for value in values))


def maximum_abs(values):
    values = list(values)
    return max(abs(float(value)) for value in values) if values else 0.0


def execution_checks(planned, observed, final_position_error,
                     final_orientation_error, left_drift,
                     maximum_cross_track, maximum_motion):
    planned_length = vector_norm(planned)
    if planned_length < 1e-9:
        raise ValueError("planned translation has zero length")
    direction = [float(value) / planned_length for value in planned]
    progress = sum(float(value) * axis for value, axis in zip(observed, direction))
    cross = [
        float(value) - progress * axis
        for value, axis in zip(observed, direction)
    ]
    cross_track = vector_norm(cross)
    motion = vector_norm(observed)
    checks = {
        "forward_progress": 0.55 * planned_length <= progress <= 1.45 * planned_length,
        "cross_track_bounded": cross_track <= float(maximum_cross_track),
        "motion_bounded": motion <= float(maximum_motion),
        "final_position_bounded": final_position_error <= 0.008,
        "final_orientation_bounded": final_orientation_error <= 4.0,
        "left_arm_bounded": left_drift <= 1.5,
    }
    return checks, progress, cross_track, motion


def build_parser():
    parser = argparse.ArgumentParser(
        description="Execute one guarded Scene3 6D retreat step"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--motion-seconds", type=float, default=4.0)
    parser.add_argument("--rollback-seconds", type=float, default=4.0)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--maximum-baseline-error-deg", type=float, default=1.0)
    parser.add_argument("--maximum-reference-error-deg", type=float, default=0.10)
    parser.add_argument("--maximum-translation", type=float, default=0.025)
    parser.add_argument("--maximum-rotation-increment-deg", type=float, default=0.10)
    parser.add_argument("--maximum-joint-step-deg", type=float, default=8.0)
    parser.add_argument("--maximum-cross-track", type=float, default=0.006)
    parser.add_argument("--maximum-motion", type=float, default=0.035)
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser


def run_ros(args):
    import rospy
    from kuavo_msgs.msg import sensorsData
    from kuavo_msgs.srv import (
        changeArmCtrlMode,
        changeArmCtrlModeRequest,
        fkSrv,
    )
    from sensor_msgs.msg import JointState

    if not args.execute or args.confirmation != CONFIRMATION:
        raise RuntimeError(
            "execution blocked; pass --execute --confirmation {}".format(
                CONFIRMATION
            )
        )

    rospy.init_node("scene3_6d_pose_execute", anonymous=True)
    if not rospy.has_param(PLAN_PARAM):
        raise RuntimeError("6D plan is unavailable; run the dry-run planner first")
    plan = rospy.get_param(PLAN_PARAM)

    required = (
        "source_arm_rad",
        "source_reference_deg",
        "target_reference_deg",
        "target_position",
        "target_quaternion_xyzw",
        "translation_increment",
        "rpy_increment_deg",
    )
    missing = [key for key in required if key not in plan]
    if missing:
        raise RuntimeError("6D plan is missing fields: {}".format(missing))

    source_arm = [float(value) for value in plan["source_arm_rad"]]
    source_reference = [
        float(value) for value in plan["source_reference_deg"]
    ]
    target_reference = [
        float(value) for value in plan["target_reference_deg"]
    ]
    target_position = [float(value) for value in plan["target_position"]]
    target_quaternion = [
        float(value) for value in plan["target_quaternion_xyzw"]
    ]
    translation = [float(value) for value in plan["translation_increment"]]
    rotation_increment = [
        float(value) for value in plan["rpy_increment_deg"]
    ]
    if not all(len(values) == expected for values, expected in (
        (source_arm, 14),
        (source_reference, 14),
        (target_reference, 14),
        (target_position, 3),
        (target_quaternion, 4),
        (translation, 3),
        (rotation_increment, 3),
    )):
        raise RuntimeError("6D plan contains invalid array lengths")

    current_reference = rospy.get_param(REFERENCE_PARAM, None)
    if not isinstance(current_reference, (list, tuple)) or len(current_reference) != 14:
        raise RuntimeError("arm command reference is unavailable")
    current_reference = [float(value) for value in current_reference]

    arm_publisher = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    deadline = time.time() + float(args.timeout)
    while arm_publisher.get_num_connections() == 0:
        if time.time() >= deadline:
            raise RuntimeError("/kuavo_arm_traj has no subscriber")
        rospy.sleep(0.05)

    rospy.wait_for_service("/ik/fk_srv", timeout=float(args.timeout))
    fk_proxy = rospy.ServiceProxy("/ik/fk_srv", fkSrv)

    def call_fk(arm):
        response = fk_proxy([float(value) for value in arm])
        if not getattr(response, "success", False):
            raise RuntimeError("/ik/fk_srv failed")
        return response.hand_poses

    def sample_arm(count=5):
        samples = []
        for _ in range(int(count)):
            message = rospy.wait_for_message(
                "/sensors_data_raw", sensorsData, timeout=float(args.timeout)
            )
            arm, _ = extract_arm_joints(message)
            samples.append(arm)
            rospy.sleep(0.06)
        return [statistics.median(row[index] for row in samples) for index in range(14)]

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
        for step in range(steps + 1):
            alpha = quintic(float(step) / float(steps))
            latest = [
                float(start[index])
                + (float(target[index]) - float(start[index])) * alpha
                for index in range(14)
            ]
            publish_once(latest)
            if step % max(1, int(args.hz / 10.0)) == 0:
                rospy.set_param(REFERENCE_PARAM, latest)
            rate.sleep()
        rospy.set_param(REFERENCE_PARAM, [float(value) for value in target])

    def hold_reference(values, duration):
        rate = rospy.Rate(float(args.hz))
        deadline_local = time.time() + float(duration)
        while not rospy.is_shutdown() and time.time() < deadline_local:
            publish_once(values)
            rate.sleep()

    current_arm = sample_arm()
    current_poses = call_fk(current_arm)
    before_position = [
        float(value) for value in current_poses.right_pose.pos_xyz
    ]

    baseline_error = max(
        abs(math.degrees(current_arm[index] - source_arm[index]))
        for index in range(14)
    )
    reference_error = max(
        abs(current_reference[index] - source_reference[index])
        for index in range(14)
    )
    joint_step = [
        target_reference[index] - source_reference[index]
        for index in range(14)
    ]
    checks = {
        "baseline_fresh": baseline_error <= args.maximum_baseline_error_deg,
        "reference_fresh": reference_error <= args.maximum_reference_error_deg,
        "translation_bounded": vector_norm(translation) <= args.maximum_translation,
        "rotation_nearly_zero": (
            maximum_abs(rotation_increment)
            <= args.maximum_rotation_increment_deg
        ),
        "joint_step_bounded": maximum_abs(joint_step[7:14]) <= args.maximum_joint_step_deg,
        "left_command_held": maximum_abs(joint_step[:7]) <= 1e-6,
        "retreat_direction": translation[0] < -0.005,
    }

    print("Saved translation: {}m".format(
        [round(value, 5) for value in translation]
    ))
    print("Saved RPY increment: {}deg".format(
        [round(value, 4) for value in rotation_increment]
    ))
    print("Right-arm command step: {}deg".format(
        [round(value, 3) for value in joint_step[7:14]]
    ))
    print("Execution baseline error: {:.3f}deg".format(baseline_error))
    print("Command-reference error: {:.3f}deg".format(reference_error))
    print("Pre-execution checks: {}".format(checks))
    if not all(checks.values()):
        print("SIX_D_RETREAT_EXECUTION_BLOCKED: stale or unsafe plan")
        print("No control command was sent")
        return 2

    mode_ok = False
    for service_name in (
        "/arm_traj_change_mode",
        "/humanoid_change_arm_ctrl_mode",
    ):
        try:
            rospy.wait_for_service(service_name, timeout=2.0)
            mode_proxy = rospy.ServiceProxy(service_name, changeArmCtrlMode)
            request = changeArmCtrlModeRequest()
            request.control_mode = 2
            response = mode_proxy(request)
            if getattr(response, "result", False):
                mode_ok = True
                print("Arm mode 2 enabled via {}".format(service_name))
                break
        except Exception:
            pass
    if not mode_ok:
        print("SIX_D_RETREAT_EXECUTION_BLOCKED: cannot enable arm mode 2")
        return 2

    moved = False
    try:
        print("Executing one smooth 2cm 6D retreat; base and claw remain untouched")
        moved = True
        move_reference(source_reference, target_reference, args.motion_seconds)
        hold_reference(target_reference, args.settle_seconds)
        after_arm = sample_arm()
        after_poses = call_fk(after_arm)
        after_position = [
            float(value) for value in after_poses.right_pose.pos_xyz
        ]
        after_quaternion = [
            float(value) for value in after_poses.right_pose.quat_xyzw
        ]
        observed = [
            after_position[index] - before_position[index]
            for index in range(3)
        ]
        final_position_error = position_error(target_position, after_position)
        final_orientation_error = quaternion_error_degrees(
            target_quaternion, after_quaternion
        )
        left_drift = max(
            abs(math.degrees(after_arm[index] - current_arm[index]))
            for index in range(7)
        )
        post_checks, progress, cross_track, motion = execution_checks(
            translation,
            observed,
            final_position_error,
            final_orientation_error,
            left_drift,
            args.maximum_cross_track,
            args.maximum_motion,
        )
        print("Observed EEF displacement: {}m".format(
            [round(value, 5) for value in observed]
        ))
        print("Retreat progress: {:.4f}m".format(progress))
        print("Cross-track error: {:.4f}m".format(cross_track))
        print("Observed motion: {:.4f}m".format(motion))
        print("Final position error: {:.4f}m".format(final_position_error))
        print("Final orientation error: {:.2f}deg".format(final_orientation_error))
        print("Left-arm maximum drift: {:.3f}deg".format(left_drift))
        print("Post-execution checks: {}".format(post_checks))
        if not all(post_checks.values()):
            raise RuntimeError("post-motion 6D safety gate failed")

        print("SIX_D_RETREAT_2CM_OK")
        print("The committed command reference has been updated")
        print("Base and claw were not controlled")
        return 0
    except Exception as exc:
        print("ROLLBACK: {}".format(exc))
        if moved:
            move_reference(
                rospy.get_param(REFERENCE_PARAM, target_reference),
                source_reference,
                args.rollback_seconds,
            )
            hold_reference(source_reference, args.settle_seconds)
            rospy.set_param(REFERENCE_PARAM, source_reference)
        print("SIX_D_RETREAT_EXECUTION_BLOCKED: rolled back")
        print("Claw remains untouched")
        return 2


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

