#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace one guarded one-degree right-wrist command through the ROS stack.

The test keeps every other arm command fixed, moves right-arm joint 5 by one
degree, compares the final ``/joint_cmd`` target with measured joint feedback,
and then returns to the original command.  It never publishes a base or claw
command.
"""

from __future__ import print_function

import argparse
import math
import statistics
import time


REFERENCE_PARAM = (
    "/challenge_cup_task_template/scene3/arm_command_reference_deg"
)
CONFIRMATION = "SCENE3_WRIST_TRACE_1DEG"
ARM_NAMES = ["arm_joint_{}".format(index) for index in range(1, 15)]

# V52 sensor and low-level command layout:
# 12 leg joints, one waist joint, 14 arm joints, two head joints.
ARM_START = 13
ARM_COUNT = 14
RIGHT_ARM_START = 7
RIGHT_WRIST_JOINT = 4
ARM_COMMAND_INDEX = RIGHT_ARM_START + RIGHT_WRIST_JOINT
LOW_LEVEL_INDEX = ARM_START + ARM_COMMAND_INDEX


def quintic(progress):
    value = max(0.0, min(1.0, float(progress)))
    return 10.0 * value ** 3 - 15.0 * value ** 4 + 6.0 * value ** 5


def maximum_abs(values):
    values = list(values)
    return max(abs(float(value)) for value in values) if values else 0.0


def classify_trace(command_delta, low_level_delta, measured_delta):
    """Return the first control layer that demonstrably lost the command."""
    command_delta = float(command_delta)
    low_level_delta = float(low_level_delta)
    measured_delta = float(measured_delta)
    if abs(command_delta) < 0.8:
        return "TRACE_INVALID_COMMAND"
    if abs(low_level_delta) < 0.30:
        return "WBC_WRIST_COMMAND_LOST"
    if command_delta * low_level_delta <= 0.0:
        return "WBC_WRIST_COMMAND_WRONG_SIGN"
    if abs(measured_delta) < 0.25:
        return "LOW_LEVEL_WRIST_NOT_FOLLOWING"
    if low_level_delta * measured_delta <= 0.0:
        return "LOW_LEVEL_WRIST_WRONG_SIGN"
    if abs(measured_delta) >= 0.65:
        return "WRIST_CONTROL_CHANNEL_OK"
    return "WRIST_CONTROL_PARTIAL_RESPONSE"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Trace one right-wrist degree through WBC and joint_cmd"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--test-degrees", type=float, default=1.0)
    parser.add_argument("--motion-seconds", type=float, default=3.0)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--return-seconds", type=float, default=3.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--maximum-handover-error-deg", type=float, default=2.5)
    parser.add_argument("--minimum-wrist-kp", type=float, default=0.1)
    parser.add_argument("--maximum-rebase-drift-deg", type=float, default=0.75)
    parser.add_argument(
        "--maximum-low-level-reference-error-deg",
        type=float,
        default=0.50,
    )
    parser.add_argument("--maximum-other-joint-motion-deg", type=float, default=1.0)
    return parser


def run_ros(args):
    import rospy
    from kuavo_msgs.msg import jointCmd, sensorsData
    from kuavo_msgs.srv import (
        ExecuteArmAction,
        ExecuteArmActionRequest,
        changeArmCtrlMode,
        changeArmCtrlModeRequest,
    )
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64, Float64MultiArray

    if not args.execute or args.confirmation != CONFIRMATION:
        raise RuntimeError(
            "execution blocked; pass --execute --confirmation {}".format(
                CONFIRMATION
            )
        )
    if not 0.5 <= abs(float(args.test_degrees)) <= 1.0:
        raise RuntimeError("test-degrees must be between 0.5 and 1.0")

    rospy.init_node("scene3_wrist_control_trace", anonymous=True)

    reference = rospy.get_param(REFERENCE_PARAM, None)
    if not isinstance(reference, (list, tuple)) or len(reference) != ARM_COUNT:
        raise RuntimeError("complete 14-joint command reference is unavailable")
    reference = [float(value) for value in reference]

    publisher = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    connection_deadline = time.time() + float(args.timeout)
    while publisher.get_num_connections() == 0:
        if time.time() >= connection_deadline:
            raise RuntimeError("/kuavo_arm_traj has no subscriber")
        rospy.sleep(0.05)

    def publish_once(values):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = list(ARM_NAMES)
        message.position = [float(value) for value in values]
        publisher.publish(message)

    def hold(values, duration):
        rate = rospy.Rate(float(args.hz))
        deadline = time.time() + float(duration)
        while not rospy.is_shutdown() and time.time() < deadline:
            publish_once(values)
            rate.sleep()

    def move(start, finish, duration):
        steps = max(1, int(float(duration) * float(args.hz)))
        rate = rospy.Rate(float(args.hz))
        for step in range(steps + 1):
            alpha = quintic(float(step) / float(steps))
            point = [
                float(start[index])
                + (float(finish[index]) - float(start[index])) * alpha
                for index in range(ARM_COUNT)
            ]
            publish_once(point)
            rate.sleep()

    def sample_arm(count=7):
        samples = []
        for _ in range(int(count)):
            message = rospy.wait_for_message(
                "/sensors_data_raw", sensorsData, timeout=float(args.timeout)
            )
            values = list(message.joint_data.joint_q)
            if len(values) < ARM_START + ARM_COUNT:
                raise RuntimeError(
                    "unexpected sensors joint_q length {}".format(len(values))
                )
            samples.append(values[ARM_START:ARM_START + ARM_COUNT])
            rospy.sleep(0.04)
        return [
            statistics.median(sample[index] for sample in samples)
            for index in range(ARM_COUNT)
        ]

    def sample_low_level(count=7):
        position_samples = []
        mode_samples = []
        gain_samples = []
        for _ in range(int(count)):
            message = rospy.wait_for_message(
                "/joint_cmd", jointCmd, timeout=float(args.timeout)
            )
            if len(message.joint_q) < ARM_START + ARM_COUNT:
                raise RuntimeError(
                    "unexpected joint_cmd length {}".format(
                        len(message.joint_q)
                    )
                )
            position_samples.append([
                float(value)
                for value in message.joint_q[
                    ARM_START:ARM_START + ARM_COUNT
                ]
            ])
            if len(message.control_modes) >= ARM_START + ARM_COUNT:
                mode_samples.append([
                    int(value)
                    for value in message.control_modes[
                        ARM_START:ARM_START + ARM_COUNT
                    ]
                ])
            if len(message.joint_kp) >= ARM_START + ARM_COUNT:
                gain_samples.append([
                    float(value)
                    for value in message.joint_kp[
                        ARM_START:ARM_START + ARM_COUNT
                    ]
                ])
            rospy.sleep(0.02)
        positions = [
            statistics.median(sample[index] for sample in position_samples)
            for index in range(ARM_COUNT)
        ]
        modes = None
        if mode_samples:
            modes = [
                int(statistics.median(
                    sample[index] for sample in mode_samples
                ))
                for index in range(ARM_COUNT)
            ]
        gains = None
        if gain_samples:
            gains = [
                statistics.median(
                    sample[index] for sample in gain_samples
                )
                for index in range(ARM_COUNT)
            ]
        return positions, modes, gains

    def call_mode_service(service_name, control_mode):
        rospy.wait_for_service(service_name, timeout=float(args.timeout))
        proxy = rospy.ServiceProxy(service_name, changeArmCtrlMode)
        request = changeArmCtrlModeRequest()
        request.control_mode = int(control_mode)
        response = proxy(request)
        if not getattr(response, "result", False):
            raise RuntimeError("{} rejected mode {}".format(
                service_name, control_mode
            ))

    def load_normal_ruiwo_gains():
        service_name = "/humanoid_controller/change_ruiwo_motor_param"
        rospy.wait_for_service(service_name, timeout=float(args.timeout))
        proxy = rospy.ServiceProxy(service_name, ExecuteArmAction)
        request = ExecuteArmActionRequest()
        request.action_name = "normal_kpkd"
        response = proxy(request)
        if not getattr(response, "success", False):
            raise RuntimeError(
                "normal Ruiwo gain service failed: {}".format(
                    getattr(response, "message", "")
                )
            )
        print("Loaded official normal_kpkd Ruiwo gains")

    def low_level_reference_error(low_level_radians, command_degrees):
        return maximum_abs([
            math.degrees(low_level_radians[index])
            - float(command_degrees[index])
            for index in range(ARM_COUNT)
        ])

    def indexed_or_none(values, index):
        if values is None:
            return None
        return values[index]

    call_mode_service("/arm_traj_change_mode", 2)

    mode_ready = False
    mode_values = []
    mode_deadline = time.time() + float(args.timeout)
    while time.time() < mode_deadline:
        message = rospy.wait_for_message(
            "/humanoid/mpc/arm_control_mode",
            Float64MultiArray,
            timeout=min(2.0, float(args.timeout)),
        )
        mode_values = [float(value) for value in message.data]
        if len(mode_values) >= 2 and all(
                abs(value - 2.0) < 0.1 for value in mode_values[:2]):
            mode_ready = True
            break
    reset_message = rospy.wait_for_message(
        "/humanoid_controller/resetting_mpc_state_",
        Float64,
        timeout=float(args.timeout),
    )
    reset_value = float(reset_message.data)
    print("Internal arm modes [actual, desired]: {}".format(mode_values))
    print("MPC resetting state: {:.1f}".format(reset_value))
    if not mode_ready or abs(reset_value) > 0.1:
        print("WRIST_TRACE_BLOCKED: internal controller is not ready")
        return 2

    # Read the physical arm before publishing anything.  A persisted command
    # can be stale after a blocked/rolled-back run; never re-enable gains while
    # such a target is active.
    initial_arm = sample_arm()
    initial_deg = [math.degrees(value) for value in initial_arm]
    initial_reference_error = maximum_abs([
        initial_deg[index] - reference[index]
        for index in range(ARM_COUNT)
    ])
    print("Saved-reference error before publishing: {:.3f}deg".format(
        initial_reference_error
    ))
    if initial_reference_error > float(args.maximum_handover_error_deg):
        reference = list(initial_deg)
        rospy.set_param(REFERENCE_PARAM, list(reference))
        print("STALE_REFERENCE_REBASED_TO_MEASURED_POSE")

    print("Priming safe 14-joint reference")
    hold(reference, 1.0)
    call_mode_service("/enable_wbc_arm_trajectory_control", 1)
    hold(reference, float(args.hold_seconds))

    before_arm = sample_arm()
    before_low, before_mode, before_kp = sample_low_level()
    before_deg = [math.degrees(value) for value in before_arm]
    handover_error = maximum_abs([
        before_deg[index] - reference[index]
        for index in range(ARM_COUNT)
    ])
    print("Command mapping: arm[{}] -> joint_cmd[{}]".format(
        ARM_COMMAND_INDEX, LOW_LEVEL_INDEX
    ))
    print("Right wrist-5 command before: {:.3f}deg".format(
        reference[ARM_COMMAND_INDEX]
    ))
    print("Right wrist-5 measured before: {:.3f}deg".format(
        before_deg[ARM_COMMAND_INDEX]
    ))
    print("Right wrist-5 joint_cmd before: {:.3f}deg".format(
        math.degrees(before_low[ARM_COMMAND_INDEX])
    ))
    print("Low-level mode before: {}  kp: {}".format(
        indexed_or_none(before_mode, ARM_COMMAND_INDEX),
        None if before_kp is None else round(
            before_kp[ARM_COMMAND_INDEX], 3
        ),
    ))
    print("Maximum command-to-measurement handover error: {:.3f}deg".format(
        handover_error
    ))

    wrist_kp = indexed_or_none(before_kp, ARM_COMMAND_INDEX)
    stale_reference = (
        handover_error > float(args.maximum_handover_error_deg)
    )
    wrist_gain_missing = (
        wrist_kp is None
        or float(wrist_kp) <= float(args.minimum_wrist_kp)
    )

    if stale_reference or wrist_gain_missing:
        print(
            "Repair required: stale_reference={} wrist_gain_missing={}".format(
                stale_reference,
                wrist_gain_missing,
            )
        )
        safe_reference = list(before_deg)
        print("Rebasing all 14 arm commands to measured pose before gain restore")
        print("Safe right wrist-5 reference: {:.3f}deg".format(
            safe_reference[ARM_COMMAND_INDEX]
        ))
        hold(safe_reference, float(args.hold_seconds))
        rebased_low, rebased_mode, rebased_kp = sample_low_level()
        rebase_target_error = low_level_reference_error(
            rebased_low,
            safe_reference,
        )
        print("Rebased /joint_cmd maximum target error: {:.3f}deg".format(
            rebase_target_error
        ))
        if rebase_target_error > float(
                args.maximum_low_level_reference_error_deg):
            print("WRIST_GAIN_REPAIR_BLOCKED: safe reference did not reach joint_cmd")
            return 2

        # Keep the persisted command synchronized before restoring a gain.  If
        # gain restoration takes effect immediately, its target is therefore
        # the measured pose rather than the stale wrist command.
        rospy.set_param(REFERENCE_PARAM, list(safe_reference))
        try:
            load_normal_ruiwo_gains()
        except Exception as error:
            hold(safe_reference, float(args.hold_seconds))
            print("WRIST_GAIN_REPAIR_BLOCKED: {}".format(error))
            return 2

        hold(safe_reference, float(args.hold_seconds))
        repaired_arm = sample_arm()
        repaired_low, repaired_mode, repaired_kp = sample_low_level()
        repaired_deg = [math.degrees(value) for value in repaired_arm]
        repair_drift = maximum_abs([
            repaired_deg[index] - before_deg[index]
            for index in range(ARM_COUNT)
        ])
        repair_target_error = low_level_reference_error(
            repaired_low,
            safe_reference,
        )
        repaired_wrist_kp = indexed_or_none(
            repaired_kp,
            ARM_COMMAND_INDEX,
        )
        print("Ruiwo wrist kp after restore: {}".format(
            None if repaired_wrist_kp is None
            else round(repaired_wrist_kp, 3)
        ))
        print("Gain-restore maximum arm drift: {:.3f}deg".format(
            repair_drift
        ))
        print("Gain-restore /joint_cmd target error: {:.3f}deg".format(
            repair_target_error
        ))
        repair_checks = {
            "wrist_kp_nonzero": (
                repaired_wrist_kp is not None
                and float(repaired_wrist_kp) > float(args.minimum_wrist_kp)
            ),
            "arm_drift_bounded": (
                repair_drift <= float(args.maximum_rebase_drift_deg)
            ),
            "joint_cmd_aligned": (
                repair_target_error <= float(
                    args.maximum_low_level_reference_error_deg
                )
            ),
        }
        print("Gain-restore checks: {}".format(repair_checks))
        if not all(repair_checks.values()):
            hold(safe_reference, float(args.hold_seconds))
            print("WRIST_GAIN_REPAIR_BLOCKED: post-restore safety gate failed")
            return 2

        reference = list(safe_reference)
        before_arm = repaired_arm
        before_low = repaired_low
        before_mode = repaired_mode
        before_kp = repaired_kp
        before_deg = repaired_deg
        print("WRIST_CONTROL_BASELINE_RECOVERED")

    target = list(reference)
    target[ARM_COMMAND_INDEX] += float(args.test_degrees)

    moved = False
    after_arm = None
    after_low = None
    after_mode = None
    after_kp = None
    try:
        print("Tracing one {:.1f}deg right wrist-5 command".format(
            float(args.test_degrees)
        ))
        print("All other arm commands are held; base and claw stay untouched")
        moved = True
        move(reference, target, float(args.motion_seconds))
        hold(target, float(args.hold_seconds))
        after_arm = sample_arm()
        after_low, after_mode, after_kp = sample_low_level()
    finally:
        if moved:
            print("Returning to the original 14-joint reference")
            move(target, reference, float(args.return_seconds))
            hold(reference, float(args.hold_seconds))
            rospy.set_param(REFERENCE_PARAM, list(reference))

    final_arm = sample_arm()
    after_deg = [math.degrees(value) for value in after_arm]
    final_deg = [math.degrees(value) for value in final_arm]
    command_delta = target[ARM_COMMAND_INDEX] - reference[ARM_COMMAND_INDEX]
    low_level_delta = math.degrees(
        after_low[ARM_COMMAND_INDEX] - before_low[ARM_COMMAND_INDEX]
    )
    measured_delta = (
        after_deg[ARM_COMMAND_INDEX] - before_deg[ARM_COMMAND_INDEX]
    )
    other_motion = maximum_abs([
        after_deg[index] - before_deg[index]
        for index in range(ARM_COUNT)
        if index != ARM_COMMAND_INDEX
    ])
    return_error = maximum_abs([
        final_deg[index] - before_deg[index]
        for index in range(ARM_COUNT)
    ])
    result = classify_trace(command_delta, low_level_delta, measured_delta)

    print("Requested wrist delta: {:.3f}deg".format(command_delta))
    print("Final /joint_cmd wrist delta: {:.3f}deg".format(low_level_delta))
    print("Measured wrist delta: {:.3f}deg".format(measured_delta))
    print("Other arm maximum measured motion: {:.3f}deg".format(
        other_motion
    ))
    print("Low-level mode during target: {}  kp: {}".format(
        indexed_or_none(after_mode, ARM_COMMAND_INDEX),
        None if after_kp is None else round(
            after_kp[ARM_COMMAND_INDEX], 3
        ),
    ))
    print("Return maximum error: {:.3f}deg".format(return_error))
    print("WRIST_TRACE_RESULT: {}".format(result))
    if other_motion > float(args.maximum_other_joint_motion_deg):
        print("WRIST_TRACE_WARNING: unrelated arm motion exceeded the limit")
    print("Test returned to the original command; base and claw were not controlled")
    return 0


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

