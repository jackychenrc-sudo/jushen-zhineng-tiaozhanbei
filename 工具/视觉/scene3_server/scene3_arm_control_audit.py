#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only audit for the Scene3 arm command chain.

The audit compares MPC and sensor arm coordinates, checks the active arm mode
and right-arm gains, detects unexpected traffic on both known arm command
topics, and refuses to approve an obviously unsafe arm configuration. It does
not create a publisher, change a mode, write a parameter, or command a robot.
"""

from __future__ import print_function

import numpy as np


SENSOR_ARM_START = 13
MPC_FIXED_STATE_PREFIX = 24
ARM_COUNT = 14
RIGHT_ARM_START = 7
LOW_LEVEL_ARM_START = 13
SAFE_ARM_TOPIC = "/kuavo_arm_traj"
EXPERIMENTAL_TIMED_TOPIC = "/kuavo_arm_target_poses"


def extract_mpc_state_values(message):
    """Return an MPC state vector from either ROS serialization shape."""

    state = message.state
    values = state.value if hasattr(state, "value") else state
    return np.asarray(list(values), dtype=float)


def compare_arm_coordinates(mpc_state, sensor_joint_q, waist_dof=1):
    """Compare the 14 arm coordinates exposed by MPC and raw sensors."""

    mpc_state = np.asarray(mpc_state, dtype=float).reshape(-1)
    sensor_joint_q = np.asarray(sensor_joint_q, dtype=float).reshape(-1)
    mpc_start = MPC_FIXED_STATE_PREFIX + int(waist_dof)
    if mpc_state.size < mpc_start + ARM_COUNT:
        raise ValueError("MPC state is too short for 14 arm joints")
    if sensor_joint_q.size < SENSOR_ARM_START + ARM_COUNT:
        raise ValueError("sensor joint_q is too short for 14 arm joints")

    mpc_deg = np.rad2deg(mpc_state[mpc_start:mpc_start + ARM_COUNT])
    sensor_deg = np.rad2deg(
        sensor_joint_q[SENSOR_ARM_START:SENSOR_ARM_START + ARM_COUNT]
    )
    difference_deg = sensor_deg - mpc_deg
    return {
        "mpc_start": int(mpc_start),
        "mpc_deg": mpc_deg,
        "sensor_deg": sensor_deg,
        "difference_deg": difference_deg,
        "maximum_difference_deg": float(np.max(np.abs(difference_deg))),
    }


def maximum_sample_spread_deg(samples_rad):
    samples = np.asarray(samples_rad, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != ARM_COUNT:
        raise ValueError("arm samples must have shape (N, 14)")
    return float(np.max(np.rad2deg(np.ptp(samples, axis=0))))


def build_control_checks(
        coordinate_report,
        measured_arm_deg,
        measured_spread_deg,
        right_modes,
        right_kp,
        reported_mode,
        safe_topic_active,
        timed_topic_active):
    measured_arm_deg = np.asarray(measured_arm_deg, dtype=float).reshape(ARM_COUNT)
    right_modes = list(right_modes)
    right_kp = list(right_kp)
    return {
        "mpc_sensor_coordinates_match": (
            float(coordinate_report["maximum_difference_deg"]) <= 0.10
        ),
        "measured_arm_stable": float(measured_spread_deg) <= 0.15,
        "third_joints_in_recovery_range": (
            abs(float(measured_arm_deg[2])) < 45.0
            and abs(float(measured_arm_deg[9])) < 45.0
        ),
        "reported_arm_mode_2": int(reported_mode) == 2,
        "right_modes_active": (
            len(right_modes) == 7 and all(int(value) == 2 for value in right_modes)
        ),
        "right_gains_active": (
            len(right_kp) == 7 and all(float(value) > 0.0 for value in right_kp)
        ),
        "validated_topic_idle": not bool(safe_topic_active),
        "experimental_timed_topic_idle": not bool(timed_topic_active),
    }


def run_ros():
    import rospy
    from kuavo_msgs.msg import armTargetPoses, jointCmd, sensorsData
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
    from ocs2_msgs.msg import mpc_observation
    from sensor_msgs.msg import JointState

    rospy.init_node("scene3_arm_control_audit", anonymous=True)

    sensor_messages = []
    arm_samples = []
    for _ in range(5):
        message = rospy.wait_for_message(
            "/sensors_data_raw", sensorsData, timeout=8.0
        )
        sensor_messages.append(message)
        arm_samples.append(np.asarray(
            message.joint_data.joint_q[
                SENSOR_ARM_START:SENSOR_ARM_START + ARM_COUNT
            ],
            dtype=float,
        ))
        rospy.sleep(0.05)

    measured_arm = np.median(np.asarray(arm_samples), axis=0)
    measured_spread = maximum_sample_spread_deg(arm_samples)
    mpc_message = rospy.wait_for_message(
        "/humanoid_wbc_observation", mpc_observation, timeout=8.0
    )
    low_level = rospy.wait_for_message("/joint_cmd", jointCmd, timeout=8.0)
    waist_dof = int(rospy.get_param("/mpc/mpcWaistDof", 1))
    coordinate_report = compare_arm_coordinates(
        extract_mpc_state_values(mpc_message),
        sensor_messages[-1].joint_data.joint_q,
        waist_dof=waist_dof,
    )

    rospy.wait_for_service("/humanoid_get_arm_ctrl_mode", timeout=8.0)
    mode_proxy = rospy.ServiceProxy(
        "/humanoid_get_arm_ctrl_mode", changeArmCtrlMode
    )
    request = changeArmCtrlModeRequest()
    request.control_mode = 0
    mode_response = mode_proxy(request)
    reported_mode = int(getattr(mode_response, "mode", -1))

    def topic_active(topic, message_type):
        try:
            rospy.wait_for_message(topic, message_type, timeout=0.8)
            return True
        except rospy.ROSException:
            return False

    safe_active = topic_active(SAFE_ARM_TOPIC, JointState)
    timed_active = topic_active(EXPERIMENTAL_TIMED_TOPIC, armTargetPoses)
    low_start = LOW_LEVEL_ARM_START + RIGHT_ARM_START
    right_modes = list(low_level.control_modes[low_start:low_start + 7])
    right_kp = list(low_level.joint_kp[low_start:low_start + 7])
    measured_deg = np.rad2deg(measured_arm)
    checks = build_control_checks(
        coordinate_report,
        measured_deg,
        measured_spread,
        right_modes,
        right_kp,
        reported_mode,
        safe_active,
        timed_active,
    )

    print("MPC arm start: {}".format(coordinate_report["mpc_start"]))
    print("MPC arm: {}deg".format(
        np.round(coordinate_report["mpc_deg"], 3).tolist()
    ))
    print("Sensor arm: {}deg".format(
        np.round(coordinate_report["sensor_deg"], 3).tolist()
    ))
    print("MPC/sensor maximum difference: {:.4f}deg".format(
        coordinate_report["maximum_difference_deg"]
    ))
    print("Measured arm five-sample spread: {:.4f}deg".format(measured_spread))
    print("Reported arm mode: {}".format(reported_mode))
    print("Right modes: {}".format(right_modes))
    print("Right kp: {}".format([round(float(v), 3) for v in right_kp]))
    print("Topic activity: {}={} {}={}".format(
        SAFE_ARM_TOPIC,
        safe_active,
        EXPERIMENTAL_TIMED_TOPIC,
        timed_active,
    ))
    print("Audit checks: {}".format(checks))
    if all(checks.values()):
        print("SCENE3_ARM_CONTROL_AUDIT_OK")
    else:
        print("SCENE3_ARM_CONTROL_AUDIT_BLOCKED")
    print("Read-only audit complete; no control command or parameter was sent")
    return 0 if all(checks.values()) else 2


def main():
    return run_ros()


if __name__ == "__main__":
    raise SystemExit(main())
