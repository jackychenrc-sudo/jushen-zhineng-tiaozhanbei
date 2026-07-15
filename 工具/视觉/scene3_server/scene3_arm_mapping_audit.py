#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validated read-only Scene3 v52 arm-index audit.

This process deliberately creates no ROS Publisher and never calls an arm,
base, gripper, simulator-state, or scoring service.  It only reads one sensor
message, calls FK, reads TF, and requests one IK calculation.
"""

from __future__ import print_function

import math
import sys

import rospy
import tf2_ros
from kuavo_msgs.msg import ikSolveParam, sensorsData, twoArmHandPoseCmd
from kuavo_msgs.srv import fkSrv, twoArmHandPoseCmdSrv


SENSOR_TOPIC = "/sensors_data_raw"
FK_SERVICE = "/ik/fk_srv"
IK_SERVICE = "/ik/two_arm_hand_pose_cmd_srv"
LEFT_FRAME = "zarm_l7_end_effector"
RIGHT_FRAME = "zarm_r7_end_effector"
TF_POSITION_LIMIT_M = 0.010
TF_ORIENTATION_LIMIT_DEG = 3.0
IK_POSITION_LIMIT_M = 0.010
IK_ORIENTATION_LIMIT_DEG = 3.0
PERTURBATION_DEG = 0.5
CROSS_POSITION_LIMIT_M = 1.0e-5
CROSS_ORIENTATION_LIMIT_DEG = 1.0e-3


def vector(values):
    return [float(value) for value in values]


def subtract(left, right):
    return [float(a) - float(b) for a, b in zip(left, right)]


def norm(values):
    return math.sqrt(sum(float(value) ** 2 for value in values))


def maximum_absolute(values):
    return max([abs(float(value)) for value in values] or [0.0])


def normalized_quaternion(values):
    result = vector(values)
    length = norm(result)
    if length <= 1.0e-12:
        raise RuntimeError("zero-length quaternion")
    return [value / length for value in result]


def quaternion_error_degrees(left, right):
    q_left = normalized_quaternion(left)
    q_right = normalized_quaternion(right)
    dot = abs(sum(a * b for a, b in zip(q_left, q_right)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def pose_from_hand(hand_pose):
    return vector(hand_pose.pos_xyz), normalized_quaternion(hand_pose.quat_xyzw)


def pose_from_transform(transform):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return (
        [translation.x, translation.y, translation.z],
        normalized_quaternion([rotation.x, rotation.y, rotation.z, rotation.w]),
    )


def pose_error(left, right):
    return (
        norm(subtract(left[0], right[0])),
        quaternion_error_degrees(left[1], right[1]),
    )


def format_vector(values, digits=6):
    return "[{}]".format(", ".join(("{: .%df}" % digits).format(v) for v in values))


def copy_hand_pose(destination, source, q0):
    destination.pos_xyz = vector(source.pos_xyz)
    destination.quat_xyzw = vector(source.quat_xyzw)
    destination.elbow_pos_xyz = [0.0, 0.0, 0.0]
    destination.joint_angles = vector(q0)


def main():
    rospy.init_node("scene3_arm_mapping_audit_read_only", anonymous=True)
    print("SCENE3_ARM_MAPPING_AUDIT_START")
    print("CONTROL_PUBLISHERS_CREATED=0")
    print("FORBIDDEN_TOPICS_OR_SERVICES_USED=0")

    for name in ("/robot_version", "/armRealDof", "/mpc/mpcArmsDof"):
        print("PARAM {}={}".format(name, rospy.get_param(name, "<missing>")))

    sensor = rospy.wait_for_message(SENSOR_TOPIC, sensorsData, timeout=8.0)
    joint_q = vector(sensor.joint_data.joint_q)
    print("SENSOR_JOINT_Q_LENGTH={}".format(len(joint_q)))
    print("SENSOR_JOINT_Q={}".format(format_vector(joint_q)))
    if len(joint_q) != 29:
        raise RuntimeError("expected exactly 29 joint_q values for v52")

    candidate_v52 = joint_q[13:27]
    candidate_legacy = joint_q[12:26]
    print("V52_LEFT_SENSOR_13_19={}".format(format_vector(candidate_v52[:7])))
    print("V52_RIGHT_SENSOR_20_26={}".format(format_vector(candidate_v52[7:])))
    print("LEGACY_WINDOW_12_25={}".format(format_vector(candidate_legacy)))

    rospy.wait_for_service(FK_SERVICE, timeout=8.0)
    fk_proxy = rospy.ServiceProxy(FK_SERVICE, fkSrv, persistent=True)

    def call_fk(q):
        response = fk_proxy(vector(q))
        if not getattr(response, "success", False):
            raise RuntimeError("{} returned success=false".format(FK_SERVICE))
        return response.hand_poses

    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
    # Keep the listener alive for the whole audit; otherwise its subscriptions
    # can be garbage-collected before the lookups below.
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(1.0)
    left_tf = pose_from_transform(
        tf_buffer.lookup_transform(
            "base_link", LEFT_FRAME, rospy.Time(0), rospy.Duration(5.0)
        )
    )
    right_tf = pose_from_transform(
        tf_buffer.lookup_transform(
            "base_link", RIGHT_FRAME, rospy.Time(0), rospy.Duration(5.0)
        )
    )
    print("TF_LEFT_POSITION={}".format(format_vector(left_tf[0])))
    print("TF_RIGHT_POSITION={}".format(format_vector(right_tf[0])))

    candidate_results = {}
    for label, candidate in (
        ("v52_13_27", candidate_v52),
        ("legacy_12_26", candidate_legacy),
    ):
        hand_poses = call_fk(candidate)
        left_fk = pose_from_hand(hand_poses.left_pose)
        right_fk = pose_from_hand(hand_poses.right_pose)
        left_error = pose_error(left_fk, left_tf)
        right_error = pose_error(right_fk, right_tf)
        candidate_results[label] = {
            "hand_poses": hand_poses,
            "left_error": left_error,
            "right_error": right_error,
        }
        print(
            "FK_TF_CANDIDATE={} LEFT_ERROR_M={:.6f} LEFT_ERROR_DEG={:.4f} "
            "RIGHT_ERROR_M={:.6f} RIGHT_ERROR_DEG={:.4f}".format(
                label,
                left_error[0],
                left_error[1],
                right_error[0],
                right_error[1],
            )
        )

    v52_hand_poses = candidate_results["v52_13_27"]["hand_poses"]
    base_left = pose_from_hand(v52_hand_poses.left_pose)
    base_right = pose_from_hand(v52_hand_poses.right_pose)
    epsilon = math.radians(PERTURBATION_DEG)
    perturbation_ok = True
    print("FK_VIRTUAL_PERTURBATION_DEG={:.3f}".format(PERTURBATION_DEG))
    for index in range(14):
        candidate = list(candidate_v52)
        candidate[index] += epsilon
        perturbed = call_fk(candidate)
        left_change = pose_error(pose_from_hand(perturbed.left_pose), base_left)
        right_change = pose_error(pose_from_hand(perturbed.right_pose), base_right)
        expected = "left" if index < 7 else "right"
        cross = right_change if index < 7 else left_change
        cross_ok = (
            cross[0] <= CROSS_POSITION_LIMIT_M
            and cross[1] <= CROSS_ORIENTATION_LIMIT_DEG
        )
        perturbation_ok = perturbation_ok and cross_ok
        print(
            "FK_PERTURB_INDEX={} EXPECTED={} LEFT_MM={:.5f} LEFT_DEG={:.5f} "
            "RIGHT_MM={:.5f} RIGHT_DEG={:.5f} CROSS_OK={}".format(
                index,
                expected,
                left_change[0] * 1000.0,
                left_change[1],
                right_change[0] * 1000.0,
                right_change[1],
                cross_ok,
            )
        )

    command = twoArmHandPoseCmd()
    command.hand_poses.header.frame_id = "base_link"
    command.use_custom_ik_param = True
    command.joint_angles_as_q0 = True
    parameters = ikSolveParam()
    parameters.major_optimality_tol = 1.0e-3
    parameters.major_feasibility_tol = 1.0e-3
    parameters.minor_feasibility_tol = 1.0e-3
    parameters.major_iterations_limit = 500
    parameters.oritation_constraint_tol = 1.0e-3
    parameters.pos_constraint_tol = 1.0e-3
    parameters.pos_cost_weight = 0.0
    parameters.constraint_mode = 3
    command.ik_param = parameters
    copy_hand_pose(
        command.hand_poses.left_pose, v52_hand_poses.left_pose, candidate_v52[:7]
    )
    copy_hand_pose(
        command.hand_poses.right_pose, v52_hand_poses.right_pose, candidate_v52[7:]
    )
    # A tiny virtual right-hand request makes the active half observable. It is
    # sent only to the calculation service and is never published to a controller.
    command.hand_poses.right_pose.pos_xyz[0] += 0.003

    rospy.wait_for_service(IK_SERVICE, timeout=8.0)
    ik_proxy = rospy.ServiceProxy(IK_SERVICE, twoArmHandPoseCmdSrv)
    ik_response = ik_proxy(command)
    print("IK_SERVICE_SUCCESS={}".format(bool(getattr(ik_response, "success", False))))
    if not getattr(ik_response, "success", False):
        raise RuntimeError(
            "{} failed: {}".format(
                IK_SERVICE, getattr(ik_response, "error_reason", "")
            )
        )

    q_arm = vector(ik_response.q_arm)
    response_left = vector(ik_response.hand_poses.left_pose.joint_angles)
    response_right = vector(ik_response.hand_poses.right_pose.joint_angles)
    if len(q_arm) != 14 or len(response_left) != 7 or len(response_right) != 7:
        raise RuntimeError("IK response dimensions are not 14/7/7")
    print("CURRENT_SENSOR_LEFT={}".format(format_vector(candidate_v52[:7])))
    print("CURRENT_SENSOR_RIGHT={}".format(format_vector(candidate_v52[7:])))
    print("IK_Q_ARM_FIRST7={}".format(format_vector(q_arm[:7])))
    print("IK_Q_ARM_LAST7={}".format(format_vector(q_arm[7:])))
    print("IK_HAND_POSES_LEFT_JOINTS={}".format(format_vector(response_left)))
    print("IK_HAND_POSES_RIGHT_JOINTS={}".format(format_vector(response_right)))

    left_contract_error = maximum_absolute(subtract(q_arm[:7], response_left))
    right_contract_error = maximum_absolute(subtract(q_arm[7:], response_right))
    left_delta_deg = [math.degrees(v) for v in subtract(q_arm[:7], candidate_v52[:7])]
    right_delta_deg = [math.degrees(v) for v in subtract(q_arm[7:], candidate_v52[7:])]
    print("IK_FIRST7_LEFT_ARRAY_MAX_ERROR={:.12g}".format(left_contract_error))
    print("IK_LAST7_RIGHT_ARRAY_MAX_ERROR={:.12g}".format(right_contract_error))
    print("IK_LEFT_DELTA_DEG={}".format(format_vector(left_delta_deg)))
    print("IK_RIGHT_DELTA_DEG={}".format(format_vector(right_delta_deg)))

    predicted = call_fk(q_arm)
    predicted_left = pose_from_hand(predicted.left_pose)
    predicted_right = pose_from_hand(predicted.right_pose)
    requested_left = pose_from_hand(command.hand_poses.left_pose)
    requested_right = pose_from_hand(command.hand_poses.right_pose)
    predicted_left_error = pose_error(predicted_left, requested_left)
    predicted_right_error = pose_error(predicted_right, requested_right)
    print(
        "IK_PREDICTED_LEFT_ERROR_M={:.6f} LEFT_ERROR_DEG={:.4f}".format(
            predicted_left_error[0], predicted_left_error[1]
        )
    )
    print(
        "IK_PREDICTED_RIGHT_ERROR_M={:.6f} RIGHT_ERROR_DEG={:.4f}".format(
            predicted_right_error[0], predicted_right_error[1]
        )
    )

    v52_left_error = candidate_results["v52_13_27"]["left_error"]
    v52_right_error = candidate_results["v52_13_27"]["right_error"]
    v52_tf_ok = all(
        (
            v52_left_error[0] <= TF_POSITION_LIMIT_M,
            v52_left_error[1] <= TF_ORIENTATION_LIMIT_DEG,
            v52_right_error[0] <= TF_POSITION_LIMIT_M,
            v52_right_error[1] <= TF_ORIENTATION_LIMIT_DEG,
        )
    )
    ik_contract_ok = left_contract_error <= 1.0e-10 and right_contract_error <= 1.0e-10
    ik_fk_ok = all(
        (
            predicted_left_error[0] <= IK_POSITION_LIMIT_M,
            predicted_left_error[1] <= IK_ORIENTATION_LIMIT_DEG,
            predicted_right_error[0] <= IK_POSITION_LIMIT_M,
            predicted_right_error[1] <= IK_ORIENTATION_LIMIT_DEG,
        )
    )
    print("CHECK_V52_SENSOR_WINDOW_TO_TF={}".format(v52_tf_ok))
    print("CHECK_FK_FIRST7_LEFT_LAST7_RIGHT={}".format(perturbation_ok))
    print("CHECK_IK_Q_ARM_TO_HAND_POSES={}".format(ik_contract_ok))
    print("CHECK_IK_PREDICTED_POSE={}".format(ik_fk_ok))
    if v52_tf_ok and perturbation_ok and ik_contract_ok and ik_fk_ok:
        print("SCENE3_ARM_MAPPING_AUDIT_PASS_NO_CONTROL_SENT")
        return 0
    print("SCENE3_ARM_MAPPING_AUDIT_BLOCKED_NO_CONTROL_SENT")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # Fail closed and preserve a clear terminal marker.
        print("SCENE3_ARM_MAPPING_AUDIT_ERROR={}".format(error))
        print("SCENE3_ARM_MAPPING_AUDIT_BLOCKED_NO_CONTROL_SENT")
        sys.exit(2)

