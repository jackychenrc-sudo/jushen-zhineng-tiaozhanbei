#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan one bounded 6D gripper-alignment step for Scene3.

The physical gripper is represented by three measured axes:

* forward: from the gripper base towards the fingers;
* gap: from the left finger towards the right finger;
* up: ``forward x gap``.

The target frame points forward horizontally towards the locked tray, keeps
the finger gap horizontal across the tray, and keeps the gripper up axis
vertical.  A fixed transform calibrated from live TF maps that physical frame
to the official IK end-effector frame.  The planner rotates only a bounded
amount and moves the official EEF as needed to keep the physical TCP fixed.

This file creates no arm/base publisher and no claw service proxy.
"""

from __future__ import print_function

import argparse
import math

import numpy as np

from scene3_6d_pose_dry_run import (
    IK_MODE_POSITION_AND_ORIENTATION_HARD,
    REFERENCE_PARAM,
    extract_arm_joints,
    extract_ik_solution,
    normalize_quaternion,
    position_error,
    quaternion_error_degrees,
)


PLAN_PARAM = "/challenge_cup_task_template/scene3/gripper_6d_align_plan"
TARGET_PARAM = "/challenge_cup_task_template/scene3/locked_target_base_xyz"
TARGET_ODOM_PARAM = (
    "/challenge_cup_task_template/scene3/locked_target_odom_xyz"
)
GRIPPER_BASE_FRAME = "right_gripper_base"
LEFT_FINGER_FRAME = "right_gripper_left_inner_knuckle"
RIGHT_FINGER_FRAME = "right_gripper_right_inner_knuckle"


def normalize(vector, name="vector"):
    values = np.asarray(vector, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("{} contains non-finite values".format(name))
    norm = float(np.linalg.norm(values))
    if norm < 1e-9:
        raise ValueError("{} has zero length".format(name))
    return values / norm


def quaternion_to_matrix(quaternion_xyzw):
    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z),
         2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),
         1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),
         2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def matrix_to_quaternion(matrix):
    """Convert a proper rotation matrix to ROS xyzw order."""

    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(
                max(0.0, 1.0 + rotation[0, 0]
                    - rotation[1, 1] - rotation[2, 2])
            ) * 2.0
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
            w = (rotation[2, 1] - rotation[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(
                max(0.0, 1.0 + rotation[1, 1]
                    - rotation[0, 0] - rotation[2, 2])
            ) * 2.0
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = math.sqrt(
                max(0.0, 1.0 + rotation[2, 2]
                    - rotation[0, 0] - rotation[1, 1])
            ) * 2.0
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    return normalize_quaternion([x, y, z, w])


def rotation_error_degrees(target, actual):
    relative = np.asarray(target, dtype=float).dot(
        np.asarray(actual, dtype=float).T
    )
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rotation_vector(matrix):
    """Return the shortest rotation vector for a rotation matrix."""

    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3, dtype=float)
    if math.pi - angle < 1e-5:
        values, vectors = np.linalg.eig(rotation)
        index = int(np.argmin(np.abs(values - 1.0)))
        axis = normalize(np.real(vectors[:, index]), "pi_rotation_axis")
        return axis * angle
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ], dtype=float) / (2.0 * math.sin(angle))
    return normalize(axis, "rotation_axis") * angle


def matrix_from_rotation_vector(vector):
    values = np.asarray(vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(values))
    if angle < 1e-12:
        return np.eye(3)
    axis = values / angle
    x, y, z = axis
    skew = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=float)
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * skew.dot(skew)
    )


def bounded_rotation(current, target, maximum_step_degrees):
    current_rotation = np.asarray(current, dtype=float).reshape(3, 3)
    target_rotation = np.asarray(target, dtype=float).reshape(3, 3)
    relative = target_rotation.dot(current_rotation.T)
    vector = rotation_vector(relative)
    full_angle = float(np.linalg.norm(vector))
    maximum = math.radians(float(maximum_step_degrees))
    if full_angle <= maximum:
        return target_rotation.copy(), math.degrees(full_angle), math.degrees(full_angle)
    step = vector * (maximum / full_angle)
    return (
        matrix_from_rotation_vector(step).dot(current_rotation),
        math.degrees(full_angle),
        float(maximum_step_degrees),
    )


def gripper_geometry(base_xyz, left_xyz, right_xyz, tcp_extension_m=0.045):
    base = np.asarray(base_xyz, dtype=float).reshape(3)
    left = np.asarray(left_xyz, dtype=float).reshape(3)
    right = np.asarray(right_xyz, dtype=float).reshape(3)
    hinge_midpoint = 0.5 * (left + right)
    forward = normalize(hinge_midpoint - base, "gripper_forward")
    gap_raw = right - left
    gap = normalize(
        gap_raw - float(np.dot(gap_raw, forward)) * forward,
        "gripper_gap",
    )
    up = normalize(np.cross(forward, gap), "gripper_up")
    gap = normalize(np.cross(up, forward), "orthogonal_gap")
    rotation = np.column_stack([forward, gap, up])
    tcp = hinge_midpoint + float(tcp_extension_m) * forward
    return {
        "tcp": tcp,
        "rotation": rotation,
        "forward": forward,
        "gap": gap,
        "up": up,
        "finger_gap_m": float(np.linalg.norm(right - left)),
    }


def desired_gripper_rotation(tcp_xyz, tray_xyz, up_sign=1.0):
    tcp = np.asarray(tcp_xyz, dtype=float).reshape(3)
    tray = np.asarray(tray_xyz, dtype=float).reshape(3)
    direction = tray - tcp
    direction[2] = 0.0
    forward = normalize(direction, "horizontal_tray_direction")
    up = np.array([0.0, 0.0, 1.0 if up_sign >= 0.0 else -1.0])
    gap = normalize(np.cross(up, forward), "desired_gap")
    up = normalize(np.cross(forward, gap), "desired_up")
    return np.column_stack([forward, gap, up])


def choose_target_gripper_rotation(current_rotation, tcp_xyz, tray_xyz):
    candidates = [
        desired_gripper_rotation(tcp_xyz, tray_xyz, up_sign=1.0),
        desired_gripper_rotation(tcp_xyz, tray_xyz, up_sign=-1.0),
    ]
    errors = [
        rotation_error_degrees(candidate, current_rotation)
        for candidate in candidates
    ]
    index = int(np.argmin(errors))
    return candidates[index], (1.0 if index == 0 else -1.0), errors[index]


def target_eef_pose(
        eef_position,
        eef_rotation,
        physical_tcp,
        physical_rotation,
        final_physical_rotation,
        maximum_step_degrees):
    """Compute a bounded EEF target while holding the physical TCP fixed."""

    eef_position = np.asarray(eef_position, dtype=float).reshape(3)
    eef_rotation = np.asarray(eef_rotation, dtype=float).reshape(3, 3)
    physical_tcp = np.asarray(physical_tcp, dtype=float).reshape(3)
    physical_rotation = np.asarray(physical_rotation, dtype=float).reshape(3, 3)
    final_physical_rotation = np.asarray(
        final_physical_rotation, dtype=float
    ).reshape(3, 3)

    eef_to_physical = eef_rotation.T.dot(physical_rotation)
    eef_to_tcp = eef_rotation.T.dot(physical_tcp - eef_position)
    final_eef_rotation = final_physical_rotation.dot(eef_to_physical.T)
    step_eef_rotation, full_angle, step_angle = bounded_rotation(
        eef_rotation,
        final_eef_rotation,
        maximum_step_degrees,
    )
    step_eef_position = physical_tcp - step_eef_rotation.dot(eef_to_tcp)
    step_physical_rotation = step_eef_rotation.dot(eef_to_physical)
    return {
        "eef_position": step_eef_position,
        "eef_rotation": step_eef_rotation,
        "physical_rotation": step_physical_rotation,
        "eef_to_tcp": eef_to_tcp,
        "eef_to_physical": eef_to_physical,
        "full_angle_deg": full_angle,
        "step_angle_deg": step_angle,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plan one TCP-held 6D gripper alignment step"
    )
    parser.add_argument("--maximum-angle-step", type=float, default=5.0)
    parser.add_argument("--maximum-right-joint-delta", type=float, default=8.0)
    parser.add_argument("--maximum-position-error", type=float, default=0.006)
    parser.add_argument("--maximum-orientation-error", type=float, default=2.5)
    parser.add_argument("--maximum-predicted-tcp-shift", type=float, default=0.008)
    parser.add_argument("--minimum-axis-standoff", type=float, default=0.070)
    parser.add_argument("--tcp-extension", type=float, default=0.045)
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser


def run_ros(args):
    import rospy
    import tf2_ros
    from geometry_msgs.msg import PointStamped
    from kuavo_msgs.msg import ikSolveParam, sensorsData, twoArmHandPoseCmd
    from kuavo_msgs.srv import fkSrv, twoArmHandPoseCmdSrv

    rospy.init_node("scene3_gripper_6d_align_plan", anonymous=True)
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(1.5)

    def frame_xyz(frame_name):
        transform = tf_buffer.lookup_transform(
            "base_link", frame_name, rospy.Time(0), rospy.Duration(3.0)
        )
        translation = transform.transform.translation
        return np.array(
            [translation.x, translation.y, translation.z], dtype=float
        )

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
        return np.array([
            transformed.point.x,
            transformed.point.y,
            transformed.point.z,
        ], dtype=float)

    sensor = rospy.wait_for_message(
        "/sensors_data_raw", sensorsData, timeout=float(args.timeout)
    )
    current_arm, mapping = extract_arm_joints(sensor)
    current_arm = [float(value) for value in current_arm]
    target_tray_odom = np.asarray(
        rospy.get_param(TARGET_ODOM_PARAM), dtype=float
    )
    if (
        target_tray_odom.shape != (3,)
        or not np.all(np.isfinite(target_tray_odom))
    ):
        raise RuntimeError("world-locked tray target is invalid")
    target_tray = point_in_base(target_tray_odom, "odom")
    local_edge = np.asarray(
        rospy.get_param(TARGET_PARAM, target_tray.tolist()), dtype=float
    )

    geometry = gripper_geometry(
        frame_xyz(GRIPPER_BASE_FRAME),
        frame_xyz(LEFT_FINGER_FRAME),
        frame_xyz(RIGHT_FINGER_FRAME),
        tcp_extension_m=args.tcp_extension,
    )

    rospy.wait_for_service("/ik/fk_srv", timeout=float(args.timeout))
    fk_proxy = rospy.ServiceProxy("/ik/fk_srv", fkSrv)

    def call_fk(arm):
        response = fk_proxy([float(value) for value in arm])
        if not getattr(response, "success", False):
            raise RuntimeError("/ik/fk_srv failed")
        return response.hand_poses

    current_poses = call_fk(current_arm)
    current_eef_position = np.asarray(
        current_poses.right_pose.pos_xyz, dtype=float
    )
    current_eef_quaternion = normalize_quaternion(
        current_poses.right_pose.quat_xyzw
    )
    current_eef_rotation = quaternion_to_matrix(current_eef_quaternion)

    final_physical_rotation, up_sign, full_physical_error = (
        choose_target_gripper_rotation(
            geometry["rotation"], geometry["tcp"], target_tray
        )
    )
    target = target_eef_pose(
        current_eef_position,
        current_eef_rotation,
        geometry["tcp"],
        geometry["rotation"],
        final_physical_rotation,
        maximum_step_degrees=args.maximum_angle_step,
    )
    target_eef_quaternion = matrix_to_quaternion(target["eef_rotation"])

    request = twoArmHandPoseCmd()
    request.hand_poses.header.frame_id = "base_link"
    request.use_custom_ik_param = True
    request.joint_angles_as_q0 = True
    ik_param = ikSolveParam()
    ik_param.major_optimality_tol = 1e-3
    ik_param.major_feasibility_tol = 1e-3
    ik_param.minor_feasibility_tol = 1e-3
    ik_param.major_iterations_limit = 500
    ik_param.oritation_constraint_tol = 1e-3
    ik_param.pos_constraint_tol = 1e-3
    ik_param.pos_cost_weight = 0.0
    ik_param.constraint_mode = IK_MODE_POSITION_AND_ORIENTATION_HARD
    request.ik_param = ik_param

    request.hand_poses.left_pose.pos_xyz = list(
        current_poses.left_pose.pos_xyz
    )
    request.hand_poses.left_pose.quat_xyzw = list(
        current_poses.left_pose.quat_xyzw
    )
    request.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.left_pose.joint_angles = list(current_arm[:7])
    request.hand_poses.right_pose.pos_xyz = target["eef_position"].tolist()
    request.hand_poses.right_pose.quat_xyzw = target_eef_quaternion
    request.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.right_pose.joint_angles = list(current_arm[7:])

    rospy.wait_for_service(
        "/ik/two_arm_hand_pose_cmd_srv", timeout=float(args.timeout)
    )
    ik_proxy = rospy.ServiceProxy(
        "/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv
    )
    response = ik_proxy(request)
    if not getattr(response, "success", False):
        print("IK success: False")
        print("IK reason: {}".format(getattr(response, "error_reason", "")))
        print("GRIPPER_6D_ALIGN_PLAN_BLOCKED")
        print("No base, arm or claw command was sent")
        return 2

    raw_solution = extract_ik_solution(response, current_arm)
    candidate_arm = list(current_arm)
    candidate_arm[7:14] = raw_solution[7:14]
    predicted = call_fk(candidate_arm)
    predicted_eef_position = np.asarray(
        predicted.right_pose.pos_xyz, dtype=float
    )
    predicted_eef_quaternion = normalize_quaternion(
        predicted.right_pose.quat_xyzw
    )
    predicted_eef_rotation = quaternion_to_matrix(predicted_eef_quaternion)
    predicted_tcp = (
        predicted_eef_position
        + predicted_eef_rotation.dot(target["eef_to_tcp"])
    )
    predicted_physical_rotation = predicted_eef_rotation.dot(
        target["eef_to_physical"]
    )

    right_delta = [
        math.degrees(candidate_arm[index] - current_arm[index])
        for index in range(7, 14)
    ]
    position_residual = position_error(
        target["eef_position"], predicted_eef_position
    )
    orientation_residual = quaternion_error_degrees(
        target_eef_quaternion, predicted_eef_quaternion
    )
    predicted_tcp_shift = float(np.linalg.norm(
        predicted_tcp - geometry["tcp"]
    ))
    before_error = rotation_error_degrees(
        final_physical_rotation, geometry["rotation"]
    )
    predicted_error = rotation_error_degrees(
        final_physical_rotation, predicted_physical_rotation
    )
    target_vector = target_tray - geometry["tcp"]
    axis_standoff = float(np.dot(
        target_vector, final_physical_rotation[:, 0]
    ))

    checks = {
        "ik_success": True,
        "position_hard": position_residual <= args.maximum_position_error,
        "orientation_hard": (
            orientation_residual <= args.maximum_orientation_error
        ),
        "right_joint_delta_bounded": (
            max(abs(value) for value in right_delta)
            <= args.maximum_right_joint_delta
        ),
        "left_command_frozen": True,
        "tcp_held": (
            predicted_tcp_shift <= args.maximum_predicted_tcp_shift
        ),
        "orientation_improved": predicted_error + 1.0 < before_error,
        "tray_still_ahead": axis_standoff >= args.minimum_axis_standoff,
        "values_finite": bool(np.all(np.isfinite(
            candidate_arm
            + predicted_eef_position.tolist()
            + predicted_eef_quaternion
        ))),
    }

    print("Joint mapping: {}".format(mapping))
    print("World-locked tray center odom: {}".format(
        np.round(target_tray_odom, 4).tolist()
    ))
    print("World-locked tray center base_link: {}".format(
        np.round(target_tray, 4).tolist()
    ))
    print("Separate local edge/prompt base_link: {}".format(
        np.round(local_edge, 4).tolist()
    ))
    print("Physical TCP before: {}".format(
        np.round(geometry["tcp"], 4).tolist()
    ))
    print("Physical forward before: {}".format(
        np.round(geometry["forward"], 4).tolist()
    ))
    print("Physical gap axis before: {}".format(
        np.round(geometry["gap"], 4).tolist()
    ))
    print("Physical up axis before: {}".format(
        np.round(geometry["up"], 4).tolist()
    ))
    print("Selected desired up sign: {:+.0f}Z".format(up_sign))
    print("Full physical orientation error: {:.2f}deg".format(
        full_physical_error
    ))
    print("Planned bounded orientation step: {:.2f}deg".format(
        target["step_angle_deg"]
    ))
    print("Target EEF translation: {}m".format(np.round(
        target["eef_position"] - current_eef_position, 5
    ).tolist()))
    print("Target EEF quaternion: {}".format(
        [round(value, 6) for value in target_eef_quaternion]
    ))
    print("Right-arm joint delta: {}deg".format(
        [round(value, 3) for value in right_delta]
    ))
    print("Predicted physical TCP: {}".format(
        np.round(predicted_tcp, 4).tolist()
    ))
    print("Predicted TCP shift: {:.1f}mm".format(
        predicted_tcp_shift * 1000.0
    ))
    print("Physical orientation error: {:.2f}deg -> {:.2f}deg".format(
        before_error, predicted_error
    ))
    print("Predicted EEF residual: position={:.4f}m orientation={:.2f}deg".format(
        position_residual, orientation_residual
    ))
    print("Tray forward standoff after full alignment: {:.1f}mm".format(
        axis_standoff * 1000.0
    ))
    print("Safety checks: {}".format(checks))

    if not all(checks.values()):
        if rospy.has_param(PLAN_PARAM):
            rospy.delete_param(PLAN_PARAM)
        print("GRIPPER_6D_ALIGN_PLAN_BLOCKED")
        print("No base, arm or claw command was sent")
        return 2

    reference = rospy.get_param(REFERENCE_PARAM, None)
    if not isinstance(reference, (list, tuple)) or len(reference) != 14:
        raise RuntimeError("arm command reference is unavailable")
    reference = [float(value) for value in reference]
    target_reference = [
        reference[index]
        + math.degrees(candidate_arm[index] - current_arm[index])
        for index in range(14)
    ]
    rospy.set_param(PLAN_PARAM, {
        "source_arm_rad": current_arm,
        "target_arm_rad": candidate_arm,
        "source_reference_deg": reference,
        "target_reference_deg": target_reference,
        "target_eef_position": target["eef_position"].tolist(),
        "target_eef_quaternion_xyzw": target_eef_quaternion,
        "physical_tcp_before": geometry["tcp"].tolist(),
        "desired_physical_rotation": final_physical_rotation.reshape(-1).tolist(),
        "physical_error_before_deg": float(before_error),
        "predicted_physical_error_deg": float(predicted_error),
        "planned_orientation_step_deg": float(target["step_angle_deg"]),
        "target_tray_base_xyz": target_tray.tolist(),
        "target_tray_odom_xyz": target_tray_odom.tolist(),
        "tcp_extension_m": float(args.tcp_extension),
        "up_sign": float(up_sign),
    })
    print("GRIPPER_6D_ALIGN_PLAN_OK")
    print("Plan saved; no base, arm or claw command was sent")
    return 0


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

