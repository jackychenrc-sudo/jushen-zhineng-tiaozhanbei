#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One bounded wrist-camera look-at correction after senior pregrasp.

The senior Scene3 code remains responsible for target selection, walking and
the pregrasp.  The exact selected tray point is latched before arm motion.  This
helper keeps the current right-hand Cartesian position and requests a verified
position-hard/orientation-hard IK solution that turns the right-camera +Z
optical axis towards that locked point.  One command is capped at eight degrees.
The claw stays open and no base, close, lift or extraction command is sent.

Default mode is calculation-only.  Real arm movement requires ``--execute``
and the exact confirmation token ``WRIST_LOOK_AT_8DEG``.
"""

from __future__ import print_function

import argparse
import math
import os
import sys

import numpy as np


EXECUTION_CONFIRMATION = "WRIST_LOOK_AT_8DEG"
DEFAULT_SENIOR_DIR = "/root/kuavo_ws/src/challenge_cup_task_template/scripts"
DEFAULT_TARGET_TOPIC = "/challenge_cup_task_template/scene3/locked_target_base"
DEFAULT_TARGET_PARAM = "/challenge_cup_task_template/scene3/locked_target_base_xyz"
IK_MODE_POS_HARD_ORI_HARD = 0x03


def normalize_vector(value, label="vector"):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("{} must be finite xyz".format(label))
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("{} has zero length".format(label))
    return vector / norm


def quaternion_to_matrix(quaternion_xyzw):
    q = np.asarray(quaternion_xyzw, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must be finite xyzw")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("quaternion has zero length")
    x, y, z, w = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def matrix_to_quaternion(rotation_matrix):
    matrix = np.asarray(rotation_matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation matrix must be finite 3x3")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(
                max(1e-12, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            ) * 2.0
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(
                max(1e-12, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            ) * 2.0
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = math.sqrt(
                max(1e-12, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            ) * 2.0
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
    result = np.array([x, y, z, w], dtype=float)
    result /= float(np.linalg.norm(result))
    if result[3] < 0.0:
        result *= -1.0
    return result


def quaternion_angle_degrees(first_xyzw, second_xyzw):
    first = np.array(first_xyzw, dtype=float, copy=True)
    second = np.array(second_xyzw, dtype=float, copy=True)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("quaternions must contain four xyzw values")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-8 or second_norm < 1e-8:
        raise ValueError("quaternions must have nonzero length")
    first /= first_norm
    second /= second_norm
    cosine_half_angle = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine_half_angle))


def locked_target_xyz(values):
    target = np.asarray(values, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("locked senior target must be finite base_link xyz")
    return target


def configure_pose_hard_ik_param(param, position_tolerance_m=0.004,
                                 orientation_tolerance_rad=0.02):
    position_tolerance = float(position_tolerance_m)
    orientation_tolerance = float(orientation_tolerance_rad)
    if not 0.001 <= position_tolerance <= 0.010:
        raise ValueError("position IK tolerance must be within 1-10mm")
    if not 0.001 <= orientation_tolerance <= 0.05:
        raise ValueError("orientation IK tolerance must be within 0.001-0.05rad")
    param.major_optimality_tol = 1e-3
    param.major_feasibility_tol = 1e-3
    param.minor_feasibility_tol = 1e-3
    param.major_iterations_limit = 500
    param.oritation_constraint_tol = orientation_tolerance
    param.pos_constraint_tol = position_tolerance
    param.pos_cost_weight = 10.0
    # Official plantIK bit mask: 0x03 = position hard + orientation hard.
    param.constraint_mode = IK_MODE_POS_HARD_ORI_HARD
    return param


def build_pose_hard_request(task, current_joints, current_poses,
                            right_position, right_quaternion,
                            position_tolerance_m=0.004,
                            orientation_tolerance_rad=0.02):
    joints = [float(value) for value in current_joints]
    if len(joints) != 14:
        raise ValueError("current arm state must contain fourteen joints")

    request = task.twoArmHandPoseCmd()
    request.hand_poses.header.frame_id = "base_link"
    request.use_custom_ik_param = True
    request.joint_angles_as_q0 = True
    configure_pose_hard_ik_param(
        request.ik_param,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
    )

    request.hand_poses.left_pose.pos_xyz = list(
        current_poses.left_pose.pos_xyz
    )
    request.hand_poses.left_pose.quat_xyzw = list(
        current_poses.left_pose.quat_xyzw
    )
    request.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.left_pose.joint_angles = joints[:7]

    request.hand_poses.right_pose.pos_xyz = list(map(float, right_position))
    request.hand_poses.right_pose.quat_xyzw = list(map(float, right_quaternion))
    request.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.right_pose.joint_angles = joints[7:]
    return request


def solve_pose_hard_ik(task, current_joints, current_poses, right_position,
                       right_quaternion, position_tolerance_m=0.004,
                       orientation_tolerance_rad=0.02):
    request = build_pose_hard_request(
        task,
        current_joints,
        current_poses,
        right_position,
        right_quaternion,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
    )
    task.rospy.wait_for_service(
        "/ik/two_arm_hand_pose_cmd_srv", timeout=5.0
    )
    proxy = task.rospy.ServiceProxy(
        "/ik/two_arm_hand_pose_cmd_srv", task.twoArmHandPoseCmdSrv
    )
    response = proxy(request)
    if not getattr(response, "success", False):
        raise RuntimeError(
            "position-hard/orientation-hard IK failed: {}".format(
                getattr(response, "error_reason", "")
            )
        )
    if len(response.q_arm) >= 14:
        return list(response.q_arm[:14])
    right_result = list(response.hand_poses.right_pose.joint_angles)
    if len(right_result) == 7:
        return list(current_joints[:7]) + right_result
    raise RuntimeError("pose-hard IK response does not contain fourteen joints")


def axis_angle_matrix(axis_xyz, angle_rad):
    axis = normalize_vector(axis_xyz, "rotation axis")
    x, y, z = axis
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    one_minus = 1.0 - cosine
    return np.array([
        [cosine + x * x * one_minus,
         x * y * one_minus - z * sine,
         x * z * one_minus + y * sine],
        [y * x * one_minus + z * sine,
         cosine + y * y * one_minus,
         y * z * one_minus - x * sine],
        [z * x * one_minus - y * sine,
         z * y * one_minus + x * sine,
         cosine + z * z * one_minus],
    ], dtype=float)


def optical_alignment_delta(camera_rotation, target_direction,
                            maximum_step_degrees=8.0):
    rotation = np.asarray(camera_rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("camera rotation must be 3x3")
    current_forward = normalize_vector(rotation[:, 2], "camera optical axis")
    desired_forward = normalize_vector(target_direction, "target direction")
    cross = np.cross(current_forward, desired_forward)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(current_forward, desired_forward), -1.0, 1.0))
    angle = math.atan2(cross_norm, dot)
    maximum = math.radians(max(0.5, float(maximum_step_degrees)))
    step = min(angle, maximum)
    if angle < math.radians(0.25):
        return np.eye(3, dtype=float), angle, 0.0
    if cross_norm < 1e-8:
        fallback = np.cross(current_forward, np.array([0.0, 0.0, 1.0]))
        if float(np.linalg.norm(fallback)) < 1e-8:
            fallback = np.cross(current_forward, np.array([0.0, 1.0, 0.0]))
        axis = normalize_vector(fallback, "fallback alignment axis")
    else:
        axis = cross / cross_norm
    return axis_angle_matrix(axis, step), angle, step


def plan_eef_orientation(current_eef_quaternion, current_camera_rotation,
                         target_direction, maximum_step_degrees=8.0):
    delta, angle, step = optical_alignment_delta(
        current_camera_rotation,
        target_direction,
        maximum_step_degrees=maximum_step_degrees,
    )
    current_eef_rotation = quaternion_to_matrix(current_eef_quaternion)
    desired_eef_rotation = np.matmul(delta, current_eef_rotation)
    desired_quaternion = matrix_to_quaternion(desired_eef_rotation)
    return desired_quaternion, angle, step


def stable_target(samples, maximum_spread_m=0.015):
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 3:
        raise ValueError("at least three target xyz samples are required")
    centre = np.median(values, axis=0)
    spread = float(np.max(np.linalg.norm(values - centre, axis=1)))
    if spread > float(maximum_spread_m):
        raise ValueError("head target is unstable: spread {:.4f}m".format(spread))
    return centre, spread


def validate_look_at(before_angle_rad, after_angle_rad, hand_translation_m,
                     minimum_reduction_degrees=1.0,
                     maximum_translation_m=0.035):
    before = math.degrees(float(before_angle_rad))
    after = math.degrees(float(after_angle_rad))
    translation = float(hand_translation_m)
    checks = {
        "optical_error_reduced": (
            before - after >= float(minimum_reduction_degrees)
        ),
        "translation_bounded": translation <= float(maximum_translation_m),
        "angles_finite": math.isfinite(before) and math.isfinite(after),
    }
    return bool(all(checks.values())), checks, before, after


def predict_camera_alignment(current_eef_position, current_eef_quaternion,
                             current_camera_origin, current_camera_rotation,
                             predicted_eef_position, predicted_eef_quaternion,
                             target_base):
    current_position = np.asarray(current_eef_position, dtype=float)
    predicted_position = np.asarray(predicted_eef_position, dtype=float)
    camera_origin = np.asarray(current_camera_origin, dtype=float)
    current_eef_rotation = quaternion_to_matrix(current_eef_quaternion)
    predicted_eef_rotation = quaternion_to_matrix(predicted_eef_quaternion)
    camera_rotation = np.asarray(current_camera_rotation, dtype=float)

    eef_to_camera_rotation = np.matmul(
        current_eef_rotation.T, camera_rotation
    )
    eef_to_camera_translation = np.matmul(
        current_eef_rotation.T, camera_origin - current_position
    )
    predicted_camera_rotation = np.matmul(
        predicted_eef_rotation, eef_to_camera_rotation
    )
    predicted_camera_origin = (
        predicted_position
        + np.matmul(predicted_eef_rotation, eef_to_camera_translation)
    )
    predicted_direction = (
        np.asarray(target_base, dtype=float) - predicted_camera_origin
    )
    predicted_forward = normalize_vector(
        predicted_camera_rotation[:, 2], "predicted camera optical axis"
    )
    predicted_direction = normalize_vector(
        predicted_direction, "predicted camera-to-target direction"
    )
    predicted_angle = math.acos(float(np.clip(
        np.dot(predicted_forward, predicted_direction), -1.0, 1.0
    )))
    return predicted_camera_origin, predicted_camera_rotation, predicted_angle


def validate_ik_plan(left_delta_degrees, right_delta_degrees,
                     fk_position_error_m, fk_orientation_error_degrees,
                     before_optical_angle_rad, predicted_optical_angle_rad,
                     minimum_right_joint_delta_degrees=0.10,
                     maximum_right_joint_delta_degrees=15.0,
                     maximum_left_joint_delta_degrees=2.0,
                     maximum_fk_position_error_m=0.008,
                     maximum_fk_orientation_error_degrees=2.0,
                     minimum_optical_reduction_degrees=1.0):
    left_delta = np.asarray(left_delta_degrees, dtype=float)
    right_delta = np.asarray(right_delta_degrees, dtype=float)
    left_maximum = float(np.max(np.abs(left_delta)))
    right_maximum = float(np.max(np.abs(right_delta)))
    optical_reduction = math.degrees(
        float(before_optical_angle_rad) - float(predicted_optical_angle_rad)
    )
    checks = {
        "right_joint_motion_nonzero": (
            right_maximum >= float(minimum_right_joint_delta_degrees)
        ),
        "right_joint_delta_bounded": (
            right_maximum <= float(maximum_right_joint_delta_degrees)
        ),
        "left_arm_held": (
            left_maximum <= float(maximum_left_joint_delta_degrees)
        ),
        "fk_position_hard": (
            float(fk_position_error_m) <= float(maximum_fk_position_error_m)
        ),
        "fk_orientation_hard": (
            float(fk_orientation_error_degrees)
            <= float(maximum_fk_orientation_error_degrees)
        ),
        "predicted_optical_error_reduced": (
            optical_reduction >= float(minimum_optical_reduction_degrees)
        ),
        "values_finite": bool(
            np.all(np.isfinite(left_delta))
            and np.all(np.isfinite(right_delta))
            and math.isfinite(float(fk_position_error_m))
            and math.isfinite(float(fk_orientation_error_degrees))
            and math.isfinite(optical_reduction)
        ),
    }
    return bool(all(checks.values())), checks, optical_reduction


def _transform_rotation(transform):
    q = transform.transform.rotation
    return quaternion_to_matrix([q.x, q.y, q.z, q.w])


def _transform_translation(transform):
    value = transform.transform.translation
    return np.array([value.x, value.y, value.z], dtype=float)


def load_senior(senior_dir):
    senior_dir = os.path.abspath(senior_dir)
    source_file = os.path.join(senior_dir, "challenge_task_3.py")
    if not os.path.isfile(source_file):
        raise RuntimeError("senior challenge_task_3.py not found: {}".format(
            source_file
        ))
    if senior_dir in sys.path:
        sys.path.remove(senior_dir)
    sys.path.insert(0, senior_dir)
    from challenge_task_3 import Scene3Task, rad_to_deg

    return Scene3Task, rad_to_deg


def wait_stable_target(args, rospy, tf_buffer, PointStamped):
    if args.target_param and rospy.has_param(args.target_param):
        target = locked_target_xyz(rospy.get_param(args.target_param))
        return target, 0.0, "locked ROS parameter"

    samples = []
    deadline = rospy.Time.now() + rospy.Duration(args.target_timeout)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        message = rospy.wait_for_message(
            args.target_topic, PointStamped, timeout=args.target_timeout
        )
        message.header.stamp = rospy.Time(0)
        target = tf_buffer.transform(
            message, "base_link", rospy.Duration(0.3)
        )
        samples.append([target.point.x, target.point.y, target.point.z])
        if len(samples) >= args.target_frames:
            target, spread = stable_target(
                samples[-args.target_frames:],
                maximum_spread_m=args.maximum_target_spread,
            )
            return target, spread, "locked ROS topic"
    raise RuntimeError(
        "no locked senior target; rerun scene3_senior_pregrasp_gate.py first"
    )


def camera_state(tf_buffer, rospy, camera_frame, target_base):
    transform = tf_buffer.lookup_transform(
        "base_link", camera_frame, rospy.Time(0), rospy.Duration(0.5)
    )
    origin = _transform_translation(transform)
    rotation = _transform_rotation(transform)
    direction = np.asarray(target_base, dtype=float) - origin
    distance = float(np.linalg.norm(direction))
    if not 0.06 <= distance <= 0.50:
        raise RuntimeError(
            "target-camera distance {:.3f}m is outside orientation gate".format(
                distance
            )
        )
    forward = normalize_vector(rotation[:, 2], "camera optical axis")
    direction_unit = normalize_vector(direction, "camera-to-target direction")
    angle = math.acos(float(np.clip(np.dot(forward, direction_unit), -1.0, 1.0)))
    return origin, rotation, direction, distance, angle


def run_ros(args):
    import rospy
    import tf2_geometry_msgs  # noqa: F401
    import tf2_ros
    from geometry_msgs.msg import PointStamped, Twist
    from sensor_msgs.msg import JointState

    rospy.init_node("scene3_wrist_camera_look_at", anonymous=True)
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    tf2_ros.TransformListener(tf_buffer)
    target, target_spread, target_source = wait_stable_target(
        args, rospy, tf_buffer, PointStamped
    )
    before_camera_origin, before_camera_rotation, direction, distance, before_angle = (
        camera_state(tf_buffer, rospy, args.camera_frame, target)
    )

    Scene3Task, rad_to_deg = load_senior(args.senior_dir)
    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    task = Scene3Task(cmd_vel_pub, arm_traj_pub)
    current_joints = task.read_current_arm_joints()
    current_poses = task.call_fk(current_joints)
    current_position = np.asarray(
        current_poses.right_pose.pos_xyz, dtype=float
    )
    current_quaternion = np.asarray(
        current_poses.right_pose.quat_xyzw, dtype=float
    )
    desired_quaternion, planned_angle, planned_step = plan_eef_orientation(
        current_quaternion,
        before_camera_rotation,
        direction,
        maximum_step_degrees=args.maximum_angle_step,
    )
    if planned_step < math.radians(0.25):
        raise RuntimeError("wrist camera already points at the target")

    print("Locked senior target base_link:", np.round(target, 4).tolist())
    print("Target source:", target_source)
    print("Target spread: {:.4f}m".format(target_spread))
    print("Camera-target distance: {:.4f}m".format(distance))
    print("Current optical error: {:.2f}deg".format(
        math.degrees(before_angle)
    ))
    print("Planned orientation step: {:.2f}deg".format(
        math.degrees(planned_step)
    ))
    print("Current hand position held at:", np.round(current_position, 4).tolist())
    print("Desired right-hand quaternion:",
          np.round(desired_quaternion, 6).tolist())

    solution = solve_pose_hard_ik(
        task,
        current_joints,
        current_poses,
        current_position.tolist(),
        desired_quaternion.tolist(),
        position_tolerance_m=args.position_ik_tolerance,
        orientation_tolerance_rad=args.orientation_ik_tolerance,
    )
    if len(solution) != 14:
        raise RuntimeError("senior IK did not return fourteen arm joints")
    joint_delta_degrees = np.degrees(
        np.asarray(solution, dtype=float) - np.asarray(current_joints, dtype=float)
    )
    left_delta_degrees = joint_delta_degrees[:7]
    right_delta_degrees = joint_delta_degrees[7:14]

    predicted_poses = task.call_fk(solution)
    predicted_position = np.asarray(
        predicted_poses.right_pose.pos_xyz, dtype=float
    )
    predicted_quaternion = np.asarray(
        predicted_poses.right_pose.quat_xyzw, dtype=float
    )
    fk_position_error = float(np.linalg.norm(
        predicted_position - current_position
    ))
    fk_orientation_error = quaternion_angle_degrees(
        desired_quaternion, predicted_quaternion
    )
    _, _, predicted_optical_angle = predict_camera_alignment(
        current_position,
        current_quaternion,
        before_camera_origin,
        before_camera_rotation,
        predicted_position,
        predicted_quaternion,
        target,
    )
    plan_ok, plan_checks, predicted_reduction = validate_ik_plan(
        left_delta_degrees,
        right_delta_degrees,
        fk_position_error,
        fk_orientation_error,
        before_angle,
        predicted_optical_angle,
        minimum_right_joint_delta_degrees=args.minimum_joint_delta,
        maximum_right_joint_delta_degrees=args.maximum_joint_delta,
        maximum_left_joint_delta_degrees=args.maximum_left_joint_delta,
        maximum_fk_position_error_m=args.maximum_fk_position_error,
        maximum_fk_orientation_error_degrees=args.maximum_fk_orientation_error,
        minimum_optical_reduction_degrees=args.minimum_angle_reduction,
    )
    print("IK constraint mode: 3 (position hard + orientation hard)")
    print("Planned left-arm joint delta:",
          np.round(left_delta_degrees, 2).tolist())
    print("Planned right-arm joint delta:",
          np.round(right_delta_degrees, 2).tolist())
    print("Predicted hand position:",
          np.round(predicted_position, 4).tolist())
    print("Predicted FK position error: {:.4f}m".format(fk_position_error))
    print("Predicted FK orientation error: {:.2f}deg".format(
        fk_orientation_error
    ))
    print("Predicted optical error: {:.2f}deg -> {:.2f}deg".format(
        math.degrees(before_angle), math.degrees(predicted_optical_angle)
    ))
    print("Predicted optical reduction: {:.2f}deg".format(
        predicted_reduction
    ))
    print("Dry-run safety checks:", plan_checks)
    if not plan_ok:
        raise RuntimeError(
            "WRIST_LOOK_AT_IK_BLOCKED: predicted solution failed safety gates"
        )
    print("WRIST_LOOK_AT_IK_OK: verified calculation; no command sent yet")

    if not args.execute:
        print("WRIST_LOOK_AT_DRY_RUN_OK: calculation only; claw remains open")
        return 0
    if args.confirmation != EXECUTION_CONFIRMATION:
        raise RuntimeError(
            "execution blocked; pass --confirmation {}".format(
                EXECUTION_CONFIRMATION
            )
        )

    task.stop_base()
    task.wait_for_arm_subscriber(timeout=8.0)
    if not task.set_arm_mode(2):
        raise RuntimeError("cannot enable senior arm external-control mode")
    if not task.open_claw():
        raise RuntimeError("cannot confirm open claw before look-at step")
    print("Executing one bounded look-at correction; hand position held")
    task.move_arm_degrees(
        rad_to_deg(solution), duration=args.motion_seconds
    )
    task.stop_base()
    rospy.sleep(args.settle_seconds)

    after_poses = task.call_fk(task.read_current_arm_joints())
    after_position = np.asarray(after_poses.right_pose.pos_xyz, dtype=float)
    hand_translation = float(np.linalg.norm(after_position - current_position))
    _, _, _, _, after_angle = camera_state(
        tf_buffer, rospy, args.camera_frame, target
    )
    ok, checks, before_degrees, after_degrees = validate_look_at(
        before_angle,
        after_angle,
        hand_translation,
        minimum_reduction_degrees=args.minimum_angle_reduction,
        maximum_translation_m=args.maximum_hand_translation,
    )
    print("Actual hand position:", np.round(after_position, 4).tolist())
    print("Hand translation: {:.4f}m".format(hand_translation))
    print("Optical error: {:.2f}deg -> {:.2f}deg".format(
        before_degrees, after_degrees
    ))
    print("Safety checks:", checks)
    if not ok:
        raise RuntimeError(
            "WRIST_LOOK_AT_STEP_BLOCKED: response failed safety checks; "
            "claw remains open"
        )
    print(
        "WRIST_LOOK_AT_STEP_OK: camera turned toward locked tray; "
        "claw remains open"
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--senior-dir", default=DEFAULT_SENIOR_DIR)
    parser.add_argument("--target-topic", default=DEFAULT_TARGET_TOPIC)
    parser.add_argument("--target-param", default=DEFAULT_TARGET_PARAM)
    parser.add_argument("--camera-frame", default="right_wrist_camera_link")
    parser.add_argument("--target-frames", type=int, default=3)
    parser.add_argument("--target-timeout", type=float, default=10.0)
    parser.add_argument("--maximum-target-spread", type=float, default=0.015)
    parser.add_argument("--maximum-angle-step", type=float, default=8.0)
    parser.add_argument("--minimum-joint-delta", type=float, default=0.10)
    parser.add_argument("--maximum-joint-delta", type=float, default=15.0)
    parser.add_argument("--maximum-left-joint-delta", type=float, default=2.0)
    parser.add_argument("--position-ik-tolerance", type=float, default=0.004)
    parser.add_argument("--orientation-ik-tolerance", type=float, default=0.02)
    parser.add_argument("--maximum-fk-position-error", type=float, default=0.008)
    parser.add_argument("--maximum-fk-orientation-error", type=float, default=2.0)
    parser.add_argument("--motion-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--minimum-angle-reduction", type=float, default=1.0)
    parser.add_argument("--maximum-hand-translation", type=float, default=0.035)
    return parser


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
