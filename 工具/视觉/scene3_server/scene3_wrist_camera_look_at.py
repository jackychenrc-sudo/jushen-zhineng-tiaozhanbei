#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One bounded wrist-camera look-at correction after senior pregrasp.

The senior Scene3 code remains responsible for target selection, walking and
arm IK.  This helper keeps the current right-hand Cartesian position and
rotates its orientation only enough to turn the right-camera +Z optical axis
towards the stable head-camera tray point.  One command is capped at eight
degrees.  The claw stays open and no base, close, lift or extraction command
is sent.

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
DEFAULT_TARGET_TOPIC = "/challenge_cup_task_template/scene3/grasp_point_base"


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
            return stable_target(
                samples[-args.target_frames:],
                maximum_spread_m=args.maximum_target_spread,
            )
    raise RuntimeError("no stable head-camera tray target")


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
    target, target_spread = wait_stable_target(
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

    print("Stable head target base_link:", np.round(target, 4).tolist())
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

    solution = task.solve_right_hand_ik(
        current_position.tolist(), desired_quaternion.tolist()
    )
    if len(solution) != 14:
        raise RuntimeError("senior IK did not return fourteen arm joints")
    delta_degrees = np.degrees(
        np.asarray(solution[7:14], dtype=float)
        - np.asarray(current_joints[7:14], dtype=float)
    )
    maximum_joint_delta = float(np.max(np.abs(delta_degrees)))
    print("Planned right-arm joint delta:",
          np.round(delta_degrees, 2).tolist())
    print("Maximum joint delta: {:.2f}deg".format(maximum_joint_delta))
    if maximum_joint_delta > args.maximum_joint_delta:
        raise RuntimeError("orientation IK exceeds joint-delta safety gate")
    print("WRIST_LOOK_AT_IK_OK: no command sent yet")

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
    parser.add_argument("--camera-frame", default="right_wrist_camera_link")
    parser.add_argument("--target-frames", type=int, default=3)
    parser.add_argument("--target-timeout", type=float, default=10.0)
    parser.add_argument("--maximum-target-spread", type=float, default=0.015)
    parser.add_argument("--maximum-angle-step", type=float, default=8.0)
    parser.add_argument("--maximum-joint-delta", type=float, default=22.0)
    parser.add_argument("--motion-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--minimum-angle-reduction", type=float, default=1.0)
    parser.add_argument("--maximum-hand-translation", type=float, default=0.035)
    return parser


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
