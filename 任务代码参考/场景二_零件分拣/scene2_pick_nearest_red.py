#!/usr/bin/env python3
"""Pick the nearest red-handled part in the current Scene2 layout."""

import argparse
import math
import os
import sys

import rospy
import tf

from kuavo_msgs.msg import armTargetPoses

import scene2_vision_locator as vision_locator


PIPELINE_DIR = (
    "/root/kuavo_ws/src/challenge_cup_simulator/"
    "test/collect_scene2_dataset"
)
sys.path.insert(0, PIPELINE_DIR)

import scene2_data_collection_pipeline as pipeline
import scene2_part_grasp_ik as grasp_ik
from scene2_part_grasp_ik import GraspRuntime


SAFE_GRASP_POSITION_TOLERANCE_M = 0.03
SAFE_TRANSIT_POSITION_TOLERANCE_M = 0.035
SAFE_OUTSIDE_POSITION_TOLERANCE_M = 0.04
MAX_IK_SEGMENT_DELTA_DEG = 27.0
MAX_TRANSIT_IK_SEGMENT_DELTA_DEG = 32.0
MAX_OUTSIDE_IK_SEGMENT_DELTA_DEG = 32.0
MAX_OUTSIDE_LIFT_IK_SEGMENT_DELTA_DEG = 32.0
MAX_LOCKED_ARM_DELTA_DEG = 5.0
MAX_TRANSFER_STEP_DEG = 20.0
JOINT_TRACKING_WARN_DEG = 8.0
MAX_ACTUAL_JOINT_ERROR_DEG = 25.0
MAX_TRANSIT_ACTUAL_JOINT_ERROR_DEG = 32.0
MAX_OUTSIDE_ACTUAL_JOINT_ERROR_DEG = 32.0
MAX_OUTSIDE_LIFT_ACTUAL_JOINT_ERROR_DEG = 32.0
SAFE_PATH_ORIENTATION_TOLERANCE_DEG = 30.0
SAFE_OUTSIDE_LIFT_ORIENTATION_TOLERANCE_DEG = 30.0
SAFE_HIGH_TRANSIT_ORIENTATION_TOLERANCE_DEG = 36.0
SAFE_TRANSIT_EE_Z_M = 0.12
BODY_SIDE_LIFT_Z_M = -0.17
MAX_RESTRICTED_LIFT_M = 0.40
RESTRICTED_SEGMENT_SETTLE_SECONDS = 1.0
RESTRICTED_MIN_SEGMENT_DURATION_S = 2.5
RESTRICTED_MIN_PROGRESS_RATIO = 0.20
RESTRICTED_HIGH_COMMAND_Z_M = SAFE_TRANSIT_EE_Z_M + 0.02
LOW_APPROACH_IK_SEEDS_RAD = (
    (0.33, 0.42, 0.05, -1.65),
    (0.40, 0.44, 0.08, -1.85),
    (0.25, 0.36, 0.00, -1.45),
)
LEFT_SIDE_CLEARANCE_Y_M = 0.44
OUTSIDE_TABLE_X_M = 0.20
MAX_CARTESIAN_STEP_M = 0.02
BODY_SIDE_STEP_M = 0.005
OUTSIDE_LIFT_STEP_M = 0.01
MIN_TRANSFER_EE_Z_M = 0.08
GRASP_ORIENTATION_BLEND = 0.0
# The two calibration samples are indexed by measured base-link y, not by a
# simulated object identity.  Linear interpolation therefore remains valid
# when a random seed swaps the two visually identical screwdrivers.
VISION_GRASP_OFFSET_CALIBRATION = (
    (0.10, (-0.017, -0.010, -0.075)),
    (0.37, (0.004, -0.004, -0.071)),
)
# Center of the inner gripping surface in each inner-finger frame.  The mesh
# contact face spans roughly local z=[0.0025, 0.0495] m; using its center
# measures where a part is actually pinched rather than the knuckle pivot.
PINCH_CONTACT_LOCAL_M = (0.0, -0.026, 0.029)


def set_humanoid_arm_mode(mode, timeout=20.0):
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

    service_name = "/humanoid_change_arm_ctrl_mode"
    rospy.wait_for_service(service_name, timeout=timeout)
    proxy = rospy.ServiceProxy(service_name, changeArmCtrlMode)
    request = changeArmCtrlModeRequest()
    request.control_mode = int(mode)
    response = proxy(request)
    if not response.result:
        raise RuntimeError(
            "%s rejected mode %s: %s"
            % (service_name, mode, response.message)
        )
    rospy.loginfo("humanoid arm mode -> %s: %s", mode, response.message)


def set_wbc_arm_trajectory_enabled(enabled, timeout=20.0):
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

    service_name = "/enable_wbc_arm_trajectory_control"
    rospy.wait_for_service(service_name, timeout=timeout)
    proxy = rospy.ServiceProxy(service_name, changeArmCtrlMode)
    request = changeArmCtrlModeRequest()
    request.control_mode = 1 if enabled else 0
    response = proxy(request)
    if not response.result:
        raise RuntimeError("%s rejected enabled=%s" % (service_name, enabled))
    rospy.loginfo("WBC ROS arm trajectory enabled -> %s", bool(enabled))


def control_scene2_claws(left_position, right_position=0.0, timeout=10.0):
    """Use Scene2's verified claw service rather than the generic command topic."""
    from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest

    service_name = "/control_robot_leju_claw"
    rospy.wait_for_service(service_name, timeout=float(timeout))
    request = controlLejuClawRequest()
    request.data.name = ["left_claw", "right_claw"]
    request.data.position = [float(left_position), float(right_position)]
    request.data.velocity = [50.0, 50.0]
    request.data.effort = [1.0, 1.0]
    response = rospy.ServiceProxy(service_name, controlLejuClaw)(request)
    if not response.success:
        raise RuntimeError("%s rejected claw command: %s" % (service_name, response.message))
    rospy.loginfo(
        "Scene2 claw service left=%.1f right=%.1f",
        left_position,
        right_position,
    )


def set_mm_wbc_arm_trajectory_enabled(enabled, timeout=20.0):
    """Enable the controller's full 14-joint mobile-manipulation path."""
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

    service_name = "/enable_mm_wbc_arm_trajectory_control"
    rospy.wait_for_service(service_name, timeout=timeout)
    proxy = rospy.ServiceProxy(service_name, changeArmCtrlMode)
    request = changeArmCtrlModeRequest()
    request.control_mode = 1 if enabled else 0
    response = proxy(request)
    if not response.result:
        raise RuntimeError("%s rejected enabled=%s" % (service_name, enabled))
    rospy.loginfo("MM WBC full-arm trajectory enabled -> %s", bool(enabled))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object",
        choices=("auto", "part_type_c_1", "part_type_c_2"),
        default="auto",
    )
    parser.add_argument("--move-time", type=float, default=5.0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--ik-check", action="store_true")
    parser.add_argument("--restricted-ik-check", action="store_true")
    parser.add_argument("--restricted-side-execute", action="store_true")
    parser.add_argument("--restricted-lift-step-execute", action="store_true")
    parser.add_argument("--restricted-high-transit-execute", action="store_true")
    parser.add_argument("--restricted-approach-execute", action="store_true")
    parser.add_argument("--restricted-pinch-align-execute", action="store_true")
    parser.add_argument("--restricted-pinch-grasp-test-execute", action="store_true")
    parser.add_argument("--fk-tf-check", action="store_true")
    parser.add_argument("--full-ik-probe", action="store_true")
    parser.add_argument("--full-ik-clearance", type=float, default=0.0)
    parser.add_argument(
        "--restricted-approach-clearance",
        type=float,
        default=0.13,
    )
    parser.add_argument("--restricted-lift-height", type=float, default=0.05)
    parser.add_argument("--pinch-clearance", type=float, default=0.04)
    parser.add_argument("--pinch-max-steps", type=int, default=18)
    parser.add_argument("--pinch-tolerance", type=float, default=0.012)
    parser.add_argument("--restricted-high-step", type=float, default=0.02)
    parser.add_argument("--restricted-approach-step", type=float, default=0.01)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--joint5-check", action="store_true")
    parser.add_argument("--joint5-delta", type=float, default=2.0)
    parser.add_argument("--joint-index", type=int, default=5)
    parser.add_argument(
        "--joint5-interface",
        choices=("target", "direct", "mm"),
        default="target",
    )
    parser.add_argument(
        "--stage",
        choices=("lift", "body", "high", "approach", "full"),
        default="high",
    )
    return parser.parse_args(rospy.myargv()[1:])


def publish_arm_target(pub, target_degrees, move_time):
    message = armTargetPoses()
    message.times = [float(move_time)]
    message.values = [float(value) for value in target_degrees]
    pub.publish(message)
    rospy.sleep(float(move_time) + 1.0)


def publish_mm_arm_target(pub, target_degrees, move_time):
    """Send one full-arm target in degrees through the MM WBC path."""
    from sensor_msgs.msg import JointState

    if len(target_degrees) != 14:
        raise ValueError("MM arm target must contain exactly 14 joints")
    message = JointState()
    message.header.stamp = rospy.Time.now()
    message.name = ["arm_joint_%d" % index for index in range(1, 15)]
    message.position = [float(value) for value in target_degrees]
    pub.publish(message)
    rospy.sleep(float(move_time) + 1.0)


def log_left_gripper_base_pose(label, timeout=1.0):
    """Log the physical gripper reference point used for final height calibration."""
    listener = tf.TransformListener()
    try:
        listener.waitForTransform(
            "base_link",
            "left_gripper_base",
            rospy.Time(0),
            rospy.Duration(float(timeout)),
        )
        translation, rotation = listener.lookupTransform(
            "base_link",
            "left_gripper_base",
            rospy.Time(0),
        )
        rospy.loginfo(
            "%s left_gripper_base xyz=%s quat=%s",
            label,
            [round(float(value), 4) for value in translation],
            [round(float(value), 4) for value in rotation],
        )
    except Exception as error:
        rospy.logwarn("%s gripper TF unavailable: %s", label, error)


def read_left_end_effector_pose(timeout=1.0):
    """Return the physical L7 end-effector pose published by the simulator."""
    listener = tf.TransformListener()
    frame = "zarm_l7_end_effector"
    listener.waitForTransform(
        "base_link",
        frame,
        rospy.Time(0),
        rospy.Duration(float(timeout)),
    )
    translation, rotation = listener.lookupTransform(
        "base_link",
        frame,
        rospy.Time(0),
    )
    return (
        [float(value) for value in translation],
        [float(value) for value in rotation],
    )


def _transform_local_point(translation, rotation, local_point):
    matrix = tf.transformations.quaternion_matrix(rotation)
    return [
        float(translation[axis])
        + sum(
            float(matrix[axis][local_axis]) * float(local_point[local_axis])
            for local_axis in range(3)
        )
        for axis in range(3)
    ]


def read_left_pinch_position(timeout=1.0):
    """Return the midpoint of the two physical inner-finger contact faces."""
    listener = tf.TransformListener()
    frames = (
        "left_gripper_left_inner_finger",
        "left_gripper_right_inner_finger",
    )
    points = []
    for frame in frames:
        listener.waitForTransform(
            "base_link",
            frame,
            rospy.Time(0),
            rospy.Duration(float(timeout)),
        )
        translation, rotation = listener.lookupTransform(
            "base_link",
            frame,
            rospy.Time(0),
        )
        points.append(
            _transform_local_point(
                translation,
                rotation,
                PINCH_CONTACT_LOCAL_M,
            )
        )
    return [
        0.5 * (points[0][axis] + points[1][axis])
        for axis in range(3)
    ]


def log_joint_diagnostic_end_effector(label):
    """Record the measured tool positions during a harmless joint check."""
    try:
        end_position, _rotation = read_left_end_effector_pose(timeout=2.0)
        pinch_position = read_left_pinch_position(timeout=2.0)
        rospy.loginfo(
            "joint diagnostic %s end=%s pinch=%s",
            label,
            [round(value, 4) for value in end_position],
            [round(value, 4) for value in pinch_position],
        )
    except Exception as error:
        rospy.logwarn(
            "joint diagnostic %s TF unavailable: %s",
            label,
            error,
        )


def run_fk_tf_check(timeout=10.0):
    """Read the FK hand pose and matching physical frames without motion."""
    current = list(pipeline._read_current_arm_joints(float(timeout)))
    fk_pose = grasp_ik._call_fk(current, float(timeout)).left_pose
    rospy.loginfo(
        "fk_tf_check FK left xyz=%s quat=%s",
        [round(float(value), 5) for value in fk_pose.pos_xyz],
        [round(float(value), 5) for value in fk_pose.quat_xyzw],
    )
    listener = tf.TransformListener()
    translations = {}
    rotations = {}
    for frame in (
        "zarm_l7_link",
        "zarm_l7_end_effector",
        "left_gripper_base",
        "left_gripper_left_inner_knuckle",
        "left_gripper_right_inner_knuckle",
        "left_gripper_left_inner_finger",
        "left_gripper_right_inner_finger",
    ):
        try:
            listener.waitForTransform(
                "base_link",
                frame,
                rospy.Time(0),
                rospy.Duration(float(timeout)),
            )
            translation, rotation = listener.lookupTransform(
                "base_link",
                frame,
                rospy.Time(0),
            )
            translations[frame] = [float(value) for value in translation]
            rotations[frame] = [float(value) for value in rotation]
            rospy.loginfo(
                "fk_tf_check TF %s xyz=%s quat=%s",
                frame,
                [round(float(value), 5) for value in translation],
                [round(float(value), 5) for value in rotation],
            )
        except Exception as error:
            rospy.logwarn("fk_tf_check TF %s unavailable: %s", frame, error)

    for label, left_frame, right_frame in (
        (
            "inner_knuckle_midpoint",
            "left_gripper_left_inner_knuckle",
            "left_gripper_right_inner_knuckle",
        ),
        (
            "inner_finger_midpoint",
            "left_gripper_left_inner_finger",
            "left_gripper_right_inner_finger",
        ),
    ):
        if left_frame not in translations or right_frame not in translations:
            continue
        midpoint = [
            0.5 * (translations[left_frame][axis] + translations[right_frame][axis])
            for axis in range(3)
        ]
        rospy.loginfo(
            "fk_tf_check %s xyz=%s",
            label,
            [round(value, 5) for value in midpoint],
        )

    finger_frames = (
        "left_gripper_left_inner_finger",
        "left_gripper_right_inner_finger",
    )
    if all(frame in translations and frame in rotations for frame in finger_frames):
        contact_points = [
            _transform_local_point(
                translations[frame],
                rotations[frame],
                PINCH_CONTACT_LOCAL_M,
            )
            for frame in finger_frames
        ]
        contact_midpoint = [
            0.5 * (contact_points[0][axis] + contact_points[1][axis])
            for axis in range(3)
        ]
        rospy.loginfo(
            "fk_tf_check inner_contact_midpoint xyz=%s",
            [round(value, 5) for value in contact_midpoint],
        )


def run_full_ik_probe(job, clearance=0.0):
    from kuavo_msgs.srv import (
        twoArmHandPoseCmdSrv,
        twoArmHandPoseCmdSrvRequest,
    )

    timeout = 20.0
    current = list(pipeline._read_current_arm_joints(timeout))
    poses = grasp_ik._call_fk(current, timeout)
    request = twoArmHandPoseCmdSrvRequest()
    command = request.twoArmHandPoseCmdRequest
    hand_poses = command.hand_poses
    hand_poses.header.frame_id = "base_link"
    target_xyz = list(job["grasp"])
    target_xyz[2] += float(clearance)
    hand_poses.left_pose.pos_xyz = target_xyz
    hand_poses.left_pose.quat_xyzw = list(poses.left_pose.quat_xyzw)
    hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    hand_poses.left_pose.joint_angles = list(current[:7])
    hand_poses.right_pose.pos_xyz = list(poses.right_pose.pos_xyz)
    hand_poses.right_pose.quat_xyzw = list(poses.right_pose.quat_xyzw)
    hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    hand_poses.right_pose.joint_angles = list(current[7:14])
    command.use_custom_ik_param = False
    command.joint_angles_as_q0 = True
    command.frame = 0

    service_name = "/ik/two_arm_hand_pose_cmd_srv"
    rospy.wait_for_service(service_name, timeout=timeout)
    response = rospy.ServiceProxy(service_name, twoArmHandPoseCmdSrv)(request)
    if not response.success:
        raise RuntimeError("full IK probe failed: %s" % response.error_reason)
    solution = list(response.q_arm)
    if len(solution) != 14:
        raise RuntimeError("full IK returned %d joints, expected 14" % len(solution))
    deltas = [
        math.degrees(solution[index] - current[index])
        for index in range(14)
    ]
    rospy.loginfo(
        "full IK probe clearance=%.3fm q_arm(deg)=%s",
        float(clearance),
        [round(math.degrees(value), 2) for value in solution],
    )
    rospy.loginfo(
        "full IK probe delta(deg)=%s time_cost=%.4fs",
        [round(value, 2) for value in deltas],
        response.time_cost,
    )


def run_joint5_check(move_time, delta_degrees, interface, joint_index):
    if abs(float(delta_degrees)) > 5.0:
        raise ValueError("joint diagnostic delta must not exceed 5 degrees")
    if int(joint_index) < 1 or int(joint_index) > 14:
        raise ValueError("joint diagnostic index must be within 1..14")
    target_index = int(joint_index) - 1

    arm_hold = None
    arm_mode_changed = False
    humanoid_mode_changed = False
    wbc_trajectory_enabled = False
    mm_wbc_trajectory_enabled = False
    try:
        start_radians = pipeline._read_current_arm_joints(20.0)
        start_degrees = [math.degrees(value) for value in start_radians]
        target_degrees = list(start_degrees)
        target_degrees[target_index] += float(delta_degrees)
        rospy.loginfo(
            "joint diagnostic start(deg): %s",
            [round(value, 3) for value in start_degrees],
        )
        rospy.loginfo(
            "joint diagnostic target(deg): %s",
            [round(value, 3) for value in target_degrees],
        )
        log_joint_diagnostic_end_effector("start")

        if interface == "direct":
            arm_hold = pipeline._start_arm_traj_hold(20.0)
        elif interface == "mm":
            from sensor_msgs.msg import JointState

            arm_target_pub = rospy.Publisher(
                "/mm_kuavo_arm_traj",
                JointState,
                queue_size=1,
            )
            pipeline._wait_for_connection(arm_target_pub, 20.0)
        else:
            arm_target_pub = rospy.Publisher(
                pipeline.ARM_TARGET_POSES_TOPIC,
                armTargetPoses,
                queue_size=1,
            )
            pipeline._wait_for_connection(arm_target_pub, 20.0)
        pipeline._set_arm_mode(pipeline.ARM_MODE_EXTERNAL_CONTROL, timeout=20.0)
        arm_mode_changed = True
        set_humanoid_arm_mode(pipeline.ARM_MODE_EXTERNAL_CONTROL, timeout=20.0)
        humanoid_mode_changed = True
        if interface == "direct":
            set_wbc_arm_trajectory_enabled(True, timeout=20.0)
            wbc_trajectory_enabled = True
        elif interface == "mm":
            set_mm_wbc_arm_trajectory_enabled(True, timeout=20.0)
            mm_wbc_trajectory_enabled = True
        if interface == "direct":
            pipeline._execute_arm_motion(
                None,
                arm_hold,
                start_degrees,
                target_degrees,
                float(move_time),
                1.0,
            )
        elif interface == "mm":
            publish_mm_arm_target(arm_target_pub, target_degrees, move_time)
        else:
            publish_arm_target(arm_target_pub, target_degrees, move_time)

        moved_radians = pipeline._read_current_arm_joints(20.0)
        moved_degrees = [math.degrees(value) for value in moved_radians]
        from kuavo_msgs.msg import jointCmd
        from sensor_msgs.msg import JointState

        if interface == "direct":
            published = rospy.wait_for_message(
                pipeline.ARM_TRAJ_TOPIC,
                JointState,
                timeout=5.0,
            )
            rospy.loginfo(
                "joint diagnostic /kuavo_arm_traj(deg): %s",
                [round(float(value), 3) for value in published.position[:14]],
            )
        try:
            final_command = rospy.wait_for_message(
                "/joint_cmd",
                jointCmd,
                timeout=2.0,
            )
            rospy.loginfo(
                "joint diagnostic /joint_cmd arm(deg): %s",
                [
                    round(math.degrees(float(value)), 3)
                    for value in final_command.joint_q[13:27]
                ],
            )
            rospy.loginfo(
                "joint diagnostic /joint_cmd arm kp: %s",
                [round(float(value), 3) for value in final_command.joint_kp[13:27]],
            )
        except rospy.ROSException:
            rospy.logwarn("joint diagnostic: /joint_cmd unavailable")
        observed_delta = moved_degrees[target_index] - start_degrees[target_index]
        all_observed_deltas = [
            moved_degrees[index] - start_degrees[index]
            for index in range(14)
        ]
        rospy.loginfo(
            "joint%d check outward: start=%.3fdeg command=%.3fdeg actual=%.3fdeg observed_delta=%.3fdeg",
            int(joint_index),
            start_degrees[target_index],
            target_degrees[target_index],
            moved_degrees[target_index],
            observed_delta,
        )
        rospy.loginfo(
            "joint5 check all joint deltas(deg): %s",
            [round(value, 3) for value in all_observed_deltas],
        )
        log_joint_diagnostic_end_effector("outward")

        if interface == "direct":
            pipeline._execute_arm_motion(
                None,
                arm_hold,
                moved_degrees,
                start_degrees,
                float(move_time),
                1.0,
            )
        elif interface == "mm":
            publish_mm_arm_target(arm_target_pub, start_degrees, move_time)
        else:
            publish_arm_target(arm_target_pub, start_degrees, move_time)
        restored_radians = pipeline._read_current_arm_joints(20.0)
        restored_degrees = [math.degrees(value) for value in restored_radians]
        rospy.loginfo(
            "joint%d check restore: target=%.3fdeg actual=%.3fdeg residual=%.3fdeg",
            int(joint_index),
            start_degrees[target_index],
            restored_degrees[target_index],
            restored_degrees[target_index] - start_degrees[target_index],
        )
        log_joint_diagnostic_end_effector("restored")
    finally:
        if mm_wbc_trajectory_enabled:
            set_mm_wbc_arm_trajectory_enabled(False, timeout=10.0)
        if wbc_trajectory_enabled:
            set_wbc_arm_trajectory_enabled(False, timeout=10.0)
        if arm_hold is not None:
            arm_hold.stop()
        if humanoid_mode_changed:
            set_humanoid_arm_mode(pipeline.ARM_MODE_AUTO_SWING, timeout=10.0)
        if arm_mode_changed:
            pipeline._set_arm_mode(pipeline.ARM_MODE_AUTO_SWING, timeout=10.0)


def _vision_grasp_offset(point_base):
    y_value = float(point_base[1])
    low_y, low_offset = VISION_GRASP_OFFSET_CALIBRATION[0]
    high_y, high_offset = VISION_GRASP_OFFSET_CALIBRATION[1]
    ratio = (y_value - low_y) / (high_y - low_y)
    ratio = min(1.0, max(0.0, ratio))
    return tuple(
        low_offset[index]
        + ratio * (high_offset[index] - low_offset[index])
        for index in range(3)
    )


def _locate_red_candidates():
    vision_args = argparse.Namespace(
        color="red",
        output_dir="/tmp/scene2_vision_test",
        min_area=300.0,
        max_area=4000.0,
        depth_radius=12,
        candidate_index=0,
        sync_slop=0.10,
        roi=None,
        continuous=True,
        sequential=True,
    )
    return vision_locator.Scene2VisionDebug(vision_args).run_sequential()


def build_vision_red_job(requested_object):
    result = _locate_red_candidates()
    candidates = []
    for item in result["candidates"]:
        if not item.get("valid_3d"):
            rospy.logwarn(
                "red candidate #%s rejected: %s",
                item.get("index"),
                item.get("rejection_reason", "no valid 3-D point"),
            )
            continue
        aspect_ratio = float(item["aspect_ratio"])
        if not 2.0 <= aspect_ratio <= 8.0:
            rospy.logwarn(
                "red candidate #%s rejected by shape ratio %.2f",
                item["index"],
                aspect_ratio,
            )
            continue
        if float(item["depth_mad_m"]) > 0.020:
            rospy.logwarn(
                "red candidate #%s rejected by depth MAD %.4fm",
                item["index"],
                float(item["depth_mad_m"]),
            )
            continue

        point_base = [float(value) for value in item["base_xyz_m"]]
        if not (
            0.15 <= point_base[0] <= 0.55
            and 0.02 <= point_base[1] <= 0.45
            and -0.12 <= point_base[2] <= 0.03
        ):
            rospy.logwarn(
                "red candidate #%s rejected outside work area: %s",
                item["index"],
                [round(value, 4) for value in point_base],
            )
            continue

        offset = _vision_grasp_offset(point_base)
        grasp_ee = [
            point_base[index] + offset[index]
            for index in range(3)
        ]
        # "Nearest" means nearest to the robot base, not nearest to a
        # hand-picked work point on the left side of the table.
        distance = math.sqrt(sum(value * value for value in point_base))
        candidates.append((distance, point_base, item, grasp_ee, offset))

    if not candidates:
        raise RuntimeError("no red screwdriver candidate passed 3-D safety filters")

    if requested_object != "auto":
        rospy.logwarn(
            "%s cannot be distinguished visually from the other red "
            "screwdriver; selecting the nearest valid candidate instead",
            requested_object,
        )
    _distance, point_base, selected_result, grasp_ee, offset = min(candidates)

    # Both red instances are the same task class and share the purple bin.
    # This name is only an action-template key; it is not inferred identity.
    selected_name = "part_type_c_1"
    grasp_world = [
        grasp_ee[0] - pipeline.WORLD_TO_EE_OFFSET_X,
        grasp_ee[1] - pipeline.WORLD_TO_EE_OFFSET_Y_LEFT,
        grasp_ee[2] - pipeline.WORLD_TO_EE_OFFSET_Z,
    ]

    jobs = pipeline._build_sorting_jobs(first_pick=selected_name)
    job = jobs[0]
    job["world_xyz"] = list(grasp_world)
    job["vision_base"] = list(point_base)
    job["grasp"] = list(grasp_ee)
    job["vision_candidate_index"] = int(selected_result["index"])
    job["vision_image_angle_deg"] = float(
        selected_result["image_angle_deg"]
    )
    job["vision_rgb_chroma_distance"] = selected_result.get(
        "rgb_chroma_distance"
    )
    job["grasp_quat"] = grasp_ik.get_object_grasp_quat_xyzw(
        selected_name,
        active_arm="left",
    )
    job["lift_quat"] = list(job["grasp_quat"])
    rospy.loginfo(
        "scene2 OpenCV red candidate #%d pixel=%s base=%s grasp=%s "
        "offset=%s image_angle=%.1fdeg rgb_distance=%s",
        int(selected_result["index"]),
        selected_result["pixel_uv"],
        [round(value, 4) for value in point_base],
        [round(value, 4) for value in grasp_ee],
        [round(value, 4) for value in offset],
        float(selected_result["image_angle_deg"]),
        selected_result.get("rgb_chroma_distance"),
    )
    return job


def make_runtime(arm_pub, arm_hold, gripper_hold, move_time, job):
    def safe_logwarn(message, *args):
        try:
            rospy.logwarn(message, *args)
        except TypeError:
            rospy.logwarn("%s args=%s", message, args)

    runtime_holder = {}

    def guarded_arm_motion(start, target, duration, settle):
        if len(start) != 14 or len(target) != 14:
            raise RuntimeError("arm trajectory must contain 14 joints")
        deltas = [
            abs(float(target[index]) - float(start[index]))
            for index in range(14)
        ]
        max_delta = max(deltas)
        ik_segment_limit = getattr(
            runtime_holder.get("runtime"),
            "ik_segment_limit_deg",
            MAX_IK_SEGMENT_DELTA_DEG,
        )
        if max_delta > ik_segment_limit:
            joint_index = deltas.index(max_delta)
            raise RuntimeError(
                "IK motion blocked: joint %d changes %.1fdeg, limit %.1fdeg"
                % (joint_index + 1, max_delta, ik_segment_limit)
            )
        locked_slice = slice(7, 14) if job["arm"] == "left" else slice(0, 7)
        locked_deltas = deltas[locked_slice]
        max_locked_delta = max(locked_deltas)
        if max_locked_delta > MAX_LOCKED_ARM_DELTA_DEG:
            raise RuntimeError(
                "IK motion blocked: locked arm changes %.1fdeg, limit %.1fdeg"
                % (max_locked_delta, MAX_LOCKED_ARM_DELTA_DEG)
            )
        pipeline._execute_arm_motion(
            arm_pub,
            arm_hold,
            start,
            target,
            max(move_time, duration),
            settle,
        )
        actual_radians = pipeline._read_current_arm_joints(20.0)
        actual_degrees = [math.degrees(value) for value in actual_radians]
        tracking_errors = [
            abs(actual_degrees[index] - float(target[index]))
            for index in range(14)
        ]
        max_tracking_error = max(tracking_errors)
        max_tracking_joint = tracking_errors.index(max_tracking_error) + 1
        rospy.loginfo(
            "arm endpoint tracking: max_joint_error=%.1fdeg joint=%d",
            max_tracking_error,
            max_tracking_joint,
        )
        if max_tracking_error > JOINT_TRACKING_WARN_DEG:
            rospy.logwarn(
                "arm joint tracking warning: joint %d error %.1fdeg",
                max_tracking_joint,
                max_tracking_error,
            )
        actual_joint_error_limit = getattr(
            runtime_holder.get("runtime"),
            "actual_joint_error_limit_deg",
            MAX_ACTUAL_JOINT_ERROR_DEG,
        )
        if max_tracking_error > actual_joint_error_limit:
            raise RuntimeError(
                "arm tracking blocked: joint %d error %.1fdeg, limit %.1fdeg"
                % (
                    max_tracking_joint,
                    max_tracking_error,
                    actual_joint_error_limit,
                )
            )

    def guarded_gripper_close(arm):
        runtime = runtime_holder["runtime"]
        actual_pos, actual_quat = grasp_ik._measure_hand_pose(runtime, arm)
        target_pos = job["grasp"]
        target_quat = job["grasp_quat"]
        position_error = math.sqrt(
            sum(
                (float(actual_pos[index]) - float(target_pos[index])) ** 2
                for index in range(3)
            )
        )
        quat_dot = abs(
            sum(
                float(actual_quat[index]) * float(target_quat[index])
                for index in range(4)
            )
        )
        orientation_error = 2.0 * math.acos(min(1.0, max(-1.0, quat_dot)))
        if (
            position_error > SAFE_GRASP_POSITION_TOLERANCE_M
            or orientation_error > runtime.orientation_tolerance_rad
        ):
            raise RuntimeError(
                "gripper close blocked: xyz_err=%.4fm/%.4fm, "
                "quat_err=%.1fdeg/%.1fdeg"
                % (
                    position_error,
                    SAFE_GRASP_POSITION_TOLERANCE_M,
                    math.degrees(orientation_error),
                    math.degrees(runtime.orientation_tolerance_rad),
                )
            )
        pipeline._publish_arm_gripper_close(gripper_hold, arm)

    runtime = GraspRuntime(
        world_to_ee_offset_x=pipeline.WORLD_TO_EE_OFFSET_X,
        world_to_ee_offset_y_left=pipeline.WORLD_TO_EE_OFFSET_Y_LEFT,
        world_to_ee_offset_y_right=pipeline.WORLD_TO_EE_OFFSET_Y_RIGHT,
        world_to_ee_offset_z=pipeline.WORLD_TO_EE_OFFSET_Z,
        pre_grasp_z_offset=pipeline.PRE_GRASP_APPROACH_Z_OFFSET,
        grasp_position_tolerance=SAFE_GRASP_POSITION_TOLERANCE_M,
        orientation_tolerance_rad=math.radians(
            SAFE_PATH_ORIENTATION_TOLERANCE_DEG
        ),
        gripper_close_time=1.0,
        timeout=20.0,
        move_time=move_time,
        settle_time=0.8,
        ik_mode_pos_hard_ori_hard=pipeline.IK_MODE_THREE_POINT_MIXED,
        read_current_arm_joints_cb=lambda: pipeline._read_current_arm_joints(20.0),
        execute_arm_motion_cb=guarded_arm_motion,
        publish_arm_gripper_close_cb=guarded_gripper_close,
        sleep_cb=rospy.sleep,
        loginfo_cb=rospy.loginfo,
        logwarn_cb=safe_logwarn,
    )
    runtime_holder["runtime"] = runtime
    return runtime


def run_ik_check(job, move_time):
    runtime = make_runtime(None, None, None, move_time, job)
    active_arm = job["arm"]
    current = list(pipeline._read_current_arm_joints(runtime.timeout))
    wrist_reference = list(current)
    weak_joint_indices = (
        (4, 5, 6) if active_arm == "left" else (11, 12, 13)
    )
    locked_other = (
        list(current[7:14]) if active_arm == "left" else list(current[0:7])
    )
    targets = build_safe_waypoints(job, runtime, active_arm, current)
    all_safe = True
    solutions = []
    for label, target, target_quat in targets:
        raw_solution = grasp_ik._call_single_arm_ik(
            runtime=runtime,
            current_joint_values=current,
            active_arm=active_arm,
            active_pos=target,
            active_quat=target_quat,
            locked_other_arm_joints=locked_other,
            constraint_mode=pipeline.IK_MODE_POS_HARD_ORI_SOFT,
            pos_cost_weight=1.0,
        )
        raw_wrist_delta = max(
            abs(
                math.degrees(raw_solution[index] - wrist_reference[index])
            )
            for index in weak_joint_indices
        )
        solution = list(raw_solution)
        for index in weak_joint_indices:
            solution[index] = wrist_reference[index]
        start_deg = [math.degrees(value) for value in current]
        target_deg = [math.degrees(value) for value in solution]
        deltas = [
            abs(target_deg[index] - start_deg[index])
            for index in range(14)
        ]
        locked_slice = slice(7, 14) if active_arm == "left" else slice(0, 7)
        max_delta = max(deltas)
        max_locked_delta = max(deltas[locked_slice])
        poses = grasp_ik._call_fk(solution, runtime.timeout)
        pose = poses.left_pose if active_arm == "left" else poses.right_pose
        predicted_pos = list(pose.pos_xyz)
        predicted_quat = list(pose.quat_xyzw)
        position_error = math.sqrt(
            sum(
                (predicted_pos[index] - target[index]) ** 2
                for index in range(3)
            )
        )
        orientation_error = grasp_ik._quat_angle_error(
            predicted_quat,
            target_quat,
        )
        orientation_tolerance = (
            math.radians(SAFE_OUTSIDE_LIFT_ORIENTATION_TOLERANCE_DEG)
            if label.startswith("outside_table_lift_")
            else math.radians(SAFE_HIGH_TRANSIT_ORIENTATION_TOLERANCE_DEG)
            if label.startswith("high_transit_")
            else runtime.orientation_tolerance_rad
        )
        stage_safe = (
            max_delta <= MAX_IK_SEGMENT_DELTA_DEG
            and max_locked_delta <= MAX_LOCKED_ARM_DELTA_DEG
            and position_error <= SAFE_GRASP_POSITION_TOLERANCE_M
            and orientation_error <= orientation_tolerance
        )
        print(
            "IK_CHECK %s safe=%s max_delta=%.1fdeg locked_delta=%.1fdeg "
            "raw_wrist_delta=%.1fdeg "
            "predicted_xyz_err=%.4fm quat_err=%.1fdeg"
            % (
                label,
                stage_safe,
                max_delta,
                max_locked_delta,
                raw_wrist_delta,
                position_error,
                math.degrees(orientation_error),
            )
        )
        all_safe = all_safe and stage_safe
        current = list(solution)
        solutions.append(list(solution))

    def check_joint_transfer(label, start_radians, target_degrees):
        start_degrees = [math.degrees(value) for value in start_radians]
        total_deltas = [
            float(target_degrees[index]) - start_degrees[index]
            for index in range(14)
        ]
        step_count = max(
            1,
            int(
                math.ceil(
                    max(abs(value) for value in total_deltas)
                    / MAX_TRANSFER_STEP_DEG
                )
            ),
        )
        minimum_z = float("inf")
        previous = list(start_degrees)
        transfer_safe = True
        for step_index in range(1, step_count + 1):
            ratio = float(step_index) / float(step_count)
            waypoint = [
                start_degrees[index] + total_deltas[index] * ratio
                for index in range(14)
            ]
            step_deltas = [
                abs(waypoint[index] - previous[index])
                for index in range(14)
            ]
            locked_slice = (
                slice(7, 14) if active_arm == "left" else slice(0, 7)
            )
            waypoint_fk = grasp_ik._call_fk(
                [math.radians(value) for value in waypoint],
                runtime.timeout,
            )
            waypoint_pose = (
                waypoint_fk.left_pose
                if active_arm == "left"
                else waypoint_fk.right_pose
            )
            minimum_z = min(minimum_z, float(waypoint_pose.pos_xyz[2]))
            transfer_safe = transfer_safe and (
                max(step_deltas) <= MAX_TRANSFER_STEP_DEG + 1.0e-6
                and max(step_deltas[locked_slice])
                <= MAX_LOCKED_ARM_DELTA_DEG
            )
            previous = waypoint
        transfer_safe = transfer_safe and minimum_z >= MIN_TRANSFER_EE_Z_M
        print(
            "IK_CHECK %s safe=%s steps=%d max_total_delta=%.1fdeg min_ee_z=%.3fm"
            % (
                label,
                transfer_safe,
                step_count,
                max(abs(value) for value in total_deltas),
                minimum_z,
            )
        )
        return transfer_safe

    # The arm returns along the already checked Cartesian waypoints after grasp.
    high_index = next(
        index for index, target in enumerate(targets)
        if target[0] == "move_high"
    )
    high_solution = solutions[high_index]
    active_place_joints = pipeline._place_active_arm_joints(
        active_arm,
        job["bin"],
    )
    place_target_degrees = pipeline._compose_single_arm_place_joints(
        active_arm,
        active_place_joints,
        locked_other,
    )
    transfer_to_bin_safe = check_joint_transfer(
        "transfer_to_bin",
        high_solution,
        place_target_degrees,
    )
    all_safe = all_safe and transfer_to_bin_safe
    print("IK_CHECK_RESULT", "PASS" if all_safe else "BLOCK")
    return all_safe


def solve_left_position_ik(
    runtime,
    start_joints,
    target_xyz,
    alternate_active_seeds=None,
):
    import numpy as np
    from scipy.optimize import least_squares

    fixed = np.asarray(start_joints, dtype=float)
    reference = fixed[:4].copy()
    joint_lower = np.asarray([-math.pi, -0.349066, -1.48353, -2.61799])
    joint_upper = np.asarray([1.5708, 3.49066, 1.48353, 0.0])
    local_span = math.radians(20.0)
    lower = np.maximum(joint_lower, reference - local_span)
    upper = np.minimum(joint_upper, reference + local_span)
    target = np.asarray(target_xyz, dtype=float)

    def residual(active):
        candidate = fixed.copy()
        candidate[:4] = active
        pose = grasp_ik._call_fk(candidate.tolist(), runtime.timeout).left_pose
        position_error = np.asarray(pose.pos_xyz, dtype=float) - target
        posture_cost = active - reference
        return np.concatenate((20.0 * position_error, 0.02 * posture_cost))

    initial_guesses = [np.clip(reference, lower, upper)]
    for seed in alternate_active_seeds or ():
        initial_guesses.append(np.clip(np.asarray(seed), lower, upper))

    candidates = []
    for initial in initial_guesses:
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=120,
            diff_step=1.0e-3,
            xtol=1e-7,
            ftol=1e-7,
            gtol=1e-7,
        )
        solution = fixed.copy()
        solution[:4] = result.x
        pose = grasp_ik._call_fk(solution.tolist(), runtime.timeout).left_pose
        error = math.sqrt(
            sum(
                (float(pose.pos_xyz[index]) - float(target_xyz[index])) ** 2
                for index in range(3)
            )
        )
        max_delta_deg = max(
            abs(math.degrees(solution[index] - reference[index]))
            for index in range(4)
        )
        total_delta_deg = sum(
            abs(math.degrees(solution[index] - reference[index]))
            for index in range(4)
        )
        converged = bool(result.success) or error <= 0.003
        candidates.append(
            (solution, error, converged, max_delta_deg, total_delta_deg)
        )

    if not candidates:
        raise RuntimeError("restricted IK produced no candidate")

    # During a near-table descent, several IK branches can reach the same
    # Cartesian point.  Keep the branch closest to the measured pose whenever
    # it is accurate enough; a lower residual is not worth a large joint jump.
    continuous_candidates = [
        item for item in candidates
        if item[2] and item[1] <= 0.015
    ]
    if continuous_candidates:
        best_solution, best_error, best_success, _max_delta, _total_delta = min(
            continuous_candidates,
            key=lambda item: (item[3], item[4], item[1]),
        )
    else:
        best_solution, best_error, best_success, _max_delta, _total_delta = min(
            candidates,
            key=lambda item: item[1],
        )
    return best_solution.tolist(), best_error, best_success


def run_restricted_ik_check(
    job,
    move_time,
    high_step=0.01,
    approach_step=0.01,
):
    runtime = make_runtime(None, None, None, move_time, job)
    if job["arm"] != "left":
        raise RuntimeError("restricted IK check currently supports left arm only")

    current = list(pipeline._read_current_arm_joints(runtime.timeout))
    start_pose = grasp_ik._call_fk(current, runtime.timeout).left_pose
    start = list(start_pose.pos_xyz)
    restricted_side_y = 0.34
    side = [start[0], restricted_side_y, start[2]]
    high = [
        OUTSIDE_TABLE_X_M,
        restricted_side_y,
        RESTRICTED_HIGH_COMMAND_Z_M,
    ]
    over_part = [
        job["grasp"][0],
        job["grasp"][1],
        RESTRICTED_HIGH_COMMAND_Z_M,
    ]
    targets = [
        ("side_clear", side),
    ]
    lift_positions = _linear_positions(
        side,
        [side[0], side[1], RESTRICTED_HIGH_COMMAND_Z_M],
        high_step,
    )
    for index, lift_position in enumerate(lift_positions, start=1):
        targets.append(
            ("side_lift_%02d" % index, lift_position)
        )
    side_high = [side[0], side[1], RESTRICTED_HIGH_COMMAND_Z_M]
    for index, position in enumerate(
        _linear_positions(side_high, high, high_step),
        start=1,
    ):
        targets.append(("high_forward_%02d" % index, position))
    for index, position in enumerate(
        _linear_positions(high, over_part, high_step),
        start=1,
    ):
        targets.append(("over_part_%02d" % index, position))

    approach_10cm = [
        job["grasp"][0],
        job["grasp"][1],
        job["grasp"][2] + 0.10,
    ]
    approach_5cm = [
        job["grasp"][0],
        job["grasp"][1],
        job["grasp"][2] + 0.05,
    ]
    for index, position in enumerate(
        _linear_positions(over_part, approach_10cm, approach_step),
        start=1,
    ):
        targets.append(("clear_descent_%02d" % index, position))
    for index, position in enumerate(
        _linear_positions(approach_10cm, approach_5cm, approach_step),
        start=1,
    ):
        targets.append(("fine_descent_%02d" % index, position))
    for index, position in enumerate(
        _linear_positions(approach_5cm, list(job["grasp"]), approach_step),
        start=1,
    ):
        targets.append(("grasp_descent_%02d" % index, position))

    all_safe = True
    for label, target in targets:
        previous = list(current)
        solution, error, solver_success = solve_left_position_ik(
            runtime,
            previous,
            target,
        )
        deltas = [
            abs(math.degrees(solution[index] - previous[index]))
            for index in range(14)
        ]
        max_delta = max(deltas[:4])
        safe = solver_success and error <= 0.025 and max_delta <= 30.0
        print(
            "RESTRICTED_IK %s safe=%s xyz_err=%.4fm max_delta=%.1fdeg joints=%s"
            % (
                label,
                safe,
                error,
                max_delta,
                [round(math.degrees(value), 1) for value in solution[:4]],
            )
        )
        all_safe = all_safe and safe
        if not safe:
            break
        current = solution
    print("RESTRICTED_IK_RESULT", "PASS" if all_safe else "BLOCK")


def run_restricted_side_motion(
    move_time,
    test_lift_step=False,
    lift_height=0.01,
):
    runtime = argparse.Namespace(timeout=20.0)
    start = list(pipeline._read_current_arm_joints(runtime.timeout))
    start_pose = grasp_ik._call_fk(start, runtime.timeout).left_pose
    target_xyz = [start_pose.pos_xyz[0], 0.34, start_pose.pos_xyz[2]]
    solution, predicted_error, solver_success = solve_left_position_ik(
        runtime,
        start,
        target_xyz,
    )
    deltas = [
        abs(math.degrees(solution[index] - start[index]))
        for index in range(14)
    ]
    if not solver_success or predicted_error > 0.025 or max(deltas[:4]) > 20.0:
        raise RuntimeError(
            "restricted side motion blocked before command: "
            "xyz_err=%.4fm max_delta=%.1fdeg"
            % (predicted_error, max(deltas[:4]))
        )

    arm_hold = None
    arm_mode_changed = False
    humanoid_mode_changed = False
    wbc_trajectory_enabled = False
    returned_to_start = False
    return_waypoints = [list(start)]
    try:
        arm_hold = pipeline._start_arm_traj_hold(20.0)
        pipeline._set_arm_mode(pipeline.ARM_MODE_EXTERNAL_CONTROL, timeout=20.0)
        arm_mode_changed = True
        set_humanoid_arm_mode(pipeline.ARM_MODE_EXTERNAL_CONTROL, timeout=20.0)
        humanoid_mode_changed = True
        set_wbc_arm_trajectory_enabled(True, timeout=20.0)
        wbc_trajectory_enabled = True

        pipeline._execute_arm_motion(
            None,
            arm_hold,
            [math.degrees(value) for value in start],
            [math.degrees(value) for value in solution],
            float(move_time),
            1.0,
        )
        actual = list(pipeline._read_current_arm_joints(20.0))
        actual_pose = grasp_ik._call_fk(actual, 20.0).left_pose
        actual_error = math.sqrt(
            sum(
                (float(actual_pose.pos_xyz[index]) - target_xyz[index]) ** 2
                for index in range(3)
            )
        )
        rospy.loginfo(
            "restricted side actual=%s target=%s xyz_err=%.4fm",
            [round(float(value), 4) for value in actual_pose.pos_xyz],
            [round(float(value), 4) for value in target_xyz],
            actual_error,
        )
        return_waypoints.append(list(actual))

        if test_lift_step:
            if (
                float(lift_height) <= 0.0
                or float(lift_height) > MAX_RESTRICTED_LIFT_M
            ):
                raise ValueError(
                    "restricted lift height must be within (0, %.2f]"
                    % MAX_RESTRICTED_LIFT_M
                )
            lift_origin = [float(value) for value in actual_pose.pos_xyz]
            step_count = int(math.ceil(float(lift_height) / 0.01))
            for step in range(1, step_count + 1):
                requested_rise = min(0.01 * step, float(lift_height))
                lift_target = [
                    lift_origin[0],
                    lift_origin[1],
                    lift_origin[2] + requested_rise,
                ]
                lift_solution, lift_prediction_error, lift_success = (
                    solve_left_position_ik(runtime, actual, lift_target)
                )
                lift_deltas = [
                    abs(math.degrees(lift_solution[index] - actual[index]))
                    for index in range(4)
                ]
                if (
                    not lift_success
                    or lift_prediction_error > 0.015
                    or max(lift_deltas) > 20.0
                ):
                    raise RuntimeError(
                        "restricted lift %d blocked before command: "
                        "xyz_err=%.4fm max_delta=%.1fdeg"
                        % (step, lift_prediction_error, max(lift_deltas))
                    )
                pipeline._execute_arm_motion(
                    None,
                    arm_hold,
                    [math.degrees(value) for value in actual],
                    [math.degrees(value) for value in lift_solution],
                    max(5.0, float(move_time) / 2.0),
                    1.0,
                )
                rospy.sleep(RESTRICTED_SEGMENT_SETTLE_SECONDS)
                actual = list(pipeline._read_current_arm_joints(20.0))
                lifted_pose = grasp_ik._call_fk(actual, 20.0).left_pose
                lift_actual_error = math.sqrt(
                    sum(
                        (
                            float(lifted_pose.pos_xyz[index])
                            - lift_target[index]
                        ) ** 2
                        for index in range(3)
                    )
                )
                rospy.loginfo(
                    "restricted lift %d/%d actual=%s target=%s xyz_err=%.4fm",
                    step,
                    step_count,
                    [round(float(value), 4) for value in lifted_pose.pos_xyz],
                    [round(float(value), 4) for value in lift_target],
                    lift_actual_error,
                )
                if lift_actual_error > 0.015:
                    raise RuntimeError(
                        "restricted lift %d tracking error %.4fm"
                        % (step, lift_actual_error)
                    )
                return_waypoints.append(list(actual))

        current_return = list(pipeline._read_current_arm_joints(20.0))
        for return_target in reversed(return_waypoints[:-1]):
            pipeline._execute_arm_motion(
                None,
                arm_hold,
                [math.degrees(value) for value in current_return],
                [math.degrees(value) for value in return_target],
                max(5.0, float(move_time) / 2.0),
                0.5,
            )
            current_return = list(pipeline._read_current_arm_joints(20.0))
        restored = list(pipeline._read_current_arm_joints(20.0))
        restored_pose = grasp_ik._call_fk(restored, 20.0).left_pose
        returned_to_start = True
        rospy.loginfo(
            "restricted side restored xyz=%s",
            [round(float(value), 4) for value in restored_pose.pos_xyz],
        )
    finally:
        if (
            not returned_to_start
            and wbc_trajectory_enabled
            and arm_hold is not None
        ):
            try:
                fallback_start = list(pipeline._read_current_arm_joints(10.0))
                for return_target in reversed(return_waypoints):
                    pipeline._execute_arm_motion(
                        None,
                        arm_hold,
                        [math.degrees(value) for value in fallback_start],
                        [math.degrees(value) for value in return_target],
                        max(5.0, float(move_time) / 2.0),
                        0.5,
                    )
                    fallback_start = list(
                        pipeline._read_current_arm_joints(10.0)
                    )
                rospy.logwarn("restricted diagnostic used fallback return")
            except Exception as error:
                rospy.logwarn("restricted fallback return failed: %s", error)
        if wbc_trajectory_enabled:
            set_wbc_arm_trajectory_enabled(False, timeout=10.0)
        if arm_hold is not None:
            arm_hold.stop()
        if humanoid_mode_changed:
            set_humanoid_arm_mode(pipeline.ARM_MODE_AUTO_SWING, timeout=10.0)
        if arm_mode_changed:
            pipeline._set_arm_mode(pipeline.ARM_MODE_AUTO_SWING, timeout=10.0)
def _execute_restricted_waypoint(
    runtime,
    arm_hold,
    current,
    target_xyz,
    label,
    move_time,
    prediction_tolerance=0.005,
    actual_tolerance=0.015,
):
    alternate_seeds = (
        LOW_APPROACH_IK_SEEDS_RAD
        if label.startswith("approach_")
        else None
    )
    solution, prediction_error, solver_success = solve_left_position_ik(
        runtime,
        current,
        target_xyz,
        alternate_active_seeds=alternate_seeds,
    )
    deltas = [
        abs(math.degrees(solution[index] - current[index]))
        for index in range(4)
    ]
    if (
        not solver_success
        or prediction_error > prediction_tolerance
        or max(deltas) > 20.0
    ):
        raise RuntimeError(
            "%s blocked before command: xyz_err=%.4fm max_delta=%.1fdeg"
            % (label, prediction_error, max(deltas))
        )
    if label.startswith("pinch_align_"):
        rospy.loginfo(
            "restricted %s joint command target(deg)=%s delta(deg)=%s",
            label,
            [round(math.degrees(solution[index]), 3) for index in range(7)],
            [
                round(math.degrees(solution[index] - current[index]), 3)
                for index in range(7)
            ],
        )

    current_pose = grasp_ik._call_fk(current, runtime.timeout).left_pose
    requested_distance = math.sqrt(
        sum(
            (float(target_xyz[index]) - float(current_pose.pos_xyz[index])) ** 2
            for index in range(3)
        )
    )
    segment_duration = _restricted_motion_duration(
        current,
        solution,
        move_time,
    )
    pipeline._execute_arm_motion(
        None,
        arm_hold,
        [math.degrees(value) for value in current],
        [math.degrees(value) for value in solution],
        segment_duration,
        1.0,
    )
    rospy.sleep(RESTRICTED_SEGMENT_SETTLE_SECONDS)
    actual = list(pipeline._read_current_arm_joints(runtime.timeout))
    actual_pose = grasp_ik._call_fk(actual, runtime.timeout).left_pose
    actual_error = math.sqrt(
        sum(
            (float(actual_pose.pos_xyz[index]) - float(target_xyz[index])) ** 2
            for index in range(3)
        )
    )
    observed_distance = math.sqrt(
        sum(
            (float(actual_pose.pos_xyz[index]) - float(current_pose.pos_xyz[index])) ** 2
            for index in range(3)
        )
    )
    if (
        requested_distance >= 0.003
        and observed_distance < RESTRICTED_MIN_PROGRESS_RATIO * requested_distance
    ):
        rospy.logwarn(
            "restricted %s progressed %.4fm of %.4fm; holding target once more",
            label,
            observed_distance,
            requested_distance,
        )
        pipeline._execute_arm_motion(
            None,
            arm_hold,
            [math.degrees(value) for value in actual],
            [math.degrees(value) for value in solution],
            max(3.0, segment_duration),
            1.0,
        )
        rospy.sleep(RESTRICTED_SEGMENT_SETTLE_SECONDS)
        actual = list(pipeline._read_current_arm_joints(runtime.timeout))
        actual_pose = grasp_ik._call_fk(actual, runtime.timeout).left_pose
        actual_error = math.sqrt(
            sum(
                (float(actual_pose.pos_xyz[index]) - float(target_xyz[index])) ** 2
                for index in range(3)
            )
        )
        rospy.loginfo(
            "restricted %s retry actual=%s target=%s xyz_err=%.4fm",
            label,
            [round(float(value), 4) for value in actual_pose.pos_xyz],
            [round(float(value), 4) for value in target_xyz],
            actual_error,
        )
    rospy.loginfo(
        "restricted %s actual=%s target=%s xyz_err=%.4fm",
        label,
        [round(float(value), 4) for value in actual_pose.pos_xyz],
        [round(float(value), 4) for value in target_xyz],
        actual_error,
    )
    if label.startswith("pinch_align_"):
        rospy.loginfo(
            "restricted %s joint feedback actual_delta(deg)=%s tracking_error(deg)=%s",
            label,
            [
                round(math.degrees(actual[index] - current[index]), 3)
                for index in range(7)
            ],
            [
                round(math.degrees(solution[index] - actual[index]), 3)
                for index in range(7)
            ],
        )
    if actual_error > actual_tolerance:
        raise RuntimeError(
            "%s tracking error %.4fm" % (label, actual_error)
        )
    return actual, actual_pose


def _restricted_motion_duration(start_radians, target_radians, move_time):
    """Scale a safe waypoint duration to its largest commanded change."""
    max_delta_deg = max(
        abs(math.degrees(float(target_radians[index]) - float(start_radians[index])))
        for index in range(4)
    )
    # The simulator needs the full, conservative duration for the first
    # large shoulder/elbow transition out of the resting pose.
    if max_delta_deg >= 12.0:
        return 15.0
    requested_cap = max(RESTRICTED_MIN_SEGMENT_DURATION_S, float(move_time))
    return min(
        requested_cap,
        max(RESTRICTED_MIN_SEGMENT_DURATION_S, 0.20 * max_delta_deg),
    )


def _return_restricted_waypoints(runtime, arm_hold, current, waypoints, move_time):
    for return_target in reversed(waypoints):
        segment_duration = _restricted_motion_duration(
            current,
            return_target,
            move_time,
        )
        pipeline._execute_arm_motion(
            None,
            arm_hold,
            [math.degrees(value) for value in current],
            [math.degrees(value) for value in return_target],
            segment_duration,
            0.5,
        )
        current = list(pipeline._read_current_arm_joints(runtime.timeout))
    return current


def _align_left_pinch_from_feedback(
    runtime,
    arm_hold,
    current,
    job,
    move_time,
    clearance,
    max_steps,
    tolerance,
    return_waypoints,
    pinch_center_clearance=None,
):
    """Use TF feedback to align the real pinch center while staying above a part."""
    clearance = float(clearance)
    if clearance < 0.01 or clearance > 0.08:
        raise ValueError("pinch clearance must be within [0.010, 0.080]")
    max_steps = int(max_steps)
    if max_steps < 1 or max_steps > 20:
        raise ValueError("pinch alignment max steps must be within [1, 20]")
    tolerance = float(tolerance)
    if tolerance < 0.003 or tolerance > 0.02:
        raise ValueError("pinch alignment tolerance must be within [0.003, 0.020]")

    previous_distance = None
    stalled_steps = 0
    for index in range(1, max_steps + 1):
        physical = read_left_pinch_position(timeout=2.0)
        if pinch_center_clearance is None:
            desired = [
                float(job["vision_base"][0]),
                float(job["vision_base"][1]),
                float(job["vision_base"][2]) + clearance,
            ]
        else:
            desired = [
                float(job["vision_base"][0]),
                float(job["vision_base"][1]),
                float(job["vision_base"][2])
                + float(pinch_center_clearance),
            ]
        difference = [desired[axis] - physical[axis] for axis in range(3)]
        distance = math.sqrt(sum(value * value for value in difference))
        rospy.loginfo(
            "pinch_align_%02d pinch=%s desired=%s dist=%.4fm",
            index,
            [round(value, 4) for value in physical],
            [round(value, 4) for value in desired],
            distance,
        )
        if distance <= tolerance:
            return current
        if distance > 0.10:
            raise RuntimeError("pinch alignment starts too far from target")
        if previous_distance is not None and distance >= previous_distance - 0.0005:
            stalled_steps += 1
        else:
            stalled_steps = 0
        if stalled_steps >= 2:
            raise RuntimeError("pinch alignment did not reduce physical error")

        xy_distance = math.hypot(difference[0], difference[1])
        if xy_distance > 0.003:
            phase = "xy"
            correction = [difference[0], difference[1], 0.0]
            correction_distance = xy_distance
        else:
            phase = "z"
            correction = [0.0, 0.0, difference[2]]
            correction_distance = abs(difference[2])
        rospy.loginfo(
            "pinch_align_%02d phase=%s correction=%s",
            index,
            phase,
            [round(value, 4) for value in correction],
        )
        maximum_step = 0.010 if phase == "z" else 0.005
        step_scale = min(
            1.0,
            maximum_step / max(correction_distance, 1.0e-6),
        )
        fk_pose = grasp_ik._call_fk(current, runtime.timeout).left_pose
        target_fk = [
            float(fk_pose.pos_xyz[axis])
            + step_scale * correction[axis]
            for axis in range(3)
        ]
        current, _ = _execute_restricted_waypoint(
            runtime,
            arm_hold,
            current,
            target_fk,
            "pinch_align_%02d" % index,
            move_time,
            prediction_tolerance=0.015,
            actual_tolerance=0.008 if phase == "z" else 0.02,
        )
        return_waypoints.append(list(current))
        previous_distance = distance

    raise RuntimeError(
        "pinch alignment exceeded the %d-step limit" % max_steps
    )


def run_restricted_high_transit_motion(
    job,
    move_time,
    approach_clearance=None,
    pinch_align_clearance=None,
    pinch_align_max_steps=18,
    pinch_align_tolerance=0.012,
    pinch_center_clearance=None,
    grasp_test=False,
    high_step=0.05,
    approach_step=0.02,
):
    if job["arm"] != "left":
        raise RuntimeError("restricted high transit currently supports left arm")
    high_step = float(high_step)
    approach_step = float(approach_step)
    if high_step < 0.01 or high_step > 0.05:
        raise ValueError("restricted high step must be within [0.010, 0.050]")
    if approach_step < 0.005 or approach_step > 0.02:
        raise ValueError("restricted approach step must be within [0.005, 0.020]")

    runtime = argparse.Namespace(timeout=20.0)
    start = list(pipeline._read_current_arm_joints(runtime.timeout))
    start_pose = grasp_ik._call_fk(start, runtime.timeout).left_pose
    side_target = [start_pose.pos_xyz[0], 0.34, start_pose.pos_xyz[2]]
    arm_hold = None
    arm_mode_changed = False
    humanoid_mode_changed = False
    wbc_trajectory_enabled = False
    returned_to_start = False
    return_waypoints = [list(start)]
    approach_start_index = None

    try:
        if grasp_test:
            control_scene2_claws(0.0, 0.0)
            rospy.sleep(0.5)
        arm_hold = pipeline._start_arm_traj_hold(20.0)
        pipeline._set_arm_mode(pipeline.ARM_MODE_EXTERNAL_CONTROL, timeout=20.0)
        arm_mode_changed = True
        set_humanoid_arm_mode(pipeline.ARM_MODE_EXTERNAL_CONTROL, timeout=20.0)
        humanoid_mode_changed = True
        set_wbc_arm_trajectory_enabled(True, timeout=20.0)
        wbc_trajectory_enabled = True
        # Let the controller consume the new external-control state before
        # issuing the first large shoulder/elbow waypoint.
        rospy.sleep(1.0)

        current, side_pose = _execute_restricted_waypoint(
            runtime,
            arm_hold,
            start,
            side_target,
            "side_clear",
            move_time,
            prediction_tolerance=0.025,
            actual_tolerance=0.025,
        )
        return_waypoints.append(list(current))

        high_target = [
            float(side_pose.pos_xyz[0]),
            float(side_pose.pos_xyz[1]),
            RESTRICTED_HIGH_COMMAND_Z_M,
        ]
        # Replan every lift segment from measured feedback.  A partially
        # tracked step must not make the following precomputed target jump.
        measured_lift_pose = side_pose
        for index in range(1, 41):
            current_z = float(measured_lift_pose.pos_xyz[2])
            if current_z >= SAFE_TRANSIT_EE_Z_M + 0.005:
                break
            step_size = 0.01 if index <= 3 else high_step
            target = [
                float(high_target[0]),
                float(high_target[1]),
                min(current_z + step_size, RESTRICTED_HIGH_COMMAND_Z_M),
            ]
            current, next_lift_pose = _execute_restricted_waypoint(
                runtime,
                arm_hold,
                current,
                target,
                "high_lift_%02d" % index,
                move_time,
            )
            return_waypoints.append(list(current))
            vertical_progress = (
                float(next_lift_pose.pos_xyz[2]) - current_z
            )
            if vertical_progress < 0.0005:
                raise RuntimeError(
                    "high_lift_%02d made insufficient vertical progress %.4fm"
                    % (index, vertical_progress)
                )
            measured_lift_pose = next_lift_pose
        else:
            raise RuntimeError("high lift exceeded the 40-step limit")

        high_pose = measured_lift_pose
        if float(high_pose.pos_xyz[2]) < SAFE_TRANSIT_EE_Z_M:
            raise RuntimeError(
                "high transit blocked: actual z %.4fm below %.4fm"
                % (float(high_pose.pos_xyz[2]), SAFE_TRANSIT_EE_Z_M)
            )

        outside_high = [
            OUTSIDE_TABLE_X_M,
            0.34,
            RESTRICTED_HIGH_COMMAND_Z_M,
        ]
        tracked_high_pose = high_pose
        for index, target in enumerate(
            _linear_positions(high_pose.pos_xyz, outside_high, high_step),
            start=1,
        ):
            corrected_target = list(target)
            corrected_target[2] += min(
                0.025,
                max(
                    0.0,
                    RESTRICTED_HIGH_COMMAND_Z_M
                    - float(tracked_high_pose.pos_xyz[2]),
                ),
            )
            current, tracked_high_pose = _execute_restricted_waypoint(
                runtime,
                arm_hold,
                current,
                corrected_target,
                "high_forward_%02d" % index,
                move_time,
                actual_tolerance=0.02,
            )
            return_waypoints.append(list(current))

        outside_pose = tracked_high_pose
        over_part = [
            float(job["grasp"][0]),
            float(job["grasp"][1]),
            RESTRICTED_HIGH_COMMAND_Z_M,
        ]
        tracked_over_pose = outside_pose
        for index, target in enumerate(
            _linear_positions(outside_pose.pos_xyz, over_part, high_step),
            start=1,
        ):
            corrected_target = list(target)
            corrected_target[2] += min(
                0.025,
                max(
                    0.0,
                    RESTRICTED_HIGH_COMMAND_Z_M
                    - float(tracked_over_pose.pos_xyz[2]),
                ),
            )
            current, tracked_over_pose = _execute_restricted_waypoint(
                runtime,
                arm_hold,
                current,
                corrected_target,
                "over_part_%02d" % index,
                move_time,
                actual_tolerance=0.02,
            )
            return_waypoints.append(list(current))

        final_pose = tracked_over_pose
        if float(final_pose.pos_xyz[2]) < SAFE_TRANSIT_EE_Z_M:
            raise RuntimeError(
                "over-part transit blocked: actual z %.4fm below %.4fm"
                % (float(final_pose.pos_xyz[2]), SAFE_TRANSIT_EE_Z_M)
            )
        rospy.loginfo(
            "restricted high transit reached over-part xyz=%s",
            [round(float(value), 4) for value in final_pose.pos_xyz],
        )

        if approach_clearance is not None:
            clearance = float(approach_clearance)
            if clearance < 0.05 or clearance > 0.15:
                raise ValueError(
                    "restricted approach clearance must be within [0.05, 0.15]"
                )
            approach_target = [
                float(job["grasp"][0]),
                float(job["grasp"][1]),
                float(job["grasp"][2]) + clearance,
            ]
            approach_start_index = len(return_waypoints) - 1
            for index, target in enumerate(
                _linear_positions(
                    final_pose.pos_xyz,
                    approach_target,
                    approach_step,
                ),
                start=1,
            ):
                current, _ = _execute_restricted_waypoint(
                    runtime,
                    arm_hold,
                    current,
                    target,
                    "approach_%02d" % index,
                    move_time,
                )
                return_waypoints.append(list(current))
            approach_pose = grasp_ik._call_fk(
                current,
                runtime.timeout,
            ).left_pose
            rospy.loginfo(
                "restricted approach reached xyz=%s clearance=%.3fm",
                [round(float(value), 4) for value in approach_pose.pos_xyz],
                clearance,
            )
            log_left_gripper_base_pose("restricted approach")
            approach_pinch = read_left_pinch_position(timeout=2.0)
            rospy.loginfo(
                "restricted approach physical pinch=%s",
                [round(float(value), 4) for value in approach_pinch],
            )

        if pinch_align_clearance is not None:
            if approach_clearance is None:
                raise RuntimeError(
                    "pinch alignment requires a verified high approach first"
                )
            current = _align_left_pinch_from_feedback(
                runtime,
                arm_hold,
                current,
                job,
                move_time,
                pinch_align_clearance,
                pinch_align_max_steps,
                pinch_align_tolerance,
                return_waypoints,
                pinch_center_clearance=pinch_center_clearance,
            )
            physical = read_left_pinch_position(timeout=2.0)
            rospy.loginfo(
                "pinch alignment complete physical=%s",
                [round(value, 4) for value in physical],
            )

        if grasp_test:
            if approach_start_index is None:
                raise RuntimeError("grasp test requires a verified approach")
            control_scene2_claws(100.0, 0.0)
            rospy.sleep(1.5)
            lift_targets = return_waypoints[approach_start_index:-1]
            current = _return_restricted_waypoints(
                runtime,
                arm_hold,
                list(current),
                lift_targets,
                move_time,
            )
            physical = read_left_pinch_position(timeout=2.0)
            rospy.loginfo(
                "grasp test lifted to physical pinch=%s",
                [round(value, 4) for value in physical],
            )
            control_scene2_claws(0.0, 0.0)
            rospy.sleep(0.8)
            return_waypoints = return_waypoints[:approach_start_index + 1]

        restored = _return_restricted_waypoints(
            runtime,
            arm_hold,
            list(current),
            return_waypoints[:-1],
            move_time,
        )
        restored_pose = grasp_ik._call_fk(restored, runtime.timeout).left_pose
        returned_to_start = True
        rospy.loginfo(
            "restricted high transit restored xyz=%s",
            [round(float(value), 4) for value in restored_pose.pos_xyz],
        )
    finally:
        if (
            not returned_to_start
            and wbc_trajectory_enabled
            and arm_hold is not None
        ):
            try:
                fallback_current = list(
                    pipeline._read_current_arm_joints(runtime.timeout)
                )
                _return_restricted_waypoints(
                    runtime,
                    arm_hold,
                    fallback_current,
                    return_waypoints,
                    move_time,
                )
                rospy.logwarn("restricted high transit used fallback return")
            except Exception as error:
                rospy.logwarn(
                    "restricted high transit fallback failed: %s", error
                )
        if wbc_trajectory_enabled:
            set_wbc_arm_trajectory_enabled(False, timeout=10.0)
        if grasp_test:
            try:
                control_scene2_claws(0.0, 0.0)
            except Exception as error:
                rospy.logwarn("failed to reopen Scene2 claws: %s", error)
        if arm_hold is not None:
            arm_hold.stop()
        if humanoid_mode_changed:
            set_humanoid_arm_mode(pipeline.ARM_MODE_AUTO_SWING, timeout=10.0)
        if arm_mode_changed:
            pipeline._set_arm_mode(pipeline.ARM_MODE_AUTO_SWING, timeout=10.0)


def _linear_positions(start, end, maximum_step):
    distance = math.sqrt(
        sum((float(end[i]) - float(start[i])) ** 2 for i in range(3))
    )
    count = max(1, int(math.ceil(distance / float(maximum_step))))
    return [
        [
            float(start[i])
            + (float(end[i]) - float(start[i])) * step / float(count)
            for i in range(3)
        ]
        for step in range(1, count + 1)
    ]


def build_safe_waypoints(job, runtime, active_arm, current_joints):
    work_poses = grasp_ik._call_fk(current_joints, runtime.timeout)
    work_pose = (
        work_poses.left_pose if active_arm == "left" else work_poses.right_pose
    )
    start_position = list(work_pose.pos_xyz)
    work_quat = list(work_pose.quat_xyzw)
    grasp_quat = tf.transformations.quaternion_slerp(
        work_quat,
        job["grasp_quat"],
        GRASP_ORIENTATION_BLEND,
    ).tolist()
    job["grasp_quat"] = list(grasp_quat)
    job["lift_quat"] = list(grasp_quat)
    transit_z = max(
        SAFE_TRANSIT_EE_Z_M,
        float(start_position[2]),
        float(job["grasp"][2]) + 0.20,
    )
    high_target = [
        job["grasp"][0],
        job["grasp"][1],
        transit_z,
    ]
    waypoints = [("start_hold", start_position, work_quat)]

    lift_target = [
        start_position[0],
        start_position[1],
        max(float(start_position[2]), BODY_SIDE_LIFT_Z_M),
    ]
    for index, position in enumerate(
        _linear_positions(start_position, lift_target, BODY_SIDE_STEP_M),
        start=1,
    ):
        waypoints.append(("body_side_lift_%02d" % index, position, work_quat))

    side_y = (
        LEFT_SIDE_CLEARANCE_Y_M
        if active_arm == "left"
        else -LEFT_SIDE_CLEARANCE_Y_M
    )
    side_target = [lift_target[0], side_y, lift_target[2]]
    for index, position in enumerate(
        _linear_positions(lift_target, side_target, MAX_CARTESIAN_STEP_M),
        start=1,
    ):
        waypoints.append(("body_side_clear_%02d" % index, position, work_quat))

    for ratio in (1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0, 1.0):
        side_quat = tf.transformations.quaternion_slerp(
            work_quat,
            grasp_quat,
            ratio,
        ).tolist()
        waypoints.append(
            (
                "outside_rotate_%d" % round(ratio * 100.0),
                list(side_target),
                side_quat,
            )
        )

    outside_forward = [OUTSIDE_TABLE_X_M, side_y, side_target[2]]
    for index, position in enumerate(
        _linear_positions(side_target, outside_forward, MAX_CARTESIAN_STEP_M),
        start=1,
    ):
        waypoints.append(("outside_forward_%02d" % index, position, grasp_quat))

    outside_high = [OUTSIDE_TABLE_X_M, side_y, transit_z]
    for index, position in enumerate(
        _linear_positions(outside_forward, outside_high, OUTSIDE_LIFT_STEP_M),
        start=1,
    ):
        waypoints.append(("outside_table_lift_%02d" % index, position, grasp_quat))

    high_positions = _linear_positions(
        outside_high,
        high_target,
        MAX_CARTESIAN_STEP_M,
    )
    for index, position in enumerate(high_positions, start=1):
        label = (
            "move_high"
            if index == len(high_positions)
            else "high_transit_%02d" % index
        )
        waypoints.append((label, position, grasp_quat))
    if waypoints[-1][0] != "move_high":
        waypoints.append(("move_high", high_target, grasp_quat))
    approach_10cm = [
        job["grasp"][0],
        job["grasp"][1],
        job["grasp"][2] + 0.10,
    ]
    approach_positions = _linear_positions(
        high_target,
        approach_10cm,
        MAX_CARTESIAN_STEP_M,
    )
    for index, position in enumerate(approach_positions, start=1):
        label = (
            "approach_10cm"
            if index == len(approach_positions)
            else "clear_descent_%02d" % index
        )
        waypoints.append((label, position, list(grasp_quat)))
    waypoints.append(
        (
            "approach_5cm",
            [job["grasp"][0], job["grasp"][1], job["grasp"][2] + 0.05],
            list(grasp_quat),
        )
    )
    waypoints.append(("grasp", list(job["grasp"]), list(grasp_quat)))
    return waypoints


def execute_cartesian_waypoint(
    runtime,
    active_arm,
    locked_other_arm_joints,
    label,
    target,
    target_quat,
):
    base_label = label[8:] if label.startswith("retreat_") else label
    transit_prefixes = (
        "start_hold",
        "body_side_",
        "outside_",
        "high_rotate_",
        "high_transit_",
        "move_high",
    )
    outside_prefixes = ("body_side_", "outside_")
    outside_lift_prefix = "outside_table_lift_"
    orientation_tolerance = (
        math.radians(SAFE_OUTSIDE_LIFT_ORIENTATION_TOLERANCE_DEG)
        if base_label.startswith(outside_lift_prefix)
        else math.radians(SAFE_HIGH_TRANSIT_ORIENTATION_TOLERANCE_DEG)
        if base_label.startswith("high_transit_")
        else runtime.orientation_tolerance_rad
    )
    if base_label.startswith(outside_prefixes):
        position_tolerance = SAFE_OUTSIDE_POSITION_TOLERANCE_M
    elif base_label.startswith(transit_prefixes):
        position_tolerance = SAFE_TRANSIT_POSITION_TOLERANCE_M
    else:
        position_tolerance = SAFE_GRASP_POSITION_TOLERANCE_M
    runtime.actual_joint_error_limit_deg = (
        MAX_OUTSIDE_LIFT_ACTUAL_JOINT_ERROR_DEG
        if base_label.startswith(outside_lift_prefix)
        else MAX_OUTSIDE_ACTUAL_JOINT_ERROR_DEG
        if base_label.startswith(outside_prefixes)
        else MAX_TRANSIT_ACTUAL_JOINT_ERROR_DEG
        if base_label.startswith(transit_prefixes)
        else MAX_ACTUAL_JOINT_ERROR_DEG
    )
    runtime.ik_segment_limit_deg = (
        MAX_OUTSIDE_LIFT_IK_SEGMENT_DELTA_DEG
        if base_label.startswith(outside_lift_prefix)
        else MAX_OUTSIDE_IK_SEGMENT_DELTA_DEG
        if base_label.startswith(outside_prefixes)
        else MAX_TRANSIT_IK_SEGMENT_DELTA_DEG
        if base_label.startswith(transit_prefixes)
        else MAX_IK_SEGMENT_DELTA_DEG
    )
    for attempt in (1, 2, 3):
        try:
            position_error, orientation_error, actual, _quat, _command = (
                grasp_ik._move_arm_ik_once(
                    runtime=runtime,
                    active_arm=active_arm,
                    active_pos=target,
                    locked_other_arm_joints=locked_other_arm_joints,
                    active_quat=target_quat,
                    label="%s_try%d" % (label, attempt),
                    constraint_mode=runtime.ik_mode_pos_hard_ori_hard,
                    pos_cost_weight=1.0,
                    move_time=runtime.move_time,
                    settle_time=runtime.settle_time,
                )
            )
        except RuntimeError as exc:
            if str(exc).startswith("IK motion blocked:") and attempt < 3:
                rospy.logwarn(
                    "%s unsafe IK branch on try %d; recomputing without motion",
                    label,
                    attempt,
                )
                continue
            raise
        if (
            position_error <= position_tolerance
            and orientation_error <= orientation_tolerance
        ):
            return
        if (
            base_label.startswith(outside_prefixes)
            and position_error <= 0.06
            and orientation_error <= orientation_tolerance
            and attempt < 3
        ):
            rospy.logwarn(
                "%s exterior correction: xyz=%.4fm target_limit=%.4fm",
                label,
                position_error,
                position_tolerance,
            )
            continue
        raise RuntimeError(
            "%s tracking error: xyz=%.4fm/%.4fm quat=%.1fdeg"
            % (
                label,
                position_error,
                position_tolerance,
                math.degrees(orientation_error),
            )
        )


def execute_joint_transfer(runtime, target_degrees, label):
    current_radians = pipeline._read_current_arm_joints(20.0)
    start_degrees = [math.degrees(value) for value in current_radians]
    total_deltas = [
        float(target_degrees[index]) - start_degrees[index]
        for index in range(14)
    ]
    step_count = max(
        1,
        int(
            math.ceil(
                max(abs(value) for value in total_deltas)
                / MAX_TRANSFER_STEP_DEG
            )
        ),
    )
    previous = list(start_degrees)
    for step_index in range(1, step_count + 1):
        ratio = float(step_index) / float(step_count)
        waypoint = [
            start_degrees[index] + total_deltas[index] * ratio
            for index in range(14)
        ]
        rospy.loginfo(
            "scene2 safe transfer: %s step %d/%d",
            label,
            step_index,
            step_count,
        )
        runtime.execute_arm_motion_cb(
            previous,
            waypoint,
            runtime.move_time,
            runtime.settle_time,
        )
        previous = waypoint


def execute_safe_stage(job, runtime, gripper_hold, locked_other_arm_joints, stage):
    active_arm = job["arm"]
    current = pipeline._read_current_arm_joints(20.0)
    waypoints = build_safe_waypoints(job, runtime, active_arm, current)
    if stage == "body":
        stage_count = max(
            index + 1
            for index, waypoint in enumerate(waypoints)
            if waypoint[0].startswith("body_side_lift_")
        )
    else:
        stage_end_label = {
            "lift": "body_side_lift_01",
            "high": "move_high",
            "approach": "approach_10cm",
            "full": "grasp",
        }[stage]
        stage_count = next(
            index + 1
            for index, waypoint in enumerate(waypoints)
            if waypoint[0] == stage_end_label
        )
    executed = waypoints[:stage_count]

    for label, target, target_quat in executed:
        execute_cartesian_waypoint(
            runtime,
            active_arm,
            locked_other_arm_joints,
            label,
            target,
            target_quat,
        )

    if stage == "full":
        runtime.publish_arm_gripper_close_cb(active_arm)
        runtime.sleep_cb(runtime.gripper_close_time)

    if stage == "full":
        high_index = next(
            index for index, waypoint in enumerate(executed)
            if waypoint[0] == "move_high"
        )
        retreat_waypoints = reversed(executed[high_index:-1])
    else:
        retreat_waypoints = reversed(executed[:-1])

    for label, target, target_quat in retreat_waypoints:
        execute_cartesian_waypoint(
            runtime,
            active_arm,
            locked_other_arm_joints,
            "retreat_" + label,
            target,
            target_quat,
        )

    if stage != "full":
        return

    active_place_joints = pipeline._place_active_arm_joints(
        active_arm,
        job["bin"],
    )
    place_target_degrees = pipeline._compose_single_arm_place_joints(
        active_arm,
        active_place_joints,
        locked_other_arm_joints,
    )
    execute_joint_transfer(runtime, place_target_degrees, "transfer_to_bin")
    pipeline._publish_gripper_open(gripper_hold)
    rospy.sleep(pipeline.PLACE_DWELL)


def main():
    args = parse_args()
    if args.move_time < 3.0:
        raise ValueError("move-time must be at least 3 seconds")

    rospy.init_node("scene2_pick_nearest_red", anonymous=True)

    if args.fk_tf_check:
        run_fk_tf_check()
        return

    if args.full_ik_probe:
        pipeline._publish_head_target(20.0)
        rospy.sleep(1.0)
        run_full_ik_probe(
            build_vision_red_job(args.object),
            clearance=args.full_ik_clearance,
        )
        return

    if args.joint5_check:
        run_joint5_check(
            args.move_time,
            args.joint5_delta,
            args.joint5_interface,
            args.joint_index,
        )
        return

    if args.restricted_side_execute:
        run_restricted_side_motion(args.move_time)
        return

    if args.restricted_lift_step_execute:
        run_restricted_side_motion(
            args.move_time,
            test_lift_step=True,
            lift_height=args.restricted_lift_height,
        )
        return

    if args.restricted_high_transit_execute:
        pipeline._publish_head_target(20.0)
        rospy.sleep(1.0)
        job = build_vision_red_job(args.object)
        run_restricted_high_transit_motion(
            job,
            args.move_time,
            high_step=args.restricted_high_step,
            approach_step=args.restricted_approach_step,
        )
        return

    if args.restricted_approach_execute:
        pipeline._publish_head_target(20.0)
        rospy.sleep(1.0)
        job = build_vision_red_job(args.object)
        run_restricted_high_transit_motion(
            job,
            args.move_time,
            approach_clearance=args.restricted_approach_clearance,
            high_step=args.restricted_high_step,
            approach_step=args.restricted_approach_step,
        )
        return

    if args.restricted_pinch_align_execute:
        pipeline._publish_head_target(20.0)
        rospy.sleep(1.0)
        job = build_vision_red_job(args.object)
        run_restricted_high_transit_motion(
            job,
            args.move_time,
            approach_clearance=args.restricted_approach_clearance,
            pinch_align_clearance=args.pinch_clearance,
            pinch_align_max_steps=args.pinch_max_steps,
            pinch_align_tolerance=args.pinch_tolerance,
            high_step=args.restricted_high_step,
            approach_step=args.restricted_approach_step,
        )
        return

    if args.restricted_pinch_grasp_test_execute:
        pipeline._publish_head_target(20.0)
        rospy.sleep(1.0)
        job = build_vision_red_job(args.object)
        run_restricted_high_transit_motion(
            job,
            args.move_time,
            approach_clearance=0.13,
            pinch_align_clearance=0.017,
            pinch_align_max_steps=18,
            pinch_align_tolerance=0.005,
            pinch_center_clearance=0.02,
            grasp_test=True,
            high_step=args.restricted_high_step,
            approach_step=args.restricted_approach_step,
        )
        return

    if (
        args.plan_only
        or args.ik_check
        or args.restricted_ik_check
        or not args.execute
    ):
        pipeline._publish_head_target(20.0)
        rospy.sleep(1.0)
        job = build_vision_red_job(args.object)
        print("selected:", job["object"])
        print("grasp:", [round(value, 4) for value in job["grasp"]])
        if args.ik_check:
            run_ik_check(job, args.move_time)
        if args.restricted_ik_check:
            run_restricted_ik_check(
                job,
                args.move_time,
                high_step=args.restricted_high_step,
                approach_step=args.restricted_approach_step,
            )
        if not args.plan_only:
            print("plan only: add --execute to allow arm motion")
        return

    pipeline.ARM_MOVE_TIME = args.move_time
    pipeline.FAST_GRASP_SETTLE_HOLD = 1.0

    gripper_hold = None
    arm_hold = None
    arm_mode_changed = False
    humanoid_mode_changed = False
    wbc_trajectory_enabled = False
    success = False

    try:
        pipeline._publish_head_target(20.0)
        gripper_hold = pipeline._start_gripper_hold(20.0)
        arm_hold = pipeline._start_arm_traj_hold(20.0)
        arm_pub = rospy.Publisher(
            pipeline.ARM_TARGET_POSES_TOPIC,
            armTargetPoses,
            queue_size=10,
        )
        pipeline._wait_for_connection(arm_pub, 20.0)

        pipeline._set_arm_mode(pipeline.ARM_MODE_EXTERNAL_CONTROL, timeout=20.0)
        arm_mode_changed = True
        set_humanoid_arm_mode(pipeline.ARM_MODE_EXTERNAL_CONTROL, timeout=20.0)
        humanoid_mode_changed = True
        set_wbc_arm_trajectory_enabled(True, timeout=20.0)
        wbc_trajectory_enabled = True

        job = build_vision_red_job(args.object)

        runtime = make_runtime(
            arm_pub,
            arm_hold,
            gripper_hold,
            args.move_time,
            job,
        )
        pipeline._publish_gripper_open(gripper_hold)

        if job["arm"] != "left" or job["bin"] != "sorting_bin_c":
            raise RuntimeError("selected red part is not a direct left-to-purple job")

        rospy.loginfo(
            "scene2 single pick: %s -> %s grasp=%s",
            job["object"],
            job["bin"],
            [round(value, 4) for value in job["grasp"]],
        )
        current_joints = pipeline._read_current_arm_joints(20.0)
        locked_other_arm = list(current_joints[7:14])
        execute_safe_stage(
            job,
            runtime,
            gripper_hold,
            locked_other_arm,
            args.stage,
        )
        success = True
        rospy.loginfo("scene2 stage %s completed successfully", args.stage)
    finally:
        if not success and gripper_hold is not None:
            try:
                pipeline._publish_gripper_open(gripper_hold)
            except Exception:
                pass
        if wbc_trajectory_enabled:
            try:
                set_wbc_arm_trajectory_enabled(False, timeout=10.0)
            except Exception as error:
                rospy.logwarn("failed to disable WBC arm trajectory: %s", error)
        if arm_hold is not None:
            arm_hold.stop()
        if gripper_hold is not None:
            gripper_hold.stop()
        if humanoid_mode_changed:
            try:
                set_humanoid_arm_mode(
                    pipeline.ARM_MODE_AUTO_SWING,
                    timeout=10.0,
                )
            except Exception as error:
                rospy.logwarn("failed to restore humanoid arm mode: %s", error)
        if arm_mode_changed:
            try:
                pipeline._set_arm_mode(
                    pipeline.ARM_MODE_AUTO_SWING,
                    timeout=10.0,
                )
            except Exception as error:
                rospy.logwarn("failed to restore arm mode: %s", error)


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSException, RuntimeError, ValueError) as error:
        rospy.logerr("scene2 single pick failed: %s", error)
        raise SystemExit(1)
