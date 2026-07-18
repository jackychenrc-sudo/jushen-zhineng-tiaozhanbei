#!/usr/bin/env python3
"""Allowed Scene2 FK/IK and quaternion helpers.

Only public robot kinematics services and pure numerical helpers live here.
There is intentionally no simulator layout, object-pose, or scene-control code.
"""

import math
from dataclasses import dataclass


USE_CUSTOM_IK_PARAM = True
JOINT_ANGLES_AS_Q0 = True


@dataclass
class GraspRuntime:
    """Runtime parameters and callbacks used by the Scene2 IK helpers."""

    world_to_ee_offset_x: float
    world_to_ee_offset_y_left: float
    world_to_ee_offset_y_right: float
    world_to_ee_offset_z: float
    pre_grasp_z_offset: float
    grasp_position_tolerance: float
    orientation_tolerance_rad: float
    gripper_close_time: float
    timeout: float
    move_time: float
    settle_time: float
    ik_mode_pos_hard_ori_hard: int
    read_current_arm_joints_cb: callable
    execute_arm_motion_cb: callable
    publish_arm_gripper_close_cb: callable
    sleep_cb: callable
    loginfo_cb: callable
    logwarn_cb: callable


def _rad_to_deg(point):
    return [math.degrees(float(value)) for value in point]


def _axis_error(actual, desired):
    return math.sqrt(
        sum(
            (float(actual[index]) - float(desired[index])) ** 2
            for index in range(3)
        )
    )


def _quat_angle_error(actual_xyzw, desired_xyzw):
    dot = sum(
        float(actual_xyzw[index]) * float(desired_xyzw[index])
        for index in range(4)
    )
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


def _make_ik_param_like_example(constraint_mode, pos_cost_weight):
    from kuavo_msgs.msg import ikSolveParam

    param = ikSolveParam()
    param.major_optimality_tol = 1e-3
    param.major_feasibility_tol = 1e-3
    param.minor_feasibility_tol = 1e-3
    param.major_iterations_limit = 100
    param.oritation_constraint_tol = 1e-3
    param.pos_constraint_tol = 1e-3
    param.pos_cost_weight = float(pos_cost_weight)
    param.constraint_mode = int(constraint_mode)
    return param


def _call_fk(joint_angles, timeout):
    import rospy
    from kuavo_msgs.srv import fkSrv

    rospy.wait_for_service("/ik/fk_srv", timeout=timeout)
    response = rospy.ServiceProxy("/ik/fk_srv", fkSrv)(list(joint_angles))
    if not response.success:
        raise RuntimeError("/ik/fk_srv returned success=false")
    return response.hand_poses


def _call_two_hands_ik(
    runtime: GraspRuntime,
    current_joint_values,
    left_pos,
    right_pos,
    left_quat,
    right_quat,
    constraint_mode,
    pos_cost_weight,
):
    import rospy
    from kuavo_msgs.msg import twoArmHandPoseCmd
    from kuavo_msgs.srv import twoArmHandPoseCmdSrv

    request = twoArmHandPoseCmd()
    request.ik_param = _make_ik_param_like_example(
        constraint_mode, pos_cost_weight
    )
    request.use_custom_ik_param = USE_CUSTOM_IK_PARAM
    request.joint_angles_as_q0 = JOINT_ANGLES_AS_Q0

    request.hand_poses.left_pose.joint_angles = list(
        current_joint_values[:7]
    )
    request.hand_poses.right_pose.joint_angles = list(
        current_joint_values[7:]
    )
    request.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]

    request.hand_poses.left_pose.pos_xyz = list(left_pos)
    request.hand_poses.left_pose.quat_xyzw = list(left_quat)
    request.hand_poses.right_pose.pos_xyz = list(right_pos)
    request.hand_poses.right_pose.quat_xyzw = list(right_quat)
    rospy.loginfo(
        "scene2 ik request: left_pos=%s left_quat=%s "
        "right_pos=%s right_quat=%s",
        [round(float(value), 6) for value in left_pos],
        [round(float(value), 6) for value in left_quat],
        [round(float(value), 6) for value in right_pos],
        [round(float(value), 6) for value in right_quat],
    )

    rospy.wait_for_service(
        "/ik/two_arm_hand_pose_cmd_srv", timeout=runtime.timeout
    )
    response = rospy.ServiceProxy(
        "/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv
    )(request)
    if not response.success:
        raise RuntimeError(
            "/ik/two_arm_hand_pose_cmd_srv failed: "
            + getattr(response, "error_reason", "")
            + f" left={list(left_pos)} right={list(right_pos)}"
        )

    q_arm = list(response.q_arm) if hasattr(response, "q_arm") else []
    left_result = list(response.hand_poses.left_pose.joint_angles)
    right_result = list(response.hand_poses.right_pose.joint_angles)
    q_arm_deg = (
        [round(math.degrees(float(value)), 3) for value in q_arm]
        if q_arm
        else []
    )
    left_result_deg = [
        round(math.degrees(float(value)), 3) for value in left_result
    ]
    right_result_deg = [
        round(math.degrees(float(value)), 3) for value in right_result
    ]
    rospy.loginfo(
        "scene2 ik result(deg): q_arm=%s "
        "left_joint_angles=%s right_joint_angles=%s",
        q_arm_deg,
        left_result_deg,
        right_result_deg,
    )

    if len(q_arm) >= 14:
        return q_arm[:14]
    if len(left_result) == 7 and len(right_result) == 7:
        return left_result + right_result
    raise RuntimeError("IK response did not contain arm joints")


def _call_single_arm_ik(
    runtime: GraspRuntime,
    current_joint_values,
    active_arm,
    active_pos,
    active_quat,
    locked_other_arm_joints,
    constraint_mode,
    pos_cost_weight,
):
    current_fk = _call_fk(current_joint_values, runtime.timeout)

    if active_arm == "left":
        right_lock = (
            [float(value) for value in locked_other_arm_joints]
            if locked_other_arm_joints is not None
            else list(current_joint_values[7:])
        )
        q0 = list(current_joint_values[:7]) + right_lock
        lock_fk = _call_fk(q0, runtime.timeout)
        left_quat = (
            list(active_quat)
            if active_quat is not None
            else list(current_fk.left_pose.quat_xyzw)
        )
        ik_full = _call_two_hands_ik(
            runtime=runtime,
            current_joint_values=q0,
            left_pos=list(active_pos),
            right_pos=list(lock_fk.right_pose.pos_xyz),
            left_quat=left_quat,
            right_quat=list(lock_fk.right_pose.quat_xyzw),
            constraint_mode=constraint_mode,
            pos_cost_weight=pos_cost_weight,
        )
        return list(ik_full[:7]) + right_lock

    if active_arm == "right":
        left_lock = (
            [float(value) for value in locked_other_arm_joints]
            if locked_other_arm_joints is not None
            else list(current_joint_values[:7])
        )
        q0 = left_lock + list(current_joint_values[7:])
        lock_fk = _call_fk(q0, runtime.timeout)
        right_quat = (
            list(active_quat)
            if active_quat is not None
            else list(current_fk.right_pose.quat_xyzw)
        )
        ik_full = _call_two_hands_ik(
            runtime=runtime,
            current_joint_values=q0,
            left_pos=list(lock_fk.left_pose.pos_xyz),
            right_pos=list(active_pos),
            left_quat=list(lock_fk.left_pose.quat_xyzw),
            right_quat=right_quat,
            constraint_mode=constraint_mode,
            pos_cost_weight=pos_cost_weight,
        )
        return left_lock + list(ik_full[7:])

    raise ValueError(f"unknown arm: {active_arm}")


def _measure_hand_pose(runtime: GraspRuntime, arm):
    q = runtime.read_current_arm_joints_cb()
    poses = _call_fk(q, runtime.timeout)
    pose = poses.left_pose if arm == "left" else poses.right_pose
    return list(pose.pos_xyz), list(pose.quat_xyzw)


def _move_arm_ik_once(
    runtime: GraspRuntime,
    active_arm,
    active_pos,
    locked_other_arm_joints,
    active_quat,
    label,
    constraint_mode,
    pos_cost_weight,
    move_time,
    settle_time,
):
    current = runtime.read_current_arm_joints_cb()
    ik_q = _call_single_arm_ik(
        runtime=runtime,
        current_joint_values=current,
        active_arm=active_arm,
        active_pos=active_pos,
        active_quat=active_quat,
        locked_other_arm_joints=locked_other_arm_joints,
        constraint_mode=constraint_mode,
        pos_cost_weight=pos_cost_weight,
    )
    cmd14 = _rad_to_deg(ik_q)
    start_q = list(current)
    if locked_other_arm_joints is not None:
        locked = [float(value) for value in locked_other_arm_joints]
        if active_arm == "left":
            start_q[7:14] = locked
        elif active_arm == "right":
            start_q[:7] = locked
    runtime.execute_arm_motion_cb(
        _rad_to_deg(start_q),
        cmd14,
        float(move_time),
        float(settle_time),
    )

    actual, actual_quat = _measure_hand_pose(runtime, active_arm)
    pos_err = _axis_error(actual, active_pos)
    quat_err = (
        _quat_angle_error(actual_quat, active_quat)
        if active_quat is not None
        else None
    )
    runtime.loginfo_cb(
        "scene2 grasp: %s %s-hand IK actual=%s "
        "pos_err=%.4f m quat_err=%s",
        label,
        active_arm,
        [round(value, 4) for value in actual],
        pos_err,
        (
            "%.1fdeg" % math.degrees(quat_err)
            if quat_err is not None
            else "n/a"
        ),
    )
    return pos_err, quat_err, actual, actual_quat, cmd14
