#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS adapter for the guarded Scene3 6D grasp pipeline.

This module centralises all arm command publication.  It deliberately uses
the active low-level ``/joint_cmd`` target as its absolute command baseline
and uses measured joints only to calculate an IK delta.
"""

from __future__ import print_function

import statistics
import time

import numpy as np

from scene3_6d_pipeline_core import (
    ARM_COUNT,
    ARM_START,
    RIGHT_ARM_START,
    command_reference_degrees,
    command_target_from_ik_delta,
    extract_arm_vector,
    interpolate_commands,
    maximum_joint_spread_degrees,
    sensor_command_offsets_degrees,
)
from scene3_6d_pose_dry_run import (
    extract_arm_joints,
    extract_ik_solution,
    normalize_quaternion,
    position_error,
    quaternion_error_degrees,
)
from scene3_gripper_6d_align_plan import (
    GRIPPER_BASE_FRAME,
    LEFT_FINGER_FRAME,
    RIGHT_FINGER_FRAME,
    gripper_geometry,
    quaternion_to_matrix,
)


IK_MODE_POSITION_HARD_ORIENTATION_SOFT = 0x02
IK_MODE_POSITION_HARD_ORIENTATION_HARD = 0x03
REFERENCE_PARAM = (
    "/challenge_cup_task_template/scene3/arm_command_reference_deg"
)
LOCKED_ODOM_PARAM = (
    "/challenge_cup_task_template/scene3/locked_target_odom_xyz"
)

ARM_NAMES = ["arm_joint_{}".format(index) for index in range(1, 15)]


def maximum_abs(values):
    values = list(values)
    return max(abs(float(value)) for value in values) if values else 0.0


class Scene36DRosController(object):
    """Live ROS services, state sampling and one guarded command segment."""

    def __init__(self, args):
        import rospy
        import tf2_geometry_msgs  # noqa: F401 - registers PointStamped
        import tf2_ros
        from geometry_msgs.msg import PointStamped
        from kuavo_msgs.msg import (
            ikSolveParam,
            jointCmd,
            sensorsData,
            twoArmHandPoseCmd,
        )
        from kuavo_msgs.srv import (
            changeArmCtrlMode,
            changeArmCtrlModeRequest,
            fkSrv,
            twoArmHandPoseCmdSrv,
        )
        from sensor_msgs.msg import JointState

        self.args = args
        self.rospy = rospy
        self.tf2_ros = tf2_ros
        self.PointStamped = PointStamped
        self.ikSolveParam = ikSolveParam
        self.jointCmd = jointCmd
        self.sensorsData = sensorsData
        self.twoArmHandPoseCmd = twoArmHandPoseCmd
        self.changeArmCtrlMode = changeArmCtrlMode
        self.changeArmCtrlModeRequest = changeArmCtrlModeRequest
        self.fkSrv = fkSrv
        self.twoArmHandPoseCmdSrv = twoArmHandPoseCmdSrv
        self.JointState = JointState

        rospy.init_node("scene3_6d_grasp_pipeline", anonymous=True)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.arm_publisher = rospy.Publisher(
            "/kuavo_arm_traj", JointState, queue_size=10
        )
        self._fk_proxy = None
        self._ik_proxy = None
        rospy.sleep(1.2)

    def _ensure_services(self):
        if self._fk_proxy is None:
            self.rospy.wait_for_service(
                "/ik/fk_srv", timeout=float(self.args.timeout)
            )
            self._fk_proxy = self.rospy.ServiceProxy(
                "/ik/fk_srv", self.fkSrv, persistent=True
            )
        if self._ik_proxy is None:
            self.rospy.wait_for_service(
                "/ik/two_arm_hand_pose_cmd_srv",
                timeout=float(self.args.timeout),
            )
            self._ik_proxy = self.rospy.ServiceProxy(
                "/ik/two_arm_hand_pose_cmd_srv",
                self.twoArmHandPoseCmdSrv,
                persistent=True,
            )

    def frame_xyz(self, frame_name):
        transform = self.tf_buffer.lookup_transform(
            "base_link",
            str(frame_name),
            self.rospy.Time(0),
            self.rospy.Duration(float(self.args.timeout)),
        )
        value = transform.transform.translation
        return np.array([value.x, value.y, value.z], dtype=float)

    def point_in_base(self, xyz, source_frame):
        point = self.PointStamped()
        point.header.frame_id = str(source_frame)
        point.header.stamp = self.rospy.Time(0)
        point.point.x = float(xyz[0])
        point.point.y = float(xyz[1])
        point.point.z = float(xyz[2])
        result = self.tf_buffer.transform(
            point,
            "base_link",
            self.rospy.Duration(float(self.args.timeout)),
        )
        return np.array(
            [result.point.x, result.point.y, result.point.z], dtype=float
        )

    def locked_tray_center_base(self):
        if not self.rospy.has_param(LOCKED_ODOM_PARAM):
            raise RuntimeError(
                "world-locked tray target is unavailable: {}".format(
                    LOCKED_ODOM_PARAM
                )
            )
        odom = np.asarray(
            self.rospy.get_param(LOCKED_ODOM_PARAM), dtype=float
        ).reshape(-1)
        if odom.size != 3 or not np.all(np.isfinite(odom)):
            raise RuntimeError("world-locked tray target is invalid")
        return self.point_in_base(odom, "odom"), odom

    def physical_geometry(self):
        return gripper_geometry(
            self.frame_xyz(GRIPPER_BASE_FRAME),
            self.frame_xyz(LEFT_FINGER_FRAME),
            self.frame_xyz(RIGHT_FINGER_FRAME),
            tcp_extension_m=float(self.args.tcp_extension),
        )

    def call_fk(self, arm_radians):
        self._ensure_services()
        response = self._fk_proxy(
            [float(value) for value in arm_radians]
        )
        if not getattr(response, "success", False):
            raise RuntimeError("/ik/fk_srv failed")
        return response.hand_poses

    def sample_measured_arm(self, count=None):
        count = int(count or self.args.state_samples)
        samples = []
        mapping = ""
        for _ in range(max(1, count)):
            message = self.rospy.wait_for_message(
                "/sensors_data_raw",
                self.sensorsData,
                timeout=float(self.args.timeout),
            )
            arm, mapping = extract_arm_joints(message)
            samples.append([float(value) for value in arm])
            self.rospy.sleep(0.04)
        median = np.array([
            statistics.median(sample[index] for sample in samples)
            for index in range(ARM_COUNT)
        ], dtype=float)
        return {
            "arm": median,
            "samples": np.asarray(samples, dtype=float),
            "spread_deg": maximum_joint_spread_degrees(samples),
            "mapping": mapping,
        }

    def sample_low_level(self, count=None):
        count = int(count or self.args.state_samples)
        samples = []
        last = None
        for _ in range(max(1, count)):
            last = self.rospy.wait_for_message(
                "/joint_cmd",
                self.jointCmd,
                timeout=float(self.args.timeout),
            )
            samples.append(extract_arm_vector(last.joint_q, "/joint_cmd"))
            self.rospy.sleep(0.04)
        median = np.median(np.asarray(samples, dtype=float), axis=0)
        modes = list(getattr(last, "control_modes", []))
        kp = list(getattr(last, "joint_kp", []))
        kd = list(getattr(last, "joint_kd", []))
        return {
            "arm_rad": median,
            "command_deg": command_reference_degrees(median),
            "samples": np.asarray(samples, dtype=float),
            "spread_deg": maximum_joint_spread_degrees(samples),
            "modes": [
                int(value)
                for value in modes[ARM_START:ARM_START + ARM_COUNT]
            ] if len(modes) >= ARM_START + ARM_COUNT else [],
            "kp": [
                float(value)
                for value in kp[ARM_START:ARM_START + ARM_COUNT]
            ] if len(kp) >= ARM_START + ARM_COUNT else [],
            "kd": [
                float(value)
                for value in kd[ARM_START:ARM_START + ARM_COUNT]
            ] if len(kd) >= ARM_START + ARM_COUNT else [],
        }

    def sample_state(self):
        measured = self.sample_measured_arm()
        low_level = self.sample_low_level()
        poses = self.call_fk(measured["arm"])
        geometry = self.physical_geometry()
        tray_base, tray_odom = self.locked_tray_center_base()
        right_position = np.asarray(
            poses.right_pose.pos_xyz, dtype=float
        )
        right_quaternion = normalize_quaternion(
            poses.right_pose.quat_xyzw
        )
        return {
            "measured_arm": measured["arm"],
            "measured_spread_deg": measured["spread_deg"],
            "mapping": measured["mapping"],
            "command_arm_rad": low_level["arm_rad"],
            "command_deg": low_level["command_deg"],
            "command_spread_deg": low_level["spread_deg"],
            "modes": low_level["modes"],
            "kp": low_level["kp"],
            "kd": low_level["kd"],
            "poses": poses,
            "eef_position": right_position,
            "eef_quaternion": right_quaternion,
            "eef_rotation": quaternion_to_matrix(right_quaternion),
            "geometry": geometry,
            "tray_base": tray_base,
            "tray_odom": tray_odom,
        }

    def audit_checks(self, state):
        right_modes = state["modes"][RIGHT_ARM_START:]
        right_kp = state["kp"][RIGHT_ARM_START:]
        return {
            "measured_arm_stable": (
                state["measured_spread_deg"]
                <= float(self.args.maximum_measured_drift_deg)
            ),
            "low_level_target_stable": (
                state["command_spread_deg"]
                <= float(self.args.maximum_command_drift_deg)
            ),
            "right_modes_active": (
                len(right_modes) == 7
                and all(int(value) == 2 for value in right_modes)
            ),
            "right_gains_active": (
                len(right_kp) == 7
                and all(
                    float(value) >= float(self.args.minimum_right_kp)
                    for value in right_kp
                )
            ),
            "tray_in_near_workspace": (
                float(self.args.minimum_tray_distance)
                <= float(np.linalg.norm(state["tray_base"][:2]))
                <= float(self.args.maximum_tray_distance)
            ),
        }

    def print_audit(self, state):
        offsets = sensor_command_offsets_degrees(
            state["measured_arm"], state["command_arm_rad"]
        )
        checks = self.audit_checks(state)
        print("Joint mapping: {}".format(state["mapping"]))
        print("Measured arm: {}deg".format(
            np.round(np.rad2deg(state["measured_arm"]), 3).tolist()
        ))
        print("Active /joint_cmd baseline: {}deg".format(
            np.round(state["command_deg"], 3).tolist()
        ))
        print("Fixed sensor-command offsets (diagnostic only): {}deg".format(
            np.round(offsets, 3).tolist()
        ))
        print("Measured spread: {:.3f}deg".format(
            state["measured_spread_deg"]
        ))
        print("Low-level target spread: {:.3f}deg".format(
            state["command_spread_deg"]
        ))
        print("Right modes: {}".format(state["modes"][7:]))
        print("Right kp: {}".format(state["kp"][7:]))
        print("Locked tray base_link: {}".format(
            np.round(state["tray_base"], 4).tolist()
        ))
        print("Audit checks: {}".format(checks))
        return checks

    def solve_pose(self, state, target_position, target_quaternion,
                   constraint_mode):
        self._ensure_services()
        target_position = np.asarray(target_position, dtype=float).reshape(3)
        target_quaternion = normalize_quaternion(target_quaternion)

        request = self.twoArmHandPoseCmd()
        request.hand_poses.header.frame_id = "base_link"
        request.use_custom_ik_param = True
        request.joint_angles_as_q0 = True

        ik_param = self.ikSolveParam()
        ik_param.major_optimality_tol = 1e-3
        ik_param.major_feasibility_tol = 1e-3
        ik_param.minor_feasibility_tol = 1e-3
        ik_param.major_iterations_limit = int(self.args.ik_iterations)
        ik_param.oritation_constraint_tol = 1e-3
        ik_param.pos_constraint_tol = 1e-3
        ik_param.pos_cost_weight = 0.0
        ik_param.constraint_mode = int(constraint_mode)
        request.ik_param = ik_param

        poses = state["poses"]
        measured = state["measured_arm"]
        request.hand_poses.left_pose.pos_xyz = list(
            poses.left_pose.pos_xyz
        )
        request.hand_poses.left_pose.quat_xyzw = list(
            poses.left_pose.quat_xyzw
        )
        request.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        request.hand_poses.left_pose.joint_angles = measured[:7].tolist()
        request.hand_poses.right_pose.pos_xyz = target_position.tolist()
        request.hand_poses.right_pose.quat_xyzw = list(target_quaternion)
        request.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        request.hand_poses.right_pose.joint_angles = measured[7:].tolist()

        response = self._ik_proxy(request)
        if not getattr(response, "success", False):
            return {
                "success": False,
                "reason": str(getattr(response, "error_reason", "")),
            }

        raw_solution = np.asarray(
            extract_ik_solution(response, measured), dtype=float
        )
        candidate = measured.copy()
        candidate[7:] = raw_solution[7:]
        predicted = self.call_fk(candidate)
        predicted_position = np.asarray(
            predicted.right_pose.pos_xyz, dtype=float
        )
        predicted_quaternion = normalize_quaternion(
            predicted.right_pose.quat_xyzw
        )
        target_command, delta_deg = command_target_from_ik_delta(
            state["command_deg"], measured, candidate, freeze_left=True
        )
        position_residual = position_error(
            target_position, predicted_position
        )
        orientation_residual = quaternion_error_degrees(
            target_quaternion, predicted_quaternion
        )
        hard_orientation = (
            int(constraint_mode)
            == IK_MODE_POSITION_HARD_ORIENTATION_HARD
        )
        checks = {
            "ik_success": True,
            "position_residual": (
                position_residual
                <= float(self.args.maximum_ik_position_error)
            ),
            "orientation_residual": (
                not hard_orientation
                or orientation_residual
                <= float(self.args.maximum_ik_orientation_error)
            ),
            "right_joint_delta": (
                maximum_abs(delta_deg[7:])
                <= float(self.args.maximum_joint_step_deg)
            ),
            "left_command_frozen": maximum_abs(delta_deg[:7]) <= 1e-9,
            "values_finite": bool(np.all(np.isfinite(
                np.concatenate([
                    candidate,
                    target_command,
                    predicted_position,
                    np.asarray(predicted_quaternion),
                ])
            ))),
        }
        return {
            "success": bool(all(checks.values())),
            "service_success": True,
            "reason": "" if all(checks.values()) else str(checks),
            "checks": checks,
            "candidate_arm": candidate,
            "delta_deg": delta_deg,
            "source_command_deg": state["command_deg"].copy(),
            "target_command_deg": target_command,
            "target_position": target_position,
            "target_quaternion": target_quaternion,
            "predicted_position": predicted_position,
            "predicted_quaternion": predicted_quaternion,
            "predicted_rotation": quaternion_to_matrix(
                predicted_quaternion
            ),
            "position_residual_m": position_residual,
            "orientation_residual_deg": orientation_residual,
            "constraint_mode": int(constraint_mode),
        }

    def wait_for_arm_subscriber(self):
        deadline = time.time() + float(self.args.timeout)
        while self.arm_publisher.get_num_connections() == 0:
            if time.time() >= deadline:
                raise RuntimeError("/kuavo_arm_traj has no subscriber")
            self.rospy.sleep(0.05)

    def arm_topic_is_quiet(self):
        """Return false when another node is actively commanding the arms."""

        try:
            self.rospy.wait_for_message(
                "/kuavo_arm_traj",
                self.JointState,
                timeout=float(self.args.arm_topic_quiet_seconds),
            )
            return False
        except self.rospy.ROSException:
            return True

    def enable_external_arm_mode(self):
        for service_name in (
            "/arm_traj_change_mode",
            "/humanoid_change_arm_ctrl_mode",
        ):
            try:
                self.rospy.wait_for_service(service_name, timeout=2.0)
                proxy = self.rospy.ServiceProxy(
                    service_name, self.changeArmCtrlMode
                )
                request = self.changeArmCtrlModeRequest()
                request.control_mode = 2
                response = proxy(request)
                if getattr(response, "result", False):
                    print("Arm mode 2 enabled via {}".format(service_name))
                    return True
            except Exception:
                pass
        return False

    def publish_once(self, command_degrees):
        values = np.asarray(command_degrees, dtype=float).reshape(-1)
        if values.size != ARM_COUNT or not np.all(np.isfinite(values)):
            raise ValueError("complete finite 14-joint command is required")
        message = self.JointState()
        message.header.stamp = self.rospy.Time.now()
        message.name = list(ARM_NAMES)
        message.position = values.tolist()
        self.arm_publisher.publish(message)

    def hold_command(self, command_degrees, duration):
        rate = self.rospy.Rate(float(self.args.hz))
        deadline = time.time() + float(duration)
        while not self.rospy.is_shutdown() and time.time() < deadline:
            self.publish_once(command_degrees)
            rate.sleep()

    def move_command(self, start_degrees, target_degrees, duration):
        count = max(1, int(float(duration) * float(self.args.hz)))
        rate = self.rospy.Rate(float(self.args.hz))
        for command in interpolate_commands(
                start_degrees, target_degrees, count):
            self.publish_once(command)
            rate.sleep()

    def _latest_command_degrees(self):
        return self.sample_low_level(count=3)["command_deg"]

    def rollback(self, source_command_deg):
        try:
            current = self._latest_command_degrees()
        except Exception:
            current = np.asarray(source_command_deg, dtype=float)
        print("ROLLBACK: returning to the pre-segment low-level command")
        self.move_command(
            current, source_command_deg, float(self.args.rollback_seconds)
        )
        self.hold_command(
            source_command_deg, float(self.args.settle_seconds)
        )
        self.rospy.set_param(
            REFERENCE_PARAM,
            [float(value) for value in source_command_deg],
        )

    def execute_plan(self, state, plan, post_check, label):
        """Execute one command-delta plan, validate it and roll back on fail."""

        source = np.asarray(plan["source_command_deg"], dtype=float)
        target = np.asarray(plan["target_command_deg"], dtype=float)
        if not self.arm_topic_is_quiet():
            print("{}_BLOCKED: /kuavo_arm_traj has another active publisher".format(
                label
            ))
            return False, None
        if not self.enable_external_arm_mode():
            print("{}_BLOCKED: cannot enable arm mode 2".format(label))
            return False, None
        self.wait_for_arm_subscriber()

        fresh = self.sample_state()
        source_error = maximum_abs(fresh["command_deg"] - source)
        measured_drift = maximum_abs(np.rad2deg(
            fresh["measured_arm"] - state["measured_arm"]
        ))
        pre_checks = {
            "low_level_source_fresh": (
                source_error
                <= float(self.args.maximum_source_freshness_deg)
            ),
            "measured_state_fresh": (
                measured_drift
                <= float(self.args.maximum_state_freshness_deg)
            ),
            "left_command_held": maximum_abs(
                target[:7] - source[:7]
            ) <= 1e-9,
            "right_step_bounded": maximum_abs(
                target[7:] - source[7:]
            ) <= float(self.args.maximum_joint_step_deg),
            "live_audit_ready": all(self.audit_checks(fresh).values()),
        }
        print("Pre-execution checks: {}".format(pre_checks))
        if not all(pre_checks.values()):
            print("{}_BLOCKED: state changed before execution".format(label))
            return False, None

        # Publishing the active low-level target is a zero-motion handover.
        self.hold_command(source, float(self.args.prime_seconds))
        primed = self.sample_state()
        handover_joint_motion = maximum_abs(np.rad2deg(
            primed["measured_arm"] - fresh["measured_arm"]
        ))
        handover_tcp_motion = float(np.linalg.norm(
            primed["geometry"]["tcp"] - fresh["geometry"]["tcp"]
        ))
        handover_checks = {
            "joint_motion_bounded": (
                handover_joint_motion
                <= float(self.args.maximum_prime_joint_motion_deg)
            ),
            "tcp_motion_bounded": (
                handover_tcp_motion
                <= float(self.args.maximum_prime_tcp_motion)
            ),
            "command_preserved": maximum_abs(
                primed["command_deg"] - source
            ) <= float(self.args.maximum_source_freshness_deg),
        }
        print("Zero-motion handover: joint={:.3f}deg tcp={:.1f}mm checks={}".format(
            handover_joint_motion,
            handover_tcp_motion * 1000.0,
            handover_checks,
        ))
        if not all(handover_checks.values()):
            print("{}_BLOCKED: zero-motion handover moved the arm".format(label))
            return False, None

        moved = False
        try:
            print("Executing {} with one quintic command-delta segment".format(
                label
            ))
            moved = True
            self.move_command(source, target, float(self.args.motion_seconds))
            self.hold_command(target, float(self.args.settle_seconds))
            after = self.sample_state()
            post_checks, details = post_check(primed, after)
            print("Post-execution checks: {}".format(post_checks))
            if not all(post_checks.values()):
                raise RuntimeError("physical 6D post-motion gate failed")
            self.rospy.set_param(
                REFERENCE_PARAM, [float(value) for value in target]
            )
            print("{}_COMMITTED".format(label))
            return True, after
        except Exception as exc:
            print("{} failed: {}".format(label, exc))
            if moved:
                self.rollback(source)
            print("{}_BLOCKED: rolled back".format(label))
            return False, None
