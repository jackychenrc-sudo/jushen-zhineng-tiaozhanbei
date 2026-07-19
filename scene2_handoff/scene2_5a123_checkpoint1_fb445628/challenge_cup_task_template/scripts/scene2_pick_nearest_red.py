#!/usr/bin/env python3
"""Pick the nearest red-handled part in the current Scene2 layout."""

import argparse
import glob
import math
import os
import queue
import threading
import time

import rospy
import tf

from kuavo_msgs.msg import armTargetPoses, sensorsData

import scene2_vision_locator as vision_locator
import scene2_robot_helpers as pipeline
import scene2_kinematics as grasp_ik
from scene2_kinematics import GraspRuntime


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
MAX_RESTRICTED_LIFT_M = 0.40
RESTRICTED_SEGMENT_SETTLE_SECONDS = 1.0
RESTRICTED_MIN_SEGMENT_DURATION_S = 2.5
RESTRICTED_MIN_PROGRESS_RATIO = 0.20
RESTRICTED_MAX_JOINT_STEP_DEG = 8.0
RESTRICTED_ORIENTATION_TOLERANCE_DEG = 20.0
FROZEN_VISION_MAX_SPREAD_M = 0.003
RESTRICTED_APPROACH_COMMAND_Z_BIAS_M = 0.024
LOW_APPROACH_IK_SEEDS_RAD = (
    (0.33, 0.42, 0.05, -1.65),
    (0.40, 0.44, 0.08, -1.85),
    (0.25, 0.36, 0.00, -1.45),
)
MAX_CARTESIAN_STEP_M = 0.02
BODY_SIDE_STEP_M = 0.005
OUTSIDE_LIFT_STEP_M = 0.01
GRASP_ORIENTATION_BLEND = 0.0
# All entries below are distances or dimensionless fractions.  The table's
# base-link XYZ is measured from the green RGB-D plane at runtime.
TABLE_OUTSIDE_CLEARANCE_M = 0.13
TABLE_SIDE_FRACTION = 0.50
TABLE_BODY_SIDE_BELOW_M = 0.087
TABLE_MIN_TRANSIT_CLEARANCE_M = 0.188
TABLE_COMMAND_TRANSIT_CLEARANCE_M = 0.203
TABLE_CARRY_CLEARANCE_M = 0.193
TABLE_MIN_CARRY_CLEARANCE_M = 0.173
TABLE_PLACE_MIN_CLEARANCE_M = 0.163
TABLE_RETREAT_CLEARANCE_M = 0.233
TABLE_RETURN_CLEARANCE_M = 0.243
TABLE_GREEN_HSV_LOWER = (35, 75, 25)
TABLE_GREEN_HSV_UPPER = (90, 255, 190)
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
PURPLE_BIN_INSET_X_M = 0.060
PURPLE_BIN_INSET_Y_M = 0.055
DEFAULT_GRASP_CENTER_CLEARANCE_M = 0.000
DEFAULT_BIN_RELEASE_CLEARANCE_M = 0.100
SAFE_CARRY_COMMAND_Z_M = 0.16
MAX_CARRY_EE_DROP_M = 0.003
MAX_CARRY_SLIP_M = 0.003
MAX_STATE_RETRIES = 2
_PINCH_TF_LISTENER = None
# TF lookup can permanently hold the Python GIL after a MuJoCo time rewind.
# Once a real pinch sample has been paired with FK, subsequent control-state
# feedback uses the live joint angles plus this calibrated offset instead of
# issuing another blocking TF lookup.
_PINCH_MODEL_OFFSET = None
_PINCH_MODEL_POINT = None
GRASP_CLAW_VELOCITY = 20.0
GRASP_CLAW_EFFORT = 2.0
TRANSFER_TIME_SCALE = 2.5
# Move the pinch point inside the red handle toward the adjoining metal shaft.
# The side is inferred from the RGB image; no object pose or fixed world
# coordinate is used.
HEADWARD_GRASP_FRACTION = 0.22
HEADWARD_GRASP_MIN_PX = 8.0
HEADWARD_GRASP_MAX_PX = 15.0
SHAFT_SCORE_MIN = 0.08
SHAFT_SCORE_MARGIN_MIN = 0.04


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


def control_scene2_claws(
    left_position,
    right_position=0.0,
    timeout=10.0,
    velocity=50.0,
    effort=1.0,
):
    """Use Scene2's verified claw service rather than the generic command topic."""
    from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest

    service_name = "/control_robot_leju_claw"
    rospy.wait_for_service(service_name, timeout=float(timeout))
    request = controlLejuClawRequest()
    request.data.name = ["left_claw", "right_claw"]
    request.data.position = [float(left_position), float(right_position)]
    request.data.velocity = [float(velocity), float(velocity)]
    request.data.effort = [float(effort), float(effort)]
    response = rospy.ServiceProxy(service_name, controlLejuClaw)(request)
    if not response.success:
        raise RuntimeError("%s rejected claw command: %s" % (service_name, response.message))
    rospy.loginfo(
        "Scene2 claw service left=%.1f right=%.1f velocity=%.1f effort=%.1f",
        left_position,
        right_position,
        velocity,
        effort,
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
    parser.add_argument("--restricted-single-loop-execute", action="store_true")
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
    parser.add_argument(
        "--grasp-center-clearance",
        type=float,
        default=DEFAULT_GRASP_CENTER_CLEARANCE_M,
    )
    parser.add_argument(
        "--bin-release-clearance",
        type=float,
        default=DEFAULT_BIN_RELEASE_CLEARANCE_M,
    )
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


def read_left_pinch_position(timeout=1.0, include_timestamps=False):
    """Return physical pinch feedback without allowing a TF lock to block control."""
    global _PINCH_MODEL_OFFSET, _PINCH_MODEL_POINT
    if _PINCH_MODEL_OFFSET is not None:
        if _PINCH_MODEL_POINT is None:
            raise RuntimeError("calibrated pinch feedback cache is empty")
        point = list(_PINCH_MODEL_POINT)
        timestamp = float(rospy.Time.now().to_sec())
        return (point, [timestamp, timestamp]) if include_timestamps else point

    frames = (
        "left_gripper_left_inner_finger",
        "left_gripper_right_inner_finger",
    )
    result_queue = queue.Queue(maxsize=1)

    def read_once():
        try:
            # A fresh listener avoids the permanently blocked cache observed
            # after MuJoCo rewinds simulated time during a trajectory step.
            listener = tf.TransformListener()
            time.sleep(0.15)
            points = []
            for frame in frames:
                translation, rotation = listener.lookupTransform(
                    "base_link",
                    frame,
                    rospy.Time(0),
                )
                # getLatestCommonTime can block indefinitely after the
                # simulator rewinds its clock.  lookupTransform(Time(0))
                # already confirms a current transform; record its sampling
                # time instead so the caller can still detect fresh feedback.
                points.append(
                    _transform_local_point(
                        translation,
                        rotation,
                        PINCH_CONTACT_LOCAL_M,
                    )
                )
            point = [
                0.5 * (points[0][axis] + points[1][axis])
                for axis in range(3)
            ]
            result_queue.put((True, point, float(rospy.Time.now().to_sec())))
        except Exception as error:
            result_queue.put((False, error, None))

    worker = threading.Thread(target=read_once)
    worker.daemon = True
    worker.start()
    try:
        success, payload, timestamp = result_queue.get(
            timeout=max(0.20, float(timeout))
        )
    except queue.Empty:
        raise RuntimeError("physical pinch TF query exceeded %.2fs" % timeout)
    if not success:
        raise payload
    timestamps = [timestamp, timestamp]
    return (payload, timestamps) if include_timestamps else payload


def read_left_claw_feedback(timeout=2.0):
    """Read the simulated left-claw state without commanding it."""
    from kuavo_msgs.msg import lejuClawState

    message = rospy.wait_for_message(
        "/leju_claw_state",
        lejuClawState,
        timeout=float(timeout),
    )
    names = list(message.data.name)
    left_index = names.index("left_claw") if "left_claw" in names else 0
    states = list(message.state)
    positions = list(message.data.position)
    return {
        "state": int(states[left_index]) if left_index < len(states) else None,
        "position": (
            float(positions[left_index])
            if left_index < len(positions)
            else None
        ),
    }


def read_visible_red_position(expected_xyz=None):
    """Return the visually closest valid red region, or None if it is occluded."""
    try:
        result = _locate_red_candidates()
    except Exception as error:
        rospy.logwarn("red object feedback unavailable: %s", error)
        return None

    points = []
    for candidate in result.get("candidates", []):
        if candidate.get("valid_3d"):
            points.append([float(value) for value in candidate["base_xyz_m"]])
    if not points:
        rospy.logwarn("red object feedback unavailable: no valid RGB-D candidate")
        return None
    if expected_xyz is None:
        return min(points, key=lambda point: math.sqrt(sum(value * value for value in point)))
    return min(
        points,
        key=lambda point: math.sqrt(
            sum((point[index] - float(expected_xyz[index])) ** 2 for index in range(3))
        ),
    )


class Scene2StateMachine:
    """Small feedback-driven state recorder for the single-part closed loop."""

    STATES = (
        "SEARCH",
        "APPROACH_ABOVE",
        "DESCEND",
        "GRASP",
        "VERIFY_GRASP",
        "LIFT",
        "TRANSFER",
        "ABOVE_TARGET",
        "DESCEND_TARGET",
        "RELEASE",
        "RETREAT",
        "VERIFY_SUCCESS",
    )

    def __init__(self):
        self.state = None
        self.retries = {}
        self.freeze_target_vision = False

    def enter(self, state, current=None, expected_red=None, target=None):
        if state not in self.STATES:
            raise ValueError("unknown Scene2 state: %s" % state)
        self.state = state
        self.retries.setdefault(state, 0)
        return self.snapshot("enter", current, expected_red, target)

    def retry(self, reason):
        count = self.retries.get(self.state, 0) + 1
        self.retries[self.state] = count
        if count > MAX_STATE_RETRIES:
            raise RuntimeError(
                "%s exceeded %d retries: %s"
                % (self.state, MAX_STATE_RETRIES, reason)
            )
        rospy.logwarn("SCENE2_STATE retry state=%s count=%d reason=%s", self.state, count, reason)

    def snapshot(self, phase, current=None, expected_red=None, target=None):
        if current is None:
            current = list(pipeline._read_current_arm_joints(5.0))
        ee_pose = grasp_ik._call_fk(current, 5.0).left_pose
        pinch = read_left_pinch_position(timeout=2.0)
        try:
            claw = read_left_claw_feedback(timeout=2.0)
        except Exception as error:
            claw = {"state": None, "position": None}
            rospy.logwarn("left claw feedback unavailable: %s", error)
        red = None
        if not (
            self.freeze_target_vision
            and self.state in ("APPROACH_ABOVE", "DESCEND", "GRASP")
        ):
            red = read_visible_red_position(expected_red)
        snapshot = {
            "ee": [float(value) for value in ee_pose.pos_xyz],
            "pinch": [float(value) for value in pinch],
            "red": red,
            "claw": claw,
        }
        rospy.loginfo(
            "SCENE2_STATE state=%s phase=%s ee=%s pinch=%s red=%s claw=%s target=%s",
            self.state,
            phase,
            [round(value, 4) for value in snapshot["ee"]],
            [round(value, 4) for value in snapshot["pinch"]],
            None if red is None else [round(value, 4) for value in red],
            claw,
            None if target is None else [round(float(value), 4) for value in target],
        )
        return snapshot


class SensorTimeStageLogger:
    """Log sparse competition timing from sensors_data_raw.sensor_time only."""

    def __init__(self, timeout=5.0):
        self.timeout = float(timeout)
        self.logged_stages = set()
        self._condition = threading.Condition()
        self._latest_sensor_time_ns = None
        self._subscriber = rospy.Subscriber(
            "/sensors_data_raw",
            sensorsData,
            self._sensor_callback,
            queue_size=1,
        )
        self.run_start_sensor_time_ns = self._read_sensor_time_ns()
        self._log("RUN_START", self.run_start_sensor_time_ns)

    def _sensor_callback(self, message):
        sensor_time_ns = (
            int(message.sensor_time.secs) * 1000000000
            + int(message.sensor_time.nsecs)
        )
        with self._condition:
            self._latest_sensor_time_ns = sensor_time_ns
            self._condition.notify_all()

    def _read_sensor_time_ns(self):
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while self._latest_sensor_time_ns is None and not rospy.is_shutdown():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("timed out waiting for sensors_data_raw sensor_time")
                self._condition.wait(timeout=min(0.1, remaining))
            if self._latest_sensor_time_ns is None:
                raise RuntimeError("sensor_time unavailable during ROS shutdown")
            return int(self._latest_sensor_time_ns)

    def _log(self, stage, raw_sensor_time_ns):
        raw_seconds = float(raw_sensor_time_ns) / 1000000000.0
        elapsed_seconds = float(
            raw_sensor_time_ns - self.run_start_sensor_time_ns
        ) / 1000000000.0
        rospy.loginfo(
            "SCENE2_SENSOR_TIME stage=%s raw_sensor_time=%.9f elapsed_sensor_time=%.3f",
            stage,
            raw_seconds,
            elapsed_seconds,
        )

    def log_stage(self, stage):
        if stage in self.logged_stages:
            return
        raw_sensor_time_ns = self._read_sensor_time_ns()
        self.logged_stages.add(stage)
        self._log(stage, raw_sensor_time_ns)


class CarryFeedbackMonitor:
    """Detect arm height loss separately from a part slipping in the gripper."""

    def __init__(self, state_machine, initial_snapshot):
        self.state_machine = state_machine
        self.initial = initial_snapshot
        self.last = initial_snapshot
        self.safe_ee_z = float(initial_snapshot["ee"][2]) - MAX_CARRY_EE_DROP_M

    def observe(self, label, current, enforce_height):
        expected_red = None
        if self.last["red"] is not None:
            pinch_delta = [
                float(value) - float(self.last["pinch"][axis])
                for axis, value in enumerate(read_left_pinch_position(timeout=2.0))
            ]
            expected_red = [
                float(self.last["red"][axis]) + pinch_delta[axis]
                for axis in range(3)
            ]
        snapshot = self.state_machine.snapshot(
            label,
            current,
            expected_red=expected_red,
        )
        if enforce_height and snapshot["ee"][2] < self.last["ee"][2] - MAX_CARRY_EE_DROP_M:
            return snapshot, True
        if self.last["red"] is not None and snapshot["red"] is not None:
            previous_relative_z = self.last["red"][2] - self.last["pinch"][2]
            current_relative_z = snapshot["red"][2] - snapshot["pinch"][2]
            if current_relative_z < previous_relative_z - MAX_CARRY_SLIP_M:
                raise RuntimeError(
                    "%s detected gripper slip: relative_z %.4fm -> %.4fm"
                    % (label, previous_relative_z, current_relative_z)
                )
        self.last = snapshot
        return snapshot, False

    def verify_lift(self):
        if self.initial["red"] is None or self.last["red"] is None:
            raise RuntimeError("VERIFY_GRASP cannot see the red object after lift")
        red_lift = self.last["red"][2] - self.initial["red"][2]
        pinch_lift = self.last["pinch"][2] - self.initial["pinch"][2]
        if red_lift < 0.025 or red_lift < pinch_lift - MAX_CARRY_SLIP_M:
            raise RuntimeError(
                "VERIFY_GRASP failed: red_lift=%.4fm pinch_lift=%.4fm"
                % (red_lift, pinch_lift)
            )


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
    debugger = vision_locator.Scene2VisionDebug(vision_args)
    result = debugger.run_sequential()
    _add_headward_grasp_points(debugger, result, vision_args.output_dir)
    return result


def locate_green_table_geometry():
    """Measure the green tabletop plane and derive all path clearances."""
    cv2 = vision_locator.cv2
    np = vision_locator.np
    CameraInfo = vision_locator.CameraInfo
    CompressedImage = vision_locator.CompressedImage

    camera_info = rospy.wait_for_message(
        vision_locator.CAMERA_INFO_TOPIC,
        CameraInfo,
        timeout=8.0,
    )
    rgb_message = rospy.wait_for_message(
        vision_locator.RGB_TOPIC,
        CompressedImage,
        timeout=15.0,
    )
    depth_message = rospy.wait_for_message(
        vision_locator.DEPTH_TOPIC,
        CompressedImage,
        timeout=15.0,
    )
    if rgb_message.header.frame_id != depth_message.header.frame_id:
        raise RuntimeError("table RGB/depth frames do not match")
    image_bgr = vision_locator.decode_rgb(rgb_message)
    depth_m = vision_locator.decode_compressed_depth(depth_message)
    if image_bgr.shape[:2] != depth_m.shape[:2]:
        raise RuntimeError("table RGB/depth images are not aligned")

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        image_hsv,
        np.array(TABLE_GREEN_HSV_LOWER, dtype=np.uint8),
        np.array(TABLE_GREEN_HSV_UPPER, dtype=np.uint8),
    )
    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )

    # Subsample the RGB-D image before the vectorized transform.  The green
    # floor and tabletop can share hue, so color selects candidate pixels and
    # the dominant horizontal depth planes separate them.
    stride = 6
    sampled_mask = green_mask[::stride, ::stride]
    sampled_depth = depth_m[::stride, ::stride]
    valid = (
        (sampled_mask > 0)
        & np.isfinite(sampled_depth)
        & (sampled_depth > 0.10)
        & (sampled_depth < 3.00)
    )
    sampled_v, sampled_u = np.nonzero(valid)
    if sampled_u.size < 1000:
        raise RuntimeError(
            "not enough green RGB-D pixels for tabletop localization: %d"
            % sampled_u.size
        )
    pixel_u = sampled_u.astype(np.float64) * stride
    pixel_v = sampled_v.astype(np.float64) * stride
    point_depth = sampled_depth[sampled_v, sampled_u].astype(np.float64)
    fx = float(camera_info.K[0])
    fy = float(camera_info.K[4])
    cx = float(camera_info.K[2])
    cy = float(camera_info.K[5])
    camera_points = np.stack(
        (
            (pixel_u - cx) * point_depth / fx,
            (pixel_v - cy) * point_depth / fy,
            point_depth,
            np.ones_like(point_depth),
        ),
        axis=0,
    )

    tf_buffer = vision_locator.tf2_ros.Buffer(
        cache_time=rospy.Duration(10.0)
    )
    tf_listener = vision_locator.tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.5)
    transform = tf_buffer.lookup_transform(
        vision_locator.BASE_FRAME,
        rgb_message.header.frame_id,
        rospy.Time(0),
        rospy.Duration(2.0),
    )
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    transform_matrix = tf.transformations.quaternion_matrix(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    )
    transform_matrix[:3, 3] = [
        translation.x,
        translation.y,
        translation.z,
    ]
    base_points = np.matmul(transform_matrix, camera_points)[:3].T

    z_values = base_points[:, 2]
    bin_start = math.floor(float(np.min(z_values)) * 100.0) / 100.0
    bin_stop = math.ceil(float(np.max(z_values)) * 100.0) / 100.0 + 0.02
    bin_edges = np.arange(bin_start, bin_stop, 0.01)
    histogram, bin_edges = np.histogram(z_values, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    significant = np.flatnonzero(
        histogram >= max(80, int(0.12 * int(np.max(histogram))))
    )
    if significant.size == 0:
        raise RuntimeError("green tabletop plane was not significant")
    # The tabletop is the highest large green horizontal plane; the similarly
    # colored floor is a separate, much lower base-link Z cluster.
    table_bin_z = float(np.max(bin_centers[significant]))
    table_points = base_points[
        np.abs(base_points[:, 2] - table_bin_z) <= 0.012
    ]
    if table_points.shape[0] < 800:
        raise RuntimeError("green tabletop plane has too few 3-D samples")

    surface_z = float(np.median(table_points[:, 2]))
    z_mad = float(
        np.median(np.abs(table_points[:, 2] - surface_z))
    )
    front_x = float(np.percentile(table_points[:, 0], 0.5))
    back_x = float(np.percentile(table_points[:, 0], 99.5))
    right_y = float(np.percentile(table_points[:, 1], 0.5))
    center_y = float(np.median(table_points[:, 1]))
    left_y = float(np.percentile(table_points[:, 1], 99.5))
    if back_x - front_x < 0.35 or left_y - right_y < 0.80 or z_mad > 0.015:
        raise RuntimeError(
            "green tabletop geometry rejected: x_span=%.3f y_span=%.3f z_mad=%.4f"
            % (back_x - front_x, left_y - right_y, z_mad)
        )

    geometry = {
        "surface_z": surface_z,
        "front_x": front_x,
        "back_x": back_x,
        "right_y": right_y,
        "center_y": center_y,
        "left_y": left_y,
        "outside_x": front_x - TABLE_OUTSIDE_CLEARANCE_M,
        "side_y_left": center_y
        + TABLE_SIDE_FRACTION * (left_y - center_y),
        "side_y_right": center_y
        - TABLE_SIDE_FRACTION * (center_y - right_y),
        "body_side_lift_z": surface_z - TABLE_BODY_SIDE_BELOW_M,
        "minimum_transit_z": surface_z
        + TABLE_MIN_TRANSIT_CLEARANCE_M,
        "command_transit_z": surface_z
        + TABLE_COMMAND_TRANSIT_CLEARANCE_M,
        "carry_z": surface_z + TABLE_CARRY_CLEARANCE_M,
        "minimum_carry_z": surface_z
        + TABLE_MIN_CARRY_CLEARANCE_M,
        "place_minimum_z": surface_z
        + TABLE_PLACE_MIN_CLEARANCE_M,
        "retreat_z": surface_z + TABLE_RETREAT_CLEARANCE_M,
        "return_z": surface_z + TABLE_RETURN_CLEARANCE_M,
    }
    rospy.loginfo(
        "TABLE_COLOR samples=%d surface_z=%.4f front_x=%.4f "
        "y=[%.4f, %.4f] outside_x=%.4f side_y_left=%.4f "
        "transit_z=%.4f",
        int(table_points.shape[0]),
        surface_z,
        front_x,
        right_y,
        left_y,
        geometry["outside_x"],
        geometry["side_y_left"],
        geometry["command_transit_z"],
    )
    return geometry


def _shaft_side_score(image_hsv, center, axis, sign, long_side_px):
    """Score low-saturation bright pixels just beyond one red-handle end."""
    np = vision_locator.np
    height, width = image_hsv.shape[:2]
    perpendicular = np.array([-axis[1], axis[0]], dtype=np.float64)
    bright_neutral = 0
    sample_count = 0
    for along in np.linspace(0.42 * long_side_px, 1.20 * long_side_px, 40):
        for across in np.linspace(-0.10 * long_side_px, 0.10 * long_side_px, 9):
            pixel = np.rint(
                np.array(center, dtype=np.float64)
                + float(sign) * axis * along
                + perpendicular * across
            ).astype(int)
            u, v = int(pixel[0]), int(pixel[1])
            if not (0 <= u < width and 0 <= v < height):
                continue
            _hue, saturation, value = image_hsv[v, u]
            sample_count += 1
            if int(saturation) < 80 and int(value) > 90:
                bright_neutral += 1
    return (
        float(bright_neutral) / float(sample_count)
        if sample_count
        else 0.0
    )


def _add_headward_grasp_points(debugger, result, output_dir):
    """Infer the metal-shaft end and deproject an in-handle grasp point."""
    cv2 = vision_locator.cv2
    np = vision_locator.np
    raw_paths = glob.glob(os.path.join(output_dir, "*_rgb_raw.jpg"))
    if not raw_paths:
        raise RuntimeError("red RGB frame was not saved for shaft-side detection")
    raw_path = max(raw_paths, key=os.path.getmtime)
    image_bgr = cv2.imread(raw_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("failed to read red RGB frame: %s" % raw_path)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    for candidate in result.get("candidates", []):
        if not candidate.get("valid_3d"):
            continue
        center = [float(value) for value in candidate["pixel_uv"]]
        angle_rad = math.radians(float(candidate["image_angle_deg"]))
        axis = np.array(
            [math.cos(angle_rad), math.sin(angle_rad)],
            dtype=np.float64,
        )
        long_side_px = float(candidate["long_side_px"])
        negative_score = _shaft_side_score(
            image_hsv, center, axis, -1.0, long_side_px
        )
        positive_score = _shaft_side_score(
            image_hsv, center, axis, 1.0, long_side_px
        )
        shaft_sign = 1.0 if positive_score >= negative_score else -1.0
        best_score = max(negative_score, positive_score)
        score_margin = abs(positive_score - negative_score)
        if best_score < SHAFT_SCORE_MIN or score_margin < SHAFT_SCORE_MARGIN_MIN:
            raise RuntimeError(
                "cannot identify screwdriver shaft side for candidate #%d: "
                "scores(-/+)=(%.3f, %.3f)"
                % (
                    int(candidate["index"]),
                    negative_score,
                    positive_score,
                )
            )

        shift_px = min(
            HEADWARD_GRASP_MAX_PX,
            max(HEADWARD_GRASP_MIN_PX, HEADWARD_GRASP_FRACTION * long_side_px),
        )
        grasp_pixel = np.rint(
            np.array(center, dtype=np.float64)
            + shaft_sign * axis * shift_px
        ).astype(int)
        point_camera = vision_locator.deproject_pixel(
            (int(grasp_pixel[0]), int(grasp_pixel[1])),
            float(candidate["depth_m"]),
            debugger.camera_info,
        )
        point_base = vision_locator.transform_point(
            debugger.tf_buffer,
            point_camera,
            result["camera_frame"],
            rospy.Time(0),
        )
        candidate["headward_pixel_uv"] = [
            int(grasp_pixel[0]),
            int(grasp_pixel[1]),
        ]
        candidate["headward_base_xyz_m"] = [
            float(value) for value in point_base
        ]
        candidate["shaft_side_scores"] = [negative_score, positive_score]
        rospy.loginfo(
            "HEADWARD_GRASP candidate=%d center_uv=%s grasp_uv=%s "
            "shaft_scores(-/+)=%.3f/%.3f base=%s",
            int(candidate["index"]),
            [int(round(value)) for value in center],
            candidate["headward_pixel_uv"],
            negative_score,
            positive_score,
            [round(float(value), 4) for value in point_base],
        )


def locate_purple_bin():
    vision_args = argparse.Namespace(
        color="purple",
        output_dir="/tmp/scene2_purple_test",
        min_area=300.0,
        max_area=500000.0,
        depth_radius=12,
        candidate_index=0,
        sync_slop=0.10,
        roi=None,
        continuous=True,
        sequential=True,
    )
    result = vision_locator.Scene2VisionDebug(vision_args).run_sequential()
    point = [float(value) for value in result["base_xyz_m"]]
    area = float(result["contour_area_px"])
    aspect = float(result["aspect_ratio"])
    if not (
        area >= 5000.0
        and 0.70 <= aspect <= 1.60
        and 0.45 <= point[0] <= 0.75
        and 0.30 <= point[1] <= 0.65
        and -0.16 <= point[2] <= 0.02
    ):
        raise RuntimeError(
            "purple bin failed visual safety filters: area=%.1f aspect=%.2f base=%s"
            % (area, aspect, [round(value, 4) for value in point])
        )
    release = [
        point[0] - PURPLE_BIN_INSET_X_M,
        point[1] - PURPLE_BIN_INSET_Y_M,
        point[2] + DEFAULT_BIN_RELEASE_CLEARANCE_M,
    ]
    rospy.loginfo(
        "scene2 OpenCV purple bin center=%s release_xy=%s",
        [round(value, 4) for value in point],
        [round(value, 4) for value in release[:2]],
    )
    return {"vision_base": point, "release": release, "vision": result}


def verify_red_in_purple_bin(purple_bin):
    result = _locate_red_candidates()
    center = purple_bin["vision_base"]
    matches = []
    for candidate in result.get("candidates", []):
        if not candidate.get("valid_3d"):
            continue
        point = [float(value) for value in candidate["base_xyz_m"]]
        horizontal_distance = math.hypot(
            point[0] - center[0],
            point[1] - center[1],
        )
        if horizontal_distance <= 0.12:
            matches.append((horizontal_distance, point))
    if not matches:
        rospy.logwarn("no red region was verified inside the purple-bin area")
        return False
    distance, point = min(matches)
    rospy.loginfo(
        "single-loop visual verification PASS red=%s purple=%s xy_distance=%.3fm",
        [round(value, 4) for value in point],
        [round(value, 4) for value in center],
        distance,
    )
    return True


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

        grasp_base = [
            float(value) for value in item["headward_base_xyz_m"]
        ]
        offset = _vision_grasp_offset(grasp_base)
        grasp_ee = [
            grasp_base[index] + offset[index]
            for index in range(3)
        ]
        # "Nearest" means nearest to the robot base, not nearest to a
        # hand-picked work point on the left side of the table.
        distance = math.sqrt(sum(value * value for value in point_base))
        candidates.append(
            (distance, point_base, grasp_base, item, grasp_ee, offset)
        )

    if not candidates:
        raise RuntimeError("no red screwdriver candidate passed 3-D safety filters")

    if requested_object != "auto":
        rospy.logwarn(
            "%s cannot be distinguished visually from the other red "
            "screwdriver; selecting the nearest valid candidate instead",
            requested_object,
        )
    (
        _distance,
        point_base,
        grasp_base,
        selected_result,
        grasp_ee,
        offset,
    ) = min(candidates)

    # Both red instances are the same task class and share the purple bin.
    # This name is only an action-template key; it is not inferred identity.
    selected_name = "part_type_c_1"
    grasp_world = [
        grasp_ee[0] - pipeline.WORLD_TO_EE_OFFSET_X,
        grasp_ee[1] - pipeline.WORLD_TO_EE_OFFSET_Y_LEFT,
        grasp_ee[2] - pipeline.WORLD_TO_EE_OFFSET_Z,
    ]

    # Do not build a simulator data-collection job here: that helper resolves
    # every object's configured world coordinate before this visually measured
    # target can replace it.  Keep the action metadata local and derive every
    # position used by this pick from RGB-D/TF feedback instead.
    job = {
        "object": selected_name,
        "bin": "sorting_bin_c",
        "arm": "left",
        "world_xyz": list(grasp_world),
    }
    job["vision_base"] = list(point_base)
    job["vision_grasp_base"] = list(grasp_base)
    job["grasp"] = list(grasp_ee)
    job["vision_candidate_index"] = int(selected_result["index"])
    job["vision_image_angle_deg"] = float(
        selected_result["image_angle_deg"]
    )
    job["vision_rgb_chroma_distance"] = selected_result.get(
        "rgb_chroma_distance"
    )
    rospy.loginfo(
        "scene2 OpenCV red candidate #%d pixel=%s base=%s "
        "headward_base=%s grasp=%s offset=%s image_angle=%.1fdeg "
        "rgb_distance=%s",
        int(selected_result["index"]),
        selected_result["pixel_uv"],
        [round(value, 4) for value in point_base],
        [round(value, 4) for value in grasp_base],
        [round(value, 4) for value in grasp_ee],
        [round(value, 4) for value in offset],
        float(selected_result["image_angle_deg"]),
        selected_result.get("rgb_chroma_distance"),
    )
    return job


def build_frozen_vision_red_job(requested_object, samples=3, attempts=3):
    """Freeze a red-object target only after three mutually consistent RGB-D reads."""
    samples = int(samples)
    for attempt in range(1, int(attempts) + 1):
        jobs = []
        for _index in range(samples):
            jobs.append(build_vision_red_job(requested_object))
            rospy.sleep(0.15)
        points = [list(job["vision_base"]) for job in jobs]
        grasp_points = [list(job["vision_grasp_base"]) for job in jobs]
        center = [
            sorted(point[axis] for point in points)[len(points) // 2]
            for axis in range(3)
        ]
        grasp_center = [
            sorted(point[axis] for point in grasp_points)[len(grasp_points) // 2]
            for axis in range(3)
        ]
        spread = max(
            math.sqrt(
                sum(
                    (float(point[axis]) - float(center[axis])) ** 2
                    for axis in range(3)
                )
            )
            for point in points
        )
        rospy.loginfo(
            "FROZEN_TARGET sample_attempt=%d samples=%s center=%s spread=%.4fm",
            attempt,
            [[round(value, 4) for value in point] for point in points],
            [round(value, 4) for value in center],
            spread,
        )
        if spread <= FROZEN_VISION_MAX_SPREAD_M:
            job = jobs[-1]
            offset = _vision_grasp_offset(grasp_center)
            grasp_ee = [
                grasp_center[axis] + offset[axis]
                for axis in range(3)
            ]
            job["vision_base"] = list(center)
            job["vision_grasp_base"] = list(grasp_center)
            job["grasp"] = list(grasp_ee)
            job["world_xyz"] = [
                grasp_ee[0] - pipeline.WORLD_TO_EE_OFFSET_X,
                grasp_ee[1] - pipeline.WORLD_TO_EE_OFFSET_Y_LEFT,
                grasp_ee[2] - pipeline.WORLD_TO_EE_OFFSET_Z,
            ]
            job["frozen_vision_base"] = list(center)
            job["vision_frozen"] = True
            job["pregrasp_pose"] = [
                grasp_ee[0], grasp_ee[1], grasp_ee[2] + 0.15,
            ]
            job["grasp_pose"] = list(grasp_ee)
            # The lift Z is filled from the measured green-table surface
            # immediately before motion planning.
            job["lift_pose"] = list(grasp_ee)
            rospy.loginfo(
                "FROZEN_TARGET accepted base=%s pregrasp=%s grasp=%s lift=%s",
                [round(value, 4) for value in center],
                [round(value, 4) for value in job["pregrasp_pose"]],
                [round(value, 4) for value in job["grasp_pose"]],
                [round(value, 4) for value in job["lift_pose"]],
            )
            return job
        rospy.logwarn(
            "FROZEN_TARGET rejected sample set spread=%.4fm limit=%.4fm",
            spread,
            FROZEN_VISION_MAX_SPREAD_M,
        )
    raise RuntimeError(
        "red target did not stabilize: three-sample spread exceeds %.4fm"
        % FROZEN_VISION_MAX_SPREAD_M
    )


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
        transfer_safe = transfer_safe and minimum_z >= float(
            job["table_geometry"]["place_minimum_z"]
        )
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
    use_wrist=False,
    fixed_joint5=None,
    orientation_quat=None,
):
    import numpy as np
    from scipy.optimize import least_squares

    fixed = np.asarray(start_joints, dtype=float)
    if fixed_joint5 is not None:
        fixed[4] = float(fixed_joint5)
    # Keep joint 5 fixed.  The proven high route uses the first four joints;
    # only the near-object and bin phases add the last two wrist joints so the
    # shoulder does not run into its limit during the final descent.
    active_indices = np.asarray(
        [0, 1, 2, 3, 5, 6] if use_wrist else [0, 1, 2, 3],
        dtype=int,
    )
    reference = fixed[active_indices].copy()
    if use_wrist:
        joint_lower = np.radians(
            np.asarray([-180.0, -20.0, -90.0, -150.0, -90.0, -40.0])
        )
        joint_upper = np.radians(
            np.asarray([90.0, 200.0, 90.0, 0.0, 90.0, 40.0])
        )
        local_span = math.radians(12.0)
    else:
        joint_lower = np.asarray(
            [-math.pi, -0.349066, -1.48353, -2.61799]
        )
        joint_upper = np.asarray([1.5708, 3.49066, 1.48353, 0.0])
        local_span = math.radians(20.0)
    lower = np.maximum(joint_lower, reference - local_span)
    upper = np.minimum(joint_upper, reference + local_span)
    target = np.asarray(target_xyz, dtype=float)
    target_orientation = None
    if orientation_quat is not None:
        target_orientation = np.asarray(orientation_quat, dtype=float)
        target_orientation /= max(np.linalg.norm(target_orientation), 1.0e-9)

    def residual(active):
        candidate = fixed.copy()
        candidate[active_indices] = active
        pose = grasp_ik._call_fk(candidate.tolist(), runtime.timeout).left_pose
        position_error = np.asarray(pose.pos_xyz, dtype=float) - target
        posture_cost = active - reference
        terms = [20.0 * position_error]
        if target_orientation is not None:
            actual_orientation = np.asarray(pose.quat_xyzw, dtype=float)
            actual_orientation /= max(np.linalg.norm(actual_orientation), 1.0e-9)
            if float(np.dot(actual_orientation, target_orientation)) < 0.0:
                actual_orientation *= -1.0
            # For small rotations the quaternion vector component is about
            # half the rotation vector.  This keeps the physical wrist near
            # its measured pre-grasp attitude without overriding position.
            terms.append(4.0 * (actual_orientation[:3] - target_orientation[:3]))
        terms.append(0.02 * posture_cost)
        return np.concatenate(terms)

    initial_guesses = [np.clip(reference, lower, upper)]
    for seed in alternate_active_seeds or ():
        guess = reference.copy()
        seed_values = np.asarray(seed, dtype=float)
        guess[:min(4, len(seed_values))] = seed_values[:4]
        initial_guesses.append(np.clip(guess, lower, upper))

    candidates = []
    for initial in initial_guesses:
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=30,
            diff_step=1.0e-3,
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
        )
        solution = fixed.copy()
        solution[active_indices] = result.x
        pose = grasp_ik._call_fk(solution.tolist(), runtime.timeout).left_pose
        error = math.sqrt(
            sum(
                (float(pose.pos_xyz[index]) - float(target_xyz[index])) ** 2
                for index in range(3)
            )
        )
        max_delta_deg = max(
            abs(math.degrees(result.x[index] - reference[index]))
            for index in range(len(active_indices))
        )
        total_delta_deg = sum(
            abs(math.degrees(result.x[index] - reference[index]))
            for index in range(len(active_indices))
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


def _restricted_lift_step_size(index, high_step):
    if index <= 6:
        return 0.005
    if index <= 9:
        return 0.010
    return float(high_step)


def run_restricted_ik_check(
    job,
    move_time,
    high_step=0.01,
    approach_step=0.01,
):
    runtime = make_runtime(None, None, None, move_time, job)
    if job["arm"] != "left":
        raise RuntimeError("restricted IK check currently supports left arm only")
    table_geometry = job.get("table_geometry")
    if table_geometry is None:
        table_geometry = locate_green_table_geometry()
        job["table_geometry"] = table_geometry
    command_transit_z = float(table_geometry["command_transit_z"])

    current = list(pipeline._read_current_arm_joints(runtime.timeout))
    start_pose = grasp_ik._call_fk(current, runtime.timeout).left_pose
    start = list(start_pose.pos_xyz)
    restricted_side_y = float(table_geometry["side_y_left"])
    side = [start[0], restricted_side_y, start[2]]
    high = [
        float(table_geometry["outside_x"]),
        restricted_side_y,
        command_transit_z,
    ]
    over_part = [
        job["grasp"][0],
        job["grasp"][1],
        command_transit_z,
    ]
    targets = [
        ("side_clear", side),
    ]
    lift_positions = []
    dry_lift_z = float(side[2])
    for index in range(1, 61):
        if dry_lift_z >= command_transit_z - 1.0e-6:
            break
        dry_lift_z = min(
            dry_lift_z + _restricted_lift_step_size(index, high_step),
            command_transit_z,
        )
        lift_positions.append([side[0], side[1], dry_lift_z])
    for index, lift_position in enumerate(lift_positions, start=1):
        targets.append(
            ("side_lift_%02d" % index, lift_position)
        )
    side_high = [side[0], side[1], command_transit_z]
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
    fixed_joint5=None,
    orientation_quat=None,
    recursion_depth=0,
    tracking_retry=0,
):
    global _PINCH_MODEL_POINT
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
        use_wrist=True,
        fixed_joint5=fixed_joint5,
        orientation_quat=orientation_quat,
    )
    deltas = [
        abs(math.degrees(solution[index] - current[index]))
        for index in range(7)
    ]
    if not solver_success:
        raise RuntimeError(
            "%s blocked before command: xyz_err=%.4fm max_delta=%.1fdeg"
            % (label, prediction_error, max(deltas))
        )
    max_delta = max(deltas)
    if max_delta > RESTRICTED_MAX_JOINT_STEP_DEG:
        joint_index = deltas.index(max_delta)
        if recursion_depth >= 7:
            raise RuntimeError(
                "%s IK subdivision exhausted: joint %d jump %.2fdeg current=%s target_xyz=%s"
                % (
                    label,
                    joint_index + 1,
                    max_delta,
                    [round(float(value), 4) for value in current],
                    [round(float(value), 4) for value in target_xyz],
                )
            )
        current_pose = grasp_ik._call_fk(current, runtime.timeout).left_pose
        midpoint = [
            0.5 * (float(current_pose.pos_xyz[axis]) + float(target_xyz[axis]))
            for axis in range(3)
        ]
        rospy.loginfo(
            "restricted %s subdividing joint=%d jump=%.2fdeg depth=%d midpoint=%s",
            label,
            joint_index + 1,
            max_delta,
            recursion_depth,
            [round(value, 4) for value in midpoint],
        )
        current, _mid_pose = _execute_restricted_waypoint(
            runtime,
            arm_hold,
            current,
            midpoint,
            "%s_mid" % label,
            move_time,
            prediction_tolerance=prediction_tolerance,
            actual_tolerance=actual_tolerance,
            fixed_joint5=fixed_joint5,
            orientation_quat=orientation_quat,
            recursion_depth=recursion_depth + 1,
        )
        return _execute_restricted_waypoint(
            runtime,
            arm_hold,
            current,
            target_xyz,
            "%s_end" % label,
            move_time,
            prediction_tolerance=prediction_tolerance,
            actual_tolerance=actual_tolerance,
            fixed_joint5=fixed_joint5,
            orientation_quat=orientation_quat,
            recursion_depth=recursion_depth + 1,
        )
    if prediction_error > prediction_tolerance:
        raise RuntimeError(
            "%s blocked before command: xyz_err=%.4fm max_delta=%.1fdeg"
            % (label, prediction_error, max_delta)
        )
    if recursion_depth > 0:
        rospy.loginfo(
            "restricted %s subdivision accepted max_delta=%.2fdeg xyz_err=%.4fm depth=%d",
            label,
            max_delta,
            prediction_error,
            recursion_depth,
        )
    verbose_joint_feedback = label.startswith(
        ("pinch_align_", "grasp_lift_", "grasp_to_outside_", "place_above_")
    )
    if verbose_joint_feedback:
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
    if label.startswith(("grasp_to_outside_", "place_above_")):
        # The controller has no separate acceleration field on this topic.
        # Extending the interpolation time to 2.5x limits the commanded
        # transport speed to 40% and reduces the resulting acceleration.
        segment_duration = max(
            segment_duration,
            float(move_time) * TRANSFER_TIME_SCALE,
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
    actual_orientation_error_deg = None
    if orientation_quat is not None:
        actual_orientation_error_deg = math.degrees(
            grasp_ik._quat_angle_error(
                actual_pose.quat_xyzw,
                orientation_quat,
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
    if _PINCH_MODEL_OFFSET is not None:
        _PINCH_MODEL_POINT = [
            float(actual_pose.pos_xyz[axis]) + float(_PINCH_MODEL_OFFSET[axis])
            for axis in range(3)
        ]
    rospy.loginfo(
        "restricted %s actual=%s target=%s xyz_err=%.4fm quat_err=%s",
        label,
        [round(float(value), 4) for value in actual_pose.pos_xyz],
        [round(float(value), 4) for value in target_xyz],
        actual_error,
        (
            None
            if actual_orientation_error_deg is None
            else round(actual_orientation_error_deg, 2)
        ),
    )
    if verbose_joint_feedback:
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
    requires_tracking_replan = (
        orientation_quat is not None
        and actual_error > actual_tolerance
    )
    if requires_tracking_replan and tracking_retry < 2:
        rospy.logwarn(
            "restricted %s feedback replan=%d xyz_err=%.4fm quat_err=%.2fdeg",
            label,
            tracking_retry + 1,
            actual_error,
            actual_orientation_error_deg,
        )
        return _execute_restricted_waypoint(
            runtime,
            arm_hold,
            actual,
            target_xyz,
            "%s_feedback" % label,
            move_time,
            prediction_tolerance=prediction_tolerance,
            actual_tolerance=actual_tolerance,
            fixed_joint5=fixed_joint5,
            orientation_quat=orientation_quat,
            recursion_depth=recursion_depth,
            tracking_retry=tracking_retry + 1,
        )
    if actual_error > actual_tolerance:
        raise RuntimeError(
            "%s tracking error %.4fm" % (label, actual_error)
        )
    if (
        actual_orientation_error_deg is not None
        and actual_orientation_error_deg > RESTRICTED_ORIENTATION_TOLERANCE_DEG
    ):
        rospy.logwarn(
            "restricted %s FK orientation offset %.2fdeg; physical TF validates final alignment",
            label,
            actual_orientation_error_deg,
        )
    return actual, actual_pose


def _restricted_motion_duration(start_radians, target_radians, move_time):
    """Scale a safe waypoint duration to its largest commanded change."""
    max_delta_deg = max(
        abs(math.degrees(float(target_radians[index]) - float(start_radians[index])))
        for index in range(7)
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


def _follow_restricted_cartesian_path(
    runtime,
    arm_hold,
    current,
    endpoint,
    label_prefix,
    move_time,
    maximum_step,
    recorded_waypoints,
    actual_tolerance=0.025,
    minimum_actual_z=None,
    require_nondecreasing_z=False,
    fixed_joint5=None,
    carry_monitor=None,
    orientation_quat=None,
):
    pose = grasp_ik._call_fk(current, runtime.timeout).left_pose
    previous_actual_z = float(pose.pos_xyz[2])
    best_actual_z = previous_actual_z
    endpoint = [float(value) for value in endpoint]
    maximum_step = float(maximum_step)
    completion_tolerance = min(0.010, 0.75 * maximum_step)
    horizontal_then_vertical = (
        minimum_actual_z is not None and not require_nondecreasing_z
    )
    stalled_steps = 0
    for index in range(1, 81):
        if (
            require_nondecreasing_z
            and float(pose.pos_xyz[2]) >= endpoint[2] - 0.020
        ):
            return current, pose
        difference = [
            endpoint[axis] - float(pose.pos_xyz[axis])
            for axis in range(3)
        ]
        if require_nondecreasing_z:
            # A lift must stay vertical in measured Cartesian space.  Feeding
            # accumulated x/y tracking error back into this interpolation
            # previously spent most of each step correcting sideways drift,
            # leaving the carried part close to the table for too long.
            difference[0] = 0.0
            difference[1] = 0.0
        horizontal_distance = math.hypot(difference[0], difference[1])
        distance = math.sqrt(sum(value * value for value in difference))
        if distance <= completion_tolerance:
            return current, pose
        if (
            horizontal_then_vertical
            and horizontal_distance <= completion_tolerance
            and abs(difference[2]) <= 0.025
            and float(pose.pos_xyz[2]) >= float(minimum_actual_z)
        ):
            return current, pose
        adaptive_step = min(
            0.035,
            maximum_step * (1.0 + float(stalled_steps)),
        )
        if horizontal_then_vertical and horizontal_distance > completion_tolerance:
            # Keep the carried part high while translating in x/y.  The WBC
            # typically trails a Cartesian z command by about 1--2 cm, so a
            # small upward bias prevents that lag from accumulating into a
            # diagonal descent toward the table.
            step_scale = min(
                1.0,
                adaptive_step / max(horizontal_distance, 1.0e-6),
            )
            target = [
                float(pose.pos_xyz[0]) + step_scale * difference[0],
                float(pose.pos_xyz[1]) + step_scale * difference[1],
                max(endpoint[2], float(pose.pos_xyz[2]) + 0.018),
            ]
            progress_before = horizontal_distance
        else:
            step_scale = min(1.0, adaptive_step / max(distance, 1.0e-6))
            target = [
                float(pose.pos_xyz[axis]) + step_scale * difference[axis]
                for axis in range(3)
            ]
            progress_before = distance
        if carry_monitor is not None:
            # Reassert a low-frequency hold at each transport interpolation
            # point.  This keeps the simulated claw effort active without a
            # high-frequency command loop.
            control_scene2_claws(
                100.0,
                0.0,
                velocity=GRASP_CLAW_VELOCITY,
                effort=GRASP_CLAW_EFFORT,
            )
        current, pose = _execute_restricted_waypoint(
            runtime,
            arm_hold,
            current,
            target,
            "%s_%02d" % (label_prefix, index),
            move_time,
            prediction_tolerance=0.008,
            actual_tolerance=actual_tolerance,
            fixed_joint5=fixed_joint5,
            orientation_quat=orientation_quat,
        )
        if carry_monitor is not None:
            _snapshot, needs_height_recovery = carry_monitor.observe(
                "%s_%02d" % (label_prefix, index),
                current,
                enforce_height=(
                    horizontal_then_vertical
                    and horizontal_distance > completion_tolerance
                ),
            )
            if needs_height_recovery:
                rospy.logwarn(
                    "%s_%02d detected EE height loss; recovering vertically",
                    label_prefix,
                    index,
                )
                recovery_target = [
                    float(pose.pos_xyz[0]),
                    float(pose.pos_xyz[1]),
                    max(
                        SAFE_CARRY_COMMAND_Z_M,
                        float(pose.pos_xyz[2]) + 0.020,
                    ),
                ]
                current, pose = _execute_restricted_waypoint(
                    runtime,
                    arm_hold,
                    current,
                    recovery_target,
                    "%s_%02d_recover" % (label_prefix, index),
                    move_time,
                    prediction_tolerance=0.010,
                    actual_tolerance=actual_tolerance,
                    fixed_joint5=None,
                    orientation_quat=orientation_quat,
                )
                carry_monitor.observe(
                    "%s_%02d_recover" % (label_prefix, index),
                    current,
                    enforce_height=False,
                )
        actual_z = float(pose.pos_xyz[2])
        if minimum_actual_z is not None and actual_z < float(minimum_actual_z):
            raise RuntimeError(
                "%s_%02d actual z %.4fm below safe floor %.4fm"
                % (label_prefix, index, actual_z, float(minimum_actual_z))
            )
        if require_nondecreasing_z and actual_z < previous_actual_z - 0.010:
            raise RuntimeError(
                "%s_%02d unexpectedly lowered by %.4fm"
                % (label_prefix, index, previous_actual_z - actual_z)
            )
        remaining = math.sqrt(
            sum(
                (endpoint[axis] - float(pose.pos_xyz[axis])) ** 2
                for axis in range(3)
            )
        )
        if horizontal_then_vertical and horizontal_distance > completion_tolerance:
            remaining_for_progress = math.hypot(
                endpoint[0] - float(pose.pos_xyz[0]),
                endpoint[1] - float(pose.pos_xyz[1]),
            )
        else:
            remaining_for_progress = remaining
        if require_nondecreasing_z:
            if actual_z >= best_actual_z + 0.005:
                best_actual_z = actual_z
                stalled_steps = 0
            else:
                stalled_steps += 1
        elif remaining_for_progress >= progress_before - 0.0001:
            stalled_steps += 1
        else:
            stalled_steps = 0
        if stalled_steps >= 8:
            raise RuntimeError(
                "%s did not make enough Cartesian progress" % label_prefix
            )
        previous_actual_z = actual_z
        recorded_waypoints.append(list(current))
    raise RuntimeError("%s exceeded the 80-step limit" % label_prefix)


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
    pinch_target_xyz=None,
    fixed_joint5=None,
    orientation_quat=None,
    initial_pinch=None,
):
    """Align the pinch center using one TF-calibrated, joint-feedback model."""
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
    consecutive_passes = 0
    retry_count = 0
    best_error = float("inf")
    last_progress_time = time.monotonic()
    last_tf_timestamps = None
    orientation_reference = None
    fk_orientation_reference = (
        list(orientation_quat) if orientation_quat is not None else None
    )
    original_vision_target = list(job.get("vision_base") or [])
    object_alignment_target = list(job.get("vision_base") or [])
    last_difference = [float("inf")] * 3
    calibration_fk = grasp_ik._call_fk(current, runtime.timeout).left_pose
    feedback_pose = calibration_fk
    if initial_pinch is None:
        initial_pinch = read_left_pinch_position(timeout=2.0)
    if len(initial_pinch) != 3:
        raise RuntimeError("invalid initial physical pinch feedback")
    global _PINCH_MODEL_OFFSET, _PINCH_MODEL_POINT
    _PINCH_MODEL_OFFSET = [
        float(initial_pinch[axis]) - float(calibration_fk.pos_xyz[axis])
        for axis in range(3)
    ]
    _PINCH_MODEL_POINT = [float(value) for value in initial_pinch]
    rospy.loginfo(
        "ALIGN_DIAG calibrated pinch offset=%s from physical=%s fk=%s",
        [round(value, 4) for value in _PINCH_MODEL_OFFSET],
        [round(float(value), 4) for value in initial_pinch],
        [round(float(value), 4) for value in calibration_fk.pos_xyz],
    )
    for index in range(1, max(max_steps, 60) + 1):
        if pinch_target_xyz is not None:
            desired = [float(value) for value in pinch_target_xyz]
        elif pinch_center_clearance is None:
            desired = [
                float(object_alignment_target[0]),
                float(object_alignment_target[1]),
                float(object_alignment_target[2]) + clearance,
            ]
        else:
            desired = [
                float(object_alignment_target[0]),
                float(object_alignment_target[1]),
                float(object_alignment_target[2])
                + float(pinch_center_clearance),
            ]
        # The real gripper TF is sampled once on entry.  Every commanded
        # waypoint updates this model point from its returned FK pose, so the
        # alignment loop must consume that cache instead of issuing another
        # TF request that can block after a simulator time jump.
        if _PINCH_MODEL_POINT is None or len(_PINCH_MODEL_POINT) != 3:
            raise RuntimeError("pinch feedback cache is unavailable")
        physical = [float(value) for value in _PINCH_MODEL_POINT]
        tf_timestamps = [time.monotonic()]
        ee_pose = feedback_pose
        ee_xyz = list(ee_pose.pos_xyz)
        ee_quat = list(ee_pose.quat_xyzw)
        ee_stamp = float(rospy.Time.now().to_sec())
        difference = [desired[axis] - physical[axis] for axis in range(3)]
        distance = math.sqrt(sum(value * value for value in difference))
        last_difference = list(difference)
        previous_best_error = best_error
        feedback_improved = distance < best_error
        if feedback_improved:
            best_error = distance
            last_progress_time = time.monotonic()
            elapsed = 0.0
            rospy.loginfo(
                "ALIGN_DIAG loop=%d best error %.4fm -> %.4fm; reset no-progress timer",
                index,
                previous_best_error,
                distance,
            )
        else:
            elapsed = time.monotonic() - last_progress_time
        # A commanded step may legitimately take more than ten seconds when
        # the controller repeats a low-progress target.  Always consume that
        # step's new feedback first; only a feedback sample that still shows
        # no useful decrease may trigger the no-progress timeout.
        if not feedback_improved and elapsed >= 10.0:
            if retry_count >= 3:
                raise RuntimeError(
                    "pinch alignment timeout: nonconvergent axis error x=%.4fm y=%.4fm z=%.4fm"
                    % tuple(last_difference)
                )
            retry_count += 1
            refreshed = None
            if pinch_target_xyz is None and not job.get("vision_frozen"):
                refreshed = read_visible_red_position(job.get("vision_base"))
            if job.get("vision_frozen") and pinch_target_xyz is None:
                rospy.loginfo(
                    "ALIGN_DIAG retry=%d retaining frozen red target=%s",
                    retry_count,
                    [round(value, 4) for value in job["vision_base"]],
                )
            elif refreshed is not None and pinch_target_xyz is None:
                refresh_shift = math.sqrt(
                    sum(
                        (float(refreshed[axis]) - float(original_vision_target[axis])) ** 2
                        for axis in range(3)
                    )
                )
                if refresh_shift <= 0.008:
                    job["vision_base"] = list(refreshed)
                    rospy.loginfo(
                        "ALIGN_DIAG retry=%d accepted refreshed target shift=%.4fm",
                        retry_count,
                        refresh_shift,
                    )
                else:
                    rospy.logwarn(
                        "ALIGN_DIAG retry=%d rejected unstable refreshed target shift=%.4fm; retaining initial target",
                        retry_count,
                        refresh_shift,
                    )
                desired = [
                    float(job["vision_base"][0]),
                    float(job["vision_base"][1]),
                    float(job["vision_base"][2])
                    + float(pinch_center_clearance or clearance),
                ]
            fk_pose = feedback_pose
            pregrasp_target = [
                float(fk_pose.pos_xyz[0]),
                float(fk_pose.pos_xyz[1]),
                float(desired[2]) + 0.04,
            ]
            rospy.logwarn(
                "ALIGN_DIAG retry=%d elapsed=%.2fs resend_pregrasp=%s",
                retry_count,
                elapsed,
                [round(value, 4) for value in pregrasp_target],
            )
            current, feedback_pose = _execute_restricted_waypoint(
                runtime,
                arm_hold,
                current,
                pregrasp_target,
                "pinch_align_retry_%d" % retry_count,
                move_time,
                prediction_tolerance=0.015,
                actual_tolerance=0.025,
                fixed_joint5=fixed_joint5,
                orientation_quat=fk_orientation_reference,
            )
            best_error = float("inf")
            last_progress_time = time.monotonic()
            previous_distance = None
            stalled_steps = 0
            consecutive_passes = 0
            continue
        if orientation_reference is None:
            orientation_reference = [float(value) for value in ee_quat]
        orientation_error_deg = math.degrees(
            grasp_ik._quat_angle_error(ee_quat, orientation_reference)
        )
        all_timestamps = list(tf_timestamps) + [ee_stamp]
        tf_updated = (
            last_tf_timestamps is None
            or any(
                abs(all_timestamps[axis] - last_tf_timestamps[axis]) > 1.0e-5
                for axis in range(len(all_timestamps))
            )
        )
        last_tf_timestamps = all_timestamps
        rospy.loginfo(
            "ALIGN_DIAG loop=%d retry=%d elapsed=%.2fs ee_xyz=%s ee_quat=%s target_xyz=%s err_xyz=%s err_norm=%.4fm angle=%.2fdeg tf=%s updated=%s",
            index,
            retry_count,
            elapsed,
            [round(float(value), 4) for value in ee_xyz],
            [round(float(value), 4) for value in ee_quat],
            [round(value, 4) for value in desired],
            [round(value, 4) for value in difference],
            distance,
            orientation_error_deg,
            [round(value, 4) for value in all_timestamps],
            tf_updated,
        )
        rospy.loginfo(
            "pinch_align_%02d pinch=%s desired=%s dist=%.4fm",
            index,
            [round(value, 4) for value in physical],
            [round(value, 4) for value in desired],
            distance,
        )
        if (
            distance <= tolerance
            and orientation_error_deg <= RESTRICTED_ORIENTATION_TOLERANCE_DEG
        ):
            job["_pinch_model_estimate"] = list(physical)
            return current
        consecutive_passes = 0
        if distance > 0.10:
            raise RuntimeError("pinch alignment starts too far from target")
        if previous_distance is not None and distance >= previous_distance - 0.0005:
            stalled_steps += 1
        else:
            stalled_steps = 0
        if stalled_steps >= 2:
            rospy.logwarn(
                "ALIGN_DIAG loop=%d error nearly unchanged; continuing bounded correction while no-progress timer runs",
                index,
            )

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
        # Reuse the only post-motion FK/feedback returned by the trajectory
        # function.  A second service call here can block after a time jump.
        fk_pose = feedback_pose
        target_fk = [
            float(fk_pose.pos_xyz[axis])
            + step_scale * correction[axis]
            for axis in range(3)
        ]
        current, feedback_pose = _execute_restricted_waypoint(
            runtime,
            arm_hold,
            current,
            target_fk,
            "pinch_align_%02d" % index,
            move_time,
            prediction_tolerance=0.015,
            actual_tolerance=0.015 if phase == "z" else 0.02,
            fixed_joint5=fixed_joint5,
        )
        return_waypoints.append(list(current))
        previous_distance = distance

    raise RuntimeError(
        "pinch alignment exceeded its bounded loop: nonconvergent axis error x=%.4fm y=%.4fm z=%.4fm"
        % tuple(last_difference)
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
    place_target_xyz=None,
    high_step=0.05,
    approach_step=0.02,
    sensor_time_logger=None,
):
    if job["arm"] != "left":
        raise RuntimeError("restricted high transit currently supports left arm")
    high_step = float(high_step)
    approach_step = float(approach_step)
    if high_step < 0.01 or high_step > 0.05:
        raise ValueError("restricted high step must be within [0.010, 0.050]")
    if approach_step < 0.005 or approach_step > 0.02:
        raise ValueError("restricted approach step must be within [0.005, 0.020]")

    table_geometry = job.get("table_geometry")
    if table_geometry is None:
        table_geometry = locate_green_table_geometry()
        job["table_geometry"] = table_geometry
    side_y = float(table_geometry["side_y_left"])
    outside_x = float(table_geometry["outside_x"])
    minimum_transit_z = float(table_geometry["minimum_transit_z"])
    command_transit_z = float(table_geometry["command_transit_z"])
    carry_z = float(table_geometry["carry_z"])
    job["lift_pose"] = [
        float(job["grasp"][0]),
        float(job["grasp"][1]),
        carry_z,
    ]

    runtime = argparse.Namespace(timeout=20.0)
    start = list(pipeline._read_current_arm_joints(runtime.timeout))
    start_pose = grasp_ik._call_fk(start, runtime.timeout).left_pose
    state_machine = Scene2StateMachine()
    state_machine.freeze_target_vision = bool(job.get("vision_frozen"))
    state_machine.enter(
        "SEARCH",
        start,
        expected_red=job.get("vision_base"),
        target=job.get("grasp"),
    )
    if sensor_time_logger is not None:
        sensor_time_logger.log_stage("SEARCH_COMPLETE")
    side_target = [
        start_pose.pos_xyz[0],
        side_y,
        start_pose.pos_xyz[2],
    ]
    arm_hold = None
    arm_mode_changed = False
    humanoid_mode_changed = False
    wbc_trajectory_enabled = False
    returned_to_start = False
    claw_closed = False
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
            command_transit_z,
        ]
        # Replan every lift segment from measured feedback.  A partially
        # tracked step must not make the following precomputed target jump.
        measured_lift_pose = side_pose
        for index in range(1, 61):
            current_z = float(measured_lift_pose.pos_xyz[2])
            if current_z >= minimum_transit_z + 0.005:
                break
            step_size = _restricted_lift_step_size(index, high_step)
            target = [
                float(high_target[0]),
                float(high_target[1]),
                min(current_z + step_size, command_transit_z),
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
            raise RuntimeError("high lift exceeded the 60-step limit")

        high_pose = measured_lift_pose
        if float(high_pose.pos_xyz[2]) < minimum_transit_z:
            raise RuntimeError(
                "high transit blocked: actual z %.4fm below %.4fm"
                % (float(high_pose.pos_xyz[2]), minimum_transit_z)
            )
        if sensor_time_logger is not None:
            sensor_time_logger.log_stage("HIGH_LIFT_COMPLETE")

        outside_high = [
            outside_x,
            side_y,
            command_transit_z,
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
                    command_transit_z
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
            command_transit_z,
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
                    command_transit_z
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
                actual_tolerance=0.03,
            )
            return_waypoints.append(list(current))

        final_pose = tracked_over_pose
        if float(final_pose.pos_xyz[2]) < minimum_transit_z:
            raise RuntimeError(
                "over-part transit blocked: actual z %.4fm below %.4fm"
                % (float(final_pose.pos_xyz[2]), minimum_transit_z)
            )
        rospy.loginfo(
            "restricted high transit reached over-part xyz=%s",
            [round(float(value), 4) for value in final_pose.pos_xyz],
        )
        if sensor_time_logger is not None:
            sensor_time_logger.log_stage("OVER_PART")

        if approach_clearance is not None:
            state_machine.enter(
                "APPROACH_ABOVE",
                current,
                expected_red=job.get("vision_base"),
                target=job.get("grasp"),
            )
            clearance = float(approach_clearance)
            if clearance < 0.05 or clearance > 0.15:
                raise ValueError(
                    "restricted approach clearance must be within [0.05, 0.15]"
                )
            approach_target = list(
                job.get(
                    "pregrasp_pose",
                    [
                        float(job["grasp"][0]),
                        float(job["grasp"][1]),
                        float(job["grasp"][2]) + clearance,
                    ],
                )
            )
            approach_orientation = list(final_pose.quat_xyzw)
            approach_start_index = len(return_waypoints) - 1
            # The high route has already centred above the frozen XY target.
            # Use the proven continuous position descent here; final physical
            # pinch alignment below holds and verifies the orientation.
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
                    actual_tolerance=0.03,
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
            if sensor_time_logger is not None:
                sensor_time_logger.log_stage("PREGRASP")

        if pinch_align_clearance is not None:
            if approach_clearance is None:
                raise RuntimeError(
                    "pinch alignment requires a verified high approach first"
                )
            descent_snapshot = state_machine.enter(
                "DESCEND",
                current,
                expected_red=job.get("vision_base"),
                target=job.get("vision_base"),
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
                initial_pinch=descent_snapshot["pinch"],
            )
            physical = job.get("_pinch_model_estimate")
            rospy.loginfo(
                "pinch alignment complete model-estimate=%s",
                [round(value, 4) for value in physical],
            )
            if sensor_time_logger is not None:
                sensor_time_logger.log_stage("ALIGN_COMPLETE")

        if grasp_test:
            if approach_start_index is None:
                raise RuntimeError("grasp test requires a verified approach")
            grasp_snapshot = state_machine.enter(
                "GRASP",
                current,
                expected_red=job.get("vision_base"),
                target=job.get("vision_base"),
            )
            control_scene2_claws(
                100.0,
                0.0,
                velocity=GRASP_CLAW_VELOCITY,
                effort=GRASP_CLAW_EFFORT,
            )
            claw_closed = True
            rospy.sleep(0.8)
            if sensor_time_logger is not None:
                sensor_time_logger.log_stage("CLAW_CLOSED")
            current = _run_postclose_reverse_lift_checkpoint(
                runtime,
                arm_hold,
                current,
                return_waypoints,
                approach_start_index,
                move_time,
                sensor_time_logger,
            )
            return
            state_machine.enter(
                "VERIFY_GRASP",
                current,
                expected_red=job.get("vision_base"),
                target=job.get("vision_base"),
            )
            # The fifth joint follows the latest measured value at every IK
            # call.  Keeping its old grasp-time value caused a growing model /
            # controller mismatch during the horizontal carry.
            carry_monitor = CarryFeedbackMonitor(state_machine, grasp_snapshot)
            state_machine.enter(
                "LIFT",
                current,
                expected_red=job.get("vision_base"),
            )
            grasp_pose = grasp_ik._call_fk(current, runtime.timeout).left_pose
            carry_orientation = list(grasp_pose.quat_xyzw)
            current, _ = _follow_restricted_cartesian_path(
                runtime,
                arm_hold,
                current,
                [
                    float(grasp_pose.pos_xyz[0]),
                    float(grasp_pose.pos_xyz[1]),
                    carry_z,
                ],
                "grasp_lift",
                move_time,
                0.005,
                return_waypoints,
                actual_tolerance=0.040,
                minimum_actual_z=float(grasp_pose.pos_xyz[2]) - 0.003,
                require_nondecreasing_z=True,
                fixed_joint5=None,
                carry_monitor=carry_monitor,
                orientation_quat=carry_orientation,
            )
            lift_snapshot, _unused = carry_monitor.observe(
                "lift_complete",
                current,
                enforce_height=False,
            )
            state_machine.enter(
                "VERIFY_GRASP",
                current,
                expected_red=carry_monitor.last["red"],
            )
            carry_monitor.verify_lift()
            rospy.sleep(0.5)
            steady_snapshot, _unused = carry_monitor.observe(
                "lift_hold",
                current,
                enforce_height=False,
            )
            if (
                lift_snapshot["red"] is not None
                and steady_snapshot["red"] is not None
                and steady_snapshot["red"][2]
                < lift_snapshot["red"][2] - MAX_CARRY_SLIP_M
            ):
                raise RuntimeError("VERIFY_GRASP failed: red object lowered while holding")
            physical = read_left_pinch_position(timeout=2.0)
            rospy.loginfo(
                "grasp test lifted to physical pinch=%s",
                [round(value, 4) for value in physical],
            )
            if sensor_time_logger is not None:
                sensor_time_logger.log_stage("POST_GRASP_LIFT")
            if place_target_xyz is not None:
                state_machine.enter("TRANSFER", current)
                current, outside_transfer_pose = _follow_restricted_cartesian_path(
                    runtime,
                    arm_hold,
                    current,
                    [outside_x, side_y, carry_z],
                    "grasp_to_outside",
                    move_time,
                    0.005,
                    return_waypoints,
                    actual_tolerance=0.040,
                    minimum_actual_z=float(
                        table_geometry["minimum_carry_z"]
                    ),
                    fixed_joint5=None,
                    carry_monitor=carry_monitor,
                    orientation_quat=carry_orientation,
                )
                rospy.loginfo(
                    "grasp transfer returned to outside-high xyz=%s",
                    [
                        round(float(value), 4)
                        for value in outside_transfer_pose.pos_xyz
                    ],
                )
                refreshed_bin = locate_purple_bin()
                release_target = [float(value) for value in refreshed_bin["release"]]
                state_machine.enter(
                    "ABOVE_TARGET",
                    current,
                    target=release_target,
                )
                current, _ = _follow_restricted_cartesian_path(
                    runtime,
                    arm_hold,
                    current,
                    [
                        release_target[0] - 0.02,
                        release_target[1] - 0.02,
                        release_target[2] + 0.09,
                    ],
                    "place_above",
                    move_time,
                    0.005,
                    return_waypoints,
                    actual_tolerance=0.040,
                    minimum_actual_z=float(
                        table_geometry["place_minimum_z"]
                    ),
                    fixed_joint5=None,
                    carry_monitor=carry_monitor,
                    orientation_quat=carry_orientation,
                )
                if sensor_time_logger is not None:
                    sensor_time_logger.log_stage("ABOVE_PURPLE_BIN")
                state_machine.enter(
                    "DESCEND_TARGET",
                    current,
                    target=release_target,
                )
                current = _align_left_pinch_from_feedback(
                    runtime,
                    arm_hold,
                    current,
                    job,
                    move_time,
                    0.04,
                    20,
                    0.012,
                    return_waypoints,
                    pinch_target_xyz=release_target,
                    fixed_joint5=None,
                    orientation_quat=carry_orientation,
                )
                rospy.loginfo(
                    "purple-bin release physical pinch=%s target=%s",
                    [
                        round(value, 4)
                        for value in read_left_pinch_position(timeout=2.0)
                    ],
                    [round(value, 4) for value in release_target],
                )
                state_machine.enter("RELEASE", current, target=release_target)
                control_scene2_claws(0.0, 0.0)
                claw_closed = False
                rospy.sleep(1.0)
                if sensor_time_logger is not None:
                    sensor_time_logger.log_stage("RELEASED")
                state_machine.enter("RETREAT", current)
                release_pose = grasp_ik._call_fk(
                    current,
                    runtime.timeout,
                ).left_pose
                current, _ = _follow_restricted_cartesian_path(
                    runtime,
                    arm_hold,
                    current,
                    [
                        float(release_pose.pos_xyz[0]),
                        float(release_pose.pos_xyz[1]),
                        float(table_geometry["retreat_z"]),
                    ],
                    "release_retreat",
                    move_time,
                    0.010,
                    return_waypoints,
                    actual_tolerance=0.040,
                    minimum_actual_z=float(release_pose.pos_xyz[2]) - 0.010,
                    require_nondecreasing_z=True,
                    fixed_joint5=None,
                )
                current, safe_pose = _follow_restricted_cartesian_path(
                    runtime,
                    arm_hold,
                    current,
                    [
                        outside_x,
                        side_y,
                        float(table_geometry["return_z"]),
                    ],
                    "return_outside",
                    move_time,
                    0.015,
                    return_waypoints,
                    actual_tolerance=0.040,
                    minimum_actual_z=minimum_transit_z,
                    fixed_joint5=None,
                )
                rospy.loginfo(
                    "single-loop released and retreated to safe xyz=%s",
                    [round(float(value), 4) for value in safe_pose.pos_xyz],
                )
                if sensor_time_logger is not None:
                    sensor_time_logger.log_stage("RETREATED")
                returned_to_start = True
                return state_machine
            else:
                control_scene2_claws(0.0, 0.0)
                claw_closed = False
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
            and not (grasp_test and claw_closed)
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
        if grasp_test and not claw_closed:
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


def _read_postclose_joint_feedback(timeout, after_sensor_time_ns=None):
    deadline = time.monotonic() + float(timeout)
    while not rospy.is_shutdown():
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise RuntimeError("timed out waiting for fresh post-close joint feedback")
        message = rospy.wait_for_message(
            "/sensors_data_raw",
            sensorsData,
            timeout=remaining,
        )
        sensor_time_ns = (
            int(message.sensor_time.secs) * 1000000000
            + int(message.sensor_time.nsecs)
        )
        if (
            after_sensor_time_ns is not None
            and sensor_time_ns <= int(after_sensor_time_ns)
        ):
            continue
        joint_q = list(message.joint_data.joint_q)
        if len(joint_q) >= 27:
            joints = [float(value) for value in joint_q[13:27]]
        elif len(joint_q) >= 26:
            joints = [float(value) for value in joint_q[12:26]]
        else:
            raise RuntimeError(
                "post-close joint_q has %d values" % len(joint_q)
            )
        if len(joints) != 14 or not all(math.isfinite(value) for value in joints):
            raise RuntimeError("post-close joint feedback is invalid")
        return joints, sensor_time_ns
    raise RuntimeError("ROS shutdown while waiting for post-close feedback")


def _build_postclose_reverse_lift_plan(
    runtime,
    closed_joints,
    return_waypoints,
    approach_start_index,
):
    if approach_start_index is None:
        raise RuntimeError("post-close reverse lift has no approach boundary")
    if not 0 <= int(approach_start_index) < len(return_waypoints):
        raise RuntimeError("post-close reverse lift approach boundary is invalid")
    samples = [
        [float(value) for value in waypoint]
        for waypoint in return_waypoints[int(approach_start_index):]
    ]
    if len(samples) < 2 or any(len(sample) != 14 for sample in samples):
        raise RuntimeError("post-close reverse lift has insufficient joint history")

    closed = [float(value) for value in closed_joints]
    endpoint_error_deg = max(
        abs(math.degrees(samples[-1][index] - closed[index]))
        for index in range(7)
    )
    if endpoint_error_deg > 3.0:
        raise RuntimeError(
            "post-close history endpoint error %.2fdeg exceeds 3.00deg"
            % endpoint_error_deg
        )

    closed_pose = grasp_ik._call_fk(closed, runtime.timeout).left_pose
    closed_xyz = [float(value) for value in closed_pose.pos_xyz]
    closed_quat = [float(value) for value in closed_pose.quat_xyzw]
    planned = list(closed)
    previous_z = float(closed_xyz[2])
    plan = []
    max_xy_span = 0.0
    max_quat_error_deg = 0.0
    max_joint_step_deg = 0.0

    for source_index in range(len(samples) - 1, -1, -1):
        target = list(samples[source_index])
        target[7:14] = closed[7:14]
        joint_step_deg = max(
            abs(math.degrees(target[index] - planned[index]))
            for index in range(7)
        )
        if joint_step_deg < 0.02:
            continue
        if joint_step_deg > RESTRICTED_MAX_JOINT_STEP_DEG:
            raise RuntimeError(
                "post-close reverse step %.2fdeg exceeds %.2fdeg"
                % (joint_step_deg, RESTRICTED_MAX_JOINT_STEP_DEG)
            )

        target_pose = grasp_ik._call_fk(target, runtime.timeout).left_pose
        target_xyz = [float(value) for value in target_pose.pos_xyz]
        target_z = float(target_xyz[2])
        lift = target_z - closed_xyz[2]
        xy_span = math.hypot(
            target_xyz[0] - closed_xyz[0],
            target_xyz[1] - closed_xyz[1],
        )
        quat_error_deg = math.degrees(
            grasp_ik._quat_angle_error(target_pose.quat_xyzw, closed_quat)
        )
        if target_z < previous_z - 0.002:
            raise RuntimeError(
                "post-close reverse path descends %.4fm"
                % (previous_z - target_z)
            )
        if lift < -0.002:
            raise RuntimeError("post-close reverse path first presses downward")
        if xy_span > 0.050:
            raise RuntimeError(
                "post-close reverse path XY span %.4fm exceeds 0.0500m"
                % xy_span
            )
        if quat_error_deg > RESTRICTED_ORIENTATION_TOLERANCE_DEG:
            raise RuntimeError(
                "post-close reverse path orientation %.2fdeg exceeds %.2fdeg"
                % (quat_error_deg, RESTRICTED_ORIENTATION_TOLERANCE_DEG)
            )
        if lift > 0.010:
            raise RuntimeError(
                "post-close reverse path first checkpoint %.4fm exceeds 0.0100m"
                % lift
            )

        plan.append(
            {
                "source_index": source_index,
                "joints": target,
                "xyz": target_xyz,
                "lift": lift,
                "xy_span": xy_span,
                "quat_error_deg": quat_error_deg,
            }
        )
        planned = target
        previous_z = target_z
        max_xy_span = max(max_xy_span, xy_span)
        max_quat_error_deg = max(max_quat_error_deg, quat_error_deg)
        max_joint_step_deg = max(max_joint_step_deg, joint_step_deg)
        if lift >= 0.007:
            rospy.loginfo(
                "POSTCLOSE_REPLAY_PRECHECK plan=%d fk_dz=%.4fm xy_span=%.4fm "
                "max_joint_step=%.2fdeg max_quat_err=%.2fdeg",
                len(plan),
                lift,
                max_xy_span,
                max_joint_step_deg,
                max_quat_error_deg,
            )
            return plan, closed_pose

    available_lift = previous_z - closed_xyz[2]
    raise RuntimeError(
        "post-close reverse path has only %.4fm lift" % available_lift
    )


def _execute_postclose_reverse_lift(
    runtime,
    arm_hold,
    closed_joints,
    initial_sensor_time_ns,
    plan,
    closed_pose,
    move_time,
    hold_state,
):
    global _PINCH_MODEL_POINT

    actual = [float(value) for value in closed_joints]
    closed_xyz = [float(value) for value in closed_pose.pos_xyz]
    closed_quat = [float(value) for value in closed_pose.quat_xyzw]
    if _PINCH_MODEL_OFFSET is not None:
        _PINCH_MODEL_POINT = [
            closed_xyz[axis] + float(_PINCH_MODEL_OFFSET[axis])
            for axis in range(3)
        ]
    initial_pinch = read_left_pinch_position(timeout=2.0)
    best_ee_z = float(closed_xyz[2])
    best_pinch_z = float(initial_pinch[2])
    last_sensor_time_ns = int(initial_sensor_time_ns)

    for index, waypoint in enumerate(plan, start=1):
        target = [float(value) for value in waypoint["joints"]]
        target[7:14] = actual[7:14]
        command_step_deg = max(
            abs(math.degrees(target[joint] - actual[joint]))
            for joint in range(7)
        )
        if command_step_deg > RESTRICTED_MAX_JOINT_STEP_DEG:
            raise RuntimeError(
                "POSTCLOSE_REPLAY_STEP %d command %.2fdeg exceeds %.2fdeg"
                % (index, command_step_deg, RESTRICTED_MAX_JOINT_STEP_DEG)
            )

        control_scene2_claws(
            100.0,
            0.0,
            velocity=GRASP_CLAW_VELOCITY,
            effort=GRASP_CLAW_EFFORT,
        )
        before = list(actual)
        hold_state["joints"] = list(target)
        segment_duration = _restricted_motion_duration(
            before,
            target,
            move_time,
        )
        pipeline._execute_arm_motion(
            None,
            arm_hold,
            [math.degrees(value) for value in before],
            [math.degrees(value) for value in target],
            segment_duration,
            RESTRICTED_SEGMENT_SETTLE_SECONDS,
        )
        actual, sensor_time_ns = _read_postclose_joint_feedback(
            runtime.timeout,
            after_sensor_time_ns=last_sensor_time_ns,
        )
        hold_state["joints"] = list(actual)
        last_sensor_time_ns = int(sensor_time_ns)

        actual_step_deg = max(
            abs(math.degrees(actual[joint] - before[joint]))
            for joint in range(7)
        )
        locked_actual_deg = max(
            abs(math.degrees(actual[joint] - before[joint]))
            for joint in range(7, 14)
        )
        target_error_deg = max(
            abs(math.degrees(target[joint] - actual[joint]))
            for joint in range(7)
        )
        if actual_step_deg > 12.0:
            raise RuntimeError(
                "POSTCLOSE_REPLAY_STEP %d actual jump %.2fdeg exceeds 12.00deg"
                % (index, actual_step_deg)
            )
        if locked_actual_deg > MAX_LOCKED_ARM_DELTA_DEG:
            raise RuntimeError(
                "POSTCLOSE_REPLAY_STEP %d locked arm moved %.2fdeg"
                % (index, locked_actual_deg)
            )
        if target_error_deg > JOINT_TRACKING_WARN_DEG:
            raise RuntimeError(
                "POSTCLOSE_REPLAY_STEP %d tracking error %.2fdeg exceeds %.2fdeg"
                % (index, target_error_deg, JOINT_TRACKING_WARN_DEG)
            )

        actual_pose = grasp_ik._call_fk(actual, runtime.timeout).left_pose
        actual_xyz = [float(value) for value in actual_pose.pos_xyz]
        if _PINCH_MODEL_OFFSET is not None:
            _PINCH_MODEL_POINT = [
                actual_xyz[axis] + float(_PINCH_MODEL_OFFSET[axis])
                for axis in range(3)
            ]
        pinch = read_left_pinch_position(timeout=2.0)
        ee_lift = actual_xyz[2] - closed_xyz[2]
        pinch_lift = float(pinch[2]) - float(initial_pinch[2])
        xy_span = math.hypot(
            actual_xyz[0] - closed_xyz[0],
            actual_xyz[1] - closed_xyz[1],
        )
        quat_error_deg = math.degrees(
            grasp_ik._quat_angle_error(actual_pose.quat_xyzw, closed_quat)
        )
        if actual_xyz[2] < best_ee_z - MAX_CARRY_EE_DROP_M:
            raise RuntimeError(
                "POSTCLOSE_REPLAY_STEP %d EE dropped %.4fm"
                % (index, best_ee_z - actual_xyz[2])
            )
        if float(pinch[2]) < best_pinch_z - MAX_CARRY_EE_DROP_M:
            raise RuntimeError(
                "POSTCLOSE_REPLAY_STEP %d pinch dropped %.4fm"
                % (index, best_pinch_z - float(pinch[2]))
            )
        if xy_span > 0.055:
            raise RuntimeError(
                "POSTCLOSE_REPLAY_STEP %d XY span %.4fm exceeds 0.0550m"
                % (index, xy_span)
            )
        if quat_error_deg > RESTRICTED_ORIENTATION_TOLERANCE_DEG:
            raise RuntimeError(
                "POSTCLOSE_REPLAY_STEP %d orientation %.2fdeg exceeds %.2fdeg"
                % (index, quat_error_deg, RESTRICTED_ORIENTATION_TOLERANCE_DEG)
            )
        best_ee_z = max(best_ee_z, float(actual_xyz[2]))
        best_pinch_z = max(best_pinch_z, float(pinch[2]))
        rospy.loginfo(
            "POSTCLOSE_REPLAY_STEP %d/%d source=%d command=%.2fdeg "
            "actual=%.2fdeg target_err=%.2fdeg ee_dz=%.4fm model_pinch_dz=%.4fm "
            "xy=%.4fm quat_err=%.2fdeg sensor_time_ns=%d",
            index,
            len(plan),
            waypoint["source_index"],
            command_step_deg,
            actual_step_deg,
            target_error_deg,
            ee_lift,
            pinch_lift,
            xy_span,
            quat_error_deg,
            sensor_time_ns,
        )
        actual = list(actual)
        measured_lift = max(ee_lift, pinch_lift)
        if measured_lift > 0.010:
            raise RuntimeError(
                "POSTCLOSE_REPLAY_STEP %d first checkpoint overshoot %.4fm"
                % (index, measured_lift)
            )
        if measured_lift >= 0.007:
            break

    final_pose = grasp_ik._call_fk(actual, runtime.timeout).left_pose
    final_xyz = [float(value) for value in final_pose.pos_xyz]
    if _PINCH_MODEL_OFFSET is not None:
        _PINCH_MODEL_POINT = [
            final_xyz[axis] + float(_PINCH_MODEL_OFFSET[axis])
            for axis in range(3)
        ]
    final_pinch = read_left_pinch_position(timeout=2.0)
    final_ee_lift = final_xyz[2] - closed_xyz[2]
    final_pinch_lift = float(final_pinch[2]) - float(initial_pinch[2])
    if final_ee_lift < 0.007 or final_pinch_lift < 0.007:
        raise RuntimeError(
            "post-close reverse lift too small ee=%.4fm pinch=%.4fm"
            % (final_ee_lift, final_pinch_lift)
        )
    if final_ee_lift > 0.010 or final_pinch_lift > 0.010:
        raise RuntimeError(
            "post-close reverse lift too large ee=%.4fm pinch=%.4fm"
            % (final_ee_lift, final_pinch_lift)
        )
    control_scene2_claws(
        100.0,
        0.0,
        velocity=GRASP_CLAW_VELOCITY,
        effort=GRASP_CLAW_EFFORT,
    )
    arm_hold.set_degrees([math.degrees(value) for value in actual])
    rospy.loginfo(
        "POSTCLOSE_REVERSE_LIFT_PASS ee_dz=%.4fm model_pinch_dz=%.4fm final=%s",
        final_ee_lift,
        final_pinch_lift,
        [round(value, 4) for value in final_xyz],
    )
    return actual


def _hold_postclose_checkpoint(runtime, arm_hold, current, reason):
    hold = [float(value) for value in current]
    arm_hold.set_degrees([math.degrees(value) for value in hold])
    rospy.logwarn("POSTCLOSE_CHECKPOINT_HOLD reason=%s", reason)
    while not rospy.is_shutdown():
        try:
            control_scene2_claws(
                100.0,
                0.0,
                velocity=GRASP_CLAW_VELOCITY,
                effort=GRASP_CLAW_EFFORT,
            )
        except Exception as error:
            rospy.logerr("POSTCLOSE_CHECKPOINT_HOLD claw refresh failed: %s", error)
        rospy.logwarn_throttle(
            5.0,
            "POSTCLOSE_CHECKPOINT_HOLD retaining closed claw and external arm control",
        )
        rospy.sleep(1.0)
    return hold


def _run_postclose_reverse_lift_checkpoint(
    runtime,
    arm_hold,
    current,
    return_waypoints,
    approach_start_index,
    move_time,
    sensor_time_logger,
):
    hold = [float(value) for value in current]
    hold_state = {"joints": list(hold)}
    try:
        closed_joints, sensor_time_ns = _read_postclose_joint_feedback(
            runtime.timeout
        )
        hold = list(closed_joints)
        hold_state["joints"] = list(closed_joints)
        plan, closed_pose = _build_postclose_reverse_lift_plan(
            runtime,
            closed_joints,
            return_waypoints,
            approach_start_index,
        )
        hold = _execute_postclose_reverse_lift(
            runtime,
            arm_hold,
            closed_joints,
            sensor_time_ns,
            plan,
            closed_pose,
            move_time,
            hold_state,
        )
        if sensor_time_logger is not None:
            sensor_time_logger.log_stage("POST_GRASP_LIFT")
        return _hold_postclose_checkpoint(
            runtime,
            arm_hold,
            hold,
            "PASS",
        )
    except Exception as error:
        if rospy.is_shutdown():
            raise
        rospy.logerr("POSTCLOSE_REVERSE_LIFT_FAILED: %s", error)
        hold = list(hold_state["joints"])
        try:
            hold, _sensor_time_ns = _read_postclose_joint_feedback(
                min(2.0, runtime.timeout)
            )
        except Exception as feedback_error:
            rospy.logwarn(
                "post-close failure feedback unavailable: %s",
                feedback_error,
            )
        return _hold_postclose_checkpoint(
            runtime,
            arm_hold,
            hold,
            "FAILED: %s" % error,
        )


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
    table_geometry = job.get("table_geometry")
    if table_geometry is None:
        table_geometry = locate_green_table_geometry()
        job["table_geometry"] = table_geometry
    work_poses = grasp_ik._call_fk(current_joints, runtime.timeout)
    work_pose = (
        work_poses.left_pose if active_arm == "left" else work_poses.right_pose
    )
    start_position = list(work_pose.pos_xyz)
    work_quat = list(work_pose.quat_xyzw)
    # GRASP_ORIENTATION_BLEND is intentionally 0.0.  Preserve the established
    # behavior explicitly: keep the measured FK hand orientation and do not
    # consult any object-template or simulator-layout orientation.
    grasp_quat = list(work_quat)
    job["grasp_quat"] = list(grasp_quat)
    job["lift_quat"] = list(grasp_quat)
    transit_z = max(
        float(table_geometry["command_transit_z"]),
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
        max(
            float(start_position[2]),
            float(table_geometry["body_side_lift_z"]),
        ),
    ]
    for index, position in enumerate(
        _linear_positions(start_position, lift_target, BODY_SIDE_STEP_M),
        start=1,
    ):
        waypoints.append(("body_side_lift_%02d" % index, position, work_quat))

    side_y = (
        float(table_geometry["side_y_left"])
        if active_arm == "left"
        else float(table_geometry["side_y_right"])
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

    outside_forward = [
        float(table_geometry["outside_x"]),
        side_y,
        side_target[2],
    ]
    for index, position in enumerate(
        _linear_positions(side_target, outside_forward, MAX_CARTESIAN_STEP_M),
        start=1,
    ):
        waypoints.append(("outside_forward_%02d" % index, position, grasp_quat))

    outside_high = [
        float(table_geometry["outside_x"]),
        side_y,
        transit_z,
    ]
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
        state_machine = run_restricted_high_transit_motion(
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

    if args.restricted_single_loop_execute:
        if not -0.015 <= float(args.grasp_center_clearance) <= 0.020:
            raise ValueError(
                "grasp-center-clearance must be within [-0.015, 0.020]"
            )
        if not 0.05 <= float(args.bin_release_clearance) <= 0.15:
            raise ValueError(
                "bin-release-clearance must be within [0.05, 0.15]"
            )
        sensor_time_logger = SensorTimeStageLogger()
        pipeline._publish_head_target(20.0)
        rospy.sleep(1.0)
        job = build_frozen_vision_red_job(args.object)
        purple_bin = locate_purple_bin()
        purple_bin["release"][2] = (
            purple_bin["vision_base"][2]
            + float(args.bin_release_clearance)
        )
        state_machine = run_restricted_high_transit_motion(
            job,
            args.move_time,
            approach_clearance=0.15,
            pinch_align_clearance=0.04,
            pinch_align_max_steps=20,
            pinch_align_tolerance=0.012,
            pinch_center_clearance=float(args.grasp_center_clearance),
            grasp_test=True,
            place_target_xyz=purple_bin["release"],
            high_step=args.restricted_high_step,
            approach_step=args.restricted_approach_step,
            sensor_time_logger=sensor_time_logger,
        )
        state_machine.enter("VERIFY_SUCCESS")
        if not verify_red_in_purple_bin(purple_bin):
            raise RuntimeError(
                "single-loop motion finished but visual bin verification failed"
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
