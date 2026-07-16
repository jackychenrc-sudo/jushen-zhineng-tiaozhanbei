#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One guarded fixed-wrist step toward the wrist RGB-D tray target.

The head-camera lock preserves tray identity.  Right-wrist depth refines the
local tray point and the gripper TF frames provide the actual TCP.  A bounded
Cartesian correction is converted to the already validated command-reference
mapping while the left arm and the three right-wrist joints stay unchanged.

The default mode is calculation-only.  Execution requires both ``--execute``
and the exact confirmation token.  This script never commands the base or
claw.  A failed post-motion check automatically returns to the previous arm
command reference.
"""

from __future__ import print_function

import argparse
import math
import threading
import time

import numpy as np

from scene3_wrist_preclose_gate import (
    DEFAULT_DEPTH_TOPIC,
    DEFAULT_GRIPPER_BASE_FRAME,
    DEFAULT_INFO_TOPIC,
    DEFAULT_LEFT_FINGER_FRAME,
    DEFAULT_RIGHT_FINGER_FRAME,
    decode_compressed_depth_payload,
    deproject_pixel,
    estimate_finger_tips,
    evaluate_preclose_gate,
    project_camera_point,
    prompted_depth_component,
)


CONFIRMATION = "SCENE3_WRIST_DEPTH_3CM"
REFERENCE_PARAM = "/challenge_cup_task_template/scene3/arm_command_reference_deg"
LOCKED_BASE_PARAM = "/challenge_cup_task_template/scene3/locked_target_base_xyz"
LOCKED_ODOM_PARAM = "/challenge_cup_task_template/scene3/locked_target_odom_xyz"
WRIST_TARGET_ODOM_PARAM = "/challenge_cup_task_template/scene3/wrist_target_odom_xyz"
BASE_TOPIC = "/challenge_cup_task_template/scene3/grasp_point_base"
ODOM_TOPIC = "/challenge_cup_task_template/scene3/grasp_point_odom"


def distance(first, second):
    return float(
        np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float))
    )


def angle_degrees(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-9 or second_norm < 1e-9:
        raise ValueError("cannot measure an angle from a zero-length vector")
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def bounded_error_step(error_xyz, maximum_step_m=0.03, residual_m=0.018):
    error = np.asarray(error_xyz, dtype=float)
    if error.shape != (3,) or not np.all(np.isfinite(error)):
        raise ValueError("TCP error must be a finite xyz vector")
    error_norm = float(np.linalg.norm(error))
    if error_norm <= float(residual_m):
        return np.zeros(3, dtype=float), 0.0, error_norm
    length = min(float(maximum_step_m), error_norm - float(residual_m))
    if length <= 0.0:
        return np.zeros(3, dtype=float), 0.0, error_norm
    return error * (length / error_norm), length, error_norm


def motion_checks(
    planned_step_m,
    before_error_m,
    after_error_m,
    tcp_motion_xyz,
    planned_direction,
    fk_target_error_m,
):
    step = float(planned_step_m)
    motion_xyz = np.asarray(tcp_motion_xyz, dtype=float)
    direction = np.asarray(planned_direction, dtype=float)
    motion = float(np.linalg.norm(motion_xyz))
    progress = float(np.dot(motion_xyz, direction))
    reduction = float(before_error_m) - float(after_error_m)
    checks = {
        "tcp_error_reduced": reduction >= max(0.003, 0.25 * step),
        "forward_progress": progress >= 0.35 * step,
        "tcp_motion_bounded": 0.30 * step <= motion <= 1.80 * step + 0.005,
        "fk_target_error_bounded": float(fk_target_error_m)
        <= max(0.005, 0.50 * step),
    }
    return checks, progress, motion, reduction


def initial_edge_prompt(head_target_base, backoff_m=0.10):
    """Move the head-camera tray point to the robot-facing tray edge.

    The head detector reports a point on the tray body.  Near the shelf, that
    point is commonly hidden from the wrist depth camera while the near edge
    is visible between the fingers.  Backing off along the base-frame planar
    line of sight produces a deterministic prompt for that visible edge.
    """

    head = np.asarray(head_target_base, dtype=float)
    if head.shape != (3,) or not np.all(np.isfinite(head)):
        raise ValueError("head tray target must be a finite xyz vector")
    planar_norm = float(np.linalg.norm(head[:2]))
    if planar_norm < 1e-6:
        raise ValueError("head tray target is too close to define an approach ray")
    if float(backoff_m) <= 0.0:
        raise ValueError("initial edge backoff must be positive")

    approach_unit = head[:2] / planar_norm
    prompt = head.copy()
    prompt[:2] -= float(backoff_m) * approach_unit
    return prompt, approach_unit


def initial_candidate_gate(
    candidate_base,
    head_target_base,
    prompt_base,
    approach_unit,
    maximum_prompt_error_m=0.080,
    minimum_edge_offset_m=0.030,
    maximum_edge_offset_m=0.170,
    maximum_lateral_offset_m=0.080,
    maximum_height_error_m=0.060,
):
    """Verify that a first wrist-depth point is the near edge of this tray."""

    candidate = np.asarray(candidate_base, dtype=float)
    head = np.asarray(head_target_base, dtype=float)
    prompt = np.asarray(prompt_base, dtype=float)
    direction = np.asarray(approach_unit, dtype=float)
    if any(value.shape != (3,) for value in (candidate, head, prompt)):
        raise ValueError("initial candidate gate expects xyz vectors")
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("approach unit must be a finite xy vector")
    if not all(np.all(np.isfinite(value)) for value in (candidate, head, prompt)):
        raise ValueError("initial candidate gate received non-finite coordinates")

    head_to_candidate_xy = head[:2] - candidate[:2]
    edge_offset = float(np.dot(head_to_candidate_xy, direction))
    lateral_vector = head_to_candidate_xy - edge_offset * direction
    details = {
        "prompt_error_m": distance(candidate, prompt),
        "edge_offset_m": edge_offset,
        "lateral_offset_m": float(np.linalg.norm(lateral_vector)),
        "height_error_m": abs(float(candidate[2] - head[2])),
    }
    checks = {
        "near_edge_prompt": details["prompt_error_m"]
        <= float(maximum_prompt_error_m),
        "robot_facing_edge": float(minimum_edge_offset_m)
        <= details["edge_offset_m"]
        <= float(maximum_edge_offset_m),
        "lateral_consistency": details["lateral_offset_m"]
        <= float(maximum_lateral_offset_m),
        "height_consistency": details["height_error_m"]
        <= float(maximum_height_error_m),
    }
    return checks, details


def _translation_xyz(transform):
    value = transform.transform.translation
    return np.array([value.x, value.y, value.z], dtype=float)


class WristDepthObserver(object):
    def __init__(
        self,
        args,
        rospy,
        tf2_ros,
        PointStamped,
        CameraInfo,
        CompressedImage,
        locked_target_base,
        locked_wrist_target_odom=None,
    ):
        self.args = args
        self.rospy = rospy
        self.PointStamped = PointStamped
        self.lock = threading.Lock()
        self.camera_info = None
        self.observations = []
        self.last_error = "waiting for wrist depth"
        self.latched_target = None
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.head_target_base = np.asarray(locked_target_base, dtype=float)
        prompt_base, approach_unit = initial_edge_prompt(
            self.head_target_base,
            backoff_m=args.initial_edge_backoff,
        )
        self.initial_prompt_base = prompt_base
        self.initial_approach_unit = approach_unit

        target = PointStamped()
        target.header.frame_id = "base_link"
        target.header.stamp = rospy.Time(0)
        target.point.x = float(prompt_base[0])
        target.point.y = float(prompt_base[1])
        target.point.z = float(prompt_base[2])
        self.head_target = target

        if locked_wrist_target_odom is not None:
            wrist_target = np.asarray(locked_wrist_target_odom, dtype=float)
            if wrist_target.shape != (3,) or not np.all(np.isfinite(wrist_target)):
                raise ValueError("saved wrist target odom coordinate is invalid")
            latched = PointStamped()
            latched.header.frame_id = "odom"
            latched.header.stamp = rospy.Time(0)
            latched.point.x = float(wrist_target[0])
            latched.point.y = float(wrist_target[1])
            latched.point.z = float(wrist_target[2])
            self.latched_target = latched
            print(
                "Loaded fixed wrist target odom={}".format(
                    np.round(wrist_target, 4).tolist()
                )
            )
        else:
            print(
                "Initial wrist edge prompt: head={} prompt={} backoff={:.3f}m".format(
                    np.round(self.head_target_base, 4).tolist(),
                    np.round(self.initial_prompt_base, 4).tolist(),
                    float(args.initial_edge_backoff),
                )
            )

        rospy.Subscriber(args.info_topic, CameraInfo, self._info, queue_size=1)
        rospy.Subscriber(args.depth_topic, CompressedImage, self._depth, queue_size=1)

    def _info(self, message):
        with self.lock:
            self.camera_info = message

    def _transform_xyz(self, xyz, source_frame, target_frame):
        message = self.PointStamped()
        message.header.frame_id = source_frame
        message.header.stamp = self.rospy.Time(0)
        message.point.x = float(xyz[0])
        message.point.y = float(xyz[1])
        message.point.z = float(xyz[2])
        result = self.tf_buffer.transform(
            message, target_frame, self.rospy.Duration(0.3)
        )
        return np.array(
            [result.point.x, result.point.y, result.point.z], dtype=float
        )

    def _source_target(self):
        with self.lock:
            return self.latched_target or self.head_target

    def _depth(self, message):
        try:
            import cv2

            with self.lock:
                info = self.camera_info
            source = self._source_target()
            if info is None or source is None:
                return

            camera_frame = str(self.args.camera_frame or info.header.frame_id)
            source_latest = self.PointStamped()
            source_latest.header.frame_id = source.header.frame_id
            source_latest.header.stamp = self.rospy.Time(0)
            source_latest.point = source.point
            expected_message = self.tf_buffer.transform(
                source_latest, camera_frame, self.rospy.Duration(0.3)
            )
            expected = np.array(
                [
                    expected_message.point.x,
                    expected_message.point.y,
                    expected_message.point.z,
                ],
                dtype=float,
            )
            camera_k = list(info.K)
            prompt = project_camera_point(expected, camera_k)
            depth = decode_compressed_depth_payload(message.data, cv2)
            component = prompted_depth_component(
                depth,
                prompt,
                expected[2] * 1000.0,
                roi_radius_px=self.args.roi_radius,
                depth_band_mm=self.args.depth_band,
                minimum_pixels=self.args.minimum_component_pixels,
            )
            refined_camera = deproject_pixel(
                component["target_pixel"], component["target_depth_mm"], camera_k
            )

            with self.lock:
                needs_latch = self.latched_target is None
            if needs_latch:
                candidate_base = self._transform_xyz(
                    refined_camera, camera_frame, "base_link"
                )
                latch_checks, latch_details = initial_candidate_gate(
                    candidate_base,
                    self.head_target_base,
                    self.initial_prompt_base,
                    self.initial_approach_unit,
                    maximum_prompt_error_m=self.args.maximum_initial_prompt_error,
                    minimum_edge_offset_m=self.args.minimum_initial_edge_offset,
                    maximum_edge_offset_m=self.args.maximum_initial_edge_offset,
                    maximum_lateral_offset_m=self.args.maximum_initial_lateral_offset,
                    maximum_height_error_m=self.args.maximum_initial_height_error,
                )
                if not all(latch_checks.values()):
                    raise RuntimeError(
                        "initial wrist edge candidate failed same-tray geometry: "
                        "checks={} details={}".format(latch_checks, latch_details)
                    )
                target_message = self.PointStamped()
                target_message.header.frame_id = camera_frame
                target_message.header.stamp = self.rospy.Time(0)
                target_message.point.x = float(refined_camera[0])
                target_message.point.y = float(refined_camera[1])
                target_message.point.z = float(refined_camera[2])
                latched = self.tf_buffer.transform(
                    target_message, "odom", self.rospy.Duration(0.3)
                )
                latched.header.stamp = self.rospy.Time(0)
                with self.lock:
                    if self.latched_target is None:
                        self.latched_target = latched
                wrist_target_odom = [
                    float(latched.point.x),
                    float(latched.point.y),
                    float(latched.point.z),
                ]
                self.rospy.set_param(WRIST_TARGET_ODOM_PARAM, wrist_target_odom)
                print(
                    "Latched fixed wrist target odom={}".format(
                        np.round(wrist_target_odom, 4).tolist()
                    )
                )
                print(
                    "Initial wrist edge candidate base={} checks={} details={}".format(
                        np.round(candidate_base, 4).tolist(),
                        latch_checks,
                        {key: round(value, 4) for key, value in latch_details.items()},
                    )
                )
                target_camera = np.asarray(refined_camera, dtype=float)
            else:
                # Once a legal wrist RGB-D point has been selected, keep that
                # exact physical point in odom.  The depth component remains a
                # visibility/occupancy check; it is no longer allowed to move
                # the servo target to a different tray pixel on every run.
                target_camera = np.asarray(expected, dtype=float)

            transforms = []
            for frame in (
                self.args.left_finger_frame,
                self.args.right_finger_frame,
                self.args.gripper_base_frame,
            ):
                transforms.append(
                    self.tf_buffer.lookup_transform(
                        camera_frame,
                        frame,
                        self.rospy.Time(0),
                        self.rospy.Duration(0.3),
                    )
                )
            left_tip, right_tip, tcp_camera, forward_camera = estimate_finger_tips(
                _translation_xyz(transforms[0]),
                _translation_xyz(transforms[1]),
                _translation_xyz(transforms[2]),
                extension_m=self.args.tcp_extension,
            )
            gate = evaluate_preclose_gate(
                depth,
                component["mask"],
                target_camera,
                left_tip,
                right_tip,
                camera_k,
                corridor_radius_px=self.args.corridor_radius,
                maximum_target_segment_distance_m=self.args.segment_tolerance,
                maximum_tcp_error_m=self.args.tcp_tolerance,
                minimum_tray_corridor_pixels=self.args.minimum_corridor_pixels,
                maximum_obstacle_ratio=self.args.maximum_obstacle_ratio,
            )
            target_base = self._transform_xyz(target_camera, camera_frame, "base_link")
            tcp_base = self._transform_xyz(tcp_camera, camera_frame, "base_link")
            axis_tip_base = self._transform_xyz(
                tcp_camera + 0.10 * forward_camera, camera_frame, "base_link"
            )
            axis_base = axis_tip_base - tcp_base
            axis_base /= float(np.linalg.norm(axis_base))
            observation = {
                "stamp": (int(message.header.stamp.secs), int(message.header.stamp.nsecs)),
                "target_base": target_base,
                "tcp_base": tcp_base,
                "target_camera": np.asarray(target_camera, dtype=float),
                "tcp_camera": np.asarray(tcp_camera, dtype=float),
                "axis_base": axis_base,
                "obstacle_ratio": float(gate["obstacle_ratio"]),
                "segment_distance": float(gate["segment_distance_m"]),
                "segment_parameter": float(gate["segment_parameter"]),
                "finger_gap": float(gate["finger_gap_m"]),
                "gate_checks": dict(gate["checks"]),
            }
            with self.lock:
                if (
                    not self.observations
                    or self.observations[-1]["stamp"] != observation["stamp"]
                ):
                    self.observations.append(observation)
                    del self.observations[:-20]
                self.last_error = ""
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
            self.rospy.logwarn_throttle(1.0, "scene3 wrist depth step: %s", exc)

    def clear(self):
        with self.lock:
            self.observations = []
            self.last_error = "waiting for fresh post-motion wrist depth"

    def wait_stable(self, timeout, sample_count=3):
        deadline = time.time() + float(timeout)
        last_error = "waiting for wrist depth"
        while not self.rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                samples = list(self.observations[-int(sample_count) :])
                last_error = self.last_error
            if len(samples) >= int(sample_count):
                targets = np.asarray([item["target_base"] for item in samples])
                tcps = np.asarray([item["tcp_base"] for item in samples])
                target = np.median(targets, axis=0)
                tcp = np.median(tcps, axis=0)
                target_spread = float(
                    np.max(np.linalg.norm(targets - target, axis=1))
                )
                tcp_spread = float(np.max(np.linalg.norm(tcps - tcp, axis=1)))
                if (
                    target_spread <= self.args.maximum_target_spread
                    and tcp_spread <= self.args.maximum_tcp_spread
                ):
                    return {
                        "target": target,
                        "tcp": tcp,
                        "target_camera": np.median(
                            np.asarray([item["target_camera"] for item in samples]),
                            axis=0,
                        ),
                        "tcp_camera": np.median(
                            np.asarray([item["tcp_camera"] for item in samples]),
                            axis=0,
                        ),
                        "axis": self._median_axis(samples),
                        "target_spread": target_spread,
                        "tcp_spread": tcp_spread,
                        "obstacle_ratio": max(
                            item["obstacle_ratio"] for item in samples
                        ),
                        "segment_distance": float(
                            np.median([item["segment_distance"] for item in samples])
                        ),
                        "segment_parameter": float(
                            np.median([item["segment_parameter"] for item in samples])
                        ),
                        "finger_gap": float(
                            np.median([item["finger_gap"] for item in samples])
                        ),
                        "gate_checks": samples[-1]["gate_checks"],
                    }
            self.rospy.sleep(0.05)
        raise RuntimeError(
            "no stable wrist depth observation: {}".format(last_error or "unstable")
        )

    @staticmethod
    def _median_axis(samples):
        axis = np.median(np.asarray([item["axis_base"] for item in samples]), axis=0)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            raise RuntimeError("stable wrist observations have a degenerate gripper axis")
        return axis / norm


def run(args):
    import rospy
    import tf2_geometry_msgs  # noqa: F401
    import tf2_ros
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import CameraInfo, CompressedImage, JointState

    from challenge_task_3 import Scene3Task
    from scene3_fixed_wrist_ik import solve_fixed_wrist_position

    rospy.init_node("scene3_wrist_depth_step", anonymous=True)

    reference = [float(value) for value in rospy.get_param(REFERENCE_PARAM)]
    locked_base = np.asarray(rospy.get_param(LOCKED_BASE_PARAM), dtype=float)
    locked_odom = np.asarray(rospy.get_param(LOCKED_ODOM_PARAM), dtype=float)
    if len(reference) != 14 or locked_base.shape != (3,) or locked_odom.shape != (3,):
        raise RuntimeError("saved arm reference or locked tray coordinates are invalid")

    arm_publisher = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    task = Scene3Task(None, arm_publisher)
    task.wait_for_arm_subscriber(timeout=args.timeout)

    def hold_reference(values, hold_time):
        values = [float(value) for value in values]
        rate = rospy.Rate(args.hz)
        deadline = time.time() + float(hold_time)
        while not rospy.is_shutdown() and time.time() < deadline:
            task.publish_arm_degrees_once(values)
            rate.sleep()
        rospy.set_param(REFERENCE_PARAM, values)

    def move_reference(start_values, target_values, duration):
        start_values = np.asarray(start_values, dtype=float)
        target_values = np.asarray(target_values, dtype=float)
        if start_values.shape != (14,) or target_values.shape != (14,):
            raise ValueError("arm command references must contain 14 values")
        steps = max(1, int(float(duration) * float(args.hz)))
        checkpoint_stride = max(1, int(float(args.hz) / 10.0))
        rate = rospy.Rate(args.hz)
        for index in range(steps + 1):
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown during arm interpolation")
            progress = float(index) / float(steps)
            alpha = (
                10.0 * progress ** 3
                - 15.0 * progress ** 4
                + 6.0 * progress ** 5
            )
            point = start_values + alpha * (target_values - start_values)
            task.publish_arm_degrees_once(point.tolist())
            if index % checkpoint_stride == 0 or index == steps:
                rospy.set_param(REFERENCE_PARAM, point.tolist())
            rate.sleep()
        hold_reference(target_values.tolist(), args.settle_seconds)

    def sample_arm(count=5):
        samples = []
        for _ in range(int(count)):
            samples.append(np.asarray(task.read_current_arm_joints(), dtype=float))
            rospy.sleep(0.08)
        return np.median(np.asarray(samples), axis=0)

    def fk_position(arm):
        poses = task.call_fk([float(value) for value in arm])
        return np.asarray(poses.right_pose.pos_xyz, dtype=float)

    def collect_points(topic, count=3):
        points = []
        for _ in range(int(count)):
            message = rospy.wait_for_message(
                topic, PointStamped, timeout=float(args.timeout)
            )
            points.append([message.point.x, message.point.y, message.point.z])
        points = np.asarray(points, dtype=float)
        center = np.median(points, axis=0)
        spread = max(distance(first, second) for first in points for second in points)
        return center, spread

    def observe_head():
        base, base_spread = collect_points(BASE_TOPIC)
        odom, odom_spread = collect_points(ODOM_TOPIC)
        identity_error = distance(odom, locked_odom)
        checks = {
            "base_stable": base_spread <= args.maximum_head_spread,
            "odom_stable": odom_spread <= args.maximum_head_spread,
            "same_tray": identity_error <= args.maximum_identity_error,
        }
        return {
            "base": base,
            "base_spread": base_spread,
            "odom_spread": odom_spread,
            "identity_error": identity_error,
            "checks": checks,
        }

    def rollback(previous_reference, reason):
        print("ROLLBACK: {}".format(reason))
        current_reference = rospy.get_param(REFERENCE_PARAM, previous_reference)
        move_reference(current_reference, previous_reference, args.motion_seconds)

    if args.execute:
        if args.confirmation != CONFIRMATION:
            raise RuntimeError(
                "execution blocked; pass --confirmation {}".format(CONFIRMATION)
            )
        if not task.set_arm_mode(2):
            raise RuntimeError("cannot enable arm external-control mode 2")
        hold_reference(reference, 1.0)
    baseline_first = sample_arm()
    if args.execute:
        hold_reference(reference, 0.8)
    else:
        rospy.sleep(0.8)
    baseline = sample_arm()
    controlled_indices = list(range(4)) + list(range(7, 11))
    baseline_drift = max(
        abs(math.degrees(baseline[index] - baseline_first[index]))
        for index in controlled_indices
    )
    if baseline_drift > args.maximum_baseline_drift_deg:
        raise RuntimeError("arm command baseline is still drifting")

    head_before = observe_head()
    print(
        "Head identity before: base={} spread={:.4f}m identity_error={:.4f}m checks={}".format(
            np.round(head_before["base"], 4).tolist(),
            head_before["base_spread"],
            head_before["identity_error"],
            head_before["checks"],
        )
    )
    if not all(head_before["checks"].values()):
        raise RuntimeError("head same-tray gate failed before motion")

    if args.reset_wrist_target and rospy.has_param(WRIST_TARGET_ODOM_PARAM):
        rospy.delete_param(WRIST_TARGET_ODOM_PARAM)
        print("Deleted previous wrist target; one new wrist point will be latched")
    fixed_wrist_target = None
    if rospy.has_param(WRIST_TARGET_ODOM_PARAM):
        fixed_wrist_target = rospy.get_param(WRIST_TARGET_ODOM_PARAM)
    observer = WristDepthObserver(
        args,
        rospy,
        tf2_ros,
        PointStamped,
        CameraInfo,
        CompressedImage,
        head_before["base"],
        locked_wrist_target_odom=fixed_wrist_target,
    )

    print("Waiting for three stable wrist-depth target/TCP observations")
    before = observer.wait_stable(args.observation_timeout)
    error_xyz = before["target"] - before["tcp"]
    step_xyz, step_length, error_norm = bounded_error_step(
        error_xyz,
        maximum_step_m=args.maximum_cartesian_step,
        residual_m=args.residual_distance,
    )
    print("Wrist target camera:", np.round(before["target_camera"], 4).tolist())
    print("Wrist TCP camera:", np.round(before["tcp_camera"], 4).tolist())
    print("Wrist target base:", np.round(before["target"], 4).tolist())
    print("Wrist TCP base:", np.round(before["tcp"], 4).tolist())
    print("TCP error: {:.1f}mm".format(error_norm * 1000.0))
    print("Obstacle ratio: {:.3f}".format(before["obstacle_ratio"]))
    print("Target-to-finger segment: {:.1f}mm".format(before["segment_distance"] * 1000.0))
    print("Target segment parameter: {:.3f}".format(before["segment_parameter"]))
    print("Finger gap: {:.1f}mm".format(before["finger_gap"] * 1000.0))
    print("Gripper-axis error: {:.1f}deg".format(angle_degrees(before["axis"], error_xyz)))
    failed_before = [
        name for name, passed in before["gate_checks"].items() if not passed
    ]
    print("Preclose conditions still failing:", failed_before)
    if before["obstacle_ratio"] > args.maximum_obstacle_ratio:
        raise RuntimeError("wrist depth sees an obstacle in the closing corridor")
    if error_norm > args.maximum_start_error:
        raise RuntimeError("wrist TCP error is outside bounded approach range")
    if step_length < args.minimum_cartesian_step:
        print("WRIST_DEPTH_STEP_NOT_NEEDED: run the preclose gate; claw remains open")
        return 0

    before_fk = fk_position(baseline)
    local_target = before_fk + step_xyz
    solved_result = solve_fixed_wrist_position(
        fk_position,
        baseline,
        local_target,
        tolerance_m=args.maximum_ik_error,
        max_iterations=args.maximum_ik_iterations,
    )
    if not solved_result["success"]:
        raise RuntimeError("fixed-wrist IK could not plan the depth-guided step")
    solved = np.asarray(solved_result["arm_joints_rad"], dtype=float)
    delta_deg = np.rad2deg(solved - baseline)
    left_delta = delta_deg[:7]
    proximal_delta = delta_deg[7:11]
    wrist_delta = delta_deg[11:14]
    target_reference = list(reference)
    for index in range(14):
        target_reference[index] += float(delta_deg[index])

    right_proximal = np.asarray(target_reference[7:11], dtype=float)
    joint_checks = {
        "left_frozen": float(np.max(np.abs(left_delta))) <= 1e-6,
        "wrist_frozen": float(np.max(np.abs(wrist_delta))) <= 1e-6,
        "proximal_bounded": float(np.max(np.abs(proximal_delta)))
        <= args.maximum_joint_step_deg,
        "joint_limits": bool(
            np.all(right_proximal >= np.asarray([-170.0, -100.0, -80.0, -115.0]))
            and np.all(right_proximal <= np.asarray([30.0, 50.0, 80.0, -1.0]))
        ),
    }
    print("Planned base-frame step:", np.round(step_xyz, 4).tolist())
    print("Planned Cartesian length: {:.1f}mm".format(step_length * 1000.0))
    print("Predicted residual: {:.1f}mm".format((error_norm - step_length) * 1000.0))
    print("Right shoulder/elbow delta:", np.round(proximal_delta, 3).tolist())
    print("Left-arm maximum delta: {:.6f}deg".format(np.max(np.abs(left_delta))))
    print("Wrist maximum delta: {:.6f}deg".format(np.max(np.abs(wrist_delta))))
    print("IK error: {:.4f}m".format(float(solved_result["final_error_m"])))
    print("Planning checks:", joint_checks)
    if not all(joint_checks.values()):
        raise RuntimeError("WRIST_DEPTH_STEP_PLAN_BLOCKED: joint safety gate failed")

    if not args.execute:
        print("WRIST_DEPTH_3CM_PLAN_OK: calculation only; no command sent")
        return 0
    previous_reference = list(reference)
    moved = False
    direction = step_xyz / step_length
    try:
        moved = True
        print("Executing one fixed-wrist depth-guided step; base and claw stay untouched")
        move_reference(reference, target_reference, args.motion_seconds)
        after_arm = sample_arm()
        after_fk = fk_position(after_arm)
        fk_target_error = distance(after_fk, local_target)

        observer.clear()
        rospy.sleep(args.vision_settle_seconds)
        after = observer.wait_stable(args.observation_timeout)
        head_after = observe_head()
        after_error = distance(after["target"], after["tcp"])
        tcp_motion_xyz = after["tcp"] - before["tcp"]
        checks, progress, tcp_motion, reduction = motion_checks(
            step_length,
            error_norm,
            after_error,
            tcp_motion_xyz,
            direction,
            fk_target_error,
        )
        checks.update(
            {
                "wrist_target_stable": after["target_spread"]
                <= args.maximum_target_spread,
                "wrist_tcp_stable": after["tcp_spread"] <= args.maximum_tcp_spread,
                "obstacle_clearance": after["obstacle_ratio"]
                <= args.maximum_obstacle_ratio,
                "same_tray": all(head_after["checks"].values()),
                "left_arm_bounded": float(
                    np.max(np.abs(np.rad2deg(after_arm[:7] - baseline[:7])))
                )
                <= args.maximum_uncommanded_motion_deg,
                "gripper_axis_motion_bounded": angle_degrees(
                    before["axis"], after["axis"]
                )
                <= args.maximum_axis_change_deg,
                "finger_gap_stable": abs(
                    after["finger_gap"] - before["finger_gap"]
                )
                <= args.maximum_finger_gap_change,
                "segment_distance_not_worse": after["segment_distance"]
                <= before["segment_distance"] + args.maximum_segment_worsening,
            }
        )
        actual_wrist_motion = float(
            np.max(np.abs(np.rad2deg(after_arm[11:14] - baseline[11:14])))
        )
        axis_change = angle_degrees(before["axis"], after["axis"])
        axis_error_after = angle_degrees(
            after["axis"], after["target"] - after["tcp"]
        )
        print("TCP error: {:.1f}mm -> {:.1f}mm".format(error_norm * 1000.0, after_error * 1000.0))
        print("Observed TCP motion: {:.1f}mm".format(tcp_motion * 1000.0))
        print("Observed forward progress: {:.1f}mm".format(progress * 1000.0))
        print("Observed error reduction: {:.1f}mm".format(reduction * 1000.0))
        print("FK target error: {:.1f}mm".format(fk_target_error * 1000.0))
        print("Obstacle ratio after: {:.3f}".format(after["obstacle_ratio"]))
        print(
            "Target-to-finger segment: {:.1f}mm -> {:.1f}mm".format(
                before["segment_distance"] * 1000.0,
                after["segment_distance"] * 1000.0,
            )
        )
        print(
            "Gripper-axis change: {:.1f}deg; axis-to-target error after: {:.1f}deg".format(
                axis_change, axis_error_after
            )
        )
        print("Observed wrist-sensor motion (diagnostic only): {:.1f}deg".format(actual_wrist_motion))
        failed_after = [
            name for name, passed in after["gate_checks"].items() if not passed
        ]
        print("Preclose conditions still failing after step:", failed_after)
        print(
            "Head identity after: spread={:.4f}m identity_error={:.4f}m checks={}".format(
                head_after["base_spread"],
                head_after["identity_error"],
                head_after["checks"],
            )
        )
        print("Post-motion checks:", checks)
        if not all(checks.values()):
            rollback(previous_reference, "post-motion depth/geometry gate failed")
            print("WRIST_DEPTH_STEP_BLOCKED: rolled back; claw remains open")
            return 2
    except Exception as exc:
        if moved:
            rollback(previous_reference, "exception: {}".format(exc))
        raise

    rospy.set_param(REFERENCE_PARAM, target_reference)
    print(
        "WRIST_DEPTH_STEP_OK: TCP error {:.1f}mm -> {:.1f}mm; claw remains open".format(
            error_norm * 1000.0, after_error * 1000.0
        )
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--reset-wrist-target",
        action="store_true",
        help="discard the saved wrist target and latch one new odom point",
    )
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--info-topic", default=DEFAULT_INFO_TOPIC)
    parser.add_argument("--camera-frame", default="right_wrist_camera_link")
    parser.add_argument("--left-finger-frame", default=DEFAULT_LEFT_FINGER_FRAME)
    parser.add_argument("--right-finger-frame", default=DEFAULT_RIGHT_FINGER_FRAME)
    parser.add_argument("--gripper-base-frame", default=DEFAULT_GRIPPER_BASE_FRAME)
    parser.add_argument("--tcp-extension", type=float, default=0.045)
    parser.add_argument(
        "--initial-edge-backoff",
        type=float,
        default=0.10,
        help="move the first wrist prompt from tray body to its robot-facing edge",
    )
    parser.add_argument("--maximum-initial-prompt-error", type=float, default=0.080)
    parser.add_argument("--minimum-initial-edge-offset", type=float, default=0.030)
    parser.add_argument("--maximum-initial-edge-offset", type=float, default=0.170)
    parser.add_argument("--maximum-initial-lateral-offset", type=float, default=0.080)
    parser.add_argument("--maximum-initial-height-error", type=float, default=0.060)
    parser.add_argument("--roi-radius", type=int, default=140)
    parser.add_argument("--depth-band", type=float, default=100.0)
    parser.add_argument("--minimum-component-pixels", type=int, default=15)
    parser.add_argument("--corridor-radius", type=int, default=8)
    parser.add_argument("--segment-tolerance", type=float, default=0.018)
    parser.add_argument("--tcp-tolerance", type=float, default=0.022)
    parser.add_argument("--minimum-corridor-pixels", type=int, default=8)
    parser.add_argument("--maximum-obstacle-ratio", type=float, default=0.20)
    parser.add_argument("--maximum-target-spread", type=float, default=0.008)
    parser.add_argument("--maximum-tcp-spread", type=float, default=0.006)
    parser.add_argument("--maximum-head-spread", type=float, default=0.020)
    parser.add_argument("--maximum-identity-error", type=float, default=0.120)
    parser.add_argument("--maximum-cartesian-step", type=float, default=0.030)
    parser.add_argument("--minimum-cartesian-step", type=float, default=0.005)
    parser.add_argument("--residual-distance", type=float, default=0.018)
    parser.add_argument("--maximum-start-error", type=float, default=0.100)
    parser.add_argument("--maximum-joint-step-deg", type=float, default=8.0)
    parser.add_argument("--maximum-ik-error", type=float, default=0.004)
    parser.add_argument("--maximum-ik-iterations", type=int, default=35)
    parser.add_argument("--maximum-baseline-drift-deg", type=float, default=0.15)
    parser.add_argument("--maximum-uncommanded-motion-deg", type=float, default=1.5)
    parser.add_argument("--maximum-axis-change-deg", type=float, default=12.0)
    parser.add_argument("--maximum-finger-gap-change", type=float, default=0.010)
    parser.add_argument("--maximum-segment-worsening", type=float, default=0.005)
    parser.add_argument("--motion-seconds", type=float, default=4.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--vision-settle-seconds", type=float, default=1.0)
    parser.add_argument("--observation-timeout", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--hz", type=float, default=50.0)
    return parser


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


