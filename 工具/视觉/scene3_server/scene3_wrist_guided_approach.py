#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One bounded wrist-guided approach step after the senior Scene3 pregrasp.

This script does not replace the senior movement or IK implementation.  It
uses right-wrist RGB-D once to latch the tray point in ``base_link``, rebuilds
the senior pregrasp/touch targets from that point, and advances at most 6 cm
along the senior pregrasp-to-touch segment.  It then re-observes the tray and
TCP.  No claw-close, lift, extraction or base command is sent.

Default mode is calculation-only.  Real arm movement requires ``--execute``
and the exact confirmation token ``WRIST_APPROACH_6CM``.
"""

from __future__ import print_function

import argparse
import os
import sys
import threading
import time

import numpy as np

from scene3_wrist_preclose_gate import (
    DEFAULT_DEPTH_TOPIC,
    DEFAULT_GRIPPER_BASE_FRAME,
    DEFAULT_INFO_TOPIC,
    DEFAULT_LEFT_FINGER_FRAME,
    DEFAULT_RIGHT_FINGER_FRAME,
    DEFAULT_TARGET_TOPIC,
    decode_compressed_depth_payload,
    deproject_pixel,
    estimate_finger_tips,
    project_camera_point,
    prompted_depth_component,
)


EXECUTION_CONFIRMATION = "WRIST_APPROACH_6CM"
DEFAULT_SENIOR_DIR = "/root/kuavo_ws/src/challenge_cup_task_template/scripts"


def plan_bounded_senior_approach(pregrasp_xyz, touch_xyz, maximum_step_m=0.06):
    pregrasp = np.asarray(pregrasp_xyz, dtype=float)
    touch = np.asarray(touch_xyz, dtype=float)
    if pregrasp.shape != (3,) or touch.shape != (3,):
        raise ValueError("pregrasp and touch must be xyz")
    delta = touch - pregrasp
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        raise ValueError("senior pregrasp and touch targets are identical")
    step_length = min(distance, max(0.005, float(maximum_step_m)))
    target = pregrasp + delta * (step_length / distance)
    return target, step_length, distance


def validate_observed_progress(
    before_error_m,
    after_error_m,
    tcp_motion_m,
    minimum_error_reduction_m=0.025,
    minimum_tcp_motion_m=0.015,
    maximum_tcp_motion_m=0.085,
):
    before = float(before_error_m)
    after = float(after_error_m)
    motion = float(tcp_motion_m)
    checks = {
        "error_reduced": before - after >= float(minimum_error_reduction_m),
        "tcp_moved": motion >= float(minimum_tcp_motion_m),
        "motion_bounded": motion <= float(maximum_tcp_motion_m),
        "error_finite": np.isfinite(before) and np.isfinite(after),
    }
    return bool(all(checks.values())), checks


def _translation_xyz(transform):
    value = transform.transform.translation
    return np.array([value.x, value.y, value.z], dtype=float)


class WristTargetObserver(object):
    def __init__(self, args, rospy, tf2_ros, PointStamped, CameraInfo, CompressedImage):
        self.args = args
        self.rospy = rospy
        self.PointStamped = PointStamped
        self.lock = threading.Lock()
        self.info = None
        self.head_target = None
        self.latched_target_base = None
        self.observations = []
        self.last_error = "waiting for input"
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.Subscriber(args.info_topic, CameraInfo, self._info_callback, queue_size=1)
        rospy.Subscriber(args.target_topic, PointStamped, self._target_callback, queue_size=1)
        rospy.Subscriber(args.depth_topic, CompressedImage, self._depth_callback, queue_size=1)

    def _info_callback(self, message):
        with self.lock:
            self.info = message

    def _target_callback(self, message):
        with self.lock:
            self.head_target = message

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
            latched = self.latched_target_base
            head = self.head_target
        if latched is not None:
            message = self.PointStamped()
            message.header.frame_id = "base_link"
            message.header.stamp = self.rospy.Time(0)
            message.point.x = float(latched[0])
            message.point.y = float(latched[1])
            message.point.z = float(latched[2])
            return message
        return head

    def _depth_callback(self, message):
        try:
            import cv2

            with self.lock:
                info = self.info
            source_target = self._source_target()
            if info is None or source_target is None:
                return
            camera_frame = str(self.args.camera_frame or info.header.frame_id)
            source_target.header.stamp = self.rospy.Time(0)
            target_camera_message = self.tf_buffer.transform(
                source_target, camera_frame, self.rospy.Duration(0.3)
            )
            expected_camera = np.array(
                [
                    target_camera_message.point.x,
                    target_camera_message.point.y,
                    target_camera_message.point.z,
                ],
                dtype=float,
            )
            camera_k = list(info.K)
            prompt = project_camera_point(expected_camera, camera_k)
            depth = decode_compressed_depth_payload(message.data, cv2)
            component = prompted_depth_component(
                depth,
                prompt,
                expected_camera[2] * 1000.0,
                roi_radius_px=self.args.roi_radius,
                depth_band_mm=self.args.depth_band,
                minimum_pixels=self.args.minimum_component_pixels,
            )
            target_camera = deproject_pixel(
                component["target_pixel"], component["target_depth_mm"], camera_k
            )
            target_base = self._transform_xyz(
                target_camera, camera_frame, "base_link"
            )
            with self.lock:
                if self.latched_target_base is None:
                    self.latched_target_base = target_base.copy()
                    print(
                        "Latched wrist tray point in base_link: {}".format(
                            np.round(target_base, 4).tolist()
                        )
                    )

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
            _, _, tcp_camera, _ = estimate_finger_tips(
                _translation_xyz(transforms[0]),
                _translation_xyz(transforms[1]),
                _translation_xyz(transforms[2]),
                extension_m=self.args.tcp_extension,
            )
            tcp_base = self._transform_xyz(tcp_camera, camera_frame, "base_link")
            observation = {
                "stamp": (int(message.header.stamp.secs), int(message.header.stamp.nsecs)),
                "wall_time": time.time(),
                "target_base": target_base,
                "tcp_base": tcp_base,
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
            self.rospy.logwarn_throttle(1.0, "scene3 wrist approach observer: %s", exc)

    def clear(self):
        with self.lock:
            self.observations = []
            self.last_error = "waiting for post-motion frame"

    def wait_stable(self, timeout=12.0, sample_count=3):
        deadline = time.time() + float(timeout)
        while not self.rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                samples = list(self.observations[-int(sample_count) :])
                last_error = self.last_error
            if len(samples) >= int(sample_count):
                targets = np.asarray(
                    [sample["target_base"] for sample in samples], dtype=float
                )
                tcps = np.asarray(
                    [sample["tcp_base"] for sample in samples], dtype=float
                )
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
                        "target_spread": target_spread,
                        "tcp_spread": tcp_spread,
                    }
            self.rospy.sleep(0.05)
        raise RuntimeError(
            "no stable wrist observations: {}".format(last_error or "unstable")
        )


def load_senior_task(senior_dir):
    senior_dir = os.path.abspath(senior_dir)
    if senior_dir in sys.path:
        sys.path.remove(senior_dir)
    sys.path.insert(0, senior_dir)
    from challenge_task_3 import Scene3Task

    return Scene3Task


def run_ros(args):
    import rospy
    import tf2_geometry_msgs  # noqa: F401
    import tf2_ros
    from geometry_msgs.msg import PointStamped, Twist
    from sensor_msgs.msg import CameraInfo, CompressedImage, JointState

    rospy.init_node("scene3_wrist_guided_approach", anonymous=True)
    observer = WristTargetObserver(
        args, rospy, tf2_ros, PointStamped, CameraInfo, CompressedImage
    )
    print("Waiting for three stable right-wrist tray/TCP observations")
    before = observer.wait_stable(timeout=args.observation_timeout)
    before_error = float(np.linalg.norm(before["target"] - before["tcp"]))

    Scene3Task = load_senior_task(args.senior_dir)
    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    task = Scene3Task(cmd_vel_pub, arm_traj_pub)
    target_message = PointStamped()
    target_message.header.frame_id = "base_link"
    target_message.header.stamp = rospy.Time(0)
    target_message.point.x = float(before["target"][0])
    target_message.point.y = float(before["target"][1])
    target_message.point.z = float(before["target"][2])
    senior_targets = task.build_scene3_grasp_targets(target_message)
    stage_target, command_step, segment_length = plan_bounded_senior_approach(
        senior_targets["pregrasp"],
        senior_targets["touch"],
        maximum_step_m=args.maximum_command_step,
    )
    print("Wrist target base:", np.round(before["target"], 4).tolist())
    print("TCP base:", np.round(before["tcp"], 4).tolist())
    print("TCP error before: {:.4f}m".format(before_error))
    print(
        "Senior segment: pregrasp={} touch={} length={:.4f}m".format(
            np.round(senior_targets["pregrasp"], 4).tolist(),
            np.round(senior_targets["touch"], 4).tolist(),
            segment_length,
        )
    )
    print(
        "Bounded stage target={} command_step={:.4f}m".format(
            np.round(stage_target, 4).tolist(), command_step
        )
    )

    if not args.execute:
        print("WRIST_APPROACH_DRY_RUN_OK: calculation only; no command sent")
        return 0
    if args.confirmation != EXECUTION_CONFIRMATION:
        raise RuntimeError(
            "execution blocked; pass --confirmation {}".format(
                EXECUTION_CONFIRMATION
            )
        )
    if not (0.08 <= before_error <= 0.20):
        raise RuntimeError("starting TCP error is outside pregrasp safety range")

    task.stop_base()
    task.wait_for_arm_subscriber(timeout=8.0)
    if not task.set_arm_mode(2):
        raise RuntimeError("cannot enable senior arm external-control mode")
    if not task.open_claw():
        raise RuntimeError("cannot confirm open claw")
    observer.clear()
    print("Executing one bounded senior-IK approach step; claw remains open")
    task.move_right_hand(stage_target.tolist(), duration=args.motion_seconds)
    task.stop_base()
    rospy.sleep(args.settle_seconds)

    after = observer.wait_stable(timeout=args.observation_timeout)
    after_error = float(np.linalg.norm(after["target"] - after["tcp"]))
    tcp_motion = float(np.linalg.norm(after["tcp"] - before["tcp"]))
    progress_ok, checks = validate_observed_progress(
        before_error,
        after_error,
        tcp_motion,
        minimum_error_reduction_m=args.minimum_error_reduction,
        maximum_tcp_motion_m=args.maximum_observed_motion,
    )
    print("TCP base after:", np.round(after["tcp"], 4).tolist())
    print("TCP error after: {:.4f}m".format(after_error))
    print("Observed TCP motion: {:.4f}m".format(tcp_motion))
    print("Progress checks:", checks)
    if not progress_ok:
        raise RuntimeError(
            "WRIST_APPROACH_STEP_BLOCKED: observed motion did not follow plan; claw remains open"
        )
    print(
        "WRIST_APPROACH_STEP_OK: error {:.4f}m -> {:.4f}m; claw remains open".format(
            before_error, after_error
        )
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--senior-dir", default=DEFAULT_SENIOR_DIR)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--info-topic", default=DEFAULT_INFO_TOPIC)
    parser.add_argument("--target-topic", default=DEFAULT_TARGET_TOPIC)
    parser.add_argument("--camera-frame", default="right_wrist_camera_link")
    parser.add_argument("--left-finger-frame", default=DEFAULT_LEFT_FINGER_FRAME)
    parser.add_argument("--right-finger-frame", default=DEFAULT_RIGHT_FINGER_FRAME)
    parser.add_argument("--gripper-base-frame", default=DEFAULT_GRIPPER_BASE_FRAME)
    parser.add_argument("--tcp-extension", type=float, default=0.045)
    parser.add_argument("--roi-radius", type=int, default=90)
    parser.add_argument("--depth-band", type=float, default=35.0)
    parser.add_argument("--minimum-component-pixels", type=int, default=30)
    parser.add_argument("--maximum-target-spread", type=float, default=0.008)
    parser.add_argument("--maximum-tcp-spread", type=float, default=0.006)
    parser.add_argument("--maximum-command-step", type=float, default=0.060)
    parser.add_argument("--maximum-observed-motion", type=float, default=0.085)
    parser.add_argument("--minimum-error-reduction", type=float, default=0.025)
    parser.add_argument("--motion-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--observation-timeout", type=float, default=12.0)
    return parser


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

