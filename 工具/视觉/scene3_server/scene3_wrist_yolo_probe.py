#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation-only right-wrist YOLO/RGB-D probe for Scene3.

The senior task remains responsible for walking, target selection and the
pregrasp IK command.  This probe answers one question before any further arm
motion: can the existing tray detector find the tray directly in the right
wrist image?

For every wrist YOLO box it extracts the dominant depth surface, chooses the
surface edge nearest the legal TF gripper TCP, and transforms that point to
``base_link``.  Candidates are matched to the target latched by the senior
pregrasp wrapper using 3-D distance instead of confidence alone.  The script never
publishes a base, arm or claw command.
"""

from __future__ import print_function

import argparse
import math
import os
import threading
import time

import numpy as np

from scene3_visual_grasp_servo import select_grasp_pixel
from scene3_wrist_preclose_gate import (
    DEFAULT_DEPTH_TOPIC,
    DEFAULT_GRIPPER_BASE_FRAME,
    DEFAULT_INFO_TOPIC,
    DEFAULT_LEFT_FINGER_FRAME,
    DEFAULT_RGB_TOPIC,
    DEFAULT_RIGHT_FINGER_FRAME,
    decode_compressed_depth_payload,
    deproject_pixel,
    estimate_finger_tips,
    project_camera_point,
)


DEFAULT_MODEL_PATH = (
    "/root/kuavo_ws/src/challenge_cup_task_template/"
    "tray_train_finetune_2/weights/best.pt"
)
DEFAULT_DEBUG_TOPIC = (
    "/challenge_cup_task_template/scene3/wrist_yolo_debug/compressed"
)
DEFAULT_TARGET_TOPIC = "/challenge_cup_task_template/scene3/locked_target_base"
DEFAULT_TARGET_PARAM = "/challenge_cup_task_template/scene3/locked_target_base_xyz"


def normalize_class_name(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def clamp_xyxy_to_xywh(image_shape, xyxy):
    """Convert a detector xyxy box to a non-empty in-image xywh box."""
    height, width = [int(value) for value in image_shape[:2]]
    if height <= 0 or width <= 0:
        raise ValueError("image shape is invalid")
    values = np.asarray(xyxy, dtype=float).reshape(-1)
    if values.size != 4 or not np.all(np.isfinite(values)):
        raise ValueError("xyxy must contain four finite values")
    x1 = max(0, min(width - 1, int(round(values[0]))))
    y1 = max(0, min(height - 1, int(round(values[1]))))
    x2 = max(x1 + 1, min(width, int(round(values[2]))))
    y2 = max(y1 + 1, min(height, int(round(values[3]))))
    return x1, y1, x2 - x1, y2 - y1


def choose_target_candidate(candidates, locked_target_base=None,
                            maximum_match_distance_m=0.18):
    """Choose the wrist detection matching the senior target when available."""
    if not candidates:
        raise ValueError("no wrist YOLO tray candidate")
    if locked_target_base is None:
        return max(
            candidates,
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                int(item["bbox"][2]) * int(item["bbox"][3]),
            ),
        )

    locked = np.asarray(locked_target_base, dtype=float)
    if locked.shape != (3,) or not np.all(np.isfinite(locked)):
        raise ValueError("locked target must be finite xyz")
    ranked = []
    for candidate in candidates:
        base = np.asarray(candidate["target_base"], dtype=float)
        distance = float(np.linalg.norm(base - locked))
        item = dict(candidate)
        item["match_distance_m"] = distance
        ranked.append(item)
    selected = min(
        ranked,
        key=lambda item: (
            float(item["match_distance_m"]),
            -float(item.get("confidence", 0.0)),
        ),
    )
    if selected["match_distance_m"] > float(maximum_match_distance_m):
        raise ValueError(
            "nearest wrist tray is {:.3f}m from locked senior target".format(
                selected["match_distance_m"]
            )
        )
    return selected


def stable_target_observation(samples, required_frames=3,
                              maximum_spread_m=0.012):
    required = max(1, int(required_frames))
    if len(samples) < required:
        return None
    selected = list(samples[-required:])
    points = np.asarray([item["target_base"] for item in selected], dtype=float)
    centre = np.median(points, axis=0)
    spread = float(np.max(np.linalg.norm(points - centre, axis=1)))
    if spread > float(maximum_spread_m):
        return None
    return {
        "target_base": centre,
        "spread_m": spread,
        "confidence": float(np.median(
            [item["confidence"] for item in selected]
        )),
        "bbox": selected[-1]["bbox"],
        "grasp_pixel": selected[-1]["grasp_pixel"],
        "match_distance_m": selected[-1].get("match_distance_m"),
    }


def _translation_xyz(transform):
    value = transform.transform.translation
    return np.array([value.x, value.y, value.z], dtype=float)


def _class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))
    try:
        return str(names[int(class_id)])
    except Exception:
        return str(class_id)


class WristYoloProbe(object):
    def __init__(self, args, model, cv2, rospy, tf2_ros, PointStamped,
                 CameraInfo, CompressedImage):
        self.args = args
        self.model = model
        self.cv2 = cv2
        self.rospy = rospy
        self.PointStamped = PointStamped
        self.CompressedImage = CompressedImage
        self.lock = threading.Lock()
        self.info = None
        self.latest_target = None
        self.locked_target_base = None
        if args.target_param and rospy.has_param(args.target_param):
            locked = np.asarray(rospy.get_param(args.target_param), dtype=float)
            if locked.shape != (3,) or not np.all(np.isfinite(locked)):
                raise RuntimeError("locked senior target parameter is invalid")
            self.locked_target_base = locked
            print(
                "Loaded locked senior target base_link={}".format(
                    np.round(locked, 4).tolist()
                )
            )
        self.samples = []
        self.processed_frames = 0
        self.last_error = "waiting for synchronized wrist RGB-D"
        self.target_classes = {
            normalize_class_name(value)
            for value in str(args.target_classes).split(",")
            if str(value).strip()
        }
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.debug_pub = rospy.Publisher(
            args.debug_topic, CompressedImage, queue_size=1
        )
        rospy.Subscriber(args.info_topic, CameraInfo, self._info_callback,
                         queue_size=1)
        if args.target_topic:
            rospy.Subscriber(args.target_topic, PointStamped,
                             self._target_callback, queue_size=1)

    def _info_callback(self, message):
        with self.lock:
            self.info = message

    def _target_callback(self, message):
        with self.lock:
            self.latest_target = message

    def _transform_xyz(self, xyz, source_frame, target_frame):
        point = self.PointStamped()
        point.header.frame_id = str(source_frame)
        point.header.stamp = self.rospy.Time(0)
        point.point.x = float(xyz[0])
        point.point.y = float(xyz[1])
        point.point.z = float(xyz[2])
        transformed = self.tf_buffer.transform(
            point, str(target_frame), self.rospy.Duration(0.3)
        )
        return np.array(
            [transformed.point.x, transformed.point.y, transformed.point.z],
            dtype=float,
        )

    def _latch_optional_target(self):
        with self.lock:
            if self.locked_target_base is not None:
                return self.locked_target_base.copy()
            message = self.latest_target
        if message is None:
            return None
        message.header.stamp = self.rospy.Time(0)
        transformed = self.tf_buffer.transform(
            message, "base_link", self.rospy.Duration(0.3)
        )
        locked = np.array(
            [transformed.point.x, transformed.point.y, transformed.point.z],
            dtype=float,
        )
        with self.lock:
            if self.locked_target_base is None:
                self.locked_target_base = locked
                print(
                    "Latched senior target base_link={}".format(
                        np.round(locked, 4).tolist()
                    )
                )
            return self.locked_target_base.copy()

    def _tcp_camera(self, camera_frame):
        transforms = []
        for frame in (
            self.args.left_finger_frame,
            self.args.right_finger_frame,
            self.args.gripper_base_frame,
        ):
            transforms.append(
                self.tf_buffer.lookup_transform(
                    camera_frame, frame, self.rospy.Time(0),
                    self.rospy.Duration(0.3)
                )
            )
        return estimate_finger_tips(
            _translation_xyz(transforms[0]),
            _translation_xyz(transforms[1]),
            _translation_xyz(transforms[2]),
            extension_m=self.args.tcp_extension,
        )[2]

    def _detections(self, image, depth, camera_frame, camera_k, tcp_pixel):
        results = self.model.predict(
            image,
            conf=float(self.args.confidence),
            imgsz=int(self.args.imgsz),
            verbose=False,
            device=self.args.device or None,
        )
        candidates = []
        raw_count = 0
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            raw_count += len(xyxy)
            for index in range(len(xyxy)):
                class_name = _class_name(result.names, classes[index])
                if (
                    self.target_classes
                    and normalize_class_name(class_name) not in self.target_classes
                ):
                    continue
                bbox = clamp_xyxy_to_xywh(image.shape, xyxy[index])
                try:
                    refined = select_grasp_pixel(
                        depth,
                        bbox,
                        tcp_pixel=tcp_pixel,
                        depth_band_mm=self.args.depth_band,
                        inward_ratio=self.args.inward_ratio,
                    )
                    target_camera = deproject_pixel(
                        refined["pixel"], refined["depth_mm"], camera_k
                    )
                    target_base = self._transform_xyz(
                        target_camera, camera_frame, "base_link"
                    )
                except Exception as exc:
                    self.rospy.logwarn_throttle(
                        1.0, "wrist YOLO box rejected: %s", exc
                    )
                    continue
                candidates.append({
                    "bbox": bbox,
                    "confidence": float(confidences[index]),
                    "class_name": class_name,
                    "grasp_pixel": refined["pixel"],
                    "edge_pixel": refined["edge_pixel"],
                    "surface_pixels": refined["surface_pixels"],
                    "target_camera": target_camera,
                    "target_base": target_base,
                })
        return candidates, raw_count

    def _publish_debug(self, image, candidates, selected, tcp_pixel, header):
        canvas = image.copy()
        for candidate in candidates:
            x, y, width, height = candidate["bbox"]
            is_selected = candidate is selected
            colour = (0, 255, 0) if is_selected else (0, 180, 255)
            self.cv2.rectangle(canvas, (x, y), (x + width, y + height), colour, 2)
            label = "{} {:.2f}".format(
                candidate["class_name"], candidate["confidence"]
            )
            self.cv2.putText(
                canvas, label, (x, max(16, y - 5)),
                self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                self.cv2.LINE_AA,
            )
        if selected is not None:
            point = tuple(int(value) for value in selected["grasp_pixel"])
            self.cv2.circle(canvas, point, 5, (0, 0, 255), -1)
        if tcp_pixel is not None:
            point = tuple(int(round(value)) for value in tcp_pixel)
            self.cv2.circle(canvas, point, 5, (255, 0, 255), 2)
        ok, encoded = self.cv2.imencode(".jpg", canvas)
        if not ok:
            return
        message = self.CompressedImage()
        message.header = header
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self.debug_pub.publish(message)

    def synchronized_callback(self, rgb_message, depth_message):
        try:
            with self.lock:
                info = self.info
            if info is None:
                return
            image = self.cv2.imdecode(
                np.frombuffer(rgb_message.data, dtype=np.uint8),
                self.cv2.IMREAD_COLOR,
            )
            depth = decode_compressed_depth_payload(
                depth_message.data, self.cv2
            )
            if image is None:
                raise RuntimeError("cannot decode right wrist RGB")
            if image.shape[:2] != depth.shape[:2]:
                raise RuntimeError("right wrist RGB/depth sizes differ")
            camera_frame = str(self.args.camera_frame or info.header.frame_id)
            if not camera_frame:
                raise RuntimeError("right wrist camera frame is empty")
            camera_k = list(info.K)
            tcp_camera = self._tcp_camera(camera_frame)
            tcp_pixel = None
            try:
                tcp_pixel = project_camera_point(tcp_camera, camera_k)
            except Exception:
                pass
            locked = self._latch_optional_target()
            candidates, raw_count = self._detections(
                image, depth, camera_frame, camera_k, tcp_pixel
            )
            selected = choose_target_candidate(
                candidates,
                locked_target_base=locked,
                maximum_match_distance_m=self.args.maximum_match_distance,
            )
            sample = dict(selected)
            sample["stamp"] = (
                int(rgb_message.header.stamp.secs),
                int(rgb_message.header.stamp.nsecs),
            )
            with self.lock:
                if not self.samples or self.samples[-1]["stamp"] != sample["stamp"]:
                    self.samples.append(sample)
                    del self.samples[:-20]
                    self.processed_frames += 1
                    frame_index = self.processed_frames
                else:
                    frame_index = self.processed_frames
                self.last_error = ""
            print(
                "frame={} raw={} valid={} class={} conf={:.3f} bbox={} "
                "grasp_px={} base={}".format(
                    frame_index,
                    raw_count,
                    len(candidates),
                    selected["class_name"],
                    selected["confidence"],
                    list(selected["bbox"]),
                    list(selected["grasp_pixel"]),
                    np.round(selected["target_base"], 4).tolist(),
                )
            )
            self._publish_debug(
                image, candidates, selected, tcp_pixel, rgb_message.header
            )
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
            self.rospy.logwarn_throttle(1.0, "scene3 wrist YOLO probe: %s", exc)

    def wait_stable(self):
        deadline = time.time() + float(self.args.timeout)
        while not self.rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                samples = list(self.samples)
                last_error = self.last_error
            stable = stable_target_observation(
                samples,
                required_frames=self.args.required_frames,
                maximum_spread_m=self.args.maximum_target_spread,
            )
            if stable is not None:
                return stable
            self.rospy.sleep(0.05)
        raise RuntimeError(
            "WRIST_YOLO_NO_STABLE_TARGET: {}".format(
                last_error or "detections were unstable"
            )
        )


def run_ros(args):
    import cv2
    import message_filters
    import rospy
    import tf2_geometry_msgs  # noqa: F401
    import tf2_ros
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import CameraInfo, CompressedImage
    from ultralytics import YOLO

    if not os.path.isfile(args.model_path):
        raise RuntimeError("wrist YOLO model does not exist: {}".format(
            args.model_path
        ))
    rospy.init_node("scene3_wrist_yolo_probe", anonymous=True)
    print("Loading wrist YOLO model: {}".format(args.model_path))
    model = YOLO(args.model_path)
    if args.device:
        model.to(args.device)
    print("Wrist YOLO classes: {}".format(model.names))

    probe = WristYoloProbe(
        args, model, cv2, rospy, tf2_ros, PointStamped, CameraInfo,
        CompressedImage,
    )
    rgb_sub = message_filters.Subscriber(args.rgb_topic, CompressedImage)
    depth_sub = message_filters.Subscriber(args.depth_topic, CompressedImage)
    synchronizer = message_filters.ApproximateTimeSynchronizer(
        [rgb_sub, depth_sub], queue_size=8, slop=0.08
    )
    synchronizer.registerCallback(probe.synchronized_callback)

    print("Waiting for three stable wrist YOLO/RGB-D tray observations")
    stable = probe.wait_stable()
    print("Wrist tray base_link:", np.round(stable["target_base"], 4).tolist())
    print("Three-frame spread: {:.4f}m".format(stable["spread_m"]))
    print("Median confidence: {:.3f}".format(stable["confidence"]))
    print("Selected bbox:", list(stable["bbox"]))
    print("Selected grasp pixel:", list(stable["grasp_pixel"]))
    if stable["match_distance_m"] is not None:
        print("Senior-target match: {:.4f}m".format(
            stable["match_distance_m"]
        ))
    print("WRIST_YOLO_TARGET_OK: observation only; no command sent")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--target-classes", default="tray,smt_tray")
    parser.add_argument("--rgb-topic", default=DEFAULT_RGB_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--info-topic", default=DEFAULT_INFO_TOPIC)
    parser.add_argument(
        "--target-topic",
        default=DEFAULT_TARGET_TOPIC,
        help="latched senior target PointStamped topic",
    )
    parser.add_argument("--target-param", default=DEFAULT_TARGET_PARAM)
    parser.add_argument("--debug-topic", default=DEFAULT_DEBUG_TOPIC)
    parser.add_argument("--camera-frame", default="right_wrist_camera_link")
    parser.add_argument("--left-finger-frame", default=DEFAULT_LEFT_FINGER_FRAME)
    parser.add_argument("--right-finger-frame", default=DEFAULT_RIGHT_FINGER_FRAME)
    parser.add_argument("--gripper-base-frame", default=DEFAULT_GRIPPER_BASE_FRAME)
    parser.add_argument("--tcp-extension", type=float, default=0.045)
    parser.add_argument("--depth-band", type=float, default=25.0)
    parser.add_argument("--inward-ratio", type=float, default=0.14)
    parser.add_argument("--maximum-match-distance", type=float, default=0.18)
    parser.add_argument("--required-frames", type=int, default=3)
    parser.add_argument("--maximum-target-spread", type=float, default=0.012)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
