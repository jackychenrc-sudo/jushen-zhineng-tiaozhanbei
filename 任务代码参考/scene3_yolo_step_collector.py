#!/usr/bin/env python3
"""Step-and-capture Scene3 RGB-D collector for YOLO training."""

import argparse
import json
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo, CompressedImage


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

CAMERAS = {
    "head": {
        "rgb": "/cam_h/color/image_raw/compressed",
        "depth": "/cam_h/depth/image_raw/compressedDepth",
        "info": "/cam_h/color/camera_info",
    },
    "left": {
        "rgb": "/cam_l/color/image_raw/compressed",
        "depth": "/cam_l/depth/image_rect_raw/compressedDepth",
        "info": "/cam_l/color/camera_info",
    },
    "right": {
        "rgb": "/cam_r/color/image_raw/compressed",
        "depth": "/cam_r/depth/image_rect_raw/compressedDepth",
        "info": "/cam_r/color/camera_info",
    },
}


def stamp_to_dict(stamp):
    return {
        "secs": int(stamp.secs),
        "nsecs": int(stamp.nsecs),
        "seconds": float(stamp.to_sec()),
    }


def camera_info_to_dict(message):
    return {
        "frame_id": message.header.frame_id,
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": message.distortion_model,
        "D": list(message.D),
        "K": list(message.K),
        "R": list(message.R),
        "P": list(message.P),
    }


def decode_rgb(message):
    encoded = np.frombuffer(message.data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode RGB image")
    return image


def decode_depth(message):
    raw = bytes(message.data)
    png_offset = raw.find(PNG_SIGNATURE)
    if png_offset < 0:
        raise RuntimeError("compressedDepth payload does not contain a PNG image")

    encoded = np.frombuffer(raw[png_offset:], dtype=np.uint8)
    depth_png = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        raise RuntimeError("failed to decode depth PNG")

    image_format = message.format or ""
    if image_format.startswith("32FC1"):
        if png_offset < 12:
            raise RuntimeError("32FC1 compressedDepth header is incomplete")
        _, depth_quant_a, depth_quant_b = struct.unpack("<iff", raw[:12])
        inverse_depth = depth_png.astype(np.float32)
        depth_m = np.full(inverse_depth.shape, np.nan, dtype=np.float32)
        valid = inverse_depth > 0
        depth_m[valid] = depth_quant_a / (inverse_depth[valid] - depth_quant_b)
        return depth_m

    depth_m = depth_png.astype(np.float32)
    valid = depth_m > 0
    if np.any(valid) and float(np.nanmedian(depth_m[valid])) > 20.0:
        depth_m *= 0.001
    depth_m[~valid] = np.nan
    return depth_m


def make_depth_preview(depth_m):
    valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 10.0)
    preview = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return preview
    low, high = np.percentile(depth_m[valid], [2, 98])
    if high <= low:
        high = low + 0.01
    scaled = np.clip((depth_m - low) / (high - low), 0.0, 1.0)
    grayscale = np.zeros(depth_m.shape, dtype=np.uint8)
    grayscale[valid] = np.round((1.0 - scaled[valid]) * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def save_image(path, image, params=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, params or [])
    if not ok:
        raise RuntimeError("failed to write image: {}".format(path))


class SyncedFrameBuffer(object):
    def __init__(self, camera_topics, sync_slop):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_seq = 0

        self.rgb_subscriber = message_filters.Subscriber(
            camera_topics["rgb"],
            CompressedImage,
        )
        self.depth_subscriber = message_filters.Subscriber(
            camera_topics["depth"],
            CompressedImage,
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_subscriber, self.depth_subscriber],
            queue_size=20,
            slop=sync_slop,
        )
        self.sync.registerCallback(self._callback)

    def _callback(self, rgb_message, depth_message):
        with self.condition:
            self.latest_rgb = rgb_message
            self.latest_depth = depth_message
            self.latest_seq += 1
            self.condition.notify_all()

    def wait_for_next_frame(self, last_seq, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.latest_seq <= last_seq and not rospy.is_shutdown():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("timed out waiting for synchronized RGB-D frame")
                self.condition.wait(remaining)
            if self.latest_rgb is None or self.latest_depth is None:
                raise RuntimeError("no synchronized RGB-D frame available")
            return self.latest_seq, self.latest_rgb, self.latest_depth


class StepCollector(object):
    def __init__(self, args, camera_info):
        self.args = args
        self.camera_info = camera_info
        self.camera_topics = CAMERAS[args.camera]
        if args.output_dir:
            self.dataset_root = Path(args.output_dir)
            self.run_dir = self.dataset_root
        else:
            run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
            self.dataset_root = Path(args.output_root)
            self.run_dir = self.dataset_root / args.split / run_name
        self.images_dir = self.run_dir / "images"
        self.depth_dir = self.run_dir / "depth"
        self.depth_preview_dir = self.run_dir / "depth_preview"
        self.metadata_dir = self.run_dir / "metadata"
        self.labels_dir = self.run_dir / "labels"
        for directory in (
            self.images_dir,
            self.depth_dir,
            self.depth_preview_dir,
            self.metadata_dir,
            self.labels_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.frame_buffer = SyncedFrameBuffer(self.camera_topics, args.sync_slop)
        self.cmd_pub = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=10)
        self.records = []
        self.capture_index = 0

        self.control_mode_enabled = False
        if args.try_set_control_mode:
            try:
                import lb_ctrl_api as ct  # pylint: disable=import-error

                ct.set_control_mode(2)
                self.control_mode_enabled = True
                rospy.loginfo("scene3 step collector: set control mode to 2")
            except Exception as exc:
                rospy.logwarn("scene3 step collector: failed to set control mode: %s", exc)

    def publish_stop(self, repeats=5):
        stop = Twist()
        for _ in range(repeats):
            self.cmd_pub.publish(stop)
            rospy.sleep(0.05)

    def execute_twist_for_duration(self, linear_x, linear_y, angular_z, duration):
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        rate = rospy.Rate(self.args.cmd_rate)
        publish_count = max(1, int(round(duration * self.args.cmd_rate)))
        for _ in range(publish_count):
            if rospy.is_shutdown():
                break
            self.cmd_pub.publish(twist)
            rate.sleep()
        self.publish_stop(repeats=max(5, int(self.args.cmd_rate * 0.2)))

    def execute_initial_forward(self):
        if self.args.initial_forward_distance <= 0.0:
            return
        duration = self.args.initial_forward_distance / self.args.initial_forward_speed
        rospy.loginfo(
            "scene3 step collector: initial forward %.3f m at %.3f m/s (%.3f s)",
            self.args.initial_forward_distance,
            self.args.initial_forward_speed,
            duration,
        )
        self.execute_twist_for_duration(
            linear_x=self.args.initial_forward_speed,
            linear_y=0.0,
            angular_z=0.0,
            duration=duration,
        )
        if self.args.initial_settle_time > 0.0:
            rospy.sleep(self.args.initial_settle_time)

    def execute_move(self, move_index):
        move_linear_y = self.args.linear_y
        if self.args.alternate_lateral:
            lateral_speed = abs(self.args.linear_y)
            move_linear_y = lateral_speed if move_index % 2 == 0 else -lateral_speed
        self.execute_twist_for_duration(
            linear_x=self.args.linear_x,
            linear_y=move_linear_y,
            angular_z=self.args.angular_z,
            duration=self.args.move_duration,
        )

    def capture_frame(self, move_index, capture_in_move, last_seq):
        seq, rgb_message, depth_message = self.frame_buffer.wait_for_next_frame(
            last_seq=last_seq,
            timeout=self.args.frame_timeout,
        )
        rgb = decode_rgb(rgb_message)
        depth_m = decode_depth(depth_message)
        if rgb.shape[:2] != depth_m.shape[:2]:
            depth_m = cv2.resize(
                depth_m,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        frame_name = "{}_{:05d}".format(self.args.prefix, self.capture_index)
        rgb_path = self.images_dir / (frame_name + ".jpg")
        depth_npz_path = self.depth_dir / (frame_name + ".npz")
        depth_png_path = self.depth_dir / (frame_name + "_mm.png")
        depth_preview_path = self.depth_preview_dir / (frame_name + ".jpg")
        metadata_path = self.metadata_dir / (frame_name + ".json")
        label_path = self.labels_dir / (frame_name + ".txt")

        valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 10.0)
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.clip(
            np.round(depth_m[valid] * 1000.0),
            1,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)

        save_image(rgb_path, rgb, [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
        save_image(depth_png_path, depth_mm)
        save_image(depth_preview_path, make_depth_preview(depth_m))
        np.savez_compressed(str(depth_npz_path), depth_m=depth_m)
        if self.args.create_empty_labels and not label_path.exists():
            label_path.write_text("", encoding="utf-8")

        depth_stats = {"valid_pixels": int(np.count_nonzero(valid))}
        if np.any(valid):
            percentiles = np.percentile(depth_m[valid], [5, 50, 95])
            depth_stats.update(
                {
                    "p05_m": float(percentiles[0]),
                    "median_m": float(percentiles[1]),
                    "p95_m": float(percentiles[2]),
                }
            )

        record = {
            "frame_index": self.capture_index,
            "camera": self.args.camera,
            "split": self.args.split,
            "run_name": self.run_dir.name,
            "move_index": move_index,
            "capture_in_move": capture_in_move,
            "captures_per_move": self.args.captures_per_move,
            "rgb_stamp": stamp_to_dict(rgb_message.header.stamp),
            "depth_stamp": stamp_to_dict(depth_message.header.stamp),
            "stamp_delta_seconds": abs(
                rgb_message.header.stamp.to_sec() - depth_message.header.stamp.to_sec()
            ),
            "motion_command": {
                "alternate_lateral": self.args.alternate_lateral,
                "linear_x": self.args.linear_x,
                "linear_y": self.args.linear_y,
                "angular_z": self.args.angular_z,
                "move_duration": self.args.move_duration,
                "settle_time": self.args.settle_time,
            },
            "files": {
                "rgb": str(rgb_path),
                "depth_npz_m": str(depth_npz_path),
                "depth_png_mm": str(depth_png_path),
                "depth_preview": str(depth_preview_path),
                "metadata": str(metadata_path),
                "label": str(label_path),
            },
            "camera_info": camera_info_to_dict(self.camera_info),
            "depth_stats": depth_stats,
        }
        metadata_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.records.append(record)
        self.capture_index += 1
        rospy.loginfo(
            "scene3 step collector: move %d/%d capture %d/%d -> %s",
            move_index + 1,
            self.args.moves,
            capture_in_move + 1,
            self.args.captures_per_move,
            rgb_path.name,
        )
        return seq

    def run(self):
        total_expected = self.args.moves * self.args.captures_per_move
        last_seq = 0
        last_seq, _, _ = self.frame_buffer.wait_for_next_frame(
            last_seq=last_seq,
            timeout=self.args.frame_timeout,
        )
        rospy.loginfo("scene3 step collector: first synchronized frame received")
        self.execute_initial_forward()

        for move_index in range(self.args.moves):
            if rospy.is_shutdown():
                break
            rospy.loginfo(
                "scene3 step collector: executing move %d/%d",
                move_index + 1,
                self.args.moves,
            )
            self.execute_move(move_index)
            rospy.sleep(self.args.settle_time)
            for capture_in_move in range(self.args.captures_per_move):
                last_seq = self.capture_frame(move_index, capture_in_move, last_seq)
                if capture_in_move + 1 < self.args.captures_per_move:
                    rospy.sleep(self.args.capture_gap)

        self.publish_stop(repeats=10)
        rospy.loginfo(
            "scene3 step collector: finished %d captures",
            len(self.records),
        )
        if len(self.records) != total_expected:
            raise RuntimeError(
                "expected {} captures but saved {}".format(
                    total_expected,
                    len(self.records),
                )
            )

    def write_manifest(self):
        manifest = {
            "scene": "scene3",
            "collector": "step_collector",
            "split": self.args.split,
            "camera": self.args.camera,
            "moves": self.args.moves,
            "captures_per_move": self.args.captures_per_move,
            "count": len(self.records),
            "motion_command": {
                "initial_forward_distance": self.args.initial_forward_distance,
                "initial_forward_speed": self.args.initial_forward_speed,
                "initial_settle_time": self.args.initial_settle_time,
                "linear_x": self.args.linear_x,
                "linear_y": self.args.linear_y,
                "angular_z": self.args.angular_z,
                "move_duration": self.args.move_duration,
                "settle_time": self.args.settle_time,
                "capture_gap": self.args.capture_gap,
                "cmd_rate": self.args.cmd_rate,
                "cmd_vel_topic": self.args.cmd_vel_topic,
            },
            "topics": self.camera_topics,
            "run_dir": str(self.run_dir),
            "frames": self.records,
        }
        manifest_path = self.run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step-and-capture Scene3 YOLO dataset collector"
    )
    parser.add_argument("--camera", choices=sorted(CAMERAS), default="head")
    parser.add_argument("--moves", type=int, default=10)
    parser.add_argument("--captures-per-move", type=int, default=2)
    parser.add_argument("--move-duration", type=float, default=1.0)
    parser.add_argument("--settle-time", type=float, default=0.8)
    parser.add_argument("--capture-gap", type=float, default=0.25)
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--cmd-rate", type=float, default=30.0)
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--initial-forward-distance", type=float, default=0.70)
    parser.add_argument("--initial-forward-speed", type=float, default=0.08)
    parser.add_argument("--initial-settle-time", type=float, default=1.0)
    parser.add_argument("--linear-x", type=float, default=0.0)
    parser.add_argument("--linear-y", type=float, default=0.08)
    parser.add_argument("--angular-z", type=float, default=0.0)
    parser.add_argument(
        "--alternate-lateral",
        dest="alternate_lateral",
        action="store_true",
        help="alternate left/right lateral motion during collection",
    )
    parser.add_argument(
        "--no-alternate-lateral",
        dest="alternate_lateral",
        action="store_false",
        help="disable alternating lateral motion during collection",
    )
    parser.set_defaults(alternate_lateral=True)
    parser.add_argument("--sync-slop", type=float, default=0.08)
    parser.add_argument("--output-root", default="scene3_yolo_dataset")
    parser.add_argument(
        "--output-dir",
        default="/home/iuucb/datasets/scene3_tray/step_run_02",
        help="write outputs directly into this directory",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--prefix", default="scene3")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--create-empty-labels", action="store_true")
    parser.add_argument("--try-set-control-mode", action="store_true")
    args = parser.parse_args(rospy.myargv()[1:])
    if args.moves <= 0 or args.captures_per_move <= 0:
        parser.error("moves and captures-per-move must be positive")
    if (
        args.move_duration <= 0.0
        or args.settle_time < 0.0
        or args.capture_gap < 0.0
        or args.frame_timeout <= 0.0
        or args.cmd_rate <= 0.0
        or args.sync_slop <= 0.0
        or args.initial_forward_distance < 0.0
        or args.initial_forward_speed <= 0.0
        or args.initial_settle_time < 0.0
    ):
        parser.error("timing arguments must be positive")
    if not (10 <= args.jpeg_quality <= 100):
        parser.error("jpeg-quality must be within [10, 100]")
    if "/" in args.split or "\\" in args.split:
        parser.error("split must be a single directory name")
    if args.run_name and ("/" in args.run_name or "\\" in args.run_name):
        parser.error("run-name must be a single directory name")
    return args


def main():
    args = parse_args()
    rospy.init_node("scene3_yolo_step_collector", anonymous=True)
    camera_info = rospy.wait_for_message(
        CAMERAS[args.camera]["info"],
        CameraInfo,
        timeout=15.0,
    )
    collector = StepCollector(args, camera_info)
    try:
        collector.run()
    finally:
        collector.publish_stop(repeats=10)
    manifest_path = collector.write_manifest()
    print("saved_frames={}".format(len(collector.records)))
    print("run_dir={}".format(collector.run_dir))
    print("manifest={}".format(manifest_path))


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSException, RuntimeError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
