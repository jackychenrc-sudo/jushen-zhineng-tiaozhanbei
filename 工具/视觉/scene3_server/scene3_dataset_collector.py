#!/usr/bin/env python3
"""Collect synchronized Scene3 RGB-D frames without moving the robot."""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo, CompressedImage

from scene3_live_perception import decode_depth, decode_rgb


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
    return {"secs": int(stamp.secs), "nsecs": int(stamp.nsecs), "seconds": stamp.to_sec()}


def transform_to_dict(transform):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return {
        "source_frame": transform.child_frame_id,
        "target_frame": transform.header.frame_id,
        "stamp": stamp_to_dict(transform.header.stamp),
        "translation_m": [translation.x, translation.y, translation.z],
        "quaternion_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
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


def depth_preview(depth_m):
    valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 10.0)
    preview = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return preview
    low, high = np.percentile(depth_m[valid], [2, 98])
    if high <= low:
        high = low + 0.01
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    scaled = (np.clip(depth_m, low, high) - low) / (high - low)
    normalized[valid] = np.round((1.0 - scaled[valid]) * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def write_image(path, image, params=None):
    success = cv2.imwrite(str(path), image, params or [])
    if not success:
        raise RuntimeError("failed to write image: {}".format(path))


class DatasetCollector:
    def __init__(self, args, camera_info, tf_buffer):
        self.args = args
        self.camera_info = camera_info
        self.tf_buffer = tf_buffer
        self.camera_topics = CAMERAS[args.camera]
        run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
        self.output_dir = (
            Path(args.output_root)
            / "seed_{:03d}".format(args.seed)
            / args.camera
            / run_name
        )
        for name in ("rgb", "depth", "depth_preview", "metadata"):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)

        self.records = []
        self.last_saved_at = -float("inf")
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.error = None

        self.rgb_subscriber = message_filters.Subscriber(
            self.camera_topics["rgb"], CompressedImage
        )
        self.depth_subscriber = message_filters.Subscriber(
            self.camera_topics["depth"], CompressedImage
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_subscriber, self.depth_subscriber],
            queue_size=20,
            slop=args.sync_slop,
        )
        self.synchronizer.registerCallback(self.callback)

    def callback(self, rgb_message, depth_message):
        now = time.monotonic()
        if now - self.last_saved_at < self.args.interval:
            return
        if not self.lock.acquire(False):
            return
        try:
            if len(self.records) >= self.args.count:
                self.done.set()
                return
            self.save_frame(rgb_message, depth_message)
            self.last_saved_at = now
            if len(self.records) >= self.args.count:
                self.done.set()
        except Exception as error:
            self.error = error
            self.done.set()
        finally:
            self.lock.release()

    def save_frame(self, rgb_message, depth_message):
        image = decode_rgb(rgb_message)
        depth_m = decode_depth(depth_message)
        if image.shape[:2] != depth_m.shape[:2]:
            raise RuntimeError(
                "RGB/depth dimensions differ: {} vs {}".format(
                    image.shape[:2], depth_m.shape[:2]
                )
            )

        index = len(self.records)
        frame_name = "seed_{:03d}_{}_frame_{:03d}".format(
            self.args.seed, self.args.camera, index
        )
        rgb_path = self.output_dir / "rgb" / (frame_name + ".jpg")
        depth_npz_path = self.output_dir / "depth" / (frame_name + ".npz")
        depth_png_path = self.output_dir / "depth" / (frame_name + "_mm.png")
        preview_path = self.output_dir / "depth_preview" / (frame_name + ".jpg")
        metadata_path = self.output_dir / "metadata" / (frame_name + ".json")

        valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 10.0)
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.clip(
            np.round(depth_m[valid] * 1000.0), 1, np.iinfo(np.uint16).max
        ).astype(np.uint16)

        write_image(rgb_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_image(depth_png_path, depth_mm)
        write_image(preview_path, depth_preview(depth_m))
        np.savez_compressed(str(depth_npz_path), depth_m=depth_m)

        transform_data = None
        transform_error = None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.args.target_frame,
                self.camera_info.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            transform_data = transform_to_dict(transform)
        except Exception as error:
            transform_error = str(error)

        depth_stats = {"valid_pixels": int(np.count_nonzero(valid))}
        if np.any(valid):
            percentiles = np.percentile(depth_m[valid], [5, 10, 50, 90, 95])
            depth_stats.update(
                {
                    "p05_m": float(percentiles[0]),
                    "p10_m": float(percentiles[1]),
                    "median_m": float(percentiles[2]),
                    "p90_m": float(percentiles[3]),
                    "p95_m": float(percentiles[4]),
                }
            )

        record = {
            "frame_index": index,
            "seed": self.args.seed,
            "camera": self.args.camera,
            "rgb_stamp": stamp_to_dict(rgb_message.header.stamp),
            "depth_stamp": stamp_to_dict(depth_message.header.stamp),
            "stamp_delta_seconds": abs(
                rgb_message.header.stamp.to_sec() - depth_message.header.stamp.to_sec()
            ),
            "files": {
                "rgb": str(rgb_path),
                "depth_npz_m": str(depth_npz_path),
                "depth_png_mm": str(depth_png_path),
                "depth_preview": str(preview_path),
                "metadata": str(metadata_path),
            },
            "camera_info": camera_info_to_dict(self.camera_info),
            "tf": transform_data,
            "tf_error": transform_error,
            "depth_stats": depth_stats,
        }
        metadata_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.records.append(record)
        rospy.loginfo(
            "saved Scene3 seed=%d camera=%s frame=%d/%d",
            self.args.seed,
            self.args.camera,
            len(self.records),
            self.args.count,
        )

    def write_manifest(self):
        manifest = {
            "scene": "scene3",
            "seed": self.args.seed,
            "camera": self.args.camera,
            "count": len(self.records),
            "requested_count": self.args.count,
            "interval_seconds": self.args.interval,
            "sync_slop_seconds": self.args.sync_slop,
            "target_frame": self.args.target_frame,
            "topics": self.camera_topics,
            "frames": self.records,
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--camera", choices=sorted(CAMERAS), default="head")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.3)
    parser.add_argument("--sync-slop", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--target-frame", default="base_link")
    parser.add_argument("--output-root", default="vision_dataset/scene3")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(rospy.myargv()[1:])
    if args.count <= 0 or args.interval < 0 or args.sync_slop <= 0 or args.timeout <= 0:
        parser.error("count, interval, sync-slop and timeout must be positive")
    if args.run_name and ("/" in args.run_name or "\\" in args.run_name):
        parser.error("run-name must be a single directory name")
    return args


def main():
    args = parse_args()
    rospy.init_node("scene3_dataset_collector", anonymous=True)
    topics = CAMERAS[args.camera]
    camera_info = rospy.wait_for_message(
        topics["info"], CameraInfo, timeout=min(args.timeout, 15.0)
    )

    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(1.0)
    collector = DatasetCollector(args, camera_info, tf_buffer)

    deadline = time.monotonic() + args.timeout
    while not collector.done.is_set() and not rospy.is_shutdown():
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "timed out after {:.1f}s with {}/{} frames".format(
                    args.timeout, len(collector.records), args.count
                )
            )
        rospy.sleep(0.05)

    if collector.error is not None:
        raise collector.error
    manifest_path = collector.write_manifest()
    print("saved_frames={}".format(len(collector.records)))
    print("dataset_dir={}".format(collector.output_dir))
    print("manifest={}".format(manifest_path))


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSException, RuntimeError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
