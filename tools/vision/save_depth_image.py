#!/usr/bin/env python3
"""
Save a ROS compressedDepth image to .npy and an optional visualization PNG.

The script tries to be robust to ROS compressed_depth_image_transport payloads by
finding the PNG signature inside msg.data and decoding from there.

Example:
  python3 tools/vision/save_depth_image.py \
    --topic /cam_h/depth/image_raw/compressedDepth \
    --output vision_debug/scene3_seed3/scene3_head_depth.npy \
    --png-output vision_debug/scene3_seed3/scene3_head_depth_vis.png \
    --count 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def decode_compressed_depth(msg: CompressedImage) -> np.ndarray:
    data = bytes(msg.data)
    png_start = data.find(PNG_SIGNATURE)
    payload = data[png_start:] if png_start >= 0 else data
    encoded = np.frombuffer(payload, dtype=np.uint8)
    depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"failed to decode compressed depth, format={msg.format!r}, bytes={len(data)}")
    return depth


def save_depth_vis(depth: np.ndarray, path: Path) -> None:
    depth_float = depth.astype(np.float32)
    valid = np.isfinite(depth_float) & (depth_float > 0)
    if not np.any(valid):
        vis = np.zeros(depth.shape[:2], dtype=np.uint8)
    else:
        lo = float(np.percentile(depth_float[valid], 2))
        hi = float(np.percentile(depth_float[valid], 98))
        if hi <= lo:
            hi = lo + 1.0
        vis = np.clip((depth_float - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


class DepthSaver:
    def __init__(self, topic: str, output: str, png_output: str | None, count: int):
        self.topic = topic
        self.output = Path(output)
        self.png_output = Path(png_output) if png_output else None
        self.count = count
        self.saved = 0
        self.output.parent.mkdir(parents=True, exist_ok=True)
        rospy.loginfo("subscribing: %s", topic)
        rospy.Subscriber(topic, CompressedImage, self.callback, queue_size=1)

    def numbered_path(self, path: Path) -> Path:
        if self.count == 1:
            return path
        return path.with_name(f"{path.stem}_{self.saved:04d}{path.suffix}")

    def callback(self, msg: CompressedImage) -> None:
        if self.saved >= self.count:
            return
        depth = decode_compressed_depth(msg)
        output_path = self.numbered_path(self.output)
        np.save(str(output_path), depth)
        if self.png_output is not None:
            save_depth_vis(depth, self.numbered_path(self.png_output))
        self.saved += 1
        rospy.loginfo(
            "saved depth %s (%d/%d), shape=%s dtype=%s format=%s",
            output_path,
            self.saved,
            self.count,
            depth.shape,
            depth.dtype,
            msg.format,
        )
        if self.saved >= self.count:
            rospy.signal_shutdown("done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cam_h/depth/image_raw/compressedDepth")
    parser.add_argument("--output", default="vision_debug/scene3_seed3/scene3_head_depth.npy")
    parser.add_argument("--png-output", default=None)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    rospy.init_node("save_depth_image_once", anonymous=True)
    DepthSaver(args.topic, args.output, args.png_output, args.count)
    rospy.spin()


if __name__ == "__main__":
    main()
