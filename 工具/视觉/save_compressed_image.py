#!/usr/bin/env python3
"""
Save one or more ROS CompressedImage frames to files.

Example:
  python3 工具/视觉/save_compressed_image.py \
    --topic /cam_h/color/image_raw/compressed \
    --output vision_debug/scene3_seed3/scene3_head_rgb.jpg \
    --count 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rospy
from sensor_msgs.msg import CompressedImage


class ImageSaver:
    def __init__(self, topic: str, output: str, count: int):
        self.topic = topic
        self.output = Path(output)
        self.count = count
        self.saved = 0
        self.output.parent.mkdir(parents=True, exist_ok=True)

        rospy.loginfo("subscribing: %s", topic)
        rospy.Subscriber(topic, CompressedImage, self.callback, queue_size=1)

    def callback(self, msg: CompressedImage):
        if self.saved >= self.count:
            return

        if self.count == 1:
            path = self.output
        else:
            stem = self.output.stem
            suffix = self.output.suffix or ".jpg"
            path = self.output.with_name(f"{stem}_{self.saved:04d}{suffix}")

        with open(path, "wb") as f:
            f.write(bytes(msg.data))

        self.saved += 1
        rospy.loginfo("saved %s (%d/%d), format=%s", path, self.saved, self.count, msg.format)

        if self.saved >= self.count:
            rospy.signal_shutdown("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cam_h/color/image_raw/compressed")
    parser.add_argument("--output", default="vision_debug/scene3_seed3/scene3_head_rgb.jpg")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    if not args.topic.strip():
        raise ValueError("topic must not be empty")

    rospy.init_node("save_compressed_image_once", anonymous=True)
    ImageSaver(args.topic, args.output, args.count)
    rospy.spin()


if __name__ == "__main__":
    main()
