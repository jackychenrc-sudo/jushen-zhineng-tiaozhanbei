#!/usr/bin/env python3
"""Read one Scene3 RGB-D frame and report the best SMT tray candidate.

This node is deliberately perception-only.  It does not publish robot motion
commands, so it is suitable for the first server-side check.
"""

import argparse
import json
import struct
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String

from label_smt_trays import (
    best_by_level,
    contour_candidates,
    draw_result,
    make_dark_mask,
    normalized_roi_to_bbox,
    parse_roi,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def decode_rgb(message):
    encoded = np.frombuffer(message.data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode compressed RGB image")
    return image


def decode_depth(message):
    """Decode ROS compressedDepth and return depth in metres."""
    raw = bytes(message.data)
    png_offset = raw.find(PNG_SIGNATURE)
    if png_offset < 0:
        raise RuntimeError("compressedDepth message does not contain a PNG payload")

    encoded = np.frombuffer(raw[png_offset:], dtype=np.uint8)
    depth_png = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        raise RuntimeError("failed to decode compressedDepth PNG")

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
    # The simulator normally publishes 16UC1 depth in millimetres.
    if np.any(valid) and float(np.nanmedian(depth_m[valid])) > 20.0:
        depth_m *= 0.001
    depth_m[~valid] = np.nan
    return depth_m


def median_depth(depth_m, center, radius):
    u, v = center
    height, width = depth_m.shape[:2]
    x1 = max(0, u - radius)
    x2 = min(width, u + radius + 1)
    y1 = max(0, v - radius)
    y2 = min(height, v + radius + 1)
    values = depth_m[y1:y2, x1:x2]
    values = values[np.isfinite(values) & (values > 0.05) & (values < 10.0)]
    if values.size == 0:
        raise RuntimeError("no valid depth around tray centre {}".format(center))
    return float(np.median(values)), int(values.size)


def project_pixel(center, depth, camera_info):
    fx = float(camera_info.K[0])
    fy = float(camera_info.K[4])
    cx = float(camera_info.K[2])
    cy = float(camera_info.K[5])
    if fx <= 0 or fy <= 0:
        raise RuntimeError("invalid camera intrinsics: fx={}, fy={}".format(fx, fy))
    u, v = center
    return [
        (float(u) - cx) * depth / fx,
        (float(v) - cy) * depth / fy,
        depth,
    ]


def draw_selected(image, candidate, depth_m, camera_xyz):
    x1, y1, x2, y2 = candidate.bbox
    u, v = candidate.center_pixel
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    text = "SELECTED {} Z={:.3f}m XYZ=({:.3f},{:.3f},{:.3f})".format(
        candidate.level, depth_m, camera_xyz[0], camera_xyz[1], camera_xyz[2]
    )
    cv2.putText(
        image,
        text,
        (max(10, x1), min(image.shape[0] - 12, y2 + 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(image, (u, v), 6, (0, 0, 255), -1)


def write_debug_images(output_dir, image, depth_image, roi, split_y, candidates, dark_threshold):
    raw_path = output_dir / "raw_rgb.jpg"
    candidates_path = output_dir / "tray_candidates.jpg"
    mask_path = output_dir / "debug_dark_mask.jpg"
    depth_path = output_dir / "depth_preview.jpg"

    cv2.imwrite(str(raw_path), image)
    cv2.imwrite(str(candidates_path), draw_result(image, candidates, roi, split_y, max_per_level=8))

    height, width = image.shape[:2]
    x1, y1, x2, y2 = normalized_roi_to_bbox(roi, width, height)
    debug_mask = np.zeros((height, width), dtype=np.uint8)
    debug_mask[y1:y2, x1:x2] = make_dark_mask(image[y1:y2, x1:x2], dark_threshold)
    cv2.imwrite(str(mask_path), debug_mask)

    valid = np.isfinite(depth_image) & (depth_image > 0.05) & (depth_image < 10.0)
    depth_preview = np.zeros(depth_image.shape[:2], dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth_image, 0.0, 5.0)
        depth_preview = (255.0 - clipped / 5.0 * 255.0).astype(np.uint8)
        depth_preview[~valid] = 0
    cv2.imwrite(str(depth_path), depth_preview)

    return {
        "raw_rgb": str(raw_path),
        "tray_candidates": str(candidates_path),
        "debug_dark_mask": str(mask_path),
        "depth_preview": str(depth_path),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=("upper", "lower"), default="upper")
    parser.add_argument("--rgb-topic", default="/cam_h/color/image_raw/compressed")
    parser.add_argument("--depth-topic", default="/cam_h/depth/image_raw/compressedDepth")
    parser.add_argument("--camera-info-topic", default="/cam_h/color/camera_info")
    parser.add_argument("--output-dir", default="/tmp/scene3_perception")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--depth-radius", type=int, default=5)
    parser.add_argument("--roi", default="0.25,0.13,0.75,0.78")
    parser.add_argument("--split-y", type=float, default=0.42)
    parser.add_argument("--dark-threshold", type=int, default=95)
    parser.add_argument("--min-area", type=int, default=70)
    parser.add_argument("--max-area", type=int, default=35000)
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def main():
    args = parse_args()
    rospy.init_node("scene3_live_perception", anonymous=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rospy.loginfo("waiting for head RGB-D topics")
    camera_info = rospy.wait_for_message(args.camera_info_topic, CameraInfo, timeout=args.timeout)
    rgb_message = rospy.wait_for_message(args.rgb_topic, CompressedImage, timeout=args.timeout)
    depth_message = rospy.wait_for_message(args.depth_topic, CompressedImage, timeout=args.timeout)

    image = decode_rgb(rgb_message)
    depth_image = decode_depth(depth_message)
    if image.shape[:2] != depth_image.shape[:2]:
        raise RuntimeError(
            "RGB/depth size mismatch: {} vs {}".format(image.shape[:2], depth_image.shape[:2])
        )

    roi = parse_roi(args.roi)
    candidates = contour_candidates(
        image=image,
        roi=roi,
        split_y=args.split_y,
        dark_threshold=args.dark_threshold,
        min_area=args.min_area,
        max_area=args.max_area,
    )
    debug_paths = write_debug_images(
        output_dir=output_dir,
        image=image,
        depth_image=depth_image,
        roi=roi,
        split_y=args.split_y,
        candidates=candidates,
        dark_threshold=args.dark_threshold,
    )
    json_path = output_dir / "tray_detection.json"

    selected = next((item for item in candidates if item.level == args.level), None)
    if selected is None:
        payload = {
            "status": "no_candidate",
            "selected_level": args.level,
            "camera_frame": camera_info.header.frame_id,
            "rgb_stamp": rgb_message.header.stamp.to_sec(),
            "depth_stamp": depth_message.header.stamp.to_sec(),
            "roi": list(roi),
            "split_y": args.split_y,
            "dark_threshold": args.dark_threshold,
            "min_area": args.min_area,
            "max_area": args.max_area,
            "best": best_by_level(candidates),
            "candidates": [asdict(candidate) for candidate in candidates],
            "debug_files": debug_paths,
            "note": "No selected tray at requested level. Inspect raw_rgb.jpg, debug_dark_mask.jpg and tray_candidates.jpg.",
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            "no {} tray candidate found; inspect {}".format(
                args.level, debug_paths["tray_candidates"]
            )
        )

    z_m, valid_depth_pixels = median_depth(
        depth_image, selected.center_pixel, args.depth_radius
    )
    camera_xyz = project_pixel(selected.center_pixel, z_m, camera_info)

    visualization = draw_result(image, candidates, roi, args.split_y, max_per_level=8)
    draw_selected(visualization, selected, z_m, camera_xyz)
    image_path = output_dir / "tray_candidates.jpg"
    cv2.imwrite(str(image_path), visualization)

    payload = {
        "status": "ok",
        "selected_level": args.level,
        "camera_frame": camera_info.header.frame_id,
        "rgb_stamp": rgb_message.header.stamp.to_sec(),
        "depth_stamp": depth_message.header.stamp.to_sec(),
        "selected": asdict(selected),
        "depth_m": z_m,
        "valid_depth_pixels": valid_depth_pixels,
        "camera_xyz_m": camera_xyz,
        "camera_model": {
            "fx": float(camera_info.K[0]),
            "fy": float(camera_info.K[4]),
            "cx": float(camera_info.K[2]),
            "cy": float(camera_info.K[5]),
        },
        "roi": list(roi),
        "split_y": args.split_y,
        "dark_threshold": args.dark_threshold,
        "best": best_by_level(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
        "debug_files": debug_paths,
        "note": "camera_xyz_m is in the optical camera frame, not the robot base frame",
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")

    publisher = rospy.Publisher("/scene3/tray_detection", String, queue_size=1, latch=True)
    rospy.sleep(0.3)
    publisher.publish(String(data=json_text))
    rospy.sleep(0.3)

    print(json_text)
    print("saved_image={}".format(image_path))
    print("saved_json={}".format(json_path))


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSException, RuntimeError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(2)

