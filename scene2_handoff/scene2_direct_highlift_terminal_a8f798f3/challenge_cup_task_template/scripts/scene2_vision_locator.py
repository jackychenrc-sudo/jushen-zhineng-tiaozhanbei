#!/usr/bin/env python3
"""Scene2 RGB-D color localization debug tool.

This node is intentionally read-only. It subscribes to the head RGB/depth
streams, detects one colored part, estimates its 3-D position, transforms the
point to ``base_link``, saves debug images, and exits by default.
"""

import argparse
import json
import math
import os
import struct
import threading

import cv2
import message_filters
import numpy as np
import rospy
import tf
import tf2_ros

from sensor_msgs.msg import CameraInfo, CompressedImage


RGB_TOPIC = "/cam_h/color/image_raw/compressed"
DEPTH_TOPIC = "/cam_h/depth/image_raw/compressedDepth"
CAMERA_INFO_TOPIC = "/cam_h/color/camera_info"
BASE_FRAME = "base_link"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# One color can have multiple HSV intervals. Red wraps around hue=0/180.
HSV_RANGES = {
    "red": [
        ((0, 80, 60), (10, 255, 255)),
        ((170, 80, 60), (180, 255, 255)),
    ],
    "orange": [
        ((5, 80, 60), (30, 255, 255)),
    ],
    "blue": [
        ((90, 70, 50), (135, 255, 255)),
    ],
    "purple": [
        ((125, 60, 40), (165, 255, 255)),
    ],
}

# Reference colors are stored as RGB because that is how they are collected
# and documented.  OpenCV images are BGR, so conversion is explicit below.
# The distance is measured between normalized RGB chromaticities; this keeps
# the red handle stable when the simulated exposure changes.
REFERENCE_RGB = {
    "red": (172.0, 36.0, 27.0),
}

RGB_CHROMA_DISTANCE_LIMIT = {
    "red": 0.23,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only Scene2 RGB-D color localization"
    )
    parser.add_argument(
        "--color",
        choices=sorted(HSV_RANGES),
        default="red",
        help="target color (default: red)",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/kuavo_ws/scene2_debug",
        help="directory for annotated images and masks",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=150.0,
        help="minimum contour area in pixels",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=50000.0,
        help="maximum contour area in pixels; helps reject large bins",
    )
    parser.add_argument(
        "--depth-radius",
        type=int,
        default=12,
        help="half-size of the depth sampling window around the centroid",
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="candidate to localize after sorting contours by area",
    )
    parser.add_argument(
        "--sync-slop",
        type=float,
        default=0.10,
        help="maximum RGB/depth timestamp difference in seconds",
    )
    parser.add_argument(
        "--roi",
        default="",
        help="optional x,y,width,height ROI; empty means full image",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="keep processing frames instead of exiting after one success",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="read one RGB frame and one depth frame separately for a settled scene",
    )
    return parser.parse_args(rospy.myargv()[1:])


def parse_roi(text):
    if not text:
        return None
    try:
        values = [int(item.strip()) for item in text.split(",")]
    except ValueError as exc:
        raise ValueError("ROI must contain integers: x,y,width,height") from exc
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise ValueError("ROI must be x,y,width,height with positive size")
    return tuple(values)


def decode_rgb(message):
    encoded = np.frombuffer(message.data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode RGB JPEG")
    return image


def decode_compressed_depth(message):
    """Decode ROS compressedDepth into float32 metres."""
    data = bytes(message.data)
    png_offset = data.find(PNG_SIGNATURE)
    if png_offset < 0:
        raise RuntimeError("compressed depth payload does not contain PNG data")

    encoded = np.frombuffer(data[png_offset:], dtype=np.uint8)
    raw = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError("failed to decode compressed depth PNG")

    encoding = message.format.split(";", 1)[0].strip()
    if encoding == "16UC1":
        if raw.dtype != np.uint16:
            raise RuntimeError(
                "16UC1 depth decoded as unexpected dtype %s" % raw.dtype
            )
        return raw.astype(np.float32) / 1000.0

    if encoding == "32FC1":
        if png_offset < 12:
            raise RuntimeError("32FC1 compressed depth header is missing")
        _compression_format, quant_a, quant_b = struct.unpack(
            "iff", data[png_offset - 12:png_offset]
        )
        inverse_depth = raw.astype(np.float32)
        depth = np.full(inverse_depth.shape, np.nan, dtype=np.float32)
        valid = inverse_depth > 0
        denominator = inverse_depth[valid] - quant_b
        good = np.abs(denominator) > 1.0e-6
        values = np.full(denominator.shape, np.nan, dtype=np.float32)
        values[good] = quant_a / denominator[good]
        depth[valid] = values
        return depth

    raise RuntimeError("unsupported depth encoding: %s" % encoding)


def normalized_rgb_distance(image_bgr, reference_rgb):
    rgb = image_bgr[:, :, ::-1].astype(np.float32)
    brightness = np.sum(rgb, axis=2)
    chromaticity = rgb / np.maximum(brightness[:, :, None], 1.0)

    reference = np.asarray(reference_rgb, dtype=np.float32)
    reference /= max(float(np.sum(reference)), 1.0)
    return np.linalg.norm(chromaticity - reference[None, None, :], axis=2)


def build_color_mask(image_bgr, color_name, roi):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES[color_name]:
        current = cv2.inRange(
            hsv,
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, current)

    color_distance = None
    if color_name in REFERENCE_RGB:
        color_distance = normalized_rgb_distance(
            image_bgr,
            REFERENCE_RGB[color_name],
        )
        distance_mask = (
            color_distance <= RGB_CHROMA_DISTANCE_LIMIT[color_name]
        ).astype(np.uint8) * 255
        # The broad HSV gate rejects orange/brown regions; the RGB distance
        # then keeps the pixels closest to the collected target color.
        mask = cv2.bitwise_and(mask, distance_mask)

    if roi is not None:
        x, y, width, height = roi
        height_px, width_px = mask.shape
        x2 = min(width_px, x + width)
        y2 = min(height_px, y + height)
        if x < 0 or y < 0 or x >= x2 or y >= y2:
            raise ValueError("ROI is outside the image")
        roi_mask = np.zeros_like(mask)
        roi_mask[y:y2, x:x2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask, color_distance


def find_candidates(mask, min_area, max_area, color_distance=None):
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1.0e-9:
            continue
        center_u = int(round(moments["m10"] / moments["m00"]))
        center_v = int(round(moments["m01"] / moments["m00"]))
        bounding_box = cv2.boundingRect(contour)
        rotated_rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rotated_rect[1]
        short_side = min(float(rect_width), float(rect_height))
        long_side = max(float(rect_width), float(rect_height))
        if short_side < 1.0:
            continue
        long_axis_angle = float(rotated_rect[2])
        if rect_width < rect_height:
            long_axis_angle += 90.0
        while long_axis_angle >= 90.0:
            long_axis_angle -= 180.0
        while long_axis_angle < -90.0:
            long_axis_angle += 180.0

        distance_median = None
        if color_distance is not None:
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(
                contour_mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED,
            )
            values = color_distance[contour_mask.astype(bool)]
            if values.size:
                distance_median = float(np.median(values))
        candidates.append(
            {
                "contour": contour,
                "area": area,
                "center": (center_u, center_v),
                "bounding_box": bounding_box,
                "rotated_box": np.intp(cv2.boxPoints(rotated_rect)),
                "long_side_px": long_side,
                "short_side_px": short_side,
                "aspect_ratio": long_side / short_side,
                "image_angle_deg": long_axis_angle,
                "rgb_chroma_distance": distance_median,
            }
        )
    candidates.sort(key=lambda item: item["area"], reverse=True)
    return candidates


def sample_depth(depth_m, contour, center, radius):
    height, width = depth_m.shape
    center_u, center_v = center
    x1 = max(0, center_u - radius)
    x2 = min(width, center_u + radius + 1)
    y1 = max(0, center_v - radius)
    y2 = min(height, center_v + radius + 1)

    object_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    object_mask = cv2.erode(object_mask, erode_kernel, iterations=1)

    window_mask = np.zeros_like(object_mask)
    window_mask[y1:y2, x1:x2] = 255
    sample_mask = cv2.bitwise_and(object_mask, window_mask).astype(bool)

    valid = (
        sample_mask
        & np.isfinite(depth_m)
        & (depth_m > 0.10)
        & (depth_m < 3.00)
    )
    values = depth_m[valid]

    # If the small central window is sparse, use the eroded object interior.
    if values.size < 10:
        valid = (
            object_mask.astype(bool)
            & np.isfinite(depth_m)
            & (depth_m > 0.10)
            & (depth_m < 3.00)
        )
        values = depth_m[valid]

    if values.size < 10:
        raise RuntimeError(
            "not enough valid depth pixels for the selected object: %d"
            % values.size
        )

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad, int(values.size), sample_mask


def deproject_pixel(center, depth_m, camera_info):
    center_u, center_v = center
    fx = float(camera_info.K[0])
    fy = float(camera_info.K[4])
    cx = float(camera_info.K[2])
    cy = float(camera_info.K[5])
    if fx <= 0.0 or fy <= 0.0:
        raise RuntimeError("invalid camera intrinsics")
    x = (center_u - cx) * depth_m / fx
    y = (center_v - cy) * depth_m / fy
    return np.array([x, y, depth_m], dtype=np.float64)


def transform_point(tf_buffer, point_camera, source_frame, stamp):
    transform = tf_buffer.lookup_transform(
        BASE_FRAME,
        source_frame,
        stamp,
        rospy.Duration(1.0),
    )
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    matrix = tf.transformations.quaternion_matrix(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    )
    matrix[0, 3] = translation.x
    matrix[1, 3] = translation.y
    matrix[2, 3] = translation.z
    homogeneous = np.array(
        [point_camera[0], point_camera[1], point_camera[2], 1.0],
        dtype=np.float64,
    )
    return np.matmul(matrix, homogeneous)[:3]


def safe_filename_stamp(stamp):
    return "%d_%09d" % (stamp.secs, stamp.nsecs)


class Scene2VisionDebug:
    def __init__(self, args):
        self.args = args
        self.roi = parse_roi(args.roi)
        self.done = False
        self.callback_lock = threading.Lock()

        rospy.loginfo("waiting for camera intrinsics: %s", CAMERA_INFO_TOPIC)
        self.camera_info = rospy.wait_for_message(
            CAMERA_INFO_TOPIC, CameraInfo, timeout=5.0
        )

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.rgb_subscriber = None
        self.depth_subscriber = None
        self.synchronizer = None
        if not args.sequential:
            self.rgb_subscriber = message_filters.Subscriber(
                RGB_TOPIC, CompressedImage
            )
            self.depth_subscriber = message_filters.Subscriber(
                DEPTH_TOPIC, CompressedImage
            )
            self.synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_subscriber, self.depth_subscriber],
                queue_size=20,
                slop=args.sync_slop,
                allow_headerless=False,
            )
            self.synchronizer.registerCallback(self.callback)

        os.makedirs(args.output_dir, exist_ok=True)
        rospy.logwarn(
            "Scene2 vision debug is read-only; it only subscribes and saves images"
        )
        rospy.loginfo("target color: %s", args.color)

    def run_sequential(self):
        rospy.loginfo("waiting for one RGB frame: %s", RGB_TOPIC)
        rgb_message = rospy.wait_for_message(
            RGB_TOPIC, CompressedImage, timeout=15.0
        )
        rospy.loginfo("waiting for one depth frame: %s", DEPTH_TOPIC)
        depth_message = rospy.wait_for_message(
            DEPTH_TOPIC, CompressedImage, timeout=15.0
        )
        time_difference = abs(
            (rgb_message.header.stamp - depth_message.header.stamp).to_sec()
        )
        rospy.loginfo(
            "sequential RGB/depth time difference: %.3f s", time_difference
        )
        return self.process(
            rgb_message,
            depth_message,
            transform_stamp=rospy.Time(0),
        )

    def validate_geometry(self, rgb, depth):
        if rgb.shape[:2] != depth.shape[:2]:
            raise RuntimeError(
                "RGB/depth are not pixel-aligned: RGB=%s depth=%s"
                % (rgb.shape[:2], depth.shape[:2])
            )
        height, width = rgb.shape[:2]
        if self.camera_info.width != width or self.camera_info.height != height:
            raise RuntimeError(
                "camera_info size does not match image: info=%dx%d image=%dx%d"
                % (
                    self.camera_info.width,
                    self.camera_info.height,
                    width,
                    height,
                )
            )

    def callback(self, rgb_message, depth_message):
        if self.done and not self.args.continuous:
            return
        if not self.callback_lock.acquire(False):
            return
        try:
            self.process(rgb_message, depth_message)
        except (RuntimeError, ValueError, tf2_ros.TransformException) as exc:
            rospy.logwarn_throttle(2.0, "vision frame rejected: %s", exc)
        except Exception as exc:
            rospy.logerr("unexpected vision error: %s", exc)
        finally:
            self.callback_lock.release()

    def process(self, rgb_message, depth_message, transform_stamp=None):
        rgb = decode_rgb(rgb_message)
        depth_m = decode_compressed_depth(depth_message)
        self.validate_geometry(rgb, depth_m)

        mask, color_distance = build_color_mask(
            rgb,
            self.args.color,
            self.roi,
        )
        candidates = find_candidates(
            mask,
            self.args.min_area,
            self.args.max_area,
            color_distance=color_distance,
        )
        if not candidates:
            failure_image_path = os.path.join(
                self.args.output_dir,
                "last_%s_no_detection.jpg" % self.args.color,
            )
            failure_mask_path = os.path.join(
                self.args.output_dir,
                "last_%s_no_detection_mask.png" % self.args.color,
            )
            cv2.imwrite(failure_image_path, rgb)
            cv2.imwrite(failure_mask_path, mask)
            raise RuntimeError(
                "no %s contour in area range %.0f..%.0f; debug frame: %s"
                % (
                    self.args.color,
                    self.args.min_area,
                    self.args.max_area,
                    failure_image_path,
                )
            )

        if self.args.candidate_index < 0 or self.args.candidate_index >= len(candidates):
            raise RuntimeError(
                "candidate index %d is outside available range 0..%d"
                % (self.args.candidate_index, len(candidates) - 1)
            )
        source_frame = rgb_message.header.frame_id
        if source_frame != depth_message.header.frame_id:
            raise RuntimeError(
                "RGB/depth frames differ: %r vs %r"
                % (source_frame, depth_message.header.frame_id)
            )
        candidate_results = []
        for index, candidate in enumerate(candidates):
            center = candidate["center"]
            item = {
                "index": index,
                "pixel_uv": [int(center[0]), int(center[1])],
                "area_px": round(candidate["area"], 2),
                "long_side_px": round(candidate["long_side_px"], 2),
                "short_side_px": round(candidate["short_side_px"], 2),
                "aspect_ratio": round(candidate["aspect_ratio"], 3),
                "image_angle_deg": round(candidate["image_angle_deg"], 2),
                "rgb_chroma_distance": (
                    None
                    if candidate["rgb_chroma_distance"] is None
                    else round(candidate["rgb_chroma_distance"], 4)
                ),
                "valid_3d": False,
            }
            try:
                depth_value, depth_mad, depth_count, _sample_mask = sample_depth(
                    depth_m,
                    candidate["contour"],
                    center,
                    self.args.depth_radius,
                )
                point_camera = deproject_pixel(
                    center,
                    depth_value,
                    self.camera_info,
                )
                point_base = transform_point(
                    self.tf_buffer,
                    point_camera,
                    source_frame,
                    rgb_message.header.stamp
                    if transform_stamp is None
                    else transform_stamp,
                )
                item.update(
                    {
                        "valid_3d": True,
                        "depth_m": round(depth_value, 5),
                        "depth_mad_m": round(depth_mad, 5),
                        "valid_depth_pixels": depth_count,
                        "camera_xyz_m": [
                            round(float(value), 5) for value in point_camera
                        ],
                        "base_xyz_m": [
                            round(float(value), 5) for value in point_base
                        ],
                    }
                )
            except (RuntimeError, tf2_ros.TransformException) as exc:
                item["rejection_reason"] = str(exc)
            candidate_results.append(item)

        selected = candidates[self.args.candidate_index]
        selected_result = candidate_results[self.args.candidate_index]
        if not selected_result["valid_3d"]:
            raise RuntimeError(
                "selected candidate has no valid 3-D point: %s"
                % selected_result.get("rejection_reason", "unknown error")
            )
        center = selected["center"]
        depth_value = selected_result["depth_m"]
        point_camera = selected_result["camera_xyz_m"]
        point_base = selected_result["base_xyz_m"]

        result = {
            "color": self.args.color,
            "pixel_uv": [int(center[0]), int(center[1])],
            "contour_area_px": round(selected["area"], 2),
            "image_angle_deg": selected_result["image_angle_deg"],
            "aspect_ratio": selected_result["aspect_ratio"],
            "rgb_chroma_distance": selected_result["rgb_chroma_distance"],
            "reference_rgb": list(REFERENCE_RGB.get(self.args.color, ())),
            "depth_m": depth_value,
            "depth_mad_m": selected_result["depth_mad_m"],
            "valid_depth_pixels": selected_result["valid_depth_pixels"],
            "camera_frame": source_frame,
            "camera_xyz_m": point_camera,
            "base_frame": BASE_FRAME,
            "base_xyz_m": point_base,
            "candidate_count": len(candidates),
            "selected_candidate_index": self.args.candidate_index,
            "candidates": candidate_results,
        }

        annotated = rgb.copy()
        for index, candidate in enumerate(candidates):
            x, y, width, height = candidate["bounding_box"]
            color = (
                (0, 255, 255)
                if index == self.args.candidate_index
                else (255, 255, 0)
            )
            cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
            cv2.polylines(
                annotated,
                [candidate["rotated_box"]],
                True,
                color,
                2,
            )
            cv2.putText(
                annotated,
                "#%d area=%.0f angle=%.1f"
                % (index, candidate["area"], candidate["image_angle_deg"]),
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        cv2.drawMarker(
            annotated,
            center,
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        cv2.putText(
            annotated,
            "uv=%s depth=%.3fm base=(%.3f, %.3f, %.3f)"
            % (
                center,
                depth_value,
                point_base[0],
                point_base[1],
                point_base[2],
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        filename_stamp = safe_filename_stamp(rgb_message.header.stamp)
        image_path = os.path.join(
            self.args.output_dir,
            "%s_%s_annotated.jpg" % (filename_stamp, self.args.color),
        )
        mask_path = os.path.join(
            self.args.output_dir,
            "%s_%s_mask.png" % (filename_stamp, self.args.color),
        )
        raw_path = os.path.join(
            self.args.output_dir,
            "%s_rgb_raw.jpg" % filename_stamp,
        )
        if not cv2.imwrite(image_path, annotated):
            raise RuntimeError("failed to save annotated image: %s" % image_path)
        if not cv2.imwrite(mask_path, mask):
            raise RuntimeError("failed to save mask: %s" % mask_path)
        if not cv2.imwrite(raw_path, rgb):
            raise RuntimeError("failed to save raw RGB image: %s" % raw_path)

        rospy.loginfo("VISION_RESULT %s", json.dumps(result, ensure_ascii=False))
        rospy.loginfo("annotated image: %s", image_path)
        rospy.loginfo("binary mask: %s", mask_path)
        rospy.loginfo("raw RGB image: %s", raw_path)

        if not self.args.continuous:
            self.done = True
            rospy.signal_shutdown("one RGB-D localization completed")
        return result


def main():
    args = parse_args()
    rospy.init_node("scene2_vision_debug", anonymous=True)
    try:
        node = Scene2VisionDebug(args)
        if args.sequential:
            node.run_sequential()
        else:
            rospy.spin()
    except (rospy.ROSException, RuntimeError, ValueError) as exc:
        rospy.logerr("Scene2 vision debug failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
