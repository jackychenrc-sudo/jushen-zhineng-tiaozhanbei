#!/usr/bin/env python3
"""Scene2 minimal read-only red-part localization."""

import argparse
import math
import os

import cv2
import numpy as np
import rospy
import tf

from sensor_msgs.msg import CameraInfo, CompressedImage


RGB_TOPIC = "/cam_h/color/image_raw/compressed"
DEPTH_TOPIC = "/cam_h/depth/image_raw/compressedDepth"
INFO_TOPIC = "/cam_h/color/camera_info"
BASE_FRAME = "base_link"
OUTPUT_FILE = "/root/kuavo_ws/scene2_debug/minimal_red_result.jpg"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=int,
        default=0,
        help="red contour index after sorting by area (default: 0)",
    )
    return parser.parse_args(rospy.myargv()[1:])


def decode_rgb(message):
    data = np.frombuffer(message.data, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("RGB image decode failed")
    return image


def decode_depth_metres(message):
    if not message.format.startswith("16UC1"):
        raise RuntimeError("expected 16UC1 depth, got %s" % message.format)

    payload = bytes(message.data)
    png_offset = payload.find(PNG_SIGNATURE)
    if png_offset < 0:
        raise RuntimeError("depth PNG header not found")

    data = np.frombuffer(payload[png_offset:], dtype=np.uint8)
    depth_mm = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if depth_mm is None or depth_mm.dtype != np.uint16:
        raise RuntimeError("depth image decode failed")
    return depth_mm.astype(np.float32) / 1000.0


def find_red_parts(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_a = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    mask_b = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    mask = cv2.bitwise_or(mask_a, mask_b)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 150]
    contours.sort(key=cv2.contourArea, reverse=True)
    return mask, contours


def contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        raise RuntimeError("selected contour has zero area")
    u = int(round(moments["m10"] / moments["m00"]))
    v = int(round(moments["m01"] / moments["m00"]))
    return u, v


def median_object_depth(depth_m, contour, center):
    height, width = depth_m.shape
    u, v = center

    object_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)
    object_mask = cv2.erode(object_mask, np.ones((3, 3), np.uint8))

    radius = 12
    window = np.zeros_like(object_mask)
    window[
        max(0, v - radius):min(height, v + radius + 1),
        max(0, u - radius):min(width, u + radius + 1),
    ] = 255

    sample_mask = (object_mask > 0) & (window > 0)
    valid = sample_mask & np.isfinite(depth_m) & (depth_m > 0.1) & (depth_m < 3.0)
    values = depth_m[valid]
    if values.size < 10:
        valid = (object_mask > 0) & np.isfinite(depth_m) & (depth_m > 0.1) & (depth_m < 3.0)
        values = depth_m[valid]
    if values.size < 10:
        raise RuntimeError("not enough valid depth pixels")
    return float(np.median(values)), int(values.size)


def deproject(center, depth, camera_info):
    u, v = center
    fx, fy = camera_info.K[0], camera_info.K[4]
    cx, cy = camera_info.K[2], camera_info.K[5]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.array([x, y, depth], dtype=np.float64)


def camera_to_base(listener, point_camera, camera_frame):
    listener.waitForTransform(
        BASE_FRAME, camera_frame, rospy.Time(0), rospy.Duration(3.0)
    )
    translation, quaternion = listener.lookupTransform(
        BASE_FRAME, camera_frame, rospy.Time(0)
    )
    matrix = tf.transformations.quaternion_matrix(quaternion)
    matrix[:3, 3] = translation
    point = np.array([*point_camera, 1.0], dtype=np.float64)
    return matrix.dot(point)[:3]


def main():
    args = parse_args()
    rospy.init_node("scene2_vision_minimal", anonymous=True)
    rospy.logwarn("minimal vision node is read-only")

    listener = tf.TransformListener()
    camera_info = rospy.wait_for_message(INFO_TOPIC, CameraInfo, timeout=5.0)
    rgb_message = rospy.wait_for_message(RGB_TOPIC, CompressedImage, timeout=5.0)
    depth_message = rospy.wait_for_message(DEPTH_TOPIC, CompressedImage, timeout=5.0)

    image = decode_rgb(rgb_message)
    depth_m = decode_depth_metres(depth_message)
    if image.shape[:2] != depth_m.shape:
        raise RuntimeError("RGB and depth image sizes do not match")

    mask, contours = find_red_parts(image)
    if not contours:
        raise RuntimeError("no red part found; check head pitch and HSV threshold")
    if args.candidate < 0 or args.candidate >= len(contours):
        raise RuntimeError(
            "candidate %d unavailable; detected %d red parts"
            % (args.candidate, len(contours))
        )

    contour = contours[args.candidate]
    center = contour_center(contour)
    depth, valid_count = median_object_depth(depth_m, contour, center)
    point_camera = deproject(center, depth, camera_info)
    point_base = camera_to_base(
        listener, point_camera, rgb_message.header.frame_id
    )

    x, y, width, height = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 255), 2)
    cv2.drawMarker(image, center, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(
        image,
        "uv=%s depth=%.3fm base=(%.3f, %.3f, %.3f)"
        % (center, depth, point_base[0], point_base[1], point_base[2]),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    cv2.imwrite(OUTPUT_FILE, image)

    print("检测到红色候选数量:", len(contours))
    print("选中候选:", args.candidate)
    print("像素中心:", center)
    print("有效深度像素:", valid_count)
    print("深度(m):", round(depth, 4))
    print("相机坐标(m):", [round(float(value), 4) for value in point_camera])
    print("base_link坐标(m):", [round(float(value), 4) for value in point_base])
    print("标注图片:", OUTPUT_FILE)


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSException, tf.Exception, RuntimeError) as error:
        rospy.logerr("minimal vision failed: %s", error)
        raise SystemExit(1)
