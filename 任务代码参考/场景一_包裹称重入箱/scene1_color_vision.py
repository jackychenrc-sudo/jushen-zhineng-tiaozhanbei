#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple color + depth parcel detector for challenge scene1.

This node does not train a neural network. It first tries to detect the yellow
cross-shaped tape on top of each parcel. The right claw should hover above the
cross center. If tape crosses are not found, it falls back to detecting
non-green parcel blobs on the green table.
"""

import math
import struct
import threading

import cv2
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage


class Scene1ColorVision:
    def __init__(self):
        self.color_topic = rospy.get_param("~color_topic", "/cam_h/color/image_raw/compressed")
        self.depth_topic = rospy.get_param("~depth_topic", "/cam_h/depth/image_raw/compressedDepth")
        self.info_topic = rospy.get_param("~camera_info_topic", "/cam_h/color/camera_info")
        self.output_topic = rospy.get_param("~output_topic", "/scene1/parcel_points")
        self.output_frame = rospy.get_param("~output_frame", "base_link")
        self.max_objects = int(rospy.get_param("~max_objects", 4))
        self.min_area = float(rospy.get_param("~min_area", 80.0))
        self.max_area = float(rospy.get_param("~max_area", 8000.0))
        self.tape_min_area = float(rospy.get_param("~tape_min_area", 45.0))
        self.tape_max_area = float(rospy.get_param("~tape_max_area", 4500.0))
        self.min_x = float(rospy.get_param("~min_x", 0.15))
        self.max_x = float(rospy.get_param("~max_x", 0.85))
        self.min_y = float(rospy.get_param("~min_y", -0.65))
        self.max_y = float(rospy.get_param("~max_y", 0.25))
        self.min_z = float(rospy.get_param("~min_z", -0.30))
        self.max_z = float(rospy.get_param("~max_z", 0.35))

        self._lock = threading.Lock()
        self._depth_msg = None
        self._camera_info = None

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._pub = rospy.Publisher(self.output_topic, PoseArray, queue_size=10)
        self._color_sub = rospy.Subscriber(self.color_topic, CompressedImage, self._color_cb, queue_size=1)
        self._depth_sub = rospy.Subscriber(self.depth_topic, CompressedImage, self._depth_cb, queue_size=1)
        self._info_sub = rospy.Subscriber(self.info_topic, CameraInfo, self._info_cb, queue_size=1)

        rospy.loginfo("scene1 color vision: color=%s depth=%s info=%s -> %s",
                      self.color_topic, self.depth_topic, self.info_topic, self.output_topic)

    def _depth_cb(self, msg):
        with self._lock:
            self._depth_msg = msg

    def _info_cb(self, msg):
        with self._lock:
            self._camera_info = msg

    def _color_cb(self, msg):
        with self._lock:
            depth_msg = self._depth_msg
            camera_info = self._camera_info

        if depth_msg is None or camera_info is None:
            return

        color = self._decode_color(msg)
        depth = self._decode_compressed_depth(depth_msg)
        if color is None or depth is None:
            return

        centers = self._detect_tape_cross_centers(color)
        source = "tape_crosses"
        detections = self._detections_from_centers(
            centers, depth, camera_info, depth_msg.header.frame_id
        )
        if not detections:
            centers = self._detect_parcel_centers(color)
            source = "parcel_blobs"
            detections = self._detections_from_centers(
                centers, depth, camera_info, depth_msg.header.frame_id
            )

        out = PoseArray()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.output_frame

        for u, v, area, point in detections:
            pose = Pose()
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = point[2]
            pose.orientation.w = 1.0
            out.poses.append(pose)

        if out.poses:
            self._pub.publish(out)
            rospy.loginfo_throttle(
                1.0,
                "scene1 color vision points: %s",
                {
                    "source": source,
                    "points": [[round(p.position.x, 3), round(p.position.y, 3), round(p.position.z, 3)]
                               for p in out.poses],
                },
            )

    def _detections_from_centers(self, centers, depth, camera_info, camera_frame):
        detections = []
        for u, v, area in centers:
            point = self._pixel_to_robot_point(u, v, depth, camera_info, camera_frame)
            if point is None:
                continue
            if not self._point_in_workspace(point):
                rospy.logwarn_throttle(
                    1.0,
                    "scene1 color vision rejected out-of-range point: %s",
                    [round(float(value), 3) for value in point],
                )
                continue
            detections.append((u, v, area, point))
            if len(detections) >= self.max_objects:
                break
        return detections

    def _point_in_workspace(self, point):
        x, y, z = [float(value) for value in point]
        if not all(math.isfinite(value) for value in (x, y, z)):
            return False
        return (
            self.min_x <= x <= self.max_x and
            self.min_y <= y <= self.max_y and
            self.min_z <= z <= self.max_z
        )

    def _decode_color(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image

    def _decode_compressed_depth(self, msg):
        raw = bytes(msg.data)
        for offset in (0, 12, 16):
            if len(raw) <= offset:
                continue
            image = cv2.imdecode(np.frombuffer(raw[offset:], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            if image.dtype == np.uint16:
                return image.astype(np.float32) / 1000.0
            if image.dtype == np.float32:
                return image

        # Some compressedDepth messages store a small config header before PNG.
        if len(raw) > 12:
            try:
                _fmt, _a, _b = struct.unpack("iff", raw[:12])
                image = cv2.imdecode(np.frombuffer(raw[12:], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if image is not None and image.dtype == np.uint16:
                    return image.astype(np.float32) / 1000.0
            except Exception:
                pass
        return None

    def _detect_tape_cross_centers(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # The parcel tops have a yellow/cream cross tape. It is much larger and
        # more stable than the small black label mark, so use it as the grasp
        # reference. Two masks are combined to cover both saturated yellow tape
        # and pale tape under simulator lighting.
        yellow = cv2.inRange(hsv, np.array([12, 35, 95]), np.array([48, 255, 255]))
        pale_yellow = cv2.inRange(hsv, np.array([15, 10, 135]), np.array([45, 130, 255]))
        mask = cv2.bitwise_or(yellow, pale_yellow)

        height, width = mask.shape[:2]
        roi = np.zeros_like(mask)
        roi[int(height * 0.12):int(height * 0.88), int(width * 0.08):int(width * 0.95)] = 255
        mask = cv2.bitwise_and(mask, roi)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.tape_min_area or area > self.tape_max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue
            aspect = float(w) / float(h)
            if aspect < 0.20 or aspect > 5.0:
                continue

            rect_area = float(w * h)
            fill_ratio = area / rect_area if rect_area > 0 else 0.0
            # A cross is neither a tiny speck nor a full filled rectangle.
            if fill_ratio < 0.12 or fill_ratio > 0.92:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            u = int(moments["m10"] / moments["m00"])
            v = int(moments["m01"] / moments["m00"])
            candidates.append((u, v, area))

        # Remove near-duplicates caused by one cross splitting into multiple contours.
        merged = []
        for candidate in sorted(candidates, key=lambda item: item[2], reverse=True):
            u, v, area = candidate
            if any((u - old_u) ** 2 + (v - old_v) ** 2 < 28 ** 2 for old_u, old_v, _ in merged):
                continue
            merged.append(candidate)

        merged.sort(key=lambda item: (item[1], item[0]))
        return merged

    def _detect_parcel_centers(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Table is green. Parcels are the colored/brown blobs on the table.
        green = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([95, 255, 255]))
        bright = cv2.inRange(hsv, np.array([0, 30, 45]), np.array([179, 255, 255]))
        mask = cv2.bitwise_and(cv2.bitwise_not(green), bright)

        height, width = mask.shape[:2]
        roi = np.zeros_like(mask)
        roi[int(height * 0.20):int(height * 0.88), int(width * 0.10):int(width * 0.90)] = 255
        mask = cv2.bitwise_and(mask, roi)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            u = int(moments["m10"] / moments["m00"])
            v = int(moments["m01"] / moments["m00"])
            centers.append((u, v, area))

        centers.sort(key=lambda item: (item[1], item[0]))
        return centers

    def _pixel_to_robot_point(self, u, v, depth, camera_info, camera_frame):
        if v < 0 or v >= depth.shape[0] or u < 0 or u >= depth.shape[1]:
            return None

        z = self._median_depth(depth, u, v)
        if z is None or not math.isfinite(z) or z <= 0.05:
            return None

        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]
        if fx == 0.0 or fy == 0.0:
            return None

        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy

        point = PointStamped()
        point.header.stamp = rospy.Time(0)
        point.header.frame_id = camera_frame or camera_info.header.frame_id
        point.point.x = x
        point.point.y = y
        point.point.z = z

        try:
            transform = self._tf_buffer.lookup_transform(
                self.output_frame,
                point.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.2),
            )
            robot_point = self._transform_point(point, transform)
            return [
                robot_point.point.x,
                robot_point.point.y,
                robot_point.point.z,
            ]
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "tf transform failed: %s", exc)
            return None

    def _transform_point(self, point, transform):
        # Avoid depending on tf2_geometry_msgs; apply translation-only fallback.
        # In the simulator camera frames are fixed by TF, so if rotation matters
        # this can be replaced by tf2_geometry_msgs after installing the package.
        q = transform.transform.rotation
        t = transform.transform.translation
        x, y, z = self._rotate_vector(
            [point.point.x, point.point.y, point.point.z],
            [q.x, q.y, q.z, q.w],
        )
        out = PointStamped()
        out.header = transform.header
        out.point.x = x + t.x
        out.point.y = y + t.y
        out.point.z = z + t.z
        return out

    def _rotate_vector(self, vector, quat_xyzw):
        x, y, z = vector
        qx, qy, qz, qw = quat_xyzw
        # q * v * q^-1, expanded.
        tx = 2.0 * (qy * z - qz * y)
        ty = 2.0 * (qz * x - qx * z)
        tz = 2.0 * (qx * y - qy * x)
        rx = x + qw * tx + (qy * tz - qz * ty)
        ry = y + qw * ty + (qz * tx - qx * tz)
        rz = z + qw * tz + (qx * ty - qy * tx)
        return rx, ry, rz

    def _median_depth(self, depth, u, v, radius=3):
        h, w = depth.shape[:2]
        x0 = max(0, u - radius)
        x1 = min(w, u + radius + 1)
        y0 = max(0, v - radius)
        y1 = min(h, v + radius + 1)
        patch = depth[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch)]
        valid = valid[valid > 0.05]
        if valid.size == 0:
            return None
        return float(np.median(valid))


def main():
    rospy.init_node("scene1_color_vision")
    Scene1ColorVision()
    rospy.spin()


if __name__ == "__main__":
    main()
