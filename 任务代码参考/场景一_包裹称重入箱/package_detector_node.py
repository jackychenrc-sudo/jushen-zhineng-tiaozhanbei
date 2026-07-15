#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOv8 package detector node for the Kuavo challenge cup simulation.

Subscribes to compressed RGB / depth images, runs YOLOv8 inference to detect
packages, computes 3D positions in the robot base_link frame, and publishes:
  - /detected_package_pose  (PoseStamped)     — best package 3D position
  - /detected_image/compressed (CompressedImage) — annotated RGB with boxes

Parameters (all under node private namespace ~):
  model_path   : path to best.pt       (default: ~/best.pt)
  conf_thresh  : confidence threshold   (default: 0.5)
  loop_rate    : main-loop frequency Hz (default: 10)
"""

import math
import struct
import threading
import os

import cv2
import numpy as np
import rospy
import tf2_ros

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import CameraInfo, CompressedImage


class PackageDetectorNode:
    def __init__(self):
        # ---- Parameters ----------------------------------------------------
        default_model = os.path.expanduser("~/best.pt")
        self.model_path  = rospy.get_param("~model_path",  default_model)
        self.conf_thresh = rospy.get_param("~conf_thresh", 0.5)
        self.loop_rate   = rospy.get_param("~loop_rate",   10.0)

        # ---- YOLO ----------------------------------------------------------
        self.model = self._load_model()

        # ---- Thread-safe cache of latest messages --------------------------
        self._lock = threading.Lock()
        self._latest_color_msg  = None   # CompressedImage
        self._latest_depth_msg  = None   # CompressedImage  (compressedDepth)
        self._camera_info       = None   # CameraInfo

        # ---- CV bridge -----------------------------------------------------
        self._bridge = CvBridge()

        # ---- TF2 -----------------------------------------------------------
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # ---- Publishers ----------------------------------------------------
        self._pose_pub = rospy.Publisher(
            "/detected_package_pose",  PoseStamped,     queue_size=10)
        self._vis_pub  = rospy.Publisher(
            "/detected_image/compressed", CompressedImage, queue_size=1)

        # ---- Subscribers ---------------------------------------------------
        rospy.Subscriber("/cam_h/color/image_raw/compressed",
                         CompressedImage, self._color_cb, queue_size=1)
        rospy.Subscriber("/cam_h/depth/image_raw/compressedDepth",
                         CompressedImage, self._depth_cb, queue_size=1)
        rospy.Subscriber("/cam_h/color/camera_info",
                         CameraInfo,      self._info_cb,  queue_size=1)

        rospy.loginfo("package_detector: model=%s  conf=%.2f  rate=%.1f Hz",
                      self.model_path, self.conf_thresh, self.loop_rate)

    # ----------------------------------------------------------------- callbacks
    def _color_cb(self, msg):
        with self._lock:
            self._latest_color_msg = msg

    def _depth_cb(self, msg):
        with self._lock:
            self._latest_depth_msg = msg

    def _info_cb(self, msg):
        with self._lock:
            self._camera_info = msg

    # ------------------------------------------------------------------ model
    def _load_model(self):
        try:
            from ultralytics import YOLO
            m = YOLO(self.model_path)
            rospy.loginfo("YOLO model loaded: %s", self.model_path)
            return m
        except Exception as e:
            rospy.logerr("Failed to load YOLO model from %s: %s",
                         self.model_path, e)
            raise

    # --------------------------------------------------------------- decoding
    def _decode_color(self, msg):
        """Decode compressed RGB via cv_bridge; fall back to manual imdecode."""
        try:
            return self._bridge.compressed_imgmsg_to_cv2(msg,
                                                         desired_encoding="bgr8")
        except CvBridgeError:
            pass
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logwarn("Color decode failed: %s", e)
            return None

    def _decode_compressed_depth(self, msg):
        """Decode compressedDepth (16UC1 PNG + possible header).

        The compressedDepth topic carries a PNG-encoded 16-bit depth image,
        sometimes with a small header prepended.  We probe several byte offsets
        and also try a struct-header interpretation.

        Returns: float32 numpy array in metres on success, None on failure.
        """
        raw = bytes(msg.data)
        if len(raw) < 8:
            rospy.logwarn("compressed depth msg too short (%d B)", len(raw))
            return None

        # Strategy 1 – probe PNG at offsets 0, 12, 16
        for offset in (0, 12, 16):
            if len(raw) <= offset:
                continue
            img = cv2.imdecode(np.frombuffer(raw[offset:], dtype=np.uint8),
                               cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.dtype == np.uint16:          # mm → m
                return img.astype(np.float32) / 1000.0
            if img.dtype == np.float32:         # already m
                return img

        # Strategy 2 – struct header (<format><compression><size> = 12 B) + PNG
        if len(raw) > 12:
            try:
                struct.unpack("iff", raw[:12])
                img = cv2.imdecode(np.frombuffer(raw[12:], dtype=np.uint8),
                                   cv2.IMREAD_UNCHANGED)
                if img is not None and img.dtype == np.uint16:
                    return img.astype(np.float32) / 1000.0
            except Exception:
                pass

        rospy.logwarn("compressed depth decode exhausted – all strategies failed")
        return None

    # -------------------------------------------------------- detection logic
    def _detect(self, color_img, depth_img, camera_info):
        """Run YOLO, keep only 'package' (cls 0) above threshold.

        Returns list of dicts:
          box          – (x1, y1, x2, y2) in pixels
          confidence   – float
          center_uv    – (u, v) box-centre pixel
          depth        – float (m) or None
          point3d_cam  – (x, y, z) in camera optical frame or None
          point3d_base – (x, y, z) in base_link or None
        """
        if self.model is None:
            return []

        results = self.model(color_img, verbose=False, imgsz=640,
                             conf=self.conf_thresh)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:                     # package class
                continue
            conf = float(box.conf[0])
            if conf < self.conf_thresh:
                continue

            # xyxy in pixels
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))

            depth       = None
            point3d_cam = None
            pt_base     = None

            if depth_img is not None and camera_info is not None:
                depth = self._sample_depth(depth_img, cx, cy, radius=2)
                if depth is not None and depth > 0.001 and math.isfinite(depth):
                    point3d_cam = self._pixel_to_camera(cx, cy, depth,
                                                        camera_info)
                    if point3d_cam is not None:
                        pt_base = self._to_base_link(point3d_cam,
                                                     camera_info.header.frame_id)

            detections.append(dict(
                box          = (x1, y1, x2, y2),
                confidence   = conf,
                center_uv    = (cx, cy),
                depth        = depth,
                point3d_cam  = point3d_cam,
                point3d_base = pt_base,
            ))

        # best confidence first
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    @staticmethod
    def _sample_depth(depth_img, u, v, radius=2):
        """Median depth in a small window around (u, v)."""
        h, w = depth_img.shape[:2]
        y0, y1 = max(0, v - radius), min(h, v + radius + 1)
        x0, x1 = max(0, u - radius), min(w, u + radius + 1)
        patch  = depth_img[y0:y1, x0:x1]
        valid  = patch[np.isfinite(patch) & (patch > 0.001)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    @staticmethod
    def _pixel_to_camera(u, v, depth, ci):
        """Pinhole: pixel → camera-frame (x, y, z)."""
        fx, _, cx = ci.K[0], ci.K[1], ci.K[2]
        fy, cy     = ci.K[4], ci.K[5]
        if fx == 0.0 or fy == 0.0:
            return None
        x = (float(u) - cx) * depth / fx
        y = (float(v) - cy) * depth / fy
        return (x, y, depth)

    def _to_base_link(self, pt_cam, cam_frame):
        """TF-lookup camera_frame → base_link, then apply transform."""
        ps = PointStamped()
        ps.header.stamp    = rospy.Time(0)
        ps.header.frame_id = cam_frame or "cam_h"
        ps.point.x, ps.point.y, ps.point.z = pt_cam

        try:
            xf = self._tf_buffer.lookup_transform("base_link",
                                                  ps.header.frame_id,
                                                  rospy.Time(0),
                                                  rospy.Duration(0.2))
            tp = self._apply_transform(ps, xf)
            return (tp.point.x, tp.point.y, tp.point.z)
        except Exception as e:
            rospy.logwarn_throttle(1.0,
                                   "TF %s→base_link: %s", ps.header.frame_id, e)
            return None

    # ------------------------------------------------------------ TF helpers
    @staticmethod
    def _apply_transform(pt, xf):
        """Manually rotate + translate (avoid tf2_geometry_msgs dep)."""
        q = xf.transform.rotation
        t = xf.transform.translation
        rx, ry, rz = PackageDetectorNode._quat_rotate(
            (pt.point.x, pt.point.y, pt.point.z), (q.x, q.y, q.z, q.w))
        out = PointStamped()
        out.header      = xf.header
        out.point.x     = rx + t.x
        out.point.y     = ry + t.y
        out.point.z     = rz + t.z
        return out

    @staticmethod
    def _quat_rotate(v, q_xyzw):
        x, y, z           = v
        qx, qy, qz, qw    = q_xyzw
        tx = 2.0 * (qy * z - qz * y)
        ty = 2.0 * (qz * x - qx * z)
        tz = 2.0 * (qx * y - qy * x)
        rx = x + qw * tx + (qy * tz - qz * ty)
        ry = y + qw * ty + (qz * tx - qx * tz)
        rz = z + qw * tz + (qx * ty - qy * tx)
        return rx, ry, rz

    # ---------------------------------------------------------- visualization
    def _draw(self, color_img, detections):
        """Draw green boxes + confidence tag + base_link coords on a copy."""
        if color_img is None:
            return None
        vis = color_img.copy()
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            conf = d["confidence"]

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, d["center_uv"], 4, (0, 255, 0), -1)

            label = "package {:.2f}".format(conf)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.55, 2)
            ly = max(0, y1 - 8)
            cv2.rectangle(vis, (x1, ly - th - 4), (x1 + tw + 4, ly + 2),
                          (0, 255, 0), -1)
            cv2.putText(vis, label, (x1 + 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
                        cv2.LINE_AA)

            pt = d.get("point3d_base")
            if pt is not None:
                info = "({:.3f}, {:.3f}, {:.3f})".format(*pt)
                cv2.putText(vis, info, (x1 + 2, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
                            cv2.LINE_AA)
        return vis

    def _publish_vis(self, color_img, detections, header):
        if color_img is None:
            return
        try:
            vis     = self._draw(color_img, detections)
            if vis is None:
                return
            encoded = cv2.imencode(
                ".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()
            msg = CompressedImage()
            msg.header.stamp    = rospy.Time.now()
            msg.header.frame_id = header.frame_id if header else "cam_h"
            msg.format          = "jpeg"
            msg.data            = encoded
            self._vis_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Vis pub failed: %s", e)

    # --------------------------------------------------------------- main loop
    def run(self):
        rate = rospy.Rate(self.loop_rate)
        while not rospy.is_shutdown():
            with self._lock:
                color_msg   = self._latest_color_msg
                depth_msg   = self._latest_depth_msg
                camera_info = self._camera_info

            if color_msg is None or camera_info is None:
                rospy.loginfo_throttle(10.0,
                    "Waiting for color image & camera_info …")
                rate.sleep()
                continue

            color_img = self._decode_color(color_msg)
            if color_img is None:
                rate.sleep()
                continue

            depth_img = None
            if depth_msg is not None:
                depth_img = self._decode_compressed_depth(depth_msg)

            # ----- detection -----
            try:
                detections = self._detect(color_img, depth_img, camera_info)
            except Exception as e:
                rospy.logwarn("YOLO inference error: %s", e)
                rate.sleep()
                continue

            # ----- publish best pose -----
            if detections and detections[0].get("point3d_base") is not None:
                p     = detections[0]["point3d_base"]
                pmsg  = PoseStamped()
                pmsg.header.stamp    = rospy.Time.now()
                pmsg.header.frame_id = "base_link"
                pmsg.pose.position.x, pmsg.pose.position.y, pmsg.pose.position.z = p
                pmsg.pose.orientation.w = 1.0
                self._pose_pub.publish(pmsg)

                rospy.loginfo(
                    "Detected %d package(s).  Best  base_link=(%.3f, %.3f, %.3f)  conf=%.2f",
                    len(detections), *p, detections[0]["confidence"])
            else:
                rospy.loginfo_throttle(5.0, "No package detected.")

            # ----- publish visualization -----
            self._publish_vis(color_img, detections,
                              color_msg.header if color_msg else None)

            rate.sleep()


def main():
    rospy.init_node("package_detector", anonymous=True)
    try:
        node = PackageDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("package_detector fatal: %s", e)
        raise


if __name__ == "__main__":
    main()
