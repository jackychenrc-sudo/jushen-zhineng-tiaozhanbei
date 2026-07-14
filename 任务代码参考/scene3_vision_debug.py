#!/usr/bin/env python3
"""
Scene 3 standalone vision validator.

Run:
  rosrun challenge_cup_task_template scene3_vision_debug.py _model_path:=/abs/path/to/best.pt
"""

import argparse
import os
import time


class Scene3VisionDebugger(object):
    def __init__(self):
        import cv2
        import numpy as np
        import rospy
        import tf2_geometry_msgs  # noqa: F401
        import tf2_ros
        from sensor_msgs.msg import CameraInfo, CompressedImage

        self.cv2 = cv2
        self.np = np
        self.rospy = rospy

        self.color_topic = rospy.get_param(
            "~color_topic",
            "/cam_h/color/image_raw/compressed",
        )
        self.depth_topic = rospy.get_param(
            "~depth_topic",
            "/cam_h/depth/image_raw/compressedDepth",
        )
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic",
            "/cam_h/color/camera_info",
        )
        self.debug_topic = rospy.get_param(
            "~debug_topic",
            "/challenge_cup_task_template/scene3/debug/compressed",
        )
        self.grasp_point_camera_topic = rospy.get_param(
            "~grasp_point_camera_topic",
            "/challenge_cup_task_template/scene3/grasp_point_camera",
        )
        self.grasp_point_base_topic = rospy.get_param(
            "~grasp_point_base_topic",
            "/challenge_cup_task_template/scene3/grasp_point_base",
        )
        self.grasp_point_odom_topic = rospy.get_param(
            "~grasp_point_odom_topic",
            "/challenge_cup_task_template/scene3/grasp_point_odom",
        )
        self.mask_topic = rospy.get_param(
            "~mask_topic",
            "/challenge_cup_task_template/scene3/mask/compressed",
        )
        self.base_frame_candidates = rospy.get_param(
            "~base_frame_candidates",
            ["base_link", "torso"],
        )
        self.world_frame = rospy.get_param("~world_frame", "odom")

        self.model_path = rospy.get_param("~model_path", "")
        self.yolo_device = rospy.get_param("~device", "")
        self.yolo_confidence = float(rospy.get_param("~confidence_threshold", 0.35))
        self.yolo_imgsz = int(rospy.get_param("~imgsz", 960))
        self.prefer_high_z = bool(rospy.get_param("~prefer_high_z", True))
        self.z_selection_margin = float(rospy.get_param("~z_selection_margin", 0.10))
        self.search_roi = self._parse_roi(
            rospy.get_param("~search_roi", [0.05, 0.05, 0.95, 0.95])
        )
        self.target_class_names = set(
            self._normalize_class_name(item)
            for item in rospy.get_param("~target_class_names", ["tray", "smt_tray"])
            if str(item).strip()
        )

        self.last_head_rgb = None
        self.last_head_depth = None
        self.last_head_cam_info = None
        self.last_detection = None
        self._last_detection_log_time = 0.0
        self._last_no_detection_log_time = 0.0

        self.yolo_model = None
        self.yolo_error = None
        self._load_yolo_model()

        from geometry_msgs.msg import PointStamped

        self.debug_image_pub = rospy.Publisher(
            self.debug_topic,
            CompressedImage,
            queue_size=1,
        )
        self.grasp_point_camera_pub = rospy.Publisher(
            self.grasp_point_camera_topic,
            PointStamped,
            queue_size=1,
        )
        self.grasp_point_base_pub = rospy.Publisher(
            self.grasp_point_base_topic,
            PointStamped,
            queue_size=1,
        )
        self.grasp_point_odom_pub = rospy.Publisher(
            self.grasp_point_odom_topic,
            PointStamped,
            queue_size=1,
        )
        self.mask_image_pub = rospy.Publisher(
            self.mask_topic,
            CompressedImage,
            queue_size=1,
        )

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber(
            self.color_topic,
            CompressedImage,
            self._head_rgb_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            self.depth_topic,
            CompressedImage,
            self._head_depth_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self._head_cam_info_cb,
            queue_size=1,
        )

        rospy.loginfo("scene3 vision debug: color_topic=%s", self.color_topic)
        rospy.loginfo("scene3 vision debug: depth_topic=%s", self.depth_topic)
        rospy.loginfo("scene3 vision debug: camera_info_topic=%s", self.camera_info_topic)
        rospy.loginfo("scene3 vision debug: debug_topic=%s", self.debug_topic)
        rospy.loginfo("scene3 vision debug: model_path=%s", self.model_path or "<unset>")
        rospy.loginfo("scene3 vision debug: world_frame=%s", self.world_frame)
        rospy.loginfo(
            "scene3 vision debug: prefer_high_z=%s z_selection_margin=%.3f",
            self.prefer_high_z,
            self.z_selection_margin,
        )
        rospy.loginfo(
            "scene3 vision debug: target_class_names=%s",
            sorted(self.target_class_names) if self.target_class_names else ["<all>"],
        )

    def _parse_roi(self, roi_value):
        if not isinstance(roi_value, (list, tuple)) or len(roi_value) != 4:
            return (0.0, 0.0, 1.0, 1.0)
        try:
            x1, y1, x2, y2 = [float(item) for item in roi_value]
        except Exception:
            return (0.0, 0.0, 1.0, 1.0)
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            return (0.0, 0.0, 1.0, 1.0)
        return (x1, y1, x2, y2)

    def _normalize_class_name(self, value):
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    def _load_yolo_model(self):
        if not self.model_path:
            self.yolo_error = "ROS param ~model_path is empty"
            self.rospy.logwarn("scene3 vision debug: %s", self.yolo_error)
            return
        if not os.path.isfile(self.model_path):
            self.yolo_error = "model file does not exist: {}".format(self.model_path)
            self.rospy.logerr("scene3 vision debug: %s", self.yolo_error)
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.yolo_error = "failed to import ultralytics: {}".format(exc)
            self.rospy.logerr("scene3 vision debug: %s", self.yolo_error)
            return
        try:
            self.yolo_model = YOLO(self.model_path)
            if self.yolo_device:
                self.yolo_model.to(self.yolo_device)
            self.yolo_error = None
            self.rospy.loginfo(
                "scene3 vision debug: YOLO model loaded successfully (%s)",
                self.model_path,
            )
        except Exception as exc:
            self.yolo_error = "failed to load YOLO model: {}".format(exc)
            self.rospy.logerr("scene3 vision debug: %s", self.yolo_error)

    def _head_rgb_cb(self, msg):
        self.last_head_rgb = msg
        self._run_vision()

    def _head_depth_cb(self, msg):
        self.last_head_depth = msg

    def _head_cam_info_cb(self, msg):
        self.last_head_cam_info = msg

    def _decode_color_image(self, msg):
        encoded = self.np.frombuffer(msg.data, dtype=self.np.uint8)
        return self.cv2.imdecode(encoded, self.cv2.IMREAD_COLOR)

    def _decode_depth_image(self, msg):
        if len(msg.data) <= 12:
            return None
        png_bytes = self.np.frombuffer(msg.data[12:], dtype=self.np.uint8)
        return self.cv2.imdecode(png_bytes, self.cv2.IMREAD_UNCHANGED)

    def _find_valid_depth_mm(self, depth_image, px, py, radius=4):
        h, w = depth_image.shape[:2]
        x0 = max(0, px - radius)
        x1 = min(w, px + radius + 1)
        y0 = max(0, py - radius)
        y1 = min(h, py + radius + 1)
        patch = depth_image[y0:y1, x0:x1]
        valid = patch[(patch > 50) & (patch < 10000)]
        if valid.size == 0:
            return None
        return float(self.np.median(valid))

    def _find_valid_depth_in_bbox_mm(self, depth_image, bbox):
        x, y, w, h = bbox
        height, width = depth_image.shape[:2]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(width, x + w)
        y1 = min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        patch = depth_image[y0:y1, x0:x1]
        valid = patch[(patch > 50) & (patch < 10000)]
        if valid.size == 0:
            return None
        return float(self.np.percentile(valid, 35))

    def _pixel_to_camera_xyz(self, u, v, depth_mm):
        if self.last_head_cam_info is None:
            return None
        fx = self.last_head_cam_info.K[0]
        fy = self.last_head_cam_info.K[4]
        cx = self.last_head_cam_info.K[2]
        cy = self.last_head_cam_info.K[5]
        if fx == 0 or fy == 0:
            return None
        z = depth_mm / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return (float(x), float(y), float(z))

    def _publish_point_stamped(self, publisher, frame_id, xyz):
        from geometry_msgs.msg import PointStamped

        msg = PointStamped()
        msg.header.stamp = self.rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.point.x = xyz[0]
        msg.point.y = xyz[1]
        msg.point.z = xyz[2]
        publisher.publish(msg)
        return msg

    def _build_point_stamped(self, frame_id, xyz):
        from geometry_msgs.msg import PointStamped

        msg = PointStamped()
        msg.header.stamp = self.rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.point.x = xyz[0]
        msg.point.y = xyz[1]
        msg.point.z = xyz[2]
        return msg

    def _transform_point_to_first_available_frame(self, point_msg, frame_names):
        for frame_name in frame_names:
            try:
                return self.tf_buffer.transform(
                    point_msg,
                    frame_name,
                    timeout=self.rospy.Duration(0.1),
                )
            except Exception:
                continue
        return None

    def _transform_point_to_base(self, point_msg):
        return self._transform_point_to_first_available_frame(
            point_msg,
            self.base_frame_candidates,
        )

    def _transform_point_to_world(self, point_msg):
        try:
            return self.tf_buffer.transform(
                point_msg,
                self.world_frame,
                timeout=self.rospy.Duration(0.1),
            )
        except Exception:
            return None

    def _publish_debug_image(self, image_bgr, stamp, frame_id):
        from sensor_msgs.msg import CompressedImage

        ok, encoded = self.cv2.imencode(".jpg", image_bgr)
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.debug_image_pub.publish(msg)

    def _publish_mask_image(self, mask_gray, stamp, frame_id):
        from sensor_msgs.msg import CompressedImage

        mask_bgr = self.cv2.cvtColor(mask_gray, self.cv2.COLOR_GRAY2BGR)
        ok, encoded = self.cv2.imencode(".jpg", mask_bgr)
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.mask_image_pub.publish(msg)

    def _match_target_class(self, class_name):
        if not self.target_class_names:
            return True
        return self._normalize_class_name(class_name) in self.target_class_names

    def _detect_tray(self, image_bgr, depth_image, frame_id):
        h, w = image_bgr.shape[:2]
        roi_x0 = int(w * self.search_roi[0])
        roi_y0 = int(h * self.search_roi[1])
        roi_x1 = int(w * self.search_roi[2])
        roi_y1 = int(h * self.search_roi[3])
        roi = image_bgr[roi_y0:roi_y1, roi_x0:roi_x1]

        full_mask = self.np.zeros((h, w), dtype=self.np.uint8)
        if self.yolo_model is None:
            return None, full_mask, []

        try:
            results = self.yolo_model.predict(
                roi,
                conf=self.yolo_confidence,
                imgsz=self.yolo_imgsz,
                verbose=False,
                device=self.yolo_device if self.yolo_device else None,
            )
        except Exception as exc:
            self.rospy.logerr_throttle(2.0, "scene3 vision debug: YOLO predict failed: %s", exc)
            return None, full_mask, []

        candidates = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            names = result.names
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for index in range(len(xyxy)):
                x1, y1, x2, y2 = [int(round(value)) for value in xyxy[index].tolist()]
                x1 = max(0, min(roi.shape[1] - 1, x1))
                y1 = max(0, min(roi.shape[0] - 1, y1))
                x2 = max(x1 + 1, min(roi.shape[1], x2))
                y2 = max(y1 + 1, min(roi.shape[0], y2))
                class_id = int(classes[index])
                class_name = str(names.get(class_id, class_id))
                if not self._match_target_class(class_name):
                    continue

                full_x = roi_x0 + x1
                full_y = roi_y0 + y1
                box_w = x2 - x1
                box_h = y2 - y1
                center_px = (full_x + box_w // 2, full_y + box_h // 2)

                depth_mm = self._find_valid_depth_in_bbox_mm(
                    depth_image,
                    (full_x, full_y, box_w, box_h),
                )
                if depth_mm is None:
                    depth_mm = self._find_valid_depth_mm(
                        depth_image,
                        center_px[0],
                        center_px[1],
                        radius=6,
                    )
                if depth_mm is None:
                    continue

                xyz = self._pixel_to_camera_xyz(
                    center_px[0],
                    center_px[1],
                    depth_mm,
                )
                if xyz is None:
                    continue

                camera_point = self._build_point_stamped(frame_id, xyz)
                base_point = self._transform_point_to_base(camera_point)
                world_point = self._transform_point_to_world(camera_point)
                selection_point = world_point if world_point is not None else base_point
                selection_z = None if selection_point is None else float(selection_point.point.z)
                selection_frame = "" if selection_point is None else str(selection_point.header.frame_id)

                confidence = float(confs[index])
                candidate = {
                    "bbox": (full_x, full_y, box_w, box_h),
                    "center_px": center_px,
                    "depth_mm": depth_mm,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "camera_xyz": xyz,
                    "grasp_point_px": center_px,
                    "base_point": base_point,
                    "world_point": world_point,
                    "selection_z": selection_z,
                    "selection_frame": selection_frame,
                }
                candidates.append(candidate)
                full_mask[full_y : full_y + box_h, full_x : full_x + box_w] = 255

        if not candidates:
            return None, full_mask, candidates

        selection_pool = candidates
        if self.prefer_high_z:
            z_candidates = [cand for cand in candidates if cand["selection_z"] is not None]
            if z_candidates:
                max_z = max(cand["selection_z"] for cand in z_candidates)
                selection_pool = [
                    cand for cand in z_candidates
                    if cand["selection_z"] >= max_z - self.z_selection_margin
                ]

        best = max(
            selection_pool,
            key=lambda cand: (
                float(cand["confidence"]),
                float(cand["bbox"][2] * cand["bbox"][3]),
            ),
        )
        return best, full_mask, candidates

    def _annotate_detection(self, image_bgr, detection):
        annotated = image_bgr.copy()
        x, y, w, h = detection["bbox"]
        px, py = detection["grasp_point_px"]
        xyz = detection["camera_xyz"]

        self.cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.cv2.circle(annotated, (px, py), 6, (0, 0, 255), -1)
        self.cv2.putText(
            annotated,
            "{} {:.2f}".format(detection["class_name"], detection["confidence"]),
            (x, max(24, y - 8)),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        self.cv2.putText(
            annotated,
            "grasp px=({}, {})".format(px, py),
            (x, y + h + 24),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        self.cv2.putText(
            annotated,
            "cam xyz=({:.3f}, {:.3f}, {:.3f})m".format(xyz[0], xyz[1], xyz[2]),
            (x, y + h + 50),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )
        return annotated

    def _run_vision(self):
        if (
            self.last_head_rgb is None
            or self.last_head_depth is None
            or self.last_head_cam_info is None
        ):
            return

        if self.yolo_model is None:
            self.rospy.logerr_throttle(
                5.0,
                "scene3 vision debug: YOLO is unavailable (%s)",
                self.yolo_error or "unknown error",
            )
            return

        try:
            color = self._decode_color_image(self.last_head_rgb)
            depth = self._decode_depth_image(self.last_head_depth)
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "scene3 vision debug: decode failed: %s", exc)
            return

        if color is None or depth is None:
            return

        if depth.shape[:2] != color.shape[:2]:
            depth = self.cv2.resize(
                depth,
                (color.shape[1], color.shape[0]),
                interpolation=self.cv2.INTER_NEAREST,
            )

        detection, full_mask, candidates = self._detect_tray(color, depth, frame_id=self.last_head_rgb.header.frame_id)
        stamp = self.last_head_rgb.header.stamp
        frame_id = self.last_head_rgb.header.frame_id

        self._publish_mask_image(full_mask, stamp, frame_id)
        annotated = color.copy()
        for cand in candidates:
            x, y, w, h = cand["bbox"]
            self.cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 1)
            self.cv2.putText(
                annotated,
                "{} {:.2f}".format(cand["class_name"], cand["confidence"]),
                (x, max(18, y - 4)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

        if detection is None:
            now = time.time()
            if now - self._last_no_detection_log_time > 2.0:
                self.rospy.loginfo(
                    "scene3 vision debug: no tray detected, YOLO candidates=%d",
                    len(candidates),
                )
                self._last_no_detection_log_time = now
            self._publish_debug_image(annotated, stamp, frame_id)
            self.last_detection = None
            return

        self.last_detection = detection
        annotated = self._annotate_detection(annotated, detection)
        self._publish_debug_image(annotated, stamp, frame_id)

        camera_point = self._publish_point_stamped(
            self.grasp_point_camera_pub,
            frame_id,
            detection["camera_xyz"],
        )
        base_point = detection.get("base_point")
        world_point = detection.get("world_point")
        if base_point is not None:
            self.grasp_point_base_pub.publish(base_point)
        if world_point is not None:
            self.grasp_point_odom_pub.publish(world_point)

        now = time.time()
        if now - self._last_detection_log_time > 1.0:
            self.rospy.loginfo(
                "scene3 vision debug: class=%s conf=%.2f bbox=%s grasp_px=%s cam_xyz=(%.3f, %.3f, %.3f) pick_z=%s%s%s",
                detection["class_name"],
                detection["confidence"],
                detection["bbox"],
                detection["grasp_point_px"],
                detection["camera_xyz"][0],
                detection["camera_xyz"][1],
                detection["camera_xyz"][2],
                "n/a" if detection.get("selection_z") is None else "{:.3f}@{}".format(
                    detection["selection_z"],
                    detection.get("selection_frame", "?"),
                ),
                "" if base_point is None else " base=({:.3f}, {:.3f}, {:.3f})".format(
                    base_point.point.x,
                    base_point.point.y,
                    base_point.point.z,
                ),
                "" if world_point is None else " odom=({:.3f}, {:.3f}, {:.3f})".format(
                    world_point.point.x,
                    world_point.point.y,
                    world_point.point.z,
                ),
            )
            self._last_detection_log_time = now


def main():
    parser = argparse.ArgumentParser(description="Scene3 standalone vision validator")
    parser.add_argument(
        "--node-name",
        default="scene3_vision_debug",
        help="ROS node name",
    )
    args, _ = parser.parse_known_args()

    import rospy

    rospy.init_node(args.node_name, anonymous=False)
    Scene3VisionDebugger()
    rospy.loginfo("scene3 vision debug: node started")
    rospy.spin()


if __name__ == "__main__":
    main()
