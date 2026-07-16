#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Noetic YOLOv8 视觉检测节点（可独立运行 / 作为库导入）
订阅 RGB、深度图、相机内参，检测 3 类零件并输出 /base_link 下的 3D 位置。

作为库导入时（如 challenge_task.py 内部使用），用 start_detector() 在后台线程启动。
独立运行时，通过命令行参数指定模型路径和置信度阈值。

依赖:
    pip install ultralytics opencv-python
    标准 ROS Noetic 环境 (rospy, sensor_msgs, cv_bridge, tf2_ros, tf2_geometry_msgs)

独立运行示例:
    rosrun your_package package_detector_node.py _model_path:=/path/to/best.pt
"""

import threading
import math
import struct
import sys

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseArray, Pose, Point
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Header
import tf2_geometry_msgs  # noqa: F401  registers transform() for PointStamped
import tf2_ros


# 类别名称（与 best.pt 训练时的 class id 一致）
CLASS_NAMES = {0: "part_A", 1: "part_B", 2: "part_C"}

# 默认话题映射
DEFAULT_RGB_TOPIC = "/cam_h/color/image_raw/compressed"
DEFAULT_DEPTH_TOPIC = "/cam_h/depth/image_raw/compressedDepth"
DEFAULT_CAMERA_INFO_TOPIC = "/cam_h/color/camera_info"
DEFAULT_PUBLISH_TOPIC = "/scene1/parcel_points"
DEFAULT_DEBUG_TOPIC = "/detected_image/compressed"


class DetectObjectsNode:
    """YOLOv8 零件检测 + 3D 定位节点。

    注意：本类不调用 rospy.init_node()，可由外部统一初始化。
    """

    def __init__(self, model_path, conf_threshold=0.5,
                 rgb_topic=DEFAULT_RGB_TOPIC,
                 depth_topic=DEFAULT_DEPTH_TOPIC,
                 camera_info_topic=DEFAULT_CAMERA_INFO_TOPIC,
                 publish_topic=DEFAULT_PUBLISH_TOPIC,
                 debug_topic=DEFAULT_DEBUG_TOPIC):
        self.model_path = model_path
        self.conf_threshold = conf_threshold

        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.publish_topic = publish_topic
        self.debug_topic = debug_topic

        # 线程安全的消息缓存
        self._lock = threading.Lock()
        self._rgb_msg = None
        self._depth_msg = None
        self._camera_info = None
        self.bridge = CvBridge()

        rospy.loginfo("Loading YOLO model from: %s", model_path)
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            rospy.logfatal(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
            sys.exit(1)
        rospy.loginfo("YOLO model loaded successfully.")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 数据订阅
        self.sub_rgb = rospy.Subscriber(
            self.rgb_topic,
            CompressedImage,
            self._cb_rgb,
            queue_size=1,
        )
        self.sub_depth = rospy.Subscriber(
            self.depth_topic,
            CompressedImage,
            self._cb_depth,
            queue_size=1,
        )
        self.sub_info = rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self._cb_info,
            queue_size=1,
        )

        # 发布器
        self.pub_poses = rospy.Publisher(self.publish_topic, PoseArray, queue_size=10)
        self.pub_debug = rospy.Publisher(
            self.debug_topic, CompressedImage, queue_size=10
        )

    # ------------------------------------------------------------------
    #  回调：只负责拷贝最新消息
    # ------------------------------------------------------------------
    def _cb_rgb(self, msg):
        with self._lock:
            self._rgb_msg = msg

    def _cb_depth(self, msg):
        with self._lock:
            self._depth_msg = msg

    def _cb_info(self, msg):
        with self._lock:
            if self._camera_info is None:
                rospy.loginfo("CameraInfo received - intrinsics available.")
            self._camera_info = msg

    # ------------------------------------------------------------------
    #  深度图解码 (健壮处理compressedDepth中可能含有的各种头部)
    # ------------------------------------------------------------------
    def _decode_compressed_depth(self, msg):
        """解码 compressedDepth 话题 (16UC1 PNG + 可选头部)。

        尝试多个字节偏移量和 struct 头部解释。
        返回: float32 numpy array (米) 成功, None 失败。
        """
        raw = bytes(msg.data)
        if len(raw) < 8:
            rospy.logwarn("compressed depth msg too short (%d B)", len(raw))
            return None

        # 策略1 — 在偏移 0, 12, 16 处探测 PNG
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

        # 策略2 — struct 头部 (<format><compression><size> = 12 B) + PNG
        if len(raw) > 12:
            try:
                struct.unpack("iff", raw[:12])
                img = cv2.imdecode(np.frombuffer(raw[12:], dtype=np.uint8),
                                   cv2.IMREAD_UNCHANGED)
                if img is not None and img.dtype == np.uint16:
                    return img.astype(np.float32) / 1000.0
            except Exception:
                pass

        rospy.logwarn("compressed depth decode exhausted — all strategies failed")
        return None

    # ------------------------------------------------------------------
    #  RGB decode (with fallback)
    # ------------------------------------------------------------------
    def _decode_color(self, msg):
        """Decode compressed RGB; try CvBridge first, fallback to imdecode."""
        try:
            return self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError:
            pass
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logwarn("Color decode failed: %s", e)
            return None

    # ------------------------------------------------------------------
    #  相机坐标系 → 3D 点
    # ------------------------------------------------------------------
    @staticmethod
    def _back_project(u, v, z_m, fx, fy, cx, cy):
        """像素 (u,v) + 深度 z_m (米) -> 相机坐标系下 3D 点。"""
        pt = Point()
        pt.x = (u - cx) * z_m / fx
        pt.y = (v - cy) * z_m / fy
        pt.z = z_m
        return pt

    # ------------------------------------------------------------------
    #  主循环
    # ------------------------------------------------------------------
    def run(self):
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            # 1. 取最新消息
            with self._lock:
                rgb = self._rgb_msg
                depth = self._depth_msg
                info = self._camera_info

            if rgb is None:
                rospy.logwarn_throttle(5, "Waiting for RGB image...")
                rate.sleep()
                continue
            if depth is None:
                rospy.logwarn_throttle(5, "Waiting for depth image...")
                rate.sleep()
                continue
            if info is None:
                rospy.logwarn_throttle(5, "Waiting for camera info...")
                rate.sleep()
                continue

            # 2. RGB decode (with fallback)
            cv_rgb = self._decode_color(rgb)
            if cv_rgb is None:
                rate.sleep()
                continue

            H, W = cv_rgb.shape[:2]

            # 3. YOLO 推理
            try:
                results = self.model(cv_rgb, conf=self.conf_threshold, verbose=False)
            except Exception as e:
                rospy.logerr("YOLO inference failed: %s", e)
                rate.sleep()
                continue

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                header = Header(stamp=rospy.Time.now(), frame_id="base_link")
                self.pub_poses.publish(PoseArray(header=header, poses=[]))
                rate.sleep()
                continue

            # 4. 解码深度图 (使用健壮的 compressedDepth 解码器)
            cv_depth = self._decode_compressed_depth(depth)
            if cv_depth is None:
                rospy.logerr(
                    "Depth decode returned None (compressedDepth parse failed)"
                )
                rate.sleep()
                continue
            # cv_depth 已经是 float32 米单位

            # 5. 提取相机内参
            K = info.K
            fx, fy = K[0], K[4]
            cx, cy = K[2], K[5]

            # 6. 遍历检测框，计算 3D 位置
            detections_3d = []
            annotated = cv_rgb.copy()

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0

                u_int = max(0, min(W - 1, int(round(u))))
                v_int = max(0, min(H - 1, int(round(v))))

                d_val = float(cv_depth[v_int, u_int])
                if d_val <= 0.001 or not math.isfinite(d_val):
                    rospy.logdebug(
                        "Skip at (%.1f, %.1f): depth=%.3f m invalid", u, v, d_val
                    )
                    continue

                d_m = d_val  # 已经是米单位

                pt_cam = self._back_project(u, v, d_m, fx, fy, cx, cy)

                cam_point = PointStamped()
                cam_point.header.frame_id = info.header.frame_id or "cam_h"
                cam_point.header.stamp = rospy.Time(0)
                cam_point.point = pt_cam

                try:
                    base_point = self.tf_buffer.transform(
                        cam_point, "base_link", rospy.Duration(0.1)
                    )
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as e:
                    rospy.logwarn("TF transform /cam_h -> /base_link failed: %s", e)
                    continue

                pose = Pose()
                pose.position = base_point.point

                # Estimate part surface yaw from depth patch
                try:
                    yaw = self._estimate_grasp_yaw(
                        u, v, cv_depth, info,
                        bbox_width=float(x2 - x1),
                        bbox_height=float(y2 - y1),
                    )
                except Exception as e:
                    rospy.logwarn("Yaw estimation failed: %s", e)
                    yaw = None

                if yaw is not None:
                    qx, qy, qz, qw = self._quaternion_from_yaw(yaw)
                    pose.orientation.x = qx
                    pose.orientation.y = qy
                    pose.orientation.z = qz
                    pose.orientation.w = qw
                else:
                    pose.orientation.w = 1.0

                detections_3d.append((label, conf, pose))

                self._draw_detection(annotated, x1, y1, x2, y2, conf, label)

            # 7. 发布 PoseArray
            header = Header(stamp=rospy.Time.now(), frame_id="base_link")
            pose_array = PoseArray(
                header=header,
                poses=[item[2] for item in detections_3d],
            )
            self.pub_poses.publish(pose_array)

            if detections_3d:
                info_str = "; ".join(
                    f"{label} [{conf:.2f}] "
                    f"({p.position.x:.3f}, {p.position.y:.3f}, {p.position.z:.3f})"
                    for label, conf, p in detections_3d
                )
                rospy.loginfo("Detected %d objects: %s", len(detections_3d), info_str)
            else:
                rospy.logdebug("No valid 3D detections this frame.")

            # 8. 发布调试图像
            self._publish_debug_image(annotated)

            rate.sleep()

    # ------------------------------------------------------------------
    #  绘图辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_detection(img, x1, y1, x2, y2, conf, label):
        """在图像上绘制检测框和标签（in-place）。"""
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, pt1, (pt1[0] + tw + 4, pt1[1] - th - 6), (0, 255, 0), -1)
        cv2.putText(
            img, text, (pt1[0] + 2, pt1[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    #  Based on depth image to estimate part surface yaw angle
    # ------------------------------------------------------------------
    def _estimate_grasp_yaw(self, box_center_u, box_center_v, depth_image, camera_info,
                            bbox_width=None, bbox_height=None, patch_scale=1.5):
        """
        Estimate part surface orientation (yaw angle) from local depth point cloud.

        Pipeline:
          1. Extract local depth patch centered on the detection box.
          2. Back-project valid pixels to camera coordinate system.
          3. Statistical outlier removal.
          4. RANSAC (preferred) or least-squares plane fitting.
          5. Transform normal vector to target frame and compute yaw on XOY plane.

        Args:
            box_center_u, box_center_v: Pixel coordinates of box center.
            depth_image: Decoded depth image (HxW, float32, meters).
            camera_info: CameraInfo message with intrinsics.
            bbox_width, bbox_height: Bounding box size in pixels.
            patch_scale: Expansion factor for local patch, default 1.5.

        Returns:
            float: yaw angle (radians) in [-pi, pi], or None on failure.
        """
        import math

        H, W = depth_image.shape[:2]
        K = camera_info.K
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]

        # ---- 1. Determine local patch boundaries ----
        if bbox_width is not None and bbox_height is not None and bbox_width > 0 and bbox_height > 0:
            half_size = patch_scale * max(bbox_width, bbox_height) / 2.0
        else:
            half_size = 30.0

        u_min = max(0, int(round(box_center_u - half_size)))
        u_max = min(W, int(round(box_center_u + half_size)))
        v_min = max(0, int(round(box_center_v - half_size)))
        v_max = min(H, int(round(box_center_v + half_size)))

        patch = depth_image[v_min:v_max, u_min:u_max]
        patch_h, patch_w = patch.shape
        if patch_h < 5 or patch_w < 5:
            rospy.logdebug("Depth patch too small (%dx%d), cannot estimate yaw", patch_w, patch_h)
            return None

        # ---- 2. Generate local point cloud (camera frame) ----
        v_grid, u_grid = np.mgrid[v_min:v_max, u_min:u_max]
        valid = (patch > 0.005) & np.isfinite(patch)
        n_valid = np.count_nonzero(valid)
        if n_valid < 20:
            rospy.logdebug("Too few valid depth points (%d) in patch", n_valid)
            return None

        u_pts = u_grid[valid].astype(np.float32)
        v_pts = v_grid[valid].astype(np.float32)
        d_pts = patch[valid].astype(np.float32)

        x_cam = (u_pts - cx) * d_pts / fx
        y_cam = (v_pts - cy) * d_pts / fy
        z_cam = d_pts
        points = np.column_stack((x_cam, y_cam, z_cam))

        # ---- 3. Statistical outlier removal ----
        center = np.mean(points, axis=0)
        dists = np.linalg.norm(points - center, axis=1)
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        if std_dist > 1e-8:
            inlier_mask = dists <= (mean_dist + 2.0 * std_dist)
            points = points[inlier_mask]

        if points.shape[0] < 20:
            rospy.logdebug("After outlier removal, only %d points remain", points.shape[0])
            return None

        # ---- 4. Plane fitting (RANSAC first, least-squares fallback) ----
        normal = self._fit_plane_ransac(points)
        if normal is None:
            normal = self._fit_plane_lstsq(points)
        if normal is None:
            rospy.logwarn("Plane fitting failed entirely")
            return None

        # ---- 5. Transform normal to target frame and compute yaw ----
        normal = self._normal_to_target_frame(normal, camera_info)
        if normal is None:
            return None

        proj = np.array([normal[0], normal[1], 0.0])
        proj_norm = np.linalg.norm(proj)
        if proj_norm < 1e-6:
            return 0.0

        proj = proj / proj_norm
        dot = max(-1.0, min(1.0, proj[0]))
        yaw = math.acos(dot)
        if proj[1] < 0:
            yaw = -yaw
        return yaw

    # ------------------------------------------------------------------
    #  RANSAC plane fitting (sklearn preferred, custom fallback)
    # ------------------------------------------------------------------
    def _fit_plane_ransac(self, points, max_trials=200, threshold=0.01, min_inlier_ratio=0.4):
        """Fit a plane using RANSAC: ax + by + cz + d = 0.
        Returns normalized normal (a, b, c) or None."""
        try:
            from sklearn.linear_model import RANSACRegressor
            Xy = points[:, :2]
            z = points[:, 2]
            ransac = RANSACRegressor(
                min_samples=3,
                residual_threshold=threshold,
                max_trials=max_trials,
                random_state=42,
            )
            ransac.fit(Xy, z)
            a, b = ransac.estimator_.coef_
            d = ransac.estimator_.intercept_
            normal_full = np.array([a, b, -1.0, d], dtype=np.float64)
        except ImportError:
            normal_full = self._ransac_custom(points, max_trials, threshold, min_inlier_ratio)
        except Exception:
            normal_full = self._ransac_custom(points, max_trials, threshold, min_inlier_ratio)

        if normal_full is None:
            return None
        a, b, c, _d = normal_full
        norm_len = math.sqrt(a*a + b*b + c*c)
        if norm_len < 1e-12:
            return None
        return np.array([a, b, c]) / norm_len

    # ------------------------------------------------------------------
    #  Custom RANSAC plane fitting (no sklearn dependency)
    # ------------------------------------------------------------------
    @staticmethod
    def _ransac_custom(points, max_trials=200, threshold=0.01, min_inlier_ratio=0.4):
        """Custom RANSAC plane fitting.
        Returns (a, b, c, d) or None."""
        N = points.shape[0]
        if N < 3:
            return None

        rng = np.random.RandomState(42)
        best_inliers = None
        best_count = 0

        for _ in range(max_trials):
            idx = rng.choice(N, 3, replace=False)
            p1, p2, p3 = points[idx]
            v1 = p2 - p1
            v2 = p3 - p1
            n_vec = np.cross(v1, v2)
            n_len = np.linalg.norm(n_vec)
            if n_len < 1e-12:
                continue
            n_vec = n_vec / n_len
            d_val = -np.dot(n_vec, p1)
            dists = np.abs(points @ n_vec + d_val)
            inlier_mask = dists < threshold
            n_inliers = np.count_nonzero(inlier_mask)
            if n_inliers > best_count:
                best_count = n_inliers
                best_inliers = inlier_mask
                if n_inliers >= min_inlier_ratio * N:
                    break

        if best_inliers is None or best_count < 20:
            return None

        inlier_pts = points[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        refined_normal = Vt[2, :]
        refined_d = -np.dot(refined_normal, centroid)
        return np.array([refined_normal[0], refined_normal[1], refined_normal[2], refined_d])

    # ------------------------------------------------------------------
    #  Least-squares plane fitting (fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_plane_lstsq(points):
        """Fit plane via SVD. Returns normalized normal (a,b,c) or None."""
        if points.shape[0] < 3:
            return None
        centroid = np.mean(points, axis=0)
        _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
        normal = Vt[2, :]
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            return None
        return normal / norm_len

    # ------------------------------------------------------------------
    #  Transform normal vector from camera frame to target frame
    # ------------------------------------------------------------------
    def _normal_to_target_frame(self, normal, camera_info, target_frame="base_link"):
        """Rotate normal from camera frame to target_frame via TF."""
        from geometry_msgs.msg import Vector3Stamped

        vec_msg = Vector3Stamped()
        vec_msg.header.frame_id = camera_info.header.frame_id or "cam_h"
        vec_msg.header.stamp = rospy.Time(0)
        vec_msg.vector.x = float(normal[0])
        vec_msg.vector.y = float(normal[1])
        vec_msg.vector.z = float(normal[2])

        try:
            transformed = self.tf_buffer.transform(
                vec_msg, target_frame, rospy.Duration(0.1)
            )
        except Exception as e:
            rospy.logwarn("Normal vector TF transform failed: %s", e)
            return None

        return np.array([
            transformed.vector.x,
            transformed.vector.y,
            transformed.vector.z,
        ])

    # ------------------------------------------------------------------
    #  Helper: quaternion from yaw (rotation about Z)
    # ------------------------------------------------------------------
    @staticmethod
    def _quaternion_from_yaw(yaw):
        """Generate quaternion (x, y, z, w) from yaw (rotation about Z)."""
        half_yaw = yaw / 2.0
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


    def _publish_debug_image(self, img):
        """将标注后的 BGR 图像以 JPEG 压缩后发布。"""
        success, jpeg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            rospy.logwarn_throttle(10, "Failed to encode debug image.")
            return

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "cam_h"
        msg.format = "jpeg"
        msg.data = jpeg_buf.tobytes()
        self.pub_debug.publish(msg)


# ---------------------------------------------------------------------------
#  Library entry  - 在后台线程中启动检测器
# ---------------------------------------------------------------------------
def start_detector(model_path, conf_threshold=0.5, daemon=True, **kwargs):
    """在后台线程中启动检测器，返回 (node, thread)。可在 challenge_task 内部调用。"""
    node = DetectObjectsNode(
        model_path=model_path,
        conf_threshold=conf_threshold,
        **kwargs
    )
    thread = threading.Thread(target=node.run, daemon=daemon)
    thread.start()
    rospy.loginfo("DetectObjectsNode started in bg thread (topic=%s)", node.publish_topic)
    return node, thread


# ---------------------------------------------------------------------------
#  CLI entry - 独立运行入口
# ---------------------------------------------------------------------------
def main():
    rospy.init_node("detect_objects_node", anonymous=False)

    model_path = rospy.get_param("~model_path")
    conf_threshold = rospy.get_param("~conf_threshold", 0.5)

    node, _thread = start_detector(model_path, conf_threshold)
    rospy.spin()


if __name__ == "__main__":
    main()
