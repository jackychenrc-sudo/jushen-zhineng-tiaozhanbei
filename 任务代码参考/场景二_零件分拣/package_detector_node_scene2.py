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
