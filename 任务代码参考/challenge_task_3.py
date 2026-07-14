#!/usr/bin/env python3
"""
挑战杯三场景统一任务入口。

推荐运行方式：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
"""

import argparse
import math
import os
import sys
import time


SCENE_CONFIGS = {
    "scene1": {
        "node_name": "challenge_task_scene1",
        "title": "场景一：包裹称重与摆放",
    },
    "scene2": {
        "node_name": "challenge_task_scene2",
        "title": "场景二：分拣归档",
    },
    "scene3": {
        "node_name": "challenge_task_scene3",
        "title": "场景三：SMT 料盘出库",
    },
}

ARM_JOINT_NAMES = ["arm_joint_{}".format(i) for i in range(1, 15)]
SCENE3_READY_ARM_POSE = [
    -20.0, 15.0, 0.0, -35.0, 0.0, -20.0, 10.0,
    -20.0, -15.0, 0.0, -35.0, 0.0, 20.0, -10.0,
]
SCENE3_HOME_ARM_POSE = [0.0] * 14
CLAW_OPEN_POS = [10.0, 10.0]
CLAW_CLOSED_POS = [90.0, 90.0]
ARM_MODE_AUTO_SWING = 1
ARM_MODE_EXTERNAL_CONTROL = 2
IK_MODE_POS_HARD_ORI_SOFT = 0x02
RIGHT_GRIPPER_QUAT_XYZW = [-0.081987, -0.152343, 0.857876, 0.483858]
SCENE3_GRASP_POINT_BASE_TOPIC = "/challenge_cup_task_template/scene3/grasp_point_base"
SCENE3_GRASP_POINT_ODOM_TOPIC = "/challenge_cup_task_template/scene3/grasp_point_odom"
SCENE3_APPROACH_DISTANCE = 0.65
SCENE3_APPROACH_DISTANCE_TOL = 0.05
SCENE3_APPROACH_KP = 0.60
SCENE3_APPROACH_TIMEOUT = 30.0
SCENE3_APPROACH_STABLE_CYCLES = 5
SCENE3_NEAR_RECOGNITION_SETTLE = 1.0
SCENE3_GRASP_WORKSPACE_PLANAR_MIN = 0.60
SCENE3_GRASP_WORKSPACE_X_MIN = -1.0
SCENE3_GRASP_WORKSPACE_X_MAX = 1.0
SCENE3_GRASP_WORKSPACE_Y_MIN = -1.0
SCENE3_GRASP_WORKSPACE_Y_MAX = 1.0
SCENE3_GRASP_WORKSPACE_Z_MIN = -1.0
SCENE3_GRASP_WORKSPACE_Z_MAX = 1.0
SCENE3_ALIGN_KP_X = 0.35
SCENE3_ALIGN_KP_Y = 0.80
SCENE3_ALIGN_MAX_VX = 0.15
SCENE3_ALIGN_MAX_VY = 0.12
SCENE3_ALIGN_LOOP_HZ = 10.0
SCENE3_ALIGN_TIMEOUT = 25.0
SCENE3_TARGET_FRESHNESS = 0.7
SCENE3_ALIGN_STABLE_CYCLES = 6
SCENE3_GRASP_PRE_OFFSET_X = -0.16
SCENE3_GRASP_TOUCH_OFFSET_X = -0.05
SCENE3_GRASP_TARGET_OFFSET_X = -0.01
SCENE3_GRASP_OFFSET_Y = 0.00
SCENE3_GRASP_OFFSET_Z = 0.02
SCENE3_GRASP_LIFT_DELTA_Z = 0.08
SCENE3_GRASP_RETREAT_X = -0.20
SCENE3_COARSE_ARM_WAIT_TIMEOUT = 4.0
SCENE3_COARSE_ARM_X = 0.30
SCENE3_COARSE_ARM_Y_MIN = -0.30
SCENE3_COARSE_ARM_Y_MAX = -0.10
SCENE3_COARSE_ARM_SECOND_SHELF_Z = -0.12


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def rad_to_deg(values):
    return [float(v) * 180.0 / math.pi for v in values]


def build_ik_param():
    from kuavo_msgs.msg import ikSolveParam

    param = ikSolveParam()
    param.major_optimality_tol = 1e-3
    param.major_feasibility_tol = 1e-3
    param.minor_feasibility_tol = 1e-3
    param.major_iterations_limit = 500
    param.oritation_constraint_tol = 1e-3
    param.pos_constraint_tol = 1e-3
    param.pos_cost_weight = 0.0
    param.constraint_mode = IK_MODE_POS_HARD_ORI_SOFT
    return param


class Scene3Task(object):
    """Scene 3 development scaffold for SMT tray outbound."""

    def __init__(self, cmd_vel_pub, arm_traj_pub,
                 vision_model_path="", vision_device="",
                 vision_confidence=0.35):
        import rospy
        import tf2_geometry_msgs  # noqa: F401
        import tf2_ros
        from kuavo_msgs.msg import lejuClawState, sensorsData, twoArmHandPoseCmd
        from kuavo_msgs.srv import changeArmCtrlMode
        from kuavo_msgs.srv import changeArmCtrlModeRequest
        from kuavo_msgs.srv import controlLejuClaw
        from kuavo_msgs.srv import fkSrv
        from kuavo_msgs.srv import twoArmHandPoseCmdSrv
        from geometry_msgs.msg import PointStamped
        from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2

        self.rospy = rospy
        self.changeArmCtrlMode = changeArmCtrlMode
        self.changeArmCtrlModeRequest = changeArmCtrlModeRequest
        self.controlLejuClaw = controlLejuClaw
        self.fkSrv = fkSrv
        self.PointStamped = PointStamped
        self.sensorsData = sensorsData
        self.twoArmHandPoseCmd = twoArmHandPoseCmd
        self.twoArmHandPoseCmdSrv = twoArmHandPoseCmdSrv
        self.cmd_vel_pub = cmd_vel_pub
        self.arm_traj_pub = arm_traj_pub

        self.last_sensor_msg = None
        self.last_claw_state = None
        self.last_head_rgb = None
        self.last_head_cam_info = None
        self.last_lidar = None
        self.last_grasp_point_base = None
        self.last_grasp_point_base_wall_time = 0.0
        self.last_grasp_point_odom = None
        self.last_grasp_point_odom_wall_time = 0.0
        self.integrated_vision = None
        self.approach_distance = SCENE3_APPROACH_DISTANCE
        self.approach_distance_tol = SCENE3_APPROACH_DISTANCE_TOL
        self.approach_kp = SCENE3_APPROACH_KP
        self.approach_timeout = SCENE3_APPROACH_TIMEOUT
        self.approach_stable_cycles = SCENE3_APPROACH_STABLE_CYCLES
        self.near_recognition_settle = SCENE3_NEAR_RECOGNITION_SETTLE
        self.workspace_planar_min = SCENE3_GRASP_WORKSPACE_PLANAR_MIN
        self.workspace_x_min = SCENE3_GRASP_WORKSPACE_X_MIN
        self.workspace_x_max = SCENE3_GRASP_WORKSPACE_X_MAX
        self.workspace_y_min = SCENE3_GRASP_WORKSPACE_Y_MIN
        self.workspace_y_max = SCENE3_GRASP_WORKSPACE_Y_MAX
        self.workspace_z_min = SCENE3_GRASP_WORKSPACE_Z_MIN
        self.workspace_z_max = SCENE3_GRASP_WORKSPACE_Z_MAX
        self.align_kp_x = SCENE3_ALIGN_KP_X
        self.align_kp_y = SCENE3_ALIGN_KP_Y
        self.align_max_vx = SCENE3_ALIGN_MAX_VX
        self.align_max_vy = SCENE3_ALIGN_MAX_VY
        self.align_loop_hz = SCENE3_ALIGN_LOOP_HZ
        self.align_timeout = SCENE3_ALIGN_TIMEOUT
        self.target_freshness = SCENE3_TARGET_FRESHNESS
        self.align_stable_cycles = SCENE3_ALIGN_STABLE_CYCLES

        self.layout_hint = {
            "rack_center_xy": (1.05, 0.0),
            "drop_box_xy": (-2.39, 0.0),
            "table_center_xy": (-2.55, 0.0),
            "lower_shelf_height": 0.4,
            "middle_shelf_height": 1.0,
        }
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber("/sensors_data_raw", sensorsData, self._sensor_cb, queue_size=1)
        rospy.Subscriber("/leju_claw_state", lejuClawState, self._claw_cb, queue_size=1)
        rospy.Subscriber(
            "/cam_h/color/image_raw/compressed",
            CompressedImage,
            self._head_rgb_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            "/cam_h/color/camera_info",
            CameraInfo,
            self._head_cam_info_cb,
            queue_size=1,
        )
        rospy.Subscriber("/lidar/points", PointCloud2, self._lidar_cb, queue_size=1)
        rospy.Subscriber(
            SCENE3_GRASP_POINT_BASE_TOPIC,
            PointStamped,
            self._grasp_point_base_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            SCENE3_GRASP_POINT_ODOM_TOPIC,
            PointStamped,
            self._grasp_point_odom_cb,
            queue_size=1,
        )

        self._claw_srv = None
        try:
            rospy.wait_for_service("/control_robot_leju_claw", timeout=5.0)
            self._claw_srv = rospy.ServiceProxy(
                "/control_robot_leju_claw",
                controlLejuClaw,
            )
        except Exception as exc:
            rospy.logwarn("scene3: claw service not ready yet: %s", exc)

        if vision_model_path:
            self._start_integrated_vision(
                model_path=vision_model_path,
                device=vision_device,
                confidence=vision_confidence,
            )

    def _sensor_cb(self, msg):
        self.last_sensor_msg = msg

    def _claw_cb(self, msg):
        self.last_claw_state = msg

    def _head_rgb_cb(self, msg):
        self.last_head_rgb = msg

    def _head_cam_info_cb(self, msg):
        self.last_head_cam_info = msg

    def _lidar_cb(self, msg):
        self.last_lidar = msg

    def _grasp_point_base_cb(self, msg):
        self.last_grasp_point_base = msg
        self.last_grasp_point_base_wall_time = time.time()

    def _grasp_point_odom_cb(self, msg):
        self.last_grasp_point_odom = msg
        self.last_grasp_point_odom_wall_time = time.time()

    def _start_integrated_vision(self, model_path, device="", confidence=0.35):
        try:
            from scene3_vision_debug import Scene3VisionDebugger
        except Exception as exc:
            self.rospy.logerr("scene3: failed to import scene3_vision_debug: %s", exc)
            return

        self.rospy.set_param("~model_path", model_path)
        self.rospy.set_param("~confidence_threshold", float(confidence))
        if device:
            self.rospy.set_param("~device", device)

        try:
            self.integrated_vision = Scene3VisionDebugger()
            self.rospy.loginfo(
                "scene3: integrated vision started, model=%s",
                model_path,
            )
        except Exception as exc:
            self.rospy.logerr("scene3: failed to start integrated vision: %s", exc)
            self.integrated_vision = None

    def wait_for_inputs(self, timeout=5.0):
        deadline = time.time() + timeout
        while not self.rospy.is_shutdown() and time.time() < deadline:
            if (
                self.last_sensor_msg is not None
                and self.last_head_rgb is not None
                and self.last_lidar is not None
            ):
                return True
            self.rospy.sleep(0.1)
        return False

    def get_recent_grasp_point_base(self, freshness=None):
        freshness = self.target_freshness if freshness is None else float(freshness)
        if self.last_grasp_point_base is None:
            return None
        if time.time() - self.last_grasp_point_base_wall_time > freshness:
            return None
        return self.last_grasp_point_base

    def get_recent_grasp_point_odom(self, freshness=None):
        freshness = self.target_freshness if freshness is None else float(freshness)
        if self.last_grasp_point_odom is None:
            return None
        if time.time() - self.last_grasp_point_odom_wall_time > freshness:
            return None
        return self.last_grasp_point_odom

    def transform_point_to_frame(self, point_msg, target_frame):
        try:
            return self.tf_buffer.transform(
                point_msg,
                target_frame,
                timeout=self.rospy.Duration(0.1),
            )
        except Exception as exc:
            self.rospy.logwarn_throttle(
                1.0,
                "scene3: failed to transform %s -> %s: %s",
                point_msg.header.frame_id,
                target_frame,
                exc,
            )
            return None

    def wait_for_arm_subscriber(self, timeout=8.0):
        start = time.time()
        while self.arm_traj_pub.get_num_connections() == 0:
            if self.rospy.is_shutdown():
                return False
            if time.time() - start > timeout:
                raise RuntimeError("/kuavo_arm_traj has no subscriber")
            self.rospy.sleep(0.1)
        return True

    def set_arm_mode(self, mode):
        service_names = ["/arm_traj_change_mode", "/humanoid_change_arm_ctrl_mode"]
        last_error = None
        for service_name in service_names:
            try:
                self.rospy.wait_for_service(service_name, timeout=2.0)
                proxy = self.rospy.ServiceProxy(service_name, self.changeArmCtrlMode)
                req = self.changeArmCtrlModeRequest()
                req.control_mode = int(mode)
                res = proxy(req)
                if getattr(res, "result", False):
                    self.rospy.loginfo("scene3: arm mode %s via %s", mode, service_name)
                    return True
                last_error = getattr(res, "message", "")
            except Exception as exc:
                last_error = exc
        self.rospy.logwarn("scene3: failed to set arm mode %s: %s", mode, last_error)
        return False

    def read_current_arm_joints(self, timeout=5.0):
        msg = self.rospy.wait_for_message("/sensors_data_raw", self.sensorsData, timeout=timeout)
        joint_q = list(msg.joint_data.joint_q)
        if len(joint_q) >= 27:
            return joint_q[13:27]
        if len(joint_q) >= 26:
            return joint_q[12:26]
        raise RuntimeError("scene3: joint_q length is too short: {}".format(len(joint_q)))

    def call_fk(self, arm_joints_rad):
        self.rospy.wait_for_service("/ik/fk_srv", timeout=5.0)
        proxy = self.rospy.ServiceProxy("/ik/fk_srv", self.fkSrv)
        res = proxy(arm_joints_rad)
        if not getattr(res, "success", False):
            raise RuntimeError("scene3: /ik/fk_srv failed")
        return res.hand_poses

    def solve_right_hand_ik(self, right_pos_xyz, right_quat_xyzw):
        current_joints = self.read_current_arm_joints()
        current_poses = self.call_fk(current_joints)

        req = self.twoArmHandPoseCmd()
        req.frame = 2
        req.use_custom_ik_param = True
        req.joint_angles_as_q0 = True
        req.ik_param = build_ik_param()

        req.hand_poses.left_pose.pos_xyz = list(current_poses.left_pose.pos_xyz)
        req.hand_poses.left_pose.quat_xyzw = list(current_poses.left_pose.quat_xyzw)
        req.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        req.hand_poses.left_pose.joint_angles = list(current_joints[:7])

        req.hand_poses.right_pose.pos_xyz = list(map(float, right_pos_xyz))
        req.hand_poses.right_pose.quat_xyzw = list(map(float, right_quat_xyzw))
        req.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        req.hand_poses.right_pose.joint_angles = list(current_joints[7:])

        self.rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=5.0)
        proxy = self.rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", self.twoArmHandPoseCmdSrv)
        res = proxy(req)
        if not getattr(res, "success", False):
            raise RuntimeError(
                "scene3: IK failed: {} target={}".format(
                    getattr(res, "error_reason", ""),
                    [round(float(v), 4) for v in right_pos_xyz],
                )
            )

        if len(res.q_arm) >= 14:
            return list(res.q_arm[:14])

        right_result = list(res.hand_poses.right_pose.joint_angles)
        if len(right_result) == 7:
            return list(current_joints[:7]) + right_result
        raise RuntimeError("scene3: IK response does not contain 14 arm joints")

    def publish_arm_pose(self, joint_positions, settle_sec=1.5):
        from sensor_msgs.msg import JointState

        if len(joint_positions) != 14:
            raise ValueError("scene3 arm pose must have 14 joints")

        msg = JointState()
        msg.header.stamp = self.rospy.Time.now()
        msg.name = ARM_JOINT_NAMES
        msg.position = joint_positions
        self.arm_traj_pub.publish(msg)
        self.rospy.sleep(settle_sec)

    def publish_arm_degrees_once(self, joint_positions):
        from sensor_msgs.msg import JointState

        msg = JointState()
        msg.header.stamp = self.rospy.Time.now()
        msg.name = ARM_JOINT_NAMES
        msg.position = [float(v) for v in joint_positions]
        self.arm_traj_pub.publish(msg)

    def hold_arm_degrees(self, joint_positions, hold_time=0.6, hz=50):
        rate = self.rospy.Rate(hz)
        end_time = time.time() + float(hold_time)
        while not self.rospy.is_shutdown() and time.time() < end_time:
            self.publish_arm_degrees_once(joint_positions)
            rate.sleep()

    def move_arm_degrees(self, target_degrees, duration=2.0, hz=50):
        start_degrees = rad_to_deg(self.read_current_arm_joints())
        target_degrees = [float(v) for v in target_degrees]
        steps = max(1, int(float(duration) * hz))
        rate = self.rospy.Rate(hz)
        for step in range(steps + 1):
            if self.rospy.is_shutdown():
                return
            alpha = float(step) / float(steps)
            point = [
                start_degrees[i] + (target_degrees[i] - start_degrees[i]) * alpha
                for i in range(14)
            ]
            self.publish_arm_degrees_once(point)
            rate.sleep()
        self.hold_arm_degrees(target_degrees, hold_time=0.3, hz=hz)

    def move_right_hand(self, pos_xyz, duration=2.0):
        self.rospy.loginfo("scene3: right hand target %s", [round(float(v), 4) for v in pos_xyz])
        joints_rad = self.solve_right_hand_ik(pos_xyz, RIGHT_GRIPPER_QUAT_XYZW)
        self.move_arm_degrees(rad_to_deg(joints_rad), duration=duration)

    def set_base_velocity(self, vx=0.0, vy=0.0, wz=0.0, duration=0.0, hz=20.0):
        from geometry_msgs.msg import Twist

        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = wz

        if duration <= 0.0:
            self.cmd_vel_pub.publish(twist)
            return

        rate = self.rospy.Rate(hz)
        end_time = time.time() + duration
        while not self.rospy.is_shutdown() and time.time() < end_time:
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        self.stop_base()

    def stop_base(self):
        self.set_base_velocity(0.0, 0.0, 0.0, duration=0.0)

    def command_claw(self, positions, velocity=50.0, effort=1.0):
        from kuavo_msgs.srv import controlLejuClawRequest

        if self._claw_srv is None:
            self.rospy.logwarn("scene3: claw service unavailable, skip claw command")
            return False

        req = controlLejuClawRequest()
        req.data.name = ["left_claw", "right_claw"]
        req.data.position = list(positions)
        req.data.velocity = [velocity, velocity]
        req.data.effort = [effort, effort]

        try:
            res = self._claw_srv(req)
        except Exception as exc:
            self.rospy.logerr("scene3: claw service call failed: %s", exc)
            return False

        if not getattr(res, "success", False):
            self.rospy.logwarn(
                "scene3: claw command rejected: %s",
                getattr(res, "message", ""),
            )
            return False
        return True

    def open_claw(self):
        ok = self.command_claw(CLAW_OPEN_POS)
        self.rospy.sleep(1.0)
        return ok

    def close_claw(self):
        ok = self.command_claw(CLAW_CLOSED_POS)
        self.rospy.sleep(1.0)
        return ok

    def log_layout_hint(self):
        self.rospy.loginfo(
            "scene3 layout hint: rack=%s drop_box=%s table=%s lower_h=%.2f middle_h=%.2f",
            self.layout_hint["rack_center_xy"],
            self.layout_hint["drop_box_xy"],
            self.layout_hint["table_center_xy"],
            self.layout_hint["lower_shelf_height"],
            self.layout_hint["middle_shelf_height"],
        )

    def wait_for_recent_base_target(self, timeout=4.0, freshness=None):
        freshness = self.target_freshness if freshness is None else float(freshness)
        deadline = time.time() + float(timeout)
        while not self.rospy.is_shutdown() and time.time() < deadline:
            target_msg = self.get_recent_grasp_point_base(freshness=freshness)
            if target_msg is not None:
                return target_msg
            self.rospy.sleep(0.05)
        return None

    def build_coarse_recognition_arm_target(self, target_msg):
        point = target_msg.point
        return [
            SCENE3_COARSE_ARM_X,
            clamp(float(point.y), SCENE3_COARSE_ARM_Y_MIN, SCENE3_COARSE_ARM_Y_MAX),
            SCENE3_COARSE_ARM_SECOND_SHELF_Z,
        ]

    def raise_arm_for_coarse_target(self, timeout=None):
        timeout = SCENE3_COARSE_ARM_WAIT_TIMEOUT if timeout is None else float(timeout)
        target_msg = self.wait_for_recent_base_target(
            timeout=timeout,
            freshness=max(self.target_freshness, 1.0),
        )
        if target_msg is None:
            self.rospy.logwarn(
                "scene3: no fresh base target for coarse arm raise within %.1fs",
                timeout,
            )
            return False

        target_point = target_msg.point
        arm_target = self.build_coarse_recognition_arm_target(target_msg)
        self.rospy.loginfo(
            "scene3: coarse target base=(%.3f, %.3f, %.3f) -> arm standby=%s",
            target_point.x,
            target_point.y,
            target_point.z,
            [round(v, 3) for v in arm_target],
        )
        try:
            self.move_right_hand(arm_target, duration=1.8)
            return True
        except Exception as exc:
            self.rospy.logwarn("scene3: coarse arm raise failed, keep ready pose: %s", exc)
            self.publish_arm_pose(SCENE3_READY_ARM_POSE, settle_sec=1.0)
            return False

    def approach_to_secondary_recognition(self, timeout=None):
        timeout = self.approach_timeout if timeout is None else float(timeout)
        self.rospy.loginfo(
            "scene3: start coarse approach, standoff=%.2fm tol=%.2fm",
            self.approach_distance,
            self.approach_distance_tol,
        )

        deadline = time.time() + timeout
        stable_cycles = 0
        last_log_time = 0.0
        rate = self.rospy.Rate(self.align_loop_hz)

        while not self.rospy.is_shutdown() and time.time() < deadline:
            odom_target = self.get_recent_grasp_point_odom()
            if odom_target is None:
                stable_cycles = 0
                self.stop_base()
                self.rospy.logwarn_throttle(
                    1.0,
                    "scene3: waiting for fresh coarse target on %s",
                    SCENE3_GRASP_POINT_ODOM_TOPIC,
                )
                rate.sleep()
                continue

            local_target = self.transform_point_to_frame(odom_target, "base_link")
            if local_target is None:
                stable_cycles = 0
                self.stop_base()
                rate.sleep()
                continue

            point = local_target.point
            planar_dist = math.hypot(float(point.x), float(point.y))
            dist_error = planar_dist - self.approach_distance

            if 0.0 <= dist_error <= self.approach_distance_tol:
                stable_cycles += 1
                self.stop_base()
                if stable_cycles >= self.approach_stable_cycles:
                    self.rospy.loginfo(
                        "scene3: reached secondary-recognition standoff at base target=(%.3f, %.3f, %.3f), dist=%.3f",
                        point.x,
                        point.y,
                        point.z,
                        planar_dist,
                    )
                    return True
            else:
                stable_cycles = 0
                if planar_dist < 1e-4:
                    self.stop_base()
                    rate.sleep()
                    continue
                ux = float(point.x) / planar_dist
                uy = float(point.y) / planar_dist
                speed = clamp(
                    self.approach_kp * dist_error,
                    -self.align_max_vx,
                    self.align_max_vx,
                )
                cmd_vx = clamp(speed * ux, -self.align_max_vx, self.align_max_vx)
                cmd_vy = clamp(speed * uy, -self.align_max_vy, self.align_max_vy)
                self.set_base_velocity(vx=cmd_vx, vy=cmd_vy, wz=0.0, duration=0.0)

            now = time.time()
            if now - last_log_time > 1.0:
                self.rospy.loginfo(
                    "scene3: coarse approach local_target=(%.3f, %.3f, %.3f) planar_dist=%.3f dist_error=%.3f stable=%d/%d",
                    point.x,
                    point.y,
                    point.z,
                    planar_dist,
                    dist_error,
                    stable_cycles,
                    self.approach_stable_cycles,
                )
                last_log_time = now
            rate.sleep()

        self.stop_base()
        self.rospy.logwarn("scene3: coarse approach timeout after %.1fs", timeout)
        return False

    def point_in_grasp_workspace(self, point):
        planar_dist = math.hypot(float(point.x), float(point.y))
        return (
            planar_dist >= self.workspace_planar_min
            and
            self.workspace_x_min <= float(point.x) <= self.workspace_x_max
            and self.workspace_y_min <= float(point.y) <= self.workspace_y_max
            and self.workspace_z_min <= float(point.z) <= self.workspace_z_max
        )

    def workspace_error(self, value, lower, upper):
        value = float(value)
        if value < lower:
            return value - float(lower)
        if value > upper:
            return value - float(upper)
        return 0.0

    def build_scene3_grasp_targets(self, target_msg):
        point = target_msg.point
        grasp = [
            float(point.x) + SCENE3_GRASP_TARGET_OFFSET_X,
            float(point.y) + SCENE3_GRASP_OFFSET_Y,
            float(point.z) + SCENE3_GRASP_OFFSET_Z,
        ]
        touch = [
            float(point.x) + SCENE3_GRASP_TOUCH_OFFSET_X,
            float(point.y) + SCENE3_GRASP_OFFSET_Y,
            float(point.z) + SCENE3_GRASP_OFFSET_Z,
        ]
        pregrasp = [
            float(point.x) + SCENE3_GRASP_PRE_OFFSET_X,
            float(point.y) + SCENE3_GRASP_OFFSET_Y,
            float(point.z) + SCENE3_GRASP_OFFSET_Z,
        ]
        lift = [
            float(point.x) + SCENE3_GRASP_PRE_OFFSET_X,
            float(point.y) + SCENE3_GRASP_OFFSET_Y,
            float(point.z) + SCENE3_GRASP_OFFSET_Z + SCENE3_GRASP_LIFT_DELTA_Z,
        ]
        retreat = [
            float(point.x) + SCENE3_GRASP_RETREAT_X,
            float(point.y) + SCENE3_GRASP_OFFSET_Y,
            float(point.z) + SCENE3_GRASP_OFFSET_Z + SCENE3_GRASP_LIFT_DELTA_Z,
        ]
        return {
            "pregrasp": pregrasp,
            "touch": touch,
            "grasp": grasp,
            "lift": lift,
            "retreat": retreat,
        }

    def grasp_tray_from_latest_target(self):
        target_msg = self.get_recent_grasp_point_base(freshness=1.0)
        if target_msg is None:
            self.rospy.logwarn("scene3: no fresh grasp point available for grasp")
            return False

        targets = self.build_scene3_grasp_targets(target_msg)
        self.rospy.loginfo(
            "scene3: grasp plan pre=%s touch=%s grasp=%s",
            [round(v, 3) for v in targets["pregrasp"]],
            [round(v, 3) for v in targets["touch"]],
            [round(v, 3) for v in targets["grasp"]],
        )

        self.stop_base()
        self.open_claw()
        self.move_right_hand(targets["pregrasp"], duration=2.0)
        self.move_right_hand(targets["touch"], duration=1.2)
        self.move_right_hand(targets["grasp"], duration=1.0)
        self.rospy.sleep(0.2)
        self.close_claw()
        self.move_right_hand(targets["lift"], duration=1.4)
        self.move_right_hand(targets["retreat"], duration=1.2)
        self.rospy.loginfo("scene3: grasp sequence finished")
        return True

    def align_base_to_tray(self, timeout=None):
        timeout = self.align_timeout if timeout is None else float(timeout)
        self.rospy.loginfo(
            "scene3: start workspace alignment planar>=%.3f x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f]",
            self.workspace_planar_min,
            self.workspace_x_min,
            self.workspace_x_max,
            self.workspace_y_min,
            self.workspace_y_max,
            self.workspace_z_min,
            self.workspace_z_max,
        )

        deadline = time.time() + timeout
        stable_cycles = 0
        last_log_time = 0.0
        rate = self.rospy.Rate(self.align_loop_hz)

        while not self.rospy.is_shutdown() and time.time() < deadline:
            target_msg = self.get_recent_grasp_point_base()
            if target_msg is None:
                stable_cycles = 0
                self.stop_base()
                self.rospy.logwarn_throttle(
                    1.0,
                    "scene3: waiting for fresh grasp point on %s",
                    SCENE3_GRASP_POINT_BASE_TOPIC,
                )
                rate.sleep()
                continue

            point = target_msg.point
            planar_dist = math.hypot(float(point.x), float(point.y))
            error_x = self.workspace_error(point.x, self.workspace_x_min, self.workspace_x_max)
            error_y = self.workspace_error(point.y, self.workspace_y_min, self.workspace_y_max)
            planar_error = min(0.0, planar_dist - self.workspace_planar_min)
            in_workspace = self.point_in_grasp_workspace(point)

            if in_workspace:
                stable_cycles += 1
                self.stop_base()
                if stable_cycles >= self.align_stable_cycles:
                    self.rospy.loginfo(
                        "scene3: target entered grasp workspace at (%.3f, %.3f, %.3f)",
                        point.x,
                        point.y,
                        point.z,
                    )
                    return True
            else:
                stable_cycles = 0
                cmd_vx = clamp(self.align_kp_x * error_x, -self.align_max_vx, self.align_max_vx)
                cmd_vy = clamp(self.align_kp_y * error_y, -self.align_max_vy, self.align_max_vy)
                if planar_error < 0.0 and planar_dist > 1e-4:
                    retreat_speed = clamp(
                        self.approach_kp * planar_error,
                        -self.align_max_vx,
                        self.align_max_vx,
                    )
                    cmd_vx += clamp(retreat_speed * float(point.x) / planar_dist, -self.align_max_vx, self.align_max_vx)
                    cmd_vy += clamp(retreat_speed * float(point.y) / planar_dist, -self.align_max_vy, self.align_max_vy)
                    cmd_vx = clamp(cmd_vx, -self.align_max_vx, self.align_max_vx)
                    cmd_vy = clamp(cmd_vy, -self.align_max_vy, self.align_max_vy)

                x_span = max(1e-6, self.workspace_x_max - self.workspace_x_min)
                y_span = max(1e-6, self.workspace_y_max - self.workspace_y_min)
                if abs(error_x) < x_span:
                    cmd_vx *= 0.5
                if abs(error_y) < y_span:
                    cmd_vy *= 0.5

                self.set_base_velocity(vx=cmd_vx, vy=cmd_vy, wz=0.0, duration=0.0)

            now = time.time()
            if now - last_log_time > 1.0:
                self.rospy.loginfo(
                    "scene3: track tray base=(%.3f, %.3f, %.3f) planar_dist=%.3f workspace_err=(%.3f, %.3f, %.3f) in_workspace=%s stable=%d/%d",
                    point.x,
                    point.y,
                    point.z,
                    planar_dist,
                    error_x,
                    error_y,
                    planar_error,
                    in_workspace,
                    stable_cycles,
                    self.align_stable_cycles,
                )
                last_log_time = now
            rate.sleep()

        self.stop_base()
        self.rospy.logwarn("scene3: alignment timeout after %.1fs", timeout)
        return False

    def run(self):
        self.rospy.loginfo("scene3: start SMT tray outbound scaffold")
        self.log_layout_hint()

        if not self.wait_for_inputs(timeout=8.0):
            self.rospy.logwarn("scene3: some inputs are still missing, continue with cached data only")

        # Start from a conservative pose so later perception/alignment code has a stable baseline.
        self.wait_for_arm_subscriber()
        self.set_arm_mode(ARM_MODE_EXTERNAL_CONTROL)
        self.open_claw()
        self.publish_arm_pose(SCENE3_HOME_ARM_POSE, settle_sec=1.0)
        self.publish_arm_pose(SCENE3_READY_ARM_POSE, settle_sec=2.0)
        self.raise_arm_for_coarse_target()
        self.stop_base()

        coarse_ready = self.approach_to_secondary_recognition()
        if coarse_ready:
            self.rospy.loginfo("scene3: coarse approach finished, wait for near recognition")
            self.stop_base()
            self.rospy.sleep(self.near_recognition_settle)
        else:
            self.rospy.logwarn("scene3: coarse approach did not converge; keep current pose")

        aligned = False
        if coarse_ready:
            aligned = self.align_base_to_tray()
            if aligned:
                self.rospy.loginfo("scene3: near target entered grasp workspace")
            else:
                self.rospy.logwarn("scene3: near alignment did not converge; keep current pose")

        grasped = False
        if aligned:
            try:
                grasped = self.grasp_tray_from_latest_target()
            except Exception as exc:
                self.rospy.logerr("scene3: grasp failed: %s", exc)
                grasped = False

        self.rospy.loginfo("scene3 scaffold ready")
        if grasped:
            self.rospy.loginfo("scene3 next step 1: move grasped tray to drop box")
        else:
            self.rospy.logwarn("scene3 next step 1: tune grasp offsets or retry alignment")
        self.rospy.loginfo("scene3 next step 2: add tray transport trajectory")
        self.rospy.loginfo("scene3 next step 3: place tray into drop box")


def _load_launcher():
    # 公共启动器位于受保护包 challenge_cup_simulator/utils/（选手不可改动），
    # 从那里导入，确保完整性校验无法被绕过。
    try:
        import rospkg
        sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
    except Exception:
        sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "challenge_cup_simulator", "utils")
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher
    return ChallengeSimLauncher


def run_scene(scene, seed, node_name=None, timeout=120,
              time_limit=None, timer_gui=True,
              scene3_model_path="", scene3_vision_device="",
              scene3_confidence_threshold=0.35):
    if scene not in SCENE_CONFIGS:
        raise ValueError("unknown scene: {}".format(scene))

    config = SCENE_CONFIGS[scene]
    ChallengeSimLauncher = _load_launcher()

    launcher = ChallengeSimLauncher(
        scene=scene,
        seed=seed,
        match_time_limit=time_limit,
        timer_gui=timer_gui,
    )
    launcher.start(node_name=node_name or config["node_name"], timeout=timeout)

    import rospy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import JointState

    rospy.loginfo("=== %s任务启动 ===", config["title"])

    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)

    rospy.sleep(1.0)
    rospy.loginfo("场景实例已初始化。")

    # ========================================
    # TODO: 在此实现三场景共用或按 scene 分支的任务逻辑
    # ========================================
    #
    # if scene == "scene1":
    #     pass  # 包裹称重与摆放
    # elif scene == "scene2":
    #     pass  # 分拣归档
    # elif scene == "scene3":
    #     pass  # SMT 料盘出库
    #
    # 可用接口：
    #   /cmd_vel                  geometry_msgs/Twist
    #   /kuavo_arm_traj           sensor_msgs/JointState
    #   /lidar/points             sensor_msgs/PointCloud2
    #   /sensors_data_raw         kuavo_msgs/sensorsData
    #   /control_robot_leju_claw  kuavo_msgs/controlLejuClaw
    #   /leju_claw_command        kuavo_msgs/lejuClawCommand
    #   /leju_claw_state          kuavo_msgs/lejuClawState
    #
    # 示例：
    # twist = Twist()
    # twist.linear.x = 0.1
    # cmd_vel_pub.publish(twist)

    if scene == "scene3":
        scene3_task = Scene3Task(
            cmd_vel_pub,
            arm_traj_pub,
            vision_model_path=scene3_model_path,
            vision_device=scene3_vision_device,
            vision_confidence=scene3_confidence_threshold,
        )
        scene3_task.run()

    rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="挑战杯三场景统一任务入口")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="scene1",
                        help="要启动的比赛场景")
    parser.add_argument("--seed", type=int, default=0,
                        help="场景种子；正式评测 seed 由组委会指定")
    parser.add_argument("--node-name", default=None,
                        help="ROS 节点名；默认按 scene 自动设置")
    parser.add_argument("--timeout", type=int, default=120,
                        help="等待仿真就绪的超时时间，单位秒")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="比赛时长，单位秒；默认读取 CHALLENGE_TIME_LIMIT，未设置则不限时")
    parser.add_argument("--no-timer-gui", action="store_true",
                        help="不弹出计时器窗口，仅保留后台计时日志")
    parser.add_argument("--scene3-model-path", default="",
                        help="场景三 YOLO 模型绝对路径；设置后在任务节点内直接启动视觉")
    parser.add_argument("--scene3-vision-device", default="",
                        help="场景三 YOLO 推理设备，例如 cuda:0 或 cpu")
    parser.add_argument("--scene3-confidence-threshold", type=float, default=0.35,
                        help="场景三 YOLO 置信度阈值")
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
        scene3_model_path=args.scene3_model_path,
        scene3_vision_device=args.scene3_vision_device,
        scene3_confidence_threshold=args.scene3_confidence_threshold,
    )


if __name__ == "__main__":
    main()
