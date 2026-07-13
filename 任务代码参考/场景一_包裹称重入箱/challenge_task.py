#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Challenge Cup task entry.

Scene 1 strategy:
1. switch arm to external control;
2. move right claw to each parcel;
3. close claw and lift;
4. pause on the weighing area;
5. move to the green box and release.

The code is intentionally written as small "basic actions", following the style
of the teacher's pytrees_actions package, but it does not depend on py_trees or
kuavo_humanoid_sdk. It only uses the ROS topics/services available in the
official challenge environment.
"""

import argparse
import math
import os
import sys
import time


SCENE_CONFIGS = {
    "scene1": {
        "node_name": "challenge_task_scene1",
        "title": "scene1 parcel weighing and placing",
    },
    "scene2": {
        "node_name": "challenge_task_scene2",
        "title": "scene2 placeholder",
    },
    "scene3": {
        "node_name": "challenge_task_scene3",
        "title": "scene3 placeholder",
    },
}


ARM_JOINT_NAMES = [
    "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
    "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
    "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
    "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
]


ARM_MODE_AUTO_SWING = 1
ARM_MODE_EXTERNAL_CONTROL = 2
IK_MODE_POS_HARD_ORI_SOFT = 0x02


# A known reachable right-claw orientation from the official collection script.
# It keeps the claw suitable for side grasping instead of pointing straight down.
RIGHT_GRIPPER_QUAT_XYZW = [-0.081987, -0.152343, 0.857876, 0.483858]


# Safe arm pose, in degrees. Left arm is relaxed; right arm is lifted near table.
SAFE_PREGRASP_DEG = [
    0.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
    45.0, -15.0, -5.0, -105.0, 60.0, 0.0, 0.0,
]


# Robot-local IK targets. These are the only values you normally tune.
# If the claw misses a parcel, run arm_keyboard_control.py, record its printed
# pos=[x,y,z], and replace the corresponding point below.
PICK_POINTS = [
    [0.42, -0.34, -0.22],
    [0.42, -0.12, -0.22],
    [0.55, -0.34, -0.22],
    [0.55, -0.12, -0.22],
]

WEIGH_POINT = [0.34, -0.39, -0.21]

DROP_POINTS = [
    [0.46, 0.15, -0.17],
    [0.46, 0.24, -0.17],
    [0.54, 0.15, -0.17],
    [0.54, 0.24, -0.17],
]

# During debugging, never descend directly to the table. First verify that the
# claw hovers above the parcel. Only then run with --execute-grasp.
SAFE_TRAVEL_Z = 0.55
SAFE_APPROACH_Z = 0.42
PICK_DESCEND_Z = 0.16
WEIGH_DESCEND_Z = 0.16
DROP_DESCEND_Z = 0.20

# YOLO/vision output is expected to be in robot base/local coordinates.
# Vision now publishes the parcel tape-cross center on the top face. Keep the
# target slightly above that surface so the gripper does not start inside the box.
VISUAL_PICK_OFFSET = [0.0, 0.0, 0.03]
VISUAL_TOPIC_CANDIDATES = [
    "/robot_yolov8_info",
    "/object_yolo_box_tf2_torso_result",
]
POSE_ARRAY_TOPIC = "/scene1/parcel_points"


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


def above(point, dz):
    return [float(point[0]), float(point[1]), float(point[2]) + float(dz)]


def travel_above(point):
    return above(point, SAFE_TRAVEL_Z)


class BasicActions:
    """Small reusable actions: arm mode, joint motion, IK motion, and claw."""

    def __init__(self, rospy, arm_pub):
        self.rospy = rospy
        self.arm_pub = arm_pub

        from kuavo_msgs.srv import changeArmCtrlMode
        from kuavo_msgs.srv import changeArmCtrlModeRequest
        from kuavo_msgs.srv import controlLejuClaw
        from kuavo_msgs.srv import controlLejuClawRequest
        from kuavo_msgs.srv import fkSrv
        from kuavo_msgs.srv import twoArmHandPoseCmdSrv
        from kuavo_msgs.msg import sensorsData
        from kuavo_msgs.msg import twoArmHandPoseCmd

        self.changeArmCtrlMode = changeArmCtrlMode
        self.changeArmCtrlModeRequest = changeArmCtrlModeRequest
        self.controlLejuClaw = controlLejuClaw
        self.controlLejuClawRequest = controlLejuClawRequest
        self.fkSrv = fkSrv
        self.twoArmHandPoseCmdSrv = twoArmHandPoseCmdSrv
        self.sensorsData = sensorsData
        self.twoArmHandPoseCmd = twoArmHandPoseCmd

    def wait_for_arm_subscriber(self, timeout=8.0):
        start = time.time()
        while self.arm_pub.get_num_connections() == 0:
            if self.rospy.is_shutdown():
                return
            if time.time() - start > timeout:
                raise RuntimeError("/kuavo_arm_traj has no subscriber")
            self.rospy.sleep(0.1)

    def set_arm_mode(self, mode):
        service_names = ["/arm_traj_change_mode", "/humanoid_change_arm_ctrl_mode"]
        last_error = None

        for service_name in service_names:
            try:
                self.rospy.wait_for_service(service_name, timeout=2.0)
                proxy = self.rospy.ServiceProxy(service_name, self.changeArmCtrlMode)
                request = self.changeArmCtrlModeRequest()
                request.control_mode = int(mode)
                response = proxy(request)
                if response.result:
                    self.rospy.loginfo("arm mode %s via %s", mode, service_name)
                    return True
                last_error = response.message
            except Exception as exc:
                last_error = exc

        self.rospy.logwarn("failed to set arm mode %s: %s", mode, last_error)
        return False

    def read_current_arm_joints(self, timeout=5.0):
        msg = self.rospy.wait_for_message(
            "/sensors_data_raw", self.sensorsData, timeout=timeout
        )
        joint_q = list(msg.joint_data.joint_q)
        if len(joint_q) >= 27:
            return joint_q[13:27]
        if len(joint_q) >= 26:
            return joint_q[12:26]
        raise RuntimeError("joint_q length is too short: {}".format(len(joint_q)))

    def call_fk(self, arm_joints_rad):
        self.rospy.wait_for_service("/ik/fk_srv", timeout=5.0)
        proxy = self.rospy.ServiceProxy("/ik/fk_srv", self.fkSrv)
        response = proxy(arm_joints_rad)
        if not response.success:
            raise RuntimeError("/ik/fk_srv failed")
        return response.hand_poses

    def solve_right_hand_ik(self, right_pos_xyz, right_quat_xyzw):
        current_joints = self.read_current_arm_joints()
        current_poses = self.call_fk(current_joints)

        request = self.twoArmHandPoseCmd()
        request.frame = 2
        request.use_custom_ik_param = True
        request.joint_angles_as_q0 = True
        request.ik_param = build_ik_param()

        request.hand_poses.left_pose.pos_xyz = list(current_poses.left_pose.pos_xyz)
        request.hand_poses.left_pose.quat_xyzw = list(current_poses.left_pose.quat_xyzw)
        request.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        request.hand_poses.left_pose.joint_angles = list(current_joints[:7])

        request.hand_poses.right_pose.pos_xyz = list(map(float, right_pos_xyz))
        request.hand_poses.right_pose.quat_xyzw = list(map(float, right_quat_xyzw))
        request.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        request.hand_poses.right_pose.joint_angles = list(current_joints[7:])

        self.rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=5.0)
        proxy = self.rospy.ServiceProxy(
            "/ik/two_arm_hand_pose_cmd_srv", self.twoArmHandPoseCmdSrv
        )
        response = proxy(request)
        if not response.success:
            raise RuntimeError(
                "IK failed: {} target={}".format(response.error_reason, right_pos_xyz)
            )

        if len(response.q_arm) >= 14:
            return list(response.q_arm[:14])

        right_result = list(response.hand_poses.right_pose.joint_angles)
        if len(right_result) == 7:
            return list(current_joints[:7]) + right_result

        raise RuntimeError("IK response does not contain 14 arm joints")

    def publish_arm_degrees_once(self, degrees):
        from sensor_msgs.msg import JointState

        msg = JointState()
        msg.header.stamp = self.rospy.Time.now()
        msg.name = ARM_JOINT_NAMES
        msg.position = [float(v) for v in degrees]
        self.arm_pub.publish(msg)

    def hold_arm_degrees(self, degrees, hold_time=0.8, hz=50):
        rate = self.rospy.Rate(hz)
        end = time.time() + float(hold_time)
        while time.time() < end and not self.rospy.is_shutdown():
            self.publish_arm_degrees_once(degrees)
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
        self.rospy.loginfo("right hand target: %s", [round(float(v), 4) for v in pos_xyz])
        joints_rad = self.solve_right_hand_ik(pos_xyz, RIGHT_GRIPPER_QUAT_XYZW)
        self.move_arm_degrees(rad_to_deg(joints_rad), duration=duration)

    def set_right_claw(self, percent, wait=0.8):
        self.rospy.wait_for_service("/control_robot_leju_claw", timeout=5.0)
        proxy = self.rospy.ServiceProxy("/control_robot_leju_claw", self.controlLejuClaw)

        request = self.controlLejuClawRequest()
        request.data.name = ["left_claw", "right_claw"]
        request.data.position = [0.0, float(percent)]
        request.data.velocity = [80.0, 80.0]
        request.data.effort = [1.0, 1.0]

        response = proxy(request)
        if not response.success:
            raise RuntimeError("claw failed: {}".format(response.message))
        self.rospy.sleep(wait)

    def open_right_claw(self):
        self.set_right_claw(0.0)

    def close_right_claw(self):
        self.set_right_claw(90.0)

    def look_down(self):
        """Lower the head camera a little so vision has a better table view."""
        try:
            from kuavo_msgs.msg import robotHeadMotionData
        except Exception as exc:
            self.rospy.logwarn("head message unavailable: %s", exc)
            return

        pub = self.rospy.Publisher("/robot_head_motion_data", robotHeadMotionData, queue_size=10)
        self.rospy.sleep(0.2)
        msg = robotHeadMotionData()
        msg.joint_data = [0.0, 20.0]
        for _ in range(8):
            if self.rospy.is_shutdown():
                return
            pub.publish(msg)
            self.rospy.sleep(0.1)


class VisionParcelLocator:
    """Read parcel positions from an existing YOLO/vision topic if available."""

    def __init__(self, rospy):
        self.rospy = rospy

    def detect_pick_points(self, max_count=4, timeout=3.0):
        pose_points = self._detect_pose_array(max_count=max_count, timeout=timeout)
        if pose_points:
            return pose_points

        try:
            from vision_msgs.msg import Detection2DArray
        except Exception as exc:
            self.rospy.logwarn("vision_msgs unavailable, skip vision: %s", exc)
            return []

        detections = []
        for topic in VISUAL_TOPIC_CANDIDATES:
            try:
                msg = self.rospy.wait_for_message(topic, Detection2DArray, timeout=timeout)
            except Exception:
                continue
            detections = self._extract_points(msg)
            if detections:
                self.rospy.loginfo("vision detected %d objects from %s", len(detections), topic)
                break

        if not detections:
            self.rospy.logwarn("no visual parcel detections; fallback to fixed PICK_POINTS")
            return []

        points = self._sort_table_points(detections)[:max_count]
        adjusted = [
            [
                p[0] + VISUAL_PICK_OFFSET[0],
                p[1] + VISUAL_PICK_OFFSET[1],
                p[2] + VISUAL_PICK_OFFSET[2],
            ]
            for p in points
        ]
        self.rospy.loginfo(
            "visual pick points: %s",
            [[round(v, 3) for v in point] for point in adjusted],
        )
        return adjusted

    def _detect_pose_array(self, max_count=4, timeout=3.0):
        try:
            from geometry_msgs.msg import PoseArray
            msg = self.rospy.wait_for_message(POSE_ARRAY_TOPIC, PoseArray, timeout=timeout)
        except Exception:
            return []

        points = [
            [float(p.position.x), float(p.position.y), float(p.position.z)]
            for p in msg.poses
            if all(math.isfinite(v) for v in [p.position.x, p.position.y, p.position.z])
        ]
        points = self._sort_table_points(points)[:max_count]
        adjusted = [
            [
                p[0] + VISUAL_PICK_OFFSET[0],
                p[1] + VISUAL_PICK_OFFSET[1],
                p[2] + VISUAL_PICK_OFFSET[2],
            ]
            for p in points
        ]
        if adjusted:
            self.rospy.loginfo(
                "pose-array pick points: %s",
                [[round(v, 3) for v in point] for point in adjusted],
            )
        return adjusted

    def _extract_points(self, msg):
        points = []
        for detection in msg.detections:
            if not detection.results:
                continue
            pos = detection.results[0].pose.pose.position
            point = [float(pos.x), float(pos.y), float(pos.z)]
            if all(math.isfinite(v) for v in point):
                points.append(point)
        return points

    def _sort_table_points(self, points):
        # Stable order for a 2x2 layout: first by x, then by y.
        return sorted(points, key=lambda p: (p[0], p[1]))


def pick_weigh_drop_one(actions, pick_point, weigh_point, drop_point, execute_grasp=False):
    actions.rospy.loginfo("pick parcel at %s", pick_point)
    actions.open_right_claw()
    actions.move_right_hand(travel_above(pick_point), duration=2.0)
    actions.move_right_hand(above(pick_point, SAFE_APPROACH_Z), duration=2.0)

    if not execute_grasp:
        actions.rospy.logwarn(
            "debug mode: stopped above parcel. If the claw is safely above the parcel, rerun with --execute-grasp"
        )
        return

    actions.move_right_hand(above(pick_point, PICK_DESCEND_Z), duration=1.8)
    actions.close_right_claw()
    actions.move_right_hand(above(pick_point, SAFE_APPROACH_Z), duration=1.8)
    actions.move_right_hand(travel_above(pick_point), duration=1.5)

    actions.rospy.loginfo("weigh parcel at %s", weigh_point)
    actions.move_right_hand(travel_above(weigh_point), duration=2.0)
    actions.move_right_hand(above(weigh_point, SAFE_APPROACH_Z), duration=2.0)
    actions.move_right_hand(above(weigh_point, WEIGH_DESCEND_Z), duration=1.8)
    actions.rospy.sleep(1.0)
    actions.move_right_hand(above(weigh_point, SAFE_APPROACH_Z), duration=1.8)
    actions.move_right_hand(travel_above(weigh_point), duration=1.5)

    actions.rospy.loginfo("drop parcel at %s", drop_point)
    actions.move_right_hand(travel_above(drop_point), duration=2.0)
    actions.move_right_hand(above(drop_point, SAFE_APPROACH_Z), duration=2.0)
    actions.move_right_hand(above(drop_point, DROP_DESCEND_Z), duration=1.8)
    actions.open_right_claw()
    actions.move_right_hand(above(drop_point, SAFE_APPROACH_Z), duration=1.8)
    actions.move_right_hand(travel_above(drop_point), duration=1.5)


def run_scene1(actions, execute_grasp=False, max_parcels=4):
    actions.rospy.loginfo("scene1: start parcel weighing and placing")
    actions.wait_for_arm_subscriber()
    actions.look_down()
    actions.set_arm_mode(ARM_MODE_EXTERNAL_CONTROL)

    actions.move_arm_degrees(SAFE_PREGRASP_DEG, duration=2.0)
    actions.open_right_claw()

    visual_points = VisionParcelLocator(actions.rospy).detect_pick_points(max_count=4)
    pick_points = visual_points if visual_points else PICK_POINTS

    for index, pick_point in enumerate(pick_points[:max_parcels]):
        if actions.rospy.is_shutdown():
            break
        pick_weigh_drop_one(
            actions,
            pick_point,
            WEIGH_POINT,
            DROP_POINTS[index],
            execute_grasp=execute_grasp,
        )
        if not execute_grasp:
            break

    actions.move_arm_degrees(SAFE_PREGRASP_DEG, duration=2.0)
    actions.open_right_claw()
    actions.set_arm_mode(ARM_MODE_AUTO_SWING)
    actions.rospy.loginfo("scene1: finished")


def run_scene2(actions):
    actions.rospy.logwarn("scene2 is not implemented in this file")


def run_scene3(actions):
    actions.rospy.logwarn("scene3 is not implemented in this file")


def execute_task(scene, actions, execute_grasp=False, max_parcels=4):
    if scene == "scene1":
        run_scene1(actions, execute_grasp=execute_grasp, max_parcels=max_parcels)
    elif scene == "scene2":
        run_scene2(actions)
    elif scene == "scene3":
        run_scene3(actions)
    else:
        raise ValueError("unknown scene: {}".format(scene))


def load_launcher():
    try:
        import rospkg
        sim_utils = os.path.join(
            rospkg.RosPack().get_path("challenge_cup_simulator"), "utils"
        )
    except Exception:
        sim_utils = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "challenge_cup_simulator", "utils",
        )
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher
    return ChallengeSimLauncher


def run_scene(scene, seed, node_name=None, timeout=120, time_limit=None,
              timer_gui=True, execute_grasp=False, max_parcels=4):
    if scene not in SCENE_CONFIGS:
        raise ValueError("unknown scene: {}".format(scene))

    ChallengeSimLauncher = load_launcher()
    launcher = ChallengeSimLauncher(
        scene=scene,
        seed=seed,
        match_time_limit=time_limit,
        timer_gui=timer_gui,
    )
    launcher.start(
        node_name=node_name or SCENE_CONFIGS[scene]["node_name"],
        timeout=timeout,
    )

    import rospy
    from sensor_msgs.msg import JointState

    rospy.loginfo("=== %s ===", SCENE_CONFIGS[scene]["title"])
    arm_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    rospy.sleep(1.0)

    actions = BasicActions(rospy, arm_pub)
    execute_task(scene, actions, execute_grasp=execute_grasp, max_parcels=max_parcels)

    rospy.loginfo("task script finished; keep node alive for timer/checking")
    rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="Challenge Cup unified task entry")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="scene1")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--node-name", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--no-timer-gui", action="store_true")
    parser.add_argument(
        "--execute-grasp",
        action="store_true",
        help="actually descend, close claw, weigh, and drop. Without this, only hover above the first parcel.",
    )
    parser.add_argument("--max-parcels", type=int, default=4)
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
        execute_grasp=args.execute_grasp,
        max_parcels=args.max_parcels,
    )


if __name__ == "__main__":
    main()
