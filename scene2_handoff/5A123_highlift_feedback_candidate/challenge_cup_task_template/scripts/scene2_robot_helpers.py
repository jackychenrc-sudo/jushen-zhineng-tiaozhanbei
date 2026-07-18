#!/usr/bin/env python3
"""Allowed Scene2 robot-control helpers used by the task script.

This module contains only public robot feedback/control interfaces and fixed
robot geometry.  It deliberately contains no simulator layout, object pose,
scene-management, recording, or process-management code.
"""

import math
import threading
import time


ARM_JOINT_NAMES = ["arm_joint_" + str(i) for i in range(1, 15)]

# Fixed robot kinematic calibration used by the existing task flow.
WORLD_TO_EE_OFFSET_X = 0.566
WORLD_TO_EE_OFFSET_Y_RIGHT = -0.014
WORLD_TO_EE_OFFSET_Y_LEFT = 0.014
WORLD_TO_EE_OFFSET_Z = -0.923783

# Joint-space release poses.  These are robot commands, not object positions.
PLACE_ACTIVE_ARM_JOINTS_DEG = {
    "right": {
        "sorting_bin_a": [-30, -10, -10, -80, 70, 0, 0],
        "sorting_bin_b": [-30, 0, 30, -80, 70, 0, 0],
    },
    "left": {
        "sorting_bin_b": [-30, 0, -30, -80, -70, 0, 0],
        "sorting_bin_c": [-30, 10, 10, -80, -70, 0, 0],
    },
}

PRE_GRASP_APPROACH_Z_OFFSET = 0.1
PLACE_DWELL = 0.4

HEAD_TARGET = [0.0, 20.0]
HEAD_SETTLE_TIME = 0.4
ARM_MODE_EXTERNAL_CONTROL = 2
ARM_MODE_AUTO_SWING = 1
ARM_MODE_SERVICE = "/arm_traj_change_mode"
ARM_TARGET_POSES_TOPIC = "/kuavo_arm_target_poses"
ARM_TRAJ_TOPIC = "/kuavo_arm_traj"
ARM_TRAJ_HZ = 100.0
ARM_MOVE_TIME = 1.4
IK_MODE_POS_HARD_ORI_SOFT = 0x02
IK_MODE_THREE_POINT_MIXED = 0x06
FAST_GRASP_SETTLE_HOLD = 0.8

RIGHT_GRIPPER_OPEN = 0.0
LEFT_GRIPPER_OPEN = 0.0
RIGHT_GRIPPER_CLOSE = 255.0
LEFT_GRIPPER_CLOSE = 255.0
GRIPPER_COMMAND_HZ = 100.0


def rad_to_deg(point):
    return [math.degrees(float(value)) for value in point]


def _wait_for_connection(pub, timeout):
    import rospy

    start = time.time()
    while (
        pub.get_num_connections() == 0
        and time.time() - start < timeout
        and not rospy.is_shutdown()
    ):
        rospy.sleep(0.2)
    if pub.get_num_connections() == 0:
        raise RuntimeError(f"topic {pub.name} has no subscriber")


def _set_arm_mode(mode, timeout):
    import rospy
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

    rospy.wait_for_service(ARM_MODE_SERVICE, timeout=timeout)
    proxy = rospy.ServiceProxy(ARM_MODE_SERVICE, changeArmCtrlMode)
    request = changeArmCtrlModeRequest()
    request.control_mode = mode
    response = proxy(request)
    if not response.result:
        raise RuntimeError(
            f"{ARM_MODE_SERVICE} rejected mode {mode}: {response.message}"
        )
    rospy.loginfo("scene2 sorting: arm mode -> %s: %s", mode, response.message)


def _publish_head_target(timeout):
    import rospy
    from kuavo_msgs.msg import robotHeadMotionData

    pub = rospy.Publisher(
        "/robot_head_motion_data", robotHeadMotionData, queue_size=10
    )
    _wait_for_connection(pub, timeout)

    msg = robotHeadMotionData()
    msg.joint_data = list(HEAD_TARGET)
    for _ in range(5):
        if rospy.is_shutdown():
            break
        pub.publish(msg)
        rospy.sleep(0.1)
    rospy.sleep(HEAD_SETTLE_TIME)


class GripperCommandHold:
    def __init__(self, pub, hz=GRIPPER_COMMAND_HZ):
        self._pub = pub
        self._hz = float(hz)
        self._left_cmd = float(LEFT_GRIPPER_OPEN)
        self._right_cmd = float(RIGHT_GRIPPER_OPEN)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="gripper_command_hold",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_open(self):
        self._set_command(LEFT_GRIPPER_OPEN, RIGHT_GRIPPER_OPEN)

    def set_right_closed(self):
        self._set_command(LEFT_GRIPPER_OPEN, RIGHT_GRIPPER_CLOSE)

    def set_left_closed(self):
        self._set_command(LEFT_GRIPPER_CLOSE, RIGHT_GRIPPER_OPEN)

    def _set_command(self, left_cmd, right_cmd):
        with self._lock:
            self._left_cmd = float(left_cmd)
            self._right_cmd = float(right_cmd)

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState

        rate = rospy.Rate(self._hz)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            with self._lock:
                left_cmd = self._left_cmd
                right_cmd = self._right_cmd
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ["left_gripper_joint", "right_gripper_joint"]
            msg.position = [left_cmd, right_cmd]
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break


def _start_gripper_hold(timeout):
    import rospy
    from sensor_msgs.msg import JointState

    pub = rospy.Publisher("/gripper/command", JointState, queue_size=10)
    _wait_for_connection(pub, timeout)
    hold = GripperCommandHold(pub)
    hold.start()
    return hold


def _publish_gripper_open(gripper_hold):
    gripper_hold.set_open()


def _publish_arm_gripper_close(gripper_hold, arm):
    if arm == "left":
        gripper_hold.set_left_closed()
        return
    if arm == "right":
        gripper_hold.set_right_closed()
        return
    raise ValueError(f"unknown arm: {arm}")


class ArmTrajHold:
    def __init__(self, pub, degrees_list, hz=ARM_TRAJ_HZ):
        self._pub = pub
        self._hz = float(hz)
        self._degrees = self._validate_degrees(degrees_list)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="arm_traj_hold",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_degrees(self, degrees_list):
        degrees = self._validate_degrees(degrees_list)
        with self._lock:
            self._degrees = degrees

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState

        rate = rospy.Rate(self._hz)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            with self._lock:
                degrees = list(self._degrees)
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ARM_JOINT_NAMES
            msg.position = degrees
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break

    @staticmethod
    def _validate_degrees(degrees_list):
        degrees = [float(value) for value in degrees_list]
        if len(degrees) != len(ARM_JOINT_NAMES):
            raise ValueError(
                f"arm traj command has {len(degrees)} joints, "
                f"expected {len(ARM_JOINT_NAMES)}"
            )
        return degrees


def _start_arm_traj_hold(timeout):
    import rospy
    from sensor_msgs.msg import JointState

    pub = rospy.Publisher(ARM_TRAJ_TOPIC, JointState, queue_size=10)
    _wait_for_connection(pub, timeout)
    initial_degrees = rad_to_deg(_read_current_arm_joints(timeout))
    hold = ArmTrajHold(pub, initial_degrees)
    hold.start()
    return hold


def _publish_arm_traj_interpolation(
    arm_hold, start_degrees, target_degrees, duration
):
    import rospy

    start = [float(value) for value in start_degrees]
    target = [float(value) for value in target_degrees]
    if len(start) != len(target):
        raise ValueError(
            f"arm traj interpolation length mismatch: "
            f"{len(start)} != {len(target)}"
        )

    steps = max(1, int(round(float(duration) * ARM_TRAJ_HZ)))
    rate = rospy.Rate(ARM_TRAJ_HZ)
    for step in range(steps + 1):
        if rospy.is_shutdown():
            break
        alpha = float(step) / float(steps)
        point = [
            start[index] + (target[index] - start[index]) * alpha
            for index in range(len(target))
        ]
        arm_hold.set_degrees(point)
        if step < steps:
            rate.sleep()


def _execute_arm_motion(
    target_pub,
    arm_hold,
    start_degrees,
    target_degrees,
    move_time,
    settle,
):
    import rospy

    _publish_arm_traj_interpolation(
        arm_hold, start_degrees, target_degrees, move_time
    )
    rospy.sleep(settle)


def _read_current_arm_joints(timeout):
    import rospy
    from kuavo_msgs.msg import sensorsData

    msg = rospy.wait_for_message(
        "/sensors_data_raw", sensorsData, timeout=timeout
    )
    joint_q = list(msg.joint_data.joint_q)
    if len(joint_q) >= 27:
        return joint_q[13:27]
    if len(joint_q) >= 26:
        return joint_q[12:26]
    raise RuntimeError(
        f"/sensors_data_raw joint_q has {len(joint_q)} values"
    )


def _place_active_arm_joints(active_arm, bin_name):
    try:
        joints = PLACE_ACTIVE_ARM_JOINTS_DEG[active_arm][bin_name]
    except KeyError:
        raise RuntimeError(
            f"no joint-space place pose configured for "
            f"{active_arm} hand -> {bin_name}"
        )
    if len(joints) != 7:
        raise ValueError(
            f"place pose for {active_arm} hand -> {bin_name} must have 7 joints"
        )
    return [float(value) for value in joints]


def _compose_single_arm_place_joints(
    active_arm, active_joints_deg, locked_other_arm_joints
):
    other_deg = rad_to_deg(locked_other_arm_joints)
    if active_arm == "left":
        return list(active_joints_deg) + other_deg
    if active_arm == "right":
        return other_deg + list(active_joints_deg)
    raise ValueError(f"unknown arm: {active_arm}")
