#!/usr/bin/env python3

import math

import rospy
import tf

from kuavo_msgs.msg import sensorsData


SENSOR_TOPIC = "/sensors_data_raw"
BASE_FRAME = "base_link"
LEFT_EE_FRAME = "zarm_l7_end_effector"
RIGHT_EE_FRAME = "zarm_r7_end_effector"


def read_arm_joint_deg():
    """读取14个机械臂关节角，只读取，不发布控制命令。"""
    state = rospy.wait_for_message(
        SENSOR_TOPIC,
        sensorsData,
        timeout=5.0
    )

    all_joint_q = list(state.joint_data.joint_q)

    if len(all_joint_q) < 26:
        raise RuntimeError(
            "关节反馈长度不足：期望至少26个，实际%d个"
            % len(all_joint_q)
        )

    if len(all_joint_q) >= 27:
        arm_rad = all_joint_q[13:27]
    else:
        arm_rad = all_joint_q[12:26]
    arm_deg = [math.degrees(value) for value in arm_rad]

    return arm_deg


def read_end_effector_tf(listener, frame_name):
    """读取末端相对于base_link的位置和姿态。"""
    listener.waitForTransform(
        BASE_FRAME,
        frame_name,
        rospy.Time(0),
        rospy.Duration(3.0)
    )

    position, quaternion = listener.lookupTransform(
        BASE_FRAME,
        frame_name,
        rospy.Time(0)
    )

    rpy_rad = tf.transformations.euler_from_quaternion(quaternion)
    rpy_deg = [math.degrees(value) for value in rpy_rad]

    return position, quaternion, rpy_deg


def print_end_effector(name, position, quaternion, rpy_deg):
    rospy.loginfo(
        "%s末端位置 xyz(m): %s",
        name,
        [round(value, 4) for value in position]
    )

    rospy.loginfo(
        "%s末端四元数 xyzw: %s",
        name,
        [round(value, 4) for value in quaternion]
    )

    rospy.loginfo(
        "%s末端姿态 RPY(deg): %s",
        name,
        [round(value, 2) for value in rpy_deg]
    )


def main():
    rospy.init_node("scene2_arm_feedback_read_only")

    rospy.logwarn("Scene2只读诊断：不会发布轨迹，不会切换控制模式")

    # TF监听器需要一点时间接收数据。
    listener = tf.TransformListener()
    rospy.sleep(1.0)

    arm_deg = read_arm_joint_deg()

    rospy.loginfo(
        "当前14个机械臂关节角(deg): %s",
        [round(value, 2) for value in arm_deg]
    )

    left_data = read_end_effector_tf(listener, LEFT_EE_FRAME)
    print_end_effector("左", *left_data)

    right_data = read_end_effector_tf(listener, RIGHT_EE_FRAME)
    print_end_effector("右", *right_data)

    rospy.loginfo("只读诊断完成，未发送任何控制命令")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as error:
        rospy.logerr("Scene2只读诊断失败：%s", error)
