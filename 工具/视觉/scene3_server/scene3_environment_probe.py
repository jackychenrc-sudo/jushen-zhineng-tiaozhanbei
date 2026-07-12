#!/usr/bin/env python3
"""Inspect the Scene3 ROS/SDK interfaces without moving the robot."""

import importlib
import json
import sys

import rosgraph
import roslib.message
import rospy


TOPICS = [
    "/cam_h/color/camera_info",
    "/cam_h/color/image_raw/compressed",
    "/cam_h/depth/image_raw/compressedDepth",
    "/cmd_vel",
    "/kuavo_arm_traj",
    "/sensors_data_raw",
    "/tf",
    "/tf_static",
]

SERVICES = [
    "/control_robot_leju_claw",
    "/humanoid_change_arm_ctrl_mode",
    "/humanoid_get_arm_ctrl_mode",
]

MODULES = [
    "cv2",
    "numpy",
    "tf2_ros",
    "kuavo_msgs",
    "kuavo_humanoid_sdk",
    "py_trees",
]


def inspect_modules():
    result = {}
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            result[name] = {
                "available": True,
                "path": getattr(module, "__file__", None),
                "version": getattr(module, "__version__", None),
            }
        except Exception as error:
            result[name] = {"available": False, "error": str(error)}
    return result


def inspect_ros_graph():
    master = rosgraph.Master(rospy.get_name())
    published = dict(master.getPublishedTopics("/"))
    system_state = master.getSystemState()
    service_names = {name for name, _providers in system_state[2]}

    topics = {}
    for name in TOPICS:
        topics[name] = {
            "available": name in published,
            "type": published.get(name),
        }

    services = {}
    for name in SERVICES:
        service_type = None
        if name in service_names:
            try:
                service_type = rosservice_type(name)
            except Exception as error:
                service_type = "ERROR: {}".format(error)
        services[name] = {
            "available": name in service_names,
            "type": service_type,
        }
    return topics, services


def rosservice_type(service_name):
    import rosservice

    service_type = rosservice.get_service_type(service_name)
    service_class = roslib.message.get_service_class(service_type)
    request_fields = []
    response_fields = []
    if service_class is not None:
        request_fields = list(getattr(service_class._request_class, "__slots__", []))
        response_fields = list(getattr(service_class._response_class, "__slots__", []))
    return {
        "name": service_type,
        "request_fields": request_fields,
        "response_fields": response_fields,
    }


def inspect_camera_frame():
    from sensor_msgs.msg import CameraInfo

    try:
        message = rospy.wait_for_message(
            "/cam_h/color/camera_info", CameraInfo, timeout=5.0
        )
        return {
            "available": True,
            "frame_id": message.header.frame_id,
            "width": message.width,
            "height": message.height,
            "K": list(message.K),
        }
    except Exception as error:
        return {"available": False, "error": str(error)}


def inspect_tf(camera_frame):
    try:
        import tf2_ros

        buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        listener = tf2_ros.TransformListener(buffer)
        rospy.sleep(1.0)
        targets = ["base_link", "base", "torso", "odom"]
        result = {}
        if not camera_frame:
            return {"error": "camera frame is unknown"}
        for target in targets:
            try:
                transform = buffer.lookup_transform(
                    target, camera_frame, rospy.Time(0), rospy.Duration(1.0)
                )
                result[target] = {
                    "available": True,
                    "source": transform.child_frame_id,
                    "target": transform.header.frame_id,
                }
            except Exception as error:
                result[target] = {"available": False, "error": str(error)}
        return result
    except Exception as error:
        return {"error": str(error)}


def main():
    rospy.init_node("scene3_environment_probe", anonymous=True, disable_signals=True)
    modules = inspect_modules()
    topics, services = inspect_ros_graph()
    camera = inspect_camera_frame()
    result = {
        "safe_probe": True,
        "robot_motion_sent": False,
        "python": sys.version,
        "modules": modules,
        "topics": topics,
        "services": services,
        "head_camera": camera,
        "tf_from_head_camera": inspect_tf(camera.get("frame_id")),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

