#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-json", required=True)
    parser.add_argument("--output", default="scene3_upper_reachability.json")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--preferred-left-y", type=float, default=0.28)
    parser.add_argument("--preferred-right-y", type=float, default=-0.28)
    parser.add_argument("--minimum-shelf-distance", type=float, default=0.78)
    parser.add_argument("--distance-step", type=float, default=0.05)
    parser.add_argument("--hand-reference-length", type=float, default=0.193)
    parser.add_argument("--surface-clearance", type=float, default=0.015)
    parser.add_argument("--pregrasp-clearance", type=float, default=0.12)
    parser.add_argument("--y-offset", type=float, default=0.0)
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=-90.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--right-shoulder-y", type=float, default=-0.12735)
    parser.add_argument("--left-shoulder-y", type=float, default=0.12735)
    parser.add_argument("--robustness-perturbation", type=float, default=0.02)
    parser.add_argument("--maximum-position-error", type=float, default=0.02)
    parser.add_argument("--maximum-orientation-error-deg", type=float, default=12.0)
    parser.add_argument("--minimum-selection-score", type=float, default=0.60)
    parser.add_argument("--minimum-depth-shape-score", type=float, default=0.75)
    parser.add_argument("--sensor-topic", default="/sensors_data_raw")
    parser.add_argument("--fk-service", default="/ik/fk_srv")
    parser.add_argument("--ik-service", default="auto")
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser.parse_args()


def raw_xyz(tray):
    values = tray.get("base_link_xyz_raw_m", tray.get("base_link_xyz_m"))
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("tray does not contain a valid base_link XYZ")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("tray base_link XYZ contains a non-finite value")
    return result


def load_stable_detection(path, minimum_score, minimum_depth_score):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    trays = data.get("upper_trays", [])
    temporal = data.get("temporal_validation", {})
    if data.get("algorithm") != "scene3_temporal_consensus_v1":
        raise ValueError("input must come from scene3 temporal consensus")
    if data.get("status") != "ok" or not temporal.get("passed"):
        raise ValueError("temporal detection is not trusted")
    if data.get("target_frame") != "base_link":
        raise ValueError("stable detections must use the base_link frame")
    if len(trays) != 3:
        raise ValueError("expected exactly 3 stable upper trays")
    for tray in trays:
        if tray.get("geometry_source") != "rgbd_vertical":
            raise ValueError("every tray must use RGB-D object geometry")
        if not tray.get("temporal_metrics", {}).get("stable"):
            raise ValueError("every tray track must be temporally stable")
        if float(tray.get("selection_score", 0.0)) < minimum_score:
            raise ValueError("tray selection score is below the safety threshold")
        if float(tray.get("depth_shape_score", 0.0)) < minimum_depth_score:
            raise ValueError("tray depth geometry score is below the safety threshold")
    return data, trays


def choose_target(trays, hand, preferred_left_y, preferred_right_y):
    preferred_y = preferred_left_y if hand == "left" else preferred_right_y
    target = min(
        trays,
        key=lambda tray: (
            abs(raw_xyz(tray)[1] - preferred_y),
            -float(tray.get("selection_score", 0.0)),
        ),
    )
    return target, preferred_y


def distance_grid(current_distance, minimum_distance, step):
    if not 0.01 <= step <= 0.20:
        raise ValueError("distance step must be within [0.01, 0.20] m")
    if not 0.60 <= minimum_distance <= current_distance:
        raise ValueError("minimum shelf distance is outside the scan range")
    count = int(math.floor((current_distance - minimum_distance) / step))
    values = [current_distance - index * step for index in range(count + 1)]
    if values[-1] - minimum_distance > 1e-6:
        values.append(minimum_distance)
    return [float(max(minimum_distance, value)) for value in values]


def quaternion_from_rpy(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def quaternion_angle_error(first, second):
    dot = abs(sum(float(first[index]) * float(second[index]) for index in range(4)))
    dot = max(-1.0, min(1.0, dot))
    return float(2.0 * math.acos(dot))


def position_error(first, second):
    return float(
        math.sqrt(sum((float(first[index]) - float(second[index])) ** 2 for index in range(3)))
    )


def target_poses(
    target_xyz,
    current_shelf_distance,
    candidate_shelf_distance,
    hand_reference_length,
    surface_clearance,
    pregrasp_clearance,
    y_offset,
    z_offset,
):
    forward_shift = current_shelf_distance - candidate_shelf_distance
    virtual_target = [
        float(target_xyz[0] - forward_shift),
        float(target_xyz[1] + y_offset),
        float(target_xyz[2] + z_offset),
    ]
    grasp_wrist = [
        float(virtual_target[0] - hand_reference_length - surface_clearance),
        virtual_target[1],
        virtual_target[2],
    ]
    pregrasp_wrist = [
        float(grasp_wrist[0] - pregrasp_clearance),
        grasp_wrist[1],
        grasp_wrist[2],
    ]
    return virtual_target, pregrasp_wrist, grasp_wrist


def extract_arm_joints(message):
    values = list(message.joint_data.joint_q)
    if len(values) >= 27:
        result = values[13:27]
    elif len(values) >= 26:
        result = values[12:26]
    else:
        raise ValueError("/sensors_data_raw does not contain 14 arm joints")
    if len(result) != 14:
        raise ValueError("failed to extract 14 current arm joints")
    return [float(value) for value in result]


def response_arm_joints(response):
    values = list(getattr(response, "q_arm", []))
    if len(values) >= 14:
        return [float(value) for value in values[:14]]
    left = list(response.hand_poses.left_pose.joint_angles)
    right = list(response.hand_poses.right_pose.joint_angles)
    if len(left) == 7 and len(right) == 7:
        return [float(value) for value in left + right]
    raise ValueError("IK response does not contain 14 arm joints")


class RosIkClient:
    def __init__(self, args):
        import rosservice
        import rospy
        from kuavo_msgs.msg import sensorsData
        from kuavo_msgs.srv import fkSrv, twoArmHandPoseCmdSrv

        self.args = args
        self.rospy = rospy
        self.fk_service_class = fkSrv
        self.ik_service_class = twoArmHandPoseCmdSrv
        rospy.init_node("scene3_upper_reachability", anonymous=True, disable_signals=True)
        available = set(rosservice.get_service_list())
        if args.fk_service not in available:
            raise RuntimeError("FK service is unavailable: {}".format(args.fk_service))
        if args.ik_service == "auto":
            candidates = [
                "/ik/two_arm_hand_pose_cmd_srv_muli_refer",
                "/ik/two_arm_hand_pose_cmd_srv",
            ]
            self.ik_service_name = next(
                (name for name in candidates if name in available), None
            )
        else:
            self.ik_service_name = args.ik_service
        if not self.ik_service_name or self.ik_service_name not in available:
            raise RuntimeError("no supported arm IK service is available")
        rospy.wait_for_service(args.fk_service, timeout=args.timeout)
        rospy.wait_for_service(self.ik_service_name, timeout=args.timeout)
        self.fk_proxy = rospy.ServiceProxy(args.fk_service, fkSrv)
        self.ik_proxy = rospy.ServiceProxy(
            self.ik_service_name, twoArmHandPoseCmdSrv
        )
        sensor = rospy.wait_for_message(args.sensor_topic, sensorsData, timeout=args.timeout)
        self.current_joints = extract_arm_joints(sensor)
        self.current_fk = self.fk(self.current_joints)

    def fk(self, joints):
        response = self.fk_proxy(list(joints))
        if not response.success:
            raise RuntimeError("FK service returned success=false")
        return response.hand_poses

    @staticmethod
    def set_pose(message, position, quaternion):
        message.pos_xyz = [float(value) for value in position]
        message.quat_xyzw = [float(value) for value in quaternion]
        message.elbow_pos_xyz = [0.0, 0.0, 0.0]

    def request(self, hand, position, quaternion):
        from kuavo_msgs.msg import ikSolveParam, twoArmHandPoseCmd

        request = twoArmHandPoseCmd()
        request.use_custom_ik_param = True
        request.joint_angles_as_q0 = True
        request.hand_poses.header.frame_id = "base_link"
        request.hand_poses.left_pose.joint_angles = list(self.current_joints[:7])
        request.hand_poses.right_pose.joint_angles = list(self.current_joints[7:])
        left = self.current_fk.left_pose
        right = self.current_fk.right_pose
        self.set_pose(request.hand_poses.left_pose, left.pos_xyz, left.quat_xyzw)
        self.set_pose(request.hand_poses.right_pose, right.pos_xyz, right.quat_xyzw)
        active = request.hand_poses.left_pose if hand == "left" else request.hand_poses.right_pose
        self.set_pose(active, position, quaternion)
        parameters = ikSolveParam()
        parameters.major_optimality_tol = 1e-3
        parameters.major_feasibility_tol = 1e-3
        parameters.minor_feasibility_tol = 1e-3
        parameters.major_iterations_limit = 200
        parameters.oritation_constraint_tol = 1e-3
        parameters.pos_constraint_tol = 1e-3
        parameters.pos_cost_weight = 0.0
        parameters.constraint_mode = 3
        request.ik_param = parameters
        return request

    def solve(self, hand, position, quaternion):
        result = {
            "requested_position_xyz_m": [float(value) for value in position],
            "requested_quaternion_xyzw": [float(value) for value in quaternion],
            "service_success": False,
            "accepted": False,
        }
        try:
            response = self.ik_proxy(self.request(hand, position, quaternion))
            result["service_success"] = bool(response.success)
            result["time_cost_ms"] = float(getattr(response, "time_cost", 0.0))
            result["error_reason"] = str(getattr(response, "error_reason", ""))
            if not response.success:
                return result
            joints = response_arm_joints(response)
            solved_fk = self.fk(joints)
            pose = solved_fk.left_pose if hand == "left" else solved_fk.right_pose
            pos_error = position_error(pose.pos_xyz, position)
            orientation_error = quaternion_angle_error(pose.quat_xyzw, quaternion)
            result.update(
                {
                    "position_error_m": pos_error,
                    "orientation_error_deg": math.degrees(orientation_error),
                    "joint_angles_rad": joints[:7] if hand == "left" else joints[7:],
                    "accepted": bool(
                        pos_error <= self.args.maximum_position_error
                        and orientation_error
                        <= math.radians(self.args.maximum_orientation_error_deg)
                    ),
                }
            )
            return result
        except Exception as error:
            result["error_reason"] = str(error)
            return result


def perturbations(amount):
    if amount <= 0.0:
        return [[0.0, 0.0, 0.0]]
    return [
        [0.0, 0.0, 0.0],
        [amount, 0.0, 0.0],
        [-amount, 0.0, 0.0],
        [0.0, amount, 0.0],
        [0.0, -amount, 0.0],
        [0.0, 0.0, amount],
        [0.0, 0.0, -amount],
    ]


def add_position(position, offset):
    return [float(position[index] + offset[index]) for index in range(3)]


def robust_check(client, hand, poses, quaternion, amount):
    checks = []
    for pose_name, position in poses.items():
        for offset in perturbations(amount):
            result = client.solve(hand, add_position(position, offset), quaternion)
            checks.append(
                {
                    "pose": pose_name,
                    "offset_xyz_m": offset,
                    "accepted": result.get("accepted", False),
                    "position_error_m": result.get("position_error_m"),
                    "orientation_error_deg": result.get("orientation_error_deg"),
                    "error_reason": result.get("error_reason", ""),
                }
            )
    return {
        "passed": all(check["accepted"] for check in checks),
        "checks": checks,
    }


def run(args):
    data, trays = load_stable_detection(
        args.stable_json,
        args.minimum_selection_score,
        args.minimum_depth_shape_score,
    )
    target, preferred_y = choose_target(
        trays, args.hand, args.preferred_left_y, args.preferred_right_y
    )
    positions = [raw_xyz(tray) for tray in trays]
    current_shelf_distance = float(
        statistics.median(position[0] for position in positions)
    )
    target_xyz = raw_xyz(target)
    shoulder_y = args.left_shoulder_y if args.hand == "left" else args.right_shoulder_y
    client = RosIkClient(args)
    scan = []
    recommendation = None
    for candidate_distance in distance_grid(
        current_shelf_distance, args.minimum_shelf_distance, args.distance_step
    ):
        virtual_target, pregrasp, grasp = target_poses(
            target_xyz,
            current_shelf_distance,
            candidate_distance,
            args.hand_reference_length,
            args.surface_clearance,
            args.pregrasp_clearance,
            args.y_offset,
            args.z_offset,
        )
        yaw = math.atan2(grasp[1] - shoulder_y, max(0.05, grasp[0]))
        yaw += math.radians(args.yaw_offset_deg)
        quaternion = quaternion_from_rpy(
            math.radians(args.roll_deg), math.radians(args.pitch_deg), yaw
        )
        pregrasp_result = client.solve(args.hand, pregrasp, quaternion)
        grasp_result = client.solve(args.hand, grasp, quaternion)
        item = {
            "candidate_shelf_distance_m": candidate_distance,
            "virtual_target_xyz_m": virtual_target,
            "pregrasp_wrist_xyz_m": pregrasp,
            "grasp_wrist_xyz_m": grasp,
            "quaternion_xyzw": quaternion,
            "pregrasp": pregrasp_result,
            "grasp": grasp_result,
            "center_poses_passed": bool(
                pregrasp_result.get("accepted") and grasp_result.get("accepted")
            ),
        }
        if item["center_poses_passed"]:
            item["robustness"] = robust_check(
                client,
                args.hand,
                {"pregrasp": pregrasp, "grasp": grasp},
                quaternion,
                args.robustness_perturbation,
            )
            if item["robustness"]["passed"]:
                recommendation = {
                    "shelf_distance_m": candidate_distance,
                    "additional_forward_motion_m": float(
                        current_shelf_distance - candidate_distance
                    ),
                    "target_id": target.get("id"),
                    "hand": args.hand,
                    "pregrasp_wrist_xyz_m": pregrasp,
                    "grasp_wrist_xyz_m": grasp,
                    "quaternion_xyzw": quaternion,
                    "basis": "pregrasp and grasp IK/FK passed with XYZ perturbations",
                }
        scan.append(item)
        if recommendation is not None:
            break
    return {
        "status": "recommended_distance_found" if recommendation else "no_robust_distance_found",
        "algorithm": "scene3_upper_ik_reachability_v1",
        "safe_analysis_only": True,
        "robot_motion_sent": False,
        "publishers_created": False,
        "source_algorithm": data.get("algorithm"),
        "ik_service": client.ik_service_name,
        "fk_service": args.fk_service,
        "current_shelf_distance_m": current_shelf_distance,
        "selected_target": {
            "id": target.get("id"),
            "hand": args.hand,
            "preferred_y_m": preferred_y,
            "base_link_xyz_raw_m": target_xyz,
            "selection_score": target.get("selection_score"),
            "depth_shape_score": target.get("depth_shape_score"),
        },
        "geometry": {
            "hand_reference_length_m": args.hand_reference_length,
            "surface_clearance_m": args.surface_clearance,
            "pregrasp_clearance_m": args.pregrasp_clearance,
            "robustness_perturbation_m": args.robustness_perturbation,
        },
        "thresholds": {
            "maximum_position_error_m": args.maximum_position_error,
            "maximum_orientation_error_deg": args.maximum_orientation_error_deg,
        },
        "recommended": recommendation,
        "scan": scan,
    }


def main():
    args = parse_args()
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.hand_reference_length <= 0.0 or args.pregrasp_clearance <= 0.0:
        raise ValueError("hand and pregrasp clearances must be positive")
    payload = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    output.write_text(text, encoding="utf-8")
    print(text)
    print("saved_json={}".format(output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)

