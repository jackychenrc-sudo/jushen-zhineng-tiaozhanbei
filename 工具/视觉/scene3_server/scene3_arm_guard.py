#!/usr/bin/env python3
"""Fail-closed Scene3 arm analysis and single-pregrasp guard.

The validated senior visual-servo walking code owns all base motion.  This
module starts only after the base has stopped.  By default it sends no arm or
claw command; the only executable stage is one explicitly confirmed pregrasp.
"""

import json
import math
import statistics
import time
from pathlib import Path


RIGHT_GRIPPER_QUAT_XYZW = [-0.081987, -0.152343, 0.857876, 0.483858]
PREGRASP_CONFIRMATION = "SCENE3_SINGLE_PREGRASP"


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def finite_vector(values, size):
    return (
        isinstance(values, (list, tuple))
        and len(values) == size
        and all(finite_number(value) for value in values)
    )


def vector_distance(first, second):
    return math.sqrt(
        sum((float(first[index]) - float(second[index])) ** 2 for index in range(3))
    )


def quaternion_norm(values):
    return math.sqrt(sum(float(value) ** 2 for value in values))


def normalize_quaternion(values):
    if not finite_vector(values, 4):
        raise ValueError("gripper quaternion must contain four finite values")
    norm = quaternion_norm(values)
    if norm < 1e-9:
        raise ValueError("gripper quaternion has zero norm")
    return [float(value) / norm for value in values]


def quaternion_angle_error_deg(first, second):
    first = normalize_quaternion(first)
    second = normalize_quaternion(second)
    dot = abs(sum(first[index] * second[index] for index in range(4)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def add_position(position, offset):
    return [float(position[index]) + float(offset[index]) for index in range(3)]


def axis_perturbations(amount):
    amount = float(amount)
    return [
        [0.0, 0.0, 0.0],
        [amount, 0.0, 0.0],
        [-amount, 0.0, 0.0],
        [0.0, amount, 0.0],
        [0.0, -amount, 0.0],
        [0.0, 0.0, amount],
        [0.0, 0.0, -amount],
    ]


def rad_to_deg(values):
    return [float(value) * 180.0 / math.pi for value in values]


class ArmGuardConfig(object):
    def __init__(self):
        self.sample_count = 5
        self.sample_timeout_sec = 3.0
        self.target_freshness_sec = 0.5
        self.maximum_target_spread_m = 0.010
        self.required_frame = "base_link"
        self.pregrasp_offset_xyz_m = [-0.16, 0.0, 0.02]
        self.touch_offset_xyz_m = [-0.05, 0.0, 0.02]
        self.grasp_offset_xyz_m = [-0.01, 0.0, 0.02]
        self.initial_lift_delta_z_m = 0.02
        self.retreat_offset_x_m = -0.20
        self.quaternion_xyzw = list(RIGHT_GRIPPER_QUAT_XYZW)
        self.robustness_perturbation_m = 0.02
        self.maximum_fk_position_error_m = 0.02
        self.maximum_fk_orientation_error_deg = 12.0
        self.pregrasp_duration_sec = 2.5


def validate_config(config):
    if config.sample_count < 3:
        raise ValueError("at least three post-motion samples are required")
    if config.sample_timeout_sec <= 0.0 or config.target_freshness_sec <= 0.0:
        raise ValueError("sample timeouts must be positive")
    if not 0.001 <= config.maximum_target_spread_m <= 0.03:
        raise ValueError("target spread gate must be within [0.001, 0.03] m")
    if config.required_frame != "base_link":
        raise ValueError("IK handoff frame must be base_link")
    for name in (
        "pregrasp_offset_xyz_m",
        "touch_offset_xyz_m",
        "grasp_offset_xyz_m",
    ):
        if not finite_vector(getattr(config, name), 3):
            raise ValueError("{} must contain three finite values".format(name))
    config.quaternion_xyzw = normalize_quaternion(config.quaternion_xyzw)
    if not 0.005 <= config.robustness_perturbation_m <= 0.03:
        raise ValueError("IK robustness perturbation must be within [0.005, 0.03] m")
    if config.maximum_fk_position_error_m <= 0.0:
        raise ValueError("FK position tolerance must be positive")
    if config.maximum_fk_orientation_error_deg <= 0.0:
        raise ValueError("FK orientation tolerance must be positive")


def message_sample(message, received_at):
    frame_id = str(getattr(getattr(message, "header", None), "frame_id", ""))
    point = getattr(message, "point", None)
    xyz = [
        getattr(point, "x", None),
        getattr(point, "y", None),
        getattr(point, "z", None),
    ]
    if not finite_vector(xyz, 3):
        raise ValueError("grasp point contains a missing or non-finite coordinate")
    return {
        "frame_id": frame_id,
        "received_at": float(received_at),
        "xyz_m": [float(value) for value in xyz],
    }


def aggregate_target_samples(samples, config):
    if len(samples) < config.sample_count:
        raise ValueError(
            "received {} unique target samples, need {}".format(
                len(samples), config.sample_count
            )
        )
    frames = {sample.get("frame_id") for sample in samples}
    if frames != {config.required_frame}:
        raise ValueError(
            "all target samples must use {}; received {}".format(
                config.required_frame, sorted(frames)
            )
        )
    positions = [sample.get("xyz_m") for sample in samples]
    if not all(finite_vector(position, 3) for position in positions):
        raise ValueError("one or more target samples have invalid XYZ")
    median_xyz = [
        float(statistics.median(position[axis] for position in positions))
        for axis in range(3)
    ]
    maximum_spread = max(
        vector_distance(position, median_xyz) for position in positions
    )
    if maximum_spread > config.maximum_target_spread_m:
        raise ValueError(
            "target spread {:.4f} m exceeds {:.4f} m".format(
                maximum_spread, config.maximum_target_spread_m
            )
        )
    return {
        "frame_id": config.required_frame,
        "sample_count": len(samples),
        "median_xyz_m": median_xyz,
        "maximum_3d_spread_m": maximum_spread,
        "first_received_at": min(sample["received_at"] for sample in samples),
        "last_received_at": max(sample["received_at"] for sample in samples),
    }


def build_grasp_targets(target_xyz, config):
    if not finite_vector(target_xyz, 3):
        raise ValueError("stable target must contain three finite values")
    pregrasp = add_position(target_xyz, config.pregrasp_offset_xyz_m)
    touch = add_position(target_xyz, config.touch_offset_xyz_m)
    grasp = add_position(target_xyz, config.grasp_offset_xyz_m)
    lift = [grasp[0], grasp[1], grasp[2] + config.initial_lift_delta_z_m]
    retreat = [
        target_xyz[0] + config.retreat_offset_x_m,
        grasp[1],
        lift[2],
    ]
    return {
        "coordinate_frame": config.required_frame,
        "pregrasp_xyz_m": pregrasp,
        "touch_xyz_m": touch,
        "grasp_xyz_m": grasp,
        "initial_lift_xyz_m": lift,
        "retreat_xyz_m": retreat,
        "quaternion_xyzw": list(config.quaternion_xyzw),
        "calibration_status": "initial_guess_requires_single_pregrasp_validation",
    }


def right_pose_from_fk(hand_poses):
    right_pose = getattr(hand_poses, "right_pose", None)
    if right_pose is None:
        raise ValueError("FK response does not contain right_pose")
    position = list(getattr(right_pose, "pos_xyz", []))
    quaternion = list(getattr(right_pose, "quat_xyzw", []))
    if not finite_vector(position, 3) or not finite_vector(quaternion, 4):
        raise ValueError("FK right hand pose is invalid")
    return position, quaternion


def solve_and_verify(task, position, quaternion, config):
    result = {
        "requested_position_xyz_m": [float(value) for value in position],
        "requested_quaternion_xyzw": [float(value) for value in quaternion],
        "ik_service_success": False,
        "fk_verified": False,
        "accepted": False,
    }
    try:
        joints = list(task.solve_right_hand_ik(position, quaternion))
        if not finite_vector(joints, 14):
            raise ValueError("IK response must contain 14 finite arm joints")
        result["ik_service_success"] = True
        hand_poses = task.call_fk(joints)
        fk_position, fk_quaternion = right_pose_from_fk(hand_poses)
        position_error = vector_distance(fk_position, position)
        orientation_error = quaternion_angle_error_deg(fk_quaternion, quaternion)
        accepted = (
            position_error <= config.maximum_fk_position_error_m
            and orientation_error <= config.maximum_fk_orientation_error_deg
        )
        result.update(
            {
                "joint_angles_rad": joints,
                "fk_position_xyz_m": fk_position,
                "fk_quaternion_xyzw": fk_quaternion,
                "fk_position_error_m": position_error,
                "fk_orientation_error_deg": orientation_error,
                "fk_verified": True,
                "accepted": bool(accepted),
            }
        )
    except Exception as error:
        result["error_reason"] = str(error)
    return result


def robust_ik_fk_scan(task, targets, config):
    checks = []
    center_solutions = {}
    for pose_name, target_key in (
        ("pregrasp", "pregrasp_xyz_m"),
        ("touch", "touch_xyz_m"),
        ("grasp", "grasp_xyz_m"),
    ):
        for offset in axis_perturbations(config.robustness_perturbation_m):
            requested = add_position(targets[target_key], offset)
            result = solve_and_verify(
                task, requested, targets["quaternion_xyzw"], config
            )
            check = {
                "pose": pose_name,
                "offset_xyz_m": list(offset),
                "accepted": result.get("accepted", False),
                "ik_service_success": result.get("ik_service_success", False),
                "fk_verified": result.get("fk_verified", False),
                "fk_position_error_m": result.get("fk_position_error_m"),
                "fk_orientation_error_deg": result.get(
                    "fk_orientation_error_deg"
                ),
                "error_reason": result.get("error_reason", ""),
            }
            checks.append(check)
            if offset == [0.0, 0.0, 0.0] and result.get("accepted"):
                center_solutions[pose_name] = list(result["joint_angles_rad"])
    passed = (
        len(checks) == 21
        and all(check["accepted"] for check in checks)
        and set(center_solutions) == {"pregrasp", "touch", "grasp"}
    )
    return {
        "passed": bool(passed),
        "perturbation_m": config.robustness_perturbation_m,
        "check_count": len(checks),
        "checks": checks,
        "center_joint_solutions_rad": center_solutions,
        "scope": (
            "independent center and +/-XYZ probes; this is not a collision check "
            "or proof of the full 3D volume"
        ),
    }


class Scene3ArmGuard(object):
    def __init__(self, task, config=None):
        self.task = task
        self.config = config or ArmGuardConfig()
        validate_config(self.config)

    def collect_stable_target(self):
        self.task.stop_base()
        deadline = time.time() + self.config.sample_timeout_sec
        samples = []
        last_marker = None
        while time.time() < deadline and len(samples) < self.config.sample_count:
            message = self.task.get_recent_grasp_point_base(
                freshness=self.config.target_freshness_sec
            )
            marker = getattr(
                self.task, "last_grasp_point_base_wall_time", None
            )
            if message is not None and marker != last_marker:
                received_at = float(marker) if finite_number(marker) else time.time()
                samples.append(message_sample(message, received_at))
                last_marker = marker
            if len(samples) < self.config.sample_count:
                self.task.rospy.sleep(0.05)
        return aggregate_target_samples(samples, self.config)

    def analyze(self):
        report = {
            "algorithm": "scene3_senior_arm_guard_v1",
            "status": "blocked",
            "execution_enabled": False,
            "base_motion_owned_by": "validated_senior_visual_servo",
            "base_command_sent": False,
            "arm_command_sent": False,
            "claw_command_sent": False,
            "command_flags_scope": "post_alignment_arm_guard_only",
            "safe_analysis_only": True,
        }
        try:
            stable_target = self.collect_stable_target()
            targets = build_grasp_targets(
                stable_target["median_xyz_m"], self.config
            )
            robustness = robust_ik_fk_scan(self.task, targets, self.config)
            report.update(
                {
                    "stable_target": stable_target,
                    "targets": targets,
                    "ik_fk_robustness": robustness,
                }
            )
            if robustness["passed"]:
                report["status"] = "single_pregrasp_ready"
                report["ready_for_single_pregrasp"] = True
                report["next_step"] = (
                    "review report, then rerun with explicit single-pregrasp confirmation"
                )
            else:
                report["status"] = "ik_fk_gate_blocked"
                report["ready_for_single_pregrasp"] = False
                report["next_step"] = "do not move the arm; inspect failed IK/FK probes"
        except Exception as error:
            report["status"] = "target_gate_blocked"
            report["ready_for_single_pregrasp"] = False
            report["error_reason"] = str(error)
            report["next_step"] = "do not move the arm; reacquire a stable base_link target"
        return report

    def run(self, execute_pregrasp=False, confirmation=""):
        report = self.analyze()
        if not execute_pregrasp:
            return report
        if confirmation != PREGRASP_CONFIRMATION:
            report["status"] = "confirmation_blocked"
            report["ready_for_single_pregrasp"] = False
            report["next_step"] = "exact pregrasp confirmation was not provided"
            return report
        if report.get("status") != "single_pregrasp_ready":
            return report

        target = report["targets"]["pregrasp_xyz_m"]
        quaternion = report["targets"]["quaternion_xyzw"]
        joints = report["ik_fk_robustness"]["center_joint_solutions_rad"][
            "pregrasp"
        ]
        if not self.task.set_arm_mode(2):
            report["status"] = "arm_mode_blocked"
            report["ready_for_single_pregrasp"] = False
            report["next_step"] = "failed to enter external arm control mode"
            return report
        self.task.stop_base()
        self.task.move_arm_degrees(
            rad_to_deg(joints), duration=self.config.pregrasp_duration_sec
        )
        report["arm_command_sent"] = True
        current_joints = self.task.read_current_arm_joints(timeout=5.0)
        hand_poses = self.task.call_fk(current_joints)
        actual_position, actual_quaternion = right_pose_from_fk(hand_poses)
        position_error = vector_distance(actual_position, target)
        orientation_error = quaternion_angle_error_deg(actual_quaternion, quaternion)
        actual_passed = (
            position_error <= self.config.maximum_fk_position_error_m
            and orientation_error <= self.config.maximum_fk_orientation_error_deg
        )
        report["post_execution_fk"] = {
            "passed": bool(actual_passed),
            "actual_position_xyz_m": actual_position,
            "actual_quaternion_xyzw": actual_quaternion,
            "position_error_m": position_error,
            "orientation_error_deg": orientation_error,
        }
        report["safe_analysis_only"] = False
        report["execution_enabled"] = True
        report["ready_for_single_pregrasp"] = False
        if actual_passed:
            report["status"] = "single_pregrasp_completed"
            report["next_step"] = (
                "stop here; visually inspect wrist-to-tray offset before enabling touch"
            )
        else:
            report["status"] = "post_execution_fk_failed"
            report["next_step"] = "stop; do not approach or close the claw"
        return report


def write_report(report, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    return path
