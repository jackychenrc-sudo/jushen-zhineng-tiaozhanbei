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
    parser.add_argument("--calibration")
    parser.add_argument("--output", default="scene3_upper_task_plan.json")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--preferred-left-y", type=float, default=0.28)
    parser.add_argument("--preferred-right-y", type=float, default=-0.28)
    parser.add_argument("--desired-shelf-distance", type=float, default=0.90)
    parser.add_argument("--approach-tolerance", type=float, default=0.03)
    parser.add_argument("--maximum-forward-pulse", type=float, default=0.08)
    return parser.parse_args()


def raw_xyz(tray):
    values = tray.get("base_link_xyz_raw_m", tray.get("base_link_xyz_m"))
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("tray does not contain a valid base_link XYZ")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("tray base_link XYZ contains a non-finite value")
    return result


def load_stable_detection(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    trays = data.get("upper_trays", [])
    temporal = data.get("temporal_validation", {})
    if (
        data.get("algorithm") != "scene3_temporal_consensus_v1"
        or data.get("status") != "ok"
        or not temporal.get("passed")
    ):
        raise ValueError("temporal detection is not trusted")
    if len(trays) != 3:
        raise ValueError("expected exactly 3 stable upper trays")
    if any(tray.get("geometry_source") != "rgbd_vertical" for tray in trays):
        raise ValueError("all trays must use RGB-D object geometry")
    if any(not tray.get("temporal_metrics", {}).get("stable") for tray in trays):
        raise ValueError("all tray tracks must be temporally stable")
    if data.get("target_frame") != "base_link":
        raise ValueError("stable detections must use the base_link target frame")
    return data, trays


def load_calibration(path):
    if not path:
        return {"calibrated": False}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_vector(calibration, key, length):
    value = calibration.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("calibration {} must contain {} values".format(key, length))
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("calibration {} contains a non-finite value".format(key))
    return result


def validate_calibration(calibration, hand):
    if not calibration.get("calibrated"):
        return None
    configured_hand = calibration.get("hand")
    if configured_hand != hand:
        raise ValueError(
            "calibration hand {} does not match requested hand {}".format(
                configured_hand, hand
            )
        )
    return {
        "hand": configured_hand,
        "pregrasp_offset_xyz_m": require_vector(
            calibration, "pregrasp_offset_xyz_m", 3
        ),
        "grasp_offset_xyz_m": require_vector(calibration, "grasp_offset_xyz_m", 3),
        "withdraw_offset_xyz_m": require_vector(
            calibration, "withdraw_offset_xyz_m", 3
        ),
        "end_effector_rpy_deg": require_vector(
            calibration, "end_effector_rpy_deg", 3
        ),
        "outbound_prepose_xyz_m": require_vector(
            calibration, "outbound_prepose_xyz_m", 3
        ),
        "outbound_release_xyz_m": require_vector(
            calibration, "outbound_release_xyz_m", 3
        ),
        "outbound_rpy_deg": require_vector(calibration, "outbound_rpy_deg", 3),
        "claw_open_positions": require_vector(
            calibration, "claw_open_positions", 2
        ),
        "claw_close_positions": require_vector(
            calibration, "claw_close_positions", 2
        ),
        "claw_velocity": require_vector(calibration, "claw_velocity", 2),
        "claw_effort": require_vector(calibration, "claw_effort", 2),
    }


def add_vector(first, second):
    return [float(first[index] + second[index]) for index in range(3)]


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


def state(name, status, safety_gate=None, command=None):
    payload = {"name": name, "status": status}
    if safety_gate:
        payload["safety_gate"] = safety_gate
    if command:
        payload["dry_run_command"] = command
    return payload


def build_plan(args, data, trays, calibration):
    positions = [raw_xyz(tray) for tray in trays]
    shelf_distance = float(statistics.median(position[0] for position in positions))
    if not 0.40 <= shelf_distance <= 2.50:
        raise ValueError("implausible shelf distance {:.3f} m".format(shelf_distance))
    target, preferred_y = choose_target(
        trays, args.hand, args.preferred_left_y, args.preferred_right_y
    )
    target_xyz = raw_xyz(target)
    remaining_forward = shelf_distance - args.desired_shelf_distance
    approach_required = remaining_forward > args.approach_tolerance
    retreat_required = remaining_forward < -args.approach_tolerance
    pulse = max(0.0, min(args.maximum_forward_pulse, remaining_forward))
    parsed_calibration = validate_calibration(calibration, args.hand)
    states = [
        state(
            "VALIDATE_TEMPORAL_VISION",
            "complete",
            "at least 3 stable RGB-D detections",
        )
    ]
    if approach_required:
        states.extend(
            [
                state(
                    "APPROACH_TO_SHELF",
                    "pending",
                    "forward pulse no greater than {:.3f} m".format(
                        args.maximum_forward_pulse
                    ),
                    {
                        "interface": "scene3_upper_approach.py",
                        "distance_m": pulse,
                    },
                ),
                state(
                    "REACQUIRE_TEMPORAL_VISION",
                    "blocked",
                    "repeat temporal consensus after every movement",
                ),
            ]
        )
        status = "approach_required"
        blocked_reason = "robot must approach and reacquire stable vision first"
    elif retreat_required:
        states.append(
            state(
                "RETREAT_FROM_SHELF",
                "blocked",
                "robot is closer than the validated grasp distance; backward control is not implemented",
            )
        )
        status = "reposition_required"
        blocked_reason = "robot is too close to the shelf for the calibrated grasp pose"
    elif parsed_calibration is None:
        states.append(
            state(
                "ARM_CALIBRATION",
                "blocked",
                "validated hand offsets and outbound poses are required",
            )
        )
        status = "calibration_required"
        blocked_reason = "arm and outbound calibration has not been validated"
    else:
        pregrasp_xyz = add_vector(
            target_xyz, parsed_calibration["pregrasp_offset_xyz_m"]
        )
        grasp_xyz = add_vector(target_xyz, parsed_calibration["grasp_offset_xyz_m"])
        withdraw_xyz = add_vector(
            target_xyz, parsed_calibration["withdraw_offset_xyz_m"]
        )
        arm_interface = "EventArmMoveKeyPoint"
        claw_interface = "control_leju_claw"
        states.extend(
            [
                state("ARM_PREPARE", "pending", "collision-free joint preparation"),
                state(
                    "CLAW_OPEN",
                    "pending",
                    "correct hand must be selected",
                    {
                        "interface": claw_interface,
                        "positions": parsed_calibration["claw_open_positions"],
                        "velocity": parsed_calibration["claw_velocity"],
                        "effort": parsed_calibration["claw_effort"],
                    },
                ),
                state(
                    "ARM_PREGRASP",
                    "pending",
                    "pose must remain outside the rack",
                    {
                        "interface": arm_interface,
                        "frame": "base_link",
                        "position_xyz_m": pregrasp_xyz,
                        "rpy_deg": parsed_calibration["end_effector_rpy_deg"],
                    },
                ),
                state(
                    "ARM_GRASP",
                    "pending",
                    "slow final approach with timeout",
                    {
                        "interface": arm_interface,
                        "frame": "base_link",
                        "position_xyz_m": grasp_xyz,
                        "rpy_deg": parsed_calibration["end_effector_rpy_deg"],
                    },
                ),
                state(
                    "CLAW_CLOSE",
                    "pending",
                    "confirm claw command completion",
                    {
                        "interface": claw_interface,
                        "positions": parsed_calibration["claw_close_positions"],
                        "velocity": parsed_calibration["claw_velocity"],
                        "effort": parsed_calibration["claw_effort"],
                    },
                ),
                state(
                    "ARM_WITHDRAW",
                    "pending",
                    "withdraw before any body movement",
                    {
                        "interface": arm_interface,
                        "frame": "base_link",
                        "position_xyz_m": withdraw_xyz,
                        "rpy_deg": parsed_calibration["end_effector_rpy_deg"],
                    },
                ),
                state(
                    "VERIFY_REMOVAL",
                    "pending",
                    "target must disappear from the rack and remain held",
                ),
                state(
                    "MOVE_TO_OUTBOUND",
                    "pending",
                    "keep tray held and monitor drop state",
                    {
                        "interface": arm_interface,
                        "frame": "base_link",
                        "position_xyz_m": parsed_calibration[
                            "outbound_prepose_xyz_m"
                        ],
                        "rpy_deg": parsed_calibration["outbound_rpy_deg"],
                    },
                ),
                state(
                    "OUTBOUND_RELEASE_POSE",
                    "pending",
                    "entire tray projection must be inside outbound box",
                    {
                        "interface": arm_interface,
                        "frame": "base_link",
                        "position_xyz_m": parsed_calibration[
                            "outbound_release_xyz_m"
                        ],
                        "rpy_deg": parsed_calibration["outbound_rpy_deg"],
                    },
                ),
                state(
                    "CLAW_RELEASE",
                    "pending",
                    "release only after outbound pose is reached",
                    {
                        "interface": claw_interface,
                        "positions": parsed_calibration["claw_open_positions"],
                        "velocity": parsed_calibration["claw_velocity"],
                        "effort": parsed_calibration["claw_effort"],
                    },
                ),
                state(
                    "VERIFY_OUTBOUND",
                    "pending",
                    "vision must confirm the tray is fully inside the box",
                ),
                state("ARM_RETRACT", "pending", "return to a collision-free pose"),
            ]
        )
        status = "ready_dry_run"
        blocked_reason = None
    return {
        "status": status,
        "algorithm": "scene3_upper_task_planner_v1",
        "execution_enabled": False,
        "source_algorithm": data.get("algorithm"),
        "temporal_validation": data.get("temporal_validation"),
        "shelf_distance_m": shelf_distance,
        "desired_shelf_distance_m": args.desired_shelf_distance,
        "remaining_forward_m": remaining_forward,
        "planned_forward_pulse_m": pulse if approach_required else 0.0,
        "selected_target": {
            "id": target.get("id"),
            "hand": args.hand,
            "preferred_hand_y_m": preferred_y,
            "center_pixel": target.get("center_pixel"),
            "base_link_xyz_raw_m": target_xyz,
            "selection_score": target.get("selection_score"),
            "depth_shape_score": target.get("depth_shape_score"),
        },
        "blocked_reason": blocked_reason,
        "reference_interfaces": {
            "arm": "EventArmMoveKeyPoint",
            "claw": "control_leju_claw",
            "source": "pytrees_actions.zip",
            "verified_in_current_simulator": False,
        },
        "states": states,
    }


def main():
    args = parse_args()
    data, trays = load_stable_detection(args.stable_json)
    calibration = load_calibration(args.calibration)
    payload = build_plan(args, data, trays, calibration)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    print("saved_plan={}".format(args.output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
