#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
from argparse import Namespace
from pathlib import Path

import scene3_multiframe_stability as stability
import scene3_upper_task_planner as planner


SCENARIOS = (
    "nominal",
    "full_dry_run",
    "unstable_vision",
    "missed_detection",
    "too_close",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, default="nominal")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--initial-distance", type=float, default=1.30)
    parser.add_argument("--desired-distance", type=float, default=0.90)
    parser.add_argument("--maximum-cycles", type=int, default=12)
    parser.add_argument("--motion-gain", type=float, default=0.72)
    parser.add_argument("--motion-noise", type=float, default=0.003)
    parser.add_argument(
        "--output", default="scene3_upper_workflow_simulation.json"
    )
    return parser.parse_args()


def stability_args():
    return Namespace(
        expected_count=3,
        minimum_frames=3,
        minimum_selection_score=0.60,
        minimum_depth_shape_score=0.75,
        maximum_pixel_spread=12.0,
        maximum_position_spread=0.02,
    )


def planner_args(desired_distance):
    return Namespace(
        hand="right",
        preferred_left_y=0.28,
        preferred_right_y=-0.28,
        desired_shelf_distance=desired_distance,
        approach_tolerance=0.03,
        maximum_forward_pulse=0.08,
    )


def synthetic_calibration():
    return {
        "calibrated": True,
        "hand": "right",
        "pregrasp_offset_xyz_m": [-0.10, 0.0, 0.02],
        "grasp_offset_xyz_m": [-0.02, 0.0, 0.01],
        "withdraw_offset_xyz_m": [-0.14, 0.0, 0.04],
        "end_effector_rpy_deg": [0.0, 90.0, 0.0],
        "outbound_prepose_xyz_m": [0.45, -0.30, 0.55],
        "outbound_release_xyz_m": [0.42, -0.30, 0.45],
        "outbound_rpy_deg": [0.0, 90.0, 0.0],
        "claw_open_positions": [0.0, 100.0],
        "claw_close_positions": [100.0, 100.0],
        "claw_velocity": [90.0, 90.0],
        "claw_effort": [1.0, 1.0],
    }


def synthetic_tray(index, distance, rng):
    lateral_positions = (0.30, 0.0, -0.29)
    reference_pixels = (500.0, 650.0, 760.0)
    scale = 1.30 / max(distance, 0.50)
    u = int(round(640.0 + (reference_pixels[index] - 640.0) * scale))
    v = int(round(245.0 + (1.30 - distance) * 70.0))
    u += rng.randint(-2, 2)
    v += rng.randint(-2, 2)
    width = max(12, int(round(16.0 * scale)))
    height = max(40, int(round(50.0 * scale)))
    xyz = [
        distance + (index - 1) * 0.002 + rng.gauss(0.0, 0.002),
        lateral_positions[index] + rng.gauss(0.0, 0.0015),
        0.27 + rng.gauss(0.0, 0.002),
    ]
    object_bbox = [
        u - width // 2,
        v - height // 2,
        u + math.ceil(width / 2.0),
        v + math.ceil(height / 2.0),
    ]
    detection_bbox = [
        object_bbox[0] + 1,
        object_bbox[1] + 2,
        object_bbox[2] - 1,
        object_bbox[3] - 2,
    ]
    return {
        "id": "upper_x{}".format(index),
        "accepted": True,
        "geometry_source": "rgbd_vertical",
        "proposal_sources": ["multiscale_template", "rgbd_vertical"],
        "object_bbox": object_bbox,
        "bbox": object_bbox,
        "detection_bbox": detection_bbox,
        "object_center_pixel": [u, v],
        "center_pixel": [u, v],
        "depth_m": max(0.10, distance + rng.gauss(0.0, 0.002)),
        "camera_xyz_m": [xyz[1], -0.40, distance],
        "base_link_xyz_m": list(xyz),
        "base_link_xyz_raw_m": list(xyz),
        "base_link_xyz_corrected_m": list(xyz),
        "selection_score": 0.80 + 0.02 * index,
        "depth_shape_score": 0.90,
        "template_score": 0.55,
    }


def synthetic_frames(distance, rng, scenario, cycle_index):
    frames = []
    for frame_index in range(3):
        trays = [synthetic_tray(index, distance, rng) for index in range(3)]
        if scenario == "unstable_vision" and cycle_index == 0 and frame_index == 2:
            trays[0]["base_link_xyz_raw_m"][0] += 0.05
            trays[0]["base_link_xyz_m"][0] += 0.05
        if scenario == "missed_detection" and cycle_index == 0 and frame_index == 1:
            trays.pop(0)
        rng.shuffle(trays)
        frames.append(
            {
                "status": "ok" if len(trays) == 3 else "count_mismatch",
                "algorithm": "multiscale_rgbd_object_geometry_v5",
                "source_frame": "Head Camera View",
                "target_frame": "base_link",
                "upper_trays": trays,
            }
        )
    return frames


def compact_consensus(payload):
    return {
        "status": payload["status"],
        "temporal_validation": payload["temporal_validation"],
        "tracks": payload.get("tracks", []),
        "upper_trays": payload.get("upper_trays", []),
    }


def compact_plan(payload):
    return {
        "status": payload["status"],
        "shelf_distance_m": payload["shelf_distance_m"],
        "remaining_forward_m": payload["remaining_forward_m"],
        "planned_forward_pulse_m": payload["planned_forward_pulse_m"],
        "selected_target": payload["selected_target"],
        "blocked_reason": payload["blocked_reason"],
        "states": payload["states"],
    }


def run_simulation(args):
    if args.maximum_cycles < 1:
        raise ValueError("maximum cycles must be positive")
    if not 0.0 < args.motion_gain <= 1.5:
        raise ValueError("motion gain must be within (0, 1.5]")
    if args.motion_noise < 0.0:
        raise ValueError("motion noise cannot be negative")
    rng = random.Random(args.seed)
    distance = float(args.initial_distance)
    if args.scenario == "too_close":
        distance = float(args.desired_distance - 0.12)
    use_synthetic_calibration = args.scenario == "full_dry_run"
    calibration = (
        synthetic_calibration() if use_synthetic_calibration else {"calibrated": False}
    )
    result = {
        "simulation_only": True,
        "physics_validated": False,
        "ros_imported": False,
        "robot_commands_sent": False,
        "scenario": args.scenario,
        "random_seed": args.seed,
        "model": {
            "type": "logic_and_measurement_abstraction",
            "motion_gain": args.motion_gain,
            "motion_noise_m": args.motion_noise,
            "collision_model": False,
            "inverse_kinematics": False,
            "contact_and_grasp_physics": False,
            "synthetic_calibration_used": use_synthetic_calibration,
        },
        "initial_distance_m": distance,
        "desired_distance_m": args.desired_distance,
        "cycles": [],
        "terminal_status": None,
    }
    for cycle_index in range(args.maximum_cycles):
        cycle = {
            "cycle": cycle_index + 1,
            "distance_before_m": distance,
        }
        frames = synthetic_frames(distance, rng, args.scenario, cycle_index)
        consensus = stability.build_consensus(
            frames,
            [Path("synthetic_frame_{:03d}.json".format(index + 1)) for index in range(3)],
            stability_args(),
        )
        cycle["vision"] = compact_consensus(consensus)
        if consensus["status"] != "ok":
            cycle["decision"] = "stop_before_motion"
            result["cycles"].append(cycle)
            result["terminal_status"] = "vision_safety_stop"
            break
        task_plan = planner.build_plan(
            planner_args(args.desired_distance),
            consensus,
            consensus["upper_trays"],
            calibration,
        )
        cycle["task_plan"] = compact_plan(task_plan)
        if task_plan["status"] == "approach_required":
            commanded = float(task_plan["planned_forward_pulse_m"])
            actual = max(
                0.0,
                commanded * args.motion_gain + rng.gauss(0.0, args.motion_noise),
            )
            actual = min(actual, commanded * 1.5)
            distance = max(0.0, distance - actual)
            cycle["motion_abstraction"] = {
                "commanded_forward_m": commanded,
                "simulated_actual_forward_m": actual,
                "distance_after_m": distance,
                "requires_new_vision": True,
            }
            cycle["decision"] = "simulate_one_pulse_then_reacquire"
            result["cycles"].append(cycle)
            continue
        if task_plan["status"] == "ready_dry_run":
            cycle["decision"] = "simulate_state_sequence_without_execution"
            cycle["simulated_state_results"] = [
                {"name": item["name"], "result": "logic_path_reached"}
                for item in task_plan["states"]
            ]
            result["cycles"].append(cycle)
            result["terminal_status"] = "ready_dry_run"
            break
        cycle["decision"] = "stop_before_arm_or_body_command"
        result["cycles"].append(cycle)
        result["terminal_status"] = task_plan["status"]
        break
    if result["terminal_status"] is None:
        result["terminal_status"] = "maximum_cycles_reached"
    result["final_distance_m"] = distance
    result["simulated_forward_total_m"] = float(
        sum(
            cycle.get("motion_abstraction", {}).get(
                "simulated_actual_forward_m", 0.0
            )
            for cycle in result["cycles"]
        )
    )
    return result


def main():
    args = parse_args()
    payload = run_simulation(args)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(text)
    print("saved_simulation={}".format(args.output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
