#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One guarded Scene3 workflow: preprocess -> align -> insert -> verify.

The stages intentionally follow the senior's advice:

1. move the physical TCP to a safe preprocessing point with position-hard,
   orientation-soft IK;
2. keep that TCP fixed while 6D IK rotates the physical gripper upright and
   towards the tray;
3. keep the corrected 6D orientation and insert on one fixed straight line;
4. run the right-wrist RGB-D pre-close gate;
5. close only after a fresh successful gate and an extra explicit flag.

No base publisher exists in this file.  Without ``--execute`` it performs a
read-only audit or calculation only.  Absolute measured joint angles are
never copied into an arm command.
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys
import time

import numpy as np

from scene3_6d_pipeline_core import (
    cartesian_motion_metrics,
    compute_grasp_targets,
    eef_target_for_physical_pose,
    insertion_line_step,
    physical_to_eef_calibration,
    position_only_eef_step,
)
from scene3_6d_pipeline_ros import (
    IK_MODE_POSITION_HARD_ORIENTATION_HARD,
    IK_MODE_POSITION_HARD_ORIENTATION_SOFT,
    Scene36DRosController,
    maximum_abs,
)
from scene3_gripper_6d_align_plan import (
    bounded_rotation,
    desired_gripper_rotation,
    matrix_to_quaternion,
    rotation_error_degrees,
)


CONFIRMATION = "SCENE3_6D_PIPELINE"
VERIFIED_PARAM = (
    "/challenge_cup_task_template/scene3/six_d_preclose_verified"
)
LOCKED_BASE_PARAM = (
    "/challenge_cup_task_template/scene3/locked_target_base_xyz"
)


def candidate_steps(maximum_step, minimum_step, remaining):
    step = min(float(maximum_step), float(remaining))
    values = []
    while step + 1e-12 >= float(minimum_step):
        values.append(step)
        step *= 0.5
    return values


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("audit", "preprocess", "align", "insert", "verify", "close", "all"),
        default="audit",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--close-claw",
        action="store_true",
        help="permit claw closure after a fresh wrist verification",
    )

    parser.add_argument("--tray-center-to-grasp", type=float, default=0.105)
    parser.add_argument("--preprocess-standoff", type=float, default=0.090)
    parser.add_argument("--grasp-height-offset", type=float, default=0.015)
    parser.add_argument("--tcp-extension", type=float, default=0.045)

    parser.add_argument("--preprocess-max-segments", type=int, default=30)
    parser.add_argument("--preprocess-step", type=float, default=0.030)
    parser.add_argument("--preprocess-minimum-step", type=float, default=0.005)
    parser.add_argument("--preprocess-tolerance", type=float, default=0.012)
    parser.add_argument("--align-max-segments", type=int, default=15)
    parser.add_argument("--align-angle-step", type=float, default=5.0)
    parser.add_argument("--align-minimum-angle-step", type=float, default=1.0)
    parser.add_argument("--align-tolerance-deg", type=float, default=6.0)
    parser.add_argument("--insert-max-segments", type=int, default=12)
    parser.add_argument("--insert-step", type=float, default=0.015)
    parser.add_argument("--insert-minimum-step", type=float, default=0.005)
    parser.add_argument("--insert-tolerance", type=float, default=0.010)

    parser.add_argument("--maximum-joint-step-deg", type=float, default=8.0)
    parser.add_argument("--maximum-ik-position-error", type=float, default=0.006)
    parser.add_argument("--maximum-ik-orientation-error", type=float, default=3.0)
    parser.add_argument("--ik-iterations", type=int, default=500)
    parser.add_argument("--maximum-measured-drift-deg", type=float, default=0.12)
    parser.add_argument("--maximum-command-drift-deg", type=float, default=0.08)
    parser.add_argument("--minimum-right-kp", type=float, default=0.1)
    parser.add_argument("--minimum-tray-distance", type=float, default=0.45)
    parser.add_argument("--maximum-tray-distance", type=float, default=0.80)
    parser.add_argument("--maximum-source-freshness-deg", type=float, default=0.20)
    parser.add_argument("--maximum-state-freshness-deg", type=float, default=0.50)
    parser.add_argument("--maximum-prime-joint-motion-deg", type=float, default=0.75)
    parser.add_argument("--maximum-prime-tcp-motion", type=float, default=0.008)
    parser.add_argument("--maximum-left-drift-deg", type=float, default=1.0)
    parser.add_argument("--maximum-tray-identity-shift", type=float, default=0.025)
    parser.add_argument("--maximum-preprocess-cross-track", type=float, default=0.015)
    parser.add_argument("--maximum-align-tcp-error", type=float, default=0.012)
    parser.add_argument("--maximum-insert-cross-track", type=float, default=0.010)
    parser.add_argument("--maximum-final-tcp-error", type=float, default=0.020)
    parser.add_argument("--maximum-final-orientation-error-deg", type=float, default=10.0)

    parser.add_argument("--state-samples", type=int, default=5)
    parser.add_argument("--arm-topic-quiet-seconds", type=float, default=0.4)
    parser.add_argument("--prime-seconds", type=float, default=0.5)
    parser.add_argument("--motion-seconds", type=float, default=3.0)
    parser.add_argument("--rollback-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--wrist-timeout", type=float, default=20.0)
    parser.add_argument("--verification-max-age", type=float, default=25.0)
    return parser


def grasp_targets(args, state):
    return compute_grasp_targets(
        state["tray_base"],
        tray_center_to_grasp_m=float(args.tray_center_to_grasp),
        preprocess_standoff_m=float(args.preprocess_standoff),
        grasp_height_offset_m=float(args.grasp_height_offset),
    )


def predicted_physical_pose(state, plan):
    calibration = physical_to_eef_calibration(
        state["eef_position"],
        state["eef_rotation"],
        state["geometry"]["tcp"],
        state["geometry"]["rotation"],
    )
    rotation = plan["predicted_rotation"]
    tcp = plan["predicted_position"] + rotation.dot(
        calibration["eef_to_tcp"]
    )
    physical_rotation = rotation.dot(
        calibration["eef_to_physical"]
    )
    return tcp, physical_rotation, calibration


def plan_preprocess_segment(controller, args, state):
    targets = grasp_targets(args, state)
    safe = targets["preprocess_tcp"]
    current_tcp = state["geometry"]["tcp"]
    remaining = float(np.linalg.norm(safe - current_tcp))
    if remaining <= float(args.preprocess_tolerance):
        return {"reached": True, "remaining_m": remaining, "targets": targets}

    reasons = []
    for step in candidate_steps(
            args.preprocess_step,
            args.preprocess_minimum_step,
            remaining):
        motion = position_only_eef_step(
            state["eef_position"], current_tcp, safe, step
        )
        plan = controller.solve_pose(
            state,
            motion["target_eef_position"],
            state["eef_quaternion"],
            IK_MODE_POSITION_HARD_ORIENTATION_SOFT,
        )
        if not plan.get("success", False):
            reasons.append("{:.1f}mm:{}".format(step * 1000.0, plan.get("reason", "IK")))
            continue
        predicted_tcp, predicted_rotation, calibration = (
            predicted_physical_pose(state, plan)
        )
        metrics = cartesian_motion_metrics(
            current_tcp, predicted_tcp, safe, motion["direction"]
        )
        predicted_checks = {
            "forward_progress": metrics["progress_m"] >= 0.25 * step,
            "cross_track_bounded": (
                metrics["cross_track_m"]
                <= max(float(args.maximum_preprocess_cross_track), 0.60 * step)
            ),
            "motion_bounded": metrics["motion_m"] <= 2.0 * step + 0.004,
            "target_error_reduced": metrics["target_error_m"] < remaining,
        }
        if not all(predicted_checks.values()):
            reasons.append("{:.1f}mm:{}".format(step * 1000.0, predicted_checks))
            continue
        plan.update({
            "stage": "preprocess",
            "targets": targets,
            "planned_step_m": step,
            "direction": motion["direction"],
            "physical_tcp_before": current_tcp.copy(),
            "predicted_physical_tcp": predicted_tcp,
            "predicted_physical_rotation": predicted_rotation,
            "calibration": calibration,
            "predicted_metrics": metrics,
            "predicted_stage_checks": predicted_checks,
        })
        return plan
    return {
        "reached": False,
        "blocked": True,
        "remaining_m": remaining,
        "targets": targets,
        "reason": "; ".join(reasons[-3:]),
    }


def print_preprocess_plan(plan):
    targets = plan["targets"]
    print("Stage 1/3: move to preprocessing position; orientation remains soft")
    print("Safe preprocessing TCP: {}".format(
        np.round(targets["preprocess_tcp"], 4).tolist()
    ))
    print("Final grasp TCP: {}".format(
        np.round(targets["final_tcp"], 4).tolist()
    ))
    if plan.get("reached", False):
        print("Preprocessing position error: {:.1f}mm".format(
            plan["remaining_m"] * 1000.0
        ))
        print("SIX_D_PREPROCESS_REACHED")
        return
    print("Physical TCP before: {}".format(
        np.round(plan["physical_tcp_before"], 4).tolist()
    ))
    print("Remaining: {:.3f}m".format(
        float(np.linalg.norm(
            targets["preprocess_tcp"] - plan["physical_tcp_before"]
        ))
    ))
    print("Planned TCP step: {:.1f}mm".format(plan["planned_step_m"] * 1000.0))
    print("IK mode: position hard + orientation soft")
    print("Right-arm command delta: {}deg".format(
        np.round(plan["delta_deg"][7:], 3).tolist()
    ))
    print("Predicted physical TCP: {}".format(
        np.round(plan["predicted_physical_tcp"], 4).tolist()
    ))
    print("Predicted checks: {}".format(plan["predicted_stage_checks"]))


def preprocess_post_check(args, plan):
    safe = plan["targets"]["preprocess_tcp"]
    direction = plan["direction"]
    before_tcp = plan["physical_tcp_before"]
    before_error = float(np.linalg.norm(safe - before_tcp))

    def check(before, after):
        metrics = cartesian_motion_metrics(
            before_tcp,
            after["geometry"]["tcp"],
            safe,
            direction,
        )
        left_drift = maximum_abs(np.rad2deg(
            after["measured_arm"][:7] - before["measured_arm"][:7]
        ))
        identity_shift = float(np.linalg.norm(
            after["tray_odom"] - before["tray_odom"]
        ))
        step = plan["planned_step_m"]
        checks = {
            "forward_progress": metrics["progress_m"] >= 0.20 * step,
            "cross_track_bounded": (
                metrics["cross_track_m"]
                <= max(float(args.maximum_preprocess_cross_track), 0.70 * step)
            ),
            "motion_bounded": metrics["motion_m"] <= 2.1 * step + 0.005,
            "target_error_reduced": metrics["target_error_m"] < before_error,
            "left_arm_held": left_drift <= float(args.maximum_left_drift_deg),
            "same_tray": identity_shift <= float(args.maximum_tray_identity_shift),
        }
        print("Observed TCP delta: {}m".format(
            np.round(metrics["movement"], 5).tolist()
        ))
        print("Progress={:.1f}mm cross={:.1f}mm remaining={:.1f}mm".format(
            metrics["progress_m"] * 1000.0,
            metrics["cross_track_m"] * 1000.0,
            metrics["target_error_m"] * 1000.0,
        ))
        return checks, metrics
    return check


def run_preprocess(controller, args, execute):
    limit = int(args.preprocess_max_segments) if execute else 1
    for index in range(max(1, limit)):
        state = controller.sample_state()
        checks = controller.print_audit(state)
        if not all(checks.values()):
            print("SIX_D_PREPROCESS_BLOCKED: live audit failed")
            return 2
        plan = plan_preprocess_segment(controller, args, state)
        print("\n=== Preprocess segment {} ===".format(index + 1))
        print_preprocess_plan(plan)
        if plan.get("reached", False):
            return 0
        if plan.get("blocked", False):
            print("SIX_D_PREPROCESS_BLOCKED: {}".format(plan.get("reason", "no safe IK")))
            return 2
        if not execute:
            print("SIX_D_PREPROCESS_PLAN_OK: calculation only; no command sent")
            return 0
        ok, _ = controller.execute_plan(
            state,
            plan,
            preprocess_post_check(args, plan),
            "SIX_D_PREPROCESS_SEGMENT",
        )
        if not ok:
            return 2
    print("SIX_D_PREPROCESS_PROGRESS_OK: segment limit reached safely")
    return 0


def desired_rotation_for_targets(targets):
    return desired_gripper_rotation(
        targets["preprocess_tcp"],
        targets["tray_center"],
        up_sign=1.0,
    )


def plan_align_segment(controller, args, state):
    targets = grasp_targets(args, state)
    safe_error = float(np.linalg.norm(
        state["geometry"]["tcp"] - targets["preprocess_tcp"]
    ))
    if safe_error > max(0.025, 2.0 * float(args.preprocess_tolerance)):
        return {
            "blocked": True,
            "reason": "TCP is not at preprocessing position ({:.1f}mm)".format(
                safe_error * 1000.0
            ),
            "targets": targets,
        }
    desired = desired_rotation_for_targets(targets)
    before_error = rotation_error_degrees(
        desired, state["geometry"]["rotation"]
    )
    if before_error <= float(args.align_tolerance_deg):
        return {
            "reached": True,
            "orientation_error_deg": before_error,
            "targets": targets,
        }
    calibration = physical_to_eef_calibration(
        state["eef_position"],
        state["eef_rotation"],
        state["geometry"]["tcp"],
        state["geometry"]["rotation"],
    )
    reasons = []
    for requested_angle in candidate_steps(
            args.align_angle_step,
            args.align_minimum_angle_step,
            before_error):
        step_rotation, _, step_angle = bounded_rotation(
            state["geometry"]["rotation"],
            desired,
            requested_angle,
        )
        target_eef = eef_target_for_physical_pose(
            targets["preprocess_tcp"],
            step_rotation,
            calibration["eef_to_tcp"],
            calibration["eef_to_physical"],
        )
        plan = controller.solve_pose(
            state,
            target_eef["eef_position"],
            matrix_to_quaternion(target_eef["eef_rotation"]),
            IK_MODE_POSITION_HARD_ORIENTATION_HARD,
        )
        if not plan.get("success", False):
            reasons.append("{:.1f}deg:{}".format(
                requested_angle, plan.get("reason", "hard 6D IK failed")
            ))
            continue
        predicted_tcp = (
            plan["predicted_position"]
            + plan["predicted_rotation"].dot(calibration["eef_to_tcp"])
        )
        predicted_rotation = plan["predicted_rotation"].dot(
            calibration["eef_to_physical"]
        )
        predicted_error = rotation_error_degrees(desired, predicted_rotation)
        predicted_tcp_error = float(np.linalg.norm(
            predicted_tcp - targets["preprocess_tcp"]
        ))
        predicted_checks = {
            "orientation_improved": predicted_error + 0.35 < before_error,
            "tcp_held": predicted_tcp_error <= float(args.maximum_align_tcp_error),
            "hard_6d_residuals": all(plan["checks"].values()),
        }
        if not all(predicted_checks.values()):
            reasons.append("{:.1f}deg:{}".format(
                requested_angle, predicted_checks
            ))
            continue
        plan.update({
            "stage": "align",
            "targets": targets,
            "desired_physical_rotation": desired,
            "physical_rotation_before": state["geometry"]["rotation"].copy(),
            "physical_tcp_before": state["geometry"]["tcp"].copy(),
            "orientation_error_before_deg": before_error,
            "predicted_orientation_error_deg": predicted_error,
            "predicted_physical_tcp": predicted_tcp,
            "predicted_physical_rotation": predicted_rotation,
            "planned_angle_step_deg": step_angle,
            "predicted_stage_checks": predicted_checks,
        })
        return plan
    return {
        "blocked": True,
        "reason": "; ".join(reasons[-3:]),
        "targets": targets,
    }


def print_align_plan(plan):
    print("Stage 2/3: hold preprocessing TCP and rotate gripper with full 6D IK")
    if plan.get("reached", False):
        print("Physical orientation error: {:.2f}deg".format(
            plan["orientation_error_deg"]
        ))
        print("SIX_D_ALIGN_REACHED")
        return
    print("Physical TCP held at: {}".format(
        np.round(plan["targets"]["preprocess_tcp"], 4).tolist()
    ))
    print("Physical orientation error: {:.2f}deg -> {:.2f}deg predicted".format(
        plan["orientation_error_before_deg"],
        plan["predicted_orientation_error_deg"],
    ))
    print("Planned bounded turn: {:.2f}deg".format(plan["planned_angle_step_deg"]))
    print("Right-arm command delta: {}deg".format(
        np.round(plan["delta_deg"][7:], 3).tolist()
    ))
    print("Predicted checks: {}".format(plan["predicted_stage_checks"]))


def align_post_check(args, plan):
    desired = plan["desired_physical_rotation"]
    safe = plan["targets"]["preprocess_tcp"]
    before_error = plan["orientation_error_before_deg"]

    def check(before, after):
        after_error = rotation_error_degrees(
            desired, after["geometry"]["rotation"]
        )
        tcp_error = float(np.linalg.norm(after["geometry"]["tcp"] - safe))
        left_drift = maximum_abs(np.rad2deg(
            after["measured_arm"][:7] - before["measured_arm"][:7]
        ))
        identity_shift = float(np.linalg.norm(
            after["tray_odom"] - before["tray_odom"]
        ))
        checks = {
            "physical_orientation_improved": after_error + 0.35 < before_error,
            "tcp_held": tcp_error <= float(args.maximum_align_tcp_error),
            "left_arm_held": left_drift <= float(args.maximum_left_drift_deg),
            "same_tray": identity_shift <= float(args.maximum_tray_identity_shift),
        }
        print("Physical orientation error: {:.2f}deg -> {:.2f}deg".format(
            before_error, after_error
        ))
        print("Preprocessing TCP error after turn: {:.1f}mm".format(
            tcp_error * 1000.0
        ))
        return checks, {"orientation_error_deg": after_error, "tcp_error_m": tcp_error}
    return check


def run_align(controller, args, execute):
    limit = int(args.align_max_segments) if execute else 1
    for index in range(max(1, limit)):
        state = controller.sample_state()
        if not all(controller.audit_checks(state).values()):
            print("SIX_D_ALIGN_BLOCKED: live audit failed")
            return 2
        plan = plan_align_segment(controller, args, state)
        print("\n=== 6D alignment segment {} ===".format(index + 1))
        if plan.get("blocked", False):
            print("SIX_D_ALIGN_BLOCKED: {}".format(plan.get("reason", "no safe IK")))
            return 2
        print_align_plan(plan)
        if plan.get("reached", False):
            return 0
        if not execute:
            print("SIX_D_ALIGN_PLAN_OK: calculation only; no command sent")
            return 0
        ok, _ = controller.execute_plan(
            state,
            plan,
            align_post_check(args, plan),
            "SIX_D_ALIGN_SEGMENT",
        )
        if not ok:
            return 2
    print("SIX_D_ALIGN_PROGRESS_OK: segment limit reached safely")
    return 0


def plan_insert_segment(controller, args, state):
    targets = grasp_targets(args, state)
    desired = desired_rotation_for_targets(targets)
    orientation_error = rotation_error_degrees(
        desired, state["geometry"]["rotation"]
    )
    if orientation_error > float(args.align_tolerance_deg) + 2.0:
        return {
            "blocked": True,
            "reason": "gripper is not aligned ({:.2f}deg)".format(orientation_error),
            "targets": targets,
        }
    line_probe = insertion_line_step(
        state["geometry"]["tcp"],
        targets["preprocess_tcp"],
        targets["final_tcp"],
        float(args.insert_step),
    )
    final_error = float(np.linalg.norm(
        state["geometry"]["tcp"] - targets["final_tcp"]
    ))
    if (
        line_probe["remaining_m"] <= float(args.insert_tolerance)
        and final_error <= float(args.maximum_final_tcp_error)
    ):
        return {
            "reached": True,
            "remaining_m": line_probe["remaining_m"],
            "final_error_m": final_error,
            "orientation_error_deg": orientation_error,
            "targets": targets,
        }
    if line_probe["cross_track_m"] > 0.030:
        return {
            "blocked": True,
            "reason": "TCP is too far from insertion line ({:.1f}mm)".format(
                line_probe["cross_track_m"] * 1000.0
            ),
            "targets": targets,
        }
    calibration = physical_to_eef_calibration(
        state["eef_position"],
        state["eef_rotation"],
        state["geometry"]["tcp"],
        state["geometry"]["rotation"],
    )
    available = max(
        line_probe["remaining_m"],
        float(args.insert_minimum_step),
    )
    reasons = []
    for requested_step in candidate_steps(
            args.insert_step,
            args.insert_minimum_step,
            available):
        line = insertion_line_step(
            state["geometry"]["tcp"],
            targets["preprocess_tcp"],
            targets["final_tcp"],
            requested_step,
        )
        target_eef = eef_target_for_physical_pose(
            line["target_tcp"],
            desired,
            calibration["eef_to_tcp"],
            calibration["eef_to_physical"],
        )
        plan = controller.solve_pose(
            state,
            target_eef["eef_position"],
            matrix_to_quaternion(target_eef["eef_rotation"]),
            IK_MODE_POSITION_HARD_ORIENTATION_HARD,
        )
        if not plan.get("success", False):
            reasons.append("{:.1f}mm:{}".format(
                requested_step * 1000.0,
                plan.get("reason", "hard 6D IK failed"),
            ))
            continue
        predicted_tcp = (
            plan["predicted_position"]
            + plan["predicted_rotation"].dot(calibration["eef_to_tcp"])
        )
        predicted_rotation = plan["predicted_rotation"].dot(
            calibration["eef_to_physical"]
        )
        predicted_line = insertion_line_step(
            predicted_tcp,
            targets["preprocess_tcp"],
            targets["final_tcp"],
            0.0,
        )
        predicted_target_error = float(np.linalg.norm(
            predicted_tcp - line["target_tcp"]
        ))
        predicted_orientation_error = rotation_error_degrees(
            desired, predicted_rotation
        )
        correction_only = (
            line["remaining_m"] <= float(args.insert_tolerance)
        )
        before_target_error = float(np.linalg.norm(
            state["geometry"]["tcp"] - line["target_tcp"]
        ))
        minimum_progress = max(0.0015, 0.20 * line["step_m"])
        predicted_checks = {
            "line_progress": (
                predicted_target_error + 0.002 < before_target_error
                if correction_only
                else predicted_line["progress_m"]
                > line["progress_m"] + minimum_progress
            ),
            "line_cross_track": (
                predicted_line["cross_track_m"]
                <= float(args.maximum_insert_cross_track)
            ),
            "target_tcp": predicted_target_error <= 0.010,
            "orientation_locked": (
                predicted_orientation_error
                <= float(args.maximum_final_orientation_error_deg)
            ),
            "hard_6d_residuals": all(plan["checks"].values()),
        }
        if not all(predicted_checks.values()):
            reasons.append("{:.1f}mm:{}".format(
                requested_step * 1000.0, predicted_checks
            ))
            continue
        plan.update({
            "stage": "insert",
            "targets": targets,
            "line_before": line,
            "desired_physical_rotation": desired,
            "physical_tcp_before": state["geometry"]["tcp"].copy(),
            "orientation_error_before_deg": orientation_error,
            "target_physical_tcp": line["target_tcp"],
            "predicted_physical_tcp": predicted_tcp,
            "predicted_physical_rotation": predicted_rotation,
            "predicted_line": predicted_line,
            "predicted_orientation_error_deg": predicted_orientation_error,
            "correction_only": correction_only,
            "before_segment_target_error_m": before_target_error,
            "predicted_stage_checks": predicted_checks,
        })
        return plan
    return {
        "blocked": True,
        "reason": "; ".join(reasons[-3:]),
        "targets": targets,
    }


def print_insert_plan(plan):
    print("Stage 3/3: keep corrected 6D orientation and insert on a straight line")
    if plan.get("reached", False):
        print("Final TCP error: {:.1f}mm".format(plan["final_error_m"] * 1000.0))
        print("Final orientation error: {:.2f}deg".format(plan["orientation_error_deg"]))
        print("SIX_D_INSERT_REACHED")
        return
    line = plan["line_before"]
    print("Insertion progress: {:.1f}/{:.1f}mm".format(
        line["progress_m"] * 1000.0,
        line["line_length_m"] * 1000.0,
    ))
    print("Next physical TCP: {}".format(
        np.round(plan["target_physical_tcp"], 4).tolist()
    ))
    print("Planned straight step: {:.1f}mm".format(line["step_m"] * 1000.0))
    print("Right-arm command delta: {}deg".format(
        np.round(plan["delta_deg"][7:], 3).tolist()
    ))
    print("Predicted checks: {}".format(plan["predicted_stage_checks"]))


def insert_post_check(args, plan):
    targets = plan["targets"]
    desired = plan["desired_physical_rotation"]
    before_progress = plan["line_before"]["progress_m"]

    def check(before, after):
        line = insertion_line_step(
            after["geometry"]["tcp"],
            targets["preprocess_tcp"],
            targets["final_tcp"],
            0.0,
        )
        target_error = float(np.linalg.norm(
            after["geometry"]["tcp"] - plan["target_physical_tcp"]
        ))
        orientation_error = rotation_error_degrees(
            desired, after["geometry"]["rotation"]
        )
        left_drift = maximum_abs(np.rad2deg(
            after["measured_arm"][:7] - before["measured_arm"][:7]
        ))
        identity_shift = float(np.linalg.norm(
            after["tray_odom"] - before["tray_odom"]
        ))
        checks = {
            "straight_progress": (
                target_error + 0.002
                < float(plan["before_segment_target_error_m"])
                if plan.get("correction_only", False)
                else line["progress_m"] > before_progress + 0.002
            ),
            "cross_track_bounded": (
                line["cross_track_m"] <= float(args.maximum_insert_cross_track)
            ),
            "segment_target_reached": target_error <= 0.012,
            "orientation_locked": (
                orientation_error <= float(args.maximum_final_orientation_error_deg)
            ),
            "left_arm_held": left_drift <= float(args.maximum_left_drift_deg),
            "same_tray": identity_shift <= float(args.maximum_tray_identity_shift),
        }
        print("Insertion progress now: {:.1f}mm; cross-track: {:.1f}mm".format(
            line["progress_m"] * 1000.0,
            line["cross_track_m"] * 1000.0,
        ))
        print("Segment TCP error: {:.1f}mm; orientation error: {:.2f}deg".format(
            target_error * 1000.0, orientation_error
        ))
        return checks, {
            "line": line,
            "target_error_m": target_error,
            "orientation_error_deg": orientation_error,
        }
    return check


def run_insert(controller, args, execute):
    limit = int(args.insert_max_segments) if execute else 1
    for index in range(max(1, limit)):
        state = controller.sample_state()
        if not all(controller.audit_checks(state).values()):
            print("SIX_D_INSERT_BLOCKED: live audit failed")
            return 2
        plan = plan_insert_segment(controller, args, state)
        print("\n=== Straight insertion segment {} ===".format(index + 1))
        if plan.get("blocked", False):
            print("SIX_D_INSERT_BLOCKED: {}".format(plan.get("reason", "no safe IK")))
            return 2
        print_insert_plan(plan)
        if plan.get("reached", False):
            return 0
        if not execute:
            print("SIX_D_INSERT_PLAN_OK: calculation only; no command sent")
            return 0
        ok, _ = controller.execute_plan(
            state,
            plan,
            insert_post_check(args, plan),
            "SIX_D_INSERT_SEGMENT",
        )
        if not ok:
            return 2
    print("SIX_D_INSERT_PROGRESS_OK: segment limit reached safely")
    return 0


def final_geometry_checks(controller, args):
    state = controller.sample_state()
    targets = grasp_targets(args, state)
    desired = desired_rotation_for_targets(targets)
    tcp_error = float(np.linalg.norm(
        state["geometry"]["tcp"] - targets["final_tcp"]
    ))
    orientation_error = rotation_error_degrees(
        desired, state["geometry"]["rotation"]
    )
    tray_standoff = float(np.dot(
        state["tray_base"] - state["geometry"]["tcp"],
        desired[:, 0],
    ))
    checks = {
        "final_tcp": tcp_error <= float(args.maximum_final_tcp_error),
        "final_orientation": (
            orientation_error <= float(args.maximum_final_orientation_error_deg)
        ),
        "tray_ahead": tray_standoff >= 0.070,
        "live_audit_ready": all(controller.audit_checks(state).values()),
    }
    print("Final physical TCP: {}".format(
        np.round(state["geometry"]["tcp"], 4).tolist()
    ))
    print("Expected final TCP: {}".format(
        np.round(targets["final_tcp"], 4).tolist()
    ))
    print("Final TCP error: {:.1f}mm".format(tcp_error * 1000.0))
    print("Final physical orientation error: {:.2f}deg".format(orientation_error))
    print("Final geometry checks: {}".format(checks))
    return checks, state, targets


def run_wrist_verification(controller, args):
    checks, state, targets = final_geometry_checks(controller, args)
    if not all(checks.values()):
        print("SIX_D_WRIST_VERIFY_BLOCKED: final 6D geometry is not ready")
        return 2
    # Prompt wrist RGB-D at the same near-edge surface used by the grasp
    # geometry.  Prompting at the tray centre would put the visual target
    # roughly 10.5 cm beyond the finger midpoint and make the pre-close gate
    # fail even when the gripper has reached the planned grasp pose.
    verification_target = targets["grasp_surface"]
    controller.rospy.set_param(
        LOCKED_BASE_PARAM,
        [float(value) for value in verification_target],
    )
    print("Wrist verification near-edge target: {}".format(
        np.round(verification_target, 4).tolist()
    ))
    gate = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scene3_wrist_preclose_gate.py",
    )
    command = [
        sys.executable,
        "-u",
        gate,
        "--target-param",
        LOCKED_BASE_PARAM,
        "--timeout",
        str(float(args.wrist_timeout)),
    ]
    print("Running observation-only right-wrist RGB-D pre-close gate")
    result = subprocess.call(command)
    if result != 0:
        if controller.rospy.has_param(VERIFIED_PARAM):
            controller.rospy.delete_param(VERIFIED_PARAM)
        print("SIX_D_WRIST_VERIFY_BLOCKED: claw remains open")
        return 2
    controller.rospy.set_param(VERIFIED_PARAM, {
        "wall_time": float(time.time()),
        "tray_odom": state["tray_odom"].tolist(),
        "grasp_surface": verification_target.tolist(),
        "final_tcp": targets["final_tcp"].tolist(),
        "observed_tcp": state["geometry"]["tcp"].tolist(),
    })
    print("SIX_D_WRIST_VERIFY_OK")
    print("The claw has not been commanded")
    return 0


def run_close(controller, args):
    from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest

    if not args.execute or not args.close_claw:
        print("SIX_D_CLOSE_BLOCKED: pass --execute --close-claw")
        return 2
    if not controller.rospy.has_param(VERIFIED_PARAM):
        print("SIX_D_CLOSE_BLOCKED: no successful wrist verification")
        return 2
    verified = controller.rospy.get_param(VERIFIED_PARAM)
    age = time.time() - float(verified.get("wall_time", 0.0))
    checks, _, _ = final_geometry_checks(controller, args)
    checks["verification_fresh"] = 0.0 <= age <= float(args.verification_max_age)
    print("Close checks: {}".format(checks))
    if not all(checks.values()):
        print("SIX_D_CLOSE_BLOCKED: verification or geometry is stale")
        return 2
    controller.rospy.wait_for_service(
        "/control_robot_leju_claw", timeout=float(args.timeout)
    )
    proxy = controller.rospy.ServiceProxy(
        "/control_robot_leju_claw", controlLejuClaw
    )
    request = controlLejuClawRequest()
    request.data.name = ["left_claw", "right_claw"]
    request.data.position = [90.0, 90.0]
    request.data.velocity = [50.0, 50.0]
    request.data.effort = [1.0, 1.0]
    response = proxy(request)
    if not getattr(response, "success", False):
        print("SIX_D_CLOSE_BLOCKED: {}".format(
            getattr(response, "message", "claw service rejected command")
        ))
        return 2
    controller.rospy.sleep(1.0)
    print("SIX_D_CLAW_CLOSED")
    return 0


def require_execution_confirmation(args):
    if args.execute and args.stage in (
            "preprocess", "align", "insert", "close", "all"):
        if args.confirmation != CONFIRMATION:
            raise RuntimeError(
                "execution blocked; pass --confirmation {}".format(
                    CONFIRMATION
                )
            )


def run_ros(args):
    require_execution_confirmation(args)
    controller = Scene36DRosController(args)

    if args.stage == "audit":
        state = controller.sample_state()
        checks = controller.print_audit(state)
        targets = grasp_targets(args, state)
        print("Current physical TCP: {}".format(
            np.round(state["geometry"]["tcp"], 4).tolist()
        ))
        print("Preprocessing TCP: {}".format(
            np.round(targets["preprocess_tcp"], 4).tolist()
        ))
        print("Final grasp TCP: {}".format(
            np.round(targets["final_tcp"], 4).tolist()
        ))
        if all(checks.values()):
            print("SIX_D_PIPELINE_AUDIT_OK")
            print("Read-only audit complete; no command was sent")
            return 0
        print("SIX_D_PIPELINE_AUDIT_BLOCKED")
        print("Read-only audit complete; no command was sent")
        return 2

    if args.stage == "preprocess":
        return run_preprocess(controller, args, args.execute)
    if args.stage == "align":
        return run_align(controller, args, args.execute)
    if args.stage == "insert":
        return run_insert(controller, args, args.execute)
    if args.stage == "verify":
        return run_wrist_verification(controller, args)
    if args.stage == "close":
        return run_close(controller, args)

    # Full workflow.  Without --execute it plans only the next preprocessing
    # segment; future stages depend on measured results and are not fabricated.
    state = controller.sample_state()
    checks = controller.print_audit(state)
    if not all(checks.values()):
        print("SIX_D_PIPELINE_BLOCKED: audit failed")
        return 2
    result = run_preprocess(controller, args, args.execute)
    if result != 0 or not args.execute:
        return result
    result = run_align(controller, args, True)
    if result != 0:
        return result
    result = run_insert(controller, args, True)
    if result != 0:
        return result
    result = run_wrist_verification(controller, args)
    if result != 0:
        return result
    if args.close_claw:
        return run_close(controller, args)
    print("SIX_D_PIPELINE_READY_TO_CLOSE")
    print("Wrist verification passed; rerun close stage with --close-claw")
    return 0


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
