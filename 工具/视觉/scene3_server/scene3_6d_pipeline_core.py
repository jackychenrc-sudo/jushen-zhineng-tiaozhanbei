#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure geometry and command-mapping helpers for the Scene3 6D pipeline.

The robot reports measured arm angles and accepts arm commands in two
different absolute zero conventions.  Absolute values must therefore never
be copied from ``/sensors_data_raw`` to ``/kuavo_arm_traj``.  The only safe
mapping is:

    next command = current /joint_cmd target + measured IK delta

The helpers in this module have no ROS dependency so that this rule and the
three-stage grasp geometry can be unit-tested on a development machine.
"""

from __future__ import print_function

import math

import numpy as np


ARM_START = 13
ARM_COUNT = 14
RIGHT_ARM_START = 7


def _finite_vector(values, length, name):
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != int(length):
        raise ValueError("{} must contain {} values".format(name, length))
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} contains non-finite values".format(name))
    return vector


def normalize(values, name="vector"):
    vector = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} contains non-finite values".format(name))
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("{} has zero length".format(name))
    return vector / norm


def extract_arm_vector(full_joint_vector, name="joint vector"):
    """Extract the V52 14-arm slice from a full 29-joint vector."""

    values = np.asarray(full_joint_vector, dtype=float).reshape(-1)
    if values.size < ARM_START + ARM_COUNT:
        raise ValueError("{} is too short: {}".format(name, values.size))
    arm = values[ARM_START:ARM_START + ARM_COUNT]
    if not np.all(np.isfinite(arm)):
        raise ValueError("{} arm slice contains non-finite values".format(name))
    return arm.copy()


def command_reference_degrees(low_level_arm_radians):
    """Convert the active low-level command target to trajectory degrees."""

    command = _finite_vector(
        low_level_arm_radians, ARM_COUNT, "low-level arm command"
    )
    return np.rad2deg(command)


def sensor_command_offsets_degrees(measured_arm_radians,
                                   low_level_arm_radians):
    """Return diagnostics only; offsets must never be used as a hard gate."""

    measured = _finite_vector(
        measured_arm_radians, ARM_COUNT, "measured arm"
    )
    command = _finite_vector(
        low_level_arm_radians, ARM_COUNT, "low-level arm command"
    )
    return np.rad2deg(command - measured)


def command_target_from_ik_delta(command_reference_deg,
                                 measured_arm_radians,
                                 solved_arm_radians,
                                 freeze_left=True):
    """Map an IK relative change onto the active command convention.

    ``measured_arm_radians`` and ``solved_arm_radians`` are in the model and
    sensor convention.  Their difference is convention-independent.  That
    difference is added to the active ``/joint_cmd`` target, which is already
    in the controller's command convention.
    """

    reference = _finite_vector(
        command_reference_deg, ARM_COUNT, "command reference"
    )
    measured = _finite_vector(
        measured_arm_radians, ARM_COUNT, "measured arm"
    )
    solved = _finite_vector(solved_arm_radians, ARM_COUNT, "IK solution")
    delta_deg = np.rad2deg(solved - measured)
    if freeze_left:
        delta_deg[:RIGHT_ARM_START] = 0.0
    return reference + delta_deg, delta_deg


def maximum_spread(samples):
    """Maximum Euclidean distance between any two equally sized samples."""

    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("at least one vector sample is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples contain non-finite values")
    maximum = 0.0
    for first in values:
        maximum = max(
            maximum,
            float(np.max(np.linalg.norm(values - first, axis=1))),
        )
    return maximum


def maximum_joint_spread_degrees(samples_radians):
    values = np.asarray(samples_radians, dtype=float)
    if values.ndim != 2 or values.shape[1] != ARM_COUNT:
        raise ValueError("14-joint samples are required")
    return float(np.max(np.ptp(np.rad2deg(values), axis=0)))


def compute_grasp_targets(tray_center_base,
                          tray_center_to_grasp_m=0.105,
                          preprocess_standoff_m=0.090,
                          grasp_height_offset_m=0.015):
    """Compute the safe turn point and final TCP point in ``base_link``.

    The approach direction is horizontal from the robot base towards the
    locked tray centre.  The final TCP is placed at the near tray edge and
    the preprocessing point is further back on the same insertion line.
    """

    tray = _finite_vector(tray_center_base, 3, "tray centre")
    approach = np.array([tray[0], tray[1], 0.0], dtype=float)
    approach = normalize(approach, "horizontal tray approach")
    final_tcp = tray - float(tray_center_to_grasp_m) * approach
    final_tcp[2] += float(grasp_height_offset_m)
    preprocess_tcp = (
        final_tcp - float(preprocess_standoff_m) * approach
    )
    return {
        "tray_center": tray,
        "approach": approach,
        "preprocess_tcp": preprocess_tcp,
        "final_tcp": final_tcp,
        "insert_distance_m": float(preprocess_standoff_m),
    }


def position_only_eef_step(current_eef_position,
                           current_physical_tcp,
                           target_physical_tcp,
                           maximum_step_m):
    """Move the EEF by one feedback step towards a physical TCP target.

    Orientation is intentionally not constrained in the preprocessing stage.
    Any orientation-induced TCP error is observed and corrected by the next
    feedback segment.
    """

    eef = _finite_vector(current_eef_position, 3, "current EEF position")
    tcp = _finite_vector(current_physical_tcp, 3, "current physical TCP")
    target = _finite_vector(target_physical_tcp, 3, "target physical TCP")
    error = target - tcp
    remaining = float(np.linalg.norm(error))
    if remaining < 1e-9:
        return {
            "target_eef_position": eef.copy(),
            "direction": np.zeros(3),
            "step_m": 0.0,
            "remaining_m": 0.0,
        }
    direction = error / remaining
    step = min(float(maximum_step_m), remaining)
    return {
        "target_eef_position": eef + step * direction,
        "direction": direction,
        "step_m": step,
        "remaining_m": remaining,
    }


def physical_to_eef_calibration(eef_position,
                                eef_rotation,
                                physical_tcp,
                                physical_rotation):
    """Calibrate the rigid physical-gripper transform from one live pose."""

    eef_position = _finite_vector(eef_position, 3, "EEF position")
    eef_rotation = np.asarray(eef_rotation, dtype=float).reshape(3, 3)
    physical_tcp = _finite_vector(physical_tcp, 3, "physical TCP")
    physical_rotation = np.asarray(
        physical_rotation, dtype=float
    ).reshape(3, 3)
    return {
        "eef_to_tcp": eef_rotation.T.dot(physical_tcp - eef_position),
        "eef_to_physical": eef_rotation.T.dot(physical_rotation),
    }


def eef_target_for_physical_pose(target_tcp,
                                 target_physical_rotation,
                                 eef_to_tcp,
                                 eef_to_physical):
    """Return the exact official EEF pose for a desired physical 6D pose."""

    target_tcp = _finite_vector(target_tcp, 3, "target physical TCP")
    target_physical_rotation = np.asarray(
        target_physical_rotation, dtype=float
    ).reshape(3, 3)
    eef_to_tcp = _finite_vector(eef_to_tcp, 3, "EEF-to-TCP offset")
    eef_to_physical = np.asarray(
        eef_to_physical, dtype=float
    ).reshape(3, 3)
    target_eef_rotation = target_physical_rotation.dot(
        eef_to_physical.T
    )
    target_eef_position = (
        target_tcp - target_eef_rotation.dot(eef_to_tcp)
    )
    return {
        "eef_position": target_eef_position,
        "eef_rotation": target_eef_rotation,
    }


def insertion_line_step(current_tcp,
                        preprocess_tcp,
                        final_tcp,
                        maximum_step_m):
    """Choose the next TCP target on the fixed straight insertion line."""

    current = _finite_vector(current_tcp, 3, "current TCP")
    start = _finite_vector(preprocess_tcp, 3, "preprocess TCP")
    final = _finite_vector(final_tcp, 3, "final TCP")
    line = final - start
    length = float(np.linalg.norm(line))
    direction = normalize(line, "insertion line")
    raw_progress = float(np.dot(current - start, direction))
    progress = max(0.0, min(length, raw_progress))
    next_progress = min(length, progress + float(maximum_step_m))
    target = start + next_progress * direction
    cross_track = float(np.linalg.norm(
        current - (start + progress * direction)
    ))
    return {
        "target_tcp": target,
        "direction": direction,
        "line_length_m": length,
        "progress_m": progress,
        "next_progress_m": next_progress,
        "remaining_m": max(0.0, length - progress),
        "cross_track_m": cross_track,
        "step_m": next_progress - progress,
    }


def cartesian_motion_metrics(before, after, target, direction):
    before = _finite_vector(before, 3, "before TCP")
    after = _finite_vector(after, 3, "after TCP")
    target = _finite_vector(target, 3, "target TCP")
    direction = normalize(direction, "motion direction")
    movement = after - before
    progress = float(np.dot(movement, direction))
    cross_track = float(np.linalg.norm(movement - progress * direction))
    return {
        "movement": movement,
        "progress_m": progress,
        "cross_track_m": cross_track,
        "motion_m": float(np.linalg.norm(movement)),
        "target_error_m": float(np.linalg.norm(target - after)),
    }


def quintic(progress):
    value = max(0.0, min(1.0, float(progress)))
    return 10.0 * value ** 3 - 15.0 * value ** 4 + 6.0 * value ** 5


def interpolate_commands(start_degrees, target_degrees, count):
    start = _finite_vector(start_degrees, ARM_COUNT, "start command")
    target = _finite_vector(target_degrees, ARM_COUNT, "target command")
    count = max(1, int(count))
    return [
        start + (target - start) * quintic(float(index) / float(count))
        for index in range(count + 1)
    ]

