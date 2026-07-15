#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the senior Scene3 flow, but stop immediately after its pregrasp.

The senior implementation remains the source of movement, perception, target
construction, IK and arm commands.  This wrapper replaces only
``grasp_tray_from_latest_target`` at runtime so the old touch, claw-close,
lift and retreat stages cannot execute before a visual TCP check.
"""

from __future__ import print_function

import copy
import importlib
import math
import os
import sys


CONFIRMATION = "SENIOR_PREGRASP_ONLY"
DEFAULT_SENIOR_DIR = "/root/kuavo_ws/src/challenge_cup_task_template/scripts"
LOCKED_TARGET_TOPIC = "/challenge_cup_task_template/scene3/locked_target_base"
LOCKED_TARGET_PARAM = "/challenge_cup_task_template/scene3/locked_target_base_xyz"


def _target_xyz(target_message):
    point = getattr(target_message, "point", None)
    values = [
        float(getattr(point, "x", float("nan"))),
        float(getattr(point, "y", float("nan"))),
        float(getattr(point, "z", float("nan"))),
    ]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("senior target is not a finite base_link point")
    return values


def lock_target(task, target_message, topic=LOCKED_TARGET_TOPIC,
                parameter=LOCKED_TARGET_PARAM):
    """Latch the exact target selected by the senior flow before arm motion."""
    header = getattr(target_message, "header", None)
    frame_id = str(getattr(header, "frame_id", "") or "")
    if frame_id and frame_id.lstrip("/") != "base_link":
        raise RuntimeError(
            "senior target must already be in base_link, got {}".format(frame_id)
        )

    locked_message = copy.deepcopy(target_message)
    locked_header = getattr(locked_message, "header", None)
    if locked_header is not None:
        locked_header.frame_id = "base_link"
        if hasattr(task.rospy, "Time"):
            locked_header.stamp = task.rospy.Time.now()

    publisher = task.rospy.Publisher(
        str(topic), target_message.__class__, queue_size=1, latch=True
    )
    publisher.publish(locked_message)
    xyz = _target_xyz(locked_message)
    if hasattr(task.rospy, "set_param"):
        task.rospy.set_param(str(parameter), xyz)

    # Keep the latched publisher alive for as long as the senior task exists.
    task._scene3_locked_target_publisher = publisher
    task._scene3_locked_target_message = locked_message
    task._scene3_locked_target_xyz = xyz
    task.rospy.loginfo(
        "SENIOR_TARGET_LOCKED base_link=%s topic=%s",
        [round(value, 4) for value in xyz],
        topic,
    )
    return xyz


def run_pregrasp_only(task, target_timeout=5.0, motion_seconds=2.5,
                      locked_target_topic=LOCKED_TARGET_TOPIC,
                      locked_target_param=LOCKED_TARGET_PARAM):
    """Execute exactly one senior pregrasp and return its Cartesian target."""
    target_message = task.wait_for_recent_base_target(
        timeout=float(target_timeout),
        freshness=max(float(task.target_freshness), 1.0),
    )
    if target_message is None:
        raise RuntimeError("no fresh Scene3 target for senior pregrasp")

    lock_target(
        task,
        target_message,
        topic=locked_target_topic,
        parameter=locked_target_param,
    )
    targets = task.build_scene3_grasp_targets(target_message)
    pregrasp = [float(value) for value in targets["pregrasp"]]

    task.stop_base()
    if not task.open_claw():
        raise RuntimeError("cannot confirm open claw before senior pregrasp")
    task.move_right_hand(pregrasp, duration=float(motion_seconds))
    task.stop_base()
    return pregrasp


def install_pregrasp_gate(senior_module, target_timeout=5.0, motion_seconds=2.5,
                          locked_target_topic=LOCKED_TARGET_TOPIC,
                          locked_target_param=LOCKED_TARGET_PARAM,
                          hold_after_pregrasp=True):
    """Patch one stage boundary without changing the senior source file."""
    def gated_grasp(task):
        pregrasp = run_pregrasp_only(
            task,
            target_timeout=target_timeout,
            motion_seconds=motion_seconds,
            locked_target_topic=locked_target_topic,
            locked_target_param=locked_target_param,
        )
        task.rospy.loginfo(
            "SENIOR_PREGRASP_READY pregrasp=%s locked_target=%s; "
            "old touch/claw/lift blocked",
            [round(value, 4) for value in pregrasp],
            [round(value, 4) for value in task._scene3_locked_target_xyz],
        )
        if hold_after_pregrasp:
            task.rospy.loginfo(
                "SENIOR_PREGRASP_HOLDING: simulation remains online for "
                "wrist-camera validation; press Ctrl-C only after the "
                "second-terminal check"
            )
            # Blocking here is intentional: returning to the senior launcher
            # immediately tears down MuJoCo and leaves only stale TF data.
            task.rospy.spin()
        return False

    senior_module.Scene3Task.grasp_tray_from_latest_target = gated_grasp
    return gated_grasp


def load_senior_module(senior_dir=DEFAULT_SENIOR_DIR):
    senior_dir = os.path.abspath(senior_dir)
    source_file = os.path.join(senior_dir, "challenge_task_3.py")
    if not os.path.isfile(source_file):
        raise RuntimeError("senior challenge_task_3.py not found: {}".format(source_file))
    if senior_dir in sys.path:
        sys.path.remove(senior_dir)
    sys.path.insert(0, senior_dir)
    return importlib.import_module("challenge_task_3")


def main():
    if os.environ.get("SCENE3_PREGRASP_CONFIRMATION", "") != CONFIRMATION:
        raise RuntimeError(
            "pregrasp blocked; export SCENE3_PREGRASP_CONFIRMATION={}".format(
                CONFIRMATION
            )
        )
    senior_dir = os.environ.get("SCENE3_SENIOR_DIR", DEFAULT_SENIOR_DIR)
    target_timeout = float(os.environ.get("SCENE3_PREGRASP_TARGET_TIMEOUT", "5.0"))
    motion_seconds = float(os.environ.get("SCENE3_PREGRASP_MOTION_SECONDS", "2.5"))
    locked_target_topic = os.environ.get(
        "SCENE3_LOCKED_TARGET_TOPIC", LOCKED_TARGET_TOPIC
    )
    locked_target_param = os.environ.get(
        "SCENE3_LOCKED_TARGET_PARAM", LOCKED_TARGET_PARAM
    )
    hold_after_pregrasp = os.environ.get(
        "SCENE3_HOLD_AFTER_PREGRASP", "1"
    ).strip().lower() not in ("0", "false", "no", "off")
    senior_module = load_senior_module(senior_dir)
    install_pregrasp_gate(
        senior_module,
        target_timeout=target_timeout,
        motion_seconds=motion_seconds,
        locked_target_topic=locked_target_topic,
        locked_target_param=locked_target_param,
        hold_after_pregrasp=hold_after_pregrasp,
    )
    return senior_module.main()


if __name__ == "__main__":
    raise SystemExit(main())
