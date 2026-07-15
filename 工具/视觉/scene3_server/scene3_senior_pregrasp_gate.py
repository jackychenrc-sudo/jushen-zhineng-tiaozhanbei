#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the senior Scene3 flow, but stop immediately after its pregrasp.

The senior implementation remains the source of movement, perception, target
construction, IK and arm commands.  This wrapper replaces only
``grasp_tray_from_latest_target`` at runtime so the old touch, claw-close,
lift and retreat stages cannot execute before a visual TCP check.
"""

from __future__ import print_function

import importlib
import os
import sys


CONFIRMATION = "SENIOR_PREGRASP_ONLY"
DEFAULT_SENIOR_DIR = "/root/kuavo_ws/src/challenge_cup_task_template/scripts"


def run_pregrasp_only(task, target_timeout=5.0, motion_seconds=2.5):
    """Execute exactly one senior pregrasp and return its Cartesian target."""
    target_message = task.wait_for_recent_base_target(
        timeout=float(target_timeout),
        freshness=max(float(task.target_freshness), 1.0),
    )
    if target_message is None:
        raise RuntimeError("no fresh Scene3 target for senior pregrasp")

    targets = task.build_scene3_grasp_targets(target_message)
    pregrasp = [float(value) for value in targets["pregrasp"]]

    task.stop_base()
    if not task.open_claw():
        raise RuntimeError("cannot confirm open claw before senior pregrasp")
    task.move_right_hand(pregrasp, duration=float(motion_seconds))
    task.stop_base()
    return pregrasp


def install_pregrasp_gate(senior_module, target_timeout=5.0, motion_seconds=2.5):
    """Patch one stage boundary without changing the senior source file."""
    def gated_grasp(task):
        pregrasp = run_pregrasp_only(
            task,
            target_timeout=target_timeout,
            motion_seconds=motion_seconds,
        )
        task.rospy.loginfo(
            "SENIOR_PREGRASP_READY target=%s; old touch/claw/lift blocked",
            [round(value, 4) for value in pregrasp],
        )
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
    senior_module = load_senior_module(senior_dir)
    install_pregrasp_gate(
        senior_module,
        target_timeout=target_timeout,
        motion_seconds=motion_seconds,
    )
    return senior_module.main()


if __name__ == "__main__":
    raise SystemExit(main())

