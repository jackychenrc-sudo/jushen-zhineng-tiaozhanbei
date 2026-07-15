#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One small senior-IK approach step before wrist secondary recognition.

At the senior pregrasp pose the target tray can be too small or too close to
the edge of the right-wrist image for the existing YOLO weights.  This helper
uses the senior ``Scene3Task`` FK/IK/control methods unchanged and advances the
current right-hand target by at most 5 cm along ``base_link`` +X, which is the
same direction as the senior pregrasp-to-touch segment.

The base remains stopped and the claw remains open.  There is no close, lift,
retreat or retry.  Default mode is IK calculation only; real arm movement
requires both ``--execute`` and the exact confirmation token
``SENIOR_BOOTSTRAP_5CM``.
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np


EXECUTION_CONFIRMATION = "SENIOR_BOOTSTRAP_5CM"
DEFAULT_SENIOR_DIR = "/root/kuavo_ws/src/challenge_cup_task_template/scripts"


def plan_forward_step(current_xyz, step_m=0.05,
                      x_bounds=(0.30, 0.60),
                      y_bounds=(-0.50, 0.20),
                      z_bounds=(0.05, 0.55)):
    current = np.asarray(current_xyz, dtype=float)
    if current.shape != (3,) or not np.all(np.isfinite(current)):
        raise ValueError("current right-hand pose must be finite xyz")
    step = min(0.05, max(0.01, float(step_m)))
    bounds = (x_bounds, y_bounds, z_bounds)
    for axis, (lower, upper) in enumerate(bounds):
        if not float(lower) <= current[axis] <= float(upper):
            raise ValueError(
                "current hand axis {} is outside bootstrap workspace".format(axis)
            )
    target = current + np.array([step, 0.0, 0.0], dtype=float)
    if target[0] > float(x_bounds[1]):
        raise ValueError("bootstrap target exceeds forward workspace gate")
    return target, step


def validate_forward_motion(before_xyz, after_xyz, target_xyz,
                            minimum_forward_m=0.015,
                            maximum_motion_m=0.080,
                            maximum_side_motion_m=0.035,
                            maximum_target_error_m=0.050):
    before = np.asarray(before_xyz, dtype=float)
    after = np.asarray(after_xyz, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    if any(value.shape != (3,) for value in (before, after, target)):
        raise ValueError("motion validation inputs must be xyz")
    delta = after - before
    motion = float(np.linalg.norm(delta))
    target_error = float(np.linalg.norm(target - after))
    checks = {
        "forward_progress": float(delta[0]) >= float(minimum_forward_m),
        "motion_bounded": motion <= float(maximum_motion_m),
        "lateral_bounded": abs(float(delta[1])) <= float(maximum_side_motion_m),
        "vertical_bounded": abs(float(delta[2])) <= float(maximum_side_motion_m),
        "target_error_bounded": target_error <= float(maximum_target_error_m),
    }
    return bool(all(checks.values())), checks, delta, motion, target_error


def load_senior_task(senior_dir):
    senior_dir = os.path.abspath(senior_dir)
    source_file = os.path.join(senior_dir, "challenge_task_3.py")
    if not os.path.isfile(source_file):
        raise RuntimeError("senior challenge_task_3.py not found: {}".format(
            source_file
        ))
    if senior_dir in sys.path:
        sys.path.remove(senior_dir)
    sys.path.insert(0, senior_dir)
    from challenge_task_3 import RIGHT_GRIPPER_QUAT_XYZW, Scene3Task

    return Scene3Task, RIGHT_GRIPPER_QUAT_XYZW


def current_right_hand(task):
    joints = task.read_current_arm_joints()
    poses = task.call_fk(joints)
    return np.asarray(poses.right_pose.pos_xyz, dtype=float)


def run_ros(args):
    import rospy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import JointState

    rospy.init_node("scene3_senior_bootstrap_approach", anonymous=True)
    Scene3Task, right_quaternion = load_senior_task(args.senior_dir)
    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    task = Scene3Task(cmd_vel_pub, arm_traj_pub)

    before = current_right_hand(task)
    target, command_step = plan_forward_step(
        before,
        step_m=args.step,
        x_bounds=(args.minimum_x, args.maximum_x),
        y_bounds=(args.minimum_y, args.maximum_y),
        z_bounds=(args.minimum_z, args.maximum_z),
    )
    print("Current senior FK right hand:", np.round(before, 4).tolist())
    print("Bootstrap target:", np.round(target, 4).tolist())
    print("Commanded forward step: {:.4f}m".format(command_step))
    print("Checking senior IK only")
    solution = task.solve_right_hand_ik(target.tolist(), right_quaternion)
    if len(solution) != 14:
        raise RuntimeError("senior IK did not return fourteen arm joints")
    print("SENIOR_BOOTSTRAP_IK_OK: no command sent yet")

    if not args.execute:
        print("SENIOR_BOOTSTRAP_DRY_RUN_OK: calculation only; claw remains open")
        return 0
    if args.confirmation != EXECUTION_CONFIRMATION:
        raise RuntimeError(
            "execution blocked; pass --confirmation {}".format(
                EXECUTION_CONFIRMATION
            )
        )

    task.stop_base()
    task.wait_for_arm_subscriber(timeout=8.0)
    if not task.set_arm_mode(2):
        raise RuntimeError("cannot enable senior arm external-control mode")
    if not task.open_claw():
        raise RuntimeError("cannot confirm open claw before bootstrap step")
    print("Executing one 5 cm maximum senior-IK approach; claw stays open")
    task.move_right_hand(target.tolist(), duration=args.motion_seconds)
    task.stop_base()
    rospy.sleep(args.settle_seconds)

    after = current_right_hand(task)
    ok, checks, delta, motion, target_error = validate_forward_motion(
        before,
        after,
        target,
        minimum_forward_m=args.minimum_forward_motion,
        maximum_motion_m=args.maximum_observed_motion,
        maximum_side_motion_m=args.maximum_side_motion,
        maximum_target_error_m=args.maximum_target_error,
    )
    print("Actual senior FK right hand:", np.round(after, 4).tolist())
    print("Observed delta:", np.round(delta, 4).tolist())
    print("Observed motion: {:.4f}m".format(motion))
    print("Target error: {:.4f}m".format(target_error))
    print("Safety checks:", checks)
    if not ok:
        raise RuntimeError(
            "SENIOR_BOOTSTRAP_STEP_BLOCKED: arm response failed safety checks; "
            "claw remains open"
        )
    print(
        "SENIOR_BOOTSTRAP_STEP_OK: one bounded approach completed; "
        "run wrist YOLO next; claw remains open"
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--senior-dir", default=DEFAULT_SENIOR_DIR)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--minimum-x", type=float, default=0.30)
    parser.add_argument("--maximum-x", type=float, default=0.60)
    parser.add_argument("--minimum-y", type=float, default=-0.50)
    parser.add_argument("--maximum-y", type=float, default=0.20)
    parser.add_argument("--minimum-z", type=float, default=0.05)
    parser.add_argument("--maximum-z", type=float, default=0.55)
    parser.add_argument("--motion-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--minimum-forward-motion", type=float, default=0.015)
    parser.add_argument("--maximum-observed-motion", type=float, default=0.080)
    parser.add_argument("--maximum-side-motion", type=float, default=0.035)
    parser.add_argument("--maximum-target-error", type=float, default=0.050)
    return parser


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
