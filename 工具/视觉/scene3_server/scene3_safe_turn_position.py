#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retired compatibility entry point for the old Scene3 position mover.

The previous implementation could copy absolute measured joint angles into
the command convention.  The real robot/controller uses different absolute
zeros, so that handover could move the arm even when no Cartesian movement
was requested.  It is intentionally fail-closed.

Use ``scene3_6d_grasp_pipeline.py --stage preprocess`` instead.  The new
pipeline always starts from the live ``/joint_cmd`` target and adds only the
relative IK delta.
"""

from __future__ import print_function

import argparse


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser


def run_ros(_args):
    print("SAFE_TURN_POSITION_RETIRED")
    print("Run scene3_6d_grasp_pipeline.py --stage preprocess instead")
    print("No base, arm or claw command was sent")
    return 2


def main(argv=None):
    return run_ros(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
