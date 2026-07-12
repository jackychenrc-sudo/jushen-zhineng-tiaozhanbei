#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection-json", required=True)
    parser.add_argument("--desired-shelf-distance", type=float, default=0.90)
    parser.add_argument("--preferred-right-hand-y", type=float, default=-0.28)
    parser.add_argument("--minimum-selection-score", type=float, default=0.60)
    parser.add_argument("--minimum-depth-shape-score", type=float, default=0.75)
    parser.add_argument("--max-forward-pulse", type=float, default=0.08)
    parser.add_argument("--forward-speed", type=float, default=0.05)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    return parser.parse_args()


def load_detection(path):
    detection_path = Path(path)
    if not detection_path.is_file():
        raise ValueError("detection JSON does not exist: {}".format(path))
    data = json.loads(detection_path.read_text(encoding="utf-8"))
    trays = data.get("upper_trays", [])
    if data.get("status") != "ok" or len(trays) != 3:
        raise ValueError(
            "upper-tray detection is not trusted: status={} count={}".format(
                data.get("status"), len(trays)
            )
        )
    return data, trays


def raw_xyz(tray):
    values = tray.get("base_link_xyz_raw_m", tray.get("base_link_xyz_m"))
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("tray does not contain a valid base_link XYZ")
    return [float(value) for value in values]


def build_plan(args, data, trays):
    for tray in trays:
        score = float(tray.get("selection_score", 0.0))
        if score < args.minimum_selection_score:
            raise ValueError(
                "tray {} selection score {:.3f} is below {:.3f}".format(
                    tray.get("id", "unknown"), score, args.minimum_selection_score
                )
            )
        depth_shape_score = float(tray.get("depth_shape_score", 0.0))
        if depth_shape_score < args.minimum_depth_shape_score:
            raise ValueError(
                "tray {} depth shape score {:.3f} is below {:.3f}".format(
                    tray.get("id", "unknown"),
                    depth_shape_score,
                    args.minimum_depth_shape_score,
                )
            )
        if tray.get("geometry_source") != "rgbd_vertical":
            raise ValueError(
                "tray {} does not have trusted RGB-D object geometry".format(
                    tray.get("id", "unknown")
                )
            )
        if not tray.get("object_bbox"):
            raise ValueError(
                "tray {} does not contain an object bbox".format(
                    tray.get("id", "unknown")
                )
            )

    tray_positions = [raw_xyz(tray) for tray in trays]
    shelf_distance = float(statistics.median(position[0] for position in tray_positions))
    if not 0.40 <= shelf_distance <= 2.50:
        raise ValueError(
            "implausible shelf distance {:.3f} m; refusing motion".format(
                shelf_distance
            )
        )

    target_index = min(
        range(len(trays)),
        key=lambda index: (
            abs(tray_positions[index][1] - args.preferred_right_hand_y),
            -float(trays[index].get("selection_score", 0.0)),
        ),
    )
    target = trays[target_index]
    target_position = tray_positions[target_index]
    remaining_forward = shelf_distance - args.desired_shelf_distance
    forward_pulse = max(0.0, min(args.max_forward_pulse, remaining_forward))
    duration = (
        forward_pulse / args.forward_speed
        if args.forward_speed > 0.0 and forward_pulse > 0.0
        else 0.0
    )

    return {
        "mode": "execute" if args.execute else "dry_run",
        "source_algorithm": data.get("algorithm"),
        "shelf_distance_m": shelf_distance,
        "desired_shelf_distance_m": args.desired_shelf_distance,
        "remaining_forward_m": remaining_forward,
        "planned_forward_pulse_m": forward_pulse,
        "forward_speed_mps": args.forward_speed,
        "planned_duration_s": duration,
        "selected_target": {
            "id": target.get("id"),
            "center_pixel": target.get("center_pixel"),
            "selection_score": target.get("selection_score"),
            "base_link_xyz_raw_m": target_position,
            "preferred_right_hand_y_m": args.preferred_right_hand_y,
        },
        "safety": {
            "arm_commanded": False,
            "gripper_commanded": False,
            "lateral_motion_commanded": False,
            "maximum_single_pulse_m": args.max_forward_pulse,
        },
    }


def execute_forward_pulse(args, plan):
    if args.confirmation != "FORWARD_ONLY":
        raise ValueError(
            "execution requires --confirmation FORWARD_ONLY"
        )
    distance = float(plan["planned_forward_pulse_m"])
    duration = float(plan["planned_duration_s"])
    if distance <= 0.0 or duration <= 0.0:
        plan["execution"] = "no_motion_needed"
        return
    if distance > args.max_forward_pulse + 1e-9 or duration > 4.0:
        raise ValueError("planned pulse exceeds safety limit")

    import rospy
    from geometry_msgs.msg import Twist

    rospy.init_node("scene3_upper_approach", anonymous=True)
    publisher = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=10)
    deadline = time.monotonic() + 5.0
    while publisher.get_num_connections() == 0 and time.monotonic() < deadline:
        rospy.sleep(0.05)
    if publisher.get_num_connections() == 0:
        raise RuntimeError("no subscriber connected to {}".format(args.cmd_vel_topic))

    stop = Twist()
    forward = Twist()
    forward.linear.x = float(args.forward_speed)
    rate = rospy.Rate(20)
    try:
        for _ in range(5):
            publisher.publish(stop)
            rate.sleep()
        started = time.monotonic()
        while not rospy.is_shutdown() and time.monotonic() - started < duration:
            publisher.publish(forward)
            rate.sleep()
    finally:
        for _ in range(12):
            publisher.publish(stop)
            rate.sleep()
    plan["execution"] = "forward_pulse_completed"


def main():
    args = parse_args()
    if args.desired_shelf_distance < 0.65:
        raise ValueError("desired shelf distance below 0.65 m is not allowed")
    if not 0.01 <= args.max_forward_pulse <= 0.08:
        raise ValueError("max forward pulse must be within [0.01, 0.08] m")
    if not 0.02 <= args.forward_speed <= 0.06:
        raise ValueError("forward speed must be within [0.02, 0.06] m/s")

    data, trays = load_detection(args.detection_json)
    plan = build_plan(args, data, trays)
    if args.execute:
        execute_forward_pulse(args, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
