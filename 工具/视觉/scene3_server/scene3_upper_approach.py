#!/usr/bin/env python3
"""鏍规嵁鍙俊瑙嗚缁撴灉瑙勫垝鎴栨墽琛屼竴娆?Scene3 鍚戝墠鐭剦鍐层€?""

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
    """璇诲彇妫€娴嬬粨鏋滐紱鏁伴噺鎴栫姸鎬佸紓甯告椂绂佹鐢熸垚杩愬姩璁″垝銆?""
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
    """闈犺繎闃舵鍙娇鐢ㄦ湭缁忕粡楠屽叕寮忎慨姝ｇ殑 base_link 鍧愭爣銆?""
    values = tray.get("base_link_xyz_raw_m", tray.get("base_link_xyz_m"))
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("tray does not contain a valid base_link XYZ")
    return [float(value) for value in values]


def build_plan(args, data, trays):
    # 鍥涢亾杩愬姩闂細鎬诲垎銆佹繁搴﹀舰鐘躲€丷GB-D 鍑犱綍鏉ユ簮銆佸畬鏁寸墿浣撴銆?    # 浠讳竴鏉′欢涓嶆弧瓒抽兘鐩存帴鎶ラ敊锛屼笉鍏佽浠呭嚟鈥滄娴嬪埌涓変釜妗嗏€濈户缁墠杩涖€?    for tray in trays:
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
    # 鐢ㄤ笁涓枡鐩樺墠鍚戣窛绂荤殑涓綅鏁颁及璁¤揣鏋惰窛绂伙紝闄嶄綆鍗曚釜娣卞害寮傚父鐨勫奖鍝嶃€?    shelf_distance = float(statistics.median(position[0] for position in tray_positions))
    if not 0.40 <= shelf_distance <= 2.50:
        raise ValueError(
            "implausible shelf distance {:.3f} m; refusing motion".format(
                shelf_distance
            )
        )

    # 杩欓噷鍙褰曟渶閫傚悎鍙虫墜鐨勫€欓€夛紝鐭剦鍐叉湰韬笉鍋氭í绉汇€佷几鑷傛垨澶圭埅鍔ㄤ綔銆?    target_index = min(
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
    """鍙彂甯冧竴娆″彈璺濈銆侀€熷害鍜屾椂闀块檺鍒剁殑鍓嶈繘鎸囦护銆?""
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
        # 鏃犺姝ｅ父缁撴潫杩樻槸涓€斿紓甯革紝閮借繛缁彂甯冮浂閫熷害锛岀‘淇濇満鍣ㄤ汉鍋滀綇銆?        for _ in range(12):
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

