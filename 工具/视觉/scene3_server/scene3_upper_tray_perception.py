#!/usr/bin/env python3
"""Detect Scene3 upper SMT trays and report their base_link coordinates."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String

from scene3_live_perception import decode_depth, decode_rgb, project_pixel


def parse_roi(text):
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 4:
        raise ValueError("--search-roi must be x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("invalid normalized search ROI")
    return values


def find_matches(image, template, roi, threshold, nms_pixels, max_matches):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    template_height, template_width = template_gray.shape
    x1, y1, x2, y2 = (
        int(width * roi[0]),
        int(height * roi[1]),
        int(width * roi[2]),
        int(height * roi[3]),
    )
    search = gray[y1:y2, x1:x2]
    if search.shape[0] < template_height or search.shape[1] < template_width:
        raise RuntimeError("template is larger than the upper-shelf search ROI")

    response = cv2.matchTemplate(search, template_gray, cv2.TM_CCOEFF_NORMED)
    peak_mask = response == cv2.dilate(response, np.ones((11, 11), np.uint8))
    peak_y, peak_x = np.where(peak_mask & (response >= threshold))
    raw = sorted(
        [(float(response[y, x]), int(x), int(y)) for x, y in zip(peak_x, peak_y)],
        reverse=True,
    )

    selected = []
    for score, x, y in raw:
        center_x = int(x1 + x + template_width // 2)
        center_y = int(y1 + y + template_height // 2)
        if any(abs(center_x - item["center_pixel"][0]) < nms_pixels for item in selected):
            continue
        selected.append(
            {
                "score": score,
                "bbox": [
                    int(x1 + x),
                    int(y1 + y),
                    int(x1 + x + template_width),
                    int(y1 + y + template_height),
                ],
                "center_pixel": [center_x, center_y],
            }
        )
        if len(selected) >= max_matches:
            break
    return selected


def percentile_depth(depth_image, bbox, percentile):
    x1, y1, x2, y2 = bbox
    values = depth_image[y1:y2, x1:x2]
    values = values[np.isfinite(values) & (values > 0.05) & (values < 5.0)]
    if values.size == 0:
        raise RuntimeError("no valid depth in tray bbox {}".format(bbox))
    return float(np.percentile(values, percentile)), int(values.size)


def rotation_matrix(quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output-dir", default="/tmp/scene3_upper_trays")
    parser.add_argument("--search-roi", default="0.30,0.25,0.70,0.43")
    parser.add_argument("--match-threshold", type=float, default=0.25)
    parser.add_argument("--minimum-score", type=float, default=0.32)
    parser.add_argument("--anchor-score", type=float, default=0.60)
    parser.add_argument("--plane-x-tolerance", type=float, default=0.06)
    parser.add_argument("--plane-z-tolerance", type=float, default=0.04)
    parser.add_argument("--expected-count", type=int, default=3)
    parser.add_argument("--depth-percentile", type=float, default=10.0)
    parser.add_argument("--nms-pixels", type=int, default=20)
    parser.add_argument("--max-matches", type=int, default=20)
    parser.add_argument("--target-frame", default="base_link")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--rgb-topic", default="/cam_h/color/image_raw/compressed")
    parser.add_argument("--depth-topic", default="/cam_h/depth/image_raw/compressedDepth")
    parser.add_argument("--camera-info-topic", default="/cam_h/color/camera_info")
    parser.add_argument("--publish-topic", default="/scene3/upper_trays")
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template = cv2.imread(args.template)
    if template is None:
        raise RuntimeError("failed to read template: {}".format(args.template))

    rospy.init_node("scene3_upper_tray_perception", anonymous=True)
    camera_info = rospy.wait_for_message(args.camera_info_topic, CameraInfo, timeout=args.timeout)
    rgb_message = rospy.wait_for_message(args.rgb_topic, CompressedImage, timeout=args.timeout)
    depth_message = rospy.wait_for_message(args.depth_topic, CompressedImage, timeout=args.timeout)
    image = decode_rgb(rgb_message)
    depth_image = decode_depth(depth_message)
    if image.shape[:2] != depth_image.shape[:2]:
        raise RuntimeError("RGB and depth dimensions do not match")

    matches = find_matches(
        image,
        template,
        parse_roi(args.search_roi),
        args.match_threshold,
        args.nms_pixels,
        args.max_matches,
    )
    if not matches:
        raise RuntimeError("no upper tray matched the template")

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(1.0)
    transform = tf_buffer.lookup_transform(
        args.target_frame,
        camera_info.header.frame_id,
        rospy.Time(0),
        rospy.Duration(3.0),
    )
    rotation = rotation_matrix(transform.transform.rotation)
    translation = np.array(
        [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ],
        dtype=np.float64,
    )

    candidates = []
    for rank_by_score, match in enumerate(matches):
        depth_m, valid_pixels = percentile_depth(
            depth_image, match["bbox"], args.depth_percentile
        )
        camera_xyz = np.array(project_pixel(match["center_pixel"], depth_m, camera_info))
        target_xyz = rotation.dot(camera_xyz) + translation
        candidates.append(
            {
                "candidate_id": "candidate_{}".format(rank_by_score),
                "rank_by_score": rank_by_score,
                "score": match["score"],
                "bbox": match["bbox"],
                "center_pixel": match["center_pixel"],
                "depth_m": depth_m,
                "depth_percentile": args.depth_percentile,
                "valid_depth_pixels": valid_pixels,
                "camera_xyz_m": camera_xyz.tolist(),
                "base_link_xyz_m": target_xyz.tolist(),
            }
        )

    anchors = [item for item in candidates if item["score"] >= args.anchor_score]
    if len(anchors) < 2:
        anchors = sorted(candidates, key=lambda item: item["score"], reverse=True)[:2]
    plane_x = float(np.median([item["base_link_xyz_m"][0] for item in anchors]))
    plane_z = float(np.median([item["base_link_xyz_m"][2] for item in anchors]))

    accepted = []
    for candidate in candidates:
        candidate["plane_dx_m"] = abs(candidate["base_link_xyz_m"][0] - plane_x)
        candidate["plane_dz_m"] = abs(candidate["base_link_xyz_m"][2] - plane_z)
        candidate["selection_score"] = candidate["score"] - 0.10 * (
            candidate["plane_dx_m"] / args.plane_x_tolerance
            + candidate["plane_dz_m"] / args.plane_z_tolerance
        )
        candidate["accepted"] = bool(
            candidate["score"] >= args.minimum_score
            and candidate["plane_dx_m"] <= args.plane_x_tolerance
            and candidate["plane_dz_m"] <= args.plane_z_tolerance
        )
        if candidate["accepted"]:
            accepted.append(candidate)

    if len(accepted) > args.expected_count:
        keep = {
            item["candidate_id"]
            for item in sorted(
                accepted, key=lambda item: item["selection_score"], reverse=True
            )[: args.expected_count]
        }
        for candidate in candidates:
            candidate["accepted"] = candidate["candidate_id"] in keep
        accepted = [item for item in candidates if item["accepted"]]

    trays = []
    for order_x, candidate in enumerate(
        sorted(accepted, key=lambda item: item["center_pixel"][0])
    ):
        tray = dict(candidate)
        tray["id"] = "upper_x{}".format(order_x)
        trays.append(tray)

    candidate_visualization = image.copy()
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["bbox"]
        color = (0, 255, 0) if candidate["accepted"] else (0, 255, 255)
        thickness = 3 if candidate["accepted"] else 1
        cv2.rectangle(candidate_visualization, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            candidate_visualization,
            "{} s={:.2f} dx={:.2f} dz={:.2f}".format(
                candidate["candidate_id"],
                candidate["score"],
                candidate["plane_dx_m"],
                candidate["plane_dz_m"],
            ),
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    visualization = image.copy()
    for tray in trays:
        x1, y1, x2, y2 = tray["bbox"]
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            visualization,
            "{} s={:.2f} z={:.3f}m".format(tray["id"], tray["score"], tray["depth_m"]),
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    status = "ok" if len(trays) == args.expected_count else "count_mismatch"
    payload = {
        "status": status,
        "source_frame": camera_info.header.frame_id,
        "target_frame": args.target_frame,
        "search_roi": parse_roi(args.search_roi),
        "match_threshold": args.match_threshold,
        "minimum_score": args.minimum_score,
        "anchor_score": args.anchor_score,
        "expected_count": args.expected_count,
        "plane_filter": {
            "reference_base_x_m": plane_x,
            "reference_base_z_m": plane_z,
            "x_tolerance_m": args.plane_x_tolerance,
            "z_tolerance_m": args.plane_z_tolerance,
        },
        "depth_method": "bbox_percentile",
        "upper_trays": trays,
        "candidates": candidates,
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    (output_dir / "upper_trays.json").write_text(json_text, encoding="utf-8")
    cv2.imwrite(str(output_dir / "upper_trays.jpg"), visualization)
    cv2.imwrite(str(output_dir / "upper_candidates.jpg"), candidate_visualization)

    publisher = rospy.Publisher(args.publish_topic, String, queue_size=1, latch=True)
    rospy.sleep(0.3)
    publisher.publish(String(data=json_text))
    rospy.sleep(0.3)
    print(json_text)
    print("saved_image={}".format(output_dir / "upper_trays.jpg"))
    print("saved_candidates={}".format(output_dir / "upper_candidates.jpg"))
    print("saved_json={}".format(output_dir / "upper_trays.json"))


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSException, RuntimeError, ValueError, tf2_ros.TransformException) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
