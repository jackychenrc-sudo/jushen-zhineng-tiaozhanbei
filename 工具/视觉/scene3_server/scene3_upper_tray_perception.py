#!/usr/bin/env python3
"""Detect randomly placed Scene3 upper SMT trays with RGB-D geometry."""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String

from scene3_live_perception import decode_depth, decode_rgb, project_pixel


def parse_csv_floats(text, expected=None):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if expected is not None and len(values) != expected:
        raise ValueError("expected {} comma-separated values".format(expected))
    return values


def parse_roi(text):
    values = parse_csv_floats(text, expected=4)
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("invalid normalized search ROI")
    return values


def pixel_roi(shape, roi):
    height, width = shape[:2]
    return (
        int(width * roi[0]),
        int(height * roi[1]),
        int(width * roi[2]),
        int(height * roi[3]),
    )


def bbox_iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    if intersection <= 0:
        return 0.0
    first_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    second_area = max(1, (bx2 - bx1) * (by2 - by1))
    return float(intersection) / float(first_area + second_area - intersection)


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) // 2, (y1 + y2) // 2]


def suppress_matches(matches, nms_pixels, max_matches):
    selected = []
    for match in sorted(matches, key=lambda item: item["score"], reverse=True):
        center_x, center_y = match["center_pixel"]
        duplicate = False
        for existing in selected:
            other_x, other_y = existing["center_pixel"]
            center_distance = math.hypot(center_x - other_x, center_y - other_y)
            if center_distance < nms_pixels or bbox_iou(
                match["bbox"], existing["bbox"]
            ) > 0.30:
                duplicate = True
                best_score = max(
                    existing.get("score", 0.0), match.get("score", 0.0)
                )
                combined_sources = sorted(
                    set(existing.get("proposal_sources", []))
                    | set(match.get("proposal_sources", []))
                )
                existing_depth_score = existing.get("depth_shape_score", 0.0)
                incoming_depth_score = match.get("depth_shape_score", 0.0)
                best_depth = match if incoming_depth_score > existing_depth_score else existing
                best_depth_fields = {}
                for key in (
                    "depth_bbox",
                    "depth_center_pixel",
                    "depth_override_m",
                ):
                    if key in best_depth:
                        value = best_depth[key]
                        best_depth_fields[key] = list(value) if isinstance(value, list) else value
                if match.get("template_score", 0.0) > existing.get(
                    "template_score", 0.0
                ):
                    existing.update(match)
                existing["score"] = best_score
                existing["proposal_sources"] = combined_sources
                existing["depth_shape_score"] = max(
                    existing_depth_score, incoming_depth_score
                )
                existing.update(best_depth_fields)
                break
        if not duplicate:
            selected.append(dict(match))
        if len(selected) >= max_matches:
            break
    return selected


def multiscale_template_matches(
    image,
    template,
    roi,
    threshold,
    width_scales,
    height_scales,
    edge_weight,
    max_matches,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    template_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(
        template_gray
    )
    image_edges = cv2.Canny(gray, 40, 120)
    x1, y1, x2, y2 = pixel_roi(gray.shape, roi)
    search_gray = gray[y1:y2, x1:x2]
    search_edges = image_edges[y1:y2, x1:x2]
    base_height, base_width = template_gray.shape
    raw = []

    for width_scale in width_scales:
        for height_scale in height_scales:
            template_width = max(5, int(round(base_width * width_scale)))
            template_height = max(12, int(round(base_height * height_scale)))
            if (
                template_height >= search_gray.shape[0]
                or template_width >= search_gray.shape[1]
            ):
                continue
            interpolation = (
                cv2.INTER_AREA
                if width_scale <= 1.0 and height_scale <= 1.0
                else cv2.INTER_CUBIC
            )
            scaled_gray = cv2.resize(
                template_gray,
                (template_width, template_height),
                interpolation=interpolation,
            )
            scaled_edges = cv2.Canny(scaled_gray, 40, 120)
            gray_response = cv2.matchTemplate(
                search_gray, scaled_gray, cv2.TM_CCOEFF_NORMED
            )
            edge_response = cv2.matchTemplate(
                search_edges, scaled_edges, cv2.TM_CCOEFF_NORMED
            )
            response = (1.0 - edge_weight) * gray_response + edge_weight * edge_response
            peak_mask = response == cv2.dilate(
                response, np.ones((9, 9), dtype=np.uint8)
            )
            peak_y, peak_x = np.where(peak_mask & (response >= threshold))
            for local_x, local_y in zip(peak_x, peak_y):
                score = float(response[local_y, local_x])
                gray_score = float(gray_response[local_y, local_x])
                edge_score = float(edge_response[local_y, local_x])
                left = int(x1 + local_x)
                top = int(y1 + local_y)
                bbox = [
                    left,
                    top,
                    left + template_width,
                    top + template_height,
                ]
                center = bbox_center(bbox)
                raw.append(
                    {
                        "score": score,
                        "template_score": score,
                        "gray_score": gray_score,
                        "edge_score": edge_score,
                        "depth_shape_score": 0.0,
                        "bbox": bbox,
                        "center_pixel": center,
                        "template_bbox": bbox,
                        "template_center_pixel": center,
                        "template_scale": [width_scale, height_scale],
                        "proposal_sources": ["multiscale_template"],
                    }
                )

    return suppress_matches(raw, nms_pixels=14, max_matches=max_matches)


def valid_depth_values(depth_image, bbox):
    x1, y1, x2, y2 = bbox
    height, width = depth_image.shape
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))
    values = depth_image[y1:y2, x1:x2]
    return values[np.isfinite(values) & (values > 0.05) & (values < 5.0)]


def percentile_depth(depth_image, bbox, percentile):
    values = valid_depth_values(depth_image, bbox)
    if values.size == 0:
        raise RuntimeError("no valid depth in tray bbox {}".format(bbox))
    return float(np.percentile(values, percentile)), int(values.size)


def depth_vertical_proposals(
    depth_image,
    roi,
    reference_depth,
    depth_tolerance,
    template_shape,
    max_matches,
):
    x1, y1, x2, y2 = pixel_roi(depth_image.shape, roi)
    search = depth_image[y1:y2, x1:x2]
    valid = np.isfinite(search) & (search > 0.05) & (search < 5.0)
    near_plane = valid & (np.abs(search - reference_depth) <= depth_tolerance)
    mask = (near_plane.astype(np.uint8) * 255)

    # Keep vertical tray surfaces while suppressing horizontal shelf rails.
    vertical = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 11))
    )
    vertical = cv2.morphologyEx(
        vertical, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)
    template_height, template_width = template_shape[:2]
    proposals = []

    for label in range(1, count):
        local_x, local_y, width, height, area = stats[label].tolist()
        if height < max(14, int(template_height * 0.35)):
            continue
        if height > int(template_height * 1.8):
            continue
        if width < 2 or width > int(template_width * 3.5):
            continue
        if area < 20:
            continue
        aspect = float(height) / float(max(width, 1))
        fill = float(area) / float(max(width * height, 1))
        vertical_score = float(np.clip((aspect - 0.45) / 2.5, 0.0, 1.0))
        height_score = math.exp(
            -0.5 * (math.log(max(height, 1) / float(template_height)) / 0.55) ** 2
        )
        shape_score = float(
            np.clip(0.45 * vertical_score + 0.35 * height_score + 0.20 * fill, 0, 1)
        )
        pad_x = max(2, int(round(width * 0.15)))
        pad_y = 2
        left = max(x1, x1 + local_x - pad_x)
        top = max(y1, y1 + local_y - pad_y)
        right = min(x2, x1 + local_x + width + pad_x)
        bottom = min(y2, y1 + local_y + height + pad_y)
        component_values = search[labels == label]
        component_values = component_values[
            np.isfinite(component_values)
            & (component_values > 0.05)
            & (component_values < 5.0)
        ]
        if component_values.size == 0:
            continue
        depth_hint = float(np.median(component_values))
        bbox = [left, top, right, bottom]
        center = bbox_center(bbox)
        proposals.append(
            {
                "score": 0.55 * shape_score,
                "template_score": 0.0,
                "gray_score": 0.0,
                "edge_score": 0.0,
                "depth_shape_score": shape_score,
                "depth_override_m": depth_hint,
                "bbox": bbox,
                "center_pixel": center,
                "depth_bbox": bbox,
                "depth_center_pixel": center,
                "template_scale": None,
                "proposal_sources": ["rgbd_vertical"],
            }
        )

    return sorted(proposals, key=lambda item: item["score"], reverse=True)[:max_matches]


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


def fit_row_model(anchors, fallback_y):
    if len(anchors) < 2:
        return 0.0, float(fallback_y)
    x = np.array([item["center_pixel"][0] for item in anchors], dtype=np.float64)
    y = np.array([item["center_pixel"][1] for item in anchors], dtype=np.float64)
    if float(np.ptp(x)) < 30.0:
        return 0.0, float(np.median(y))
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.clip(slope, -0.12, 0.12)), float(intercept)


def load_shelf_calibration(path):
    if not path:
        return []
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise ValueError("calibration file does not exist: {}".format(path))
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    knots = data.get("shelf_knots", [])
    if not isinstance(knots, list) or not knots:
        raise ValueError("calibration file must contain non-empty shelf_knots")
    parsed = []
    for knot in knots:
        coordinate = float(knot["coordinate"])
        offset = [float(value) for value in knot["offset_xyz_m"]]
        if len(offset) != 3:
            raise ValueError("each calibration offset_xyz_m must contain 3 values")
        parsed.append(
            {"coordinate": coordinate, "offset_xyz_m": offset}
        )
    return sorted(parsed, key=lambda item: item["coordinate"])


def interpolate_shelf_offset(knots, coordinate):
    if not knots:
        return np.zeros(3, dtype=np.float64)
    coordinates = np.array(
        [item["coordinate"] for item in knots], dtype=np.float64
    )
    result = []
    for axis in range(3):
        values = np.array(
            [item["offset_xyz_m"][axis] for item in knots], dtype=np.float64
        )
        result.append(float(np.interp(coordinate, coordinates, values)))
    return np.array(result, dtype=np.float64)


def shelf_plane_correction(
    candidate,
    plane_x,
    plane_z,
    shelf_coordinate,
    base_weight,
    edge_weight,
    uncertainty_weight,
    maximum_weight,
):
    template_uncertainty = float(
        np.clip(1.0 - candidate["template_score"], 0.0, 1.0)
    )
    edge_factor = float(np.clip(abs(shelf_coordinate) * 2.0, 0.0, 1.0))
    correction_weight = float(
        np.clip(
            base_weight
            + edge_weight * edge_factor
            + uncertainty_weight * template_uncertainty,
            0.0,
            maximum_weight,
        )
    )
    raw_xyz = np.array(candidate["base_link_xyz_m"], dtype=np.float64)
    plane_delta = np.array(
        [plane_x - raw_xyz[0], 0.0, plane_z - raw_xyz[2]],
        dtype=np.float64,
    )
    return correction_weight, correction_weight * plane_delta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output-dir", default="/tmp/scene3_upper_trays")
    parser.add_argument("--search-roi", default="0.06,0.20,0.94,0.46")
    parser.add_argument("--match-threshold", type=float, default=0.18)
    parser.add_argument("--minimum-score", type=float, default=0.26)
    parser.add_argument("--anchor-score", type=float, default=0.55)
    parser.add_argument("--plane-x-tolerance", type=float, default=0.06)
    parser.add_argument("--plane-z-tolerance", type=float, default=0.04)
    parser.add_argument("--perspective-x-weight", type=float, default=0.16)
    parser.add_argument("--perspective-z-weight", type=float, default=0.08)
    parser.add_argument("--row-tolerance-pixels", type=float, default=42.0)
    parser.add_argument("--expected-count", type=int, default=3)
    parser.add_argument("--depth-percentile", type=float, default=10.0)
    parser.add_argument("--depth-band-tolerance", type=float, default=0.08)
    parser.add_argument("--geometry-depth-score", type=float, default=0.55)
    parser.add_argument("--width-scales", default="0.65,0.80,1.0,1.25,1.55,1.9,2.3")
    parser.add_argument("--height-scales", default="0.80,0.95,1.10")
    parser.add_argument("--edge-weight", type=float, default=0.35)
    parser.add_argument("--calibration-file", default="")
    parser.add_argument("--plane-correction-base", type=float, default=0.05)
    parser.add_argument("--plane-correction-edge-weight", type=float, default=0.25)
    parser.add_argument(
        "--plane-correction-uncertainty-weight", type=float, default=0.60
    )
    parser.add_argument("--plane-correction-max", type=float, default=0.80)
    parser.add_argument("--use-corrected-output", action="store_true")
    parser.add_argument("--nms-pixels", type=int, default=22)
    parser.add_argument("--max-matches", type=int, default=60)
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
    roi = parse_roi(args.search_roi)
    width_scales = parse_csv_floats(args.width_scales)
    height_scales = parse_csv_floats(args.height_scales)
    calibration_knots = load_shelf_calibration(args.calibration_file)

    rospy.init_node("scene3_upper_tray_perception", anonymous=True)
    camera_info = rospy.wait_for_message(args.camera_info_topic, CameraInfo, timeout=args.timeout)
    rgb_message = rospy.wait_for_message(args.rgb_topic, CompressedImage, timeout=args.timeout)
    depth_message = rospy.wait_for_message(args.depth_topic, CompressedImage, timeout=args.timeout)
    image = decode_rgb(rgb_message)
    depth_image = decode_depth(depth_message)
    if image.shape[:2] != depth_image.shape[:2]:
        raise RuntimeError("RGB and depth dimensions do not match")

    template_matches = multiscale_template_matches(
        image,
        template,
        roi,
        args.match_threshold,
        width_scales,
        height_scales,
        args.edge_weight,
        args.max_matches,
    )
    if not template_matches:
        raise RuntimeError("no tray-like appearance found in the upper shelf")

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

    for match in template_matches:
        depth_m, _ = percentile_depth(depth_image, match["bbox"], args.depth_percentile)
        match["proposal_depth_m"] = depth_m
    appearance_anchors = [
        item for item in template_matches if item["template_score"] >= args.anchor_score
    ]
    if len(appearance_anchors) < 2:
        appearance_anchors = template_matches[:2]
    reference_depth = float(
        np.median([item["proposal_depth_m"] for item in appearance_anchors])
    )

    depth_matches = depth_vertical_proposals(
        depth_image,
        roi,
        reference_depth,
        args.depth_band_tolerance,
        template.shape,
        args.max_matches,
    )
    matches = suppress_matches(
        template_matches + depth_matches,
        nms_pixels=args.nms_pixels,
        max_matches=args.max_matches,
    )

    candidates = []
    for rank_by_score, match in enumerate(
        sorted(matches, key=lambda item: item["score"], reverse=True)
    ):
        detection_bbox = list(match.get("template_bbox", match["bbox"]))
        detection_center = list(
            match.get("template_center_pixel", match["center_pixel"])
        )
        use_depth_geometry = bool(
            match.get("depth_shape_score", 0.0) >= args.geometry_depth_score
            and match.get("depth_bbox") is not None
        )
        object_bbox = list(
            match["depth_bbox"] if use_depth_geometry else detection_bbox
        )
        object_center = list(
            match.get("depth_center_pixel", bbox_center(object_bbox))
            if use_depth_geometry
            else detection_center
        )
        if use_depth_geometry and match.get("depth_override_m") is not None:
            depth_m = float(match["depth_override_m"])
            valid_pixels = int(valid_depth_values(depth_image, object_bbox).size)
        else:
            depth_m, valid_pixels = percentile_depth(
                depth_image, detection_bbox, args.depth_percentile
            )
        camera_xyz = np.array(project_pixel(object_center, depth_m, camera_info))
        target_xyz = rotation.dot(camera_xyz) + translation
        candidate = dict(match)
        candidate.update(
            {
                "candidate_id": "candidate_{}".format(rank_by_score),
                "rank_by_score": rank_by_score,
                "depth_m": depth_m,
                "depth_percentile": args.depth_percentile,
                "valid_depth_pixels": valid_pixels,
                "camera_xyz_m": camera_xyz.tolist(),
                "base_link_xyz_m": target_xyz.tolist(),
                "detection_bbox": detection_bbox,
                "detection_center_pixel": detection_center,
                "object_bbox": object_bbox,
                "object_center_pixel": object_center,
                "geometry_source": (
                    "rgbd_vertical" if use_depth_geometry else "template"
                ),
                "bbox": object_bbox,
                "center_pixel": object_center,
            }
        )
        candidate.pop("depth_override_m", None)
        candidate.pop("proposal_depth_m", None)
        candidates.append(candidate)

    anchors = [
        item for item in candidates if item["template_score"] >= args.anchor_score
    ]
    if len(anchors) < 2:
        anchors = sorted(
            candidates, key=lambda item: item["template_score"], reverse=True
        )[:2]
    plane_x = float(np.median([item["base_link_xyz_m"][0] for item in anchors]))
    plane_z = float(np.median([item["base_link_xyz_m"][2] for item in anchors]))
    anchor_u = float(np.median([item["center_pixel"][0] for item in anchors]))
    anchor_y = float(np.median([item["center_pixel"][1] for item in anchors]))
    row_slope, row_intercept = fit_row_model(anchors, anchor_y)
    roi_x1, _, roi_x2, _ = pixel_roi(image.shape, roi)
    shelf_width = float(max(1, roi_x2 - roi_x1))

    eligible = []
    for candidate in candidates:
        u, v = candidate["center_pixel"]
        relative_offset = abs(float(u) - anchor_u) / shelf_width
        x_tolerance = args.plane_x_tolerance + args.perspective_x_weight * relative_offset
        z_tolerance = args.plane_z_tolerance + args.perspective_z_weight * relative_offset
        predicted_row_y = row_slope * float(u) + row_intercept
        row_error = abs(float(v) - predicted_row_y)
        plane_dx = abs(candidate["base_link_xyz_m"][0] - plane_x)
        plane_dz = abs(candidate["base_link_xyz_m"][2] - plane_z)
        plane_score = math.exp(
            -0.5 * (plane_dx / max(x_tolerance, 1e-6)) ** 2
            -0.5 * (plane_dz / max(z_tolerance, 1e-6)) ** 2
        )
        row_score = math.exp(
            -0.5 * (row_error / max(args.row_tolerance_pixels, 1e-6)) ** 2
        )
        appearance_score = max(
            candidate["template_score"], 0.75 * candidate["depth_shape_score"]
        )
        selection_score = (
            0.55 * appearance_score + 0.25 * plane_score + 0.20 * row_score
        )
        candidate.update(
            {
                "relative_shelf_offset": (float(u) - anchor_u) / shelf_width,
                "adaptive_x_tolerance_m": x_tolerance,
                "adaptive_z_tolerance_m": z_tolerance,
                "predicted_row_y": predicted_row_y,
                "row_error_pixels": row_error,
                "plane_dx_m": plane_dx,
                "plane_dz_m": plane_dz,
                "plane_score": plane_score,
                "row_score": row_score,
                "appearance_score": appearance_score,
                "selection_score": selection_score,
            }
        )
        shelf_center = 0.5 * float(roi_x1 + roi_x2)
        shelf_coordinate = (float(u) - shelf_center) / shelf_width
        correction_weight, plane_correction = shelf_plane_correction(
            candidate,
            plane_x,
            plane_z,
            shelf_coordinate,
            args.plane_correction_base,
            args.plane_correction_edge_weight,
            args.plane_correction_uncertainty_weight,
            args.plane_correction_max,
        )
        calibration_offset = interpolate_shelf_offset(
            calibration_knots, shelf_coordinate
        )
        raw_xyz = np.array(candidate["base_link_xyz_m"], dtype=np.float64)
        corrected_xyz = raw_xyz + plane_correction + calibration_offset
        candidate.update(
            {
                "shelf_coordinate": shelf_coordinate,
                "plane_correction_weight": correction_weight,
                "plane_correction_xyz_m": plane_correction.tolist(),
                "calibration_offset_xyz_m": calibration_offset.tolist(),
                "base_link_xyz_raw_m": raw_xyz.tolist(),
                "base_link_xyz_corrected_m": corrected_xyz.tolist(),
            }
        )
        candidate["base_link_xyz_m"] = (
            corrected_xyz.tolist()
            if args.use_corrected_output
            else raw_xyz.tolist()
        )
        source_ok = (
            candidate["template_score"] >= args.minimum_score
            or candidate["depth_shape_score"] >= 0.42
        )
        candidate["eligible"] = bool(
            source_ok
            and plane_dx <= x_tolerance
            and plane_dz <= z_tolerance
            and row_error <= args.row_tolerance_pixels
        )
        candidate["accepted"] = False
        if candidate["eligible"]:
            eligible.append(candidate)

    selected = sorted(
        eligible, key=lambda item: item["selection_score"], reverse=True
    )[: args.expected_count]
    selected_ids = {item["candidate_id"] for item in selected}
    for candidate in candidates:
        candidate["accepted"] = candidate["candidate_id"] in selected_ids

    trays = []
    for order_x, candidate in enumerate(
        sorted(selected, key=lambda item: item["center_pixel"][0])
    ):
        tray = dict(candidate)
        tray["id"] = "upper_x{}".format(order_x)
        trays.append(tray)

    candidate_visualization = image.copy()
    for candidate in sorted(
        candidates, key=lambda item: item["selection_score"], reverse=True
    )[:24]:
        x1, y1, x2, y2 = candidate["bbox"]
        color = (0, 255, 0) if candidate["accepted"] else (0, 255, 255)
        thickness = 3 if candidate["accepted"] else 1
        cv2.rectangle(candidate_visualization, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            candidate_visualization,
            "{} {:.2f}".format(candidate["candidate_id"], candidate["selection_score"]),
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )

    visualization = image.copy()
    for index, tray in enumerate(trays):
        detection_bbox = tray.get("detection_bbox", tray["bbox"])
        if detection_bbox != tray["bbox"]:
            dx1, dy1, dx2, dy2 = detection_bbox
            cv2.rectangle(
                visualization, (dx1, dy1), (dx2, dy2), (255, 255, 0), 1
            )
        x1, y1, x2, y2 = tray["bbox"]
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 3)
        center_x, center_y = tray["center_pixel"]
        cv2.circle(visualization, (center_x, center_y), 4, (0, 0, 255), -1)
        label = "{} s={:.2f} z={:.3f}m {}".format(
            tray["id"],
            tray["selection_score"],
            tray["depth_m"],
            tray.get("geometry_source", "unknown"),
        )
        label_position = (18, 25 + 24 * index)
        cv2.putText(
            visualization,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            visualization,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    status = "ok" if len(trays) == args.expected_count else "count_mismatch"
    payload = {
        "status": status,
        "algorithm": "multiscale_rgbd_object_geometry_v5",
        "source_frame": camera_info.header.frame_id,
        "target_frame": args.target_frame,
        "search_roi": roi,
        "expected_count": args.expected_count,
        "proposal_settings": {
            "match_threshold": args.match_threshold,
            "minimum_score": args.minimum_score,
            "anchor_score": args.anchor_score,
            "width_scales": width_scales,
            "height_scales": height_scales,
            "edge_weight": args.edge_weight,
            "depth_band_tolerance_m": args.depth_band_tolerance,
            "geometry_depth_score": args.geometry_depth_score,
            "appearance_depth_scale": 0.75,
            "selection_weights": {
                "appearance": 0.55,
                "plane": 0.25,
                "row": 0.20,
            },
        },
        "perspective_model": {
            "anchor_center_u": anchor_u,
            "reference_camera_depth_m": reference_depth,
            "reference_base_x_m": plane_x,
            "reference_base_z_m": plane_z,
            "base_x_tolerance_m": args.plane_x_tolerance,
            "base_z_tolerance_m": args.plane_z_tolerance,
            "x_tolerance_weight_m": args.perspective_x_weight,
            "z_tolerance_weight_m": args.perspective_z_weight,
            "row_slope": row_slope,
            "row_intercept": row_intercept,
            "row_tolerance_pixels": args.row_tolerance_pixels,
        },
        "shelf_coordinate_calibration": {
            "coordinate_definition": "(u - roi_center_u) / roi_width",
            "base_link_xyz_output": (
                "corrected" if args.use_corrected_output else "raw"
            ),
            "plane_correction": {
                "base_weight": args.plane_correction_base,
                "edge_weight": args.plane_correction_edge_weight,
                "uncertainty_weight": args.plane_correction_uncertainty_weight,
                "maximum_weight": args.plane_correction_max,
            },
            "calibration_file": args.calibration_file or None,
            "shelf_knots": calibration_knots,
        },
        "depth_method": "rgbd_component_geometry_or_template_percentile",
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

