#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scene3 right-wrist RGB-D gate used immediately before claw closure.

The senior task remains responsible for target selection, walking, secondary
head-camera recognition and the pregrasp IK command.  This node performs one
small, observation-only job after pregrasp:

* project the latest legal ``base_link`` tray point into the right camera;
* refine the prompted tray surface with wrist depth;
* obtain both right-finger poses from legal TF;
* verify that the tray is inside the finger-closing corridor and that no
  nearer non-tray surface (for example a shelf rail) occupies that corridor.

It never publishes an arm, base or claw command.  A later executor may close
the claw only after this node reports ``WRIST_PRECLOSE_GATE_OK``.
"""

from __future__ import print_function

import argparse
import math
import threading
import time

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

DEFAULT_RGB_TOPIC = "/cam_r/color/image_raw/compressed"
DEFAULT_DEPTH_TOPIC = "/cam_r/depth/image_rect_raw/compressedDepth"
DEFAULT_INFO_TOPIC = "/cam_r/color/camera_info"
DEFAULT_TARGET_TOPIC = "/challenge_cup_task_template/scene3/locked_target_base"
DEFAULT_TARGET_PARAM = "/challenge_cup_task_template/scene3/locked_target_base_xyz"
DEFAULT_DEBUG_TOPIC = "/challenge_cup_task_template/scene3/wrist_preclose_debug/compressed"

DEFAULT_LEFT_FINGER_FRAME = "right_gripper_left_inner_knuckle"
DEFAULT_RIGHT_FINGER_FRAME = "right_gripper_right_inner_knuckle"
DEFAULT_GRIPPER_BASE_FRAME = "right_gripper_base"


def intrinsics_from_k(camera_k):
    values = np.asarray(camera_k, dtype=float).reshape(-1)
    if values.size != 9:
        raise ValueError("camera K must contain nine values")
    fx, fy = float(values[0]), float(values[4])
    cx, cy = float(values[2]), float(values[5])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal length must be positive")
    return fx, fy, cx, cy


def project_camera_point(point_xyz, camera_k):
    point = np.asarray(point_xyz, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("camera point must be finite xyz")
    if point[2] <= 1e-4:
        raise ValueError("camera point is behind the image plane")
    fx, fy, cx, cy = intrinsics_from_k(camera_k)
    return np.array(
        [fx * point[0] / point[2] + cx, fy * point[1] / point[2] + cy],
        dtype=float,
    )


def deproject_pixel(pixel_uv, depth_mm, camera_k):
    pixel = np.asarray(pixel_uv, dtype=float)
    depth_m = float(depth_mm) * 0.001
    if pixel.shape != (2,) or depth_m <= 0.0 or not math.isfinite(depth_m):
        raise ValueError("pixel and positive depth are required")
    fx, fy, cx, cy = intrinsics_from_k(camera_k)
    return np.array(
        [
            (pixel[0] - cx) * depth_m / fx,
            (pixel[1] - cy) * depth_m / fy,
            depth_m,
        ],
        dtype=float,
    )


def decode_compressed_depth_payload(payload, cv2_module):
    raw = bytes(payload)
    offset = raw.find(PNG_SIGNATURE)
    if offset < 0:
        raise ValueError("compressedDepth payload has no PNG image")
    encoded = np.frombuffer(raw[offset:], dtype=np.uint8)
    depth = cv2_module.imdecode(encoded, cv2_module.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise ValueError("failed to decode compressedDepth image")
    return depth.astype(np.float32, copy=False)


def _nearest_pixel(mask, prompt_uv):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("mask is empty")
    prompt = np.asarray(prompt_uv, dtype=float)
    distance_sq = (xs - prompt[0]) ** 2 + (ys - prompt[1]) ** 2
    index = int(np.argmin(distance_sq))
    return int(xs[index]), int(ys[index]), float(math.sqrt(distance_sq[index]))


def _binary_open3(mask):
    source = np.asarray(mask, dtype=bool)
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    neighbour_count = np.zeros(source.shape, dtype=np.uint8)
    for row_offset in range(3):
        for column_offset in range(3):
            neighbour_count += padded[
                row_offset : row_offset + source.shape[0],
                column_offset : column_offset + source.shape[1],
            ]
    eroded = neighbour_count == 9
    padded_eroded = np.pad(eroded, 1, mode="constant", constant_values=False)
    dilated = np.zeros(source.shape, dtype=bool)
    for row_offset in range(3):
        for column_offset in range(3):
            dilated |= padded_eroded[
                row_offset : row_offset + source.shape[0],
                column_offset : column_offset + source.shape[1],
            ]
    return dilated


def _connected_components(mask):
    """Small NumPy fallback returning (component_mask, stats) pairs."""

    source = np.asarray(mask, dtype=bool)
    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            source.astype(np.uint8), 8
        )
        result = []
        for label in range(1, int(count)):
            component = labels == label
            result.append(
                (
                    component,
                    {
                        "area": int(stats[label, cv2.CC_STAT_AREA]),
                        "width": int(stats[label, cv2.CC_STAT_WIDTH]),
                        "height": int(stats[label, cv2.CC_STAT_HEIGHT]),
                    },
                )
            )
        return result
    except ImportError:
        pass

    height, width = source.shape
    visited = np.zeros(source.shape, dtype=bool)
    components = []
    for start_y, start_x in zip(*np.nonzero(source & ~visited)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and source[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        component = np.zeros(source.shape, dtype=bool)
        ys = np.asarray([item[0] for item in pixels], dtype=int)
        xs = np.asarray([item[1] for item in pixels], dtype=int)
        component[ys, xs] = True
        components.append(
            (
                component,
                {
                    "area": int(len(pixels)),
                    "width": int(np.max(xs) - np.min(xs) + 1),
                    "height": int(np.max(ys) - np.min(ys) + 1),
                },
            )
        )
    return components


def prompted_depth_component(
    depth_image,
    prompt_uv,
    expected_depth_mm,
    roi_radius_px=90,
    depth_band_mm=35.0,
    minimum_pixels=30,
):
    """Select the wrist-depth component nearest a projected head-camera target.

    This is a geometric prompt, not a fixed scene pixel.  The expected depth
    comes from TF-transforming the latest head RGB-D target into the wrist
    optical frame.  Thin components receive a score penalty so a shelf rail
    does not beat a tray surface merely because it crosses the prompt ROI.
    """

    depth = np.asarray(depth_image, dtype=float)
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    prompt = np.asarray(prompt_uv, dtype=float)
    if prompt.shape != (2,) or not np.all(np.isfinite(prompt)):
        raise ValueError("prompt must be a finite image pixel")
    expected = float(expected_depth_mm)
    if expected <= 50.0 or not math.isfinite(expected):
        raise ValueError("expected depth is invalid")

    height, width = depth.shape
    u = int(round(prompt[0]))
    v = int(round(prompt[1]))
    radius = max(12, int(roi_radius_px))
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("prompt ROI does not overlap the image")

    crop = depth[y0:y1, x0:x1]
    band = max(8.0, float(depth_band_mm))
    candidate = (
        np.isfinite(crop)
        & (crop > 50.0)
        & (crop < 10000.0)
        & (np.abs(crop - expected) <= band)
    ).astype(np.uint8)
    if int(np.count_nonzero(candidate)) < int(minimum_pixels):
        raise ValueError("not enough wrist-depth pixels near projected target")

    # Remove isolated noise but preserve object edges.  A 3x3 opening removes
    # one-pixel rails/noise without assuming a fixed tray location or size.
    cleaned = _binary_open3(candidate)
    if int(np.count_nonzero(cleaned)) < int(minimum_pixels):
        cleaned = candidate

    components = _connected_components(cleaned)
    best_component = None
    best_score = None
    prompt_local = np.array([prompt[0] - x0, prompt[1] - y0], dtype=float)
    for component, stats in components:
        area = int(stats["area"])
        if area < int(minimum_pixels):
            continue
        _, _, distance = _nearest_pixel(component, prompt_local)
        box_width = max(1, int(stats["width"]))
        box_height = max(1, int(stats["height"]))
        major = float(max(box_width, box_height))
        minor = float(min(box_width, box_height))
        thinness = minor / major
        fill = float(area) / float(box_width * box_height)
        line_penalty = 35.0 if thinness < 0.10 or fill < 0.08 else 0.0
        score = distance + line_penalty - min(float(area), 2000.0) * 0.004
        if best_score is None or score < best_score:
            best_score = score
            best_component = component
    if best_component is None:
        raise ValueError("no tray-like depth component near projected target")

    local_mask = best_component
    local_x, local_y, prompt_distance = _nearest_pixel(local_mask, prompt_local)
    full_mask = np.zeros_like(depth, dtype=bool)
    full_mask[y0:y1, x0:x1] = local_mask
    target_u, target_v = int(x0 + local_x), int(y0 + local_y)

    component_depth = depth[full_mask]
    valid_component_depth = component_depth[
        np.isfinite(component_depth)
        & (component_depth > 50.0)
        & (component_depth < 10000.0)
    ]
    if valid_component_depth.size < int(minimum_pixels):
        raise ValueError("selected tray component has insufficient depth")
    point_depth = float(depth[target_v, target_u])
    if not math.isfinite(point_depth) or point_depth <= 50.0:
        point_depth = float(np.median(valid_component_depth))

    return {
        "mask": full_mask,
        "target_pixel": (target_u, target_v),
        "target_depth_mm": point_depth,
        "median_depth_mm": float(np.median(valid_component_depth)),
        "surface_pixels": int(np.count_nonzero(full_mask)),
        "prompt_distance_px": float(prompt_distance),
        "roi": (x0, y0, x1, y1),
    }


def estimate_finger_tips(
    left_origin_xyz,
    right_origin_xyz,
    gripper_base_xyz,
    extension_m=0.045,
):
    left = np.asarray(left_origin_xyz, dtype=float)
    right = np.asarray(right_origin_xyz, dtype=float)
    base = np.asarray(gripper_base_xyz, dtype=float)
    if left.shape != (3,) or right.shape != (3,) or base.shape != (3,):
        raise ValueError("finger and gripper origins must be xyz")
    midpoint = 0.5 * (left + right)
    forward = midpoint - base
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        raise ValueError("cannot infer gripper forward axis")
    forward /= norm
    extension = max(0.0, float(extension_m))
    left_tip = left + extension * forward
    right_tip = right + extension * forward
    tcp = 0.5 * (left_tip + right_tip)
    return left_tip, right_tip, tcp, forward


def point_to_segment_metrics(point_xyz, segment_start_xyz, segment_end_xyz):
    point = np.asarray(point_xyz, dtype=float)
    start = np.asarray(segment_start_xyz, dtype=float)
    end = np.asarray(segment_end_xyz, dtype=float)
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator < 1e-10:
        raise ValueError("finger-tip segment is degenerate")
    parameter = float(np.dot(point - start, segment) / denominator)
    closest = start + min(1.0, max(0.0, parameter)) * segment
    distance = float(np.linalg.norm(point - closest))
    return parameter, distance, closest


def closing_corridor_mask(image_shape, left_uv, right_uv, radius_px=8):
    height, width = [int(value) for value in image_shape[:2]]
    if height <= 0 or width <= 0:
        raise ValueError("image shape is invalid")
    left = np.asarray(left_uv, dtype=float)
    right = np.asarray(right_uv, dtype=float)
    segment = right - left
    denominator = float(np.dot(segment, segment))
    if denominator < 1e-8:
        raise ValueError("projected finger segment is degenerate")
    yy, xx = np.indices((height, width), dtype=float)
    relative_x = xx - left[0]
    relative_y = yy - left[1]
    parameter = np.clip(
        (relative_x * segment[0] + relative_y * segment[1]) / denominator,
        0.0,
        1.0,
    )
    closest_x = left[0] + parameter * segment[0]
    closest_y = left[1] + parameter * segment[1]
    distance_sq = (xx - closest_x) ** 2 + (yy - closest_y) ** 2
    return distance_sq <= float(max(1, int(radius_px))) ** 2


def evaluate_preclose_gate(
    depth_image,
    tray_mask,
    target_xyz,
    left_tip_xyz,
    right_tip_xyz,
    camera_k,
    corridor_radius_px=8,
    maximum_target_segment_distance_m=0.018,
    maximum_tcp_error_m=0.022,
    minimum_tray_corridor_pixels=8,
    maximum_obstacle_ratio=0.20,
    obstacle_depth_margin_mm=12.0,
    self_mask_radius_px=9,
):
    """Evaluate whether closing the fingers would pinch tray, not shelf rail."""

    depth = np.asarray(depth_image, dtype=float)
    tray = np.asarray(tray_mask, dtype=bool)
    if depth.shape != tray.shape or depth.ndim != 2:
        raise ValueError("depth and tray mask must have the same 2-D shape")
    target = np.asarray(target_xyz, dtype=float)
    left_tip = np.asarray(left_tip_xyz, dtype=float)
    right_tip = np.asarray(right_tip_xyz, dtype=float)
    tcp = 0.5 * (left_tip + right_tip)

    target_uv = project_camera_point(target, camera_k)
    left_uv = project_camera_point(left_tip, camera_k)
    right_uv = project_camera_point(right_tip, camera_k)
    tcp_uv = project_camera_point(tcp, camera_k)
    parameter, segment_distance, _ = point_to_segment_metrics(
        target, left_tip, right_tip
    )
    tcp_error = target - tcp
    tcp_error_norm = float(np.linalg.norm(tcp_error))
    finger_gap = float(np.linalg.norm(right_tip - left_tip))

    corridor = closing_corridor_mask(
        depth.shape, left_uv, right_uv, radius_px=corridor_radius_px
    )
    target_u, target_v = [int(round(value)) for value in target_uv]
    target_in_image = (
        0 <= target_u < depth.shape[1] and 0 <= target_v < depth.shape[0]
    )
    target_in_corridor = bool(target_in_image and corridor[target_v, target_u])
    tray_corridor_pixels = int(np.count_nonzero(corridor & tray))

    self_mask = np.zeros(depth.shape, dtype=bool)
    yy, xx = np.indices(depth.shape, dtype=float)
    for pixel in (left_uv, right_uv):
        self_mask |= (
            (xx - float(pixel[0])) ** 2 + (yy - float(pixel[1])) ** 2
            <= float(max(1, int(self_mask_radius_px))) ** 2
        )

    target_depth_mm = float(target[2] * 1000.0)
    valid = np.isfinite(depth) & (depth > 50.0) & (depth < 10000.0)
    obstacle = (
        corridor
        & valid
        & ~tray
        & ~self_mask
        & (depth <= target_depth_mm + float(obstacle_depth_margin_mm))
    )
    obstacle_pixels = int(np.count_nonzero(obstacle))
    relevant_pixels = tray_corridor_pixels + obstacle_pixels
    obstacle_ratio = float(obstacle_pixels) / float(max(1, relevant_pixels))

    checks = {
        "target_in_corridor": target_in_corridor,
        "target_between_fingers": 0.02 <= parameter <= 0.98,
        "segment_distance": segment_distance
        <= float(maximum_target_segment_distance_m),
        "tcp_error": tcp_error_norm <= float(maximum_tcp_error_m),
        "tray_occupancy": tray_corridor_pixels
        >= int(minimum_tray_corridor_pixels),
        "obstacle_clearance": obstacle_ratio <= float(maximum_obstacle_ratio),
        "finger_gap": 0.012 <= finger_gap <= 0.16,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "target_uv": target_uv,
        "left_uv": left_uv,
        "right_uv": right_uv,
        "tcp_uv": tcp_uv,
        "tcp_xyz": tcp,
        "tcp_error_xyz": tcp_error,
        "tcp_error_m": tcp_error_norm,
        "segment_parameter": parameter,
        "segment_distance_m": segment_distance,
        "finger_gap_m": finger_gap,
        "tray_corridor_pixels": tray_corridor_pixels,
        "obstacle_pixels": obstacle_pixels,
        "obstacle_ratio": obstacle_ratio,
        "corridor_mask": corridor,
        "obstacle_mask": obstacle,
    }


def stable_preclose_gate(
    observations,
    required_count=3,
    maximum_target_spread_m=0.008,
    maximum_tcp_spread_m=0.006,
):
    samples = list(observations)[-max(1, int(required_count)) :]
    if len(samples) < int(required_count):
        return False, {"reason": "not_enough_samples"}
    if not all(bool(sample.get("passed", False)) for sample in samples):
        return False, {"reason": "one_or_more_frames_failed"}
    targets = np.asarray([sample["target_xyz"] for sample in samples], dtype=float)
    tcps = np.asarray([sample["tcp_xyz"] for sample in samples], dtype=float)
    target_median = np.median(targets, axis=0)
    tcp_median = np.median(tcps, axis=0)
    target_spread = float(np.max(np.linalg.norm(targets - target_median, axis=1)))
    tcp_spread = float(np.max(np.linalg.norm(tcps - tcp_median, axis=1)))
    passed = (
        target_spread <= float(maximum_target_spread_m)
        and tcp_spread <= float(maximum_tcp_spread_m)
    )
    return passed, {
        "reason": "ok" if passed else "unstable",
        "target_median": target_median,
        "tcp_median": tcp_median,
        "target_spread_m": target_spread,
        "tcp_spread_m": tcp_spread,
    }


def _translation_xyz(transform):
    value = transform.transform.translation
    return np.array([value.x, value.y, value.z], dtype=float)


def run_ros(args):
    import cv2
    import message_filters
    import rospy
    import tf2_geometry_msgs  # noqa: F401
    import tf2_ros
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import CameraInfo, CompressedImage

    rospy.init_node("scene3_wrist_preclose_gate", anonymous=True)
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    tf2_ros.TransformListener(tf_buffer)

    lock = threading.Lock()
    state = {
        "camera_info": None,
        "target": None,
        "latched_target": None,
        "observations": [],
        "last_error": "waiting for input",
    }
    if args.target_param and rospy.has_param(args.target_param):
        locked_xyz = np.asarray(rospy.get_param(args.target_param), dtype=float)
        if locked_xyz.shape != (3,) or not np.all(np.isfinite(locked_xyz)):
            raise RuntimeError("locked senior target parameter is invalid")
        locked_message = PointStamped()
        locked_message.header.frame_id = "base_link"
        locked_message.header.stamp = rospy.Time(0)
        locked_message.point.x = float(locked_xyz[0])
        locked_message.point.y = float(locked_xyz[1])
        locked_message.point.z = float(locked_xyz[2])
        state["target"] = locked_message
    debug_pub = rospy.Publisher(args.debug_topic, CompressedImage, queue_size=1)

    def info_callback(message):
        with lock:
            state["camera_info"] = message

    def target_callback(message):
        with lock:
            state["target"] = message

    rospy.Subscriber(args.info_topic, CameraInfo, info_callback, queue_size=1)
    rospy.Subscriber(args.target_topic, PointStamped, target_callback, queue_size=1)

    def synchronized_callback(rgb_message, depth_message):
        try:
            with lock:
                info = state["camera_info"]
                source_target = state["latched_target"] or state["target"]
            if info is None or source_target is None:
                return
            # The Scene3 simulator currently labels CameraInfo with
            # ``Right wrist Camera View`` although the image projection axes
            # are provided by ``right_wrist_camera_link``.  Keep the runtime
            # override explicit instead of silently assuming either frame.
            camera_frame = str(args.camera_frame or info.header.frame_id)
            if not camera_frame:
                raise RuntimeError("right camera CameraInfo has no frame_id")

            image_data = np.frombuffer(rgb_message.data, dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("failed to decode right RGB image")
            depth = decode_compressed_depth_payload(depth_message.data, cv2)
            if depth.shape[:2] != image.shape[:2]:
                raise RuntimeError("right RGB and depth image sizes differ")

            target_latest = PointStamped()
            target_latest.header.frame_id = source_target.header.frame_id
            target_latest.header.stamp = rospy.Time(0)
            target_latest.point = source_target.point
            target_camera = tf_buffer.transform(
                target_latest, camera_frame, rospy.Duration(0.3)
            )
            expected_xyz = np.array(
                [
                    target_camera.point.x,
                    target_camera.point.y,
                    target_camera.point.z,
                ],
                dtype=float,
            )
            camera_k = list(info.K)
            prompt_uv = project_camera_point(expected_xyz, camera_k)
            component = prompted_depth_component(
                depth,
                prompt_uv,
                expected_xyz[2] * 1000.0,
                roi_radius_px=args.roi_radius,
                depth_band_mm=args.depth_band,
                minimum_pixels=args.minimum_component_pixels,
            )
            refined_xyz = deproject_pixel(
                component["target_pixel"], component["target_depth_mm"], camera_k
            )

            # Once wrist RGB-D has found the prompted tray surface, preserve
            # that physical point in the original base frame.  The head YOLO
            # box can jump to an adjacent tray when the arm occludes its view;
            # continuing to chase that box would change target identity.
            with lock:
                needs_latch = state["latched_target"] is None
            if needs_latch:
                refined_camera = PointStamped()
                refined_camera.header.frame_id = camera_frame
                refined_camera.header.stamp = rospy.Time(0)
                refined_camera.point.x = float(refined_xyz[0])
                refined_camera.point.y = float(refined_xyz[1])
                refined_camera.point.z = float(refined_xyz[2])
                latched_target = tf_buffer.transform(
                    refined_camera,
                    source_target.header.frame_id,
                    rospy.Duration(0.3),
                )
                latched_target.header.stamp = rospy.Time(0)
                with lock:
                    if state["latched_target"] is None:
                        state["latched_target"] = latched_target
                print(
                    "Latched wrist target in {}: {}".format(
                        latched_target.header.frame_id,
                        [
                            round(float(latched_target.point.x), 4),
                            round(float(latched_target.point.y), 4),
                            round(float(latched_target.point.z), 4),
                        ],
                    )
                )

            transforms = []
            for frame in (
                args.left_finger_frame,
                args.right_finger_frame,
                args.gripper_base_frame,
            ):
                transforms.append(
                    tf_buffer.lookup_transform(
                        camera_frame, frame, rospy.Time(0), rospy.Duration(0.3)
                    )
                )
            left_tip, right_tip, tcp, _ = estimate_finger_tips(
                _translation_xyz(transforms[0]),
                _translation_xyz(transforms[1]),
                _translation_xyz(transforms[2]),
                extension_m=args.tcp_extension,
            )
            result = evaluate_preclose_gate(
                depth,
                component["mask"],
                refined_xyz,
                left_tip,
                right_tip,
                camera_k,
                corridor_radius_px=args.corridor_radius,
                maximum_target_segment_distance_m=args.segment_tolerance,
                maximum_tcp_error_m=args.tcp_tolerance,
                minimum_tray_corridor_pixels=args.minimum_corridor_pixels,
                maximum_obstacle_ratio=args.maximum_obstacle_ratio,
            )
            observation = dict(result)
            observation["target_xyz"] = refined_xyz
            observation["tcp_xyz"] = tcp
            observation["stamp"] = (
                int(rgb_message.header.stamp.secs),
                int(rgb_message.header.stamp.nsecs),
            )
            with lock:
                observations = state["observations"]
                if not observations or observations[-1]["stamp"] != observation["stamp"]:
                    observations.append(observation)
                    del observations[:-max(12, args.required_frames * 3)]
                state["last_error"] = ""

            overlay = image.copy()
            overlay[result["corridor_mask"]] = (
                0.55 * overlay[result["corridor_mask"]]
                + 0.45 * np.array([0, 255, 255])
            ).astype(np.uint8)
            overlay[result["obstacle_mask"]] = np.array([0, 0, 255], dtype=np.uint8)
            contours, _ = cv2.findContours(
                component["mask"].astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            colours = [(255, 0, 0), (255, 0, 0), (255, 255, 255)]
            for pixel, colour in zip(
                (result["left_uv"], result["right_uv"], result["target_uv"]),
                colours,
            ):
                cv2.circle(
                    overlay,
                    tuple(int(round(value)) for value in pixel),
                    6,
                    colour,
                    2,
                )
            label = "GATE PASS" if result["passed"] else "GATE BLOCK"
            cv2.putText(
                overlay,
                "{} err={:.1f}mm obs={:.2f}".format(
                    label, result["tcp_error_m"] * 1000.0, result["obstacle_ratio"]
                ),
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0) if result["passed"] else (0, 0, 255),
                2,
            )
            ok, encoded = cv2.imencode(".jpg", overlay)
            if ok:
                debug = CompressedImage()
                debug.header = rgb_message.header
                debug.format = "jpeg"
                debug.data = encoded.tobytes()
                debug_pub.publish(debug)
        except Exception as exc:
            with lock:
                state["last_error"] = str(exc)
            rospy.logwarn_throttle(1.0, "scene3 wrist preclose gate: %s", exc)

    rgb_sub = message_filters.Subscriber(args.rgb_topic, CompressedImage)
    depth_sub = message_filters.Subscriber(args.depth_topic, CompressedImage)
    synchronizer = message_filters.ApproximateTimeSynchronizer(
        [rgb_sub, depth_sub], queue_size=8, slop=0.08
    )
    synchronizer.registerCallback(synchronized_callback)

    print("Waiting for right-wrist RGB-D preclose observations")
    deadline = time.time() + float(args.timeout)
    last_reported_count = -1
    while not rospy.is_shutdown() and time.time() < deadline:
        with lock:
            observations = list(state["observations"])
            last_error = state["last_error"]
        if len(observations) != last_reported_count and observations:
            latest = observations[-1]
            failed = [name for name, passed in latest["checks"].items() if not passed]
            print(
                "frame={} pass={} target={} tcp={} error_mm={:.1f} obstacle={:.3f} failed={}".format(
                    len(observations),
                    latest["passed"],
                    np.round(latest["target_xyz"], 4).tolist(),
                    np.round(latest["tcp_xyz"], 4).tolist(),
                    latest["tcp_error_m"] * 1000.0,
                    latest["obstacle_ratio"],
                    failed,
                )
            )
            last_reported_count = len(observations)
        passed, stable = stable_preclose_gate(
            observations,
            required_count=args.required_frames,
            maximum_target_spread_m=args.maximum_target_spread,
            maximum_tcp_spread_m=args.maximum_tcp_spread,
        )
        if passed:
            print(
                "WRIST_PRECLOSE_GATE_OK target_spread={:.4f}m tcp_spread={:.4f}m; observation only, claw remains open".format(
                    stable["target_spread_m"], stable["tcp_spread_m"]
                )
            )
            return 0
        rospy.sleep(0.05)

    with lock:
        last_error = state["last_error"]
        observations = list(state["observations"])
    if observations:
        latest = observations[-1]
        failed = [name for name, passed in latest["checks"].items() if not passed]
        raise RuntimeError(
            "WRIST_PRECLOSE_GATE_BLOCKED failed={} last_error={}".format(
                failed, last_error or "none"
            )
        )
    raise RuntimeError(
        "WRIST_PRECLOSE_GATE_NO_DATA: {}".format(last_error or "no synchronized frame")
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-topic", default=DEFAULT_RGB_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--info-topic", default=DEFAULT_INFO_TOPIC)
    parser.add_argument(
        "--camera-frame",
        default="",
        help="TF frame whose +Z axis matches the right image projection",
    )
    parser.add_argument("--target-topic", default=DEFAULT_TARGET_TOPIC)
    parser.add_argument("--target-param", default=DEFAULT_TARGET_PARAM)
    parser.add_argument("--debug-topic", default=DEFAULT_DEBUG_TOPIC)
    parser.add_argument("--left-finger-frame", default=DEFAULT_LEFT_FINGER_FRAME)
    parser.add_argument("--right-finger-frame", default=DEFAULT_RIGHT_FINGER_FRAME)
    parser.add_argument("--gripper-base-frame", default=DEFAULT_GRIPPER_BASE_FRAME)
    parser.add_argument("--tcp-extension", type=float, default=0.045)
    parser.add_argument("--roi-radius", type=int, default=90)
    parser.add_argument("--depth-band", type=float, default=35.0)
    parser.add_argument("--minimum-component-pixels", type=int, default=30)
    parser.add_argument("--corridor-radius", type=int, default=8)
    parser.add_argument("--segment-tolerance", type=float, default=0.018)
    parser.add_argument("--tcp-tolerance", type=float, default=0.022)
    parser.add_argument("--minimum-corridor-pixels", type=int, default=8)
    parser.add_argument("--maximum-obstacle-ratio", type=float, default=0.20)
    parser.add_argument("--required-frames", type=int, default=3)
    parser.add_argument("--maximum-target-spread", type=float, default=0.008)
    parser.add_argument("--maximum-tcp-spread", type=float, default=0.006)
    parser.add_argument("--timeout", type=float, default=15.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_ros(args)


if __name__ == "__main__":
    raise SystemExit(main())
