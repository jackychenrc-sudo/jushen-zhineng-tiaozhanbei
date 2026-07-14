#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scene3 visual-servo grasping built on the senior movement and vision code.

The existing detector calls the YOLO box centre a grasp point and uses a low
depth percentile.  A shelf rail can therefore win the depth estimate.  This
module keeps the detector and movement flow unchanged, but refines the final
grasp point from the dominant tray-depth surface and closes the claw only when
the actual gripper TCP is visually aligned.

Default mode is observation-only.  Real arm/claw commands require both
``--execute`` and the exact confirmation token ``VISUAL_SERVO_GRASP``.
"""

from __future__ import print_function

import argparse
import math
import time

import numpy as np


EXECUTION_CONFIRMATION = "VISUAL_SERVO_GRASP"
DEFAULT_LEFT_FINGER_FRAME = "right_gripper_left_inner_knuckle"
DEFAULT_RIGHT_FINGER_FRAME = "right_gripper_right_inner_knuckle"
DEFAULT_GRIPPER_BASE_FRAME = "right_gripper_base"


def _validate_bbox(depth_image, bbox):
    if depth_image is None or np.asarray(depth_image).ndim < 2:
        raise ValueError("depth image must be two-dimensional")
    x, y, width, height = [int(value) for value in bbox]
    image_height, image_width = np.asarray(depth_image).shape[:2]
    x0 = max(0, min(image_width, x))
    y0 = max(0, min(image_height, y))
    x1 = max(x0, min(image_width, x + max(0, width)))
    y1 = max(y0, min(image_height, y + max(0, height)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox does not overlap the depth image")
    return x0, y0, x1, y1


def dominant_surface_mask(
    depth_image,
    bbox,
    depth_band_mm=25.0,
    histogram_bin_mm=10.0,
    central_fraction=0.60,
    minimum_pixels=24,
):
    """Return the dominant object-depth mask inside a YOLO box.

    The mode is estimated from the central part of the box, where the tray has
    much more area than a thin foreground shelf rail.  The returned mask is in
    bbox-local coordinates and is independent of absolute image position.
    """

    depth = np.asarray(depth_image, dtype=float)
    x0, y0, x1, y1 = _validate_bbox(depth, bbox)
    crop = depth[y0:y1, x0:x1]
    crop_height, crop_width = crop.shape[:2]

    fraction = min(1.0, max(0.20, float(central_fraction)))
    margin_x = int(round(0.5 * (1.0 - fraction) * crop_width))
    margin_y = int(round(0.5 * (1.0 - fraction) * crop_height))
    centre = crop[
        margin_y : max(margin_y + 1, crop_height - margin_y),
        margin_x : max(margin_x + 1, crop_width - margin_x),
    ]
    valid_centre = centre[(centre > 50.0) & (centre < 10000.0)]
    if valid_centre.size < int(minimum_pixels):
        valid_centre = crop[(crop > 50.0) & (crop < 10000.0)]
    if valid_centre.size < int(minimum_pixels):
        raise ValueError("not enough valid depth pixels in tray bbox")

    bin_width = max(1.0, float(histogram_bin_mm))
    quantized = np.rint(valid_centre / bin_width).astype(np.int64)
    values, counts = np.unique(quantized, return_counts=True)
    winning_bin = int(values[int(np.argmax(counts))])
    near_mode = valid_centre[np.abs(valid_centre / bin_width - winning_bin) <= 0.75]
    if near_mode.size == 0:
        mode_depth_mm = float(winning_bin * bin_width)
    else:
        mode_depth_mm = float(np.median(near_mode))

    band = max(5.0, float(depth_band_mm))
    surface_mask = (
        (crop > 50.0)
        & (crop < 10000.0)
        & (np.abs(crop - mode_depth_mm) <= band)
    )
    if int(np.count_nonzero(surface_mask)) < int(minimum_pixels):
        raise ValueError("dominant tray surface is too small")
    return surface_mask, mode_depth_mm, (x0, y0, x1, y1)


def binary_boundary(mask):
    """Return the four-neighbour inner boundary of a binary mask."""

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    interior = (
        padded[0:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, 0:-2]
        & padded[1:-1, 2:]
    )
    return source & ~interior


def _nearest_mask_pixel(mask, desired_x, desired_y):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("surface mask is empty")
    distances = (xs.astype(float) - float(desired_x)) ** 2 + (
        ys.astype(float) - float(desired_y)
    ) ** 2
    index = int(np.argmin(distances))
    return int(xs[index]), int(ys[index])


def select_grasp_pixel(
    depth_image,
    bbox,
    tcp_pixel,
    depth_band_mm=25.0,
    inward_ratio=0.14,
):
    """Select an accessible tray-edge point nearest the projected gripper TCP.

    The edge is derived from the dominant depth surface, not from a fixed pixel
    or seed.  Moving the selected boundary point slightly toward the surface
    centroid avoids attempting to pinch a one-pixel silhouette.
    """

    mask, mode_depth_mm, bounds = dominant_surface_mask(
        depth_image,
        bbox,
        depth_band_mm=depth_band_mm,
    )
    x0, y0, _, _ = bounds
    boundary = binary_boundary(mask)
    boundary_ys, boundary_xs = np.nonzero(boundary)
    if boundary_xs.size < 8:
        raise ValueError("tray surface boundary is too small")

    if tcp_pixel is None:
        target_x = float(np.median(boundary_xs))
        target_y = float(np.median(boundary_ys))
    else:
        target_x = float(tcp_pixel[0]) - float(x0)
        target_y = float(tcp_pixel[1]) - float(y0)

    normalized_x = (boundary_xs - target_x) / max(1.0, float(mask.shape[1]))
    normalized_y = (boundary_ys - target_y) / max(1.0, float(mask.shape[0]))
    boundary_index = int(np.argmin(normalized_x ** 2 + normalized_y ** 2))
    edge_x = float(boundary_xs[boundary_index])
    edge_y = float(boundary_ys[boundary_index])

    surface_ys, surface_xs = np.nonzero(mask)
    centroid_x = float(np.median(surface_xs))
    centroid_y = float(np.median(surface_ys))
    ratio = min(0.35, max(0.0, float(inward_ratio)))
    desired_x = edge_x + ratio * (centroid_x - edge_x)
    desired_y = edge_y + ratio * (centroid_y - edge_y)
    local_x, local_y = _nearest_mask_pixel(mask, desired_x, desired_y)

    depth = np.asarray(depth_image, dtype=float)
    full_x = int(x0 + local_x)
    full_y = int(y0 + local_y)
    radius = 3
    patch = depth[
        max(0, full_y - radius) : full_y + radius + 1,
        max(0, full_x - radius) : full_x + radius + 1,
    ]
    valid_patch = patch[
        (patch > 50.0)
        & (patch < 10000.0)
        & (np.abs(patch - mode_depth_mm) <= max(5.0, float(depth_band_mm)))
    ]
    point_depth_mm = (
        float(np.median(valid_patch)) if valid_patch.size else float(mode_depth_mm)
    )
    return {
        "pixel": (full_x, full_y),
        "depth_mm": point_depth_mm,
        "mode_depth_mm": float(mode_depth_mm),
        "edge_pixel": (int(round(x0 + edge_x)), int(round(y0 + edge_y))),
        "surface_pixels": int(np.count_nonzero(mask)),
    }


def gripper_tcp_from_origins(left_xyz, right_xyz, base_xyz, extension_m=0.045):
    """Estimate the pinch TCP from legal TF link origins.

    The midpoint of the two inner-knuckle origins defines the jaw centre.  The
    fixed extension follows the physical gripper axis from ``gripper_base`` to
    that midpoint.  It is a tool calibration, not a scene coordinate.
    """

    left = np.asarray(left_xyz, dtype=float)
    right = np.asarray(right_xyz, dtype=float)
    base = np.asarray(base_xyz, dtype=float)
    if left.shape != (3,) or right.shape != (3,) or base.shape != (3,):
        raise ValueError("TF origins must be three-dimensional")
    midpoint = 0.5 * (left + right)
    axis = midpoint - base
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        raise ValueError("cannot infer gripper forward axis")
    return midpoint + max(0.0, float(extension_m)) * axis / axis_norm


def clamp_cartesian_step(error_xyz, maximum_step_m=0.01):
    error = np.asarray(error_xyz, dtype=float)
    norm = float(np.linalg.norm(error))
    maximum = max(1e-4, float(maximum_step_m))
    if norm <= maximum:
        return error.copy()
    return error * (maximum / norm)


def grasp_gate(error_xyz, lateral_tolerance_m=0.008, vertical_tolerance_m=0.008,
               forward_tolerance_m=0.012):
    error = np.asarray(error_xyz, dtype=float)
    if error.shape != (3,):
        raise ValueError("error must be xyz")
    return bool(
        abs(float(error[0])) <= float(forward_tolerance_m)
        and abs(float(error[1])) <= float(lateral_tolerance_m)
        and abs(float(error[2])) <= float(vertical_tolerance_m)
    )


def _translation_xyz(transform):
    value = transform.transform.translation
    return np.array([value.x, value.y, value.z], dtype=float)


def _point_xyz(point_message):
    return np.array(
        [point_message.point.x, point_message.point.y, point_message.point.z],
        dtype=float,
    )


def run_ros(args):
    import rospy
    import tf2_geometry_msgs  # noqa: F401
    from geometry_msgs.msg import PointStamped, Twist
    from sensor_msgs.msg import JointState

    from challenge_task_3 import Scene3Task
    from scene3_vision_debug import Scene3VisionDebugger

    rospy.init_node("scene3_visual_grasp_servo", anonymous=True)

    # Keep the senior detector alive, but publish refined outputs separately.
    rospy.set_param("~model_path", args.model_path)
    rospy.set_param("~device", args.device)
    rospy.set_param("~confidence_threshold", args.confidence)
    rospy.set_param(
        "~debug_topic", "/challenge_cup_task_template/scene3/servo_debug/compressed"
    )
    rospy.set_param(
        "~grasp_point_camera_topic",
        "/challenge_cup_task_template/scene3/refined_grasp_point_camera",
    )
    rospy.set_param(
        "~grasp_point_base_topic",
        "/challenge_cup_task_template/scene3/refined_grasp_point_base",
    )
    rospy.set_param(
        "~grasp_point_odom_topic",
        "/challenge_cup_task_template/scene3/refined_grasp_point_odom",
    )
    rospy.set_param(
        "~mask_topic", "/challenge_cup_task_template/scene3/servo_mask/compressed"
    )

    class RefinedVision(Scene3VisionDebugger):
        def __init__(self):
            self.left_finger_frame = args.left_finger_frame
            self.right_finger_frame = args.right_finger_frame
            self.gripper_base_frame = args.gripper_base_frame
            self.tcp_extension_m = args.tcp_extension
            self.depth_band_mm = args.depth_band_mm
            self.inward_ratio = args.inward_ratio
            self.last_refined_wall_time = 0.0
            super(RefinedVision, self).__init__()
            self.tcp_point_pub = rospy.Publisher(
                "/challenge_cup_task_template/scene3/gripper_tcp_base",
                PointStamped,
                queue_size=1,
            )

        def lookup_tcp(self, target_frame):
            transforms = []
            for frame in (
                self.left_finger_frame,
                self.right_finger_frame,
                self.gripper_base_frame,
            ):
                transforms.append(
                    self.tf_buffer.lookup_transform(
                        target_frame,
                        frame,
                        rospy.Time(0),
                        rospy.Duration(0.20),
                    )
                )
            return gripper_tcp_from_origins(
                _translation_xyz(transforms[0]),
                _translation_xyz(transforms[1]),
                _translation_xyz(transforms[2]),
                extension_m=self.tcp_extension_m,
            )

        def _project_tcp(self, camera_frame):
            if self.last_head_cam_info is None:
                return None
            tcp = self.lookup_tcp(camera_frame)
            if tcp[2] <= 0.05:
                return None
            camera_info = self.last_head_cam_info
            fx, fy = float(camera_info.K[0]), float(camera_info.K[4])
            cx, cy = float(camera_info.K[2]), float(camera_info.K[5])
            if fx <= 0.0 or fy <= 0.0:
                return None
            return (
                fx * float(tcp[0]) / float(tcp[2]) + cx,
                fy * float(tcp[1]) / float(tcp[2]) + cy,
            )

        def _detect_tray(self, image_bgr, depth_image, frame_id):
            detection, full_mask, candidates = super(RefinedVision, self)._detect_tray(
                image_bgr,
                depth_image,
                frame_id,
            )
            if detection is None:
                return detection, full_mask, candidates
            try:
                tcp_pixel = self._project_tcp(frame_id)
                refined = select_grasp_pixel(
                    depth_image,
                    detection["bbox"],
                    tcp_pixel=tcp_pixel,
                    depth_band_mm=self.depth_band_mm,
                    inward_ratio=self.inward_ratio,
                )
                xyz = self._pixel_to_camera_xyz(
                    refined["pixel"][0],
                    refined["pixel"][1],
                    refined["depth_mm"],
                )
                if xyz is None:
                    raise ValueError("camera intrinsics unavailable")
                camera_point = self._build_point_stamped(frame_id, xyz)
                base_point = self._transform_point_to_base(camera_point)
                world_point = self._transform_point_to_world(camera_point)
                if base_point is None:
                    raise ValueError("camera-to-base transform unavailable")
                result = dict(detection)
                result.update(
                    {
                        "camera_xyz": xyz,
                        "grasp_point_px": refined["pixel"],
                        "edge_point_px": refined["edge_pixel"],
                        "tcp_pixel": tcp_pixel,
                        "base_point": base_point,
                        "world_point": world_point,
                        "depth_mode_mm": refined["mode_depth_mm"],
                        "refined": True,
                    }
                )
                selection_point = world_point if world_point is not None else base_point
                result["selection_z"] = float(selection_point.point.z)
                result["selection_frame"] = str(selection_point.header.frame_id)
                self.last_refined_wall_time = time.time()
                return result, full_mask, candidates
            except Exception as exc:
                rospy.logwarn_throttle(1.0, "scene3 grasp refinement blocked: %s", exc)
                # Fail closed: never publish the old box centre as a refined point.
                return None, full_mask, candidates

        def _annotate_detection(self, image_bgr, detection):
            image = super(RefinedVision, self)._annotate_detection(
                image_bgr, detection
            )
            tcp_pixel = detection.get("tcp_pixel")
            edge_pixel = detection.get("edge_point_px")
            if edge_pixel is not None:
                self.cv2.circle(image, tuple(map(int, edge_pixel)), 6, (255, 0, 255), 2)
            if tcp_pixel is not None:
                tcp_draw = tuple(int(round(value)) for value in tcp_pixel)
                grasp_draw = tuple(map(int, detection["grasp_point_px"]))
                self.cv2.circle(image, tcp_draw, 7, (255, 0, 0), 2)
                self.cv2.line(image, tcp_draw, grasp_draw, (255, 255, 255), 2)
            return image

        def publish_tcp_base(self):
            tcp = self.lookup_tcp("base_link")
            message = PointStamped()
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = "base_link"
            message.point.x, message.point.y, message.point.z = map(float, tcp)
            self.tcp_point_pub.publish(message)
            return tcp

    vision = RefinedVision()
    arm_publisher = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    velocity_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    task = Scene3Task(velocity_publisher, arm_publisher, vision_model_path="")

    def observe(timeout=12.0, sample_count=3):
        deadline = time.time() + float(timeout)
        target_samples = []
        tcp_samples = []
        observed_stamps = set()
        while not rospy.is_shutdown() and time.time() < deadline:
            detection = vision.last_detection
            if detection is None or not detection.get("refined", False):
                rospy.sleep(0.05)
                continue
            point = detection.get("base_point")
            if point is None:
                rospy.sleep(0.05)
                continue
            stamp = (int(point.header.stamp.secs), int(point.header.stamp.nsecs))
            if stamp in observed_stamps:
                rospy.sleep(0.03)
                continue
            try:
                tcp = vision.publish_tcp_base()
            except Exception:
                rospy.sleep(0.05)
                continue
            observed_stamps.add(stamp)
            target_samples.append(_point_xyz(point))
            tcp_samples.append(tcp)
            if len(target_samples) >= int(sample_count):
                targets = np.asarray(target_samples, dtype=float)
                tcps = np.asarray(tcp_samples, dtype=float)
                target = np.median(targets, axis=0)
                tcp_median = np.median(tcps, axis=0)
                target_spread = float(
                    np.max(np.linalg.norm(targets - target, axis=1))
                )
                tcp_spread = float(
                    np.max(np.linalg.norm(tcps - tcp_median, axis=1))
                )
                if target_spread > args.maximum_target_spread:
                    raise RuntimeError(
                        "refined grasp point unstable: {:.4f}m".format(target_spread)
                    )
                if tcp_spread > args.maximum_tcp_spread:
                    raise RuntimeError(
                        "gripper TCP unstable: {:.4f}m".format(tcp_spread)
                    )
                return target, tcp_median, target_spread, tcp_spread
        raise RuntimeError("timed out waiting for refined visual grasp observations")

    print("绛夊緟涓夊抚绮剧偧鎶撳彇鐐瑰拰澶圭埅TCP锛涗笉浣跨敤鍥哄畾鍍忕礌鎴栧満鏅潗鏍?)
    target, tcp, target_spread, tcp_spread = observe()
    error = target - tcp
    print("绮剧偧鎶撳彇鐐?", np.round(target, 4).tolist())
    print("澶圭埅TCP:", np.round(tcp, 4).tolist())
    print("TCP璇樊:", np.round(error, 4).tolist(), "norm={:.4f}m".format(np.linalg.norm(error)))
    print("涓夊抚娉㈠姩: target={:.4f}m tcp={:.4f}m".format(target_spread, tcp_spread))

    if not args.execute:
        print("VISUAL_SERVO_DRY_RUN_OK锛氬彧瀹屾垚鎶撳彇鐐?TCP鏍￠獙锛屾湭鎺у埗鏈烘鑷傚拰澶圭埅")
        return 0
    if args.confirmation != EXECUTION_CONFIRMATION:
        raise RuntimeError(
            "execution blocked; pass --confirmation {}".format(EXECUTION_CONFIRMATION)
        )
    if float(np.linalg.norm(error)) > args.maximum_start_error:
        raise RuntimeError("starting TCP error exceeds safety gate")

    task.wait_for_arm_subscriber(timeout=8.0)
    if not task.set_arm_mode(2):
        raise RuntimeError("cannot enable senior arm external-control mode")
    if not task.open_claw():
        raise RuntimeError("cannot open claw")

    aligned_observations = 0
    previous_error_norm = float(np.linalg.norm(error))
    for iteration in range(1, args.maximum_iterations + 1):
        target, tcp, target_spread, tcp_spread = observe()
        error = target - tcp
        error_norm = float(np.linalg.norm(error))
        aligned = grasp_gate(
            error,
            lateral_tolerance_m=args.lateral_tolerance,
            vertical_tolerance_m=args.vertical_tolerance,
            forward_tolerance_m=args.forward_tolerance,
        )
        print(
            "瑙嗚闂幆{:02d}: target={} tcp={} error={} norm={:.4f}m gate={}".format(
                iteration,
                np.round(target, 4).tolist(),
                np.round(tcp, 4).tolist(),
                np.round(error, 4).tolist(),
                error_norm,
                aligned,
            )
        )
        if aligned:
            aligned_observations += 1
            if aligned_observations >= args.required_aligned_observations:
                if not task.close_claw():
                    raise RuntimeError("visual gate passed but claw close failed")
                print("VISUAL_SERVO_GRASP_OK锛氭寚灏朤CP杩炵画瀵瑰噯鍚庡凡闂埅锛涘皻鏈娊鍑?)
                return 0
            rospy.sleep(args.settle_seconds)
            continue

        aligned_observations = 0
        if iteration > 1 and error_norm > previous_error_norm + args.maximum_worsening:
            raise RuntimeError("visual error worsened beyond safety gate; stop with claw open")

        step = clamp_cartesian_step(error, args.maximum_cartesian_step)
        # Align lateral/vertical axes before inserting farther into the rack.
        if abs(error[1]) > args.coarse_axis_gate or abs(error[2]) > args.coarse_axis_gate:
            step[0] = min(0.0, float(step[0]))
            step = clamp_cartesian_step(step, args.maximum_cartesian_step)

        current_joints = task.read_current_arm_joints()
        current_poses = task.call_fk(current_joints)
        wrist = np.asarray(current_poses.right_pose.pos_xyz, dtype=float)
        wrist_target = wrist + step
        print(
            "  IK寰={}m wrist_target={}".format(
                np.round(step, 4).tolist(),
                np.round(wrist_target, 4).tolist(),
            )
        )
        task.move_right_hand(wrist_target.tolist(), duration=args.motion_seconds)
        rospy.sleep(args.settle_seconds)
        new_target, new_tcp, _, _ = observe()
        tcp_motion = float(np.linalg.norm(new_tcp - tcp))
        new_error_norm = float(np.linalg.norm(new_target - new_tcp))
        if tcp_motion > args.maximum_observed_tcp_motion:
            raise RuntimeError(
                "observed TCP motion {:.4f}m exceeded safety gate".format(tcp_motion)
            )
        if new_error_norm > error_norm + args.maximum_worsening:
            raise RuntimeError("post-motion visual error worsened; stop with claw open")
        previous_error_norm = new_error_norm

    raise RuntimeError("visual servo iteration limit reached; claw remains open")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--left-finger-frame", default=DEFAULT_LEFT_FINGER_FRAME)
    parser.add_argument("--right-finger-frame", default=DEFAULT_RIGHT_FINGER_FRAME)
    parser.add_argument("--gripper-base-frame", default=DEFAULT_GRIPPER_BASE_FRAME)
    parser.add_argument("--tcp-extension", type=float, default=0.045)
    parser.add_argument("--depth-band-mm", type=float, default=25.0)
    parser.add_argument("--inward-ratio", type=float, default=0.14)
    parser.add_argument("--maximum-target-spread", type=float, default=0.012)
    parser.add_argument("--maximum-tcp-spread", type=float, default=0.006)
    parser.add_argument("--maximum-start-error", type=float, default=0.25)
    parser.add_argument("--maximum-cartesian-step", type=float, default=0.010)
    parser.add_argument("--maximum-observed-tcp-motion", type=float, default=0.025)
    parser.add_argument("--maximum-worsening", type=float, default=0.006)
    parser.add_argument("--coarse-axis-gate", type=float, default=0.015)
    parser.add_argument("--lateral-tolerance", type=float, default=0.008)
    parser.add_argument("--vertical-tolerance", type=float, default=0.008)
    parser.add_argument("--forward-tolerance", type=float, default=0.012)
    parser.add_argument("--required-aligned-observations", type=int, default=3)
    parser.add_argument("--maximum-iterations", type=int, default=20)
    parser.add_argument("--motion-seconds", type=float, default=1.0)
    parser.add_argument("--settle-seconds", type=float, default=0.7)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_ros(args)


if __name__ == "__main__":
    raise SystemExit(main())

