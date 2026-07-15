#!/usr/bin/env python3
"""Scene 3 head-vision node that preserves the selected tray identity.

The ordinary Scene3 vision node still detects every tray.  This wrapper changes
only the final candidate selection: it chooses the candidate whose legal RGB-D
world coordinate is nearest to the per-run locked target coordinate.

No seed, image pixel, or scene coordinate is embedded in this file.  A fresh
``locked_target_odom_xyz`` value must be produced for every selected tray.
"""

import argparse
import math
import time


DEFAULT_LOCK_PARAM = (
    "/challenge_cup_task_template/scene3/locked_target_odom_xyz"
)
DEFAULT_MAX_DISTANCE_M = 0.12


def parse_xyz(value):
    """Return a finite XYZ tuple, or None when a ROS parameter is invalid."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        xyz = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in xyz):
        return None
    return xyz


def candidate_world_xyz(candidate):
    """Extract a candidate's transformed world point without importing ROS."""
    point_stamped = candidate.get("world_point")
    if point_stamped is None or not hasattr(point_stamped, "point"):
        return None
    point = point_stamped.point
    return parse_xyz((point.x, point.y, point.z))


def select_locked_candidate(candidates, lock_xyz, maximum_distance_m):
    """Select the nearest world-space candidate and fail closed when too far.

    Returns ``(candidate, distance, reason)``.  Confidence and box area are used
    only as deterministic tie-breakers; identity distance always has priority.
    """
    lock_xyz = parse_xyz(lock_xyz)
    if lock_xyz is None:
        return None, None, "invalid target lock"
    try:
        maximum_distance_m = float(maximum_distance_m)
    except (TypeError, ValueError):
        return None, None, "invalid maximum distance"
    if not math.isfinite(maximum_distance_m) or maximum_distance_m <= 0.0:
        return None, None, "invalid maximum distance"

    scored = []
    for candidate in candidates:
        world_xyz = candidate_world_xyz(candidate)
        if world_xyz is None:
            continue
        distance = math.sqrt(sum(
            (world_xyz[axis] - lock_xyz[axis]) ** 2
            for axis in range(3)
        ))
        confidence = float(candidate.get("confidence", 0.0))
        bbox = candidate.get("bbox", (0, 0, 0, 0))
        area = float(bbox[2] * bbox[3]) if len(bbox) >= 4 else 0.0
        scored.append((distance, -confidence, -area, candidate))

    if not scored:
        return None, None, "no candidate has a world coordinate"

    distance, _, _, selected = min(scored, key=lambda item: item[:3])
    if distance > maximum_distance_m:
        return None, distance, "nearest candidate is outside identity gate"

    selected["target_lock_distance_m"] = float(distance)
    selected["target_lock_world_xyz"] = lock_xyz
    return selected, float(distance), "ok"


class LockedTargetSelectionMixin(object):
    """Mixin placed before Scene3VisionDebugger in the runtime MRO."""

    def __init__(self):
        # Defaults exist before the base class starts ROS subscribers, avoiding a
        # callback race during node startup.
        self.target_lock_param = DEFAULT_LOCK_PARAM
        self.target_lock_max_distance_m = DEFAULT_MAX_DISTANCE_M
        self.require_target_lock = True
        self._last_target_lock_log_time = 0.0
        super(LockedTargetSelectionMixin, self).__init__()

        self.target_lock_param = self.rospy.get_param(
            "~target_lock_param",
            DEFAULT_LOCK_PARAM,
        )
        self.target_lock_max_distance_m = float(self.rospy.get_param(
            "~target_lock_max_distance_m",
            DEFAULT_MAX_DISTANCE_M,
        ))
        self.require_target_lock = bool(self.rospy.get_param(
            "~require_target_lock",
            True,
        ))
        self.rospy.loginfo(
            "scene3 locked vision: lock_param=%s max_distance=%.3fm required=%s",
            self.target_lock_param,
            self.target_lock_max_distance_m,
            self.require_target_lock,
        )

    def _detect_tray(self, image_bgr, depth_image, frame_id):
        ordinary_best, full_mask, candidates = super(
            LockedTargetSelectionMixin,
            self,
        )._detect_tray(image_bgr, depth_image, frame_id)

        raw_lock = self.rospy.get_param(self.target_lock_param, None)
        lock_xyz = parse_xyz(raw_lock)
        if lock_xyz is None:
            self.rospy.logwarn_throttle(
                2.0,
                "scene3 locked vision: valid target identity is not available",
            )
            if self.require_target_lock:
                return None, full_mask, candidates
            return ordinary_best, full_mask, candidates

        selected, distance, reason = select_locked_candidate(
            candidates,
            lock_xyz,
            self.target_lock_max_distance_m,
        )
        if selected is None:
            distance_text = "n/a" if distance is None else "{:.3f}m".format(distance)
            self.rospy.logwarn_throttle(
                2.0,
                "scene3 locked vision: target blocked (%s, nearest=%s)",
                reason,
                distance_text,
            )
            return None, full_mask, candidates

        now = time.time()
        if now - self._last_target_lock_log_time > 1.0:
            world_xyz = candidate_world_xyz(selected)
            self.rospy.loginfo(
                "scene3 locked vision: SAME_TARGET distance=%.3fm odom=(%.3f, %.3f, %.3f) candidates=%d",
                distance,
                world_xyz[0],
                world_xyz[1],
                world_xyz[2],
                len(candidates),
            )
            self._last_target_lock_log_time = now
        return selected, full_mask, candidates


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Scene3 vision with per-run world-coordinate identity lock",
    )
    parser.add_argument(
        "--node-name",
        default="scene3_locked_target_vision",
        help="ROS node name",
    )
    args, _ = parser.parse_known_args(argv)

    import rospy
    from scene3_vision_debug import Scene3VisionDebugger

    class LockedTargetVision(
        LockedTargetSelectionMixin,
        Scene3VisionDebugger,
    ):
        pass

    rospy.init_node(args.node_name, anonymous=False)
    LockedTargetVision()
    rospy.loginfo("scene3 locked vision: node started")
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
