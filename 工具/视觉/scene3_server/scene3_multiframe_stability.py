#!/usr/bin/env python3
import argparse
import glob
import itertools
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", action="append", default=[])
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--capture-count", type=int, default=3)
    parser.add_argument("--frame-delay", type=float, default=0.35)
    parser.add_argument(
        "--perception-script",
        default=str(Path(__file__).with_name("scene3_upper_tray_perception.py")),
    )
    parser.add_argument("--perception-arg", action="append", default=[])
    parser.add_argument("--template")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=3)
    parser.add_argument("--minimum-frames", type=int, default=3)
    parser.add_argument("--minimum-selection-score", type=float, default=0.60)
    parser.add_argument("--minimum-depth-shape-score", type=float, default=0.75)
    parser.add_argument("--maximum-pixel-spread", type=float, default=12.0)
    parser.add_argument("--maximum-position-spread", type=float, default=0.02)
    return parser.parse_args()


def raw_xyz(tray):
    values = tray.get("base_link_xyz_raw_m", tray.get("base_link_xyz_m"))
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("tray does not contain a valid base_link XYZ")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("tray base_link XYZ contains a non-finite value")
    return result


def center_pixel(tray):
    values = tray.get("object_center_pixel", tray.get("center_pixel"))
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("tray does not contain a valid center pixel")
    result = [float(values[0]), float(values[1])]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("tray center pixel contains a non-finite value")
    return result


def valid_bbox(tray, key):
    values = tray.get(key)
    if not isinstance(values, list) or len(values) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    return bool(
        all(math.isfinite(value) for value in (x1, y1, x2, y2))
        and x2 > x1
        and y2 > y1
    )


def validate_frame(data, frame_index, args):
    errors = []
    trays = data.get("upper_trays", [])
    if data.get("status") != "ok":
        errors.append("frame {} status is {}".format(frame_index, data.get("status")))
    if len(trays) != args.expected_count:
        errors.append(
            "frame {} count is {}, expected {}".format(
                frame_index, len(trays), args.expected_count
            )
        )
    for tray_index, tray in enumerate(trays):
        label = "frame {} tray {}".format(frame_index, tray_index)
        if tray.get("geometry_source") != "rgbd_vertical":
            errors.append("{} lacks RGB-D geometry".format(label))
        if not valid_bbox(tray, "object_bbox"):
            errors.append("{} lacks a valid object_bbox".format(label))
        if tray.get("accepted") is False:
            errors.append("{} is not an accepted V5 detection".format(label))
        if float(tray.get("selection_score", 0.0)) < args.minimum_selection_score:
            errors.append("{} selection score is too low".format(label))
        if (
            float(tray.get("depth_shape_score", 0.0))
            < args.minimum_depth_shape_score
        ):
            errors.append("{} depth shape score is too low".format(label))
        try:
            raw_xyz(tray)
            center_pixel(tray)
        except ValueError as error:
            errors.append("{}: {}".format(label, error))
        try:
            depth_m = float(tray.get("depth_m"))
            if not math.isfinite(depth_m) or depth_m <= 0.0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("{} lacks a valid positive depth".format(label))
    return errors


def distance_2d(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def distance_3d(first, second):
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def best_assignment(reference, candidates, args):
    count = len(reference)
    best = None
    for permutation in itertools.permutations(range(count)):
        cost = 0.0
        for reference_index, candidate_index in enumerate(permutation):
            pixel_distance = distance_2d(
                center_pixel(reference[reference_index]),
                center_pixel(candidates[candidate_index]),
            )
            position_distance = distance_3d(
                raw_xyz(reference[reference_index]),
                raw_xyz(candidates[candidate_index]),
            )
            cost += pixel_distance / max(args.maximum_pixel_spread, 1e-6)
            cost += position_distance / max(args.maximum_position_spread, 1e-6)
        if best is None or cost < best[0]:
            best = (cost, permutation)
    return best[1]


def median_vector(values):
    return [float(statistics.median(axis)) for axis in zip(*values)]


def median_int_vector(values):
    return [int(round(value)) for value in median_vector(values)]


def maximum_pairwise(values, distance_function):
    maximum = 0.0
    for first, second in itertools.combinations(values, 2):
        maximum = max(maximum, distance_function(first, second))
    return maximum


def axis_ranges(values):
    return [float(max(axis) - min(axis)) for axis in zip(*values)]


def aggregate_track(track, track_index, args):
    centers = [center_pixel(tray) for tray in track]
    positions = [raw_xyz(tray) for tray in track]
    object_boxes = [tray["object_bbox"] for tray in track]
    detection_boxes = [
        tray.get("detection_bbox", tray["object_bbox"]) for tray in track
    ]
    pixel_spread = maximum_pairwise(centers, distance_2d)
    position_spread = maximum_pairwise(positions, distance_3d)
    stable = bool(
        pixel_spread <= args.maximum_pixel_spread
        and position_spread <= args.maximum_position_spread
    )
    median_position = median_vector(positions)
    median_center = median_int_vector(centers)
    payload = {
        "id": "upper_x{}".format(track_index),
        "accepted": True,
        "geometry_source": "rgbd_vertical",
        "proposal_sources": sorted(
            {
                str(source)
                for tray in track
                for source in tray.get("proposal_sources", [])
            }
        ),
        "center_pixel": median_center,
        "object_center_pixel": median_center,
        "object_bbox": median_int_vector(object_boxes),
        "bbox": median_int_vector(object_boxes),
        "detection_bbox": median_int_vector(detection_boxes),
        "depth_m": float(
            statistics.median(float(tray["depth_m"]) for tray in track)
        ),
        "base_link_xyz_raw_m": median_position,
        "base_link_xyz_m": median_position,
        "selection_score": float(
            min(float(tray["selection_score"]) for tray in track)
        ),
        "depth_shape_score": float(
            min(float(tray["depth_shape_score"]) for tray in track)
        ),
        "template_score": float(
            min(float(tray.get("template_score", 0.0)) for tray in track)
        ),
        "temporal_metrics": {
            "stable": stable,
            "frame_count": len(track),
            "maximum_pixel_spread": pixel_spread,
            "pixel_axis_ranges": axis_ranges(centers),
            "maximum_position_spread_m": position_spread,
            "position_axis_ranges_m": axis_ranges(positions),
            "minimum_selection_score": float(
                min(float(tray["selection_score"]) for tray in track)
            ),
            "minimum_depth_shape_score": float(
                min(float(tray["depth_shape_score"]) for tray in track)
            ),
        },
    }
    camera_positions = [tray.get("camera_xyz_m") for tray in track]
    if all(isinstance(values, list) and len(values) == 3 for values in camera_positions):
        payload["camera_xyz_m"] = median_vector(camera_positions)
    corrected_positions = [tray.get("base_link_xyz_corrected_m") for tray in track]
    if all(
        isinstance(values, list) and len(values) == 3
        for values in corrected_positions
    ):
        payload["base_link_xyz_corrected_m"] = median_vector(corrected_positions)
    return payload


def build_consensus(frames, source_files, args):
    frame_errors = []
    if len(frames) < args.minimum_frames:
        frame_errors.append(
            "received {} frames, need at least {}".format(
                len(frames), args.minimum_frames
            )
        )
    for frame_index, frame in enumerate(frames):
        frame_errors.extend(validate_frame(frame, frame_index, args))
    source_frames = {frame.get("source_frame") for frame in frames}
    target_frames = {frame.get("target_frame") for frame in frames}
    if len(source_frames) > 1:
        frame_errors.append("source camera frame changed between samples")
    if len(target_frames) > 1:
        frame_errors.append("target coordinate frame changed between samples")
    base_payload = {
        "algorithm": "scene3_temporal_consensus_v1",
        "source_algorithms": sorted(
            {str(frame.get("algorithm")) for frame in frames if frame.get("algorithm")}
        ),
        "source_files": [str(path) for path in source_files],
        "frame_count": len(frames),
        "thresholds": {
            "expected_count": args.expected_count,
            "minimum_frames": args.minimum_frames,
            "minimum_selection_score": args.minimum_selection_score,
            "minimum_depth_shape_score": args.minimum_depth_shape_score,
            "maximum_pixel_spread": args.maximum_pixel_spread,
            "maximum_position_spread_m": args.maximum_position_spread,
        },
    }
    if frame_errors:
        base_payload.update(
            {
                "status": "rejected_input",
                "temporal_validation": {
                    "passed": False,
                    "errors": frame_errors,
                },
                "upper_trays": [],
                "tracks": [],
            }
        )
        return base_payload

    reference = sorted(frames[0]["upper_trays"], key=lambda tray: center_pixel(tray)[0])
    tracks = [[tray] for tray in reference]
    for frame in frames[1:]:
        candidates = frame["upper_trays"]
        assignment = best_assignment(reference, candidates, args)
        for reference_index, candidate_index in enumerate(assignment):
            tracks[reference_index].append(candidates[candidate_index])

    aggregated = [
        aggregate_track(track, track_index, args)
        for track_index, track in enumerate(tracks)
    ]
    aggregated.sort(key=lambda tray: tray["center_pixel"][0])
    for index, tray in enumerate(aggregated):
        tray["id"] = "upper_x{}".format(index)
    passed = all(tray["temporal_metrics"]["stable"] for tray in aggregated)
    base_payload.update(
        {
            "status": "ok" if passed else "unstable",
            "source_frame": frames[0].get("source_frame"),
            "target_frame": frames[0].get("target_frame"),
            "temporal_validation": {
                "passed": passed,
                "errors": [] if passed else ["one or more tray tracks are unstable"],
            },
            "upper_trays": aggregated if passed else [],
            "tracks": [
                {
                    "id": tray["id"],
                    "center_pixel": tray["center_pixel"],
                    "base_link_xyz_raw_m": tray["base_link_xyz_raw_m"],
                    "temporal_metrics": tray["temporal_metrics"],
                }
                for tray in aggregated
            ],
        }
    )
    return base_payload


def capture_frames(args, output_dir):
    if not args.template:
        raise ValueError("--template is required with --capture")
    script = Path(args.perception_script)
    if not script.is_file():
        raise ValueError("perception script does not exist: {}".format(script))
    frames = []
    files = []
    for index in range(args.capture_count):
        frame_dir = output_dir / "frames" / "frame_{:03d}".format(index + 1)
        frame_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(script),
            "--template",
            args.template,
            "--output-dir",
            str(frame_dir),
        ] + list(args.perception_arg)
        log_path = frame_dir / "perception.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        json_path = frame_dir / "upper_trays.json"
        if result.returncode != 0 or not json_path.is_file():
            raise RuntimeError(
                "perception frame {} failed; inspect {}".format(index + 1, log_path)
            )
        frames.append(json.loads(json_path.read_text(encoding="utf-8")))
        files.append(json_path)
        if index + 1 < args.capture_count:
            time.sleep(args.frame_delay)
    return frames, files


def input_paths(args):
    paths = [Path(path) for path in args.input_json]
    for pattern in args.input_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def main():
    args = parse_args()
    if args.minimum_frames < 2:
        raise ValueError("minimum frames must be at least 2")
    if args.capture and args.capture_count < args.minimum_frames:
        raise ValueError("capture count must be at least minimum frames")
    if args.expected_count < 1:
        raise ValueError("expected count must be positive")
    if args.maximum_pixel_spread <= 0.0 or args.maximum_position_spread <= 0.0:
        raise ValueError("stability spreads must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.capture:
        if args.input_json or args.input_glob:
            raise ValueError("capture mode cannot be combined with input JSON files")
        frames, files = capture_frames(args, output_dir)
    else:
        files = input_paths(args)
        if not files:
            raise ValueError("provide --capture or at least one input JSON file")
        frames = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    payload = build_consensus(frames, files, args)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    output_path = output_dir / "stable_upper_trays.json"
    output_path.write_text(json_text, encoding="utf-8")
    print(json_text)
    print("saved_json={}".format(output_path))
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
