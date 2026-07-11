#!/usr/bin/env python3
"""
Convert a tray 2D candidate + depth image into camera-frame XYZ JSON.

This is an offline Day2 bridge script. It does not subscribe to ROS; it reads
saved RGB/depth files and produces the JSON that the action module should use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np


def parse_pair(text: str) -> Tuple[int, int]:
    vals = [int(float(x)) for x in text.split(",")]
    if len(vals) != 2:
        raise ValueError("expected u,v")
    return vals[0], vals[1]


def parse_bbox(text: str) -> Tuple[int, int, int, int]:
    vals = [int(float(x)) for x in text.split(",")]
    if len(vals) != 4:
        raise ValueError("expected x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def load_camera_info(path: str | None, fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float, float]:
    if not path:
        return fx, fy, cx, cy
    info = json.loads(Path(path).read_text(encoding="utf-8"))
    if "K" in info and len(info["K"]) >= 6:
        k = info["K"]
        return float(k[0]), float(k[4]), float(k[2]), float(k[5])
    if all(key in info for key in ("fx", "fy", "cx", "cy")):
        return float(info["fx"]), float(info["fy"]), float(info["cx"]), float(info["cy"])
    raise ValueError(f"unsupported camera_info JSON format: {path}")


def median_depth_meters(depth: np.ndarray, u: int, v: int, radius: int, depth_scale: float) -> float:
    h, w = depth.shape[:2]
    x1 = max(0, u - radius)
    x2 = min(w, u + radius + 1)
    y1 = max(0, v - radius)
    y2 = min(h, v + radius + 1)
    patch = depth[y1:y2, x1:x2].astype(np.float32)
    valid = np.isfinite(patch) & (patch > 0)
    if not np.any(valid):
        raise ValueError(f"no valid depth around center=({u},{v}), radius={radius}")
    value = float(np.median(patch[valid]))
    if np.issubdtype(depth.dtype, np.integer):
        value *= depth_scale
    return value


def pixel_to_camera(u: int, v: int, z: float, fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float]:
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return float(x), float(y), float(z)


def draw_result(rgb_path: Path, bbox: Tuple[int, int, int, int], center: Tuple[int, int], xyz: Tuple[float, float, float], output: Path) -> None:
    image = cv2.imread(str(rgb_path))
    if image is None:
        return
    x1, y1, x2, y2 = bbox
    u, v = center
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(image, (u, v), 5, (0, 0, 255), -1)
    text = f"xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})m"
    cv2.putText(image, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True, help=".npy depth image saved by save_depth_image.py")
    parser.add_argument("--center", required=True, help="u,v pixel center, for example 640,368")
    parser.add_argument("--bbox", required=True, help="x1,y1,x2,y2 bbox")
    parser.add_argument("--level", default="upper", choices=["upper", "lower", "unknown"])
    parser.add_argument("--camera-info", default=None, help="optional camera_info JSON with K or fx/fy/cx/cy")
    parser.add_argument("--fx", type=float, default=392.871)
    parser.add_argument("--fy", type=float, default=392.871)
    parser.add_argument("--cx", type=float, default=640.0)
    parser.add_argument("--cy", type=float, default=360.0)
    parser.add_argument("--depth-scale", type=float, default=0.001, help="scale integer depth units to meters, default mm->m")
    parser.add_argument("--radius", type=int, default=5, help="median depth window radius")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-vis", default=None)
    args = parser.parse_args()

    rgb_path = Path(args.rgb)
    depth_path = Path(args.depth)
    depth = np.load(str(depth_path))
    u, v = parse_pair(args.center)
    bbox = parse_bbox(args.bbox)
    fx, fy, cx, cy = load_camera_info(args.camera_info, args.fx, args.fy, args.cx, args.cy)
    z = median_depth_meters(depth, u, v, args.radius, args.depth_scale)
    xyz = pixel_to_camera(u, v, z, fx, fy, cx, cy)

    result: dict[str, Any] = {
        "scene": "scene3",
        "object": "smt_tray",
        "level": args.level,
        "bbox": list(bbox),
        "center_pixel": [u, v],
        "depth": z,
        "camera_xyz": list(xyz),
        "grasp_hint": "front_center",
        "confidence": 0.5,
        "camera_model": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "depth_file": str(depth_path),
        "rgb_file": str(rgb_path),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved_json={output_json}")

    if args.output_vis:
        output_vis = Path(args.output_vis)
        draw_result(rgb_path, bbox, (u, v), xyz, output_vis)
        print(f"saved_vis={output_vis}")


if __name__ == "__main__":
    main()
