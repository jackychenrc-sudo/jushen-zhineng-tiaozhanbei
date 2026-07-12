#!/usr/bin/env python3
"""
Offline Scene3 SMT tray labeler.

This script reads a saved RGB image and marks likely tray candidates clearly.
It is meant for fast Day2 visual debugging before the result is connected to
depth and robot motion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


Color = Tuple[int, int, int]
BBox = Tuple[int, int, int, int]
Roi = Tuple[float, float, float, float]


@dataclass
class TrayCandidate:
    candidate_id: str
    level: str
    bbox: BBox
    center_pixel: Tuple[int, int]
    area: float
    fill_ratio: float
    aspect_h_over_w: float
    score: float
    rank_in_level: int
    order_in_level_x: int


def parse_roi(text: str) -> Roi:
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 4:
        raise ValueError("--roi must be x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("--roi values must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1")
    return x1, y1, x2, y2


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def normalized_roi_to_bbox(roi: Roi, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = roi
    return clamp_bbox((int(width * x1), int(height * y1), int(width * x2), int(height * y2)), width, height)


def make_dark_mask(crop: np.ndarray, dark_threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    dark_gray = cv2.inRange(gray, 0, dark_threshold)
    dark_value = cv2.inRange(value, 0, min(255, dark_threshold + 25))
    low_saturation_dark = cv2.bitwise_and(dark_value, cv2.inRange(saturation, 0, 120))
    mask = cv2.bitwise_or(dark_gray, low_saturation_dark)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return mask


def contour_candidates(
    image: np.ndarray,
    roi: Roi,
    split_y: float,
    dark_threshold: int,
    min_area: int,
    max_area: int,
) -> List[TrayCandidate]:
    height, width = image.shape[:2]
    roi_x1, roi_y1, roi_x2, roi_y2 = normalized_roi_to_bbox(roi, width, height)
    crop = image[roi_y1:roi_y2, roi_x1:roi_x2]
    mask = make_dark_mask(crop, dark_threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw: List[TrayCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue

        fill_ratio = area / float(bw * bh)
        aspect = bh / float(max(bw, 1))
        if fill_ratio < 0.12:
            continue
        if not (0.45 <= aspect <= 9.0):
            continue
        if not (5 <= bw <= 180 and 14 <= bh <= 240):
            continue

        x1 = roi_x1 + bx
        y1 = roi_y1 + by
        x2 = roi_x1 + bx + bw
        y2 = roi_y1 + by + bh
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        level = "upper" if cy < int(height * split_y) else "lower"
        center_bonus = 1.0 - min(abs(cx - width / 2.0) / (width / 2.0), 1.0)
        vertical_bonus = min(aspect / 4.0, 1.0)
        size_bonus = min(area / 2200.0, 1.0)
        score = area * (1.0 + 0.45 * center_bonus + 0.35 * vertical_bonus + 0.20 * size_bonus)
        raw.append(
            TrayCandidate(
                candidate_id="",
                level=level,
                bbox=(x1, y1, x2, y2),
                center_pixel=(cx, cy),
                area=area,
                fill_ratio=fill_ratio,
                aspect_h_over_w=aspect,
                score=float(score),
                rank_in_level=-1,
                order_in_level_x=-1,
            )
        )

    return assign_candidate_ids(raw)


def assign_candidate_ids(candidates: Iterable[TrayCandidate]) -> List[TrayCandidate]:
    result: List[TrayCandidate] = []
    for level in ("upper", "lower"):
        level_candidates = [candidate for candidate in candidates if candidate.level == level]
        by_score = sorted(level_candidates, key=lambda item: item.score, reverse=True)
        score_rank: Dict[Tuple[int, int, int, int], int] = {
            candidate.bbox: index for index, candidate in enumerate(by_score)
        }
        by_x = sorted(level_candidates, key=lambda item: item.center_pixel[0])
        x_rank: Dict[Tuple[int, int, int, int], int] = {
            candidate.bbox: index for index, candidate in enumerate(by_x)
        }
        for candidate in by_score:
            candidate.rank_in_level = score_rank[candidate.bbox]
            candidate.order_in_level_x = x_rank[candidate.bbox]
            candidate.candidate_id = f"{level}_{candidate.rank_in_level}"
            result.append(candidate)
    return sorted(result, key=lambda item: (item.level != "upper", item.rank_in_level))


def draw_label_box(image: np.ndarray, candidate: TrayCandidate, color: Color) -> None:
    x1, y1, x2, y2 = candidate.bbox
    cx, cy = candidate.center_pixel
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    label = f"{candidate.candidate_id} x{candidate.order_in_level_x} ({cx},{cy})"
    label_y = max(20, y1 - 7)
    cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def draw_result(
    image: np.ndarray,
    candidates: List[TrayCandidate],
    roi: Roi,
    split_y: float,
    max_per_level: int,
) -> np.ndarray:
    vis = image.copy()
    height, width = image.shape[:2]
    roi_box = normalized_roi_to_bbox(roi, width, height)
    cv2.rectangle(vis, roi_box[:2], roi_box[2:], (255, 255, 0), 2)
    cv2.putText(vis, "search ROI", (roi_box[0], max(20, roi_box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    split_pixel_y = int(height * split_y)
    cv2.line(vis, (roi_box[0], split_pixel_y), (roi_box[2], split_pixel_y), (255, 0, 255), 2)
    cv2.putText(vis, "upper / lower split", (roi_box[0], split_pixel_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)

    colors = {"upper": (0, 220, 0), "lower": (0, 165, 255)}
    for level in ("upper", "lower"):
        selected = [candidate for candidate in candidates if candidate.level == level][:max_per_level]
        for candidate in selected:
            draw_label_box(vis, candidate, colors[level])

    summary = f"upper={sum(c.level == 'upper' for c in candidates)} lower={sum(c.level == 'lower' for c in candidates)}"
    cv2.putText(vis, summary, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, summary, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 1, cv2.LINE_AA)
    return vis


def best_by_level(candidates: List[TrayCandidate]) -> Dict[str, Optional[dict]]:
    best: Dict[str, Optional[dict]] = {"upper": None, "lower": None}
    for level in ("upper", "lower"):
        level_candidates = [candidate for candidate in candidates if candidate.level == level]
        if level_candidates:
            best[level] = asdict(level_candidates[0])
    return best


def write_debug_mask(image: np.ndarray, roi: Roi, dark_threshold: int, output_path: Path) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = normalized_roi_to_bbox(roi, width, height)
    mask = make_dark_mask(image[y1:y2, x1:x2], dark_threshold)
    debug = np.zeros((height, width), dtype=np.uint8)
    debug[y1:y2, x1:x2] = mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="input RGB image")
    parser.add_argument("--output", default=None, help="output labeled image")
    parser.add_argument("--json-output", default=None, help="output JSON candidate file")
    parser.add_argument("--debug-mask", default=None, help="optional output mask image for debugging")
    parser.add_argument("--roi", default="0.25,0.13,0.75,0.78", help="normalized search ROI x1,y1,x2,y2")
    parser.add_argument("--split-y", type=float, default=0.42, help="normalized y split between upper and lower trays")
    parser.add_argument("--dark-threshold", type=int, default=95)
    parser.add_argument("--min-area", type=int, default=70)
    parser.add_argument("--max-area", type=int, default=35000)
    parser.add_argument("--max-per-level", type=int, default=8)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    roi = parse_roi(args.roi)
    if not (0.0 < args.split_y < 1.0):
        raise ValueError("--split-y must be between 0 and 1")

    candidates = contour_candidates(
        image=image,
        roi=roi,
        split_y=args.split_y,
        dark_threshold=args.dark_threshold,
        min_area=args.min_area,
        max_area=args.max_area,
    )

    output_path = Path(args.output) if args.output else image_path.with_name(image_path.stem + "_smt_labeled.jpg")
    json_path = Path(args.json_output) if args.json_output else image_path.with_name(image_path.stem + "_smt_labeled.json")

    labeled = draw_result(image, candidates, roi=roi, split_y=args.split_y, max_per_level=args.max_per_level)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), labeled)

    payload = {
        "image": str(image_path),
        "output_image": str(output_path),
        "roi": list(roi),
        "split_y": args.split_y,
        "best": best_by_level(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.debug_mask:
        write_debug_mask(image, roi=roi, dark_threshold=args.dark_threshold, output_path=Path(args.debug_mask))

    print(f"image={image_path}")
    print(f"num_candidates={len(candidates)}")
    for candidate in candidates[: args.max_per_level * 2]:
        print(
            f"{candidate.candidate_id}: level={candidate.level} "
            f"bbox={list(candidate.bbox)} center={list(candidate.center_pixel)} "
            f"score={candidate.score:.1f}"
        )
    print(f"saved_image={output_path}")
    print(f"saved_json={json_path}")


if __name__ == "__main__":
    main()
