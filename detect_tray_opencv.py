#!/usr/bin/env python3
"""
scene3 料盘 OpenCV 初版检测脚本。

输入：
  一张机器人头部相机图，例如 scene3_head_rgb.jpg

输出：
  1. 控制台打印候选 bbox 和 center_pixel
  2. 保存画框后的结果图

说明：
  这不是最终算法，只是第一版 baseline：
  - 在画面中间货架区域做 ROI；
  - 找偏暗、竖直、细长的候选区域；
  - 输出候选 bbox；
  - 后续再用深度图和 offset 选抓取点。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Candidate:
    bbox: Tuple[int, int, int, int]
    area: float
    score: float

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2


def detect_dark_vertical_candidates(
    image: np.ndarray,
    roi: Tuple[float, float, float, float],
    min_area: int = 80,
) -> List[Candidate]:
    h, w = image.shape[:2]
    rx1, ry1, rx2, ry2 = roi
    x1 = int(w * rx1)
    y1 = int(h * ry1)
    x2 = int(w * rx2)
    y2 = int(h * ry2)

    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 黑色/深色料盘候选。阈值故意宽一点，先多找候选。
    mask_dark = cv2.inRange(gray, 0, 90)

    # 去掉噪声，连接细长区域。
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    mask = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Candidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue

        aspect_h_over_w = bh / max(bw, 1)
        aspect_w_over_h = bw / max(bh, 1)

        # scene3 图里料盘常表现为货架上的深色薄竖片，也可能是小矩形。
        vertical_like = aspect_h_over_w >= 1.2 and 8 <= bw <= 90 and 20 <= bh <= 180
        small_tray_like = 0.4 <= aspect_w_over_h <= 4.0 and area >= min_area
        if not (vertical_like or small_tray_like):
            continue

        gx1, gy1 = x1 + bx, y1 + by
        gx2, gy2 = x1 + bx + bw, y1 + by + bh

        # 越靠近画面中间、越竖直、面积适中，分数越高。
        cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        center_bonus = 1.0 - min(abs(cx - w / 2) / (w / 2), 1.0)
        vertical_bonus = min(aspect_h_over_w / 4.0, 1.0)
        score = float(area) * (1.0 + 0.5 * center_bonus + 0.5 * vertical_bonus)
        candidates.append(Candidate((gx1, gy1, gx2, gy2), float(area), score))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def draw_candidates(
    image: np.ndarray,
    candidates: List[Candidate],
    roi: Tuple[float, float, float, float],
    max_draw: int = 20,
) -> np.ndarray:
    vis = image.copy()
    h, w = image.shape[:2]
    rx1, ry1, rx2, ry2 = roi
    roi_box = (int(w * rx1), int(h * ry1), int(w * rx2), int(h * ry2))
    cv2.rectangle(vis, roi_box[:2], roi_box[2:], (255, 255, 0), 2)
    cv2.putText(
        vis,
        "ROI",
        (roi_box[0], max(20, roi_box[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    for i, cand in enumerate(candidates[:max_draw]):
        x1, y1, x2, y2 = cand.bbox
        u, v = cand.center
        color = (0, 255, 0) if i == 0 else (0, 180, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (u, v), 4, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"{i}: ({u},{v})",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return vis


def parse_roi(s: str) -> Tuple[float, float, float, float]:
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 4:
        raise ValueError("--roi must be x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("--roi values must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1")
    return x1, y1, x2, y2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="输入图片")
    parser.add_argument("--output", default=None, help="输出画框图片")
    parser.add_argument(
        "--roi",
        default="0.25,0.15,0.75,0.78",
        help="检测区域，归一化坐标 x1,y1,x2,y2；默认取画面中间货架区域",
    )
    parser.add_argument("--min-area", type=int, default=80)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    roi = parse_roi(args.roi)
    candidates = detect_dark_vertical_candidates(image, roi=roi, min_area=args.min_area)

    print(f"image={image_path}")
    print(f"num_candidates={len(candidates)}")
    for i, cand in enumerate(candidates[:20]):
        print(
            f"candidate_{i}: bbox={list(cand.bbox)} "
            f"center={list(cand.center)} area={cand.area:.1f} score={cand.score:.1f}"
        )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.with_name(image_path.stem + "_tray_detect.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis = draw_candidates(image, candidates, roi=roi)
    cv2.imwrite(str(output_path), vis)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()

