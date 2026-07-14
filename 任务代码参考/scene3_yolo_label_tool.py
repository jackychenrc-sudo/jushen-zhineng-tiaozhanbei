#!/usr/bin/env python3
"""Simple OpenCV label tool for Scene3 YOLO datasets."""

import argparse
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_class_names(text):
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise ValueError("at least one class name is required")
    return names


def clamp(value, low, high):
    return max(low, min(high, value))


def find_image_files(path):
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def image_to_label_path(image_path, image_root, label_root):
    relative = image_path.relative_to(image_root)
    return (label_root / relative).with_suffix(".txt")


def ensure_dataset_yaml(dataset_root, class_names):
    val_dir = dataset_root / "val"
    yaml_path = dataset_root / "dataset.yaml"
    lines = [
        "path: {}".format(dataset_root.resolve()),
        "train: train",
        "val: {}".format("val" if val_dir.exists() else "train"),
        "names:",
    ]
    for index, name in enumerate(class_names):
        lines.append("  {}: {}".format(index, name))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def infer_dataset_root(image_root):
    image_root = image_root.resolve()
    current = image_root
    while current != current.parent:
        if (current / "train").exists() or (current / "val").exists():
            return current
        current = current.parent
    return image_root


class LabelTool(object):
    def __init__(self, args):
        self.args = args
        self.image_root = Path(args.image_root).resolve()
        self.label_root = Path(args.label_root).resolve()
        self.label_root.mkdir(parents=True, exist_ok=True)
        self.class_names = parse_class_names(args.class_names)
        self.images = find_image_files(self.image_root)
        if not self.images:
            raise RuntimeError("no images found under {}".format(self.image_root))

        self.index = max(0, min(args.start_index, len(self.images) - 1))
        self.current_image = None
        self.current_path = None
        self.current_boxes = []
        self.selected_class = 0
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.dirty = False
        self.window_name = "scene3_yolo_label_tool"

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _load_boxes(self, label_path, image_width, image_height):
        boxes = []
        if not label_path.exists():
            return boxes
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            box_width = float(parts[3]) * image_width
            box_height = float(parts[4]) * image_height
            x1 = int(round(x_center - box_width / 2.0))
            y1 = int(round(y_center - box_height / 2.0))
            x2 = int(round(x_center + box_width / 2.0))
            y2 = int(round(y_center + box_height / 2.0))
            boxes.append(
                {
                    "class_id": class_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        return boxes

    def _save_boxes(self):
        if self.current_image is None or self.current_path is None:
            return
        label_path = image_to_label_path(
            self.current_path,
            self.image_root,
            self.label_root,
        )
        label_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = self.current_image.shape[:2]
        lines = []
        for box in self.current_boxes:
            x1 = clamp(min(box["x1"], box["x2"]), 0, width - 1)
            y1 = clamp(min(box["y1"], box["y2"]), 0, height - 1)
            x2 = clamp(max(box["x1"], box["x2"]), x1 + 1, width)
            y2 = clamp(max(box["y1"], box["y2"]), y1 + 1, height)
            x_center = ((x1 + x2) / 2.0) / float(width)
            y_center = ((y1 + y2) / 2.0) / float(height)
            box_width = (x2 - x1) / float(width)
            box_height = (y2 - y1) / float(height)
            lines.append(
                "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                    int(box["class_id"]),
                    x_center,
                    y_center,
                    box_width,
                    box_height,
                )
            )
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.dirty = False

    def _load_current(self):
        self.current_path = self.images[self.index]
        self.current_image = cv2.imread(str(self.current_path))
        if self.current_image is None:
            raise RuntimeError("failed to read image: {}".format(self.current_path))
        label_path = image_to_label_path(
            self.current_path,
            self.image_root,
            self.label_root,
        )
        height, width = self.current_image.shape[:2]
        self.current_boxes = self._load_boxes(label_path, width, height)
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.dirty = False

    def _change_index(self, delta):
        if self.dirty:
            self._save_boxes()
        self.index = clamp(self.index + delta, 0, len(self.images) - 1)
        self._load_current()

    def _on_mouse(self, event, x, y, flags, param):
        if self.current_image is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.drag_end = (x, y)
            x1 = min(self.drag_start[0], self.drag_end[0])
            y1 = min(self.drag_start[1], self.drag_end[1])
            x2 = max(self.drag_start[0], self.drag_end[0])
            y2 = max(self.drag_start[1], self.drag_end[1])
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                self.current_boxes.append(
                    {
                        "class_id": self.selected_class,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
                self.dirty = True

    def _draw(self):
        canvas = self.current_image.copy()
        for idx, box in enumerate(self.current_boxes):
            class_id = int(box["class_id"])
            color = (0, 255, 0) if class_id == self.selected_class else (0, 255, 255)
            x1 = int(box["x1"])
            y1 = int(box["y1"])
            x2 = int(box["x2"])
            y2 = int(box["y2"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = "{}:{}".format(class_id, self.class_names[class_id])
            cv2.putText(
                canvas,
                label,
                (x1, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            cv2.putText(
                canvas,
                "#{}".format(idx),
                (x1, min(canvas.shape[0] - 8, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        if self.dragging and self.drag_start and self.drag_end:
            cv2.rectangle(canvas, self.drag_start, self.drag_end, (255, 0, 0), 1)

        lines = [
            "image {}/{}".format(self.index + 1, len(self.images)),
            "class {} -> {}".format(self.selected_class, self.class_names[self.selected_class]),
            "boxes {}".format(len(self.current_boxes)),
            "keys: n-next p-prev s-save d-del c-clear 0-9-class q-quit",
            str(self.current_path.relative_to(self.image_root)),
        ]
        y = 24
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                canvas,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (20, 20, 20),
                1,
            )
            y += 24
        return canvas

    def run(self):
        self._load_current()
        while True:
            canvas = self._draw()
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):
                if self.dirty:
                    self._save_boxes()
                break
            if key in (ord("n"), ord("l"), 83):
                if self.index < len(self.images) - 1:
                    self._change_index(1)
                continue
            if key in (ord("p"), ord("h"), 81):
                if self.index > 0:
                    self._change_index(-1)
                continue
            if key == ord("s"):
                self._save_boxes()
                continue
            if key == ord("d"):
                if self.current_boxes:
                    self.current_boxes.pop()
                    self.dirty = True
                continue
            if key == ord("c"):
                self.current_boxes = []
                self.dirty = True
                continue
            if ord("0") <= key <= ord("9"):
                class_id = key - ord("0")
                if class_id < len(self.class_names):
                    self.selected_class = class_id
                continue
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCV YOLO label tool for Scene3 data")
    parser.add_argument("--image-root", required=True, help="directory containing images")
    parser.add_argument("--label-root", default=None, help="directory to save YOLO txt labels")
    parser.add_argument("--class-names", default="tray", help="comma separated class names")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--write-dataset-yaml", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    image_root = Path(args.image_root).resolve()
    if args.label_root is None:
        if image_root.name == "images":
            label_root = image_root.parent / "labels"
        else:
            label_root = image_root / "labels"
        args.label_root = str(label_root)
    tool = LabelTool(args)
    if args.write_dataset_yaml:
        dataset_root = infer_dataset_root(image_root)
        yaml_path = ensure_dataset_yaml(dataset_root, tool.class_names)
        print("dataset_yaml={}".format(yaml_path))
    tool.run()


if __name__ == "__main__":
    main()
