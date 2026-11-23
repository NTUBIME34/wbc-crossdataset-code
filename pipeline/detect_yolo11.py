#!/usr/bin/env python3
"""Stage 1 of the inference pipeline: YOLO11 WBC detection and crop.

The defaults are the paper's pipeline settings (main Methods 2.4, Detection-and-cropping
stage):
inference at 512 px, confidence threshold 0.30, NMS IoU 0.70; the highest-confidence
box of each image (any class id of the detector head) is expanded by 10% of its
width and height on each side, clipped to the image and saved as a PNG crop. An image
without a box is written to detections.csv with status no_detection and no crop.
Crops mirror the input folder layout, so class-folder inputs give class-folder crops.

The default weights are models/yolo11_wbc_detection_best.pt (file name of the
weights deposit).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTOR = "yolo11_wbc_detection_best.pt"


def default_model_path(filename: str) -> Path:
    """$MODEL_DIR/<file>, else pipeline/models/<file>, else <repo>/models/<file>."""
    candidates = []
    if os.environ.get("MODEL_DIR"):
        candidates.append(Path(os.environ["MODEL_DIR"]).expanduser() / filename)
    candidates += [ROOT / "models" / filename, ROOT.parent / "models" / filename]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def clamp_box(box, width: int, height: int, padding_ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = x2 - x1
    bh = y2 - y1
    pad_w = bw * padding_ratio
    pad_h = bh * padding_ratio
    x1 = max(0, int(round(x1 - pad_w)))
    y1 = max(0, int(round(y1 - pad_h)))
    x2 = min(width, int(round(x2 + pad_w)))
    y2 = min(height, int(round(y2 + pad_h)))
    return x1, y1, x2, y2


def unique_crop_path(output_dir: Path, rel_path: Path, det_idx: int, ext: str = ".png") -> Path:
    clean_parts = list(rel_path.with_suffix("").parts)
    filename = clean_parts[-1] + f"__det{det_idx:02d}{ext}"
    return output_dir.joinpath(*clean_parts[:-1], filename)


def save_annotated(image, box: tuple[int, int, int, int], output_path: Path, label: str) -> None:
    annotated = image.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        annotated,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 WBC detection and crop stage (defaults = paper settings)")
    parser.add_argument("--input-dir", required=True, help="Folder containing source images (searched recursively).")
    parser.add_argument("--output-dir", required=True, help="Folder where cropped images and detections.csv are saved.")
    parser.add_argument(
        "--model",
        default=None,
        help=f"YOLO detection .pt path. Default: {DEFAULT_DETECTOR} in $MODEL_DIR, pipeline/models/ or models/.",
    )
    parser.add_argument("--conf", type=float, default=0.30, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=512, help="YOLO inference image size.")
    parser.add_argument("--padding-ratio", type=float, default=0.10,
                        help="Fraction of box width/height added on each side of the box.")
    parser.add_argument("--device", default="auto",
                        help="auto (Ultralytics default: first GPU if available, else CPU), cpu, cuda or cuda:N.")
    parser.add_argument(
        "--select",
        choices=["best", "all"],
        default="best",
        help="Save only the highest-confidence box per image or all boxes.",
    )
    parser.add_argument(
        "--fallback",
        choices=["skip", "copy"],
        default="skip",
        help="When no box is detected, skip the image (paper setting) or copy the original image as the crop.",
    )
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--save-crops",
        dest="save_crops",
        action="store_true",
        help="Save cropped cell images for the classification stage.",
    )
    crop_group.add_argument(
        "--no-save-crops",
        dest="save_crops",
        action="store_false",
        help="Do not save cropped images; only write detections.csv and optional annotated images.",
    )
    parser.set_defaults(save_crops=True)
    parser.add_argument("--save-annotated", action="store_true", help="Save images with detection boxes for checking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    crop_dir = output_dir / "crops"
    annotated_dir = output_dir / "annotated"
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).expanduser() if args.model else default_model_path(DEFAULT_DETECTOR)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}. Download the weights (pipeline/README.md) "
            "into pipeline/models/, set MODEL_DIR, or pass --model.")

    model = YOLO(str(model_path))
    predict_kwargs = {} if args.device == "auto" else {"device": args.device}
    images = list_images(input_dir)
    rows: list[dict[str, object]] = []

    for image_path in tqdm(images, desc="Detecting"):
        rel_path = image_path.relative_to(input_dir)
        image = cv2.imread(str(image_path))
        if image is None:
            rows.append({
                "source_path": str(image_path),
                "crop_path": "",
                "status": "read_failed",
                "det_index": "",
                "yolo_class_id": "",
                "yolo_class_name": "",
                "yolo_conf": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
            })
            continue

        height, width = image.shape[:2]
        result = model.predict(
            source=image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
            **predict_kwargs,
        )[0]
        boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            order = sorted(range(len(confs)), key=lambda i: float(confs[i]), reverse=True)
            if args.select == "best":
                order = order[:1]
            boxes = [(idx, xyxy[idx], float(confs[idx]), int(classes[idx])) for idx in order]

        if not boxes:
            if args.fallback == "copy":
                if args.save_crops:
                    crop_path = unique_crop_path(crop_dir, rel_path, 0)
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(crop_path), image)
                    crop_path_value = str(crop_path)
                else:
                    crop_path_value = ""
                rows.append({
                    "source_path": str(image_path),
                    "crop_path": crop_path_value,
                    "status": "fallback_original",
                    "det_index": 0,
                    "yolo_class_id": "",
                    "yolo_class_name": "",
                    "yolo_conf": "",
                    "x1": 0,
                    "y1": 0,
                    "x2": width,
                    "y2": height,
                })
            else:
                rows.append({
                    "source_path": str(image_path),
                    "crop_path": "",
                    "status": "no_detection",
                    "det_index": "",
                    "yolo_class_id": "",
                    "yolo_class_name": "",
                    "yolo_conf": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                })
            continue

        for out_idx, (_raw_idx, raw_box, conf, cls_id) in enumerate(boxes):
            x1, y1, x2, y2 = clamp_box(raw_box, width, height, args.padding_ratio)
            if x2 <= x1 or y2 <= y1:
                status = "empty_crop"
                crop_path = ""
            else:
                if args.save_crops:
                    crop = image[y1:y2, x1:x2]
                    crop_path_obj = unique_crop_path(crop_dir, rel_path, out_idx)
                    crop_path_obj.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(crop_path_obj), crop)
                    crop_path = str(crop_path_obj)
                else:
                    crop_path = ""
                status = "detected"
                if args.save_annotated:
                    cls_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else str(cls_id)
                    ann_path = unique_crop_path(annotated_dir, rel_path, out_idx)
                    save_annotated(image, (x1, y1, x2, y2), ann_path, f"{cls_name} {conf:.3f}")

            cls_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else str(cls_id)
            rows.append({
                "source_path": str(image_path),
                "crop_path": crop_path,
                "status": status,
                "det_index": out_idx,
                "yolo_class_id": cls_id,
                "yolo_class_name": cls_name,
                "yolo_conf": f"{conf:.6f}",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

    csv_path = output_dir / "detections.csv"
    fieldnames = [
        "source_path",
        "crop_path",
        "status",
        "det_index",
        "yolo_class_id",
        "yolo_class_name",
        "yolo_conf",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    detected = sum(1 for r in rows if r["status"] == "detected")
    crops_saved = sum(1 for r in rows if r["crop_path"])
    failed = sum(1 for r in rows if r["status"] in {"no_detection", "read_failed", "empty_crop"})
    print(f"Detector: {model_path.name} (imgsz {args.imgsz}, conf {args.conf}, iou {args.iou}, "
          f"padding {args.padding_ratio}, select {args.select}, fallback {args.fallback})")
    print(f"Images found: {len(images)}")
    print(f"Detections found: {detected}")
    print(f"Crops saved: {crops_saved}")
    print(f"Failed/skipped: {failed}")
    print(f"Detection CSV: {csv_path}")
    print(f"Crop folder: {crop_dir}")


if __name__ == "__main__":
    main()
