#!/usr/bin/env python3
"""Detector crop of class-folder images: highest-confidence box, no margin, threshold.

This is the crop that turns each classifier training source, and the WBCBench2026
Phase-1 fields, into single-cell images (main Methods "Datasets"; Supplementary S1,
Table S1; the pre-cropped external set, route A of Supplementary Table S6):

  * the YOLO11 WBC detector runs at 512 px (the size it was trained at) with NMS IoU
    0.70, on the image as read by OpenCV (BGR);
  * only the single highest-confidence box of each image is used (any class id of
    the detector head), and only if its confidence is at least --conf;
  * that box is cropped WITHOUT margin: its corner coordinates are truncated to
    integers (int()), clipped to the image, and the crop is written with the same
    file name into <output-dir>/<class folder>/;
  * an image whose best box scores below --conf is excluded (no crop is written);
    it is not cropped at a lower confidence. Unreadable files are excluded too.

Thresholds as run (see data_prep/README.md, "Detector thresholds as run"): 0.70 for
the merged internal corpus, the BMCMHF pool and the WBCBench2026 pre-cropped set;
0.30 for the AML-LMU, PBC, Raabin-WBC and HRLS pools used by the leave-one-dataset-out
analysis. crop_training_sources.sh passes the per-source values.

Provenance. The authors' crops were made with an earlier script that applied the same
rule (best box, int() truncation, no margin); that script was later edited and the
Ultralytics version used then was not recorded (Supplementary Table S5). Re-running
the released detector with this script (CPU, Ultralytics 8.4.82) on the authors' copies
of the five sources and of the Phase-1 fields reproduced the authors' crop box exactly
for about 95% of sampled images; the others in that sample differ by at most one pixel
in a box coordinate. This sample-specific observation is not a bound for other runs,
and it does not identify which software or device setting caused the differences.
Preserve crop_log.csv and the generated crops when exact input replay is required;
regenerated inputs must be compared separately from historical inputs. An image
whose best-box confidence lies right at the threshold can switch sides.

The deployable pipeline's detector (pipeline/detect_yolo11.py) is a different
setting on purpose: threshold 0.30, a 10% margin on each side, rounded coordinates
and PNG crops named <stem>__det00.png.

Input layout: <input-dir>/<class>/<image>. Images in sub-folders of a class folder
(for example the numbered range folders of the BMCMHF release) are found
recursively and written flat into the class folder. Hidden files are ignored.

Outputs (under --output-dir, never inside the repository):
  <class>/<image>     crops (same file name; with --png the suffix becomes .png)
  crop_log.csv        one row per input image: status, best-box confidence and box.
                      harmonize_labels.py --merged-dir reads the confidences.

Examples:
  # one training source (crop_training_sources.sh runs all five)
  python data_prep/detector_crop.py --input-dir "$SOURCES_ROOT/dataset_DIB" \\
      --output-dir "$DATA_ROOT/dataset_DIB_single_wbc" --skip-classes ig,platelet --png --conf 0.30
  # WBCBench2026 Phase-1 pre-cropped set (route A; WBCB_P1_CROPS)
  python data_prep/detector_crop.py --input-dir "$DATA_ROOT/wbc-bench-2026_phase1_nocrop_formatted" \\
      --output-dir "$DATA_ROOT/wbc-bench-2026_phase1_cropped_formatted" --conf 0.70
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETECTOR = "yolo11_wbc_detection_best.pt"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LOG_FIELDS = ["class", "source_file", "crop_file", "status", "conf", "x1", "y1", "x2", "y2"]


def default_model_path() -> Path:
    """$MODEL_DIR/<file>, else pipeline/models/<file>, else <repo>/models/<file>."""
    candidates = []
    if os.environ.get("MODEL_DIR"):
        candidates.append(Path(os.environ["MODEL_DIR"]).expanduser() / DEFAULT_DETECTOR)
    candidates += [REPO_ROOT / "pipeline" / "models" / DEFAULT_DETECTOR,
                   REPO_ROOT / "models" / DEFAULT_DETECTOR]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop the highest-confidence detector box (no margin) of every image "
                    "in a class-folder dataset; images below the threshold are excluded.")
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Dataset root with one sub-folder per class.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where the cropped class folders and crop_log.csv are written.")
    parser.add_argument("--model", type=Path, default=None,
                        help=f"YOLO detector weights. Default: {DEFAULT_DETECTOR} in $MODEL_DIR, "
                             "pipeline/models/ or models/.")
    parser.add_argument("--conf", type=float, default=0.70,
                        help="Confidence threshold of the best box (default 0.70: merged corpus, "
                             "BMCMHF pool, WBCBench2026 pre-cropped set; the AML-LMU, PBC, "
                             "Raabin-WBC and HRLS pools were cropped at 0.30, see data_prep/README.md).")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size.")
    parser.add_argument("--device", default=None,
                        help="Inference device passed to Ultralytics, e.g. cpu or 0 "
                             "(default: Ultralytics picks one).")
    parser.add_argument("--skip-classes", default="",
                        help="Comma-separated class folders not passed to the detector "
                             "(case-insensitive), e.g. ig,platelet for PBC.")
    parser.add_argument("--png", action="store_true",
                        help="Write every crop as PNG (<stem>.png) instead of keeping the input "
                             "format. Used for AML-LMU, PBC and HRLS, whose crops the authors "
                             "made from PNG copies (see data_prep/README.md).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-crop images whose crop already exists (default: keep a crop whose "
                             "logged confidence passes --conf, re-check the others).")
    return parser.parse_args()


def read_previous_log(log_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Rows of an existing crop_log.csv, keyed by (class, source_file)."""
    if not log_path.exists():
        return {}
    with log_path.open(newline="", encoding="utf-8") as f:
        return {(r["class"], r["source_file"]): r for r in csv.DictReader(f)}


def list_class_images(class_dir: Path) -> tuple[list[Path], int]:
    images, hidden = [], 0
    for path in sorted(class_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        if any(part.startswith(".") for part in path.relative_to(class_dir).parts):
            hidden += 1
            continue
        images.append(path)
    return images, hidden


def best_box_crop(model, image, conf: float, iou: float, imgsz: int, device):
    """Return (crop, best_conf, (x1, y1, x2, y2)) or (None, best_conf, None)."""
    kwargs = {"conf": conf, "iou": iou, "imgsz": imgsz, "verbose": False}
    if device is not None:
        kwargs["device"] = device
    results = model.predict(source=image, **kwargs)
    best_box, best_conf = None, 0.0
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for box, box_conf in zip(boxes, confidences):
            if box_conf > best_conf:  # strict: the first of equal maxima is kept
                best_conf = float(box_conf)
                best_box = box
    if best_box is None or best_conf < conf:
        return None, best_conf, None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, best_box)  # truncation, no margin
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None, best_conf, (x1, y1, x2, y2)
    return crop, best_conf, (x1, y1, x2, y2)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise SystemExit("--output-dir must not be the input folder or lie inside it.")
    if output_dir == REPO_ROOT or REPO_ROOT in output_dir.parents:
        raise SystemExit("--output-dir must be outside the repository.")
    model_path = (args.model or default_model_path()).expanduser()
    if not model_path.exists():
        raise SystemExit(f"Detector weights not found: {model_path}. Download "
                         f"{DEFAULT_DETECTOR} (see the top-level README), set MODEL_DIR or pass --model.")

    skip = {name.strip().lower() for name in args.skip_classes.split(",") if name.strip()}
    class_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not class_dirs:
        raise SystemExit(f"No class sub-folders found in {input_dir}")

    import cv2
    from tqdm import tqdm
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "crop_log.csv"
    previous = read_previous_log(log_path)
    rows: list[dict[str, object]] = []
    totals = {"images": 0, "cropped": 0, "excluded": 0, "unreadable": 0}
    print(f"Detector: {model_path.name}  conf>={args.conf}  imgsz={args.imgsz}  iou={args.iou}")
    print(f"{'class':28s} {'images':>8s} {'cropped':>8s} {'excluded':>8s} {'unreadable':>10s}")

    processed: set[str] = set()
    for class_dir in class_dirs:
        if class_dir.name.strip().lower() in skip:
            print(f"{class_dir.name:28s} skipped (--skip-classes)")
            continue
        processed.add(class_dir.name)
        images, hidden = list_class_images(class_dir)
        out_class = output_dir / class_dir.name
        targets: dict[Path, Path] = {}
        seen: set[Path] = set()
        for src in images:
            name = src.with_suffix(".png").name if args.png else src.name
            dst = out_class / name
            if dst in seen:
                raise SystemExit(f"Two inputs map to the same crop name {dst}; "
                                 "rename one of them before cropping.")
            seen.add(dst)
            targets[src] = dst
        counts = {"images": len(images), "cropped": 0, "excluded": 0, "unreadable": 0}
        for src in tqdm(images, desc=class_dir.name, leave=False):
            dst = targets[src]
            row = {"class": class_dir.name, "source_file": src.relative_to(class_dir).as_posix(),
                   "crop_file": "", "status": "", "conf": "", "x1": "", "y1": "", "x2": "", "y2": ""}
            if dst.exists() and not args.overwrite:
                # Resume: keep a crop made earlier when its logged confidence passes this
                # threshold, and carry the confidence and box over (harmonize_labels.py
                # --merged-dir needs the confidence). Crops without a usable log entry
                # are checked again below.
                old = previous.get((row["class"], row["source_file"]), {})
                try:
                    old_ok = float(old.get("conf") or "nan") >= args.conf
                except ValueError:
                    old_ok = False
                if old_ok:
                    row.update({k: old.get(k, "") for k in ("conf", "x1", "y1", "x2", "y2")})
                    row.update(status="kept_existing", crop_file=dst.relative_to(output_dir).as_posix())
                    counts["cropped"] += 1
                    rows.append(row)
                    continue
            image = cv2.imread(str(src))
            if image is None:
                row["status"] = "unreadable"
                counts["unreadable"] += 1
                rows.append(row)
                dst.unlink(missing_ok=True)  # a stale crop from an earlier run
                continue
            crop, best_conf, box = best_box_crop(model, image, args.conf, args.iou, args.imgsz, args.device)
            if best_conf > 0:
                row["conf"] = f"{best_conf:.6f}"
            if box is not None:
                row.update(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
            if crop is None:
                row["status"] = "excluded_below_threshold" if box is None else "empty_crop"
                counts["excluded"] += 1
                dst.unlink(missing_ok=True)  # a stale crop from an earlier run
            else:
                out_class.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(dst), crop):
                    raise SystemExit(f"Could not write {dst}")
                row.update(status="cropped", crop_file=dst.relative_to(output_dir).as_posix())
                counts["cropped"] += 1
            rows.append(row)
        for key in totals:
            totals[key] += counts[key]
        note = f"  ({hidden} hidden files ignored)" if hidden else ""
        print(f"{class_dir.name:28s} {counts['images']:8d} {counts['cropped']:8d} "
              f"{counts['excluded']:8d} {counts['unreadable']:10d}{note}")

    # Rows of classes not processed in this call (e.g. a partial re-run) are kept.
    done = {(str(r["class"]), str(r["source_file"])) for r in rows}
    rows += [r for key, r in previous.items() if key not in done and key[0] not in processed]
    rows.sort(key=lambda r: (str(r["class"]), str(r["source_file"])))
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'TOTAL':28s} {totals['images']:8d} {totals['cropped']:8d} "
          f"{totals['excluded']:8d} {totals['unreadable']:10d}")
    print(f"Crop log: {log_path}")


if __name__ == "__main__":
    main()
