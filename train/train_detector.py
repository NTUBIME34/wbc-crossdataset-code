#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the Stage-1 YOLO11-M white blood cell detector on TXL-PBC.

Reproduces the detector of the main Methods 2.4 ("Detection-and-cropping stage") and the
three-class variant of Supplementary Section S1:

* COCO-pretrained YOLO11-M (``yolo11m.pt``, downloaded by Ultralytics on first use),
  input 512 px, up to 150 epochs with early stopping on the validation split
  (patience 30), batch 32, seed 3407, ``optimizer="auto"`` (which selected AdamW,
  learning rate 1.43e-3 decaying linearly), RAM caching and the augmentation settings in
  FIXED_TRAIN_ARGS below.
* The released detector was trained with Ultralytics 8.3.170. Other versions may
  resolve the automatic settings differently; the script warns when they differ.
* Deployed detector: WBC boxes only (``--labels wbc_only``, the default). Every YOLO
  label file of TXL-PBC is reduced to its class-0 (WBC) lines in a linked copy of the
  dataset; ``data.yaml`` keeps its three class names, so the checkpoint has a three-class
  head of which only WBC has training instances.
* ``--labels as_released`` trains on TXL-PBC as distributed (WBC, RBC, platelets): the
  three-class variant reported in Supplementary Section S1, which was not deployed.

TXL-PBC (Scientific Data 2025; figshare doi:10.6084/m9.figshare.27073186;
CC BY 4.0) ships ``data.yaml`` with ``images/{train,val,test}`` (882 training and 252
validation images of 1,260) and matching ``labels/`` folders. Ultralytics is AGPL-3.0
licensed and installed separately.

Example:
    python train/train_detector.py --data /path/to/TXL-PBC/data.yaml
    python train/train_detector.py --data /path/to/TXL-PBC/data.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

DETECTOR_ULTRALYTICS_VERSION = "8.3.170"

# Training arguments of the reported detector (all other settings: Ultralytics defaults).
FIXED_TRAIN_ARGS = dict(
    cache=True,            # cache images in RAM
    plots=True,            # loss / PR curves
    pretrained=True,       # start from the COCO-pretrained checkpoint
    optimizer="auto",
    # color and geometric augmentation for staining / imaging differences
    hsv_h=0.015,
    hsv_s=0.70,
    hsv_v=0.40,
    fliplr=0.5,
    flipud=0.0,            # no vertical flips
    degrees=10.0,
    translate=0.10,
    scale=0.30,            # mild scaling
    shear=0.0,
    perspective=0.0005,    # almost no perspective warp
    mosaic=0.30,           # reduced mosaic probability
    mixup=0.10,
    copy_paste=0.0,        # off: would create unrealistic cell boundaries
)

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}
WBC_ONLY_MARKER = ".wbc_only_dataset"


def parse_args():
    out_root = Path(os.environ.get("OUT_ROOT", "./outputs"))
    p = argparse.ArgumentParser(
        prog="train_detector.py",
        description="Train the YOLO11-M WBC detector on TXL-PBC with the paper's settings.")
    p.add_argument("--data", required=True, type=Path,
                   help="TXL-PBC data.yaml as distributed (train/val/test image folders, 3 class names).")
    p.add_argument("--labels", choices=["wbc_only", "as_released"], default="wbc_only",
                   help="wbc_only: deployed detector (WBC boxes only); as_released: three-class "
                        "variant of Supplementary Section S1.")
    p.add_argument("--model", default="yolo11m.pt",
                   help="Initial weights (default: COCO-pretrained yolo11m.pt).")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--patience", type=int, default=30, help="Early-stopping patience (epochs).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default=None,
                   help="Ultralytics device string, e.g. 0, 0,1 or cpu (default: Ultralytics picks).")
    p.add_argument("--project", type=Path, default=out_root / "detector",
                   help="Output folder for runs (default: $OUT_ROOT/detector, OUT_ROOT=./outputs).")
    p.add_argument("--name", default=None,
                   help="Run name (default: yolo11m_txlpbc_<labels>_s<imgsz>).")
    p.add_argument("--dry-run", action="store_true",
                   help="Prepare the data, validate the arguments with Ultralytics, print them and exit.")
    return p.parse_args()


def label_path_for(image_path: Path) -> Path:
    """Label file Ultralytics pairs with an image: last '/images/' -> '/labels/', suffix -> .txt."""
    s = str(image_path)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa not in s:
        raise ValueError(f"Image path has no '{sa}' component, cannot locate its labels: {s}")
    return Path(sb.join(s.rsplit(sa, 1))).with_suffix(".txt")


def keep_wbc_lines(text: str) -> list[str]:
    """Lines of a YOLO label file that are WBC boxes: at least five fields, class id '0'."""
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "0":
            kept.append(line)
    return kept


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_wbc_only_dataset(data_yaml: Path, dst: Path) -> Path:
    """Linked copy of the dataset whose label files keep only class-0 (WBC) boxes.

    Images are linked file by file (not as directories: Ultralytics resolves directory
    links and would read the original labels). A label file left without boxes is
    written empty, so its image stays in the split as background. Returns the new data.yaml.
    """
    import yaml

    data_yaml = data_yaml.expanduser().resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    dst = dst.expanduser().resolve()
    if dst.exists():
        if not (dst / WBC_ONLY_MARKER).exists():
            raise SystemExit(f"[ERROR] {dst} exists and was not made by this script; choose another --project/--name.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    (dst / WBC_ONLY_MARKER).write_text(f"WBC-only copy of {data_yaml}\n", encoding="utf-8")

    out_cfg = {"path": str(dst)}
    for split in ("train", "val", "test"):
        if not cfg.get(split):
            continue
        if not isinstance(cfg[split], str):
            raise SystemExit(f"[ERROR] {split}: only a single image folder per split is supported.")
        src_images = (root / cfg[split]).resolve()
        if not src_images.is_dir():
            raise SystemExit(f"[ERROR] {split} image folder not found: {src_images}")
        n_images = n_boxes = n_empty = 0
        for img in sorted(src_images.rglob("*")):
            if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rel = img.relative_to(src_images)
            link_or_copy(img, dst / "images" / split / rel)
            n_images += 1
            src_label = label_path_for(img)
            if src_label.is_file():
                kept = keep_wbc_lines(src_label.read_text(encoding="utf-8"))
                out_label = (dst / "labels" / split / rel).with_suffix(".txt")
                out_label.parent.mkdir(parents=True, exist_ok=True)
                out_label.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
                n_boxes += len(kept)
                n_empty += int(not kept)
        out_cfg[split] = f"images/{split}"
        print(f"[INFO] {split}: {n_images} images, {n_boxes} WBC boxes, {n_empty} label files without WBC")
    for key in ("nc", "names"):
        if key in cfg:
            out_cfg[key] = cfg[key]
    out_yaml = dst / "data.yaml"
    out_yaml.write_text(yaml.safe_dump(out_cfg, sort_keys=False), encoding="utf-8")
    print(f"[INFO] WBC-only dataset: {out_yaml}")
    return out_yaml


def main():
    args = parse_args()
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != DETECTOR_ULTRALYTICS_VERSION:
        print(f"[WARN] Ultralytics {ultralytics.__version__} is installed; the released detector was "
              f"trained with {DETECTOR_ULTRALYTICS_VERSION} (pip install ultralytics=={DETECTOR_ULTRALYTICS_VERSION}).")

    project = args.project.expanduser().resolve()
    name = args.name or f"yolo11m_txlpbc_{args.labels}_s{args.imgsz}"
    data_yaml = args.data.expanduser().resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"[ERROR] data.yaml not found: {data_yaml}")
    if args.labels == "wbc_only":
        data_yaml = build_wbc_only_dataset(data_yaml, project / f"{name}_dataset")

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project),
        name=name,
        exist_ok=True,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        **FIXED_TRAIN_ARGS,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    if args.dry_run:
        from ultralytics.cfg import get_cfg
        get_cfg(overrides=dict(train_kwargs, model=args.model))  # raises on unknown or mistyped keys
        print(json.dumps(dict(train_kwargs, model=args.model), indent=2))
        print("[INFO] Dry run: arguments validated, no training started.")
        return

    start = time.time()
    print(f"[INFO] Training {args.model} on {data_yaml}")
    model = YOLO(args.model)
    model.train(**train_kwargs)
    print(f"[INFO] Training finished in {(time.time() - start) / 60:.2f} min")
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        print(f"[INFO] Best checkpoint: {trainer.best}")


if __name__ == "__main__":
    main()
