#!/usr/bin/env python3
"""
Crop single-cell WBC images with Cellpose masks (cropping routes C and D of main-text
Table 2b and Supplementary Table S6).

For each image, Cellpose (model cpsam, Cellpose 4.1.1) predicts instance masks. The
script keeps the mask whose centroid is closest to the image center and crops a fixed
224 x 224 window centered on that centroid (white padding where the window leaves the
image). Two variants:

  default             route D: every pixel outside the selected mask is set to white
                      before cropping ("background removed");
  --keep-background   route C: the window keeps the original background.

If an image yields no mask, the whole image is treated as the mask and the window is
centered on the image center (in the paper a mask was found in every Phase-1 field; the
script reports how often this fallback happened).

Precision: the paper's crops were made on a GPU with Cellpose's default bfloat16
weights. Without a GPU the script loads float32 weights instead (--precision auto),
because bfloat16 inference is very slow on processors without native bfloat16 support;
float32 masks can differ slightly from bfloat16 ones, so an occasional crop window can
shift. --precision bf16 forces the paper's setting on any device. Route E re-crops the route-C crops
with the pipeline detector (pipeline/detect_yolo11.py, confidence 0.30, 5% margin); see
eval/run_routes_ablation.sh.

Input layouts:
    <input-dir>/<class>/*.png                  (--layout class)
    <input-dir>/<split>/<class>/*.png          (--layout split; train/val/test)
For the paper the input was the WBCBench2026 Phase-1 class folders written by
data_prep/prepare_wbcbench.py --split phase1 (the default --input-dir when DATA_ROOT is
set). Output keeps the same layout and file names; the default --output-dir is the
input folder name plus _cellpose_224 (route D) or _cellpose_keepbg_224 (route C), the
folders eval/run_routes_ablation.sh reads. Step 6 (optional) of data_prep/README.md;
runs in the separate Cellpose environment (environment/cellpose_env.yml).

Usage:
  # route D -> $DATA_ROOT/wbc-bench-2026_phase1_nocrop_formatted_cellpose_224
  python data_prep/cellpose_crop.py --layout class
  # route C -> $DATA_ROOT/wbc-bench-2026_phase1_nocrop_formatted_cellpose_keepbg_224
  python data_prep/cellpose_crop.py --layout class --keep-background
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SPLIT_DIR_NAMES = {"train", "val", "valid", "validation", "test"}
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_NAME = "wbc-bench-2026_phase1_nocrop_formatted"  # prepare_wbcbench.py --split phase1

np = None
Image = None
tqdm = None


def load_runtime_dependencies() -> None:
    global Image, np, tqdm

    try:
        import cv2  # noqa: F401
        import numpy as numpy_module
        from PIL import Image as pil_image
        from PIL import ImageFile
        from tqdm.auto import tqdm as tqdm_auto
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime package. Install the image-processing dependencies first, for example:\n"
            "  pip install numpy pillow tqdm opencv-python cellpose torch"
        ) from exc

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    np = numpy_module
    Image = pil_image
    tqdm = tqdm_auto


@dataclass(frozen=True)
class DataItem:
    src: Path
    dst: Path
    label: str
    split: str


@dataclass(frozen=True)
class ImageItem:
    src: Path
    label: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Cellpose-centered single-WBC crops from class folders "
                    "(route D by default; route C with --keep-background)."
    )
    parser.add_argument("--input-dir", type=Path, default=None,
                        help=f"Class-folder (or split/class-folder) dataset to crop. "
                             f"Default: $DATA_ROOT/{DEFAULT_INPUT_NAME}.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where the crops are written (same layout and file names). Default: "
                             "<input-dir>_cellpose_<crop-size> (route D) or "
                             "<input-dir>_cellpose_keepbg_<crop-size> (route C).")
    parser.add_argument("--keep-background", action="store_true",
                        help="Route C: keep the original background (default: route D, "
                             "white outside the selected mask).")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--model-type", default="cpsam", help="Cellpose model_type, e.g. cpsam, cyto3.")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto",
                        help="Model weights: auto = bfloat16 on a GPU (the Cellpose default, as run "
                             "for the paper) and float32 on CPU, where bfloat16 is very slow on "
                             "processors without native bfloat16 support; bf16 or fp32 force one.")
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument(
        "--layout",
        choices=["auto", "class", "split"],
        default="auto",
        help="class = class/image, split = split/class/image, auto = detect from folders.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.0, help="If > 0, write train/val/class folders.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit for quick tests.")
    parser.add_argument("--save-manifest", action="store_true", help="Write manifest.csv under output-dir.")
    args = parser.parse_args()
    if args.input_dir is None:
        if not os.environ.get("DATA_ROOT"):
            parser.error("pass --input-dir or set DATA_ROOT")
        args.input_dir = Path(os.environ["DATA_ROOT"]) / DEFAULT_INPUT_NAME
    args.input_dir = args.input_dir.expanduser().resolve()
    if args.output_dir is None:
        tag = "cellpose_keepbg" if args.keep_background else "cellpose"
        args.output_dir = args.input_dir.with_name(f"{args.input_dir.name}_{tag}_{args.crop_size}")
    return args


def load_cellpose_model(model_type: str, precision: str = "auto"):
    try:
        import inspect
        import torch
        from cellpose import models
    except ImportError as exc:
        raise SystemExit(
            "Cellpose/Torch is not installed in this Python environment. Install it first, for example:\n"
            "  pip install cellpose torch\n"
            "or run this script inside the conda environment that already has Cellpose and Torch."
        ) from exc

    use_gpu = torch.cuda.is_available()
    signature = inspect.signature(models.CellposeModel)
    kwargs = {"gpu": use_gpu}
    if "use_bfloat16" in signature.parameters:  # Cellpose 4 (default True)
        kwargs["use_bfloat16"] = precision == "bf16" or (precision == "auto" and use_gpu)
    weights = "bfloat16" if kwargs.get("use_bfloat16", False) else "float32"
    print(f"Hardware config: GPU={use_gpu}, model_type={model_type}, weights={weights}")
    if "pretrained_model" in signature.parameters:
        return models.CellposeModel(pretrained_model=model_type, **kwargs)
    if "model_type" in signature.parameters:
        return models.CellposeModel(model_type=model_type, **kwargs)
    return models.CellposeModel(**kwargs)


def has_images(path: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in path.iterdir())


def detect_layout(input_dir: Path) -> str:
    split_dirs = [p for p in input_dir.iterdir() if p.is_dir() and p.name.lower() in SPLIT_DIR_NAMES]
    for split_dir in split_dirs:
        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if any(has_images(class_dir) for class_dir in class_dirs):
            return "split"
    return "class"


def iter_class_images(input_dir: Path, layout: str) -> list[ImageItem]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if layout == "auto":
        layout = detect_layout(input_dir)
    print(f"Detected layout: {layout}")

    image_items: list[ImageItem] = []
    if layout == "split":
        for split_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
            split = split_dir.name
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                label = class_dir.name
                for img_path in sorted(class_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        image_items.append(ImageItem(src=img_path, label=label, split=split))
    else:
        for class_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
            label = class_dir.name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_items.append(ImageItem(src=img_path, label=label, split=""))
    return image_items


def assign_split(src: Path, label: str, val_ratio: float, seed: int) -> str:
    if val_ratio <= 0:
        return ""

    # Stable per-file pseudo-random split without depending on global ordering.
    digest = hashlib.sha1(f"{seed}:{label}:{src.name}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) % 10_000
    return "val" if value / 10_000 < val_ratio else "train"


def build_items(
    image_items: list[ImageItem],
    output_dir: Path,
    val_ratio: float,
    seed: int,
    overwrite: bool,
    limit: int | None,
) -> list[DataItem]:
    by_label: dict[str, list[ImageItem]] = {}
    for item in image_items:
        by_label.setdefault(item.label, []).append(item)

    data_items: list[DataItem] = []
    for label in sorted(by_label):
        for image_item in by_label[label]:
            src = image_item.src
            split = image_item.split or assign_split(src, label, val_ratio, seed)
            rel_dir = Path(split) / label if split else Path(label)
            dst = output_dir / rel_dir / src.name
            if overwrite or not dst.exists():
                data_items.append(DataItem(src=src, dst=dst, label=label, split=split or "all"))
            if limit is not None and len(data_items) >= limit:
                return data_items
    return data_items


def select_center_mask(masks: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
    """Mask closest to the image center, its centroid, and whether a mask was found."""
    height, width = masks.shape[:2]
    center_x, center_y = width / 2.0, height / 2.0

    if masks.max() == 0:
        return np.ones((height, width), dtype=np.uint8), center_x, center_y, False

    best_id = None
    best_dist = float("inf")
    for mask_id in np.unique(masks):
        if mask_id == 0:
            continue
        ys, xs = np.where(masks == mask_id)
        if len(xs) == 0:
            continue
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        dist = (cx - center_x) ** 2 + (cy - center_y) ** 2
        if dist < best_dist:
            best_dist = dist
            best_id = mask_id

    if best_id is None:
        return np.ones((height, width), dtype=np.uint8), center_x, center_y, False

    ys, xs = np.where(masks == best_id)
    return (masks == best_id).astype(np.uint8), float(np.mean(xs)), float(np.mean(ys)), True


def crop_with_padding(image_rgb: np.ndarray, cx: float, cy: float, crop_size: int) -> np.ndarray:
    import cv2

    height, width = image_rgb.shape[:2]
    half = crop_size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2 - height)
    pad_right = max(0, x2 - width)

    if any((pad_top, pad_bottom, pad_left, pad_right)):
        image_rgb = cv2.copyMakeBorder(
            image_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        x1 += pad_left
        y1 += pad_top

    crop = image_rgb[y1 : y1 + crop_size, x1 : x1 + crop_size]
    if crop.shape[:2] != (crop_size, crop_size):
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    return crop


def process_batch(
    images_rgb: list[np.ndarray], masks_batch: Iterable[np.ndarray], crop_size: int, keep_background: bool
) -> tuple[list[np.ndarray], int]:
    crops: list[np.ndarray] = []
    no_mask = 0
    for image_rgb, masks in zip(images_rgb, masks_batch):
        mask_binary, cx, cy, found = select_center_mask(masks)
        no_mask += int(not found)
        if keep_background:
            crops.append(crop_with_padding(image_rgb, cx, cy, crop_size))
        else:
            white_bg = np.full_like(image_rgb, 255)
            clean_image = np.where(mask_binary[..., None] == 1, image_rgb, white_bg)
            crops.append(crop_with_padding(clean_image, cx, cy, crop_size))
    return crops, no_mask


def read_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def save_image_rgb(image_rgb: np.ndarray, save_path: Path) -> None:
    import cv2

    save_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), image_bgr)


def write_manifest(output_dir: Path, items: list[DataItem]) -> None:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "dst", "label", "split"])
        writer.writeheader()
        for item in sorted(items, key=lambda i: (i.split, i.label, i.src.name)):
            writer.writerow(
                {
                    "src": str(item.src),
                    "dst": str(item.dst),
                    "label": item.label,
                    "split": item.split,
                }
            )
    print(f"Wrote manifest: {manifest_path}")


def run() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == REPO_ROOT or REPO_ROOT in output_dir.parents:
        raise SystemExit("--output-dir must be outside the repository.")
    if output_dir == args.input_dir or args.input_dir in output_dir.parents:
        raise SystemExit("--output-dir must not be the input folder or lie inside it.")
    print(f"Input:  {args.input_dir}\nOutput: {output_dir}")
    load_runtime_dependencies()
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be >= 0 and < 1")

    route = "C (keep background)" if args.keep_background else "D (background removed)"
    print(f"Route {route}")
    image_items = iter_class_images(args.input_dir, args.layout)
    print(f"Found {len(image_items)} images in {args.input_dir}")
    labels = sorted({item.label for item in image_items})
    print(f"Classes: {', '.join(labels)}")
    splits = sorted({item.split for item in image_items if item.split})
    if splits:
        print(f"Splits: {', '.join(splits)}")

    data_items = build_items(
        image_items=image_items,
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(f"Need to process {len(data_items)} images")
    if not data_items:
        print("All images already exist. Use --overwrite to regenerate them.")
        return

    model = load_cellpose_model(args.model_type, args.precision)
    io_executor = ThreadPoolExecutor(max_workers=args.num_workers)
    no_mask_total = 0

    try:
        for start in tqdm(range(0, len(data_items), args.batch_size)):
            batch_items = data_items[start : start + args.batch_size]
            batch_images: list[np.ndarray] = []
            valid_items: list[DataItem] = []

            for item in batch_items:
                try:
                    batch_images.append(read_image_rgb(item.src))
                    valid_items.append(item)
                except Exception as exc:
                    print(f"Skip unreadable image: {item.src} ({exc})")

            if not batch_images:
                continue

            masks, _, _ = model.eval(
                batch_images,
                diameter=args.diameter,
                channels=None,
                normalize=True,
                batch_size=args.batch_size,
            )
            crops, no_mask = process_batch(batch_images, masks, args.crop_size, args.keep_background)
            no_mask_total += no_mask
            list(io_executor.map(lambda pair: save_image_rgb(pair[0], pair[1].dst), zip(crops, valid_items)))
    finally:
        io_executor.shutdown(wait=True)

    if args.save_manifest:
        write_manifest(output_dir, data_items)

    print(f"Images without any Cellpose mask (centered fallback): {no_mask_total}")
    print(f"DONE: {output_dir}")


if __name__ == "__main__":
    run()
