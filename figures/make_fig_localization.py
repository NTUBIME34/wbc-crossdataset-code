#!/usr/bin/env python3
"""Figure 2 (Stage-1 detection and cropping): for each of the nine training classes, the original
field image with the YOLO11 predicted box + confidence (top row), and the resulting
single-cell crop that is fed to the classifier (bottom row).

Model  : the exact detector deployed in the pipeline (WBC-only YOLO11-m, TXL-PBC),
         run with the pipeline's settings (imgsz 512, conf 0.30, NMS IoU 0.70).
Images : one field image per class plus its detector crop. They are not distributed
         with this repository; the published figure uses PBC, AML-LMU and HRLS fields
         (licenses in the paper's Data availability statement).

Inputs:
    --field-dir  <Class>_<id>.png   uncropped field images, one per class
                                    (Basophil_..., Eosinophil_..., ..., Promyelocyte_...;
                                    .jpg/.tif/.bmp also accepted)
    --crop-dir   the matching crops: the same file name, or <name>__det00.png as written
                 by pipeline/detect_yolo11.py (crops/ folder of its output). The published
                 figure uses its default crops, the deployed crop with the 10% margin
                 ("Crops in the published figure" in figures/README.md)
    --weights    the YOLO11 WBC detector (default: models/yolo11_wbc_detection_best.pt)

The box drawn is the detection nearest the image center (the target cell). The
pipeline crops the highest-confidence box instead; on these single-cell fields the
detector returns one box per image, so the two coincide. The bottom row shows the
crop after resize-to-256 / center-crop-224; the square padding the classifier applies
first is not shown (as stated in the figure caption).

Outputs (in --out-dir):
    fig_localization_grid.png  2 x 9 grid (Fig. 2)
    fig_detection_example.png  one field with its box; used as the detection panel
                               inside the hand-drawn Fig. 1
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DETECTOR = "yolo11_wbc_detection_best.pt"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

CONF, IOU, IMGSZ = 0.30, 0.70, 512          # same settings as the deployed pipeline
C_LOC, INK = "#2e8b57", "#1a1a1a"

# the model's nine training classes, in its own class order
CLASSES = ["Basophil", "Eosinophil", "Erythroblast", "Lymphoblast", "Lymphocyte",
           "Monocyte", "Myeloblast", "Neutrophil", "Promyelocyte"]


def default_weights() -> Path:
    """$MODEL_DIR/<file>, else pipeline/models/<file>, else <repo>/models/<file>."""
    candidates = []
    if os.environ.get("MODEL_DIR"):
        candidates.append(Path(os.environ["MODEL_DIR"]).expanduser() / DETECTOR)
    candidates += [REPO / "pipeline" / "models" / DETECTOR, REPO / "models" / DETECTOR]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-1 detection-and-cropping grid (Fig. 2) and the single detection example "
                    "used in Fig. 1.",
    )
    parser.add_argument("--field-dir", required=True,
                        help="Folder with one uncropped field image per class, named <Class>_<id>.png "
                             "(required).")
    parser.add_argument("--crop-dir", required=True,
                        help="Folder with the matching detector crops (same file name, or "
                             "<name>__det00.png; required).")
    parser.add_argument("--weights", default=None,
                        help=f"YOLO11 WBC detector weights (default: {DETECTOR} in $MODEL_DIR, "
                             "pipeline/models/ or models/).")
    parser.add_argument("--device", default=None,
                        help="Inference device, e.g. cpu or 0 (default: Ultralytics picks one).")
    parser.add_argument("--example-class", default="Lymphocyte", choices=CLASSES,
                        help="Class whose field is drawn as the single detection example "
                             "(default: %(default)s).")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: $OUT_ROOT/figures, else ./outputs/figures).")
    return parser.parse_args()


def out_dir_from(arg) -> Path:
    """--out-dir, else $OUT_ROOT/figures, else ./outputs/figures; never inside the repository."""
    out = Path(arg) if arg else Path(os.environ.get("OUT_ROOT", "outputs")) / "figures"
    out = out.expanduser().resolve()
    if out == REPO or REPO in out.parents:
        sys.exit(f"[ERROR] output folder {out} is inside the repository; "
                 "pass --out-dir or set OUT_ROOT to a folder outside it")
    return out


def find_field(field_dir: Path, cls: str) -> Path:
    matches = sorted(p for p in field_dir.glob(f"{cls}_*") if p.suffix.lower() in IMG_EXTS)
    if not matches:
        sys.exit(f"[ERROR] no field image named {cls}_*.png in {field_dir}")
    if len(matches) > 1:
        print(f"[WARN] {len(matches)} field images for {cls}; using {matches[0].name}")
    return matches[0]


def find_crop(crop_dir: Path, field: Path) -> Path:
    for p in (crop_dir / field.name, crop_dir / f"{field.stem}__det00.png"):
        if p.is_file():
            return p
    sys.exit(f"[ERROR] no crop for {field.name} in {crop_dir} "
             f"(expected {field.name} or {field.stem}__det00.png)")


def classifier_view(path, size=224):
    """The classifier's resize (short side 256) and center crop (224); the square padding
    it applies first is not shown."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = int(size * 256 / 224)
    if w < h: nw, nh = s, max(1, round(h * s / w))
    else:     nw, nh = max(1, round(w * s / h)), s
    im = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - size) // 2, (nh - size) // 2
    return im.crop((left, top, left + size, top + size))


def detect_center(model, img_path, predict_kw):
    """Return (image, box, conf) for the WBC nearest the image center (the target cell)."""
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    r = model.predict(str(img_path), conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False, **predict_kw)[0]
    if len(r.boxes) == 0:
        return im, None, None
    best, bd = None, None
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        d = ((x1+x2)/2 - W/2)**2 + ((y1+y2)/2 - H/2)**2
        if bd is None or d < bd:
            bd, best = d, ((x1, y1, x2, y2), float(b.conf[0]))
    return im, best[0], best[1]


def main() -> None:
    args = parse_args()
    field_dir = Path(args.field_dir).expanduser().resolve()
    crop_dir = Path(args.crop_dir).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve() if args.weights else default_weights()
    out = out_dir_from(args.out_dir)
    if not weights.is_file():
        sys.exit(f"[ERROR] detector weights not found: {weights} (pass --weights or set MODEL_DIR)")

    from ultralytics import YOLO
    model = YOLO(str(weights))
    predict_kw = {"device": args.device} if args.device else {}

    files = {c: find_field(field_dir, c) for c in CLASSES}
    crops = {c: find_crop(crop_dir, files[c]) for c in CLASSES}
    dets = {c: detect_center(model, files[c], predict_kw) for c in CLASSES}

    out.mkdir(parents=True, exist_ok=True)
    # ---------- Figure A: the 2 x 9 localization grid ----------
    n = len(CLASSES)
    fig, axes = plt.subplots(2, n, figsize=(1.55*n, 4.3))
    for j, c in enumerate(CLASSES):
        im, box, conf = dets[c]
        ax = axes[0, j]
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        if box:
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=C_LOC, lw=1.8))
            # The label is set at 14 pt so that it prints at ~7 pt after the 0.51 reduction
            # this figure undergoes at \textwidth. At that size it is wider than the space
            # left of the box in several panels, so it is pinned to the panel's top-left in
            # axes coordinates instead of to the box corner: one box per panel, so the
            # association is unambiguous, and the nine labels line up.
            ax.text(0.02, 0.98, f"WBC {conf:.2f}", transform=ax.transAxes,
                    fontsize=14, weight="bold", color="white", ha="left", va="top",
                    bbox=dict(facecolor=C_LOC, edgecolor="none", pad=1.0))
        for s in ax.spines.values(): s.set_color("#999"); s.set_linewidth(0.6)
        ax.set_title(c, fontsize=14, weight="bold", color=INK, pad=3)

        axc = axes[1, j]
        axc.imshow(classifier_view(crops[c]))
        axc.set_xticks([]); axc.set_yticks([])
        for s in axc.spines.values(): s.set_color(C_LOC); s.set_linewidth(1.1)

    axes[0, 0].set_ylabel("Uncropped\nimage",  fontsize=14, weight="bold", color=INK)
    axes[1, 0].set_ylabel("Detection-and-\ncropping output", fontsize=14, weight="bold", color=C_LOC)
    fig.subplots_adjust(left=0.075, right=0.996, top=0.885, bottom=0.010, wspace=0.05, hspace=0.10)
    fig.savefig(out/"fig_localization_grid.png", dpi=400, facecolor="white")
    print(f"saved {out / 'fig_localization_grid.png'}")

    # ---------- Figure B: single detection example (box + confidence) ----------
    c = args.example_class
    im, box, conf = dets[c]
    f2, a2 = plt.subplots(figsize=(3.4, 3.4))
    a2.imshow(im); a2.set_xticks([]); a2.set_yticks([])
    if box:
        x1, y1, x2, y2 = box
        a2.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=C_LOC, lw=2.6))
        a2.text(x1+3, y1+3, f"WBC  {conf:.2f}", fontsize=11, weight="bold", color="white",
                ha="left", va="top",
                bbox=dict(facecolor=C_LOC, edgecolor="none", pad=2.5))
    f2.tight_layout(pad=0.2)
    f2.savefig(out/"fig_detection_example.png", dpi=400, facecolor="white")
    print(f"saved {out / 'fig_detection_example.png'}  "
          + (f"({c}: conf={conf:.3f})" if box else f"({c}: NO DETECTION)"))

    print("\n--- per-class detection confidence ---")
    for c in CLASSES:
        _, b, cf = dets[c]
        print(f"  {c:14s} conf = {cf:.3f}" if b else f"  {c:14s} NO DETECTION")


if __name__ == "__main__":
    main()
