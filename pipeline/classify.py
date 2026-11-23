#!/usr/bin/env python3
"""Stage 2 of the inference pipeline: classify YOLO crops with the Swin-B classifier.

Preprocessing, test-time augmentation and the nine-to-seven class mapping are the
same as in eval/evaluate_classifier.py, the common evaluator behind the paper's
WBCBench2026 numbers (the analyses with their own settings are listed in
eval/README.md):

    square padding with black, resize to 256 px (img_size * 256 / 224),
    center crop to 224, ImageNet normalization; with --tta hflip (default) the
    softmax probabilities of the crop and its horizontal flip are averaged.

Decision rules (main Methods, evaluation metrics; Supplementary Section S17):

    one checkpoint (--checkpoint, default): the nine-class argmax is mapped to the
        seven evaluation classes (the per-seed rule);
    several checkpoints (--checkpoints a,b,c): the nine-class probabilities of each
        checkpoint are summed into the seven classes, averaged over the checkpoints,
        and the seven-class argmax is taken (the probability-ensemble rule).

The default checkpoint is the proposed seed-62 model,
models/swin_base_colour_aug_cwce_seed62.pth (file name of the weights deposit).

Output: one row per crop with the nine-class prediction (pred_class, pred_index,
pred_prob, top-k), the detection fields merged from detections.csv, and the
seven-class label (pred_label), its probability (pred_confidence) and the seven
class probabilities (prob_<class>). With --detections-csv, every input image that
produced no crop (status no_detection, read_failed or empty_crop) gets one row whose
pred_class and pred_label repeat that status, so it never matches a class.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import timm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
DEFAULT_CHECKPOINT = "swin_base_colour_aug_cwce_seed62.pth"
DEFAULT_MODEL_NAME = "swin_base_patch4_window7_224"
DEFAULT_IMG_SIZE = 224

# Used only when a checkpoint carries no "classes" entry (the released checkpoints do).
DEFAULT_CLASSES = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "Lymphoblast",
    "Lymphocyte",
    "Monocyte",
    "Myeloblast",
    "Neutrophil",
    "Promyelocyte",
]

# Nine-to-seven mapping: the default rules of eval/evaluate_classifier.py --use-mappings.
MERGE_RULES = [
    "Neutrophil=Band_neutrophil|Segmented_neutrophil|Neutrophil|Neutrophil_sum",
    "Lymphocyte=Lymphocyte|Variant_lymphocyte|Prolymphocyte",
    "Blast_Group=Blast|Lymphoblast|Myeloblast|Erythroblast",
]
IGNORE_CLASSES = {"Plasma_cell", "Metamyelocyte", "Myelocyte"}
SKIPPED_CLASS = "Skipped_Class"

# Detection statuses (pipeline/detect_yolo11.py) for images that yield no crop.
NO_CROP_STATUSES = ("no_detection", "read_failed", "empty_crop")

BASE_FIELDNAMES = [
    "crop_path",
    "source_path",
    "pred_class",
    "pred_index",
    "pred_prob",
    "topk_classes",
    "topk_probs",
    "det_status",
    "yolo_class_name",
    "yolo_conf",
    "x1",
    "y1",
    "x2",
    "y2",
]


def default_model_path(filename: str) -> Path:
    """$MODEL_DIR/<file>, else pipeline/models/<file>, else <repo>/models/<file>."""
    candidates = []
    if os.environ.get("MODEL_DIR"):
        candidates.append(Path(os.environ["MODEL_DIR"]).expanduser() / filename)
    candidates += [PIPELINE_DIR / "models" / filename, REPO_ROOT / "models" / filename]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


# ---------------------------------------------------------------- class mapping
def parse_merge_rules(merge_rules: list[str]) -> dict[str, str]:
    alias_to_canonical: dict[str, str] = {}
    for rule in merge_rules:
        canonical, aliases_raw = rule.split("=", 1)
        for alias in (x.strip() for x in aliases_raw.split("|")):
            if alias:
                alias_to_canonical[alias] = canonical.strip()
    return alias_to_canonical


ALIAS_TO_CANONICAL = parse_merge_rules(MERGE_RULES)


def get_canonical_label(name: str) -> str | None:
    name = name.strip()
    if name in IGNORE_CLASSES:
        return None
    return ALIAS_TO_CANONICAL.get(name, name)


# ---------------------------------------------------------------- preprocessing
def square_pad(img: Image.Image, fill: int = 0) -> Image.Image:
    """Pad the shorter side so both sides equal max(w, h), preserving aspect ratio."""
    w, h = img.size
    side = max(w, h)
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)


def build_val_transform(img_size: int = 224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Lambda(square_pad),
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_tta_transforms(img_size: int = 224, mode: str = "hflip"):
    base = build_val_transform(img_size)
    if mode == "none":
        return [("center", base)]
    if mode == "hflip":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        flip = transforms.Compose([
            transforms.Lambda(square_pad),
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return [("center", base), ("hflip", flip)]
    raise ValueError(f"Unknown TTA mode: {mode}")


class CropDataset(Dataset):
    """Returns all TTA views of one crop as a [views, 3, H, W] tensor."""

    def __init__(self, image_paths: list[Path], views):
        self.image_paths = image_paths
        self.views = views

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        return torch.stack([tf(image) for _, tf in self.views], dim=0), str(path)


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


# ---------------------------------------------------------------- models
class BPaCoClassifier(nn.Module):
    """Wrapper for full BPaCo training states (*_bpaco_full.pth, key "model_state")."""

    def __init__(self, model_name: str, num_classes: int, moco_dim: int = 128, mlp: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.feat_dim = int(self.backbone.num_features)
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        if mlp:
            self.projector = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, moco_dim),
            )
        else:
            self.projector = nn.Linear(self.feat_dim, moco_dim)
        self.center_head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, moco_dim),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


def strip_module_prefix(state: dict) -> dict:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def read_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # full training states also hold optimizer objects
        print(f"[WARN] Weights-only loading failed for {path.name} ({type(exc).__name__}); "
              "loading with full unpickling. Do this only for checkpoints you trust.")
        return torch.load(path, map_location="cpu", weights_only=False)


def load_model(checkpoint_path: Path, model_name: str | None, img_size: int | None, device: torch.device):
    checkpoint = read_checkpoint(checkpoint_path)
    ckpt_args: dict = {}
    classes = None
    if isinstance(checkpoint, dict):
        ckpt_args = checkpoint.get("args") or {}
        if not isinstance(ckpt_args, dict):
            ckpt_args = dict(vars(ckpt_args))
        classes = checkpoint.get("classes")
    if classes is None:
        print(f"[WARN] {checkpoint_path.name} has no 'classes' entry; assuming the nine training classes.")
        classes = DEFAULT_CLASSES
    classes = list(classes)
    model_name = model_name or ckpt_args.get("model_name") or DEFAULT_MODEL_NAME
    img_size = int(img_size or ckpt_args.get("img_size") or DEFAULT_IMG_SIZE)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model = BPaCoClassifier(
            model_name=model_name,
            num_classes=len(classes),
            moco_dim=int(ckpt_args.get("moco_dim") or 128),
            mlp=bool(ckpt_args.get("mlp", True)),
        )
        model.load_state_dict(strip_module_prefix(checkpoint["model_state"]), strict=True)
        kind = "bpaco_full"
    else:
        state = checkpoint
        if isinstance(checkpoint, dict):
            state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
        model.load_state_dict(strip_module_prefix(state), strict=True)
        kind = "timm_state_dict"

    model.to(device)
    model.eval()
    return model, classes, model_name, img_size, kind


def predict_probs(model, loader, device: torch.device, n_views: int, num_classes: int) -> torch.Tensor:
    """Softmax probabilities averaged over the TTA views, in loader order."""
    chunks = []
    with torch.no_grad():
        for views, _paths in tqdm(loader, desc="Classifying"):
            probs_sum = None
            for view_idx in range(n_views):
                inputs = views[:, view_idx].contiguous().to(device, non_blocking=True)
                probs_view = F.softmax(model(inputs), dim=1)
                probs_sum = probs_view if probs_sum is None else probs_sum + probs_view
            chunks.append((probs_sum / float(n_views)).cpu())
    if not chunks:
        return torch.zeros((0, num_classes))
    return torch.cat(chunks, dim=0)


def seven_class_probs(probs9: torch.Tensor, canonical_names: list[str | None],
                      canonical_classes: list[str]) -> torch.Tensor:
    """Sum the nine-class probabilities into the seven classes (float64, class-index order)."""
    probs7 = torch.zeros((probs9.shape[0], len(canonical_classes)), dtype=torch.float64)
    for cls_idx, name in enumerate(canonical_names):
        if name is not None:
            k = canonical_classes.index(name)
            probs7[:, k] = probs7[:, k] + probs9[:, cls_idx].double()
    return probs7


def decide(per_model_probs: list[torch.Tensor], classes: list[str]):
    """Nine- and seven-class decisions from per-checkpoint nine-class probabilities.

    One checkpoint: the nine-class argmax mapped to the seven classes (per-seed rule).
    Several: per-checkpoint seven-class sums averaged over checkpoints, then argmax.
    Returns probs9, probs7, canonical_classes, pred_idx9, labels7, confidences7.
    """
    canonical_names = [get_canonical_label(c) for c in classes]
    canonical_classes = sorted({c for c in canonical_names if c is not None})
    ensemble = len(per_model_probs) > 1
    probs9 = torch.stack(per_model_probs, dim=0).mean(dim=0) if ensemble else per_model_probs[0]
    probs7 = torch.stack(
        [seven_class_probs(p, canonical_names, canonical_classes) for p in per_model_probs], dim=0
    ).mean(dim=0)
    pred_idx = torch.argmax(probs9, dim=1)
    labels, confidences = [], []
    for i in range(probs9.shape[0]):
        if ensemble:
            k = int(torch.argmax(probs7[i]).item())
            labels.append(canonical_classes[k])
            confidences.append(float(probs7[i, k].item()))
        else:
            label = canonical_names[int(pred_idx[i].item())] or SKIPPED_CLASS
            labels.append(label)
            confidences.append(
                float(probs7[i, canonical_classes.index(label)].item()) if label in canonical_classes else 0.0)
    return probs9, probs7, canonical_classes, pred_idx, labels, confidences


# ---------------------------------------------------------------- detections
def read_detection_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Detections CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swin-B WBC classification stage (preprocessing, TTA and 9->7 mapping "
                    "as in eval/evaluate_classifier.py).")
    parser.add_argument("--input-dir", required=True, help="Folder containing cropped cell images (searched recursively).")
    parser.add_argument("--output-csv", required=True, help="Prediction CSV path.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=f"Classifier checkpoint. Default: {DEFAULT_CHECKPOINT} in $MODEL_DIR, pipeline/models/ or models/.",
    )
    parser.add_argument(
        "--checkpoints",
        default="",
        help="Comma-separated checkpoints (e.g. the three seeds) for a probability ensemble; overrides --checkpoint.",
    )
    parser.add_argument("--model-name", default=None,
                        help=f"timm model name. Default: from the checkpoint, else {DEFAULT_MODEL_NAME}.")
    parser.add_argument("--img-size", type=int, default=None,
                        help=f"Classifier input size. Default: from the checkpoint, else {DEFAULT_IMG_SIZE}.")
    parser.add_argument("--tta", choices=["hflip", "none"], default="hflip",
                        help="Test-time augmentation: hflip averages the crop and its horizontal flip (default, as in the paper).")
    parser.add_argument("--detections-csv", default="", help="detections.csv from detect_yolo11.py; merges box info and adds no-crop rows.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU), cpu, cuda or cuda:N.")
    parser.add_argument("--topk", type=int, default=3, help="Number of nine-class predictions kept in topk_*.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    detections_csv = Path(args.detections_csv).expanduser().resolve() if args.detections_csv else None

    if args.checkpoints.strip():
        checkpoint_paths = [Path(p.strip()).expanduser().resolve() for p in args.checkpoints.split(",") if p.strip()]
    elif args.checkpoint:
        checkpoint_paths = [Path(args.checkpoint).expanduser().resolve()]
    else:
        checkpoint_paths = [default_model_path(DEFAULT_CHECKPOINT).resolve()]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    for path in checkpoint_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {path}. Download the weights (pipeline/README.md) "
                "into pipeline/models/, set MODEL_DIR, or pass --checkpoint / --checkpoints.")

    device = resolve_device(args.device)
    image_paths = list_images(input_dir)
    detection_rows = read_detection_rows(detections_csv)
    det_by_crop = {
        str(Path(row["crop_path"]).resolve()): row
        for row in detection_rows if row.get("crop_path")
    }

    # Per-checkpoint nine-class probabilities, TTA-averaged.
    per_model_probs = []
    classes = None
    model_name = img_size = None
    for checkpoint_path in checkpoint_paths:
        model, ck_classes, ck_model_name, ck_img_size, kind = load_model(
            checkpoint_path, args.model_name, args.img_size, device)
        print(f"Checkpoint: {checkpoint_path.name} ({kind})")
        if classes is None:
            classes, model_name, img_size = ck_classes, ck_model_name, ck_img_size
        elif ck_classes != classes or ck_img_size != img_size:
            raise ValueError(f"{checkpoint_path.name}: classes or img_size differ from the first checkpoint.")
        views = build_tta_transforms(img_size, args.tta)
        loader = DataLoader(
            CropDataset(image_paths, views),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        per_model_probs.append(predict_probs(model, loader, device, len(views), len(classes)))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble = len(per_model_probs) > 1
    probs9, probs7, canonical_classes, pred_indices, labels7, confidences7 = decide(per_model_probs, classes)

    topk = max(1, min(args.topk, len(classes)))
    rows = []
    for i, path in enumerate(image_paths):
        prob_row = probs9[i]
        top_probs, top_indices = torch.topk(prob_row, k=topk)
        pred_idx = int(pred_indices[i].item())
        pred_label, pred_confidence = labels7[i], confidences7[i]
        det = det_by_crop.get(str(path.resolve()), {})
        row = {
            "crop_path": str(path),
            "source_path": det.get("source_path", ""),
            "pred_class": classes[pred_idx],
            "pred_index": pred_idx,
            "pred_prob": f"{float(prob_row[pred_idx].item()):.6f}",
            "topk_classes": "|".join(classes[int(j.item())] for j in top_indices),
            "topk_probs": "|".join(f"{float(p.item()):.6f}" for p in top_probs),
            "det_status": det.get("status", ""),
            "yolo_class_name": det.get("yolo_class_name", ""),
            "yolo_conf": det.get("yolo_conf", ""),
            "x1": det.get("x1", ""),
            "y1": det.get("y1", ""),
            "x2": det.get("x2", ""),
            "y2": det.get("y2", ""),
            "pred_label": pred_label,
            "pred_confidence": pred_confidence,
        }
        for k, name in enumerate(canonical_classes):
            row[f"prob_{name}"] = float(probs7[i, k].item())
        rows.append(row)

    # Images without a crop: one row each, never matching a class.
    n_no_crop = 0
    for det in detection_rows:
        status = det.get("status", "")
        if det.get("crop_path") or status not in NO_CROP_STATUSES:
            continue
        rows.append({
            "crop_path": "",
            "source_path": det.get("source_path", ""),
            "pred_class": status,
            "det_status": status,
            "pred_label": status,
        })
        n_no_crop += 1

    rows.sort(key=lambda r: (r["source_path"] or r["crop_path"], r["crop_path"]))

    fieldnames = BASE_FIELDNAMES + ["pred_label", "pred_confidence"] + [f"prob_{c}" for c in canonical_classes]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Model: {model_name}, input {img_size} px, TTA: {args.tta}, checkpoints: {len(checkpoint_paths)}")
    print("Seven-class rule: " + (
        "nine-class probabilities summed into seven classes, averaged over checkpoints, argmax"
        if ensemble else "nine-class argmax mapped to the seven classes"))
    print(f"Classes: {classes}")
    print(f"Crops classified: {len(image_paths)}")
    if detection_rows:
        print(f"Images without a crop (listed with their status): {n_no_crop}")
    print(f"Prediction CSV: {output_csv}")


if __name__ == "__main__":
    main()
