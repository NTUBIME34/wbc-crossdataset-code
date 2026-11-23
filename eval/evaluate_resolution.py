#!/usr/bin/env python3
"""Evaluate Swin checkpoints at non-224 inference resolutions (main-text Section 3.7).

The common evaluator (eval/evaluate_classifier.py) creates timm Swin with the default
fixed img_size=224. This script passes img_size explicitly to timm.create_model so that
inference-only 224/256/384 comparisons can be executed without retraining: the same
weights, crops square-padded, resized to int(img_size * 256 / 224) and centre-cropped to
img_size, with the patch grid scaled to match.

Two settings differ from the common evaluator and are kept as run (Supplementary
Section S17): horizontal-flip test-time augmentation is the default, and the nine-class
probabilities are summed into the seven evaluation classes before the argmax
(probability-summed mapping), instead of mapping the nine-class argmax.

Outputs in --output-dir: metrics.csv, classification_report.txt, confusion_matrix.csv
and predictions.csv (per-image; keep WBCBench2026 outputs on your own machine).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REPO_ROOT = Path(__file__).resolve().parent.parent

IGNORE_CLASSES = {"Plasma_cell", "Metamyelocyte", "Myelocyte"}
MERGE = {
    "Band_neutrophil": "Neutrophil",
    "Segmented_neutrophil": "Neutrophil",
    "Neutrophil_sum": "Neutrophil",
    "Variant_lymphocyte": "Lymphocyte",
    "Prolymphocyte": "Lymphocyte",
    "Blast": "Blast_Group",
    "Lymphoblast": "Blast_Group",
    "Myeloblast": "Blast_Group",
    "Erythroblast": "Blast_Group",
}


def load_checkpoint(path: Path):
    """torch.load with weights_only=True first (every checkpoint of the paper loads this way);
    other files fall back to weights_only=False, which can run code stored in the file."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise
    except Exception as exc:
        print(f"[WARN] weights-only load of {path} failed ({type(exc).__name__}); "
              "loading with weights_only=False. Only do this for files you trust.")
        return torch.load(path, map_location="cpu", weights_only=False)


def canonical(name: str) -> str | None:
    if name in IGNORE_CLASSES:
        return None
    return MERGE.get(name, name)


def square_pad(img: Image.Image, fill: int = 0) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    left = (side - w) // 2
    right = side - w - left
    top = (side - h) // 2
    bottom = side - h - top
    return ImageOps.expand(img, border=(left, top, right, bottom), fill=fill)


def build_transform(img_size: int, hflip: bool = False):
    ops = [
        transforms.Lambda(square_pad),
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
    ]
    if hflip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(ops)


class FolderDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.root = root
        self.class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
        self.transform = transform
        self.samples: list[tuple[Path, str]] = []
        for cname in self.class_names:
            for p in sorted((root / cname).rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, cname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.transform(img), label, str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only resolution sweep of a 224-trained Swin checkpoint")
    parser.add_argument("--test-root", required=True, type=Path, help="Class-folder test set (one subfolder per class).")
    parser.add_argument("--model-path", required=True, type=Path, help="Checkpoint with 'state_dict' and 'classes'.")
    parser.add_argument("--model-name", default="swin_base_patch4_window7_224", help="timm model name.")
    parser.add_argument("--img-size", required=True, type=int, help="Inference resolution, e.g. 224, 256 or 384.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", help="cuda or cpu (falls back to cpu without a GPU).")
    parser.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir == REPO_ROOT or REPO_ROOT in args.output_dir.parents:
        raise SystemExit(f"[ERR] --output-dir {args.output_dir} is inside the repository; "
                         "write outputs to a folder outside it.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.model_path)
    classes = checkpoint.get("classes")
    if classes is None:
        raise ValueError("Checkpoint must contain classes for mapped evaluation.")
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

    model = timm.create_model(
        args.model_name,
        pretrained=False,
        num_classes=len(classes),
        img_size=args.img_size,
    )
    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"[WARN] missing={msg.missing_keys}, unexpected={msg.unexpected_keys}")
    model.to(device).eval()

    transforms_to_run = [("center", build_transform(args.img_size, False))]
    if args.tta == "hflip":
        transforms_to_run.append(("hflip", build_transform(args.img_size, True)))

    dataset = FolderDataset(args.test_root, transforms_to_run[0][1])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    model_canon = [canonical(c) for c in classes]
    y_true, y_pred, rows = [], [], []
    skipped = 0
    total = 0

    with torch.no_grad():
        for images, labels, paths in loader:
            total += len(labels)
            probs_sum = None
            for _name, tfm in transforms_to_run:
                if _name == "center":
                    batch = images
                else:
                    batch_imgs = []
                    for p in paths:
                        batch_imgs.append(tfm(Image.open(p).convert("RGB")))
                    batch = torch.stack(batch_imgs, dim=0)
                logits = model(batch.to(device, non_blocking=True))
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_sum = probs if probs_sum is None else probs_sum + probs
            probs_np = probs_sum / len(transforms_to_run)

            for i, raw_true in enumerate(labels):
                true = canonical(raw_true)
                if true is None:
                    skipped += 1
                    continue
                canonical_probs: dict[str, float] = {}
                for j, c in enumerate(model_canon):
                    if c is not None:
                        canonical_probs[c] = canonical_probs.get(c, 0.0) + float(probs_np[i, j])
                pred = max(canonical_probs, key=canonical_probs.get)
                y_true.append(true)
                y_pred.append(pred)
                row = {
                    "image_path": paths[i],
                    "source_dataset": args.test_root.name,
                    "raw_true": raw_true,
                    "raw_pred": classes[int(np.argmax(probs_np[i]))],
                    "true_label": true,
                    "pred_label": pred,
                    "pred_confidence": canonical_probs[pred],
                    "correct": int(true == pred),
                }
                for c, p in sorted(canonical_probs.items()):
                    row[f"prob_{c}"] = p
                rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_classes = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=final_classes, digits=4, zero_division=0)
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    metrics = {
        "img_size": args.img_size,
        "total": total,
        "evaluated": len(y_true),
        "skipped": skipped,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=final_classes, average="macro", zero_division=0),
    }
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    if rows:
        fields = list(rows[0].keys())
        with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    cm = confusion_matrix(y_true, y_pred, labels=final_classes)
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *final_classes])
        for c, row in zip(final_classes, cm):
            writer.writerow([c, *row.tolist()])
    print(report)
    print(metrics)


if __name__ == "__main__":
    main()
