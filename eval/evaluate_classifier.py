#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_classifier.py

The common evaluator behind the paper's WBCBench2026 classifier numbers (main Methods;
Supplementary Section S17), apart from the analyses that keep their own settings: the
resolution sweep (evaluate_resolution.py), the missed-versus-detected comparison of
Supplementary S12 (detector_abstention_quality_test.py) and the frozen DinoBloom-B probe
(frozen_probe.py); see eval/README.md. Loads one timm checkpoint and evaluates it on a
class-folder test set (one subfolder per class).

Protocol (unchanged from the runs reported in the paper):
    1) square padding with black to max(w, h), resize to int(img_size * 256 / 224)
       (256 px for 224), centre crop to img_size, ImageNet normalisation;
    2) --tta hflip (default) averages the softmax probabilities of the crop and its
       horizontal flip; --tta none uses the crop only;
    3) the nine-class argmax is taken first and then mapped (--use-mappings):
         merge   Neutrophil  = Band_neutrophil|Segmented_neutrophil|Neutrophil|Neutrophil_sum
                 Lymphocyte  = Lymphocyte|Variant_lymphocyte|Prolymphocyte
                 Blast_Group = Blast|Lymphoblast|Myeloblast|Erythroblast
         ignore  Plasma_cell, Metamyelocyte, Myelocyte (ground-truth images skipped)
       Basophil, Eosinophil, Monocyte and Promyelocyte map to themselves.

Features:
    1) Standard evaluation metrics (accuracy, balanced accuracy, classification report)
    2) Optional class mapping (merge / ignore classes)
    3) Optional output artifacts (--output-dir):
         - predictions.csv (one row per evaluated image: image_path, source_dataset,
           raw_true, raw_pred, true_label, pred_label, true_confidence,
           pred_confidence, correct, prob_<class>; rows sorted by class folder, then
           by file path)
         - confusion_counts.npy / confusion_props.npy
         - confusion_counts.png / confusion_props.png
         - classification_report.txt
         - montage.png (top-K high-confidence images per true class)
    4) Optional --dump-preds FILE: path,true,pred CSV of the evaluated images

Per-image outputs on WBCBench2026 (predictions.csv, --dump-preds files, montage.png)
contain benchmark image IDs or images. Keep them on your own machine; they must not be
committed or redistributed.

Usage (pre-cropped external set, the paper's protocol):
    python eval/evaluate_classifier.py \
      --test-root /path/to/wbcbench_phase1_cropped \
      --model-path /path/to/swin_base_colour_aug_cwce_seed62.pth \
      --model-name swin_base_patch4_window7_224 \
      --img-size 224 \
      --batch-size 32 \
      --use-mappings \
      --tta hflip \
      --output-dir /path/to/outputs/example_seed62 \
      --top-k 4

Usage (custom ignore/merge rules):
    python eval/evaluate_classifier.py \
      --test-root /path/to/test_data \
      --model-path /path/to/checkpoint.pth \
      --model-name swin_base_patch4_window7_224 \
      --use-mappings \
      --ignore-class Plasma_cell \
      --ignore-class Metamyelocyte \
      --merge "Neutrophil=Band_neutrophil|Segmented_neutrophil|Neutrophil" \
      --merge "Lymphocyte=Lymphocyte|Variant_lymphocyte|Prolymphocyte" \
      --merge "Blast_Group=Blast|Lymphoblast|Myeloblast|Erythroblast" \
      --output-dir /path/to/output_dir \
      --highlight-class Myeloblast \
      --highlight-class Promyelocyte

The runners in this folder (run_*.sh) call this script with the exact arguments of each
paper analysis; see eval/README.md.
"""

import argparse
import os
import sys
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageOps
import timm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, confusion_matrix
from pathlib import Path
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- CUSTOM MAPPING RULE ---
DEFAULT_IGNORE_CLASSES = ("Plasma_cell", "Metamyelocyte", "Myelocyte")


def default_merge_rules() -> list[str]:
    return [
        "Neutrophil=Band_neutrophil|Segmented_neutrophil|Neutrophil|Neutrophil_sum",
        "Lymphocyte=Lymphocyte|Variant_lymphocyte|Prolymphocyte",
        "Blast_Group=Blast|Lymphoblast|Myeloblast|Erythroblast",
    ]


def parse_merge_rules(merge_rules: list[str]) -> dict[str, str]:
    alias_to_canonical: dict[str, str] = {}
    for rule in merge_rules:
        if "=" not in rule:
            raise ValueError(f"Invalid --merge rule: {rule}. Expected Canonical=a|b|c")
        canonical, aliases_raw = rule.split("=", 1)
        canonical = canonical.strip()
        aliases = [x.strip() for x in aliases_raw.split("|") if x.strip()]
        if not canonical or not aliases:
            raise ValueError(f"Invalid --merge rule: {rule}. Empty canonical or aliases.")
        for a in aliases:
            alias_to_canonical[a] = canonical
    return alias_to_canonical


def get_canonical_label(name: str, alias_to_canonical: dict[str, str], ignore_set: set[str]) -> str | None:
    name = name.strip()
    if name in ignore_set:
        return None
    return alias_to_canonical.get(name, name)
# ---------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REPO_ROOT = Path(__file__).resolve().parent.parent


def outside_repo(path, what: str) -> Path:
    """Resolve an output path and refuse one inside the repository: per-image outputs (on
    WBCBench2026 they carry benchmark image IDs) must stay out of it."""
    p = Path(path).expanduser().resolve()
    if p == REPO_ROOT or REPO_ROOT in p.parents:
        sys.exit(f"[ERR] {what} {p} is inside the repository; write outputs to a folder outside it.")
    return p


def list_images(folder):
    # Sorted so that row order (and every per-image table) is the same on every machine.
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)

class FolderDataset(Dataset):
    def __init__(self, root, class_names, transform=None):
        self.root = Path(root)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform

        self.samples = []
        self.targets = []

        for cname in class_names:
            cdir = self.root / cname
            if not cdir.exists():
                print(f"[WARN] Class directory not found: {cdir}")
                continue

            imgs = list_images(cdir)
            for img_path in imgs:
                self.samples.append(img_path)
                self.targets.append(self.class_to_idx[cname])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        target = self.targets[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[ERR] Failed to load {path}: {e}")
            # Return a blank image or handle error appropriately
            img = Image.new("RGB", (224, 224))

        if self.transform:
            img = self.transform(img)

        return img, target, str(path)

def square_pad(img: Image.Image, fill: int = 0) -> Image.Image:
    """Pad the shorter side so both sides equal max(w, h), preserving aspect ratio."""
    w, h = img.size
    side = max(w, h)
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)


def build_val_transform(img_size=224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    return transforms.Compose([
        transforms.Lambda(square_pad),
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def build_tta_transforms(img_size=224, mode="none"):
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
            transforms.Normalize(mean, std)
        ])
        return [("center", base), ("hflip", flip)]
    raise ValueError(f"Unknown TTA mode: {mode}")


def load_checkpoint(path, map_location="cpu"):
    """torch.load that tries weights_only=True first (tensors and plain containers only).

    Every checkpoint of the paper (trainer exports and released weights) loads this way.
    Other files fall back to weights_only=False, which can execute code stored in the
    file: only evaluate checkpoints you trust. The loaded tensors are the same either way.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except FileNotFoundError:
        raise
    except Exception as exc:  # e.g. pickle.UnpicklingError for objects outside the allowlist
        print(f"[WARN] weights-only load of {path} failed ({type(exc).__name__}); "
              "loading with weights_only=False. Only do this for files you trust.")
        return torch.load(path, map_location=map_location, weights_only=False)


def is_hub_vit(model_name: str) -> bool:
    name = model_name.lower()
    return name.startswith(("hf-hub:", "hf_hub:")) or "dinov2" in name or "dinobloom" in name


def create_model(model_name: str, num_classes: int, img_size: int = 224) -> nn.Module:
    """Build the timm architecture; the weights are loaded from the checkpoint afterwards.

    Swin and CNN models are created with timm's defaults, exactly as in the reported runs.
    For hf-hub / DINOv2 / DinoBloom ViTs the input size is passed explicitly: the hub
    configuration of DinoBloom used at run time advertised the DINOv2 base resolution
    (518 px), which would give a 37x37 patch grid. With img_size=224 and patch size 14 timm
    builds a 16x16 grid, i.e. 256 patch tokens plus the class token = 257 position
    embeddings, which is the shape stored in the DinoBloom checkpoint and in the
    fine-tuned checkpoints (train/train_dinobloom.py). The stored weights are therefore
    loaded as they are; no position-embedding interpolation takes place
    (Supplementary Section S7).
    """
    extra = {"img_size": img_size} if is_hub_vit(model_name) else {}
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes, **extra)

def make_montage_per_class(
    img_paths: list[str],
    true_labels: list[str],
    conf_scores: np.ndarray,
    class_names: list[str],
    out_path: Path,
    top_k: int = 4,
    img_size: int = 224,
) -> None:
    per_class_indices: dict[str, list[int]] = {c: [] for c in class_names}
    per_class_scores: dict[str, list[float]] = {c: [] for c in class_names}

    for idx, c in enumerate(true_labels):
        if c not in per_class_indices:
            continue
        per_class_indices[c].append(idx)
        per_class_scores[c].append(float(conf_scores[idx]))

    rows = len(class_names)
    cols = top_k
    label_col_width = 220
    tile_w = img_size
    tile_h = img_size
    montage_w = label_col_width + cols * tile_w
    montage_h = rows * tile_h

    montage = Image.new("RGB", (montage_w, montage_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(montage)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    for r, cname in enumerate(class_names):
        display_name = cname.replace("_sum", "s").replace("_", " ")
        try:
            bbox = draw.textbbox((0, 0), display_name, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(display_name, font=font)
        text_x = (label_col_width - text_w) / 2
        text_y = r * tile_h + (tile_h - text_h) / 2
        draw.text((text_x, text_y), display_name, fill=(0, 0, 0), font=font)

        inds = per_class_indices[cname]
        if not inds:
            continue
        scores = np.array(per_class_scores[cname])
        order = np.argsort(-scores)
        top_inds = [inds[i] for i in order[:cols]]

        for cidx, sample_idx in enumerate(top_inds):
            img = Image.open(img_paths[sample_idx]).convert("RGB").resize((img_size, img_size))
            x0 = label_col_width + cidx * tile_w
            y0 = r * tile_h
            montage.paste(img, (x0, y0))

    montage.save(out_path)
    print(f"[INFO] Saved montage to: {out_path}")


def save_cm_png(
    cm: np.ndarray,
    labels: list[str],
    path: Path,
    fmt: str = "d",
    highlight_classes: list[str] | None = None,
) -> None:
    clean_labels = [l.replace("_sum", "s").replace("_", " ") for l in labels]
    plt.figure(figsize=(max(8, len(labels) * 0.8), max(7, len(labels) * 0.7)))
    im = plt.imshow(cm, cmap="Blues")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    plt.xticks(range(len(labels)), clean_labels, rotation=45, ha="right", fontsize=12)
    plt.yticks(range(len(labels)), clean_labels, fontsize=12)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val == 0:
                continue
            txt = f"{val:{fmt}}" if fmt != ".2f" else f"{val:.2f}"
            color = "white" if val > thresh else "black"
            plt.text(j, i, txt, ha="center", va="center", color=color, fontsize=14 if fmt == ".2f" else 10)

    if highlight_classes:
        ax = plt.gca()
        for cls_name in highlight_classes:
            idx = -1
            for k, l in enumerate(labels):
                if cls_name.lower() in l.lower().replace("_sum", "s").replace("_", " "):
                    idx = k
                    break
            if idx != -1:
                rect = Rectangle((idx - 0.5, idx - 0.5), 1, 1, fill=False, edgecolor="red", lw=3)
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion PNG to: {path}")


def build_parser(description: str = "Evaluate a trained model on a test folder (the paper's common evaluator)",
                 default_model_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--test-root", required=True, type=str, help="Path to test dataset root (subfolders as classes)")
    parser.add_argument("--model-path", required=True, type=str, help="Path to .pth checkpoint")
    if default_model_name is None:
        parser.add_argument("--model-name", required=True, type=str,
                            help="timm model name (e.g. resnet50, swin_base_patch4_window7_224)")
    else:
        parser.add_argument("--model-name", default=default_model_name, type=str,
                            help=f"timm model name (default: {default_model_name})")
    parser.add_argument("--img-size", default=224, type=int, help="Input image size")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--device", default="cuda", type=str, help="cuda or cpu (falls back to cpu without a GPU)")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--use-mappings", action="store_true", help="Enable custom class mapping rules (merge Neutrophils, etc.)")
    parser.add_argument("--ignore-class", action="append", default=None,
                        help="Class to ignore during evaluation (repeatable). "
                             f"Default with --use-mappings: {', '.join(DEFAULT_IGNORE_CLASSES)}.")
    parser.add_argument("--merge", action="append", default=None,
                        help="Merge rule: Canonical=a|b|c (repeatable). Replaces the default rules.")
    parser.add_argument("--output-dir", default=None, type=str,
                        help="Directory to save predictions/confusion/report/montage outputs. If not set, only print metrics.")
    parser.add_argument("--top-k", default=4, type=int, help="Top-K images per class in montage")
    parser.add_argument("--highlight-class", action="append", default=[],
                        help="Class name keywords to highlight on confusion_props diagonal (repeatable).")
    parser.add_argument("--tta", choices=["none", "hflip"], default="hflip",
                        help="Test-time augmentation (default: hflip, the external protocol). hflip averages "
                             "center-crop and horizontal-flip probabilities; none uses the center crop only.")
    parser.add_argument("--dump-preds", default=None, type=str,
                        help="Also write a path,true,pred CSV of the evaluated images (input of stats/paired_tests.py).")
    return parser


def main(argv=None, parser=None):
    parser = parser if parser is not None else build_parser()
    args = parser.parse_args(argv)
    out_dir = outside_repo(args.output_dir, "--output-dir") if args.output_dir else None
    dump_path = outside_repo(args.dump_preds, "--dump-preds") if args.dump_preds else None

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Running with --use-mappings: {args.use_mappings}")

    if args.use_mappings:
        merge_rules = args.merge if args.merge else default_merge_rules()
        ignore_classes = set(args.ignore_class) if args.ignore_class else set(DEFAULT_IGNORE_CLASSES)
        alias_to_canonical = parse_merge_rules(merge_rules)
        print(f"[INFO] Mapping enabled. Ignore: {sorted(ignore_classes)}")
        print(f"[INFO] Merge rules: {merge_rules}")
    else:
        merge_rules = []
        ignore_classes = set()
        alias_to_canonical = {}

    # 1. Load Checkpoint to check for saved class names or config
    print(f"[INFO] Loading checkpoint: {args.model_path}")
    checkpoint = load_checkpoint(args.model_path, map_location=device)

    # Try to find class names in checkpoint
    ckpt_classes = checkpoint.get("classes", None)

    # If not in checkpoint, infer from test directory
    test_root = Path(args.test_root)
    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    # Discover classes present in test folder
    test_folder_classes = sorted([d.name for d in test_root.iterdir() if d.is_dir()])

    if ckpt_classes is None:
        print("[WARN] 'classes' key not found in checkpoint. Assuming model classes match test folders.")
        model_classes = test_folder_classes
    else:
        print(f"[INFO] Found {len(ckpt_classes)} classes in checkpoint.")
        model_classes = ckpt_classes

    num_classes = len(model_classes)
    print(f"[INFO] Model Num Classes: {num_classes}")

    # 2. Build Dataset and Loader
    # We load ALL folders in test_root. We map labels later.
    # To load correctly, we just assign arbitrary indices based on sorted names.
    tta_transforms = build_tta_transforms(args.img_size, args.tta)
    dataset = FolderDataset(test_root, class_names=test_folder_classes, transform=tta_transforms[0][1])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")

    print(f"[INFO] Test samples: {len(dataset)}")
    print(f"[INFO] TTA mode: {args.tta} ({len(tta_transforms)} view(s))")

    # 3. Build Model
    print(f"[INFO] Creating model: {args.model_name}")
    model = create_model(args.model_name, num_classes, args.img_size)

    # Load Weights
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    # Cleanup state_dict keys (remove 'module.' prefix if distributed)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"[INFO] Model loaded. Missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"[INFO] Ignored {len(msg.unexpected_keys)} checkpoint keys not used by the model.")
    if msg.missing_keys:
        # Every reported run loaded with no missing keys. Missing keys mean the file is not an
        # exported classifier checkpoint (e.g. a *_bpaco_full.pth training state), and the
        # model would be evaluated with uninitialised weights.
        raise RuntimeError(
            f"{len(msg.missing_keys)} model weights were not found in {args.model_path}. "
            "Pass an exported classifier checkpoint (best_macro_f1.pth or a released .pth with a "
            "'state_dict' entry), not a full training state.")

    model.to(device)
    model.eval()

    # 4. Inference
    y_true_mapped = []
    y_pred_mapped = []
    eval_img_paths = []
    eval_true_conf = []
    prediction_rows = []

    model_canonical_names = []
    for name in model_classes:
        if args.use_mappings:
            model_canonical_names.append(get_canonical_label(name, alias_to_canonical, ignore_classes))
        else:
            model_canonical_names.append(name)

    print("[INFO] Starting inference...")
    total_samples = 0
    skipped_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, paths) in enumerate(loader):
            probs_sum = None
            for view_idx, (view_name, view_tf) in enumerate(tta_transforms):
                if view_idx == 0:
                    view_inputs = inputs
                else:
                    view_tensors = []
                    for pth in paths:
                        img = Image.open(pth).convert("RGB")
                        view_tensors.append(view_tf(img))
                    view_inputs = torch.stack(view_tensors, dim=0)
                view_inputs = view_inputs.to(device)
                outputs = model(view_inputs)
                probs_view = F.softmax(outputs, dim=1)
                probs_sum = probs_view if probs_sum is None else probs_sum + probs_view
            probs = probs_sum / float(len(tta_transforms))
            preds = torch.argmax(probs, dim=1)

            # Map batch
            # target indices correspond to test_folder_classes
            # preds indices correspond to model_classes

            targets_np = targets.cpu().numpy()
            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()

            for i in range(len(targets_np)):
                total_samples += 1

                # Get raw names
                gt_name = test_folder_classes[targets_np[i]]
                pred_name = model_classes[preds_np[i]]

                if args.use_mappings:
                    # Apply custom rules
                    mapped_gt = get_canonical_label(gt_name, alias_to_canonical, ignore_classes)
                    mapped_pred = get_canonical_label(pred_name, alias_to_canonical, ignore_classes)

                    if mapped_gt is None:
                        skipped_samples += 1
                        continue # Skip this sample (e.g. Plasma_cell)

                    # If pred maps to None (shouldn't happen with current rules unless model predicts Plasma_cell/Meta),
                    if mapped_pred is None:
                        mapped_pred = "Skipped_Class"

                    y_true_mapped.append(mapped_gt)
                    y_pred_mapped.append(mapped_pred)

                    # Confidence for mapped true class = sum of probs over model classes
                    # that map to this canonical class
                    canonical_probs = {c: 0.0 for c in sorted({x for x in model_canonical_names if x is not None})}
                    conf_val = 0.0
                    for cls_idx, canonical_name in enumerate(model_canonical_names):
                        if canonical_name == mapped_gt:
                            conf_val += float(probs_np[i, cls_idx])
                        if canonical_name is not None:
                            canonical_probs[canonical_name] = canonical_probs.get(canonical_name, 0.0) + float(probs_np[i, cls_idx])
                    eval_true_conf.append(conf_val)
                    eval_img_paths.append(paths[i])
                    prediction_rows.append({
                        "image_path": paths[i],
                        "source_dataset": Path(paths[i]).parts[-3] if len(Path(paths[i]).parts) >= 3 else "",
                        "raw_true": gt_name,
                        "raw_pred": pred_name,
                        "true_label": mapped_gt,
                        "pred_label": mapped_pred,
                        "true_confidence": conf_val,
                        "pred_confidence": float(canonical_probs.get(mapped_pred, 0.0)),
                        "correct": int(mapped_gt == mapped_pred),
                        "probs": canonical_probs,
                    })
                else:
                    y_true_mapped.append(gt_name)
                    y_pred_mapped.append(pred_name)

                    # Confidence for true class from model output if class exists in model classes
                    conf_val = 0.0
                    if gt_name in model_classes:
                        gt_idx = model_classes.index(gt_name)
                        conf_val = float(probs_np[i, gt_idx])
                    eval_true_conf.append(conf_val)
                    eval_img_paths.append(paths[i])
                    class_probs = {model_classes[cls_idx]: float(probs_np[i, cls_idx]) for cls_idx in range(len(model_classes))}
                    prediction_rows.append({
                        "image_path": paths[i],
                        "source_dataset": Path(paths[i]).parts[-3] if len(Path(paths[i]).parts) >= 3 else "",
                        "raw_true": gt_name,
                        "raw_pred": pred_name,
                        "true_label": gt_name,
                        "pred_label": pred_name,
                        "true_confidence": conf_val,
                        "pred_confidence": float(class_probs.get(pred_name, 0.0)),
                        "correct": int(gt_name == pred_name),
                        "probs": class_probs,
                    })

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(loader)} done.")

    # 5. Metrics
    print("\n" + "="*40)
    print("TEST REPORT")
    print("="*40)

    final_classes = sorted(list(set(y_true_mapped) | set(y_pred_mapped)))
    if len(y_true_mapped) == 0:
        print("[ERR] No samples left after mapping/ignoring. Please adjust --ignore-class / --merge settings.")
        return

    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    bal_acc = balanced_accuracy_score(y_true_mapped, y_pred_mapped)

    print(f"Total Processed: {total_samples}")
    print(f"Evaluated:       {len(y_true_mapped)}")
    print(f"Skipped:         {skipped_samples}")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Balanced Acc:    {bal_acc:.4f}")
    print("-" * 40)

    # Classification Report
    # Use zero_division=0 to handle cases where a class is never predicted
    print("Classification Report:")
    print(classification_report(y_true_mapped, y_pred_mapped, target_names=final_classes, digits=4, zero_division=0))

    print("-" * 40)

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "true", "pred"])
            writer.writerows(zip(eval_img_paths, y_true_mapped, y_pred_mapped))
        print(f"[INFO] Saved {len(eval_img_paths)} path,true,pred rows to: {dump_path}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_true_mapped, y_pred_mapped, target_names=final_classes, digits=4, zero_division=0))
        print(f"[INFO] Saved classification_report.txt to: {out_dir}")

        if prediction_rows:
            prob_classes = sorted({c for row in prediction_rows for c in row["probs"].keys()})
            pred_csv = out_dir / "predictions.csv"
            with open(pred_csv, "w", encoding="utf-8", newline="") as f:
                fieldnames = [
                    "image_path",
                    "source_dataset",
                    "raw_true",
                    "raw_pred",
                    "true_label",
                    "pred_label",
                    "true_confidence",
                    "pred_confidence",
                    "correct",
                ] + [f"prob_{c}" for c in prob_classes]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in prediction_rows:
                    out_row = {k: row[k] for k in fieldnames if k in row}
                    for c in prob_classes:
                        out_row[f"prob_{c}"] = row["probs"].get(c, 0.0)
                    writer.writerow(out_row)
            print(f"[INFO] Saved predictions.csv to: {pred_csv}")

        cm_counts = confusion_matrix(y_true_mapped, y_pred_mapped, labels=final_classes)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = cm_counts.sum(axis=1, keepdims=True)
            cm_props = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums != 0)

        np.save(out_dir / "confusion_counts.npy", cm_counts)
        np.save(out_dir / "confusion_props.npy", cm_props)
        print(f"[INFO] Saved confusion arrays to: {out_dir}")

        save_cm_png(cm_counts, final_classes, out_dir / "confusion_counts.png", fmt="d")
        save_cm_png(cm_props, final_classes, out_dir / "confusion_props.png", fmt=".2f", highlight_classes=args.highlight_class)

        if len(eval_img_paths) > 0:
            make_montage_per_class(
                img_paths=eval_img_paths,
                true_labels=y_true_mapped,
                conf_scores=np.array(eval_true_conf, dtype=float),
                class_names=sorted(set(y_true_mapped)),
                out_path=out_dir / "montage.png",
                top_k=args.top_k,
                img_size=args.img_size,
            )


if __name__ == "__main__":
    main()
