#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plain timm classifier training on a class-folder dataset (root/{train,val,test}/<ClassName>/*).

This trainer produced the class-weighted cross-entropy CNN baselines of main Table 3
(VGG16-BN, EfficientNet-B0, ResNet50; seeds 42/43/44) and the earlier mixup/CutMix
Swin-B leave-one-dataset-out arm of Supplementary Note 1. The exact flags are in
train/scripts/run_cnn_baselines.sh and train/scripts/run_lodo_mixup_arm.sh. The CNN
baselines were trained in the paper's second software environment (environment/cnn_env.yml:
Python 3.12, PyTorch 2.9.0, timm 0.9.12). Training runs in full precision (no AMP).

What the reported runs use: square padding, resize 1.1x, random-resized crop 224
(scale 0.8-1.0), horizontal flip, color jitter 0.2/0.2/0.2/0.05; class weights
1/sqrt(n_c) (--loss-type ce_cw); AdamW with cosine annealing; checkpoint selected by
validation macro-F1 (best_macro_f1.pth); test_results.json holds the internal score.

Other options kept from the original script (not used by any reported run):
- Other loss types: ce / ldam / ldam_drw / cb_focal
- WeightedRandomSampler (inv / sqrt_inv)
- Minority strong augmentation (--minority-boost)
- Stain augmentation (--stain-aug), early stopping (--patience)
- Decoupled head fine-tuning Stage 2 (cRT)
Mixup / CutMix (--use-mixup) is used by the earlier LODO arm only.
"""

import argparse, os, random
from pathlib import Path
from typing import List, Tuple
from collections import Counter

import numpy as np
from PIL import Image, ImageFile, ImageOps, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    from skimage import color as skcolor
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score, recall_score, confusion_matrix
import time
import json

# --------------------
# Utils
# --------------------
def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]

# --------------------
# Loss helpers
# --------------------
def make_ldam_loss(weight_tensor=None, m_list=None, s: float = 30.0):
    class LDAMLoss(nn.Module):
        def __init__(self, weight=None, m_list=None, s=30.0):
            super().__init__()
            self.weight = weight
            self.m_list = m_list
            self.s = s
        def forward(self, logits, target):
            if self.m_list is None:
                return F.cross_entropy(logits, target, weight=self.weight)
            batch_m = self.m_list[target]  # (B,)
            one_hot = torch.zeros_like(logits).scatter_(1, target.view(-1,1), 1.0)
            adjusted_logits = logits - one_hot * batch_m.view(-1,1)
            return F.cross_entropy(self.s * adjusted_logits, target, weight=self.weight)
    return LDAMLoss(weight=weight_tensor, m_list=m_list, s=s)

class WeightedSoftTargetCrossEntropy(nn.Module):
    """SoftTargetCrossEntropy with per-class weights for mixup + ce_cw."""
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight  # (C,)
    def forward(self, logits, target):
        # target: (B, C) soft labels from mixup
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -target * log_probs  # (B, C)
        if self.weight is not None:
            w = self.weight.to(logits.device)
            loss = loss * w.unsqueeze(0)  # broadcast (1,C) * (B,C)
        return loss.sum(dim=-1).mean()

def make_cb_focal_loss(beta=0.9999, gamma=2.0, samples_per_cls=None, num_classes=None):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.clip(effective_num, 1e-8, None)
    weights = weights / weights.sum() * num_classes
    class_weight = torch.tensor(weights, dtype=torch.float32)

    class CBFocal(nn.Module):
        def __init__(self, cw, gamma):
            super().__init__()
            self.cw = cw
            self.gamma = gamma
        def forward(self, logits, target):
            ce = F.cross_entropy(logits, target, weight=self.cw.to(logits.device), reduction="none")
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.gamma) * ce
            return focal.mean()
    return CBFocal(class_weight, gamma)

# --------------------
# Dataset
# --------------------
class FolderWBCDataset(Dataset):
    def __init__(self, root_split: Path, class_names: List[str],
                 label2idx: dict, transform=None, strong_transform=None,
                 split: str = "train", minority_boost=False, minority_label_set=None):
        self.root_split = Path(root_split)
        self.class_names = class_names
        self.label2idx = label2idx
        self.idx2label = {v:k for k,v in label2idx.items()}
        self.transform = transform
        self.strong_transform = strong_transform
        self.split = split
        self.minority_boost = bool(minority_boost)
        self.minority_label_set = set(minority_label_set or [])

        paths = []
        labels = []
        for cname in class_names:
            cdir = self.root_split / cname
            if not cdir.exists(): continue
            imgs = list_images(cdir)
            for img in imgs:
                paths.append(str(img))
                labels.append(cname)
        self.paths = paths
        self.labels = labels

    def __len__(self): return len(self.paths)

    def label_is_minority(self, label_name: str) -> bool:
        return label_name in self.minority_label_set

    def __getitem__(self, idx: int):
        p = Path(self.paths[idx])
        lab_name = self.labels[idx]
        y = self.label2idx[lab_name]
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                if self.transform:
                    if self.split == "train" and self.minority_boost and self.strong_transform and self.label_is_minority(lab_name):
                        img = self.strong_transform(img)
                    else:
                        img = self.transform(img)
                return img, y
        except Exception as e:
            print(f"[WARN] Failed to load image: {p}. Error: {e}")
            # Fallback: pick another random image from the same class
            # This is a simple retry logic
            new_idx = random.randint(0, len(self.paths)-1)
            return self.__getitem__(new_idx)

# --------------------
# Stain-aware augmentation
# --------------------
class RandomReinhardStyle:
    """Reinhard LAB color transfer toward a random reference image.
    Simulates cross-source / cross-stain-batch colour shifts.
    Requires scikit-image.
    """
    def __init__(self, ref_paths, p=0.30, alpha_range=(0.6, 0.9), eps=1e-6):
        if not HAS_SKIMAGE:
            raise ImportError("scikit-image is required for RandomReinhardStyle. "
                              "Install with: pip install scikit-image")
        self.p = p
        self.alpha_range = alpha_range
        self.eps = eps
        self.ref_stats = [self._get_lab_stats(Image.open(str(rp)).convert("RGB"))
                          for rp in ref_paths]
        print(f"[INFO] RandomReinhardStyle: loaded {len(self.ref_stats)} reference images")

    def _valid_mask(self, rgb01):
        return np.any(rgb01 > 0.01, axis=2)

    def _get_lab_stats(self, img):
        rgb01 = np.asarray(img).astype(np.float32) / 255.0
        lab = skcolor.rgb2lab(rgb01)
        mask = self._valid_mask(rgb01)
        pixels = lab[mask] if mask.sum() >= 10 else lab.reshape(-1, 3)
        mu = pixels.mean(axis=0)
        sigma = pixels.std(axis=0) + self.eps
        return mu, sigma

    def __call__(self, img):
        if random.random() > self.p or len(self.ref_stats) == 0:
            return img
        rgb01 = np.asarray(img).astype(np.float32) / 255.0
        lab = skcolor.rgb2lab(rgb01)
        mask = self._valid_mask(rgb01)
        pixels = lab[mask] if mask.sum() >= 10 else lab.reshape(-1, 3)
        src_mu = pixels.mean(axis=0)
        src_sigma = pixels.std(axis=0) + self.eps
        ref_mu, ref_sigma = random.choice(self.ref_stats)
        alpha = random.uniform(*self.alpha_range)
        tgt_mu = (1 - alpha) * src_mu + alpha * ref_mu
        tgt_sigma = (1 - alpha) * src_sigma + alpha * ref_sigma
        lab_out = (lab - src_mu) / src_sigma * tgt_sigma + tgt_mu
        rgb_out = np.clip(skcolor.lab2rgb(lab_out), 0, 1)
        return Image.fromarray((rgb_out * 255).astype(np.uint8))


class RandomWBAndGamma:
    """Lightweight white-balance + gamma shift (no extra deps).
    Fallback stain-like augmentation when scikit-image is unavailable.
    """
    def __init__(self, p=0.3, gamma_range=(0.9, 1.1), color_range=(0.9, 1.1)):
        self.p = p
        self.gamma_range = gamma_range
        self.color_range = color_range

    def __call__(self, img):
        if random.random() > self.p:
            return img
        factor = random.uniform(*self.color_range)
        img = ImageEnhance.Color(img).enhance(factor)
        gamma = random.uniform(*self.gamma_range)
        lut = [min(255, max(0, int(((i / 255.0) ** gamma) * 255.0))) for i in range(256)]
        return img.point(lut * 3)


# --------------------
# Transforms
# --------------------
def square_pad(img: Image.Image, fill: int = 0) -> Image.Image:
    """Pad the shorter side so both sides equal max(w, h), preserving aspect ratio."""
    w, h = img.size
    side = max(w, h)
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)


def _make_square_pad_fn(mode="black"):
    """Return a square-pad callable with the chosen fill strategy."""
    if mode == "reflect":
        def _pad(img):
            w, h = img.size
            side = max(w, h)
            pl = (side - w) // 2; pr = side - w - pl
            pt = (side - h) // 2; pb = side - h - pt
            arr = np.array(img)
            arr = np.pad(arr, ((pt, pb), (pl, pr), (0, 0)), mode="reflect")
            return Image.fromarray(arr)
        return _pad
    elif mode == "replicate":
        def _pad(img):
            w, h = img.size
            side = max(w, h)
            pl = (side - w) // 2; pr = side - w - pl
            pt = (side - h) // 2; pb = side - h - pt
            arr = np.array(img)
            arr = np.pad(arr, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
            return Image.fromarray(arr)
        return _pad
    else:  # "black" (default)
        return square_pad


def build_transforms(img_size=224, rrc_scale_lo=0.8, rrc_scale_hi=1.0,
                     cj_brightness=0.2, cj_contrast=0.2, cj_saturation=0.2, cj_hue=0.05,
                     pad_mode="black", stain_aug_fn=None):
    mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
    pad_fn = _make_square_pad_fn(pad_mode)

    train_steps = []
    # 1) stain-aware augmentation BEFORE padding (avoid black-border pollution)
    if stain_aug_fn is not None:
        train_steps.append(stain_aug_fn)
    # 2) geometry
    train_steps += [
        transforms.Lambda(pad_fn),
        transforms.Resize(int(img_size * 1.1)),
        transforms.RandomResizedCrop(img_size, scale=(rrc_scale_lo, rrc_scale_hi)),
        transforms.RandomHorizontalFlip(),
    ]
    # 3) color jitter
    if any(v > 0 for v in [cj_brightness, cj_contrast, cj_saturation, cj_hue]):
        train_steps.append(transforms.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue))
    # 4) tensor
    train_steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    train_tfms = transforms.Compose(train_steps)

    val_tfms = transforms.Compose([
        transforms.Lambda(pad_fn),
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tfms, val_tfms

def build_transforms_strong(img_size: int = 224, pad_mode="black"):
    mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
    pad_fn = _make_square_pad_fn(pad_mode)
    return transforms.Compose([
        transforms.Lambda(pad_fn),
        transforms.Resize(int(img_size*1.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.6,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

# --------------------
# Model / checkpoint
# --------------------
def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

def save_checkpoint(state: dict, outdir: str, filename: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    torch.save(state, path)
    print(f"[INFO] Saved: {path}")

# --------------------
# Train / eval
# --------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval=50, mixup_fn=None,
                    num_classes=None):
    model.train()
    running_loss=0.0
    preds_all=[]; targets_all=[]
    for step,(images,labels) in enumerate(loader):
        images=images.to(device, non_blocking=True)
        labels=labels.to(device, non_blocking=True)

        if mixup_fn is not None:
            if images.size(0) % 2 == 0:
                images, labels = mixup_fn(images, labels)

        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*images.size(0)
        pred = out.argmax(1).detach().cpu()
        if labels.ndim > 1:  # soft targets
            tgt = labels.argmax(1).detach().cpu()
        else:
            tgt = labels.detach().cpu()
        preds_all.append(pred); targets_all.append(tgt)

        if (step+1) % log_interval == 0:
            avg = running_loss / ((step+1)*loader.batch_size)
            print(f"[Train] Epoch {epoch} Step {step+1}/{len(loader)} Loss: {avg:.4f}")

    preds = torch.cat(preds_all).numpy()
    tgts  = torch.cat(targets_all).numpy()
    epoch_loss = running_loss / len(loader.dataset)
    labels = list(range(num_classes)) if num_classes is not None else None
    macro_f1 = f1_score(tgts, preds, labels=labels, average="macro", zero_division=0)
    ba = balanced_accuracy_score(tgts, preds)
    recall = recall_score(tgts, preds, labels=labels, average="macro", zero_division=0)
    return {"loss": epoch_loss, "f1": macro_f1, "ba": ba, "recall": recall}

@torch.no_grad()
def evaluate(model, loader, criterion, device, split="Val", num_classes=None):
    model.eval()
    running_loss=0.0
    preds_all=[]; targets_all=[]
    for images, labels in loader:
        images=images.to(device, non_blocking=True)
        labels=labels.to(device, non_blocking=True)
        out = model(images)
        loss = criterion(out, labels)
        running_loss += loss.item()*images.size(0)
        preds_all.append(out.argmax(1).detach().cpu())
        targets_all.append(labels.detach().cpu())
    preds = torch.cat(preds_all).numpy()
    tgts  = torch.cat(targets_all).numpy()
    loss = running_loss / len(loader.dataset)
    labels = list(range(num_classes)) if num_classes is not None else None
    f1 = f1_score(tgts, preds, labels=labels, average="macro", zero_division=0)
    ba = balanced_accuracy_score(tgts, preds)
    recall = recall_score(tgts, preds, labels=labels, average="macro", zero_division=0)
    per_class_f1 = f1_score(tgts, preds, labels=labels, average=None, zero_division=0).tolist()
    per_class_recall = recall_score(tgts, preds, labels=labels, average=None, zero_division=0).tolist()
    print(f"[{split}] Loss: {loss:.4f} | F1: {f1:.4f} | BA: {ba:.4f} | Rec: {recall:.4f}")
    return {"loss": loss, "f1": f1, "ba": ba, "recall": recall,
            "per_class_f1": per_class_f1, "per_class_recall": per_class_recall}, tgts, preds

# --------------------
# Args
# --------------------
def parse_args():
    p = argparse.ArgumentParser(
        prog="train_cnn_baseline.py",
        description="Train a timm classifier (CNN baselines, earlier mixup/CutMix LODO arm) on a "
                    "class-folder dataset. The exact flags of the paper rows are in train/scripts/.")
    p.add_argument("--root", type=str, required=True, help="Folder root containing train/val/test")
    p.add_argument("--model-name", type=str, required=True,
                   help="timm model name, e.g. resnet50, vgg16_bn, tf_efficientnet_b0")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda (current GPU), cuda:N, a GPU index N, or cpu. Falls back to cpu without CUDA.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--loss-type", type=str, default="ce_cw",
                   choices=["ce","ce_cw","ldam","ldam_drw","cb_focal"])
    p.add_argument("--sampler-weighting", type=str, default="none",
                   choices=["none","inv","sqrt_inv"])
    p.add_argument("--ldam-scale-s", type=float, default=30.0)
    p.add_argument("--ldam-max-margin", type=float, default=0.5)
    p.add_argument("--drw-epoch", type=int, default=15)
    p.add_argument("--cb-beta", type=float, default=0.9999)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--outdir", type=str, required=True,
                   help="Run folder for checkpoints, logs and test results.")
    p.add_argument("--log-interval", type=int, default=50)
    # minority boost
    p.add_argument("--minority-boost", action="store_true")
    p.add_argument("--minority-quantile", type=float, default=0.25)
    # mixup
    p.add_argument("--use-mixup", action="store_true")
    p.add_argument("--mixup-alpha", type=float, default=0.4)
    p.add_argument("--cutmix-alpha", type=float, default=0.4)
    # cRT
    p.add_argument("--crt-head-epochs", type=int, default=0)
    p.add_argument("--crt-loss-type", type=str, default="cb_focal",
                   choices=["cb_focal","ce_cw"])
    p.add_argument("--crt-lr", type=float, default=1e-4)
    p.add_argument("--crt-weight-decay", type=float, default=1e-5)
    p.add_argument("--crt-sampler-weighting", type=str, default="inv",
                   choices=["inv","sqrt_inv"])
    p.add_argument("--crt-reinit-head", action="store_true")
    # early stopping
    p.add_argument("--patience", type=int, default=0,
                   help="Early stopping patience on val macro-F1 (0=disabled)")
    # augmentation tuning
    p.add_argument("--rrc-scale-lo", type=float, default=0.8,
                   help="RandomResizedCrop min scale (default 0.8)")
    p.add_argument("--rrc-scale-hi", type=float, default=1.0,
                   help="RandomResizedCrop max scale (default 1.0)")
    p.add_argument("--pad-mode", type=str, default="black",
                   choices=["black", "reflect", "replicate"],
                   help="Square padding mode")
    p.add_argument("--cj-brightness", type=float, default=0.2)
    p.add_argument("--cj-contrast", type=float, default=0.2)
    p.add_argument("--cj-saturation", type=float, default=0.2)
    p.add_argument("--cj-hue", type=float, default=0.05)
    # stain-aware augmentation
    p.add_argument("--stain-aug", type=str, default="none",
                   choices=["none", "reinhard", "wb_gamma"],
                   help="Stain-aware augmentation type")
    p.add_argument("--style-refs-dir", type=str, default="",
                   help="Directory of reference images for Reinhard stain aug")
    p.add_argument("--stain-aug-p", type=float, default=0.3,
                   help="Probability of applying stain augmentation")
    p.add_argument("--stain-aug-alpha-lo", type=float, default=0.6,
                   help="Reinhard alpha range lower bound")
    p.add_argument("--stain-aug-alpha-hi", type=float, default=0.9,
                   help="Reinhard alpha range upper bound")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Start from random weights; for smoke tests only.")
    return p.parse_args()


def resolve_device(spec: str) -> torch.device:
    """Map --device (cuda, cuda:N, a bare GPU index N, or cpu) to a torch device.

    Falls back to the CPU when CUDA is not available, as before.
    """
    spec = str(spec).strip().lower()
    if spec.isdigit():
        spec = f"cuda:{spec}"
    if spec.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(spec)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device
    return torch.device("cpu")

# --------------------
# Main
# --------------------
def main():
    args = parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"[INFO] Device: {device}")

    root = Path(args.root).expanduser().resolve()
    train_root = root / "train"
    val_root   = root / "val"
    test_root  = root / "test"

    # Discover class names from train folder
    class_dirs = [d for d in train_root.iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in class_dirs])
    if args.num_classes and args.num_classes != len(class_names):
        print("[WARN] --num-classes ignored (derived from folders).")
    num_classes = len(class_names)
    label2idx = {c:i for i,c in enumerate(class_names)}

    # Count per class
    counts = []
    for cname in class_names:
        imgs = list_images(train_root / cname)
        counts.append(len(imgs))
    counts_arr = np.array(counts, dtype=np.int64)

    # Minority classes
    q_thr = np.quantile(counts_arr, args.minority_quantile)
    minority_set = {c for c,cnt in zip(class_names, counts_arr) if cnt <= q_thr}
    print(f"[INFO] Classes ({num_classes}): {class_names}")
    print(f"[INFO] Train counts: {counts_arr.tolist()}")
    print(f"[INFO] Minority threshold={q_thr} | minority={sorted(list(minority_set))}")

    # Stain augmentation
    stain_aug_fn = None
    if args.stain_aug == "reinhard":
        if not HAS_SKIMAGE:
            print("[WARN] scikit-image not installed, falling back to wb_gamma")
            stain_aug_fn = RandomWBAndGamma(p=args.stain_aug_p)
        else:
            ref_dir = Path(args.style_refs_dir).expanduser().resolve()
            if ref_dir.is_dir():
                ref_paths = list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.jpg")) + list(ref_dir.glob("*.jpeg"))
                if len(ref_paths) == 0:
                    print(f"[WARN] No reference images in {ref_dir}, stain aug disabled")
                else:
                    stain_aug_fn = RandomReinhardStyle(
                        ref_paths=ref_paths, p=args.stain_aug_p,
                        alpha_range=(args.stain_aug_alpha_lo, args.stain_aug_alpha_hi))
            else:
                print(f"[WARN] --style-refs-dir '{args.style_refs_dir}' not found, stain aug disabled")
    elif args.stain_aug == "wb_gamma":
        stain_aug_fn = RandomWBAndGamma(p=args.stain_aug_p)
        print(f"[INFO] Using RandomWBAndGamma (p={args.stain_aug_p})")

    # Transforms
    train_tfms, val_tfms = build_transforms(
        img_size=args.img_size,
        rrc_scale_lo=args.rrc_scale_lo,
        rrc_scale_hi=args.rrc_scale_hi,
        cj_brightness=args.cj_brightness,
        cj_contrast=args.cj_contrast,
        cj_saturation=args.cj_saturation,
        cj_hue=args.cj_hue,
        pad_mode=args.pad_mode,
        stain_aug_fn=stain_aug_fn,
    )
    print(f"[INFO] RRC scale=({args.rrc_scale_lo}, {args.rrc_scale_hi}) | pad_mode={args.pad_mode}")
    print(f"[INFO] ColorJitter(b={args.cj_brightness}, c={args.cj_contrast}, s={args.cj_saturation}, h={args.cj_hue})")
    print(f"[INFO] Stain aug: {args.stain_aug}")
    strong_tfms = build_transforms_strong(args.img_size, pad_mode=args.pad_mode) if args.minority_boost else None

    # Datasets
    train_ds = FolderWBCDataset(train_root, class_names, label2idx, transform=train_tfms,
                                strong_transform=strong_tfms, split="train",
                                minority_boost=args.minority_boost, minority_label_set=minority_set)
    val_ds = FolderWBCDataset(val_root, class_names, label2idx, transform=val_tfms, split="val")
    test_ds = FolderWBCDataset(test_root, class_names, label2idx, transform=val_tfms, split="test")

    # Numeric labels for sampler
    train_numeric_labels = [label2idx[l] for l in train_ds.labels]
    samples_per_cls = np.zeros(num_classes, dtype=np.int64)
    for i in range(num_classes):
        samples_per_cls[i] = sum(x==i for x in train_numeric_labels)
    print(f"[INFO] samples_per_cls: {samples_per_cls.tolist()}")

    # Class weight (inverse sqrt)
    class_weight = torch.tensor([1.0/np.sqrt(max(int(n),1)) for n in samples_per_cls], dtype=torch.float32)
    print(f"[INFO] class_weight -> {class_weight}")

    # LDAM margins
    m_list = torch.tensor([args.ldam_max_margin / (n**0.25) if n>0 else 0.0 for n in samples_per_cls], dtype=torch.float32)
    print(f"[INFO] LDAM m_list -> {m_list}")

    # Sampler
    sampler = None
    if args.sampler_weighting != "none":
        weights=[]
        for y in train_numeric_labels:
            freq = samples_per_cls[y]
            if args.sampler_weighting == "inv":
                w = 1.0 / max(freq,1)
            else:
                w = 1.0 / np.sqrt(max(freq,1))
            weights.append(w)
        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        print(f"[INFO] Using WeightedRandomSampler scheme={args.sampler_weighting}")

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=args.use_mixup,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True,
                            worker_init_fn=seed_worker)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=seed_worker)

    print(f"[INFO] Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")
    print(f"[INFO] Minority boost: {args.minority_boost}")
    if args.use_mixup:
        print("[INFO] drop_last=True (Mixup)")

    # Model
    model = get_model(args.model_name, num_classes=num_classes, pretrained=not args.no_pretrained).to(device)

    # Mixup/CutMix
    mixup_fn=None
    if args.use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            label_smoothing=0.0,
            num_classes=num_classes
        )
        print(f"[INFO] Mixup/CutMix enabled (mixup_alpha={args.mixup_alpha}, cutmix_alpha={args.cutmix_alpha})")

    # Loss selection
    loss_type = args.loss_type
    print(f"[INFO] Loss-type: {loss_type}")
    if args.use_mixup:
        if loss_type == "ce_cw":
            criterion = WeightedSoftTargetCrossEntropy(weight=class_weight.to(device))
            print("[INFO] Using WeightedSoftTargetCrossEntropy (mixup + ce_cw)")
        else:
            criterion = SoftTargetCrossEntropy()
    else:
        if loss_type == "ce":
            criterion = nn.CrossEntropyLoss()
        elif loss_type == "ce_cw":
            criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))
        elif loss_type == "ldam":
            criterion = make_ldam_loss(weight_tensor=None, m_list=m_list.to(device), s=args.ldam_scale_s)
        elif loss_type == "ldam_drw":
            criterion = None  # will set per epoch
        elif loss_type == "cb_focal":
            criterion = make_cb_focal_loss(beta=args.cb_beta, gamma=args.focal_gamma,
                                           samples_per_cls=samples_per_cls, num_classes=num_classes)
        else:
            raise ValueError("Unsupported loss-type")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.outdir, exist_ok=True)

    # Save initial metadata
    meta = {
        "args": vars(args),
        "classes": class_names,
        "counts": counts_arr.tolist(),
        "minority": sorted(list(minority_set))
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    best_val_f1 = 0.0
    best_val_ba = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    total_start = time.time()

    for epoch in range(1, args.epochs+1):
        ep_start = time.time()
        if loss_type == "ldam_drw" and not args.use_mixup:
            use_w = class_weight.to(device) if epoch >= args.drw_epoch else None
            criterion = make_ldam_loss(weight_tensor=use_w, m_list=m_list.to(device), s=args.ldam_scale_s)

        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        # train
        tr_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                     log_interval=args.log_interval, mixup_fn=mixup_fn,
                                     num_classes=num_classes)
        print(f"[Train] Loss: {tr_metrics['loss']:.4f} | F1: {tr_metrics['f1']:.4f} | BA: {tr_metrics['ba']:.4f} | Rec: {tr_metrics['recall']:.4f}")

        # Eval criterion when mixup used
        eval_crit = criterion
        if args.use_mixup:
            if loss_type == "ce_cw":
                eval_crit = nn.CrossEntropyLoss(weight=class_weight.to(device))
            else:
                eval_crit = nn.CrossEntropyLoss()

        val_metrics, y_true, y_pred = evaluate(model, val_loader, eval_crit, device, split="Val",
                                               num_classes=num_classes)

        # Per-class val report
        per_cls_report = {}
        for ci, cname in enumerate(class_names):
            per_cls_report[cname] = {
                "f1": val_metrics["per_class_f1"][ci],
                "recall": val_metrics["per_class_recall"][ci],
            }
        print(f"[Val] Per-class F1: { {c: round(v['f1'], 4) for c,v in per_cls_report.items()} }")

        scheduler.step()
        ep_time = time.time() - ep_start
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

        # Log
        log_entry = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "time": ep_time,
            "gpu_mem_mb": gpu_mem,
            "train": tr_metrics,
            "val": val_metrics
        }
        history.append(log_entry)
        with open(os.path.join(args.outdir, "train_log.json"), "w") as f:
            json.dump(history, f, indent=2)

        ckpt_base = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "classes": class_names,
            "args": vars(args)
        }

        # --- Save best_val_f1 ---
        improved_f1 = False
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            improved_f1 = True
            save_checkpoint({**ckpt_base, "best_macro_f1": best_val_f1},
                            args.outdir, "best_macro_f1.pth")

        # --- Save best_val_ba ---
        if val_metrics["ba"] > best_val_ba:
            best_val_ba = val_metrics["ba"]
            save_checkpoint({**ckpt_base, "best_ba": best_val_ba},
                            args.outdir, "best_ba.pth")

        # --- Save best_val_loss ---
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint({**ckpt_base, "best_val_loss": best_val_loss},
                            args.outdir, "best_val_loss.pth")

        # Log best
        with open(os.path.join(args.outdir, "best_info.txt"), "w") as f:
            f.write(f"Best F1 Epoch: {best_val_f1:.4f}\n")
            f.write(f"Best BA Epoch: {best_val_ba:.4f}\n")
            f.write(f"Best Val Loss: {best_val_loss:.4f}\n")

        # --- Early stopping ---
        if args.patience > 0:
            if improved_f1:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[INFO] Early stopping at epoch {epoch} (patience={args.patience})")
                    break

    total_time = time.time() - total_start
    print(f"\n[INFO] Stage 1 done. Best Val Macro-F1: {best_val_f1:.4f}. Total time: {total_time/60:.2f}m")

    # Stage 2 cRT
    if args.crt_head_epochs > 0:
        print("\n===== Stage 2: cRT head fine-tuning =====")
        ckpt_path = os.path.join(args.outdir, "best_macro_f1.pth")
        if not os.path.isfile(ckpt_path):
            print(f"[WARN] cRT skipped (no checkpoint {ckpt_path})")
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            print(f"[INFO] Loaded stage1 ckpt: {ckpt_path}")

            # Freeze backbone
            for name, param in model.named_parameters():
                lname = name.lower()
                if any(k in lname for k in ["head","fc","classifier","cls_head"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            if args.crt_reinit_head:
                for name, module in model.named_modules():
                    lname = name.lower()
                    if any(k in lname for k in ["head","fc","classifier","cls_head"]) and isinstance(module, nn.Linear):
                        nn.init.trunc_normal_(module.weight, std=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                        print(f"[INFO] Reinit head: {name}")

            head_params = [p for p in model.parameters() if p.requires_grad]
            optimizer_crt = torch.optim.AdamW(head_params, lr=args.crt_lr,
                                              weight_decay=args.crt_weight_decay)

            # Balanced sampler
            crt_weights=[]
            for y in train_numeric_labels:
                freq = samples_per_cls[y]
                if args.crt_sampler_weighting == "inv":
                    w = 1.0 / max(freq,1)
                else:
                    w = 1.0 / np.sqrt(max(freq,1))
                crt_weights.append(w)
            crt_weights = torch.DoubleTensor(crt_weights)
            crt_sampler = WeightedRandomSampler(crt_weights, num_samples=len(crt_weights), replacement=True)
            crt_train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                          sampler=crt_sampler, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=True)

            if args.crt_loss_type == "cb_focal":
                crt_criterion = make_cb_focal_loss(beta=args.cb_beta, gamma=args.focal_gamma,
                                                   samples_per_cls=samples_per_cls, num_classes=num_classes)
            else:
                crt_criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

            best_crt_f1 = 0.0
            crt_history = []
            for e2 in range(1, args.crt_head_epochs+1):
                ep_start = time.time()
                model.train()
                run_loss=0.0
                pa=[]; ta=[]
                for images, labels in crt_train_loader:
                    images=images.to(device); labels=labels.to(device)
                    optimizer_crt.zero_grad(set_to_none=True)
                    out=model(images)
                    loss=crt_criterion(out, labels)
                    loss.backward(); optimizer_crt.step()
                    run_loss += loss.item()*images.size(0)
                    pa.append(out.argmax(1).detach().cpu())
                    ta.append(labels.detach().cpu())
                
                # metrics
                y_true_tr = torch.cat(ta).numpy()
                y_pred_tr = torch.cat(pa).numpy()
                tl = run_loss / len(crt_train_loader.dataset)
                tr_metrics = {
                    "loss": tl,
                    "f1": f1_score(y_true_tr, y_pred_tr, labels=list(range(num_classes)),
                                   average="macro", zero_division=0),
                    "ba": balanced_accuracy_score(y_true_tr, y_pred_tr),
                    "recall": recall_score(y_true_tr, y_pred_tr, labels=list(range(num_classes)),
                                           average="macro", zero_division=0)
                }

                val_metrics, _, _ = evaluate(model, val_loader, crt_criterion, device,
                                             split="Val(cRT)", num_classes=num_classes)
                ep_time = time.time() - ep_start
                print(f"[cRT] Ep {e2} TrLoss:{tl:.4f} TrF1:{tr_metrics['f1']:.4f} ValF1:{val_metrics['f1']:.4f}")

                log_entry = {
                    "epoch": e2,
                    "lr": optimizer_crt.param_groups[0]["lr"],
                    "time": ep_time,
                    "train": tr_metrics,
                    "val": val_metrics
                }
                crt_history.append(log_entry)
                with open(os.path.join(args.outdir, "crt_log.json"), "w") as f:
                    json.dump(crt_history, f, indent=2)

                if val_metrics["f1"] > best_crt_f1:
                    best_crt_f1 = val_metrics["f1"]
                    save_checkpoint({
                        "stage":"crt_head",
                        "epoch": e2,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer_crt.state_dict(),
                        "best_macro_f1": best_crt_f1,
                        "classes": class_names,
                        "args": vars(args)
                    }, args.outdir, "crt_best_macro_f1.pth")
            
            print(f"[INFO] cRT finished. Best Val Macro-F1: {best_crt_f1:.4f}")
            crt_best = os.path.join(args.outdir, "crt_best_macro_f1.pth")
            if os.path.isfile(crt_best):
                ck = torch.load(crt_best, map_location=device)
                model.load_state_dict(ck["state_dict"])
                print(f"[INFO] Loaded best cRT head: {crt_best}")
            test_metrics, yt, yp = evaluate(model, test_loader, crt_criterion, device,
                                            split="Test(cRT)", num_classes=num_classes)
            print("[Test(cRT)] Classification report:")
            print(classification_report(yt, yp, labels=list(range(num_classes)),
                                        target_names=class_names, digits=4,
                                        zero_division=0))
    else:
        # Stage1 test - evaluate all 3 best checkpoints
        eval_crit = criterion
        if args.use_mixup:
            if loss_type == "ce_cw":
                eval_crit = nn.CrossEntropyLoss(weight=class_weight.to(device))
            else:
                eval_crit = nn.CrossEntropyLoss()

        ckpt_candidates = [
            ("best_macro_f1", "best_macro_f1.pth"),
            ("best_ba", "best_ba.pth"),
            ("best_val_loss", "best_val_loss.pth"),
        ]
        all_test_results = {}
        for ckpt_name, ckpt_file in ckpt_candidates:
            ckpt_path = os.path.join(args.outdir, ckpt_file)
            if not os.path.isfile(ckpt_path):
                print(f"[WARN] {ckpt_file} not found, skipping.")
                continue
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            ckpt_epoch = ckpt.get("epoch", "?")
            print(f"\n[INFO] Testing with {ckpt_name} (epoch {ckpt_epoch})")

            test_metrics, yt, yp = evaluate(model, test_loader, eval_crit, device,
                                            split=f"Test({ckpt_name})", num_classes=num_classes)
            print(f"[Test({ckpt_name})] Classification report:")
            print(classification_report(yt, yp, labels=list(range(num_classes)),
                                        target_names=class_names, digits=4,
                                        zero_division=0))

            # Confusion matrix
            cm = confusion_matrix(yt, yp, labels=list(range(num_classes))).tolist()

            test_report = classification_report(yt, yp, labels=list(range(num_classes)),
                                                target_names=class_names, digits=4,
                                                output_dict=True, zero_division=0)
            all_test_results[ckpt_name] = {
                "ckpt_epoch": int(ckpt_epoch) if isinstance(ckpt_epoch, (int, float)) else ckpt_epoch,
                "metrics": test_metrics,
                "report": test_report,
                "confusion_matrix": cm,
            }

        # Save combined test results
        with open(os.path.join(args.outdir, "test_results.json"), "w") as f:
            json.dump({"class_names": class_names, "results": all_test_results}, f, indent=2)
        print(f"\n[INFO] Test results saved to {os.path.join(args.outdir, 'test_results.json')}")

        # Also save confusion matrix as readable CSV for the primary (best_macro_f1) checkpoint
        if "best_macro_f1" in all_test_results:
            cm = all_test_results["best_macro_f1"]["confusion_matrix"]
            import csv
            cm_path = os.path.join(args.outdir, "confusion_matrix.csv")
            with open(cm_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([""] + class_names)
                for i, row in enumerate(cm):
                    writer.writerow([class_names[i]] + row)
            print(f"[INFO] Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
