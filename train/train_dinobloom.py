#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tuning trainer for ViT foundation backbones (DinoBloom-B) on class-folder WBC data.

This is train_classifier.py with two additions for hf-hub ViT models such as
hf-hub:1aurent/vit_base_patch14_224.dinobloom:
  * img_size is passed to timm.create_model (see _timm_extra_kwargs), and
  * layer-wise LR decay is applied per transformer block instead of per Swin stage.
For Swin and CNN backbones it behaves exactly like train_classifier.py.

It produced the fine-tuned DinoBloom-B row of main Table 3 and both DinoBloom-B
configurations of Supplementary Table S16 (runners in train/scripts/). The objective is

    loss = CE + bpaco_weight * BPaCo

and the BPaCo term (--bpaco-weight > 0, S16 BPaCo variant only) needs the upstream
loss file fetched with train/third_party/fetch_bpaco_loss.sh. The classifier branch is
exported as a plain timm checkpoint (best_macro_f1.pth) for eval/evaluate_dinobloom.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

import timm
from timm.data.mixup import Mixup
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAIN_DIR = Path(__file__).resolve().parent
# Upstream BPaCo loss (github.com/Davidczy/BPaCo, commit cbe5775, losses.py). It has no
# licence, so it is downloaded by the user instead of being shipped with this code.
BPACO_LOSS_FILE = TRAIN_DIR / "third_party" / "bpaco_losses.py"
BPACO_FETCH_SCRIPT = TRAIN_DIR / "third_party" / "fetch_bpaco_loss.sh"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def import_from_file(module_name: str, path: Path):
    """Import a helper module from an explicit file path (independent of sys.path).

    The module is registered in sys.modules so objects built from it stay picklable
    for data-loader workers.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_bpaco_loss():
    """Return the upstream BPaCoLoss class; only called when --bpaco-weight > 0.

    The module defines classes only (no random state is touched at import), so
    importing it here instead of at the top of the file does not change training.
    """
    if not BPACO_LOSS_FILE.is_file():
        raise SystemExit(
            "[ERROR] --bpaco-weight > 0 needs the BPaCo loss, which is not redistributed "
            f"with this code. Download and verify it once with:\n    bash {BPACO_FETCH_SCRIPT}"
        )
    return import_from_file("bpaco_losses", BPACO_LOSS_FILE).BPaCoLoss


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


def list_images(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def square_pad(img: Image.Image, fill: int = 0) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)


class TwoViewTransform:
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def __call__(self, img):
        return self.t1(img), self.t2(img)


class FolderWBCDataset(Dataset):
    def __init__(self, root_split: Path, class_names: list[str], transform=None):
        self.root_split = Path(root_split)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform
        self.paths: list[str] = []
        self.targets: list[int] = []
        for cname in class_names:
            for p in list_images(self.root_split / cname):
                self.paths.append(str(p))
                self.targets.append(self.class_to_idx[cname])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        target = self.targets[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            idx = random.randint(0, len(self.paths) - 1)
            return self.__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        return img, target


class WeightedSoftTargetCrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -target * log_probs
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device).unsqueeze(0)
        return loss.sum(dim=-1).mean()


class RandomWBAndGamma:
    def __init__(self, p=0.3, gamma_range=(0.9, 1.1), color_range=(0.9, 1.1)):
        self.p = p
        self.gamma_range = gamma_range
        self.color_range = color_range

    def __call__(self, img):
        if random.random() > self.p:
            return img
        gamma = random.uniform(*self.gamma_range)
        lut = [min(255, max(0, int(((i / 255.0) ** gamma) * 255.0))) for i in range(256)]
        img = img.point(lut * 3)
        arr = np.asarray(img).astype(np.float32)
        gains = np.array([random.uniform(*self.color_range) for _ in range(3)], dtype=np.float32)
        arr = np.clip(arr * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class RandomTensorGaussianNoise:
    def __init__(self, p=0.15, std=0.01):
        self.p = p
        self.std = std

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def _timm_extra_kwargs(model_name: str, img_size: int) -> dict:
    """Extra timm.create_model kwargs for ViT foundation backbones.

    The timm configuration that DinoBloom inherits from DINOv2 advertises a 518 px
    input, but the released DinoBloom checkpoint stores 224 x 224 position embeddings
    (16 x 16 patches + class token = 257 tokens), the resolution DinoBloom was
    self-supervised at. Passing img_size=224 makes timm build that same patch grid, so
    the pretrained position embeddings load unchanged (no interpolation). Swin/CNN
    backbones already encode their resolution in the model name, so we leave them
    untouched to keep the proposed Swin-B / ResNet50 runs byte-for-byte identical.
    """
    n = model_name.lower()
    if n.startswith(("hf-hub:", "hf_hub:")) or "dinov2" in n or "dinobloom" in n:
        return {"img_size": img_size}
    return {}


class BPaCoClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        moco_dim: int = 128,
        pretrained: bool = True,
        mlp: bool = True,
        grad_checkpointing: bool = False,
        img_size: int = 224,
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            **_timm_extra_kwargs(model_name, img_size))
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
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

    def encode(self, x):
        feat = self.backbone(x)
        z = F.normalize(self.projector(feat), dim=1)
        logits = self.classifier(feat)
        return feat, z, logits

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

    def center_logits(self):
        return F.normalize(self.center_head(self.classifier.weight), dim=1)


class Queue:
    def __init__(self, size: int, dim: int, num_classes: int, device: torch.device):
        self.size = size
        self.ptr = 0
        self.features = F.normalize(torch.randn(size, dim, device=device), dim=1)
        self.labels = torch.randint(0, num_classes, (size,), device=device)

    @torch.no_grad()
    def enqueue(self, keys, labels):
        keys = keys.detach()
        labels = labels.detach()
        bsz = keys.shape[0]
        if bsz >= self.size:
            self.features.copy_(keys[-self.size :])
            self.labels.copy_(labels[-self.size :])
            self.ptr = 0
            return
        end = self.ptr + bsz
        if end <= self.size:
            self.features[self.ptr : end] = keys
            self.labels[self.ptr : end] = labels
        else:
            first = self.size - self.ptr
            self.features[self.ptr :] = keys[:first]
            self.labels[self.ptr :] = labels[:first]
            self.features[: end % self.size] = keys[first:]
            self.labels[: end % self.size] = labels[first:]
        self.ptr = end % self.size


def build_transforms(args):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean, std)
    rgb_mean = mean
    ra_params = dict(
        translate_const=int(args.img_size * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
    )

    stain_aug = RandomWBAndGamma(p=args.stain_aug_p) if args.stain_aug == "wb_gamma" else None

    if args.recipe == "paper":
        # RandAugment (timm implementation, see randaugment.py) is only used by the BPaCo views.
        rand_augment_transform = import_from_file(
            "randaugment", TRAIN_DIR / "randaugment.py").rand_augment_transform

        pre = [stain_aug] if stain_aug is not None else []
        view1 = transforms.Compose([
            *pre,
            transforms.RandomResizedCrop(args.img_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0),
            rand_augment_transform(f"rand-n{args.randaug_n}-m{args.randaug_m}-mstd0.5", ra_params),
            transforms.ToTensor(),
            normalize,
        ])
        view2 = transforms.Compose([
            *pre,
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val = transforms.Compose([
            transforms.Resize(int(args.img_size * 256 / 224)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        common = [transforms.Lambda(square_pad)]
        policy = args.aug_policy
        if policy in {"stain_only", "stain_blur_noise"} and stain_aug is None:
            stain_aug = RandomWBAndGamma(p=args.stain_aug_p)
        if stain_aug is not None:
            common.append(stain_aug)

        if policy == "meta_default":
            common += [
                transforms.Resize(int(args.img_size * 1.1)),
                transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            ]
        elif policy == "conservative":
            common += [
                transforms.Resize(int(args.img_size * 1.08)),
                transforms.RandomResizedCrop(args.img_size, scale=(args.safe_crop_scale, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(args.rotation_degrees),
                transforms.ColorJitter(
                    0.08 * args.color_jitter_strength,
                    0.08 * args.color_jitter_strength,
                    0.08 * args.color_jitter_strength,
                    0.02 * args.color_jitter_strength,
                ),
            ]
        elif policy == "color_only":
            common += [
                transforms.Resize(int(args.img_size * 1.08)),
                transforms.CenterCrop(args.img_size),
                transforms.ColorJitter(
                    0.12 * args.color_jitter_strength,
                    0.12 * args.color_jitter_strength,
                    0.12 * args.color_jitter_strength,
                    0.03 * args.color_jitter_strength,
                ),
            ]
        elif policy == "stain_only":
            common += [
                transforms.Resize(int(args.img_size * 1.08)),
                transforms.CenterCrop(args.img_size),
            ]
        elif policy == "geometry_only":
            common += [
                transforms.Resize(int(args.img_size * 1.08)),
                transforms.RandomResizedCrop(args.img_size, scale=(args.safe_crop_scale, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(args.rotation_degrees),
            ]
        elif policy in {"blur_noise", "stain_blur_noise"}:
            common += [
                transforms.Resize(int(args.img_size * 1.08)),
                transforms.RandomResizedCrop(args.img_size, scale=(args.safe_crop_scale, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    0.08 * args.color_jitter_strength,
                    0.08 * args.color_jitter_strength,
                    0.08 * args.color_jitter_strength,
                    0.02 * args.color_jitter_strength,
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=args.blur_p),
            ]
        else:
            raise ValueError(f"Unknown --aug-policy: {policy}")

        common += [
            transforms.ToTensor(),
        ]
        if policy in {"blur_noise", "stain_blur_noise"}:
            common.append(RandomTensorGaussianNoise(p=args.noise_p, std=args.noise_std))
        common += [
            normalize,
        ]
        view1 = transforms.Compose(common)
        view2 = transforms.Compose(common)
        val = transforms.Compose([
            transforms.Lambda(square_pad),
            transforms.Resize(int(args.img_size * 1.1)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            normalize,
        ])
    return TwoViewTransform(view1, view2), val


def class_weights_sqrt_inv(counts: np.ndarray) -> torch.Tensor:
    weights = np.array([1.0 / np.sqrt(max(int(n), 1)) for n in counts], dtype=np.float32)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def soft_ce(logits, soft_targets, weight=None):
    log_probs = F.log_softmax(logits, dim=1)
    loss = -soft_targets * log_probs
    if weight is not None:
        loss = loss * weight.to(logits.device).unsqueeze(0)
    return loss.sum(dim=1).mean()


def mixup_targets_to_hard(soft_targets):
    return soft_targets.argmax(dim=1)


def get_swin_layer_id(name: str, max_layer_id: int = 5) -> int:
    if not name.startswith("backbone."):
        return max_layer_id
    inner = name[len("backbone."):]
    if inner.startswith("patch_embed"):
        return 0
    if inner.startswith("layers."):
        parts = inner.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1]) + 1
    return max_layer_id - 1


def get_vit_layer_id(name: str, num_blocks: int) -> int:
    """Per-block layer id for ViT / DINOv2 backbones (e.g. DinoBloom).

    Mirrors the Swin convention used for the matched-recipe control so the SAME
    layer-wise-LR-decay semantics apply across architectures:
      stem (patch_embed / cls_token / pos_embed / reg|mask token) -> 0
      transformer block i                                         -> i + 1
      final norm / anything else                                  -> num_blocks
    The deepest backbone level therefore gets lr * decay**1 (== 0.8*lr with
    decay=0.80) and the classifier/projector heads get lr * head_lr_mult, exactly
    as in the Swin-B / ResNet50 runs. Head params are handled in make_optimizer.
    """
    if not name.startswith("backbone."):
        return num_blocks + 1
    inner = name[len("backbone."):]
    if inner.startswith(("patch_embed", "cls_token", "pos_embed", "reg_token", "mask_token")):
        return 0
    if inner.startswith("blocks."):
        parts = inner.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1]) + 1
    return num_blocks


def _detect_vit_num_blocks(model) -> int:
    """Number of transformer blocks if the backbone is a ViT, else 0 (Swin/CNN)."""
    idxs = set()
    for name, _ in model.named_parameters():
        m = re.search(r"backbone\.blocks\.(\d+)\.", name)
        if m:
            idxs.add(int(m.group(1)))
    return (max(idxs) + 1) if idxs else 0


def make_optimizer(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if not args.layer_wise_lr_decay:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    vit_blocks = _detect_vit_num_blocks(model)
    if vit_blocks > 0:
        max_layer_id = vit_blocks + 1
        layer_id_fn = lambda n: get_vit_layer_id(n, vit_blocks)
        print(f"[INFO] LLRD: ViT backbone detected ({vit_blocks} blocks); per-block "
              f"layer-wise LR decay (decay={args.layer_decay}, head_mult={args.head_lr_mult}).")
    else:
        max_layer_id = 5
        layer_id_fn = lambda n: get_swin_layer_id(n, max_layer_id=max_layer_id)
        print("[INFO] LLRD: Swin/CNN layer-id scheme (max_layer_id=5).")

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("classifier.", "projector.", "center_head.")):
            lr = args.lr * args.head_lr_mult
            layer_id = "head"
        else:
            lid = layer_id_fn(name)
            lr = args.lr * (args.layer_decay ** (max_layer_id - lid))
            layer_id = f"layer_{lid}"
        decay = 0.0 if param.ndim == 1 or name.endswith(".bias") else args.weight_decay
        key = (layer_id, lr, decay)
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": decay, "name": str(layer_id)})
        groups[key]["params"].append(param)
    print("[INFO] Layer-wise LR decay param groups:")
    for group in sorted(groups.values(), key=lambda g: g["lr"]):
        print(f"  {group['name']}: lr={group['lr']:.8f}, wd={group['weight_decay']}, n={len(group['params'])}")
    return torch.optim.AdamW(list(groups.values()), lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(optimizer, args):
    if args.warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def lr_lambda(epoch_idx):
        epoch_num = epoch_idx + 1
        if epoch_num <= args.warmup_epochs:
            return max(epoch_num / float(args.warmup_epochs), 1e-8)
        progress = (epoch_num - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def export_timm_checkpoint(model, args, class_names, path: Path, epoch: int, metric_name: str, metric_value: float):
    export_model = timm.create_model(
        args.model_name, pretrained=False, num_classes=len(class_names),
        **_timm_extra_kwargs(args.model_name, args.img_size))
    src = model.module if hasattr(model, "module") else model
    export_state = export_model.state_dict()
    src_backbone_state = src.backbone.state_dict()
    for key in export_state:
        if key in src_backbone_state and export_state[key].shape == src_backbone_state[key].shape:
            export_state[key] = src_backbone_state[key].detach().cpu()
    export_model.load_state_dict(export_state, strict=False)
    classifier = export_model.get_classifier()
    classifier.weight.data.copy_(src.classifier.weight.detach().cpu())
    classifier.bias.data.copy_(src.classifier.bias.detach().cpu())
    torch.save({
        "epoch": epoch,
        "state_dict": export_model.state_dict(),
        "classes": class_names,
        "args": vars(args),
        metric_name: metric_value,
    }, path)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names, max_batches: int = 0):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    seen = 0
    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        losses.append(loss.item() * images.size(0))
        seen += images.size(0)
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
        if max_batches and batch_idx >= max_batches:
            break
    labels = list(range(len(class_names)))
    return {
        "loss": float(np.sum(losses) / max(seen, 1)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "ba": float(balanced_accuracy_score(y_true, y_pred)),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            digits=4, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_one_epoch(model, queue, loader, criterion_bpaco, ce_criterion, optimizer, scaler, device, args, mixup_fn, class_weight):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_bpaco = 0.0
    seen = 0
    y_true = []
    y_pred = []
    for step, ((im_q, im_k), targets) in enumerate(loader, start=1):
        im_q = im_q.to(device, non_blocking=True)
        im_k = im_k.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        ce_targets = targets
        if args.use_mixup:
            im_q, ce_targets = mixup_fn(im_q, targets)
            hard_for_bpaco = mixup_targets_to_hard(ce_targets)
        else:
            hard_for_bpaco = targets

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
            _, q, logits_q = model.encode(im_q)
            if args.use_mixup:
                ce_loss = soft_ce(logits_q, ce_targets, class_weight if args.loss_type == "ce_cw" else None)
            else:
                ce_loss = ce_criterion(logits_q, targets)
            if args.bpaco_weight > 0:
                _, k, logits_k = model.encode(im_k)
                features = torch.cat([q, k, queue.features.detach()], dim=0)
                labels = torch.cat([hard_for_bpaco, targets, queue.labels.detach()], dim=0)
                logits = torch.cat([logits_q, logits_k], dim=0)
                center_logits = model.center_logits()[: args.num_classes]
                bpaco_loss = criterion_bpaco(features, labels, logits, center_logits)
            else:
                k = q.detach()
                bpaco_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            loss = ce_loss + args.bpaco_weight * bpaco_loss

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.bpaco_weight > 0:
            queue.enqueue(k, targets)

        total_loss += loss.item() * im_q.size(0)
        total_ce += ce_loss.item() * im_q.size(0)
        total_bpaco += bpaco_loss.item() * im_q.size(0)
        seen += im_q.size(0)
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(logits_q.argmax(dim=1).detach().cpu().tolist())

        if step % args.log_interval == 0:
            print(
                f"[Train] step {step}/{len(loader)} "
                f"loss={total_loss/(step*im_q.size(0)):.4f} "
                f"ce={total_ce/(step*im_q.size(0)):.4f} "
                f"bpaco={total_bpaco/(step*im_q.size(0)):.4f}"
            )
        if args.max_train_steps and step >= args.max_train_steps:
            break
    labels_np = list(range(args.num_classes))
    return {
        "loss": float(total_loss / max(seen, 1)),
        "ce": float(total_ce / max(seen, 1)),
        "bpaco": float(total_bpaco / max(seen, 1)),
        "f1": float(f1_score(y_true, y_pred, labels=labels_np, average="macro", zero_division=0)),
        "ba": float(balanced_accuracy_score(y_true, y_pred)),
    }


def build_balanced_sampler(dataset, num_classes, scheme="inv"):
    labels = dataset.targets
    counts = np.array([sum(y == i for y in labels) for i in range(num_classes)], dtype=np.float64)
    weights = []
    for y in labels:
        freq = max(counts[y], 1.0)
        weights.append(1.0 / freq if scheme == "inv" else 1.0 / np.sqrt(freq))
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def make_eval_ce(args, class_weight, device):
    return nn.CrossEntropyLoss(weight=class_weight.to(device)) if args.loss_type == "ce_cw" else nn.CrossEntropyLoss()


def train_crt(model, train_root, val_loader, test_loader, class_names, label2idx, val_tf, class_weight, device, args, outdir):
    if args.crt_epochs <= 0:
        return
    if (outdir / "crt_test_results.json").exists():
        print("[INFO] cRT skipped; crt_test_results.json already exists.")
        return
    print(f"\n===== cRT Stage: {args.crt_epochs} epochs =====")
    best_path = outdir / "best_macro_f1_bpaco_full.pth"
    if not best_path.exists():
        print(f"[WARN] cRT skipped; missing {best_path}")
        return
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if args.crt_freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

    crt_train_ds = FolderWBCDataset(train_root, class_names, val_tf)
    sampler = build_balanced_sampler(crt_train_ds, len(class_names), args.crt_sampler_weighting)
    crt_loader = DataLoader(
        crt_train_ds, batch_size=args.batch_size, shuffle=False, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=False, worker_init_fn=seed_worker)

    if args.crt_loss_type == "ce_cw":
        criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.crt_lr,
        weight_decay=args.crt_weight_decay,
    )
    best_f1 = -1.0
    crt_history = []
    for epoch in range(1, args.crt_epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        y_true = []
        y_pred = []
        for images, targets in crt_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * images.size(0)
            seen += images.size(0)
            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(logits.argmax(1).detach().cpu().tolist())
        tr = {
            "loss": float(total / max(seen, 1)),
            "f1": float(f1_score(y_true, y_pred, labels=list(range(len(class_names))), average="macro", zero_division=0)),
            "ba": float(balanced_accuracy_score(y_true, y_pred)),
        }
        val = evaluate(model, val_loader, criterion, device, class_names, args.max_eval_batches)
        crt_history.append({"epoch": epoch, "train": tr, "val": {k: v for k, v in val.items() if k not in {"report", "confusion_matrix"}}})
        (outdir / "crt_log.json").write_text(json.dumps(crt_history, indent=2), encoding="utf-8")
        print(f"[cRT] epoch={epoch} train_f1={tr['f1']:.4f} val_f1={val['f1']:.4f} val_ba={val['ba']:.4f}")
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer": opt.state_dict(),
                "classes": class_names,
                "args": vars(args),
                "best_macro_f1": best_f1,
            }, outdir / "crt_best_macro_f1_bpaco_full.pth")
            export_timm_checkpoint(model, args, class_names, outdir / "crt_best_macro_f1.pth", epoch, "crt_best_macro_f1", best_f1)

    best_crt = outdir / "crt_best_macro_f1_bpaco_full.pth"
    if best_crt.exists():
        ckpt = torch.load(best_crt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test = evaluate(model, test_loader, criterion, device, class_names, args.max_eval_batches)
        (outdir / "crt_test_results.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
        print(f"[cRT Test] f1={test['f1']:.4f} ba={test['ba']:.4f}")


def parse_args():
    p = argparse.ArgumentParser(
        prog="train_dinobloom.py",
        description="Fine-tune a ViT foundation backbone (DinoBloom-B) or any timm model on a "
                    "class-folder dataset. The exact flags of every paper row are in "
                    "train/scripts/; the defaults below are not a paper recipe.")
    p.add_argument("--root", required=True,
                   help="Dataset root with train/, val/ and test/, one sub-folder per class.")
    p.add_argument("--model-name", default="swin_base_patch4_window7_224",
                   help="timm model name, e.g. hf-hub:1aurent/vit_base_patch14_224.dinobloom "
                        "(append @<revision> to pin a Hugging Face revision).")
    p.add_argument("--recipe", choices=["paper", "meta"], default="paper",
                   help="'paper': BPaCo two-view augmentation; 'meta': single-view policy chosen by --aug-policy.")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--device", default="cuda",
                   help="cuda (current GPU), cuda:N, a GPU index N, or cpu. Falls back to cpu without CUDA.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", required=True, help="Run folder for checkpoints, logs and test results.")
    p.add_argument("--moco-dim", type=int, default=128)
    p.add_argument("--moco-k", type=int, default=8192)
    p.add_argument("--moco-t", type=float, default=0.2)
    p.add_argument("--mlp", action="store_true", default=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--bpaco-weight", type=float, default=0.25,
                   help="Weight of the BPaCo term; 0 disables it. > 0 needs third_party/bpaco_losses.py.")
    p.add_argument("--loss-type", choices=["ce", "ce_cw"], default="ce",
                   help="ce_cw: class weights 1/sqrt(n_c), normalized to mean 1.")
    p.add_argument("--use-mixup", action="store_true")
    p.add_argument("--mixup-alpha", type=float, default=0.4)
    p.add_argument("--cutmix-alpha", type=float, default=0.4)
    p.add_argument("--randaug-m", type=int, default=10)
    p.add_argument("--randaug-n", type=int, default=2)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    p.add_argument("--grad-checkpointing", action="store_true", help="Enable backbone gradient checkpointing when supported.")
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--layer-wise-lr-decay", action="store_true")
    p.add_argument("--layer-decay", type=float, default=0.8)
    p.add_argument("--head-lr-mult", type=float, default=2.0)
    p.add_argument("--stain-aug", choices=["none", "wb_gamma"], default="none")
    p.add_argument("--stain-aug-p", type=float, default=0.3)
    p.add_argument(
        "--aug-policy",
        choices=[
            "meta_default",
            "conservative",
            "color_only",
            "stain_only",
            "geometry_only",
            "blur_noise",
            "stain_blur_noise",
        ],
        default="meta_default",
        help="Morphology-preserving augmentation policy for --recipe meta.",
    )
    p.add_argument("--safe-crop-scale", type=float, default=0.9)
    p.add_argument("--rotation-degrees", type=float, default=10.0)
    p.add_argument("--color-jitter-strength", type=float, default=1.0)
    p.add_argument("--blur-p", type=float, default=0.15)
    p.add_argument("--noise-p", type=float, default=0.15)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--crt-epochs", type=int, default=0)
    p.add_argument("--crt-loss-type", choices=["ce", "ce_cw"], default="ce_cw")
    p.add_argument("--crt-lr", type=float, default=1e-4)
    p.add_argument("--crt-weight-decay", type=float, default=1e-5)
    p.add_argument("--crt-sampler-weighting", choices=["inv", "sqrt_inv"], default="inv")
    p.add_argument("--crt-freeze-backbone", action="store_true")
    p.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights; useful for smoke tests.")
    p.add_argument("--max-train-steps", type=int, default=0, help="Debug only: stop each epoch after N train steps.")
    p.add_argument("--max-eval-batches", type=int, default=0, help="Debug only: evaluate only N batches.")
    p.add_argument(
        "--resume",
        choices=["auto", "none"],
        default="auto",
        help="Resume from last_bpaco_full.pth in outdir when possible.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    # Imported before seeding, as the module-level import was; fails early if the file is missing.
    BPaCoLoss = load_bpaco_loss() if args.bpaco_weight > 0 else None
    seed_everything(args.seed)

    root = Path(args.root).expanduser().resolve()
    train_root = root / "train"
    val_root = root / "val"
    test_root = root / "test"
    class_names = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
    label2idx = {c: i for i, c in enumerate(class_names)}
    args.num_classes = len(class_names)
    counts = np.array([len(list_images(train_root / c)) for c in class_names], dtype=np.int64)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.resume == "auto" and (outdir / "test_results.json").exists():
        print(f"[INFO] Existing completed run found, skipping: {outdir / 'test_results.json'}")
        return

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"[INFO] device={device}")
    print(f"[INFO] classes={class_names}")
    print(f"[INFO] counts={counts.tolist()}")
    print(f"[INFO] objective = CE + {args.bpaco_weight} * BPaCo")

    train_tf, val_tf = build_transforms(args)
    train_ds = FolderWBCDataset(train_root, class_names, train_tf)
    val_ds = FolderWBCDataset(val_root, class_names, val_tf)
    test_ds = FolderWBCDataset(test_root, class_names, val_tf)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, worker_init_fn=seed_worker)

    model = BPaCoClassifier(
        args.model_name, args.num_classes, args.moco_dim,
        pretrained=not args.no_pretrained, mlp=args.mlp,
        grad_checkpointing=args.grad_checkpointing, img_size=args.img_size).to(device)
    queue = Queue(args.moco_k, args.moco_dim, args.num_classes, device)
    criterion_bpaco = None
    if args.bpaco_weight > 0:
        criterion_bpaco = BPaCoLoss(
            alpha=args.alpha, beta=args.beta, gamma=args.gamma, temperature=args.moco_t,
            K=args.moco_k, num_classes=args.num_classes, cls_num_list1=counts.tolist()).to(device)
        criterion_bpaco.cal_weight_for_classes(counts.tolist())

    class_weight = class_weights_sqrt_inv(counts).to(device)
    ce_criterion = (
        nn.CrossEntropyLoss(weight=class_weight)
        if args.loss_type == "ce_cw"
        else nn.CrossEntropyLoss()
    )
    eval_criterion = ce_criterion
    mixup_fn = None
    if args.use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            label_smoothing=0.0,
            num_classes=args.num_classes,
        )

    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    meta = {
        "args": vars(args),
        "classes": class_names,
        "counts": counts.tolist(),
        "note": "BPaCo objective: CE + bpaco_weight * BPaCo; exported checkpoints are plain timm classifier checkpoints.",
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best_f1 = -1.0
    best_ba = -1.0
    history = []
    start_epoch = 0
    log_path = outdir / "train_log.json"
    if args.resume == "auto" and log_path.exists():
        try:
            history = json.loads(log_path.read_text(encoding="utf-8"))
            if history:
                start_epoch = int(history[-1].get("epoch", len(history)))
                best_f1 = max(float(r.get("val", {}).get("f1", -1.0)) for r in history)
                best_ba = max(float(r.get("val", {}).get("ba", -1.0)) for r in history)
        except Exception as exc:
            print(f"[WARN] Could not parse existing train_log.json; starting log fresh: {exc}")
            history = []
            start_epoch = 0

    resume_ckpt = outdir / "last_bpaco_full.pth"
    if args.resume == "auto" and resume_ckpt.exists() and start_epoch > 0:
        print(f"[INFO] Resuming from {resume_ckpt} at epoch {start_epoch}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            for _ in range(start_epoch):
                scheduler.step()
        if scaler.is_enabled() and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if "queue_features" in ckpt:
            queue.features.copy_(ckpt["queue_features"].to(device))
        if "queue_labels" in ckpt:
            queue.labels.copy_(ckpt["queue_labels"].to(device))
        if "queue_ptr" in ckpt:
            queue.ptr = int(ckpt["queue_ptr"])
    elif args.resume == "auto" and start_epoch > 0:
        print("[WARN] train_log.json exists but last_bpaco_full.pth is missing; starting training from scratch.")
        history = []
        start_epoch = 0

    if start_epoch >= args.epochs:
        print(f"[INFO] Training already reached epoch {start_epoch}/{args.epochs}; running final test/cRT if needed.")

    start = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_metrics = train_one_epoch(
            model, queue, train_loader, criterion_bpaco, ce_criterion,
            optimizer, scaler if scaler.is_enabled() else None, device, args, mixup_fn, class_weight)
        val_metrics = evaluate(model, val_loader, eval_criterion, device, class_names, args.max_eval_batches)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": {k: v for k, v in val_metrics.items() if k not in {"report", "confusion_matrix"}},
        }
        history.append(row)
        (outdir / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"[Train] {train_metrics}")
        print(f"[Val] loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.4f} ba={val_metrics['ba']:.4f}")

        raw_ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "queue_features": queue.features.detach().cpu(),
            "queue_labels": queue.labels.detach().cpu(),
            "queue_ptr": queue.ptr,
            "classes": class_names,
            "args": vars(args),
        }
        torch.save(raw_ckpt, outdir / "last_bpaco_full.pth")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            shutil.copyfile(outdir / "last_bpaco_full.pth", outdir / "best_macro_f1_bpaco_full.pth")
            export_timm_checkpoint(
                model, args, class_names, outdir / "best_macro_f1.pth",
                epoch, "best_macro_f1", best_f1)
        if val_metrics["ba"] > best_ba:
            best_ba = val_metrics["ba"]
            export_timm_checkpoint(
                model, args, class_names, outdir / "best_ba.pth",
                epoch, "best_ba", best_ba)

        (outdir / "best_info.txt").write_text(
            f"Best macro F1: {best_f1:.6f}\nBest balanced acc: {best_ba:.6f}\n",
            encoding="utf-8",
        )

    print(f"\n[INFO] Training finished in {(time.time() - start)/60:.2f} min")
    best_ckpt = torch.load(outdir / "best_macro_f1_bpaco_full.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, eval_criterion, device, class_names, args.max_eval_batches)
    (outdir / "test_results.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    with open(outdir / "confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for i, row in enumerate(test_metrics["confusion_matrix"]):
            writer.writerow([class_names[i]] + row)
    print(f"[Test] loss={test_metrics['loss']:.4f} f1={test_metrics['f1']:.4f} ba={test_metrics['ba']:.4f}")
    train_crt(model, train_root, val_loader, test_loader, class_names, label2idx, val_tf, class_weight, device, args, outdir)


if __name__ == "__main__":
    main()
