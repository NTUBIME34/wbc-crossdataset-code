#!/usr/bin/env python3
"""Are the detector's failures informative? Missed-versus-detected comparison
(Supplementary Section S12, table "detector_abstention_quality_test").

Question: on the partly degraded Phase-2 split the YOLO detector returns no box for 348
labelled images. Are these cells harder for the classifier than cells it did find?

Test: feed BOTH groups to the classifier in EXACTLY the same way (full field -> eval
transform -> Swin, no cropping), so the only difference between the groups is whether
the detector found a cell:

  Group MISSED   : the no-detection images with a harmonizable label (348 in the paper)
  Group DETECTED : the same number of images matched on TRUE CLASS from the detected set
                   (control), drawn once with random.Random(0)

Class-matching the control is essential: the missed set is 63% neutrophils (219/348),
and neutrophils are the easiest class, so an unmatched control would be confounded.

Settings, kept as run (Supplementary Section S17): Resize(256) -> CenterCrop(224) without
square padding and without flip averaging; three-seed probability sum; nine-class argmax
mapped to the seven evaluation classes. The group difference is descriptive: the control
is a single class-matched draw, so no two-sample test applies.

Inputs: detections.csv written by pipeline/detect_yolo11.py on the Phase-2 evaluation
fields (columns source_path and status; the true class is the name of each source image's
parent folder) and the three proposed checkpoints (seeds 42/62/82). The paper reports
accuracy 0.727 (missed) versus 0.862 (class-matched detected).
"""
import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO = Path(__file__).resolve().parent.parent
CKPT_NAMES = [f"swin_base_colour_aug_cwce_seed{s}.pth" for s in (42, 62, 82)]

TRAIN_CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'Lymphoblast', 'Lymphocyte',
                 'Monocyte', 'Myeloblast', 'Neutrophil', 'Promyelocyte']
# 9-class prediction -> harmonized 7-class space
TO7 = {'Basophil': 'Basophil', 'Eosinophil': 'Eosinophil', 'Lymphocyte': 'Lymphocyte',
       'Monocyte': 'Monocyte', 'Neutrophil': 'Neutrophil', 'Promyelocyte': 'Promyelocyte',
       'Erythroblast': 'Blast_Group', 'Lymphoblast': 'Blast_Group', 'Myeloblast': 'Blast_Group'}
# benchmark folder name -> harmonized 7-class label
GT7 = {'Segmented_neutrophil': 'Neutrophil', 'Band_neutrophil': 'Neutrophil',
       'Lymphocyte': 'Lymphocyte', 'Variant_lymphocyte': 'Lymphocyte', 'Prolymphocyte': 'Lymphocyte',
       'Monocyte': 'Monocyte', 'Eosinophil': 'Eosinophil', 'Basophil': 'Basophil',
       'Promyelocyte': 'Promyelocyte', 'Blast': 'Blast_Group',
       'Myelocyte': None, 'Metamyelocyte': None, 'Plasma_cell': None}

TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def default_checkpoints() -> list[str]:
    """$MODEL_DIR, pipeline/models/ or models/ (first that holds the file), as in pipeline/classify.py."""
    dirs = ([Path(os.environ["MODEL_DIR"]).expanduser()] if os.environ.get("MODEL_DIR") else []) + \
           [REPO / "pipeline" / "models", REPO / "models"]
    out = []
    for name in CKPT_NAMES:
        hits = [d / name for d in dirs if (d / name).is_file()]
        out.append(str(hits[0] if hits else dirs[0] / name))
    return out


def load_checkpoint(path):
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


class DS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return TF(Image.open(p).convert("RGB")), y


def build_groups(detections_csv):
    with open(detections_csv, newline="") as f:
        det = list(csv.DictReader(f))
    missed, detected = [], []
    for r in det:
        p = r["source_path"]
        lab = GT7.get(os.path.basename(os.path.dirname(p)))
        if lab is None:
            continue                                  # excluded by harmonization
        (missed if r["status"] == "no_detection" else detected).append((p, lab))

    # class-matched control: same true-class histogram as the missed group
    need = Counter(l for _, l in missed)
    pool = defaultdict(list)
    for p, l in detected:
        pool[l].append((p, l))
    rng = random.Random(0)
    control = []
    for l, k in need.items():
        rng.shuffle(pool[l])
        control += pool[l][:k]
    return missed, control


@torch.no_grad()
def evaluate(items, name, ckpts, device, batch_size, workers):
    dl = DataLoader(DS(items), batch_size=batch_size, num_workers=workers)
    probs = None
    for ck in ckpts:
        m = timm.create_model("swin_base_patch4_window7_224", pretrained=False,
                              num_classes=len(TRAIN_CLASSES))
        sd = load_checkpoint(ck)
        sd = sd.get("model", sd.get("state_dict", sd))
        m.load_state_dict(sd); m.to(device).eval()
        pr = []
        for x, _ in dl:
            pr.append(torch.softmax(m(x.to(device)), 1).cpu())
        pr = torch.cat(pr)
        probs = pr if probs is None else probs + pr     # 3-seed ensemble
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pred9 = [TRAIN_CLASSES[i] for i in probs.argmax(1).tolist()]
    pred7 = [TO7[p] for p in pred9]
    true7 = [y for _, y in items]
    acc = np.mean([p == t for p, t in zip(pred7, true7)])
    print(f"\n=== {name}  (n={len(items)}) ===")
    print(f"  accuracy = {acc:.4f}")
    per = defaultdict(lambda: [0, 0])
    for p, t in zip(pred7, true7):
        per[t][1] += 1; per[t][0] += (p == t)
    for c in sorted(per):
        c_ok, n = per[c]
        print(f"    {c:14s} {c_ok:4d}/{n:4d} = {c_ok/n:.3f}")
    return acc, {c: ok / n for c, (ok, n) in per.items()}


def main():
    ap = argparse.ArgumentParser(description="Missed-versus-detected comparison of Supplementary Section S12.")
    ap.add_argument("--detections", required=True,
                    help="detections.csv from pipeline/detect_yolo11.py on the Phase-2 evaluation fields.")
    ap.add_argument("--checkpoints", nargs=3, default=None, metavar="CKPT",
                    help="The three proposed checkpoints (seeds 42 62 82). Default: "
                         + ", ".join(CKPT_NAMES) + " in $MODEL_DIR, pipeline/models/ or models/.")
    ap.add_argument("--output-csv", default=None,
                    help="Optional summary CSV (group, n, accuracy, per-class accuracy), the format of "
                         "analysis/tables/detector_abstention_quality_test.csv.")
    ap.add_argument("--device", default="cuda", help="cuda or cpu (falls back to cpu without a GPU).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.output_csv:
        out_csv = Path(args.output_csv).expanduser().resolve()
        if out_csv == REPO or REPO in out_csv.parents:
            raise SystemExit(f"[ERR] --output-csv {out_csv} is inside the repository; "
                             "write outputs to a folder outside it.")

    ckpts = args.checkpoints or default_checkpoints()
    for ck in ckpts:
        if not Path(ck).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ck}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    missed, control = build_groups(args.detections)
    print(f"MISSED (no detection, labelled): {len(missed)}")
    print(f"CONTROL (detected, class-matched): {len(control)}")
    print(f"class histogram identical: {Counter(l for _,l in missed) == Counter(l for _,l in control)}")
    a, per_a = evaluate(missed, "A. YOLO MISSED these cells", ckpts, device, args.batch_size, args.workers)
    b, per_b = evaluate(control, "B. YOLO DETECTED these cells (class-matched control)", ckpts, device,
                        args.batch_size, args.workers)
    print("\n" + "=" * 60)
    print(f"  MISSED   accuracy = {a:.4f}")
    print(f"  CONTROL  accuracy = {b:.4f}")
    print(f"  difference        = {a-b:+.4f}")

    if args.output_csv:
        classes = sorted(set(per_a) | set(per_b))
        out = Path(args.output_csv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")  # LF, as in analysis/tables/
            w.writerow(["group", "n", "accuracy", *classes])
            for group, items, acc, per in (("missed_by_yolo", missed, a, per_a),
                                           ("detected_class_matched_control", control, b, per_b)):
                w.writerow([group, len(items), f"{acc:.4f}", *(f"{per[c]:.3f}" if c in per else "" for c in classes)])
        print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    main()
