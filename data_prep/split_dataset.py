"""Per-class random train/val/test split of a class-folder dataset (default 80/10/10, seed 42).

Step 3 of data_prep/README.md. This is the split used for every partition in the paper:
the merged internal corpus (main Methods "Datasets": a per-class random 80/10/10 split
at the cell level, seed 42) and the per-source pools that the leave-one-dataset-out
folds are built from (Supplementary Note 1).

How the split is drawn (do not change it if you want the paper's partitions):
  * the random generator is seeded once (random.seed(seed)); the class folders are then
    processed in sorted order and each class's sorted file list is shuffled with that one
    generator, so the split of a class depends on the classes that sort before it;
  * per class, n_train = int(n * 0.8) and n_val = int(n * 0.1) (truncation), and the
    test split takes the remainder.
Re-running this script on the authors' folders reproduces their per-source splits and
the merged corpus split file for file.

The merged corpus was split while its Metamyelocyte and Myelocyte folders were still
present (11 class folders); those two classes were removed from the split afterwards,
leaving the nine internal classes (102,345 training, 12,790 validation and 12,803 test
images of the 127,938). --drop-after-split does this in one step: the listed classes
take part in the draw, so the other classes get the same split, but are not written.

Usage:
  # one per-source pool
  python data_prep/split_dataset.py \\
    --src "$DATA_ROOT/dataset_DIB_single_wbc_copy" \\
    --dst "$DATA_ROOT/dataset_DIB_single_wbc_copy_split"
  # the merged internal corpus
  python data_prep/split_dataset.py \\
    --src "$DATA_ROOT/Mixed_data_mix_mono" \\
    --dst "$DATA_ROOT/Mixed_data_mix_mono_9_split_copy" \\
    --drop-after-split Metamyelocyte Myelocyte
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
REPO_ROOT = Path(__file__).resolve().parent.parent

def list_images(d: Path) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

def split_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    r1, r2, r3 = ratios
    t = int(n * r1)
    v = int(n * r2)
    te = n - t - v
    return t, v, te

def transfer(files: List[Path], dst: Path, mode: str = "copy"):
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        out = dst / f.name
        if mode == "copy":
            shutil.copy2(f, out)
        elif mode == "move":
            shutil.move(str(f), out)
        elif mode == "link":
            # Hard link to save space; if it fails across filesystems, fall back to copy
            try:
                if out.exists():
                    out.unlink()
                os.link(f, out)
            except OSError:
                shutil.copy2(f, out)
        else:
            raise ValueError(f"unknown mode: {mode}")

def safe_ratios(r: List[float]) -> Tuple[float, float, float]:
    if len(r) != 3:
        raise ValueError("ratios must be three numbers, e.g. 0.8 0.1 0.1 or 8 1 1")
    r = [float(x) for x in r]
    s = sum(r)
    if s <= 0:
        raise ValueError("sum of ratios must be > 0")
    # Normalize
    return (r[0]/s, r[1]/s, r[2]/s)

def main():
    ap = argparse.ArgumentParser(description="Split a folder-of-folders dataset into train/val/test by 8:1:1 (default).")
    ap.add_argument("--src", required=True, help="Input folder (each subfolder is one class)")
    ap.add_argument("--dst", required=True, help="Output folder (will create train/val/test/<each class>)")
    ap.add_argument("--ratios", nargs=3, default=[8, 1, 1], help="train val test ratios, e.g. 8 1 1 or 0.8 0.1 0.1")
    ap.add_argument("--mode", choices=["copy", "move", "link"], default="copy", help="Transfer mode: copy/move/link (hard link)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--drop-after-split", nargs="+", default=[], metavar="CLASS",
                    help="Class folders that take part in the random draw but are not written "
                         "(the merged corpus: Metamyelocyte Myelocyte)")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    ratios = safe_ratios(args.ratios)
    random.seed(args.seed)

    if not src.exists():
        raise FileNotFoundError(src)

    # Refuse a destination that is the source itself or lies inside it (the split would
    # then be written into the class folders it is reading). Sibling folders whose names
    # merely start with the source name are fine.
    if dst == src or src in dst.parents:
        raise ValueError("dst must not be identical to src or located inside src, to avoid overwriting.")
    if dst == REPO_ROOT or REPO_ROOT in dst.parents:
        raise SystemExit(f"--dst must be outside the repository: {dst}")

    classes = [d for d in sorted(src.iterdir()) if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No subfolders (classes) found in {src}")
    drop = set(args.drop_after_split)
    missing = drop - {d.name for d in classes}
    if missing:
        raise SystemExit(f"--drop-after-split: no such class folder in {src}: {sorted(missing)}")

    # Create the output root directories
    for sp in ("train", "val", "test"):
        (dst / sp).mkdir(parents=True, exist_ok=True)

    summary = {}
    for cls_dir in classes:
        cls_name = cls_dir.name
        imgs = list_images(cls_dir)
        if not imgs:
            print(f"[WARN] {cls_name}: no images, skip")
            continue

        random.shuffle(imgs)
        n_tr, n_val, n_te = split_counts(len(imgs), ratios)
        tr = imgs[:n_tr]
        val = imgs[n_tr:n_tr+n_val]
        te = imgs[n_tr+n_val:]
        if cls_name in drop:
            print(f"[DROP] {cls_name:30s} -> drawn (train {len(tr)} | val {len(val)} | test {len(te)}), not written")
            continue

        transfer(tr, dst / "train" / cls_name, mode=args.mode)
        transfer(val, dst / "val" / cls_name, mode=args.mode)
        transfer(te, dst / "test" / cls_name, mode=args.mode)

        summary[cls_name] = (len(tr), len(val), len(te))
        print(f"[OK] {cls_name:30s} -> train {len(tr):5d} | val {len(val):5d} | test {len(te):5d}")

    # Summary
    t_tot = sum(v[0] for v in summary.values())
    v_tot = sum(v[1] for v in summary.values())
    te_tot = sum(v[2] for v in summary.values())
    print(f"\n[DONE] total -> train {t_tot} | val {v_tot} | test {te_tot} | classes {len(summary)}")
    print(f"Output: {dst}")

if __name__ == "__main__":
    main()
