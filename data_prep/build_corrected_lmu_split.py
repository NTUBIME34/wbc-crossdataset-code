#!/usr/bin/env python3
"""Build the CORRECTED AML-LMU nine-class split (step 4b of data_prep/README.md).

WHY THIS EXISTS
---------------
The first AML-LMU nine-class split carried only the 11 ATYPICAL (LYA) lymphocytes in its
Lymphocyte class (in the authors' nine-class pool that folder was named "Lymphocyte "
with a trailing space); the 3,936 TYPICAL (LYT) lymphocytes were never carried into it.
Both LYA and LYT map to the harmonized Lymphocyte class (Supplementary Table S3), so the
pool should contain all 3,947 (Supplementary Table S15).

Consequences of the original omission (Supplementary Note 1):
  * the leave-one-dataset-out (LODO) evaluation of the held-out AML-LMU domain scored the
    Lymphocyte class on only 11 cells;
  * every LODO model that kept AML-LMU *in* its training pool was trained with far fewer
    lymphocytes than intended -- most severely the ALL-minus-BMCMHF model.

The internal (non-LODO) corpus is NOT affected: it already contained the typical
lymphocytes.

This script mirrors the existing split (hard links, class-folder names stripped of
spaces) and adds the typical lymphocytes with their own deterministic 80/10/10 split
(random.Random(42).shuffle of the sorted file list; counts rounded). The original split
is left untouched. This reproduces the corrected split used for the paper's LODO folds
file for file when --src is the split of the AML-LMU pool written by harmonize_labels.py
(default --lmu-typical defer) and --typical is the folder of typical-lymphocyte crops.
Next: build_corrected_lodo_splits.py, then build_lmu_lodo_test.py.

Usage:
  python data_prep/build_corrected_lmu_split.py --data-root "$DATA_ROOT"
"""
import argparse
import os
import random
import shutil
from pathlib import Path

SRC_NAME = "PKG_AML_Cytomorphology_LMU_single_wbc_for_train_9class_split"
CROPS_NAME = "PKG_AML_Cytomorphology_LMU_single_wbc"
TYPICAL_NAMES = ("LYT", "Lymphocyte_typical")  # release code, authors' folder name

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
RATIO = (0.8, 0.1, 0.1)  # train / val / test, matching the original split


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Add the AML-LMU typical lymphocytes to the AML-LMU split (corrected split).")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"),
                    help="Default location of the folders below (default: $DATA_ROOT).")
    ap.add_argument("--src", default=None,
                    help=f"Existing AML-LMU split (train/val/test). Default: DATA_ROOT/{SRC_NAME}")
    ap.add_argument("--typical", default=None,
                    help="Folder of typical-lymphocyte (LYT) crops. Default: the LYT or "
                         f"Lymphocyte_typical folder in DATA_ROOT/{CROPS_NAME}")
    ap.add_argument("--dst", default=None, help="Output folder. Default: <src>_corrected")
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete an existing output folder first (default: refuse).")
    return ap.parse_args()


def link_or_copy(src: Path, tgt: Path) -> None:
    try:
        os.link(src, tgt)
    except OSError:  # e.g. across file systems
        shutil.copy2(src, tgt)


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    if args.src is None or args.typical is None:
        if root is None:
            raise SystemExit("Set DATA_ROOT or pass --data-root, or give --src and --typical.")
    src = Path(args.src).expanduser().resolve() if args.src else root / SRC_NAME
    if args.typical:
        typical_dir = Path(args.typical).expanduser().resolve()
    else:
        found = [root / CROPS_NAME / n for n in TYPICAL_NAMES if (root / CROPS_NAME / n).is_dir()]
        if not found:
            raise SystemExit(f"No typical-lymphocyte folder ({' or '.join(TYPICAL_NAMES)}) in "
                             f"{root / CROPS_NAME}; pass --typical.")
        typical_dir = found[0]
    dst = Path(args.dst).expanduser().resolve() if args.dst else src.with_name(src.name + "_corrected")
    if not src.is_dir() or not typical_dir.is_dir():
        raise SystemExit(f"Missing input: {src if not src.is_dir() else typical_dir}")
    if dst == src or src in dst.parents or dst in src.parents:
        raise SystemExit("--dst must be a separate folder (not the source split or its parent).")
    if dst == REPO_ROOT or REPO_ROOT in dst.parents:
        raise SystemExit(f"--dst must be outside the repository: {dst}")

    typ = sorted(p for p in typical_dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in EXTS)
    # Guard: the typical lymphocytes must not be in the source split already (that happens
    # if the pool was harmonized with --lmu-typical include); adding them again would
    # place copies of one image in two splits.
    typ_names = {p.name for p in typ}
    already = [p for p in src.rglob("*") if p.is_file() and p.name in typ_names]
    if already:
        raise SystemExit(f"{len(already)} typical-lymphocyte files are already in {src}; "
                         "this split needs no correction.")

    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; rerun with --overwrite to rebuild it.")
        shutil.rmtree(dst)

    # 1) mirror the existing split (hardlinks), normalising class-dir names
    #    so that "Lymphocyte " (trailing space) becomes "Lymphocyte".
    for split in ("train", "val", "test"):
        for cdir in sorted(p for p in (src / split).iterdir() if p.is_dir()):
            out = dst / split / cdir.name.strip()
            out.mkdir(parents=True, exist_ok=True)
            for f in sorted(cdir.rglob("*")):
                if f.is_file() and f.suffix.lower() in EXTS:
                    tgt = out / f.name
                    if not tgt.exists():
                        link_or_copy(f, tgt)

    # 2) add the typical lymphocytes, split deterministically at the same ratio
    random.Random(SEED).shuffle(typ)
    n = len(typ)
    n_tr = int(round(n * RATIO[0]))
    n_va = int(round(n * RATIO[1]))
    parts = {
        "train": typ[:n_tr],
        "val": typ[n_tr:n_tr + n_va],
        "test": typ[n_tr + n_va:],
    }
    for split, files in parts.items():
        out = dst / split / "Lymphocyte"
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            tgt = out / f.name
            if not tgt.exists():
                link_or_copy(f, tgt)

    # 3) report
    print(f"typical lymphocytes added: {n} -> "
          f"train {len(parts['train'])}, val {len(parts['val'])}, test {len(parts['test'])}")
    total = 0
    for split in ("train", "val", "test"):
        tot = sum(1 for p in (dst / split).rglob("*") if p.is_file())
        ly = sum(1 for p in (dst / split / "Lymphocyte").rglob("*") if p.is_file())
        total += tot
        print(f"  {split:5s}: total={tot:6d}  Lymphocyte={ly:5d}")
    ly_all = sum(1 for split in ("train", "val", "test")
                 for p in (dst / split / "Lymphocyte").rglob("*") if p.is_file())
    print(f"  TOTAL: {total}   Lymphocyte overall = {ly_all} "
          f"(expected 3947 = 11 atypical + 3936 typical; AML-LMU pool 18128)")
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()
