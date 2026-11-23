#!/usr/bin/env python3
"""Build the AML-LMU held-out test set of the leave-one-dataset-out (LODO) analysis
(step 4d of data_prep/README.md).

A LODO fold is scored on every pooled cell of its held-out source (Supplementary Note 1
and Table S15). For four sources that is simply the harmonized pool folder. For AML-LMU
the pool folder written by harmonize_labels.py (default --lmu-typical defer) lacks the
typical lymphocytes, which the paper's corrected split adds back, so this script builds
the complete AML-LMU pool as one class-folder test set:

    held-out set = AML-LMU nine-class pool + typical-lymphocyte (LYT) crops
                 = train + val + test of the corrected AML-LMU split
                 = 18,128 cells in eight classes (Supplementary Table S15)

Class-folder names are stripped of surrounding spaces ("Lymphocyte " -> "Lymphocyte").
Files are hard-linked (copied when hard links are not possible) under their original
names. The paper's AML-LMU fold was evaluated on exactly this set (folder name
LMU_LODO_corrected_test, the default of eval/evaluate_lodo.py). The step that built the
authors' folder was not kept; this script reproduces that folder file for file.

Usage:
  python data_prep/build_lmu_lodo_test.py --data-root "$DATA_ROOT"
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

POOL_NAME = "PKG_AML_Cytomorphology_LMU_single_wbc_for_train_9class"
CROPS_NAME = "PKG_AML_Cytomorphology_LMU_single_wbc"
TYPICAL_NAMES = ("LYT", "Lymphocyte_typical")  # release code, authors' folder name
DST_NAME = "LMU_LODO_corrected_test"
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parent.parent

# AML-LMU column of Supplementary Table S15.
EXPECTED = {"Neutrophil": 8483, "Lymphocyte": 3947, "Monocyte": 1779, "Eosinophil": 424,
            "Basophil": 79, "Erythroblast": 78, "Promyelocyte": 70, "Myeloblast": 3268}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build the AML-LMU LODO held-out set (nine-class pool + typical lymphocytes).")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"),
                    help="Default location of the folders below (default: $DATA_ROOT).")
    ap.add_argument("--pool", default=None,
                    help=f"AML-LMU pool from harmonize_labels.py. Default: DATA_ROOT/{POOL_NAME}")
    ap.add_argument("--typical", default=None,
                    help="Folder of typical-lymphocyte (LYT) crops. Default: the LYT or "
                         f"Lymphocyte_typical folder in DATA_ROOT/{CROPS_NAME}. Pass 'none' "
                         "if the pool already includes them (--lmu-typical include).")
    ap.add_argument("--dst", default=None, help=f"Output folder. Default: DATA_ROOT/{DST_NAME}")
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete an existing output folder first (default: refuse).")
    return ap.parse_args()


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    if root is None and (args.pool is None or args.typical is None or args.dst is None):
        raise SystemExit("Set DATA_ROOT or pass --data-root, or give --pool, --typical and --dst.")
    pool = Path(args.pool).expanduser().resolve() if args.pool else root / POOL_NAME
    dst = Path(args.dst).expanduser().resolve() if args.dst else root / DST_NAME
    if args.typical and args.typical.lower() == "none":
        typical_dir = None
    elif args.typical:
        typical_dir = Path(args.typical).expanduser().resolve()
    else:
        found = [root / CROPS_NAME / n for n in TYPICAL_NAMES if (root / CROPS_NAME / n).is_dir()]
        if not found:
            raise SystemExit(f"No typical-lymphocyte folder ({' or '.join(TYPICAL_NAMES)}) in "
                             f"{root / CROPS_NAME}; pass --typical (or --typical none).")
        typical_dir = found[0]
    if not pool.is_dir():
        raise SystemExit(f"Pool folder not found: {pool}")
    if typical_dir is not None and not typical_dir.is_dir():
        raise SystemExit(f"Typical-lymphocyte folder not found: {typical_dir}")
    if dst in (pool, typical_dir) or pool in dst.parents or dst in pool.parents:
        raise SystemExit("--dst must be a separate folder.")
    if dst == REPO_ROOT or REPO_ROOT in dst.parents:
        raise SystemExit(f"--dst must be outside the repository: {dst}")

    plan: dict[Path, Path] = {}
    for cdir in sorted(p for p in pool.iterdir() if p.is_dir()):
        for f in images(cdir):
            plan[dst / cdir.name.strip() / f.name] = f
    if typical_dir is not None:
        for f in images(typical_dir):
            tgt = dst / "Lymphocyte" / f.name
            if tgt in plan:
                raise SystemExit(f"{f.name} is already in the pool; use --typical none.")
            plan[tgt] = f

    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; rerun with --overwrite to rebuild it.")
        shutil.rmtree(dst)
    for tgt, src in sorted(plan.items()):
        tgt.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, tgt)
        except OSError:
            shutil.copy2(src, tgt)

    counts = Counter(t.parent.name for t in plan)
    ok = True
    print(f"AML-LMU held-out set: {dst}")
    for label in sorted(set(counts) | set(EXPECTED)):
        n, exp = counts.get(label, 0), EXPECTED.get(label, 0)
        ok &= n == exp
        print(f"  {label:13s} {n:6d}   expected {exp:6d}  {'ok' if n == exp else 'DIFFERS'}")
    total, exp_total = sum(counts.values()), sum(EXPECTED.values())
    ok &= total == exp_total
    print(f"  {'total':13s} {total:6d}   expected {exp_total:6d}  {'ok' if total == exp_total else 'DIFFERS'}")
    if not ok:
        print("WARNING: counts differ from Supplementary Table S15.")


if __name__ == "__main__":
    main()
