#!/usr/bin/env python3
"""Rebuild the LODO ALL-minus-X folds using the CORRECTED AML-LMU split
(which contains all 3,947 AML-LMU lymphocytes instead of 11). Step 4c of
data_prep/README.md; run build_corrected_lmu_split.py first.

Mirrors create_lodo_all_datasets.py exactly:
  - held-out dataset: its train+val+test -> out/test
  - every other dataset: train -> out/train, val -> out/val   (test is NOT used for training)
  - symlinks named <source code>__<split>__<file name>; class names normalised with .strip()

Only the four folds whose TRAINING pool contains AML-LMU are rebuilt (held out: BMCMHF,
HRLS, PBC, Raabin-WBC). ALL-minus-LMU needs no rebuild: AML-LMU is held out there, so it
never entered that model's training set, and its fold from create_lodo_all_datasets.py
is the one used in the paper.

Output: LODO_corrected_ALL_minus_<X>_test_<X>_split   (original folds left untouched;
an existing corrected fold folder is replaced). These are the training roots of
train/scripts/ (LODO_ROOT_<FOLD>); held-out scoring reads the complete pools instead
(eval/evaluate_lodo.py).

Usage:
  python data_prep/build_corrected_lodo_splits.py --base-dir "$DATA_ROOT"
"""
import argparse
import os
import shutil
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "BM":     "BM_cytomorphology_data_single_wbc_07_for_train_9class_split",
    "DIB":    "dataset_DIB_single_wbc_copy_split",
    "HRLS":   "HRLS_dataset_single_wbc_copy_split",
    "LMU":    "PKG_AML_Cytomorphology_LMU_single_wbc_for_train_9class_split_corrected",  # <-- FIXED
    "Raabin": "Raabin_WBC_single_wbc_copy_split",
}
HELDOUT = ["BM", "HRLS", "DIB", "Raabin"]   # order: most-affected / fastest first


def iter_images(d: Path):
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p


def union_classes(base: Path, datasets: dict):
    cs = set()
    for rel in datasets.values():
        for sp in ("train", "val", "test"):
            r = base / rel / sp
            if r.is_dir():
                cs.update(c.name.strip() for c in r.iterdir() if c.is_dir())
    return sorted(cs)


def link(src_root: Path, dst_root: Path, ds: str, splits):
    counts = {}
    for sp in splits:
        sr = src_root / sp
        if not sr.is_dir():
            continue
        for cdir in sorted(p for p in sr.iterdir() if p.is_dir()):
            cname = cdir.name.strip()
            out = dst_root / cname
            out.mkdir(parents=True, exist_ok=True)
            for src in iter_images(cdir):
                tgt = out / f"{ds}__{sp}__{src.name}"
                if not (tgt.exists() or tgt.is_symlink()):
                    os.symlink(src, tgt)
                    counts[cname] = counts.get(cname, 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rebuild the four LODO folds that train on AML-LMU, using the corrected AML-LMU split.")
    ap.add_argument("--base-dir", default=os.environ.get("DATA_ROOT"),
                    help="Directory containing the per-source split folders (default: $DATA_ROOT).")
    ap.add_argument("--out-dir", default=None,
                    help="Where the corrected fold folders go (default: --base-dir).")
    ap.add_argument("--dataset", action="append", default=[], metavar="CODE=FOLDER",
                    help="Override a source's split folder name, e.g. LMU=my_lmu_split_corrected.")
    args = ap.parse_args()
    if not args.base_dir:
        ap.error("set DATA_ROOT or pass --base-dir")

    base = Path(args.base_dir).expanduser().resolve()
    out_base = Path(args.out_dir).expanduser().resolve() if args.out_dir else base
    if out_base == REPO_ROOT or REPO_ROOT in out_base.parents:
        ap.error(f"--out-dir must be outside the repository: {out_base}")
    datasets = dict(DATASETS)
    for item in args.dataset:
        code, sep, folder = item.partition("=")
        if not sep or code not in DATASETS:
            ap.error(f"--dataset expects CODE=FOLDER with CODE in {sorted(DATASETS)}: {item}")
        datasets[code] = folder
    for ds, rel in datasets.items():
        if not (base / rel).is_dir():
            raise SystemExit(f"Missing split folder for {ds}: {base / rel}")

    classes = union_classes(base, datasets)
    print("union classes:", classes, "\n")

    for held in HELDOUT:
        out_root = out_base / f"LODO_corrected_ALL_minus_{held}_test_{held}_split"
        if out_root.exists():
            shutil.rmtree(out_root)
        for sp in ("train", "val", "test"):
            for c in classes:
                (out_root / sp / c).mkdir(parents=True, exist_ok=True)

        for ds, rel in datasets.items():
            src = base / rel
            if ds == held:
                link(src, out_root / "test", ds, ["train", "val", "test"])
            else:
                link(src, out_root / "train", ds, ["train"])
                link(src, out_root / "val",   ds, ["val"])

        def cnt(sp, cls=None):
            root = out_root / sp
            if cls:
                root = root / cls
            return sum(1 for p in root.rglob("*") if p.is_symlink() or p.is_file())

        print(f"{out_root.name}")
        print(f"   train={cnt('train'):6d}  val={cnt('val'):6d}  test={cnt('test'):6d}  "
              f"| train Lymphocyte={cnt('train','Lymphocyte'):6d}")


if __name__ == "__main__":
    main()
