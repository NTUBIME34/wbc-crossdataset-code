#!/usr/bin/env python3
"""Create the leave-one-dataset-out (LODO) ALL-minus-X fold folders with symlinks
(step 4a of data_prep/README.md).

For each held-out source X (Supplementary Note 1):
  * every split (train + val + test) of X becomes the fold's test/ folder;
  * every other source contributes its train split to train/ and its val split to val/
    (their test splits are not used);
  * links are named <source code>__<split>__<file name>; class names are stripped of
    surrounding spaces.

The inputs are the per-source 80/10/10 splits written by split_dataset.py. This script
builds the original folds LODO_ALL_minus_<X>_test_<X>_split for all five sources. The
paper's AML-LMU-held-out fold is the one built here (its training pool holds no AML-LMU
cells); the other four folds were rebuilt with the corrected AML-LMU split by
build_corrected_lodo_splits.py (run it after this script). Held-out scoring uses each
source's complete pool, not these test/ folders (eval/evaluate_lodo.py; for AML-LMU
the set written by build_lmu_lodo_test.py). Symlink targets are absolute paths, so do
not move the data folder afterwards.

Source codes: BM = BMCMHF, DIB = PBC, HRLS = HRLS, LMU = AML-LMU, Raabin = Raabin-WBC.

Usage:
  python data_prep/create_lodo_all_datasets.py --base-dir "$DATA_ROOT"
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
REPO_ROOT = Path(__file__).resolve().parent.parent

# Source code -> per-source split folder (split_dataset.py output) under --base-dir.
DATASETS = {
    "BM": "BM_cytomorphology_data_single_wbc_07_for_train_9class_split",
    "DIB": "dataset_DIB_single_wbc_copy_split",
    "HRLS": "HRLS_dataset_single_wbc_copy_split",
    "LMU": "PKG_AML_Cytomorphology_LMU_single_wbc_for_train_9class_split",
    "Raabin": "Raabin_WBC_single_wbc_copy_split",
}


def norm_class_name(name: str) -> str:
    return name.strip()


def iter_images(class_dir: Path):
    for p in sorted(class_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def safe_link_name(dataset: str, split: str, src: Path) -> str:
    return f"{dataset}__{split}__{src.name}"


def link_images(src_root: Path, dst_root: Path, dataset: str, src_splits: list[str]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for split in src_splits:
        split_root = src_root / split
        if not split_root.is_dir():
            continue
        for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            cname = norm_class_name(class_dir.name)
            out_dir = dst_root / cname
            out_dir.mkdir(parents=True, exist_ok=True)
            for src in iter_images(class_dir):
                out = out_dir / safe_link_name(dataset, split, src)
                if out.exists() or out.is_symlink():
                    raise FileExistsError(out)
                os.symlink(src, out)
                counts[cname] = counts.get(cname, 0) + 1
    return sorted(counts.items())


def discover_union_classes(base_dir: Path, datasets: dict[str, str]) -> list[str]:
    classes = set()
    for rel in datasets.values():
        root = base_dir / rel
        for split in ("train", "val", "test"):
            split_root = root / split
            if not split_root.is_dir():
                continue
            for class_dir in split_root.iterdir():
                if class_dir.is_dir():
                    classes.add(norm_class_name(class_dir.name))
    return sorted(classes)


def parse_dataset_overrides(items: list[str]) -> dict[str, str]:
    datasets = dict(DATASETS)
    for item in items:
        code, sep, folder = item.partition("=")
        if not sep or code not in DATASETS:
            raise SystemExit(f"--dataset expects CODE=FOLDER with CODE in {sorted(DATASETS)}: {item}")
        datasets[code] = folder
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leave-one-dataset-out ALL-minus-X folder datasets with symlinks."
    )
    parser.add_argument(
        "--base-dir",
        default=os.environ.get("DATA_ROOT"),
        help="Directory containing the five per-source split folders (default: $DATA_ROOT).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where the fold folders and LODO_ALL_minus_manifest.tsv go (default: --base-dir).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="CODE=FOLDER",
        help="Override a source's split folder name, e.g. DIB=my_pbc_split (repeatable).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing LODO output folders before rebuilding.",
    )
    args = parser.parse_args()
    if not args.base_dir:
        parser.error("set DATA_ROOT or pass --base-dir")

    base_dir = Path(args.base_dir).expanduser().resolve()
    out_base = Path(args.out_dir).expanduser().resolve() if args.out_dir else base_dir
    if out_base == REPO_ROOT or REPO_ROOT in out_base.parents:
        raise SystemExit(f"--out-dir must be outside the repository: {out_base}")
    datasets = parse_dataset_overrides(args.dataset)
    union_classes = discover_union_classes(base_dir, datasets)
    manifest_path = out_base / "LODO_ALL_minus_manifest.tsv"

    rows = []
    for heldout in ("LMU", "DIB", "Raabin", "HRLS", "BM"):
        out_root = out_base / f"LODO_ALL_minus_{heldout}_test_{heldout}_split"
        if out_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"{out_root} exists; rerun with --overwrite to rebuild.")
            for p in sorted(out_root.rglob("*"), reverse=True):
                if p.is_symlink() or p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            out_root.rmdir()

        for split in ("train", "val", "test"):
            for cname in union_classes:
                (out_root / split / cname).mkdir(parents=True, exist_ok=True)

        for dataset, rel in datasets.items():
            src_root = base_dir / rel
            if not src_root.is_dir():
                raise FileNotFoundError(src_root)

            if dataset == heldout:
                linked = link_images(src_root, out_root / "test", dataset, ["train", "val", "test"])
                for cname, count in linked:
                    rows.append([out_root.name, f"ALL - {heldout}", heldout, "test", dataset, cname, count])
            else:
                linked_train = link_images(src_root, out_root / "train", dataset, ["train"])
                for cname, count in linked_train:
                    rows.append([out_root.name, f"ALL - {heldout}", heldout, "train", dataset, cname, count])
                linked_val = link_images(src_root, out_root / "val", dataset, ["val"])
                for cname, count in linked_val:
                    rows.append([out_root.name, f"ALL - {heldout}", heldout, "val", dataset, cname, count])

    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["root", "train_definition", "test_dataset", "split", "source_dataset", "class", "count"])
        writer.writerows(rows)

    print(f"[DONE] Created {len(datasets)} LODO datasets under {out_base}")
    print(f"[DONE] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
