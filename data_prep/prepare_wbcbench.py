#!/usr/bin/env python3
"""
prepare_wbcbench.py

Step 5 of data_prep/README.md. Turns one labelled split of the WBCBench2026 competition
release (CSV labels + a flat image folder) into class sub-folders named after the
benchmark's native classes, the layout that the evaluators and the pipeline scoring
expect.

Nothing from WBCBench2026 is included in this repository. Download the competition files
yourself (Kaggle competition wbc-bench-2026; the paper used the files as downloaded in
November 2025) and accept the competition rules, which restrict redistribution. A later
Hugging Face release corrects 257 labels, removes 10 images and adds a patient hash, so
its counts differ from the ones below.

Input (the competition download):
  <src>/
    phase1_label.csv      phase1/        (Phase-1 release, 8,288 labelled fields)
    phase2_eval.csv       phase2/eval/   (Phase-2 evaluation split, 5,350 fields)

Output (one folder per native class):
  <dst>/Segmented_neutrophil/, Band_neutrophil/, Lymphocyte/, Variant_lymphocyte/, ...

--split phase1       -> the primary external evaluation set (main Methods "Study design");
                        default <dst> $DATA_ROOT/wbc-bench-2026_phase1_nocrop_formatted,
                        called WBCB_P1_FIELDS by eval/ and analysis/evidence/MANIFEST.csv
--split phase2_eval  -> the robustness test of the deployable pipeline (main Table 2a);
                        default <dst> $DATA_ROOT/wbc-bench-2026_phase2_eval_nocrop_formatted
                        (WBCB_P2_FIELDS)

Downstream (see data_prep/README.md): the full-pipeline route runs the detector on these
fields. The pre-cropped set (route A; WBCB_P1_CROPS) is the highest-confidence detector
box per Phase-1 field at 0.70, no margin (8,269 crops):
  python data_prep/detector_crop.py --input-dir "$DATA_ROOT/wbc-bench-2026_phase1_nocrop_formatted" \\
      --output-dir "$DATA_ROOT/wbc-bench-2026_phase1_cropped_formatted" --conf 0.70
Plasma_cell, Metamyelocyte and Myelocyte have no counterpart in the output taxonomy and
are ignored by the evaluators (208 Phase-1 cells, 154 Phase-2 fields; Supplementary S1).

Usage:
  python data_prep/prepare_wbcbench.py --src /path/to/wbc-bench-2026 --split phase1
  python data_prep/prepare_wbcbench.py --src /path/to/wbc-bench-2026 --split phase2_eval
"""

import argparse
import csv
import shutil
import os
from pathlib import Path
from collections import defaultdict

# Map the benchmark's class codes to the folder names the evaluators expect.
LABEL_MAP = {
    "SNE": "Segmented_neutrophil",
    "LY":  "Lymphocyte",
    "MO":  "Monocyte",
    "EO":  "Eosinophil",
    "BA":  "Basophil",
    "VLY": "Variant_lymphocyte", # (atypical) lymphocyte
    "BNE": "Band_neutrophil",    # Band-form neutrophil
    "MMY": "Metamyelocyte",
    "MY":  "Myelocyte",
    "PMY": "Promyelocyte",
    "BL":  "Blast",              # Blast cell
    "PC":  "Plasma_cell",
    "PLY": "Prolymphocyte"
}

# (label CSV, image folder) relative to --src, and the released split size.
SPLITS = {
    "phase1": ("phase1_label.csv", "phase1", 8288),
    "phase2_eval": ("phase2_eval.csv", "phase2/eval", 5350),
}
DEFAULT_DST = {
    "phase1": "wbc-bench-2026_phase1_nocrop_formatted",
    "phase2_eval": "wbc-bench-2026_phase2_eval_nocrop_formatted",
}

def load_csv(path):
    """Returns list of (filename, label) tuples."""
    data = []
    if not path.exists():
        print(f"[WARN] CSV not found: {path}")
        return data

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Handle cases where header might have extra spaces
        fieldnames = [n.strip() for n in reader.fieldnames] if reader.fieldnames else []

        # Try to identify ID and Label columns
        id_col = next((n for n in fieldnames if n.lower() in ["id", "filename", "image"]), None)
        lbl_col = next((n for n in fieldnames if n.lower() in ["label", "labels", "class"]), None)

        if not id_col or not lbl_col:
            print(f"[ERROR] Could not identify columns in {path}. Found: {fieldnames}")
            return data

        for row in reader:
            # Clean up keys if needed (strip spaces)
            clean_row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            if id_col in clean_row and lbl_col in clean_row:
                img_id = clean_row[id_col]
                label = clean_row[lbl_col]
                if img_id and label:
                    data.append((img_id, label))
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Arrange one labelled WBCBench2026 split into native-class folders.")
    parser.add_argument("--src", required=True, help="Root of the WBCBench2026 competition download")
    parser.add_argument("--split", required=True, choices=sorted(SPLITS),
                        help="phase1 (8,288 fields) or phase2_eval (5,350 fields)")
    parser.add_argument("--dst", default=None,
                        help="Output folder (default: $DATA_ROOT/wbc-bench-2026_<split>_nocrop_formatted)")
    parser.add_argument("--mode", default="copy", choices=["copy", "move", "symlink"], help="File transfer mode")
    args = parser.parse_args()

    if args.dst is None:
        if not os.environ.get("DATA_ROOT"):
            parser.error("pass --dst or set DATA_ROOT")
        args.dst = str(Path(os.environ["DATA_ROOT"]) / DEFAULT_DST[args.split])

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    repo_root = Path(__file__).resolve().parent.parent
    if dst_root == repo_root or repo_root in dst_root.parents:
        parser.error("--dst must be outside the repository (WBCBench2026 data must never enter it)")

    csv_name, img_rel, expected_n = SPLITS[args.split]
    sources = [(src_root / csv_name, src_root / img_rel)]

    print(f"[INFO] Scanning sources in {src_root}...")

    tasks = [] # List of (src_file, dst_folder)
    stats = defaultdict(int)
    missing = 0
    unknown = defaultdict(int)

    for csv_path, img_dir in sources:
        if not csv_path.exists():
            parser.error(f"label CSV not found: {csv_path}")
        if not img_dir.is_dir():
            parser.error(f"image folder not found: {img_dir}")

        print(f"Processing {csv_path.name} -> Images in {img_dir.relative_to(src_root)}")
        entries = load_csv(csv_path)

        for img_name, label in entries:
            # Apply mapping if exists
            final_label = LABEL_MAP.get(label, label)
            if label not in LABEL_MAP:
                unknown[label] += 1

            src_file = img_dir / img_name
            if not src_file.exists():
                missing += 1
                continue

            dst_folder = dst_root / final_label
            tasks.append((src_file, dst_folder))
            stats[final_label] += 1

    print(f"[INFO] Found {len(tasks)} images to process across {len(stats)} classes.")
    for cls, count in sorted(stats.items()):
        print(f"  - {cls}: {count}")
    if missing:
        print(f"[WARN] {missing} images listed in the CSV were not found in the image folder.")
    if unknown:
        print(f"[WARN] labels without a folder-name mapping (kept as-is): {dict(unknown)}")
    if len(tasks) != expected_n:
        print(f"[WARN] expected {expected_n} images for {args.split} (November-2025 competition files); "
              f"found {len(tasks)}.")

    if not tasks:
        print("[WARN] No tasks found. Check paths.")
        return

    dst_root.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for src_file, dst_folder in sorted(tasks):
        dst_folder.mkdir(parents=True, exist_ok=True)
        dst_file = dst_folder / src_file.name

        # Handle duplicates
        if dst_file.exists():
            # Check if same file
            if dst_file.stat().st_size == src_file.stat().st_size:
                continue # Skip duplicate
            else:
                # Rename collision
                stem = src_file.stem
                suffix = src_file.suffix
                dst_file = dst_folder / f"{stem}_dup{suffix}"

        try:
            if args.mode == "copy":
                shutil.copy2(src_file, dst_file)
            elif args.mode == "move":
                shutil.move(src_file, dst_file)
            elif args.mode == "symlink":
                os.symlink(src_file, dst_file)
            success_count += 1
        except Exception as e:
            print(f"[ERR] Failed to process {src_file}: {e}")

    print(f"\n[DONE] Successfully processed {success_count} images to {dst_root}")

if __name__ == "__main__":
    main()
