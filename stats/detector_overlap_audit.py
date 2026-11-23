#!/usr/bin/env python3
"""Exact overlap between the detector's training images and the LODO held-out sets.

TXL-PBC (the detector's training data) ships a metadata table mapping each of its
1,260 images to the source dataset and the original file name it came from, so the
overlap with the classifier datasets is established by exact file-name match rather
than estimated. Two points fix the number:
  1. The detector was trained on TXL-PBC's train split only (882 images); the val and
     test splits were never trained on.
  2. The comparison is against the leave-one-dataset-out held-out sets: when PBC or
     Raabin-WBC is the held-out classifier domain, the detector that cropped it may
     have seen some of those images during its own training.
For every source given with --heldout, the script reports the held-out cell count,
the number of detector-training images from that source, their exact file-name
overlap (file names compared without extension) and its share of the held-out set.

Inputs:
  --txl-root     TXL-PBC dataset folder holding metadata_file.xlsx (columns
                 New_Image_Name, Original_Image_Name, Source_Dataset) and
                 images/{train,val,test}/ (or give --metadata, .xlsx or .csv)
  --heldout SOURCE FILE [FILE ...]
                 TXL-PBC source name (e.g. Raabin-WBC, PBC) and the per-image files of
                 that held-out fold: eval/evaluate_classifier.py predictions.csv or
                 analysis/evidence/lodo*/heldout_*.csv.gz (image_id column); the ids of
                 all files are pooled.
The .xlsx is read with pandas when openpyxl is installed, otherwise with a small
standard-library reader of the first worksheet.

Paper: Supplementary S1, "Overlap between detector and classifier training sources"
(TXL-PBC composition 500/500/160/100, train split 882 = 358/353/101/70; exactly 358 of
10,154 held-out Raabin-WBC cells, 3.53%, and 353 of 11,845 PBC cells, 2.98%), and the
"about 3%" statements of main Section 2.2 and the main Table 4 caption.

Example:
    python stats/detector_overlap_audit.py --txl-root /data/TXL-PBC \\
        --heldout Raabin-WBC analysis/evidence/lodo/heldout_RaabinWBC_seed42.csv.gz \\
        --heldout PBC analysis/evidence/lodo/heldout_PBC_seed42.csv.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
COLS = ("New_Image_Name", "Original_Image_Name", "Source_Dataset")
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def stem(x: str) -> str:
    return str(x).rsplit(".", 1)[0]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _col_index(ref: str) -> int:
    idx = 0
    for ch in ref:
        if ch.isalpha():
            idx = idx * 26 + (ord(ch.upper()) - 64)
    return idx - 1


def read_xlsx_stdlib(path: Path) -> list[dict[str, str]]:
    """First worksheet of an .xlsx file as a list of dicts (header = first row)."""
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{XLSX_NS}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{XLSX_NS}t")))
        sheet = "xl/worksheets/sheet1.xml"
        if sheet not in z.namelist():
            sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
        table = []
        for row in ET.fromstring(z.read(sheet)).iter(f"{XLSX_NS}row"):
            cells: dict[int, str] = {}
            for c in row.iter(f"{XLSX_NS}c"):
                v = c.find(f"{XLSX_NS}v")
                if c.get("t") == "s":
                    val = shared[int(v.text)] if v is not None else ""
                elif c.get("t") == "inlineStr":
                    val = "".join(t.text or "" for t in c.iter(f"{XLSX_NS}t"))
                else:
                    val = v.text if v is not None and v.text is not None else ""
                cells[_col_index(c.get("r"))] = val
            if cells:
                table.append([cells.get(i, "") for i in range(max(cells) + 1)])
    header = [h.strip() for h in table[0]]
    return [dict(zip(header, r)) for r in table[1:]]


def read_metadata(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        rows = read_rows(path)
    else:
        try:
            import pandas as pd
            rows = pd.read_excel(path).astype(str).to_dict("records")
        except ImportError:
            rows = read_xlsx_stdlib(path)
    if not rows or any(c not in rows[0] for c in COLS):
        sys.exit(f"[ERR] {path}: needs columns {', '.join(COLS)}")
    return rows


def heldout_ids(files: list[str]) -> set[str]:
    out: set[str] = set()
    for f in files:
        rows = read_rows(f)
        col = next((c for c in ("image_id", "image_path", "path") if c in rows[0]), None)
        if col is None:
            sys.exit(f"[ERR] {f}: needs an image_id or image_path column")
        out.update(stem(Path(r[col]).name) for r in rows)
    return out


def out_dir(arg: str | None) -> Path:
    """--out-dir, else $OUT_ROOT/stats (OUT_ROOT defaults to ./outputs); never the repo."""
    d = Path(arg) if arg else Path(os.environ.get("OUT_ROOT", "outputs")) / "stats"
    d = d.expanduser().resolve()
    if d == REPO_ROOT or REPO_ROOT in d.parents:
        sys.exit(f"[ERR] output directory {d} is inside the repository; "
                 "set OUT_ROOT or --out-dir to a location outside it")
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exact file-name overlap between TXL-PBC's detector-training split and "
                    "the leave-one-dataset-out held-out sets (Supplementary S1).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="Inputs:" + __doc__.split("Inputs:", 1)[1])
    ap.add_argument("--txl-root", required=True, help="TXL-PBC folder (metadata + images/<split>)")
    ap.add_argument("--metadata", default=None, help="metadata table (default <txl-root>/metadata_file.xlsx)")
    ap.add_argument("--heldout", nargs="+", action="append", required=True, metavar=("SOURCE", "FILE"),
                    help="TXL-PBC source name followed by the held-out per-image files")
    ap.add_argument("--train-split", default="train", help="split the detector was trained on")
    ap.add_argument("--out-dir", default=None, help="default $OUT_ROOT/stats")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()

    txl = Path(args.txl_root)
    meta = read_metadata(Path(args.metadata) if args.metadata else txl / "metadata_file.xlsx")
    composition = dict(Counter(r["Source_Dataset"] for r in meta))
    splits = {sp: {stem(p.name) for p in (txl / "images" / sp).iterdir() if p.is_file()}
              for sp in SPLITS if (txl / "images" / sp).is_dir()}
    if args.train_split not in splits:
        sys.exit(f"[ERR] {txl / 'images' / args.train_split} not found")
    by_new = {stem(r["New_Image_Name"]): (r["Source_Dataset"], stem(r["Original_Image_Name"]))
              for r in meta}
    split_by_source: dict[str, dict[str, set[str]]] = {sp: {} for sp in splits}
    for sp, names in splits.items():
        for n in names:
            if n in by_new:
                src, orig = by_new[n]
                split_by_source[sp].setdefault(src, set()).add(orig)

    print("TXL-PBC composition (from its own metadata table)")
    for k, v in sorted(composition.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<12} {v:>5}")
    print(f"  {'TOTAL':<12} {sum(composition.values()):>5}")
    print("\nTXL-PBC splits")
    for sp in splits:
        print(f"  {sp:<6} {len(splits[sp]):>5} images   "
              + ", ".join(f"{s}={len(v)}" for s, v in sorted(split_by_source[sp].items())))

    result = {"txl_pbc_composition": composition,
              "txl_pbc_splits": {sp: len(v) for sp, v in splits.items()},
              "txl_pbc_split_by_source": {sp: {s: len(v) for s, v in sorted(d.items())}
                                          for sp, d in split_by_source.items()},
              "detector_trained_on": f"{args.train_split} split only", "overlap": {}}
    print(f"\nOverlap with the held-out sets (detector {args.train_split} split)")
    print(f"{'source':<12}{'held-out cells':>16}{'detector train':>16}{'overlap':>10}{'% of held-out':>15}")
    print("-" * 70)
    for spec in args.heldout:
        if len(spec) < 2:
            ap.error("--heldout needs a source name and at least one file")
        src, files = spec[0], spec[1:]
        if src not in composition:
            print(f"[WARN] {src} is not a TXL-PBC source ({', '.join(sorted(composition))})")
        ho = heldout_ids(files)
        trained = split_by_source[args.train_split].get(src, set())
        inter = trained & ho
        pct = 100 * len(inter) / len(ho) if ho else 0.0
        print(f"{src:<12}{len(ho):>16}{len(trained):>16}{len(inter):>10}{pct:>14.2f}%")
        result["overlap"][src] = {
            "heldout_cells": len(ho),
            "detector_train_images_from_this_source": len(trained),
            "exact_filename_overlap": len(inter),
            "pct_of_heldout": round(pct, 2),
            "all_detector_train_images_appear_in_heldout": len(inter) == len(trained),
        }

    if args.no_write:
        return
    path = out_dir(args.out_dir) / "detector_overlap_audit.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
