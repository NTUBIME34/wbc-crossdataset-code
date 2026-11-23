#!/usr/bin/env python3
"""Internal macro-F1 in the nine-class space and mapped to the seven evaluation classes.

Every model was tested on the same internal held-out split (n = 12,803), so each
run's nine-class confusion matrix can be collapsed to the external seven-class space
(Erythroblast, Lymphoblast and Myeloblast -> Blast_Group; the other classes keep
their names) and macro-F1 recomputed without new inference: F1 per class from the
collapsed matrix, F1 = 0 when a class has no prediction and no support, macro-F1 =
mean over the seven classes. The nine-class macro-F1 of the same matrix is printed
alongside. Both are reported as the mean over the given seeds.

Inputs (--model NAME FILE [FILE ...], one file per seed), any of:
  confusion_matrix.csv written by train/train_classifier.py (first row: empty cell
      then the class names; then one row per true class), also as .csv.gz, as in
      analysis/evidence/ internal matrices;
  test_results.json written by the trainer ("confusion_matrix" key; rows and columns
      in the alphabetical nine-class order);
  confusion_counts.npy written by eval/evaluate_classifier.py without --use-mappings
      (rows and columns in the sorted order of the labels; nine-class order assumed);
  a nine-class predictions.csv (true_label, pred_label).

Paper: Supplementary S17, "Internal seven-class sensitivity" (0.952 proposed Swin,
0.944 VGG16-BN, 0.939 EfficientNet-B0, 0.937 ResNet50, 0.945 Swin-B BPaCo, 0.930 and
0.931 ResNet50 BPaCo with AdamW and with SGD, 0.953 fine-tuned DinoBloom-B) and the
nine-class internal column of main Table 3.

Example:
    python stats/internal_seven_class.py \\
        --model "Swin (proposed)" s42/confusion_matrix.csv s62/confusion_matrix.csv s82/confusion_matrix.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CLS9 = ["Basophil", "Eosinophil", "Erythroblast", "Lymphoblast", "Lymphocyte",
        "Monocyte", "Myeloblast", "Neutrophil", "Promyelocyte"]
CLS7 = ["Basophil", "Blast_Group", "Eosinophil", "Lymphocyte",
        "Monocyte", "Neutrophil", "Promyelocyte"]
BLAST = {"Erythroblast", "Lymphoblast", "Myeloblast"}


def to7(name: str) -> int:
    return CLS7.index("Blast_Group" if name in BLAST else name)


def read_matrix(path: str) -> tuple[list[str], np.ndarray]:
    """Return (class names in row order, square count matrix)."""
    p = Path(path)
    if p.suffix == ".json":
        cm = np.array(json.loads(p.read_text())["confusion_matrix"])
        return CLS9[:len(cm)], cm
    if p.suffix == ".npy":
        cm = np.load(p)
        if cm.shape != (len(CLS9), len(CLS9)):
            sys.exit(f"[ERR] {path}: expected a 9x9 matrix, got {cm.shape}; pass predictions.csv")
        return CLS9, cm
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    if "true_label" in header and "pred_label" in header:      # per-image predictions
        ti, pi = header.index("true_label"), header.index("pred_label")
        names = sorted({r[ti] for r in rows[1:]} | {r[pi] for r in rows[1:]})
        cm = np.zeros((len(names), len(names)), dtype=int)
        for r in rows[1:]:
            cm[names.index(r[ti]), names.index(r[pi])] += 1
        return names, cm
    names = [h for h in header[1:]]
    if [r[0] for r in rows[1:]] != names:
        sys.exit(f"[ERR] {path}: row labels differ from the column labels")
    return names, np.array([[int(float(x)) for x in r[1:]] for r in rows[1:]])


def macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for k in range(len(cm)):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp
        f1s.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def collapse(names: list[str], cm9: np.ndarray) -> np.ndarray:
    cm7 = np.zeros((len(CLS7), len(CLS7)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            cm7[to7(a), to7(b)] += cm9[i, j]
    return cm7


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
        description="Internal nine-class macro-F1 and its collapse to the seven evaluation "
                    "classes from stored confusion matrices (Supplementary S17).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="Inputs" + __doc__.split("Inputs", 1)[1])
    ap.add_argument("--model", nargs="+", action="append", required=True, metavar=("NAME", "FILE"),
                    help="model name followed by one confusion matrix per seed (repeatable)")
    ap.add_argument("--out-dir", default=None, help="default $OUT_ROOT/stats")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()

    rows = []
    print(f"{'model':<34}{'n':>7}  {'internal 9-class':<18}{'internal 7-class':<18}per seed (7-class)")
    for spec in args.model:
        if len(spec) < 2:
            ap.error("--model needs a name and at least one file")
        name, files = spec[0], spec[1:]
        m9, m7, ns = [], [], []
        for f in files:
            names, cm = read_matrix(f)
            unknown = [c for c in names if c not in CLS9]
            if unknown:
                sys.exit(f"[ERR] {f}: classes outside the nine-class taxonomy: {unknown}")
            m9.append(macro_f1(cm))
            cm7 = collapse(names, cm)
            m7.append(macro_f1(cm7))
            ns.append(int(cm7.sum()))
        if len(set(ns)) != 1:
            print(f"[WARN] {name}: the matrices hold different numbers of cells: {ns}")
        rows.append({"model": name, "n": ns[0], "internal9_mean": st.mean(m9),
                     "internal7_mean": st.mean(m7), "internal9_per_seed": m9,
                     "internal7_per_seed": m7})
        print(f"{name:<34}{ns[0]:>7}  {st.mean(m9):<18.3f}{st.mean(m7):<18.3f}"
              + "/".join(f"{v:.4f}" for v in m7) + f"  (means {st.mean(m9):.4f} / {st.mean(m7):.4f})")

    if args.no_write:
        return
    path = out_dir(args.out_dir) / "internal_seven_class.json"
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
