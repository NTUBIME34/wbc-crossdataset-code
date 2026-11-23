#!/usr/bin/env python3
"""Seed mean and population SD of per-seed external metrics.

For each run (one per-image prediction file per seed) this prints the mean and the
population standard deviation (ddof=0) over seeds of macro-F1, weighted F1,
accuracy and balanced accuracy, the seed means of the per-class F1 (and, with
--per-class, recall), and the per-seed macro-F1 values. This is aggregation (i) of
the main Methods: each seed's nine-class argmax mapped to the seven evaluation
classes, scored per seed, then averaged.

Metric definitions (as in the paper): macro-F1 is the unweighted mean of the F1 of
the seven evaluation classes (scikit-learn f1_score with labels= the seven classes,
zero_division=0); weighted F1 uses the same labels; balanced accuracy is the mean
recall over the classes present in the truth.

Input formats (one file per seed, .csv or .csv.gz):
  eval/evaluate_classifier.py predictions.csv (run with --use-mappings):
      image_path, ..., true_label, pred_label[, prob_<class> ...]
  pipeline/classify.py output on class-folder input: source_path, pred_label,
      det_status, ...; the true label is the class folder of source_path, mapped
      with the evaluator's merge/ignore rules
  archive files with image_id,true_label,pred_label and path,true,pred dumps.

Paper: main Table 3 (external columns; the per-seed rows behind every "mean +- sd"),
the per-seed values quoted in Supplementary S7, and Supplementary Table S10
(per-class external F1 by backbone, three-seed means).

Example:
    python stats/seed_summary.py \\
        --run "Swin (proposed)" seed42/predictions.csv seed62/predictions.csv seed82/predictions.csv \\
        --run VGG16-BN vgg_s42.csv vgg_s43.csv vgg_s44.csv --per-class
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CLS7 = ["Basophil", "Blast_Group", "Eosinophil", "Lymphocyte",
        "Monocyte", "Neutrophil", "Promyelocyte"]
SHORT = {"Basophil": "BASO", "Blast_Group": "BLAST", "Eosinophil": "EOS", "Lymphocyte": "LYM",
         "Monocyte": "MONO", "Neutrophil": "NEU", "Promyelocyte": "PROMY"}

# Default merge / ignore rules of eval/evaluate_classifier.py --use-mappings.
MERGE_RULES = [
    "Neutrophil=Band_neutrophil|Segmented_neutrophil|Neutrophil|Neutrophil_sum",
    "Lymphocyte=Lymphocyte|Variant_lymphocyte|Prolymphocyte",
    "Blast_Group=Blast|Lymphoblast|Myeloblast|Erythroblast",
]
IGNORE_CLASSES = {"Plasma_cell", "Metamyelocyte", "Myelocyte"}
ALIAS = {a.strip(): rule.split("=", 1)[0].strip()
         for rule in MERGE_RULES for a in rule.split("=", 1)[1].split("|") if a.strip()}


# ---------------------------------------------------------------- input
def canonical(name: str) -> str | None:
    name = name.strip()
    return None if name in IGNORE_CLASSES else ALIAS.get(name, name)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def field_id(path: str) -> str:
    """Image id shared by all routes: file name without extension and without the
    detector's crop suffix ('.../Lymphocyte/cell0001__det00.png' -> 'cell0001')."""
    return Path(path).name.split("__")[0].split(".")[0]


def load_predictions(path: str | Path) -> dict[str, list[str]]:
    """Return {'id', 'true', 'pred'} lists (one entry per scored image)."""
    rows = read_rows(path)
    if not rows:
        sys.exit(f"[ERR] {path}: no rows")
    cols = set(rows[0])
    path_col = next((c for c in ("image_path", "path", "image_id", "source_path", "crop_path")
                     if c in cols), None)
    pred_col = next((c for c in ("pred_label", "pred", "pred_class") if c in cols), None)
    if path_col is None or pred_col is None:
        sys.exit(f"[ERR] {path}: unrecognised columns {sorted(cols)}")
    true_col = "true_label" if "true_label" in cols else ("true" if "true" in cols else None)
    out: dict[str, list[str]] = {"id": [], "true": [], "pred": []}
    for r in rows:
        src = r[path_col] or r.get("crop_path", "")  # pipeline crop rows may lack source_path
        if true_col is not None:
            t = r[true_col]
        else:  # pipeline output on class-folder input: label = class folder of the image
            t = canonical(Path(src).parent.name)
            if t is None:
                continue
        p = r[pred_col]
        if true_col is None and pred_col == "pred_class":
            p = canonical(p) or "Skipped_Class"
        out["id"].append(field_id(src))
        out["true"].append(t)
        out["pred"].append(p)
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


# ---------------------------------------------------------------- metrics
def seed_metrics(true: list[str], pred: list[str], labels: list[str]) -> dict[str, float]:
    f1s = f1_score(true, pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(true, pred, labels=labels, average=None, zero_division=0)
    m = {
        "n": len(true),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(f1_score(true, pred, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
    }
    m.update({f"f1_{c}": float(v) for c, v in zip(labels, f1s)})
    m.update({f"recall_{c}": float(v) for c, v in zip(labels, rec)})
    return m


def summarise(name: str, files: list[str], labels: list[str]) -> dict:
    per_seed = []
    for f in files:
        d = load_predictions(f)
        per_seed.append(seed_metrics(d["true"], d["pred"], labels))
    res = {"run": name, "n_seeds": len(files), "n": per_seed[0]["n"],
           "n_per_seed": [p["n"] for p in per_seed], "per_seed": per_seed}
    for k in per_seed[0]:
        if k == "n":
            continue
        a = np.array([p[k] for p in per_seed], dtype=float)
        res[k] = (float(a.mean()), float(a.std(ddof=0)))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Seed mean +- population SD (ddof=0) of per-seed macro-F1, weighted F1, "
                    "accuracy, balanced accuracy and per-class F1.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="Example:" + __doc__.split("Example:", 1)[1])
    ap.add_argument("--run", nargs="+", action="append", required=True, metavar=("NAME", "FILE"),
                    help="run name followed by one per-image prediction file per seed (repeatable)")
    ap.add_argument("--labels", nargs="+", default=CLS7,
                    help="label space for F1 (default: the seven evaluation classes)")
    ap.add_argument("--per-class", action="store_true",
                    help="also print per-class F1 and recall seed means")
    ap.add_argument("--digits", type=int, default=3, help="decimals in the summary table (paper: 3)")
    ap.add_argument("--out-dir", default=None,
                    help="where seed_summary.csv is written (default $OUT_ROOT/stats)")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

    results = []
    for spec in args.run:
        if len(spec) < 2:
            ap.error("--run needs a name and at least one file")
        results.append(summarise(spec[0], spec[1:], args.labels))

    d = args.digits
    wn = max([24] + [len(r["run"]) + 2 for r in results])
    fmt = lambda ms: f"{ms[0]:.{d}f} +- {ms[1]:.{d}f}"  # noqa: E731
    print(f"{'run':<{wn}}{'n':>6}{'seeds':>6}  {'macro-F1':<16}{'weighted F1':<16}"
          f"{'accuracy':<16}{'bal. acc':<16}" + "".join(f"{SHORT.get(c, c)[:6]:>7}" for c in args.labels))
    for r in results:
        print(f"{r['run']:<{wn}}{r['n']:>6}{r['n_seeds']:>6}  {fmt(r['macro_f1']):<16}{fmt(r['weighted_f1']):<16}"
              f"{fmt(r['accuracy']):<16}{fmt(r['balanced_accuracy']):<16}"
              + "".join(f"{r['f1_' + c][0]:>7.{d}f}" for c in args.labels))
    print()
    for r in results:
        seeds = ", ".join(f"{p['macro_f1']:.4f}" for p in r["per_seed"])
        print(f"{r['run']}: per-seed macro-F1 {seeds}; mean {r['macro_f1'][0]:.4f} "
              f"(sd {r['macro_f1'][1]:.4f}); n per seed {r['n_per_seed']}")
        if args.per_class:
            for c in args.labels:
                print(f"    {c:<14} F1 {r['f1_' + c][0]:.{d}f} +- {r['f1_' + c][1]:.{d}f}   "
                      f"recall {r['recall_' + c][0]:.{d}f} +- {r['recall_' + c][1]:.{d}f}")

    if args.no_write:
        return
    path = out_dir(args.out_dir) / "seed_summary.csv"
    keys = [k for k in results[0] if isinstance(results[0][k], tuple)]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "n", "n_seeds"] + [f"{k}_{s}" for k in keys for s in ("mean", "sd")]
                   + ["per_seed_macro_f1"])
        for r in results:
            w.writerow([r["run"], r["n"], r["n_seeds"]]
                       + [f"{v:.6f}" for k in keys for v in r[k]]
                       + [";".join(f"{p['macro_f1']:.6f}" for p in r["per_seed"])])
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
