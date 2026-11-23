#!/usr/bin/env python3
"""Image-level bootstrap 95% CI of each model's three-seed probability-ensemble macro-F1.

Recipe (Supplementary Table S9; main Results 3.3):
  1. Probability ensemble: for every image, the mean over seeds of the seven-class
     probabilities (prob_<class> columns of eval/evaluate_classifier.py, i.e. the
     nine-class softmax summed into the seven classes, horizontal-flip TTA averaged);
     the ensemble label is the argmax (ties go to the first class in alphabetical
     order). Seeds are joined by image id.
  2. Rows are sorted by image file name.
  3. For each model a fresh numpy.random.default_rng(0) draws 2,000 resamples
     rng.integers(0, n, n) of the n images; macro-F1 (mean F1 over the seven classes,
     F1 = 0 when a class has no prediction and no support) is recomputed on each.
  4. The CI is the percentile interval (numpy.percentile, linear interpolation,
     2.5 / 97.5).
The row order matters: the same draws applied to another order give a different
interval (sorting by full path instead of file name moves the Swin bounds by about
0.0004), so step 2 is part of the recipe.

Input: one per-image prediction file per seed with prob_<class> columns
(eval/evaluate_classifier.py predictions.csv with --use-mappings, or archive files
image_id,true_label,pred_label,prob_<class>).

Output: external_macrof1_bootstrap_ci.csv (backbone, ensemble_macro_f1, ci95_low,
ci95_high; 4 decimals), the format of analysis/tables/external_macrof1_bootstrap_ci.csv
cited by S9, written to --out-dir (default $OUT_ROOT/stats).

Example:
    python stats/bootstrap_ensemble_ci.py \\
        --model "Swin (proposed)" s42/predictions.csv s62/predictions.csv s82/predictions.csv \\
        --model VGG16-BN vgg_s42.csv vgg_s43.csv vgg_s44.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CLS7 = ["Basophil", "Blast_Group", "Eosinophil", "Lymphocyte",
        "Monocyte", "Neutrophil", "Promyelocyte"]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def field_id(path: str) -> str:
    return Path(path).name.split("__")[0].split(".")[0]


def load_ensemble(files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true codes, ensemble probabilities), rows sorted by image file name."""
    acc: dict[str, list] = {}
    for f in files:
        rows = read_rows(f)
        path_col = next((c for c in ("image_path", "path", "image_id") if c in rows[0]), None)
        missing = [c for c in CLS7 if f"prob_{c}" not in rows[0]]
        if path_col is None or "true_label" not in rows[0] or missing:
            sys.exit(f"[ERR] {f}: needs an image path/id column, true_label and prob_<class> "
                     f"columns for {', '.join(CLS7)}")
        seen = set()
        for r in rows:
            fid = field_id(r[path_col])
            if fid in seen:
                sys.exit(f"[ERR] {f}: image id {fid} occurs twice")
            seen.add(fid)
            p = np.array([float(r[f"prob_{c}"]) for c in CLS7])
            if fid not in acc:
                acc[fid] = [r["true_label"], np.zeros(len(CLS7)), 0, Path(r[path_col]).name]
            elif acc[fid][0] != r["true_label"]:
                sys.exit(f"[ERR] true label of {fid} differs between seed files")
            acc[fid][1] += p
            acc[fid][2] += 1
    incomplete = [k for k, v in acc.items() if v[2] != len(files)]
    if incomplete:
        sys.exit(f"[ERR] {len(incomplete)} images are missing from at least one seed file")
    keys = sorted(acc, key=lambda k: acc[k][3])          # sort by image file name
    bad = sorted({acc[k][0] for k in keys} - set(CLS7))
    if bad:
        sys.exit(f"[ERR] true labels outside the seven classes: {bad}")
    y = np.array([CLS7.index(acc[k][0]) for k in keys])
    P = np.array([acc[k][1] / len(files) for k in keys])
    return y, P


def macro_f1(t: np.ndarray, p: np.ndarray, k: int = len(CLS7)) -> float:
    cm = np.bincount(t * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)
    return float(f.mean())


def bootstrap(y: np.ndarray, yp: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    n = len(y)
    rng = np.random.default_rng(seed)
    return np.array([macro_f1(y[i], yp[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])


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
        description="Bootstrap 95% CI (2,000 replicates, percentile) of three-seed "
                    "probability-ensemble macro-F1 (Supplementary Table S9 recipe).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="Input:" + __doc__.split("Input:", 1)[1])
    ap.add_argument("--model", nargs="+", action="append", required=True, metavar=("NAME", "FILE"),
                    help="model name followed by one prediction file per seed (repeatable)")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap replicates (paper: 2,000)")
    ap.add_argument("--rng-seed", type=int, default=0,
                    help="seed of the per-model numpy default_rng (paper: 0)")
    ap.add_argument("--out-dir", default=None, help="default $OUT_ROOT/stats")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()

    rows = []
    print(f"{'backbone':<24}{'n':>6}  {'ensemble macro-F1':<18}{'95% CI':<16}{'(4 decimals)'}")
    for spec in args.model:
        if len(spec) < 2:
            ap.error("--model needs a name and at least one file")
        name, files = spec[0], spec[1:]
        y, P = load_ensemble(files)
        yp = P.argmax(1)
        point = macro_f1(y, yp)
        lo, hi = np.percentile(bootstrap(y, yp, args.n_boot, args.rng_seed), [2.5, 97.5])
        rows.append((name, point, float(lo), float(hi)))
        print(f"{name:<24}{len(y):>6}  {point:<18.3f}{lo:.3f}--{hi:.3f}      "
              f"{point:.4f} [{lo:.4f}, {hi:.4f}]")

    if args.no_write:
        return
    path = out_dir(args.out_dir) / "external_macrof1_bootstrap_ci.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "ensemble_macro_f1", "ci95_low", "ci95_high"])
        for name, point, lo, hi in rows:
            w.writerow([name, f"{point:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
