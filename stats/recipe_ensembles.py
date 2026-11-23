#!/usr/bin/env python3
"""Three-seed probability ensembles of the augmentation / imbalance recipes, with the
recipe-selection bootstraps.

For each recipe group (one per-image prediction file per seed):
  * Probability ensemble: mean over seeds of the prob_<class> columns (seeds joined by
    image id; a class column missing from a file counts as 0), label = argmax over the
    classes in alphabetical order.
  * Ensemble metrics over the labels present in truth or prediction: accuracy,
    balanced accuracy (mean recall), macro-F1, minimum class recall, minimum class F1,
    Blast group and Promyelocyte recall / F1, and the balanced-utility score
    0.35 macro-F1 + 0.30 balanced accuracy + 0.20 min recall + 0.15 min F1.
  * Per-group image bootstrap (--n-boot-ci replicates, percentile 2.5 / 97.5) of those
    metrics, then, for every pair of groups in the order given, a paired image
    bootstrap (--n-boot-paired replicates) of the difference of each metric.
All draws come from ONE numpy.random.default_rng(20260629), consumed in that order
(group CIs first, then the pairs, metric by metric), each draw rng.integers(0, n, n).
The bootstrap therefore depends on the group order, the replicate counts and the image
order. --order file (default) keeps the row order of the first file of the first
group, as the archived run did; --order image_id sorts by image id instead.

Inputs: eval/evaluate_classifier.py predictions.csv (--use-mappings, --tta hflip) with
prob_<class> columns, .csv or .csv.gz.

Paper: Supplementary Table S8 (accuracy, balanced accuracy, macro-F1, minimum recall,
BLAST F1, PROMY F1 of the three Swin-B recipe ensembles) and main Results 3.6, the
recipe-selection bootstraps (1,000 replicates): +0.013 [+0.0004, +0.025] over the
class-weighted-CE baseline and +0.030 [+0.016, +0.044] over mixup + BPaCo + cRT.
Pass the groups in the paper's order: color aug, class-weighted CE, mixup + BPaCo + cRT.

Example:
    python stats/recipe_ensembles.py \\
        --group color_only_hflip co_s42.csv co_s62.csv co_s82.csv \\
        --group ce_baseline_hflip ce_s42.csv ce_s62.csv ce_s82.csv \\
        --group mixup_bpaco_crt_hflip mx_s42.csv mx_s62.csv mx_s82.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_METRICS = ("macro_f1", "balanced_accuracy", "min_recall", "min_f1", "blast_f1", "promy_f1")


# ---------------------------------------------------------------- input
def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def field_id(path: str) -> str:
    return Path(path).name.split("__")[0].split(".")[0]


def row_key(row: dict[str, str]) -> str:
    for col in ("image_path", "image_id", "path"):
        if col in row:
            return field_id(row[col])
    sys.exit("[ERR] prediction file without an image_path / image_id column")


def prob_columns(rows: list[dict[str, str]]) -> list[str]:
    return sorted([c for c in rows[0] if c.startswith("prob_")])


def out_dir(arg: str | None) -> Path:
    """--out-dir, else $OUT_ROOT/stats/recipe_ensembles (OUT_ROOT defaults to ./outputs)."""
    d = Path(arg) if arg else Path(os.environ.get("OUT_ROOT", "outputs")) / "stats" / "recipe_ensembles"
    d = d.expanduser().resolve()
    if d == REPO_ROOT or REPO_ROOT in d.parents:
        sys.exit(f"[ERR] output directory {d} is inside the repository; "
                 "set OUT_ROOT or --out-dir to a location outside it")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- metrics
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = sorted(set(y_true) | set(y_pred))
    return fast_metrics(y_true, y_pred, labels)


def fast_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    recalls = []
    f1s = []
    correct = float((y_true == y_pred).sum())
    for label in labels:
        tp = float(((y_true == label) & (y_pred == label)).sum())
        fp = float(((y_true != label) & (y_pred == label)).sum())
        fn = float(((y_true == label) & (y_pred != label)).sum())
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        recalls.append(recall)
        f1s.append(f1)
    per_label = {label: {"recall": recalls[i], "f1": f1s[i]} for i, label in enumerate(labels)}
    return {
        "accuracy": correct / float(len(y_true)) if len(y_true) else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "min_recall": min(recalls),
        "min_f1": min(f1s),
        "blast_recall": per_label.get("Blast_Group", {}).get("recall", math.nan),
        "blast_f1": per_label.get("Blast_Group", {}).get("f1", math.nan),
        "promy_recall": per_label.get("Promyelocyte", {}).get("recall", math.nan),
        "promy_f1": per_label.get("Promyelocyte", {}).get("f1", math.nan),
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, rng: np.random.Generator,
                 n_boot: int) -> dict[str, tuple[float, float, float]]:
    n = len(y_true)
    labels = sorted(set(y_true) | set(y_pred))
    vals: dict[str, list[float]] = {}
    point = fast_metrics(y_true, y_pred, labels)
    for key in point:
        vals[key] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = fast_metrics(y_true[idx], y_pred[idx], labels)
        for key, val in m.items():
            vals[key].append(val)
    return {
        key: (point[key], float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for key, v in vals.items()
    }


def load_group_ensemble(paths: list[str], key_order: list[str] | None
                        ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Mean seed probabilities -> (image order, y_true, ensemble y_pred)."""
    per_seed = [read_csv(p) for p in paths]
    own_order = [row_key(r) for r in per_seed[0]]
    if len(set(own_order)) != len(own_order):
        sys.exit(f"[ERR] {paths[0]}: an image id occurs twice")
    key_order = own_order if key_order is None else key_order
    first = {row_key(r): r for r in per_seed[0]}
    missing = [k for k in key_order if k not in first]
    if missing or len(first) != len(key_order):
        sys.exit(f"[ERR] {paths[0]}: image set differs from the first group")
    y_true = np.array([first[k]["true_label"] for k in key_order])
    classes = sorted({c[5:] for rows in per_seed for c in prob_columns(rows)})
    prob_sum = {c: np.zeros(len(key_order), dtype=float) for c in classes}
    for p, rows in zip(paths, per_seed):
        by_key = {row_key(r): r for r in rows}
        for i, key in enumerate(key_order):
            if key not in by_key:
                sys.exit(f"[ERR] {p}: image {key} missing")
            row = by_key[key]
            for c in classes:
                prob_sum[c][i] += float(row.get(f"prob_{c}", 0.0))
    for c in classes:
        prob_sum[c] /= len(per_seed)
    prob_matrix = np.vstack([prob_sum[c] for c in classes]).T
    y_pred = np.array([classes[i] for i in prob_matrix.argmax(axis=1)])
    return key_order, y_true, y_pred


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recipe probability ensembles (Supplementary S8) and the 1,000-replicate "
                    "recipe-selection bootstraps (main Results 3.6).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="Inputs:" + __doc__.split("Inputs:", 1)[1])
    ap.add_argument("--group", nargs="+", action="append", required=True, metavar=("NAME", "FILE"),
                    help="group name followed by one prediction file per seed (repeatable; "
                         "order matters for the shared RNG)")
    ap.add_argument("--rng-seed", type=int, default=20260629, help="seed of the shared RNG")
    ap.add_argument("--n-boot-ci", type=int, default=1000, help="per-group bootstrap replicates")
    ap.add_argument("--n-boot-paired", type=int, default=1000, help="paired bootstrap replicates")
    ap.add_argument("--order", choices=["file", "image_id"], default="file",
                    help="image order: rows of the first file (default, archived run) or sorted ids")
    ap.add_argument("--out-dir", default=None, help="default $OUT_ROOT/stats/recipe_ensembles")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()
    rng = np.random.default_rng(args.rng_seed)

    groups = {}
    for spec in args.group:
        if len(spec) < 2:
            ap.error("--group needs a name and at least one file")
        groups[spec[0]] = spec[1:]

    summary_rows, ci_rows, ensembles = [], [], {}
    key_order = None
    for group, paths in groups.items():
        if key_order is None and args.order == "image_id":
            key_order = sorted(row_key(r) for r in read_csv(paths[0]))
        key_order, y_true, y_pred = load_group_ensemble(paths, key_order)
        ensembles[group] = (y_true, y_pred)
        m = metrics(y_true, y_pred)
        utility = (0.35 * m["macro_f1"] + 0.30 * m["balanced_accuracy"]
                   + 0.20 * m["min_recall"] + 0.15 * m["min_f1"])
        summary_rows.append({"group": group, "n": len(y_true), **m, "balanced_utility": utility})
        ci = bootstrap_ci(y_true, y_pred, rng, n_boot=args.n_boot_ci)
        for metric_name, (point, lo, hi) in ci.items():
            ci_rows.append({"group": group, "metric": metric_name, "point": point,
                            "ci95_low": lo, "ci95_high": hi})

    pair_rows = []
    names = list(ensembles)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            y_true_a, pred_a = ensembles[a]
            y_true_b, pred_b = ensembles[b]
            if not np.array_equal(y_true_a, y_true_b):
                sys.exit(f"[ERR] true labels of {a} and {b} differ")
            n = len(y_true_a)
            labels = sorted(set(y_true_a) | set(pred_a) | set(pred_b))
            point_a = fast_metrics(y_true_a, pred_a, labels)
            point_b = fast_metrics(y_true_b, pred_b, labels)
            for metric_name in PAIR_METRICS:
                diffs = []
                for _ in range(args.n_boot_paired):
                    idx = rng.integers(0, n, size=n)
                    diffs.append(fast_metrics(y_true_a[idx], pred_a[idx], labels)[metric_name]
                                 - fast_metrics(y_true_b[idx], pred_b[idx], labels)[metric_name])
                d = np.array(diffs)
                pair_rows.append({
                    "model_a": a, "model_b": b, "metric": metric_name,
                    "point_delta_a_minus_b": point_a[metric_name] - point_b[metric_name],
                    "ci95_low": float(np.nanpercentile(d, 2.5)),
                    "ci95_high": float(np.nanpercentile(d, 97.5)),
                    "p_two_sided_bootstrap": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                })

    print(f"{'ensemble':<26}{'n':>6}  {'Acc':>6} {'Bal.acc':>7} {'Macro-F1':>8} {'MinRec':>7} "
          f"{'BLAST F1':>8} {'PROMY F1':>8}")
    for r in summary_rows:
        print(f"{r['group']:<26}{r['n']:>6}  {r['accuracy']:>6.3f} {r['balanced_accuracy']:>7.3f} "
              f"{r['macro_f1']:>8.3f} {r['min_recall']:>7.3f} {r['blast_f1']:>8.3f} {r['promy_f1']:>8.3f}")
    print(f"\nPaired bootstrap of the macro-F1 difference ({args.n_boot_paired} replicates, "
          f"order: {args.order}):")
    for r in pair_rows:
        if r["metric"] == "macro_f1":
            print(f"  {r['model_a']} - {r['model_b']}: {r['point_delta_a_minus_b']:+.3f} "
                  f"(95% CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]; "
                  f"delta {r['point_delta_a_minus_b']:+.4f})")

    if args.no_write:
        return
    d = out_dir(args.out_dir)
    write_rows(d / "ensemble_summary.csv", summary_rows)
    write_rows(d / "ensemble_bootstrap_ci.csv", ci_rows)
    write_rows(d / "paired_bootstrap.csv", pair_rows)
    print(f"\n[saved] {d}/ensemble_summary.csv, ensemble_bootstrap_ci.csv, paired_bootstrap.csv")


if __name__ == "__main__":
    main()
