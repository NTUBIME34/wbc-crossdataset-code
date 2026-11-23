#!/usr/bin/env python3
"""Present-class metrics of the leave-one-dataset-out (LODO) folds.

Each fold's held-out dataset contains only some of the nine training classes, so the
primary metric is the present-class macro-F1: the mean F1 over the classes with
support > 0 in the held-out truth (a prediction of an absent class is a false negative
for the true class; the absent class itself is not averaged). Per fold and seed the
script computes, from the nine-class predictions.csv written by eval/evaluate_lodo.py
(the common evaluator with the LODO settings: nine classes, no TTA):
  * present-class macro-F1 (per seed, mean and population SD over seeds),
  * accuracy and present-class balanced accuracy (mean recall over present classes),
  * per-class support n, recall and F1 (seed means),
  * the full-label-space values of Supplementary Note 1: macro-F1 over the fixed nine
    classes, and over the union of true and predicted labels,
  * the across-fold mean of each column, with folds weighted equally (main Table 4)
    and weighted by held-out cell count (Supplementary S16),
  * for every class with mean recall below --collapse-recall, the share of its cells
    (pooled over seeds) sent to its most frequent wrong class.

Inputs: --fold NAME FILE [FILE ...], one predictions.csv per seed (pass the seeds in
the order you want them printed; the paper uses seeds 42/52/62). Archive files with
image_id,true_label,pred_label are read too.

Paper: main Table 4 (number of present classes, accuracy, present-class macro-F1
mean +- sd, present-class balanced accuracy, Mean row), Supplementary Table S14
(per-class n / recall / F1) and its routing percentages (78.5% of AML-LMU
Myeloblasts called Lymphocyte, 95.2% of HRLS Lymphoblasts called Myeloblast),
Supplementary Note 1 (fixed nine-class and union macro-F1; per-seed values) and the
proposed column of Supplementary S16.

Example:
    python stats/lodo_present_class.py \\
        --fold AML-LMU lmu_s42/predictions.csv lmu_s52/predictions.csv lmu_s62/predictions.csv \\
        --fold PBC pbc_s42/predictions.csv pbc_s52/predictions.csv pbc_s62/predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CLS9 = ["Basophil", "Eosinophil", "Erythroblast", "Lymphoblast", "Lymphocyte",
        "Monocyte", "Myeloblast", "Neutrophil", "Promyelocyte"]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_labels(path: str | Path) -> tuple[list[str], list[str]]:
    rows = read_rows(path)
    if not rows or "true_label" not in rows[0] or "pred_label" not in rows[0]:
        sys.exit(f"[ERR] {path}: needs true_label and pred_label columns")
    return [r["true_label"] for r in rows], [r["pred_label"] for r in rows]


def per_class_f1(true: list[str], pred: list[str]):
    """Return (present_macro_f1, {class: f1}) over classes present in true_label."""
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int); present = set()
    for t, p in zip(true, pred):
        present.add(t)
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1; fn[t] += 1
    f1 = {}
    for c in present:
        d = 2 * tp[c] + fp[c] + fn[c]
        f1[c] = 0.0 if d == 0 else 2 * tp[c] / d
    macro = st.mean(f1[c] for c in present) if present else float("nan")
    return macro, f1


def macro_over(true: list[str], pred: list[str], labels: list[str]) -> float:
    """Mean F1 over a fixed label list (zero for a label with no support and no prediction)."""
    tp = Counter(t for t, p in zip(true, pred) if t == p)
    pc = Counter(pred)
    tc = Counter(true)
    f = [0.0 if (pc[c] + tc[c]) == 0 else 2 * tp[c] / (pc[c] + tc[c]) for c in labels]
    return float(np.mean(f))


def seed_stats(true: list[str], pred: list[str]) -> dict:
    present_macro, f1 = per_class_f1(true, pred)
    support = Counter(true)
    hits = Counter(t for t, p in zip(true, pred) if t == p)
    recall = {c: hits[c] / support[c] for c in support}
    return {
        "present_macro_f1": present_macro,
        "accuracy": sum(hits.values()) / len(true),
        "present_balanced_accuracy": float(np.mean([recall[c] for c in support])),
        "nine_class_macro_f1": macro_over(true, pred, CLS9),
        "union_macro_f1": macro_over(true, pred, sorted(set(true) | set(pred))),
        "f1": f1, "recall": recall, "support": dict(support), "n": len(true),
    }


def mean_sd(v: list[float]) -> tuple[float, float]:
    return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0)


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
        description="Present-class macro-F1, accuracy, balanced accuracy and per-class "
                    "recall/F1 of leave-one-dataset-out folds (seed mean +- population SD).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="Inputs:" + __doc__.split("Inputs:", 1)[1])
    ap.add_argument("--fold", nargs="+", action="append", required=True, metavar=("NAME", "FILE"),
                    help="held-out dataset name followed by one predictions.csv per seed")
    ap.add_argument("--seeds", nargs="+", default=None, help="seed names for the printout")
    ap.add_argument("--collapse-recall", type=float, default=0.10,
                    help="report error routing for classes with mean recall below this")
    ap.add_argument("--out-dir", default=None, help="default $OUT_ROOT/stats")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()

    report: dict = {"protocol": "present-class macro-F1 = mean F1 over classes with support > 0 "
                                "in the held-out truth; seed mean and population SD",
                    "folds": {}}
    cols = {"present_macro_f1": [], "accuracy": [], "present_balanced_accuracy": [],
            "nine_class_macro_f1": [], "union_macro_f1": []}
    print(f"{'held-out':<12}{'#cls':>5}{'n':>8}  {'accuracy':>8}  {'macro-F1 (present)':<20}"
          f"{'bal.acc':>7}  {'nine-class':<16}{'union':<16}per-seed present macro-F1")
    blocks = []
    for spec in args.fold:
        if len(spec) < 2:
            ap.error("--fold needs a name and at least one file")
        name, files = spec[0], spec[1:]
        seeds = args.seeds or [str(i + 1) for i in range(len(files))]
        if len(seeds) != len(files):
            ap.error("--seeds must match the number of files per fold")
        runs = [load_labels(f) for f in files]
        stats = [seed_stats(t, p) for t, p in runs]
        if len({s["n"] for s in stats}) != 1 or len({tuple(sorted(s["support"].items())) for s in stats}) != 1:
            sys.exit(f"[ERR] {name}: the seed files do not hold the same held-out cells")
        fold = {"n": stats[0]["n"], "n_present": len(stats[0]["support"]), "seeds": seeds}
        for k in cols:
            v = [s[k] for s in stats]
            fold[k] = {"per_seed": dict(zip(seeds, v)), "mean": mean_sd(v)[0], "sd": mean_sd(v)[1]}
            cols[k].append(fold[k]["mean"])
        classes = sorted(stats[0]["support"], key=lambda c: np.mean([s["recall"][c] for s in stats]))
        fold["per_class"] = {c: {"n": stats[0]["support"][c],
                                 "recall": float(np.mean([s["recall"][c] for s in stats])),
                                 "f1": float(np.mean([s["f1"][c] for s in stats])),
                                 "f1_per_seed": [s["f1"][c] for s in stats]} for c in classes}
        routing = {}
        for c in classes:
            if fold["per_class"][c]["recall"] < args.collapse_recall:
                wrong = Counter(p for t, pr in runs for tt, p in zip(t, pr) if tt == c and p != c)
                total = sum(1 for t, _ in runs for tt in t if tt == c)
                if wrong:
                    dest, k = wrong.most_common(1)[0]
                    routing[c] = {"to": dest, "share": k / total, "cells_pooled": total}
        fold["routing"] = routing
        report["folds"][name] = fold
        pm = fold["present_macro_f1"]
        print(f"{name:<12}{fold['n_present']:>5}{fold['n']:>8}  {fold['accuracy']['mean']:>8.3f}  "
              f"{pm['mean']:.3f} +- {pm['sd']:.3f}      {fold['present_balanced_accuracy']['mean']:>7.3f}  "
              f"{fold['nine_class_macro_f1']['mean']:.3f} +- {fold['nine_class_macro_f1']['sd']:.3f}  "
              f"{fold['union_macro_f1']['mean']:.3f} +- {fold['union_macro_f1']['sd']:.3f}  "
              + ", ".join(f"{s}: {v:.3f}" for s, v in pm["per_seed"].items()))
        blocks.append((name, fold))
    report["mean_over_folds"] = {k: st.mean(v) for k, v in cols.items()}
    n_tot = sum(f["n"] for _, f in blocks)
    report["mean_over_folds_cell_weighted"] = {
        k: sum(f[k]["mean"] * f["n"] for _, f in blocks) / n_tot for k in cols}
    for label, mo in (("Mean", report["mean_over_folds"]),
                      ("Mean (cells)", report["mean_over_folds_cell_weighted"])):
        print(f"{label:<12}{'':>5}{'':>8}  {mo['accuracy']:>8.3f}  {mo['present_macro_f1']:.3f}{'':<15}"
              f"{mo['present_balanced_accuracy']:>7.3f}  {mo['nine_class_macro_f1']:.3f}{'':<11}"
              f"{mo['union_macro_f1']:.3f}")
    print("(Mean: folds weighted equally, as in main Table 4; Mean (cells): weighted by held-out "
          "cell count)")
    print("\nPer-class (seed means; classes ordered by recall):")
    print(f"{'held-out':<12}{'class':<14}{'n':>8}{'recall':>8}{'F1':>8}   F1 per seed")
    for name, fold in blocks:
        for c, v in fold["per_class"].items():
            print(f"{name:<12}{c:<14}{v['n']:>8}{v['recall']:>8.3f}{v['f1']:>8.3f}   "
                  + "/".join(f"{x:.3f}" for x in v["f1_per_seed"]))
        for c, r in fold["routing"].items():
            print(f"{'':<12}-> {100 * r['share']:.1f}% of the {fold['per_class'][c]['n']:,} {c} cells "
                  f"(pooled over seeds) were predicted {r['to']}")

    if args.no_write:
        return
    d = out_dir(args.out_dir)
    path = d / "lodo_present_class.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    with (d / "lodo_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["held_out", "class", "n", "recall_mean", "f1_mean"])
        for name, fold in blocks:
            for c, v in fold["per_class"].items():
                w.writerow([name, c, v["n"], f"{v['recall']:.6f}", f"{v['f1']:.6f}"])
    print(f"\n[saved] {path}, lodo_per_class.csv")


if __name__ == "__main__":
    main()
