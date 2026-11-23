#!/usr/bin/env python3
"""Calibration of the three-seed probability ensemble on the external set.

Ensemble: mean over seeds of the seven-class probabilities (prob_<class> columns of
eval/evaluate_classifier.py; seeds joined by image id). Reported:
  * global top-label expected calibration error (ECE) over 15 equal-width confidence
    bins (bin i = (lo, hi], the first bin also includes 0), weighted by bin size;
  * multiclass Brier score: mean over images of the squared error of the
    seven-class probability vector against the one-hot truth;
  * accuracy and mean top-label confidence, and the share, mean confidence and
    accuracy of the top confidence bin;
  * the per-bin reliability table (n, mean confidence, accuracy);
  * per-class one-vs-rest Brier score and ECE (same 15 bins, where "accuracy" is the
    fraction of positives and "confidence" the mean class probability) with their
    reliability tables.
The bin count matters (ECE is 0.016 at 10 bins and 0.017 at 15-20 bins), so it is
fixed at 15.

Inputs: one external predictions.csv per seed (eval/evaluate_classifier.py,
--use-mappings, --tta hflip) with prob_<class> columns.

Outputs (in --out-dir, default $OUT_ROOT/stats): calibration_ensemble.json (all of the
above), calibration_bins.csv (global reliability table) and calibration_ensemble.txt.

Paper: main Results 3.6 (expected calibration error 0.017, Brier score 0.074, mean
confidence 0.970 versus accuracy 0.954, top bin 90% of predictions with 0.998 versus
0.984) and Supplementary Fig. S2 (reliability diagram, 15 bins, n = 8,061).

Example:
    python stats/calibration.py --predictions s42/predictions.csv s62/predictions.csv s82/predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CLS7 = ["Basophil", "Blast_Group", "Eosinophil", "Lymphocyte",
        "Monocyte", "Neutrophil", "Promyelocyte"]
PROB = [f"prob_{c}" for c in CLS7]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def field_id(path: str) -> str:
    return Path(path).name.split("__")[0].split(".")[0]


def load_ensemble(files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean probabilities over seeds, aligned by image id to the first file's rows."""
    base_ids, y_true, total = None, None, None
    for f in files:
        rows = read_rows(f)
        key_col = next((c for c in ("image_path", "image_id") if c in rows[0]), None)
        if key_col is None or "true_label" not in rows[0] or any(c not in rows[0] for c in PROB):
            sys.exit(f"[ERR] {f}: needs image_path/image_id, true_label and {', '.join(PROB)}")
        ids = [field_id(r[key_col]) for r in rows]
        if len(set(ids)) != len(ids):
            sys.exit(f"[ERR] {f}: image ids are not unique")
        by_id = {i: r for i, r in zip(ids, rows)}
        if base_ids is None:
            base_ids = ids
            y_true = np.array([by_id[i]["true_label"] for i in base_ids])
            total = np.zeros((len(ids), len(CLS7)))
        elif set(ids) != set(base_ids):
            sys.exit(f"[ERR] {f}: image set differs from {files[0]}")
        total += np.array([[float(by_id[i][c]) for c in PROB] for i in base_ids])
    return y_true, total / len(files)


def ece_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15):
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": f"({lo:.3f},{hi:.3f}]", "n": 0, "mean_conf": None, "acc": None})
            continue
        mc, ac = float(conf[m].mean()), float(correct[m].mean())
        ece += abs(ac - mc) * n / len(conf)
        rows.append({"bin": f"({lo:.3f},{hi:.3f}]", "n": n, "mean_conf": round(mc, 4), "acc": round(ac, 4)})
    return ece, rows


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
        description="ECE (15 bins), multiclass Brier score and reliability tables of the "
                    "three-seed probability ensemble.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="Inputs:" + __doc__.split("Inputs:", 1)[1])
    ap.add_argument("--predictions", nargs="+", required=True, help="one predictions.csv per seed")
    ap.add_argument("--bins", type=int, default=15, help="equal-width confidence bins (paper: 15)")
    ap.add_argument("--out-dir", default=None, help="default $OUT_ROOT/stats")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()

    y_true, probs = load_ensemble(args.predictions)
    N = len(y_true)
    bad = sorted(set(y_true) - set(CLS7))
    if bad:
        sys.exit(f"[ERR] true labels outside the seven classes: {bad}")

    # global top-label ECE
    conf = probs.max(1)
    pred = np.array(CLS7)[probs.argmax(1)]
    correct = (pred == y_true).astype(float)
    g_ece, g_rows = ece_bins(conf, correct, args.bins)
    # multiclass Brier
    onehot = np.stack([(y_true == c).astype(float) for c in CLS7], 1)
    mc_brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    top = next(r for r in reversed(g_rows) if r["n"] > 0)
    # per-class one-vs-rest
    per_class = {}
    for j, c in enumerate(CLS7):
        y = (y_true == c).astype(float)
        p = probs[:, j]
        ovr_ece, rows = ece_bins(p, y, args.bins)
        per_class[c] = {"support": int(y.sum()), "ovr_brier": round(float(np.mean((p - y) ** 2)), 4),
                        "ovr_ece": round(ovr_ece, 4), "reliability": [r for r in rows if r["n"] > 0]}

    out = {"n": N, "n_seeds": len(args.predictions), "bins": args.bins,
           "global_toplabel_ece": round(g_ece, 4), "multiclass_brier": round(mc_brier, 4),
           "accuracy": round(float(correct.mean()), 4), "mean_confidence": round(float(conf.mean()), 4),
           "top_bin": {**top, "share": round(top["n"] / N, 4)},
           "global_reliability_bins": g_rows, "per_class_one_vs_rest": per_class}

    L = [f"Three-seed probability ensemble, n = {N} ({len(args.predictions)} seed files)",
         f"  expected calibration error ({args.bins} bins) = {g_ece:.3f}   ({g_ece:.4f})",
         f"  multiclass Brier score       = {mc_brier:.3f}   ({mc_brier:.4f})",
         f"  mean confidence {conf.mean():.3f} versus accuracy {correct.mean():.3f}   "
         f"({conf.mean():.4f} / {correct.mean():.4f})",
         f"  top confidence bin {top['bin']}: {top['n']} of {N} predictions "
         f"({100 * top['n'] / N:.0f}%), mean confidence {top['mean_conf']:.3f} versus accuracy "
         f"{top['acc']:.3f}",
         "  reliability bins (n, mean confidence, accuracy):"]
    L += [f"    {r['bin']}: n={r['n']:<5} conf={r['mean_conf']}  acc={r['acc']}" for r in g_rows if r["n"]]
    L.append("  per-class one-vs-rest (support / Brier / ECE):")
    L += [f"    {c:<13} n={v['support']:<5} Brier={v['ovr_brier']:.4f}  ECE={v['ovr_ece']:.4f}"
          for c, v in per_class.items()]
    text = "\n".join(L)
    print(text)
    if args.no_write:
        return
    d = out_dir(args.out_dir)
    (d / "calibration_ensemble.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (d / "calibration_ensemble.txt").write_text(text + "\n", encoding="utf-8")
    with (d / "calibration_bins.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bin", "n", "mean_conf", "acc"])
        w.writeheader()
        w.writerows(g_rows)
    print(f"[saved] {d / 'calibration_ensemble.json'}, calibration_bins.csv, calibration_ensemble.txt")


if __name__ == "__main__":
    main()
