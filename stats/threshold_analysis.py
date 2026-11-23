#!/usr/bin/env python3
"""Post hoc one-vs-rest probability thresholds for the minority classes (Supplementary S11).

Protocol:
  1. Build the three-seed probability ensemble of the proposed model (mean of the
     seven-class probabilities over seeds, joined by image id) on (a) the internal
     held-out test split, evaluated with the seven-class mapping and horizontal-flip TTA,
     and (b) the external WBCBench2026 Phase-1 pre-cropped set.
  2. For Blast_Group and Promyelocyte, choose the one-vs-rest threshold t* that
     maximizes one-vs-rest F1 on the INTERNAL ensemble over t = 0.01, 0.02, ..., 0.99
     (first maximum). An image is positive for class c if its ensemble probability
     p_c >= t, regardless of the other classes; no multiclass label or macro-F1 is
     recomputed. The internal split is used here as a validation proxy, so for this
     analysis it is not an untouched test set.
  3. Apply t* unchanged to the external ensemble ("internal-tuned").
  4. Compare with the argmax rule on the external ensemble and with the oracle
     threshold chosen on the external set itself (an upper bound that cannot be
     achieved in deployment).
This is a secondary analysis; every main result uses argmax decisions.

Inputs: eval/evaluate_classifier.py predictions.csv files (--use-mappings, --tta hflip)
with prob_<class> columns: --internal (internal test split, one per seed) and
--external (external set, one per seed).

Output: threshold_validation_locked.csv (columns cls, t_star, internal_f1,
ext_argmax_f1, ext_locked_f1, ext_locked_prec, ext_locked_rec, t_oracle,
ext_oracle_f1; 4 decimals), the format of analysis/tables/threshold_validation_locked.csv
cited by S11, written to --out-dir (default $OUT_ROOT/stats).

Paper: Supplementary Table S11 and main Results 3.6 (Promyelocyte 0.702 -> 0.719,
Blast group 0.841 -> 0.839).

Example:
    python stats/threshold_analysis.py \\
        --internal int_s42.csv int_s62.csv int_s82.csv \\
        --external ext_s42.csv ext_s62.csv ext_s82.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Basophil", "Blast_Group", "Eosinophil", "Lymphocyte",
           "Monocyte", "Neutrophil", "Promyelocyte"]
TARGETS = ["Blast_Group", "Promyelocyte"]
FIELDS = ["cls", "t_star", "internal_f1", "ext_argmax_f1", "ext_locked_f1",
          "ext_locked_prec", "ext_locked_rec", "t_oracle", "ext_oracle_f1"]


def load(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def field_id(path: str) -> str:
    return Path(path).name.split("__")[0].split(".")[0]


def ensemble(paths: list[str]):
    """Average prob vectors across seeds, keyed by image id. Returns a list of
    (true_label, {class: prob}, argmax_label)."""
    acc = {}
    for p in paths:
        rows = load(p)
        key_col = next((c for c in ("image_path", "image_id") if c in rows[0]), None)
        if key_col is None or any(f"prob_{c}" not in rows[0] for c in CLASSES):
            sys.exit(f"[ERR] {p}: needs image_path/image_id and prob_<class> for all seven classes")
        keys = [field_id(r[key_col]) for r in rows]
        if len(set(keys)) != len(keys):
            sys.exit(f"[ERR] {p}: image ids are not unique, so seeds cannot be joined by id")
        for key, r in zip(keys, rows):
            probs = {c: float(r[f"prob_{c}"]) for c in CLASSES}
            if key not in acc:
                acc[key] = [r["true_label"], {c: 0.0 for c in CLASSES}, 0]
            for c in CLASSES:
                acc[key][1][c] += probs[c]
            acc[key][2] += 1
    if any(k != len(paths) for _, _, k in acc.values()):
        sys.exit("[ERR] the seed files do not cover the same images")
    out = []
    for key, (true, probs, k) in acc.items():
        mean_probs = {c: probs[c] / k for c in CLASSES}
        argmax = max(CLASSES, key=lambda c: mean_probs[c])
        out.append((true, mean_probs, argmax))
    return out


def ovr_f1(data, cls, threshold=None):
    """One-vs-rest F1 for class cls. If threshold is None, use the argmax rule."""
    tp = fp = fn = 0
    for true, probs, argmax in data:
        pos = (probs[cls] >= threshold) if threshold is not None else (argmax == cls)
        if true == cls and pos:
            tp += 1
        elif true == cls and not pos:
            fn += 1
        elif true != cls and pos:
            fp += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec


def best_threshold(data, cls):
    grid = [i / 100 for i in range(1, 100)]
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1, _, _ = ovr_f1(data, cls, t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


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
        description="Internal-tuned vs oracle one-vs-rest thresholds for Blast_Group and "
                    "Promyelocyte (post hoc; Supplementary S11).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="Inputs:" + __doc__.split("Inputs:", 1)[1])
    ap.add_argument("--internal", nargs="+", required=True,
                    help="internal-test predictions.csv, one per seed")
    ap.add_argument("--external", nargs="+", required=True,
                    help="external predictions.csv, one per seed")
    ap.add_argument("--out-dir", default=None, help="default $OUT_ROOT/stats")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()

    internal = ensemble(args.internal)
    external = ensemble(args.external)
    print(f"internal ensemble n={len(internal)}, external ensemble n={len(external)}")

    rows = []
    print(f"\n{'class':13} {'t*(int)':>8} {'int F1':>7} {'ext argmax F1':>14} {'ext F1 (t*)':>12} "
          f"{'t oracle':>9} {'ext oracle F1':>14}")
    for cls in TARGETS:
        t_star, int_f1 = best_threshold(internal, cls)          # chosen on internal
        ext_argmax_f1, _, _ = ovr_f1(external, cls, None)        # baseline
        ext_locked_f1, lp, lr = ovr_f1(external, cls, t_star)    # internal-tuned
        t_oracle, ext_oracle_f1 = best_threshold(external, cls)  # optimistic upper bound
        rows.append(dict(cls=cls, t_star=t_star, internal_f1=int_f1,
                         ext_argmax_f1=ext_argmax_f1, ext_locked_f1=ext_locked_f1,
                         ext_locked_prec=lp, ext_locked_rec=lr,
                         t_oracle=t_oracle, ext_oracle_f1=ext_oracle_f1))
        print(f"{cls:13} {t_star:>8.3f} {int_f1:>7.3f} {ext_argmax_f1:>14.3f} {ext_locked_f1:>12.3f} "
              f"{t_oracle:>9.3f} {ext_oracle_f1:>14.3f}")

    print()
    for r in rows:
        delta = r["ext_locked_f1"] - r["ext_argmax_f1"]
        gap = r["ext_oracle_f1"] - r["ext_locked_f1"]
        print(f"  {r['cls']}: internal-tuned change vs argmax = {delta:+.4f}; "
              f"oracle minus internal-tuned = {gap:+.4f}")

    if args.no_write:
        return
    path = out_dir(args.out_dir) / "threshold_validation_locked.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
