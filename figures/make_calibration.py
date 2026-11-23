#!/usr/bin/env python3
"""Reliability diagram + calibration metrics for the proposed model (Supplementary
Fig. S2; calibration sentence of Results section 3.6).

Three-seed probability ensemble (seeds 42/62/82, horizontal-flip TTA) on the
external WBCBench2026 Phase-1 evaluated set (n = 8,061), in the harmonized
seven-class space. The ensemble probability vector of an image is the mean of the
three seeds' seven-class probability vectors; the seeds are joined on image_path.

Input: the per-seed predictions.csv files that eval/evaluate_classifier.py writes
(--use-mappings --tta hflip) for the proposed model on your own copy of the
WBCBench2026 Phase-1 pre-cropped set, laid out as

    <pred-dir>/seed_42/predictions.csv
    <pred-dir>/seed_62/predictions.csv
    <pred-dir>/seed_82/predictions.csv

(use --pattern for another layout). Columns used: image_path, true_label and
prob_<class> for the seven classes. WBCBench2026 predictions are not distributed
with this repository; regenerate them with the released weights.

Reported metrics:
  ECE   expected calibration error, 15 equal-width confidence bins
  Brier multiclass Brier score (mean squared error of the probability vector)
plus accuracy, mean confidence and the per-bin table (count, share of predictions,
mean confidence, accuracy, contribution to ECE).

The bin count matters: ECE is 0.016 at 10 bins and 0.017 at 15-20 bins. We fix it
at 15 and state it, so the number is reproducible.

Paper values: ECE 0.017, Brier 0.074, mean confidence 0.970 versus accuracy 0.954;
the top bin holds 90% of the predictions (mean confidence 0.998 versus accuracy 0.984).

Outputs (in --out-dir): fig_calibration.pdf / .png and fig_calibration_bins.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEEDS = (42, 62, 82)
CLASSES = ["Basophil", "Blast_Group", "Eosinophil", "Lymphocyte",
           "Monocyte", "Neutrophil", "Promyelocyte"]
N_BINS = 15
C_CLS = "#7a3fb0"
DEFAULT_PATTERN = "seed_{seed}/predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reliability diagram, ECE and Brier score of the three-seed probability "
                    "ensemble (Supplementary Fig. S2).",
    )
    parser.add_argument("--pred-dir", default=os.environ.get("PRED_DIR"),
                        help="Directory with one evaluator output per seed (required unless "
                             "PRED_DIR is set).")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help="Per-seed predictions file relative to --pred-dir; {seed} is "
                             "replaced by the seed (default: %(default)s).")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help="Seeds to ensemble (default: 42 62 82).")
    parser.add_argument("--bins", type=int, default=N_BINS,
                        help="Number of equal-width confidence bins (default: %(default)s, "
                             "as in the paper).")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: $OUT_ROOT/figures, else ./outputs/figures).")
    args = parser.parse_args()
    if not args.pred_dir:
        parser.error("--pred-dir is required (or set PRED_DIR)")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds lists a seed twice")
    return args


def out_dir_from(arg) -> Path:
    """--out-dir, else $OUT_ROOT/figures, else ./outputs/figures; never inside the repository."""
    out = Path(arg) if arg else Path(os.environ.get("OUT_ROOT", "outputs")) / "figures"
    out = out.expanduser().resolve()
    if out == REPO or REPO in out.parents:
        sys.exit(f"[ERROR] output folder {out} is inside the repository; "
                 "pass --out-dir or set OUT_ROOT to a folder outside it")
    return out


def read_seed(path: Path) -> dict:
    """image_path -> (true_label, seven-class probability vector) for one seed."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = ["image_path", "true_label"] + [f"prob_{c}" for c in CLASSES]
        missing = [c for c in need if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"[ERROR] {path} lacks columns: {', '.join(missing)}")
        for r in reader:
            key = r["image_path"]
            if key in rows:
                sys.exit(f"[ERROR] {path}: image_path occurs twice: {key}")
            rows[key] = (r["true_label"], np.array([float(r[f"prob_{c}"]) for c in CLASSES]))
    return rows


def load_ensemble(pred_dir: Path, pattern: str, seeds):
    """Average the per-class probabilities over the seeds, joined on image_path."""
    per_seed = []
    for s in seeds:
        path = pred_dir / pattern.format(seed=s)
        if not path.is_file():
            sys.exit(f"[ERROR] predictions for seed {s} not found: {path}")
        per_seed.append((s, read_seed(path)))
        print(f"  seed {s}: {len(per_seed[-1][1])} images  ({path})")

    ref_seed, ref = per_seed[0]
    for s, rows in per_seed[1:]:
        if rows.keys() != ref.keys():
            only_ref = sorted(ref.keys() - rows.keys())
            only_s = sorted(rows.keys() - ref.keys())
            sys.exit(f"[ERROR] seeds {ref_seed} and {s} cover different images "
                     f"({len(only_ref)} only in seed {ref_seed}, {len(only_s)} only in seed {s}; "
                     f"e.g. {(only_ref + only_s)[:3]}). Evaluate every seed on the same set.")

    keys = sorted(ref)
    probs = np.zeros((len(keys), len(CLASSES)))
    true = []
    for i, k in enumerate(keys):
        labels = {rows[k][0] for _, rows in per_seed}
        if len(labels) != 1:
            sys.exit(f"[ERROR] true_label differs between seeds for {k}: {sorted(labels)}")
        label = ref[k][0]
        if label not in CLASSES:
            sys.exit(f"[ERROR] true_label '{label}' of {k} is not one of the seven harmonized "
                     f"classes; evaluate with --use-mappings.")
        true.append(label)
        for _, rows in per_seed:
            probs[i] += rows[k][1]
    return probs / len(seeds), true


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir).expanduser().resolve()
    out = out_dir_from(args.out_dir)
    n_bins = args.bins

    print(f"Calibration of the {len(args.seeds)}-seed probability ensemble")
    probs, true = load_ensemble(pred_dir, args.pattern, args.seeds)
    n = len(true)
    pred = [CLASSES[i] for i in probs.argmax(1)]
    conf = probs.max(1)
    correct = np.array([p == t for p, t in zip(pred, true)], dtype=float)

    # ECE over n_bins equal-width bins (lower edge exclusive, upper edge inclusive)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, xs, ys, table = 0.0, [], [], []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        k = int(m.sum())
        row = {"bin_lower": edges[i], "bin_upper": edges[i + 1], "n": k, "share": k / n,
               "mean_confidence": None, "accuracy": None, "ece_contribution": None}
        table.append(row)
        if not k:
            continue
        acc_b, conf_b = correct[m].mean(), conf[m].mean()
        contrib = m.sum() / n * abs(acc_b - conf_b)
        ece += contrib
        xs.append(conf_b)
        ys.append(acc_b)
        row.update(mean_confidence=conf_b, accuracy=acc_b, ece_contribution=contrib)

    # multiclass Brier score
    Y = np.zeros_like(probs)
    for i, t in enumerate(true):
        Y[i, CLASSES.index(t)] = 1.0
    brier = ((probs - Y) ** 2).sum(1).mean()

    print(f"  n               = {n}")
    print(f"  accuracy        = {correct.mean():.4f}")
    print(f"  mean confidence = {conf.mean():.4f}")
    print(f"  ECE ({n_bins} bins)  = {ece:.4f}")
    print(f"  Brier           = {brier:.4f}")

    print("\nPer-bin reliability table (confidence bins are (lower, upper]):")
    print(f"  {'bin':<16}{'n':>6}{'share':>8}{'mean_conf':>11}{'accuracy':>10}"
          f"{'conf-acc':>10}{'ECE_part':>10}")
    for r in table:
        label = f"({r['bin_lower']:.3f}, {r['bin_upper']:.3f}]"
        if r["n"]:
            print(f"  {label:<16}{r['n']:>6}{r['share']:>8.1%}{r['mean_confidence']:>11.4f}"
                  f"{r['accuracy']:>10.4f}{r['mean_confidence'] - r['accuracy']:>10.4f}"
                  f"{r['ece_contribution']:>10.4f}")
        else:
            print(f"  {label:<16}{0:>6}{0:>8.1%}{'-':>11}{'-':>10}{'-':>10}{'-':>10}")
    top = max(table, key=lambda r: r["n"])
    print(f"\nMost populated bin ({top['bin_lower']:.3f}, {top['bin_upper']:.3f}]: "
          f"{top['n']} of {n} predictions ({top['share']:.1%}), mean confidence "
          f"{top['mean_confidence']:.4f} versus accuracy {top['accuracy']:.4f}")

    out.mkdir(parents=True, exist_ok=True)
    bins_csv = out / "fig_calibration_bins.csv"
    with open(bins_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        for r in table:
            writer.writerow({k: ("" if v is None else (f"{v:.6f}" if isinstance(v, float) else v))
                             for k, v in r.items()})

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1.2, label="perfect calibration")
    ax.plot(xs, ys, "o-", color=C_CLS, lw=2, ms=6, label=f"model (ECE={ece:.3f})")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram (external, 3-seed ensemble)", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_calibration.pdf")
    fig.savefig(out / "fig_calibration.png", dpi=300)
    print(f"\nsaved {out / 'fig_calibration.pdf'} / .png and {bins_csv.name}")


if __name__ == "__main__":
    main()
