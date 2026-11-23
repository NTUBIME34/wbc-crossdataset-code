#!/usr/bin/env python3
"""Figure 3: row-normalized confusion matrices (internal vs external), 3-seed ensemble
of the proposed color-augmentation Swin model on the 7-class mapped label space.
Uses only the model's own predictions (no dataset images) -> no licensing issue.

The ensemble prediction of an image is the argmax of the seven-class probabilities
summed over the seeds (seeds 42/62/82, joined on image_path). The script also prints
the numbers quoted in Results section 3.8: per-class recall on both sets, the largest
external confusions, and the breakdown of the external errors by class pair.

Inputs: the per-seed predictions.csv files that eval/evaluate_classifier.py writes
(--use-mappings --tta hflip) for the proposed model, one directory per evaluation set:

    <internal-dir>/seed_42/predictions.csv   internal test split   (n = 12,803)
    <external-dir>/seed_42/predictions.csv   WBCBench2026 Phase-1  (n = 8,061)
    ... and the same for seeds 62 and 82

(use --internal-pattern / --external-pattern for another layout). Columns used:
image_path, true_label and prob_<class> for the seven classes.

Output (in --out-dir): fig_confusion.pdf and .png
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["Basophil", "Blast_Group", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil", "Promyelocyte"]
SHORT = ["BASO", "BLAST", "EOS", "LYM", "MONO", "NEU", "PROMY"]  # canonical (match manuscript tables)
SEEDS = [42, 62, 82]
DEFAULT_PATTERN = "seed_{seed}/predictions.csv"
GRANULOCYTES = ["Basophil", "Eosinophil", "Neutrophil"]  # mature granulocyte classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Row-normalized confusion matrices of the three-seed ensemble, internal "
                    "vs external (Fig. 3), plus the error breakdown of Results section 3.8.",
    )
    parser.add_argument("--internal-dir", default=os.environ.get("INTERNAL_PRED_DIR"),
                        help="Evaluator outputs on the internal test split, one per seed "
                             "(required unless INTERNAL_PRED_DIR is set).")
    parser.add_argument("--external-dir", default=os.environ.get("PRED_DIR"),
                        help="Evaluator outputs on WBCBench2026 Phase-1, one per seed "
                             "(required unless PRED_DIR is set).")
    parser.add_argument("--internal-pattern", default=DEFAULT_PATTERN,
                        help="Per-seed predictions file relative to --internal-dir; {seed} is "
                             "replaced by the seed (default: %(default)s).")
    parser.add_argument("--external-pattern", default=DEFAULT_PATTERN,
                        help="Per-seed predictions file relative to --external-dir "
                             "(default: %(default)s).")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Seeds to ensemble (default: 42 62 82).")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: $OUT_ROOT/figures, else ./outputs/figures).")
    args = parser.parse_args()
    if not args.internal_dir:
        parser.error("--internal-dir is required (or set INTERNAL_PRED_DIR)")
    if not args.external_dir:
        parser.error("--external-dir is required (or set PRED_DIR)")
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


def seed_files(pred_dir, pattern, seeds):
    paths = {}
    for s in seeds:
        p = Path(pred_dir).expanduser().resolve() / pattern.format(seed=s)
        if not p.is_file():
            sys.exit(f"[ERROR] predictions for seed {s} not found: {p}")
        paths[s] = p
    return paths


def ensemble_conf(paths):
    acc = {}
    seen = {}
    for s, p in paths.items():
        seen[s] = set()
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            need = ["image_path", "true_label"] + [f"prob_{c}" for c in CLASSES]
            missing = [c for c in need if c not in (reader.fieldnames or [])]
            if missing:
                sys.exit(f"[ERROR] {p} lacks columns: {', '.join(missing)}")
            for r in reader:
                k = r["image_path"]
                if k in seen[s]:
                    sys.exit(f"[ERROR] {p}: image_path occurs twice: {k}")
                seen[s].add(k)
                probs = {c: float(r[f"prob_{c}"]) for c in CLASSES}
                if k not in acc:
                    acc[k] = [r["true_label"], {c: 0.0 for c in CLASSES}]
                elif acc[k][0] != r["true_label"]:
                    sys.exit(f"[ERROR] true_label differs between seeds for {k}")
                for c in CLASSES:
                    acc[k][1][c] += probs[c]
    # every seed must cover the same images, or the summed probabilities are not comparable
    ref = next(iter(seen.values()))
    for s, keys in seen.items():
        if keys != ref:
            sys.exit(f"[ERROR] seed {s} covers a different image set ({len(keys)} vs {len(ref)} "
                     f"images, {len(keys ^ ref)} differ). Evaluate every seed on the same set.")
    cm = np.zeros((7, 7))
    n = 0
    for k, (t, pr) in sorted(acc.items()):
        if t not in CLASSES:
            continue
        pred = max(CLASSES, key=lambda c: pr[c])
        cm[CLASSES.index(t), CLASSES.index(pred)] += 1
        n += 1
    row = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    return cm, cmn, n, row.flatten().astype(int)


def panel(ax, cmn, cm, title, n):
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(SHORT, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("True", fontsize=9)
    ax.set_title(f"{title}  (n={n:,})", fontsize=10, weight="bold")
    for i in range(7):
        for j in range(7):
            v = cmn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "#333")
    return im


def print_counts(cm, title):
    print(f"\n{title} (rows = true, columns = predicted):")
    print("        " + "".join(f"{s:>7}" for s in SHORT))
    for i, s in enumerate(SHORT):
        print(f"  {s:<6}" + "".join(f"{int(v):>7d}" for v in cm[i]))


def error_breakdown(cm, title):
    """Errors by class pair, granulocyte share and per-class false positives / negatives."""
    n = int(cm.sum())
    err = n - int(np.trace(cm))
    print(f"\n{title}: {err} errors of {n:,} predictions ({err / n:.1%})")
    pairs = [(int(cm[i, j] + cm[j, i]), i, j) for i in range(7) for j in range(i + 1, 7)
             if cm[i, j] + cm[j, i] > 0]
    print("  largest confusions by class pair (counts in each direction, share of errors):")
    for k, i, j in sorted(pairs, key=lambda x: (-x[0], x[1], x[2]))[:6]:
        print(f"    {SHORT[i]}->{SHORT[j]} {int(cm[i, j])}, {SHORT[j]}->{SHORT[i]} {int(cm[j, i])}"
              f"  (total {k}, {k / err:.0%})")
    g = [CLASSES.index(c) for c in GRANULOCYTES]
    gran = int(sum(cm[i, j] for i in g for j in g if i != j))
    print(f"  confusions among mature granulocytes (BASO/EOS/NEU): {gran} ({gran / err:.0%})")
    print("  per class: false positives, false negatives, prevalence")
    for i, s in enumerate(SHORT):
        fp = int(cm[:, i].sum() - cm[i, i])
        fn = int(cm[i, :].sum() - cm[i, i])
        print(f"    {s:<6} FP {fp:>4d}  FN {fn:>4d}  prevalence {cm[i].sum() / n:.1%}")


def main() -> None:
    args = parse_args()
    out = out_dir_from(args.out_dir)
    internal = seed_files(args.internal_dir, args.internal_pattern, args.seeds)
    external = seed_files(args.external_dir, args.external_pattern, args.seeds)

    ci, cin, ni, _ = ensemble_conf(internal)
    ce, cen, ne, _ = ensemble_conf(external)

    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    panel(axes[0], cin, ci, "Internal test", ni)
    im = panel(axes[1], cen, ce, "External evaluation\n(WBCBench2026 Phase-1)", ne)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Row-normalized fraction (recall)", fontsize=8)
    fig.suptitle("Confusion matrices, proposed Swin model (3-seed ensemble, 7-class)", fontsize=11, weight="bold")
    fig.savefig(out / "fig_confusion.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(out / "fig_confusion.png", dpi=200, bbox_inches="tight")

    print(f"internal n = {ni:,}   external n = {ne:,}   seeds = {', '.join(map(str, args.seeds))}")
    # report diagonal recalls for narration
    print("\nclass     internal_recall  external_recall")
    for i, c in enumerate(CLASSES):
        print(f"{c:12} {cin[i,i]:.3f}          {cen[i,i]:.3f}")
    # biggest external off-diagonal confusions
    print("\nTop external off-diagonal confusions (true -> pred, fraction):")
    offs = [(CLASSES[i], CLASSES[j], cen[i, j]) for i in range(7) for j in range(7) if i != j]
    for t, p, v in sorted(offs, key=lambda x: -x[2])[:6]:
        print(f"  {t} -> {p}: {v:.2f}")
    print_counts(ci, "Internal counts")
    print_counts(ce, "External counts")
    error_breakdown(ce, "External errors")
    print(f"\nsaved {out / 'fig_confusion.pdf'} / .png")


if __name__ == "__main__":
    main()
