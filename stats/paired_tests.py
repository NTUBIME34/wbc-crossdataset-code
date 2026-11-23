#!/usr/bin/env python3
"""Paired comparisons of two three-seed models on the shared external images.

For every comparison "A minus B" (Supplementary Table S7 and its footnote):
  * Majority-vote ensemble of the per-seed seven-class labels: the most frequent label
    wins (Counter.most_common over the seed labels in the order the files are given);
    a three-way 1-1-1 tie takes the label of the FIRST file, so pass the seed-42 file
    first to apply the paper's rule ("a three-way tie takes the seed-42 label").
  * McNemar's exact test on per-image correctness (binomial test on the discordant
    counts b = A right / B wrong and c = A wrong / B right).
  * Paired image-level bootstrap of the macro-F1 difference: 5,000 replicates drawn
    with numpy.random.default_rng(12345), rng.choice(arange(n), size=n, replace=True)
    per replicate; the same draws for every comparison. 95% CI = percentiles
    [2.5, 97.5], 90% CI = [5, 95].
  * TOST at a +-0.02 macro-F1 margin, decided on the 90% CI: equivalence is
    established only if the 90% interval lies strictly inside (-0.02, +0.02).
    Reported as "equivalent", "not equivalent" (90% CI entirely beyond the margin)
    or "not established" (otherwise).
  * Tie-rule sensitivity (1-1-1 ties broken by each seed in turn, for both models),
    per-class F1 of the two vote ensembles, and, when both models carry prob_<class>
    columns, the tie-free probability ensemble (mean seven-class probabilities over
    seeds, argmax) with the same tests and the number of images on which the vote and
    probability ensembles of each model disagree.
  * Seed level: per-seed macro-F1, seed-paired differences (files paired by position)
    and a paired t-test over the seed pairs (scipy.stats.ttest_rel).
Macro-F1 is the mean F1 over the seven evaluation classes; a predicted label outside
them counts as an error.

Row order: images are joined by image id and ordered by "class folder/file name"
(the order of the full paths when all files share one dataset root, which is the
order of the archived S7 run). The order fixes which image each bootstrap index
picks, so another order gives slightly different intervals; --order name sorts by
file name only.

Inputs: one per-image file per seed and model: eval/evaluate_classifier.py
predictions.csv (--use-mappings; image_path, true_label, pred_label[, prob_<class>])
or path,true,pred dumps, .csv or .csv.gz.

Paper: Supplementary Table S7 (McNemar p, delta, 95% CI, 90% CI, TOST) and its
footnote (tie-rule range, probability-ensemble delta), the S7 interpretation
paragraph (per-class F1 differences, seed-paired differences, p ~ 0.15), main
Results 3.4 and Methods (64 vote/probability disagreements).

Example (proposed vs fine-tuned DinoBloom-B):
    python stats/paired_tests.py \\
        --model proposed sw_s42.csv sw_s62.csv sw_s82.csv \\
        --model dinobloom db_s42.csv db_s62.csv db_s82.csv \\
        --compare proposed dinobloom
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, ttest_rel

REPO_ROOT = Path(__file__).resolve().parents[1]
CLS7 = ["Basophil", "Blast_Group", "Eosinophil", "Lymphocyte",
        "Monocyte", "Neutrophil", "Promyelocyte"]
K = len(CLS7)
LAB2I = {c: i for i, c in enumerate(CLS7)}


# ---------------------------------------------------------------- input
def read_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def field_id(path: str) -> str:
    return Path(path).name.split("__")[0].split(".")[0]


def load_seed_file(path: str) -> dict[str, tuple]:
    """{image id: (true, pred, probability vector or None, sort key)}."""
    rows = read_rows(path)
    if not rows:
        sys.exit(f"[ERR] {path}: no rows")
    cols = set(rows[0])
    path_col = next((c for c in ("image_path", "path", "image_id") if c in cols), None)
    true_col = "true_label" if "true_label" in cols else ("true" if "true" in cols else None)
    pred_col = "pred_label" if "pred_label" in cols else ("pred" if "pred" in cols else None)
    if path_col is None or true_col is None or pred_col is None:
        sys.exit(f"[ERR] {path}: needs image_path/path, true_label/true and pred_label/pred")
    has_prob = all(f"prob_{c}" in cols for c in CLS7)
    out = {}
    for r in rows:
        p = Path(r[path_col])
        fid = field_id(r[path_col])
        if fid in out:
            sys.exit(f"[ERR] {path}: image id {fid} occurs twice")
        prob = np.array([float(r[f"prob_{c}"]) for c in CLS7]) if has_prob else None
        out[fid] = (r[true_col], r[pred_col], prob, f"{p.parent.name}/{p.name}", p.name)
    return out


def encode(labels: list[str]) -> np.ndarray:
    return np.array([LAB2I.get(x, K) for x in labels], dtype=np.int64)  # outside CLS7 -> bucket K


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
def confusion(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    m = K + 1
    return np.bincount(yt * m + yp, minlength=m * m).reshape(m, m)


def class_f1(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    cm = confusion(yt, yp)
    tp = np.diag(cm)[:K].astype(float)
    fp = cm[:, :K].sum(0) - tp
    fn = cm[:K, :].sum(1) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)


def macro_f1(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(class_f1(yt, yp).mean())


def vote(labels_by_seed: list[list[str]]) -> list[str]:
    """Counter.most_common(1) over the seed labels in file order (1-1-1 -> first file)."""
    return [Counter(v).most_common(1)[0][0] for v in zip(*labels_by_seed)]


def tost_words(lo90: float, hi90: float, margin: float) -> tuple[str, str]:
    if lo90 > -margin and hi90 < margin:
        return "equivalent", "EQUIVALENT (90% CI inside the margin)"
    if lo90 >= margin or hi90 <= -margin:
        return "not equivalent", "NOT equivalent; 90% CI entirely beyond the margin"
    if lo90 > 0 or hi90 < 0:
        return "not established", "NOT equivalent; 90% CI excludes 0 but crosses a margin bound"
    return "not established", "NOT equivalent; 90% CI includes 0 and crosses a margin bound"


def paired(yt: np.ndarray, pa: np.ndarray, pb: np.ndarray, nboot: int, rng_seed: int,
           margin: float) -> dict:
    n = len(yt)
    rng = np.random.default_rng(rng_seed)
    idx = np.arange(n)
    diffs = np.empty(nboot)
    for k in range(nboot):
        s = rng.choice(idx, size=n, replace=True)
        diffs[k] = macro_f1(yt[s], pa[s]) - macro_f1(yt[s], pb[s])
    fa, fb = macro_f1(yt, pa), macro_f1(yt, pb)
    l95, h95 = np.percentile(diffs, [2.5, 97.5])
    l90, h90 = np.percentile(diffs, [5, 95])
    ca, cb = pa == yt, pb == yt
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    mcn = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    p_boot = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    words, verdict = tost_words(l90, h90, margin)
    return {"f1_A": fa, "f1_B": fb, "delta": fa - fb,
            "ci95": [float(l95), float(h95)], "ci90": [float(l90), float(h90)],
            "boot_mean": float(diffs.mean()), "boot_p_two_sided": float(p_boot),
            "mcnemar_b_A_right_B_wrong": b, "mcnemar_c_A_wrong_B_right": c,
            "mcnemar_exact_p": float(mcn), "tost": words, "tost_verdict": verdict,
            "tost_95ci_rule_equivalent": bool(l95 > -margin and h95 < margin)}


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.2f}"


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="McNemar, paired bootstrap (95%/90% CI), TOST on the 90% CI, tie-rule "
                    "sensitivity and seed-level paired t-test for three-seed models.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="Inputs:" + __doc__.split("Inputs:", 1)[1])
    ap.add_argument("--model", nargs="+", action="append", required=True, metavar=("NAME", "FILE"),
                    help="model name and one prediction file per seed; the first file breaks "
                         "1-1-1 ties (pass seed 42 first)")
    ap.add_argument("--compare", nargs=2, action="append", metavar=("A", "B"),
                    help="comparison A minus B (repeatable; default: the two models given)")
    ap.add_argument("--nboot", type=int, default=5000, help="bootstrap replicates (paper: 5,000)")
    ap.add_argument("--rng-seed", type=int, default=12345, help="bootstrap RNG seed (paper: 12345)")
    ap.add_argument("--margin", type=float, default=0.02, help="TOST margin (paper: 0.02)")
    ap.add_argument("--order", choices=["path", "name"], default="path",
                    help="image order for the bootstrap: class folder/file name (default, the "
                         "archived run) or file name")
    ap.add_argument("--no-tie-sensitivity", action="store_true",
                    help="skip the tie-rule sensitivity bootstraps")
    ap.add_argument("--out-dir", default=None, help="default $OUT_ROOT/stats")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()

    models: dict[str, list[dict]] = {}
    for spec in args.model:
        if len(spec) < 2:
            ap.error("--model needs a name and at least one file")
        models[spec[0]] = [load_seed_file(f) for f in spec[1:]]
    comps = args.compare or ([list(models)[:2]] if len(models) == 2 else None)
    if not comps:
        ap.error("give --compare A B (more than two models were loaded)")
    for a, b in comps:
        for m in (a, b):
            if m not in models:
                ap.error(f"unknown model in --compare: {m}")

    # ------------------------------------------------ shared image set and truth
    id_sets = [set(d) for m in models.values() for d in m]
    shared_ids = set.intersection(*id_sets)
    union_ids = set.union(*id_sets)
    true, key = {}, {}
    for m in models.values():
        for d in m:
            for fid in shared_ids:
                t = d[fid][0]
                if true.setdefault(fid, t) != t:
                    sys.exit(f"[ERR] true label of {fid} differs between files")
                key.setdefault(fid, d[fid][3] if args.order == "path" else d[fid][4])
    shared = sorted(shared_ids, key=lambda f: key[f])
    n = len(shared)
    yt = encode([true[f] for f in shared])
    if (yt == K).any():
        sys.exit("[ERR] true labels outside the seven evaluation classes")
    res: dict = {"meta": {"n_shared": n, "n_union": len(union_ids),
                          "identical_image_sets": len(union_ids) == n, "nboot": args.nboot,
                          "rng": f"numpy.random.default_rng({args.rng_seed}); rng.choice(arange(n), "
                                 "size=n, replace=True) per replicate",
                          "ci": "percentile: 95% = [2.5, 97.5], 90% = [5, 95]",
                          "tost": f"equivalent iff the 90% CI lies inside +-{args.margin}",
                          "vote": "Counter.most_common over seed labels in file order; 1-1-1 -> first file",
                          "order": args.order,
                          "support": {c: int((yt == i).sum()) for i, c in enumerate(CLS7)}},
                 "models": {}, "comparisons": {}}

    # ------------------------------------------------ per-model ensembles
    ens_vote, ens_prob, alt = {}, {}, {}
    for name, seeds in models.items():
        P = [[d[f][1] for f in shared] for d in seeds]
        v = encode(vote(P))
        codes = np.stack([encode(p) for p in P], 1)
        info: dict = {"n_seeds": len(seeds),
                      "per_seed_macro_f1": [macro_f1(yt, codes[:, j]) for j in range(len(seeds))]}
        if len(seeds) == 3:
            maj = np.where(codes[:, 1] == codes[:, 2], codes[:, 1], codes[:, 0])
            if not np.array_equal(maj, v):
                sys.exit(f"[ERR] vote rule check failed for {name}")
            tie = ((codes[:, 0] != codes[:, 1]) & (codes[:, 0] != codes[:, 2])
                   & (codes[:, 1] != codes[:, 2]))
            unan = (codes[:, 0] == codes[:, 1]) & (codes[:, 1] == codes[:, 2])
            alt[name] = [np.where(tie, codes[:, j], maj) for j in range(3)]
            info.update({"n_1_1_1": int(tie.sum()), "n_2_1": int((~tie & ~unan).sum()),
                         "n_unanimous": int(unan.sum()),
                         "vote_macro_f1_by_tie_seed": [macro_f1(yt, x) for x in alt[name]]})
        ens_vote[name] = v
        info["vote_macro_f1"] = macro_f1(yt, v)
        info["vote_class_f1"] = dict(zip(CLS7, class_f1(yt, v).tolist()))
        if all(d[shared[0]][2] is not None for d in seeds):
            avg = np.mean([np.stack([d[f][2] for f in shared]) for d in seeds], axis=0)
            ens_prob[name] = avg.argmax(1).astype(np.int64)
            info["prob_macro_f1"] = macro_f1(yt, ens_prob[name])
            info["vote_vs_prob_disagreements"] = int((ens_prob[name] != v).sum())
        res["models"][name] = info

    # ------------------------------------------------ comparisons
    for a, b in comps:
        cid = f"{a} - {b}"
        r = {"vote": paired(yt, ens_vote[a], ens_vote[b], args.nboot, args.rng_seed, args.margin)}
        r["vote_class_f1_delta"] = {c: res["models"][a]["vote_class_f1"][c]
                                    - res["models"][b]["vote_class_f1"][c] for c in CLS7}
        if not args.no_tie_sensitivity and a in alt and b in alt:
            r["vote_by_tie_seed"] = [paired(yt, alt[a][j], alt[b][j], args.nboot, args.rng_seed,
                                            args.margin) for j in range(3)]
        if a in ens_prob and b in ens_prob:
            r["prob"] = paired(yt, ens_prob[a], ens_prob[b], args.nboot, args.rng_seed, args.margin)
        sa, sb = res["models"][a]["per_seed_macro_f1"], res["models"][b]["per_seed_macro_f1"]
        if len(sa) == len(sb) and len(sa) > 1:
            d = [x - y for x, y in zip(sa, sb)]
            t = ttest_rel(sa, sb)
            r["seed_level"] = {"per_seed_diff": d, "mean_diff": statistics.fmean(d),
                               "n_positive": sum(x > 0 for x in d),
                               "paired_t": float(t.statistic), "paired_t_p": float(t.pvalue)}
        res["comparisons"][cid] = r

    # ------------------------------------------------ printout
    L: list[str] = []
    w = L.append
    m = res["meta"]
    w(f"shared images n = {n} (identical image sets: {m['identical_image_sets']}); "
      f"{args.nboot} paired replicates, {m['rng']}; order: {args.order}")
    w(f"TOST margin +-{args.margin}, decided on the 90% CI; vote: 1-1-1 ties -> first file")
    w("")
    for name, info in res["models"].items():
        extra = ""
        if "n_1_1_1" in info:
            extra = (f"; 1-1-1 ties {info['n_1_1_1']}, vote macro-F1 by tie seed "
                     + "/".join(f"{x:.4f}" for x in info["vote_macro_f1_by_tie_seed"]))
        if "prob_macro_f1" in info:
            extra += (f"; probability ensemble {info['prob_macro_f1']:.4f}, vote/probability "
                      f"disagreements {info['vote_vs_prob_disagreements']} of {n}")
        w(f"{name}: per-seed macro-F1 " + "/".join(f"{x:.4f}" for x in info["per_seed_macro_f1"])
          + f"; vote ensemble {info['vote_macro_f1']:.4f}" + extra)
    w("")
    w(f"{'Comparison':<34}{'McNemar p':>10}{'delta':>9}  {'95% CI':<18}{'90% CI':<18}TOST")
    for cid, r in res["comparisons"].items():
        v = r["vote"]
        w(f"{cid:<34}{fmt_p(v['mcnemar_exact_p']):>10}{v['delta']:>+9.3f}  "
          f"[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]  [{v['ci90'][0]:+.3f},{v['ci90'][1]:+.3f}]  "
          f"{v['tost']}")
    for cid, r in res["comparisons"].items():
        w("")
        w(f"-- {cid}")
        for kind in ("vote", "prob"):
            if kind not in r:
                continue
            v = r[kind]
            w(f"   [{kind}] A={v['f1_A']:.4f} B={v['f1_B']:.4f} delta={v['delta']:+.4f} "
              f"95% CI [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] 90% CI [{v['ci90'][0]:+.4f},"
              f"{v['ci90'][1]:+.4f}] bootstrap p={v['boot_p_two_sided']:.4f}")
            w(f"          McNemar b={v['mcnemar_b_A_right_B_wrong']} c={v['mcnemar_c_A_wrong_B_right']}"
              f" exact p={v['mcnemar_exact_p']:.3g}; TOST: {v['tost_verdict']}")
        for j, q in enumerate(r.get("vote_by_tie_seed", [])):
            w(f"   [vote, 1-1-1 ties -> file {j + 1} for both] delta={q['delta']:+.4f} "
              f"95% CI [{q['ci95'][0]:+.4f},{q['ci95'][1]:+.4f}] 90% CI [{q['ci90'][0]:+.4f},"
              f"{q['ci90'][1]:+.4f}]  {q['tost']}")
        w("   vote-ensemble class F1 difference: "
          + ", ".join(f"{c} {x:+.3f}" for c, x in r["vote_class_f1_delta"].items()))
        if "seed_level" in r:
            s = r["seed_level"]
            w(f"   seed level: differences " + ", ".join(f"{x:+.3f}" for x in s["per_seed_diff"])
              + f"; mean {s['mean_diff']:+.4f}; A higher on {s['n_positive']}/{len(s['per_seed_diff'])}"
              f"; paired t = {s['paired_t']:.3f}, p = {s['paired_t_p']:.3f}")
    text = "\n".join(L)
    print(text)
    if args.no_write:
        return
    d = out_dir(args.out_dir)
    (d / "paired_tests.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (d / "paired_tests.txt").write_text(text + "\n", encoding="utf-8")
    print(f"[saved] {d / 'paired_tests.json'}")


if __name__ == "__main__":
    main()
