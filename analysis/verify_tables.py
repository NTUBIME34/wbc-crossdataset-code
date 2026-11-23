#!/usr/bin/env python
"""Recompute the manuscript's reported numbers from analysis/evidence/ and compare them,
at the precision the paper prints, with the values printed in the paper.

    python analysis/verify_tables.py                          # public evidence only
    python analysis/verify_tables.py --private-evidence DIR   # adds the per-image checks

Default mode needs no WBCBench2026 per-image file. It reads the internal confusion
matrices, the leave-one-dataset-out (LODO) per-image files of the public training
sources, the LODO foundation-model grid, the two hash lists, the count-only WBCBench
aggregates in evidence/wbcbench_confusion/ (confusion matrices, ensemble matrices,
McNemar discordant counts, calibration bins, threshold sweeps, field counts) and the
summary tables in analysis/tables/.

--private-evidence DIR points at a folder laid out like analysis/evidence/ (either DIR
itself or DIR/analysis/evidence/) that holds WBCBench2026 per-image predictions, for
example ones you regenerated with the released weights (see analysis/README.md). It adds
the checks that need per-image data: it rebuilds every count file from the per-image
predictions and compares it with the shipped copy, reruns the image-level bootstraps
(Supplementary S7 and S9), the per-image change counts of the unified re-evaluation and,
when the two hematologists' verdict files are present (they are not distributed), the
Supplementary S13 verdict counts and printed rows, and that the reviewed cells, as a multiset
of (type, benchmark label, model prediction), are the model's 205 panel cells.

Every expected value below is a value printed in the paper (manuscript and supplement as
of 2026-11-21), written as the paper prints it; a few counts appear in the paper only as
words ("four", "fifteen") or as the size of a table. A number passes when the recomputed
value, formatted to the same number of decimals, equals the printed string; integer counts
must match exactly; a value the paper quotes from a cited 4-decimal table passes when the two
rounding intervals overlap (analysis/README.md). Seed summaries are mean and population SD
(ddof=0) over seeds, as in Methods.
The script ends with a PASS/FAIL/SKIPPED summary and exits non-zero on any FAIL.

Author option: --check-tex MAIN SUPP confirms that every expected value still occurs in
the LaTeX sources of the paper as a whole value (not part of a longer, signed or
comma-grouped number), including the values of checks that were skipped for want of
private data. A value can also occur elsewhere in the sources, so this catches a value
that disappears from the paper but not every edit; the rows of Tables S7, S9 and S13,
whose cells recur elsewhere, are therefore also looked up as whole printed rows.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import re
import sys
import warnings
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

# The checks never call scikit-learn, but keep its "y_pred contains classes not in
# y_true" warning (raised whenever a no-detection sentinel is scored) out of the log in
# case a future check does.
warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

ANALYSIS = Path(__file__).resolve().parent
EV = ANALYSIS / "evidence"
TABLES = ANALYSIS / "tables"
WBC = EV / "wbcbench_confusion"

CLS7 = ["Basophil", "Blast_Group", "Eosinophil", "Lymphocyte",
        "Monocyte", "Neutrophil", "Promyelocyte"]
SHORT = {"Basophil": "BASO", "Blast_Group": "BLAST", "Eosinophil": "EOS", "Lymphocyte": "LYM",
         "Monocyte": "MONO", "Neutrophil": "NEU", "Promyelocyte": "PROMY"}
ND = "__no_detection__"
NINE = ["Basophil", "Eosinophil", "Erythroblast", "Lymphoblast", "Lymphocyte",
        "Monocyte", "Myeloblast", "Neutrophil", "Promyelocyte"]
SEEDS = (42, 62, 82)
CNN_SEEDS = (42, 43, 44)
LODO_SEEDS = (42, 52, 62)
LODO_TAGS = {"AML-LMU": "AMLLMU", "PBC": "PBC", "Raabin-WBC": "RaabinWBC",
             "HRLS": "HRLS", "BMCMHF": "BMCMHF"}
# WBCBench2026 native folder -> seven-class evaluation label ("" = no counterpart; those
# fields are excluded from scoring), the mapping the evaluator applies with --use-mappings.
GT7 = {"Segmented_neutrophil": "Neutrophil", "Band_neutrophil": "Neutrophil", "Lymphocyte": "Lymphocyte",
       "Variant_lymphocyte": "Lymphocyte", "Prolymphocyte": "Lymphocyte", "Monocyte": "Monocyte",
       "Eosinophil": "Eosinophil", "Basophil": "Basophil", "Promyelocyte": "Promyelocyte",
       "Blast": "Blast_Group", "Myelocyte": "", "Metamyelocyte": "", "Plasma_cell": ""}


# =========================================================================== metrics
class Scores:
    """Metrics of a square confusion matrix (rows true, columns predicted).

    The definitions are scikit-learn's with zero_division=0: F1 of a class with no
    true, predicted or correct sample is 0, balanced accuracy averages recall over the
    classes that occur in y_true, and a prediction outside `labels` (for example the
    no-detection sentinel) is an error for its true class and a false positive for no
    listed class."""

    def __init__(self, cm: pd.DataFrame):
        labels = list(dict.fromkeys(list(cm.index) + list(cm.columns)))
        self.cm = cm.reindex(index=labels, columns=labels, fill_value=0).astype(np.int64)
        self.labels = labels
        m = self.cm.values
        self.tp = pd.Series(np.diag(m), index=labels)
        self.support = pd.Series(m.sum(1), index=labels)
        self.predicted = pd.Series(m.sum(0), index=labels)

    def _get(self, s: pd.Series, c: str) -> int:
        return int(s.get(c, 0))

    def f1(self, c: str) -> float:
        tp, sup, pr = self._get(self.tp, c), self._get(self.support, c), self._get(self.predicted, c)
        den = 2 * tp + (pr - tp) + (sup - tp)
        return 2 * tp / den if den else 0.0

    def recall(self, c: str) -> float:
        sup = self._get(self.support, c)
        return self._get(self.tp, c) / sup if sup else float("nan")

    def macro_f1(self, labels=CLS7) -> float:
        return float(np.mean([self.f1(c) for c in labels]))

    def weighted_f1(self, labels=CLS7) -> float:
        w = np.array([self._get(self.support, c) for c in labels], dtype=float)
        f = np.array([self.f1(c) for c in labels])
        return float((f * w).sum() / w.sum()) if w.sum() else 0.0

    def accuracy(self) -> float:
        return float(self.tp.sum() / self.cm.values.sum())

    def balanced_accuracy(self) -> float:
        present = [c for c in self.labels if self._get(self.support, c) > 0]
        return float(np.mean([self.recall(c) for c in present]))

    def min_recall(self, labels=CLS7) -> float:
        return float(min(self.recall(c) for c in labels if self._get(self.support, c) > 0))

    def n(self) -> int:
        return int(self.cm.values.sum())


def confusion(y_true, y_pred) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    ct = pd.crosstab(pd.Categorical(list(y_true), labels), pd.Categorical(list(y_pred), labels),
                     dropna=False)
    ct.index, ct.columns = labels, labels
    return ct


def ms(values) -> tuple[float, float]:
    """Seed mean and population SD (ddof=0)."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std())


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p (binomial, p = 0.5), as scipy.stats.binomtest."""
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# =========================================================================== registry
# Every WBCBench2026 evaluation run behind a reported number. The key is group/run_id;
# public counts for it are in wbcbench_confusion/run_confusion.csv.gz and its per-image
# file (withheld) sits at <private evidence>/<group>/<run_id>.csv.gz. "coverage" names
# the detection file whose missed-but-labelled fields count as errors for that run.
class Run:
    def __init__(self, group, run_id, coverage=None, per_image=True):
        self.group, self.run_id, self.coverage, self.per_image = group, run_id, coverage, per_image
        self.key = f"{group}/{run_id}"


def _registry() -> list[Run]:
    R = []
    add = lambda *a, **k: R.append(Run(*a, **k))  # noqa: E731
    for s in SEEDS:
        add("backbone_comparison", f"swin_proposed_seed{s}")
    for t in ("vgg16bn", "resnet50", "efficientnet_b0"):
        for s in CNN_SEEDS:
            add("backbone_comparison", f"{t}_seed{s}")             # superseded corpus
    for t in ("vgg16bn", "resnet50", "efficientnet_b0"):
        for s in CNN_SEEDS:
            add("backbone_common_manifold", f"{t}_seed{s}_copy")
    for tag in ("swinbpaco", "rn50bpaco_adamw", "rn50bpaco_sgd", "dinobloom"):
        for s in SEEDS:
            add("matched_recipe", f"{tag}_seed{s}")
    for arm in ("adamw", "sgd"):
        for s in SEEDS:
            add("recipe_transfer", f"rn50_color_{arm}_seed{s}")
    for tag in ("swin_cwce_default", "rn50_cwce_default"):
        for s in SEEDS:
            add("cwce_default", f"{tag}_seed{s}")
    for s in SEEDS:
        add("aggressive_recipe", f"swin_mixup_bpaco_crt_seed{s}")
    for pol in POLICIES:
        for s in SEEDS:
            add("augmentation_policy", f"swin_{pol}_seed{s}", per_image=False)
    for s in SEEDS:
        add("tta_ablation", f"swin_proposed_noflip_seed{s}", per_image=False)
    add("frozen_probe", "dinobloom_probe", per_image=False)
    for s in SEEDS:
        for px in (224, 256, 384):
            add("resolution", f"swin_proposed_seed{s}_img{px}")
    for s in SEEDS:
        add("localization_ablation", f"nocrop_swin_seed{s}")
    for grp in ("localization_ablation", "localization_ablation_preunified"):
        for r in "BCDE":
            for s in SEEDS:
                add(grp, f"route{r}_seed{s}", coverage="phase1_seed62_detections" if r == "B" else None)
    for grp in ("phase2_degradation", "phase2_degradation_preunified"):
        for s in SEEDS:
            add(grp, f"phase2_pipeline_seed{s}", coverage=f"phase2_seed{s}_detections")
    return R


POLICIES = ("stain_only", "geometry_only", "conservative", "blur_noise", "stain_blur_noise")
RUNS = _registry()

# (name, rule, member run keys). "probability": mean of the seven-class probabilities over
# seeds, then argmax. "vote_tieNN": majority of the per-seed labels, a three-way tie taking
# the seed-NN label.
_pre = lambda tag: [f"backbone_comparison/{tag}_seed{s}" for s in SEEDS]  # noqa: E731
_mr = lambda tag: [f"matched_recipe/{tag}_seed{s}" for s in SEEDS]  # noqa: E731
_cm = lambda tag: [f"backbone_common_manifold/{tag}_seed{s}_copy" for s in CNN_SEEDS]  # noqa: E731
ENSEMBLES = [
    ("swin_proposed", "probability", _pre("swin_proposed")),
    ("vgg16bn", "probability", _cm("vgg16bn")),
    ("resnet50", "probability", _cm("resnet50")),
    ("efficientnet_b0", "probability", _cm("efficientnet_b0")),
    ("swin_cwce_default", "probability", [f"cwce_default/swin_cwce_default_seed{s}" for s in SEEDS]),
    ("swin_mixup_bpaco_crt", "probability", [f"aggressive_recipe/swin_mixup_bpaco_crt_seed{s}" for s in SEEDS]),
    ("dinobloom_ft", "probability", _mr("dinobloom")),
]
VOTE_MODELS = [("swin_bpaco", _mr("swinbpaco")), ("rn50_bpaco_adamw", _mr("rn50bpaco_adamw")),
               ("rn50_bpaco_sgd", _mr("rn50bpaco_sgd")), ("swin_proposed", _pre("swin_proposed")),
               ("dinobloom_ft", _mr("dinobloom"))]
for _name, _members in VOTE_MODELS:
    for _j in SEEDS:
        ENSEMBLES.append((_name, f"vote_tie{_j}", _members))
# Supplementary S7 paired comparisons (model A minus model B)
COMPARISONS = [("bpaco_shared_adamw", "swin_bpaco", "rn50_bpaco_adamw"),
               ("bpaco_adamw_vs_sgd", "swin_bpaco", "rn50_bpaco_sgd"),
               ("proposed_vs_dinobloom", "swin_proposed", "dinobloom_ft")]
MCNEMAR = [(c, "vote_tie42", a, b) for c, a, b in COMPARISONS] + \
          [("proposed_vs_dinobloom", "probability", "swin_proposed", "dinobloom_ft")]
REEVAL_PAIRS = [(f"localization_ablation/route{r}_seed{s}", f"localization_ablation_preunified/route{r}_seed{s}")
                for r in "BCDE" for s in SEEDS] + \
               [(f"phase2_degradation/phase2_pipeline_seed{s}", f"phase2_degradation_preunified/phase2_pipeline_seed{s}")
                for s in SEEDS]
N_BINS = 15
THRESHOLDS = [i / 100 for i in range(1, 100)]
THRESHOLD_CLASSES = ("Blast_Group", "Promyelocyte")


# =========================================================================== count builders
# Used by --private-evidence (rebuild and compare) and by the authors' build step, so the
# shipped counts and the reader's rebuild come from the same code.
def _stem(image_id: str) -> str:
    m = re.match(r"(\d{8})", str(image_id).split("/")[-1])
    return m.group(1) if m else str(image_id)


class Private:
    """Per-image WBCBench evidence in the analysis/evidence/<group>/<run>.csv.gz layout."""

    def __init__(self, root: Path):
        root = Path(root).resolve()
        if (root / "analysis" / "evidence").is_dir():
            self.top, self.ev = root, root / "analysis" / "evidence"
        elif root.name == "evidence" and root.parent.name == "analysis":
            self.top, self.ev = root.parent.parent, root
        else:
            self.top, self.ev = root, root
        if not self.ev.is_dir():
            raise SystemExit(f"--private-evidence: no evidence folder at {root}")
        if (Path(__file__).resolve().parent.parent in [self.ev, *self.ev.parents]):
            raise SystemExit("--private-evidence must point outside the repository")
        self._cache: dict[str, pd.DataFrame] = {}

    def resolve(self, rel: str) -> Path:
        """A file of the layout, accepting .csv for .csv.gz."""
        cands = [rel] + ([rel[3:]] if rel.startswith("../") else [])
        for c in cands:
            for q in (c, c[:-3] if c.endswith(".csv.gz") else None):
                if q and (self.ev / q).exists():
                    return self.ev / q
        return self.ev / rel

    def path(self, key: str) -> Path:
        return self.resolve(f"{key}.csv.gz")

    def has(self, key: str) -> bool:
        return self.path(key).exists()

    def load(self, key: str) -> pd.DataFrame:
        if key not in self._cache:
            self._cache[key] = _normalise_private(pd.read_csv(self.path(key)))
        return self._cache[key]

    def has_predictions(self) -> bool:
        d = self.predictions_dir()
        return all((d / f"proposed_swin_seed{s}_wbcbench_phase1.csv").exists() for s in SEEDS)

    def predictions_dir(self) -> Path:
        for d in (self.top / "analysis" / "predictions", self.ev / "predictions", self.ev.parent / "predictions"):
            if d.is_dir():
                return d
        return self.top / "analysis" / "predictions"


def _basename(p) -> str:
    return str(p).replace("\\", "/").split("/")[-1]


def _normalise_private(d: pd.DataFrame) -> pd.DataFrame:
    """Accept the archive layout and the raw outputs readers regenerate: the evaluator's
    predictions.csv (image_path, true_label, pred_label, prob_*), its --dump-preds file
    (path, true, pred) and the detector's detections.csv (source_path, status, yolo_conf;
    run on class folders, so the native class is the parent folder)."""
    if "image_id" in d.columns:
        return d
    if {"source_path", "status"} <= set(d.columns):
        native = d.source_path.map(lambda p: str(p).replace("\\", "/").split("/")[-2])
        return pd.DataFrame({"image_id": d.source_path.map(_basename), "det_status": d.status,
                             "true_label": native.map(lambda c: GT7.get(c) or np.nan),
                             "yolo_conf": d.get("yolo_conf")})
    if {"path", "true", "pred"} <= set(d.columns):
        d = d.rename(columns={"path": "image_path", "true": "true_label", "pred": "pred_label"})
    if "image_path" in d.columns:
        d = d.copy()
        d.insert(0, "image_id", d.image_path.map(_basename))
    return d


def missed_labelled(priv: Private, det_run: str) -> pd.Series:
    d = priv.load(f"detections/{det_run}")
    return d[(d.det_status != "detected") & d.true_label.notna()].true_label


def run_matrix(priv: Private, run: Run) -> pd.DataFrame:
    d = priv.load(run.key)
    yt, yp = list(d.true_label), list(d.pred_label)
    if run.coverage:
        miss = list(missed_labelled(priv, run.coverage))
        yt += miss
        yp += [ND] * len(miss)
    ct = pd.crosstab(pd.Categorical(yt, CLS7), pd.Categorical(yp, CLS7 + [ND]), dropna=False)
    ct.index, ct.columns = CLS7, CLS7 + [ND]
    if len(yt) != int(ct.values.sum()):
        raise SystemExit(f"{run.key}: labels outside the seven-class space")
    return ct.astype(np.int64)


def _aligned(priv: Private, keys: list[str]) -> list[pd.DataFrame]:
    ds = [priv.load(k).sort_values("image_id").reset_index(drop=True) for k in keys]
    for k, d in zip(keys[1:], ds[1:]):
        if not (d.image_id.values == ds[0].image_id.values).all() or \
                not (d.true_label.values == ds[0].true_label.values).all():
            raise SystemExit(f"ensemble members disagree on images or labels: {keys[0]} vs {k}")
    return ds


def ensemble_labels(priv: Private, rule: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(true codes, predicted codes, image ids) of a three-seed ensemble, ids sorted."""
    ds = _aligned(priv, keys)
    y = np.array([CLS7.index(t) for t in ds[0].true_label])
    if rule == "probability":
        P = np.mean([d[[f"prob_{c}" for c in CLS7]].values for d in ds], axis=0)
        p = P.argmax(1)
    else:
        j = SEEDS.index(int(rule.replace("vote_tie", "")))
        V = np.stack([np.array([CLS7.index(x) for x in d.pred_label]) for d in ds], 1)
        maj = np.where(V[:, 1] == V[:, 2], V[:, 1], V[:, 0])
        tie = (V[:, 0] != V[:, 1]) & (V[:, 0] != V[:, 2]) & (V[:, 1] != V[:, 2])
        p = np.where(tie, V[:, j], maj)
    return y, p, list(ds[0].image_id)


def _cm7(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.bincount(y * 7 + p, minlength=49).reshape(7, 7)


def full_precision_ensemble(priv: Private) -> tuple[np.ndarray, np.ndarray]:
    """Proposed model's ensemble probabilities at full precision (calibration, thresholds),
    joined on image_path and summed seed 42 + 62 + 82 before dividing, as the figure and
    threshold scripts do."""
    d = priv.predictions_dir()
    frames = [pd.read_csv(d / f"proposed_swin_seed{s}_wbcbench_phase1.csv").set_index("image_path").sort_index()
              for s in SEEDS]
    for f in frames[1:]:
        if not f.index.equals(frames[0].index):
            raise SystemExit("full-precision prediction files cover different images")
    acc = np.zeros((len(frames[0]), 7))
    for f in frames:
        acc = acc + f[[f"prob_{c}" for c in CLS7]].values
    return acc / len(frames), np.array([CLS7.index(t) for t in frames[0].true_label])


def calibration_table(name: str, P: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    conf, pred = P.max(1), P.argmax(1)
    correct = pred == y
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    brier = ((P - Y) ** 2).sum(1)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    rows = []
    for i in range(N_BINS):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        rows.append(dict(ensemble=name, bin=i, lower=round(edges[i], 6), upper=round(edges[i + 1], 6),
                         n=int(m.sum()), n_correct=int(correct[m].sum()),
                         sum_confidence=float(conf[m].sum()), sum_brier=float(brier[m].sum())))
    return pd.DataFrame(rows)


def threshold_table(name: str, P: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in THRESHOLD_CLASSES:
        k = CLS7.index(c)
        pos_true = y == k
        for t in THRESHOLDS:
            pos = P[:, k] >= t
            rows.append(dict(ensemble=name, cls=c, threshold=t, tp=int((pos & pos_true).sum()),
                             fp=int((pos & ~pos_true).sum()), fn=int((~pos & pos_true).sum()),
                             tn=int((~pos & ~pos_true).sum())))
    return pd.DataFrame(rows)


def rebuild_counts(priv: Private) -> dict[str, pd.DataFrame]:
    """Every count file of evidence/wbcbench_confusion/ that per-image data can rebuild."""
    out = {}
    rows = []
    for run in RUNS:
        if not run.per_image or not priv.has(run.key) or \
                (run.coverage and not priv.has(f"detections/{run.coverage}")):
            continue
        ct = run_matrix(priv, run)
        for c in CLS7:
            rows.append(dict(run=run.key, true_label=c, **{k: int(ct.loc[c, k]) for k in CLS7 + [ND]}))
    if rows:
        out["run_confusion"] = pd.DataFrame(rows)

    rows, codes = [], {}
    for name, rule, keys in ENSEMBLES:
        if not all(priv.has(k) for k in keys):
            continue
        y, p, ids = ensemble_labels(priv, rule, keys)
        codes[(name, rule)] = (y, p, ids)
        cm = _cm7(y, p)
        for i, c in enumerate(CLS7):
            rows.append(dict(ensemble=name, rule=rule, members="|".join(keys), true_label=c,
                             **{k: int(cm[i, j]) for j, k in enumerate(CLS7)}))
    if rows:
        out["ensemble_confusion"] = pd.DataFrame(rows)

    rows = []
    for comp, rule, a, b in MCNEMAR:
        if (a, rule) not in codes or (b, rule) not in codes:
            continue
        ya, pa, ia = codes[(a, rule)]
        yb, pb, ib = codes[(b, rule)]
        if ia != ib or not (ya == yb).all():
            raise SystemExit(f"McNemar {comp}: the two ensembles cover different images")
        ca, cb = pa == ya, pb == yb
        rows.append(dict(comparison=comp, rule=rule, model_a=a, model_b=b, n=len(ya),
                         both_correct=int((ca & cb).sum()), a_correct_b_wrong=int((ca & ~cb).sum()),
                         a_wrong_b_correct=int((~ca & cb).sum()), both_wrong=int((~ca & ~cb).sum())))
    if rows:
        out["mcnemar_discordant"] = pd.DataFrame(rows)

    rows = []
    for name in ("swin_proposed", "dinobloom_ft"):
        if (name, "probability") not in codes or (name, "vote_tie42") not in codes:
            continue
        y, pp, ids = codes[(name, "probability")]
        _, pv, ids2 = codes[(name, "vote_tie42")]
        keys = dict(ENSEMBLES_BY_NAME)[(name, "vote_tie42")]
        V = np.stack([np.array([CLS7.index(x) for x in d.pred_label]) for d in _aligned(priv, keys)], 1)
        tie = (V[:, 0] != V[:, 1]) & (V[:, 0] != V[:, 2]) & (V[:, 1] != V[:, 2])
        rows.append(dict(ensemble=name, n=len(y), n_vote_vs_probability_disagree=int((pp != pv).sum()),
                         n_three_way_ties=int(tie.sum())))
    if rows:
        out["ensemble_agreement"] = pd.DataFrame(rows)

    if priv.has_predictions():
        P, y = full_precision_ensemble(priv)
        out["calibration_bins"] = calibration_table("swin_proposed", P, y)
        out["threshold_sweep"] = threshold_table("swin_proposed", P, y)

    rows = []
    lab1 = priv.load("wbcbench_labels/phase1_raw_labels") if priv.has("wbcbench_labels/phase1_raw_labels") else None
    lab2 = priv.load("wbcbench_labels/phase2_raw_labels") if priv.has("wbcbench_labels/phase2_raw_labels") else None
    for split, lab, det in (("phase1", lab1, "phase1_seed62_detections"), ("phase2", lab2, "phase2_seed42_detections")):
        if lab is None or not priv.has(f"detections/{det}"):
            continue
        dd = priv.load(f"detections/{det}").assign(stem=lambda x: x.image_id.map(_stem))
        detected = set(dd.stem[dd.det_status == "detected"])
        lab = lab.assign(stem=lab.image_id.map(_stem))
        for raw, g in lab.groupby("raw_label", sort=True):
            rows.append(dict(split=split, raw_label=raw, label7=g.label7.iloc[0], n_fields=len(g),
                             n_precropped=float(g.in_precropped.sum()) if "in_precropped" in g else np.nan,
                             n_detected=int(g.stem.isin(detected).sum())))
    if rows:
        out["field_counts"] = pd.DataFrame(rows)

    rows = []
    for new, old in REEVAL_PAIRS:
        if not (priv.has(new) and priv.has(old)):
            continue
        a = priv.load(new).assign(stem=lambda x: x.image_id.map(_stem)).set_index("stem").pred_label
        b = priv.load(old).assign(stem=lambda x: x.image_id.map(_stem)).set_index("stem").pred_label
        common = a.index.intersection(b.index)
        rows.append(dict(run=new, earlier_run=old, n_compared=len(common),
                         n_changed=int((a.loc[common] != b.loc[common]).sum())))
    if rows:
        out["reeval_changes"] = pd.DataFrame(rows)
    return out


ENSEMBLES_BY_NAME = [((n, r), k) for n, r, k in ENSEMBLES]


# =========================================================================== checker
class Checker:
    def __init__(self, quiet: bool = False):
        self.results: list[tuple[str, str, str, str, str]] = []
        self.quiet = quiet
        self.current = ""
        self.expected: list[tuple[str, str, str]] = []      # (label, printed, paper file)

    def section(self, title: str) -> None:
        self.current = title
        if not self.quiet:
            print(f"\n{title}\n" + "-" * min(len(title), 100))

    def _emit(self, status: str, label: str, printed: str, got: str) -> None:
        self.results.append((self.current, label, status, printed, got))
        if status == "FAIL" or not self.quiet:
            print(f"  {status:<7} {label:<66} printed {printed:<16} recomputed {got}")

    @staticmethod
    def _fmt(printed: str, got: float) -> str:
        p = printed.strip().replace("\u2212", "-").replace(",", "")
        if p.endswith("%"):
            p = p[:-1]
        d = len(p.split(".")[1]) if "." in p else 0
        s = f"{got:+.{d}f}" if p[:1] in "+-" else f"{got:.{d}f}"
        if re.fullmatch(r"[+-]?0(\.0*)?", s):
            s = s.lstrip("+-")
            if p[:1] == "+":
                s = "+" + s
        return s

    def num(self, label: str, printed: str, got: float, paper: str = "") -> None:
        """Compare at the printed number of decimals ('35.6%' expects a percentage)."""
        self.expected.append((label, printed, paper))
        if got is None or (isinstance(got, float) and math.isnan(got)):
            self._emit("FAIL", label, printed, "nan")
            return
        want = self._fmt(printed, float(printed.strip().replace("\u2212", "-").replace(",", "").rstrip("%")))
        ok = self._fmt(printed, float(got)) == want
        self._emit("PASS" if ok else "FAIL", label, printed, f"{got:.6g}" if abs(got) < 1e5 else f"{got:,.0f}")

    def from_table(self, label: str, printed: str, table_value: float, paper: str = "", table_decimals: int = 4) -> None:
        """The printed value is a correct rounding of a value the paper takes from a cited table
        that gives it to `table_decimals` places (the interval of the table value and the
        interval of the printed value overlap)."""
        self.expected.append((label, printed, paper))
        p = printed.strip().replace("\u2212", "-").replace(",", "").rstrip("%")
        d = len(p.split(".")[1]) if "." in p else 0
        ok = abs(float(table_value) - float(p)) <= 0.5 * 10 ** -d + 0.5 * 10 ** -table_decimals + 1e-12
        self._emit("PASS" if ok else "FAIL", label, printed, f"{float(table_value):.{table_decimals}f} (table)")

    def count(self, label: str, printed, got: int, paper: str = "") -> None:
        want = int(str(printed).replace(",", ""))
        self.expected.append((label, str(printed), paper))
        self._emit("PASS" if int(got) == want else "FAIL", label, str(printed), f"{int(got):,}")

    def text(self, label: str, printed: str, got: str, paper: str = "") -> None:
        self.expected.append((label, printed, paper))
        self._emit("PASS" if str(got) == printed else "FAIL", label, printed, str(got))

    def row(self, label: str, printed: str, paper: str = "") -> None:
        """A printed table row whose cells are checked one by one elsewhere: registered whole for
        --check-tex only (no result line), so that an edit of one cell is caught even where that
        cell's value also occurs elsewhere in the paper."""
        self.expected.append((label, printed, paper))

    def true(self, label: str, ok: bool, detail: str = "") -> None:
        self._emit("PASS" if ok else "FAIL", label, "holds", detail or ("yes" if ok else "no"))

    def close(self, label: str, expected: float, got: float, tol: float) -> None:
        ok = abs(expected - got) <= tol
        self._emit("PASS" if ok else "FAIL", label, f"{expected}", f"{got:.6g} (tol {tol})")

    def skip(self, label: str, reason: str, printed: str | None = None, paper: str = "") -> None:
        if printed is not None:                 # the printed value still goes to --check-tex
            self.expected.append((label, printed, paper))
        self.results.append((self.current, label, "SKIPPED", "", reason))
        if not self.quiet:
            print(f"  {'SKIPPED':<7} {label:<66} {reason}")

    def summary(self) -> int:
        c = Counter(r[2] for r in self.results)
        print("\n" + "=" * 78)
        print(f"Checks: {len(self.results)}   PASS {c['PASS']}   FAIL {c['FAIL']}   SKIPPED {c['SKIPPED']}")
        if c["SKIPPED"]:
            reasons = Counter(r[4] for r in self.results if r[2] == "SKIPPED")
            for reason, n in reasons.most_common():
                print(f"  skipped {n:>4}: {reason}")
        if c["FAIL"]:
            print(f"\n{c['FAIL']} FAILED:")
            for sec, label, st, printed, got in self.results:
                if st == "FAIL":
                    print(f"  [{sec[:40]}] {label}: printed {printed}, recomputed {got}")
            return 1
        print("All non-skipped checks pass.")
        return 0


# =========================================================================== public loaders
def read_counts(name: str) -> pd.DataFrame:
    return pd.read_csv(WBC / f"{name}.csv.gz")


class Public:
    def __init__(self):
        rc = read_counts("run_confusion")
        self.run = {k: g.set_index("true_label")[CLS7 + [ND]].reindex(CLS7) for k, g in rc.groupby("run")}
        ec = read_counts("ensemble_confusion")
        self.ens = {(n, r): g.set_index("true_label")[CLS7].reindex(CLS7) for (n, r), g in ec.groupby(["ensemble", "rule"])}
        self.mcn = read_counts("mcnemar_discordant")
        self.cal = read_counts("calibration_bins")
        self.thr = read_counts("threshold_sweep")
        self.fields = read_counts("field_counts")
        self.agree = read_counts("ensemble_agreement")
        self.reeval = read_counts("reeval_changes")

    def scores(self, key: str, detected_only: bool = False) -> Scores:
        cm = self.run[key]
        if detected_only:
            cm = cm[CLS7]
        return Scores(cm)

    def seeds(self, group: str, fmt: str, seeds=SEEDS, detected_only=False) -> list[Scores]:
        return [self.scores(f"{group}/{fmt.format(s=s)}", detected_only) for s in seeds]


def internal(run: str) -> pd.DataFrame:
    return pd.read_csv(EV / "backbone_common_manifold_internal" / f"{run}.csv.gz", index_col=0)


def to7(cm: pd.DataFrame) -> pd.DataFrame:
    m = {c: ("Blast_Group" if c in ("Myeloblast", "Lymphoblast", "Erythroblast") else c) for c in cm.index}
    return cm.groupby(m).sum().T.groupby(m).sum().T


def lodo(group: str, tag: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(EV / group / f"heldout_{tag}_seed{seed}.csv.gz")


# =========================================================================== default checks
INTERNAL_ROWS = OrderedDict([
    ("Swin-B (colour aug + CW-CE; proposed)", ([f"swin_proposed_seed{s}" for s in SEEDS], "0.956")),
    ("VGG16-BN (CW-CE)", ([f"vgg16bn_seed{s}_copy" for s in CNN_SEEDS], "0.948")),
    ("EfficientNet-B0 (CW-CE)", ([f"efficientnet_b0_seed{s}_copy" for s in CNN_SEEDS], "0.944")),
    ("ResNet50 (CW-CE)", ([f"resnet50_seed{s}_copy" for s in CNN_SEEDS], "0.942")),
    ("Swin-B (BPaCo; AdamW)", ([f"swin_bpaco_seed{s}" for s in SEEDS], "0.950")),
    ("ResNet50 (BPaCo; AdamW, shared protocol)", ([f"resnet50_bpaco_adamw_seed{s}" for s in SEEDS], "0.933")),
    ("ResNet50 (BPaCo; SGD)", ([f"resnet50_bpaco_sgd_seed{s}" for s in SEEDS], "0.933")),
    ("DinoBloom-B (frozen; linear probe)", (["dinobloom_probe"], "0.938")),
    ("DinoBloom-B (colour aug; fine-tuned)", ([f"dinobloom_ft_seed{s}" for s in SEEDS], "0.957")),
])
FIRST_BLOCK = ["Swin-B (colour aug + CW-CE; proposed)", "VGG16-BN (CW-CE)",
               "EfficientNet-B0 (CW-CE)", "ResNet50 (CW-CE)"]

# Table 3 external columns: key template, seeds, printed (mF1, sd, wF1, wsd, bal, bsd, BLAST, PROMY)
TABLE2_EXT = OrderedDict([
    ("Swin-B (colour aug + CW-CE; proposed)", ("backbone_comparison/swin_proposed_seed{s}", SEEDS,
                                               ("0.863", "0.028", "0.947", "0.004", "0.887", "0.005", "0.793", "0.601"))),
    ("VGG16-BN (CW-CE)", ("backbone_common_manifold/vgg16bn_seed{s}_copy", CNN_SEEDS,
                          ("0.813", "0.032", "0.927", "0.010", "0.828", "0.026", "0.676", "0.446"))),
    ("EfficientNet-B0 (CW-CE)", ("backbone_common_manifold/efficientnet_b0_seed{s}_copy", CNN_SEEDS,
                                 ("0.796", "0.030", "0.911", "0.013", "0.862", "0.021", "0.654", "0.491"))),
    ("ResNet50 (CW-CE)", ("backbone_common_manifold/resnet50_seed{s}_copy", CNN_SEEDS,
                          ("0.718", "0.010", "0.886", "0.007", "0.842", "0.008", "0.686", "0.177"))),
    ("Swin-B (BPaCo; AdamW)", ("matched_recipe/swinbpaco_seed{s}", SEEDS,
                               ("0.819", "0.018", "0.921", "0.007", "0.864", "0.009", "0.636", "0.539"))),
    ("ResNet50 (BPaCo; AdamW, shared protocol)", ("matched_recipe/rn50bpaco_adamw_seed{s}", SEEDS,
                                                  ("0.762", "0.017", "0.904", "0.003", "0.833", "0.004", "0.516", "0.407"))),
    ("ResNet50 (BPaCo; SGD)", ("matched_recipe/rn50bpaco_sgd_seed{s}", SEEDS,
                               ("0.815", "0.004", "0.921", "0.005", "0.887", "0.003", "0.709", "0.428"))),
    ("DinoBloom-B (colour aug; fine-tuned)", ("matched_recipe/dinobloom_seed{s}", SEEDS,
                                              ("0.822", "0.047", "0.929", "0.009", "0.863", "0.017", "0.659", "0.596"))),
])


def seed_values(pub: Public, tmpl: str, seeds, fn) -> list[float]:
    return [fn(pub.scores(tmpl.format(s=s))) for s in seeds]


def check_internal(chk: Checker, pub: Public | None) -> dict:
    chk.section("Table 3 internal column, Supp S4, S7 and S17: internal confusion matrices (n = 12,803)")
    vals, sds = {}, {}
    for name, (runs, printed) in INTERNAL_ROWS.items():
        cms = [internal(r) for r in runs]
        for r, cm in zip(runs, cms):
            chk.count(f"{name}: held-out cells ({r})", "12,803", int(cm.values.sum()), "ms")
            chk.true(f"{name}: nine-class matrix ({r})", list(cm.index) == NINE and list(cm.columns) == NINE)
        v = [Scores(cm).macro_f1(NINE) for cm in cms]
        vals[name], sds[name] = ms(v)
        chk.num(f"{name}: internal macro-F1", printed, vals[name], "ms")
    mx = max(sds[n] for n in FIRST_BLOCK)
    chk.true("Section 3.3: class-weighted-CE rows, across-seed internal sd <= 0.001 (at the printed precision)",
             float(f"{mx:.3f}") <= 0.001, f"largest sd {mx:.5f}")
    int_mean = dict(vals)
    lo, hi = min(int_mean[n] for n in FIRST_BLOCK), max(int_mean[n] for n in FIRST_BLOCK)
    chk.num("Table 3b class-weighted-CE rows: internal range low", "0.942", lo, "ms")
    chk.num("Table 3b class-weighted-CE rows: internal range high", "0.956", hi, "ms")
    chk.num("Section 3.3: class-weighted-CE rows, internal spread", "0.014", hi - lo, "ms")
    if pub is not None:
        ext = {n: np.mean(seed_values(pub, TABLE2_EXT[n][0], TABLE2_EXT[n][1], lambda s: s.macro_f1()))
               for n in FIRST_BLOCK}
        chk.true("Table 3b class-weighted-CE rows: internal and external columns rank the four models identically",
                 sorted(FIRST_BLOCK, key=lambda k: -int_mean[k]) == sorted(FIRST_BLOCK, key=lambda k: -ext[k]))
        chk.num("Section 3.3: class-weighted-CE rows, external spread", "0.145", max(ext.values()) - min(ext.values()), "ms")
    chk.true("DinoBloom-B fine-tuned has the highest internal macro-F1 of the table",
             max(int_mean, key=int_mean.get) == "DinoBloom-B (colour aug; fine-tuned)")
    chk.num("ResNet50: internal drop under BPaCo with SGD", "0.009",
            int_mean["ResNet50 (CW-CE)"] - int_mean["ResNet50 (BPaCo; SGD)"], "ms")

    # Supp S4: blast sub-types (row-wise confusion with the other two sub-types)
    cms = [internal(f"swin_proposed_seed{s}") for s in SEEDS]
    for sub, n, f1, sd, pct in (("Myeloblast", "582", "0.974", "0.001", "0.2"),
                                ("Lymphoblast", "257", "0.992", "0.002", "1.4"),
                                ("Erythroblast", "1,506", "0.924", "0.001", "0.0")):
        others = [o for o in ("Myeloblast", "Lymphoblast", "Erythroblast") if o != sub]
        chk.count(f"S4 {sub}: internal support", n, int(cms[0].loc[sub].sum()), "supp")
        m, s = ms([Scores(c).f1(sub) for c in cms])
        chk.num(f"S4 {sub}: internal F1", f1, m, "supp")
        chk.num(f"S4 {sub}: internal F1 sd", sd, s, "supp")
        chk.num(f"S4 {sub}: confused with the other two blast types (%)", pct + "%",
                float(np.mean([100 * c.loc[sub, others].sum() / c.loc[sub].sum() for c in cms])), "supp")

    # Supp S17: internal macro-F1 with the matrices mapped to the seven evaluation classes
    seven = {}
    for name, label, printed in (("Swin-B (colour aug + CW-CE; proposed)", "proposed Swin", "0.952"),
                                 ("VGG16-BN (CW-CE)", "VGG16-BN", "0.944"),
                                 ("EfficientNet-B0 (CW-CE)", "EfficientNet-B0", "0.939"),
                                 ("ResNet50 (CW-CE)", "ResNet50", "0.937"),
                                 ("Swin-B (BPaCo; AdamW)", "Swin-B, BPaCo", "0.945"),
                                 ("ResNet50 (BPaCo; AdamW, shared protocol)", "ResNet50, BPaCo with AdamW", "0.930"),
                                 ("ResNet50 (BPaCo; SGD)", "ResNet50, BPaCo with SGD", "0.931"),
                                 ("DinoBloom-B (colour aug; fine-tuned)", "fine-tuned DinoBloom-B", "0.953")):
        seven[name] = float(np.mean([Scores(to7(internal(r))).macro_f1(CLS7) for r in INTERNAL_ROWS[name][0]]))
        chk.num(f"S17 internal seven-class macro-F1: {label}", printed, seven[name], "supp")
    chk.true("S17: ordering among the four class-weighted-CE reference models is the same in the seven-class space",
             sorted(FIRST_BLOCK, key=lambda k: -seven[k]) == sorted(FIRST_BLOCK, key=lambda k: -int_mean[k]))
    chk.true("S17: DinoBloom-B fine-tuned stays the highest internal score in the seven-class space",
             max(seven, key=seven.get) == "DinoBloom-B (colour aug; fine-tuned)")
    chk.true("S17: ResNet50 BPaCo-SGD below ResNet50 CW-CE in the seven-class space (0.931 versus 0.937)",
             seven["ResNet50 (BPaCo; SGD)"] < seven["ResNet50 (CW-CE)"])

    # Supp S7: internal scores of the other ResNet50 / Swin-B pairs
    for runs, printed, label in (([f"rn50_color_adamw_seed{s}" for s in SEEDS], "0.931", "ResNet50 colour recipe (AdamW)"),
                                 ([f"rn50_color_sgd_seed{s}" for s in SEEDS], "0.939", "ResNet50 colour recipe (SGD)"),
                                 ([f"swin_cwce_default_seed{s}" for s in SEEDS], "0.959", "Swin-B CW-CE, default augmentation"),
                                 ([f"rn50_cwce_default_seed{s}" for s in SEEDS], "0.941", "ResNet50 CW-CE, default augmentation")):
        cms = [internal(r) for r in runs]
        for r, cm in zip(runs, cms):
            chk.count(f"S7 {label}: held-out cells ({r})", "12,803", int(cm.values.sum()), "supp")
        chk.num(f"S7 {label}: internal macro-F1", printed, float(np.mean([Scores(c).macro_f1(NINE) for c in cms])), "supp")
    return int_mean


def check_internal_ensemble(chk: Checker) -> None:
    chk.section("Section 3.8 and Supp S11: internal three-seed ensemble (seven classes, n = 12,803)")
    cm = pd.read_csv(EV / "internal_ensemble" / "swin_proposed_confusion7.csv.gz", index_col=0).reindex(index=CLS7, columns=CLS7)
    s = Scores(cm)
    chk.count("internal ensemble: evaluated cells (Fig. 3)", "12,803", s.n(), "ms")
    rec = [s.recall(c) for c in CLS7]
    chk.num("internal per-class recall: lowest (0.90--0.98)", "0.90", min(rec), "ms")
    chk.num("internal per-class recall: highest (0.90--0.98)", "0.98", max(rec), "ms")
    chk.num("internal Blast group recall (0.95 to 0.84)", "0.95", s.recall("Blast_Group"), "ms")
    chk.num("internal Promyelocyte recall (0.96 to 0.79)", "0.96", s.recall("Promyelocyte"), "ms")
    sw = pd.read_csv(EV / "internal_ensemble" / "swin_proposed_threshold_sweep.csv.gz")
    for c, t_print, f_print, label in (("Blast_Group", "0.500", "0.949", "BLAST"), ("Promyelocyte", "0.480", "0.962", "PROMY")):
        g = sw[sw.cls == c].copy()
        g["f1"] = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
        best = g.loc[g.f1.idxmax()]          # first maximum in threshold order, as the script
        chk.num(f"S11 {label}: internal-tuned threshold t*", t_print, float(best.threshold), "supp")
        chk.num(f"S11 {label}: internal F1 at t*", f_print, float(best.f1), "supp")


def check_lodo(chk: Checker) -> dict:
    chk.section("Table 1 pools, Table 4, Supp Note 1, S14, S15: leave-one-dataset-out (public sources)")
    G = "lodo_proposed"
    data = {h: [lodo(G, t, s) for s in LODO_SEEDS] for h, t in LODO_TAGS.items()}
    sc = {h: [Scores(confusion(d.true_label, d.pred_label)) for d in data[h]] for h in data}
    pools = {"AML-LMU": "18,128", "BMCMHF": "73,096", "PBC": "11,845", "Raabin-WBC": "10,154", "HRLS": "15,405"}
    for h, n in pools.items():
        chk.count(f"Table 1 pool size: {h}", n, len(data[h][0]), "ms")
    chk.count("Table 1 caption: the five pools sum to", "128,628", sum(len(data[h][0]) for h in data), "ms")

    T4 = {"AML-LMU": ("8", "0.782", "0.590", "0.008", "0.718"), "PBC": ("6", "0.919", "0.909", "0.015", "0.888"),
          "Raabin-WBC": ("5", "0.844", "0.742", "0.015", "0.716"), "HRLS": ("7", "0.778", "0.763", "0.008", "0.808"),
          "BMCMHF": ("7", "0.702", "0.606", "0.030", "0.643")}
    present = {}
    for h, (k, acc, m, sd, bal) in T4.items():
        labs = sorted(set(data[h][0].true_label))
        chk.count(f"Table 4 {h}: classes present", k, len(labs), "ms")
        chk.num(f"Table 4 {h}: accuracy", acc, np.mean([s.accuracy() for s in sc[h]]), "ms")
        pm = [s.macro_f1(sorted(set(d.true_label))) for s, d in zip(sc[h], data[h])]
        present[h] = pm
        chk.num(f"Table 4 {h}: present-class macro-F1 mean", m, ms(pm)[0], "ms")
        chk.num(f"Table 4 {h}: present-class macro-F1 sd", sd, ms(pm)[1], "ms")
        chk.num(f"Table 4 {h}: balanced accuracy (present)", bal, np.mean([s.balanced_accuracy() for s in sc[h]]), "ms")
    fold_means = {h: ms(v)[0] for h, v in present.items()}
    chk.num("Table 4 Mean row: present-class macro-F1", "0.722", np.mean(list(fold_means.values())), "ms")
    chk.num("Table 4 Mean row: balanced accuracy", "0.754",
            np.mean([np.mean([s.balanced_accuracy() for s in sc[h]]) for h in sc]), "ms")
    chk.num("Section 3.5: present-class macro-F1 range low (AML-LMU)", "0.59", fold_means["AML-LMU"], "ms")
    chk.num("Section 3.5: present-class macro-F1 range high (PBC)", "0.91", fold_means["PBC"], "ms")
    chk.num("Section 3.5: present-class mean", "0.72", np.mean(list(fold_means.values())), "ms")
    chk.true("Section 3.5: AML-LMU lowest and PBC highest fold",
             min(fold_means, key=fold_means.get) == "AML-LMU" and max(fold_means, key=fold_means.get) == "PBC")
    for s, printed in zip(LODO_SEEDS, ("0.639", "0.612", "0.566")):
        chk.num(f"Note 1 BMCMHF per-seed present-class macro-F1 (seed {s})", printed,
                present["BMCMHF"][LODO_SEEDS.index(s)], "supp")
    chk.true("Note 1: BMCMHF has the widest seed spread", max(T4, key=lambda h: ms(present[h])[1]) == "BMCMHF")

    # Note 1: full nine-class label space (fixed) and the union convention
    fixed = {h: [s.macro_f1(NINE) for s in sc[h]] for h in sc}
    union = {h: [s.macro_f1(sorted(set(d.true_label) | set(d.pred_label))) for s, d in zip(sc[h], data[h])] for h in sc}
    for h, m, sd in (("AML-LMU", "0.525", "0.007"), ("PBC", "0.606", "0.010"), ("Raabin-WBC", "0.412", "0.008"),
                     ("HRLS", "0.593", "0.006"), ("BMCMHF", "0.471", "0.023")):
        chk.num(f"Note 1 fixed nine-class macro-F1 {h}: mean", m, ms(fixed[h])[0], "supp")
        chk.num(f"Note 1 fixed nine-class macro-F1 {h}: sd", sd, ms(fixed[h])[1], "supp")
    fixed_mean = np.mean([ms(v)[0] for v in fixed.values()])
    chk.num("Note 1 fixed nine-class: across-dataset mean", "0.522", fixed_mean, "supp")
    chk.num("Note 1: present-class mean minus fixed nine-class mean", "0.20",
            np.mean(list(fold_means.values())) - fixed_mean, "supp")
    chk.num("Note 1 union convention: PBC mean", "0.631", ms(union["PBC"])[0], "supp")
    chk.num("Note 1 union convention: PBC sd", "0.027", ms(union["PBC"])[1], "supp")
    chk.num("Note 1 union convention: across-dataset mean", "0.526", np.mean([ms(v)[0] for v in union.values()]), "supp")
    chk.count("Note 1: folds unchanged by the union convention", "4",
              sum(np.allclose(fixed[h], union[h]) for h in sc), "supp")
    pbc52 = data["PBC"][LODO_SEEDS.index(52)]
    chk.true("Note 1: on PBC seed 52 the model never predicts Lymphoblast (union of eight labels)",
             "Lymphoblast" not in set(pbc52.pred_label) and len(set(pbc52.true_label) | set(pbc52.pred_label)) == 8)

    # Section 3.5 / Note 1: HRLS Lymphoblast F1 in every seed of both arms; mixup arm
    for grp, arm in (("lodo_proposed", "proposed"), ("lodo", "mixup/CutMix")):
        for s in LODO_SEEDS:
            d = lodo(grp, "HRLS", s)
            chk.num(f"HRLS Lymphoblast F1, {arm} arm, seed {s}", "0.000",
                    Scores(confusion(d.true_label, d.pred_label)).f1("Lymphoblast"), "ms")
    mix = {h: np.mean([Scores(confusion(d.true_label, d.pred_label)).macro_f1(sorted(set(d.true_label)))
                       for d in (lodo("lodo", t, s) for s in LODO_SEEDS)]) for h, t in LODO_TAGS.items()}
    chk.num("Note 1: mixup/CutMix arm mean present-class macro-F1", "0.685", np.mean(list(mix.values())), "supp")
    chk.count("Note 1: folds on which the mixup/CutMix arm is lower", "4",
              sum(mix[h] < fold_means[h] for h in mix), "supp")

    # Supp S14: every printed cell
    S14 = {
        "AML-LMU": [("Myeloblast", "3,268", "0.025", "0.048"), ("Basophil", "79", "0.274", "0.376"), ("Monocyte", "1,779", "0.765", "0.779"), ("Promyelocyte", "70", "0.857", "0.467"), ("Erythroblast", "78", "0.927", "0.416"), ("Lymphocyte", "3,947", "0.947", "0.710"), ("Eosinophil", "424", "0.952", "0.947"), ("Neutrophil", "8,483", "0.994", "0.982")],
        "PBC": [("Lymphocyte", "1,214", "0.666", "0.724"), ("Erythroblast", "1,547", "0.781", "0.876"), ("Monocyte", "1,420", "0.942", "0.916"), ("Basophil", "1,218", "0.968", "0.976"), ("Eosinophil", "3,117", "0.983", "0.987"), ("Neutrophil", "3,329", "0.987", "0.975")],
        "Raabin-WBC": [("Monocyte", "561", "0.285", "0.431"), ("Eosinophil", "744", "0.767", "0.730"), ("Lymphocyte", "2,420", "0.778", "0.855"), ("Basophil", "212", "0.818", "0.764"), ("Neutrophil", "6,217", "0.931", "0.932")],
        "HRLS": [("Lymphoblast", "2,557", "0.000", "0.000"), ("Myeloblast", "2,534", "0.763", "0.548"), ("Lymphocyte", "3,037", "0.943", "0.920"), ("Monocyte", "2,039", "0.965", "0.893"), ("Eosinophil", "1,017", "0.987", "0.993"), ("Basophil", "1,023", "0.999", "0.993"), ("Neutrophil", "3,198", "0.999", "0.993")],
        "BMCMHF": [("Erythroblast", "13,470", "0.272", "0.421"), ("Promyelocyte", "5,918", "0.469", "0.626"), ("Basophil", "355", "0.550", "0.353"), ("Monocyte", "3,049", "0.671", "0.379"), ("Neutrophil", "26,536", "0.825", "0.864"), ("Eosinophil", "3,484", "0.832", "0.838"), ("Lymphocyte", "20,284", "0.880", "0.758")],
    }
    for h, rows in S14.items():
        labs = sorted(set().union(*[set(d.true_label) for d in data[h]]))
        chk.count(f"S14 {h}: classes listed", str(len(rows)), len(labs), "supp")
        recs = {c: np.mean([s.recall(c) for s in sc[h]]) for c in labs}
        f1s = {c: np.mean([s.f1(c) for s in sc[h]]) for c in labs}
        for c, n, r, f in rows:
            chk.count(f"S14 {h}/{c}: n", n, int((data[h][0].true_label == c).sum()), "supp")
            chk.num(f"S14 {h}/{c}: recall", r, recs[c], "supp")
            chk.num(f"S14 {h}/{c}: F1", f, f1s[c], "supp")
        chk.true(f"S14 {h}: classes ordered by recall", [c for c, *_ in rows] == sorted(labs, key=lambda c: recs[c]))
        # the block's F1 column averages back to Table 4 (mean of three-decimal values, so within 0.001)
        chk.close(f"S14 {h}: mean of printed F1 column vs Table 4 present macro-F1", float(T4[h][2]),
                  float(np.mean([float(f) for *_, f in rows])), 0.001)
    for h, c, dest, pct in (("AML-LMU", "Myeloblast", "Lymphocyte", "78.5%"), ("HRLS", "Lymphoblast", "Myeloblast", "95.2%")):
        sub = pd.concat(data[h])
        sub = sub[sub.true_label == c]
        chk.num(f"S14: {h} {c} predicted {dest} across seeds", pct, 100 * (sub.pred_label == dest).mean(), "supp")

    # Supp S15: per-dataset class composition (the held-out set of each fold)
    S15 = {"Neutrophil": ("8,483", "3,329", "6,217", "3,198", "26,536", "47,763"),
           "Lymphocyte": ("3,947", "1,214", "2,420", "3,037", "20,284", "30,902"),
           "Monocyte": ("1,779", "1,420", "561", "2,039", "3,049", "8,848"),
           "Eosinophil": ("424", "3,117", "744", "1,017", "3,484", "8,786"),
           "Basophil": ("79", "1,218", "212", "1,023", "355", "2,887"),
           "Erythroblast": ("78", "1,547", "0", "0", "13,470", "15,095"),
           "Promyelocyte": ("70", "0", "0", "0", "5,918", "5,988"),
           "Myeloblast": ("3,268", "0", "0", "2,534", "0", "5,802"),
           "Lymphoblast": ("0", "0", "0", "2,557", "0", "2,557")}
    order = ["AML-LMU", "PBC", "Raabin-WBC", "HRLS", "BMCMHF"]
    vc = {h: data[h][0].true_label.value_counts() for h in order}
    for c, row in S15.items():
        for h, printed in zip(order, row[:5]):
            chk.count(f"S15 {c} / {h}", printed, int(vc[h].get(c, 0)), "supp")
        chk.count(f"S15 {c} / pooled", row[5], sum(int(vc[h].get(c, 0)) for h in order), "supp")
    for h, printed in zip(order, ("18,128", "11,845", "10,154", "15,405", "73,096")):
        chk.count(f"S15 total / {h}", printed, int(vc[h].sum()), "supp")
    chk.count("S15 total / pooled", "128,628", sum(int(vc[h].sum()) for h in order), "supp")
    pooled = {c: sum(int(vc[h].get(c, 0)) for h in order) for c in S15}
    chk.count("S15 prose: Lymphoblast cells left when HRLS is held out", "0",
              pooled["Lymphoblast"] - int(vc["HRLS"].get("Lymphoblast", 0)), "supp")
    chk.count("S15 prose: Promyelocyte cells left when BMCMHF is held out", "70",
              pooled["Promyelocyte"] - int(vc["BMCMHF"]["Promyelocyte"]), "supp")
    chk.count("S15 prose: Erythroblast cells left when BMCMHF is held out", "1,625",
              pooled["Erythroblast"] - int(vc["BMCMHF"]["Erythroblast"]), "supp")
    chk.num("S15 prose: share of Promyelocyte support BMCMHF removes", "98.8%",
            100 * vc["BMCMHF"]["Promyelocyte"] / pooled["Promyelocyte"], "supp")
    chk.num("S15 prose: share of Erythroblast support BMCMHF removes", "89.2%",
            100 * vc["BMCMHF"]["Erythroblast"] / pooled["Erythroblast"], "supp")
    chk.num("S15 prose: Myeloblast support the AML-LMU fold retains", "43.7%",
            100 * (pooled["Myeloblast"] - vc["AML-LMU"]["Myeloblast"]) / pooled["Myeloblast"], "supp")

    # tables/lodo_corrected_perseed.csv (cited by Supplementary Note 1)
    t = pd.read_csv(TABLES / "lodo_corrected_perseed.csv")
    ok = True
    for _, r in t.iterrows():
        i = LODO_SEEDS.index(int(r.seed))
        s, d = sc[r.held_out][i], data[r.held_out][i]
        want = (f"{present[r.held_out][i]:.4f}", f"{union[r.held_out][i]:.4f}", f"{s.accuracy():.4f}",
                f"{s.balanced_accuracy():.4f}", len(set(d.true_label)), len(d))
        got = (f"{r.present_macro_f1:.4f}", f"{r.nine_class_macro_f1:.4f}", f"{r.accuracy:.4f}",
               f"{r.balanced_accuracy:.4f}", int(r.n_present), int(r.n_eval))
        ok &= want == got
    chk.true("tables/lodo_corrected_perseed.csv: all 15 rows recompute from lodo_proposed/ (4 decimals)", ok and len(t) == 15)
    return fold_means


def check_lodo_foundation(chk: Checker, fold_means: dict) -> None:
    chk.section("Supp S16: DinoBloom-B on the identical LODO folds")
    gf = pd.read_csv(EV / "lodo_foundation" / "grid_foundation_lodo.csv.gz")
    chk.count("S16 grid: unique (configuration, fold, seed) rows", "30", len(gf.drop_duplicates(["cfg", "fold", "seed"])), "supp")
    gf = gf.drop_duplicates(["cfg", "fold", "seed"])
    folds = {"AML-LMU": "LMU", "PBC": "DIB", "Raabin-WBC": "Raabin", "HRLS": "HRLS", "BMCMHF": "BM"}
    S16 = {"AML-LMU": ("18,128", "0.590", "0.008", "0.677", "0.004", "0.647", "0.018"),
           "PBC": ("11,845", "0.909", "0.015", "0.918", "0.025", "0.931", "0.011"),
           "Raabin-WBC": ("10,154", "0.742", "0.015", "0.844", "0.013", "0.792", "0.034"),
           "HRLS": ("15,405", "0.763", "0.008", "0.781", "0.011", "0.797", "0.000"),
           "BMCMHF": ("73,096", "0.606", "0.030", "0.660", "0.009", "0.628", "0.017")}
    means = {"proposed": {}, "dino_meta": {}, "dino_ft": {}}
    sizes = {}
    for h, (n, pm, psd, mm, msd, fm, fsd) in S16.items():
        prop = [Scores(confusion(d.true_label, d.pred_label)).macro_f1(sorted(set(d.true_label)))
                for d in (lodo("lodo_proposed", LODO_TAGS[h], s) for s in LODO_SEEDS)]
        chk.num(f"S16 {h}: Swin-B (proposed) mean", pm, ms(prop)[0], "supp")
        chk.num(f"S16 {h}: Swin-B (proposed) sd", psd, ms(prop)[1], "supp")
        means["proposed"][h] = ms(prop)[0]
        for cfg, m, sd in (("dino_meta", mm, msd), ("dino_ft", fm, fsd)):
            sub = gf[(gf.cfg == cfg) & (gf.fold == folds[h])]
            chk.count(f"S16 {h}: {cfg} seeds", "3", len(sub), "supp")
            chk.count(f"S16 {h}: n ({cfg})", n, int(sub.n_rows.iloc[0]), "supp")
            v = sub.macroF1_present.values
            chk.num(f"S16 {h}: DinoBloom-B {cfg} mean", m, float(v.mean()), "supp")
            chk.num(f"S16 {h}: DinoBloom-B {cfg} sd", sd, float(v.std()), "supp")
            means[cfg][h] = float(v.mean())
        sizes[h] = int(n.replace(",", ""))
    for cfg, eq, cw in (("proposed", "0.722", "0.661"), ("dino_meta", "0.776", "0.715"), ("dino_ft", "0.759", "0.692")):
        chk.num(f"S16 mean over folds (equal weight): {cfg}", eq, np.mean(list(means[cfg].values())), "supp")
        chk.num(f"S16 mean over folds (cell-weighted): {cfg}", cw,
                sum(means[cfg][h] * sizes[h] for h in sizes) / sum(sizes.values()), "supp")
    for cfg in ("dino_meta", "dino_ft"):
        chk.true(f"S16: DinoBloom-B ({cfg}) exceeds the proposed model on every fold",
                 all(means[cfg][h] > means["proposed"][h] for h in sizes))
    for cfg, (pbc, hrls) in (("dino_meta", ("0.009", "0.018")), ("dino_ft", ("0.022", "0.034"))):
        where = "S16 / Section 3.4" if cfg == "dino_meta" else "S16 (BPaCo variant)"
        chk.num(f"{where}: {cfg} lead on PBC", pbc, means[cfg]["PBC"] - means["proposed"]["PBC"], "supp")
        chk.num(f"{where}: {cfg} lead on HRLS", hrls, means[cfg]["HRLS"] - means["proposed"]["HRLS"], "supp")


def check_hashes(chk: Checker) -> None:
    chk.section("Supp S1: detector-training versus external-benchmark images (SHA-256 lists)")
    det = pd.read_csv(EV / "leakage_audit" / "detector_train_image_hashes.csv.gz").sha256
    ext = pd.read_csv(EV / "leakage_audit" / "external_field_image_hashes.csv.gz").sha256
    chk.count("TXL-PBC detector-training images hashed", "1,260", len(det), "supp")
    chk.count("WBCBench2026 Phase-1 fields hashed", "8,288", len(ext), "supp")
    chk.true("all 1,260 and all 8,288 hashes distinct within their own sets", det.is_unique and ext.is_unique)
    chk.count("images shared between the two sets", "0", len(set(det) & set(ext)), "supp")


def check_wbcbench(chk: Checker, pub: Public) -> None:
    S = pub.scores
    # ------------------------------------------------------------ Table 3 external, S10, S2
    chk.section("Table 3 external columns, Section 3.3-3.4 (WBCBench2026 Phase-1 pre-cropped, n = 8,061)")
    ext_means = {}
    for name, (tmpl, seeds, p) in TABLE2_EXT.items():
        sc = [S(tmpl.format(s=s)) for s in seeds]
        for s_, x in zip(seeds, sc):
            chk.count(f"{name}: evaluated cells (seed {s_})", "8,061", x.n(), "ms")
        for (lab, fn), (pm, psd) in zip((("macro-F1", lambda x: x.macro_f1()), ("weighted-F1", lambda x: x.weighted_f1()),
                                         ("balanced accuracy", lambda x: x.balanced_accuracy())),
                                        ((p[0], p[1]), (p[2], p[3]), (p[4], p[5]))):
            m, sd = ms([fn(x) for x in sc])
            chk.num(f"{name}: {lab} mean", pm, m, "ms")
            chk.num(f"{name}: {lab} sd", psd, sd, "ms")
            if lab == "macro-F1":
                ext_means[name] = m
        chk.num(f"{name}: BLAST F1", p[6], np.mean([x.f1("Blast_Group") for x in sc]), "ms")
        chk.num(f"{name}: PROMY F1", p[7], np.mean([x.f1("Promyelocyte") for x in sc]), "ms")
    probe = S("frozen_probe/dinobloom_probe")
    chk.count("DinoBloom-B frozen probe: evaluated cells", "8,061", probe.n(), "ms")
    for lab, val, printed in (("macro-F1", probe.macro_f1(), "0.809"), ("weighted-F1", probe.weighted_f1(), "0.911"),
                              ("balanced accuracy", probe.balanced_accuracy(), "0.822"),
                              ("BLAST F1", probe.f1("Blast_Group"), "0.547"), ("PROMY F1", probe.f1("Promyelocyte"), "0.558")):
        chk.num(f"DinoBloom-B (frozen; linear probe): {lab}", printed, val, "ms")
    best_blast = max(TABLE2_EXT, key=lambda n: np.mean([S(TABLE2_EXT[n][0].format(s=s)).f1("Blast_Group") for s in TABLE2_EXT[n][1]]))
    best_promy = max(TABLE2_EXT, key=lambda n: np.mean([S(TABLE2_EXT[n][0].format(s=s)).f1("Promyelocyte") for s in TABLE2_EXT[n][1]]))
    chk.true("Table 3: highest external macro-F1, BLAST and PROMY on the proposed row (bold)",
             max(ext_means, key=ext_means.get) == best_blast == best_promy == "Swin-B (colour aug + CW-CE; proposed)"
             and ext_means["Swin-B (colour aug + CW-CE; proposed)"] > probe.macro_f1())
    cnn = ["VGG16-BN (CW-CE)", "EfficientNet-B0 (CW-CE)", "ResNet50 (CW-CE)"]
    chk.num("Abstract / Section 3.4: CNN baseline macro-F1 low end", "0.718", min(ext_means[n] for n in cnn), "ms")
    chk.num("Abstract / Section 3.4: CNN baseline macro-F1 high end", "0.813", max(ext_means[n] for n in cnn), "ms")
    cnn_w = [np.mean([S(TABLE2_EXT[n][0].format(s=s)).weighted_f1() for s in CNN_SEEDS]) for n in cnn]
    chk.num("Section 3.3: CNN weighted-F1 low end", "0.886", min(cnn_w), "ms")
    chk.num("Section 3.3: CNN weighted-F1 high end", "0.927", max(cnn_w), "ms")
    cnn_b = [np.mean([S(TABLE2_EXT[n][0].format(s=s)).f1("Blast_Group") for s in CNN_SEEDS]) for n in cnn]
    chk.num("Section 3.3: CNN Blast group F1 low end (0.65--0.69)", "0.65", min(cnn_b), "ms")
    chk.num("Section 3.3: CNN Blast group F1 high end (0.65--0.69)", "0.69", max(cnn_b), "ms")
    chk.num("Section 3.3: ResNet50 external gain under BPaCo with SGD", "0.097",
            ext_means["ResNet50 (BPaCo; SGD)"] - ext_means["ResNet50 (CW-CE)"], "ms")
    sds = {n: ms(seed_values(pub, TABLE2_EXT[n][0], TABLE2_EXT[n][1], lambda x: x.macro_f1()))[1] for n in TABLE2_EXT}
    gap = ext_means["VGG16-BN (CW-CE)"] - ext_means["EfficientNet-B0 (CW-CE)"]
    chk.num("S9 footnote: VGG16-BN minus EfficientNet-B0 seed-mean gap", "0.017", gap, "supp")
    chk.true("S9 footnote: that gap is smaller than either across-seed sd",
             gap < sds["VGG16-BN (CW-CE)"] and gap < sds["EfficientNet-B0 (CW-CE)"])

    chk.section("Supp S10 per-class external F1 and Supp S2 external class support")
    support = {"Neutrophil": ("4,018", "49.8"), "Lymphocyte": ("2,433", "30.2"), "Blast_Group": ("572", "7.1"),
               "Eosinophil": ("181", "2.2"), "Monocyte": ("741", "9.2"), "Promyelocyte": ("42", "0.5"),
               "Basophil": ("74", "0.9")}
    sw = S("backbone_comparison/swin_proposed_seed42")
    for c, (n, pct) in support.items():
        chk.count(f"S2 / S10 external support {SHORT[c]}", n, int(sw.support[c]), "supp")
        chk.num(f"S2 external share {SHORT[c]} (%)", pct, 100 * sw.support[c] / sw.n(), "supp")
    chk.count("S2 / S10 external total", "8,061", sw.n(), "supp")
    odd = [r.key for r in RUNS if r.group in ("backbone_comparison", "backbone_common_manifold", "matched_recipe",
                                                "recipe_transfer", "cwce_default", "aggressive_recipe", "resolution",
                                                "augmentation_policy", "tta_ablation", "frozen_probe")
           and [int(S(r.key).support[c]) for c in CLS7] != [int(sw.support[c]) for c in CLS7]]
    chk.true("every pre-cropped run scores the same 8,061 labelled cells (class support identical)", not odd,
             ", ".join(odd[:3]))
    S10 = {"Basophil": ("0.899", "0.883", "0.870", "0.897"), "Blast_Group": ("0.793", "0.676", "0.686", "0.654"),
           "Eosinophil": ("0.945", "0.940", "0.757", "0.857"), "Lymphocyte": ("0.943", "0.925", "0.882", "0.906"),
           "Monocyte": ("0.870", "0.831", "0.690", "0.785"), "Neutrophil": ("0.990", "0.986", "0.966", "0.981"),
           "Promyelocyte": ("0.601", "0.446", "0.177", "0.491")}
    cols = ["Swin-B (colour aug + CW-CE; proposed)", "VGG16-BN (CW-CE)", "ResNet50 (CW-CE)", "EfficientNet-B0 (CW-CE)"]
    for c, row in S10.items():
        for name, printed in zip(cols, row):
            tmpl, seeds, _ = TABLE2_EXT[name]
            chk.num(f"S10 {SHORT[c]} F1: {name.split(' (')[0]}", printed,
                    np.mean([S(tmpl.format(s=s)).f1(c) for s in seeds]), "supp")

    # ------------------------------------------------------------ S7 per-seed, Section 3.4
    chk.section("Section 3.4 and Supp S7: backbone comparisons across training configurations")
    per = {}
    for label, tmpl, printed in (("Swin-B BPaCo (AdamW)", "matched_recipe/swinbpaco_seed{s}", ("0.836", "0.827", "0.795")),
                                 ("ResNet50 BPaCo (AdamW, shared protocol)", "matched_recipe/rn50bpaco_adamw_seed{s}", ("0.741", "0.761", "0.783")),
                                 ("ResNet50 colour recipe (AdamW)", "recipe_transfer/rn50_color_adamw_seed{s}", ("0.672", "0.713", "0.697")),
                                 ("ResNet50 colour recipe (SGD)", "recipe_transfer/rn50_color_sgd_seed{s}", ("0.707", "0.721", "0.755")),
                                 ("Swin-B CW-CE, default augmentation", "cwce_default/swin_cwce_default_seed{s}", ("0.849", "0.869", "0.849")),
                                 ("ResNet50 CW-CE, default augmentation", "cwce_default/rn50_cwce_default_seed{s}", ("0.678", "0.777", "0.738")),
                                 ("DinoBloom-B fine-tuned", "matched_recipe/dinobloom_seed{s}", ("0.835", "0.873", "0.759")),
                                 ("proposed Swin-B", "backbone_comparison/swin_proposed_seed{s}", ("0.886", "0.878", "0.824"))):
        v = seed_values(pub, tmpl, SEEDS, lambda x: x.macro_f1())
        per[label] = v
        for s, x, pr in zip(SEEDS, v, printed):
            chk.num(f"S7 per-seed macro-F1, {label} (seed {s})", pr, x, "supp")
    sgd = seed_values(pub, "matched_recipe/rn50bpaco_sgd_seed{s}", SEEDS, lambda x: x.macro_f1())
    mean = {k: ms(v)[0] for k, v in per.items()}
    for label, (pm, psd) in (("ResNet50 colour recipe (AdamW)", ("0.694", "0.017")), ("ResNet50 colour recipe (SGD)", ("0.727", "0.020")),
                             ("Swin-B CW-CE, default augmentation", ("0.856", "0.009")),
                             ("ResNet50 CW-CE, default augmentation", ("0.731", "0.041"))):
        chk.num(f"Section 3.4 / S7: {label} mean", pm, mean[label], "ms")
        chk.num(f"Section 3.4 / S7: {label} sd", psd, ms(per[label])[1], "ms")
    chk.true("Section 3.4: ResNet50 colour recipe (AdamW) below ResNet50's CW-CE baseline (0.718)",
             mean["ResNet50 colour recipe (AdamW)"] < ext_means["ResNet50 (CW-CE)"])
    margins = {"BPaCo (shared AdamW)": mean["Swin-B BPaCo (AdamW)"] - mean["ResNet50 BPaCo (AdamW, shared protocol)"],
               "colour augmentation": mean["proposed Swin-B"] - mean["ResNet50 colour recipe (AdamW)"],
               "default augmentation": mean["Swin-B CW-CE, default augmentation"] - mean["ResNet50 CW-CE, default augmentation"]}
    for k, printed in (("BPaCo (shared AdamW)", "0.058"), ("colour augmentation", "0.169"), ("default augmentation", "0.125")):
        where = "Table 3a / S7" if k == "default augmentation" else "Table 3a / Section 3.4 / S7"
        chk.num(f"{where}: Swin-B minus ResNet50 seed-mean margin, {k}", printed, margins[k], "supp")
    chk.num("Discussion: smallest margin (+0.058 to +0.169)", "+0.058", min(margins.values()), "ms")
    chk.num("Discussion: largest margin (+0.058 to +0.169)", "+0.169", max(margins.values()), "ms")
    t3 = list(ext_means.values()) + [probe.macro_f1()] + [mean[k] for k in (
        "ResNet50 colour recipe (AdamW)", "Swin-B CW-CE, default augmentation", "ResNet50 CW-CE, default augmentation")]
    chk.num("Discussion: entire external macro-F1 spread of Table 3 (both panels)", "0.169", max(t3) - min(t3), "ms")
    for a, b in (("Swin-B BPaCo (AdamW)", "ResNet50 BPaCo (AdamW, shared protocol)"),
                 ("proposed Swin-B", "ResNet50 colour recipe (AdamW)"),
                 ("Swin-B CW-CE, default augmentation", "ResNet50 CW-CE, default augmentation")):
        chk.true(f"S7: every {a} seed exceeds every {b} seed", min(per[a]) > max(per[b]))
    chk.num("Section 3.4 / S7: SGD arm minus shared-protocol ResNet50 arm (seed means)", "0.053",
            ms(sgd)[0] - mean["ResNet50 BPaCo (AdamW, shared protocol)"], "ms")
    chk.num("S7: Swin-B BPaCo seed mean (0.8195 versus 0.8148)", "0.8195", mean["Swin-B BPaCo (AdamW)"], "supp")
    chk.num("S7: ResNet50 BPaCo-SGD seed mean (0.8195 versus 0.8148)", "0.8148", ms(sgd)[0], "supp")
    chk.num("S7: those seed means differ by", "0.005", mean["Swin-B BPaCo (AdamW)"] - ms(sgd)[0], "supp")
    for s, x, printed in zip(SEEDS, np.array(per["Swin-B BPaCo (AdamW)"]) - np.array(sgd), ("+0.016", "+0.017", "-0.019")):
        chk.num(f"S7: Swin-B BPaCo minus ResNet50 BPaCo-SGD, seed {s}", printed, x, "supp")
    chk.count("S7: seed-matched pairs on which Swin-B BPaCo is higher than the SGD arm", "2",
              int(sum(np.array(per["Swin-B BPaCo (AdamW)"]) > np.array(sgd))), "supp")
    for s, x, printed in zip(SEEDS, np.array(per["proposed Swin-B"]) - np.array(per["ResNet50 colour recipe (SGD)"]),
                             ("+0.179", "+0.158", "+0.069")):
        chk.num(f"S7: proposed minus ResNet50 colour recipe (SGD), seed {s}", printed, x, "supp")
    # The S7 DinoBloom seed differences are exact differences of the per-seed macro-F1 values
    # (0.0512 / 0.0056 / 0.0644), not differences of the rounded per-seed values printed beside them.
    for s, x, y, printed in zip(SEEDS, per["proposed Swin-B"], per["DinoBloom-B fine-tuned"], ("+0.051", "+0.006", "+0.064")):
        chk.num(f"S7: proposed minus DinoBloom-B, seed {s}", printed, x - y, "supp")
    chk.true("Section 3.4: DinoBloom-B below the proposed model on every matched seed",
             all(a > b for a, b in zip(per["proposed Swin-B"], per["DinoBloom-B fine-tuned"])))
    from scipy.stats import ttest_rel
    chk.num("Section 3.4 / S7: paired t-test, proposed versus DinoBloom-B (p~0.15)", "0.15",
            float(ttest_rel(per["proposed Swin-B"], per["DinoBloom-B fine-tuned"]).pvalue), "ms")
    chk.num("Section 3.4: proposed Swin-B macro-F1 (frozen probe compared with)", "0.863", mean["proposed Swin-B"], "ms")
    chk.true("Section 3.4: frozen probe inside the CW-CE baseline range (0.718--0.813)",
             min(ext_means[n] for n in cnn) <= probe.macro_f1() <= max(ext_means[n] for n in cnn) + 5e-4)

    # ------------------------------------------------------------ ensembles, paired tests
    chk.section("Section 3.6, Supp S7-S9 and S13: three-seed ensembles and paired tests")
    E = pub.ens

    def es(name, rule="probability"):
        return Scores(E[(name, rule)])
    for name, printed in (("swin_proposed", "0.893"), ("vgg16bn", "0.841"), ("resnet50", "0.751"), ("efficientnet_b0", "0.848")):
        where = "S9" if name == "vgg16bn" else "S9 / Section 3.3"
        chk.num(f"{where}: {name} probability-ensemble macro-F1", printed, es(name).macro_f1(), "supp")
    chk.num("Section 3.3: CNN ensemble range low (0.751--0.848)", "0.751", min(es(n).macro_f1() for n in ("vgg16bn", "resnet50", "efficientnet_b0")), "ms")
    chk.num("Section 3.3: CNN ensemble range high (0.751--0.848)", "0.848", max(es(n).macro_f1() for n in ("vgg16bn", "resnet50", "efficientnet_b0")), "ms")
    chk.true("S9 footnote: VGG16-BN and EfficientNet-B0 ensembles rank in reverse of their seed means",
             es("vgg16bn").macro_f1() < es("efficientnet_b0").macro_f1() and ext_means["VGG16-BN (CW-CE)"] > ext_means["EfficientNet-B0 (CW-CE)"])
    S8 = (("swin_proposed", ("0.954", "0.908", "0.893", "0.786", "0.841", "0.702")),
          ("swin_cwce_default", ("0.946", "0.913", "0.881", "0.810", "0.829", "0.687")),
          ("swin_mixup_bpaco_crt", ("0.945", "0.906", "0.864", "0.786", "0.822", "0.611")))
    for name, p in S8:
        x = es(name)
        for lab, val, printed in zip(("accuracy", "balanced accuracy", "macro-F1", "minimum recall", "BLAST F1", "PROMY F1"),
                                     (x.accuracy(), x.balanced_accuracy(), x.macro_f1(), x.min_recall(),
                                      x.f1("Blast_Group"), x.f1("Promyelocyte")), p):
            chk.num(f"S8 {name}: {lab}", printed, val, "supp")
    chk.true("S8: proposed row has the best ensemble macro-F1, the CW-CE row the best minimum recall",
             max((n for n, _ in S8), key=lambda n: es(n).macro_f1()) == "swin_proposed"
             and max((n for n, _ in S8), key=lambda n: es(n).min_recall()) == "swin_cwce_default")
    chk.num("Section 3.6: CW-CE baseline ensemble (0.893 versus 0.881)", "0.881", es("swin_cwce_default").macro_f1(), "ms")
    chk.num("Section 3.6: aggressive recipe ensemble", "0.864", es("swin_mixup_bpaco_crt").macro_f1(), "ms")
    x = es("swin_proposed")
    chk.num("S11 BLAST external F1 (argmax)", "0.841", x.f1("Blast_Group"), "supp")
    chk.num("S11 PROMY external F1 (argmax)", "0.702", x.f1("Promyelocyte"), "supp")
    chk.num("Overview: pre-cropped seed mean to ensemble range low (0.863--0.893)", "0.863", ext_means["Swin-B (colour aug + CW-CE; proposed)"], "ms")
    chk.num("Overview: pre-cropped seed mean to ensemble range high (0.863--0.893)", "0.893", x.macro_f1(), "ms")

    # Section 3.8 error structure and Supp S13 panel, from the proposed ensemble matrix
    cm = E[("swin_proposed", "probability")]
    for c, printed in (("Neutrophil", "0.99"), ("Eosinophil", "0.97"), ("Lymphocyte", "0.94"),
                       ("Blast_Group", "0.84"), ("Promyelocyte", "0.79")):
        chk.num(f"Section 3.8: external recall {SHORT[c]}", printed, x.recall(c), "ms")
    chk.num("Section 3.8: blasts read as Lymphocyte (%)", "11%", 100 * cm.loc["Blast_Group", "Lymphocyte"] / x.support["Blast_Group"], "ms")
    chk.num("Section 3.8: promyelocytes called Basophil (%)", "10%", 100 * cm.loc["Promyelocyte", "Basophil"] / x.support["Promyelocyte"], "ms")
    chk.num("Section 3.8: promyelocytes called Monocyte (%)", "7%", 100 * cm.loc["Promyelocyte", "Monocyte"] / x.support["Promyelocyte"], "ms")
    errors = int(cm.values.sum() - np.trace(cm.values))
    chk.count("Section 3.8: external errors of the ensemble", "368", errors, "ms")
    chk.num("Section 3.8: errors as a share of 8,061 predictions", "4.6%", 100 * errors / x.n(), "ms")
    chk.count("Section 3.8: Blast group read as Lymphocyte", "65", int(cm.loc["Blast_Group", "Lymphocyte"]), "ms")
    chk.count("Section 3.8: Lymphocyte read as Blast group", "64", int(cm.loc["Lymphocyte", "Blast_Group"]), "ms")
    chk.num("Section 3.8: Blast<->Lymphocyte share of all errors", "35%",
            100 * (cm.loc["Blast_Group", "Lymphocyte"] + cm.loc["Lymphocyte", "Blast_Group"]) / errors, "ms")
    chk.count("Section 3.8: Lymphocyte read as Monocyte", "63", int(cm.loc["Lymphocyte", "Monocyte"]), "ms")
    chk.count("Section 3.8: Monocyte read as Lymphocyte", "27", int(cm.loc["Monocyte", "Lymphocyte"]), "ms")
    chk.num("Section 3.8: Lymphocyte<->Monocyte share of all errors", "24%",
            100 * (cm.loc["Lymphocyte", "Monocyte"] + cm.loc["Monocyte", "Lymphocyte"]) / errors, "ms")
    gran = ["Neutrophil", "Eosinophil", "Basophil"]
    chk.num("Section 3.8: confusions among the mature granulocyte classes (%)", "3%",
            100 * sum(cm.loc[a, b] for a in gran for b in gran if a != b) / errors, "ms")
    promy_fp = int(cm["Promyelocyte"].sum() - cm.loc["Promyelocyte", "Promyelocyte"])
    promy_fn = int(cm.loc["Promyelocyte"].sum() - cm.loc["Promyelocyte", "Promyelocyte"])
    chk.count("Section 3.8 / S13: Promyelocyte false alarms", "19", promy_fp, "ms")
    chk.count("Section 3.8: Promyelocyte false negatives", "9", promy_fn, "ms")
    chk.num("Section 3.8: external Promyelocyte prevalence", "0.5%", 100 * x.support["Promyelocyte"] / x.n(), "ms")
    blast_fn = int(cm.loc["Blast_Group"].sum() - cm.loc["Blast_Group", "Blast_Group"])
    blast_fp = int(cm["Blast_Group"].sum() - cm.loc["Blast_Group", "Blast_Group"])
    oth = [c for c in CLS7 if c not in ("Blast_Group", "Promyelocyte")]
    p_fn = int(cm.loc["Promyelocyte", oth].sum())       # Blast takes precedence
    p_fp = int(cm.loc[oth, "Promyelocyte"].sum())
    for lab, val, printed in (("Blast_FN", blast_fn, "93"), ("Blast_FP", blast_fp, "88"),
                              ("Promy_FN", p_fn, "9"), ("Promy_FP", p_fp, "15")):
        chk.count(f"S13 panel size {lab} (Blast group takes precedence)", printed, val, "supp")
    chk.count("S13 / Section 3.9: reviewed panel (Blast group or Promyelocyte disagreements)", "205",
              blast_fn + blast_fp + p_fn + p_fp, "supp")
    chk.count("S13 caption: true Blast group predicted Promyelocyte (counted under Blast_FN)", "4",
              int(cm.loc["Blast_Group", "Promyelocyte"]), "supp")
    ag = pub.agree.set_index("ensemble").loc["swin_proposed"]
    chk.count("Methods: probability ensemble and majority vote disagree (of 8,061)", "64", int(ag.n_vote_vs_probability_disagree), "ms")
    chk.count("Methods: images in the vote/probability comparison", "8,061", int(ag.n), "ms")

    # Supp S7 paired comparisons: point differences, McNemar, tie-rule sensitivity
    def vdelta(a, b, rule):
        return es(a, rule).macro_f1() - es(b, rule).macro_f1()
    S7T = {"bpaco_shared_adamw": ("<0.001", "+0.062"), "bpaco_adamw_vs_sgd": ("0.21", "+0.010"),
           "proposed_vs_dinobloom": ("<0.001", "+0.031")}
    for comp, a, b in COMPARISONS:
        p_print, d_print = S7T[comp]
        chk.num(f"S7 {comp}: majority-vote macro-F1 difference", d_print, vdelta(a, b, "vote_tie42"), "supp")
        r = pub.mcn[(pub.mcn.comparison == comp) & (pub.mcn.rule == "vote_tie42")].iloc[0]
        chk.count(f"S7 {comp}: McNemar images", "8,061", int(r.n), "supp")
        p = mcnemar_p(int(r.a_correct_b_wrong), int(r.a_wrong_b_correct))
        if p_print.startswith("<"):
            chk.true(f"S7 {comp}: McNemar exact p {p_print}", p < float(p_print[1:]), f"p = {p:.3g}")
        else:
            chk.num(f"S7 {comp}: McNemar exact p", p_print, p, "supp")
    for comp, a, b in COMPARISONS[:2]:
        base = vdelta(a, b, "vote_tie42")
        dev = max(abs(vdelta(a, b, f"vote_tie{j}") - base) for j in SEEDS)
        chk.true(f"S7 footnote: {comp} difference moves at most 0.003 under other tie-breaking seeds",
                 dev <= 0.0035, f"max shift {dev:.4f}")
    dd = [vdelta("swin_proposed", "dinobloom_ft", f"vote_tie{j}") for j in SEEDS]
    chk.num("S7 footnote: DinoBloom-B difference under tie rules, low end", "+0.017", min(dd), "supp")
    chk.num("S7 footnote: DinoBloom-B difference under tie rules, high end", "+0.031", max(dd), "supp")
    chk.num("S7 footnote: tie-free probability-ensemble difference, proposed minus DinoBloom-B", "+0.029",
            es("swin_proposed").macro_f1() - es("dinobloom_ft").macro_f1(), "supp")
    r = pub.mcn[(pub.mcn.comparison == "proposed_vs_dinobloom") & (pub.mcn.rule == "probability")].iloc[0]
    chk.true("Section 3.4: proposed ensemble beats DinoBloom-B's on shared images also under the probability rule (McNemar p<0.001)",
             mcnemar_p(int(r.a_correct_b_wrong), int(r.a_wrong_b_correct)) < 0.001)
    sb, ra = es("swin_bpaco", "vote_tie42"), es("rn50_bpaco_adamw", "vote_tie42")
    for c, printed in (("Blast_Group", "+0.122"), ("Eosinophil", "+0.114"), ("Promyelocyte", "+0.182"), ("Monocyte", "-0.022")):
        chk.num(f"S7: Swin-B minus ResNet50 BPaCo (AdamW) vote-ensemble F1, {SHORT[c]}", printed, sb.f1(c) - ra.f1(c), "supp")
    chk.true("S7: ResNet50 higher only for Monocyte",
             [c for c in CLS7 if ra.f1(c) > sb.f1(c)] == ["Monocyte"])

    # ------------------------------------------------------------ calibration and thresholds
    chk.section("Section 3.6, Supp S11 and Fig. S2: calibration and post hoc thresholds")
    cal = pub.cal[pub.cal.ensemble == "swin_proposed"]
    n = int(cal.n.sum())
    chk.count("calibration: predictions binned", "8,061", n, "supp")
    acc_b = cal.n_correct / cal.n.where(cal.n > 0)
    conf_b = cal.sum_confidence / cal.n.where(cal.n > 0)
    ece = float(((cal.n / n) * (acc_b - conf_b).abs()).fillna(0).sum())
    chk.num("ECE (15 equal-width bins)", "0.017", ece, "ms")
    chk.num("multiclass Brier score", "0.074", float(cal.sum_brier.sum() / n), "ms")
    chk.num("mean confidence (0.970 versus 0.954)", "0.970", float(cal.sum_confidence.sum() / n), "ms")
    chk.num("accuracy (0.970 versus 0.954)", "0.954", float(cal.n_correct.sum() / n), "ms")
    top = cal.loc[cal.n.idxmax()]
    chk.true("the most populated bin is the top bin", int(top.bin) == N_BINS - 1)
    chk.num("top bin share of predictions", "90%", 100 * top.n / n, "ms")
    chk.num("top bin mean confidence (0.998 versus 0.984)", "0.998", float(top.sum_confidence / top.n), "ms")
    chk.num("top bin accuracy (0.998 versus 0.984)", "0.984", float(top.n_correct / top.n), "ms")
    thr = pub.thr[pub.thr.ensemble == "swin_proposed"].copy()
    thr["f1"] = 2 * thr.tp / (2 * thr.tp + thr.fp + thr.fn)
    tvl = pd.read_csv(TABLES / "threshold_validation_locked.csv").set_index("cls")
    sweep_int = pd.read_csv(EV / "internal_ensemble" / "swin_proposed_threshold_sweep.csv.gz")
    sweep_int["f1"] = 2 * sweep_int.tp / (2 * sweep_int.tp + sweep_int.fp + sweep_int.fn)
    for c, label, t_star, locked, t_or, f_or in (("Blast_Group", "BLAST", "0.500", "0.839", "0.380", "0.849"),
                                                ("Promyelocyte", "PROMY", "0.480", "0.719", "0.720", "0.744")):
        g = thr[thr.cls == c].reset_index(drop=True)
        gi = sweep_int[sweep_int.cls == c].reset_index(drop=True)
        ts = float(gi.loc[gi.f1.idxmax(), "threshold"])
        at = g[np.isclose(g.threshold, ts)].iloc[0]
        best = g.loc[g.f1.idxmax()]
        row = tvl.loc[c]
        # S11 cites tables/threshold_validation_locked.csv as its source: the printed cells agree
        # with that table within rounding (from_table), and the table recomputes exactly from the
        # sweeps (below). The BLAST oracle F1 is printed 0.849, the exact 982/1156 = 0.84948 that
        # the table stores as 0.8495; S11 cites the sweep counts for that cell.
        chk.from_table(f"S11 {label}: internal-tuned threshold t* (cited table)", t_star, row.t_star, "supp")
        chk.from_table(f"S11 {label}: external F1 at the internal t* (cited table)", locked, row.ext_locked_f1, "supp")
        chk.from_table(f"S11 {label}: oracle threshold (cited table)", t_or, row.t_oracle, "supp")
        chk.from_table(f"S11 {label}: external F1 at the oracle threshold (cited table)", f_or, row.ext_oracle_f1, "supp")
        want = {"t_star": f"{ts:.4f}", "internal_f1": f"{float(gi.f1.max()):.4f}", "ext_argmax_f1": f"{x.f1(c):.4f}",
                "ext_locked_f1": f"{float(at.f1):.4f}", "ext_locked_prec": f"{at.tp / (at.tp + at.fp):.4f}",
                "ext_locked_rec": f"{at.tp / (at.tp + at.fn):.4f}", "t_oracle": f"{float(best.threshold):.4f}",
                "ext_oracle_f1": f"{float(best.f1):.4f}"}
        got = {k: f"{float(row[k]):.4f}" for k in want}
        chk.true(f"tables/threshold_validation_locked.csv {label}: every column recomputes (4 decimals)", want == got,
                 "" if want == got else f"{got} vs {want}")
    chk.from_table("Section 3.6: PROMY external F1 rises to (0.702 to 0.719)", "0.719", tvl.loc["Promyelocyte", "ext_locked_f1"], "ms")
    chk.from_table("Section 3.6: BLAST external F1 changes to (0.841 to 0.839)", "0.839", tvl.loc["Blast_Group", "ext_locked_f1"], "ms")

    # ------------------------------------------------------------ resolution
    chk.section("Section 3.7 and Supp S17: naive resolution scaling (pre-cropped, n = 8,061)")
    res = {}
    for px, pm, psd in ((224, "0.863", "0.027"), (256, "0.830", "0.039"), (384, "0.464", "0.084")):
        sc = pub.seeds("resolution", f"swin_proposed_seed{{s}}_img{px}")
        res[px] = sc
        chk.num(f"{px} px: macro-F1 mean", pm, ms([x.macro_f1() for x in sc])[0], "ms")
        chk.num(f"{px} px: macro-F1 sd", psd, ms([x.macro_f1() for x in sc])[1], "ms")
    chk.true("Section 3.7: macro-F1 falls 224 > 256 > 384 px for each of the three seeds",
             all(res[224][i].macro_f1() > res[256][i].macro_f1() > res[384][i].macro_f1() for i in range(3)))
    for c, lo, hi in (("Blast_Group", "0.07", "0.52"), ("Promyelocyte", "0.00", "0.35")):
        v = [x.f1(c) for x in res[384]]
        chk.num(f"Section 3.7: 384 px per-seed {SHORT[c]} F1, lowest", lo, min(v), "ms")
        chk.num(f"Section 3.7: 384 px per-seed {SHORT[c]} F1, highest", hi, max(v), "ms")
    drop = {c: np.mean([a.f1(c) for a in res[224]]) - np.mean([b.f1(c) for b in res[384]]) for c in CLS7}
    chk.true("Section 3.7: every class degrades at 384 px", all(v > 0 for v in drop.values()))
    small = sorted(CLS7, key=lambda c: res[224][0].support[c])[:4]
    chk.true("Section 3.7: the four smallest classes degrade most on average",
             np.mean([drop[c] for c in small]) > np.mean([drop[c] for c in CLS7 if c not in small]),
             f"{np.mean([drop[c] for c in small]):.3f} versus {np.mean([drop[c] for c in CLS7 if c not in small]):.3f}")
    chk.text("S17: the sweep's own 224 px value", "0.863\u00b10.027",
             "{:.3f}\u00b1{:.3f}".format(*ms([x.macro_f1() for x in res[224]])), "supp")

    # ------------------------------------------------------------ localization routes
    chk.section("Abstract, Table 2b, Section 3.1 and Supp S6: cropping routes (n = 8,080 labelled fields)")
    route = {"noloc": pub.seeds("localization_ablation", "nocrop_swin_seed{s}"),
             "A": pub.seeds("backbone_comparison", "swin_proposed_seed{s}"),
             **{r: pub.seeds("localization_ablation", f"route{r}_seed{{s}}") for r in "BCDE"}}
    for r in ("noloc", "B", "C", "D", "E"):
        for s, x in zip(SEEDS, route[r]):
            chk.count(f"route {r}: labelled fields scored (seed {s})", "8,080", x.n(), "ms")
    for s, x in zip(SEEDS, route["B"]):
        chk.count(f"route B: labelled field without a detection, counted as an error (seed {s})", "1",
                  int(x.cm[ND].sum()), "ms")
    chk.true("routes no-localization, A, C, D and E: every labelled field scored (no missing detection)",
             all(int(x.cm[ND].sum()) == 0 for r in ("noloc", "A", "C", "D", "E") for x in route[r]))
    T3 = {"noloc": ("0.802", "0.641", "0.627", "0.039", "0.466", "0.175"),
          "B": ("0.944", "0.882", "0.879", "0.006", "0.820", "0.711"),
          "C": ("0.934", "0.883", "0.867", "0.013", "0.783", "0.730"),
          "D": ("0.849", "0.839", "0.760", "0.055", "0.546", "0.569")}
    for r, (acc, bal, m, sd, bl, pr) in T3.items():
        sc = route[r]
        chk.num(f"Table 2b {r}: accuracy", acc, np.mean([x.accuracy() for x in sc]), "ms")
        chk.num(f"Table 2b {r}: balanced accuracy", bal, np.mean([x.balanced_accuracy() for x in sc]), "ms")
        chk.num(f"Table 2b {r}: macro-F1 mean", m, ms([x.macro_f1() for x in sc])[0], "ms")
        chk.num(f"Table 2b {r}: macro-F1 sd", sd, ms([x.macro_f1() for x in sc])[1], "ms")
        chk.num(f"Table 2b {r}: BLAST F1", bl, np.mean([x.f1("Blast_Group") for x in sc]), "ms")
        chk.num(f"Table 2b {r}: PROMY F1", pr, np.mean([x.f1("Promyelocyte") for x in sc]), "ms")
    chk.true("Section 3.1: the YOLO route has the smaller across-seed sd",
             ms([x.macro_f1() for x in route["B"]])[1] < ms([x.macro_f1() for x in route["noloc"]])[1])
    S6 = {"noloc": ("0.802", "0.641", "0.627", "0.039", "0.466", "0.175"),
          "A": ("0.946", "0.887", "0.863", "0.028", "0.793", "0.601"),
          "B": ("0.944", "0.882", "0.879", "0.006", "0.820", "0.711"),
          "C": ("0.934", "0.883", "0.867", "0.013", "0.783", "0.730"),
          "D": ("0.849", "0.839", "0.760", "0.055", "0.546", "0.569"),
          "E": ("0.951", "0.899", "0.882", "0.019", "0.817", "0.680")}
    rmean = {}
    for r, (acc, bal, m, sd, bl, pr) in S6.items():
        sc = route[r]
        chk.num(f"S6 {r}: accuracy", acc, np.mean([x.accuracy() for x in sc]), "supp")
        chk.num(f"S6 {r}: balanced accuracy", bal, np.mean([x.balanced_accuracy() for x in sc]), "supp")
        mm, sdd = ms([x.macro_f1() for x in sc])
        rmean[r] = (mm, sdd)
        chk.num(f"S6 {r}: macro-F1 mean", m, mm, "supp")
        chk.num(f"S6 {r}: macro-F1 sd", sd, sdd, "supp")
        chk.num(f"S6 {r}: BLAST F1", bl, np.mean([x.f1("Blast_Group") for x in sc]), "supp")
        chk.num(f"S6 {r}: PROMY F1", pr, np.mean([x.f1("Promyelocyte") for x in sc]), "supp")
    chk.true("S6: route E has the highest point estimate and a larger seed sd than route B",
             max(("B", "C", "D", "E"), key=lambda r: rmean[r][0]) == "E" and rmean["E"][1] > rmean["B"][1])
    chk.true("S6: Cellpose keep-background (C) is below the trained detector (B)", rmean["C"][0] < rmean["B"][0])

    # ------------------------------------------------------------ attrition and detector coverage
    chk.section("Section 2.1, Section 3.2, Table 2a, Supp S1 and S12: fields, detector coverage and Phase-2")
    f = pub.fields
    p1, p2 = f[f.split == "phase1"], f[f.split == "phase2"]
    chk.count("Phase-1 fields released", "8,288", int(p1.n_fields.sum()), "ms")
    chk.count("Phase-2 evaluation fields released", "5,350", int(p2.n_fields.sum()), "ms")
    un1 = p1[p1.label7.isna() | (p1.label7 == "")]
    chk.count("Phase-1 fields without a seven-class counterpart", "208", int(un1.n_fields.sum()), "ms")
    chk.num("S1: those fields as a share of Phase-1", "2.5%", 100 * un1.n_fields.sum() / p1.n_fields.sum(), "supp")
    for raw, printed in (("Myelocyte", "108"), ("Metamyelocyte", "98"), ("Plasma_cell", "2")):
        chk.count(f"S1: unmapped Phase-1 fields, {raw}", printed, int(p1.set_index("raw_label").n_fields.get(raw, 0)), "supp")
    lab1 = p1[~(p1.label7.isna() | (p1.label7 == ""))]
    chk.count("Phase-1 labelled fields (full-pipeline n)", "8,080", int(lab1.n_fields.sum()), "ms")
    chk.count("S1: Phase-1 crops at detector confidence 0.70", "8,269", int(p1.n_precropped.astype(float).sum()), "supp")
    lost = (p1.n_fields - p1.n_precropped.astype(float)).astype(int)
    chk.count("S1: Phase-1 fields with no box at 0.70", "19", int(lost.sum()), "supp")
    chk.count("S1: of which Lymphocyte", "13", int(lost[p1.raw_label == "Lymphocyte"].sum()), "supp")
    chk.count("S1: of which Segmented neutrophil", "6", int(lost[p1.raw_label == "Segmented_neutrophil"].sum()), "supp")
    chk.count("pre-cropped evaluated set (n)", "8,061", int(lab1.n_precropped.astype(float).sum()), "ms")
    un2 = p2[p2.label7.isna() | (p2.label7 == "")]
    chk.count("S1: Phase-2 fields without a seven-class counterpart", "154", int(un2.n_fields.sum()), "supp")
    chk.count("S1: Phase-2 fields without a detection", "357", int((p2.n_fields - p2.n_detected).sum()), "supp")
    lab2 = p2[~(p2.label7.isna() | (p2.label7 == ""))]
    det1 = int(lab1.n_detected.sum())
    chk.count("Section 3.1 / 3.2: Phase-1 labelled fields with a detection", "8,079", det1, "ms")
    chk.num("Table 2a / Section 3.2: Phase-1 coverage", "99.99%", 100 * det1 / lab1.n_fields.sum(), "ms")
    det2, lab2n = int(lab2.n_detected.sum()), int(lab2.n_fields.sum())
    chk.count("Table 2a / Section 3.2: Phase-2 labelled fields", "5,196", lab2n, "ms")
    chk.count("Table 2a / Section 3.2: Phase-2 labelled fields with a detection", "4,848", det2, "ms")
    chk.count("Section 3.2 / S12: Phase-2 labelled fields without a detection", "348", lab2n - det2, "ms")
    chk.num("Table 2a / Section 3.2: Phase-2 coverage", "93.3%", 100 * det2 / lab2n, "ms")
    chk.num("S12: missed fields as a share of the labelled Phase-2 fields", "6.7%", 100 * (lab2n - det2) / lab2n, "supp")

    p2s = pub.seeds("phase2_degradation", "phase2_pipeline_seed{s}")
    p2d = pub.seeds("phase2_degradation", "phase2_pipeline_seed{s}", detected_only=True)
    for s, x, xd in zip(SEEDS, p2s, p2d):
        chk.count(f"Phase-2 seed {s}: labelled fields scored (coverage-aware)", "5,196", x.n(), "ms")
        chk.count(f"Phase-2 seed {s}: detected cells scored", "4,848", xd.n(), "ms")
    B = route["B"]
    chk.num("Table 2a Phase-1: balanced accuracy", "0.882", np.mean([x.balanced_accuracy() for x in B]), "ms")
    chk.num("Table 2a Phase-1: macro-F1 mean", "0.879", ms([x.macro_f1() for x in B])[0], "ms")
    chk.num("Table 2a Phase-1: macro-F1 sd", "0.006", ms([x.macro_f1() for x in B])[1], "ms")
    for lab, sc, bal, m, sd in (("Phase-2 (coverage-aware)", p2s, "0.795", "0.769", "0.002"),
                                ("Phase-2, detected cells only", p2d, "0.833", "0.788", "0.002")):
        chk.num(f"Table 2a {lab}: balanced accuracy", bal, np.mean([x.balanced_accuracy() for x in sc]), "ms")
        chk.num(f"Table 2a {lab}: macro-F1 mean", m, ms([x.macro_f1() for x in sc])[0], "ms")
        chk.num(f"Table 2a {lab}: macro-F1 sd", sd, ms([x.macro_f1() for x in sc])[1], "ms")
    miss = p2s[0].cm[ND]
    chk.true("Phase-2: the three seeds share the same 348 missed fields (class by class)",
             all(x.cm[ND].equals(miss) for x in p2s))
    sizes = {"Neutrophil": "219", "Lymphocyte": "83", "Monocyte": "30", "Eosinophil": "8", "Blast_Group": "7",
             "Promyelocyte": "1", "Basophil": "0"}
    for c, printed in sizes.items():
        chk.count(f"S12: missed-group size {SHORT[c]}", printed, int(miss[c]), "supp")
    chk.num("S12: neutrophils as a share of the missed set", "63%", 100 * miss["Neutrophil"] / miss.sum(), "supp")

    # ------------------------------------------------------------ provenance of the unified re-evaluation
    chk.section("Supp S17: unified re-evaluation and development use of the external labels")
    rv = pub.reeval.set_index("run")
    pipeline_first = ["localization_ablation/routeB_seed62", "localization_ablation/routeE_seed62"] + \
                     [f"phase2_degradation/phase2_pipeline_seed{s}" for s in SEEDS]
    others = [k for k in rv.index if k not in pipeline_first]
    chk.num("S17: changed predictions in the runs first scored by the pipeline evaluator, fewest", "139",
            int(rv.loc[pipeline_first, "n_changed"].min()), "supp")
    chk.num("S17: changed predictions in the runs first scored by the pipeline evaluator, most", "175",
            int(rv.loc[pipeline_first, "n_changed"].max()), "supp")
    chk.true("S17: at most 51 changed predictions in the other runs", int(rv.loc[others, "n_changed"].max()) <= 51,
             str(int(rv.loc[others, "n_changed"].max())))
    chk.count("S17: largest change among the other runs", "51", int(rv.loc[others, "n_changed"].max()), "supp")
    pre = {r: pub.seeds("localization_ablation_preunified", f"route{r}_seed{{s}}") for r in "BCDE"}
    pre_p2 = pub.seeds("phase2_degradation_preunified", "phase2_pipeline_seed{s}")
    pre_p2d = pub.seeds("phase2_degradation_preunified", "phase2_pipeline_seed{s}", detected_only=True)
    mE0, mE1 = ms([x.macro_f1() for x in pre["E"]])[0], rmean["E"][0]
    chk.num("S17: route E macro-F1 before the unified re-evaluation", "0.8842", mE0, "supp")
    chk.num("S17: route E macro-F1 after", "0.8817", mE1, "supp")
    chk.num("S17: route E change", "-0.0025", mE1 - mE0, "supp")
    for lab, a, b in (("coverage-aware", pre_p2, p2s), ("detected-only", pre_p2d, p2d)):
        chk.num(f"S17: Phase-2 {lab} macro-F1 change", "+0.002",
                ms([x.macro_f1() for x in b])[0] - ms([x.macro_f1() for x in a])[0], "supp")
    chk.true("S17: macro-F1 seed means of routes B, C and D changed by less than 0.001",
             all(abs(rmean[r][0] - ms([x.macro_f1() for x in pre[r]])[0]) < 0.001 for r in "BCD"))
    chk.num("S17: route E Promyelocyte F1 before", "0.698", np.mean([x.f1("Promyelocyte") for x in pre["E"]]), "supp")
    chk.num("S17: route E Promyelocyte F1 after", "0.680", np.mean([x.f1("Promyelocyte") for x in route["E"]]), "supp")
    chk.num("S17: Phase-2 balanced accuracy before", "0.786", np.mean([x.balanced_accuracy() for x in pre_p2]), "supp")
    chk.num("S17: Phase-2 balanced accuracy after", "0.795", np.mean([x.balanced_accuracy() for x in p2s]), "supp")
    pol = {p: ms([x.macro_f1() for x in pub.seeds("augmentation_policy", f"swin_{p}_seed{{s}}")])[0] for p in POLICIES}
    pol["color_only"] = ext_means["Swin-B (colour aug + CW-CE; proposed)"]
    chk.num("S17: six augmentation policies, lowest (stain-blur-noise)", "0.773", pol["stain_blur_noise"], "supp")
    chk.num("S17: six augmentation policies, highest (colour-only)", "0.863", pol["color_only"], "supp")
    chk.true("S17: stain-blur-noise lowest and colour-only highest of the six",
             min(pol, key=pol.get) == "stain_blur_noise" and max(pol, key=pol.get) == "color_only")
    chk.num("S17: that baseline (Swin-B CW-CE, default augmentation)", "0.856", mean["Swin-B CW-CE, default augmentation"], "supp")
    chk.num("S17: same checkpoints without flip averaging", "0.860",
            ms([x.macro_f1() for x in pub.seeds("tta_ablation", "swin_proposed_noflip_seed{s}")])[0], "supp")

    # ------------------------------------------------------------ Abstract and overview statements
    chk.section("Abstract and Results overview")
    chk.num("Abstract: full-pipeline macro-F1", "0.879", ms([x.macro_f1() for x in B])[0], "ms")
    chk.num("Abstract: full-pipeline macro-F1 sd", "0.006", ms([x.macro_f1() for x in B])[1], "ms")
    chk.num("Abstract: no-localization macro-F1", "0.627", ms([x.macro_f1() for x in route["noloc"]])[0], "ms")
    chk.num("Abstract: no-localization macro-F1 sd", "0.039", ms([x.macro_f1() for x in route["noloc"]])[1], "ms")
    chk.num("Abstract: Cellpose keep-background", "0.867", rmean["C"][0], "ms")
    chk.num("Abstract: Cellpose background masked", "0.760", rmean["D"][0], "ms")
    chk.num("Abstract: pre-cropped Swin-B", "0.863", ext_means["Swin-B (colour aug + CW-CE; proposed)"], "ms")
    chk.num("Abstract: Phase-2 macro-F1", "0.769", ms([x.macro_f1() for x in p2s])[0], "ms")
    chk.num("Abstract: Phase-2 macro-F1 sd", "0.002", ms([x.macro_f1() for x in p2s])[1], "ms")
    chk.true("Overview: the full-pipeline endpoint lies inside 0.863--0.893",
             0.8625 <= ms([x.macro_f1() for x in B])[0] < 0.8935)


def check_frozen_probe(chk: Checker, pub: Public) -> None:
    chk.section("Supp S7: frozen DinoBloom-B linear probe (grid of fifteen configurations)")
    g = pd.read_csv(EV / "frozen_probe" / "grid.csv.gz")
    chk.count("probe grid cells", "15", len(g), "supp")
    sel = g.loc[g.val_macro_f1_9.idxmax()]
    chk.true("selected configuration (highest validation macro-F1): unweighted, C = 0.01",
             sel.weighting == "none" and float(sel.C) == 0.01 and float(sel.C) == g.C.min())
    probe = pub.scores("frozen_probe/dinobloom_probe")
    chk.true("selected probe's external metrics equal the probe's confusion matrix (6 decimals)",
             all(f"{a:.6f}" == f"{b:.6f}" for a, b in ((sel.ext_macro_f1, probe.macro_f1()),
                                                        (sel.ext_weighted_f1, probe.weighted_f1()),
                                                        (sel.ext_balanced_acc, probe.balanced_accuracy()))))
    chk.num("probe internal macro-F1 (grid)", "0.938", float(sel.test_macro_f1_9), "supp")
    chk.num("probe internal macro-F1 (deposited matrix)", "0.938", Scores(internal("dinobloom_probe")).macro_f1(NINE), "supp")
    for w, mf, ba in (("inverse_sqrt", "0.789", "0.833"), ("inverse_frequency", "0.757", "0.838")):
        sub = g[g.weighting == w]
        best = sub.loc[sub.val_macro_f1_9.idxmax()]
        chk.true(f"{w} probe: validation-selected C = 0.01", float(best.C) == 0.01)
        chk.num(f"{w} probe: external macro-F1", mf, float(best.ext_macro_f1), "supp")
        chk.num(f"{w} probe: external balanced accuracy", ba, float(best.ext_balanced_acc), "supp")


# --------------------------------------------------------------------------- tables
def _fmt4(x: float) -> str:
    return f"{x:.4f}"


def check_tables(chk: Checker, pub: Public, int_mean: dict) -> None:
    chk.section("analysis/tables/: cited tables against the paper and the evidence")
    # S9
    t = pd.read_csv(TABLES / "external_macrof1_bootstrap_ci.csv").set_index("backbone")
    for row, ens, pe, lo, hi in (("Swin (proposed)", "swin_proposed", "0.893", "0.874", "0.911"),
                                 ("VGG16-BN", "vgg16bn", "0.841", "0.820", "0.860"),
                                 ("ResNet50", "resnet50", "0.751", "0.737", "0.766"),
                                 ("EfficientNet-B0", "efficientnet_b0", "0.848", "0.830", "0.865")):
        chk.from_table(f"S9 {row}: ensemble macro-F1 (cited table)", pe, t.loc[row, "ensemble_macro_f1"], "supp")
        chk.from_table(f"S9 {row}: 95% CI low (cited table)", lo, t.loc[row, "ci95_low"], "supp")
        chk.from_table(f"S9 {row}: 95% CI high (cited table)", hi, t.loc[row, "ci95_high"], "supp")
        got = f"{float(t.loc[row, 'ensemble_macro_f1']):.4f}"
        want = _fmt4(Scores(pub.ens[(ens, "probability")]).macro_f1())
        chk.true(f"S9 table file {row}: ensemble column recomputes from ensemble_confusion (4 decimals)",
                 got == want, f"{got} vs {want}")
        chk.row(f"S9 {row} as printed: ensemble and 95% CI", f"{pe} & {lo}--{hi}", "supp")
    lo = float(t.loc["Swin (proposed)", "ci95_low"])
    cnn_hi = [float(t.loc[r, "ci95_high"]) for r in ("VGG16-BN", "ResNet50", "EfficientNet-B0")]
    chk.from_table("Section 3.3: highest CNN ensemble upper bound (<= 0.865)", "0.865", max(cnn_hi), "ms")
    chk.true("Section 3.3: every CNN interval lies below the Swin interval (non-overlapping)", all(h < lo for h in cnn_hi))
    chk.true("S9 footnote: VGG16-BN and EfficientNet-B0 ensembles lie inside each other's 95% interval",
             t.loc["EfficientNet-B0", "ci95_low"] <= t.loc["VGG16-BN", "ensemble_macro_f1"] <= t.loc["EfficientNet-B0", "ci95_high"]
             and t.loc["VGG16-BN", "ci95_low"] <= t.loc["EfficientNet-B0", "ensemble_macro_f1"] <= t.loc["VGG16-BN", "ci95_high"])
    # S10
    m = pd.read_csv(TABLES / "master_external_results.csv").set_index("metric")
    want = master_external_table(pub)
    chk.true("tables/master_external_results.csv (S10 source) recomputes cell for cell (4 decimals)",
             m.astype(str).equals(want.astype(str)), "" if m.astype(str).equals(want.astype(str)) else "differs")
    # S12
    a = pd.read_csv(TABLES / "detector_abstention_quality_test.csv").set_index("group")
    for grp, label, vals in (("missed_by_yolo", "missed", ("348", "0.727", "0.429", "0.500", "0.566", "0.100", "0.895", "0.000")),
                             ("detected_class_matched_control", "class-matched control",
                              ("348", "0.862", "0.714", "0.250", "0.723", "0.633", "0.977", "0.000"))):
        r = a.loc[grp]
        chk.count(f"S12 table file, {label}: n", vals[0], int(r.n), "supp")
        for col, printed in zip(("accuracy", "Blast_Group", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil", "Promyelocyte"), vals[1:]):
            chk.num(f"S12 table file, {label}: {col}", printed, float(r[col]), "supp")
    diff = float(a.loc["missed_by_yolo", "accuracy"] - a.loc["detected_class_matched_control", "accuracy"])
    chk.num("S12: accuracy difference", "-0.135", diff, "supp")
    chk.num("S12: accuracy points harder", "13.5", -100 * diff, "supp")
    chk.num("Section 3.2 / S12: abstained cells the classifier would label correctly", "73%",
            100 * float(a.loc["missed_by_yolo", "accuracy"]), "ms")
    chk.num("Section 3.2: missed-cell accuracy (0.727 versus 0.862)", "0.727", float(a.loc["missed_by_yolo", "accuracy"]), "ms")
    chk.num("Section 3.2: control accuracy (0.727 versus 0.862)", "0.862", float(a.loc["detected_class_matched_control", "accuracy"]), "ms")
    # Refreshed aggregate tables: every value recomputes from the public evidence
    for name, fn in (("ae_crop_ablation_3seed_summary", ae_summary_table), ("ae_crop_ablation_perseed", ae_perseed_table),
                     ("nocrop_baseline_perseed", nocrop_perseed_table), ("nocrop_vs_crop_summary", nocrop_vs_crop_table),
                     ("phase1_vs_phase2_degradation", phase_table), ("backbone_comparison_pm_sd", backbone_pm_sd_table),
                     ("matched_recipe_control_perseed", matched_recipe_table)):
        shipped = pd.read_csv(TABLES / f"{name}.csv", dtype=str, keep_default_na=False)
        want = fn(pub).astype(str)
        same = list(shipped.columns) == list(want.columns) and shipped.values.tolist() == want.values.tolist()
        chk.true(f"tables/{name}.csv recomputes from the public evidence", same)


# Table builders (also used by the authors' build step to write the files)
def _route_scores(pub: Public) -> dict:
    return {"A": pub.seeds("backbone_comparison", "swin_proposed_seed{s}"),
            **{r: pub.seeds("localization_ablation", f"route{r}_seed{{s}}") for r in "BCDE"},
            "NOCROP": pub.seeds("localization_ablation", "nocrop_swin_seed{s}")}


ROUTE_NAMES = {"A": "original crop -> Swin (classifier reference, n=8061)",
               "B": "no-crop -> YOLO -> Swin (deployable pipeline, coverage-aware, n=8080)",
               "C": "no-crop -> Cellpose keep-bg -> Swin (n=8080)",
               "D": "no-crop -> Cellpose no-bg -> Swin (n=8080)",
               "E": "Cellpose keep-bg ROI -> YOLO refine -> Swin (n=8080)"}


def ae_summary_table(pub: Public) -> pd.DataFrame:
    rs, rows = _route_scores(pub), []
    for r in "ABCDE":
        sc = rs[r]
        row = dict(route=r, route_name=ROUTE_NAMES[r])
        for col, fn in (("accuracy", lambda x: x.accuracy()), ("bal_acc", lambda x: x.balanced_accuracy()),
                        ("macro_f1", lambda x: x.macro_f1()), ("blast_f1", lambda x: x.f1("Blast_Group")),
                        ("promy_f1", lambda x: x.f1("Promyelocyte"))):
            m, sd = ms([fn(x) for x in sc])
            row[f"{col}_mean"], row[f"{col}_sd"] = _fmt4(m), _fmt4(sd)
        rows.append(row)
    return pd.DataFrame(rows)


def ae_perseed_table(pub: Public) -> pd.DataFrame:
    rs, rows = _route_scores(pub), []
    for r in "ABCDE":
        for s, x in zip(SEEDS, rs[r]):
            rows.append(dict(route=r, route_name=ROUTE_NAMES[r], seed=s, accuracy=f"{x.accuracy():.6f}",
                             balanced_accuracy=f"{x.balanced_accuracy():.6f}", macro_f1=f"{x.macro_f1():.6f}",
                             blast_f1=f"{x.f1('Blast_Group'):.6f}", promy_f1=f"{x.f1('Promyelocyte'):.6f}",
                             min_recall=f"{x.min_recall():.6f}"))
    return pd.DataFrame(rows)


def nocrop_perseed_table(pub: Public) -> pd.DataFrame:
    return pd.DataFrame([dict(route="NOCROP", seed=s, accuracy=_fmt4(x.accuracy()), balanced_accuracy=_fmt4(x.balanced_accuracy()),
                              macro_f1=_fmt4(x.macro_f1()), blast_f1=_fmt4(x.f1("Blast_Group")),
                              promy_f1=_fmt4(x.f1("Promyelocyte")))
                         for s, x in zip(SEEDS, _route_scores(pub)["NOCROP"])])


def nocrop_vs_crop_table(pub: Public) -> pd.DataFrame:
    rs, rows = _route_scores(pub), []
    for r, desc in (("NOCROP", "full 368 field -> Swin (no localization, n=8080)"),
                    ("B", "no-crop field -> YOLO -> Swin (deployable, coverage-aware, n=8080)"),
                    ("A", "pre-cropped by the same detector (classifier reference, n=8061)")):
        sc = rs[r]
        m, sd = ms([x.macro_f1() for x in sc])
        rows.append(dict(route=r, description=desc, macro_f1_mean=_fmt4(m), macro_f1_sd=_fmt4(sd),
                         bal_acc_mean=_fmt4(np.mean([x.balanced_accuracy() for x in sc])),
                         blast_f1_mean=_fmt4(np.mean([x.f1("Blast_Group") for x in sc])),
                         promy_f1_mean=_fmt4(np.mean([x.f1("Promyelocyte") for x in sc]))))
    return pd.DataFrame(rows)


def phase_table(pub: Public) -> pd.DataFrame:
    rows = []
    for split, cond, sc in (("phase1", "clean field->YOLO->Swin, coverage-aware (n=8080; coverage 99.99%)",
                             pub.seeds("localization_ablation", "routeB_seed{s}")),
                            ("phase2_coverage_aware", "partly degraded; 348/5196 no-detections counted as errors; headline (coverage 93.3%)",
                             pub.seeds("phase2_degradation", "phase2_pipeline_seed{s}")),
                            ("phase2_detected_only", "partly degraded; detected cells only (n=4848); optimistic",
                             pub.seeds("phase2_degradation", "phase2_pipeline_seed{s}", detected_only=True))):
        m, sd = ms([x.macro_f1() for x in sc])
        rows.append(dict(split=split, condition=cond, macro_f1_mean=_fmt4(m), macro_f1_sd=_fmt4(sd),
                         bal_acc_mean=_fmt4(np.mean([x.balanced_accuracy() for x in sc])),
                         acc_mean=_fmt4(np.mean([x.accuracy() for x in sc])), n_seeds=len(sc)))
    return pd.DataFrame(rows)


def backbone_pm_sd_table(pub: Public) -> pd.DataFrame:
    rows = []
    for label, name in (("Swin (proposed: +color aug +CW-CE)", "Swin-B (colour aug + CW-CE; proposed)"),
                        ("VGG16-BN", "VGG16-BN (CW-CE)"), ("ResNet50", "ResNet50 (CW-CE)"),
                        ("EfficientNet-B0", "EfficientNet-B0 (CW-CE)")):
        iv = [Scores(internal(r)).macro_f1(NINE) for r in INTERNAL_ROWS[name][0]]
        tmpl, seeds, _ = TABLE2_EXT[name]
        ev = seed_values(pub, tmpl, seeds, lambda x: x.macro_f1())
        rows.append(dict(backbone=label, internal9_macro_f1_mean=_fmt4(ms(iv)[0]), internal9_sd=_fmt4(ms(iv)[1]),
                         internal_n=len(iv), external7_macro_f1_mean=_fmt4(ms(ev)[0]), external7_sd=_fmt4(ms(ev)[1]),
                         external_n=len(ev), external_per_seed="|".join(_fmt4(v) for v in ev)))
    return pd.DataFrame(rows)


def matched_recipe_table(pub: Public) -> pd.DataFrame:
    rows = []
    for backbone, recipe, ext, intl in (("Swin-B", "BPaCo (AdamW)", "matched_recipe/swinbpaco_seed{s}", "swin_bpaco_seed{s}"),
                                        ("ResNet50", "BPaCo (AdamW, shared protocol)", "matched_recipe/rn50bpaco_adamw_seed{s}", "resnet50_bpaco_adamw_seed{s}"),
                                        ("ResNet50", "BPaCo (SGD)", "matched_recipe/rn50bpaco_sgd_seed{s}", "resnet50_bpaco_sgd_seed{s}")):
        for s in SEEDS:
            x = pub.scores(ext.format(s=s))
            rows.append(dict(backbone=backbone, recipe=recipe, seed=s,
                             internal_macro_f1=_fmt4(Scores(internal(intl.format(s=s))).macro_f1(NINE)),
                             external_macro_f1=_fmt4(x.macro_f1()), external_n=x.n(), tta="hflip"))
    return pd.DataFrame(rows)


def master_external_table(pub: Public) -> pd.DataFrame:
    out = OrderedDict()
    for col, name in (("Swin (proposed)", "Swin-B (colour aug + CW-CE; proposed)"), ("VGG16-BN", "VGG16-BN (CW-CE)"),
                      ("ResNet50", "ResNet50 (CW-CE)"), ("EfficientNet-B0", "EfficientNet-B0 (CW-CE)")):
        tmpl, seeds, _ = TABLE2_EXT[name]
        sc = [pub.scores(tmpl.format(s=s)) for s in seeds]
        d = OrderedDict()
        for k, fn in (("macro_f1", lambda x: x.macro_f1()), ("weighted_f1", lambda x: x.weighted_f1()),
                      ("balanced_acc", lambda x: x.balanced_accuracy())):
            m, sd = ms([fn(x) for x in sc])
            d[k] = f"{m:.4f}\u00b1{sd:.4f}"
        d["n_seeds"] = str(len(sc))
        for c in CLS7:
            d[f"F1_{c}"] = _fmt4(np.mean([x.f1(c) for x in sc]))
        out[col] = d
    df = pd.DataFrame(out)
    df.index.name = "metric"
    return df


# =========================================================================== integrity
WBC_ID = re.compile(r"(?<![A-Za-z0-9_])\d{8}(?:__det\d+)?\.(?:jpg|png|jpeg|bmp|tif)", re.I)


def check_integrity(chk: Checker) -> None:
    chk.section("Archive integrity: MANIFEST.csv, public files, no WBCBench per-image material")
    man = pd.read_csv(EV / "MANIFEST.csv", dtype=str, keep_default_na=False)
    pub_rows = man[man.release == "public"]
    wh_rows = man[man.release == "withheld_wbcbench"]
    chk.true("MANIFEST: release is public or withheld_wbcbench on every row",
             set(man.release) <= {"public", "withheld_wbcbench"})
    chk.true("MANIFEST: status is current or superseded on every row", set(man.status) <= {"current", "superseded"})
    hashes = {f: sha256(EV / f) for f in set(man.file) if f and (EV / f).is_file()}
    bad = [f"{r.group}/{r.run_id}" for r in man.itertuples() if r.file and hashes.get(r.file) != r.sha256]
    chk.true(f"MANIFEST: every listed public file exists with its SHA-256 ({len(hashes)} files, {len(man)} rows)",
             not bad, ", ".join(bad[:4]))
    chk.true("MANIFEST: every public row names a public file", (pub_rows.file != "").all())
    present = [r.withheld_file for r in wh_rows.itertuples() if not r.withheld_file or (EV / r.withheld_file).exists()]
    chk.true(f"MANIFEST: none of the {len(wh_rows)} withheld per-image files is in the public tree", not present,
             ", ".join(present[:4]))
    listed = set(man.file)
    on_disk = {str(p.relative_to(EV)) for p in EV.rglob("*") if p.is_file() and p.name != "MANIFEST.csv"}
    stray = sorted(on_disk - listed)
    chk.true("every file under analysis/evidence/ is listed in MANIFEST.csv", not stray, ", ".join(stray[:4]))
    registry = {r.key for r in RUNS}
    rc = set(pd.read_csv(WBC / "run_confusion.csv.gz").run)
    chk.true("every WBCBench run in the registry has public counts and a MANIFEST row",
             registry == rc and registry <= set(man.group + "/" + man.run_id))
    hits = []
    for p in list(EV.rglob("*")) + list(TABLES.rglob("*")):
        if p.is_file() and p.suffix in (".gz", ".csv"):
            txt = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
            if WBC_ID.search(txt):
                hits.append(str(p.relative_to(ANALYSIS)))
    chk.true("no WBCBench-style image identifier (8-digit file name) in any public evidence or table file",
             not hits, ", ".join(hits[:4]))
    forbidden = [p for p in ANALYSIS.rglob("*") if p.is_file() and
                 (p.name.startswith(("error_review", "blast_promy_review", "first_hematologist", "second_hematologist"))
                  or {"predictions", "hematologist_review"} & set(p.relative_to(ANALYSIS).parts))]
    chk.true("no hematologist verdict file, review export or analysis/predictions/ folder in the public tree",
             not forbidden, ", ".join(str(p) for p in forbidden[:4]))


# =========================================================================== private checks
def private_checks(chk: Checker, priv: Private | None, pub: Public) -> None:
    reason = "needs --private-evidence (WBCBench per-image data)"

    # ---------------------------------------------------------------- manifest and counts
    chk.section("Private evidence: withheld runs present, MANIFEST metrics and count files rebuild")
    man = pd.read_csv(EV / "MANIFEST.csv", dtype=str, keep_default_na=False)
    wh = man[man.release == "withheld_wbcbench"]
    count_files = ["run_confusion", "ensemble_confusion", "mcnemar_discordant", "ensemble_agreement",
                   "calibration_bins", "threshold_sweep", "field_counts", "reeval_changes"]
    if priv is None:
        chk.skip("private folder: SHA256SUMS.txt verifies", reason)
        chk.skip(f"withheld per-image files present ({len(wh)} MANIFEST rows)", reason)
        chk.skip("MANIFEST n_eval and macro_f1 recompute from the per-image files", reason)
        for name in count_files:
            chk.skip(f"wbcbench_confusion/{name}.csv.gz rebuilds from per-image predictions", reason)
        chk.skip("the three Phase-2 detection files agree field by field", reason)
    else:
        sums = priv.top / "SHA256SUMS.txt"
        if not sums.exists():
            chk.skip("private folder: SHA256SUMS.txt verifies", "the private folder has no SHA256SUMS.txt")
        else:
            bad = []
            for line in sums.read_text().splitlines():
                h, rel = line.split(maxsplit=1)
                if not (priv.top / rel).is_file() or sha256(priv.top / rel) != h:
                    bad.append(rel)
            chk.true("private folder: SHA256SUMS.txt verifies", not bad, ", ".join(bad[:3]))
        missing = [r.withheld_file for r in wh.itertuples() if not priv.resolve(r.withheld_file).exists()]
        if missing:
            chk.skip(f"withheld per-image files present ({len(wh)} MANIFEST rows)",
                     f"{len(missing)} of {len(wh)} not in the private folder; their checks are skipped")
        else:
            chk.true(f"withheld per-image files present ({len(wh)} MANIFEST rows)", True)
        mism, n_read = [], 0
        for r in wh.itertuples():
            p = priv.resolve(r.withheld_file)
            if not p.exists() or not r.n_eval:
                continue
            n_read += 1
            d = _normalise_private(pd.read_csv(p))
            if str(len(d)) != r.n_eval:
                mism.append(f"{r.group}/{r.run_id} n")
            if r.macro_f1 and "pred_label" in d:
                if f"{Scores(confusion(d.true_label, d.pred_label)).macro_f1():.6f}" != r.macro_f1:
                    mism.append(f"{r.group}/{r.run_id} macro_f1")
        n_rows = int((wh.n_eval != "").sum())
        if n_read == 0:
            chk.skip("MANIFEST n_eval and macro_f1 recompute from the per-image files",
                     "none of the withheld per-image files is in the private folder")
        else:
            chk.true("MANIFEST n_eval and macro_f1 recompute from the per-image files", not mism,
                     ", ".join(mism[:4]) or f"{n_read} of {n_rows} rows recomputed")
        rebuilt = rebuild_counts(priv)
        keys = {"run_confusion": ["run"], "ensemble_confusion": ["ensemble", "rule"],
                "mcnemar_discordant": ["comparison", "rule"], "ensemble_agreement": ["ensemble"],
                "calibration_bins": ["ensemble"], "threshold_sweep": ["ensemble"], "field_counts": ["split"],
                "reeval_changes": ["run"]}
        for name in count_files:
            label = f"wbcbench_confusion/{name}.csv.gz rebuilds from per-image predictions"
            if name not in rebuilt:
                chk.skip(label, "the private folder lacks the inputs for this file")
                continue
            new, shipped = rebuilt[name], read_counts(name)
            k = keys[name]
            have = set(map(tuple, new[k].astype(str).values))
            sub = shipped[[tuple(r) in have for r in shipped[k].astype(str).values]]
            total = len(set(map(tuple, shipped[k].astype(str).values)))
            ok = _frames_equal(sub.reset_index(drop=True), new.reset_index(drop=True))
            bad = []
            if not ok and name == "run_confusion":
                for run, g in new.groupby("run", sort=False):
                    if not _frames_equal(sub[sub.run == run].reset_index(drop=True), g.reset_index(drop=True)):
                        bad.append(run)
            chk.true(label + f" ({len(have)} of {total} entries)", ok, ", ".join(bad[:4]) if bad else "")
        if all(priv.has(f"detections/phase2_seed{s}_detections") for s in SEEDS):
            dets = [priv.load(f"detections/phase2_seed{s}_detections").set_index("image_id").det_status for s in SEEDS]
            chk.true("the three Phase-2 detection files agree field by field", all(d.equals(dets[0]) for d in dets))
        else:
            chk.skip("the three Phase-2 detection files agree field by field", "the private folder lacks a Phase-2 detection file")

    # ---------------------------------------------------------------- S9 bootstrap
    chk.section("Supp S9 (private): image-level bootstrap of the ensemble macro-F1 (2,000 replicates)")
    S9 = (("swin_proposed", "Swin (proposed)", "0.874", "0.911"), ("vgg16bn", "VGG16-BN", "0.820", "0.860"),
          ("resnet50", "ResNet50", "0.737", "0.766"), ("efficientnet_b0", "EfficientNet-B0", "0.830", "0.865"))
    t = pd.read_csv(TABLES / "external_macrof1_bootstrap_ci.csv").set_index("backbone")
    for name, row, lo, hi in S9:
        keys = dict(ENSEMBLES_BY_NAME)[(name, "probability")]
        if priv is not None and not all(priv.has(k) for k in keys):
            for lab, printed in ((f"S9 {row}: bootstrap 95% CI low", lo), (f"S9 {row}: bootstrap 95% CI high", hi),
                                 (f"S9 {row}: CI equals tables/external_macrof1_bootstrap_ci.csv (4 decimals)", None)):
                chk.skip(lab, "the private folder lacks this ensemble's per-image files", printed, "supp")
            continue
        if priv is None:
            chk.skip(f"S9 {row}: bootstrap 95% CI low", reason, lo, "supp")
            chk.skip(f"S9 {row}: bootstrap 95% CI high", reason, hi, "supp")
            chk.skip(f"S9 {row}: CI equals tables/external_macrof1_bootstrap_ci.csv (4 decimals)", reason)
            continue
        y, p, _ = ensemble_labels(priv, "probability", keys)
        rng = np.random.default_rng(0)                  # a fresh generator per model, seed 0
        n = len(y)
        vals = np.empty(2000)
        for b in range(2000):
            idx = rng.integers(0, n, n)
            vals[b] = _macro7(y[idx], p[idx])
        l, h = np.percentile(vals, [2.5, 97.5])
        chk.num(f"S9 {row}: bootstrap 95% CI low", lo, l, "supp")
        chk.num(f"S9 {row}: bootstrap 95% CI high", hi, h, "supp")
        chk.true(f"S9 {row}: CI equals tables/external_macrof1_bootstrap_ci.csv (4 decimals)",
                 (f"{l:.4f}", f"{h:.4f}") == (f"{t.loc[row, 'ci95_low']:.4f}", f"{t.loc[row, 'ci95_high']:.4f}"))

    # ---------------------------------------------------------------- S7 bootstrap
    chk.section("Supp S7 (private): paired image-level bootstrap (5,000 replicates), TOST at +/-0.02")
    S7 = {"bpaco_shared_adamw": ("+0.044", "+0.079", "+0.047", "+0.076", "not equivalent"),
          "bpaco_adamw_vs_sgd": ("-0.006", "+0.026", "-0.003", "+0.023", "not established"),
          "proposed_vs_dinobloom": ("+0.015", "+0.046", "+0.018", "+0.044", "not established")}
    labels = []                                     # (label, printed value; None for a yes/no check)
    for comp, printed in S7.items():
        labels += list(zip((f"S7 {comp}: 95% CI low", f"S7 {comp}: 95% CI high", f"S7 {comp}: 90% CI low",
                            f"S7 {comp}: 90% CI high", f"S7 {comp}: TOST verdict"), printed))
    labels += [("S7 footnote: every DinoBloom-B 95% interval excludes zero under the three tie rules", None),
               ("S7 footnote: probability-ensemble 95% CI low", "+0.016"),
               ("S7 footnote: probability-ensemble 95% CI high", "+0.043"),
               ("Section 3.4: main-text CI for the shared-protocol difference [+0.044,+0.079]", None)]
    S7P = {"bpaco_shared_adamw": ("<0.001", "+0.062"), "bpaco_adamw_vs_sgd": ("0.21", "+0.010"),
           "proposed_vs_dinobloom": ("<0.001", "+0.031")}   # checked in check_wbcbench from the public counts
    for comp, (l95, h95, l90, h90, verdict) in S7.items():
        chk.row(f"S7 {comp} as printed: p, difference, intervals, verdict",
                f"{S7P[comp][0]} & {S7P[comp][1]} & [{l95},{h95}] & [{l90},{h90}] & {verdict}", "supp")
    chk.row("S7 footnote as printed: probability-ensemble 95% CI", "[+0.016,+0.043]", "supp")
    needed = {k for n, r, ks in ENSEMBLES if n in ("swin_bpaco", "rn50_bpaco_adamw", "rn50_bpaco_sgd",
                                                   "swin_proposed", "dinobloom_ft") for k in ks}
    order_ok = priv is not None and priv.has("wbcbench_labels/phase1_raw_labels") and all(priv.has(k) for k in needed)
    if not order_ok:
        for lab, printed in labels:
            chk.skip(lab, reason if priv is None else
                     "the private folder lacks wbcbench_labels/phase1_raw_labels or a member run of the S7 ensembles",
                     printed, "supp")
    else:
        boot = _s7_bootstrap(priv)
        for comp, (l95, h95, l90, h90, verdict) in S7.items():
            r = boot[(comp, "vote_tie42")]
            chk.num(f"S7 {comp}: 95% CI low", l95, r[0], "supp")
            chk.num(f"S7 {comp}: 95% CI high", h95, r[1], "supp")
            chk.num(f"S7 {comp}: 90% CI low", l90, r[2], "supp")
            chk.num(f"S7 {comp}: 90% CI high", h90, r[3], "supp")
            if r[2] > -0.02 and r[3] < 0.02:
                got = "equivalent"
            elif r[2] >= 0.02 or r[3] <= -0.02:
                got = "not equivalent"
            else:
                got = "not established"
            chk.text(f"S7 {comp}: TOST verdict", verdict, got, "supp")
        chk.true("S7 footnote: every DinoBloom-B 95% interval excludes zero under the three tie rules",
                 all(boot[("proposed_vs_dinobloom", f"vote_tie{j}")][0] > 0 for j in SEEDS))
        rp = boot[("proposed_vs_dinobloom", "probability")]
        chk.num("S7 footnote: probability-ensemble 95% CI low", "+0.016", rp[0], "supp")
        chk.num("S7 footnote: probability-ensemble 95% CI high", "+0.043", rp[1], "supp")
        r = boot[("bpaco_shared_adamw", "vote_tie42")]
        chk.true("Section 3.4: main-text CI for the shared-protocol difference [+0.044,+0.079]",
                 (f"{r[0]:+.3f}", f"{r[1]:+.3f}") == ("+0.044", "+0.079"))

    # ---------------------------------------------------------------- S13 two-hematologist review
    chk.section("Supp S13 (private): two-hematologist review of the 205-cell panel")
    files13 = {"first": "hematologist_review/first_hematologist_verdicts",
               "second": "hematologist_review/second_hematologist_verdicts"}
    verdict13 = {"model_right": "R", "model_wrong": "W", "ambiguous": "A"}
    name13 = {"R": "model right", "W": "model wrong", "A": "ambiguous"}
    types13 = ("Blast_FN", "Blast_FP", "Promy_FN", "Promy_FP")
    # paired outcome codes; "RW" counts cells rated R by one hematologist and W by the other, either way round
    pairs13 = {"RR": ("R-R", "both model right", "Abstract / Introduction / Section 3.9 / Discussion"),
               "WW": ("W-W", "both model wrong", "Abstract / Introduction / Section 3.9 / Discussion"),
               "AA": ("A-A", "both ambiguous", "Abstract / Introduction / Section 3.9 / Discussion / S13 reading"),
               "RW": ("R-W", "one model right, the other model wrong", "Section 3.9 / S13 reading"),
               "RA": ("R-A", "one model right, the other ambiguous", ""),
               "WA": ("W-A", "one model wrong, the other ambiguous", "")}
    # printed values: (a) R / W / A of each hematologist by type and over all types, (b) the first hematologist's
    # verdict (rows) by the second's (columns), (c) paired outcomes by type and over all types
    s13a = {"first": {"Blast_FN": "57 30 6", "Blast_FP": "8 73 7", "Promy_FN": "7 2 0", "Promy_FP": "5 9 1",
                      "all": "77 114 14"},
            "second": {"Blast_FN": "12 72 9", "Blast_FP": "69 12 7", "Promy_FN": "0 9 0", "Promy_FP": "12 2 1",
                       "all": "93 95 17"}}
    s13a_pct = {"first": "37.6% 55.6% 6.8%", "second": "45.4% 46.3% 8.3%"}
    s13b = {"R": "21 50 6", "W": "69 43 2", "A": "3 2 9"}
    s13c = {"Blast_FN": "11 29 4 42 5 2", "Blast_FP": "7 11 5 61 2 2", "Promy_FN": "0 2 0 7 0 0",
            "Promy_FP": "3 1 0 9 2 0", "all": "21 43 9 119 9 4"}
    s13c_pct = "10.2% 21.0% 4.4% 58.0% 4.4% 2.0%"
    plan = []           # (method, label, printed, paper, value from the derived counts D)

    def add13(method, label, printed, paper, get):
        plan.append((method, label, printed, paper, get))
    for w in ("first", "second"):
        add13("count", f"S13: {w} hematologist's verdict rows", "205", "supp", lambda D, w=w: D["rows"][w])
    add13("true", "S13: both verdict files hold the same cells, pre-filled columns and valid verdicts", None, "",
          lambda D: (D["same"], ""))
    add13("true", "S13: the reviewed cells are the model's own 205 panel cells (joint multiset)", None, "",
          lambda D: (D["panel_ok"], ""))
    for w in ("first", "second"):
        for et in (*types13, "all"):
            for c, printed in zip("RWA", s13a[w][et].split()):
                where = "S13a / S13 reading" if (et == "all" and c == "R") else "S13a"
                add13("count", f"{where}: {w} hematologist, {'all types' if et == 'all' else et}, {name13[c]}",
                      printed, "supp", lambda D, k=(w, et, c): D["single"][k])
        for c, printed in zip("RWA", s13a_pct[w].split()):
            where = "S13a / S13 reading" if c == "R" else "S13a"
            add13("num", f"{where}: {w} hematologist, all types, {name13[c]} (%)", printed, "supp",
                  lambda D, k=(w, "all", c): 100 * D["single"][k] / D["n"])
    for x in "RWA":
        for y, printed in zip("RWA", s13b[x].split()):
            add13("count", f"S13b: first hematologist {name13[x]}, second {name13[y]}", printed, "supp",
                  lambda D, k=(x, y): D["cross"][k])
    add13("count", "S13 caption / Abstract / Introduction / Section 3.9 / Discussion: same verdict (S13b diagonal)",
          "73", "supp", lambda D: D["agree"])
    add13("num", "S13 caption / Abstract / Section 3.9: same verdict (%)", "35.6%", "supp",
          lambda D: 100 * D["agree"] / D["n"])
    for et in (*types13, "all"):
        for p, printed in zip(pairs13, s13c[et].split()):
            code, desc, where = pairs13[p]
            label = (f"S13c{' / ' + where if where else ''}: all types, {desc} ({code})" if et == "all"
                     else f"S13c: {et}, {code}")
            add13("count", label, printed, "supp", lambda D, k=(et, p): D["pairs"][k])
    for p, printed in zip(pairs13, s13c_pct.split()):
        code, desc, _ = pairs13[p]
        where = " / Section 3.9" if p in ("RR", "WW", "AA", "RW") else ""
        add13("num", f"S13c{where}: all types, {desc} ({code}, %)", printed, "supp",
              lambda D, k=("all", p): 100 * D["pairs"][k] / D["n"])
    add13("count", "Section 3.9: at least one hematologist rated ambiguous", "22", "ms",
          lambda D: sum(D["pairs"][("all", p)] for p in ("AA", "RA", "WA")))
    add13("num", "Section 3.9: at least one hematologist rated ambiguous (%)", "10.7%", "ms",
          lambda D: 100 * sum(D["pairs"][("all", p)] for p in ("AA", "RA", "WA")) / D["n"])
    add13("true", "S13 reading: most R-W pairs are Blast_FN or Blast_FP cells", None, "",
          lambda D: (2 * D["rw_blast"] > D["pairs"][("all", "RW")], f"{D['rw_blast']} of {D['pairs'][('all', 'RW')]}"))

    # the printed rows of Table S13 (a)-(c), as --check-tex sees them, so that an edited cell is caught
    tex13 = {t: t.replace("_", "\\_") for t in types13}
    n13 = {"Blast_FN": "93", "Blast_FP": "88", "Promy_FN": "9", "Promy_FP": "15"}

    def row13(*cells):
        return " & ".join(str(c) for c in cells)

    def pct13(k, n):
        return f"{k} ({100 * k / n:.1f}%)"
    for et in types13:
        add13("text", f"S13a as printed: {et} row", row13(tex13[et], n13[et], *s13a["first"][et].split(),
                                                         *s13a["second"][et].split()), "supp",
              lambda D, et=et: row13(tex13[et], D["n_type"][et],
                                     *(D["single"][(w, et, c)] for w in ("first", "second") for c in "RWA")))
    add13("text", "S13a as printed: total row",
          row13("Total", "205", *(f"{k} ({p})" for w in ("first", "second")
                                  for k, p in zip(s13a[w]["all"].split(), s13a_pct[w].split()))), "supp",
          lambda D: row13("Total", D["n"], *(pct13(D["single"][(w, "all", c)], D["n"])
                                             for w in ("first", "second") for c in "RWA")))
    for x in "RWA":
        add13("text", f"S13b as printed: {x} row", row13(x, *s13b[x].split(), s13a["first"]["all"].split()["RWA".index(x)]),
              "supp", lambda D, x=x: row13(x, *(D["cross"][(x, y)] for y in "RWA"), D["single"][("first", "all", x)]))
    add13("text", "S13b as printed: total row", row13("Total", *s13a["second"]["all"].split(), "205"), "supp",
          lambda D: row13("Total", *(D["single"][("second", "all", y)] for y in "RWA"), D["n"]))
    for et in types13:
        add13("text", f"S13c as printed: {et} row", row13(tex13[et], n13[et], *s13c[et].split()), "supp",
              lambda D, et=et: row13(tex13[et], D["n_type"][et], *(D["pairs"][(et, p)] for p in pairs13)))
    add13("text", "S13c as printed: total row",
          row13("Total", "205", *(f"{k} ({p})" for k, p in zip(s13c["all"].split(), s13c_pct.split()))), "supp",
          lambda D: row13("Total", D["n"], *(pct13(D["pairs"][("all", p)], D["n"]) for p in pairs13)))

    swin_keys = dict(ENSEMBLES_BY_NAME)[("swin_proposed", "probability")]
    if priv is None or not all(priv.has(f) for f in files13.values()) or not all(priv.has(k) for k in swin_keys):
        for method, label, printed, paper, _ in plan:
            chk.skip(label, reason if priv is None else
                     "the private folder holds no hematologist verdict files (not distributed) or lacks the proposed model's runs",
                     None if method == "true" else printed, paper)
    else:
        pre = ["error_type", "benchmark_label", "model_prediction", "model_confidence"]
        a, b = (priv.load(files13[w]).set_index("review_id") for w in ("first", "second"))
        same = (a.index.is_unique and b.index.is_unique and set(a.index) == set(b.index)
                and a.loc[b.index, pre].reset_index(drop=True).equals(b[pre].reset_index(drop=True))
                and a.verdict.isin(list(verdict13)).all() and b.verdict.isin(list(verdict13)).all())
        f = a.verdict.map(verdict13).fillna("?")
        s = b.verdict.reindex(a.index).map(verdict13).fillna("?")
        pair = [x + y if x == y else "".join(sorted((x, y), key="RWA?".index)) for x, y in zip(f, s)]
        D = {"n": len(a), "rows": {"first": len(a), "second": len(b)}, "same": same,
             "single": Counter(), "cross": Counter(zip(f, s)), "pairs": Counter(),
             "agree": int((f == s).sum()), "n_type": Counter(a.error_type)}
        for et, x, y, p in zip(a.error_type, f, s, pair):
            for t in (et, "all"):
                D["single"][("first", t, x)] += 1
                D["single"][("second", t, y)] += 1
                D["pairs"][(t, p)] += 1
        D["rw_blast"] = D["pairs"][("Blast_FN", "RW")] + D["pairs"][("Blast_FP", "RW")]
        y, p, _ = ensemble_labels(priv, "probability", swin_keys)
        panel = Counter()
        for t_, p_ in zip(y, p):
            t, q = CLS7[t_], CLS7[p_]
            et = ("Blast_FN" if t == "Blast_Group" and q != "Blast_Group" else
                  "Blast_FP" if t != "Blast_Group" and q == "Blast_Group" else
                  "Promy_FN" if t == "Promyelocyte" and q != "Promyelocyte" else
                  "Promy_FP" if t != "Promyelocyte" and q == "Promyelocyte" else None)
            if et:
                panel[(et, t, q)] += 1
        rev = Counter(zip(a.error_type, a.benchmark_label, a.model_prediction))
        D["panel_ok"] = sum((panel & rev).values()) == 205 and not (rev - panel) and not (panel - rev)
        for method, label, printed, paper, get in plan:
            if method == "true":
                ok, detail = get(D)
                chk.true(label, bool(ok), detail)
            else:
                getattr(chk, method)(label, printed, get(D), paper)


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_float_dtype(x) or pd.api.types.is_float_dtype(y):
            if not np.allclose(x.astype(float), y.astype(float), rtol=0, atol=1e-6, equal_nan=True):
                return False
        elif x.astype(str).tolist() != y.astype(str).tolist():
            return False
    return True


def _macro7(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * 7 + p, minlength=49).reshape(7, 7).astype(float)
    tp = np.diag(cm)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0).mean())


def _s7_bootstrap(priv: Private) -> dict:
    """Paired bootstrap of the S7 comparisons: images in the order of their full path
    (native WBCBench class folder, then file name), numpy.random.default_rng(12345),
    rng.choice(n, n, replace=True) per replicate, percentile intervals."""
    raw = priv.load("wbcbench_labels/phase1_raw_labels").set_index("image_id").raw_label
    lab = {}
    used = {(a, f"vote_tie{j}") for c, a, b in COMPARISONS for j in SEEDS} | \
           {(b, f"vote_tie{j}") for c, a, b in COMPARISONS for j in SEEDS} | \
           {("swin_proposed", "probability"), ("dinobloom_ft", "probability")}
    for name, rule, keys in ENSEMBLES:
        if (name, rule) not in used:
            continue
        y, p, ids = ensemble_labels(priv, rule, keys)
        order = np.argsort([f"{raw[i]}/{i}" for i in ids], kind="stable")
        lab[(name, rule)] = (y[order], p[order])
    n = len(next(iter(lab.values()))[0])
    tasks = [(c, f"vote_tie{j}", a, b) for c, a, b in COMPARISONS for j in SEEDS] + \
            [("proposed_vs_dinobloom", "probability", "swin_proposed", "dinobloom_ft")]
    diffs = {t: np.empty(5000) for t in tasks}
    rng = np.random.default_rng(12345)
    idx = np.arange(n)
    for k in range(5000):
        s = rng.choice(idx, size=n, replace=True)
        for t in tasks:
            comp, rule, a, b = t
            ya, pa = lab[(a, rule)]
            yb, pb = lab[(b, rule)]
            diffs[t][k] = _macro7(ya[s], pa[s]) - _macro7(yb[s], pb[s])
    out = {}
    for (comp, rule, a, b), d in diffs.items():
        l95, h95 = np.percentile(d, [2.5, 97.5])
        l90, h90 = np.percentile(d, [5, 95])
        out[(comp, rule)] = (l95, h95, l90, h90)
    return out


# =========================================================================== tex check
def check_tex(chk: Checker, main_tex: Path, supp_tex: Path) -> int:
    def norm(t: str) -> str:
        t = re.sub(r"\\(?:mathbf|textbf|emph|mathrm)\{", "", t)
        for a, b in (("{,}", ","), ("\\%", "%"), ("\\pm", "\u00b1"), ("$", ""), ("\\,", ""), ("\u2212", "-"),
                     ("{", ""), ("}", ""), ("~", " ")):
            t = t.replace(a, b)
        return re.sub(r"[ \t]+", " ", t)          # aligned table cells: runs of blanks count as one
    text = {"ms": norm(main_tex.read_text()), "supp": norm(supp_tex.read_text())}
    both = text["ms"] + text["supp"]
    missing = []
    def occurs(tok: str, hay: str) -> bool:
        # as a whole value: not part of a longer number, of a signed number (a value printed with its sign
        # is expected with it) or of a comma-grouped number ('154' in '10,154'); '--' of a range is no sign
        pre = r"(?<![\d.])" + (r"(?<!\d,)(?<!\+)(?<![^-]-)" if tok[:1].isdigit() else "")
        post = r"(?!\.?\d)" + (r"(?!,\d{3}(?!\d))" if tok[-1:].isdigit() else "")
        return re.search(pre + re.escape(tok) + post, hay) is not None
    for label, printed, paper in chk.expected:
        tok = printed.replace("\u2212", "-")
        hay = text.get(paper, both)
        if not occurs(tok, hay) and not occurs(tok, both):
            missing.append((label, printed, paper))
    print("\n" + "=" * 78)
    print(f"--check-tex: {len(chk.expected) - len(missing)} of {len(chk.expected)} expected values occur in the LaTeX sources")
    for label, printed, paper in missing:
        print(f"  MISSING  {printed!r:<12} ({paper}) {label}")
    return 1 if missing else 0


# =========================================================================== main
def run_checks(priv: Private | None, quiet: bool) -> Checker:
    chk = Checker(quiet=quiet)
    pub = Public()
    int_mean = check_internal(chk, pub)
    check_internal_ensemble(chk)
    fold_means = check_lodo(chk)
    check_lodo_foundation(chk, fold_means)
    check_hashes(chk)
    check_wbcbench(chk, pub)
    check_frozen_probe(chk, pub)
    check_tables(chk, pub, int_mean)
    check_integrity(chk)
    private_checks(chk, priv, pub)
    return chk


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recompute the manuscript's reported numbers from analysis/evidence/ and compare them "
                    "with the printed values. Default mode needs no WBCBench per-image file.")
    ap.add_argument("--private-evidence", metavar="DIR", type=Path, default=None,
                    help="folder with WBCBench per-image predictions in the analysis/evidence/<group>/<run>.csv.gz "
                         "layout (DIR or DIR/analysis/evidence); adds the checks that need per-image data")
    ap.add_argument("--quiet", action="store_true", help="print only failures and the summary")
    ap.add_argument("--dump-counts", metavar="OUT", type=Path, default=None,
                    help="with --private-evidence: also write the rebuilt count files to OUT (never inside the repo)")
    ap.add_argument("--check-tex", nargs=2, metavar=("MAIN_TEX", "SUPP_TEX"), type=Path, default=None,
                    help="author option: confirm every expected value, also of skipped checks, occurs in the "
                         "paper's LaTeX sources")
    args = ap.parse_args()
    priv = Private(args.private_evidence) if args.private_evidence else None
    chk = run_checks(priv, args.quiet)
    rc = chk.summary()
    if args.dump_counts is not None:
        if priv is None:
            raise SystemExit("--dump-counts needs --private-evidence")
        out = args.dump_counts.resolve()
        if ANALYSIS.parent in out.parents or out == ANALYSIS.parent:
            raise SystemExit("--dump-counts: choose a folder outside the repository")
        out.mkdir(parents=True, exist_ok=True)
        for name, df in rebuild_counts(priv).items():
            df.to_csv(out / f"{name}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        print(f"rebuilt count files written to {out}")
    if args.check_tex:
        rc |= check_tex(chk, *args.check_tex)
    sys.exit(rc)


if __name__ == "__main__":
    main()
