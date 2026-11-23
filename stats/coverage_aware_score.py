#!/usr/bin/env python3
"""Coverage-aware scoring of the full pipeline (per seed, then mean +- population SD).

A labelled field on which the detector returned no box is counted as an error: it is
appended as a never-matching prediction ('__no_detection__') before scoring, so it
adds a false negative to its true class; macro-F1 is still the unweighted mean of the
seven class F1 scores (labels= the seven classes), so the sentinel is not averaged as
an eighth class. The detected-cells-only view ("as scored") is reported alongside.
Without --detections the script scores the prediction files as they are (used for
the no-localization row and the Cellpose routes, which classify every field).

Inputs
  --predictions  one per-image file per seed, in either format:
      eval/evaluate_classifier.py predictions.csv run (with --use-mappings) on the
          detector's crops folder (crops keep the class folders of the input), or
      pipeline/classify.py output (source_path, det_status, pred_label, ...) on
          class-folder input; its no-crop rows already carry a non-class label.
  --detections   pipeline/detect_yolo11.py detections.csv (run with --select best
      --fallback skip), one shared file or one per seed. A field counts as missed if
      none of its rows has status 'detected'. Its true label is its class folder,
      mapped with the evaluator's merge/ignore rules (Plasma_cell, Metamyelocyte and
      Myelocyte are not labelled), or comes from --labels.

Image ids: file name without extension and without the '__detNN' crop suffix.

Truth and prediction labels use case-insensitive matching of the seven class names
and the explicit merge/ignore aliases below; spelling or separator variants are not
guessed. Unknown labels raise an error instead of silently producing misleading
scores. Ignored true classes are excluded; predictions of ignored classes remain
errors ('Skipped_Class'). Documented no-crop sentinels retain their existing meaning.
Missing entries in an optional --labels file mean unlabelled fields, as before.
With --detections, every detected labelled field must have a usable prediction and
the truth labels must agree across the two files. Incomplete or mismatched inputs
raise an error; they are never silently scored on a smaller subset.

Paper: main Table 2b (detection-and-cropping ablation) and Table 2a (Phase-1 vs Phase-2: n,
coverage, balanced accuracy, macro-F1; detected-only row), the primary endpoint
0.879 +- 0.006, Supplementary Table S6 (cropping routes) and the S1 counts of
labelled / detected fields.

Example (Phase-2, three seeds, one detections file):
    python stats/coverage_aware_score.py --name "Phase-2" \\
        --predictions p2_s42/predictions.csv p2_s62/predictions.csv p2_s82/predictions.csv \\
        --detections p2_detect/detections.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CLS7 = ["Basophil", "Blast_Group", "Eosinophil", "Lymphocyte",
        "Monocyte", "Neutrophil", "Promyelocyte"]
MISS = "__no_detection__"
NO_CROP_STATUSES = {"no_detection", "read_failed", "empty_crop"}

# Default merge / ignore rules of eval/evaluate_classifier.py --use-mappings.
MERGE_RULES = [
    "Neutrophil=Band_neutrophil|Segmented_neutrophil|Neutrophil|Neutrophil_sum",
    "Lymphocyte=Lymphocyte|Variant_lymphocyte|Prolymphocyte",
    "Blast_Group=Blast|Lymphoblast|Myeloblast|Erythroblast",
]
IGNORE_CLASSES = {"Plasma_cell", "Metamyelocyte", "Myelocyte"}
ALIAS = {a.strip(): rule.split("=", 1)[0].strip()
         for rule in MERGE_RULES for a in rule.split("=", 1)[1].split("|") if a.strip()}
LABEL_LOOKUP = {name.casefold(): name for name in CLS7}
LABEL_LOOKUP.update({name.casefold(): target for name, target in ALIAS.items()})
IGNORE_LOOKUP = {name.casefold() for name in IGNORE_CLASSES}
PREDICTION_SENTINELS = {MISS, "Skipped_Class", *NO_CROP_STATUSES}
SENTINEL_LOOKUP = {name.casefold(): name for name in PREDICTION_SENTINELS}


# ---------------------------------------------------------------- input
def canonical(name: str, *, context: str = "") -> str | None:
    """Normalize documented names only; return None for ignored true classes."""
    key = name.strip().casefold() if isinstance(name, str) else ""
    if key in IGNORE_LOOKUP:
        return None
    if key not in LABEL_LOOKUP:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}unsupported class label {name!r}; use one of {CLS7}, "
            f"a documented merge alias, or an ignored class {sorted(IGNORE_CLASSES)}"
        )
    return LABEL_LOOKUP[key]


def canonical_prediction(name: str, *, context: str = "") -> str:
    """Keep non-class predictions as errors, not extra averaged classes."""
    key = name.strip().casefold() if isinstance(name, str) else ""
    if key in SENTINEL_LOOKUP:
        return SENTINEL_LOOKUP[key]
    return canonical(name, context=context) or "Skipped_Class"


def read_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def field_id(path: str) -> str:
    """'.../Lymphocyte/cell0001__det00.png' and '.../Lymphocyte/cell0001.jpg' -> 'cell0001'."""
    return Path(path).name.split("__")[0].split(".")[0]


def load_predictions(path: str | Path, labels: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Return {'id', 'true', 'pred', 'status'} lists, one entry per labelled image."""
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
    out: dict[str, list[str]] = {"id": [], "true": [], "pred": [], "status": []}
    for row_number, r in enumerate(rows, start=2):
        src = r[path_col] or r.get("crop_path", "")  # pipeline crop rows may lack source_path
        if true_col is not None:
            t = canonical(r[true_col], context=f"{path}: row {row_number}, {true_col}")
        else:  # pipeline output: label = class folder of the field, or from --labels
            raw = labels.get(field_id(src)) if labels is not None else Path(src).parent.name
            t = canonical(raw, context=f"{path}: row {row_number}, truth") if raw else None
        if t is None:
            continue
        p = canonical_prediction(r[pred_col], context=f"{path}: row {row_number}, {pred_col}")
        status = r.get("det_status", "") or "detected"
        out["id"].append(field_id(src))
        out["true"].append(t)
        out["pred"].append(p)
        out["status"].append(status)
    if len(set(out["id"])) != len(out["id"]):
        sys.exit(f"[ERR] {path}: more than one row per field; run the detector with --select best")
    return out


def load_detections(path: str | Path, labels: dict[str, str] | None) -> dict[str, dict]:
    """{field id: {'label': seven-class label or None, 'detected': bool}} from detections.csv."""
    fields: dict[str, dict] = {}
    for row_number, r in enumerate(read_rows(path), start=2):
        src = r["source_path"]
        fid = field_id(src)
        if labels is not None:
            raw = labels.get(fid)
            lab = canonical(raw, context=f"{path}: row {row_number}, truth") if raw else None
        else:
            lab = canonical(Path(src).parent.name, context=f"{path}: row {row_number}, truth")
        info = fields.setdefault(fid, {"label": lab, "detected": False, "status": set()})
        if info["label"] != lab:
            raise ValueError(f"{path}: row {row_number}, conflicting truth labels for field {fid!r}")
        info["detected"] |= r["status"] == "detected"
        info["status"].add(r["status"])
    return fields


def load_labels(path: str | Path) -> dict[str, str]:
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"{path}: no label rows")
    id_col = next((c for c in ("image_id", "image_path", "path", "source_path") if c in rows[0]), None)
    lab_col = next((c for c in ("true_label", "label", "class") if c in rows[0]), None)
    if id_col is None or lab_col is None:
        raise ValueError(f"{path}: unrecognised label columns {sorted(rows[0])}")
    labels: dict[str, str] = {}
    for row_number, r in enumerate(rows, start=2):
        fid, raw = field_id(r[id_col]), r[lab_col]
        value = canonical(raw, context=f"{path}: row {row_number}, {lab_col}")
        if fid in labels and canonical(labels[fid]) != value:
            raise ValueError(f"{path}: row {row_number}, conflicting truth labels for field {fid!r}")
        labels[fid] = raw
    return labels


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
def scores(yt: list[str], yp: list[str]) -> dict[str, float]:
    unknown = sorted(set(yt) - set(CLS7))
    if unknown:
        raise ValueError(f"unsupported true labels {unknown}; expected canonical labels {CLS7}")
    unknown_predictions = sorted(set(yp) - set(CLS7) - PREDICTION_SENTINELS)
    if unknown_predictions:
        raise ValueError(f"unsupported predicted labels {unknown_predictions}")
    f1s = f1_score(yt, yp, labels=CLS7, average=None, zero_division=0)
    return {
        "n": len(yt),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(f1_score(yt, yp, labels=CLS7, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        **{f"f1_{c}": float(v) for c, v in zip(CLS7, f1s)},
    }


def agg(vals: list[dict]) -> dict:
    out = {"n": vals[0]["n"], "per_seed_macro_f1": [v["macro_f1"] for v in vals]}
    for k in vals[0]:
        if k == "n":
            continue
        a = np.array([v[k] for v in vals])
        out[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=0))}
    return out


def score_seed(pred: dict[str, list[str]], fields: dict[str, dict] | None) -> dict:
    """As-scored (detected cells) and coverage-aware views for one seed."""
    res: dict = {}
    if fields is None:
        res["as_scored"] = scores(pred["true"], pred["pred"])
        res["no_crop_rows"] = sum(s in NO_CROP_STATUSES for s in pred["status"])
        return res
    labelled = {f: i for f, i in fields.items() if i["label"] is not None}
    missed = [f for f, i in labelled.items() if not i["detected"]]
    detected = [f for f, i in labelled.items() if i["detected"]]
    row_of = {f: k for k, f in enumerate(pred["id"])}
    # scored cells: every row of the prediction file except no-crop rows
    det_idx = [k for k, s in enumerate(pred["status"]) if s not in NO_CROP_STATUSES]
    yt = [pred["true"][k] for k in det_idx]
    yp = [pred["pred"][k] for k in det_idx]
    not_scored = [f for f in detected if f not in row_of]
    if not_scored:
        raise ValueError(f"{len(not_scored)} detected labelled fields have no prediction row; "
                         "refusing to score incomplete predictions")
    extra_rows = [f for f in pred["id"] if f not in fields]
    if extra_rows:
        raise ValueError(f"{len(extra_rows)} prediction rows have no field in the detections file")
    for fid, k in row_of.items():
        if pred["true"][k] != fields[fid]["label"]:
            raise ValueError(f"conflicting truth labels for field {fid!r} between predictions and detections")
        if fields[fid]["detected"] and pred["status"][k] in NO_CROP_STATUSES:
            raise ValueError(f"detected labelled field {fid!r} has no usable classification; "
                             "refusing to omit it from coverage-aware scoring")
    for f in missed:
        if f in row_of and pred["status"][row_of[f]] not in NO_CROP_STATUSES:
            sys.exit(f"[ERR] field {f} has no detection but was classified; coverage-aware "
                     "scoring expects the detector's --fallback skip")
    res["as_scored"] = scores(yt, yp)
    # a missed field is appended once, as a never-matching prediction for its true label
    miss_labels = [labelled[f]["label"] for f in missed]
    res["coverage_aware"] = scores(yt + miss_labels, yp + [MISS] * len(miss_labels))
    res["coverage"] = {"labelled": len(labelled), "detected": len(detected), "missed": len(missed),
                       "fraction": len(detected) / len(labelled) if labelled else float("nan"),
                       "fields_in_file": len(fields),
                       "missed_unlabelled": sum(1 for i in fields.values()
                                                if i["label"] is None and not i["detected"])}
    return res


def changed_predictions(new: dict[str, list[str]], old_path: str) -> dict[str, int]:
    old = load_predictions(old_path)
    o = dict(zip(old["id"], old["pred"]))
    n = dict(zip(new["id"], new["pred"]))
    shared = [f for f in n if f in o]
    return {"n_new": len(n), "n_old": len(o), "n_matched": len(shared),
            "n_changed": sum(1 for f in shared if n[f] != o[f])}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Coverage-aware macro-F1 (missed detections counted as errors), "
                    "per seed and mean +- population SD.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="Inputs" + __doc__.split("Inputs", 1)[1])
    ap.add_argument("--predictions", nargs="+", required=True, help="one per-image file per seed")
    ap.add_argument("--detections", nargs="+", default=None,
                    help="detections.csv: one shared file or one per seed (omit for routes "
                         "without a detector)")
    ap.add_argument("--labels", default=None,
                    help="optional CSV image_id,true_label for fields whose label is not their "
                         "class folder (used for detections.csv and for pipeline output)")
    ap.add_argument("--seeds", nargs="+", default=None, help="seed names for the printout")
    ap.add_argument("--name", default="run", help="row name for the printout")
    ap.add_argument("--compare-to", nargs="+", default=None,
                    help="earlier per-seed prediction files: count changed predictions per seed")
    ap.add_argument("--out-dir", default=None, help="default $OUT_ROOT/stats")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

    k = len(args.predictions)
    seeds = args.seeds or [str(i + 1) for i in range(k)]
    if len(seeds) != k:
        ap.error("--seeds must match --predictions")
    dets = args.detections
    if dets and len(dets) not in (1, k):
        ap.error("--detections takes one file or one per prediction file")
    labels = load_labels(args.labels) if args.labels else None

    per_seed, preds, field_cache = [], [], {}
    for i, p in enumerate(args.predictions):
        pr = load_predictions(p, labels)
        preds.append(pr)
        fields = None
        if dets:
            dpath = dets[0] if len(dets) == 1 else dets[i]
            if dpath not in field_cache:
                field_cache[dpath] = load_detections(dpath, labels)
            fields = field_cache[dpath]
        per_seed.append(score_seed(pr, fields))

    summary = {"name": args.name, "seeds": seeds,
               "per_seed": {s: r for s, r in zip(seeds, per_seed)}}
    views = [v for v in ("as_scored", "coverage_aware") if v in per_seed[0]]
    for v in views:
        summary[v] = agg([r[v] for r in per_seed])
    if dets:
        summary["coverage"] = [r["coverage"] for r in per_seed]

    # ------------------------------------------------------------ printout
    print(f"{args.name}: {k} seed(s) {', '.join(seeds)}")
    if not dets and per_seed[0].get("no_crop_rows"):
        print(f"  the file lists {per_seed[0]['no_crop_rows']} labelled images without a crop as "
              "non-class predictions, so 'as scored' is already coverage-aware")
    if dets:
        c = per_seed[0]["coverage"]
        print(f"  fields in detections file {c['fields_in_file']}: labelled {c['labelled']} "
              f"(unlabelled {c['fields_in_file'] - c['labelled']}), detected {c['detected']}, "
              f"missed {c['missed']} (coverage {c['detected']}/{c['labelled']} = "
              f"{100 * c['fraction']:.2f}%); fields without a box, labelled or not: "
              f"{c['missed'] + c['missed_unlabelled']}")
    title = {"as_scored": "detected cells only" if dets else "as scored",
             "coverage_aware": "coverage-aware"}
    print(f"  {'view':<22}{'n':>6}  {'acc':>6} {'bal.acc':>7}  {'macro-F1':<15} {'wF1':>6} "
          f"{'BLAST':>6} {'PROMY':>6}")
    for v in views:
        a = summary[v]
        print(f"  {title[v]:<22}{a['n']:>6}  {a['accuracy']['mean']:>6.3f} "
              f"{a['balanced_accuracy']['mean']:>7.3f}  "
              f"{a['macro_f1']['mean']:.3f} +- {a['macro_f1']['sd']:.3f}  "
              f"{a['weighted_f1']['mean']:>6.3f} {a['f1_Blast_Group']['mean']:>6.3f} "
              f"{a['f1_Promyelocyte']['mean']:>6.3f}")
    for v in views:
        a = summary[v]
        print(f"  {title[v]}: macro-F1 {a['macro_f1']['mean']:.6f} +- {a['macro_f1']['sd']:.6f}; "
              f"per seed {', '.join(f'{x:.6f}' for x in a['per_seed_macro_f1'])}")
    if args.compare_to:
        if len(args.compare_to) != k:
            ap.error("--compare-to needs one file per prediction file")
        summary["changed_vs_earlier"] = {}
        for s, pr, old in zip(seeds, preds, args.compare_to):
            ch = changed_predictions(pr, old)
            summary["changed_vs_earlier"][s] = ch
            print(f"  seed {s}: {ch['n_changed']} of {ch['n_matched']} matched predictions changed "
                  f"(new {ch['n_new']}, earlier {ch['n_old']})")

    if args.no_write:
        return
    d = out_dir(args.out_dir)
    tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.name)
    path = d / f"coverage_aware_{tag}.json"
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
