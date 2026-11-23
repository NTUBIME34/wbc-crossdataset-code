#!/usr/bin/env python3
"""Frozen DinoBloom-B linear probe (main-text Table 3, row "DinoBloom-B (frozen; linear
probe)"; Supplementary Section S7, "Foundation-model baseline").

The probe was run once under a protocol written down before the run; the protocol is
reproduced here and was not changed afterwards.

    Backbone        timm.create_model(<DinoBloom-B hub id>, pretrained=True, num_classes=0,
                    img_size=224), frozen, eval mode (the same construction as the
                    fine-tuned DinoBloom-B in train/train_dinobloom.py).
    Features        the backbone's pooled output, 768-dimensional: timm's default
                    global_pool='token' for this model, i.e. the class token (the value
                    actually used is written to meta.json).
    Preprocessing   square padding -> Resize(256) -> CenterCrop(224) -> ImageNet mean/std,
                    the transforms of the common evaluator (eval/evaluate_classifier.py),
                    on every split.
    Training data   the train split of the internal nine-class corpus (102,345 images),
                    centre view only, no augmentation.
    Standardising   StandardScaler fitted on the training features only.
    Classifier      sklearn LogisticRegression (multinomial, lbfgs, max_iter=5000,
                    tol=1e-4, random_state=0), fitted on the training split only (no refit
                    on train + val).
    C grid          {0.01, 0.1, 1, 10, 100}.
    Class weighting none; inverse-square-root (w_c proportional to 1/sqrt(n_c), scaled to a
                    mean sample weight of 1, the form used by the proposed model's loss);
                    inverse-frequency (sklearn 'balanced', w_c = n / (K n_c)).
    Selection       highest nine-class macro-F1 on the val split (12,790 images); ties go to
                    the smaller C, then to the order none -> inverse-sqrt ->
                    inverse-frequency.
    Internal score  nine-class macro-F1 of the selected configuration on the test split
                    (12,803 images), no test-time augmentation.
    External score  WBCBench2026 Phase-1 pre-cropped set with the common evaluator's
                    seven-class mapping: the probe probabilities of the centre and the
                    horizontally flipped crop are averaged, the nine-class argmax is mapped
                    to the seven classes; n = 8,061.
    Reporting       the full 15-cell grid (val / test / external) is written to grid.csv;
                    selection uses the val column only.

Reported result (single fit, no seed replication): selected configuration unweighted,
C = 0.01; internal macro-F1 0.938; external macro-F1 0.809, weighted F1 0.911, balanced
accuracy 0.822, BLAST F1 0.547, PROMY F1 0.558. At their own val-selected C (also 0.01)
the inverse-square-root and inverse-frequency probes reached external macro-F1 0.789 and
0.757.

Outputs in --out: emb_{train,val,test}.npy, emb_external_{center,hflip}.npy (embeddings
are cached and re-used when present), embedding_index.json, grid.csv, selected.json,
probe.joblib (scaler + classifier), external_predictions.csv (path,true,pred), meta.json
and summary.txt. The external files list WBCBench2026 image paths: keep them on your own
machine.

    python eval/frozen_probe.py --data-root /path/to/internal_corpus \
        --external-root /path/to/wbcbench_phase1_cropped --out /path/to/outputs/frozen_probe \
        [--device cuda] [--limit N  # smoke test only]
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_classifier as EC  # noqa: E402

MODEL = "hf-hub:1aurent/vit_base_patch14_224.dinobloom@4f2a9d8593a7d12efe210778ea8560863fcacd03"
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
WEIGHTINGS = ["none", "inverse_sqrt", "inverse_frequency"]   # also the tie-break order


class ImagePaths(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def folder_items(data_root, split, classes, limit):
    paths, labels = [], []
    for k, c in enumerate(classes):
        files = sorted(p for p in (data_root / split / c).iterdir() if p.suffix.lower() in EC.IMG_EXTS)
        if limit:
            files = files[:limit]
        paths += [str(p) for p in files]
        labels += [k] * len(files)
    return paths, np.array(labels)


def external_items(crop_root):
    """(path, None, ground-truth folder name) for every image, class folders and files sorted."""
    items = []
    for cls_dir in sorted(crop_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in EC.IMG_EXTS:
                continue
            items.append((str(p), None, cls_dir.name))
    return items


@torch.no_grad()
def embed(model, paths, tf, device, bs, workers):
    loader = DataLoader(ImagePaths(paths, tf), batch_size=bs, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    out = []
    for x in loader:
        out.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out)


def cached(path, fn):
    if path.exists():
        return np.load(path)
    arr = fn()
    np.save(path, arr)
    return arr


def class_weight(kind, y, k):
    if kind == "none":
        return None
    if kind == "inverse_frequency":
        return "balanced"
    counts = np.bincount(y, minlength=k).astype(float)
    w = 1.0 / np.sqrt(counts)
    w *= len(y) / float((counts * w).sum())        # sample-average weight = 1
    return {c: float(w[c]) for c in range(k)}


def external_metrics(probs, ext_gts, ext_paths, classes, a2c, ignore):
    preds = probs.argmax(1)
    y_true, y_pred, records, skipped = [], [], [], 0
    for i, gt in enumerate(ext_gts):
        mgt = EC.get_canonical_label(gt, a2c, ignore)
        if mgt is None:
            skipped += 1
            continue
        mp = EC.get_canonical_label(classes[preds[i]], a2c, ignore) or "Skipped_Class"
        y_true.append(mgt); y_pred.append(mp); records.append((ext_paths[i], mgt, mp))
    labels = sorted(set(y_true) | set(y_pred))
    rep = classification_report(y_true, y_pred, labels=labels, digits=4,
                                output_dict=True, zero_division=0)
    return {
        "n_total": len(ext_gts), "n_skipped_ignored": skipped, "n_eval": len(y_true),
        "canonical_labels": labels,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "report": rep,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }, records


def main():
    ap = argparse.ArgumentParser(description="Frozen DinoBloom-B linear probe under the pre-specified protocol.")
    ap.add_argument("--data-root", required=True,
                    help="Internal nine-class corpus with train/, val/ and test/ class folders.")
    ap.add_argument("--external-root", required=True,
                    help="WBCBench2026 Phase-1 pre-cropped class-folder set (external evaluation).")
    ap.add_argument("--out", required=True, help="Output directory (embeddings are cached here).")
    ap.add_argument("--model-name", default=MODEL, help=f"timm hub id of DinoBloom-B (default: {MODEL}).")
    ap.add_argument("--check-classes-ckpt", default=None,
                    help="Optional checkpoint (e.g. a fine-tuned DinoBloom-B best_macro_f1.pth) whose "
                         "'classes' must equal the class order of --data-root/train.")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu (falls back to cpu without a GPU).")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="images per class/split; smoke test only")
    args = ap.parse_args()
    data_root = Path(args.data_root).expanduser()
    ext_root = Path(args.external_root).expanduser()
    out = EC.outside_repo(args.out, "--out"); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    classes = sorted(p.name for p in (data_root / "train").iterdir() if p.is_dir())
    if args.check_classes_ckpt:
        ft_classes = EC.load_checkpoint(args.check_classes_ckpt, map_location="cpu")["classes"]
        assert classes == list(ft_classes), f"class order {classes} != fine-tune {ft_classes}"
    K = len(classes)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0, img_size=224).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    views = dict(EC.build_tta_transforms(224, "hflip"))
    center, flip = views["center"], views["hflip"]

    # ---- embeddings ----------------------------------------------------------------------
    split = {}
    for s in ("train", "val", "test"):
        paths, y = folder_items(data_root, s, classes, args.limit)
        X = cached(out / f"emb_{s}.npy", lambda: embed(model, paths, center, device, args.batch_size, args.workers))
        split[s] = (paths, y, X)
        print(f"[EMB] {s}: {X.shape}  ({time.time()-t0:.0f}s)", flush=True)
    all_ext = external_items(ext_root)
    ext_items = all_ext[:: max(1, len(all_ext) // (args.limit * 13))] if args.limit else all_ext
    ext_paths = [c for c, _, _ in ext_items]
    ext_gts = [g for _, _, g in ext_items]
    Xe_c = cached(out / "emb_external_center.npy", lambda: embed(model, ext_paths, center, device, args.batch_size, args.workers))
    Xe_f = cached(out / "emb_external_hflip.npy", lambda: embed(model, ext_paths, flip, device, args.batch_size, args.workers))
    print(f"[EMB] external: {Xe_c.shape} x2 views  ({time.time()-t0:.0f}s)", flush=True)
    with open(out / "embedding_index.json", "w") as f:
        json.dump({s: {"paths": split[s][0], "labels": split[s][1].tolist()} for s in split}
                  | {"external": {"paths": ext_paths, "gts": ext_gts}}, f)

    # ---- probe grid ----------------------------------------------------------------------
    (_, ytr, Xtr), (_, yva, Xva), (_, yte, Xte) = split["train"], split["val"], split["test"]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    Xe_cs, Xe_fs = scaler.transform(Xe_c), scaler.transform(Xe_f)
    a2c = EC.parse_merge_rules(EC.default_merge_rules())
    ignore = set(EC.DEFAULT_IGNORE_CLASSES)
    lab9 = list(range(K))

    grid, fitted = [], {}
    for wk in WEIGHTINGS:
        for C in C_GRID:
            t1 = time.time()
            clf = LogisticRegression(C=C, class_weight=class_weight(wk, ytr, K), max_iter=5000,
                                     tol=1e-4, solver="lbfgs", random_state=0)
            clf.fit(Xtr_s, ytr)
            val_f1 = f1_score(yva, clf.predict(Xva_s), labels=lab9, average="macro", zero_division=0)
            test_f1 = f1_score(yte, clf.predict(Xte_s), labels=lab9, average="macro", zero_division=0)
            probs = (clf.predict_proba(Xe_cs) + clf.predict_proba(Xe_fs)) / 2.0
            ext, _ = external_metrics(probs, ext_gts, ext_paths, classes, a2c, ignore)
            row = {"weighting": wk, "C": C, "val_macro_f1_9": val_f1, "test_macro_f1_9": test_f1,
                   "ext_macro_f1": ext["macro_f1"], "ext_weighted_f1": ext["weighted_f1"],
                   "ext_balanced_acc": ext["balanced_acc"], "ext_n_eval": ext["n_eval"],
                   "n_iter": int(np.max(clf.n_iter_)), "converged": bool(np.max(clf.n_iter_) < 5000),
                   "fit_seconds": round(time.time() - t1, 1)}
            grid.append(row); fitted[(wk, C)] = clf
            print(f"[GRID] {wk:17s} C={C:<6} val={val_f1:.4f} test={test_f1:.4f} "
                  f"ext={ext['macro_f1']:.4f} iters={row['n_iter']} ({row['fit_seconds']}s)", flush=True)

    with open(out / "grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0])); w.writeheader(); w.writerows(grid)

    # ---- pre-specified selection: max val macro-F1; ties -> smaller C, then WEIGHTINGS order
    order = {wk: i for i, wk in enumerate(WEIGHTINGS)}
    best = sorted(grid, key=lambda r: (-round(r["val_macro_f1_9"], 12), r["C"], order[r["weighting"]]))[0]
    clf = fitted[(best["weighting"], best["C"])]
    probs = (clf.predict_proba(Xe_cs) + clf.predict_proba(Xe_fs)) / 2.0
    ext, records = external_metrics(probs, ext_gts, ext_paths, classes, a2c, ignore)
    te_pred = clf.predict(Xte_s)
    per_weighting = {}
    for wk in WEIGHTINGS:
        rows = [r for r in grid if r["weighting"] == wk]
        b = sorted(rows, key=lambda r: (-round(r["val_macro_f1_9"], 12), r["C"]))[0]
        per_weighting[wk] = {"C": b["C"], "val_macro_f1_9": b["val_macro_f1_9"],
                             "test_macro_f1_9": b["test_macro_f1_9"], "ext_macro_f1": b["ext_macro_f1"]}

    selected = {
        "selected_config": {"weighting": best["weighting"], "C": best["C"]},
        "internal_test_macro_f1_9": best["test_macro_f1_9"],
        "internal_val_macro_f1_9": best["val_macro_f1_9"],
        "internal_test_n": int(len(yte)),
        "internal_test_confusion_9": confusion_matrix(yte, te_pred, labels=lab9).tolist(),
        "external": ext,
        "best_per_weighting": per_weighting,
        "classes": classes,
    }
    (out / "selected.json").write_text(json.dumps(selected, indent=2))
    with open(out / "external_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "true", "pred"]); w.writerows(records)
    joblib.dump({"scaler": scaler, "classifier": clf, "classes": classes,
                 "config": selected["selected_config"]}, out / "probe.joblib")

    meta = {
        "model": args.model_name, "img_size": 224, "global_pool": getattr(model, "global_pool", None),
        "num_features": int(model.num_features),
        "pretrained_cfg": {k: v for k, v in model.pretrained_cfg.items() if isinstance(v, (str, int, float, list, tuple, type(None)))},
        "timm": timm.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__,
        "numpy": np.__version__, "device": str(device), "limit": args.limit,
        "protocol": "module docstring of eval/frozen_probe.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "c_grid": C_GRID, "weightings": WEIGHTINGS, "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    rep = ext["report"]
    lines = [
        "==== SELECTED (pre-specified rule: max val macro-F1) ====",
        f"config: {selected['selected_config']}",
        f"internal test macro-F1 (9-class, n={len(yte)}): {best['test_macro_f1_9']:.4f}",
        f"external macro-F1 (n={ext['n_eval']}): {ext['macro_f1']:.4f}  weighted-F1 {ext['weighted_f1']:.4f}  "
        f"bal-acc {ext['balanced_acc']:.4f}  BLAST F1 {rep.get('Blast_Group', {}).get('f1-score', float('nan')):.4f}  "
        f"PROMY F1 {rep.get('Promyelocyte', {}).get('f1-score', float('nan')):.4f}",
    ]
    for wk, b in per_weighting.items():
        lines.append(f"  best {wk:17s}: C={b['C']:<6} val={b['val_macro_f1_9']:.4f} "
                     f"test={b['test_macro_f1_9']:.4f} ext={b['ext_macro_f1']:.4f}")
    lines.append(f"total {time.time()-t0:.0f}s")
    print("\n" + "\n".join(lines))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
