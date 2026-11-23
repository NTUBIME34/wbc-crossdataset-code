#!/usr/bin/env python3
"""Evaluate the retained frozen probe and original local backbone, without refitting.

The external mode follows frozen_probe.py: each view is standardized and classified
separately, probabilities are averaged, then the nine-class argmax is mapped to seven
classes. Internal mode uses the center view and original ordered classes only.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import timm
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_probe as protocol
import evaluate_classifier as common


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def checked(path, expected):
    if not path.is_file() or digest(path).lower() != expected.lower():
        raise ValueError('Missing or checksum-mismatched artifact: ' + path.name)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--probe-sha256', required=True)
    ap.add_argument('--trust-probe', action='store_true', help='Required: joblib files can execute code; trust the checksum-verified author artifact only')
    ap.add_argument('--model-name', required=True, help='local-dir:/path/to/original_dinobloom_config_and_weights')
    ap.add_argument('--backbone-sha256', required=True)
    ap.add_argument('--config-sha256', required=True)
    ap.add_argument('--test-root', type=Path, required=True)
    ap.add_argument('--mode', choices=('external', 'internal'), required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    os.umask(0o077)
    if not args.trust_probe:
        raise ValueError('Explicit --trust-probe is required before joblib loading')
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError('Invalid batch size or worker count')
    if not args.model_name.startswith('local-dir:'):
        raise ValueError('Use the original locally packaged backbone; network model lookup is not used')
    local = Path(args.model_name.split(':', 1)[1]).expanduser().resolve()
    backbone, config = local / 'model.safetensors', local / 'config.json'
    checked(args.probe, args.probe_sha256)
    checked(backbone, args.backbone_sha256)
    checked(config, args.config_sha256)
    root = args.test_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Test root is missing')
    out = common.outside_repo(str(args.output_dir), '--output-dir')
    if out.exists() or out == root or root in out.parents or out in root.parents:
        raise ValueError('Use a new output directory separate from the dataset')
    checkpoint = joblib.load(args.probe)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {'scaler', 'classifier', 'classes', 'config'}:
        raise ValueError('Unexpected fitted probe payload')
    classes = checkpoint['classes']
    if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes):
        raise ValueError('Invalid ordered probe classes')
    if not all(isinstance(c, str) and c and c not in {'.', '..'} and '/' not in c and '\\' not in c for c in classes):
        raise ValueError('Unsafe probe class name')
    scaler, clf = checkpoint['scaler'], checkpoint['classifier']
    if not np.array_equal(clf.classes_, np.arange(len(classes))):
        raise ValueError('Probe numeric class order differs from stored names')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; explicitly use --device cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.time()
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0, img_size=224).to(device).eval()
    if int(model.num_features) != int(scaler.n_features_in_):
        raise ValueError('Backbone feature dimension differs from the retained probe')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    views = dict(common.build_tta_transforms(224, 'hflip'))
    if args.mode == 'external':
        items = protocol.external_items(root)
        paths, truths = [row[0] for row in items], [row[2] for row in items]
    else:
        actual = {p.name for p in root.iterdir() if p.is_dir()}
        if actual != set(classes):
            raise ValueError('Internal class folders differ from ordered probe classes')
        paths, numeric_truth = protocol.folder_items(root.parent, root.name, classes, 0)
        truths = [classes[int(i)] for i in numeric_truth]
    if not paths:
        raise ValueError('No evaluation images')
    features = protocol.embed(model, paths, views['center'], device, args.batch_size, args.workers)
    probabilities = clf.predict_proba(scaler.transform(features))
    if args.mode == 'external':
        flipped = protocol.embed(model, paths, views['hflip'], device, args.batch_size, args.workers)
        probabilities = (probabilities + clf.predict_proba(scaler.transform(flipped))) / 2.0
    if probabilities.shape != (len(paths), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid retained-probe probability matrix')
    pred = probabilities.argmax(1)
    out.mkdir(parents=True)
    if args.mode == 'external':
        aliases = common.parse_merge_rules(common.default_merge_rules())
        ignore = set(common.DEFAULT_IGNORE_CLASSES)
        metrics, _ = protocol.external_metrics(probabilities, truths, paths, classes, aliases, ignore)
        canonical = sorted({common.get_canonical_label(c, aliases, ignore) for c in classes} - {None})
        with (out / 'predictions.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_path', 'raw_true', 'raw_pred', 'true_label', 'pred_label', 'correct'] + ['prob_' + c for c in canonical])
            for i, (path, truth) in enumerate(zip(paths, truths)):
                mapped = common.get_canonical_label(truth, aliases, ignore)
                if mapped is None:
                    continue
                raw_prediction = classes[pred[i]]
                prediction = common.get_canonical_label(raw_prediction, aliases, ignore) or 'Skipped_Class'
                sums = {c:0.0 for c in canonical}
                for j, name in enumerate(classes):
                    merged = common.get_canonical_label(name, aliases, ignore)
                    if merged in sums:
                        sums[merged] += float(probabilities[i, j])
                writer.writerow([path, truth, raw_prediction, mapped, prediction, int(mapped == prediction)] + [sums[c] for c in canonical])
    else:
        label_indices = np.arange(len(classes))
        metrics = dict(n_eval=len(paths), classes=classes,
                       macro_f1=float(f1_score(numeric_truth, pred, labels=label_indices, average='macro', zero_division=0)),
                       weighted_f1=float(f1_score(numeric_truth, pred, labels=label_indices, average='weighted', zero_division=0)),
                       accuracy=float(accuracy_score(numeric_truth, pred)),
                       balanced_acc=float(balanced_accuracy_score(numeric_truth, pred)),
                       confusion_matrix=confusion_matrix(numeric_truth, pred, labels=label_indices).tolist())
        with (out / 'predictions.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_path', 'true_label', 'pred_label'] + ['prob_' + c for c in classes])
            for i, path in enumerate(paths):
                writer.writerow([path, truths[i], classes[pred[i]]] + probabilities[i].tolist())
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    provenance = dict(status='COMPLETE', mode=args.mode, model_name=args.model_name,
                      selected_config=checkpoint['config'], class_order=classes,
                      fitted_or_selected=False, original_backbone_pretrained=True,
                      probe_sha256=digest(args.probe), backbone_sha256=digest(backbone), config_sha256=digest(config),
                      evaluator_sha256=digest(Path(__file__)), protocol_sha256=digest(Path(protocol.__file__)),
                      inputs=len(paths), evaluated=metrics['n_eval'], device=str(device),
                      feature_dimensions=int(model.num_features), elapsed_seconds=time.time()-started,
                      note='Retained-model inference; exact historical input identity and numerical agreement require separate checks.')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(status='COMPLETE', macro_f1=metrics['macro_f1'], n=metrics['n_eval']), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
