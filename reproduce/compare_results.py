#!/usr/bin/env python3
"""Compare newly generated predictions with retained reference counts, without changing either."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'stats'))
import coverage_aware_score as scorer


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def references(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    result = defaultdict(dict)
    with opener(path, 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            run, truth = row['run'], row['true_label']
            if truth in result[run]:
                raise ValueError('Duplicate reference truth row for ' + run)
            result[run][truth] = {key: int(value) for key, value in row.items()
                                  if key not in ('run', 'true_label') and int(value)}
    return dict(result)


def confusion_from(predictions, fields=None):
    # score_seed supplies the same strict coverage, ID and truth validation used by
    # the deployable scorer. Counts below are independently assembled from rows.
    scorer.score_seed(predictions, fields)
    actual = defaultdict(Counter)
    if fields is None:
        for truth, pred in zip(predictions['true'], predictions['pred']):
            actual[truth][pred] += 1
    else:
        lookup = {identifier:(truth, pred) for identifier, truth, pred in
                  zip(predictions['id'], predictions['true'], predictions['pred'])}
        for identifier, field in fields.items():
            truth = field['label']
            if truth not in scorer.CLS7:
                continue
            pred = lookup[identifier][1] if field['detected'] else scorer.MISS
            actual[truth][pred] += 1
    return {truth:dict(row) for truth, row in actual.items()}


def metrics(matrix):
    support = Counter()
    predicted = Counter()
    hits = Counter()
    for truth, row in matrix.items():
        for pred, count in row.items():
            support[truth] += count
            predicted[pred] += count
            if truth == pred:
                hits[truth] += count
    total = sum(support.values())
    if total == 0:
        raise ValueError('Empty confusion matrix')
    f1 = {c: (2 * hits[c] / (support[c] + predicted[c])
              if support[c] + predicted[c] else 0.0) for c in scorer.CLS7}
    present = [c for c in scorer.CLS7 if support[c]]
    return dict(n=total, macro_f1=sum(f1.values()) / len(scorer.CLS7),
                weighted_f1=sum(support[c] * f1[c] for c in scorer.CLS7) / total,
                accuracy=sum(hits.values()) / total,
                balanced_accuracy=sum(hits[c] / support[c] for c in present) / len(present),
                per_class_f1=f1)


def compare(reference, actual):
    difference = []
    for truth in sorted(set(reference) | set(actual)):
        for pred in sorted(set(reference.get(truth, {})) | set(actual.get(truth, {}))):
            old, new = reference.get(truth, {}).get(pred, 0), actual.get(truth, {}).get(pred, 0)
            if old != new:
                difference.append(dict(true_label=truth, pred_label=pred, reference=old, generated=new))
    ref_metrics, new_metrics = metrics(reference), metrics(actual)
    numeric = ('macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy')
    return dict(state='EXACT_COUNT_MATCH' if not difference else 'NUMERICAL_DIFFERENCE',
                exact_confusion_match=not difference, reference=ref_metrics, generated=new_metrics,
                delta={key:new_metrics[key] - ref_metrics[key] for key in numeric},
                three_decimal_match={key:f'{new_metrics[key]:.3f}' == f'{ref_metrics[key]:.3f}' for key in numeric},
                count_differences=difference)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', nargs=2, action='append', required=True, metavar=('REFERENCE_KEY', 'PREDICTIONS'))
    ap.add_argument('--detections', nargs=2, action='append', default=[], metavar=('REFERENCE_KEY', 'DETECTIONS'))
    ap.add_argument('--reference', type=Path, default=REPO / 'analysis/evidence/wbcbench_confusion/run_confusion.csv.gz')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    output = args.out.expanduser().resolve()
    if output == REPO or REPO in output.parents:
        ap.error('--out must be outside the repository')
    runs = dict(args.run)
    if len(runs) != len(args.run):
        ap.error('Duplicate run key')
    detections = dict(args.detections)
    if len(detections) != len(args.detections) or set(detections) - set(runs):
        ap.error('Duplicate or unrequested detection key')
    refs = references(args.reference)
    results = []
    for key, prediction in args.run:
        if key not in refs:
            ap.error('Unknown reference key: ' + key)
        fields = scorer.load_detections(detections[key], None) if key in detections else None
        actual = confusion_from(scorer.load_predictions(prediction), fields)
        result = dict(run=key, predictions_sha256=digest(prediction), **compare(refs[key], actual))
        if key in detections:
            result['detections_sha256'] = digest(detections[key])
        results.append(result)
        print(key + ': ' + result['state'])
    report = dict(reference_sha256=digest(args.reference), checker_sha256=digest(Path(__file__)),
                  all_selected_counts_match=all(r['exact_confusion_match'] for r in results),
                  scope='Selected regenerated per-seed confusion counts; not all study experiments or training.',
                  results=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    return 0 if report['all_selected_counts_match'] else 1


if __name__ == '__main__':
    sys.exit(main())
