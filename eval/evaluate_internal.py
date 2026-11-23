#!/usr/bin/env python3
"""Evaluate an exact internal split with the checkpoint's trainer-native test transform.

This is the nine-class internal endpoint of Table 3 and Supplementary S4, not the
external square-pad/256/hflip protocol or the seven-class ensemble of Fig. 3.
The actual trainer supplies its validation transform. No training, augmentation,
test-time averaging or label harmonization is performed.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score


REPO = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
TRAINERS = {'classifier': 'train_classifier.py', 'cnn': 'train_cnn_baseline.py',
            'dinobloom': 'train_dinobloom.py'}


def import_file(path, name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def transform_from_trainer(trainer_name, saved_args):
    if trainer_name not in TRAINERS:
        raise ValueError('Unknown trainer family')
    if not isinstance(saved_args, dict):
        raise ValueError('Checkpoint args must be a dictionary')
    trainer_path = REPO / 'train' / TRAINERS[trainer_name]
    trainer = import_file(trainer_path, '_internal_' + trainer_name)
    # parse_args constructs the real current defaults but never enters trainer.main.
    # The dummy required paths are not read; saved args override all matching defaults.
    previous_argv = sys.argv
    sys.argv = ['trainer-defaults', '--root', 'unused', '--outdir', 'unused']
    if trainer_name == 'cnn':
        sys.argv += ['--model-name', str(saved_args.get('model_name', 'unused'))]
    try:
        args = trainer.parse_args()
    finally:
        sys.argv = previous_argv
    defaults = vars(args).copy()
    vars(args).update(saved_args)
    if isinstance(args.img_size, bool) or not isinstance(args.img_size, int) or args.img_size <= 0:
        raise ValueError('Saved img_size must be a positive integer')
    if trainer_name == 'cnn':
        if args.pad_mode not in {'black', 'reflect', 'replicate'}:
            raise ValueError('Unsupported saved CNN pad_mode')
        _, transform = trainer.build_transforms(img_size=args.img_size, pad_mode=args.pad_mode)
        protocol = dict(img_size=args.img_size, pad_mode=args.pad_mode,
                        resize=int(args.img_size * 1.1), square_padding=True)
    else:
        if args.recipe not in {'paper', 'meta'}:
            raise ValueError('Unsupported saved recipe')
        _, transform = trainer.build_transforms(args)
        protocol = dict(img_size=args.img_size, recipe=args.recipe,
                        resize=int(args.img_size * (256 / 224 if args.recipe == 'paper' else 1.1)),
                        square_padding=args.recipe != 'paper')
    protocol.update(center_crop=args.img_size, tta='none', label_mapping='none',
                    source_file='train/' + TRAINERS[trainer_name],
                    source_sha256=sha256(trainer_path),
                    saved_argument_names=sorted(saved_args),
                    defaulted_argument_names=sorted(set(defaults) - set(saved_args)))
    return transform, args, protocol


def validate_classes(classes):
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError('Checkpoint must contain a nonempty ordered classes list')
    if not all(isinstance(c, str) and c and c not in {'.', '..'} and '/' not in c and '\\' not in c for c in classes):
        raise ValueError('Checkpoint class names must be safe folder names')
    if len(set(classes)) != len(classes):
        raise ValueError('Duplicate checkpoint classes')
    return list(classes)


class ExactSplitDataset(Dataset):
    def __init__(self, root, classes, transform):
        self.root = Path(root).resolve()
        self.classes = validate_classes(classes)
        self.transform = transform
        if not self.root.is_dir():
            raise ValueError('Test root is not a directory')
        folders = {p.name for p in self.root.iterdir() if p.is_dir()}
        if folders != set(self.classes):
            raise ValueError('Test class folders differ from checkpoint classes: missing=' +
                             repr(sorted(set(self.classes) - folders)) + ', extra=' + repr(sorted(folders - set(self.classes))))
        self.samples = []
        for index, name in enumerate(self.classes):
            images = sorted(p for p in (self.root / name).rglob('*')
                            if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not images:
                raise ValueError('Empty internal test class: ' + name)
            self.samples.extend((p, index) for p in images)
        self.targets = [index for _, index in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        # A corrupt file is a failed run, never a silently substituted blank image.
        with Image.open(path) as image:
            transformed = self.transform(image.convert('RGB'))
        return transformed, target, str(path.relative_to(self.root))


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-path', type=Path, required=True)
    ap.add_argument('--trainer', choices=sorted(TRAINERS), required=True)
    ap.add_argument('--test-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--model-name', help='Explicit architecture override, recorded in provenance')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=4)
    return ap


def validate_cli(args):
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('batch-size must be positive and num-workers nonnegative')
    if not args.model_path.is_file():
        raise ValueError('Checkpoint is missing')
    if not args.test_root.is_dir():
        raise ValueError('Test split is missing')
    out = args.output_dir.expanduser().resolve()
    if out == REPO or REPO in out.parents:
        raise ValueError('Output directory must be outside the repository')
    test_root = args.test_root.expanduser().resolve()
    if out == test_root or test_root in out.parents or out in test_root.parents:
        raise ValueError('Output directory must be separate from the input test split')
    if out.exists():
        raise ValueError('Output directory already exists; use a new directory to preserve evidence')
    return out


def main(argv=None):
    args = make_parser().parse_args(argv)
    os.umask(0o077)
    out = validate_cli(args)
    started = time.time()
    # Exported research classifiers contain tensors and plain Python metadata only.
    # Do not execute a checkpoint's pickle fallback here.
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('state_dict'), dict):
        raise ValueError('Expected an exported classifier checkpoint with state_dict')
    classes = validate_classes(checkpoint.get('classes'))
    transform, train_args, protocol = transform_from_trainer(args.trainer, checkpoint.get('args'))
    model_name = args.model_name or checkpoint.get('model_name') or getattr(train_args, 'model_name', None)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError('Model architecture is missing; supply --model-name')
    dataset = ExactSplitDataset(args.test_root, classes, transform)
    device_name = 'cuda:' + args.device if args.device.isdigit() else args.device
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; explicitly use --device cpu')
    common = import_file(REPO / 'eval/evaluate_classifier.py', '_internal_model_factory')
    model = common.create_model(model_name, len(classes), train_args.img_size)
    state = {}
    for key, value in checkpoint['state_dict'].items():
        target = key[7:] if key.startswith('module.') else key
        if target in state:
            raise ValueError('Duplicate state key after removing module prefix')
        state[target] = value
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    out.mkdir(parents=True)
    prediction_path = out / 'predictions.csv'
    truth, predicted = [], []
    with prediction_path.open('x', newline='') as stream, torch.inference_mode():
        writer = csv.writer(stream)
        writer.writerow(['image_path', 'raw_true', 'raw_pred', 'true_label', 'pred_label', 'correct'] +
                        ['prob_' + name for name in classes])
        for images, targets, paths in loader:
            logits = model(images.to(device, non_blocking=True))
            if logits.shape != (len(targets), len(classes)) or not torch.isfinite(logits).all():
                raise RuntimeError('Model output is nonfinite or has the wrong class dimension')
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels = logits.argmax(dim=1).cpu().tolist()
            targets = targets.tolist()
            for path, target, prediction, prob in zip(paths, targets, labels, probs):
                writer.writerow([path, classes[target], classes[prediction], classes[target], classes[prediction],
                                 int(target == prediction)] + [float(p) for p in prob])
            truth.extend(targets)
            predicted.extend(labels)
    if len(truth) != len(dataset):
        raise RuntimeError('Incomplete inference: output count does not equal test split size')
    labels = list(range(len(classes)))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    metrics = dict(n=len(truth), classes=classes,
                   f1=float(f1_score(truth, predicted, labels=labels, average='macro', zero_division=0)),
                   ba=float(balanced_accuracy_score(truth, predicted)),
                   accuracy=float(accuracy_score(truth, predicted)),
                   weighted_f1=float(f1_score(truth, predicted, labels=labels, average='weighted', zero_division=0)),
                   report=classification_report(truth, predicted, labels=labels, target_names=classes,
                                                output_dict=True, zero_division=0),
                   confusion_matrix=matrix.tolist())
    for name in ('Myeloblast', 'Lymphoblast', 'Erythroblast'):
        if name in classes:
            i = classes.index(name)
            others = [classes.index(c) for c in ('Myeloblast', 'Lymphoblast', 'Erythroblast') if c in classes and c != name]
            metrics.setdefault('blast_subtypes', {})[name] = dict(
                f1=metrics['report'][name]['f1-score'], support=int(matrix[i].sum()),
                mutual_confusion_fraction=float(matrix[i, others].sum() / matrix[i].sum()))
    (out / 'test_results.json').write_text(json.dumps(metrics, indent=2) + '\n')
    with (out / 'confusion_matrix.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([''] + classes)
        for name, row in zip(classes, matrix):
            writer.writerow([name] + row.tolist())
    provenance = dict(status='COMPLETE', scope='trainer-native internal class decision, no TTA or mapping',
                      trainer=args.trainer, model_name=model_name, class_order=classes,
                      checkpoint_sha256=sha256(args.model_path), predictions_sha256=sha256(prediction_path),
                      evaluator_sha256=sha256(Path(__file__)), protocol=protocol,
                      torch_version=torch.__version__, batch_size=args.batch_size,
                      n=len(dataset), elapsed_seconds=time.time() - started,
                      note='Exact split identity must be established separately by its image manifest; this evaluator does not infer it from sample counts.')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(status='COMPLETE', n=metrics['n'], f1=metrics['f1'], ba=metrics['ba'],
                          output_dir=str(out)), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
