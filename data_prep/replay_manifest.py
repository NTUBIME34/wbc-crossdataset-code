#!/usr/bin/env python3
"""Export private historical input manifests and replay them without model inference.

Manifests may contain dataset identifiers/labels. Keep them outside this repository;
this script does not grant permission to redistribute a dataset or its metadata.
Exact-input replay is distinct from rerunning a detector or reproducing training.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import threading

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
REPO = Path(__file__).resolve().parents[1]
SCHEMA = 'wbc-input-replay-v1'
LOADED_SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
STARTED_AT = datetime.datetime.now().astimezone().isoformat()
_CV_LOCK = threading.Lock()
_CV_READY = False


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def opencv():
    global _CV_READY
    import cv2
    # OpenCV documents setNumThreads as non-thread-safe. Initialize it once
    # before any of this module's worker threads can start image operations.
    if not _CV_READY:
        with _CV_LOCK:
            if not _CV_READY:
                cv2.setNumThreads(1)
                _CV_READY = True
    return cv2


def image(path):
    cv2 = opencv()
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise ValueError(f'Unreadable image: {path}')
    return value


def rgb_digest(value):
    cv2 = opencv()
    rgb = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    result = hashlib.sha256()
    result.update(f'RGB:{rgb.shape[1]}x{rgb.shape[0]}:'.encode('ascii'))
    result.update(rgb.tobytes())
    return result.hexdigest()


def signature(path):
    pixels = image(path)
    return signature_from_pixels(path, pixels)


def signature_from_pixels(path, pixels):
    return dict(sha256=digest(path), rgb_sha256=rgb_digest(pixels),
                shape=list(pixels.shape))


def relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError(f'Unsafe relative path: {value!r}')
    return Path(*path.parts)


def external(path):
    path = Path(path).expanduser().resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError('Historical manifests and replay outputs must stay outside the repository')
    return path


def image_paths(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and
                  p.suffix.lower() in IMAGE_EXTENSIONS and
                  not any(part.startswith('.') for part in p.relative_to(root).parts))


def save_json(path, value):
    path = external(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(value, generator=dict(script_sha256_at_import=LOADED_SCRIPT_SHA256,
                 script_sha256_at_save=digest(Path(__file__)),
                 process_started_at=STARTED_AT,
                 artifact_written_at=datetime.datetime.now().astimezone().isoformat()))
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    return dict(path=str(path), sha256=digest(path))


def new_manifest_path(path):
    path = external(path)
    if path.exists():
        raise FileExistsError(f'Manifest or report already exists: {path}')
    return path


def load_manifest(path, kind):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('kind') != kind:
        raise ValueError('Unexpected manifest schema or kind')
    if not manifest.get('entries'):
        raise ValueError('An empty manifest is not a successful replay')
    return manifest


def unique_targets(entries, key):
    targets = [str(relative(entry[key])) for entry in entries]
    if len(set(targets)) != len(targets):
        raise ValueError('Duplicate target paths in manifest')


class Resolver:
    """Prefer recorded paths, then names; full content matching is explicit."""
    def __init__(self, root, match_content=False, allow_equivalent=False, allowed_roots=None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f'Input folder missing: {self.root}')
        self.paths = image_paths(self.root)
        if not self.paths:
            raise ValueError('No input images found')
        # Prepared class folders may legitimately link to separately downloaded
        # images. Such access must be explicitly declared; preserve the logical
        # folder layout rather than substituting physical paths for labels/IDs.
        self.allowed_roots = normalize_allowed_roots(allowed_roots)
        parent_cache = {}
        for path in self.paths:
            rooted_image(path, self.root, parent_cache, self.allowed_roots)
        self.by_name = collections.defaultdict(list)
        for path in self.paths:
            self.by_name[path.name].append(path)
        self.cache = {}
        self.content = None
        self.match_content = match_content
        self.allow_equivalent = allow_equivalent

    def info(self, path):
        if path not in self.cache:
            self.cache[path] = signature(path)
        return self.cache[path]

    def warm(self, entries, workers, key='relative_path'):
        """Parallelize exact signatures, not label or split decisions."""
        if workers <= 1:
            return
        candidates = set()
        for entry in entries:
            rel = relative(entry[key])
            direct = self.root / rel
            if direct.is_file():
                candidates.add(direct)
            candidates.update(self.by_name[rel.name])
        if self.match_content:
            candidates.update(self.paths)
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            paths = sorted(candidates)
            for number, (path, info) in enumerate(executor.map(path_signature, paths, chunksize=32), 1):
                self.cache[Path(path)] = info
                if number % 2048 == 0:
                    print(f'Validated {number}/{len(paths)} candidate images', file=sys.stderr, flush=True)

    @staticmethod
    def compatible(actual, expected):
        return actual['rgb_sha256'] == expected['rgb_sha256'] and actual['shape'] == expected['shape']

    def resolve(self, name, expected):
        rel = relative(name)
        direct = self.root / rel
        if direct.is_file() and self.compatible(self.info(direct), expected):
            return direct
        matches = [p for p in self.by_name[rel.name] if self.compatible(self.info(p), expected)]
        if not matches and self.match_content:
            if self.content is None:
                self.content = collections.defaultdict(list)
                for path in self.paths:
                    self.content[self.info(path)['rgb_sha256']].append(path)
            matches = [p for p in self.content[expected['rgb_sha256']]
                       if self.compatible(self.info(p), expected)]
        if not matches:
            raise ValueError(f'No exact decoded-RGB input for {name}; rebuilt detector crops are not silently accepted')
        exact_bytes = [p for p in matches if self.info(p)['sha256'] == expected['sha256']]
        if exact_bytes:
            matches = exact_bytes
        if len(matches) > 1 and not self.allow_equivalent:
            raise ValueError(f'Ambiguous input match ({len(matches)} equivalent files) for {name}; resolve layout or explicitly allow equivalent copies')
        return sorted(matches)[0]


def tree_entry(pair):
    path, name = pair
    return dict(relative_path=name, **signature(path))


def path_signature(path):
    return str(path), signature(path)


def export_tree(args):
    new_manifest_path(args.manifest)
    root = args.input_root.expanduser().resolve()
    paths = image_paths(root)
    if not paths:
        raise ValueError('No images found')
    parent_cache = {}
    allowed_roots = normalize_allowed_roots(getattr(args, 'allow_input_root', None))
    for path in paths:
        rooted_image(path, root, parent_cache, allowed_roots)
    expected = getattr(args, 'expected_images', None)
    if expected is not None and len(paths) != expected:
        raise ValueError(f'Expected {expected} images, found {len(paths)}')
    workers = getattr(args, 'workers', 1)
    if workers < 1:
        raise ValueError('--workers must be positive')
    pairs = [(path, path.relative_to(root).as_posix()) for path in paths]
    entries = []
    if workers == 1:
        entries = [tree_entry(pair) for pair in pairs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for row in executor.map(tree_entry, pairs, chunksize=32):
                entries.append(row)
                if len(entries) % 2048 == 0:
                    print(f'Indexed {len(entries)}/{len(paths)} images', file=sys.stderr, flush=True)
    result = dict(schema=SCHEMA, kind='tree', decoder='OpenCV IMREAD_COLOR -> BGR -> RGB',
                  purpose='Historical membership, relative layout and exact image-content replay; not new random splitting',
                  entries=entries)
    summary = collections.Counter('/'.join(Path(r['relative_path']).parts[:2]) for r in entries)
    result['counts'] = dict(summary)
    result['n_images'] = len(entries)
    return save_json(args.manifest, result)


def export_crops(args):
    new_manifest_path(args.manifest)
    source = args.input_root.expanduser().resolve()
    crops = args.crop_root.expanduser().resolve()
    with args.detections.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    if not rows or not {'source_path', 'crop_path', 'status'}.issubset(columns or []):
        raise ValueError('Expected a nonempty pipeline detections.csv')
    workers = getattr(args, 'workers', 1)
    if workers < 1:
        raise ValueError('--workers must be positive')
    allowed_roots = normalize_allowed_roots(getattr(args, 'allow_input_root', None))
    local = threading.local()
    def export_row(row):
        if not hasattr(local, 'parent_cache'):
            local.parent_cache = {}
        parent_cache = local.parent_cache
        src, src_rel = rooted_image(row['source_path'], source, parent_cache, allowed_roots)
        raw = image(src)
        entry = dict(source_relative=src_rel, source=signature_from_pixels(src, raw),
                     detection={k: v for k, v in row.items() if k not in {'source_path', 'crop_path'}})
        if row['status'] == 'detected':
            crop, crop_rel = rooted_image(row['crop_path'], crops, parent_cache)
            if Path(crop_rel).suffix.lower() != '.png':
                raise ValueError('Exact pixel crop replay requires lossless PNG historical crops')
            box = [int(row[k]) for k in ('x1', 'y1', 'x2', 'y2')]
            target = image(crop)
            check_box(box, raw)
            x1, y1, x2, y2 = box
            cut = raw[y1:y2, x1:x2]
            if rgb_digest(cut) != rgb_digest(target) or list(cut.shape) != list(target.shape):
                raise ValueError(f'Historical box does not exactly reproduce historical crop: {src_rel}')
            entry.update(crop_relative=crop_rel, crop=signature_from_pixels(crop, target), box_xyxy=box)
        elif row['status'] != 'no_detection':
            raise ValueError(f'Unsupported detector status: {row["status"]}')
        return entry
    if workers == 1:
        entries = [export_row(row) for row in rows]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            entries = []
            for entry in executor.map(export_row, rows):
                entries.append(entry)
                if len(entries) % 1024 == 0:
                    print(f'Checked {len(entries)}/{len(rows)} source/crop pairs', file=sys.stderr, flush=True)
    unique_targets(entries, 'source_relative')
    unique_targets([r for r in entries if 'crop_relative' in r], 'crop_relative')
    result = dict(schema=SCHEMA, kind='crops', decoder='OpenCV IMREAD_COLOR; integer half-open xyxy slicing',
                  purpose='Replay historical inputs, not fresh YOLO detection',
                  detection_columns=columns, detections_sha256=digest(args.detections), entries=entries,
                  counts=dict(collections.Counter(r['detection']['status'] for r in entries)))
    return save_json(args.manifest, result)


def normalize_allowed_roots(values):
    roots = []
    for value in values or []:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f'Allowed input root is not a directory: {root}')
        roots.append(root)
    return roots


def rooted_image(value, root, parent_cache, allowed_roots=None):
    """Validate physical access but retain the logical relative class/ID path.

    A final/parent symlink escaping the input tree is rejected unless its resolved
    target lies under an explicitly allowed additional input root. The recorded
    relative path always comes from the input tree, never the symlink target.
    """
    root = Path(root).expanduser().resolve()
    path = Path(os.path.abspath(str(Path(value).expanduser())))
    name = path.relative_to(root).as_posix()
    relative(name)
    current, seen = path, set()
    for _ in range(40):
        if current.parent not in parent_cache:
            parent_cache[current.parent] = current.parent.resolve()
        resolved = parent_cache[current.parent] / current.name
        if not resolved.is_symlink():
            break
        if resolved in seen:
            raise ValueError(f'Input symlink cycle: {name}')
        seen.add(resolved)
        target = Path(os.readlink(resolved))
        current = target if target.is_absolute() else resolved.parent / target
    else:
        raise ValueError(f'Too many input symlinks: {name}')
    permitted = [root] + list(allowed_roots or [])
    if not any(resolved == allowed or allowed in resolved.parents for allowed in permitted):
        raise ValueError(f'Resolved input escapes declared roots: {name}')
    return resolved, name


def recover_exact_box(raw, crop):
    """Locate a lossless sub-image by exact pixels, never morphological similarity.

    Pixel anchors avoid the rounding and FFT errors of template-matching scores.
    Every surviving candidate must equal the entire crop; ambiguous matches fail.
    """
    import numpy as np
    height, width = crop.shape[:2]
    if height > raw.shape[0] or width > raw.shape[1]:
        raise ValueError('Historical crop is larger than its candidate raw image')
    anchors = [(height // 2, width // 2), (height // 3, width // 3),
               (2 * height // 3, 2 * width // 3), (0, 0), (height - 1, width - 1)]
    ay, ax = anchors[0]
    ys, xs = np.where(np.all(raw == crop[ay, ax], axis=2))
    ys, xs = ys - ay, xs - ax
    valid = (ys >= 0) & (xs >= 0) & (ys + height <= raw.shape[0]) & (xs + width <= raw.shape[1])
    ys, xs = ys[valid], xs[valid]
    for ay, ax in anchors[1:]:
        valid = np.all(raw[ys + ay, xs + ax] == crop[ay, ax], axis=1)
        ys, xs = ys[valid], xs[valid]
    matches = []
    for y, x in zip(ys, xs):
        if np.array_equal(raw[y:y + height, x:x + width], crop):
            matches.append((int(x), int(y)))
            if len(matches) > 1:
                raise ValueError('Expected one exact raw-to-crop pixel location, found multiple')
    if len(matches) != 1:
        raise ValueError(f'Expected one exact raw-to-crop pixel location, found {len(matches)}')
    x, y = matches[0]
    return [x, y, x + width, y + height]


def aligned_entry(pair):
    """Worker for author-side raw/crop coordinate recovery."""
    raw_path, crop_path, name = pair
    try:
        raw, cropped = image(raw_path), image(crop_path)
        box = recover_exact_box(raw, cropped)
        row = dict(source_relative=name, source=signature_from_pixels(raw_path, raw),
                   crop_relative=name, crop=signature_from_pixels(crop_path, cropped), box_xyxy=box,
                   detection=dict(status='detected', det_index='0', yolo_class_id='',
                                  yolo_class_name='', yolo_conf='',
                                  **dict(zip(('x1', 'y1', 'x2', 'y2'), map(str, box)))))
        return dict(entry=row)
    except (ValueError, OSError) as exc:
        return dict(relative_path=name, error=str(exc))


def export_aligned_crops(args):
    """Recover geometry only when the entire historical crop equals an exact ROI."""
    from concurrent.futures import ProcessPoolExecutor
    target = new_manifest_path(args.manifest)
    report_path = new_manifest_path(args.report)
    if target == report_path:
        raise ValueError('--report and --manifest must be different files')
    source = args.input_root.expanduser().resolve()
    crops = args.crop_root.expanduser().resolve()
    paths = image_paths(crops)
    if not paths:
        raise ValueError('No crop images found')
    pairs, errors = [], []
    raw_parent_cache, crop_parent_cache = {}, {}
    allowed_roots = normalize_allowed_roots(getattr(args, 'allow_input_root', None))
    for crop in paths:
        name = crop.relative_to(crops).as_posix()
        src = source / relative(name)
        if crop.suffix.lower() != '.png':
            errors.append(dict(relative_path=name, error='Historical target is not lossless PNG'))
        elif not src.is_file():
            errors.append(dict(relative_path=name, error='No raw image at aligned relative path'))
        else:
            raw_path, _ = rooted_image(src, source, raw_parent_cache, allowed_roots)
            crop_path, _ = rooted_image(crop, crops, crop_parent_cache)
            pairs.append((raw_path, crop_path, name))
    entries = []
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for number, result in enumerate(executor.map(aligned_entry, pairs, chunksize=16), 1):
            if 'entry' in result:
                entries.append(result['entry'])
            else:
                errors.append(result)
            if number % 2048 == 0:
                print(f'Checked {number}/{len(pairs)} aligned pairs; failures {len(errors)}', file=sys.stderr, flush=True)
    report = dict(schema=SCHEMA, kind='aligned_crop_recovery_report',
                  historical_crops=len(paths), exact_unique_subimage_matches=len(entries),
                  failures=len(errors), failure_details=errors,
                  geometry_source='Retrospectively recovered by complete exact pixel-subimage equality, not detector logs or visual similarity',
                  note='No inference confidence or detector exclusion decisions are inferred from these matches')
    save_json(args.report, report)
    if errors:
        raise ValueError(f'{len(errors)} aligned crops did not match exactly; no complete replay manifest was exported (see --report)')
    unique_targets(entries, 'source_relative')
    result = dict(schema=SCHEMA, kind='crops', decoder='OpenCV IMREAD_COLOR; integer half-open xyxy slicing',
                  purpose='Replay historical crops with retrospectively recovered exact pixel geometry; not fresh detector output',
                  geometry_provenance=report['geometry_source'],
                  detection_columns=['source_path', 'crop_path', 'status', 'det_index', 'yolo_class_id', 'yolo_class_name', 'yolo_conf', 'x1', 'y1', 'x2', 'y2'],
                  entries=entries, counts=dict(detected=len(entries)))
    return save_json(args.manifest, result)


def check_box(box, pixels):
    if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in box):
        raise ValueError('Box must contain four integers')
    x1, y1, x2, y2 = box
    h, w = pixels.shape[:2]
    if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
        raise ValueError(f'Invalid or out-of-bounds box {box} for {w}x{h}')


def new_output_path(path, input_root):
    path = external(path)
    source = input_root.expanduser().resolve()
    if path == source or source in path.parents or path in source.parents:
        raise ValueError('Output must not overlap the source input tree')
    if path.exists():
        raise ValueError('Output folder already exists; choose a new empty destination')
    return path


def output_folder(path, input_root):
    path = new_output_path(path, input_root)
    path.mkdir(parents=True)
    return path


def resolver(args):
    return Resolver(args.input_root, args.match_content, args.allow_equivalent_copies,
                    getattr(args, 'allow_input_root', None))


def save_tree_image(plan):
    entry, src, dst, copy_mode, same_bytes = plan
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if same_bytes:
        if copy_mode == 'hardlink':
            os.link(src, dst)
        else:
            shutil.copyfile(src, dst)
    else:
        import cv2
        if not cv2.imwrite(str(dst), image(src)):
            raise ValueError(f'Cannot save {dst}')
    if not Resolver.compatible(signature(dst), entry):
        raise ValueError(f'Replayed image does not match: {entry["relative_path"]}')
    return int(same_bytes)


def replay_tree(args):
    new_output_path(args.output_dir, args.input_root)
    manifest = load_manifest(args.manifest, 'tree')
    unique_targets(manifest['entries'], 'relative_path')
    index = resolver(args)
    workers = getattr(args, 'workers', 1)
    if workers < 1:
        raise ValueError('--workers must be positive')
    index.warm(manifest['entries'], workers)
    planned = []
    for entry in manifest['entries']:
        src = index.resolve(entry['relative_path'], entry)
        if index.info(src)['sha256'] != entry['sha256'] and Path(entry['relative_path']).suffix.lower() != '.png':
            raise ValueError('A different encoding can only be replayed to a lossless PNG target')
        planned.append((entry, src))
    output = output_folder(args.output_dir, args.input_root)
    save_plans = [(entry, src, output / relative(entry['relative_path']),
                   getattr(args, 'copy_mode', 'copy'), index.info(src)['sha256'] == entry['sha256'])
                  for entry, src in planned]
    if workers == 1:
        byte_equal = sum(save_tree_image(plan) for plan in save_plans)
    else:
        from concurrent.futures import ProcessPoolExecutor
        byte_equal = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for number, same_bytes in enumerate(executor.map(save_tree_image, save_plans, chunksize=32), 1):
                byte_equal += same_bytes
                if number % 2048 == 0:
                    print(f'Replayed {number}/{len(save_plans)} images', file=sys.stderr, flush=True)
    report = dict(kind='tree_replay', manifest_sha256=digest(args.manifest),
                  images=len(planned), decoded_rgb_equal=len(planned), bytes_equal=byte_equal,
                  note='Exact split membership and decoded image pixels; not a training reproducibility claim')
    save_json(output / 'REPLAY_REPORT.json', report)
    return report


def replay_crops(args):
    new_output_path(args.output_dir, args.input_root)
    manifest = load_manifest(args.manifest, 'crops')
    unique_targets(manifest['entries'], 'source_relative')
    unique_targets([e for e in manifest['entries'] if e['detection']['status'] == 'detected'], 'crop_relative')
    index = resolver(args)
    workers = getattr(args, 'workers', 1)
    if workers < 1:
        raise ValueError('--workers must be positive')
    if workers > 1 and index.match_content:
        # Construct one immutable content index before worker threads use it.
        index.content = collections.defaultdict(list)
        for path in index.paths:
            index.content[index.info(path)['rgb_sha256']].append(path)
    def validate_entry(entry):
        if entry['detection']['status'] not in {'detected', 'no_detection'}:
            raise ValueError('Unsupported detector status')
        src = index.resolve(entry['source_relative'], entry['source'])
        if entry['detection']['status'] == 'detected':
            raw = image(src)
            check_box(entry['box_xyxy'], raw)
            x1, y1, x2, y2 = entry['box_xyxy']
            cut = raw[y1:y2, x1:x2]
            if rgb_digest(cut) != entry['crop']['rgb_sha256'] or list(cut.shape) != entry['crop']['shape']:
                raise ValueError(f'Replayed crop differs: {entry["source_relative"]}')
            if Path(entry['crop_relative']).suffix.lower() != '.png':
                raise ValueError('Crop target must be a lossless PNG')
        return entry, src
    if workers == 1:
        planned = [validate_entry(entry) for entry in manifest['entries']]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            planned = []
            for plan in executor.map(validate_entry, manifest['entries']):
                planned.append(plan)
                if len(planned) % 1024 == 0:
                    print(f'Validated {len(planned)}/{len(manifest["entries"])} fields', file=sys.stderr, flush=True)
    output = output_folder(args.output_dir, args.input_root)
    def save_entry(plan):
        entry, src = plan
        # Downstream scoring takes field ID and native label from the source path.
        # A content-matched download may have different filenames/class folders;
        # relocate a copy to the recorded layout rather than silently changing IDs
        # or labels. Direct matching prepared inputs are not duplicated.
        field_src = src
        if src.relative_to(index.root).as_posix() != entry['source_relative']:
            field_src = output / 'fields' / relative(entry['source_relative'])
            field_src.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, field_src)
        row = dict(entry['detection'], source_path=str(field_src), crop_path='')
        if entry['detection']['status'] == 'detected':
            import cv2
            x1, y1, x2, y2 = entry['box_xyxy']
            dst = output / 'crops' / relative(entry['crop_relative'])
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dst), image(src)[y1:y2, x1:x2]):
                raise ValueError(f'Cannot save {dst}')
            if not Resolver.compatible(signature(dst), entry['crop']):
                raise ValueError('Saved crop differs from the manifest')
            row['crop_path'] = str(dst)
        return row
    if workers == 1:
        rows = [save_entry(plan) for plan in planned]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = []
            for row in executor.map(save_entry, planned):
                rows.append(row)
                if len(rows) % 1024 == 0:
                    print(f'Replayed {len(rows)}/{len(planned)} fields', file=sys.stderr, flush=True)
    n = sum(row['status'] == 'detected' for row in rows)
    with (output / 'detections.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest['detection_columns'])
        writer.writeheader()
        writer.writerows(rows)
    report = dict(kind='historical_crop_replay', manifest_sha256=digest(args.manifest),
                  fields=len(rows), crops=n, decoded_rgb_equal=n,
                  statuses=dict(collections.Counter(row['status'] for row in rows)),
                  note='Saved historical coordinates, not a fresh detector test; PNG encoding bytes may differ')
    save_json(output / 'REPLAY_REPORT.json', report)
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for command in ('export-tree', 'export-crops', 'export-aligned-crops', 'replay-tree', 'replay-crops'):
        p = sub.add_parser(command)
        p.add_argument('--input-root', type=Path, required=True)
        p.add_argument('--manifest', type=Path, required=True)
        p.add_argument('--allow-input-root', type=Path, action='append', default=[],
                       help='Explicit additional root for prepared input symlink targets; repeatable. Logical labels/IDs remain under --input-root.')
        if command == 'export-tree':
            p.add_argument('--workers', type=int, default=1)
            p.add_argument('--expected-images', type=int, help='Optional reference image count; a snapshot alone does not establish paper completeness')
        if command == 'export-crops':
            p.add_argument('--detections', type=Path, required=True)
            p.add_argument('--crop-root', type=Path, required=True)
            p.add_argument('--workers', type=int, default=1)
        if command == 'export-aligned-crops':
            p.add_argument('--crop-root', type=Path, required=True)
            p.add_argument('--report', type=Path, required=True)
            p.add_argument('--workers', type=int, default=1)
        if command.startswith('replay'):
            p.add_argument('--output-dir', type=Path, required=True)
            p.add_argument('--match-content', action='store_true', help='Build a decoded-RGB content index if paths/names do not match (can be expensive)')
            p.add_argument('--allow-equivalent-copies', action='store_true', help='Explicitly allow multiple exact-content source matches; does not assert cell identity')
        if command == 'replay-crops':
            p.add_argument('--workers', type=int, default=1)
        if command == 'replay-tree':
            p.add_argument('--copy-mode', choices=('copy', 'hardlink'), default='copy', help='Hardlink requires one filesystem; never edit replayed linked images')
            p.add_argument('--workers', type=int, default=1)
    return ap


def main():
    os.umask(0o077)
    args = parser().parse_args()
    functions = {'export-tree': export_tree, 'export-crops': export_crops,
                 'export-aligned-crops': export_aligned_crops,
                 'replay-tree': replay_tree, 'replay-crops': replay_crops}
    try:
        print(json.dumps(functions[args.command](args), indent=2))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
