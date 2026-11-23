# Historical input and split replay

`replay_manifest.py` closes two different input-reconstruction gaps without changing
weights or substituting new reference scores:

- Export an author's saved integer detector boxes with exact source/crop content
  fingerprints, then rebuild those **historical** crops from the source images.
- Export the exact relative membership of a historical training/validation/test tree,
  then reconstruct that tree from matching image content rather than drawing a new
  random split from renamed files.

This is an input replay utility, not a detector evaluation, dataset downloader, or
claim of deterministic retraining. Use `pipeline/detect_yolo11.py` separately to test
fresh detection. A fresh detector may return different rounded boxes.

## Privacy and prerequisites

No historical manifests are distributed in this code directory. Manifests can contain
image identifiers, class labels and split membership. Keep restricted-dataset
manifests in a private evidence package and obtain source data under its terms. Every
manifest and replay output must be outside the repository. The script needs OpenCV
and NumPy from `environment/journal_env.yml`; no torch, CUDA or model weights are
required.

## Saved detector boxes

The author's export requires the existing `detections.csv`, original fields and PNG
crops. Every detected box is checked against the historical crop before any manifest
is written. Invalid boxes, missing images and crop mismatches fail explicitly.

```bash
python data_prep/replay_manifest.py export-crops \
  --detections /private/historical_detection/detections.csv \
  --input-root /private/original_fields \
  --crop-root /private/historical_detection/crops \
  --manifest /private/evidence/field_boxes.json

python data_prep/replay_manifest.py replay-crops \
  --manifest /private/evidence/field_boxes.json \
  --input-root /licensed_download/prepared_fields \
  --output-dir /private/new_historical_replay
```

The output mirrors the pipeline layout: `crops/`, `detections.csv` with relocated
paths, and `REPLAY_REPORT.json`. Saved `no_detection` rows are retained; they are not
silently deleted. The reconstruction must match the decoded RGB pixel hash of every
historical crop, although PNG compression bytes need not match. Inputs are fully
validated before the new output directory is created.

If content matching resolves an input whose original filename or class folder differs,
the script copies it to a `fields/` subdirectory using the recorded relative path.
This preserves the field IDs and native labels expected by coverage-aware scoring;
matching prepared inputs need no additional field-image copy.

Prepared class folders sometimes contain file symlinks into a separately downloaded
raw-image directory. Such targets are rejected by default. Declare each permitted
physical data root explicitly on both export and replay, for example:

```bash
python data_prep/replay_manifest.py replay-crops \
  --manifest /private/evidence/field_boxes.json \
  --input-root /licensed_download/prepared_fields \
  --allow-input-root /licensed_download/raw_images \
  --output-dir /private/new_historical_replay
```

`--allow-input-root` is repeatable and does not authorize access outside those roots.
The manifest retains the logical prepared path, native class and field ID, not the
symlink target's folder or filename. Content hashes still have to match. Keep input
files and links unchanged throughout validation and replay.

`export-crops --workers 8` and `replay-crops --workers 8` can amortize image-file I/O
latency with ordered worker threads. The default is one worker. Replay still validates
every input and crop before creating its output folder; parallel writes retain the
original manifest/CSV row order. OpenCV's internal thread setting is initialized once,
not changed concurrently by image workers.

## Exact split membership

```bash
python data_prep/replay_manifest.py export-tree \
  --input-root /private/historical_corpus_split \
  --manifest /private/evidence/internal_split.json

python data_prep/replay_manifest.py replay-tree \
  --manifest /private/evidence/internal_split.json \
  --input-root /licensed_download/matching_merged_crops \
  --output-dir /private/replayed_internal_split
```

Each manifest entry fixes the target path (for example `test/Neutrophil/image.png`),
file SHA-256, RGB SHA-256 and image dimensions. Replay first tries the recorded
relative path and filename. `--match-content` explicitly builds a decoded-RGB index
when names changed; this can take substantial time. Ambiguous equivalent source
matches fail by default. `--allow-equivalent-copies` permits selection only among
exact-content matches; it does not assert that matching images are the same cell.

`export-tree --workers 4` can parallelize indexing. If a reference image count is
known, pass `--expected-images N`; an unexpected count fails before export. Without
this flag the utility snapshots the actual input tree only: an internally consistent
snapshot is not evidence that every class or image expected by a paper was present.

**Boundary:** the split manifest requires the actual historical crop pixels, not
approximately equivalent crops from a newly run detector. It preserves membership
and labels but cannot recover missing historical inputs. Internal-corpus raw-to-crop
provenance and source-file matching must be established separately before claiming
independent reconstruction from public raw downloads. A replay that cannot resolve
an exact image fails rather than quietly constructing a different test set.

Existing output directories and manifest files are never overwritten. Paths that
escape the target directory, duplicate targets, unsupported statuses and empty
manifests are errors. `python -B -m unittest discover -s tests -p
test_replay_manifest.py -v` runs synthetic offline fixtures.

For an author-side copy on one filesystem, `replay-tree --copy-mode hardlink` avoids
duplicating image storage. Do not edit any linked image: it shares storage with its
input. The default is an independent file copy.

## Recover geometry from aligned historical raw/crop pairs

If an author's uncropped merged corpus and lossless historical crops still have the
same relative filenames, geometry can be recovered without rerunning the detector:

```bash
python data_prep/replay_manifest.py export-aligned-crops \
  --input-root /private/historical_merged_uncropped \
  --crop-root /private/historical_merged_crops \
  --manifest /private/evidence/internal_raw_to_crop.json \
  --report /private/evidence/internal_geometry_recovery.json --workers 4
```

Only a **unique, complete pixel-exact sub-image** is accepted. Matching one pixel,
perceptual similarity, a high template-correlation score, or a nearby detector box
is not sufficient. If any crop is absent, ambiguous or nonidentical, the command
exits unsuccessfully and writes only a diagnostic report, not a complete manifest.
Recovered coordinates are labeled **retrospectively recovered geometry**, not
contemporaneous detector logs. Confidence values and excluded-cell decisions are
not inferred. The resulting crop manifest can be used with `replay-crops`; raw
images can be matched by content when a public download has different names.
