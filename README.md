# White blood cell detection and classification across datasets: code

This repository holds the code behind a paper on cross-dataset white blood cell (WBC)
classification. A YOLO11 detector finds the white blood cell in a blood-smear field image
and crops it, and a Swin-B classifier assigns one of nine cell types; for external
evaluation the nine classes are mapped to the seven classes of the WBCBench2026 benchmark.
The repository covers the deployable inference pipeline, data preparation, training of
every reported model, evaluation, statistics, figures, and a verifier that recomputes the
paper's numbers from public evidence.

No dataset images and no trained weights are stored here. The weights are distributed
separately (see [Trained weights](#trained-weights)); the datasets are public and are
downloaded from their sources (see [Data](#data)).

## Repository layout

```text
pipeline/      inference: field image -> YOLO11 detection and crop -> Swin-B classification
data_prep/     rebuilds the dataset folders from the public downloads: detector crops, label
               harmonization, splits, leave-one-dataset-out folds, WBCBench2026 class
               folders, Cellpose crops
train/         one trainer per model family, one runner per paper row (train/scripts/),
               the fetch script for the BPaCo loss (train/third_party/)
eval/          evaluators and one runner per paper analysis (WBCBench2026 and
               leave-one-dataset-out)
stats/         statistics from per-image outputs: seed summaries, coverage-aware scores,
               bootstrap intervals, paired tests, recipe ensembles, thresholds, calibration
figures/       Fig. 2, Fig. 3 and Supplementary Figs. S1 and S2
analysis/      public evidence behind the reported numbers and verify_tables.py
environment/   conda environment files and the step-to-environment table
docs/          PAPER_MAP.md: every table and figure -> scripts, inputs, expected values
reproduce/     study-wide task registry, preflight/execution logs and numerical comparisons
LICENSE        MIT licence of the authors' code
THIRD_PARTY_NOTICES.md
```

Each code folder has its own README with the details. `models/` (weights) and `outputs/`
are git-ignored.

## Installation

```bash
conda env create -f environment/journal_env.yml
conda activate journal_env
```

`journal_env` (Python 3.10.20, PyTorch 2.5.1 with CUDA 12.1, timm 1.0.27, Ultralytics
8.4.82) runs the pipeline, every WBCBench2026 evaluation, the statistics, the figures and
the verifier. `environment/cnn_env.yml` reproduces the CNN-baseline training and the
leave-one-dataset-out evaluation as run, `environment/cellpose_env.yml` makes the Cellpose
crops of Supplementary routes C and D, and detector training uses Ultralytics 8.3.170.
`environment/README.md` maps every step to its environment and explains CPU-only installs.

## Quick start: run the pipeline on your own images

1. Download the weight files from the weights record (see [Trained weights](#trained-weights))
   and put them in `models/` at the repository root, under their record names. The
   pipeline needs the detector and one classifier seed; the other two seeds are for the
   three-seed ensemble:

   ```text
   models/yolo11_wbc_detection_best.pt
   models/swin_base_colour_aug_cwce_seed62.pth      (default classifier)
   models/swin_base_colour_aug_cwce_seed42.pth
   models/swin_base_colour_aug_cwce_seed82.pth
   ```

   The scripts look for weights in `$MODEL_DIR`, then `pipeline/models/`, then `models/`.

2. Run detection and classification on a folder of field images (searched recursively):

   ```bash
   export OUT_ROOT=/path/to/outputs        # a folder outside the repository
   bash pipeline/run_pipeline.sh /path/to/field_images "$OUT_ROOT/my_run"
   ```

   `DEVICE=cpu` forces CPU inference. The three-seed probability ensemble:

   ```bash
   M=models
   CLASSIFIER_CKPTS=$M/swin_base_colour_aug_cwce_seed42.pth,$M/swin_base_colour_aug_cwce_seed62.pth,$M/swin_base_colour_aug_cwce_seed82.pth \
     bash pipeline/run_pipeline.sh /path/to/field_images "$OUT_ROOT/my_run_ensemble"
   ```

   A run writes `01_detection/detections.csv` with the crops and
   `02_classification/predictions.csv`, which holds the nine-class prediction, the
   seven-class label `pred_label` and the seven class probabilities. An image without a
   detected cell gets the status `no_detection` and is not classified.

3. Optional, with labelled images (one sub-folder per class, for example `Neutrophil/`):
   score the run so that every missed cell counts as an error, as in the paper's primary
   endpoint.

   ```bash
   python stats/coverage_aware_score.py --name my_run \
     --predictions "$OUT_ROOT/my_run/02_classification/predictions.csv" \
     --detections "$OUT_ROOT/my_run/01_detection/detections.csv"
   ```

The detector settings are the paper's (512 px, confidence 0.30, NMS IoU 0.70, the
highest-confidence box expanded by 10% on each side), and the classifier uses the
paper's WBCBench2026 evaluation transform (square padding, resize to 256 px, centre crop
224, horizontal-flip test-time augmentation). `pipeline/README.md` lists all options
and output columns.

The scorer accepts the supported class names and aliases case-insensitively, including
the native lowercase PBC folder names. Unknown class names are errors, not silently
counted as incorrect predictions. With a detection CSV, missing classifier predictions,
extra image IDs or inconsistent truth labels are also errors. A genuine detector miss
continues to count as an error in the coverage-aware metric.

Run the offline scoring and full-pipeline preflight regression tests with:

```bash
python -B -m unittest discover -s tests -v
```

These tests need no dataset download or model weights. The full-pipeline evaluation
runner requires every seed requested by `SEEDS`; a missing checkpoint fails the run.

## Reproducing the paper

### Study-wide execution and result comparison

`reproduce/run_study.py` provides one registry for the main text and supplement,
including retained-weight inference, trainer-native internal evaluation, input replay,
statistics and the original training entry points. It checks every requested checkpoint
before starting a task, records exact commands and source hashes, and rejects missing,
stale or failed outputs. A completed command is not automatically a numerical match.

```bash
python -B reproduce/run_study.py --plan
python -B reproduce/run_study.py --check --tasks cnn-external,dinobloom-external \
  --config /path/to/local_config.json --keep-going
python -B reproduce/run_study.py --run --tasks cnn-external,dinobloom-external \
  --config /path/to/local_config.json --keep-going
```

The configuration supplies `DATA_ROOT`, `MODEL_DIR` (the original 13-weight core),
`MODEL_ROOT` (additional retained study checkpoints), `OUT_ROOT`, and the interpreters
for the environments. See [`reproduce/README.md`](reproduce/README.md) for the model
tree and task order. Network training requires explicit `--allow-training`.

`reproduce/compare_results.py` independently compares generated per-seed confusion
counts and metrics with the retained public reference. Numerical differences remain
differences even if both rounded scores happen to match. Image IDs and per-image outputs
stay outside the repository.

Exact internal membership and historical crop geometry can be reconstructed with
[`data_prep/replay_manifest.py`](data_prep/REPLAY.md), using a verified input manifest
and your own licensed image files. This is separate from rerunning the detector on raw
images: replay reproduces historical inputs; fresh detection tests the pipeline in the
current environment. Neither route silently substitutes for the other.

### Evidence verification and regeneration

Successful command execution, numerical agreement with the paper, and complete
retraining are separate checks. The released weights support inference without
retraining. A run using locally supplied, checksum-verified dataset files does not
require that the inference code download them again. Regenerating historical detector
crops is a separate input-reconstruction test: preserve the regenerated and historical
inputs separately, and report numerical differences rather than replacing the paper's
reference values or treating process exit status as a numerical pass.

[`docs/PAPER_MAP.md`](docs/PAPER_MAP.md) lists, for every table and figure of the main
text and the supplement, the scripts that produce it, their inputs, the expected values
with their line numbers in the LaTeX sources, the public evidence that checks it, and
whether it can be regenerated with the released weights.

**Check the numbers without any per-image WBCBench2026 data:**

```bash
python analysis/verify_tables.py
```

The default mode recomputes the printed values from the public files in `analysis/`
(WBCBench2026 count matrices, internal confusion matrices, leave-one-dataset-out
predictions on the public training sources, SHA-256 hash lists and the cited summary
tables) and compares each with the paper at the printed precision. Checks that need
per-image WBCBench2026 predictions are reported as SKIPPED; the run fails if any check
fails.

**Add the per-image checks with outputs you regenerate yourself:**

```bash
python analysis/verify_tables.py --private-evidence /path/to/your_evidence
```

The folder (outside the repository) holds your own evaluator and detector outputs on
WBCBench2026, laid out as `analysis/README.md` describes. This mode rebuilds the public
count files from them and adds the checks that need the images one by one, such as the
image-level bootstraps of Supplementary S7 and S9; checks whose inputs are missing stay
SKIPPED. `analysis/README.md` gives the current check counts of both modes.

**Regenerate the results.** The usual order is:

1. `data_prep/`: build the dataset folders under `DATA_ROOT` from the public downloads
   (`data_prep/README.md`).
2. `train/`: retrain the models whose weights are not released, one runner per paper row
   (`train/README.md`).
3. `eval/`: run the evaluation runner of each analysis; the runners call the matching
   `stats/` script for seed summaries and for coverage-aware and leave-one-dataset-out
   scores (`eval/README.md`).
4. `stats/` and `figures/`: ensembles, bootstrap intervals, paired tests, calibration,
   thresholds and the figures, from the per-image outputs (`stats/README.md`,
   `figures/README.md`).
5. `python analysis/verify_tables.py --private-evidence ...` on your outputs.

**Configurations.** There are no separate configuration files. Each training
configuration is the set of flags in its `train/scripts/` runner, checked against the
arguments saved by every reported run (`train/README.md`, "Provenance of the runner
flags"); the detector recipe is the defaults of `train/train_detector.py`, which match
the Ultralytics arguments recorded by the reported run (`train/scripts/run_detector.sh`);
and each evaluation setting is a flag that the matching `eval/` runner passes to its
evaluator (`eval/README.md`).

The 13 core weight files regenerate the full-pipeline results (Table 2, the primary
endpoint), the pre-cropped results of the proposed model and of the BPaCo comparison
models, the cropping routes of Table 2b and Supplementary Table S6, the
missed-versus-detected comparison (S12), the resolution sweep and the external
calibration and confusion matrix. The CNN baselines, fine-tuned DinoBloom-B, the other
training configurations and the leave-one-dataset-out models are not in the weights
record; `train/scripts/` retrains them (run folders under `MODEL_ROOT`, `eval/README.md`).
The frozen DinoBloom-B probe is refitted by `eval/frozen_probe.py` from the pinned public
backbone. Table 3's internal column (its eight trained rows; the checkpoints of the
proposed model and the three BPaCo models are released) and S4 can be regenerated
without training by
`eval/evaluate_internal.py`, which uses each checkpoint's actual trainer-native transform
and preserves its ordered nine classes. Fig. 3 internal and S11 use the separate
seven-class common-evaluator path. All internal endpoints require the exact historical
split, not a new random split with the same seed; the input-replay manifest supplies that
membership when available. The public internal confusion matrices independently check
the retained results. Hematologist judgments (S13, Section 3.9) are human evidence, not
model outputs, and their per-image records remain withheld.

**Conventions.**

- `DATA_ROOT` holds the prepared datasets, `OUT_ROOT` the outputs (default `./outputs`,
  resolved against the current directory), `MODEL_DIR` the weights. The scripts in
  `data_prep/`, `eval/`, `stats/` and `figures/` and the pipeline runner
  `pipeline/run_pipeline.sh` refuse to write inside the repository, so set `OUT_ROOT` to
  a folder outside it.
- Seeds: 42/62/82 for the Swin-B, BPaCo and DinoBloom-B models, 42/43/44 for the CNN
  baselines, 42/52/62 for leave-one-dataset-out. Seed summaries are the mean and the
  population standard deviation.
- External results are in the seven evaluation classes, with horizontal-flip
  test-time augmentation.
- GPU inference and training are not bit-reproducible across hardware and library
  versions, so a few individual predictions, and seed-level values of retrained models,
  can differ slightly from the paper's runs.
- Keep every per-image output on WBCBench2026 on your own machine (see [Data](#data)).
  The repository's `.gitignore` ignores CSV and image files by default.

## Data

All datasets are public; download them from their sources and follow each licence or set
of terms. No dataset image or label file is redistributed here.

| Dataset | Used for | Source | Licence or terms |
|---|---|---|---|
| AML-Cytomorphology_LMU (AML-LMU) | classifier training; leave-one-dataset-out | TCIA, doi:10.7937/tcia.2019.36f5o9ld | CC BY 3.0 |
| Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer (BMCMHF) | classifier training; leave-one-dataset-out | TCIA, doi:10.7937/TCIA.AXH3-T579 | CC BY 4.0 |
| PBC | classifier training; leave-one-dataset-out | Mendeley Data, doi:10.17632/snkd93bnjr.1 | CC BY 4.0 |
| Raabin-WBC | classifier training; leave-one-dataset-out | <https://raabindata.com> | free access with citation under the Raabin Health Database terms |
| HRLS | classifier training; leave-one-dataset-out | figshare, doi:10.6084/m9.figshare.22680517 | CC0 1.0 |
| TXL-PBC | detector training | figshare, doi:10.6084/m9.figshare.27073186 | CC BY 4.0 |
| WBCBench2026 | external evaluation only | Kaggle competition `wbc-bench-2026` | listed as CC BY-NC-SA 4.0; the competition rules restrict redistribution |

**WBCBench2026.** The paper used the competition files downloaded in November 2025
(Phase-1: 8,288 field images; Phase-2 evaluation split: 5,350), whose rules restrict
redistribution. A later Hugging Face release corrects 257 labels, removes 10 images and
adds a patient hash, so it gives different counts. The benchmark was used only for
evaluation, never for training or checkpoint selection (Phase-1 results did inform the
choice of augmentation and test-time augmentation; Supplementary Section S17). This
repository contains no WBCBench2026 images, labels, image IDs or per-image predictions:
only aggregate counts (`analysis/evidence/wbcbench_confusion/`) and the SHA-256 hashes of
the Phase-1 fields used for the detector-overlap audit
(`analysis/evidence/leakage_audit/`, hashes only). To reproduce the WBCBench2026 results,
download the competition files yourself, build the class folders with
`data_prep/prepare_wbcbench.py`, and regenerate the predictions with the released weights
(`eval/README.md`). Keep these per-image outputs on your own machine; only aggregate
tables may be shared.

**Raabin-WBC.** Its images are free to access with citation under the Raabin Health
Database terms, not under a Creative Commons licence. They are not reproduced in any
figure. The leave-one-dataset-out evidence files (`analysis/evidence/lodo*/`) list file
names, labels and predictions for the public training sources, including Raabin-WBC;
these rows follow that dataset's terms.

Example images in Figs. 1 and 2 and Supplementary Fig. S1 come only from PBC, HRLS,
AML-LMU and BMCMHF, under the licences above; `figures/` reads them from your own
download.

## Trained weights

The weights record holds the 13 core detector and classifier checkpoints in
`weights_core/`, with a README, `MANIFEST.csv` and `SHA256SUMS.txt`. It is distributed
separately from the code. The release candidate is not itself evidence that a public
download record is live.
[`docs/RELEASE_METADATA.json`](docs/RELEASE_METADATA.json) records public release fields
explicitly: a `null` value means that no publicly verified value is available yet.

<!-- named-only:start -->
Configured repository origin: <https://github.com/NTUBIME34/wbc-crossdataset-code>.
This is the Git origin, not a claim that the repository is already publicly accessible.

Reserved weights DOI: `10.5281/zenodo.21465935`. A reservation is not a published
record or a usable weight-download link. The actual public record and file URLs must
be recorded after publication.
<!-- named-only:end -->

| File | Model | Seeds |
|---|---|---|
| `yolo11_wbc_detection_best.pt` | YOLO11-M WBC detector (Stage 1), trained on TXL-PBC WBC boxes; best epoch (107) | 3407 |
| `swin_base_colour_aug_cwce_seed{42,62,82}.pth` | proposed Swin-B classifier: colour augmentation, class-weighted cross-entropy | 42, 62, 82 |
| `swin_base_bpaco_seed{42,62,82}.pth` | Swin-B, BPaCo, AdamW (shared protocol) | 42, 62, 82 |
| `resnet50_bpaco_adamw_seed{42,62,82}.pth` | ResNet50, BPaCo, AdamW (shared protocol) | 42, 62, 82 |
| `resnet50_bpaco_sgd_seed{42,62,82}.pth` | ResNet50, BPaCo, SGD (earlier arm) | 42, 62, 82 |

The classifier files are checkpoint dictionaries: the timm model weights (`state_dict`), the
class list (`classes`), the training arguments (`args`), the selected epoch (`epoch`) and
its validation macro-F1 (`best_macro_f1`). `pipeline/classify.py` reads the model name and
class order from them, `eval/evaluate_classifier.py` the class order (it takes the model
name from `--model-name`). The detector file is an Ultralytics checkpoint (a pickled model
object); check its SHA-256 against the record's `SHA256SUMS.txt` before loading it. Models
that are not in the record (CNN baselines, fine-tuned DinoBloom-B, the other ResNet50 and
Swin-B configurations, mixup + BPaCo + cRT, the augmentation-policy models, the
leave-one-dataset-out models and the three-class detector variant) are retrained with
`train/scripts/` (`train/README.md`).

## Licences

- **Code.** The authors' code is released under the MIT License (`LICENSE`).
- **Third-party code.** `train/randaugment.py` is third-party code redistributed under its
  own licences (Apache-2.0, BSD-2-Clause and MIT); the notices and licence texts are in
  `THIRD_PARTY_NOTICES.md`. The BPaCo loss is not included: its upstream repository
  publishes no licence, so `train/third_party/fetch_bpaco_loss.sh` downloads the pinned
  file and verifies it, and the file stays out of version control.
- **Dependencies.** Ultralytics YOLO, imported by `pipeline/detect_yolo11.py`,
  `data_prep/detector_crop.py`, `figures/make_fig_localization.py` and
  `train/train_detector.py`, is licensed under AGPL-3.0 and installed separately; it is not
  part of this repository. The other dependencies and the pretrained weights downloaded at
  run time are listed with their licences in `THIRD_PARTY_NOTICES.md`.
- **Weights.** The weight files carry their own licences, stated in the weights record:
  the twelve classifier files under CC BY-NC-SA 4.0 and the detector under AGPL-3.0,
  following Ultralytics' licensing of models trained with it.
- **Data.** Each dataset keeps its own licence or terms (see [Data](#data)).

## Research use only

This code and the trained weights are research software. They are not a medical device,
have not been cleared or approved by any regulator, and must not be used for clinical
diagnosis, treatment decisions or patient management.

<!-- named-only:start -->
## Citation

The associated article is currently an unpublished manuscript:

Chang CW, Sun KJ, Tsai CH, Chen CY. *A Two-Stage YOLO-Swin Transformer Pipeline for
Cross-Dataset White Blood Cell Classification*. Unpublished manuscript, 2026.

No article DOI or preprint identifier is assigned here. Replace this citation with the
actual public article or preprint citation when one is available; do not use the
weights DOI as the article DOI.
<!-- named-only:end -->

## Release metadata and final checksums

Validate a local draft without pretending that missing publication fields are complete:

```bash
python -B docs/check_release_metadata.py
```

Before a public upload, fill the actual version, public repository/release URL, weights
record DOI/URL and file-download URL in `docs/RELEASE_METADATA.json`, then run:

```bash
python -B docs/check_release_metadata.py --publication --online
python -B docs/check_release_metadata.py --artifact /path/to/final-code.zip \
  --sha256 ACTUAL_FINAL_ZIP_SHA256
```

The publication check intentionally fails while required values are missing. A paper
that remains an unpublished manuscript does not need a fabricated article DOI. The
per-file manifest checks extracted contents; the ZIP SHA-256 identifies the exact
archive. Rebuild both after any final edits. Publication-metadata validation does not
replace the study's execution and numerical-reproduction checks.
