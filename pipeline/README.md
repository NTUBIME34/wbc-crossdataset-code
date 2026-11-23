# Inference pipeline: YOLO11 detection + Swin-B classification

This folder runs the paper's deployable two-stage pipeline on blood-smear field images:
full field → YOLO11 detection and crop → Swin-B classification. It is inference only.
Training code is in `train/`; the scripts that recompute the paper's numbers are in `eval/`
and `stats/`.

```text
pipeline/
├── detect_yolo11.py   # stage 1: detection and crop
├── classify.py        # stage 2: classification (nine classes + seven evaluation classes)
├── run_pipeline.sh    # both stages in one command
├── requirements.txt   # pip pins of the paper's software stack
└── models/            # put the downloaded weights here (not in the repository)
```

## Settings

The defaults are the settings behind the paper's full-pipeline results (main Methods,
sections "Detection-and-cropping stage", "Evaluation metrics and statistics" and "Implementation and
reproducibility"; Supplementary Section S17).

| Stage | Setting | Default |
|---|---|---|
| Detection | weights | `yolo11_wbc_detection_best.pt` (YOLO11-M) |
| | input size | 512 px |
| | confidence threshold / NMS IoU | 0.30 / 0.70 |
| | box | the highest-confidence box of each image, whatever its class id (the checkpoint keeps the three-class head of TXL-PBC, but only WBC has training instances) |
| | crop | box expanded by 10% of its width and height on each side, clipped to the image, saved as PNG |
| | image without a box | status `no_detection`, no crop, not classified |
| Classification | weights | `swin_base_colour_aug_cwce_seed62.pth` (`swin_base_patch4_window7_224`) |
| | preprocessing | square padding with black to the longer side, resize to 256 px, center crop 224, ImageNet normalization |
| | test-time augmentation | softmax probabilities averaged over the crop and its horizontal flip (`--tta none` turns this off) |
| | output | nine classes: Basophil, Eosinophil, Erythroblast, Lymphoblast, Lymphocyte, Monocyte, Myeloblast, Neutrophil, Promyelocyte |
| | seven evaluation classes | Blast_Group = Myeloblast + Lymphoblast + Erythroblast; Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil and Promyelocyte map to themselves |

Seven-class decision rule:

- **One checkpoint** (default): the nine-class argmax is mapped to the seven classes. This is
  the per-seed rule behind the paper's seed means.
- **Several checkpoints** (`--checkpoints a,b,c`, or `CLASSIFIER_CKPTS` in the runner): each
  checkpoint's nine-class probabilities are summed into the seven classes, the sums are
  averaged over the checkpoints, and the argmax is taken. This is the paper's
  probability-ensemble rule.

`classify.py` uses the same transform, test-time augmentation and nine-to-seven mapping
as `eval/evaluate_classifier.py`, the common evaluator behind the paper's WBCBench2026
numbers; the resolution sweep, the missed-versus-detected comparison of Supplementary S12
and the frozen DinoBloom-B probe keep their own settings (`eval/README.md`). With the same
checkpoint and crops, the two give the same predicted classes and the same seven-class
probabilities.

## Weights

The weights are not part of the repository. Use the separately distributed core
checkpoint bundle and verify it against its `SHA256SUMS.txt` before loading.
The root README and `docs/RELEASE_METADATA.json` distinguish the local release
candidate from an actual public download record.

<!-- named-only:start -->
Weights DOI reserved: `10.5281/zenodo.21465935` (not a published download record).
<!-- named-only:end -->

The pipeline uses these files from the core bundle:

| File | Content | Used as |
|---|---|---|
| `yolo11_wbc_detection_best.pt` | YOLO11-M detector trained on TXL-PBC (WBC boxes only); best epoch (107) of the training run | detector (default) |
| `swin_base_colour_aug_cwce_seed62.pth` | proposed Swin-B classifier, seed 62: class-weighted cross-entropy and morphology-preserving color augmentation; the checkpoint with the best internal-validation macro-F1 | classifier (default) |
| `swin_base_colour_aug_cwce_seed42.pth`, `swin_base_colour_aug_cwce_seed82.pth` | the same model trained with seeds 42 and 82 | other seeds, or a three-seed ensemble |

Put the files in `pipeline/models/`. You can instead keep them elsewhere and set
`MODEL_DIR=/path/to/weights`, or pass each path explicitly (`YOLO_MODEL`,
`CLASSIFIER_CKPT`, `--model`, `--checkpoint`). Without these, the scripts look in
`$MODEL_DIR`, then `pipeline/models/`, then `models/` at the repository root. Weight files
(`*.pt`, `*.pth`) are git-ignored.

Each classifier file is a checkpoint dictionary, not a bare state dict: `state_dict` (the
timm model weights), `classes` (the class order), `args` (the training arguments), `epoch`
and `best_macro_f1` (the validation macro-F1 of that epoch). `classify.py` reads the model
name, input size and class order from it. To load the weights yourself, call
`model.load_state_dict(torch.load(f, weights_only=True)["state_dict"])` on the timm model
(the weights record's README shows the full code). The deposit's other classifier
checkpoints (`swin_base_bpaco_seed*.pth`, `resnet50_bpaco_adamw_seed*.pth`,
`resnet50_bpaco_sgd_seed*.pth`; the BPaCo comparison models) load the same way through
`--checkpoint`. They are not the pipeline's classifier.
The detector is an Ultralytics model; see `THIRD_PARTY_NOTICES.md` for the Ultralytics
license (AGPL-3.0).

## Installation

Python 3.10. The pins in `requirements.txt` are the paper's software stack: PyTorch 2.5.1
(CUDA 12.1 wheels), torchvision 0.20.1, timm 1.0.27, Ultralytics 8.4.82, NumPy 2.2.6 and
scikit-learn 1.7.2.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r pipeline/requirements.txt
```

For the exact environment use `conda env create -f environment/journal_env.yml`. For a
CPU-only machine, or a different CUDA version, change the PyTorch index URL and the
`+cu121` suffixes in `requirements.txt`, for example to
`https://download.pytorch.org/whl/cpu` and `+cpu`.

## Usage

Input: a folder of field images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`,
`.webp`), searched recursively. The crops keep the input's folder layout.

```bash
bash pipeline/run_pipeline.sh /path/to/field_images /path/to/run_dir
```

Without a second argument the run goes to `${OUT_ROOT:-./outputs}/pipeline_<date_time>`,
resolved against the current directory. The runner refuses a run folder inside the
repository, as the `eval/` runners do (detector crops and predictions of a labelled
benchmark must not end up in the tree), so pass a folder outside it or set `OUT_ROOT`.
If the first argument is omitted, `$DATA_ROOT` is used as the input folder.

Three-seed probability ensemble:

```bash
M=pipeline/models
CLASSIFIER_CKPTS=$M/swin_base_colour_aug_cwce_seed42.pth,$M/swin_base_colour_aug_cwce_seed62.pth,$M/swin_base_colour_aug_cwce_seed82.pth \
  bash pipeline/run_pipeline.sh /path/to/field_images /path/to/run_dir
```

Settings are environment variables (`bash pipeline/run_pipeline.sh --help` lists them all):

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON_BIN` | `python3` | Python interpreter |
| `MODEL_DIR` | unset | weights folder searched first (then `pipeline/models/`, then `models/`) |
| `YOLO_MODEL` | `yolo11_wbc_detection_best.pt` in the weights folder | detector |
| `CLASSIFIER_CKPT` | `swin_base_colour_aug_cwce_seed62.pth` in the weights folder | classifier |
| `CLASSIFIER_CKPTS` | empty | comma-separated classifiers for an ensemble; overrides `CLASSIFIER_CKPT` |
| `TTA` | `hflip` | `hflip` or `none` |
| `CONF`, `IOU`, `YOLO_IMGSZ` | `0.30`, `0.70`, `512` | detector threshold, NMS IoU, input size |
| `PADDING_RATIO` | `0.10` | box expansion on each side, as a fraction of box width/height |
| `SELECT` | `best` | `best` = highest-confidence box; `all` = every box (not a paper setting) |
| `FALLBACK` | `skip` | `skip` = no crop for an image without a box; `copy` = classify the whole image (not a paper setting) |
| `SAVE_CROPS` | `1` | `0` = detection only |
| `SAVE_ANNOTATED` | `0` | `1` = also save images with the box drawn |
| `DEVICE` | `auto` | `auto` (GPU if available), `cpu`, `cuda`, `cuda:N` |
| `BATCH_SIZE`, `WORKERS`, `TOPK` | `32`, `4`, `3` | classifier batch size, DataLoader workers, top-k kept |

The stages can also run on their own:

```bash
python pipeline/detect_yolo11.py --input-dir /path/to/field_images --output-dir /path/to/run_dir/01_detection
python pipeline/classify.py --input-dir /path/to/run_dir/01_detection/crops \
  --detections-csv /path/to/run_dir/01_detection/detections.csv \
  --output-csv /path/to/run_dir/02_classification/predictions.csv
```

`classify.py` also works on any folder of single-cell crops. Without `--detections-csv`,
the detection columns stay empty.

## Outputs

```text
run_dir/
├── 01_detection/
│   ├── detections.csv
│   ├── crops/          # <image name>__det00.png, in the input's folder layout
│   └── annotated/      # only with SAVE_ANNOTATED=1
└── 02_classification/
    └── predictions.csv
```

`detections.csv` has one row per input image (one row per box with `SELECT=all`):

| Column | Content |
|---|---|
| `source_path`, `crop_path` | input image; saved crop (empty if there is none) |
| `status` | `detected`, `no_detection`, `read_failed`, `empty_crop`, or `fallback_original` (with `FALLBACK=copy`) |
| `det_index` | box rank within the image (0 = highest confidence) |
| `yolo_class_id`, `yolo_class_name`, `yolo_conf` | class id, class name and confidence of the box |
| `x1`, `y1`, `x2`, `y2` | expanded and clipped box in pixels (the crop) |

`predictions.csv` has one row per crop, plus one row for each image without a crop. Rows
are sorted by source image path.

| Column | Content |
|---|---|
| `crop_path`, `source_path` | crop and its source image |
| `pred_class`, `pred_index`, `pred_prob` | nine-class prediction: argmax of the test-time-augmented probabilities, averaged over checkpoints for an ensemble |
| `topk_classes`, `topk_probs` | top-k nine-class predictions, separated by `\|` |
| `det_status`, `yolo_class_name`, `yolo_conf`, `x1`, `y1`, `x2`, `y2` | copied from `detections.csv` |
| `pred_label` | seven-class label (decision rule above) |
| `pred_confidence` | seven-class probability of `pred_label` |
| `prob_Basophil`, `prob_Blast_Group`, `prob_Eosinophil`, `prob_Lymphocyte`, `prob_Monocyte`, `prob_Neutrophil`, `prob_Promyelocyte` | seven-class probabilities: nine-class probabilities summed, then averaged over checkpoints; full precision, as in `eval/evaluate_classifier.py` |

A row for an image without a crop has an empty `crop_path`, its detection status in
`det_status`, and the same status (for example `no_detection`) in `pred_class` and
`pred_label`. That label never matches a class, and the probability columns are empty.

## Scoring with labels: missed detections count as errors

The pipeline needs no labels. To score it, put the field images in class folders named by
their label (for WBCBench2026, see `data_prep/`). Run the pipeline once, then:

```bash
python stats/coverage_aware_score.py --name my_run \
  --predictions /path/to/run_dir/02_classification/predictions.csv \
  --detections /path/to/run_dir/01_detection/detections.csv \
  --out-dir /path/to/run_dir/03_scores
```

The scores are printed and saved as `coverage_aware_my_run.json` in `--out-dir`. Without
`--out-dir` the file goes to `$OUT_ROOT/stats` (`./outputs/stats` when `OUT_ROOT` is
unset). The script refuses a folder inside the repository, so when you run it from the
repository root, pass `--out-dir` or set `OUT_ROOT` to a folder outside it.

A labelled image on which the detector finds no box counts as an error. It receives a
no-detection output that matches no class, so it adds a false negative to its true class.
Macro-F1 stays the unweighted mean of the seven class F1 scores; the no-detection output is
not averaged as an eighth class. This is how the paper's primary endpoint is computed
(full-pipeline macro-F1 0.879 ± 0.006 on the WBCBench2026 Phase-1 split, n = 8,080, of
which 8,079 fields were detected).

The script also prints the detected-cells-only view. That view drops the missed images and
overstates performance when coverage falls. On Phase-2, the detector missed 348 of 5,196
labelled fields; macro-F1 is 0.769 ± 0.002 with the misses counted and 0.788 ± 0.002
without them. Score with `SELECT=best` and `FALLBACK=skip`, the paper's settings: the
scorer expects one row per field and no classified fallback crops.

For the paper's seed mean ± population SD, classify the same crops once per seed and pass
the three `predictions.csv` files to `--predictions`. The detection stage does not depend
on the classifier seed. The repository's evaluation route (`eval/run_full_pipeline_eval.sh`)
scores the same crops with `eval/evaluate_classifier.py`, which gives the same predictions.

## Notes

- A lower `CONF` or `FALLBACK=copy` changes coverage. Results are then not comparable
  with the paper's.
- CPU and GPU runs can differ at floating-point level, for example in an occasional box
  edge or in the last digits of a probability.
