# Figures

Scripts that draw the paper's code-generated figures. Each script reads inputs you
provide (your own evaluation outputs or your copy of the public datasets) and writes into
`--out-dir` (default `$OUT_ROOT/figures`, else `./outputs/figures`; a folder inside the
repository is refused). All four run on CPU in the `journal_env` environment
(`environment/journal_env.yml`). No images, predictions or rendered figures are
distributed here.

```text
figures/
├── make_fig_localization.py   # Fig. 2 (and the detection panel of Fig. 1a)
├── make_confusion_matrix.py   # Fig. 3 and the error analysis of Results 3.8
├── make_fig_domainshift.py    # Supplementary Fig. S1
└── make_calibration.py        # Supplementary Fig. S2 and the calibration sentence of Results 3.6
```

| Paper figure | Script | Output files | Inputs |
|---|---|---|---|
| Fig. 1, study design and pipeline | none: drawn by hand | – | – |
| Fig. 2, Stage-1 detection and cropping | `make_fig_localization.py` | `fig_localization_grid.png` (also `fig_detection_example.png`) | nine field images and their crops from `pipeline/detect_yolo11.py` (default settings); detector weights |
| Fig. 3, confusion matrices | `make_confusion_matrix.py` | `fig_confusion.pdf`, `fig_confusion.png` | proposed model, seeds 42/62/82: predictions on the internal test split and on WBCBench2026 Phase-1 |
| Supplementary Fig. S1, domain shift across sources | `make_fig_domainshift.py` | `fig_domainshift.pdf`, `fig_domainshift.png` | twelve dataset-original images from PBC, HRLS, AML-LMU and BMCMHF |
| Supplementary Fig. S2, reliability diagram (Section S11) | `make_calibration.py` | `fig_calibration.pdf`, `fig_calibration.png`, `fig_calibration_bins.csv` | proposed model, seeds 42/62/82: predictions on WBCBench2026 Phase-1 |

Fig. 1 was assembled by hand. The field image with the detector's box inside its panel (a)
is `fig_detection_example.png`, which `make_fig_localization.py` writes next to Fig. 2.

## Prediction inputs (Fig. 3 and Supplementary Fig. S2)

Both scripts read the per-seed `predictions.csv` files that `eval/evaluate_classifier.py`
writes with `--use-mappings --tta hflip` for the proposed model (color augmentation +
class-weighted CE). The eval runners put them here:

```text
$OUT_ROOT/augmentation_recipes/colour_aug_proposed/seed_<S>/predictions.csv   external, n = 8,061  (eval/run_augmentation_recipe_eval.sh)
$OUT_ROOT/internal_mapped/colour_aug_proposed/seed_<S>/predictions.csv        internal, n = 12,803 (eval/run_internal_mapped_eval.sh)
```

A directory holding `seed_42/`, `seed_62/` and `seed_82/` is the default layout; for any
other layout pass `--pattern` (`--internal-pattern`, `--external-pattern`), in which
`{seed}` stands for the seed. The scripts use the columns `image_path`, `true_label` and
`prob_<class>` (seven classes). They join the seeds on `image_path`, stop if the seeds
cover different images or disagree on a label, and combine the seven-class probabilities
(mean over seeds for calibration, sum for the confusion matrices; the ensemble label is
the argmax either way). WBCBench2026 predictions carry image IDs: keep them on your own
machine.

```bash
# Supplementary Fig. S2 and Results 3.6 (PRED_DIR may replace --pred-dir)
python figures/make_calibration.py \
    --pred-dir "$OUT_ROOT/augmentation_recipes/colour_aug_proposed"

# Fig. 3 and Results 3.8
python figures/make_confusion_matrix.py \
    --internal-dir "$OUT_ROOT/internal_mapped/colour_aug_proposed" \
    --external-dir "$OUT_ROOT/augmentation_recipes/colour_aug_proposed"
```

What they print, with the paper's values:

- `make_calibration.py`: n = 8,061; expected calibration error 0.017 over 15 equal-width
  confidence bins; multiclass Brier score 0.074; mean confidence 0.970 versus accuracy
  0.954; and the per-bin table (count, share, mean confidence, accuracy, contribution to
  the ECE; also saved as `fig_calibration_bins.csv`). The top bin holds 90% of the
  predictions, with mean confidence 0.998 versus accuracy 0.984. `stats/calibration.py`
  computes the same numbers, plus per-class one-vs-rest calibration, without drawing.
- `make_confusion_matrix.py`: n = 12,803 internal and 8,061 external; per-class recall
  (internal 0.90–0.98; external Neutrophil 0.99, Eosinophil 0.97, Lymphocyte 0.94,
  Blast group 0.84 and Promyelocyte 0.79, against 0.95 and 0.96 internally); the largest
  external confusions (11% of blasts called Lymphocyte, 10% of promyelocytes called
  Basophil and 7% Monocyte); both count matrices; and the external error breakdown:
  368 errors (4.6%), Blast group↔Lymphocyte 65 and 64 (35% of errors),
  Lymphocyte↔Monocyte 63 and 27 (24%), 3% among the mature granulocytes, and
  Promyelocyte false positives versus false negatives 19 versus 9 at 0.5% prevalence.

## Image inputs (Fig. 2 and Supplementary Fig. S1)

### `make_fig_domainshift.py`

```bash
python figures/make_fig_domainshift.py --sources-root /path/to/sources
python figures/make_fig_domainshift.py --list      # the twelve pinned paths
```

The figure shows dataset-original files (no detector cropping) for three cell types
(neutrophil, lymphocyte, monocyte) in four training sources, each center-cropped to a
square at a common display size. The pinned paths are relative to `--sources-root`
(default `$SOURCES_ROOT`), the folder with the datasets as downloaded that
`data_prep/crop_training_sources.sh` also reads; it is not the prepared-data
`DATA_ROOT`. The pins use that script's default folder names (`dataset_DIB/`,
`HRLS_dataset/`, `PKG_AML_Cytomorphology_LMU/`, `BM_cytomorphology_data/`), the
authors' PNG copies of PBC and AML-LMU and the authors' renamed BMCMHF class folders
(`Band_neutrophil/`, `Lymphocyte/`, `Monocyte/`). A pinned file that is missing is
looked up in the release's BMCMHF class-code folder (`NGB/`, `LYT/`, `MON/`) and under
the same name with another image extension, which finds the BMCMHF, JPEG (PBC) and
TIFF (AML-LMU) files of the public releases; `--list` shows both BMCMHF paths. Replace
any panel with `--image ROW/COLUMN=PATH`, e.g.
`--image Neutrophil/PBC=/path/to/BNE_495973.jpg`. A missing file is drawn as "missing"
and the script exits with status 1.

Licenses of the reproduced images: PBC CC BY 4.0, HRLS CC0 1.0, AML-LMU CC BY 3.0,
BMCMHF CC BY 4.0; cite each source when you reuse the figure. Raabin-WBC, also a
training source, is not shown: its images are available under the Raabin Health
Database terms (free access with citation) rather than a Creative Commons license, and,
as the paper's Data availability statement says, neither Raabin-WBC nor WBCBench2026
images are reproduced.

### `make_fig_localization.py`

The exact nine inputs are listed in [`fig2_source_mapping.csv`](fig2_source_mapping.csv).
The mapping includes the original source IDs, the renamed field names expected by the
script, file hashes of the retained PNG copies, and decoded-RGB hashes. It contains no
images or WBCBench image identifiers.

| Panel | Dataset | Source-relative path in the retained source copy | Rename for `--field-dir` |
|---|---|---|---|
| Basophil | AML-LMU | `PKG_AML_Cytomorphology_LMU/BAS/BAS_0009.png` | `Basophil_000009.png` |
| Eosinophil | PBC | `dataset_DIB/eosinophil/EO_19043.png` | `Eosinophil_000707.png` |
| Erythroblast | AML-LMU | `PKG_AML_Cytomorphology_LMU/EBO/EBO_0014.png` | `Erythroblast_000014.png` |
| Lymphoblast | HRLS | `HRLS_dataset/Lymphoblast/   9.png` | `Lymphoblast_000010.png` |
| Lymphocyte | PBC | `dataset_DIB/lymphocyte/LY_109025.png` | `Lymphocyte_000009.png` |
| Monocyte | AML-LMU | `PKG_AML_Cytomorphology_LMU/MON/MON_0008.png` | `Monocyte_000008.png` |
| Myeloblast | AML-LMU | `PKG_AML_Cytomorphology_LMU/MYO/MYO_0791.png` | `Myeloblast_000791.png` |
| Neutrophil | PBC | `dataset_DIB/neutrophil/BNE_101082.png` | `Neutrophil_000003.png` |
| Promyelocyte | AML-LMU | `PKG_AML_Cytomorphology_LMU/PMO/PMO_0038.png` | `Promyelocyte_000001.png` |

Obtain the datasets separately: PBC ([Mendeley Data](https://doi.org/10.17632/snkd93bnjr.1),
CC BY 4.0), AML-LMU ([TCIA](https://doi.org/10.7937/tcia.2019.36f5o9ld), CC BY 3.0),
and HRLS ([figshare](https://doi.org/10.6084/m9.figshare.22680517), CC0 1.0).
Cite the source datasets when reusing the figure. These paths describe the authors'
retained source copies, not a directory layout promised by every download mirror.
PBC and AML-LMU are distributed as JPEG and TIFF, respectively; identify the matching
source stem, decode it, and save a PNG under the listed field name without resizing or
other image processing. Do not merely rename the extension. The HRLS source filename
has **three leading spaces** before `9.png`; the CSV preserves them.

The `panel_png_sha256` and `retained_source_sha256` columns refer to encoded files in
the authors' retained copies, not to unseen public JPEG/TIFF archives. Eight pairs are
byte-identical; the Promyelocyte pair has different PNG encoding but identical RGB
pixels. All nine pairs were rechecked against the retained source files on 2026-11-05.
For another lossless encoding, check `width`, `height`, and `rgb_sha256`: the latter is
SHA-256 of `PIL.Image.open(path).convert("RGB").tobytes()` (row-major, interleaved RGB,
without a dimensions header). A byte-hash difference alone does not establish a pixel
difference. If the decoded RGB check fails, record the differing source version rather
than silently treating it as the original figure input.

```bash
python pipeline/detect_yolo11.py --input-dir /path/to/fields \
    --output-dir /path/to/fig2_detection \
    --model models/yolo11_wbc_detection_best.pt --device cpu
python figures/make_fig_localization.py \
    --field-dir /path/to/fields --crop-dir /path/to/fig2_detection/crops \
    --weights models/yolo11_wbc_detection_best.pt --device cpu
```

- `--field-dir`: one uncropped field image per training class, named `<Class>_<id>.png`
  (Basophil, Eosinophil, Erythroblast, Lymphoblast, Lymphocyte, Monocyte, Myeloblast,
  Neutrophil, Promyelocyte). The published figure uses fields from PBC, AML-LMU and HRLS.
- `--crop-dir`: the matching crops, under the same file name or as `<name>__det00.png`,
  the name `pipeline/detect_yolo11.py` gives them in the `crops/` folder of its output.
  The published figure uses the crops that `detect_yolo11.py` writes with its default
  settings: the deployed crop, the highest-confidence box expanded by 10% of its width
  and height on each side and clipped to the image (main Methods 2.4).
- `--weights`: the YOLO11 detector `yolo11_wbc_detection_best.pt`; without the flag it is
  looked up in `$MODEL_DIR`, `pipeline/models/` and `models/`. `--device` is passed to
  Ultralytics (default: its own choice).

The detector runs with the pipeline's settings (512 px, confidence 0.30, NMS IoU 0.70).
The box drawn is the detection nearest the image center; the pipeline crops the
highest-confidence box, and on these single-cell fields the detector returns one box
per image, so the two coincide. The top row draws the box itself; the crop below it
adds the 10% margin. The bottom row shows each crop resized to 256 px and
center-cropped to 224; the square padding the classifier applies first is not shown,
as the figure caption states. The script also prints the detector confidence of each
of the nine cells.

**Crops in the published figure.** The bottom row of the published Fig. 2 shows, for
all nine classes, the crop that `pipeline/detect_yolo11.py` writes with its default
settings. In the Monocyte field the box reaches the top-left corner of the image, so
the margin is clipped on the left and top. Run on CPU with the released detector,
`detect_yolo11.py` reproduces the nine crops pixel for pixel.

## Reproduction check

Run on CPU with the authors' evaluation outputs and example images,
`make_calibration.py`, `make_confusion_matrix.py` and `make_fig_domainshift.py`
rebuild the paper's figure files exactly: the PNG files byte for byte, the PDF files
except for their creation date. With the authors' field images and the default crops of
`pipeline/detect_yolo11.py`, `make_fig_localization.py` reproduces the published Fig. 2
except for detector numerics in the top row: run on CPU, one of the nine confidence tags
(Lymphoblast) rounds 0.01 higher than in the published figure and the edges of one box
(Myeloblast) shift by less than a pixel (the script leaves the device to Ultralytics,
which uses a GPU when one is present). The bottom row is identical pixel for pixel.
