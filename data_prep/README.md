# data_prep: building the datasets

This folder rebuilds every dataset folder the paper uses from the public releases:
the five per-source single-cell pools and their leave-one-dataset-out (LODO) folds,
the merged internal corpus with its 80/10/10 split, and the WBCBench2026 evaluation
folders. No image, label file or per-image list of any dataset is included in the
repository.

All steps run in `journal_env` (`environment/journal_env.yml`) except
`cellpose_crop.py`, which runs in the Cellpose environment
(`environment/cellpose_env.yml`). The detector steps need the released detector
`yolo11_wbc_detection_best.pt` (looked up in `$MODEL_DIR`, `pipeline/models/`, then
`models/`). Inputs come from flags or `DATA_ROOT`; every script refuses to write inside
the repository and prints its usage with `--help`.

Line numbers below refer to the manuscript (`main`) and supplement (`supp`) sources
of 2026-11-19.

## What gets built

The folder names are the authors' working names. `train/scripts/common.sh`, the `eval/`
runners and `analysis/evidence/MANIFEST.csv` use them as defaults under `$DATA_ROOT`.

| Paper item | Folder under `$DATA_ROOT` | Made by | Read by |
|---|---|---|---|
| Detector crops of the five sources (Supp Table S1) | `PKG_AML_Cytomorphology_LMU_single_wbc`, `BM_cytomorphology_data_single_wbc_07`, `dataset_DIB_single_wbc`, `Raabin_WBC_single_wbc`, `HRLS_dataset_single_wbc` (each with `crop_log.csv`) | `crop_training_sources.sh` (uses `detector_crop.py`) | `harmonize_labels.py`; the AML-LMU typical lymphocytes also feed steps 4b and 4d |
| Harmonized pools (Table 1 "Pool size"; Supp Table S15) | `PKG_AML_Cytomorphology_LMU_single_wbc_for_train_9class`, `BM_cytomorphology_data_single_wbc_07_for_train_9class`, `dataset_DIB_single_wbc_copy`, `Raabin_WBC_single_wbc_copy`, `HRLS_dataset_single_wbc_copy` | `harmonize_labels.py` | `split_dataset.py`; `eval/evaluate_lodo.py` (held-out PBC, Raabin-WBC, HRLS, BMCMHF) |
| Per-source 80/10/10 splits (Supp Note 1) | `<pool>_split` | `split_dataset.py` | the LODO builders |
| Merged internal corpus (127,938 cells; Supp Table S2) | `Mixed_data_mix_mono` (11 class folders) and its split `Mixed_data_mix_mono_9_split_copy` | `harmonize_labels.py --merged-dir`, `split_dataset.py --drop-after-split` | `train/scripts/` (`CORPUS_ROOT`), `eval/run_internal_mapped_eval.sh` |
| LODO training folds (Table 4) | `LODO_ALL_minus_LMU_test_LMU_split`; `LODO_corrected_ALL_minus_{DIB,Raabin,HRLS,BM}_test_<X>_split` | `create_lodo_all_datasets.py`; `build_corrected_lmu_split.py` then `build_corrected_lodo_splits.py` | `train/scripts/` (`LODO_ROOT_<FOLD>`) |
| AML-LMU held-out set (18,128 cells) | `LMU_LODO_corrected_test` | `build_lmu_lodo_test.py` | `eval/evaluate_lodo.py` |
| WBCBench2026 Phase-1 full fields (8,288) | `wbc-bench-2026_phase1_nocrop_formatted` (`WBCB_P1_FIELDS`) | `prepare_wbcbench.py --split phase1` | `eval/` runners, pipeline scoring |
| WBCBench2026 Phase-2 evaluation fields (5,350) | `wbc-bench-2026_phase2_eval_nocrop_formatted` (`WBCB_P2_FIELDS`) | `prepare_wbcbench.py --split phase2_eval` | `eval/run_full_pipeline_eval.sh` |
| Phase-1 pre-cropped set (8,269 crops; route A) | `wbc-bench-2026_phase1_cropped_formatted` (`WBCB_P1_CROPS`) | `detector_crop.py --conf 0.70` | `eval/` runners |
| Cellpose crops, routes C and D (Table 2b; Supp Table S6) | `wbc-bench-2026_phase1_nocrop_formatted_cellpose_keepbg_224`, `wbc-bench-2026_phase1_nocrop_formatted_cellpose_224` (`ROUTE_C_DIR`, `ROUTE_D_DIR`) | `cellpose_crop.py` | `eval/run_routes_ablation.sh` |
| TXL-PBC (detector training) | `TXL-PBC/data.yaml`, as distributed | no step | `train/scripts/run_detector.sh` (`TXLPBC_DATA_YAML`) |

LODO fold codes: `LMU` = AML-LMU, `DIB` = PBC, `Raabin` = Raabin-WBC, `HRLS` = HRLS,
`BM` = BMCMHF.

## Downloads and licences

As listed in the paper's Data availability statement (main 1023-1040):

- AML-Cytomorphology_LMU (TCIA, doi:10.7937/tcia.2019.36f5o9ld; CC BY 3.0)
- Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer, short name BMCMHF (TCIA,
  doi:10.7937/TCIA.AXH3-T579; CC BY 4.0)
- PBC (Mendeley Data, doi:10.17632/snkd93bnjr.1; CC BY 4.0)
- Raabin-WBC (https://raabindata.com; free access with citation under the Raabin
  Health Database terms)
- HRLS (figshare, doi:10.6084/m9.figshare.22680517; CC0 1.0)
- TXL-PBC (figshare, doi:10.6084/m9.figshare.27073186; CC BY 4.0)
- WBCBench2026 (Kaggle competition wbc-bench-2026, listed as CC BY-NC-SA 4.0; the paper
  used the competition files downloaded in November 2025, whose rules restrict
  redistribution; a later Hugging Face release corrects 257 labels, removes 10 images
  and adds a patient hash). Download it yourself; nothing from it is in this repository.

Put the five classifier sources under one `SOURCES_ROOT`. For each source,
`crop_training_sources.sh` needs the folder that directly holds its class
sub-folders; the defaults are the authors' folder names (see the script header) and
`LMU_DIR`, `BMCMHF_DIR`, `PBC_DIR`, `RAABIN_DIR`, `HRLS_DIR` point it elsewhere:

- AML-LMU: the class-code folders of the release (`BAS`, `EBO`, ..., as in its
  `abbreviations.txt`). Crops are written as PNG.
- BMCMHF: the class-code folders; images inside numbered range sub-folders are found.
- PBC: the eight lower-case class folders (`ig` and `platelet` are not cropped).
- Raabin-WBC: the training split (`Train/`), the only Raabin-WBC split used.
- HRLS: its nine class folders.

Long folder names as in the authors' copies (`Lymphocyte_typical`,
`Segmented_neutrophil`, ...) work as well. `harmonize_labels.py` stops with a list of
any class folder it cannot map.

## Steps

```bash
export DATA_ROOT=/path/to/prepared_data     # outside the repository
export MODEL_DIR=/path/to/weights            # holds yolo11_wbc_detection_best.pt

# 1. detector crops of the five sources
bash data_prep/crop_training_sources.sh /path/to/sources "$DATA_ROOT"

# 2. label harmonization (Supp Table S3): five pools + the merged corpus
python data_prep/harmonize_labels.py --data-root "$DATA_ROOT" \
    --merged-dir "$DATA_ROOT/Mixed_data_mix_mono"

# 3. per-class 80/10/10 splits, seed 42
for pool in PKG_AML_Cytomorphology_LMU_single_wbc_for_train_9class \
            BM_cytomorphology_data_single_wbc_07_for_train_9class \
            dataset_DIB_single_wbc_copy Raabin_WBC_single_wbc_copy HRLS_dataset_single_wbc_copy; do
  python data_prep/split_dataset.py --src "$DATA_ROOT/$pool" --dst "$DATA_ROOT/${pool}_split"
done
python data_prep/split_dataset.py --src "$DATA_ROOT/Mixed_data_mix_mono" \
    --dst "$DATA_ROOT/Mixed_data_mix_mono_9_split_copy" --drop-after-split Metamyelocyte Myelocyte

# 4. LODO folds, the corrected AML-LMU split and the AML-LMU held-out set
python data_prep/create_lodo_all_datasets.py --base-dir "$DATA_ROOT"     # 4a
python data_prep/build_corrected_lmu_split.py --data-root "$DATA_ROOT"   # 4b
python data_prep/build_corrected_lodo_splits.py --base-dir "$DATA_ROOT"  # 4c
python data_prep/build_lmu_lodo_test.py --data-root "$DATA_ROOT"         # 4d

# 5. WBCBench2026 class folders and the pre-cropped set
python data_prep/prepare_wbcbench.py --src /path/to/wbc-bench-2026 --split phase1
python data_prep/prepare_wbcbench.py --src /path/to/wbc-bench-2026 --split phase2_eval
python data_prep/detector_crop.py --input-dir "$DATA_ROOT/wbc-bench-2026_phase1_nocrop_formatted" \
    --output-dir "$DATA_ROOT/wbc-bench-2026_phase1_cropped_formatted" --conf 0.70

# 6. optional: Cellpose crops for routes C and D (Cellpose environment)
python data_prep/cellpose_crop.py --layout class --keep-background   # route C
python data_prep/cellpose_crop.py --layout class                     # route D
```

`harmonize_labels.py` and `split_dataset.py` copy files by default; `--mode link`
uses hard links instead. The LODO folds are symlink trees and the corrected AML-LMU
split and held-out set are hard links, so keep `DATA_ROOT` on one file system and do
not move it after step 4.

### 1. Detector crop (`crop_training_sources.sh`, `detector_crop.py`)

Every source image goes through the Stage-1 detector at 512 px (NMS IoU 0.70); the
single highest-confidence box is cropped without margin (corner coordinates
truncated to integers), and an image whose best box scores below the threshold is
excluded rather than cropped at a lower confidence (main 288; supp 54-57). Not
passed to the detector: PBC `ig` and `platelet`, AML-LMU smudge cells (`KSC`), BMCMHF
artefact, not-identifiable, other-cell and smudge-cell images (`ART`, `NIF`, `OTH`,
`KSC`) (Supp Table S1 caption). Thresholds: 0.30 for AML-LMU, PBC, Raabin-WBC and
HRLS and 0.70 for BMCMHF, as run; see "Detector thresholds as run" below.
`crop_log.csv` records each image's status, best-box confidence and box.

### 2. Label harmonization (`harmonize_labels.py`)

Native classes are mapped to the nine internal labels by the crosswalk of Supp Table
S3 (supp 225-273); unmapped classes are dropped. One pool per source, original file
names. The AML-LMU pool is written without its typical lymphocytes (`LYT`), as in the
authors' first nine-class pool; steps 4b and 4d add them back, which reproduces the
corrected split and held-out set exactly (`--lmu-typical include` writes the full pool
in one step, but then the AML-LMU split differs from the paper's).

`--merged-dir` writes the merged internal corpus as it was built: crops whose logged
confidence is at least 0.70, the S3 crosswalk with typical lymphocytes, the native
classes the corpus kept outside the crosswalk (supp 117-121: AML-LMU and HRLS band
neutrophils under Neutrophil, BMCMHF immature lymphocytes under Lymphocyte, AML-LMU
monoblasts under Myeloblast) and Metamyelocyte and Myelocyte folders (AML-LMU and
BMCMHF), which take part in the split and are then removed. The script compares the
seven-class totals with Supp Table S2. `--merged-crosswalk-only` builds a cleaner
corpus without those extras (not the paper's).

### 3. Splits (`split_dataset.py`)

One `random.seed(42)`, class folders in sorted order, each class's sorted file list
shuffled, `int(n*0.8)` training and `int(n*0.1)` validation images, the rest test
(main 291-294; supp 659-660). The merged corpus was split with its Metamyelocyte and
Myelocyte folders present and those two classes deleted afterwards;
`--drop-after-split` does the same in one step and leaves 102,345 training, 12,790
validation and 12,803 test images (main 392-393, which prints the training and test counts).

### 4. LODO folds

- 4a `create_lodo_all_datasets.py` builds `LODO_ALL_minus_<X>_test_<X>_split` for all
  five sources: the held-out source's train+val+test as `test/`, the other four
  sources' training splits as `train/` and validation splits as `val/` (supp 659-662),
  as symlinks named `<code>__<split>__<file>`. Only the AML-LMU fold is used.
- 4b `build_corrected_lmu_split.py` adds the AML-LMU typical lymphocytes (3,936 crops;
  the first split held only the 11 atypical ones) to the AML-LMU split with their own
  seed-42 80/10/10 draw: `PKG_AML_Cytomorphology_LMU_single_wbc_for_train_9class_split_corrected`
  (supp 667-669).
- 4c `build_corrected_lodo_splits.py` rebuilds the four folds whose training pool
  contains AML-LMU from the corrected split:
  `LODO_corrected_ALL_minus_{BM,HRLS,DIB,Raabin}_test_<X>_split`.
- 4d `build_lmu_lodo_test.py` builds `LMU_LODO_corrected_test`, the complete AML-LMU
  pool (18,128 cells).

A fold is scored on every pooled cell of its held-out source (supp 661-662 and 943-944):
`eval/evaluate_lodo.py` reads `LMU_LODO_corrected_test` for AML-LMU and the pool
folders for the other four sources, not the folds' `test/` folders.

### 5. WBCBench2026 (`prepare_wbcbench.py`, `detector_crop.py`)

`prepare_wbcbench.py` turns the label CSV and image folder of one released split into
one folder per native class: `--split phase1` (8,288 fields) and `--split phase2_eval`
(5,350 fields) (main 262-263). The pre-cropped set is the detector's best box per
Phase-1 field at 0.70 without margin: 8,269 crops; the 19 dropped fields are 13
lymphocytes and 6 segmented neutrophils (supp 98-111). Plasma cells, metamyelocytes
and myelocytes stay in their folders; the evaluators ignore them (208 Phase-1 cells,
154 Phase-2 fields). The full-pipeline route runs the deployable detector on the
fields themselves (`pipeline/`, `eval/run_full_pipeline_eval.sh`).

### 6. Cellpose crops (`cellpose_crop.py`, optional)

Cellpose 4.1.1 with the `cpsam` model; the mask nearest the field centre gives a
224 x 224 window centred on its centroid, with the background kept (route C,
`--keep-background`) or whitened outside the mask (route D) (Supp Table S6). The
defaults read `WBCB_P1_FIELDS` and write the two folders `eval/run_routes_ablation.sh`
expects. Route E needs no extra data step: it re-crops the route-C crops with the
detector inside `eval/run_routes_ablation.sh`.

The paper's crops were made on a GPU with Cellpose's default bfloat16 weights. Without
a GPU, `--precision auto` (the default) loads float32 weights, because bfloat16
inference is very slow on processors without native bfloat16 support. Even in float32 a
CPU run takes roughly 20 s per field on four cores (about two days per route for the
8,288 Phase-1 fields), so use a GPU for the full set. Float32 masks can differ slightly
from bfloat16 ones, so an occasional crop can shift; `--precision bf16` forces the
paper's setting.

## Expected counts

The scripts print their counts next to these values.

**Table 1, pool sizes (main 320-324)**: AML-LMU 18,128; BMCMHF 73,096; PBC 11,845;
Raabin-WBC 10,154; HRLS 15,405; sum 128,628 (main 307). WBCBench2026 Phase-1 8,288
images (main 325); TXL-PBC 1,260 images (main 326).

**Supp Table S1, detector filtering (supp 79-83)**

| Source | Passed to the detector | Excluded | Retained |
|---|---:|---:|---:|
| AML-LMU | 18,350 | 12 | 18,338 |
| PBC | 11,849 | 4 | 11,845 |
| Raabin-WBC | 10,175 | 21 | 10,154 |
| HRLS | 16,027 | 16 | 16,011 |
| BMCMHF | 147,869 | 55,862 | 92,007 |

Of the 92,007 BMCMHF crops, 79,880 fall in nine categories after harmonization and
73,096 in the seven internal labels (supp 58-62).

**Supp Table S15, pools by class (supp 980-991)**

| Class | AML-LMU | PBC | Raabin-WBC | HRLS | BMCMHF |
|---|---:|---:|---:|---:|---:|
| Neutrophil | 8,483 | 3,329 | 6,217 | 3,198 | 26,536 |
| Lymphocyte | 3,947 | 1,214 | 2,420 | 3,037 | 20,284 |
| Monocyte | 1,779 | 1,420 | 561 | 2,039 | 3,049 |
| Eosinophil | 424 | 3,117 | 744 | 1,017 | 3,484 |
| Basophil | 79 | 1,218 | 212 | 1,023 | 355 |
| Erythroblast | 78 | 1,547 | 0 | 0 | 13,470 |
| Promyelocyte | 70 | 0 | 0 | 0 | 5,918 |
| Myeloblast | 3,268 | 0 | 0 | 2,534 | 0 |
| Lymphoblast | 0 | 0 | 0 | 2,557 | 0 |
| Total | 18,128 | 11,845 | 10,154 | 15,405 | 73,096 |

**Supp Table S2, merged internal corpus (supp 199-207)**: Neutrophil 47,551;
Lymphocyte 30,811; Blast group (Erythroblast + Lymphoblast + Myeloblast) 23,422;
Eosinophil 8,766; Monocyte 8,526; Promyelocyte 5,988; Basophil 2,874; total 127,938
(main 291). Split: 102,345 training and 12,803 test images (main 392-393).

**WBCBench2026 (supp 98-111)**: Phase-1 8,288 fields; pre-cropped set 8,269 crops,
8,061 evaluated; full pipeline 8,080 labelled fields. Phase-2 5,350 fields, 5,196 with a
label in the seven classes.

## Detector thresholds as run

Supplementary S1 states the thresholds below (supp 54-57, 116-117 and the Table S1
caption, supp 67-68); the main text gives 0.70 for the merged corpus and the pre-cropped set and calls
the pool thresholds source-specific (main 268, 288). The authors' crop folders show
two crop runs:

- The per-source crops behind the pools, Table S1 and Table S15 were made in November
  2025: BMCMHF at 0.70, but AML-LMU, PBC, Raabin-WBC and HRLS at 0.30. Re-running the
  detector on the authors' copies, every readable image that run excluded from those
  four sources scores below 0.30 (highest 0.298), while the images it kept go down to
  0.300; for BMCMHF the excluded images score at most 0.698 and the kept ones at least
  0.700. At 0.30 the four peripheral-blood sources lose under 0.25% of their images
  (Supp Table S1, supp 79-82).
- The merged internal corpus was cropped again, for every source at 0.70 (file dates
  of March 2026; checked on a sample of the images it includes and excludes). It
  therefore lacks the pool images that score between 0.30 and 0.70: 599 AML-LMU, 66
  PBC, 146 HRLS and 155 Raabin-WBC pool images (0.6% to 3.3% of each pool; none for
  BMCMHF), 966 in total, which are the 966 pool images Supp S1 attributes to this
  threshold (supp 116-117). The AML-LMU, PBC and HRLS counts come from matching file
  contents against the corpus (sampled confidences of the missing images all lie
  between 0.30 and 0.70), the Raabin-WBC count from re-running the detector on all
  10,154 of its kept images.
- The WBCBench2026 pre-cropped set used 0.70: its 19 dropped fields all score below
  0.70.

`crop_training_sources.sh` therefore crops AML-LMU, PBC, Raabin-WBC and HRLS at 0.30
and BMCMHF at 0.70, which reproduces the pools, and `harmonize_labels.py --merged-dir`
keeps only crops logged at 0.70 or above, which reproduces the corpus composition.
`CONF=0.70 bash data_prep/crop_training_sources.sh ...` applies 0.70 to every source;
the four peripheral-blood pools then shrink by 0.6% to 3.3% and no longer match
Table S15.

## What was checked against the authors' folders

Run read-only on the authors' prepared folders (file operations patched to record
names only):

- `harmonize_labels.py` reproduces all five pool folders file for file.
- `split_dataset.py` reproduces all five per-source splits, and with
  `--drop-after-split Metamyelocyte Myelocyte` the merged corpus split
  (`Mixed_data_mix_mono_9_split_copy`), file for file.
- `build_corrected_lmu_split.py`, `build_corrected_lodo_splits.py` (four folds) and
  `build_lmu_lodo_test.py` reproduce the corrected AML-LMU split, the four corrected
  folds and `LMU_LODO_corrected_test` file for file; `create_lodo_all_datasets.py`
  reproduces the PBC, Raabin-WBC, HRLS and BMCMHF folds and the `train/` and `val/`
  folders of the AML-LMU fold (its `test/` folder, which no evaluation reads, has since
  lost the 11 atypical-lymphocyte links in the authors' copy).
- `detector_crop.py` (Ultralytics 8.4.82, CPU) kept every sampled image the authors
  kept, at the thresholds above, and reproduced the authors' crop box exactly for
  about 95% of sampled images of the five sources and of the Phase-1 fields; the rest
  differ by at most one pixel in a box coordinate. The 19 Phase-1 fields the authors
  dropped are also dropped.
- `harmonize_labels.py --merged-dir` follows the corpus composition worked out from the
  authors' folders: pool images at 0.70 or above, plus the merge extras (103 AML-LMU
  and 99 HRLS band neutrophils, 24 AML-LMU monoblasts and 50 BMCMHF immature
  lymphocytes, the 276 of supp 117-121, matched by file content where possible) and the
  Metamyelocyte and Myelocyte folders. Per source and class, this composition gives the
  authors' corpus counts exactly. The script itself was run end to end on a small
  sample of the public datasets, not on the full data.
- The whole chain of steps 1 to 4 ran on that small sample (CPU) and wrote every
  training-side folder of the table above; `prepare_wbcbench.py` ran on a synthetic download with the competition
  file layout; `cellpose_crop.py` ran in the Cellpose environment on one image (route
  C) and its crop logic was tested for both routes with a stand-in segmentation model.

## Not recorded, and known differences

- **Crop script and version.** The authors' crop runs used an earlier version of the
  crop script that applied the same rule; it was edited afterwards, and the Ultralytics
  version of those runs was not recorded (supp 370-371). `detector_crop.py` is a
  re-implementation of the rule, checked as described above.
- **Merged corpus file names.** The corpus was built by merging the uncropped sources
  into one folder with a renaming script that is not shipped (files were renamed
  `<class>_<n>.png`, which discards their source names), cropping that folder at 0.70
  and splitting it. The merge order was not recorded, so the original file names
  cannot be regenerated. The rebuilt corpus holds the same cells with the same class
  counts, but `split_dataset.py` shuffles sorted file names, so its seed-42 split puts
  different cells in training, validation and test than the authors' split (the
  per-class split sizes are identical). Internal-test numbers from a rebuilt corpus can therefore
  differ slightly from Table 3's internal column. The authors' corpus stored every crop
  as PNG cut from the decoded source; the rebuilt one keeps the BMCMHF and Raabin-WBC
  crops as JPEG (one more JPEG encoding).
- **AML-LMU typical lymphocytes.** The authors' first nine-class AML-LMU pool held only
  the 11 atypical lymphocytes; the 3,936 typical ones were restored by the corrected
  split (steps 4b to 4d; supp 667-669). The internal corpus always contained them.
- **HRLS.** 12 of the 16 HRLS images excluded in Table S1 are files that cannot be
  decoded in the authors' download; `detector_crop.py` logs them as `unreadable`.
- **AML-LMU file format.** The authors' AML-LMU copy is PNG, while the release's
  checksum list names TIFF files; `--png` writes PNG crops with the same stems either
  way.
- **WBCBench2026 version.** Counts above are for the November-2025 competition files.
  The later Hugging Face release (257 labels corrected, 10 images removed) gives
  different counts.
- **Box precision.** The one-pixel differences above describe that CPU sample, not a
  guaranteed bound. An independent-environment inference check on the complete
  full-pipeline inputs found different rounded boxes for 361 Phase-1 and 229 Phase-2
  crops (maximum coordinate differences 3 and 5 pixels, respectively), with unchanged
  detection coverage. Replaying the saved original boxes on the source images rebuilt
  every original full-pipeline crop exactly in decoded RGB. Using those historical
  inputs with the released classifiers reproduced every historical class prediction
  for all three seeds in both splits, including the coverage-aware macro-F1 values
  0.879 and 0.769. Thus those score differences were localized to regenerated detector
  inputs; their specific software or hardware cause was not established. This check
  does not resolve the separate historical no-margin pre-crop provenance gap.
