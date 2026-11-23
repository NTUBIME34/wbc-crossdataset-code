# eval/

Evaluation code behind every classifier number the paper reports on WBCBench2026 and in
the leave-one-dataset-out (LODO) analysis, with one runner per paper analysis. Everything
here is inference only: checkpoints come from the weights deposit or from `train/scripts/`.
The statistics (seed means, ensembles, bootstrap intervals, paired tests, coverage-aware
scores) are computed from the per-image outputs by `stats/`; figures by `figures/`.

## The common evaluator

`evaluate_classifier.py` (formerly `test.py`) is the evaluator behind main-text Table 3
(external columns), Table 2, Supplementary Tables S6 to S11 and the unified
re-evaluation of Supplementary Section S17. Every external number is scored with it
unless listed under "Analyses with their own settings" below.

| Step | Setting |
|---|---|
| Input | a class-folder test set (one subfolder per class; the folder name is the label) |
| Preprocessing | square padding with black to max(w, h), resize to `int(img_size * 256 / 224)` = 256 px, centre crop 224, ImageNet mean/std |
| Test-time augmentation | `--tta hflip` (default): softmax probabilities of the crop and its horizontal flip are averaged; `--tta none`: crop only |
| Decision rule | nine-class argmax first, then mapped to the seven evaluation classes (`--use-mappings`) |
| Merge rules | Neutrophil = Band_neutrophil, Segmented_neutrophil, Neutrophil (and Neutrophil_sum); Lymphocyte = Lymphocyte, Variant_lymphocyte, Prolymphocyte; Blast_Group = Blast, Lymphoblast, Myeloblast, Erythroblast; Basophil, Eosinophil, Monocyte and Promyelocyte map to themselves |
| Ignore rules | ground-truth images of Plasma_cell, Metamyelocyte and Myelocyte are skipped (no counterpart in the seven classes); on the pre-cropped Phase-1 set this leaves n = 8,061 of 8,269 crops |

`--merge` and `--ignore-class` replace the default rules (the LODO wrapper uses this).
Probability ensembles (nine-class probabilities summed into the seven classes, averaged
over seeds, then argmax) and majority votes (three-way ties take the seed-42 label) are
formed later by `stats/` from the `prob_<class>` and `pred_label` columns.

Outputs with `--output-dir` (column names are those of the evaluator as run; `stats/`
and `figures/` read them):

- `predictions.csv`: `image_path, source_dataset, raw_true, raw_pred, true_label,
  pred_label, true_confidence, pred_confidence, correct, prob_<class>...`, one row per
  evaluated image, sorted by class folder and then by file path;
- `classification_report.txt`, `confusion_counts.npy`, `confusion_props.npy` and PNGs,
  `montage.png` (top-K images per true class);
- with `--dump-preds FILE`: a `path,true,pred` CSV of the evaluated images (an input
  format of `stats/paired_tests.py`).

The checkpoint is a dict with `state_dict` (or `model`) and `classes`, as written by the
trainers and in the weights deposit; it is loaded with `weights_only=True` first (every
checkpoint of the paper loads this way). A file whose weights do not cover the model (for
example a `*_bpaco_full.pth` training state) is refused instead of being evaluated with
uninitialised weights. For hub ViTs (DinoBloom-B) the input size is passed to timm: the
16 x 16 patch grid has 257 position embeddings, the shape stored in the DinoBloom
checkpoint and in the fine-tuned checkpoints, so the weights load as stored with no
position-embedding interpolation (Supplementary Section S7).

## Analyses with their own settings

| Analysis | Script | Settings (as run; Supplementary Section S17) |
|---|---|---|
| LODO: Table 4, Note 1, S14, S16 | `evaluate_lodo.py` | common evaluator without test-time augmentation; nine classes (only neutrophil and lymphocyte sub-types merged, blast sub-types kept apart); test set = every pooled cell of the held-out source |
| Resolution sweep, Section 3.7 | `evaluate_resolution.py` | `img_size` passed to `timm.create_model`; resize `img_size * 256 / 224`, crop `img_size`; flip averaging; nine-class probabilities summed into seven classes before the argmax |
| Missed versus detected, S12 | `detector_abstention_quality_test.py` | Resize 256, CenterCrop 224, no padding, no flip; three-seed probability sum; nine-class argmax mapped; control group drawn once with `random.Random(0)` |
| Frozen DinoBloom-B probe, Table 3 | `frozen_probe.py` | the pre-specified protocol in the script's docstring: square padding, 256, crop 224 on every split, class-token embeddings, standardised, logistic regression selected on the val split; flip averaging externally |

The internal macro-F1 column of Table 3 is the nine-class score in each trainer's
`test_results.json` (no test-time augmentation, each model's own preprocessing;
see `train/README.md`). To regenerate this endpoint from an exported checkpoint
without retraining, use `evaluate_internal.py` on the exact internal test split:

```bash
python eval/evaluate_internal.py --trainer classifier \
  --model-path /path/to/swin_base_colour_aug_cwce_seed42.pth \
  --test-root /path/to/exact_internal_split/test \
  --output-dir /path/to/new_outputs/internal_seed42 --device cuda
```

Use `--trainer cnn` for the CNN-baseline trainer and `--trainer dinobloom` for
fine-tuned DinoBloom. The evaluator imports the actual trainer's validation
transform and overlays the checkpoint's saved arguments on the trainer defaults.
The `paper` recipe resizes to 256 and center crops without square padding; the
`meta` recipe and CNN default square-pad, resize to 246 and center crop. These are
224-pixel examples, not a single protocol interchangeable with the external
256-pixel/hflip evaluator. Checkpoint class order is retained and no class merge
or TTA is applied. Invalid weights, corrupt images and class-folder mismatches
fail rather than being silently skipped. The output includes nine-class scores,
the confusion matrix, per-image predictions and S4 blast-subtype statistics.
Loss is not regenerated, since it additionally depends on training class weights.

The exact input split must separately match its image manifest; a test count alone
does not establish split identity. These outputs do not replace the seven-class
probability ensemble of Fig. 3 or S11. The internal seven-class summaries of S17
come from `stats/internal_seven_class.py`.

### Offline architecture configuration for fine-tuned DinoBloom

Fine-tuned DinoBloom checkpoints contain classifier weights, but an `hf-hub:`
model name can still ask timm for its architecture configuration. If the study
model bundle includes `assets/dinobloom_config/config.json`, avoid that network
dependency with the retained, revision-pinned configuration:

```bash
DINO_MODEL_NAME="local-dir:$MODEL_ROOT/assets/dinobloom_config" \
  bash eval/run_dinobloom_external_eval.sh
python eval/evaluate_internal.py --trainer dinobloom \
  --model-path /path/to/dinobloom_checkpoint.pth \
  --model-name "local-dir:$MODEL_ROOT/assets/dinobloom_config" \
  --test-root /path/to/exact_internal_split/test \
  --output-dir /path/to/new_outputs/dinobloom_internal
```

Use timm 1.0.27 for this `local-dir:` mode. The explicit 224-pixel input override
in the common model factory is retained; a generic configuration's advertised
518-pixel input must not replace the checkpoint's 224-pixel patch grid. This
configuration is not a replacement for the original pretrained model weights
needed to fit the frozen probe or to retrain a model.

## Files

| File | Purpose |
|---|---|
| `evaluate_classifier.py` | the common evaluator (formerly `test.py`) |
| `evaluate_internal.py` | native internal classifier endpoint (Table 3 and S4), no TTA or label mapping |
| `evaluate_dinobloom.py` | the common evaluator with the DinoBloom-B hub id as default model name (formerly `test_dino.py`) |
| `evaluate_resolution.py` | inference-only resolution sweep (formerly `eval_flexible_resolution.py`) |
| `evaluate_lodo.py` | LODO wrapper around the common evaluator |
| `frozen_probe.py` | frozen DinoBloom-B linear probe: fits the 15-configuration grid, protocol in the docstring |
| `evaluate_frozen_probe.py` | inference with an already-fitted probe file, no refitting (the fitted probe is not in the weights record; `RETAINED_PROBE.md`) |
| `RETAINED_PROBE.md` | how `evaluate_frozen_probe.py` is run with a fitted probe file and the pinned DinoBloom-B backbone files |
| `detector_abstention_quality_test.py` | missed-versus-detected comparison of S12 |
| `run_backbone_external_eval.sh` | Table 3 CNN baselines, BPaCo block and the ResNet50 configurations of Section 3.4 |
| `run_augmentation_recipe_eval.sh` | the proposed model and the S8 recipes (also the S17 policy sweep and no-flip run) |
| `run_dinobloom_external_eval.sh` | fine-tuned DinoBloom-B, Table 3 |
| `run_resolution_scaling_eval.sh` | Section 3.7 |
| `run_internal_mapped_eval.sh` | internal test split in the seven-class space (Fig. 3, S11) |
| `run_routes_ablation.sh` | Table 2b and S6: uncropped images and cropping routes A to E |
| `run_full_pipeline_eval.sh` | Table 2a and the primary endpoint: detect, classify, coverage-aware score; S12 |
| `run_lodo_eval.sh` | Table 4, Note 1, S14, S16 |

Every script prints its usage with `--help`; each runner's header lists its environment
variables and defaults.

## Which runner gives which paper number

External = pre-cropped WBCBench2026 Phase-1 (n = 8,061) unless stated; values are mean
± population SD over seeds (seeds 42/62/82; CNN baselines 42/43/44; LODO 42/52/62).

| Paper | Runner (settings) | Then | Paper value |
|---|---|---|---|
| Table 3, Swin-B proposed | `run_augmentation_recipe_eval.sh` (`colour_aug_proposed`) | `stats/seed_summary.py` (run by the runner) | 0.863 ± 0.028 (0.886 / 0.878 / 0.824); weighted F1 0.947 ± 0.004, balanced accuracy 0.887 ± 0.005, BLAST 0.793, PROMY 0.601 |
| Table 3, CNN baselines | `run_backbone_external_eval.sh` (`RUN_GROUPS=cnn`) | `stats/seed_summary.py` (run by the runner) | VGG16-BN 0.813 ± 0.032, EfficientNet-B0 0.796 ± 0.030, ResNet50 0.718 ± 0.010 |
| Table 3, BPaCo block | `run_backbone_external_eval.sh` (`RUN_GROUPS=bpaco`, released weights) | `stats/seed_summary.py` (run by the runner) | Swin-B 0.819 ± 0.018, ResNet50 AdamW 0.762 ± 0.017, ResNet50 SGD 0.815 ± 0.004 |
| Table 3, DinoBloom-B frozen probe | `frozen_probe.py` | printed, `summary.txt` | internal 0.938; external 0.809, weighted F1 0.911, balanced accuracy 0.822, BLAST 0.547, PROMY 0.558 |
| Table 3, DinoBloom-B fine-tuned | `run_dinobloom_external_eval.sh` | `stats/seed_summary.py` (run by the runner) | 0.822 ± 0.047 (0.835 / 0.873 / 0.759) |
| Section 3.4, S7: other configurations | `run_backbone_external_eval.sh` (`RUN_GROUPS=resnet50_configs`); `run_augmentation_recipe_eval.sh` (`class_weighted_ce`) | `stats/seed_summary.py` | ResNet50 colour AdamW 0.694 ± 0.017, colour SGD 0.727 ± 0.020, CW-CE default augmentation 0.731 ± 0.041; Swin-B CW-CE 0.856 ± 0.009 |
| S7 paired tests | the per-image outputs of the BPaCo group, the proposed model and DinoBloom-B (`path_true_pred.csv` or `predictions.csv`, seed 42 first) | `stats/paired_tests.py` | e.g. Swin-B vs ResNet50 BPaCo AdamW: Δ +0.062, 95% CI [+0.044, +0.079]; proposed vs DinoBloom-B: Δ +0.031, seed-level p ≈ 0.15 |
| S9 ensemble intervals | proposed model and CNN baselines as above | `stats/bootstrap_ensemble_ci.py` | Swin 0.893 (0.874–0.911); VGG16-BN 0.841 (0.820–0.860); ResNet50 0.751 (0.737–0.766); EfficientNet-B0 0.848 (0.830–0.865) |
| S10 per-class F1 | as above | `stats/seed_summary.py --per-class` | e.g. PROMY 0.601 / 0.446 / 0.177 / 0.491 (Swin / VGG16-BN / ResNet50 / EfficientNet-B0) |
| Section 3.6, S8 | `run_augmentation_recipe_eval.sh` (three recipes) | `stats/recipe_ensembles.py` | ensembles 0.893 / 0.881 / 0.864; recipe-selection bootstraps +0.013 [+0.0004, +0.025] and +0.030 [+0.016, +0.044] |
| Calibration (Section 3.6, Supplementary Fig. S2) | `run_augmentation_recipe_eval.sh` (`colour_aug_proposed`) | `figures/make_calibration.py`, `stats/calibration.py` | ECE 0.017, Brier score 0.074 |
| S11 thresholds | `run_internal_mapped_eval.sh` and `run_augmentation_recipe_eval.sh` (`colour_aug_proposed`) | `stats/threshold_analysis.py` | BLAST t* 0.500: 0.841 → 0.839 (oracle 0.849); PROMY t* 0.480: 0.702 → 0.719 (oracle 0.744) |
| Fig. 3, Section 3.8 | `run_internal_mapped_eval.sh` (internal, n = 12,803) and `run_augmentation_recipe_eval.sh` (external) | `figures/make_confusion_matrix.py` | internal recall 0.90–0.98; 368 external errors |
| Section 3.7 resolution sweep | `run_resolution_scaling_eval.sh` | printed by the runner | 224 px 0.863 ± 0.027, 256 px 0.830 ± 0.039, 384 px 0.464 ± 0.084 |
| Table 2b, S6 routes | `run_routes_ablation.sh` | `stats/coverage_aware_score.py` (run by the runner) | uncropped 0.627 ± 0.039; A 0.863 ± 0.028; B 0.879 ± 0.006; C 0.867 ± 0.013; D 0.760 ± 0.055; E 0.882 ± 0.019 |
| Table 2a, primary endpoint | `run_full_pipeline_eval.sh` | `stats/coverage_aware_score.py` (run by the runner) | Phase-1 0.879 ± 0.006 (coverage 8,079/8,080, balanced accuracy 0.882); Phase-2 0.769 ± 0.002 (4,848/5,196 detected, 0.795); detected cells only 0.788 ± 0.002 (0.833) |
| S12 | `run_full_pipeline_eval.sh` (`RUN_ABSTENTION=1`) or `detector_abstention_quality_test.py --detections` | printed / `--output-csv` | accuracy 0.727 (missed) vs 0.862 (class-matched detected), n = 348 each |
| Table 4, Note 1, S14 | `run_lodo_eval.sh` (`ARM=proposed`; `ARM=mixup`) | `stats/lodo_present_class.py` | AML-LMU 0.590 ± 0.008, PBC 0.909 ± 0.015, Raabin-WBC 0.742 ± 0.015, HRLS 0.763 ± 0.008, BMCMHF 0.606 ± 0.030, mean 0.722; earlier mixup arm mean 0.685 |
| S16 | `run_lodo_eval.sh` (`ARM=dinobloom_proposed`; `ARM=dinobloom_bpaco`) | `stats/lodo_present_class.py` | fold-equal means 0.776 and 0.759 (cell-weighted 0.715 and 0.692) |
| S17 policy sweep | `run_augmentation_recipe_eval.sh` (`RECIPES="aug_stain_only aug_geometry_only aug_conservative aug_blur_noise aug_stain_blur_noise"`; `TTA=none`) | `stats/seed_summary.py` | seed means 0.773 (stain-blur-noise) to 0.863 (color-only), baseline 0.856; without flip averaging 0.860 versus 0.863 |

Notes on these rows:

- Route B of `run_routes_ablation.sh` and the Phase-1 split of
  `run_full_pipeline_eval.sh` are the same computation.
- The paper's route and Phase-2 values are those of the unified re-evaluation (S17):
  every route and Phase-2 re-scored post hoc with the common evaluator. The runners do
  exactly that. Re-scoring reproduced the earlier pre-cropped and uncropped runs
  image for image.
- Route E runs the detector on each route-C crop with a 5% margin and keeps the
  Cellpose crop when no box is found (`--fallback copy`, as run); in the paper every
  crop yielded a box.
- Resolution sweep: the paper's 224-px seed-62 value was taken from the common evaluator
  (S17). For that checkpoint both mapping rules give identical predictions on all 8,061
  images (macro-F1 0.8785), so `run_resolution_scaling_eval.sh` reproduces 0.863 ± 0.027
  with `evaluate_resolution.py` alone; the common evaluator gives 0.863 ± 0.028 (Table 3).
- Rows that need weights outside the deposit (CNN baselines, fine-tuned DinoBloom-B, the
  ResNet50 and CW-CE configurations, mixup + BPaCo + cRT, the policy sweep, the LODO
  arms) read checkpoints retrained with the matching `train/scripts/` runner.
- The S13 two-hematologist review is per-image WBCBench2026 material and is not part of
  this repository.

## Inputs

**Software.** Everything runs in `journal_env` (`environment/journal_env.yml`). The paper's
LODO evaluations of Table 4 ran in the second stack (`environment/cnn_env.yml`); the
evaluator code is the same. Cellpose crops for routes C and D are made beforehand in
the Cellpose environment (`environment/cellpose_env.yml`, `data_prep/cellpose_crop.py`).
Runners call `$PYTHON` (default `python`): activate the environment first or point
`PYTHON` at its interpreter.

**Data** (`DATA_ROOT`, default `./data`; every folder can be overridden by the variable
named in the runner header). The `WBCB_*` names are the ones the `regenerate` commands
of `analysis/evidence/MANIFEST.csv` use, so one set of variables serves both.

| Variable | Default folder under `DATA_ROOT` | Content |
|---|---|---|
| `WBCB_P1_FIELDS` | `wbc-bench-2026_phase1_nocrop_formatted` | Phase-1 full fields, one folder per native class (8,288 fields; 8,080 labelled in the seven classes); `data_prep/prepare_wbcbench.py --split phase1` |
| `WBCB_P2_FIELDS` | `wbc-bench-2026_phase2_eval_nocrop_formatted` | Phase-2 evaluation fields (5,350; 5,196 labelled); `data_prep/prepare_wbcbench.py --split phase2_eval` |
| `WBCB_P1_CROPS` | `wbc-bench-2026_phase1_cropped_formatted` | Phase-1 pre-cropped set (route A): the detector's highest-confidence box per field at confidence 0.70, no margin (8,269 crops; `data_prep/detector_crop.py --input-dir "$WBCB_P1_FIELDS" --output-dir "$WBCB_P1_CROPS"`) |
| `ROUTE_C_DIR`, `ROUTE_D_DIR` | `wbc-bench-2026_phase1_nocrop_formatted_cellpose_keepbg_224`, `..._cellpose_224` | Cellpose crops with and without background (`data_prep/cellpose_crop.py`) |
| `CORPUS_ROOT`, `INTERNAL_TEST_ROOT` | `Mixed_data_mix_mono_9_split_copy` (`test/` = 12,803 cells) | merged nine-class corpus (same default as `train/scripts/`) |
| LODO held-out pools (`LODO_TEST_ROOT_<FOLD>`) | `LMU_LODO_corrected_test`, `BM_cytomorphology_data_single_wbc_07_for_train_9class`, `dataset_DIB_single_wbc_copy`, `HRLS_dataset_single_wbc_copy`, `Raabin_WBC_single_wbc_copy` | AML-LMU 18,128, BMCMHF 73,096, PBC 11,845, HRLS 15,405, Raabin-WBC 10,154 cells |

**Core weights** (the 13 files of the weights record's `weights_core/`) are looked up in
`$MODEL_DIR`, then `pipeline/models/`, then `models/`, as in `pipeline/classify.py`:
`swin_base_colour_aug_cwce_seed{42,62,82}.pth` (the proposed model),
`swin_base_bpaco_seed*.pth`, `resnet50_bpaco_adamw_seed*.pth`,
`resnet50_bpaco_sgd_seed*.pth` and `yolo11_wbc_detection_best.pt`. The released files
hold the same weights as the checkpoints behind the paper.

**Locally trained checkpoints** are read from the run folders of `train/scripts/`
(`MODEL_ROOT`, default `$OUT_ROOT`, i.e. the same `OUT_ROOT` as training), each with
`seed_<seed>/best_macro_f1.pth` unless stated:

- in place of a core file that is not found: `swin_base_color_aug_cwce/` (the
  proposed model, in `run_augmentation_recipe_eval.sh` and `run_internal_mapped_eval.sh`)
  and `swin_base_bpaco/`, `resnet50_bpaco_adamw/`, `resnet50_bpaco_sgd/` (the BPaCo
  comparison models, in `run_backbone_external_eval.sh`). The resolution, routes and
  full-pipeline runners read only the released files, or the files named by
  `CKPT_PATTERN`;
- models without released weights: `cnn_cwce/<model>/`, `swin_base_cwce_default_aug/`,
  `swin_base_mixup_bpaco_crt/` (`crt_best_macro_f1.pth`), `resnet50_color_aug_cwce_adamw/`,
  `resnet50_color_aug_cwce_sgd/`, `resnet50_cwce_default_aug/`,
  `dinobloom_b_color_aug_cwce/`, `swin_base_aug_<policy>/`;
- LODO: `lodo_swin_base_color_aug_cwce/`, `lodo_swin_base_mixup_cutmix_cwce/` and
  `lodo_dinobloom_b_{proposed,bpaco}/`, with `ALL_minus_<FOLD>/seed_<seed>/`.

## Outputs, and WBCBench2026 data

Runners write under `OUT_ROOT` (default `./outputs`, resolved against the current
directory) and refuse to run when it lies inside the repository:
`backbone_external/`, `augmentation_recipes/`, `dinobloom_external/`, `resolution/`,
`internal_mapped/`, `routes/`, `full_pipeline/` and `lodo/`, each with
`<run>/seed_<seed>/`. The `stats/` scripts they call write to `$OUT_ROOT/stats/`: seed
summaries to `stats/<runner folder>/seed_summary.csv`, coverage-aware scores to
`stats/coverage_aware_route_<route>.json` and `stats/coverage_aware_full_pipeline_<split>.json`.
A folder that already holds its result is skipped; `FORCE=1` re-runs it.

**WBCBench2026 outputs stay on your own machine.** `predictions.csv`,
`path_true_pred.csv` and `detections.csv` carry benchmark image IDs, and `montage.png`
and the detector crops are benchmark images; the competition rules restrict
redistribution and the paper does not redistribute images, labels or per-image
predictions. Do not commit or share them (the repository's `.gitignore` ignores CSV and
image files by default). Only aggregate tables (confusion matrices, per-class metrics,
summary statistics) may leave your machine. The LODO outputs are public-dataset cells;
their montages reproduce dataset images, so check each dataset's licence before sharing.

## Reproducibility notes

- The evaluator code is the code that produced the paper's numbers. Changes made for the
  release do not alter a prediction: images are listed in sorted order, `--tta` defaults
  to `hflip` (every runner passes it explicitly and `evaluate_lodo.py` passes `none`, as
  the LODO runs used), `--dump-preds` was added, checkpoints are loaded weights-only
  first, checkpoints with missing weights are refused, paths come from the command
  line, and output paths inside the repository are refused.
- Release check (CPU, on public PBC and HRLS crops arranged to exercise the merge and
  ignore rules): `evaluate_classifier.py` against the evaluator as run, with and without
  flip averaging and mapping, gave identical labels, probabilities, reports and confusion
  matrices (rows reordered only); `evaluate_dinobloom.py` against the DinoBloom-B
  evaluator as run, `evaluate_resolution.py` (224, 256 and 384 px) and `evaluate_lodo.py`
  against their originals gave identical outputs; `frozen_probe.py` reproduced the
  original probe's embeddings, grid, selection and predictions on a small corpus;
  `detector_abstention_quality_test.py` printed the same groups and accuracies; and the
  `--dump-preds` labels of the two BPaCo backbones equalled those of the dual-view
  evaluator that exported the S7 BPaCo predictions.
- GPU inference is not bit-reproducible across hardware, and re-running the detector can
  move a few boxes (also when the pre-cropped set is rebuilt with `data_prep/`), so a
  handful of individual predictions may differ from the paper's runs.
- `predictions.csv` rows are now sorted; the paper's runs listed images in filesystem
  order. Statistics that resample rows in file order (the recipe-selection bootstrap of
  `stats/recipe_ensembles.py`) can therefore differ slightly from the printed intervals;
  `stats/README.md` explains the order options.
- DinoBloom-B is read from the Hugging Face hub pinned to revision
  `4f2a9d8593a7d12efe210778ea8560863fcacd03`, the snapshot attributed to the reported runs
  (S7). Later revisions keep the same weights and, with `img_size` passed explicitly,
  build the same model (checked: identical outputs). Set `HF_HUB_OFFLINE=1` once the
  files are cached.
- `frozen_probe.py` caches the embeddings in `--out`; a re-run with the same folder
  re-fits the probe grid from them.
