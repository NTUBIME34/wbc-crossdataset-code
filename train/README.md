# Training

Everything needed to retrain the models the paper reports: one trainer per model family,
one runner per paper row in `scripts/`, and the detector trainer. Each runner's header
names the row it reproduces, the reported values and where its flags come from.

| Trainer | Models | Environment |
|---|---|---|
| `train_classifier.py` | proposed Swin-B; Swin-B and ResNet50 class-weighted-CE configurations; all BPaCo arms; mixup + BPaCo + cRT; augmentation-policy sweep; LODO of the proposed recipe | `journal_env` |
| `train_dinobloom.py` | DinoBloom-B fine-tuning (Table 3, S16). `train_classifier.py` plus `img_size` handling for hf-hub ViTs and per-block layer-wise LR decay; identical for Swin/CNN backbones | `journal_env` |
| `train_cnn_baseline.py` | class-weighted-CE CNN baselines (VGG16-BN, EfficientNet-B0, ResNet50); earlier mixup/CutMix LODO arm | CNN environment |
| `train_detector.py` | YOLO11-M WBC detector on TXL-PBC (deployed WBC-only model and the three-class variant) | Ultralytics 8.3.170 |

Environments (see `environment/README.md`):

* `journal_env` (`environment/journal_env.yml`): Python 3.10, PyTorch 2.5.1, timm 1.0.27.
* CNN environment (`environment/cnn_env.yml`): Python 3.12, PyTorch 2.9.0, timm 0.9.12. The
  CNN baselines were trained in it; timm resolves their pretrained weights to
  `resnet50.a1_in1k`, `vgg16_bn.tv_in1k` and `tf_efficientnet_b0.ns_jft_in1k`, the same
  tags as in `journal_env`.
* Detector: the released detector was trained with Ultralytics 8.3.170
  (`pip install ultralytics==8.3.170`); `train_detector.py` warns about other versions.
  The full pipeline and its evaluation use Ultralytics 8.4.82 from `journal_env`.

`randaugment.py` is third-party code (timm / PyContrast / PaCo, see its header) used only
by the BPaCo recipe. The BPaCo loss itself is not redistributed; see
[BPaCo loss](#bpaco-loss).

## Data

The runners read the folders built by `data_prep/` under `DATA_ROOT`:

| Variable | Default | Content |
|---|---|---|
| `CORPUS_ROOT` | `$DATA_ROOT/Mixed_data_mix_mono_9_split_copy` | merged nine-class corpus (127,938 cells; per-class 80/10/10 split, seed 42): `train/` 102,345, `val/`, `test/` 12,803 |
| `LODO_ROOT_LMU` | `$DATA_ROOT/LODO_ALL_minus_LMU_test_LMU_split` | AML-LMU held out |
| `LODO_ROOT_DIB` | `$DATA_ROOT/LODO_corrected_ALL_minus_DIB_test_DIB_split` | PBC held out |
| `LODO_ROOT_Raabin` | `$DATA_ROOT/LODO_corrected_ALL_minus_Raabin_test_Raabin_split` | Raabin-WBC held out |
| `LODO_ROOT_HRLS` | `$DATA_ROOT/LODO_corrected_ALL_minus_HRLS_test_HRLS_split` | HRLS held out |
| `LODO_ROOT_BM` | `$DATA_ROOT/LODO_corrected_ALL_minus_BM_test_BM_split` | BMCMHF held out |
| `TXLPBC_DATA_YAML` | `$DATA_ROOT/TXL-PBC/data.yaml` | TXL-PBC as distributed (detector only) |

Every classifier root holds `train/`, `val/` and `test/` with one sub-folder per class. The
four folds whose training pool contains AML-LMU use the corrected splits; the AML-LMU fold
uses the original build, whose training pool does not contain AML-LMU. No WBCBench2026 data
is used for training or checkpoint selection.

## Running

```bash
conda activate journal_env
export DATA_ROOT=/path/to/prepared/data OUT_ROOT=/path/to/outputs
bash train/scripts/run_proposed.sh                  # three seeds of the proposed model
SEEDS=62 bash train/scripts/run_proposed.sh         # one seed
DRY_RUN=1 bash train/scripts/run_proposed.sh        # print the exact trainer commands
```

Runner settings (all optional apart from `DATA_ROOT`): `OUT_ROOT` (default `./outputs`;
keep it outside the repository), `PYTHON` (default `python`), `DEVICE` (default `cuda`;
`cuda:1`, `cpu`, ...), `SEEDS`, `EXTRA_ARGS` (appended last, for smoke tests only) and
`DRY_RUN=1`. Run folders are `$OUT_ROOT/<tag>/seed_<seed>/` (LODO:
`$OUT_ROOT/<tag>/ALL_minus_<FOLD>/seed_<seed>/`). A finished run is skipped, and the
classifier trainers resume an interrupted run from `last_bpaco_full.pth`.

Each run folder holds `best_macro_f1.pth` (a plain timm checkpoint: `state_dict`,
`classes`, `args`; the checkpoint selected by validation macro-F1), `test_results.json`
(the internal nine-class test score, `f1` = macro-F1; for `train_cnn_baseline.py` under
`results.best_macro_f1.metrics`), `confusion_matrix.csv`, `meta.json` and the training
log. External WBCBench2026 metrics are computed from `best_macro_f1.pth` by the scripts
in `eval/` (see `eval/README.md`), with the common evaluator (square padding, resize to
256 px, center-crop 224, horizontal-flip test-time augmentation).

## Runner per paper row

Reported values are copied from the paper (external = pre-cropped WBCBench2026 Phase-1,
n = 8,061, mean ± population sd over seeds; internal = nine-class macro-F1 on the
12,803-cell held-out set).

| Runner | Paper row | Seeds | Reported |
|---|---|---|---|
| `run_proposed.sh` | Table 3, Swin-B (color aug + CW-CE; proposed); the classifier of Table 2, S8 to S11, Fig. 3 | 42/62/82 | internal 0.956, external 0.863 ± 0.028 |
| `run_cnn_baselines.sh` | Table 3, VGG16-BN / EfficientNet-B0 / ResNet50 (CW-CE) | 42/43/44 | external 0.813 ± 0.032 / 0.796 ± 0.030 / 0.718 ± 0.010 |
| `run_swin_bpaco_adamw.sh` | Table 3, Swin-B (BPaCo; AdamW) | 42/62/82 | internal 0.950, external 0.819 ± 0.018 |
| `run_resnet50_bpaco_adamw.sh` | Table 3, ResNet50 (BPaCo; AdamW, as Swin-B) | 42/62/82 | internal 0.933, external 0.762 ± 0.017 |
| `run_resnet50_bpaco_sgd.sh` | Table 3, ResNet50 (BPaCo; SGD) | 42/62/82 | internal 0.933, external 0.815 ± 0.004 |
| `run_dinobloom_finetune.sh` | Table 3, DinoBloom-B (color aug; fine-tuned) | 42/62/82 | internal 0.957, external 0.822 ± 0.047 |
| `run_swin_cwce_default_aug.sh` | Section 3.4 / S7, Swin-B class-weighted CE with default augmentation; S8 "Class-weighted CE (Swin-B)" | 42/62/82 | external 0.856 ± 0.009; S8 ensemble 0.881 |
| `run_resnet50_cwce_default_aug.sh` | Section 3.4 / S7, ResNet50 under the same configuration | 42/62/82 | external 0.731 ± 0.041 |
| `run_resnet50_color_aug_adamw.sh` | Section 3.4 / S7, proposed color-augmentation configuration on ResNet50 (AdamW) | 42/62/82 | external 0.694 ± 0.017 |
| `run_resnet50_color_aug_sgd.sh` | Section 3.4 / S7, the same with the SGD settings of the earlier BPaCo arm | 42/62/82 | external 0.727 ± 0.020 |
| `run_swin_mixup_bpaco_crt.sh` | S8, Mixup + BPaCo + cRT (checkpoint: `crt_best_macro_f1.pth`) | 42/62/82 | ensemble macro-F1 0.864 |
| `run_aug_policy_sweep.sh` | S17, the six augmentation policies the color-only policy was chosen from | 42/62/82 | seed means 0.773 (stain-blur-noise) to 0.863 (color-only) |
| `run_lodo_proposed.sh` | Table 4, Note 1, S14: LODO with the proposed recipe | 42/52/62 | present-class macro-F1 0.590 to 0.909, mean 0.722 |
| `run_lodo_mixup_arm.sh` | Note 1, earlier mixup/CutMix LODO arm | 42/52/62 | mean present-class macro-F1 0.685 |
| `run_lodo_dinobloom.sh CONFIG FOLD SEED` | S16, DinoBloom-B LODO, `proposed` and `bpaco` configurations (one cell per call) | 42/52/62 | fold-equal means 0.776 and 0.759 |
| `run_detector.sh [wbc_only\|three_class]` | Methods, YOLO11-M detector; S1 three-class variant | 3407 | validation precision 0.996, recall 0.987, mAP@50 0.994, mAP@50-95 0.880 |

The frozen DinoBloom-B linear probe of Table 3 is fitted by `eval/frozen_probe.py`; it has
no training run here. The LODO scores (Table 4, Note 1, S14, S16) come from the LODO
evaluation in `eval/` (common evaluator without test-time augmentation, present-class
macro-F1, AML-LMU scored on its corrected 18,128-cell test set), not from the trainers'
own `test_results.json`.

The 13-file core weights bundle contains the checkpoints below. This lists bundle
contents, not a claim that its public download record is already live; consult
`docs/RELEASE_METADATA.json` and the root README for release status. Additional
study models are separate from this core set. The runner that produces each core
checkpoint is:

* `swin_base_colour_aug_cwce_seed{42,62,82}.pth`: `run_proposed.sh`
* `swin_base_bpaco_seed*.pth`: `run_swin_bpaco_adamw.sh`
* `resnet50_bpaco_adamw_seed*.pth`: `run_resnet50_bpaco_adamw.sh`
* `resnet50_bpaco_sgd_seed*.pth`: `run_resnet50_bpaco_sgd.sh`
* `yolo11_wbc_detection_best.pt`: `run_detector.sh wbc_only` (best epoch)

## BPaCo loss

The BPaCo arms (`run_swin_bpaco_adamw.sh`, `run_resnet50_bpaco_adamw.sh`,
`run_resnet50_bpaco_sgd.sh`, `run_swin_mixup_bpaco_crt.sh`, `run_lodo_dinobloom.sh bpaco`)
use `BPaCoLoss` from the BPaCo authors' repository, which has no licence and is therefore
not included. Download and verify the pinned file once (commit `cbe5775`, git blob
`8a802682`):

```bash
bash train/third_party/fetch_bpaco_loss.sh   # -> train/third_party/bpaco_losses.py
```

The trainers import it only when `--bpaco-weight > 0`, so every other run works without it.
The upstream loss places its class weights on `cuda`, so these arms need a GPU. Details:
`third_party/README.md`.

## Provenance of the runner flags

The flags in every runner were checked against the arguments saved by the reported runs
(`meta.json` / checkpoint `args`): parsing each runner's command with the trainer's own
argument parser gives the saved arguments of every reported run (114 run folders), apart
from the data root, the output folder and the GPU index. Two intentional differences:

* DinoBloom-B runners pin the Hugging Face revision
  (`hf-hub:1aurent/vit_base_patch14_224.dinobloom@4f2a9d85...`, override with
  `DINOBLOOM_MODEL`). The reported runs used the unpinned name, which resolved to this
  snapshot; later revisions keep the same weights file and only edit `config.json` and the
  model card.
* `run_lodo_dinobloom.sh bpaco` turns gradient checkpointing on for every cell; 10 of the
  15 reported cells ran without it on a 48 GB GPU. It only trades memory for compute.

Four run sets predate the last edits of the trainer: the Swin-B and the ResNet50 (SGD)
BPaCo arms, the Swin-B class-weighted-CE baseline and mixup + BPaCo + cRT. Their saved
arguments lack flags added later, whose defaults keep the earlier code paths (for example
`--aug-policy meta_default`, which the two Swin-B class-weighted runners pass explicitly, or
`--warmup-epochs 0`). For the Swin-B BPaCo arm a one-epoch re-run with the current trainer
reproduced the first epoch exactly; the other three have no such check.

Seeds fix the data order and augmentation, but GPU training is not bitwise reproducible
across hardware and library versions, and the augmentation random streams also depend on
the number of data-loader workers (8 in every reported run). Expect seed-level values close
to, not identical with, the reported ones.

## Smoke test (CPU)

```bash
DATA_ROOT=/tmp/tiny CORPUS_ROOT=/tmp/tiny/corpus OUT_ROOT=/tmp/tiny_out SEEDS=42 DEVICE=cpu \
EXTRA_ARGS="--epochs 1 --batch-size 4 --workers 0 --max-train-steps 2 --max-eval-batches 1" \
  bash train/scripts/run_proposed.sh
```

Any class-folder dataset with `train/`, `val/` and `test/` works; `--no-pretrained`
avoids downloading weights. For `run_cnn_baselines.sh` use
`MODELS=resnet50 EXTRA_ARGS="--epochs 1 --batch-size 4 --num-workers 0"` (without
`MODELS` it trains all three backbones); for the detector,
`python train/train_detector.py --data /path/to/TXL-PBC/data.yaml --dry-run` prepares the
WBC-only labels and validates the arguments without training.

A smoke run still writes full-size checkpoints to `OUT_ROOT`: about 2.7 GB for the Swin-B
run above (`best_ba.pth`, `best_macro_f1.pth`, `best_macro_f1_bpaco_full.pth`,
`last_bpaco_full.pth`), about 0.8 GB per ResNet50 run and about 5 GB per VGG16-BN run. Put
`OUT_ROOT` on a disk with room and delete it afterwards.
