# Software environments

The paper's results came from three conda environments plus one pinned Ultralytics
version for detector training (Supplementary Table S5). The files here are exports of the
authors' environments (`conda env export --no-builds`, machine-specific lines removed).

| File | Python | Main packages | Used for |
|---|---|---|---|
| `journal_env.yml` | 3.10.20 | PyTorch 2.5.1 (CUDA 12.1), torchvision 0.20.1, timm 1.0.27, Ultralytics 8.4.82, scikit-learn 1.7.2, NumPy 2.2.6 | almost everything (table below) |
| `cnn_env.yml` | 3.12.12 | PyTorch 2.9.0 (CUDA 12.8), torchvision 0.24.0, timm 0.9.12 | CNN baselines; leave-one-dataset-out evaluation as run |
| `cellpose_env.yml` | 3.12.12 | Cellpose 4.1.1 (`cpsam` model), PyTorch 2.5.1 (CUDA 12.1) | Cellpose crops of Supplementary routes C and D |
| `journal_env` + `ultralytics==8.3.170` | 3.10.20 | as `journal_env`, with Ultralytics 8.3.170 | detector training |

```bash
conda env create -f environment/journal_env.yml
conda env create -f environment/cnn_env.yml        # CNN baselines; LODO evaluation as run
conda env create -f environment/cellpose_env.yml   # only for Cellpose routes C and D
```

`cnn_env.yml` omits packages of the authors' environment that no script here imports and
that no listed package depends on; every remaining version is as installed (see the file
header).

## Which step runs where

| Step | Scripts | Environment |
|---|---|---|
| Inference pipeline | `pipeline/run_pipeline.sh`, `pipeline/detect_yolo11.py`, `pipeline/classify.py` | `journal_env` (or `pip install -r pipeline/requirements.txt`, the same pins) |
| Data preparation | `data_prep/*.py`, `data_prep/crop_training_sources.sh`, except `cellpose_crop.py` | `journal_env` |
| Cellpose crops (routes C and D) | `data_prep/cellpose_crop.py` | `cellpose_env` |
| Detector training | `train/train_detector.py`, `train/scripts/run_detector.sh` | `journal_env` with `ultralytics==8.3.170` (below) |
| Swin-B, BPaCo and DinoBloom-B training, the ResNet50 configurations of Section 3.4, leave-one-dataset-out training of the proposed recipe and of DinoBloom-B | `train/train_classifier.py`, `train/train_dinobloom.py`; the corresponding `train/scripts/run_*.sh` | `journal_env` |
| CNN baselines (VGG16-BN, EfficientNet-B0, ResNet50) | `train/train_cnn_baseline.py`, `train/scripts/run_cnn_baselines.sh` | `cnn_env` |
| Earlier mixup/CutMix leave-one-dataset-out arm | `train/scripts/run_lodo_mixup_arm.sh` (`train/train_cnn_baseline.py`) | not recorded; the trainer runs in `journal_env` and in `cnn_env` |
| WBCBench2026 evaluation (every external number), frozen DinoBloom-B probe | `eval/` runners and evaluators other than the LODO ones, `eval/frozen_probe.py` | `journal_env` |
| Leave-one-dataset-out evaluation | `eval/run_lodo_eval.sh`, `eval/evaluate_lodo.py` | `cnn_env` as run; the evaluator code is the same in `journal_env` |
| Statistics, figures, verification | `stats/`, `figures/`, `analysis/verify_tables.py` | `journal_env` (CPU is enough) |

Shell runners call `$PYTHON` (the `eval/`, `train/scripts/` and `data_prep/` runners) or
`$PYTHON_BIN` (`pipeline/run_pipeline.sh`); activate the environment first or point the
variable at its interpreter.

## Detector training: Ultralytics 8.3.170

The released detector was trained with Ultralytics 8.3.170. Full-pipeline detector
inference and the detector's validation metrics used Ultralytics 8.4.82 from `journal_env`;
the version behind the detector crops of the training sources and of the pre-cropped set
was not recorded (Supplementary Table S5). For the training run itself the paper records
only the Ultralytics version. To retrain, use a second copy of `journal_env` with that
version (pip adds `py-cpuinfo`, which 8.3.170 requires):

```bash
conda env create -f environment/journal_env.yml -n detector_train_env
conda activate detector_train_env
pip install ultralytics==8.3.170
```

`train/train_detector.py` prints a warning when another Ultralytics version is installed.

## Notes

- **GPU wheels.** `journal_env.yml` and `cellpose_env.yml` install PyTorch's CUDA 12.1
  wheels from PyTorch's own index (the `--extra-index-url` line). On a CPU-only machine,
  or with another CUDA version, change that URL and the `+cu121` suffixes (for example
  `https://download.pytorch.org/whl/cpu` and `+cpu`); the `nvidia-*` and `triton` lines
  can then be deleted. The PyPI wheel of PyTorch 2.9.0 in `cnn_env.yml` runs on CPU as
  well.
- **Hardware.** Training used NVIDIA RTX 4090 (24 GB) and RTX 4060 Ti (16 GB) GPUs and a
  48 GB GPU (main Methods 2.9; Supplementary Table S5). By the run records, 10 of the
  15 BPaCo-variant cells of the DinoBloom-B leave-one-dataset-out grid (Supplementary
  Table S16) ran on the 48 GB GPU without gradient checkpointing and the other five on
  the 24 GB and 16 GB cards (`train/README.md`); the proposed-recipe cells of that grid
  have no host record. The BPaCo arms need a CUDA GPU, because the upstream BPaCo loss
  places its class weights on `cuda`. Inference, statistics, figures and the verifier
  run on CPU. The Cellpose crops run on CPU as well (float32, about 20 s per field), but
  the full Phase-1 set needs a GPU in practice (`data_prep/README.md`, step 6).
- **Downloads on first use.** timm downloads the pretrained backbones, the Hugging Face
  client downloads DinoBloom-B (pinned revision), Ultralytics downloads `yolo11m.pt` for
  detector training, and Cellpose downloads `cpsam`. Once DinoBloom-B is cached, set
  `HF_HUB_OFFLINE=1` to run without network access. Licences: `THIRD_PARTY_NOTICES.md`.
- **Optional package.** `stats/detector_overlap_audit.py` reads the TXL-PBC `.xlsx`
  metadata with pandas when `openpyxl` is installed and with a built-in reader otherwise,
  so no environment needs `openpyxl`.
