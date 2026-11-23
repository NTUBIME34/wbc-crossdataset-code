#!/usr/bin/env bash
# Paper row: main Section 3.4 / Supplementary S7, ResNet50 under the configuration of the Swin-B
#   class-weighted-CE baseline (default augmentation), seeds 42/62/82.
#   Reported: external 0.731 +/- 0.041 (per seed 0.678/0.777/0.738), internal 0.941;
#   Swin-B counterpart: run_swin_cwce_default_aug.sh (0.856).
# Recipe: run_swin_cwce_default_aug.sh with --model-name resnet50 and without gradient
#   checkpointing (a memory-saving setting, on for Swin-B only). The layer-wise LR decay is
#   defined over Swin stages and gives ResNet50 a single backbone learning rate (0.8x; head x2).
# Provenance: the backbone x recipe factorial runner (default-augmentation cell); the saved
#   arguments of the reported runs equal these flags except --root, --outdir and --device.
#   Added post hoc (Supplementary S7).
# Output: $OUT_ROOT/resnet50_cwce_default_aug/seed_<seed>/best_macro_f1.pth.
# Environment: journal_env. Needs no third-party file.
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_resnet50_cwce_default_aug.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
no_args "$@"

CORPUS="$(corpus_root)"
check_split_root "${CORPUS}"
for seed in $(seed_list 42 62 82); do
  run_trainer train_classifier.py "${OUT_ROOT}/resnet50_cwce_default_aug/seed_${seed}" test_results.json \
    --root "${CORPUS}" \
    --model-name resnet50 \
    --seed "${seed}" \
    --img-size 224 \
    --epochs 30 \
    --batch-size 128 \
    --workers 8 \
    --optimizer adamw \
    --amp \
    --moco-dim 128 \
    --moco-k 8192 \
    --moco-t 0.2 \
    --alpha 0.05 \
    --beta 1.0 \
    --gamma 1.0 \
    --recipe meta \
    --loss-type ce_cw \
    --bpaco-weight 0.0 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --warmup-epochs 5 \
    --layer-wise-lr-decay \
    --layer-decay 0.80 \
    --head-lr-mult 2.0 \
    --log-interval 25 \
    --aug-policy meta_default
done
