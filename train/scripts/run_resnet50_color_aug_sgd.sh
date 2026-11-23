#!/usr/bin/env bash
# Paper row: main Section 3.4 / Supplementary S7, the color-augmentation configuration on
#   ResNet50 with the SGD settings of the earlier ResNet50 BPaCo arm, seeds 42/62/82.
#   Reported: external 0.727 +/- 0.020 (per seed 0.707/0.721/0.755), internal 0.939.
# Recipe: color-only augmentation and class-weighted CE as in run_proposed.sh, but SGD
#   (momentum 0.9, lr 0.02, wd 5e-4) with cosine decay and no warmup, layer-wise decay or
#   mixed precision (as in run_resnet50_bpaco_sgd.sh); batch 128; 30 epochs.
# Provenance: the SGD arm of the factorial runner; the saved arguments of the reported runs
#   equal these flags except --root, --outdir and --device.
# Output: $OUT_ROOT/resnet50_color_aug_cwce_sgd/seed_<seed>/best_macro_f1.pth.
# Environment: journal_env. Needs no third-party file.
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_resnet50_color_aug_sgd.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
no_args "$@"

CORPUS="$(corpus_root)"
check_split_root "${CORPUS}"
for seed in $(seed_list 42 62 82); do
  run_trainer train_classifier.py "${OUT_ROOT}/resnet50_color_aug_cwce_sgd/seed_${seed}" test_results.json \
    --root "${CORPUS}" \
    --model-name resnet50 \
    --seed "${seed}" \
    --img-size 224 \
    --epochs 30 \
    --batch-size 128 \
    --workers 8 \
    --optimizer sgd \
    --moco-dim 128 \
    --moco-k 8192 \
    --moco-t 0.2 \
    --alpha 0.05 \
    --beta 1.0 \
    --gamma 1.0 \
    --recipe meta \
    --loss-type ce_cw \
    --bpaco-weight 0.0 \
    --lr 0.02 \
    --weight-decay 0.0005 \
    --log-interval 25 \
    --aug-policy color_only \
    --color-jitter-strength 1.0
done
