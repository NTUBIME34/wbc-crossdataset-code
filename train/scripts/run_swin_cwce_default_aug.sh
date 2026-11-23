#!/usr/bin/env bash
# Paper rows: the Swin-B class-weighted-CE baseline with the default augmentation, seeds 42/62/82.
#   Main Section 3.4 / Supplementary S7 (third configuration evaluated on both backbones):
#     external 0.856 +/- 0.009 (per seed 0.849/0.869/0.849), internal 0.959.
#   Supplementary Table S8, "Class-weighted CE (Swin-B)": three-seed ensemble macro-F1 0.881.
#   Supplementary S17: the baseline the six augmentation policies replace the augmentation of.
# Recipe: the proposed configuration (run_proposed.sh) with the default augmentation instead of
#   the color-only policy: square padding, resize 1.1x, random-resized crop 224 (scale
#   0.8-1.0), horizontal flip, color jitter 0.2/0.2/0.2/0.05.
# Provenance: run A0 of the original Swin-B ablation runner. The reported runs predate the
#   --aug-policy flag. meta_default is its default and the transform the paper describes for
#   this baseline (random-resized crop, flip, color jitter); the earlier trainer version was
#   not kept, so this equivalence is not bit-verified. The other saved arguments equal these
#   flags except --root, --outdir and --device.
# Output: $OUT_ROOT/swin_base_cwce_default_aug/seed_<seed>/best_macro_f1.pth.
# Environment: journal_env. Needs no third-party file.
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_swin_cwce_default_aug.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
no_args "$@"

CORPUS="$(corpus_root)"
check_split_root "${CORPUS}"
for seed in $(seed_list 42 62 82); do
  run_trainer train_classifier.py "${OUT_ROOT}/swin_base_cwce_default_aug/seed_${seed}" test_results.json \
    --root "${CORPUS}" \
    --model-name swin_base_patch4_window7_224 \
    --seed "${seed}" \
    --img-size 224 \
    --epochs 30 \
    --batch-size 128 \
    --workers 8 \
    --optimizer adamw \
    --amp \
    --grad-checkpointing \
    --moco-dim 128 \
    --moco-k 8192 \
    --moco-t 0.2 \
    --alpha 0.05 \
    --beta 1.0 \
    --gamma 1.0 \
    --log-interval 25 \
    --recipe meta \
    --aug-policy meta_default \
    --loss-type ce_cw \
    --bpaco-weight 0.0 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --warmup-epochs 5 \
    --layer-wise-lr-decay \
    --layer-decay 0.80 \
    --head-lr-mult 2.0
done
