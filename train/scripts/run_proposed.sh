#!/usr/bin/env bash
# Paper row: main Table 3, "Swin-B (color aug + CW-CE; proposed)", seeds 42/62/82.
#   Reported: internal macro-F1 0.956; external 0.863 +/- 0.028 (per seed 0.886/0.878/0.824).
#   The same three checkpoints are the classifier of the full pipeline (main Table 2),
#   of the "Color aug (proposed)" ensemble (Supplementary Tables S8, S9) and of S10, S11, Fig. 3.
# Recipe: Supplementary Table S5, proposed column (color-only augmentation, class-weighted CE,
#   AdamW 1e-4 / 1e-4, 5-epoch warmup + cosine, layer-wise LR decay 0.8, head x2, batch 128,
#   30 epochs, AMP, gradient checkpointing). The MoCo/alpha/beta/gamma flags are inert at
#   --bpaco-weight 0 and only keep the saved arguments identical to the reported runs.
# Provenance: the color_only cell of the original augmentation-policy runner; the saved
#   arguments of the reported runs equal these flags except --root, --outdir and --device.
# Output: $OUT_ROOT/swin_base_color_aug_cwce/seed_<seed>/best_macro_f1.pth (same weights as the
#   released swin_base_colour_aug_cwce_seed<seed>.pth); test_results.json = internal score.
# Environment: journal_env. Needs no third-party file.
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_proposed.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
no_args "$@"

CORPUS="$(corpus_root)"
check_split_root "${CORPUS}"
for seed in $(seed_list 42 62 82); do
  run_trainer train_classifier.py "${OUT_ROOT}/swin_base_color_aug_cwce/seed_${seed}" test_results.json \
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
    --aug-policy color_only \
    --color-jitter-strength 1.0
done
