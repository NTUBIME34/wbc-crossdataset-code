#!/usr/bin/env bash
# Paper row: main Table 3, "DinoBloom-B (color aug; fine-tuned)", seeds 42/62/82.
#   Reported: internal macro-F1 0.957; external 0.822 +/- 0.047 (per seed 0.835/0.873/0.759).
# Recipe: the proposed model's hyperparameters (run_proposed.sh) on the DinoBloom-B backbone
#   (Supplementary S7 "Fully fine-tuned foundation model"): run at 224 x 224 with no
#   position-embedding interpolation, layer-wise LR decay applied per transformer block.
# Provenance: the original DinoBloom three-seed runner; the saved arguments of the reported
#   runs equal these flags except --root, --outdir, --device and the revision pin of
#   --model-name (see DINOBLOOM_MODEL in common.sh).
# Output: $OUT_ROOT/dinobloom_b_color_aug_cwce/seed_<seed>/best_macro_f1.pth.
# Environment: journal_env (downloads the DinoBloom-B weights from the Hugging Face Hub on
#   first use). Needs no third-party file.
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_dinobloom_finetune.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
no_args "$@"

CORPUS="$(corpus_root)"
check_split_root "${CORPUS}"
for seed in $(seed_list 42 62 82); do
  run_trainer train_dinobloom.py "${OUT_ROOT}/dinobloom_b_color_aug_cwce/seed_${seed}" test_results.json \
    --root "${CORPUS}" \
    --model-name "${DINOBLOOM_MODEL}" \
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
    --log-interval 25 \
    --recipe meta \
    --aug-policy color_only \
    --color-jitter-strength 1.0 \
    --loss-type ce_cw \
    --bpaco-weight 0.0 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --warmup-epochs 5 \
    --layer-wise-lr-decay \
    --layer-decay 0.80 \
    --head-lr-mult 2.0
done
