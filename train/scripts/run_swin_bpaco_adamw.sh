#!/usr/bin/env bash
# Paper row: main Table 3, "Swin-B (BPaCo; AdamW)", seeds 42/62/82.
#   Reported: internal macro-F1 0.950; external 0.819 +/- 0.018 (per seed 0.836/0.827/0.795).
#   Swin-B arm of the shared-protocol BPaCo comparison (main Section 3.4, Supplementary S7).
# Recipe: Supplementary S7 "BPaCo hyperparameters": CE + 0.25 x BPaCo, two augmented views
#   (random-resized crop 0.08-1, flip, color jitter 0.4, RandAugment n=2 m=10), MoCo queue
#   8,192, 128-d MLP projection, temperature 0.2, (alpha, beta, gamma) = (0.05, 1.0, 1.0);
#   AdamW lr 1e-4, wd 0.05, cosine, no warmup or layer-wise decay; batch 128; 30 epochs;
#   AMP and gradient checkpointing.
# Provenance: block 1 of the original BPaCo runner; the saved arguments of the reported runs
#   equal these flags except --root, --outdir and --device. A later one-epoch re-run with the
#   current trainer reproduced epoch 1 of the seed-42 run exactly.
# Output: $OUT_ROOT/swin_base_bpaco/seed_<seed>/best_macro_f1.pth (released as
#   swin_base_bpaco_seed<seed>.pth).
# Environment: journal_env, a CUDA GPU, and the BPaCo loss:
#   bash train/third_party/fetch_bpaco_loss.sh
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_swin_bpaco_adamw.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
no_args "$@"

CORPUS="$(corpus_root)"
check_split_root "${CORPUS}"
for seed in $(seed_list 42 62 82); do
  run_trainer train_classifier.py "${OUT_ROOT}/swin_base_bpaco/seed_${seed}" test_results.json \
    --root "${CORPUS}" \
    --model-name swin_base_patch4_window7_224 \
    --seed "${seed}" \
    --recipe paper \
    --epochs 30 \
    --batch-size 128 \
    --workers 8 \
    --optimizer adamw \
    --amp \
    --grad-checkpointing \
    --lr 0.0001 \
    --weight-decay 0.05 \
    --loss-type ce \
    --moco-dim 128 \
    --moco-k 8192 \
    --moco-t 0.2 \
    --alpha 0.05 \
    --beta 1.0 \
    --gamma 1.0 \
    --bpaco-weight 0.25 \
    --randaug-m 10 \
    --randaug-n 2
done
