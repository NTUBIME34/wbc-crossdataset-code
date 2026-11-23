#!/usr/bin/env bash
# Paper rows: main Table 4, Supplementary Note 1 and Table S14: leave-one-dataset-out (LODO)
#   with the proposed recipe, five folds x seeds 42/52/62.
#   Reported present-class macro-F1 (Table 4): AML-LMU 0.590 +/- 0.008, PBC 0.909 +/- 0.015,
#   Raabin-WBC 0.742 +/- 0.015, HRLS 0.763 +/- 0.008, BMCMHF 0.606 +/- 0.030; mean 0.722.
#   These come from eval/run_lodo_eval.sh (common evaluator without test-time augmentation,
#   present-class macro-F1; AML-LMU on its corrected 18,128-cell test set), not from the
#   trainer's own test_results.json (246-px internal preprocessing, nine-class macro-F1).
# Recipe: run_proposed.sh, trained on the other four sources' training splits and
#   checkpoint-selected on their validation splits (Supplementary Note 1).
# Provenance: the original LODO runner of the proposed recipe; the saved arguments of the
#   reported runs equal these flags except --root, --outdir and --device. That runner also
#   trained seed 82, which the paper does not report.
# Folds: LMU (AML-LMU), DIB (PBC), Raabin (Raabin-WBC), HRLS, BM (BMCMHF); split roots from
#   lodo_root in common.sh (LODO_ROOT_<FOLD> overrides).
# Output: $OUT_ROOT/lodo_swin_base_color_aug_cwce/ALL_minus_<FOLD>/seed_<seed>/best_macro_f1.pth.
# Environment: journal_env. Needs no third-party file.
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_lodo_proposed.sh [FOLD ...]
#        (no FOLD = all five; SEEDS="42" etc. for a subset of seeds)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

FOLDS=("$@")
[[ ${#FOLDS[@]} -gt 0 ]] || FOLDS=(LMU DIB Raabin HRLS BM)
for fold_arg in "${FOLDS[@]}"; do
  fold="$(canonical_fold "${fold_arg}")"
  root="$(lodo_root "${fold}")"
  check_split_root "${root}"
  for seed in $(seed_list 42 52 62); do
    run_trainer train_classifier.py "${OUT_ROOT}/lodo_swin_base_color_aug_cwce/ALL_minus_${fold}/seed_${seed}" test_results.json \
      --root "${root}" \
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
done
