#!/usr/bin/env bash
# Paper rows: Supplementary Note 1 (and main Section 3.5): the earlier LODO arm,
#   Swin-B with class-weighted CE plus mixup/CutMix (alpha 0.4), same folds, seeds 42/52/62.
#   Reported: mean present-class macro-F1 0.685 (versus 0.722 for the proposed recipe), lower
#   on four of the five held-out sources; per-image evidence in analysis/evidence/lodo/.
#   Scores come from eval/run_lodo_eval.sh, not from the trainer's test_results.json.
# Recipe: train_cnn_baseline.py defaults (square padding, resize 1.1x, random-resized crop
#   0.8-1.0, flip, color jitter 0.2/0.2/0.2/0.05; AdamW 1e-4 / 1e-4, cosine; full precision)
#   with mixup 0.4 + CutMix 0.4, class-weighted soft-target CE, batch 64, 30 epochs.
# Provenance: the original LODO runner of this arm; the saved arguments of the reported runs
#   equal these flags except --root, --outdir and --device. They used the same fold roots as
#   run_lodo_proposed.sh (corrected splits for the four folds that train on AML-LMU).
# Output: $OUT_ROOT/lodo_swin_base_mixup_cutmix_cwce/ALL_minus_<FOLD>/seed_<seed>/best_macro_f1.pth.
# Environment: the trainer runs in journal_env or the CNN environment (environment/cnn_env.yml);
#   the environment of the reported runs was not recorded.
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_lodo_mixup_arm.sh [FOLD ...]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

FOLDS=("$@")
[[ ${#FOLDS[@]} -gt 0 ]] || FOLDS=(LMU DIB Raabin HRLS BM)
for fold_arg in "${FOLDS[@]}"; do
  fold="$(canonical_fold "${fold_arg}")"
  root="$(lodo_root "${fold}")"
  check_split_root "${root}"
  for seed in $(seed_list 42 52 62); do
    run_trainer train_cnn_baseline.py "${OUT_ROOT}/lodo_swin_base_mixup_cutmix_cwce/ALL_minus_${fold}/seed_${seed}" test_results.json \
      --root "${root}" \
      --model-name swin_base_patch4_window7_224 \
      --use-mixup \
      --mixup-alpha 0.4 \
      --cutmix-alpha 0.4 \
      --loss-type ce_cw \
      --epochs 30 \
      --batch-size 64 \
      --seed "${seed}"
  done
done
