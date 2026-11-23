#!/usr/bin/env bash
# Paper rows: Supplementary S17 "Development use of the external labels": the six augmentation
#   policies the color-only policy was chosen from, each replacing the default augmentation of
#   the Swin-B class-weighted-CE baseline (run_swin_cwce_default_aug.sh), seeds 42/62/82.
#   Reported: pre-cropped external macro-F1 seed means from 0.773 (stain-blur-noise) to 0.863
#   (color-only), against 0.856 for the baseline.
# Policies (train_classifier.py --aug-policy):
#   color_only        color jitter 0.12/0.12/0.12/0.03 (= the proposed model, run_proposed.sh)
#   stain_only        white-balance/gamma jitter (p 0.3)
#   geometry_only     mild random-resized crop (scale 0.9-1), flip, +/-10 degree rotation
#   conservative      geometry_only plus milder color jitter (0.08/0.08/0.08/0.02)
#   blur_noise        mild crop, flip, milder color jitter, light blur (p 0.15), noise (p 0.15)
#   stain_blur_noise  blur_noise plus white-balance/gamma jitter
# Provenance: the original augmentation-policy runner; the saved arguments of all reported
#   runs equal these flags except --root, --outdir and --device.
# Output: $OUT_ROOT/swin_base_aug_<policy>/seed_<seed>/best_macro_f1.pth.
# Environment: journal_env. Needs no third-party file.
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_aug_policy_sweep.sh
#   POLICIES="..." selects policies. color_only is left out by default because it is the
#   proposed model; add it to POLICIES to retrain it under this runner's folder name.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
no_args "$@"

POLICIES="${POLICIES:-conservative stain_only geometry_only blur_noise stain_blur_noise}"

policy_flags() {
  case "$1" in
    conservative)     echo "--aug-policy conservative --safe-crop-scale 0.90 --rotation-degrees 10 --color-jitter-strength 1.0" ;;
    color_only)       echo "--aug-policy color_only --color-jitter-strength 1.0" ;;
    stain_only)       echo "--aug-policy stain_only --stain-aug wb_gamma --stain-aug-p 0.30" ;;
    geometry_only)    echo "--aug-policy geometry_only --safe-crop-scale 0.90 --rotation-degrees 10" ;;
    blur_noise)       echo "--aug-policy blur_noise --safe-crop-scale 0.90 --blur-p 0.15 --noise-p 0.15 --noise-std 0.01" ;;
    stain_blur_noise) echo "--aug-policy stain_blur_noise --stain-aug wb_gamma --stain-aug-p 0.30 --safe-crop-scale 0.90 --blur-p 0.15 --noise-p 0.15 --noise-std 0.01" ;;
    *) die "unknown policy '$1'" ;;
  esac
}

CORPUS="$(corpus_root)"
check_split_root "${CORPUS}"
for policy in ${POLICIES}; do
  flags_str="$(policy_flags "${policy}")"
  read -r -a flags <<< "${flags_str}"
  for seed in $(seed_list 42 62 82); do
    run_trainer train_classifier.py "${OUT_ROOT}/swin_base_aug_${policy}/seed_${seed}" test_results.json \
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
      "${flags[@]}"
  done
done
