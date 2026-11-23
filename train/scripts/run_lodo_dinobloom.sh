#!/usr/bin/env bash
# Paper rows: Supplementary Table S16, DinoBloom-B fine-tuned on the identical LODO folds and
#   seeds (42/52/62) as the proposed model, in two configurations. Single-cell runner: one
#   call trains one (configuration, fold, seed) cell.
#     proposed  the proposed recipe's hyperparameters on DinoBloom-B (as run_dinobloom_finetune.sh)
#               reported fold-equal mean present-class macro-F1 0.776
#     bpaco     BPaCo variant: the BPaCo term (weight 0.25) and two-view augmentation of
#               Supplementary S7 with the proposed model's class-weighted CE and AdamW schedule
#               (lr 1e-4, wd 1e-4, 5-epoch warmup, per-block layer-wise decay 0.8, head x2)
#               reported fold-equal mean present-class macro-F1 0.759
#   Per-fold values: Supplementary Table S16 and analysis/evidence/lodo_foundation/; scores come
#   from eval/run_lodo_eval.sh, not from the trainer's test_results.json.
# Provenance: the saved arguments of the 15 reported cells of each configuration equal these
#   flags except --root, --outdir, --device, the revision pin of --model-name (DINOBLOOM_MODEL
#   in common.sh) and gradient checkpointing: 10 of the 15 bpaco cells ran without it on a
#   48 GB GPU. Gradient checkpointing only trades memory for compute (the ViT has no dropout
#   or drop-path), so it is on here for every cell. In the bpaco cells --recipe paper ignores
#   --aug-policy; the flag is kept because the reported runs recorded it.
# Output: $OUT_ROOT/lodo_dinobloom_b_<config>/ALL_minus_<FOLD>/seed_<seed>/best_macro_f1.pth.
# Environment: journal_env; the bpaco configuration also needs a CUDA GPU and the BPaCo loss
#   (bash train/third_party/fetch_bpaco_loss.sh).
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_lodo_dinobloom.sh CONFIG FOLD SEED
#   CONFIG: proposed | bpaco    FOLD: LMU DIB Raabin HRLS BM (or AML-LMU PBC Raabin-WBC HRLS BMCMHF)
#   All 30 cells:  for c in proposed bpaco; do for f in LMU DIB Raabin HRLS BM; do
#                  for s in 42 52 62; do bash train/scripts/run_lodo_dinobloom.sh $c $f $s; done; done; done
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[[ $# -eq 3 ]] || die "usage: run_lodo_dinobloom.sh CONFIG FOLD SEED (see the header of this file)"
config="$1"
fold="$(canonical_fold "$2")"
seed="$3"
[[ "${seed}" =~ ^[0-9]+$ ]] || die "SEED must be an integer, got '${seed}'"
case "${config}" in
  proposed) recipe_flags=(--recipe meta --bpaco-weight 0) ;;
  # BPaCo settings are the trainer defaults the reported cells ran with, spelled out here.
  bpaco)    recipe_flags=(--recipe paper --bpaco-weight 0.25 --moco-dim 128 --moco-k 8192 --moco-t 0.2
                          --alpha 0.05 --beta 1.0 --gamma 1.0 --randaug-m 10 --randaug-n 2) ;;
  *) die "CONFIG must be 'proposed' or 'bpaco', got '${config}'" ;;
esac

root="$(lodo_root "${fold}")"
check_split_root "${root}"
run_trainer train_dinobloom.py "${OUT_ROOT}/lodo_dinobloom_b_${config}/ALL_minus_${fold}/seed_${seed}" test_results.json \
  --root "${root}" \
  --model-name "${DINOBLOOM_MODEL}" \
  --seed "${seed}" \
  --img-size 224 \
  --epochs 30 \
  --batch-size 128 \
  --workers 8 \
  --optimizer adamw \
  --amp \
  --grad-checkpointing \
  "${recipe_flags[@]}" \
  --loss-type ce_cw \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --warmup-epochs 5 \
  --layer-wise-lr-decay \
  --layer-decay 0.80 \
  --head-lr-mult 2.0 \
  --aug-policy color_only \
  --color-jitter-strength 1.0 \
  --log-interval 50
