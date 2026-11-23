#!/usr/bin/env bash
# Paper row: Supplementary Table S8, "Mixup + BPaCo + cRT" (the aggressive recipe), seeds 42/62/82.
#   Reported: three-seed ensemble macro-F1 0.864 (main Section 3.6). The reported checkpoint
#   is the cRT stage, crt_best_macro_f1.pth, not best_macro_f1.pth.
# Recipe (Supplementary S7, last sentences of "BPaCo hyperparameters"): the configuration of the
#   Swin-B class-weighted-CE baseline (run_swin_cwce_default_aug.sh) plus a BPaCo term (weight
#   0.1), mixup (alpha 0.1, no CutMix) and eight epochs of classifier re-training (cRT) with
#   class-balanced sampling and the backbone frozen.
# Provenance: run A4 of the original Swin-B ablation runner. The reported runs predate the
#   --aug-policy flag. meta_default is its default and the transform the paper describes for
#   this baseline (random-resized crop, flip, color jitter); the earlier trainer version was
#   not kept, so this equivalence is not bit-verified. The other saved arguments equal these
#   flags except --root, --outdir and --device.
# Output: $OUT_ROOT/swin_base_mixup_bpaco_crt/seed_<seed>/crt_best_macro_f1.pth
#   (crt_test_results.json = internal score after cRT).
# Environment: journal_env, a CUDA GPU, and the BPaCo loss:
#   bash train/third_party/fetch_bpaco_loss.sh
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_swin_mixup_bpaco_crt.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
no_args "$@"

CORPUS="$(corpus_root)"
check_split_root "${CORPUS}"
for seed in $(seed_list 42 62 82); do
  run_trainer train_classifier.py "${OUT_ROOT}/swin_base_mixup_bpaco_crt/seed_${seed}" crt_test_results.json \
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
    --use-mixup \
    --mixup-alpha 0.1 \
    --cutmix-alpha 0.0 \
    --bpaco-weight 0.10 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --warmup-epochs 5 \
    --layer-wise-lr-decay \
    --layer-decay 0.80 \
    --head-lr-mult 2.0 \
    --crt-epochs 8 \
    --crt-loss-type ce_cw \
    --crt-lr 0.0001 \
    --crt-weight-decay 0.00001 \
    --crt-sampler-weighting inv \
    --crt-freeze-backbone
done
