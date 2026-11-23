#!/usr/bin/env bash
# Paper row: main Table 3, "ResNet50 (BPaCo; AdamW, as Swin-B)", seeds 42/62/82.
#   Reported: internal macro-F1 0.933; external 0.762 +/- 0.017 (per seed 0.741/0.761/0.783).
#   ResNet50 arm of the shared-protocol BPaCo comparison (main Section 3.4, Supplementary S7);
#   added post hoc with the Swin-B arm's exact training arguments, only --model-name differs.
# Recipe: identical to run_swin_bpaco_adamw.sh (see there).
# Provenance: the optimizer-matched BPaCo control runner; the saved arguments of the
#   reported runs equal these flags except --root, --outdir and --device.
# Output: $OUT_ROOT/resnet50_bpaco_adamw/seed_<seed>/best_macro_f1.pth (released as
#   resnet50_bpaco_adamw_seed<seed>.pth).
# Environment: journal_env, a CUDA GPU, and the BPaCo loss:
#   bash train/third_party/fetch_bpaco_loss.sh
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_resnet50_bpaco_adamw.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
no_args "$@"

CORPUS="$(corpus_root)"
check_split_root "${CORPUS}"
for seed in $(seed_list 42 62 82); do
  run_trainer train_classifier.py "${OUT_ROOT}/resnet50_bpaco_adamw/seed_${seed}" test_results.json \
    --root "${CORPUS}" \
    --model-name resnet50 \
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
