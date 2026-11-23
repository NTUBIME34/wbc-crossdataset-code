#!/usr/bin/env bash
# Paper row: main Table 3, "ResNet50 (BPaCo; SGD)", seeds 42/62/82.
#   Reported: internal macro-F1 0.933; external 0.815 +/- 0.004.
#   The earlier BPaCo arm with backbone-specific optimization (main Section 3.4,
#   Supplementary S7); it is not one of the shared-protocol comparisons.
# Recipe: BPaCo objective and views as in run_swin_bpaco_adamw.sh, but SGD (momentum 0.9,
#   lr 0.02, wd 5e-4), full precision, no gradient checkpointing; batch 128; 30 epochs.
# Provenance: block 3 of the original BPaCo runner; the saved arguments of the reported runs
#   equal these flags except --root, --outdir and --device.
# Output: $OUT_ROOT/resnet50_bpaco_sgd/seed_<seed>/best_macro_f1.pth (released as
#   resnet50_bpaco_sgd_seed<seed>.pth).
# Environment: journal_env, a CUDA GPU, and the BPaCo loss:
#   bash train/third_party/fetch_bpaco_loss.sh
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_resnet50_bpaco_sgd.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
no_args "$@"

CORPUS="$(corpus_root)"
check_split_root "${CORPUS}"
for seed in $(seed_list 42 62 82); do
  run_trainer train_classifier.py "${OUT_ROOT}/resnet50_bpaco_sgd/seed_${seed}" test_results.json \
    --root "${CORPUS}" \
    --model-name resnet50 \
    --seed "${seed}" \
    --recipe paper \
    --epochs 30 \
    --batch-size 128 \
    --workers 8 \
    --optimizer sgd \
    --lr 0.02 \
    --weight-decay 0.0005 \
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
