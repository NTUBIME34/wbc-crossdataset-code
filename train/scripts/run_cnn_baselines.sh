#!/usr/bin/env bash
# Paper rows: main Table 3, class-weighted-CE CNN baselines on the common corpus, seeds 42/43/44:
#   VGG16-BN (CW-CE)         internal 0.948, external 0.813 +/- 0.032
#   EfficientNet-B0 (CW-CE)  internal 0.944, external 0.796 +/- 0.030
#   ResNet50 (CW-CE)         internal 0.942, external 0.718 +/- 0.010
# Recipe: Supplementary Table S5, CNN column (square padding, resize 1.1x, random-resized crop
#   224 at scale 0.8-1.0, horizontal flip, color jitter 0.2/0.2/0.2/0.05; class-weighted CE;
#   AdamW 1e-4 / 1e-4 with cosine annealing; batch 64; 30 epochs; full precision). These are
#   the trainer's defaults; the flags below are the ones the reported runs were launched with.
# Provenance: the common-corpus CNN runner; the saved arguments of all nine reported runs
#   equal these flags except --root, --outdir and --device.
# Output: $OUT_ROOT/cnn_cwce/<model>/seed_<seed>/best_macro_f1.pth; test_results.json holds the
#   internal score under results.best_macro_f1.
# Environment: the second environment (environment/cnn_env.yml: Python 3.12, PyTorch 2.9.0,
#   timm 0.9.12). timm resolves the pretrained weights to resnet50.a1_in1k, vgg16_bn.tv_in1k
#   and tf_efficientnet_b0.ns_jft_in1k.
#
# Usage: DATA_ROOT=/path/to/prepared/data bash train/scripts/run_cnn_baselines.sh
#        MODELS="resnet50" SEEDS="42" ... to run a subset.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
no_args "$@"

CORPUS="$(corpus_root)"
check_split_root "${CORPUS}"
MODELS="${MODELS:-vgg16_bn tf_efficientnet_b0 resnet50}"
for seed in $(seed_list 42 43 44); do
  for model in ${MODELS}; do
    run_trainer train_cnn_baseline.py "${OUT_ROOT}/cnn_cwce/${model}/seed_${seed}" test_results.json \
      --root "${CORPUS}" \
      --model-name "${model}" \
      --seed "${seed}" \
      --img-size 224 \
      --epochs 30 \
      --batch-size 64 \
      --num-workers 8 \
      --lr 0.0001 \
      --weight-decay 0.0001 \
      --loss-type ce_cw
  done
done
