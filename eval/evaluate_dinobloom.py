#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_dinobloom.py

External evaluation of the fine-tuned DinoBloom-B checkpoints (main-text Table 3, row
"DinoBloom-B (color aug; fine-tuned)"). It is the common evaluator of
evaluate_classifier.py with a DinoBloom default for --model-name: the same
preprocessing (square padding, resize to 256 px, centre crop 224), horizontal-flip
test-time augmentation, nine-to-seven class mapping and output files, so the result is
directly comparable to every other row of Table 3.

Position embeddings: the timm hub configuration of DinoBloom used at run time
advertised the DINOv2 base resolution (518 px). evaluate_classifier.create_model passes
img_size=224 for hub ViTs, so timm builds the 16x16 patch grid whose 257 position
embeddings (256 patches + class token) match the 224-px embeddings stored in the
DinoBloom checkpoint and in the fine-tuned checkpoints. The weights are loaded as
stored; there is no position-embedding interpolation (Supplementary Section S7).

The model configuration is read from the Hugging Face hub (or the local cache). The
default name pins the hub revision attributed to the reported runs (Supplementary
Section S7; the runs used the unpinned name); with img_size passed explicitly, later
revisions build the same architecture. Set HF_HUB_OFFLINE=1 once the files are cached
to run without network access.

Usage:
    python eval/evaluate_dinobloom.py \
      --test-root /path/to/wbcbench_phase1_cropped \
      --model-path /path/to/dinobloom_finetune/seed_42/best_macro_f1.pth \
      --img-size 224 \
      --use-mappings \
      --tta hflip \
      --output-dir /path/to/outputs/dinobloom_external/seed_42

All other options are those of evaluate_classifier.py (see --help).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_classifier as base  # noqa: E402

DEFAULT_MODEL_NAME = "hf-hub:1aurent/vit_base_patch14_224.dinobloom@4f2a9d8593a7d12efe210778ea8560863fcacd03"


def main(argv=None):
    parser = base.build_parser(
        description="Evaluate a fine-tuned DinoBloom-B checkpoint with the common evaluator",
        default_model_name=DEFAULT_MODEL_NAME,
    )
    base.main(argv, parser=parser)


if __name__ == "__main__":
    main()
