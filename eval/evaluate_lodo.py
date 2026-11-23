#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LODO evaluation wrapper for ALL-minus checkpoints (main-text Table 4, Supplementary
Note 1, S14 and S16).

The actual inference and report generation stay in evaluate_classifier.py (the common
evaluator); this wrapper fixes the leave-one-dataset-out settings used for the paper:

  * no test-time augmentation (--tta none);
  * nine-class evaluation: only neutrophil and lymphocyte sub-types are merged (rules
    below); the blast sub-types are NOT merged into a blast group, and Plasma_cell,
    Metamyelocyte and Myelocyte are ignored;
  * the held-out source dataset is the test set: every cell of that source's harmonized
    pool. The pools were cropped at detector confidence 0.30 for AML-LMU, PBC,
    Raabin-WBC and HRLS and at 0.70 for BMCMHF, as run and as Supplementary S1 states
    (data_prep/README.md, "Detector thresholds as run").

--model-name is the backbone of the checkpoint: swin_base_patch4_window7_224 for the Swin
arms, the DinoBloom-B hub id of evaluate_dinobloom.py for the S16 arms (the input size is
then passed to timm, as in evaluate_dinobloom.py).

Present-class macro-F1 (Table 4) is computed afterwards from predictions.csv by
stats/lodo_present_class.py. The paper's LODO evaluations were run in the second
software stack (environment/cnn_env.yml); the evaluator code is the same.

Held-out roots default to DATA_ROOT/<folder> with the folder names below; pass
--test-root to use any other location.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_classifier as base_test  # noqa: E402


# Held-out pool of each source (fold code -> default folder under DATA_ROOT).
DEFAULT_TEST_DIRS = {
    "BM": "BM_cytomorphology_data_single_wbc_07_for_train_9class",   # BMCMHF, 73,096 cells
    "LMU": "LMU_LODO_corrected_test",                                 # AML-LMU (corrected), 18,128
    "DIB": "dataset_DIB_single_wbc_copy",                             # PBC, 11,845
    "HRLS": "HRLS_dataset_single_wbc_copy",                           # HRLS, 15,405
    "Raabin": "Raabin_WBC_single_wbc_copy",                           # Raabin-WBC, 10,154
}
# Paper names of the five sources, and the spellings of the evidence file names
# (heldout_AMLLMU_*, heldout_RaabinWBC_*; also accepted by train/scripts/common.sh)
# -> fold codes used by the LODO folders (ALL_minus_<code>).
PAPER_NAMES = {
    "BMCMHF": "BM",
    "AML-LMU": "LMU",
    "AMLLMU": "LMU",
    "PBC": "DIB",
    "HRLS": "HRLS",
    "Raabin-WBC": "Raabin",
    "RaabinWBC": "Raabin",
}


def lodo_merge_rules() -> list[str]:
    return [
        "Neutrophil=Band_neutrophil|Segmented_neutrophil|Neutrophil|Neutrophil_sum|Neutrophil_band|Neutrophil_seg",
        "Lymphocyte=Lymphocyte|Lymphocyte_typical|Variant_lymphocyte|Prolymphocyte",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one ALL-minus checkpoint on its held-out LODO source dataset."
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=sorted(DEFAULT_TEST_DIRS) + sorted(set(PAPER_NAMES) - set(DEFAULT_TEST_DIRS)),
        help="Held-out source domain, as fold code or paper name (AMLLMU and RaabinWBC are "
             "also accepted), e.g. BM (= BMCMHF) for ALL-minus-BM.",
    )
    parser.add_argument("--model-path", required=True, type=str, help="Path to .pth checkpoint.")
    parser.add_argument(
        "--test-root",
        default=None,
        type=str,
        help="Override the default test root for this domain (default: $DATA_ROOT/<folder>, see the module docstring).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to save predictions.csv, classification report, confusion matrices and montage.",
    )
    parser.add_argument("--model-name", default="swin_base_patch4_window7_224", type=str)
    parser.add_argument("--img-size", default=224, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--top-k", default=4, type=int)

    args = parser.parse_args()

    domain = PAPER_NAMES.get(args.domain, args.domain)
    data_root = Path(os.environ.get("DATA_ROOT", "data")).expanduser()
    test_root = Path(args.test_root or data_root / DEFAULT_TEST_DIRS[domain]).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not test_root.is_dir():
        raise FileNotFoundError(f"Test root not found for domain {domain}: {test_root}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Delegate to the common evaluator so outputs stay compatible with every other run.
    argv = [
        "--test-root",
        str(test_root),
        "--model-path",
        str(model_path),
        "--model-name",
        args.model_name,
        "--img-size",
        str(args.img_size),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--use-mappings",
        "--tta",
        "none",
        "--output-dir",
        str(output_dir),
        "--top-k",
        str(args.top_k),
        "--highlight-class",
        "Blast",
        "--highlight-class",
        "Promyelocyte",
    ]

    for cls_name in ("Plasma_cell", "Metamyelocyte", "Myelocyte"):
        argv.extend(["--ignore-class", cls_name])

    for rule in lodo_merge_rules():
        argv.extend(["--merge", rule])

    print(f"[LODO] Domain:     {domain}")
    print(f"[LODO] Test root:  {test_root}")
    print(f"[LODO] Model path: {model_path}")
    print(f"[LODO] Output dir: {output_dir}")

    base_test.main(argv)


if __name__ == "__main__":
    main()
