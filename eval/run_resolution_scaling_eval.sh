#!/usr/bin/env bash
# Inference-only resolution scaling of the 224-trained proposed model (main-text
# Section 3.7; reported in the text, no table). Inference only.
#
# The SAME proposed Swin-B checkpoints (released weights
# swin_base_colour_aug_cwce_seed{42,62,82}.pth) are evaluated on the WBCBench2026
# Phase-1 pre-cropped set (n = 8,061) at 224 / 256 / 384 px by passing img_size
# explicitly to timm.create_model (eval/evaluate_resolution.py: square padding, resize to
# img_size * 256 / 224, centre crop img_size, horizontal-flip TTA, probability-summed
# nine -> seven mapping); no retraining.
# Paper (mean +- population SD over seeds 42/62/82): 224 px 0.863 +- 0.027,
# 256 px 0.830 +- 0.039, 384 px 0.464 +- 0.084.
# Supplementary Section S17: the paper's seed-62 value at 224 px was taken from the common
# evaluator (eval/evaluate_classifier.py). For that checkpoint the two mapping rules give
# identical predictions (macro-F1 0.8785 either way), so this runner reproduces
# 0.863 +- 0.027 with evaluate_resolution.py alone; the common evaluator gives
# 0.863 +- 0.028 over the three seeds (main-text Table 3).
#
# Usage: bash eval/run_resolution_scaling_eval.sh [-h|--help]
#
# Environment (default in brackets):
#   DATA_ROOT      prepared data                                        [./data]
#   WBCB_P1_CROPS  pre-cropped Phase-1 class folders
#                  [$DATA_ROOT/wbc-bench-2026_phase1_cropped_formatted]
#   MODEL_DIR      released weight files                                [pipeline/models or models]
#   CKPT_PATTERN   checkpoint pattern, {seed} is replaced  [released swin_base_colour_aug_cwce_seed{seed}.pth]
#   SEEDS          seeds                                                [42 62 82]
#   SIZES          inference resolutions                                [224 256 384]
#   OUT_ROOT       outputs, must be outside the repository              [./outputs]
#   PYTHON         interpreter of the journal_env environment           [python]
#   DEVICE         cuda or cpu                                          [cuda]
#   FORCE          1 = re-run folders that already hold metrics.csv     [0]
# Outputs: $OUT_ROOT/resolution/img<size>/seed_<seed>/{metrics.csv, predictions.csv,
# classification_report.txt, confusion_matrix.csv, run.log}; summary table printed at the
# end. Per-image outputs contain WBCBench2026 image IDs: keep them on your own machine.
set -uo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
case "${1:-}" in -h|--help) usage; exit 0 ;; "") ;; *) usage >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_ROOT="${OUT_ROOT:-./outputs}"
[[ "${OUT_ROOT}" == /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT#./}"
# Refuse an OUT_ROOT inside the repository. Both paths are compared with symbolic links and
# '..' resolved; a part that does not exist yet is resolved the way mkdir -p will create it.
physical_path() {
  local rest="$1" part out="" phys
  [[ "${rest}" == /* ]] || rest="${PWD}/${rest}"
  while [[ -n "${rest}" ]]; do
    part="${rest%%/*}"
    if [[ "${rest}" == */* ]]; then rest="${rest#*/}"; else rest=""; fi
    case "${part}" in
      ""|.) ;;
      ..) out="${out%/*}" ;;
      *) out="${out}/${part}"
         if [[ -d "${out}" ]] && phys="$(cd "${out}" 2>/dev/null && pwd -P)"; then out="${phys}"; fi ;;
    esac
  done
  echo "${out:-/}"
}
case "$(physical_path "${OUT_ROOT}")/" in "$(physical_path "${REPO}")/"*)
  echo "[ERR] OUT_ROOT (${OUT_ROOT}) is inside the repository; set OUT_ROOT to a folder outside it." >&2; exit 2 ;;
esac
export OUT_ROOT
WBCB_P1_CROPS="${WBCB_P1_CROPS:-${DATA_ROOT}/wbc-bench-2026_phase1_cropped_formatted}"
SEEDS="${SEEDS:-42 62 82}"
SIZES="${SIZES:-224 256 384}"
EVAL_PY="${REPO}/eval/evaluate_resolution.py"
OUT_BASE="${OUT_ROOT}/resolution"

weight_path() {  # released weight file: $MODEL_DIR, then pipeline/models/, then models/
  local d
  for d in ${MODEL_DIR:+"${MODEL_DIR}"} "${REPO}/pipeline/models" "${REPO}/models"; do
    if [[ -f "${d}/$1" ]]; then echo "${d}/$1"; return 0; fi
  done
  echo "${MODEL_DIR:-${REPO}/models}/$1"
}
ckpt_for() {  # $1 seed
  if [[ -n "${CKPT_PATTERN:-}" ]]; then echo "${CKPT_PATTERN//\{seed\}/$1}"
  else weight_path "swin_base_colour_aug_cwce_seed$1.pth"; fi
}

if [[ ! -d "${WBCB_P1_CROPS}" ]]; then echo "[ERR] external set not found: ${WBCB_P1_CROPS}" >&2; exit 2; fi
FAILED=0
for size in ${SIZES}; do
  for seed in ${SEEDS}; do
    ckpt="$(ckpt_for "${seed}")"
    out="${OUT_BASE}/img${size}/seed_${seed}"
    if [[ ! -f "${ckpt}" ]]; then echo "[FAIL] seed ${seed}: checkpoint not found (${ckpt})" >&2; FAILED=$((FAILED + 1)); continue; fi
    if [[ -f "${out}/metrics.csv" && "${FORCE:-0}" != "1" ]]; then
      echo "[SKIP] ${size} px seed ${seed}: ${out}/metrics.csv exists"; continue
    fi
    mkdir -p "${out}"
    echo "[EVAL] resolution ${size} seed ${seed} -> ${out}"
    "${PYTHON}" -u "${EVAL_PY}" \
      --test-root "${WBCB_P1_CROPS}" \
      --model-path "${ckpt}" \
      --model-name swin_base_patch4_window7_224 \
      --img-size "${size}" \
      --tta hflip \
      --device "${DEVICE}" \
      --output-dir "${out}" 2>&1 | tee "${out}/run.log"
    rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then echo "[FAIL] ${size} px seed ${seed} (rc=${rc})" >&2; FAILED=$((FAILED + 1)); fi
  done
done

echo "[SUMMARY] macro-F1 per resolution, mean +- population SD over seeds"
"${PYTHON}" - "${OUT_BASE}" ${SIZES} <<'PY' || FAILED=$((FAILED + 1))
import csv, sys
from pathlib import Path
import numpy as np
base, sizes = Path(sys.argv[1]), sys.argv[2:]
for size in sizes:
    vals = []
    for m in sorted((base / f"img{size}").glob("seed_*/metrics.csv")):
        with open(m, newline="") as f:
            vals.append((m.parent.name, float(next(csv.DictReader(f))["macro_f1"])))
    if vals:
        a = np.array([v for _, v in vals])
        per = ", ".join(f"{s} {v:.4f}" for s, v in vals)
        print(f"  {size:>4} px  macro-F1 {a.mean():.3f} +- {a.std(ddof=0):.3f}  ({per})")
PY
echo "[RESOLUTION SCALING] done (${FAILED} failure(s))"
[[ ${FAILED} -eq 0 ]]
