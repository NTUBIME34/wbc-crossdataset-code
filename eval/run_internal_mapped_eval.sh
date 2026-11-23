#!/usr/bin/env bash
# Internal held-out test split scored in the seven-class evaluation space with the common
# evaluator (Fig. 3 internal panel and Section 3.8; Supplementary Table S11 threshold
# tuning). Inference only.
#
# The internal test split of the nine-class corpus (n = 12,803) is passed through
# eval/evaluate_classifier.py with exactly the external settings: square padding, resize
# 256, centre crop 224, horizontal-flip TTA, --use-mappings (nine -> seven classes).
# Recipes:
#   colour_aug_proposed  the proposed model, released weights swin_base_colour_aug_cwce_seed*.pth
#   class_weighted_ce    the Swin-B CW-CE baseline of Supplementary Table S8
# Downstream (three-seed probability ensemble of colour_aug_proposed):
#   figures/make_confusion_matrix.py  internal confusion matrix of Fig. 3 (recall 0.90-0.98)
#   stats/threshold_analysis.py       S11 one-vs-rest thresholds tuned here and applied to the
#                                     external ensemble (BLAST t* 0.500, PROMY t* 0.480)
# The internal macro-F1 of main-text Table 3 is NOT from this runner: it is the nine-class
# score in each trainer's test_results.json (no TTA, each model's own preprocessing).
#
# Usage: bash eval/run_internal_mapped_eval.sh [-h|--help]
#
# Environment (default in brackets):
#   DATA_ROOT          prepared data                                    [./data]
#   CORPUS_ROOT        nine-class corpus with train/ val/ test/  [$DATA_ROOT/Mixed_data_mix_mono_9_split_copy]
#   INTERNAL_TEST_ROOT internal test split (nine class folders)         [$CORPUS_ROOT/test]
#   MODEL_ROOT         run folders of train/scripts/ (its OUT_ROOT)     [$OUT_ROOT]
#   CKPT_COLOUR_AUG    checkpoint pattern, {seed} is replaced
#                      [released file; else $MODEL_ROOT/swin_base_color_aug_cwce/seed_{seed}/best_macro_f1.pth]
#   CKPT_CW_CE         [$MODEL_ROOT/swin_base_cwce_default_aug/seed_{seed}/best_macro_f1.pth]
#   MODEL_DIR          released weight files                            [pipeline/models or models]
#   RECIPES            recipes to run                     [colour_aug_proposed class_weighted_ce]
#   SEEDS              seeds                                            [42 62 82]
#   OUT_ROOT           outputs, must be outside the repository          [./outputs]
#   PYTHON             interpreter of the journal_env environment       [python]
#   DEVICE             cuda or cpu                                      [cuda]
#   FORCE              1 = re-run folders that already hold predictions.csv [0]
# Outputs: $OUT_ROOT/internal_mapped/<recipe>/seed_<seed>/{predictions.csv,
# classification_report.txt, confusion_*, montage.png, run.log}.
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
CORPUS_ROOT="${CORPUS_ROOT:-${DATA_ROOT}/Mixed_data_mix_mono_9_split_copy}"
INTERNAL_TEST_ROOT="${INTERNAL_TEST_ROOT:-${CORPUS_ROOT%/}/test}"
MODEL_ROOT="${MODEL_ROOT:-${OUT_ROOT}}"
SEEDS="${SEEDS:-42 62 82}"
RECIPES="${RECIPES:-colour_aug_proposed class_weighted_ce}"
# Default checkpoint patterns ({seed} is replaced per run).
DEFAULT_COLOUR_AUG="${MODEL_ROOT}/swin_base_color_aug_cwce/seed_{seed}/best_macro_f1.pth"
DEFAULT_CW_CE="${MODEL_ROOT}/swin_base_cwce_default_aug/seed_{seed}/best_macro_f1.pth"
EVAL_PY="${REPO}/eval/evaluate_classifier.py"
OUT_BASE="${OUT_ROOT}/internal_mapped"

weight_path() {  # released weight file: $MODEL_DIR, then pipeline/models/, then models/
  local d
  for d in ${MODEL_DIR:+"${MODEL_DIR}"} "${REPO}/pipeline/models" "${REPO}/models"; do
    if [[ -f "${d}/$1" ]]; then echo "${d}/$1"; return 0; fi
  done
  echo "${MODEL_DIR:-${REPO}/models}/$1"
}
ckpt_for() {  # $1 recipe, $2 seed
  local p
  case "$1" in
    colour_aug_proposed)
      if [[ -n "${CKPT_COLOUR_AUG:-}" ]]; then p="${CKPT_COLOUR_AUG}"
      else
        p="$(weight_path "swin_base_colour_aug_cwce_seed$2.pth")"
        [[ -f "${p}" ]] || p="${DEFAULT_COLOUR_AUG}"
      fi ;;
    class_weighted_ce)
      p="${CKPT_CW_CE:-${DEFAULT_CW_CE}}" ;;
    *) echo "[ERR] unknown recipe: $1" >&2; return 1 ;;
  esac
  echo "${p//\{seed\}/$2}"
}

if [[ ! -d "${INTERNAL_TEST_ROOT}" ]]; then echo "[ERR] internal test split not found: ${INTERNAL_TEST_ROOT}" >&2; exit 2; fi
FAILED=0
for recipe in ${RECIPES}; do
  for seed in ${SEEDS}; do
    ckpt="$(ckpt_for "${recipe}" "${seed}")" || exit 2
    out="${OUT_BASE}/${recipe}/seed_${seed}"
    if [[ ! -f "${ckpt}" ]]; then echo "[FAIL] ${recipe} seed ${seed}: checkpoint not found (${ckpt})" >&2; FAILED=$((FAILED + 1)); continue; fi
    if [[ -f "${out}/predictions.csv" && "${FORCE:-0}" != "1" ]]; then
      echo "[SKIP] ${recipe} seed ${seed}: ${out}/predictions.csv exists"; continue
    fi
    mkdir -p "${out}"
    echo "[EVAL] internal mapped ${recipe} seed ${seed} -> ${out}"
    "${PYTHON}" -u "${EVAL_PY}" \
      --test-root "${INTERNAL_TEST_ROOT}" \
      --model-path "${ckpt}" \
      --model-name swin_base_patch4_window7_224 \
      --img-size 224 \
      --batch-size 64 \
      --num-workers 8 \
      --device "${DEVICE}" \
      --use-mappings \
      --tta hflip \
      --output-dir "${out}" \
      --top-k 4 \
      --highlight-class Blast \
      --highlight-class Promyelocyte 2>&1 | tee "${out}/run.log"
    rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then echo "[FAIL] ${recipe} seed ${seed} (rc=${rc})" >&2; FAILED=$((FAILED + 1)); fi
  done
done
echo "[INTERNAL MAPPED EVAL] done (${FAILED} failure(s))"
[[ ${FAILED} -eq 0 ]]
