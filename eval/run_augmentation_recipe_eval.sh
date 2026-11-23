#!/usr/bin/env bash
# Augmentation / imbalance recipe comparison on the external benchmark (Supplementary
# Table S8; main-text Section 3.6; the proposed row of main-text Table 3). Inference only.
#
# Three Swin-B training recipes are evaluated on the WBCBench2026 Phase-1 pre-cropped set
# (n = 8,061) under the SAME protocol: eval/evaluate_classifier.py, --use-mappings (nine
# classes -> seven), horizontal-flip TTA, 224 px, with the arguments of the original
# prediction exports. Only the checkpoint changes:
#   colour_aug_proposed  colour augmentation + CW-CE (the proposed model)
#                        released weights swin_base_colour_aug_cwce_seed{42,62,82}.pth
#                        (MODEL_DIR, pipeline/models/ or models/); per-seed 0.886/0.878/0.824,
#                        seed mean 0.863 +- 0.028, ensemble 0.893 [train/scripts/run_proposed.sh]
#   class_weighted_ce    CW-CE with the default augmentation (Swin-B baseline);
#                        seed mean 0.856 +- 0.009 (Section 3.4), ensemble 0.881
#                        [train/scripts/run_swin_cwce_default_aug.sh]
#   mixup_bpaco_crt      mixup + BPaCo + classifier re-training; ensemble 0.864. cRT re-fits
#                        only the final linear head, so this recipe uses crt_best_macro_f1.pth
#                        [train/scripts/run_swin_mixup_bpaco_crt.sh].
# The S8 three-seed probability ensembles, the 1,000-replicate recipe-selection bootstrap
# (Section 3.6) and the S9 intervals are computed from these predictions.csv files by
# stats/recipe_ensembles.py and stats/bootstrap_ensemble_ci.py.
# Supplementary S17 ("Development use of the external labels") uses the same runner:
#   RECIPES="aug_stain_only aug_geometry_only aug_conservative aug_blur_noise aug_stain_blur_noise"
#     evaluates the policy sweep of train/scripts/run_aug_policy_sweep.sh
#     ($MODEL_ROOT/swin_base_aug_<policy>/seed_<seed>/best_macro_f1.pth); seed means range from
#     0.773 (stain-blur-noise) to 0.863 (color-only, = colour_aug_proposed), baseline 0.856;
#   TTA=none RECIPES=colour_aug_proposed gives the no-flip comparison (0.860 versus 0.863).
#
# Usage: bash eval/run_augmentation_recipe_eval.sh [-h|--help]
#
# Environment (default in brackets):
#   DATA_ROOT        prepared data                                      [./data]
#   WBCB_P1_CROPS    pre-cropped Phase-1 class folders
#                    [$DATA_ROOT/wbc-bench-2026_phase1_cropped_formatted]
#   MODEL_ROOT       run folders of train/scripts/ (its OUT_ROOT)       [$OUT_ROOT]
#   CKPT_COLOUR_AUG  checkpoint pattern, {seed} is replaced
#                    [released file; else $MODEL_ROOT/swin_base_color_aug_cwce/seed_{seed}/best_macro_f1.pth]
#   CKPT_CW_CE       [$MODEL_ROOT/swin_base_cwce_default_aug/seed_{seed}/best_macro_f1.pth]
#   CKPT_MIXUP_BPACO_CRT
#                    [$MODEL_ROOT/swin_base_mixup_bpaco_crt/seed_{seed}/crt_best_macro_f1.pth]
#   MODEL_DIR        released weight files                              [pipeline/models or models]
#   RECIPES          recipes to run           [colour_aug_proposed class_weighted_ce mixup_bpaco_crt]
#                    (also aug_<policy> for the S17 policy sweep, see above)
#   TTA              hflip, or none for the S17 no-flip comparison      [hflip]
#   BATCH_SIZE       inference batch size [64; the reported S17 policy sweep used 32]
#   SEEDS            seeds                                              [42 62 82]
#   OUT_ROOT         outputs, must be outside the repository            [./outputs]
#   PYTHON           interpreter of the journal_env environment         [python]
#   DEVICE           cuda or cpu                                        [cuda]
#   FORCE            1 = re-run folders that already hold predictions.csv [0]
# Outputs: $OUT_ROOT/augmentation_recipes/<recipe>/seed_<seed>/{predictions.csv,
# path_true_pred.csv, classification_report.txt, confusion_*, montage.png, run.log}
# (<recipe>_no_tta/ with TTA=none); seed summary in
# $OUT_ROOT/stats/augmentation_recipes[_no_tta]/seed_summary.csv.
# Per-image outputs contain WBCBench2026 image IDs: keep them on your own machine.
set -uo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
case "${1:-}" in -h|--help) usage; exit 0 ;; "") ;; *) usage >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERR] BATCH_SIZE must be a positive integer" >&2; exit 2
fi
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
MODEL_ROOT="${MODEL_ROOT:-${OUT_ROOT}}"
SEEDS="${SEEDS:-42 62 82}"
RECIPES="${RECIPES:-colour_aug_proposed class_weighted_ce mixup_bpaco_crt}"
TTA="${TTA:-hflip}"
case "${TTA}" in hflip) TTA_SUFFIX="" ;; none) TTA_SUFFIX="_no_tta" ;; *) echo "[ERR] TTA must be hflip or none" >&2; exit 2 ;; esac
# Default checkpoint patterns ({seed} is replaced per run).
DEFAULT_COLOUR_AUG="${MODEL_ROOT}/swin_base_color_aug_cwce/seed_{seed}/best_macro_f1.pth"
DEFAULT_CW_CE="${MODEL_ROOT}/swin_base_cwce_default_aug/seed_{seed}/best_macro_f1.pth"
DEFAULT_MIXUP_BPACO_CRT="${MODEL_ROOT}/swin_base_mixup_bpaco_crt/seed_{seed}/crt_best_macro_f1.pth"
EVAL_PY="${REPO}/eval/evaluate_classifier.py"
OUT_BASE="${OUT_ROOT}/augmentation_recipes"

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
    mixup_bpaco_crt)
      p="${CKPT_MIXUP_BPACO_CRT:-${DEFAULT_MIXUP_BPACO_CRT}}" ;;
    aug_*)
      p="${MODEL_ROOT}/swin_base_aug_${1#aug_}/seed_{seed}/best_macro_f1.pth" ;;
    *) echo "[ERR] unknown recipe: $1" >&2; return 1 ;;
  esac
  echo "${p//\{seed\}/$2}"
}

if [[ ! -d "${WBCB_P1_CROPS}" ]]; then echo "[ERR] external set not found: ${WBCB_P1_CROPS}" >&2; exit 2; fi
FAILED=0
SUMMARY_ARGS=()
for recipe in ${RECIPES}; do
  files=()
  for seed in ${SEEDS}; do
    ckpt="$(ckpt_for "${recipe}" "${seed}")" || exit 2
    out="${OUT_BASE}/${recipe}${TTA_SUFFIX}/seed_${seed}"
    if [[ ! -f "${ckpt}" ]]; then echo "[FAIL] ${recipe} seed ${seed}: checkpoint not found (${ckpt})" >&2; FAILED=$((FAILED + 1)); continue; fi
    if [[ -f "${out}/predictions.csv" && "${FORCE:-0}" != "1" ]]; then
      echo "[SKIP] ${recipe} seed ${seed}: ${out}/predictions.csv exists"
    else
      mkdir -p "${out}"
      echo "[EVAL] ${recipe} seed ${seed} -> ${out}"
      "${PYTHON}" -u "${EVAL_PY}" \
        --test-root "${WBCB_P1_CROPS}" \
        --model-path "${ckpt}" \
        --model-name swin_base_patch4_window7_224 \
        --img-size 224 \
        --batch-size "${BATCH_SIZE}" \
        --num-workers 8 \
        --device "${DEVICE}" \
        --use-mappings \
        --tta "${TTA}" \
        --output-dir "${out}" \
        --top-k 4 \
        --highlight-class Blast \
        --highlight-class Promyelocyte \
        --dump-preds "${out}/path_true_pred.csv" 2>&1 | tee "${out}/run.log"
      rc=${PIPESTATUS[0]}
      if [[ ${rc} -ne 0 ]]; then echo "[FAIL] ${recipe} seed ${seed} (rc=${rc})" >&2; FAILED=$((FAILED + 1)); continue; fi
    fi
    files+=("${out}/predictions.csv")
  done
  if [[ ${#files[@]} -gt 0 ]]; then SUMMARY_ARGS+=(--run "${recipe}${TTA_SUFFIX}" "${files[@]}"); fi
done

if [[ ${#SUMMARY_ARGS[@]} -gt 0 ]]; then
  echo "[SUMMARY] per-seed mean +- population SD (stats/seed_summary.py)"
  "${PYTHON}" "${REPO}/stats/seed_summary.py" "${SUMMARY_ARGS[@]}" \
    --out-dir "${OUT_ROOT}/stats/augmentation_recipes${TTA_SUFFIX}" || FAILED=$((FAILED + 1))
fi
echo "[AUGMENTATION RECIPE EVAL] done (${FAILED} failure(s)). Build the S8 three-seed probability"
echo "ensembles and the recipe-selection bootstrap with stats/recipe_ensembles.py."
[[ ${FAILED} -eq 0 ]]
