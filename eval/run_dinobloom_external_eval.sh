#!/usr/bin/env bash
# External evaluation of the fine-tuned DinoBloom-B seeds on the WBCBench2026 Phase-1
# pre-cropped set (main-text Table 3, row "DinoBloom-B (color aug; fine-tuned)";
# Supplementary Section S7). Inference only.
#
# Same protocol as the proposed Swin model: eval/evaluate_dinobloom.py is the common
# evaluator with the DinoBloom hub id as model name (square padding, resize 256, centre
# crop 224, --use-mappings nine -> seven classes, horizontal-flip TTA), so the external
# macro-F1 is directly comparable to the proposed model's 0.863 on the same n = 8,061.
# Paper: 0.822 +- 0.047 (per seed 0.835 / 0.873 / 0.759 for seeds 42 / 62 / 82).
#
# The DinoBloom configuration is read from the Hugging Face hub (or the local cache); set
# HF_HUB_OFFLINE=1 once it is cached to run without network access.
#
# Usage: bash eval/run_dinobloom_external_eval.sh [-h|--help]
#
# Environment (default in brackets):
#   DATA_ROOT      prepared data                                        [./data]
#   WBCB_P1_CROPS  pre-cropped Phase-1 class folders
#                  [$DATA_ROOT/wbc-bench-2026_phase1_cropped_formatted]
#   MODEL_ROOT     run folders of train/scripts/ (its OUT_ROOT)         [$OUT_ROOT]
#   DINO_ROOT      fine-tuned DinoBloom-B checkpoints, <DINO_ROOT>/seed_<seed>/best_macro_f1.pth
#                  (train/scripts/run_dinobloom_finetune.sh)  [$MODEL_ROOT/dinobloom_b_color_aug_cwce]
#   DINO_MODEL_NAME  timm hub id (default: the one in eval/evaluate_dinobloom.py)
#   SEEDS          seeds                                                [42 62 82]
#   OUT_ROOT       outputs, must be outside the repository              [./outputs]
#   PYTHON         interpreter of the journal_env environment           [python]
#   DEVICE         cuda or cpu                                          [cuda]
#   FORCE          1 = re-run folders that already hold predictions.csv [0]
# Outputs: $OUT_ROOT/dinobloom_external/seed_<seed>/{predictions.csv, path_true_pred.csv,
# classification_report.txt, confusion_*, montage.png, run.log}; seed summary in
# $OUT_ROOT/stats/dinobloom_external/seed_summary.csv. Per-image outputs contain
# WBCBench2026 image IDs: keep them on your own machine.
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
MODEL_ROOT="${MODEL_ROOT:-${OUT_ROOT}}"
DINO_ROOT="${DINO_ROOT:-${MODEL_ROOT}/dinobloom_b_color_aug_cwce}"
SEEDS="${SEEDS:-42 62 82}"
EVAL_PY="${REPO}/eval/evaluate_dinobloom.py"
OUT_BASE="${OUT_ROOT}/dinobloom_external"
MODEL_ARGS=()
if [[ -n "${DINO_MODEL_NAME:-}" ]]; then MODEL_ARGS=(--model-name "${DINO_MODEL_NAME}"); fi

if [[ ! -d "${WBCB_P1_CROPS}" ]]; then echo "[ERR] external set not found: ${WBCB_P1_CROPS}" >&2; exit 2; fi
FAILED=0
files=()
for seed in ${SEEDS}; do
  ckpt="${DINO_ROOT}/seed_${seed}/best_macro_f1.pth"
  out="${OUT_BASE}/seed_${seed}"
  if [[ ! -f "${ckpt}" ]]; then echo "[FAIL] seed ${seed}: checkpoint not found (${ckpt})" >&2; FAILED=$((FAILED + 1)); continue; fi
  if [[ -f "${out}/predictions.csv" && "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP] seed ${seed}: ${out}/predictions.csv exists"
  else
    mkdir -p "${out}"
    echo "[EVAL] DinoBloom-B external seed ${seed} -> ${out}"
    "${PYTHON}" -u "${EVAL_PY}" \
      --test-root "${WBCB_P1_CROPS}" \
      --model-path "${ckpt}" \
      ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} \
      --img-size 224 \
      --batch-size 64 \
      --num-workers 8 \
      --device "${DEVICE}" \
      --use-mappings \
      --tta hflip \
      --output-dir "${out}" \
      --top-k 4 \
      --dump-preds "${out}/path_true_pred.csv" 2>&1 | tee "${out}/run.log"
    rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then echo "[FAIL] seed ${seed} (rc=${rc})" >&2; FAILED=$((FAILED + 1)); continue; fi
  fi
  files+=("${out}/predictions.csv")
done

if [[ ${#files[@]} -gt 0 ]]; then
  echo "[SUMMARY] seed mean +- population SD (stats/seed_summary.py)"
  "${PYTHON}" "${REPO}/stats/seed_summary.py" --run dinobloom_b_finetuned "${files[@]}" \
    --out-dir "${OUT_ROOT}/stats/dinobloom_external" || FAILED=$((FAILED + 1))
fi
echo "[DINOBLOOM EXTERNAL EVAL] done (${FAILED} failure(s))"
[[ ${FAILED} -eq 0 ]]
