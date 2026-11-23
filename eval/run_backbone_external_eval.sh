#!/usr/bin/env bash
# External evaluation of the backbone comparison on the WBCBench2026 Phase-1 pre-cropped
# set (n = 8,061 after mapping): main-text Table 3 (convolutional baselines and BPaCo
# block), Section 3.4, Supplementary Sections S7, S9 and S10. Inference only.
#
# Every checkpoint goes through the common evaluator (eval/evaluate_classifier.py):
# square padding, resize 256, centre crop 224, horizontal-flip TTA, nine-class argmax
# mapped to the seven evaluation classes (--use-mappings, default merge/ignore rules).
#
# Groups (RUN_GROUPS, default "cnn bpaco"):
#   cnn              class-weighted-CE baselines retrained on the common corpus, seeds 42 43 44:
#                    ResNet50 0.718, VGG16-BN 0.813, EfficientNet-B0 0.796 (external macro-F1,
#                    seed mean). Not in the weights deposit: train/scripts/run_cnn_baselines.sh
#                    writes <CNN_ROOT>/<timm name>/seed_<seed>/best_macro_f1.pth.
#   bpaco            BPaCo block, seeds 42 62 82: swin_base_bpaco_seed*.pth 0.819,
#                    resnet50_bpaco_adamw_seed*.pth 0.762, resnet50_bpaco_sgd_seed*.pth 0.815
#                    (released weights in MODEL_DIR, pipeline/models/ or models/; if absent, the
#                    run folders swin_base_bpaco, resnet50_bpaco_adamw, resnet50_bpaco_sgd under
#                    MODEL_ROOT).
#   resnet50_configs ResNet50 under the proposed colour configuration with AdamW (0.694) and
#                    with SGD (0.727), and under the Swin-B CW-CE baseline configuration (0.731);
#                    seeds 42 62 82; run folders resnet50_color_aug_cwce_adamw,
#                    resnet50_color_aug_cwce_sgd, resnet50_cwce_default_aug under MODEL_ROOT.
# The proposed Swin-B row (0.863) is run_augmentation_recipe_eval.sh (colour_aug_proposed),
# the fine-tuned DinoBloom-B row run_dinobloom_external_eval.sh, the frozen probe
# frozen_probe.py.
#
# Usage: bash eval/run_backbone_external_eval.sh [-h|--help]
#
# Environment (default in brackets):
#   DATA_ROOT     prepared data                                         [./data]
#   WBCB_P1_CROPS pre-cropped Phase-1 class folders
#                 [$DATA_ROOT/wbc-bench-2026_phase1_cropped_formatted]
#   MODEL_ROOT    run folders of train/scripts/ (its OUT_ROOT)          [$OUT_ROOT]
#   CNN_ROOT      common-corpus CNN checkpoints                         [$MODEL_ROOT/cnn_cwce]
#   MODEL_DIR     released weight files                                 [pipeline/models or models]
#   OUT_ROOT      outputs, must be outside the repository               [./outputs]
#   RUN_GROUPS    groups to run                                         [cnn bpaco]
#   PYTHON        interpreter of the journal_env environment            [python]
#   DEVICE        cuda or cpu                                           [cuda]
#   FORCE         1 = re-run folders that already hold predictions.csv  [0]
# Outputs: $OUT_ROOT/backbone_external/<run>/seed_<seed>/{predictions.csv, path_true_pred.csv,
# classification_report.txt, confusion_*, montage.png, run.log}; seed summary in
# $OUT_ROOT/stats/backbone_external/seed_summary.csv. Per-image outputs contain WBCBench2026
# image IDs: keep them on your own machine.
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
CNN_ROOT="${CNN_ROOT:-${MODEL_ROOT}/cnn_cwce}"
RUN_GROUPS="${RUN_GROUPS:-cnn bpaco}"
EVAL_PY="${REPO}/eval/evaluate_classifier.py"
OUT_BASE="${OUT_ROOT}/backbone_external"

weight_path() {  # released weight file: $MODEL_DIR, then pipeline/models/, then models/
  local d
  for d in ${MODEL_DIR:+"${MODEL_DIR}"} "${REPO}/pipeline/models" "${REPO}/models"; do
    if [[ -f "${d}/$1" ]]; then echo "${d}/$1"; return 0; fi
  done
  echo "${MODEL_DIR:-${REPO}/models}/$1"
}
resolve_ckpt() {  # $1 pattern with {seed} ("@weights/<file>" = released weight file), $2 seed,
                  # $3 optional fallback pattern used when the first file does not exist
  local p="${1//\{seed\}/$2}"
  if [[ "${p}" == @weights/* ]]; then p="$(weight_path "${p#@weights/}")"; fi
  if [[ ! -f "${p}" && -n "${3:-}" ]]; then p="${3//\{seed\}/$2}"; fi
  echo "${p}"
}

# run name | timm model | checkpoint pattern | seeds | fallback pattern
RUNS_cnn=(
  "resnet50|resnet50|${CNN_ROOT}/resnet50/seed_{seed}/best_macro_f1.pth|42 43 44"
  "vgg16_bn|vgg16_bn|${CNN_ROOT}/vgg16_bn/seed_{seed}/best_macro_f1.pth|42 43 44"
  "tf_efficientnet_b0|tf_efficientnet_b0|${CNN_ROOT}/tf_efficientnet_b0/seed_{seed}/best_macro_f1.pth|42 43 44"
)
RUNS_bpaco=(
  "swin_base_bpaco|swin_base_patch4_window7_224|@weights/swin_base_bpaco_seed{seed}.pth|42 62 82|${MODEL_ROOT}/swin_base_bpaco/seed_{seed}/best_macro_f1.pth"
  "resnet50_bpaco_adamw|resnet50|@weights/resnet50_bpaco_adamw_seed{seed}.pth|42 62 82|${MODEL_ROOT}/resnet50_bpaco_adamw/seed_{seed}/best_macro_f1.pth"
  "resnet50_bpaco_sgd|resnet50|@weights/resnet50_bpaco_sgd_seed{seed}.pth|42 62 82|${MODEL_ROOT}/resnet50_bpaco_sgd/seed_{seed}/best_macro_f1.pth"
)
RUNS_resnet50_configs=(
  "resnet50_color_aug_cwce_adamw|resnet50|${MODEL_ROOT}/resnet50_color_aug_cwce_adamw/seed_{seed}/best_macro_f1.pth|42 62 82"
  "resnet50_color_aug_cwce_sgd|resnet50|${MODEL_ROOT}/resnet50_color_aug_cwce_sgd/seed_{seed}/best_macro_f1.pth|42 62 82"
  "resnet50_cwce_default_aug|resnet50|${MODEL_ROOT}/resnet50_cwce_default_aug/seed_{seed}/best_macro_f1.pth|42 62 82"
)

if [[ ! -d "${WBCB_P1_CROPS}" ]]; then echo "[ERR] external set not found: ${WBCB_P1_CROPS}" >&2; exit 2; fi
FAILED=0
SUMMARY_ARGS=()
for group in ${RUN_GROUPS}; do
  case "${group}" in cnn|bpaco|resnet50_configs) ;; *) echo "[ERR] unknown group: ${group}" >&2; exit 2 ;; esac
  declare -n runs="RUNS_${group}"
  for entry in "${runs[@]}"; do
    IFS='|' read -r run model pattern seeds fallback <<< "${entry}"
    files=()
    for seed in ${seeds}; do
      ckpt="$(resolve_ckpt "${pattern}" "${seed}" "${fallback}")"
      out="${OUT_BASE}/${run}/seed_${seed}"
      if [[ ! -f "${ckpt}" ]]; then echo "[FAIL] ${run} seed ${seed}: checkpoint not found (${ckpt})" >&2; FAILED=$((FAILED + 1)); continue; fi
      if [[ -f "${out}/predictions.csv" && "${FORCE:-0}" != "1" ]]; then
        echo "[SKIP] ${run} seed ${seed}: ${out}/predictions.csv exists"
      else
        mkdir -p "${out}"
        echo "[EVAL] ${run} seed ${seed} -> ${out}"
        "${PYTHON}" -u "${EVAL_PY}" \
          --test-root "${WBCB_P1_CROPS}" \
          --model-path "${ckpt}" \
          --model-name "${model}" \
          --img-size 224 \
          --batch-size 32 \
          --num-workers 4 \
          --device "${DEVICE}" \
          --use-mappings \
          --tta hflip \
          --output-dir "${out}" \
          --top-k 4 \
          --dump-preds "${out}/path_true_pred.csv" 2>&1 | tee "${out}/run.log"
        rc=${PIPESTATUS[0]}
        if [[ ${rc} -ne 0 ]]; then echo "[FAIL] ${run} seed ${seed} (rc=${rc})" >&2; FAILED=$((FAILED + 1)); continue; fi
      fi
      files+=("${out}/predictions.csv")
    done
    if [[ ${#files[@]} -gt 0 ]]; then SUMMARY_ARGS+=(--run "${run}" "${files[@]}"); fi
  done
  unset -n runs
done

if [[ ${#SUMMARY_ARGS[@]} -gt 0 ]]; then
  echo "[SUMMARY] seed mean +- population SD (stats/seed_summary.py)"
  "${PYTHON}" "${REPO}/stats/seed_summary.py" "${SUMMARY_ARGS[@]}" --per-class \
    --out-dir "${OUT_ROOT}/stats/backbone_external" || FAILED=$((FAILED + 1))
fi
echo "[BACKBONE EXTERNAL EVAL] done (${FAILED} failure(s))"
[[ ${FAILED} -eq 0 ]]
