#!/usr/bin/env bash
# Deployable pipeline (field -> YOLO -> Swin) on the clean Phase-1 and the partly degraded
# Phase-2 evaluation split of WBCBench2026 (main-text Table 2a and the primary endpoint;
# Supplementary Section S17 "Unified re-evaluation"; optional Supplementary Section S12).
# Inference only.
#
# Per split: (1) pipeline/detect_yolo11.py on the full fields (512 px, conf 0.30, NMS IoU
# 0.70, highest-confidence box with a 10% margin, no fallback); (2) the crops are scored by
# the common evaluator (eval/evaluate_classifier.py: square padding, resize 256, centre crop
# 224, horizontal-flip TTA, --use-mappings) for the proposed seeds 42/62/82; (3)
# stats/coverage_aware_score.py counts every labelled field without a detection as an
# error (never-matching prediction, not averaged as an eighth class) and also reports the
# detected-cells-only view.
# Paper (mean +- population SD; balanced accuracy is a seed mean):
#   Phase-1  n 8,080  coverage 8,079/8,080  bal. acc 0.882  macro-F1 0.879 +- 0.006
#   Phase-2  n 5,196  coverage 4,848/5,196  bal. acc 0.795  macro-F1 0.769 +- 0.002
#   Phase-2, detected cells only  n 4,848   bal. acc 0.833  macro-F1 0.788 +- 0.002
# The Phase-1 computation is route B of run_routes_ablation.sh. With RUN_ABSTENTION=1 the
# Phase-2 detections also feed eval/detector_abstention_quality_test.py (S12: accuracy
# 0.727 on the 348 missed cells versus 0.862 on a class-matched detected control).
# Re-running the detector on other hardware can move a few boxes, so individual
# predictions may differ slightly from the paper's runs.
# Every requested classifier checkpoint is required. A missing seed is an error,
# not a successful evaluation of a smaller ensemble or an empty run.
#
# Usage: bash eval/run_full_pipeline_eval.sh [-h|--help]
#
# Environment (default in brackets):
#   DATA_ROOT      prepared data                                        [./data]
#   WBCB_P1_FIELDS Phase-1 full fields, class folders  [$DATA_ROOT/wbc-bench-2026_phase1_nocrop_formatted]
#   WBCB_P2_FIELDS Phase-2 evaluation fields, class folders
#                  [$DATA_ROOT/wbc-bench-2026_phase2_eval_nocrop_formatted]
#   YOLO_MODEL     detector weights   [yolo11_wbc_detection_best.pt in MODEL_DIR, pipeline/models or models]
#   MODEL_DIR      released weight files                                [pipeline/models or models]
#   CKPT_PATTERN   classifier pattern, {seed} replaced [released swin_base_colour_aug_cwce_seed{seed}.pth]
#   SPLITS         splits to run                                        [phase1 phase2]
#   SEEDS          seeds                                                [42 62 82]
#   RUN_ABSTENTION 1 = run the S12 missed-versus-detected comparison    [1]
#   OUT_ROOT       outputs, must be outside the repository              [./outputs]
#   PYTHON         interpreter of the journal_env environment           [python]
#   DEVICE         cuda or cpu (classifier)                             [cuda]
#   DETECT_DEVICE  detector device: auto, cpu, cuda or cuda:N      [auto; cpu if DEVICE=cpu]
#   FORCE          1 = re-run folders that already hold results         [0]
# Outputs: $OUT_ROOT/full_pipeline/<split>/detection/{detections.csv, crops/},
# $OUT_ROOT/full_pipeline/<split>/seed_<seed>/{predictions.csv, path_true_pred.csv, ...},
# scores in $OUT_ROOT/stats/coverage_aware_full_pipeline_<split>.json and
# $OUT_ROOT/full_pipeline/phase2/detector_abstention_quality_test.csv. Per-image outputs and
# crops are WBCBench2026 material: keep them on your own machine.
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
WBCB_P1_FIELDS="${WBCB_P1_FIELDS:-${DATA_ROOT}/wbc-bench-2026_phase1_nocrop_formatted}"
WBCB_P2_FIELDS="${WBCB_P2_FIELDS:-${DATA_ROOT}/wbc-bench-2026_phase2_eval_nocrop_formatted}"
SPLITS="${SPLITS:-phase1 phase2}"
SEEDS="${SEEDS:-42 62 82}"
RUN_ABSTENTION="${RUN_ABSTENTION:-1}"
EVAL_PY="${REPO}/eval/evaluate_classifier.py"
DETECT_PY="${REPO}/pipeline/detect_yolo11.py"
OUT_BASE="${OUT_ROOT}/full_pipeline"

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
YOLO_MODEL="${YOLO_MODEL:-$(weight_path yolo11_wbc_detection_best.pt)}"
if [[ "${DEVICE}" == "cpu" ]]; then DETECT_DEVICE="${DETECT_DEVICE:-cpu}"; else DETECT_DEVICE="${DETECT_DEVICE:-auto}"; fi
FAILED=0

# Validate the requested run before detection, including when cached CSV files
# exist. Previously missing classifier weights were only warnings, so an empty
# or partial seed set could exit successfully.
read -r -a REQUESTED_SEEDS <<< "${SEEDS}"
if [[ ${#REQUESTED_SEEDS[@]} -eq 0 ]]; then
  echo "[ERR] SEEDS must contain at least one seed" >&2; exit 2
fi
declare -A SEEN_SEEDS=()
for seed in "${REQUESTED_SEEDS[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ || -n "${SEEN_SEEDS[${seed}]:-}" ]]; then
    echo "[ERR] SEEDS must contain distinct non-negative integers" >&2; exit 2
  fi
  SEEN_SEEDS["${seed}"]=1
  ckpt="$(ckpt_for "${seed}")"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERR] requested seed ${seed}: checkpoint not found (${ckpt})" >&2; exit 2
  fi
done
if [[ ! -f "${YOLO_MODEL}" ]]; then
  echo "[ERR] detector weights not found: ${YOLO_MODEL}" >&2; exit 2
fi
read -r -a REQUESTED_SPLITS <<< "${SPLITS}"
if [[ ${#REQUESTED_SPLITS[@]} -eq 0 ]]; then
  echo "[ERR] SPLITS must contain phase1 and/or phase2" >&2; exit 2
fi
for split in "${REQUESTED_SPLITS[@]}"; do
  case "${split}" in phase1|phase2) ;; *) echo "[ERR] unknown split: ${split}" >&2; exit 2 ;; esac
  if [[ "${split}" == "phase2" && "${RUN_ABSTENTION}" == "1" ]]; then
    for seed in 42 62 82; do
      ckpt="$(ckpt_for "${seed}")"
      if [[ ! -f "${ckpt}" ]]; then
        echo "[ERR] S12 requires seed ${seed}: checkpoint not found (${ckpt})" >&2; exit 2
      fi
    done
  fi
done

for split in ${SPLITS}; do
  case "${split}" in
    phase1) fields="${WBCB_P1_FIELDS}"; name="Phase-1" ;;
    phase2) fields="${WBCB_P2_FIELDS}"; name="Phase-2" ;;
    *) echo "[ERR] unknown split: ${split} (phase1 or phase2)" >&2; exit 2 ;;
  esac
  det="${OUT_BASE}/${split}/detection"
  # (1) detection
  if [[ -f "${det}/detections.csv" && "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP] ${name} detection: ${det}/detections.csv exists"
  else
    if [[ ! -d "${fields}" ]]; then echo "[ERR] ${name} fields not found: ${fields}" >&2; FAILED=$((FAILED + 1)); continue; fi
    if [[ ! -f "${YOLO_MODEL}" ]]; then echo "[ERR] detector weights not found: ${YOLO_MODEL}" >&2; exit 2; fi
    mkdir -p "${det}"
    echo "[DETECT] ${name}: ${fields} -> ${det}"
    "${PYTHON}" -u "${DETECT_PY}" \
      --input-dir "${fields}" \
      --output-dir "${det}" \
      --model "${YOLO_MODEL}" \
      --conf 0.30 \
      --iou 0.70 \
      --imgsz 512 \
      --padding-ratio 0.10 \
      --device "${DETECT_DEVICE}" \
      --select best \
      --fallback skip 2>&1 | tee "${det}/run.log"
    rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then echo "[FAIL] ${name} detection (rc=${rc})" >&2; FAILED=$((FAILED + 1)); continue; fi
  fi
  # (2) classification of the crops with the common evaluator
  files=(); seeds_done=()
  for seed in ${SEEDS}; do
    ckpt="$(ckpt_for "${seed}")"
    out="${OUT_BASE}/${split}/seed_${seed}"
    if [[ ! -f "${ckpt}" ]]; then echo "[FAIL] seed ${seed}: checkpoint disappeared (${ckpt})" >&2; FAILED=$((FAILED + 1)); continue; fi
    if [[ -f "${out}/predictions.csv" && "${FORCE:-0}" != "1" ]]; then
      echo "[SKIP] ${name} seed ${seed}: ${out}/predictions.csv exists"
    else
      mkdir -p "${out}"
      echo "[EVAL] ${name} seed ${seed} -> ${out}"
      "${PYTHON}" -u "${EVAL_PY}" \
        --test-root "${det}/crops" \
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
        --highlight-class Promyelocyte \
        --dump-preds "${out}/path_true_pred.csv" 2>&1 | tee "${out}/run.log"
      rc=${PIPESTATUS[0]}
      if [[ ${rc} -ne 0 ]]; then echo "[FAIL] ${name} seed ${seed} (rc=${rc})" >&2; FAILED=$((FAILED + 1)); continue; fi
    fi
    files+=("${out}/predictions.csv"); seeds_done+=("${seed}")
  done
  # (3) coverage-aware scoring
  if [[ ${#files[@]} -gt 0 ]]; then
    echo "[SCORE] ${name} (stats/coverage_aware_score.py)"
    "${PYTHON}" "${REPO}/stats/coverage_aware_score.py" --name "full_pipeline_${split}" \
      --predictions "${files[@]}" --seeds "${seeds_done[@]}" \
      --detections "${det}/detections.csv" || FAILED=$((FAILED + 1))
  fi
  # optional: S12 missed-versus-detected comparison on the Phase-2 detections
  if [[ "${split}" == "phase2" && "${RUN_ABSTENTION}" == "1" ]]; then
    csv_out="${OUT_BASE}/phase2/detector_abstention_quality_test.csv"
    if [[ -f "${csv_out}" && "${FORCE:-0}" != "1" ]]; then
      echo "[SKIP] S12 comparison: ${csv_out} exists"
    else
      ck=(); for seed in 42 62 82; do ck+=("$(ckpt_for "${seed}")"); done
      echo "[S12] missed versus class-matched detected cells -> ${csv_out}"
      "${PYTHON}" -u "${REPO}/eval/detector_abstention_quality_test.py" \
        --detections "${det}/detections.csv" \
        --checkpoints "${ck[@]}" \
        --device "${DEVICE}" \
        --output-csv "${csv_out}" 2>&1 | tee "${OUT_BASE}/phase2/detector_abstention_quality_test.log"
      rc=${PIPESTATUS[0]}
      if [[ ${rc} -ne 0 ]]; then echo "[FAIL] S12 comparison (rc=${rc})" >&2; FAILED=$((FAILED + 1)); fi
    fi
  fi
done
echo "[FULL PIPELINE EVAL] done (${FAILED} failure(s))"
[[ ${FAILED} -eq 0 ]]
