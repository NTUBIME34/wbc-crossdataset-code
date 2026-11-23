#!/usr/bin/env bash
# Detection-and-cropping ablation on WBCBench2026 Phase-1: uncropped images and cropping
# routes (main-text Table 2b; Supplementary Table S6; the unified re-evaluation of
# Supplementary Section S17).
# Inference only.
#
# One frozen evaluator for every route: eval/evaluate_classifier.py with the unified
# settings (swin_base_patch4_window7_224, 224 px, square padding, resize 256, centre crop
# 224, horizontal-flip TTA, --use-mappings, nine-class argmax mapped to seven classes), on
# the proposed checkpoints swin_base_colour_aug_cwce_seed{42,62,82}.pth. Only the input
# route changes:
#   noloc  full 368-px fields, no localization                       0.627 +- 0.039 (n 8,080)
#   A      pre-cropped set (detector box at 0.70, no margin);
#          classifier-only reference                                 0.863 +- 0.028 (n 8,061)
#   B      field -> YOLO (pipeline/detect_yolo11.py: 512 px, conf 0.30, NMS IoU 0.70, best box,
#          10% margin, no fallback) -> Swin; the deployable pipeline 0.879 +- 0.006 (n 8,080,
#          the one field without a detection counted as an error)
#   C      field -> Cellpose crop keeping the background -> Swin     0.867 +- 0.013
#   D      field -> Cellpose crop, background removed -> Swin        0.760 +- 0.055
#   E      route-C crop -> YOLO re-crop (5% margin, the Cellpose crop is kept when no box
#          is found) -> Swin                                         0.882 +- 0.019
# Routes C and D read Cellpose crop folders made beforehand with data_prep/cellpose_crop.py
# (separate environment, environment/cellpose_env.yml). The detector and Cellpose stages do
# not depend on the classifier seed, so each runs once. Scores (accuracy, balanced
# accuracy, macro-F1, BLAST/PROMY F1; mean +- population SD) come from
# stats/coverage_aware_score.py, run at the end; route B is scored coverage-aware with its
# detections.csv. Re-running the detector on other hardware can move a few boxes, so
# individual predictions may differ slightly from the paper's runs.
#
# Usage: bash eval/run_routes_ablation.sh [-h|--help]
#
# Environment (default in brackets):
#   DATA_ROOT      prepared data                                        [./data]
#   WBCB_P1_FIELDS Phase-1 full fields, class folders  [$DATA_ROOT/wbc-bench-2026_phase1_nocrop_formatted]
#   WBCB_P1_CROPS  Phase-1 pre-cropped set           [$DATA_ROOT/wbc-bench-2026_phase1_cropped_formatted]
#   ROUTE_C_DIR    Cellpose crops keeping the background
#                  [$DATA_ROOT/wbc-bench-2026_phase1_nocrop_formatted_cellpose_keepbg_224]
#   ROUTE_D_DIR    Cellpose crops, background removed
#                  [$DATA_ROOT/wbc-bench-2026_phase1_nocrop_formatted_cellpose_224]
#   YOLO_MODEL     detector weights   [yolo11_wbc_detection_best.pt in MODEL_DIR, pipeline/models or models]
#   MODEL_DIR      released weight files                                [pipeline/models or models]
#   CKPT_PATTERN   classifier pattern, {seed} replaced [released swin_base_colour_aug_cwce_seed{seed}.pth]
#   ROUTES         routes to run                                        [noloc A B C D E]
#   SEEDS          seeds                                                [42 62 82]
#   OUT_ROOT       outputs, must be outside the repository              [./outputs]
#   PYTHON         interpreter of the journal_env environment           [python]
#   DEVICE         cuda or cpu (classifier)                             [cuda]
#   DETECT_DEVICE  detector device: auto, cpu, cuda or cuda:N      [auto; cpu if DEVICE=cpu]
#   FORCE          1 = re-run folders that already hold results         [0]
# Outputs: $OUT_ROOT/routes/detect_{B,E}/{detections.csv, crops/}, $OUT_ROOT/routes/<route>/
# seed_<seed>/{predictions.csv, path_true_pred.csv, ...}, scores in
# $OUT_ROOT/stats/coverage_aware_route_<route>.json. Per-image outputs and crops are
# WBCBench2026 material: keep them on your own machine.
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
WBCB_P1_CROPS="${WBCB_P1_CROPS:-${DATA_ROOT}/wbc-bench-2026_phase1_cropped_formatted}"
ROUTE_C_DIR="${ROUTE_C_DIR:-${DATA_ROOT}/wbc-bench-2026_phase1_nocrop_formatted_cellpose_keepbg_224}"
ROUTE_D_DIR="${ROUTE_D_DIR:-${DATA_ROOT}/wbc-bench-2026_phase1_nocrop_formatted_cellpose_224}"
ROUTES="${ROUTES:-noloc A B C D E}"
SEEDS="${SEEDS:-42 62 82}"
EVAL_PY="${REPO}/eval/evaluate_classifier.py"
DETECT_PY="${REPO}/pipeline/detect_yolo11.py"
OUT_BASE="${OUT_ROOT}/routes"

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

detect() {  # $1 input dir, $2 output dir, $3 padding ratio, $4 fallback
  if [[ -f "$2/detections.csv" && "${FORCE:-0}" != "1" ]]; then echo "[SKIP] detection: $2/detections.csv exists"; return 0; fi
  if [[ ! -d "$1" ]]; then echo "[ERR] detector input not found: $1" >&2; return 1; fi
  if [[ ! -f "${YOLO_MODEL}" ]]; then echo "[ERR] detector weights not found: ${YOLO_MODEL}" >&2; return 1; fi
  mkdir -p "$2"
  echo "[DETECT] $1 -> $2 (padding $3, fallback $4)"
  "${PYTHON}" -u "${DETECT_PY}" \
    --input-dir "$1" \
    --output-dir "$2" \
    --model "${YOLO_MODEL}" \
    --conf 0.30 \
    --iou 0.70 \
    --imgsz 512 \
    --padding-ratio "$3" \
    --device "${DETECT_DEVICE}" \
    --select best \
    --fallback "$4" 2>&1 | tee "$2/run.log"
  return "${PIPESTATUS[0]}"
}

route_root() {  # $1 route -> test root (runs the detector for B and E)
  case "$1" in
    noloc) echo "${WBCB_P1_FIELDS}" ;;
    A) echo "${WBCB_P1_CROPS}" ;;
    B) detect "${WBCB_P1_FIELDS}" "${OUT_BASE}/detect_B" 0.10 skip >&2 || return 1
       echo "${OUT_BASE}/detect_B/crops" ;;
    C) echo "${ROUTE_C_DIR}" ;;
    D) echo "${ROUTE_D_DIR}" ;;
    E) detect "${ROUTE_C_DIR}" "${OUT_BASE}/detect_E" 0.05 copy >&2 || return 1
       echo "${OUT_BASE}/detect_E/crops" ;;
    *) echo "[ERR] unknown route: $1" >&2; return 1 ;;
  esac
}

for route in ${ROUTES}; do
  if ! root="$(route_root "${route}")"; then echo "[FAIL] route ${route}: input stage failed" >&2; FAILED=$((FAILED + 1)); continue; fi
  if [[ ! -d "${root}" ]]; then echo "[FAIL] route ${route}: input not found (${root})" >&2; FAILED=$((FAILED + 1)); continue; fi
  files=(); seeds_done=()
  for seed in ${SEEDS}; do
    ckpt="$(ckpt_for "${seed}")"
    out="${OUT_BASE}/${route}/seed_${seed}"
    if [[ ! -f "${ckpt}" ]]; then echo "[FAIL] seed ${seed}: checkpoint not found (${ckpt})" >&2; FAILED=$((FAILED + 1)); continue; fi
    if [[ -f "${out}/predictions.csv" && "${FORCE:-0}" != "1" ]]; then
      echo "[SKIP] route ${route} seed ${seed}: ${out}/predictions.csv exists"
    else
      mkdir -p "${out}"
      echo "[EVAL] route ${route} seed ${seed}: ${root} -> ${out}"
      "${PYTHON}" -u "${EVAL_PY}" \
        --test-root "${root}" \
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
      if [[ ${rc} -ne 0 ]]; then echo "[FAIL] route ${route} seed ${seed} (rc=${rc})" >&2; FAILED=$((FAILED + 1)); continue; fi
    fi
    files+=("${out}/predictions.csv"); seeds_done+=("${seed}")
  done
  if [[ ${#files[@]} -eq 0 ]]; then continue; fi
  det_args=()
  if [[ "${route}" == "B" ]]; then det_args=(--detections "${OUT_BASE}/detect_B/detections.csv"); fi
  echo "[SCORE] route ${route} (stats/coverage_aware_score.py)"
  "${PYTHON}" "${REPO}/stats/coverage_aware_score.py" --name "route_${route}" \
    --predictions "${files[@]}" --seeds "${seeds_done[@]}" ${det_args[@]+"${det_args[@]}"} || FAILED=$((FAILED + 1))
done
echo "[ROUTES ABLATION] done (${FAILED} failure(s))"
[[ ${FAILED} -eq 0 ]]
