#!/usr/bin/env bash
# Two-stage inference: YOLO11 detection and crop, then Swin-B classification.
# The defaults are the paper's pipeline settings; see pipeline/README.md.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash pipeline/run_pipeline.sh INPUT_DIR [OUTPUT_DIR]

  INPUT_DIR   folder of field images, searched recursively (default: $DATA_ROOT)
  OUTPUT_DIR  run folder, outside the repository; relative paths are resolved against
              the current directory (default: ${OUT_ROOT:-./outputs}/pipeline_${RUN_NAME})

Writes OUTPUT_DIR/01_detection/{detections.csv,crops/} and
OUTPUT_DIR/02_classification/predictions.csv.

Environment variables (defaults in brackets):
  PYTHON_BIN        Python interpreter [python3]
  MODEL_DIR         weights folder searched first [unset; then pipeline/models, then <repo>/models]
  YOLO_MODEL        detector [yolo11_wbc_detection_best.pt in the weights folder]
  CLASSIFIER_CKPT   classifier [swin_base_colour_aug_cwce_seed62.pth in the weights folder]
  CLASSIFIER_CKPTS  comma-separated classifiers for a seed ensemble; overrides CLASSIFIER_CKPT []
  TTA               hflip or none [hflip]
  CONF              detector confidence threshold [0.30]
  IOU               detector NMS IoU [0.70]
  YOLO_IMGSZ        detector input size [512]
  PADDING_RATIO     box expansion per side, fraction of box size [0.10]
  SELECT            best (highest-confidence box) or all [best]
  FALLBACK          skip (no crop for a missed image) or copy (whole image as crop) [skip]
  SAVE_CROPS        1 = save crops and classify; 0 = detection only [1]
  SAVE_ANNOTATED    1 = also save images with the box drawn [0]
  DEVICE            auto, cpu, cuda or cuda:N [auto]
  BATCH_SIZE        classifier batch size [32]
  WORKERS           DataLoader workers [4]
  TOPK              nine-class top-k kept in predictions.csv [3]
  RUN_NAME          name of the default output folder [current date and time]
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR="${1:-${DATA_ROOT:-}}"
if [[ -z "${INPUT_DIR}" ]]; then
  usage >&2
  exit 1
fi
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${2:-${OUT_ROOT:-./outputs}/pipeline_${RUN_NAME}}"

# Refuse a run folder inside the repository, as the eval/ runners do: detector crops and
# predictions of a labelled benchmark (for example WBCBench2026) must stay out of the tree.
# The run folder is compared with symbolic links and '..' resolved; a part that does not
# exist yet is resolved the way mkdir -p will create it.
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd -P)"
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
case "$(physical_path "${OUTPUT_DIR}")/" in "${REPO_DIR}/"*)
  echo "[ERR] OUTPUT_DIR (${OUTPUT_DIR}) is inside the repository; pass a run folder outside it or set OUT_ROOT to a folder outside it." >&2
  exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
# Weights: explicit YOLO_MODEL / CLASSIFIER_CKPT(S), else the Python scripts look for the
# default file names in $MODEL_DIR, pipeline/models/ and <repo>/models/, in that order.
if [[ -n "${MODEL_DIR:-}" ]]; then
  export MODEL_DIR
fi
YOLO_MODEL="${YOLO_MODEL:-}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-}"
CLASSIFIER_CKPTS="${CLASSIFIER_CKPTS:-}"
TTA="${TTA:-hflip}"

CONF="${CONF:-0.30}"
IOU="${IOU:-0.70}"
YOLO_IMGSZ="${YOLO_IMGSZ:-512}"
PADDING_RATIO="${PADDING_RATIO:-0.10}"
SELECT="${SELECT:-best}"
FALLBACK="${FALLBACK:-skip}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"
TOPK="${TOPK:-3}"
SAVE_ANNOTATED="${SAVE_ANNOTATED:-0}"
SAVE_CROPS="${SAVE_CROPS:-1}"

mkdir -p "${OUTPUT_DIR}"

DETECT_ARGS=(
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}/01_detection"
  --conf "${CONF}"
  --iou "${IOU}"
  --imgsz "${YOLO_IMGSZ}"
  --padding-ratio "${PADDING_RATIO}"
  --select "${SELECT}"
  --fallback "${FALLBACK}"
  --device "${DEVICE}"
)

if [[ -n "${YOLO_MODEL}" ]]; then
  DETECT_ARGS+=(--model "${YOLO_MODEL}")
fi

if [[ "${SAVE_ANNOTATED}" == "1" ]]; then
  DETECT_ARGS+=(--save-annotated)
fi

if [[ "${SAVE_CROPS}" != "1" ]]; then
  DETECT_ARGS+=(--no-save-crops)
fi

echo "[1/2] YOLO11 detection and crop"
"${PYTHON_BIN}" "${PIPELINE_DIR}/detect_yolo11.py" "${DETECT_ARGS[@]}"

if [[ "${SAVE_CROPS}" != "1" ]]; then
  echo "SAVE_CROPS=${SAVE_CROPS}: detection only, classification skipped."
  echo "Detection CSV: ${OUTPUT_DIR}/01_detection/detections.csv"
  exit 0
fi

CLASSIFY_ARGS=(
  --input-dir "${OUTPUT_DIR}/01_detection/crops"
  --output-csv "${OUTPUT_DIR}/02_classification/predictions.csv"
  --detections-csv "${OUTPUT_DIR}/01_detection/detections.csv"
  --tta "${TTA}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --topk "${TOPK}"
)

if [[ -n "${CLASSIFIER_CKPTS}" ]]; then
  CLASSIFY_ARGS+=(--checkpoints "${CLASSIFIER_CKPTS}")
elif [[ -n "${CLASSIFIER_CKPT}" ]]; then
  CLASSIFY_ARGS+=(--checkpoint "${CLASSIFIER_CKPT}")
fi

echo "[2/2] Swin-B classification"
"${PYTHON_BIN}" "${PIPELINE_DIR}/classify.py" "${CLASSIFY_ARGS[@]}"

echo "Done."
echo "Detection CSV: ${OUTPUT_DIR}/01_detection/detections.csv"
echo "Crops: ${OUTPUT_DIR}/01_detection/crops"
echo "Predictions CSV: ${OUTPUT_DIR}/02_classification/predictions.csv"
