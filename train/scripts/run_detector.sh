#!/usr/bin/env bash
# Paper rows: main Methods 2.4 "Detection-and-cropping stage": the deployed YOLO11-M detector behind every
#   full-pipeline result (main Table 2); and Supplementary S1 "Detector variant that
#   was not used" (three-class variant).
#   Reported for the deployed detector (TXL-PBC validation split, 252 images / 257 WBC boxes,
#   Ultralytics 8.4.82 default validation at 512 px): precision 0.996, recall 0.987,
#   mAP@50 0.994, mAP@50-95 0.880; early stopping ended training after epoch 137 and the
#   best epoch, 107, is deployed. Three-class variant, WBC class: precision 0.996,
#   recall 0.996, mAP@50 0.994, mAP@50-95 0.894.
# Recipe: train/train_detector.py defaults (COCO-pretrained YOLO11-M, 512 px, up to 150
#   epochs with patience 30, batch 32, seed 3407, optimizer auto, augmentation listed there).
# Provenance: the original detector training script without its server backup step; the
#   Ultralytics arguments recorded by the reported runs match train_detector.py.
# Output: $OUT_ROOT/detector/yolo11m_txlpbc_<labels>_s512/weights/best.pt (the deployed
#   detector is released as yolo11_wbc_detection_best.pt).
# Environment: Ultralytics 8.3.170, the version that trained the released detector
#   (pip install ultralytics==8.3.170); train_detector.py warns about other versions.
#
# Usage: TXLPBC_DATA_YAML=/path/to/TXL-PBC/data.yaml bash train/scripts/run_detector.sh [wbc_only|three_class]
#   wbc_only (default): the deployed detector, trained on WBC boxes only.
#   three_class: TXL-PBC labels as released (WBC, RBC, platelets).
#   Without TXLPBC_DATA_YAML the data.yaml is looked up at $DATA_ROOT/TXL-PBC/data.yaml.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

variant="${1:-wbc_only}"
case "${variant}" in
  wbc_only)    labels=wbc_only ;;
  three_class) labels=as_released ;;
  *) die "variant must be wbc_only or three_class, got '${variant}'" ;;
esac
if [[ -n "${TXLPBC_DATA_YAML:-}" ]]; then
  data_yaml="${TXLPBC_DATA_YAML}"
else
  data_yaml="$(data_root)/TXL-PBC/data.yaml"
fi

extra=()
[[ -n "${EXTRA_ARGS:-}" ]] && read -r -a extra <<< "${EXTRA_ARGS}"
cmd=("${PYTHON}" -u "${TRAIN_DIR}/train_detector.py" --data "${data_yaml}" --labels "${labels}"
     --project "${OUT_ROOT}/detector" ${extra[@]+"${extra[@]}"})
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi
[[ -f "${data_yaml}" ]] || die "TXL-PBC data.yaml not found: ${data_yaml} (set TXLPBC_DATA_YAML)"
mkdir -p "${OUT_ROOT}"
warn_if_inside_repo
"${cmd[@]}"
