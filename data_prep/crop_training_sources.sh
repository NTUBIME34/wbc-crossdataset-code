#!/usr/bin/env bash
# Detector crops of the five classifier training sources (step 1 of data_prep/README.md;
# main Methods "Datasets"; Supplementary S1 and Table S1).
#
# Every source image is passed through the Stage-1 detector; the highest-confidence box
# is cropped without margin, and an image whose best box scores below the threshold is
# excluded (data_prep/detector_crop.py). Each output folder gets a crop_log.csv with the
# best-box confidence of every image.
#
# Thresholds as run. The per-source crops behind the leave-one-dataset-out pools
# (Supplementary Tables S1 and S15) were made at 0.30 for AML-LMU, PBC, Raabin-WBC and
# HRLS and at 0.70 for BMCMHF; these are the defaults below. The merged internal corpus
# was cropped at 0.70 for every source; harmonize_labels.py --merged-dir rebuilds it
# from these crops by keeping the ones whose logged confidence is at least 0.70, so one
# detector pass serves both. Supplementary S1 states these thresholds; data_prep/README.md
# ("Detector thresholds as run") gives the evidence. CONF=0.70 applies 0.70 everywhere.
#
# Usage:
#   bash data_prep/crop_training_sources.sh [SOURCES_ROOT] [DATA_ROOT]
#
#   SOURCES_ROOT  folder holding the five datasets as downloaded (default: $SOURCES_ROOT)
#   DATA_ROOT     folder that receives the crop folders            (default: $DATA_ROOT)
#
# Environment (all optional):
#   LMU_DIR, BMCMHF_DIR, PBC_DIR, RAABIN_DIR, HRLS_DIR
#             the folder that directly contains each source's class sub-folders;
#             defaults are listed in the table below (relative to SOURCES_ROOT)
#   SOURCES   which sources to crop, e.g. "PBC HRLS" (default: all five)
#   CONF_LMU, CONF_BMCMHF, CONF_PBC, CONF_RAABIN, CONF_HRLS
#             per-source thresholds (defaults 0.30, 0.70, 0.30, 0.30, 0.30: as run)
#   CONF      one threshold for every source (overrides the CONF_* values)
#   DETECTOR  detector weights (default: yolo11_wbc_detection_best.pt in $MODEL_DIR,
#             pipeline/models/ or models/)
#   DEVICE    Ultralytics device, e.g. cpu or 0 (default: chosen by Ultralytics)
#   OVERWRITE=1  re-crop images whose crop already exists (default: resume)
#   PYTHON    Python interpreter (default: python)
#
# Source        class-folder root (default)          output folder in DATA_ROOT
# AML-LMU       PKG_AML_Cytomorphology_LMU           PKG_AML_Cytomorphology_LMU_single_wbc
# BMCMHF        BM_cytomorphology_data               BM_cytomorphology_data_single_wbc_07
# PBC           dataset_DIB                          dataset_DIB_single_wbc
# Raabin-WBC    Raabin_WBC/Train (training split)    Raabin_WBC_single_wbc
# HRLS          HRLS_dataset                         HRLS_dataset_single_wbc
#
# Classes not passed to the detector (Supplementary Table S1): PBC ig and platelet;
# AML-LMU smudge cells (KSC); BMCMHF artefact, not identifiable, other cell and smudge
# cell (ART, NIF, OTH, KSC). AML-LMU, PBC and HRLS crops are written as PNG (--png),
# BMCMHF and Raabin-WBC crops keep JPEG, as in the authors' copies.
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
SRC_ROOT="${1:-${SOURCES_ROOT:-}}"
DST_ROOT="${2:-${DATA_ROOT:-}}"
SOURCES="${SOURCES:-AML-LMU BMCMHF PBC Raabin-WBC HRLS}"

if [[ -z "${SRC_ROOT}" || -z "${DST_ROOT}" ]]; then
  echo "error: give SOURCES_ROOT and DATA_ROOT as arguments or environment variables" >&2
  usage >&2
  exit 2
fi

EXTRA=()
[[ -n "${DETECTOR:-}" ]] && EXTRA+=(--model "${DETECTOR}")
[[ -n "${DEVICE:-}" ]] && EXTRA+=(--device "${DEVICE}")
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA+=(--overwrite)

threshold() {  # threshold <per-source default>: CONF overrides every source
  if [[ -n "${CONF:-}" ]]; then echo "${CONF}"; else echo "$1"; fi
}

crop() {  # crop <source name> <input dir> <output dir> <threshold> <skip classes> [--png]
  local name="$1" input="$2" output="$3" conf="$4" skip="$5"
  shift 5
  if [[ ! -d "${input}" ]]; then
    echo "error: ${name}: class-folder root not found: ${input}" >&2
    exit 1
  fi
  echo "==== ${name}: ${input} -> ${output} (confidence >= ${conf})"
  "${PYTHON}" "${HERE}/detector_crop.py" --input-dir "${input}" --output-dir "${output}" \
    --conf "${conf}" --skip-classes "${skip}" "$@" ${EXTRA[@]+"${EXTRA[@]}"}
}

for source in ${SOURCES}; do
  case "${source}" in
    AML-LMU)
      crop "${source}" "${LMU_DIR:-${SRC_ROOT}/PKG_AML_Cytomorphology_LMU}" \
        "${DST_ROOT}/PKG_AML_Cytomorphology_LMU_single_wbc" "$(threshold "${CONF_LMU:-0.30}")" \
        "KSC,Smudge_cell" --png ;;
    BMCMHF)
      crop "${source}" "${BMCMHF_DIR:-${SRC_ROOT}/BM_cytomorphology_data}" \
        "${DST_ROOT}/BM_cytomorphology_data_single_wbc_07" "$(threshold "${CONF_BMCMHF:-0.70}")" \
        "ART,NIF,OTH,KSC,Artefact,Artifact,Not_identifiable,Other_cell,Smudge_cell" ;;
    PBC)
      crop "${source}" "${PBC_DIR:-${SRC_ROOT}/dataset_DIB}" \
        "${DST_ROOT}/dataset_DIB_single_wbc" "$(threshold "${CONF_PBC:-0.30}")" \
        "ig,platelet" --png ;;
    Raabin-WBC)
      crop "${source}" "${RAABIN_DIR:-${SRC_ROOT}/Raabin_WBC/Train}" \
        "${DST_ROOT}/Raabin_WBC_single_wbc" "$(threshold "${CONF_RAABIN:-0.30}")" "" ;;
    HRLS)
      crop "${source}" "${HRLS_DIR:-${SRC_ROOT}/HRLS_dataset}" \
        "${DST_ROOT}/HRLS_dataset_single_wbc" "$(threshold "${CONF_HRLS:-0.30}")" "" --png ;;
    *)
      echo "error: unknown source '${source}' (use AML-LMU BMCMHF PBC Raabin-WBC HRLS)" >&2
      exit 2 ;;
  esac
done
echo "Done. Next step: python data_prep/harmonize_labels.py --data-root \"${DST_ROOT}\" \\"
echo "    --merged-dir \"${DST_ROOT}/Mixed_data_mix_mono\""
