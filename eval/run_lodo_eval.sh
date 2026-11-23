#!/usr/bin/env bash
# Leave-one-dataset-out evaluation (main-text Table 4; Supplementary Note 1, S14 and S16).
# Inference only.
#
# Each ALL-minus-<X> checkpoint is evaluated on every pooled cell of its held-out source
# with eval/evaluate_lodo.py: the common evaluator without test-time augmentation, in the
# nine-class space (only neutrophil and lymphocyte sub-types merged; blast sub-types kept
# apart). Present-class macro-F1 (classes with support > 0 in the held-out truth), seed
# mean +- population SD, is computed afterwards from predictions.csv by
# stats/lodo_present_class.py.
# Arms (ARM), checkpoints from the train/scripts/ runner named in brackets:
#   proposed            the proposed recipe, seeds 42/52/62 [run_lodo_proposed.sh] (Table 4):
#                       AML-LMU 0.590 +- 0.008, PBC 0.909 +- 0.015, Raabin-WBC 0.742 +- 0.015,
#                       HRLS 0.763 +- 0.008, BMCMHF 0.606 +- 0.030; mean 0.722
#   mixup               the earlier class-weighted CE + mixup/CutMix arm, same folds and seeds
#                       [run_lodo_mixup_arm.sh] (Supplementary Note 1: mean 0.685)
#   dinobloom_proposed  DinoBloom-B, proposed recipe [run_lodo_dinobloom.sh proposed]
#                       (S16: fold-equal mean 0.776, cell-weighted 0.715)
#   dinobloom_bpaco     DinoBloom-B, BPaCo variant [run_lodo_dinobloom.sh bpaco]
#                       (S16: fold-equal mean 0.759, cell-weighted 0.692)
# Folds: LMU = AML-LMU, DIB = PBC, Raabin = Raabin-WBC, HRLS, BM = BMCMHF (paper names and
# the evidence-file spellings AMLLMU and RaabinWBC are accepted, as in train/scripts/; run
# folders always use the fold code). The AML-LMU held-out set is the corrected
# 18,128-cell packaging (data_prep/build_lmu_lodo_test.py). The paper's Table 4
# evaluations ran in the second software stack (environment/cnn_env.yml); the evaluator
# code is the same.
#
# Usage: bash eval/run_lodo_eval.sh [-h|--help]
#
# Environment (default in brackets):
#   DATA_ROOT     folder holding the five held-out pools (folder names in eval/evaluate_lodo.py)
#                 [./data]
#   LODO_TEST_ROOT_<FOLD>  override one held-out root, e.g. LODO_TEST_ROOT_LMU
#   ARM           proposed, mixup, dinobloom_proposed or dinobloom_bpaco      [proposed]
#   MODEL_ROOT    run folders of train/scripts/ (its OUT_ROOT)                [$OUT_ROOT]
#   LODO_ROOT     checkpoints of the arm   [$MODEL_ROOT/lodo_swin_base_color_aug_cwce,
#                 lodo_swin_base_mixup_cutmix_cwce, lodo_dinobloom_b_proposed or
#                 lodo_dinobloom_b_bpaco]
#   CKPT_PATTERN  checkpoint path under LODO_ROOT, {fold} and {seed} replaced
#                 [ALL_minus_{fold}/seed_{seed}/best_macro_f1.pth]
#   MODEL_NAME    timm model name [swin_base_patch4_window7_224; the DinoBloom-B hub id of
#                 eval/evaluate_dinobloom.py for the dinobloom arms]
#   FOLDS         folds to run                                           [LMU DIB Raabin HRLS BM]
#   SEEDS         seeds                                                  [42 52 62]
#   OUT_ROOT      outputs, must be outside the repository                [./outputs]
#   PYTHON        interpreter of the cnn_env environment (or journal_env) [python]
#   DEVICE        cuda or cpu                                            [cuda]
#   FORCE         1 = re-run folders that already hold predictions.csv   [0]
# Outputs: $OUT_ROOT/lodo/<ARM>/ALL_minus_<fold>/seed_<seed>/{predictions.csv,
# classification_report.txt, confusion_*, montage.png, run.log}. These are public-dataset
# cells; the montage reproduces dataset images, so check each dataset's licence before
# sharing it.
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
export OUT_ROOT DATA_ROOT
MODEL_ROOT="${MODEL_ROOT:-${OUT_ROOT}}"
ARM="${ARM:-proposed}"
DINO_HUB_ID="hf-hub:1aurent/vit_base_patch14_224.dinobloom@4f2a9d8593a7d12efe210778ea8560863fcacd03"
case "${ARM}" in
  proposed)           default_root="${MODEL_ROOT}/lodo_swin_base_color_aug_cwce";    default_model="swin_base_patch4_window7_224" ;;
  mixup)              default_root="${MODEL_ROOT}/lodo_swin_base_mixup_cutmix_cwce"; default_model="swin_base_patch4_window7_224" ;;
  dinobloom_proposed) default_root="${MODEL_ROOT}/lodo_dinobloom_b_proposed";        default_model="${DINO_HUB_ID}" ;;
  dinobloom_bpaco)    default_root="${MODEL_ROOT}/lodo_dinobloom_b_bpaco";           default_model="${DINO_HUB_ID}" ;;
  *) echo "[ERR] unknown ARM: ${ARM} (proposed, mixup, dinobloom_proposed, dinobloom_bpaco)" >&2; exit 2 ;;
esac
LODO_ROOT="${LODO_ROOT:-${default_root}}"
MODEL_NAME="${MODEL_NAME:-${default_model}}"
DEFAULT_CKPT_PATTERN='ALL_minus_{fold}/seed_{seed}/best_macro_f1.pth'
CKPT_PATTERN="${CKPT_PATTERN:-${DEFAULT_CKPT_PATTERN}}"
FOLDS="${FOLDS:-LMU DIB Raabin HRLS BM}"
SEEDS="${SEEDS:-42 52 62}"
EVAL_PY="${REPO}/eval/evaluate_lodo.py"
OUT_BASE="${OUT_ROOT}/lodo/${ARM}"

canonical_fold() {  # fold code, paper name or evidence-file spelling -> fold code
  case "$1" in      # the same aliases as train/scripts/common.sh
    LMU|AML-LMU|AMLLMU) echo LMU ;;
    DIB|PBC) echo DIB ;;
    Raabin|Raabin-WBC|RaabinWBC) echo Raabin ;;
    HRLS) echo HRLS ;;
    BM|BMCMHF) echo BM ;;
    *) return 1 ;;
  esac
}

FAILED=0
for fold_arg in ${FOLDS}; do
  fold="$(canonical_fold "${fold_arg}")" || { echo "[ERR] unknown fold: ${fold_arg}" >&2; exit 2; }
  override_var="LODO_TEST_ROOT_${fold}"
  root_args=()
  if [[ -n "${!override_var:-}" ]]; then root_args=(--test-root "${!override_var}"); fi
  for seed in ${SEEDS}; do
    rel="${CKPT_PATTERN//\{fold\}/${fold}}"
    ckpt="${LODO_ROOT}/${rel//\{seed\}/${seed}}"
    out="${OUT_BASE}/ALL_minus_${fold}/seed_${seed}"
    if [[ ! -f "${ckpt}" ]]; then echo "[FAIL] ${ARM} ${fold} seed ${seed}: checkpoint not found (${ckpt})" >&2; FAILED=$((FAILED + 1)); continue; fi
    if [[ -f "${out}/predictions.csv" && "${FORCE:-0}" != "1" ]]; then
      echo "[SKIP] ${ARM} ${fold} seed ${seed}: ${out}/predictions.csv exists"; continue
    fi
    mkdir -p "${out}"
    echo "[EVAL] LODO ${ARM} ALL_minus_${fold} seed ${seed} -> ${out}"
    "${PYTHON}" -u "${EVAL_PY}" \
      --domain "${fold}" \
      ${root_args[@]+"${root_args[@]}"} \
      --model-path "${ckpt}" \
      --model-name "${MODEL_NAME}" \
      --img-size 224 \
      --batch-size 32 \
      --device "${DEVICE}" \
      --num-workers 4 \
      --output-dir "${out}" \
      --top-k 4 2>&1 | tee "${out}/run.log"
    rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then echo "[FAIL] ${ARM} ${fold} seed ${seed} (rc=${rc})" >&2; FAILED=$((FAILED + 1)); fi
  done
done
echo "[LODO EVAL] done (${FAILED} failure(s)). Present-class macro-F1: stats/lodo_present_class.py"
echo "on ${OUT_BASE}/ALL_minus_<fold>/seed_<seed>/predictions.csv."
[[ ${FAILED} -eq 0 ]]
