#!/usr/bin/env bash
# Shared settings for the training runners in train/scripts/. Sourced by every runner;
# not meant to be run on its own.
#
# Environment variables (only DATA_ROOT is needed in the usual case):
#   DATA_ROOT         folder that holds the prepared datasets (see data_prep/README.md)
#   CORPUS_ROOT       merged nine-class corpus with train/ val/ test/ (127,938 cells;
#                     102,345 train, 12,803 test); default $DATA_ROOT/Mixed_data_mix_mono_9_split_copy
#   LODO_ROOT_<FOLD>  leave-one-dataset-out split root for FOLD in LMU DIB Raabin HRLS BM;
#                     defaults are the data_prep folder names under $DATA_ROOT (lodo_root)
#   OUT_ROOT          parent of all run folders; default ./outputs (relative to the
#                     current directory; keep it outside the repository)
#   PYTHON            interpreter; default python (activate the right environment first)
#   DEVICE            --device for the trainers; default cuda
#   SEEDS             space-separated seeds overriding the runner's list, e.g. SEEDS="42"
#   EXTRA_ARGS        extra trainer flags appended last (they override the recipe; for
#                     smoke tests only, e.g. "--epochs 1 --max-train-steps 2")
#   DRY_RUN=1         print the trainer commands instead of running them
#
# Run folders: $OUT_ROOT/<tag>/seed_<seed>/ (LODO: $OUT_ROOT/<tag>/ALL_minus_<FOLD>/seed_<seed>/).
# A run whose folder already holds its final result file is skipped, so a runner can be
# restarted; train_classifier.py and train_dinobloom.py also resume an interrupted run.

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "${SCRIPTS_DIR}")"
REPO_ROOT="$(dirname "${TRAIN_DIR}")"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-./outputs}"

# DinoBloom-B (timm weights on the Hugging Face Hub, Apache-2.0), pinned to revision 4f2a9d85,
# the snapshot attributed to the reported runs (Supplementary S7). Later revisions keep the
# same model.safetensors and only edit config.json and README.md. The runs themselves used the
# unpinned name, which resolved to this snapshot.
DINOBLOOM_MODEL="${DINOBLOOM_MODEL:-hf-hub:1aurent/vit_base_patch14_224.dinobloom@4f2a9d8593a7d12efe210778ea8560863fcacd03}"

# As in most original runners: deterministic cuBLAS workspace (pairs with the trainers'
# torch.use_deterministic_algorithms) and less GPU memory fragmentation.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() { echo "[ERROR] $*" >&2; exit 1; }

# -h / --help on any runner prints its header comment.
for _arg in "$@"; do
  case "${_arg}" in
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
  esac
done
unset _arg

# For runners that take no positional arguments.
no_args() { [[ $# -eq 0 ]] || die "unexpected arguments: $* (configure with environment variables, see --help)"; }

# Warn once when the run folders end up inside the repository tree.
_WARNED_OUT_ROOT=0
warn_if_inside_repo() {
  [[ "${_WARNED_OUT_ROOT}" == "0" ]] || return 0
  _WARNED_OUT_ROOT=1
  local out_abs repo_abs
  out_abs="$(cd "${OUT_ROOT}" && pwd -P)" || return 0
  repo_abs="$(cd "${REPO_ROOT}" && pwd -P)"
  case "${out_abs}/" in
    "${repo_abs}/"*) echo "[WARN] OUT_ROOT=${out_abs} is inside the repository; checkpoints are large, prefer a folder outside it." >&2 ;;
  esac
}

data_root() {
  [[ -n "${DATA_ROOT:-}" ]] || die "set DATA_ROOT to the folder with the prepared datasets (see train/README.md)"
  echo "${DATA_ROOT%/}"
}

# Merged nine-class training corpus (internal train/val/test split, seed 42).
corpus_root() {
  if [[ -n "${CORPUS_ROOT:-}" ]]; then echo "${CORPUS_ROOT%/}"; return; fi
  local base
  base="$(data_root)" || exit 1
  echo "${base}/Mixed_data_mix_mono_9_split_copy"
}

# Fold keys follow the data_prep folder names: LMU = AML-LMU, DIB = PBC, Raabin = Raabin-WBC,
# HRLS = HRLS, BM = BMCMHF. Paper names are accepted as aliases.
canonical_fold() {
  case "$1" in
    LMU|AML-LMU|AMLLMU) echo LMU ;;
    DIB|PBC) echo DIB ;;
    Raabin|Raabin-WBC|RaabinWBC) echo Raabin ;;
    HRLS) echo HRLS ;;
    BM|BMCMHF) echo BM ;;
    *) die "unknown fold '$1' (use LMU DIB Raabin HRLS BM, or AML-LMU PBC Raabin-WBC HRLS BMCMHF)" ;;
  esac
}

# Training root of a LODO fold. The four folds whose training pool contains AML-LMU use the
# corrected splits (data_prep/build_corrected_lodo_splits.py); the AML-LMU fold itself uses the
# original build, whose training pool does not contain AML-LMU.
lodo_root() {
  local fold default var base
  fold="$(canonical_fold "$1")" || exit 1
  case "${fold}" in
    LMU) default="LODO_ALL_minus_LMU_test_LMU_split" ;;
    *)   default="LODO_corrected_ALL_minus_${fold}_test_${fold}_split" ;;
  esac
  var="LODO_ROOT_${fold}"
  if [[ -n "${!var:-}" ]]; then echo "${!var%/}"; return; fi
  base="$(data_root)" || exit 1
  echo "${base}/${default}"
}

# Seeds of a runner, overridable with SEEDS.
seed_list() {
  if [[ -n "${SEEDS:-}" ]]; then echo "${SEEDS}"; else echo "$*"; fi
}

check_split_root() {
  local root="$1"
  [[ "${DRY_RUN:-0}" == "1" ]] && return 0
  [[ -d "${root}/train" && -d "${root}/val" && -d "${root}/test" ]] \
    || die "${root} must contain train/ val/ test/ class folders (set DATA_ROOT or the *_ROOT variables)"
}

# run_trainer <trainer.py> <run folder> <final result file> <trainer flags...>
# Appends --outdir and --device, then EXTRA_ARGS, and logs to <run folder>/train_stdout.log.
run_trainer() {
  local trainer="$1" outdir="$2" done_file="$3"
  shift 3
  local extra=()
  [[ -n "${EXTRA_ARGS:-}" ]] && read -r -a extra <<< "${EXTRA_ARGS}"
  local cmd=("${PYTHON}" -u "${TRAIN_DIR}/${trainer}" "$@" --outdir "${outdir}" --device "${DEVICE}" ${extra[@]+"${extra[@]}"})
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
    return 0
  fi
  if [[ -f "${outdir}/${done_file}" ]]; then
    echo "[SKIP] ${outdir} already has ${done_file}"
    return 0
  fi
  mkdir -p "${outdir}"
  warn_if_inside_repo
  echo "[RUN] $(date '+%F %T') ${trainer} -> ${outdir}" | tee -a "${outdir}/train_stdout.log"
  "${cmd[@]}" 2>&1 | tee -a "${outdir}/train_stdout.log"
}
