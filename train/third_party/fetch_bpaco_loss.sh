#!/usr/bin/env bash
# Download the BPaCo loss from the authors' repository at a pinned commit and verify it.
#
# The BPaCo arms of the paper (main Table 3, Supplementary Tables S7/S8/S16) use
# BPaCoLoss from losses.py of https://github.com/Davidczy/BPaCo (BPaCo, MICCAI 2024).
# That repository has no licence file, so the loss cannot be redistributed here; this
# script fetches the exact file the experiments used and checks it byte for byte.
# The proposed model and every class-weighted-CE run do not need it.
#
# Usage:
#   bash train/third_party/fetch_bpaco_loss.sh                 # download + verify
#   bash train/third_party/fetch_bpaco_loss.sh --check         # verify an existing copy only
#   bash train/third_party/fetch_bpaco_loss.sh --from-file F   # verify and install a local copy
#                                                               # (e.g. from a clone at cbe5775)
#   bash train/third_party/fetch_bpaco_loss.sh --dest PATH     # write somewhere else
#
# Result: train/third_party/bpaco_losses.py (keep it out of version control).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/Davidczy/BPaCo"
COMMIT="cbe577570a0f23e6a49191c531ac27e4523df7da"
UPSTREAM_PATH="losses.py"
RAW_URL="https://raw.githubusercontent.com/Davidczy/BPaCo/${COMMIT}/${UPSTREAM_PATH}"
EXPECTED_BLOB="8a802682d6d634ae9e791ad6d32c1ae2f79fa7ca"   # git blob id of losses.py at COMMIT
EXPECTED_SHA256="d37005638fecd5892a7bb9c9706398faab775581b2f5e236123acb94853a1559"
EXPECTED_SIZE=23835

DEST="${HERE}/bpaco_losses.py"
MODE="download"
SRC=""

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --check) MODE="check"; shift ;;
    --from-file) MODE="local"; SRC="${2:?--from-file needs a path}"; shift 2 ;;
    --dest) DEST="${2:?--dest needs a path}"; shift 2 ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON=python3; else PYTHON=python; fi
fi

# Prints "<git blob sha1> <sha256> <size>" for a file.
digest() {
  "${PYTHON}" - "$1" <<'PYEOF'
import hashlib, sys
data = open(sys.argv[1], "rb").read()
blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
print(blob, hashlib.sha256(data).hexdigest(), len(data))
PYEOF
}

verify() {
  local file="$1" blob sha size
  read -r blob sha size < <(digest "${file}")
  if [[ "${blob}" != "${EXPECTED_BLOB}" || "${sha}" != "${EXPECTED_SHA256}" || "${size}" != "${EXPECTED_SIZE}" ]]; then
    echo "[FAIL] ${file} does not match ${REPO_URL}/blob/${COMMIT}/${UPSTREAM_PATH}" >&2
    echo "       git blob ${blob} (expected ${EXPECTED_BLOB})" >&2
    echo "       sha256   ${sha} (expected ${EXPECTED_SHA256})" >&2
    echo "       bytes    ${size} (expected ${EXPECTED_SIZE})" >&2
    return 1
  fi
  echo "[OK] ${file}: git blob ${blob}, sha256 ${sha}"
}

if [[ "${MODE}" == "check" ]]; then
  [[ -f "${DEST}" ]] || { echo "[FAIL] ${DEST} not found; run this script without --check" >&2; exit 1; }
  verify "${DEST}"
  exit $?
fi

if [[ -f "${DEST}" ]] && verify "${DEST}" >/dev/null 2>&1; then
  echo "[OK] ${DEST} is already present and verified."
  exit 0
fi

mkdir -p "$(dirname "${DEST}")"
TMP="$(mktemp "${DEST}.part.XXXXXX")"
trap 'rm -f "${TMP}"' EXIT

if [[ "${MODE}" == "local" ]]; then
  [[ -f "${SRC}" ]] || { echo "[FAIL] ${SRC} not found" >&2; exit 1; }
  cp "${SRC}" "${TMP}"
else
  echo "[INFO] Downloading ${RAW_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "${TMP}" "${RAW_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "${TMP}" "${RAW_URL}"
  else
    "${PYTHON}" -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
      "${RAW_URL}" "${TMP}"
  fi
fi

verify "${TMP}"
chmod 644 "${TMP}"
mv "${TMP}" "${DEST}"
trap - EXIT
echo "[DONE] Installed ${DEST}"
