#!/usr/bin/env bash
# Convert Bones-SEED G1 CSVs → UFO train/inference ufo_pkl (+ optional manifest).
#
# Two stages:
#   1) gear_sonic CSV → motion_lib individual pkls  (needs conda env with gear_sonic deps, e.g. isaaclab)
#   2) merge + ~10s clip → UFO ufo_pkl + manifest  (needs conda env ufo)
#
# Example:
#   bash scripts/convert_bones_seed_to_ufo.sh \
#     --session 220713 \
#     --bones-csv-root /home/liu/lyx/GR00T-WholeBodyControl/bones-seed/g1/csv \
#     --gear-sonic-root /home/liu/lyx/GR00T-WholeBodyControl/gear_sonic \
#     --convert-env isaaclab \
#     --ufo-env ufo

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION=""
BONES_CSV_ROOT=""
GEAR_SONIC_ROOT=""
CONVERT_ENV="isaaclab"
UFO_ENV="ufo"
FPS=30
FPS_SOURCE=120
NUM_WORKERS=16
CLIP_SECONDS=10
SKIP_CONVERT=0
FORCE=0

usage() {
  sed -n '2,20p' "$0"
  echo
  echo "Required:"
  echo "  --session DATE                 e.g. 220713"
  echo "  --bones-csv-root DIR           .../bones-seed/g1/csv"
  echo "  --gear-sonic-root DIR          .../gear_sonic"
  echo "Optional:"
  echo "  --convert-env NAME             conda env for stage1 (default: isaaclab)"
  echo "  --ufo-env NAME                 conda env for stage2 (default: ufo)"
  echo "  --fps N                        output fps (default: 30)"
  echo "  --fps-source N                 Bones-SEED source fps (default: 120)"
  echo "  --num-workers N                convert workers (default: 16)"
  echo "  --clip-seconds N               UFO train clip length (default: 10; 0=skip)"
  echo "  --skip-convert                 only run UFO merge (reuse existing motion_lib pkls)"
  echo "  --force                        overwrite UFO outputs / manifest"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="$2"; shift 2 ;;
    --bones-csv-root) BONES_CSV_ROOT="$2"; shift 2 ;;
    --gear-sonic-root) GEAR_SONIC_ROOT="$2"; shift 2 ;;
    --convert-env) CONVERT_ENV="$2"; shift 2 ;;
    --ufo-env) UFO_ENV="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --fps-source) FPS_SOURCE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --clip-seconds) CLIP_SECONDS="$2"; shift 2 ;;
    --skip-convert) SKIP_CONVERT=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${SESSION}" || -z "${BONES_CSV_ROOT}" || -z "${GEAR_SONIC_ROOT}" ]]; then
  usage
  exit 1
fi

CSV_DIR="${BONES_CSV_ROOT%/}/${SESSION}"
MOTION_LIB_DIR="${GEAR_SONIC_ROOT%/}/data/motion_lib_bones_seed/robot/${SESSION}"
UFO_OUT_DIR="${ROOT_DIR}/humanoidverse/data/bones_seed_g1_${SESSION}"
MANIFEST="${ROOT_DIR}/configs/data/bones_seed_g1_${SESSION}.yaml"
CONVERT_SCRIPT="${GEAR_SONIC_ROOT%/}/data_process/convert_soma_csv_to_motion_lib.py"

if [[ ! -d "${CSV_DIR}" ]]; then
  echo "CSV dir not found: ${CSV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
  echo "Convert script not found: ${CONVERT_SCRIPT}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${SKIP_CONVERT}" -eq 0 ]]; then
  echo "==> [1/2] Bones-SEED CSV → motion_lib individual pkls"
  echo "    env=${CONVERT_ENV}"
  echo "    input=${CSV_DIR}"
  echo "    output=${MOTION_LIB_DIR}"
  conda activate "${CONVERT_ENV}"
  mkdir -p "${MOTION_LIB_DIR}"
  python "${CONVERT_SCRIPT}" \
    --input "${CSV_DIR}" \
    --output "${MOTION_LIB_DIR}" \
    --robot g1 \
    --fps "${FPS}" \
    --fps_source "${FPS_SOURCE}" \
    --individual \
    --num_workers "${NUM_WORKERS}"
else
  echo "==> [1/2] skipped (--skip-convert); using ${MOTION_LIB_DIR}"
fi

if [[ ! -d "${MOTION_LIB_DIR}" ]] || ! compgen -G "${MOTION_LIB_DIR}/*.pkl" >/dev/null; then
  echo "No motion_lib pkls found under ${MOTION_LIB_DIR}" >&2
  exit 1
fi

echo "==> [2/2] merge + clip → UFO ufo_pkl"
echo "    env=${UFO_ENV}"
echo "    out=${UFO_OUT_DIR}"
conda activate "${UFO_ENV}"
cd "${ROOT_DIR}"

FORCE_FLAG=()
if [[ "${FORCE}" -eq 1 ]]; then
  FORCE_FLAG=(--force)
fi

python -m humanoidverse.tools.merge_motion_lib_to_ufo \
  --input "${MOTION_LIB_DIR}" \
  --out-dir "${UFO_OUT_DIR}" \
  --name "bones_seed_g1_${SESSION}" \
  --clip-seconds "${CLIP_SECONDS}" \
  --write-manifest "${MANIFEST}" \
  "${FORCE_FLAG[@]}"

echo
echo "Done."
echo "  full:      ${UFO_OUT_DIR}/full_ufo.pkl"
echo "  train:     ${UFO_OUT_DIR}/train_near10s_ufo.pkl"
echo "  manifest:  ${MANIFEST}"
echo
echo "Train smoke:"
echo "  ./run_train.sh --agent fb --robot-config configs/robots/g1_29dof.yaml \\"
echo "    --data-manifest ${MANIFEST#"${ROOT_DIR}"/} --gpu-ids single --smoke"
