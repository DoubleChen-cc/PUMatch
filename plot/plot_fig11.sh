#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash plot/plot_fig11.sh --graph data/bin_graph/com-friendster.ungraph/snap.txt
# Optional:
#   --patterns "pattern/3.g,pattern/4.g"
#   --exec "./test {graph} {query}"
#   --dataset "Friendster"
#
# This script performs 3 builds/runs in order:
#   1) PACKAGE_ONLY=true
#   2) PACKAGE_ONLY=false, hop2neighbor=false
#   3) hop2neighbor=true, DEGREE_THRESHOLD=256
# Then it computes speedups and updates result.xlsx/Sheet1 for plot_fig11.py.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOT_DIR="${ROOT_DIR}/plot"
CONFIG_H="${ROOT_DIR}/src/config.h"
EXCEL_PATH="${PLOT_DIR}/result.xlsx"

GRAPH_PATH=""
PATTERNS_CSV=""
EXEC_TEMPLATE="./test {graph} {query}"
DATASET_NAME="Friendster"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --graph)
      GRAPH_PATH="$2"; shift 2;;
    --patterns)
      PATTERNS_CSV="$2"; shift 2;;
    --exec)
      EXEC_TEMPLATE="$2"; shift 2;;
    --dataset)
      DATASET_NAME="$2"; shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

if [[ -z "${GRAPH_PATH}" ]]; then
  echo "Missing --graph"
  exit 1
fi

if [[ "${GRAPH_PATH}" != /* ]]; then
  GRAPH_PATH="${ROOT_DIR}/${GRAPH_PATH}"
fi

if [[ ! -f "${CONFIG_H}" ]]; then
  echo "config.h not found: ${CONFIG_H}"
  exit 1
fi

if [[ -z "${PATTERNS_CSV}" ]]; then
  mapfile -t PATTERNS < <(cd "${ROOT_DIR}" && ls pattern/*.g 2>/dev/null | sort)
else
  IFS=',' read -r -a PATTERNS <<< "${PATTERNS_CSV}"
fi

if [[ ${#PATTERNS[@]} -eq 0 ]]; then
  echo "No query patterns found."
  exit 1
fi

for i in "${!PATTERNS[@]}"; do
  p="${PATTERNS[$i]}"
  if [[ "${p}" != /* ]]; then
    p="${ROOT_DIR}/${p}"
  fi
  PATTERNS[$i]="$p"
done

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
RAW_TSV="${TMP_DIR}/fig11_raw.tsv"
echo -e "query\tpackage_only\td256\tnon_prefetch" > "${RAW_TSV}"

backup_cfg="${TMP_DIR}/config.h.bak"
cp "${CONFIG_H}" "${backup_cfg}"

restore_config() {
  cp "${backup_cfg}" "${CONFIG_H}" || true
}
trap 'restore_config; rm -rf "${TMP_DIR}"' EXIT

set_cfg_bool() {
  local key="$1" val="$2"
  sed -i -E "s|(inline constexpr bool ${key} = )[a-z]+;|\1${val};|g" "${CONFIG_H}"
}

set_cfg_int() {
  local key="$1" val="$2"
  sed -i -E "s|(inline constexpr int ${key} = )[0-9]+;|\1${val};|g" "${CONFIG_H}"
}

build_target() {
  echo "[BUILD] make clean && make"
  (cd "${ROOT_DIR}" && make clean && make)
}

run_and_get_seconds() {
  local query="$1"
  local cmd="${EXEC_TEMPLATE}"
  cmd="${cmd//\{graph\}/${GRAPH_PATH}}"
  cmd="${cmd//\{query\}/${query}}"
  echo "[RUN] ${cmd}"
  local out
  out="$(cd "${ROOT_DIR}" && eval "${cmd}")"
  # Expect final numeric pair "... <ms> <matches>"
  local ms
  ms="$(printf "%s\n" "${out}" | awk '
    {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^[-+]?[0-9]*\.?[0-9]+$/ && (i+1)<=NF && $(i+1) ~ /^[-+]?[0-9]+$/) {
          last=$i
        }
      }
    }
    END { if (last=="") print "NaN"; else print last }'
  )"
  if [[ "${ms}" == "NaN" ]]; then
    echo "Failed to parse runtime(ms) from command output"
    printf "%s\n" "${out}"
    exit 1
  fi
  python3 - <<PY
ms=float("${ms}")
print(ms/1000.0)
PY
}

declare -A T_PACKAGE_ONLY
declare -A T_D256
declare -A T_NON_PREFETCH

echo "===== Phase 1: PACKAGE_ONLY=true ====="
set_cfg_bool "PACKAGE_ONLY" "true"
build_target
for q in "${PATTERNS[@]}"; do
  name="$(basename "${q}")"
  name="${name%.g}"
  T_PACKAGE_ONLY["$name"]="$(run_and_get_seconds "${q}")"
done

echo "===== Phase 2: PACKAGE_ONLY=false, hop2neighbor=false ====="
set_cfg_bool "PACKAGE_ONLY" "false"
set_cfg_bool "hop2neighbor" "false"
build_target
for q in "${PATTERNS[@]}"; do
  name="$(basename "${q}")"
  name="${name%.g}"
  T_NON_PREFETCH["$name"]="$(run_and_get_seconds "${q}")"
done

echo "===== Phase 3: hop2neighbor=true, DEGREE_THRESHOLD=256 ====="
set_cfg_bool "hop2neighbor" "true"
set_cfg_int "DEGREE_THRESHOLD" "256"
build_target
for q in "${PATTERNS[@]}"; do
  name="$(basename "${q}")"
  name="${name%.g}"
  T_D256["$name"]="$(run_and_get_seconds "${q}")"
done

for q in "${PATTERNS[@]}"; do
  name="$(basename "${q}")"
  name="${name%.g}"
  echo -e "${name}\t${T_PACKAGE_ONLY[$name]}\t${T_D256[$name]}\t${T_NON_PREFETCH[$name]}" >> "${RAW_TSV}"
done

python3 "${PLOT_DIR}/prepare_fig11_input.py" \
  --raw "${RAW_TSV}" \
  --excel "${EXCEL_PATH}" \
  --dataset "${DATASET_NAME}"

echo "===== Plot Fig11 ====="
(cd "${PLOT_DIR}" && python3 plot_fig11.py)
echo "[DONE] Fig11 pipeline finished."

