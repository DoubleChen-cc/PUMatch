#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash plot/plot_fig12.sh --graph data/bin_graph/com-friendster.ungraph/snap.txt
# Optional:
#   --patterns "pattern/3.g,pattern/4.g"
#   --stmatch-cmd "./compared_systems/STMatch_UM/cu_test {graph} {query}"
#   --pumatch-cmd "./test {graph} {query}"
#
# Output:
#   - Updates plot/result.xlsx Sheet3:
#       Query graphs | UM-only/PU | Time Speedup
#   - Calls plot/plot_fig12.py

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOT_DIR="${ROOT_DIR}/plot"
EXCEL_PATH="${PLOT_DIR}/result.xlsx"

GRAPH_PATH=""
PATTERNS_CSV=""
STMATCH_CMD_TEMPLATE="./compared_systems/STMatch_UM/cu_test {graph} {query}"
PUMATCH_CMD_TEMPLATE="./test {graph} {query}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --graph)
      GRAPH_PATH="$2"; shift 2;;
    --patterns)
      PATTERNS_CSV="$2"; shift 2;;
    --stmatch-cmd)
      STMATCH_CMD_TEMPLATE="$2"; shift 2;;
    --pumatch-cmd)
      PUMATCH_CMD_TEMPLATE="$2"; shift 2;;
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
RAW_TSV="${TMP_DIR}/fig12_raw.tsv"
echo -e "query\tst_transfer_mb\tpu_transfer_mb\tum_only_over_um" > "${RAW_TSV}"

parse_cuda_mem_stats_total_mb() {
  local report_file="$1"
  local stats
  stats="$(nsys stats "${report_file}" -r cuda_gpu_mem_size_sum 2>/dev/null || true)"
  if [[ -z "${stats}" ]]; then
    echo "NaN"
    return
  fi

  # Only extract UM HtD (Unified Host-to-Device), in MB.
  local um_htd
  um_htd="$(echo "${stats}" | awk '/CUDA memcpy Unified Host-to-Device/ {gsub(/,/, "", $1); print $1; exit}')"

  um_htd="${um_htd:-0}"

  python3 - <<PY
try:
  a=float("${um_htd}")
  print(a)
except Exception:
  print("NaN")
PY
}

profile_transfer_mb() {
  local cmd_template="$1"
  local query="$2"
  local tag="$3"

  local cmd="${cmd_template}"
  cmd="${cmd//\{graph\}/${GRAPH_PATH}}"
  cmd="${cmd//\{query\}/${query}}"

  local rep_prefix="${TMP_DIR}/${tag}"
  echo "[NSYS] ${cmd}"
  nsys profile \
    --stats=true \
    --trace=cuda \
    --cuda-memory-usage=true \
    --cuda-um-cpu-page-faults=true \
    --export=none \
    --force-overwrite=true \
    -o "${rep_prefix}" \
    bash -lc "cd \"${ROOT_DIR}\" && ${cmd}" >/dev/null 2>&1

  local total_mb
  total_mb="$(parse_cuda_mem_stats_total_mb "${rep_prefix}.nsys-rep")"
  rm -f "${rep_prefix}.nsys-rep" "${rep_prefix}.sqlite"
  echo "${total_mb}"
}

for q in "${PATTERNS[@]}"; do
  q_stem="$(basename "${q}")"
  q_stem="${q_stem%.g}"
  echo "===== Query ${q_stem}.g ====="

  st_mb="$(profile_transfer_mb "${STMATCH_CMD_TEMPLATE}" "${q}" "st_${q_stem}")"
  pu_mb="$(profile_transfer_mb "${PUMATCH_CMD_TEMPLATE}" "${q}" "pu_${q_stem}")"

  ratio="$(python3 - <<PY
import math
try:
  st=float("${st_mb}")
  pu=float("${pu_mb}")
  if pu<=0 or math.isnan(st) or math.isnan(pu):
    print("NaN")
  else:
    print(st/pu)
except Exception:
  print("NaN")
PY
)"
  echo -e "${q_stem}\t${st_mb}\t${pu_mb}\t${ratio}" >> "${RAW_TSV}"
done

python3 "${PLOT_DIR}/prepare_fig12_input.py" \
  --raw "${RAW_TSV}" \
  --excel "${EXCEL_PATH}"

echo "===== Plot Fig12 ====="
(cd "${PLOT_DIR}" && python3 plot_fig12.py)
echo "[DONE] Fig12 pipeline finished."

