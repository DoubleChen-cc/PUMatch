#!/usr/bin/env bash
set -euo pipefail

# Enron-only pipeline (LWA format):
# 1) Read enron.graph + enron.properties from data/LWA
# 2) Convert to txt via BV2Ascii
# 3) Convert txt -> bin_graph/enron via convert.sh
# 4) Optional: run ./test for a quick check
#
# Usage:
#   bash graph_converter/test_enron.sh
#   bash graph_converter/test_enron.sh --run --pattern pattern/1.g

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEBGRAPH_DIR="${ROOT_DIR}/webgraph"

DATA_DIR="${ROOT_DIR}/data"
LWA_DIR="${DATA_DIR}/LWA"
TXT_DIR="${DATA_DIR}/txt_graph"
BIN_DIR="${DATA_DIR}/bin_graph"

NAME="enron"
RUN_TEST=0
PATTERN_REL="pattern/1.g"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_TEST=1
      shift
      ;;
    --pattern)
      PATTERN_REL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

INPUT_PREFIX="${LWA_DIR}/${NAME}"
INPUT_GRAPH="${INPUT_PREFIX}.graph"
INPUT_PROP="${INPUT_PREFIX}.properties"
OUT_TXT="${TXT_DIR}/${NAME}.txt"
OUT_BIN_DIR="${BIN_DIR}/${NAME}"

WEBGRAPH_CP=".:webgraph-3.6.10.jar:dsiutils-2.6.17.jar:fastutil-8.5.5.jar:log4j-1.2.17.jar:slf4j-simple-2.0.0-alpha3.jar:slf4j-api-2.0.0-alpha3.jar:jsap-2.0a.jar"

mkdir -p "${LWA_DIR}" "${TXT_DIR}" "${BIN_DIR}"

if [[ ! -f "${INPUT_GRAPH}" ]]; then
  echo "Missing file: ${INPUT_GRAPH}"
  exit 1
fi
if [[ ! -f "${INPUT_PROP}" ]]; then
  echo "Missing file: ${INPUT_PROP}"
  exit 1
fi

echo "=== Step 1/3: BV2Ascii -> ${OUT_TXT} ==="
(cd "${WEBGRAPH_DIR}" && java -classpath "${WEBGRAPH_CP}" BV2Ascii "${INPUT_PREFIX}") > "${OUT_TXT}"

echo "=== Step 2/3: txt -> bin_graph/${NAME} ==="
mkdir -p "${OUT_BIN_DIR}"
cp "${OUT_TXT}" "${OUT_BIN_DIR}/snap.txt"
# Guard against CRLF line endings in convert.sh (common on Windows checkout).
sed -i 's/\r$//' "${SCRIPT_DIR}/convert.sh"
mkdir -p "${SCRIPT_DIR}/bin"
bash "${SCRIPT_DIR}/convert.sh" "${OUT_BIN_DIR}"

echo "=== Step 3/3: done ==="
echo "TXT: ${OUT_TXT}"
echo "BIN: ${OUT_BIN_DIR}"

if [[ "${RUN_TEST}" -eq 1 ]]; then
  PATTERN_PATH="${ROOT_DIR}/${PATTERN_REL}"
  if [[ ! -f "${PATTERN_PATH}" ]]; then
    echo "Pattern not found: ${PATTERN_PATH}"
    exit 1
  fi
  if [[ ! -x "${ROOT_DIR}/test" ]]; then
    echo "Executable ./test not found. Build first: make"
    exit 1
  fi
  echo "=== Optional check: ./test ${OUT_BIN_DIR}/snap.txt ${PATTERN_PATH} ==="
  (cd "${ROOT_DIR}" && ./test "${OUT_BIN_DIR}/snap.txt" "${PATTERN_PATH}")
fi

