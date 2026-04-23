#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
SNAP_DIR="${DATA_DIR}/SNAP"
LWA_DIR="${DATA_DIR}/LWA"
TXT_DIR="${DATA_DIR}/txt_graph"
BIN_DIR="${DATA_DIR}/bin_graph"

# 5 datasets:
# - SNAP: Friendster
# - LWA: gsh-2015-host, sk-2005, uk-2005, twitter-2010
FRIENDSTER_URL="https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz"
LWA_NAMES=("gsh-2015-host" "sk-2005" "uk-2005" "twitter-2010")

WEBGRAPH_DIR="${ROOT_DIR}/webgraph"
WEBGRAPH_CP=".:webgraph-3.6.10.jar:dsiutils-2.6.17.jar:fastutil-8.5.5.jar:log4j-1.2.17.jar:slf4j-simple-2.0.0-alpha3.jar:slf4j-api-2.0.0-alpha3.jar:jsap-2.0a.jar"

mkdir -p "${SNAP_DIR}" "${LWA_DIR}" "${TXT_DIR}" "${BIN_DIR}"

echo "=== Step 1/4: download datasets ==="
# Friendster -> SNAP
wget -nc -P "${SNAP_DIR}" "${FRIENDSTER_URL}"

# LWA datasets (.graph + .properties) -> LWA
for name in "${LWA_NAMES[@]}"; do
  base="http://data.law.di.unimi.it/webdata/${name}/${name}"
  wget -nc -P "${LWA_DIR}" "${base}.graph"
  wget -nc -P "${LWA_DIR}" "${base}.properties"
done

echo "=== Step 2/4: process LWA to txt_graph (BV2Ascii, like run.sh) ==="
for name in "${LWA_NAMES[@]}"; do
  input_prefix="${LWA_DIR}/${name}"               # expects ${input_prefix}.graph/.properties
  output_txt="${TXT_DIR}/${name}.txt"
  (cd "${WEBGRAPH_DIR}" && java -classpath "${WEBGRAPH_CP}" BV2Ascii "${input_prefix}") > "${output_txt}"
  echo "Generated ${output_txt}"
done

echo "=== Step 3/4: unzip Friendster to txt_graph ==="
if [[ -f "${SNAP_DIR}/com-friendster.ungraph.txt.gz" ]]; then
  gzip -cd "${SNAP_DIR}/com-friendster.ungraph.txt.gz" > "${TXT_DIR}/com-friendster.ungraph.txt"
  echo "Generated ${TXT_DIR}/com-friendster.ungraph.txt"
fi

echo "=== Step 4/4: convert txt_graph to bin_graph ==="
# Guard against CRLF line endings in convert.sh (common on Windows checkout).
sed -i 's/\r$//' "${SCRIPT_DIR}/convert.sh"
for graph_txt in "${TXT_DIR}"/*.txt; do
  [[ -f "${graph_txt}" ]] || continue
  filename="$(basename "${graph_txt}")"
  filename="${filename%.txt}"
  out_dir="${BIN_DIR}/${filename}"
  mkdir -p "${out_dir}"
  cp "${graph_txt}" "${out_dir}/snap.txt"
  bash "${SCRIPT_DIR}/convert.sh" "${out_dir}"
done

echo "Done. txt_graph: ${TXT_DIR}, bin_graph: ${BIN_DIR}"
