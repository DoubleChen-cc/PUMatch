#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/run_fig9_pipeline.py"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 未安装或不在 PATH 中"
  exit 1
fi

exec python3 "${PY_SCRIPT}" "$@"

