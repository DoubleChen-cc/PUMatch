#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible wrapper: use unified prepare_results.py fig9."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    cmd = [sys.executable, str(here / "prepare_results.py"), "fig9", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(here.parent)))


if __name__ == "__main__":
    main()

