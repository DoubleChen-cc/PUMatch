#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare/update Sheet1 columns needed by plot_fig11.py from raw TSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Fig11 input in Sheet1")
    parser.add_argument("--raw", type=Path, required=True, help="raw TSV from plot_fig11.sh")
    parser.add_argument("--excel", type=Path, required=True, help="plot/result.xlsx path")
    parser.add_argument("--dataset", default="Friendster")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw, sep="\t")
    raw["Data graph"] = args.dataset
    raw["query graph"] = raw["query"].astype(str)
    raw["PUMatch(package-only)"] = raw["package_only"].astype(float)
    raw["PUMactch(D=256,a=0.5)"] = raw["d256"].astype(float)
    raw["PUMatch(non-prefetch)"] = raw["non_prefetch"].astype(float)
    raw["Speedup.2"] = raw["PUMatch(non-prefetch)"] / raw["PUMatch(package-only)"]
    raw["Speedup.3"] = raw["PUMatch(non-prefetch)"] / raw["PUMactch(D=256,a=0.5)"]
    raw["Speedup.4"] = 1.0

    need_cols = [
        "Data graph",
        "query graph",
        "PUMatch(package-only)",
        "PUMactch(D=256,a=0.5)",
        "PUMatch(non-prefetch)",
        "Speedup.2",
        "Speedup.3",
        "Speedup.4",
    ]
    raw = raw[need_cols]

    excel = args.excel
    if excel.exists():
        try:
            sheet1 = pd.read_excel(excel, sheet_name="Sheet1")
        except Exception:
            sheet1 = pd.DataFrame(columns=["Data graph", "query graph"])
    else:
        sheet1 = pd.DataFrame(columns=["Data graph", "query graph"])

    if "Data graph" not in sheet1.columns:
        sheet1["Data graph"] = np.nan
    if "query graph" not in sheet1.columns:
        sheet1["query graph"] = np.nan
    for c in need_cols[2:]:
        if c not in sheet1.columns:
            sheet1[c] = np.nan

    for _, r in raw.iterrows():
        m = (sheet1["Data graph"] == r["Data graph"]) & (sheet1["query graph"].astype(str) == str(r["query graph"]))
        if m.any():
            idx = sheet1.index[m][0]
        else:
            idx = len(sheet1)
            sheet1.loc[idx, "Data graph"] = r["Data graph"]
            sheet1.loc[idx, "query graph"] = r["query graph"]
        for c in need_cols[2:]:
            sheet1.loc[idx, c] = r[c]

    excel.parent.mkdir(parents=True, exist_ok=True)
    if excel.exists():
        with pd.ExcelWriter(excel, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            sheet1.to_excel(w, sheet_name="Sheet1", index=False)
    else:
        with pd.ExcelWriter(excel, engine="openpyxl") as w:
            sheet1.to_excel(w, sheet_name="Sheet1", index=False)

    print(f"Updated {excel} Sheet1 with {len(raw)} rows.")


if __name__ == "__main__":
    main()

