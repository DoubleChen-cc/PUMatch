#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare/update Sheet3 for plot_fig12.py from raw TSV + Sheet1 time speedup."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Fig12 input in Sheet3")
    parser.add_argument("--raw", type=Path, required=True, help="raw TSV from plot_fig12.sh")
    parser.add_argument("--excel", type=Path, required=True, help="plot/result.xlsx path")
    args = parser.parse_args()

    if not args.excel.exists():
        raise RuntimeError(f"Excel not found: {args.excel} (Sheet1 required).")

    raw = pd.read_csv(args.raw, sep="\t")
    raw["query_norm"] = raw["query"].astype(str).str.replace(r"\.g$", "", regex=True)

    sheet1 = pd.read_excel(args.excel, sheet_name="Sheet1")
    if "query graph" not in sheet1.columns:
        raise RuntimeError("Sheet1 missing column: query graph")
    sheet1["query_norm"] = sheet1["query graph"].astype(str).str.replace(r"\.g$", "", regex=True)

    if "STMatch(UM)" in sheet1.columns and "PUMatch(UM+Package)" in sheet1.columns:
        sheet1["time_speedup_calc"] = sheet1["STMatch(UM)"] / sheet1["PUMatch(UM+Package)"]
    elif "Speedup.1" in sheet1.columns:
        sheet1["time_speedup_calc"] = sheet1["Speedup.1"]
    else:
        raise RuntimeError("Sheet1 missing STMatch/PUMatch columns for Time Speedup.")

    merged = raw.merge(sheet1[["query_norm", "time_speedup_calc"]], on="query_norm", how="left")
    sheet3 = pd.DataFrame(
        {
            "Query graphs": pd.to_numeric(merged["query_norm"], errors="coerce"),
            "UM-only/PU": pd.to_numeric(merged["um_only_over_um"], errors="coerce"),
            "Time Speedup": pd.to_numeric(merged["time_speedup_calc"], errors="coerce"),
        }
    )
    sheet3 = sheet3.dropna(subset=["Query graphs"]).sort_values("Query graphs")

    with pd.ExcelWriter(args.excel, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        sheet3.to_excel(w, sheet_name="Sheet3", index=False)

    print(f"Updated {args.excel} Sheet3 with {len(sheet3)} rows.")


if __name__ == "__main__":
    main()

