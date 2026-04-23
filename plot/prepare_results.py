#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified data preparation for plotting.

Subcommands:
  - fig9  -> write Sheet1 for plot_fig9.py
  - fig10 -> write Sheet2 for plot_fig10.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _normalize_name(path_like: str) -> str:
    s = str(path_like).strip().strip('"').strip("'")
    base = os.path.basename(s)
    stem, _ = os.path.splitext(base)
    return stem if stem else s


def _parse_simple_result_file(path: Path, time_unit: str) -> Dict[Tuple[str, str], float]:
    """Parse lines: <graph> <query> <time> <matches>."""
    data: Dict[Tuple[str, str], float] = {}
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            g, q, t_str = parts[0], parts[1], parts[2]
            try:
                t = float(t_str)
            except ValueError:
                continue
            if time_unit == "ms":
                t = t / 1000.0
            elif time_unit != "s":
                raise ValueError(f"Unsupported time_unit: {time_unit}")
            data[(_normalize_name(g), _normalize_name(q))] = t
    return data


def _parse_dist_result(path: Path, time_unit: str) -> Dict[Tuple[str, str, int], float]:
    """Parse lines: <graph> <query> t0 ... t(n-1) <matches>; keep max(ti)."""
    out: Dict[Tuple[str, str, int], float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            g = _normalize_name(parts[0])
            q = _normalize_name(parts[1])
            nums = parts[2:]
            if len(nums) < 2:
                continue
            try:
                times = [float(x) for x in nums[:-1]]
            except ValueError:
                continue
            if not times:
                continue
            t = max(times)
            if time_unit == "ms":
                t = t / 1000.0
            elif time_unit != "s":
                raise ValueError(f"Unsupported time unit: {time_unit}")
            out[(g, q, len(times))] = t
    return out


def _write_sheet(excel_path: Path, sheet_name: str, df: pd.DataFrame) -> None:
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    if excel_path.exists():
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            df.to_excel(w, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name=sheet_name, index=False)


def prepare_fig9(args: argparse.Namespace) -> None:
    pu = _parse_simple_result_file(args.pumatch, args.pumatch_unit)
    ga = _parse_simple_result_file(args.gamma, args.gamma_unit)
    st = _parse_simple_result_file(args.stmatch, args.stmatch_unit)

    keys = sorted(set(pu.keys()) | set(ga.keys()) | set(st.keys()))
    rows: List[dict] = []
    for g, q in keys:
        t_pu = pu.get((g, q), np.nan)
        t_ga = ga.get((g, q), np.nan)
        t_st = st.get((g, q), np.nan)
        rows.append(
            {
                "Data graph": g,
                "query graph": q,
                "PUMatch(UM+Package)": t_pu,
                "GAMMA": t_ga,
                "STMatch(UM)": t_st,
                "Speedup": (t_ga / t_pu) if (pd.notna(t_pu) and pd.notna(t_ga) and t_pu > 0) else np.nan,
                "Speedup.1": (t_st / t_pu) if (pd.notna(t_pu) and pd.notna(t_st) and t_pu > 0) else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid rows parsed for fig9.")
    _write_sheet(args.out, "Sheet1", df)
    print(f"Updated {args.out} Sheet1, rows={len(df)}")


def prepare_fig10(args: argparse.Namespace) -> None:
    pu = _parse_dist_result(args.pu, args.pu_unit)
    ga = _parse_dist_result(args.ga, args.ga_unit)
    keys = sorted(set((g, q) for g, q, _ in pu.keys()) | set((g, q) for g, q, _ in ga.keys()))
    if args.graph_filter:
        gf = args.graph_filter.strip().lower()
        keys = [(g, q) for g, q in keys if g.lower() == gf]

    rows: List[dict] = []
    for g, q in keys:
        row = {"query graph": q}
        for n in (2, 3, 4):
            t_pu = pu.get((g, q, n), np.nan)
            t_ga = ga.get((g, q, n), np.nan)
            row[f"PU{n}"] = t_pu
            row[f"GA{n}"] = t_ga
            row[f"Speedup{n}"] = (t_ga / t_pu) if (pd.notna(t_pu) and pd.notna(t_ga) and t_pu > 0) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows for Sheet2.")
    _write_sheet(args.out, "Sheet2", df)
    print(f"Updated {args.out} Sheet2, rows={len(df)}")


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent

    parser = argparse.ArgumentParser(description="Unified prepare script for plot inputs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p9 = sub.add_parser("fig9", help="Prepare Sheet1")
    p9.add_argument("--pumatch", type=Path, default=(root / "result.txt"))
    p9.add_argument("--gamma", type=Path, default=(root / "compared_systems" / "GAMMA" / "result.txt"))
    p9.add_argument("--stmatch", type=Path, default=(root / "compared_systems" / "STMatch_UM" / "result.txt"))
    p9.add_argument("--pumatch-unit", choices=["ms", "s"], default="ms")
    p9.add_argument("--gamma-unit", choices=["ms", "s"], default="s")
    p9.add_argument("--stmatch-unit", choices=["ms", "s"], default="ms")
    p9.add_argument("--out", type=Path, default=(here / "result.xlsx"))
    p9.set_defaults(func=prepare_fig9)

    p10 = sub.add_parser("fig10", help="Prepare Sheet2")
    p10.add_argument("--pu", type=Path, default=(root / "pu_result.txt"))
    p10.add_argument("--ga", type=Path, default=(root / "compared_systems" / "GAMMA" / "gamma_result.txt"))
    p10.add_argument("--pu-unit", choices=["ms", "s"], default="ms")
    p10.add_argument("--ga-unit", choices=["ms", "s"], default="s")
    p10.add_argument("--out", type=Path, default=(here / "result.xlsx"))
    p10.add_argument("--graph-filter", default="")
    p10.set_defaults(func=prepare_fig10)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

