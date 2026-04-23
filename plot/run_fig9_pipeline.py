#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键流程：
1) 运行 PUMatch / GAMMA / STMatch
2) 汇总 result.txt -> result.xlsx (Sheet1)
3) 调用 plot_fig9.py 出图
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"[RUN] ({cwd}) {' '.join(shlex.quote(c) for c in cmd)}")
    p = subprocess.Popen(cmd, cwd=str(cwd))
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (exit={rc}): {' '.join(cmd)}")


def parse_patterns(value: str | None, pattern_dir: Path) -> List[Path]:
    if value:
        return [Path(x.strip()) for x in value.split(",") if x.strip()]
    pats = sorted(pattern_dir.glob("*.g"), key=lambda p: p.name)
    return [p for p in pats]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    plot_dir = root / "plot"
    gamma_dir = root / "compared_systems" / "GAMMA"
    st_dir = root / "compared_systems" / "STMatch_UM"
    pattern_dir = root / "pattern"

    parser = argparse.ArgumentParser(description="Run 3 systems + build fig9 inputs + plot")
    parser.add_argument("--graph", required=True, help="数据图路径（传给三个程序）")
    parser.add_argument(
        "--patterns",
        default=None,
        help="query 列表，逗号分隔；默认使用 pattern/*.g 全部",
    )
    parser.add_argument(
        "--clear-result",
        action="store_true",
        help="运行前清空三个 result.txt",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="跳过运行程序，只做汇总+画图",
    )
    parser.add_argument(
        "--pumatch-bin",
        default="./test",
        help="PUMatch 可执行文件（相对项目根目录）",
    )
    parser.add_argument(
        "--gamma-bin",
        default="./sm",
        help="GAMMA 可执行文件（相对 compared_systems/GAMMA）",
    )
    parser.add_argument(
        "--stmatch-bin",
        default="./cu_test",
        help="STMatch 可执行文件（相对 compared_systems/STMatch_UM）",
    )
    parser.add_argument(
        "--gamma-mt",
        default="1",
        help="GAMMA 第3参数 graph_mt（默认1）",
    )
    parser.add_argument(
        "--gamma-mode",
        default="full",
        choices=["full", "truncate10"],
        help="GAMMA 第4参数（默认full）",
    )
    parser.add_argument(
        "--gamma-debug-flag",
        default="debug",
        help="GAMMA 最后一个参数（默认debug）",
    )
    parser.add_argument(
        "--stmatch-extra-args",
        default="",
        help="附加给 STMatch 的额外参数（原样拆词拼接）",
    )
    parser.add_argument(
        "--pumatch-extra-args",
        default="",
        help="附加给 PUMatch 的额外参数（原样拆词拼接）",
    )
    args = parser.parse_args()

    patterns = parse_patterns(args.patterns, pattern_dir)
    if not patterns:
        raise RuntimeError("No pattern files found/provided.")

    graph = Path(args.graph)
    if not graph.is_absolute():
        graph = (root / graph).resolve()

    if args.clear_result:
        for rp in [
            root / "result.txt",
            gamma_dir / "result.txt",
            st_dir / "result.txt",
        ]:
            rp.write_text("", encoding="utf-8")
            print(f"[CLEAR] {rp}")

    if not args.skip_run:
        pumatch_bin = Path(args.pumatch_bin)
        if not pumatch_bin.is_absolute():
            pumatch_bin = (root / pumatch_bin).resolve()
        gamma_bin = Path(args.gamma_bin)
        if not gamma_bin.is_absolute():
            gamma_bin = (gamma_dir / gamma_bin).resolve()
        st_bin = Path(args.stmatch_bin)
        if not st_bin.is_absolute():
            st_bin = (st_dir / st_bin).resolve()

        pu_extra = shlex.split(args.pumatch_extra_args) if args.pumatch_extra_args else []
        st_extra = shlex.split(args.stmatch_extra_args) if args.stmatch_extra_args else []

        for q in patterns:
            q_abs = q if q.is_absolute() else (root / q).resolve()
            print(f"\n===== Query: {q_abs.name} =====")

            run_cmd([str(pumatch_bin), str(graph), str(q_abs)] + pu_extra, root)
            run_cmd(
                [
                    str(gamma_bin),
                    str(graph),
                    str(q_abs),
                    str(args.gamma_mt),
                    args.gamma_mode,
                    args.gamma_debug_flag,
                ],
                gamma_dir,
            )
            run_cmd([str(st_bin), str(graph), str(q_abs)] + st_extra, st_dir)

    # 汇总成 result.xlsx (Sheet1)
    prep = plot_dir / "prepare_results.py"
    run_cmd(
        [
            sys.executable,
            str(prep),
            "fig9",
            "--pumatch",
            str((root / "result.txt").resolve()),
            "--gamma",
            str((gamma_dir / "result.txt").resolve()),
            "--stmatch",
            str((st_dir / "result.txt").resolve()),
            "--out",
            str((plot_dir / "result.xlsx").resolve()),
        ],
        root,
    )

    # 画图
    plot_py = plot_dir / "plot_fig9.py"
    run_cmd([sys.executable, str(plot_py)], plot_dir)
    print("\n[DONE] Fig9 pipeline completed.")


if __name__ == "__main__":
    main()

