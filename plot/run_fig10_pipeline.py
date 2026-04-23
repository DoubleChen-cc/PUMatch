#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click pipeline for Fig10:
1) run distributed PUMatch(test_mpi) and GAMMA(sm_mpi) at node counts 2/3/4
2) aggregate max per-node times to Sheet2
3) plot with plot_fig10.py
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def run_cmd(cmd: str, cwd: Path) -> None:
    print(f"[RUN] ({cwd}) {cmd}")
    rc = subprocess.call(cmd, cwd=str(cwd), shell=True)
    if rc != 0:
        raise RuntimeError(f"Command failed (exit={rc}): {cmd}")


def parse_patterns(s: str | None, pattern_dir: Path) -> List[Path]:
    if s:
        return [Path(x.strip()) for x in s.split(",") if x.strip()]
    return sorted(pattern_dir.glob("*.g"), key=lambda p: p.name)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    plot_dir = root / "plot"
    gamma_dir = root / "compared_systems" / "GAMMA"
    pattern_dir = root / "pattern"

    parser = argparse.ArgumentParser(description="Run distributed benchmarks and plot Fig10")
    parser.add_argument("--graph", required=True, help="Data graph path")
    parser.add_argument("--patterns", default=None, help="Comma-separated query paths; default pattern/*.g")
    parser.add_argument("--nodes", default="2,3,4", help="Node counts, comma-separated, e.g. 2,3,4")
    parser.add_argument("--clear-result", action="store_true", help="Clear pu_result.txt and gamma_result.txt")
    parser.add_argument("--skip-run", action="store_true", help="Skip benchmark execution")

    parser.add_argument("--pu-exe", default="./test_mpi")
    parser.add_argument("--ga-exe", default="./sm_mpi")

    parser.add_argument("--pu-mpirun-template", default="mpirun -np {np} {exe} {graph} {query}")
    parser.add_argument("--ga-mpirun-template", default="mpirun -np {np} {exe} {graph} {query} 3 full debug")

    parser.add_argument("--pu-extra-args", default="")
    parser.add_argument("--ga-extra-args", default="")
    parser.add_argument("--graph-filter", default="", help="Optional graph stem filter for Sheet2")
    args = parser.parse_args()

    graph = Path(args.graph)
    if not graph.is_absolute():
        graph = (root / graph).resolve()

    patterns = parse_patterns(args.patterns, pattern_dir)
    if not patterns:
        raise RuntimeError("No pattern files found.")

    nodes = [int(x.strip()) for x in args.nodes.split(",") if x.strip()]
    for n in nodes:
        if n <= 0:
            raise ValueError(f"Invalid node count: {n}")

    pu_result = root / "pu_result.txt"
    ga_result = gamma_dir / "gamma_result.txt"
    if args.clear_result:
        pu_result.write_text("", encoding="utf-8")
        ga_result.write_text("", encoding="utf-8")
        print(f"[CLEAR] {pu_result}")
        print(f"[CLEAR] {ga_result}")

    if not args.skip_run:
        pu_exe = Path(args.pu_exe)
        if not pu_exe.is_absolute():
            pu_exe = (root / pu_exe).resolve()
        ga_exe = Path(args.ga_exe)
        if not ga_exe.is_absolute():
            ga_exe = (gamma_dir / ga_exe).resolve()

        pu_extra = args.pu_extra_args.strip()
        ga_extra = args.ga_extra_args.strip()

        for n in nodes:
            for q in patterns:
                q_abs = q if q.is_absolute() else (root / q).resolve()
                print(f"\n===== np={n}, query={q_abs.name} =====")
                pu_cmd = args.pu_mpirun_template.format(np=n, exe=shlex.quote(str(pu_exe)),
                                                        graph=shlex.quote(str(graph)), query=shlex.quote(str(q_abs)))
                if pu_extra:
                    pu_cmd = pu_cmd + " " + pu_extra
                run_cmd(pu_cmd, root)

                ga_cmd = args.ga_mpirun_template.format(np=n, exe=shlex.quote(str(ga_exe)),
                                                        graph=shlex.quote(str(graph)), query=shlex.quote(str(q_abs)))
                if ga_extra:
                    ga_cmd = ga_cmd + " " + ga_extra
                run_cmd(ga_cmd, gamma_dir)

    prep = plot_dir / "prepare_results.py"
    prep_cmd = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(prep))} "
        f"fig10 "
        f"--pu {shlex.quote(str(pu_result.resolve()))} "
        f"--ga {shlex.quote(str(ga_result.resolve()))} "
        f"--out {shlex.quote(str((plot_dir / 'result.xlsx').resolve()))}"
    )
    if args.graph_filter:
        prep_cmd += f" --graph-filter {shlex.quote(args.graph_filter)}"
    run_cmd(prep_cmd, root)

    plot_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(plot_dir / 'plot_fig10.py'))}"
    run_cmd(plot_cmd, plot_dir)
    print("\n[DONE] Fig10 pipeline completed.")


if __name__ == "__main__":
    main()

