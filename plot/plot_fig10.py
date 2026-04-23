# -*- coding: utf-8 -*-
"""
Sheet2: 每个 query graph 6 根柱 — PU2, GA2, PU3, GA3, PU4, GA4
在 PU 柱上标注对应 Speedup（一位小数），样式与 plot_sheet2_dataset_bars.py 一致。
"""
import re

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
    }
)

FILE_PATH = "result.xlsx"
SHEET_NAME = "Sheet2"

COL_QUERY = "query graph"

# 与 plot_sheet2_dataset_bars 一致
FRIEND_WIDTH_PT = 507
FRIEND_HEIGHT_PT = FRIEND_WIDTH_PT / 3.0
BAR_WIDTH = 0.25

# 6 柱顺序：节点 2,3,4 各一对 PU / GA
BAR_SPECS = [
    ("PU", "PU2", "Speedup2"),
    ("GA", "GA2", None),
    ("PU", "PU3", "Speedup3"),
    ("GA", "GA3", None),
    ("PU", "PU4", "Speedup4"),
    ("GA", "GA4", None),
]

COLOR_PU = "#1f77b4"
COLOR_GA = "#ff7f0e"


def _pt_to_inch(pt: float) -> float:
    return float(pt) / 72.0


def _query_sort_key(q: str):
    q = "" if q is None else str(q)
    m = re.fullmatch(r"([A-Za-z_ -]*?)(\d+)", q.strip())
    if not m:
        return (q, 10**18)
    prefix, num = m.group(1), int(m.group(2))
    return (prefix, num)


def _annotate_speedup_pu(ax, x, y, speedup_value: float):
    ax.annotate(
        f"{speedup_value:.1f}x",
        xy=(x, y),
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="bold",
        rotation=90,
        clip_on=False,
    )


def main():
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    df = df.sort_values(by=COL_QUERY, key=lambda s: s.map(_query_sort_key))
    queries = df[COL_QUERY].astype(str).tolist()
    n_queries = len(queries)
    if n_queries == 0:
        raise RuntimeError("Sheet2 has no rows.")

    # 宽度随 query 数量缩放（约 20 个 query 对应 507pt），高度与 Friendster 一致
    width_pt = FRIEND_WIDTH_PT * max(n_queries, 1) / 20.0
    width_pt = max(width_pt, FRIEND_WIDTH_PT * 0.5)

    fig, ax = plt.subplots(
        figsize=(_pt_to_inch(width_pt), _pt_to_inch(FRIEND_HEIGHT_PT))
    )

    x = np.arange(n_queries)
    n_bars = len(BAR_SPECS)
    total_width = 0.90
    bar_w = total_width / n_bars

    max_time = 0.0
    min_pos_time = float("inf")

    for qi in range(n_queries):
        row = df.iloc[qi]
        for bi, (kind, col_time, col_sp) in enumerate(BAR_SPECS):
            if col_time not in df.columns:
                continue
            t = row[col_time]
            if pd.isna(t):
                continue
            t = float(t)
            max_time = max(max_time, t)
            if t > 0:
                min_pos_time = min(min_pos_time, t)

            offset = (bi - (n_bars - 1) / 2) * bar_w
            bx = x[qi] + offset
            color = COLOR_PU if kind == "PU" else COLOR_GA
            ax.bar(
                bx,
                t,
                width=bar_w,
                color=color,
                alpha=0.85,
                linewidth=0,
            )

            if kind == "PU" and col_sp and col_sp in df.columns:
                sp = row[col_sp]
                if not pd.isna(sp):
                    _annotate_speedup_pu(ax, bx, t, float(sp))

    ax.set_xlabel("Query graphs")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(n_queries)])
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)

    if max_time > 0 and min_pos_time < float("inf"):
        # 始终使用 log 纵坐标
        ax.set_yscale("log")
        ax.set_ylim(bottom=min_pos_time * 0.7, top=max_time * 1.5)
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda v, p: f"{np.log10(v):.0f}" if v > 0 else ""
            )
        )
        ax.set_ylabel("log Execution time(s)", labelpad=2)
        ax.tick_params(axis="y", pad=1)

    # 图例：PU2/GA2 等（英文）
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor=COLOR_PU,
            edgecolor="none",
            alpha=0.85,
            label="PUMatch (GPUs=2/3/4)",
        ),
        Patch(
            facecolor=COLOR_GA,
            edgecolor="none",
            alpha=0.85,
            label="GAMMA (GPUs=2/3/4)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.2,
    )

    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    out_file = "sheet13_pu_ga_bars.pdf"
    fig.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
