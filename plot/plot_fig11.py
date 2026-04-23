# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator


# 字体和字号与合成图保持一致
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
SHEET_NAME = "Sheet1"

COL_DATASET = "Data graph"
COL_QUERY = "query graph"

COL_PACKAGE_ONLY = "PUMatch(package-only)"
COL_D256 = "PUMactch(D=256,a=0.5)"
COL_NON_PREFETCH = "PUMatch(non-prefetch)"

COL_PACKAGE_ONLY_SPEEDUP = "Speedup.2"
COL_D256_SPEEDUP = "Speedup.3"
COL_NON_PREFETCH_SPEEDUP = "Speedup.4"


FRIEND_WIDTH_PT = 507  # 与其他图统一宽度
FRIEND_HEIGHT_PT = FRIEND_WIDTH_PT / 3.0  # 与最初 Friendster 图相同高度
BAR_WIDTH = 0.25


def _pt_to_inch(pt: float) -> float:
    return float(pt) / 72.0


def _query_sort_key(q: str):
    q = "" if q is None else str(q)
    # 按 q1, q2, ... 的数字部分排序
    import re

    m = re.fullmatch(r"([A-Za-z_ -]*?)(\d+)", q.strip())
    if not m:
        return (q, 10**18)
    prefix, num = m.group(1), int(m.group(2))
    return (prefix, num)


def _annotate_speedup(ax, x, y, speedup_value: float):
    ax.annotate(
        f"{speedup_value:.2f}x",
        xy=(x, y),
        xytext=(0, 0),  # 字体底边紧贴柱顶
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="bold",
        rotation=90,  # 竖排
        clip_on=False,
    )


def main():
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

    # 只取 Friendster
    df_f = df[df[COL_DATASET] == "Friendster"].copy()
    if df_f.empty:
        raise RuntimeError("Sheet1 中没有 Friendster 数据。")

    # 只保留至少有一个配置有时间的数据
    time_cols = [COL_PACKAGE_ONLY, COL_D256, COL_NON_PREFETCH]
    df_f = df_f[df_f[time_cols].notna().any(axis=1)].copy()
    if df_f.empty:
        raise RuntimeError("Friendster 中没有有效的运行时间数据。")

    # 按查询图排序
    df_f = df_f.sort_values(by=COL_QUERY, key=lambda s: s.map(_query_sort_key))
    queries = df_f[COL_QUERY].astype(str).tolist()
    n_queries = len(queries)

    # 画布尺寸：宽 507pt，高 507/3 pt（与 Friendster 图一致）
    fig, ax = plt.subplots(
        figsize=(_pt_to_inch(FRIEND_WIDTH_PT), _pt_to_inch(FRIEND_HEIGHT_PT))
    )

    # 横坐标位置：与 Friendster 图保持一致，组中心间隔为 1
    x = np.arange(n_queries)

    methods = [
        ("package-only", COL_PACKAGE_ONLY, COL_PACKAGE_ONLY_SPEEDUP),
        ("D=256,a=0.5", COL_D256, COL_D256_SPEEDUP),
        ("non-prefetch", COL_NON_PREFETCH, COL_NON_PREFETCH_SPEEDUP),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 蓝 / 橙 / 绿

    max_time = 0.0
    min_pos_time = float("inf")

    for mi, (label, time_col, speedup_col) in enumerate(methods):
        if time_col not in df_f.columns:
            continue
        offset = (mi - (len(methods) - 1) / 2) * BAR_WIDTH
        for i in range(n_queries):
            t = df_f.iloc[i][time_col]
            if pd.isna(t):
                continue
            t = float(t)
            max_time = max(max_time, t)
            if t > 0:
                min_pos_time = min(min_pos_time, t)
            bar_x = x[i] + offset
            ax.bar(
                bar_x,
                t,
                width=BAR_WIDTH,
                label=label if i == 0 else "",
                color=colors[mi % len(colors)],
                alpha=0.85,
                linewidth=0,
            )
            if speedup_col in df_f.columns:
                sp = df_f.iloc[i][speedup_col]
                if not pd.isna(sp):
                    _annotate_speedup(ax, bar_x, t, float(sp))

    # 坐标轴与标签
    ax.set_xlabel("Query graphs")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(n_queries)])
    ax.set_title("Friendster")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)

    # y 轴范围与刻度样式（与之前逻辑一致）
    if max_time > 0:
        if min_pos_time < float("inf") and max_time / max(min_pos_time, 1e-12) >= 100:
            ax.set_yscale("log")
            ax.set_ylim(bottom=min_pos_time * 0.7, top=max_time * 1.2)
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, p: f"{np.log10(v):.0f}" if v > 0 else "")
            )
            ax.set_ylabel("log Execution time(s)", labelpad=2)
        else:
            ax.set_ylim(top=max_time * 1.2)
            ax.set_ylabel("Execution time(s)", labelpad=2)
        ax.tick_params(axis="y", pad=1)

    # 图例
    # 图例稍微抬高一点，避免压住 Friendster 的加速比标注
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=len(methods),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
        handlelength=1.2,
    )

    fig.tight_layout()
    out_file = "friendster_prefetch_configs.pdf"
    fig.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()

