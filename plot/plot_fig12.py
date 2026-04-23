# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# 字体和风格与其他图保持一致
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
SHEET_NAME = "Sheet3"

COL_X = "Query graphs"
COL_UM_ONLY_PU = "UM-only/PU"
COL_TIME_SPEEDUP = "Time Speedup"

# 图尺寸：宽 242pt，高为三分之一
FIG_WIDTH_PT = 242
FIG_HEIGHT_PT = FIG_WIDTH_PT / 3.0


def _pt_to_inch(pt: float) -> float:
    return float(pt) / 72.0


def main():
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

    # 丢弃缺失行，按 Queries 排序
    df = df[[COL_X, COL_UM_ONLY_PU, COL_TIME_SPEEDUP]].dropna()
    df = df.sort_values(by=COL_X)

    x = df[COL_X].values
    y_um_only_pu = df[COL_UM_ONLY_PU].values
    y_time_speedup = df[COL_TIME_SPEEDUP].values

    fig, ax = plt.subplots(
        figsize=(_pt_to_inch(FIG_WIDTH_PT), _pt_to_inch(FIG_HEIGHT_PT))
    )

    # 两条折线
    ax.plot(
        x,
        y_um_only_pu,
        marker="o",
        linestyle="-",
        linewidth=0.8,
        markersize=3,
        color="#1f77b4",
        label="Data transfer (UM/PU)",
    )
    ax.plot(
        x,
        y_time_speedup,
        marker="s",
        linestyle="-",
        linewidth=0.8,
        markersize=2,
        color="#ff7f0e",
        label="Execution time (UM/PU)",
    )

    ax.set_xlabel("Query graphs", labelpad=1)
    # 不显示纵轴名称，留白更紧凑

    # x 轴用整数刻度
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])

    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    # 图注放在图上方，避免与折线重合
    # 图注再往上抬一点，避免压到折线
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.32),
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.2,
    )

    # 减少图周围留白，并缩小底部边距，让图整体略往下移
    fig.tight_layout(rect=[0.08, 0.0, 0.98, 0.98])
    out_file = "sheet9_speedup_lines.pdf"
    fig.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()

