# -*- coding: utf-8 -*-
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
SHEET_NAME = "Sheet1"

COL_DATASET = "Data graph"
COL_QUERY = "query graph"

COL_PUMATCH = "PUMatch(UM+Package)"
COL_GAMMA = "GAMMA"
COL_STMATCH = "STMatch(UM)"

# Speedup columns (in Sheet1):
COL_GAMMA_SPEEDUP = "Speedup"
COL_STMATCH_SPEEDUP = "Speedup.1"

# Figure size in points (pt). 1 inch = 72 pt.
# Friendster: width 507pt, height = width / 3
FRIEND_WIDTH_PT = 507
FRIEND_HEIGHT_PT = FRIEND_WIDTH_PT / 3.0

# Other datasets: use the same height; width will be computed
# from the number of query graphs for each dataset.
OTHERS_HEIGHT_PT = FRIEND_HEIGHT_PT

# Common bar width in data coordinates (same for all figures)
BAR_WIDTH = 0.25


def _safe_filename(s: str) -> str:
    s = str(s).strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s) or "dataset"


def _query_sort_key(q: str):
    q = "" if q is None else str(q)
    m = re.fullmatch(r"([A-Za-z_ -]*?)(\d+)", q.strip())
    if not m:
        return (q, 10**18)
    prefix, num = m.group(1), int(m.group(2))
    return (prefix, num)


def _pt_to_inch(pt: float) -> float:
    return float(pt) / 72.0


def _annotate_speedup(ax, x, y, speedup_value: float):
    ax.annotate(
        f"{speedup_value:.2f}x",
        xy=(x, y),
        # 竖向排版，靠近柱顶
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

    if COL_DATASET not in df.columns or COL_QUERY not in df.columns:
        raise KeyError(
            f"Expected columns '{COL_DATASET}' and '{COL_QUERY}' in Sheet1. "
            f"Got: {df.columns.tolist()}"
        )

    datasets = df[COL_DATASET].dropna().unique().tolist()
    print("Datasets:", datasets)

    # ---------- Friendster: single figure ----------
    friendster_name = None
    friendster_queries_count = None
    friendster_payload = None  # (df_friendster, queries, methods)
    for d in datasets:
        if str(d).strip().lower() == "friendster":
            friendster_name = d
            break

    if friendster_name is not None:
        df_f = df[df[COL_DATASET] == friendster_name].copy()
        methods_f = [
            ("PUMatch(UM+Package)", COL_PUMATCH, None),
            ("GAMMA", COL_GAMMA, COL_GAMMA_SPEEDUP),
            ("STMatch(UM)", COL_STMATCH, COL_STMATCH_SPEEDUP),
        ]

        time_cols_f = [c for _, c, _ in methods_f if c in df_f.columns]
        if time_cols_f:
            df_f = df_f[df_f[time_cols_f].notna().any(axis=1)].copy()

        if not df_f.empty:
            df_f = df_f.sort_values(
                by=COL_QUERY, key=lambda s: s.map(_query_sort_key)
            )
            queries_f = df_f[COL_QUERY].astype(str).tolist()
            friendster_queries_count = len(queries_f)
            friendster_payload = (df_f, queries_f, methods_f)

            fig_f, ax_f = plt.subplots(
                figsize=(
                    _pt_to_inch(FRIEND_WIDTH_PT),
                    _pt_to_inch(FRIEND_HEIGHT_PT),
                )
            )

            x = np.arange(len(queries_f))
            n_methods_f = len(methods_f)
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

            max_time = 0.0
            min_pos_time = float("inf")
            for mi, (label, time_col, speedup_col) in enumerate(methods_f):
                if time_col not in df_f.columns:
                    continue

                offset = (mi - (n_methods_f - 1) / 2) * BAR_WIDTH
                for i in range(len(queries_f)):
                    t = df_f.iloc[i][time_col]
                    if pd.isna(t):
                        continue

                    t = float(t)
                    max_time = max(max_time, t)
                    if t > 0:
                        min_pos_time = min(min_pos_time, t)
                    bar_x = x[i] + offset

                    ax_f.bar(
                        bar_x,
                        t,
                        width=BAR_WIDTH,
                        label=label if i == 0 else "",
                        color=colors[mi % len(colors)],
                        alpha=0.85,
                        linewidth=0,
                    )

                    if speedup_col and speedup_col in df_f.columns:
                        sp = df_f.iloc[i][speedup_col]
                        if not pd.isna(sp):
                            _annotate_speedup(ax_f, bar_x, t, float(sp))

            ax_f.set_xlabel("Query graphs")
            ax_f.set_ylabel("Execution time(s)")
            ax_f.set_title(str(friendster_name))

            ax_f.set_xticks(x)
            ax_f.set_xticklabels([str(i + 1) for i in range(len(queries_f))])
            ax_f.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)

            if max_time > 0:
                if (
                    min_pos_time < float("inf")
                    and max_time / max(min_pos_time, 1e-12) >= 100
                ):
                    # Log scale with extra headroom for labels
                    ax_f.set_yscale("log")
                    ax_f.set_ylim(bottom=min_pos_time * 0.7, top=max_time * 2)
                    # y 轴刻度显示为指数 n，并在标签上标注 log
                    ax_f.yaxis.set_major_locator(LogLocator(base=10.0))
                    ax_f.yaxis.set_major_formatter(
                        FuncFormatter(
                            lambda v, p: f"{np.log10(v):.0f}" if v > 0 else ""
                        )
                    )
                    ax_f.set_ylabel("log Execution time(s)")
                else:
                    # Linear scale
                    ax_f.set_ylim(top=max_time * 2)
                    ax_f.set_ylabel("Execution time(s)")
                # y 轴刻度文本尽量贴近坐标轴
                ax_f.tick_params(axis="y", pad=1)

            ax_f.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.25),
                ncol=n_methods_f,
                frameon=False,
                columnspacing=1.2,
                handletextpad=0.6,
                handlelength=1.2,
            )

            fig_f.tight_layout()
            out_f = f"sheet2_{_safe_filename(friendster_name)}.pdf"
            fig_f.savefig(out_f, format="pdf", bbox_inches="tight")
            plt.close(fig_f)
            print(f"[{friendster_name}] saved: {out_f}")

    # ---------- Other datasets: each in its own figure ----------
    other_names = [d for d in datasets if d != friendster_name]

    # 用 Friendster 的“每个 query 对应的宽度”作为参考，以保证视觉密度一致
    if friendster_queries_count:
        per_query_pt = FRIEND_WIDTH_PT / float(friendster_queries_count)
    else:
        per_query_pt = 20.0  # 兜底值

    for dataset in other_names:
        df_d = df[df[COL_DATASET] == dataset].copy()
        # Two methods, with STMatch speedup annotation
        methods = [
            ("PUMatch(UM+Package)", COL_PUMATCH, None),
            ("STMatch(UM)", COL_STMATCH, COL_STMATCH_SPEEDUP),
        ]
        time_cols = [c for _, c, _ in methods if c in df_d.columns]
        if not time_cols:
            continue
        df_d = df_d[df_d[time_cols].notna().any(axis=1)].copy()
        if df_d.empty:
            continue
        df_d = df_d.sort_values(by=COL_QUERY, key=lambda s: s.map(_query_sort_key))

        queries = df_d[COL_QUERY].astype(str).tolist()
        n_queries = len(queries)
        if n_queries == 0:
            continue

        # Slightly tighter horizontal layout than Friendster
        width_pt = per_query_pt * n_queries * 0.75
        fig, ax = plt.subplots(
            figsize=(_pt_to_inch(width_pt), _pt_to_inch(OTHERS_HEIGHT_PT))
        )

        # Use a smaller group spacing on x so that query groups are closer,
        # while keeping bar width (BAR_WIDTH) identical to Friendster.
        group_spacing = 0.7
        x = np.arange(n_queries) * group_spacing
        n_methods = len(methods)

        # PUMatch(UM+Package) -> blue, STMatch(UM) -> green
        colors = ["#1f77b4", "#2ca02c"]

        max_time = 0.0
        min_pos_time = float("inf")
        for mi, (label, time_col, speedup_col) in enumerate(methods):
            if time_col not in df_d.columns:
                continue
            offset = (mi - (n_methods - 1) / 2) * BAR_WIDTH
            for i in range(n_queries):
                t = df_d.iloc[i][time_col]
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
                if speedup_col and speedup_col in df_d.columns:
                    sp = df_d.iloc[i][speedup_col]
                    if not pd.isna(sp):
                        _annotate_speedup(ax, bar_x, t, float(sp))

        ax.set_xlabel("Query graphs")
        ax.set_ylabel("Execution time(s)", labelpad=2)
        ax.set_title(str(dataset))
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(n_queries)])
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)

        if max_time > 0:
            if (
                min_pos_time < float("inf")
                and max_time / max(min_pos_time, 1e-12) >= 100
            ):
                # Log scale with extra headroom for labels
                ax.set_yscale("log")
                ax.set_ylim(bottom=min_pos_time * 0.7, top=max_time * 1.5)
                ax.yaxis.set_major_locator(LogLocator(base=10.0))
                ax.yaxis.set_major_formatter(
                    FuncFormatter(
                        lambda v, p: f"{np.log10(v):.0f}" if v > 0 else ""
                    )
                )
                ax.set_ylabel("log Execution time(s)", labelpad=2)
            else:
                # Linear scale with more headroom for labels
                ax.set_ylim(top=max_time * 1.3)
                ax.set_ylabel("Execution time(s)", labelpad=2)
            ax.tick_params(axis="y", pad=1)

        fig.tight_layout()
        out_file = f"sheet2_{_safe_filename(dataset)}.pdf"
        fig.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[{dataset}] saved: {out_file}")

    # 额外：把后 4 个数据集拼成一张横向大图
    other_names = [d for d in datasets if d != friendster_name]
    combined_items = []
    for dataset in other_names:
        df_d = df[df[COL_DATASET] == dataset].copy()
        methods = [
            ("PUMatch(UM+Package)", COL_PUMATCH, None),
            ("STMatch(UM)", COL_STMATCH, COL_STMATCH_SPEEDUP),
        ]
        time_cols = [c for _, c, _ in methods if c in df_d.columns]
        if not time_cols:
            continue
        df_d = df_d[df_d[time_cols].notna().any(axis=1)].copy()
        if df_d.empty:
            continue
        df_d = df_d.sort_values(by=COL_QUERY, key=lambda s: s.map(_query_sort_key))
        queries = df_d[COL_QUERY].astype(str).tolist()
        if not queries:
            continue
        combined_items.append((dataset, df_d, methods, queries))

    if combined_items:
        n = len(combined_items)
        width_ratios = [len(qs) for (_, _, _, qs) in combined_items]
        total_queries = sum(width_ratios)
        combined_width_pt = per_query_pt * total_queries * 0.75

        # 将拼接好的大图按当前长宽比整体缩放到宽度 507pt
        target_width_pt = 507
        scale = target_width_pt / combined_width_pt if combined_width_pt > 0 else 1.0
        scaled_height_pt = OTHERS_HEIGHT_PT * scale

        fig_c, axes_c = plt.subplots(
            1,
            n,
            figsize=(
                _pt_to_inch(target_width_pt),
                _pt_to_inch(scaled_height_pt),
            ),
            gridspec_kw={"width_ratios": width_ratios},
        )
        if n == 1:
            axes_c = [axes_c]

        colors = ["#1f77b4", "#2ca02c"]

        for ax, (dataset, df_d, methods, queries) in zip(axes_c, combined_items):
            n_queries = len(queries)
            group_spacing = 0.7
            x = np.arange(n_queries) * group_spacing
            n_methods = len(methods)

            max_time = 0.0
            min_pos_time = float("inf")
            for mi, (label, time_col, speedup_col) in enumerate(methods):
                if time_col not in df_d.columns:
                    continue
                offset = (mi - (n_methods - 1) / 2) * BAR_WIDTH
                for i in range(n_queries):
                    t = df_d.iloc[i][time_col]
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
                    if speedup_col and speedup_col in df_d.columns:
                        sp = df_d.iloc[i][speedup_col]
                        if not pd.isna(sp):
                            _annotate_speedup(ax, bar_x, t, float(sp))

            ax.set_xlabel("Query graphs")
            if ax is axes_c[0]:
                ax.set_ylabel("Execution time(s)", labelpad=2)
            ax.set_title(str(dataset))
            ax.set_xticks(x)
            ax.set_xticklabels([str(i + 1) for i in range(n_queries)])
            ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)

            if max_time > 0:
                if (
                    min_pos_time < float("inf")
                    and max_time / max(min_pos_time, 1e-12) >= 100
                ):
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=min_pos_time * 0.7, top=max_time * 2)
                    ax.yaxis.set_major_locator(LogLocator(base=10.0))
                    ax.yaxis.set_major_formatter(
                        FuncFormatter(
                            lambda v, p: f"{np.log10(v):.0f}" if v > 0 else ""
                        )
                    )
                    if ax is axes_c[0]:
                        ax.set_ylabel("log Execution time(s)", labelpad=2)
                else:
                    ax.set_ylim(top=max_time * 2)
                    if ax is axes_c[0]:
                        ax.set_ylabel("Execution time(s)", labelpad=2)
                ax.tick_params(axis="y", pad=1)

        # 先自适应布局，再压缩图之间和两侧的空隙，让每个子图更宽一些
        fig_c.tight_layout()
        fig_c.subplots_adjust(left=0.05, right=0.99, wspace=0.08)
        out_combined = "sheet2_other_datasets_combined.pdf"
        fig_c.savefig(out_combined, format="pdf", bbox_inches="tight")
        plt.close(fig_c)
        print(f"[others_combined] saved: {out_combined}")

        # 额外：将 Friendster 和合成图上下拼在同一张图里
        # - 合成图大小不变（本段不会修改 fig_c 的尺寸逻辑）
        # - Friendster 宽度不变（507pt），高度改为与合成图一致（scaled_height_pt）
        if friendster_name is not None and friendster_payload is not None:
            df_f2, queries_f2, methods_f2 = friendster_payload

            # 重新生成 Friendster（覆盖原来的 PDF），高度与合成图一致
            fig_f2, ax_f2 = plt.subplots(
                figsize=(
                    _pt_to_inch(FRIEND_WIDTH_PT),
                    _pt_to_inch(scaled_height_pt),
                )
            )

            x_f = np.arange(len(queries_f2))
            n_methods_f = len(methods_f2)
            colors_f = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue/orange/green

            max_time = 0.0
            min_pos_time = float("inf")
            for mi, (label, time_col, speedup_col) in enumerate(methods_f2):
                if time_col not in df_f2.columns:
                    continue
                offset = (mi - (n_methods_f - 1) / 2) * BAR_WIDTH
                for i in range(len(queries_f2)):
                    t = df_f2.iloc[i][time_col]
                    if pd.isna(t):
                        continue
                    t = float(t)
                    max_time = max(max_time, t)
                    if t > 0:
                        min_pos_time = min(min_pos_time, t)
                    bar_x = x_f[i] + offset
                    ax_f2.bar(
                        bar_x,
                        t,
                        width=BAR_WIDTH,
                        label=label if i == 0 else "",
                        color=colors_f[mi % len(colors_f)],
                        alpha=0.85,
                        linewidth=0,
                    )
                    if speedup_col and speedup_col in df_f2.columns:
                        sp = df_f2.iloc[i][speedup_col]
                        if not pd.isna(sp):
                            _annotate_speedup(ax_f2, bar_x, t, float(sp))

            ax_f2.set_xlabel("Query graphs")
            ax_f2.set_ylabel("Execution time(s)")
            ax_f2.set_title(str(friendster_name))
            ax_f2.set_xticks(x_f)
            ax_f2.set_xticklabels([str(i + 1) for i in range(len(queries_f2))])
            ax_f2.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)

            if max_time > 0:
                if (
                    min_pos_time < float("inf")
                    and max_time / max(min_pos_time, 1e-12) >= 100
                ):
                    ax_f2.set_yscale("log")
                    ax_f2.set_ylim(bottom=min_pos_time * 0.7, top=max_time * 1.3)
                    ax_f2.yaxis.set_major_locator(LogLocator(base=10.0))
                    ax_f2.yaxis.set_major_formatter(
                        FuncFormatter(
                            lambda v, p: f"{np.log10(v):.0f}" if v > 0 else ""
                        )
                    )
                    ax_f2.set_ylabel("log Execution time(s)")
                else:
                    ax_f2.set_ylim(top=max_time * 1.3)
                    ax_f2.set_ylabel("Execution time(s)")
                ax_f2.tick_params(axis="y", pad=1)

            ax_f2.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                ncol=n_methods_f,
                frameon=False,
                columnspacing=1.2,
                handletextpad=0.6,
                handlelength=1.2,
            )

            fig_f2.tight_layout()
            out_friend = f"sheet2_{_safe_filename(friendster_name)}.pdf"
            fig_f2.savefig(out_friend, format="pdf", bbox_inches="tight")
            plt.close(fig_f2)
            print(f"[{friendster_name}] re-saved (matched height): {out_friend}")

            # 上下拼接：上 Friendster，下 4 个数据集合成图
            # 最终上下拼接图：整体稍微加高一些，为图例和标注留出空间
            fig_s = plt.figure(
                figsize=(
                    _pt_to_inch(FRIEND_WIDTH_PT),
                    _pt_to_inch(scaled_height_pt * 2.3),
                )
            )
            gs = fig_s.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)

            ax_top = fig_s.add_subplot(gs[0, 0])
            # 复用与上面一致的 Friendster 绘制逻辑（不再标注 query 名，仅 1..n）
            for mi, (label, time_col, speedup_col) in enumerate(methods_f2):
                if time_col not in df_f2.columns:
                    continue
                offset = (mi - (n_methods_f - 1) / 2) * BAR_WIDTH
                for i in range(len(queries_f2)):
                    t = df_f2.iloc[i][time_col]
                    if pd.isna(t):
                        continue
                    t = float(t)
                    bar_x = x_f[i] + offset
                    ax_top.bar(
                        bar_x,
                        t,
                        width=BAR_WIDTH,
                        label=label if i == 0 else "",
                        color=colors_f[mi % len(colors_f)],
                        alpha=0.85,
                        linewidth=0,
                    )
                    if speedup_col and speedup_col in df_f2.columns:
                        sp = df_f2.iloc[i][speedup_col]
                        if not pd.isna(sp):
                            _annotate_speedup(ax_top, bar_x, t, float(sp))

            ax_top.set_xlabel("Query graphs")
            ax_top.set_ylabel(ax_f2.get_ylabel(), labelpad=2)
            ax_top.set_title(str(friendster_name))
            ax_top.set_xticks(x_f)
            ax_top.set_xticklabels([str(i + 1) for i in range(len(queries_f2))])
            ax_top.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
            ax_top.set_yscale(ax_f2.get_yscale())
            ax_top.set_ylim(ax_f2.get_ylim())
            if ax_f2.get_yscale() == "log":
                ax_top.yaxis.set_major_locator(LogLocator(base=10.0))
                ax_top.yaxis.set_major_formatter(
                    FuncFormatter(
                        lambda v, p: f"{np.log10(v):.0f}" if v > 0 else ""
                    )
                )
            ax_top.tick_params(axis="y", pad=1)
            # 图例抬高一些，避免遮住柱顶的加速比标注
            ax_top.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.28),
                ncol=n_methods_f,
                frameon=False,
                columnspacing=1.2,
                handletextpad=0.6,
                handlelength=1.2,
            )

            # Bottom: subplots for combined datasets
            gs_bottom = gs[1, 0].subgridspec(
                1, n, width_ratios=width_ratios, wspace=0.08
            )
            axes_bottom = [fig_s.add_subplot(gs_bottom[0, i]) for i in range(n)]

            colors_o = ["#1f77b4", "#2ca02c"]  # blue/green
            for ax, (dataset, df_d, methods, queries) in zip(axes_bottom, combined_items):
                n_queries = len(queries)
                group_spacing = 0.7
                x = np.arange(n_queries) * group_spacing
                n_methods = len(methods)

                max_time = 0.0
                min_pos_time = float("inf")
                for mi, (label, time_col, speedup_col) in enumerate(methods):
                    if time_col not in df_d.columns:
                        continue
                    offset = (mi - (n_methods - 1) / 2) * BAR_WIDTH
                    for i in range(n_queries):
                        t = df_d.iloc[i][time_col]
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
                            color=colors_o[mi % len(colors_o)],
                            alpha=0.85,
                            linewidth=0,
                        )
                        if speedup_col and speedup_col in df_d.columns:
                            sp = df_d.iloc[i][speedup_col]
                            if not pd.isna(sp):
                                _annotate_speedup(ax, bar_x, t, float(sp))

                ax.set_xlabel("Query graphs")
                if ax is axes_bottom[0]:
                    ax.set_ylabel("Execution time(s)", labelpad=2)
                ax.set_title(str(dataset))
                ax.set_xticks(x)
                ax.set_xticklabels([str(i + 1) for i in range(n_queries)])
                ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)

                if max_time > 0:
                    if (
                        min_pos_time < float("inf")
                        and max_time / max(min_pos_time, 1e-12) >= 100
                    ):
                        ax.set_yscale("log")
                        ax.set_ylim(bottom=min_pos_time * 0.7, top=max_time * 5)
                        ax.yaxis.set_major_locator(LogLocator(base=10.0))
                        ax.yaxis.set_major_formatter(
                            FuncFormatter(
                                lambda v, p: f"{np.log10(v):.0f}" if v > 0 else ""
                            )
                        )
                        if ax is axes_bottom[0]:
                            ax.set_ylabel("log Execution time(s)", labelpad=2)
                    else:
                        ax.set_ylim(top=max_time * 5)
                        if ax is axes_bottom[0]:
                            ax.set_ylabel("Execution time(s)", labelpad=2)
                    ax.tick_params(axis="y", pad=1)

            fig_s.subplots_adjust(left=0.05, right=0.99, top=0.98, bottom=0.06)
            out_stack = "sheet2_Friendster_plus_other_datasets.pdf"
            fig_s.savefig(out_stack, format="pdf", bbox_inches="tight")
            plt.close(fig_s)
            print(f"[stacked] saved: {out_stack}")


if __name__ == "__main__":
    main()

