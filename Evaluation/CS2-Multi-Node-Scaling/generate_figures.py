#!/usr/bin/env python3
"""generate_figures.py — Publication-quality figures for SigmaTrace evaluation.

Shows jitter via error bars (bar charts) and shaded bands (line charts).

Usage:
    python generate_figures.py [--datadir DIR] [--outdir DIR] [--format pdf|png]
                               [--best-only]

Options:
    --best-only    Use only the best (lowest overhead / highest bandwidth)
                   value from all repetitions, without jitter visualization.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════
# Style
# ═══════════════════════════════════════════════════════════════════

sns.set_theme(style="whitegrid", font_scale=1.1)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "baseline": "#2c3e50", "aggregate": "#27ae60",
    "simple": "#2980b9", "full": "#c0392b",
}
MARKERS = {"baseline": "o", "aggregate": "s", "simple": "D", "full": "^"}
LABELS = {
    "baseline": "Baseline", "aggregate": "Aggregate",
    "simple": "Simple", "full": "Full",
}
CONFIG_ORDER = ["baseline", "aggregate", "simple", "full"]


def human_bytes(n):
    for unit in ["B", "KB", "MB"]:
        if n < 1024:
            return f"{int(n)}{unit}" if n == int(n) else f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}GB"


def compute_overhead(df, group_cols, val_col="time_us"):
    """Compute per-rep overhead vs baseline, then aggregate stats."""
    # Median per (config, *group_cols, rep) — handles sweep multi-row
    per_rep = df.groupby(["config"] + group_cols + ["rep"])[val_col].median().reset_index()

    # Baseline per (group_cols, rep)
    bl = per_rep[per_rep["config"] == "baseline"][group_cols + ["rep", val_col]].rename(
        columns={val_col: "bl_val"})

    merged = per_rep[per_rep["config"] != "baseline"].merge(bl, on=group_cols + ["rep"])
    merged["overhead_pct"] = (merged[val_col] / merged["bl_val"] - 1) * 100

    # Aggregate across reps
    agg = merged.groupby(["config"] + group_cols)["overhead_pct"].agg(
        oh_mean="mean", oh_median="median", oh_std="std",
        oh_min="min", oh_max="max", n_reps="count"
    ).reset_index()
    agg["oh_std"] = agg["oh_std"].fillna(0)
    return agg


# ═══════════════════════════════════════════════════════════════════
# Figure: Combined scaling + sweep
# ═══════════════════════════════════════════════════════════════════

def fig_combined(df_fixed, df_sweep, outpath, best_only=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3),
                                     gridspec_kw={"width_ratios": [1, 1.4]})

    # configs = [c for c in CONFIG_ORDER if c != "baseline"]
    configs = ["full"]

    # ── Panel (a): overhead bars with error bars ──
    oh_fixed = compute_overhead(df_fixed, ["nodes", "gpus"])

    node_vals = sorted(oh_fixed["nodes"].unique())
    x = np.arange(len(node_vals))
    n_c = len(configs)
    width = 0.7 / n_c

    val_col = "oh_min" if best_only else "oh_median"
    for i, config in enumerate(configs):
        sub = oh_fixed[oh_fixed["config"] == config].set_index("nodes")
        vals = [sub.loc[n, val_col] if n in sub.index else 0 for n in node_vals]
        offset = (i - (n_c - 1) / 2) * width
        bar_kwargs = dict(color=COLORS[config], label=LABELS[config],
                          edgecolor="white", linewidth=0.5, zorder=3)
        if not best_only:
            errs = [sub.loc[n, "oh_std"] if n in sub.index else 0 for n in node_vals]
            bar_kwargs.update(yerr=errs, capsize=3, error_kw={"linewidth": 1})
        else:
            errs = [0] * len(node_vals)
        bars = ax1.bar(x + offset, vals, width * 0.9, **bar_kwargs)
        for bar, v, e in zip(bars, vals, errs):
            y_pos = bar.get_height() + e + 0.15
            ax1.text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=6.5,
                     color=COLORS[config], fontweight="medium")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n} Nodes\n{n*4} GPUs" for n in node_vals], fontsize=8)
    ax1.set_ylabel("Overhead (%)", fontsize=9)
    # ax1.set_title("(a) Fixed 1 MB, varying nodes", fontsize=9, fontweight="medium")
    ax1.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax1.grid(axis="y", alpha=0.3)
    ax1.grid(axis="x", visible=False)

    # ── Panel (b): overhead vs msg size with jitter bands ──
    oh_sweep = compute_overhead(df_sweep, ["size_bytes"])

    for config in configs:
        sub = oh_sweep[oh_sweep["config"] == config].sort_values("size_bytes")
        if sub.empty:
            continue
        sz = sub["size_bytes"].values
        vals = sub["oh_min"].values if best_only else sub["oh_median"].values

        ax2.plot(sz, vals, color=COLORS[config], marker=MARKERS[config],
                 label=LABELS[config], markersize=4.5,
                 markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        if not best_only:
            lo = sub["oh_min"].values
            hi = sub["oh_max"].values
            ax2.fill_between(sz, lo, hi, color=COLORS[config], alpha=0.12, zorder=2)

    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: human_bytes(x)))
    ax2.set_ylabel("Overhead (%)", fontsize=9)
    # ax2.set_title("(b) Fixed nodes, varying message size", fontsize=9, fontweight="medium")
    ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5, zorder=1)
    ax2.axhspan(-5, 5, alpha=0.06, color="green", zorder=0)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    # Clip y and annotate out-of-range
    YMAX = 15
    ax2.set_ylim(-5, YMAX)
    clip_col = "oh_min" if best_only else "oh_median"
    for config in configs:
        sub = oh_sweep[oh_sweep["config"] == config]
        clipped = sub[sub[clip_col] > YMAX]
        for _, row in clipped.iterrows():
            ax2.annotate(f"{row[clip_col]:.0f}%",
                         xy=(row["size_bytes"], YMAX - 0.3),
                         fontsize=6, color=COLORS[config], fontweight="bold",
                         ha="center", va="top")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.06), frameon=True, framealpha=0.9,
               edgecolor="lightgray")
    sns.despine()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


# ═══════════════════════════════════════════════════════════════════
# Figure: Standalone scaling bars
# ═══════════════════════════════════════════════════════════════════

def fig_scaling_overhead(df, outpath, best_only=False):
    oh = compute_overhead(df, ["nodes", "gpus"])
    configs = [c for c in CONFIG_ORDER if c != "baseline" and c in oh["config"].unique()]

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    node_vals = sorted(oh["nodes"].unique())
    x = np.arange(len(node_vals))
    n_c = len(configs)
    width = 0.7 / n_c

    val_col = "oh_min" if best_only else "oh_median"
    for i, config in enumerate(configs):
        sub = oh[oh["config"] == config].set_index("nodes")
        vals = [sub.loc[n, val_col] if n in sub.index else 0 for n in node_vals]
        offset = (i - (n_c - 1) / 2) * width
        bar_kwargs = dict(color=COLORS[config], label=LABELS[config],
                          edgecolor="white", linewidth=0.5, zorder=3)
        if not best_only:
            errs = [sub.loc[n, "oh_std"] if n in sub.index else 0 for n in node_vals]
            bar_kwargs.update(yerr=errs, capsize=3, error_kw={"linewidth": 1})
        else:
            errs = [0] * len(node_vals)
        bars = ax.bar(x + offset, vals, width * 0.9, **bar_kwargs)
        for bar, v, e in zip(bars, vals, errs):
            y_pos = bar.get_height() + e + 0.15
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=6.5,
                    color=COLORS[config], fontweight="medium")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} Nodes\n{n*4} GPUs" for n in node_vals])
    ax.set_ylabel("Overhead (%)")
    # ax.set_title("NCCL AllReduce Overhead", fontsize=11, fontweight="medium")
    ax.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    sns.despine(left=False, bottom=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(configs), fontsize=8,
               bbox_to_anchor=(0.5, 1.02), frameon=True, framealpha=0.9,
               edgecolor="lightgray")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


# ═══════════════════════════════════════════════════════════════════
# Figure: Sweep bandwidth with jitter bands
# ═══════════════════════════════════════════════════════════════════

def fig_sweep_bandwidth(df, outpath, best_only=False):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for config in CONFIG_ORDER:
        sub = df[df["config"] == config]
        if sub.empty:
            continue
        agg = sub.groupby("size_bytes")["bus_bw_gbps"].agg(
            ["median", "min", "max"]).reset_index()
        # For bandwidth, "best" means max (higher is better)
        val_col = "max" if best_only else "median"
        ax.plot(agg["size_bytes"], agg[val_col],
                color=COLORS[config], marker=MARKERS[config],
                label=LABELS[config], markersize=5,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        if not best_only:
            ax.fill_between(agg["size_bytes"], agg["min"], agg["max"],
                            color=COLORS[config], alpha=0.1, zorder=2)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: human_bytes(x)))
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Bus Bandwidth (GB/s)")
    # ax.set_title("NCCL AllReduce Bandwidth", fontsize=11, fontweight="medium")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    sns.despine()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


# ═══════════════════════════════════════════════════════════════════
# Figure: Doorbell three-panel
# ═══════════════════════════════════════════════════════════════════

def fig_doorbell(lat_df, tp_df, burst_df, outpath):
    db_colors = {
        "baseline": COLORS["baseline"],
        "uar_only": "#f39c12",
        "full_trace": COLORS["full"],
    }
    db_labels = {"baseline": "Baseline", "uar_only": "UAR redirect", "full_trace": "Full trace"}
    db_markers = {"baseline": "o", "uar_only": "D", "full_trace": "^"}
    db_order = ["baseline", "uar_only", "full_trace"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.5, 3))

    # ── (a) Latency: individual runs as scatter + median line ──
    for config in db_order:
        sub = lat_df[lat_df["config"] == config]
        ax1.scatter(sub["run"], sub["median_us"],
                    color=db_colors[config], marker=db_markers[config],
                    label=db_labels[config], s=50, edgecolors="white",
                    linewidths=0.5, zorder=3)
        m = sub["median_us"].median()
        ax1.axhline(y=m, color=db_colors[config], linewidth=1, linestyle="--", alpha=0.5)

    ax1.set_xlabel("Run #", fontsize=9)
    ax1.set_ylabel("Per-doorbell latency (us)", fontsize=9)
    # ax1.set_title("(a) Doorbell latency", fontsize=9, fontweight="medium")
    ax1.grid(True, alpha=0.3)

    # ── (b) Throughput with min-max bands ──
    for config in db_order:
        sub = tp_df[tp_df["config"] == config]
        agg = sub.groupby("msg_bytes")["median_gbps"].agg(
            ["median", "min", "max"]).reset_index()
        if agg.empty:
            continue
        ax2.plot(agg["msg_bytes"], agg["median"],
                 color=db_colors[config], marker=db_markers[config],
                 label=db_labels[config], markersize=5,
                 markeredgecolor="white", markeredgewidth=0.4)
        ax2.fill_between(agg["msg_bytes"], agg["min"], agg["max"],
                         color=db_colors[config], alpha=0.1)

    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: human_bytes(x)))
    ax2.set_xlabel("Message Size", fontsize=9)
    ax2.set_ylabel("Throughput (GB/s)", fontsize=9)
    # ax2.set_title("(b) Bulk throughput", fontsize=9, fontweight="medium")
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # ── (c) Burst rate with min-max bands ──
    for config in db_order:
        sub = burst_df[burst_df["config"] == config]
        agg = sub.groupby("burst_size")["median_mdbps"].agg(
            ["median", "min", "max"]).reset_index()
        if agg.empty:
            continue
        ax3.plot(agg["burst_size"], agg["median"],
                 color=db_colors[config], marker=db_markers[config],
                 label=db_labels[config], markersize=5,
                 markeredgecolor="white", markeredgewidth=0.4)
        ax3.fill_between(agg["burst_size"], agg["min"], agg["max"],
                         color=db_colors[config], alpha=0.1)

    ax3.set_xscale("log")
    ax3.set_xlabel("Burst size (doorbells)", fontsize=9)
    ax3.set_ylabel("Rate (M doorbells/s)", fontsize=9)
    ax3.set_title("(c) Sustained rate", fontsize=9, fontweight="medium")
    ax3.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.06), frameon=True, framealpha=0.9,
               edgecolor="lightgray")
    sns.despine()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default=".")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    parser.add_argument("--best-only", action="store_true",
                        help="Use only the lowest overhead value (no jitter)")
    args = parser.parse_args()

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    ext = args.format

    f_fixed = datadir / "scaling_fixed.csv"
    f_sweep = datadir / "scaling_sweep.csv"

    if f_fixed.exists():
        df_fixed = pd.read_csv(f_fixed)
        fig_scaling_overhead(df_fixed, outdir / f"scaling_overhead.{ext}",
                             best_only=args.best_only)

    if f_sweep.exists():
        df_sweep = pd.read_csv(f_sweep)
        fig_sweep_bandwidth(df_sweep, outdir / f"sweep_bandwidth.{ext}",
                            best_only=args.best_only)

    if f_fixed.exists() and f_sweep.exists():
        fig_combined(df_fixed, df_sweep, outdir / f"nccl_combined.{ext}",
                     best_only=args.best_only)

    f_lat = datadir / "doorbell_latency.csv"
    f_tp = datadir / "doorbell_throughput.csv"
    f_br = datadir / "doorbell_burst.csv"

    if f_lat.exists() and f_tp.exists() and f_br.exists():
        fig_doorbell(pd.read_csv(f_lat), pd.read_csv(f_tp), pd.read_csv(f_br),
                     outdir / f"doorbell_overhead.{ext}")
    else:
        print(f"Doorbell CSVs not found in {datadir}, skipping")

    print(f"\nAll figures in {outdir}/")


if __name__ == "__main__":
    main()
