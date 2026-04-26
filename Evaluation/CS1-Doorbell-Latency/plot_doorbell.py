#!/usr/bin/env python3
"""plot_doorbell.py — Generate figures from doorbell benchmark v3 results.

Reads results/<config>_r<N>.csv files. Each CSV has columns:
  op, [bytes,] iter, ns, us

Generates:
  1. doorbell_latency: Grouped bar chart of median latency per operation type, by config
  2. doorbell_overhead_relative: Bar chart of % overhead vs baseline per op type
  3. doorbell_overhead_absolute: Bar chart of absolute added latency vs baseline
  4. doorbell_bulk_latency: Bulk put latency vs message size line chart
  5. doorbell_bulk_throughput: Bulk put throughput vs message size line chart

Usage:
    python plot_doorbell.py [--datadir results] [--outdir figures] [--format pdf]
"""

import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {"baseline": "#2c3e50", "uar_only": "#e67e22", "full_trace": "#c0392b"}
LABELS = {"baseline": "Baseline", "uar_only": "UAR forward", "full_trace": "Forward + Read WQEs"}
CONFIG_ORDER = ["baseline", "uar_only", "full_trace"]
OP_ORDER = ["put", "get", "atomic", "pingpong"]
OP_LABELS = {"put": "Put+Quiet", "get": "Get", "atomic": "Atomic\nFetchAdd", "pingpong": "Ping-Pong\n(RTT)"}

# Unit convention: All raw data is in nanoseconds (ns column).
# Convert to microseconds (μs) for display: ns / 1000


def human_bytes(n):
    for unit in ["B", "KB", "MB"]:
        if n < 1024:
            return f"{int(n)}{unit}"
        n /= 1024
    return f"{n:.0f}GB"


def load_data(datadir):
    """Load all CSV files, tag with config and run number."""
    frames = []
    for f in sorted(Path(datadir).glob("*_r*.csv")):
        # filename: baseline_r1.csv, full_trace_r3.csv, etc.
        stem = f.stem  # e.g. "full_trace_r3"
        parts = stem.rsplit("_r", 1)
        if len(parts) != 2:
            continue
        config = parts[0]
        run = int(parts[1])
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  WARN: {f}: {e}")
            continue
        df["config"] = config
        df["run"] = run
        frames.append(df)

    if not frames:
        print("No data found.")
        return None
    return pd.concat(frames, ignore_index=True)


def fig_latency_bars(df, outpath):
    """Grouped bar chart: median ns per operation type, per config."""
    scalar = df[df["op"].isin(OP_ORDER)]
    # Median across iterations, then median across runs
    per_run = scalar.groupby(["config", "op", "run"])["ns"].median().reset_index()
    agg = per_run.groupby(["config", "op"])["ns"].agg(
        med="median", std="std").reset_index()
    agg["std"] = agg["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(6, 2.8))
    ops = [o for o in OP_ORDER if o in agg["op"].unique()]
    configs = [c for c in CONFIG_ORDER if c in agg["config"].unique()]
    x = np.arange(len(ops))
    n_c = len(configs)
    width = 0.7 / n_c

    for i, config in enumerate(configs):
        sub = agg[agg["config"] == config].set_index("op")
        vals_us = [sub.loc[o, "med"] / 1000 if o in sub.index else 0 for o in ops]
        errs_us = [sub.loc[o, "std"] / 1000 if o in sub.index else 0 for o in ops]
        offset = (i - (n_c - 1) / 2) * width
        ax.bar(x + offset, vals_us, width * 0.88,
               yerr=errs_us, capsize=3, error_kw={"linewidth": 0.8},
               color=COLORS[config], label=LABELS[config],
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([OP_LABELS.get(o, o) for o in ops], fontsize=9)
    ax.set_ylabel("Median latency (μs)", fontsize=10)
    ax.set_title("Per-Operation Completion Latency", fontsize=11, fontweight="medium")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    sns.despine()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


def fig_overhead_relative(df, outpath):
    """Overhead % vs baseline per operation type."""
    scalar = df[df["op"].isin(OP_ORDER)]
    per_run = scalar.groupby(["config", "op", "run"])["ns"].median().reset_index()

    # Paired overhead: each run's config vs same run's baseline
    bl = per_run[per_run["config"] == "baseline"][["op", "run", "ns"]].rename(
        columns={"ns": "bl_ns"})
    merged = per_run[per_run["config"] != "baseline"].merge(bl, on=["op", "run"])
    merged["overhead_pct"] = (merged["ns"] / merged["bl_ns"] - 1) * 100

    agg = merged.groupby(["config", "op"]).agg(
        oh_med=("overhead_pct", "median"),
        oh_std=("overhead_pct", "std"),
    ).reset_index()
    agg["oh_std"] = agg["oh_std"].fillna(0)

    fig, ax = plt.subplots(figsize=(5, 2.8))

    ops = [o for o in OP_ORDER if o in agg["op"].unique()]
    configs = [c for c in CONFIG_ORDER if c != "baseline" and c in agg["config"].unique()]
    x = np.arange(len(ops))
    n_c = len(configs)
    width = 0.65 / n_c

    for i, config in enumerate(configs):
        sub = agg[agg["config"] == config].set_index("op")
        vals = [sub.loc[o, "oh_med"] if o in sub.index else 0 for o in ops]
        errs = [sub.loc[o, "oh_std"] if o in sub.index else 0 for o in ops]
        offset = (i - (n_c - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.88,
                      yerr=errs, capsize=3, error_kw={"linewidth": 0.8},
                      color=COLORS[config], label=LABELS[config],
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=7, color=COLORS[config])

    ax.set_xticks(x)
    ax.set_xticklabels([OP_LABELS.get(o, o) for o in ops], fontsize=8)
    ax.set_ylabel("Overhead (%)", fontsize=9)
    ax.set_title("Relative Overhead vs Baseline", fontsize=10, fontweight="medium")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


def fig_overhead_absolute(df, outpath):
    """Absolute added latency vs baseline per operation type."""
    scalar = df[df["op"].isin(OP_ORDER)]
    per_run = scalar.groupby(["config", "op", "run"])["ns"].median().reset_index()

    # Paired overhead: each run's config vs same run's baseline
    bl = per_run[per_run["config"] == "baseline"][["op", "run", "ns"]].rename(
        columns={"ns": "bl_ns"})
    merged = per_run[per_run["config"] != "baseline"].merge(bl, on=["op", "run"])
    merged["added_us"] = (merged["ns"] - merged["bl_ns"]) / 1000

    agg = merged.groupby(["config", "op"]).agg(
        add_med=("added_us", "median"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(5, 2.8))

    ops = [o for o in OP_ORDER if o in agg["op"].unique()]
    configs = [c for c in CONFIG_ORDER if c != "baseline" and c in agg["config"].unique()]
    x = np.arange(len(ops))
    n_c = len(configs)
    width = 0.65 / n_c

    for i, config in enumerate(configs):
        sub = agg[agg["config"] == config].set_index("op")
        vals_us = [sub.loc[o, "add_med"] if o in sub.index else 0 for o in ops]
        offset = (i - (n_c - 1) / 2) * width
        bars = ax.bar(x + offset, vals_us, width * 0.88,
                      color=COLORS[config], label=LABELS[config],
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v_us in zip(bars, vals_us):
            if abs(v_us) > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        f"{v_us:.1f}", ha="center", va="bottom",
                        fontsize=7, color=COLORS[config])

    ax.set_xticks(x)
    ax.set_xticklabels([OP_LABELS.get(o, o) for o in ops], fontsize=8)
    ax.set_ylabel("Added latency (μs)", fontsize=9)
    ax.set_title("Absolute Added Latency vs Baseline", fontsize=10, fontweight="medium")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


def fig_bulk_latency(df, outpath):
    """Line chart: bulk put latency vs message size."""
    bulk = df[df["op"] == "put_bulk"].copy()
    if bulk.empty:
        print("No put_bulk data, skipping bulk latency.")
        return
    bulk["bytes"] = bulk["bytes"].astype(int)

    per_run = bulk.groupby(["config", "bytes", "run"])["ns"].median().reset_index()
    agg = per_run.groupby(["config", "bytes"]).agg(
        ns_med=("ns", "median"), ns_min=("ns", "min"), ns_max=("ns", "max")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(5, 2.8))
    markers = {"baseline": "o", "uar_only": "D", "full_trace": "^"}

    for config in CONFIG_ORDER:
        sub = agg[agg["config"] == config].sort_values("bytes")
        if sub.empty:
            continue
        sz = sub["bytes"].values
        med_us = sub["ns_med"].values / 1000
        lo_us = sub["ns_min"].values / 1000
        hi_us = sub["ns_max"].values / 1000

        ax.plot(sz, med_us, color=COLORS[config], marker=markers[config],
                label=LABELS[config], markersize=5,
                markeredgecolor="white", markeredgewidth=0.4)
        ax.fill_between(sz, lo_us, hi_us, color=COLORS[config], alpha=0.1)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: human_bytes(x)))
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Latency (μs)", fontsize=9)
    ax.set_xlabel("Message size")
    ax.set_title("Bulk Put Completion Latency", fontsize=10, fontweight="medium")
    ax.legend(fontsize=8, loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


def fig_bulk_throughput(df, outpath):
    """Line chart: bulk put throughput vs message size."""
    bulk = df[df["op"] == "put_bulk"].copy()
    if bulk.empty:
        print("No put_bulk data, skipping bulk throughput.")
        return
    bulk["bytes"] = bulk["bytes"].astype(int)

    per_run = bulk.groupby(["config", "bytes", "run"])["ns"].median().reset_index()
    agg = per_run.groupby(["config", "bytes"]).agg(
        ns_med=("ns", "median"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(5, 2.8))
    markers = {"baseline": "o", "uar_only": "D", "full_trace": "^"}

    for config in CONFIG_ORDER:
        sub = agg[agg["config"] == config].sort_values("bytes")
        if sub.empty:
            continue
        sz = sub["bytes"].values
        med_ns = sub["ns_med"].values
        # throughput: bytes / seconds -> GB/s
        med_s = med_ns * 1e-9
        gbps = np.where(med_s > 0, (sz / med_s) / 1e9, 0)

        ax.plot(sz, gbps, color=COLORS[config], marker=markers[config],
                label=LABELS[config], markersize=5,
                markeredgecolor="white", markeredgewidth=0.4)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: human_bytes(x)))
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Throughput (GB/s)", fontsize=9)
    ax.set_xlabel("Message size")
    ax.set_title("Bulk Put Effective Throughput", fontsize=10, fontweight="medium")
    ax.legend(fontsize=8, loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


def print_summary(df):
    """Print summary table."""
    print("\n" + "=" * 72)
    print("DOORBELL BENCHMARK SUMMARY (median ns across runs)")
    print("=" * 72)

    scalar = df[df["op"].isin(OP_ORDER)]
    per_run = scalar.groupby(["config", "op", "run"])["ns"].median().reset_index()
    agg = per_run.groupby(["config", "op"])["ns"].median().reset_index()
    pivot = agg.pivot(index="op", columns="config", values="ns")

    configs = [c for c in CONFIG_ORDER if c in pivot.columns]
    print(f"\n  {'op':<12s}", end="")
    for c in configs:
        print(f"  {LABELS.get(c,c):>14s}", end="")
    print()
    print(f"  {'─'*12}", end="")
    for _ in configs:
        print(f"  {'─'*14}", end="")
    print()

    for op in OP_ORDER:
        if op not in pivot.index:
            continue
        row = f"  {OP_LABELS.get(op,op).replace(chr(10),' '):<12s}"
        bl = pivot.loc[op, "baseline"] if "baseline" in configs else None
        for c in configs:
            v = pivot.loc[op, c] if c in pivot.columns else 0
            if c == "baseline":
                row += f"  {v:12.0f}ns"
            elif bl and bl > 0:
                oh = (v / bl - 1) * 100
                row += f"  {v:8.0f}ns({oh:+.0f}%)"
            else:
                row += f"  {v:12.0f}ns"
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="results")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    ext = args.format

    df = load_data(args.datadir)
    if df is None:
        return

    print(f"Loaded {len(df)} rows, configs: {sorted(df['config'].unique())}, "
          f"ops: {sorted(df['op'].unique())}")

    print_summary(df)

    fig_latency_bars(df, outdir / f"doorbell_latency.{ext}")
    fig_overhead_relative(df, outdir / f"doorbell_overhead_relative.{ext}")
    fig_overhead_absolute(df, outdir / f"doorbell_overhead_absolute.{ext}")
    fig_bulk_latency(df, outdir / f"doorbell_bulk_latency.{ext}")
    fig_bulk_throughput(df, outdir / f"doorbell_bulk_throughput.{ext}")


if __name__ == "__main__":
    main()
