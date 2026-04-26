#!/usr/bin/env python3
"""parse_nccl_results.py — Extract nccl-tests/rccl-tests output into CSV.

Reports both time_us and bus_bw_gbps with jitter metrics.
Also extracts nccl-tests' own "Avg bus bandwidth" line.

Usage:
    python parse_nccl_results.py scaling_results_*/
"""

import re
import sys
import csv
import statistics
from pathlib import Path
from collections import defaultdict

# Per-size result line (out-of-place columns):
#   1048576  262144  float  sum  -1  523.62  2.00  3.75  0  ...
RESULT_RE = re.compile(
    r"^\s*(\d+)\s+\d+\s+\w+\s+\w+\s+\S+\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)

# nccl-tests footer: # Avg bus bandwidth    : 3.67018
AVG_BW_RE = re.compile(r"^#\s*Avg bus bandwidth\s*:\s*([\d.]+)")

def parse_file(path: Path) -> tuple[list[dict], float | None]:
    rows = []
    avg_bw = None
    for line in path.read_text().splitlines():
        m = RESULT_RE.match(line)
        if m:
            rows.append({
                "size_bytes": int(m.group(1)),
                "time_us": float(m.group(2)),
                "alg_bw_gbps": float(m.group(3)),
                "bus_bw_gbps": float(m.group(4)),
            })
        m2 = AVG_BW_RE.match(line)
        if m2:
            avg_bw = float(m2.group(1))
    return rows, avg_bw

FILE_A_RE = re.compile(r"A_(\w+)_(\d+)n_(\d+)g(?:_r(\d+))?\.txt")
FILE_B_RE = re.compile(r"B_(\w+)_(\d+)n_sweep(?:_r(\d+))?\.txt")

def aggregate_stats(rows, group_keys):
    """Compute stats on both time_us and bus_bw_gbps."""
    groups = defaultdict(lambda: {"time": [], "bw": []})
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups[key]["time"].append(r["time_us"])
        groups[key]["bw"].append(r["bus_bw_gbps"])

    stats = []
    for key, vals in sorted(groups.items()):
        d = dict(zip(group_keys, key))
        t, b = vals["time"], vals["bw"]
        d["n_reps"] = len(t)
        for prefix, v in [("time", t), ("bw", b)]:
            d[f"{prefix}_mean"] = statistics.mean(v)
            d[f"{prefix}_median"] = statistics.median(v)
            d[f"{prefix}_std"] = statistics.stdev(v) if len(v) > 1 else 0.0
            d[f"{prefix}_min"] = min(v)
            d[f"{prefix}_max"] = max(v)
            d[f"{prefix}_range"] = max(v) - min(v)
            d[f"{prefix}_cv"] = (d[f"{prefix}_std"] / d[f"{prefix}_mean"] * 100
                                 if d[f"{prefix}_mean"] > 0 else 0.0)
        stats.append(d)
    return stats

STAT_FIELDS = ["n_reps",
               "time_mean", "time_median", "time_std", "time_min", "time_max", "time_range", "time_cv",
               "bw_mean", "bw_median", "bw_std", "bw_min", "bw_max", "bw_range", "bw_cv"]

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir> [results_dir2 ...]")
        sys.exit(1)

    fixed_rows = []
    sweep_rows = []

    for d in sys.argv[1:]:
        dirpath = Path(d)
        platform = "cluster-slingshot" if "slingshot" in dirpath.name.lower() else "cluster-ib"

        for f in sorted(dirpath.glob("A_*.txt")):
            m = FILE_A_RE.match(f.name)
            if not m:
                continue
            config = m.group(1)
            nodes, gpus = int(m.group(2)), int(m.group(3))
            rep = int(m.group(4)) if m.group(4) else 1
            data, avg_bw = parse_file(f)
            if not data:
                print(f"  WARN: no data in {f}")
                continue
            row = {
                "platform": platform, "config": config,
                "nodes": nodes, "gpus": gpus, "rep": rep,
                **data[-1],
            }
            if avg_bw is not None:
                row["nccl_avg_bw"] = avg_bw
            fixed_rows.append(row)

        for f in sorted(dirpath.glob("B_*.txt")):
            m = FILE_B_RE.match(f.name)
            if not m:
                continue
            config = m.group(1)
            nodes = int(m.group(2))
            rep = int(m.group(3)) if m.group(3) else 1
            gpus = nodes * (8 if platform == "cluster-slingshot" else 4)
            data, avg_bw = parse_file(f)
            for row in data:
                sweep_rows.append({
                    "platform": platform, "config": config,
                    "nodes": nodes, "gpus": gpus, "rep": rep,
                    **row,
                })

    # ─── Write raw CSVs ─────────────────────────────────────────
    raw_fields = ["platform", "config", "nodes", "gpus", "rep",
                  "size_bytes", "time_us", "alg_bw_gbps", "bus_bw_gbps"]
    raw_fields_a = raw_fields + (["nccl_avg_bw"] if any("nccl_avg_bw" in r for r in fixed_rows) else [])

    if fixed_rows:
        out = Path("scaling_fixed.csv")
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=raw_fields_a, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(fixed_rows, key=lambda r: (
                r["platform"], r["nodes"], r["config"], r["rep"])))
        print(f"Wrote {out} ({len(fixed_rows)} rows)")

    if sweep_rows:
        out = Path("scaling_sweep.csv")
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=raw_fields)
            w.writeheader()
            w.writerows(sorted(sweep_rows, key=lambda r: (
                r["platform"], r["nodes"], r["config"], r.get("size_bytes", 0), r["rep"])))
        print(f"Wrote {out} ({len(sweep_rows)} rows)")

    # ─── Stats ──────────────────────────────────────────────────
    if fixed_rows:
        gkeys = ["platform", "config", "nodes", "gpus"]
        stats = aggregate_stats(fixed_rows, gkeys)

        # Also aggregate nccl_avg_bw if available
        if any("nccl_avg_bw" in r for r in fixed_rows):
            avg_bw_groups = defaultdict(list)
            for r in fixed_rows:
                if "nccl_avg_bw" in r:
                    key = tuple(r[k] for k in gkeys)
                    avg_bw_groups[key].append(r["nccl_avg_bw"])
            for s in stats:
                key = tuple(s[k] for k in gkeys)
                vals = avg_bw_groups.get(key, [])
                if vals:
                    s["nccl_avg_bw_mean"] = statistics.mean(vals)
                    s["nccl_avg_bw_median"] = statistics.median(vals)

        out = Path("scaling_fixed_stats.csv")
        extra = ["nccl_avg_bw_mean", "nccl_avg_bw_median"] if any("nccl_avg_bw_mean" in s for s in stats) else []
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=gkeys + STAT_FIELDS + extra, extrasaction="ignore")
            w.writeheader()
            w.writerows(stats)
        print(f"Wrote {out} ({len(stats)} rows)")

        # ─── Summary ───────────────────────────────────────────
        baselines = {}
        for s in stats:
            if s["config"] == "baseline":
                baselines[(s["platform"], s["nodes"])] = s

        print("\n── Overhead summary ──")
        print(f"  {'plat':5s}  {'n':>3s}  {'config':10s}  "
              f"{'t_med(us)':>10s} {'+-std':>7s} {'cv%':>5s}  "
              f"{'bw_med':>7s} {'+-std':>7s}  "
              f"{'t_oh%':>7s}  {'bw_oh%':>7s}")
        print(f"  {'─'*5}  {'─'*3}  {'─'*10}  "
              f"{'─'*10} {'─'*7} {'─'*5}  "
              f"{'─'*7} {'─'*7}  "
              f"{'─'*7}  {'─'*7}")

        for s in stats:
            key = (s["platform"], s["nodes"])
            if key not in baselines:
                continue
            bl = baselines[key]

            if s["config"] == "baseline":
                print(f"  {s['platform']:5s}  {s['nodes']:3d}n  {'baseline':10s}  "
                      f"{s['time_median']:10.1f} {s['time_std']:6.1f}  {s['time_cv']:4.1f}  "
                      f"{s['bw_median']:7.2f} {s['bw_std']:6.2f}  "
                      f"{'—':>7s}  {'—':>7s}")
                continue

            t_oh = (s["time_median"] / bl["time_median"] - 1) * 100 if bl["time_median"] > 0 else 0
            bw_oh = (s["bw_median"] / bl["bw_median"] - 1) * 100 if bl["bw_median"] > 0 else 0

            print(f"  {s['platform']:5s}  {s['nodes']:3d}n  {s['config']:10s}  "
                  f"{s['time_median']:10.1f} {s['time_std']:6.1f}  {s['time_cv']:4.1f}  "
                  f"{s['bw_median']:7.2f} {s['bw_std']:6.2f}  "
                  f"{t_oh:+6.1f}%  {bw_oh:+6.1f}%")

    if sweep_rows:
        gkeys = ["platform", "config", "nodes", "gpus", "size_bytes"]
        stats = aggregate_stats(sweep_rows, gkeys)
        out = Path("scaling_sweep_stats.csv")
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=gkeys + STAT_FIELDS)
            w.writeheader()
            w.writerows(stats)
        print(f"Wrote {out} ({len(stats)} rows)")

if __name__ == "__main__":
    main()
