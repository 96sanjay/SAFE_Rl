import argparse
from pathlib import Path
from math import pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Plot styling
# -----------------------------
plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

# Default color hints (fixed names you care about)
COLORS = {
    "No-Control": "#333333",
    "Default-RBC": "#1f77b4",
    "Advanced-RBC": "#d62728",
    "RBC-EV-ChargeOnly": "#ff7f0e",
    "EV-DepartureAware": "#9467bd",
    "Ref-RBC-V2G": "#8c564b",
    "RL-Agent": "#2ca02c",
    "Solar": "#f1c40f",
}

def build_palette(agent_names):
    """Build palette that never KeyErrors when new agents appear."""
    palette = dict(COLORS)
    cmap = plt.get_cmap("tab20")
    auto_colors = [cmap(i) for i in range(cmap.N)]
    auto_i = 0
    for name in agent_names:
        if name in palette:
            continue
        palette[name] = auto_colors[auto_i % len(auto_colors)]
        auto_i += 1
    return palette

def _export_series(df: pd.DataFrame) -> pd.Series:
    """Prefer explicit grid_export_kwh; else fall back to solar_waste_kwh (proxy)."""
    if "grid_export_kwh" in df.columns:
        return df["grid_export_kwh"]
    if "solar_waste_kwh" in df.columns:
        return df["solar_waste_kwh"]
    return pd.Series(np.zeros(len(df)), index=df.index)

# -----------------------------
# District-level loading
# -----------------------------
def load_data(kpi_path, eval_path):
    data = {}
    kpi_file = Path(kpi_path)
    if not kpi_file.exists():
        print(f"[ERROR] KPI file not found: {kpi_path}")
        return None

    try:
        df = pd.read_csv(kpi_file)
        data["kpi"] = df
        print(f"[INFO] Loaded {len(df)} steps from {kpi_file.name}")
    except Exception as e:
        print(f"[ERROR] Failed to load KPI file: {e}")
        return None

    eval_file = Path(eval_path)
    if eval_file.exists():
        try:
            data["eval"] = pd.read_csv(eval_file)
        except Exception as e:
            print(f"[WARN] Failed to load eval file: {e}")
            data["eval"] = None
    else:
        print(f"[WARN] Eval file not found: {eval_path}")
        data["eval"] = None

    # Backwards compat: if grid_export_kwh missing, create from solar_waste_kwh if present
    if "grid_export_kwh" not in data["kpi"].columns and "solar_waste_kwh" in data["kpi"].columns:
        data["kpi"]["grid_export_kwh"] = data["kpi"]["solar_waste_kwh"]

    # stable plotting index
    data["kpi"] = data["kpi"].copy()
    data["kpi"].index = np.arange(len(data["kpi"]))
    return data

def align_datasets(agents_data):
    valid = {k: v for k, v in agents_data.items() if v is not None}
    if not valid:
        return {}
    min_len = min(len(d["kpi"]) for d in valid.values())
    print(f"[INFO] Aligning all KPI datasets to {min_len} steps")
    for agent in agents_data:
        if agents_data[agent]:
            agents_data[agent]["kpi"] = agents_data[agent]["kpi"].iloc[:min_len].copy()
    return agents_data

# -----------------------------
# District-level plots (core set)
# -----------------------------
def plot_spider_chart(agents_data, baseline, output_dir):
    print("... 1. Spider Chart")
    metrics = [
        "electricity_consumption_total",
        "carbon_emissions_total",
        "cost_total",
        "ramping_average",
        "daily_peak_average",
    ]
    if baseline not in agents_data or agents_data[baseline].get("eval") is None:
        print(f"[WARN] Baseline {baseline} not available for spider chart")
        return

    base_df = agents_data[baseline]["eval"]
    if base_df is None or "cost_function" not in base_df.columns:
        print("[WARN] eval.csv missing expected columns for spider chart")
        return

    base_vals = {}
    for m in metrics:
        vals = base_df[base_df["cost_function"] == m]["value"].values
        if len(vals) == 0:
            print(f"[WARN] Metric {m} not found in baseline eval")
            return
        base_vals[m] = float(vals[0]) if float(vals[0]) != 0 else 1.0

    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    max_val = 0.0

    for agent, data in agents_data.items():
        if data is None or data.get("eval") is None:
            continue
        vals = []
        for m in metrics:
            agent_vals = data["eval"][data["eval"]["cost_function"] == m]["value"].values
            normalized = float(agent_vals[0]) / base_vals[m] if len(agent_vals) > 0 else 1.0
            vals.append(normalized)
            max_val = max(max_val, normalized)

        vals += vals[:1]
        color = COLORS.get(agent, "black")
        ax.plot(angles, vals, linewidth=2, label=agent, color=color)
        ax.fill(angles, vals, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=10)
    ax.set_ylim(0, max(2.0, max_val * 1.1))
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)
    plt.title(f"Performance Overview (Normalized to {baseline})", fontsize=14, pad=20)
    plt.savefig(output_dir / "1_summary_spider.png", dpi=300)
    plt.close()

def plot_cumulative(agents_data, output_dir):
    print("... 6. Cumulative Metrics")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for agent, d in agents_data.items():
        df = d["kpi"].copy()
        df["cum_cost"] = df["step_cost"].cumsum() if "step_cost" in df.columns else 0.0
        df["cum_deficit"] = df["ev_departure_deficit_kwh"].cumsum() if "ev_departure_deficit_kwh" in df.columns else 0.0
        df_plot = df.iloc[::24]
        color = COLORS.get(agent, "black")
        axes[0].plot(df_plot.index, df_plot["cum_cost"], label=agent, color=color, linewidth=2)
        axes[1].plot(df_plot.index, df_plot["cum_deficit"], label=agent, color=color, linewidth=2)

    axes[0].set_title("Cumulative Cost", fontweight="bold")
    axes[0].set_ylabel("Cost")
    axes[0].legend(frameon=True)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Cumulative EV Deficit", fontweight="bold")
    axes[1].set_ylabel("Deficit (kWh)")
    axes[1].set_xlabel("Hour")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "6_cumulative.png", dpi=300)
    plt.close()

def plot_safety_counts(agents_data, output_dir):
    print("... 3. Safety Violation Counts")
    data = []
    for agent, d in agents_data.items():
        df = d["kpi"]
        batt = int((df["battery_abuse_kwh"] > 0.001).sum()) if "battery_abuse_kwh" in df.columns else 0
        evf = int((df["ev_departure_deficit_kwh"] > 0.001).sum()) if "ev_departure_deficit_kwh" in df.columns else 0
        data.append({"Agent": agent, "Type": "Battery Abuse", "Count": batt})
        data.append({"Agent": agent, "Type": "EV Failure", "Count": evf})

    df_plot = pd.DataFrame(data)
    palette = build_palette(df_plot["Agent"].unique().tolist())
    g = sns.catplot(
        data=df_plot, x="Agent", y="Count", col="Type", hue="Agent",
        kind="bar", sharey=False, height=5, aspect=1.2, palette=palette
    )
    g.set_titles("{col_name} Events", fontweight="bold")
    plt.savefig(output_dir / "3_safety_counts.png", dpi=300)
    plt.close()

def plot_magnitudes(agents_data, output_dir):
    print("... 4. Violation Magnitudes")
    data = []
    for agent, d in agents_data.items():
        df = d["kpi"]
        if "ev_departure_deficit_kwh" in df.columns:
            data.append({"Agent": agent, "Metric": "EV Deficit", "Value": df["ev_departure_deficit_kwh"].sum()})
        if "battery_abuse_kwh" in df.columns:
            data.append({"Agent": agent, "Metric": "Battery Abuse", "Value": df["battery_abuse_kwh"].sum()})
        data.append({"Agent": agent, "Metric": "Grid Export (proxy)", "Value": float(_export_series(df).sum())})

    df_plot = pd.DataFrame(data)
    palette = build_palette(df_plot["Agent"].unique().tolist())
    g = sns.catplot(
        data=df_plot, x="Agent", y="Value", col="Metric", hue="Agent",
        kind="bar", sharey=False, height=5, aspect=1.2, palette=palette
    )
    g.set_titles("{col_name}", fontweight="bold")
    plt.savefig(output_dir / "4_magnitudes.png", dpi=300)
    plt.close()

# -----------------------------
# Per-building loading + plots
# -----------------------------
def load_per_building_data(specs):
    out = {}
    for s in specs:
        agent, path = s.split("=", 1)
        agent = agent.strip()
        path = path.strip()
        p = Path(path)
        if not p.exists():
            print(f"[ERROR] Per-building CSV not found: {path}")
            continue
        df = pd.read_csv(p)
        if "building" not in df.columns:
            print(f"[ERROR] Per-building CSV missing 'building' column: {path}")
            continue
        out[agent] = df
        print(f"[INFO] Loaded per-building CSV for {agent}: {p.name} ({len(df)} buildings)")
    return out

def _default_pb_metrics(per_bldg: dict):
    if not per_bldg:
        return []
    agents = list(per_bldg.keys())
    common = set(per_bldg[agents[0]].columns)
    for a in agents[1:]:
        common &= set(per_bldg[a].columns)

    candidates = [
        "ev_departure_deficit_kwh__sum",
        "ev_departure_deficit_kwh__count_gt_0.001",
        "battery_abuse_excess_kwh_equiv__sum",
        "battery_abuse_hours__count_gt_0.95",
        "grid_import_kwh__sum",
        "grid_export_kwh__sum",
        "step_cost__sum",
        "step_carbon_kg__sum",
        "thermal_discomfort__sum",
        "soc_max__max",
        "soc_mean__mean",
    ]
    picked = [c for c in candidates if c in common]
    if not picked:
        df0 = per_bldg[agents[0]]
        numeric = [c for c in df0.columns if c != "building" and pd.api.types.is_numeric_dtype(df0[c])]
        picked = numeric[:6]
    return picked

def plot_per_building_heatmap(per_bldg, output_dir, metric, baseline=None):
    agents = list(per_bldg.keys())
    frames = []
    for a in agents:
        if metric not in per_bldg[a].columns:
            return
        df = per_bldg[a][["building", metric]].copy()
        df["Agent"] = a
        df.rename(columns={metric: "Value"}, inplace=True)
        frames.append(df)

    dfp = pd.concat(frames, ignore_index=True)
    pivot = dfp.pivot(index="building", columns="Agent", values="Value").fillna(0.0)

    if baseline and baseline in pivot.columns:
        pivot = pivot.sort_values(by=baseline, ascending=False)
    else:
        pivot["__sum__"] = pivot.sum(axis=1)
        pivot = pivot.sort_values(by="__sum__", ascending=False).drop(columns="__sum__")

    plt.figure(figsize=(1.2 * max(6, len(agents)), 0.35 * max(10, len(pivot))))
    sns.heatmap(pivot, annot=False, cmap="viridis")
    plt.title(f"Per-building Heatmap: {metric}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"PB_heatmap_{metric}.png", dpi=300)
    plt.close()

def plot_per_building_bars(per_bldg, output_dir, metric, baseline=None, top_n=12):
    agents = list(per_bldg.keys())
    palette = build_palette(agents)
    frames = []
    for a in agents:
        if metric not in per_bldg[a].columns:
            return
        df = per_bldg[a][["building", metric]].copy()
        df["Agent"] = a
        df.rename(columns={metric: "Value"}, inplace=True)
        frames.append(df)
    dfp = pd.concat(frames, ignore_index=True)

    if baseline and baseline in agents:
        base = dfp[dfp["Agent"] == baseline].sort_values("Value", ascending=False)
        top_buildings = base["building"].head(top_n).tolist()
    else:
        top_buildings = dfp.groupby("building")["Value"].mean().sort_values(ascending=False).head(top_n).index.tolist()

    dfp = dfp[dfp["building"].isin(top_buildings)].copy()
    plt.figure(figsize=(14, 6))
    sns.barplot(data=dfp, x="building", y="Value", hue="Agent", palette=palette, order=top_buildings)
    plt.title(f"Per-building (Top {top_n}): {metric}", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"PB_bars_{metric}.png", dpi=300)
    plt.close()

def plot_per_building_totals_summary(per_bldg, output_dir, metric):
    agents = list(per_bldg.keys())
    palette = build_palette(agents)
    rows = []
    for a in agents:
        if metric not in per_bldg[a].columns:
            continue
        rows.append({"Agent": a, "Total": float(per_bldg[a][metric].sum())})
    if not rows:
        return
    dfp = pd.DataFrame(rows)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=dfp, x="Agent", y="Total", palette=palette)
    plt.title(f"Per-building summed total (sanity): {metric}", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"PB_totalcheck_{metric}.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate thesis evaluation plots (district + per-building)")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="District runs: Name=kpi_path,eval_path")
    parser.add_argument("--per_building", nargs="*", default=None,
                        help="Per-building totals: Name=per_building_csv")
    parser.add_argument("--baseline", default="No-Control",
                        help="Baseline agent name for normalization/sorting")
    parser.add_argument("--out", default="runs/plots_thesis_final",
                        help="Output directory for plots")
    parser.add_argument("--pb_metrics", nargs="*", default=None,
                        help="Explicit per-building metric columns to plot")
    parser.add_argument("--pb_topn", type=int, default=12,
                        help="Top-N buildings to show in per-building bar charts")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" THESIS EVALUATION SUITE (PB)")
    print("=" * 60)

    # District
    agents_data = {}
    if args.runs:
        for run_spec in args.runs:
            try:
                name, paths = run_spec.split("=")
                kpi_path, eval_path = paths.split(",")
                print(f"\n[INFO] Loading district run: {name} ...")
                agents_data[name] = load_data(kpi_path, eval_path)
            except ValueError:
                print(f"[ERROR] Invalid run specification: {run_spec}")
                continue

        if agents_data:
            agents_data = align_datasets(agents_data)
            print("\n" + "=" * 60)
            print("GENERATING DISTRICT-LEVEL PLOTS")
            print("=" * 60)
            plot_spider_chart(agents_data, args.baseline, output_dir)
            plot_cumulative(agents_data, output_dir)
            plot_safety_counts(agents_data, output_dir)
            plot_magnitudes(agents_data, output_dir)
        else:
            print("[WARN] No district runs loaded; skipping district plots.")

    # Per-building
    if args.per_building:
        print("\n" + "=" * 60)
        print("GENERATING PER-BUILDING PLOTS")
        print("=" * 60)
        per_bldg = load_per_building_data(args.per_building)
        if per_bldg:
            metrics = args.pb_metrics if args.pb_metrics else _default_pb_metrics(per_bldg)
            print(f"[INFO] Per-building metrics to plot: {metrics}")
            for m in metrics:
                plot_per_building_heatmap(per_bldg, output_dir, m, baseline=args.baseline)
                plot_per_building_bars(per_bldg, output_dir, m, baseline=args.baseline, top_n=args.pb_topn)
                plot_per_building_totals_summary(per_bldg, output_dir, m)
        else:
            print("[WARN] No per-building CSVs loaded; skipping per-building plots.")

    print("\n" + "=" * 60)
    print(f"✓ All plots saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

