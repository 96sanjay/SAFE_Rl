
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict

# Set publication-quality style for academic use
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.facecolor': 'white'
})

def load_data():
    """Loads the district and per-building KPI CSV files."""
    dist_csv = Path("runs/baselines/intelligent_rbc/kpis_district_v2.csv")
    full_csv = Path("runs/baselines/intelligent_rbc/kpis_with_buildings_v2.csv")
    
    if not dist_csv.exists():
        raise FileNotFoundError(f"District CSV not found at: {dist_csv}")
    
    df_dist = pd.read_csv(dist_csv)
    df_full = pd.read_csv(full_csv) if full_csv.exists() else None
    return df_dist, df_full

def extract_metrics(df):
    """Calculates summary statistics and final CityLearn metrics."""
    metrics = {}
    final_step = df.iloc[-1]
    
    # Normalized CityLearn Metrics (Official Challenge KPIs)
    cl_keys = [
        'citylearn_electricity_consumption_total', 'citylearn_carbon_emissions_total',
        'citylearn_cost_total', 'citylearn_daily_peak_average', 
        'citylearn_all_time_peak_average', 'citylearn_ramping_average', 
        'citylearn_discomfort_proportion', 'citylearn_zero_net_energy'
    ]
    for key in cl_keys:
        if key in final_step:
            metrics[key] = final_step[key]
            
    # Physical/Raw Metrics (Engineering Units)
    metrics['raw_total_consumption_kwh'] = df['grid_import_kwh'].sum()
    metrics['raw_total_cost_usd'] = df['step_cost'].sum()
    metrics['cmdp_total_cost'] = df['cost'].sum()
    metrics['cmdp_violation_rate'] = (df['constraint_violation'] > 0).sum() / len(df) * 100
    metrics['battery_worst_soc'] = df['soc_max'].max()
    
    # EV Reliability Metrics
    deps = (df['ev_departure_departures'] > 0).sum()
    defs = (df['ev_departure_deficit_kwh'] > 0).sum()
    metrics['ev_satisfaction_rate'] = (deps - defs) / deps * 100 if deps > 0 else 0
    metrics['ev_total_deficit_kwh'] = df['ev_departure_deficit_kwh'].sum()
    
    return metrics

def plot_all(metrics, df_dist, df_full, out_dir):
    """Generates the three primary visualization assets for the thesis."""
    
    # --- 1. Normalized Metrics Bar (Baseline Comparison) ---
    plt.figure(figsize=(12, 6))
    keys = [k.replace('citylearn_', '').replace('_', '\n') for k in metrics if k.startswith('citylearn')]
    vals = [metrics[k] for k in metrics if k.startswith('citylearn')]
    plt.bar(keys, vals, color='steelblue', alpha=0.8)
    plt.axhline(1.0, color='red', ls='--', label='Baseline Ref (1.0)')
    plt.title("CityLearn Normalized Evaluation Metrics (Lower is Better)")
    plt.tight_layout()
    plt.savefig(out_dir / "01_citylearn_metrics.png")
    plt.close()
    
    # --- 2. Hourly Pattern Dashboard (District Temporal Analysis) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    hourly = df_dist.groupby('hour').mean()
    
    # Top Left: CMDP Cost
    axes[0,0].bar(hourly.index, hourly['cost'], color='firebrick', alpha=0.7)
    axes[0,0].set_title("Avg CMDP Cost by Hour")
    
    # Top Right: Battery SoC
    axes[0,1].plot(hourly.index, hourly['soc_max'], 'o-', color='darkblue')
    axes[0,1].axhline(0.95, color='red', ls='--', label='Safety Limit')
    axes[0,1].set_title("Avg Max Battery SoC by Hour")
    
    # Bottom Left: Grid Import
    axes[1,0].bar(hourly.index, hourly['grid_import_kwh'], color='green', alpha=0.7)
    axes[1,0].set_title("Avg Grid Import (kWh) by Hour")
    
    # Bottom Right: EV Deficits
    hourly_ev = df_dist.groupby('hour')['ev_departure_deficit_kwh'].sum()
    axes[1,1].bar(hourly_ev.index, hourly_ev.values, color='orange', alpha=0.7)
    axes[1,1].set_title("Total EV Deficit Events by Hour")
    
    plt.tight_layout()
    plt.savefig(out_dir / "02_hourly_dashboard.png")
    plt.close()

    # --- 3. Per-Building Analysis (Spatial Heterogeneity) ---
    if df_full is not None:
        b_data = []
        for b in range(17):
            soc_col = f'soc_b{b}'
            def_col = f'ev_deficit_kwh_b{b}'
            if soc_col in df_full.columns:
                b_data.append({
                    'Bldg': b, 
                    'Violation_Rate': (df_full[soc_col] > 0.95).mean() * 100,
                    'Deficit': df_full[def_col].sum() if def_col in df_full.columns else 0
                })
        b_df = pd.DataFrame(b_data)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(b_df['Bldg'].astype(str), b_df['Violation_Rate'], color='red', alpha=0.5, label='SoC Violation %')
        ax1.set_ylabel('Violation Rate (%)')
        
        ax2 = ax1.twinx()
        ax2.plot(b_df['Bldg'].astype(str), b_df['Deficit'], 'D-', color='black', label='EV Deficit (kWh)')
        ax2.set_ylabel('Total Deficit (kWh)')
        
        plt.title("Per-Building Safety (SoC) vs. Reliability (EV Deficit)")
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        plt.tight_layout()
        plt.savefig(out_dir / "03_building_analysis.png")
        plt.close()

def save_tables(metrics, out_dir):
    """Exports the summary metrics to a LaTeX table for direct thesis inclusion."""
    df_m = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    df_m.to_latex(out_dir / "summary_metrics.tex", index=False)
    print(f"✅ LaTeX Table saved: {out_dir / 'summary_metrics.tex'}")

def main():
    # Define and create output directory
    out_dir = Path("runs/baselines/intelligent_rbc/comprehensive_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis pipeline
    try:
        df_dist, df_full = load_data()
        metrics = extract_metrics(df_dist)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE BASELINE ANALYSIS COMPLETE".center(80))
        print("="*80)
        
        plot_all(metrics, df_dist, df_full, out_dir)
        save_tables(metrics, out_dir)
        
        print(f"\n📁 Assets generated in: {out_dir}")
        print(f"   • Plots: 01_citylearn_metrics.png, 02_hourly_dashboard.png, 03_building_analysis.png")
        print(f"   • LaTeX: summary_metrics.tex")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()