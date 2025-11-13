#!/usr/bin/env python3
"""
Plot KPIs from a specific training run.
Usage: python scripts/plot_kpis.py <run_directory>
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np
from pathlib import Path

def plot_kpis(run_dir: str):
    """Plot KPIs from a training run directory."""
    
    # Construct paths
    run_path = Path(run_dir)
    kpi_file = run_path / "kpis_kpis.csv"
    
    if not kpi_file.exists():
        print(f"KPI file not found: {kpi_file}")
        return
    
    # Load data
    print(f"Loading KPI data from: {kpi_file}")
    df = pd.read_csv(kpi_file)
    df['global_step'] = np.arange(len(df))
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Episodes: {df['episode'].min()} to {df['episode'].max()}")
    print(f"Steps: {df['step_count'].min()} to {df['step_count'].max()}")
    
    # Downsample large datasets for plotting to avoid rendering overflow
    stride = max(1, len(df) // 20000)
    plot_df = df.iloc[::stride].copy()

    # Create subplots - expanded to include CityLearn KPIs and reward/violation views
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f'Training KPIs - {run_path.name}', fontsize=16)
    
    # 1. SoC over time
    axes[0, 0].plot(plot_df['step_count'], plot_df['soc_mean'], 'b-', alpha=0.7, linewidth=1)
    # Try to detect SoC band from data, default to [0.0, 0.95] if not found
    soc_min_band = 0.0
    soc_max_band = 0.95
    # Check if we can infer from violation pattern
    if 'cost' in plot_df.columns:
        violating = plot_df[plot_df['cost'] > 0]
        if len(violating) > 0:
            # If violations occur when soc_max > 0.95, that's our upper bound
            if 'soc_max' in violating.columns:
                soc_max_band = 0.95  # Based on current implementation
    axes[0, 0].axhline(y=soc_min_band, color='g', linestyle='--', alpha=0.5, label=f'SoC Min ({soc_min_band})')
    axes[0, 0].axhline(y=soc_max_band, color='r', linestyle='--', label=f'SoC Max ({soc_max_band})')
    axes[0, 0].set_xlabel('Step Count')
    axes[0, 0].set_ylabel('State of Charge (SoC)')
    axes[0, 0].set_title('Battery State of Charge')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Constraint violations over time
    axes[0, 1].plot(plot_df['step_count'], plot_df['constraint_violation'], 'r-', alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('Step Count')
    axes[0, 1].set_ylabel('Constraint Violation (1=violation, 0=safe)')
    axes[0, 1].set_title('Safety Constraint Violations')
    axes[0, 1].set_ylim(-0.1, 1.1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Safety cost over time
    axes[0, 2].plot(plot_df['step_count'], plot_df['cost'], 'orange', alpha=0.7, linewidth=1)
    axes[0, 2].set_xlabel('Step Count')
    axes[0, 2].set_ylabel('Safety Cost')
    axes[0, 2].set_title('Safety Cost Over Time')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Net consumption over time
    if 'step_net_consumption_kwh' in df.columns:
        consumption_data = plot_df['step_net_consumption_kwh']
        ylabel = 'Step Net Consumption (kWh)'
        title = 'Step-by-Step Energy Consumption'
    else:
        consumption_data = df['total_net_consumption_kwh']
        ylabel = 'Total Net Consumption (kWh)'
        title = 'Cumulative Energy Consumption'
    
    axes[1, 0].plot(plot_df['step_count'], consumption_data, 'g-', alpha=0.7, linewidth=1)
    axes[1, 0].set_xlabel('Step Count')
    axes[1, 0].set_ylabel(ylabel)
    axes[1, 0].set_title(title)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Action statistics over time
    axes[1, 1].plot(plot_df['step_count'], plot_df['action_mean'], 'purple', alpha=0.7, linewidth=1, label='Mean')
    axes[1, 1].fill_between(plot_df['step_count'], 
                           plot_df['action_mean'] - plot_df['action_std'], 
                           plot_df['action_mean'] + plot_df['action_std'], 
                           alpha=0.3, color='purple', label='±1 Std')
    axes[1, 1].set_xlabel('Step Count')
    axes[1, 1].set_ylabel('Action Value')
    axes[1, 1].set_title('Battery Control Actions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Non-shiftable load over time (replacing observation stats)
    if 'non_shiftable_load' in plot_df.columns:
        axes[1, 2].plot(plot_df['step_count'], plot_df['non_shiftable_load'], 'brown', alpha=0.7, linewidth=1, label='Non-shiftable Load')
        axes[1, 2].set_xlabel('Step Count')
        axes[1, 2].set_ylabel('Normalized Load')
        axes[1, 2].set_title('Non-shiftable Load')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No observation data available', 
                        ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Observation Statistics (N/A)')
    
    # 7. CityLearn Electricity Consumption (episode-level)
    def episode_series(col: str):
        if col not in df.columns:
            return None
        data = df[df[col] > 0][['episode', col]].copy()
        if data.empty:
            return None
        data = data.groupby('episode', as_index=False).last().sort_values('episode')
        return data

    ep_elec = episode_series('citylearn_electricity_consumption_total')
    if ep_elec is not None:
        axes[2, 0].plot(ep_elec['episode'], ep_elec['citylearn_electricity_consumption_total'], 'g-o', markersize=4)
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Electricity Consumption (ratio)')
        axes[2, 0].set_title('CityLearn: Total Electricity Consumption')
        axes[2, 0].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, 'No CityLearn data', ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('CityLearn: Total Electricity Consumption')
    
    # 8. CityLearn Carbon Emissions (episode-level)
    ep_carbon = episode_series('citylearn_carbon_emissions_total')
    if ep_carbon is not None:
        axes[2, 1].plot(ep_carbon['episode'], ep_carbon['citylearn_carbon_emissions_total'], 'orange', marker='o', markersize=4)
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Carbon Emissions (ratio)')
        axes[2, 1].set_title('CityLearn: Total Carbon Emissions')
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, 'No CityLearn data', ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('CityLearn: Total Carbon Emissions')
    
    # 9. CityLearn Cost (episode-level)
    ep_cost = episode_series('citylearn_cost_total')
    if ep_cost is not None:
        axes[2, 2].plot(ep_cost['episode'], ep_cost['citylearn_cost_total'], 'red', marker='o', markersize=4)
        axes[2, 2].set_xlabel('Episode')
        axes[2, 2].set_ylabel('Total Cost (ratio)')
        axes[2, 2].set_title('CityLearn: Total Energy Cost')
        axes[2, 2].grid(True, alpha=0.3)
    else:
        axes[2, 2].text(0.5, 0.5, 'No CityLearn data', ha='center', va='center', transform=axes[2, 2].transAxes)
        axes[2, 2].set_title('CityLearn: Total Energy Cost')

    # 10. Reward over time
    if 'reward' in df.columns:
        axes[3, 0].plot(plot_df['global_step'], plot_df['reward'], color='teal', alpha=0.4, linewidth=0.8, label='Reward')

        # Rolling mean (window in steps)
        rolling_window = 168  # one week of hourly steps
        rolling = df['reward'].rolling(window=rolling_window, min_periods=1).mean()
        axes[3, 0].plot(
            df['global_step'].iloc[::stride],
            rolling.iloc[::stride],
            color='black',
            linewidth=1.2,
            label=f'{rolling_window}-step MA'
        )

        axes[3, 0].set_xlabel('Global Step')
        axes[3, 0].set_ylabel('Reward')
        axes[3, 0].set_title('Step Reward')
        axes[3, 0].legend(loc='upper right')
        axes[3, 0].grid(True, alpha=0.3)
    else:
        axes[3, 0].text(0.5, 0.5, 'Reward not logged', ha='center', va='center', transform=axes[3, 0].transAxes)
        axes[3, 0].set_title('Step Reward')

    # 11. Cumulative constraint violation percentage
    if 'constraint_violation' in df.columns:
        running_pct = df['constraint_violation'].astype(float).expanding().mean() * 100.0
        axes[3, 1].plot(df['global_step'].iloc[::stride], running_pct.iloc[::stride], color='magenta', alpha=0.7, linewidth=1.2)
        axes[3, 1].set_xlabel('Global Step')
        axes[3, 1].set_ylabel('Violations (%)')
        axes[3, 1].set_title('Cumulative Violation Percentage')
        axes[3, 1].set_ylim(0, 100)
        axes[3, 1].grid(True, alpha=0.3)
    else:
        axes[3, 1].text(0.5, 0.5, 'Constraint data missing', ha='center', va='center', transform=axes[3, 1].transAxes)
        axes[3, 1].set_title('Cumulative Violation Percentage')

    # 12. CityLearn Zero Net Energy (episode-level)
    ep_zne = episode_series('citylearn_zero_net_energy')
    if ep_zne is not None:
        axes[3, 2].plot(ep_zne['episode'], ep_zne['citylearn_zero_net_energy'], 'c-o', markersize=4)
        axes[3, 2].set_xlabel('Episode')
        axes[3, 2].set_ylabel('Zero Net Energy (ratio)')
        axes[3, 2].set_title('CityLearn: Zero Net Energy')
        axes[3, 2].grid(True, alpha=0.3)
    else:
        axes[3, 2].text(0.5, 0.5, 'No CityLearn data', ha='center', va='center', transform=axes[3, 2].transAxes)
        axes[3, 2].set_title('CityLearn: Zero Net Energy')
    
    plt.tight_layout()
    
    # Save plot
    output_file = run_path / "kpi_plots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total steps: {len(df)}")
    print(f"Constraint violations: {df['constraint_violation'].sum()} / {len(df)} ({df['constraint_violation'].mean()*100:.1f}%)")
    print(f"Average SoC: {df['soc_mean'].mean():.3f}")
    print(f"SoC range: {df['soc_min'].min():.3f} - {df['soc_max'].max():.3f}")
    print(f"Average safety cost: {df['cost'].mean():.3f}")
    if 'step_net_consumption_kwh' in df.columns:
        print(f"Average step energy consumption: {df['step_net_consumption_kwh'].mean():.3f} kWh")
        print(f"Step energy consumption range: {df['step_net_consumption_kwh'].min():.3f} to {df['step_net_consumption_kwh'].max():.3f} kWh")
    else:
        print(f"Total energy consumption: {df['total_net_consumption_kwh'].max():.1f} kWh")
    print(f"Average action: {df['action_mean'].mean():.3f} ± {df['action_std'].mean():.3f}")
    if 'reward' in df.columns:
        print(f"Average reward: {df['reward'].mean():.3f}")
        running_pct = df['constraint_violation'].astype(float).expanding().mean() * 100.0
        print(f"Cumulative violation percentage (final): {running_pct.iloc[-1]:.1f}%")
    
    # CityLearn KPI statistics
    print("\n=== CityLearn KPIs ===")
    citylearn_columns = [col for col in df.columns if col.startswith('citylearn_')]
    if citylearn_columns:
        for col in citylearn_columns:
            # Get non-zero values (episode-end data)
            non_zero_data = df[df[col] > 0][col]
            if not non_zero_data.empty:
                print(f"{col}: {non_zero_data.iloc[-1]:.3f} (latest episode)")
            else:
                print(f"{col}: No data available")
    else:
        print("No CityLearn KPIs found in data")

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/plot_kpis.py <run_directory>")
        print("Example: python scripts/plot_kpis.py runs/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2025-10-09-18-21-46")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    plot_kpis(run_dir)

if __name__ == "__main__":
    main()
