#!/usr/bin/env python3
"""
Comprehensive analysis script for multiple algorithms.
Generates KPI plots and day dynamics for RCPO, CPO, PCPO, TD3, TD3Lag, SAC.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

plt.rcParams['agg.path.chunksize'] = 10000

# Algorithms to analyze
ALGORITHMS = {
    'RCPO': 'RCPO',
    'CPO': 'CPO',
    'PCPO': 'PCPO',
    'TD3': 'TD3',
    'TD3Lag': 'TD3Lag',
    'SAC': 'SAC'
}

COLORS = {
    'RCPO': '#e377c2',      # Pink
    'CPO': '#9467bd',       # Purple
    'PCPO': '#8c564b',      # Brown
    'TD3': '#2ca02c',       # Green
    'TD3Lag': '#ff7f0e',    # Orange
    'SAC': '#1f77b4'        # Blue
}

def find_latest_runs(runs_dir="runs"):
    """Find latest runs for each algorithm."""
    latest_runs = {}
    for algo_name, algo_pattern in ALGORITHMS.items():
        pattern = os.path.join(runs_dir, f"{algo_pattern}-*")
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            algo_dir = matches[0]
            seed_dirs = [d for d in os.listdir(algo_dir) 
                         if os.path.isdir(os.path.join(algo_dir, d)) and d.startswith('seed-')]
            if seed_dirs:
                seed_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(algo_dir, x)), reverse=True)
                latest_runs[algo_name] = os.path.join(algo_dir, seed_dirs[0])
    return latest_runs

def plot_kpis_for_algorithm(run_dir, algo_name, output_dir):
    """Plot KPIs for a single algorithm."""
    run_path = Path(run_dir)
    kpi_file = run_path / "kpis_kpis.csv"
    
    if not kpi_file.exists():
        print(f"  ✗ KPI file not found: {kpi_file}")
        return None
    
    # Load data
    df = pd.read_csv(kpi_file)
    df['global_step'] = np.arange(len(df))
    
    # Downsample for plotting
    stride = max(1, len(df) // 20000)
    plot_df = df.iloc[::stride].copy()
    
    # Detect SoC band
    soc_min_band = 0.0
    soc_max_band = 0.95
    
    # Create figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f'{algo_name} - Training KPIs', fontsize=16, fontweight='bold')
    
    # 1. SoC over time
    axes[0, 0].plot(plot_df['step_count'], plot_df['soc_mean'], 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].axhline(y=soc_max_band, color='r', linestyle='--', linewidth=2, label=f'Max SoC ({soc_max_band})')
    axes[0, 0].axhline(y=soc_min_band, color='r', linestyle='--', linewidth=2, label=f'Min SoC ({soc_min_band})')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('SoC')
    axes[0, 0].set_title('State of Charge (SoC)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Reward over time
    if 'reward' in plot_df.columns:
        axes[0, 1].plot(plot_df['step_count'], plot_df['reward'], 'g-', alpha=0.7, linewidth=1)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Reward per Step')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Cost over time
    if 'cost' in plot_df.columns:
        axes[0, 2].plot(plot_df['step_count'], plot_df['cost'], 'r-', alpha=0.7, linewidth=1)
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Cost')
        axes[0, 2].set_title('Cost per Step')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Action over time
    if 'action_mean' in plot_df.columns:
        axes[1, 0].plot(plot_df['step_count'], plot_df['action_mean'], 'm-', alpha=0.7, linewidth=1)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Action')
        axes[1, 0].set_title('Action Mean')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Grid Import
    if 'grid_import_kwh' in plot_df.columns:
        axes[1, 1].plot(plot_df['step_count'], plot_df['grid_import_kwh'], 'orange', alpha=0.7, linewidth=1)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Import (kWh)')
        axes[1, 1].set_title('Grid Import')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Grid Export
    if 'grid_export_kwh' in plot_df.columns:
        axes[1, 2].plot(plot_df['step_count'], plot_df['grid_export_kwh'], 'cyan', alpha=0.7, linewidth=1)
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Export (kWh)')
        axes[1, 2].set_title('Grid Export')
        axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Solar Generation
    if 'solar_generation_kwh' in plot_df.columns:
        axes[2, 0].plot(plot_df['step_count'], plot_df['solar_generation_kwh'], 'y-', alpha=0.7, linewidth=1)
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('Generation (kWh)')
        axes[2, 0].set_title('Solar Generation')
        axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Non-shiftable Load (NOTE: This is NORMALIZED 0-1, not raw kWh)
    if 'non_shiftable_load' in plot_df.columns:
        axes[2, 1].plot(plot_df['step_count'], plot_df['non_shiftable_load'], 'brown', alpha=0.7, linewidth=1)
        axes[2, 1].set_xlabel('Step')
        axes[2, 1].set_ylabel('Load (Normalized 0-1)')
        axes[2, 1].set_title('Non-shiftable Load (Normalized)')
        axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Episode Rewards
    if 'episode' in plot_df.columns and 'reward' in plot_df.columns:
        episode_rewards = plot_df.groupby('episode')['reward'].sum()
        axes[2, 2].plot(episode_rewards.index, episode_rewards.values, 'g-', alpha=0.7, linewidth=2)
        axes[2, 2].set_xlabel('Episode')
        axes[2, 2].set_ylabel('Total Reward')
        axes[2, 2].set_title('Episode Total Reward')
        axes[2, 2].grid(True, alpha=0.3)
    
    # 10. CityLearn KPIs
    citylearn_cols = [col for col in plot_df.columns if col.startswith('citylearn_')]
    if citylearn_cols:
        for idx, col in enumerate(citylearn_cols[:3]):
            row = 3
            col_idx = idx
            if col_idx >= 3:
                break
            values = plot_df[col].dropna()
            if len(values) > 0:
                # Get episode numbers for these values
                episodes = plot_df.loc[values.index, 'episode'].values
                axes[row, col_idx].plot(episodes, values.values, 'purple', alpha=0.7, linewidth=2, marker='o', markersize=3)
                axes[row, col_idx].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0)')
                axes[row, col_idx].set_xlabel('Episode')
                axes[row, col_idx].set_ylabel('Ratio')
                axes[row, col_idx].set_title(col.replace('citylearn_', '').replace('_', ' ').title())
                axes[row, col_idx].legend()
                axes[row, col_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f"{algo_name}_kpis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ KPI plot saved: {output_file}")
    return output_file

def find_common_day(latest_runs):
    """Find a common day (month and day_type) across all algorithms."""
    common_days = None
    
    for algo_name, run_path in latest_runs.items():
        kpi_file = Path(run_path) / "kpis_kpis.csv"
        if kpi_file.exists():
            df = pd.read_csv(kpi_file)
            if 'month' in df.columns and 'day_type' in df.columns:
                # Get unique (month, day_type) combinations from last 25% of episodes
                available_episodes = sorted(df['episode'].unique())
                if len(available_episodes) > 0:
                    last_quarter_start = int(len(available_episodes) * 0.75)
                    candidate_episodes = available_episodes[last_quarter_start:]
                    # Get days from candidate episodes
                    candidate_data = df[df['episode'].isin(candidate_episodes)]
                    days = candidate_data[['month', 'day_type']].drop_duplicates()
                    if common_days is None:
                        common_days = set(zip(days['month'], days['day_type']))
                    else:
                        common_days = common_days.intersection(set(zip(days['month'], days['day_type'])))
    
    if common_days and len(common_days) > 0:
        # Pick the first common day (prefer a weekday if available)
        # day_type: 0=Monday, 1=Tuesday, ..., 6=Sunday
        sorted_days = sorted(common_days, key=lambda x: (x[0], x[1]))
        return sorted_days[0]  # Return (month_norm, day_type_norm) - normalized values
    return None

def load_raw_data_from_building_csv(building_csv_path="data/citylearn/Building_5.csv"):
    """Load raw load and solar data from building CSV file."""
    if not Path(building_csv_path).exists():
        return None, None
    df_building = pd.read_csv(building_csv_path)
    # Create lookups: (month, day_type, hour) -> value
    # Note: hour in CSV is 1-24, we'll map to 0-23 for indexing
    load_lookup = {}
    solar_lookup = {}
    for _, row in df_building.iterrows():
        month = int(row['month'])
        day_type = int(row['day_type'])
        hour_csv = int(row['hour'])  # 1-24 in CSV
        hour_idx = (hour_csv - 1) % 24  # Convert to 0-23
        load_kw = float(row['non_shiftable_load'])
        solar_kw = float(row['solar_generation'])
        load_lookup[(month, day_type, hour_idx)] = load_kw
        solar_lookup[(month, day_type, hour_idx)] = solar_kw
    return load_lookup, solar_lookup

def plot_day_dynamics(run_dir, algo_name, output_dir, day_episode=None, day_start_idx=None, target_month_norm=None, target_day_type_norm=None, target_month_raw=None, target_day_type_raw=None):
    """Plot day dynamics for a single algorithm."""
    run_path = Path(run_dir)
    kpi_file = run_path / "kpis_kpis.csv"
    
    if not kpi_file.exists():
        print(f"  ✗ KPI file not found: {kpi_file}")
        return None
    
    # Load data
    df = pd.read_csv(kpi_file)
    
    # Load raw load and solar lookup from building CSV
    load_lookup, solar_lookup = load_raw_data_from_building_csv()
    
    # Select episode - prefer one with target month and day_type
    day_start_idx = None  # Initialize
    if day_episode is None:
        available_episodes = sorted(df['episode'].unique())
        if len(available_episodes) > 0:
            # If we have target month/day_type, find an episode that CONTAINS that day
            # and extract the 24 hours corresponding to that day
            if target_month_norm is not None and target_day_type_norm is not None:
                # Look in ALL episodes to find one that contains the target day
                # Prefer episodes from last 25% but search all if needed
                last_quarter_start = int(len(available_episodes) * 0.75)
                # Search last 25% first, then all others
                candidate_episodes = list(reversed(available_episodes[last_quarter_start:])) + list(reversed(available_episodes[:last_quarter_start]))
                
                day_episode = None
                
                for ep in candidate_episodes:
                    ep_data = df[df['episode'] == ep]
                    if len(ep_data) < 24:
                        continue
                    
                    # Search through the episode to find where the target day occurs
                    # Look for a 24-hour window where month and day_type match the target
                    for start_idx in range(len(ep_data) - 23):  # Need at least 24 hours
                        window = ep_data.iloc[start_idx:start_idx+24]
                        # Check if this window matches the target day
                        # Find first non-null month/day_type in the window
                        month_match = False
                        day_type_match = False
                        for _, row in window.iterrows():
                            month_val = row.get('month')
                            day_type_val = row.get('day_type')
                            if pd.notna(month_val) and pd.notna(day_type_val):
                                if abs(month_val - target_month_norm) < 0.001:
                                    month_match = True
                                if abs(day_type_val - target_day_type_norm) < 0.001:
                                    day_type_match = True
                                if month_match and day_type_match:
                                    # Found a window matching the target day!
                                    day_episode = ep
                                    day_start_idx = start_idx
                                    break
                        if day_episode is not None:
                            break
                    if day_episode is not None:
                        break
                
                if day_episode is not None:
                    # Verify it has import/export data
                    ep_data = df[df['episode'] == day_episode]
                    window_data = ep_data.iloc[day_start_idx:day_start_idx+24]
                    if 'grid_import_kwh' in window_data.columns:
                        import_nonzero = (window_data['grid_import_kwh'] > 0).sum()
                        export_nonzero = (window_data['grid_export_kwh'] > 0).sum() if 'grid_export_kwh' in window_data.columns else 0
                        print(f"  Selected episode {day_episode}, hours {day_start_idx}-{day_start_idx+23} (matches target day, import={import_nonzero} non-zero, export={export_nonzero} non-zero)")
                    else:
                        print(f"  Selected episode {day_episode}, hours {day_start_idx}-{day_start_idx+23} (matches target day)")
                else:
                    # Fallback: use last 25% episode, first 24 hours
                    day_episode = available_episodes[last_quarter_start]
                    day_start_idx = 0
                    print(f"  ⚠️  Selected episode {day_episode} (from last 25%, target day not found - import/export may not match load/solar day)")
            else:
                # Use an episode from the last 25% of training
                last_quarter_start = int(len(available_episodes) * 0.75)
                day_episode = available_episodes[last_quarter_start]
                print(f"  Selected episode {day_episode} (from last 25% of training)")
        else:
            print(f"  ✗ No episodes found")
            return None
    
    # Filter data for selected episode
    episode_data = df[df['episode'] == day_episode].copy()
    
    if len(episode_data) == 0:
        print(f"  ✗ No data for episode {day_episode}")
        return None
    
    # Extract the specific 24-hour window for the target day
    # If day_start_idx was set, use it; otherwise use first 24 hours
    if day_start_idx is not None:
        if len(episode_data) >= day_start_idx + 24:
            day_data = episode_data.iloc[day_start_idx:day_start_idx+24].copy()
            print(f"  Using hours {day_start_idx}-{day_start_idx+23} of episode (total: {len(episode_data)} steps)")
        else:
            day_data = episode_data.iloc[day_start_idx:].copy()
            print(f"  Using hours {day_start_idx}-{len(episode_data)-1} of episode (less than 24 hours)")
    else:
        # Fallback: use first 24 hours
        if len(episode_data) >= 24:
            day_data = episode_data.head(24).copy()
            print(f"  Using first 24 hours of episode (total: {len(episode_data)} steps)")
        else:
            day_data = episode_data.copy()
            print(f"  Using all {len(day_data)} steps (less than 24 hours)")
    
    # Get month and day_type - use target if provided, otherwise from first row
    if target_month_raw is not None and target_day_type_raw is not None:
        # Use the target day (same for all algorithms)
        plot_month = target_month_raw
        plot_day_type = target_day_type_raw
        print(f"  Using target day: Month {plot_month}, Day Type {plot_day_type}")
    else:
        # Fallback: get from first row (denormalize if needed)
        first_row = day_data.iloc[0]
        month_norm = first_row.get('month', 0.5) if pd.notna(first_row.get('month')) else 0.5
        day_type_norm = first_row.get('day_type', 0.5) if pd.notna(first_row.get('day_type')) else 0.5
        
        # Denormalize: month [0,1] -> [1,12], day_type [0,1] -> [1,7]
        plot_month = max(1, min(12, int(round(month_norm * 11 + 1))))
        plot_day_type = max(1, min(7, int(round(day_type_norm * 6 + 1))))
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # day_type in CSV is 1-7, but we need 0-6 for indexing
    day_idx = (plot_day_type - 1) % 7
    day_name = day_names[day_idx] if 0 <= day_idx < 7 else f"Day {plot_day_type}"
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_name = month_names[plot_month] if 1 <= plot_month <= 12 else f"Month {plot_month}"
    
    # Extract raw load and solar from building CSV for the selected day
    # Use the denormalized month and day_type we calculated
    if load_lookup and solar_lookup:
        raw_loads = []
        raw_solars = []
        for hour in range(len(day_data)):
            key = (plot_month, plot_day_type, hour)
            if key in load_lookup:
                # Load is in kW, convert to kWh (1 hour timestep)
                load_kwh = load_lookup[key] * 1.0
                raw_loads.append(load_kwh)
            else:
                raw_loads.append(0.0)
            
            if key in solar_lookup:
                # Solar is in kW, convert to kWh (1 hour timestep)
                solar_kwh = solar_lookup[key] * 1.0
                raw_solars.append(solar_kwh)
            else:
                raw_solars.append(0.0)
        
        day_data['non_shiftable_load_kwh'] = raw_loads
        day_data['solar_generation_kwh_raw'] = raw_solars
        print(f"  ✓ Extracted raw load and solar from building CSV for {month_name} {day_name} (24 hours)")
    elif 'non_shiftable_load_kwh' not in day_data.columns:
        print(f"  ⚠️  Raw load/solar not available from building CSV")
    
    # Get hours (assuming 1 step = 1 hour)
    hours = np.arange(len(day_data))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{algo_name} - Day Dynamics (Episode {day_episode}, {month_name} {day_name})', 
                 fontsize=16, fontweight='bold')
    
    # 1. SoC trajectory
    ax = axes[0, 0]
    if 'soc_mean' in day_data.columns:
        ax.plot(hours, day_data['soc_mean'], linewidth=2, label='SoC', color=COLORS.get(algo_name, 'blue'))
        ax.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Max SoC (0.95)')
        ax.axhline(y=0.0, color='r', linestyle='--', linewidth=2, label='Min SoC (0.0)')
        ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax.set_ylabel('State of Charge', fontsize=11, fontweight='bold')
        ax.set_title('SoC Trajectory Over Day', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)
    
    # 2. Energy flows
    ax = axes[0, 1]
    # Use raw load and solar from building CSV (same for all algorithms)
    if 'non_shiftable_load_kwh' in day_data.columns:
        ax.plot(hours, day_data['non_shiftable_load_kwh'], 'brown', linewidth=2, label='Load (kWh)', alpha=0.8)
    if 'solar_generation_kwh_raw' in day_data.columns:
        # Raw solar from building CSV (positive values)
        ax.plot(hours, day_data['solar_generation_kwh_raw'], 'y-', linewidth=2, label='Solar (kWh)', alpha=0.8)
    # Agent-dependent values
    if 'grid_import_kwh' in day_data.columns:
        ax.plot(hours, day_data['grid_import_kwh'], 'g-', linewidth=2, label='Import (kWh)', alpha=0.8)
    if 'grid_export_kwh' in day_data.columns:
        ax.plot(hours, day_data['grid_export_kwh'], 'r-', linewidth=2, label='Export (kWh)', alpha=0.8)
    ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Energy (kWh)', fontsize=11, fontweight='bold')
    ax.set_title(f'Energy Flows Over Day - {month_name} {day_name} (Load & Solar from Building CSV)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    
    # 3. Actions (bar plot for charging/discharging)
    ax = axes[1, 0]
    if 'action_mean' in day_data.columns:
        actions = day_data['action_mean'].values
        # Create colors: green for charging (positive), red for discharging (negative)
        colors = ['green' if a > 0 else 'red' if a < 0 else 'gray' for a in actions]
        ax.bar(hours, actions, color=colors, alpha=0.7, width=0.8, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax.set_ylabel('Action Value', fontsize=11, fontweight='bold')
        ax.set_title('Battery Actions Over Day (Green=Charging, Red=Discharging)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(-0.5, 24)
    
    # 4. Reward and Cost
    # Reward = CityLearn reward (negative of net electricity consumption)
    #   - Minimizes grid imports (negative reward for importing)
    #   - Maximizes grid exports (zero reward when exporting)
    # Cost = Constraint violation cost (SoC band violation)
    #   - Binary mode: 1.0 if SoC outside [0.0, 0.95], 0.0 otherwise
    #   - Hinge mode: magnitude of violation normalized by band width
    #   - Measures how far SoC is outside the safety band
    ax = axes[1, 1]
    if 'reward' in day_data.columns:
        ax2 = ax.twinx()
        ax.plot(hours, day_data['reward'], 'g-', linewidth=2, label='Reward (CityLearn)', alpha=0.8)
        ax.set_ylabel('Reward (minimize imports)', fontsize=11, fontweight='bold', color='g')
        ax.tick_params(axis='y', labelcolor='g')
    if 'cost' in day_data.columns:
        if 'reward' not in day_data.columns:
            ax2 = ax
        ax2.plot(hours, day_data['cost'], 'r-', linewidth=2, label='Cost (SoC Violation)', alpha=0.8, marker='o', markersize=3)
        ax2.set_ylabel('Cost (SoC outside [0.0, 0.95])', fontsize=11, fontweight='bold', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax.set_title('Reward (CityLearn) and Constraint Violation Cost Over Day', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    
    plt.tight_layout()
    output_file = output_dir / f"{algo_name}_day_dynamics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Day dynamics plot saved: {output_file}")
    return output_file

def plot_comparison_kpis(latest_runs, output_dir):
    """Create comparison plots for all algorithms."""
    print("\n" + "="*120)
    print("CREATING COMPARISON PLOTS")
    print("="*120)
    
    # Collect CityLearn KPIs
    all_kpi_data = {}
    
    for algo_name, run_path in latest_runs.items():
        kpi_file = Path(run_path) / "kpis_kpis.csv"
        if kpi_file.exists():
            df = pd.read_csv(kpi_file)
            citylearn_cols = [col for col in df.columns if col.startswith('citylearn_')]
            if citylearn_cols:
                episode_rows = df[
                    df[citylearn_cols[0]].notna() & 
                    (df[citylearn_cols[0]] != 0.0) &
                    (df[citylearn_cols[0]].abs() > 1e-6)
                ]
                if len(episode_rows) > 0:
                    algo_kpis = {}
                    for col in citylearn_cols:
                        values = episode_rows[col].dropna()
                        values = values[values.abs() > 1e-6]
                        if len(values) > 0:
                            episodes = episode_rows['episode'].astype(int).tolist()
                            valid_indices = [i for i, v in enumerate(episode_rows[col]) 
                                           if pd.notna(v) and abs(v) > 1e-6]
                            valid_episodes = [episodes[i] for i in valid_indices]
                            valid_values = [float(v) for i, v in enumerate(episode_rows[col]) if i in valid_indices]
                            
                            algo_kpis[col] = {
                                'episodes': valid_episodes,
                                'values': valid_values,
                                'mean': float(np.mean(valid_values)),
                                'min': float(np.min(valid_values)),
                                'max': float(np.max(valid_values))
                            }
                    all_kpi_data[algo_name] = algo_kpis
    
    # Create comparison plot
    if all_kpi_data:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CityLearn KPIs Comparison - All Algorithms', fontsize=16, fontweight='bold')
        
        kpi_configs = [
            ('citylearn_electricity_consumption_total', 'Electricity Consumption\n(Ratio vs Baseline)', 0, 0, True),
            ('citylearn_carbon_emissions_total', 'Carbon Emissions\n(Ratio vs Baseline)', 0, 1, True),
            ('citylearn_cost_total', 'Total Cost\n(Ratio vs Baseline)', 1, 0, True),
            ('citylearn_zero_net_energy', 'Zero Net Energy\n(Metric)', 1, 1, False)
        ]
        
        for kpi_key, kpi_title, row, col, lower_is_better in kpi_configs:
            ax = axes[row, col]
            
            for algo_name in ALGORITHMS.keys():
                if algo_name in all_kpi_data and kpi_key in all_kpi_data[algo_name]:
                    data = all_kpi_data[algo_name][kpi_key]
                    episodes = data['episodes']
                    values = data['values']
                    ax.plot(episodes, values, 'o-', label=algo_name, color=COLORS[algo_name], 
                           alpha=0.7, markersize=5, linewidth=2)
            
            if lower_is_better:
                ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0)', alpha=0.7)
            
            ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
            ax.set_ylabel('Ratio', fontsize=11, fontweight='bold')
            ax.set_title(kpi_title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / "citylearn_kpis_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Comparison plot saved: {output_file}")

def main():
    """Main function."""
    print("="*120)
    print("COMPREHENSIVE KPI AND DAY DYNAMICS ANALYSIS")
    print("="*120)
    
    # Find latest runs
    latest_runs = find_latest_runs()
    print(f"\nFound {len(latest_runs)} algorithms:")
    for algo, path in latest_runs.items():
        print(f"  {algo}: {path}")
    
    # Create output directory
    output_dir = Path("comprehensive_analysis")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate individual KPI plots
    print("\n" + "="*120)
    print("GENERATING INDIVIDUAL KPI PLOTS")
    print("="*120)
    for algo_name, run_path in latest_runs.items():
        print(f"\n{algo_name}:")
        plot_kpis_for_algorithm(run_path, algo_name, output_dir)
    
    # Find common day across all algorithms and force same day for all
    print("\n" + "="*120)
    print("FINDING COMMON DAY ACROSS ALL ALGORITHMS")
    print("="*120)
    common_day = find_common_day(latest_runs)
    if common_day:
        target_month_norm, target_day_type_norm = common_day
        # Denormalize to get raw values for building CSV lookup
        target_month_raw = max(1, min(12, int(round(target_month_norm * 11 + 1))))
        target_day_type_raw = max(1, min(7, int(round(target_day_type_norm * 6 + 1))))
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_idx = (target_day_type_raw - 1) % 7
        day_name = day_names[day_idx] if 0 <= day_idx < 7 else f"Day {target_day_type_raw}"
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_name = month_names[target_month_raw] if 1 <= target_month_raw <= 12 else f"Month {target_month_raw}"
        print(f"\n✓ Found common day: {month_name} {day_name} (Month {target_month_raw}, Day Type {target_day_type_raw})")
        print(f"  Using this EXACT day for all algorithms for fair comparison")
        print(f"  All algorithms will show same load and solar generation")
    else:
        target_month_norm, target_day_type_norm = None, None
        target_month_raw, target_day_type_raw = None, None
        print(f"\n⚠️  No common day found, using last 25% episodes")
    
    # Generate day dynamics plots
    print("\n" + "="*120)
    print("GENERATING DAY DYNAMICS PLOTS")
    print("="*120)
    for algo_name, run_path in latest_runs.items():
        print(f"\n{algo_name}:")
        # Find episode and day_start_idx for this algorithm
        result = plot_day_dynamics(run_path, algo_name, output_dir, 
                                   target_month_norm=target_month_norm if common_day else None,
                                   target_day_type_norm=target_day_type_norm if common_day else None,
                                   target_month_raw=target_month_raw if common_day else None,
                                   target_day_type_raw=target_day_type_raw if common_day else None)
    
    # Generate comparison plots
    plot_comparison_kpis(latest_runs, output_dir)
    
    print("\n" + "="*120)
    print("ANALYSIS COMPLETE")
    print("="*120)
    print(f"All plots saved to: {output_dir}")
    print("="*120)

if __name__ == "__main__":
    main()

