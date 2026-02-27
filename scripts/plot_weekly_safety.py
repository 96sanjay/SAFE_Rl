import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from pathlib import Path

# --- CONFIGURATION ---
LIMIT_LOW = 0.00
LIMIT_HIGH = 0.95
START_STEP = 4000
END_STEP = 4000 + 168  # 1 Week

COLORS = {
    "No-Control": "black",
    "Default-RBC": "#1f77b4",  # Blue
    "Advanced-RBC": "#d62728", # Red
}

def load_data(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None
    return pd.read_csv(path)

def plot_weekly_behavior(output_dir, files):
    # Setup plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    dfs = {}
    
    # Load and Preprocess
    for label, path in files.items():
        df = load_data(path)
        if df is None: continue
        
        # Fallback for missing min/max columns
        # (If soc_min is missing, we assume soc_mean is the value to check)
        if 'soc_min' not in df.columns: df['soc_min'] = df['soc_mean']
        if 'soc_max' not in df.columns: df['soc_max'] = df['soc_mean']
        
        # Slice the week
        if len(df) > END_STEP:
            df = df.iloc[START_STEP:END_STEP].copy()
        
        dfs[label] = df

    # Plotting
    for label, df in dfs.items():
        steps = df['step']
        c = COLORS.get(label, 'gray')
        
        # Panel 1: SoC Behavior
        # Plot the Mean line
        axes[0].plot(steps, df['soc_mean'], label=label, color=c, linewidth=2)
        # Plot the Range (Min to Max) as a shaded area
        axes[0].fill_between(steps, df['soc_min'], df['soc_max'], color=c, alpha=0.1)

        # Panel 2: Low Violations (soc_min < 0.0)
        # We check if the WORST building went below the limit
        vio_low = (df['soc_min'] < LIMIT_LOW).astype(int)
        axes[1].plot(steps, vio_low, label=label, color=c, linestyle='-', linewidth=1.5)
        
        # Panel 3: High Violations (soc_max > 0.95)
        # We check if the WORST building went above the limit
        vio_high = (df['soc_max'] > LIMIT_HIGH).astype(int)
        axes[2].plot(steps, vio_high, label=label, color=c, linestyle='-', linewidth=1.5)

    # Formatting Panel 1 (SoC)
    axes[0].set_ylabel("Battery SoC (0-1)")
    axes[0].set_title(f"Weekly Battery Behavior (Steps {START_STEP}-{END_STEP})")
    axes[0].axhline(LIMIT_HIGH, color='red', linestyle='--', label=f'Max Limit ({LIMIT_HIGH})')
    axes[0].axhline(LIMIT_LOW, color='red', linestyle='--', label=f'Min Limit ({LIMIT_LOW})')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(-0.05, 1.05)

    # Formatting Panel 2 (Low Violations)
    axes[1].set_ylabel("Violation Flag (0/1)")
    axes[1].set_title(f"Lower Bound Violations (SoC < {LIMIT_LOW})")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_yticks([0, 1])
    
    # Formatting Panel 3 (High Violations)
    axes[2].set_ylabel("Violation Flag (0/1)")
    axes[2].set_title(f"Upper Bound Violations (SoC > {LIMIT_HIGH})")
    axes[2].set_xlabel("Time Step (Hour of Year)")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([0, 1])
    
    # Save
    out_path = os.path.join(output_dir, "weekly_safety_check.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[SUCCESS] Plot saved to: {out_path}")

if __name__ == "__main__":
    files = {
        "No-Control": "runs/kpi_no_control_kpis.csv",
        "Default-RBC": "runs/kpi_citylearn_default_kpis.csv",
        "Advanced-RBC": "runs/kpi_rbc_advanced_kpis.csv"
    }
    
    output_dir = "runs/plots_3way"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Weekly Safety Check...")
    plot_weekly_behavior(output_dir, files)
