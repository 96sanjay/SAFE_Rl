#!/usr/bin/env python3
"""Analyze RCPO battery violations from KPI logs"""

import pandas as pd
import numpy as np

print("="*80)
print("RCPO BATTERY VIOLATION ANALYSIS FROM KPI LOGS")
print("="*80)

# Load the KPI file
kpi_file = "./runs/kpi_logs/EVAL_RCPO_Aggressive.csv"
df = pd.read_csv(kpi_file)

print(f"\nLoaded {len(df)} timesteps from KPI log")
print(f"\nColumns available: {len(df.columns)} columns")

# Find battery-related columns
battery_cols = [col for col in df.columns if 'battery' in col.lower() or 'soc' in col.lower() or 'building' in col.lower()]
print(f"\nBattery-related columns found:")
for col in battery_cols:
    print(f"  - {col}")

# Check for cost columns
cost_cols = [col for col in df.columns if 'cost' in col.lower()]
print(f"\nCost columns found:")
for col in cost_cols[:10]:  # Show first 10
    print(f"  - {col}")

# Analyze battery violations
if 'cost_building_soc' in df.columns:
    battery_viol_steps = (df['cost_building_soc'] > 0).sum()
    total_steps = len(df)
    viol_rate = 100 * battery_viol_steps / total_steps
    
    print(f"\n" + "="*80)
    print("BATTERY VIOLATION ANALYSIS")
    print("="*80)
    print(f"Total steps: {total_steps}")
    print(f"Steps with battery violations: {battery_viol_steps}")
    print(f"Battery violation rate: {viol_rate:.2f}%")
    print(f"Total battery cost: {df['cost_building_soc'].sum():.2f}")
    
    # Check if it's really 99.98%
    if viol_rate > 99.0:
        print(f"\n⚠️  CONFIRMED: Battery violations at {viol_rate:.2f}% - nearly every step!")
    
    # Analyze SOC values if available
    soc_cols = [col for col in df.columns if 'building_soc' in col]
    if soc_cols:
        print(f"\n" + "="*80)
        print("BATTERY SOC ANALYSIS")
        print("="*80)
        for col in soc_cols[:5]:  # Analyze first 5 buildings
            soc_values = df[col].dropna()
            if len(soc_values) > 0:
                print(f"\n{col}:")
                print(f"  Mean: {soc_values.mean():.4f}")
                print(f"  Min: {soc_values.min():.4f}")
                print(f"  Max: {soc_values.max():.4f}")
                print(f"  Std: {soc_values.std():.4f}")
                
                # Check boundaries
                at_zero = (soc_values < 0.01).sum()
                at_max = (soc_values > 0.94).sum()
                print(f"  At SOC≈0: {at_zero} ({100*at_zero/len(soc_values):.1f}%)")
                print(f"  At SOC≈max: {at_max} ({100*at_max/len(soc_values):.1f}%)")
else:
    print("\n⚠️  'cost_building_soc' column not found!")
    print("Available columns with 'cost':")
    for col in cost_cols[:20]:
        print(f"  - {col}")

print("\n" + "="*80)
