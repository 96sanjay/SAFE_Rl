#!/usr/bin/env python3
"""Analyze RCPO training actions from progress.csv"""

import pandas as pd
import numpy as np

print("="*80)
print("RCPO TRAINING ACTIONS ANALYSIS")
print("="*80)

# Load training progress
df = pd.read_csv("runs/rcpo_aggressive_v2/RCPO-{CityLearnSafety-SoC-v0}/seed-042-2026-02-06-00-05-56/progress.csv")

print(f"\nLoaded {len(df)} training epochs")
print(f"Columns: {len(df.columns)}")

# Show all column names
print("\nAll columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. {col}")

# Find action and battery-related columns
action_cols = [col for col in df.columns if 'action' in col.lower()]
battery_cols = [col for col in df.columns if 'battery' in col.lower() or 'soc' in col.lower()]

print(f"\n{'='*80}")
print(f"ACTION COLUMNS ({len(action_cols)} found):")
print(f"{'='*80}")
for col in action_cols:
    print(f"  - {col}")

print(f"\n{'='*80}")
print(f"BATTERY/SOC COLUMNS ({len(battery_cols)} found):")
print(f"{'='*80}")
for col in battery_cols:
    print(f"  - {col}")

# Analyze first, middle, and last epochs
epochs_to_check = [0, 49, 99]
print(f"\n{'='*80}")
print(f"SAMPLE EPOCHS: {epochs_to_check}")
print(f"{'='*80}")

for epoch in epochs_to_check:
    if epoch < len(df):
        print(f"\nEpoch {epoch}:")
        row = df.iloc[epoch]
        
        # Show cost and reward
        if 'EpCost' in df.columns:
            print(f"  Cost: {row['EpCost']:.2f}")
        if 'EpRet' in df.columns:
            print(f"  Reward: {row['EpRet']:.2f}")
        
        # Show any action values
        if action_cols:
            print(f"  Actions:")
            for col in action_cols[:10]:  # Show first 10 action columns
                print(f"    {col}: {row[col]:.4f}")
        
        # Show battery metrics
        if battery_cols:
            print(f"  Battery metrics:")
            for col in battery_cols[:10]:  # Show first 10 battery columns
                if col in row.index:
                    print(f"    {col}: {row[col]:.4f}")

print("\n" + "="*80)
