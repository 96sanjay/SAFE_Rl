#!/usr/bin/env python3
"""Analyze RCPO training KPIs to verify battery actions"""

import pandas as pd
import numpy as np

print("="*80)
print("RCPO TRAINING KPI ANALYSIS - BATTERY ACTIONS")
print("="*80)

# Load the massive training KPI file (this will take a moment)
print("\nLoading training KPIs (1.4GB file, please wait)...")
df = pd.read_csv("runs/kpi_logs/CityLearnSafety_kpis_v3.csv")

print(f"\n✓ Loaded {len(df):,} timesteps from training")
print(f"  Episodes: {len(df) / 8760:.1f}")
print(f"  Columns: {len(df.columns)}")

# Find battery action columns
action_cols = [col for col in df.columns if 'action' in col.lower()]
battery_action_cols = [col for col in action_cols if 'battery' in col.lower() or any(f'action_{i}' in col for i in range(18))]

print(f"\n{'='*80}")
print(f"BATTERY ACTION COLUMNS FOUND:")
print(f"{'='*80}")
for col in battery_action_cols[:20]:
    print(f"  - {col}")

# Sample actions from early, middle, and late training
sample_points = [
    ("Early (Epoch 0)", 100, 200),      # Steps 100-200 of epoch 0
    ("Middle (Epoch 50)", 50*8760+100, 50*8760+200),  # Epoch 50
    ("Late (Epoch 99)", 99*8760+100, 99*8760+200),    # Epoch 99
]

print(f"\n{'='*80}")
print(f"BATTERY ACTIONS AT DIFFERENT TRAINING STAGES")
print(f"{'='*80}")

for label, start, end in sample_points:
    if end < len(df):
        print(f"\n{label} (steps {start}-{end}):")
        sample = df.iloc[start:end]
        
        # Check first few battery actions
        for col in battery_action_cols[:5]:
            if col in df.columns:
                values = sample[col]
                print(f"  {col}:")
                print(f"    Mean: {values.mean():.4f}")
                print(f"    Min: {values.min():.4f}")
                print(f"    Max: {values.max():.4f}")
                print(f"    Std: {values.std():.4f}")

# Check battery SOC values throughout training
print(f"\n{'='*80}")
print(f"BATTERY SOC THROUGHOUT TRAINING")
print(f"{'='*80}")

soc_cols = [col for col in df.columns if 'battery_soc' in col]
if soc_cols:
    print(f"\nFound {len(soc_cols)} battery SOC columns")
    
    # Analyze first building's battery
    if 'battery_soc_b1' in df.columns:
        soc = df['battery_soc_b1']
        print(f"\nbattery_soc_b1 (entire training):")
        print(f"  Mean: {soc.mean():.4f}")
        print(f"  Min: {soc.min():.4f}")
        print(f"  Max: {soc.max():.4f}")
        print(f"  At SOC≈0 (<0.01): {(soc < 0.01).sum():,} ({100*(soc < 0.01).sum()/len(soc):.2f}%)")
        print(f"  At SOC≈max (>0.90): {(soc > 0.90).sum():,} ({100*(soc > 0.90).sum()/len(soc):.2f}%)")

# Check battery violations
if 'battery_soc_violation' in df.columns:
    print(f"\n{'='*80}")
    print(f"BATTERY VIOLATIONS DURING TRAINING")
    print(f"{'='*80}")
    violations = df['battery_soc_violation']
    print(f"Total timesteps with violations: {violations.sum():,}/{len(violations):,}")
    print(f"Violation rate: {100*violations.mean():.2f}%")

print("\n" + "="*80)
