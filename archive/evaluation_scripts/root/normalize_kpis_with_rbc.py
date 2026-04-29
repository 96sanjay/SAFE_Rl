#!/usr/bin/env python3
"""Normalize CityLearn KPIs using RBC as baseline"""

import pandas as pd
from pathlib import Path

print("="*80)
print("NORMALIZING CITYLEARN KPIs WITH RBC AS BASELINE")
print("="*80)

# Load the comparison file
comparison_file = Path("runs/evaluations/comparison_citylearn_kpis.csv")
df = pd.read_csv(comparison_file)

print(f"\n✓ Loaded comparison with {len(df)} models")
print(f"  Columns: {list(df.columns)}")
print(f"  Models: {list(df['Agent'])}")

# Find RBC row
rbc_row = df[df['Agent'] == 'RBC']
if len(rbc_row) == 0:
    print("\n❌ ERROR: RBC not found in the data!")
    exit(1)

rbc_row = rbc_row.iloc[0]
print(f"\n✓ Found RBC baseline")

# Get metric columns (all except 'Agent')
metric_cols = [col for col in df.columns if col != 'Agent']
print(f"\n✓ Will normalize {len(metric_cols)} metrics:")
for col in metric_cols:
    print(f"    - {col}")

# Create normalized dataframe
df_normalized = df.copy()

# Normalize each metric column by dividing by RBC value
for col in metric_cols:
    rbc_value = rbc_row[col]
    if rbc_value == 0 or pd.isna(rbc_value):
        print(f"\n⚠️  Skipping {col} (RBC value is {rbc_value})")
        df_normalized[col] = df[col]  # Keep original
    else:
        df_normalized[col] = df[col] / rbc_value
        print(f"  {col}: RBC baseline = {rbc_value:.4f}")

# Display results
print(f"\n" + "="*80)
print("NORMALIZED METRICS (RBC = 1.0)")
print("="*80)
print(df_normalized.to_string(index=False))

# Save to new file
output_file = Path("runs/evaluations/comparison_citylearn_kpis_normalized.csv")
df_normalized.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# Also create a comparison showing both original and normalized side-by-side
print(f"\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80)
print("""
Normalized value interpretation:
  1.0   = Same as RBC (baseline)
  < 1.0 = Better than RBC (e.g., 0.9 = 10% better)
  > 1.0 = Worse than RBC (e.g., 1.5 = 50% worse)

For metrics where LOWER is better (consumption, cost, carbon, ramping):
  - Values < 1.0 are GOOD (using less than RBC)
  - Values > 1.0 are BAD (using more than RBC)

For metrics where HIGHER might be better (depends on context):
  - Check the original values to understand direction
""")

# Create a summary table
print(f"\n" + "="*80)
print("PERFORMANCE SUMMARY (Relative to RBC)")
print("="*80)

summary_cols = ['CL_Consumption', 'CL_Carbon', 'CL_Cost', 'CL_Peak_Daily', 'CL_Ramping']
available_cols = [col for col in summary_cols if col in df_normalized.columns]

if available_cols:
    summary = df_normalized[['Agent'] + available_cols].copy()
    
    # Add a "Better/Worse" interpretation
    for col in available_cols:
        summary[f'{col}_Status'] = summary[col].apply(
            lambda x: '✅ Better' if x < 1.0 else ('⚖️ Same' if abs(x - 1.0) < 0.01 else '❌ Worse')
        )
    
    print(summary.to_string(index=False))

print("\n" + "="*80)
