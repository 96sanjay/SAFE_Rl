import pandas as pd

print("="*80)
print("COST COMPUTATION DIAGNOSTIC")
print("="*80)

# Load the evaluation KPI log
kpi_file = "runs/kpi_logs/Lambda40_FreshEval.csv"
df = pd.read_csv(kpi_file)

print(f"\nLoaded {len(df)} timesteps")
print()

# Check if cost columns exist and have non-zero values
print("1. CHECKING COST COLUMNS")
print("-"*80)

cost_cols = ['cost', 'cost_ev_departure', 'cost_grid_peak', 'cost_grid_ramp', 
             'cost_grid_peak_raw', 'cost_grid_ramp_raw']

for col in cost_cols:
    if col in df.columns:
        nonzero = (df[col] > 0).sum()
        total = df[col].sum()
        print(f"{col:<25} NonZero: {nonzero:>5} steps  Total: {total:>10.2f}")
    else:
        print(f"{col:<25} ❌ MISSING")

print()

# Check thresholds vs actual values
print("2. THRESHOLD VERIFICATION")
print("-"*80)

peak_threshold = 96.10
ramp_threshold = 48.41

if 'grid_import_kwh' in df.columns:
    grid_import = df['grid_import_kwh']
    print(f"Peak Threshold:      {peak_threshold:.2f} kW")
    print(f"Grid Import Max:     {grid_import.max():.2f} kW")
    print(f"Grid Import Mean:    {grid_import.mean():.2f} kW")
    print(f"Steps > Threshold:   {(grid_import > peak_threshold).sum()} ({(grid_import > peak_threshold).sum()/len(df)*100:.2f}%)")
else:
    print("❌ grid_import_kwh column missing")

print()

if 'grid_ramp_delta' in df.columns:
    ramp_delta = df['grid_ramp_delta'].abs()
    print(f"Ramp Threshold:      {ramp_threshold:.2f} kW/h")
    print(f"Ramp Delta Max:      {ramp_delta.max():.2f} kW/h")
    print(f"Ramp Delta Mean:     {ramp_delta.mean():.2f} kW/h")
    print(f"Steps > Threshold:   {(ramp_delta > ramp_threshold).sum()} ({(ramp_delta > ramp_threshold).sum()/len(df)*100:.2f}%)")
else:
    print("❌ grid_ramp_delta column missing")

print()

# Check if violations are flagged correctly
print("3. VIOLATION FLAG VERIFICATION")
print("-"*80)

if 'grid_peak_violation' in df.columns:
    peak_viols = (df['grid_peak_violation'] > 0).sum()
    print(f"Peak Violations Flagged: {peak_viols}")
else:
    print("❌ grid_peak_violation missing")

if 'grid_ramp_violation' in df.columns:
    ramp_viols = (df['grid_ramp_violation'] > 0).sum()
    print(f"Ramp Violations Flagged: {ramp_viols}")
else:
    print("❌ grid_ramp_violation missing")

print()

# Show some examples where violations SHOULD occur
print("4. SAMPLE TIMESTEPS WITH HIGH VALUES")
print("-"*80)

if 'grid_import_kwh' in df.columns and 'cost_grid_peak_raw' in df.columns:
    high_import = df.nlargest(5, 'grid_import_kwh')[['step', 'grid_import_kwh', 'cost_grid_peak_raw', 'cost_grid_peak', 'grid_peak_violation']]
    print("\nTop 5 Grid Import Timesteps:")
    print(high_import.to_string(index=False))

print()

if 'grid_ramp_delta' in df.columns and 'cost_grid_ramp_raw' in df.columns:
    df['ramp_abs'] = df['grid_ramp_delta'].abs()
    high_ramp = df.nlargest(5, 'ramp_abs')[['step', 'grid_ramp_delta', 'cost_grid_ramp_raw', 'cost_grid_ramp', 'grid_ramp_violation']]
    print("\nTop 5 Ramp Delta Timesteps:")
    print(high_ramp.to_string(index=False))

print()

# Check environment parameters
print("5. ENVIRONMENT PARAMETERS CHECK")
print("-"*80)
import os
print(f"EV_COST_SCALE:       {os.environ.get('CITYLEARN_EV_COST_SCALE', 'NOT SET')}")
print(f"EXPORT_FACTOR:       {os.environ.get('CITYLEARN_EXPORT_FACTOR', 'NOT SET')}")
print(f"REWARD_SCALE:        {os.environ.get('CITYLEARN_REWARD_SCALE', 'NOT SET')}")

print("="*80)
