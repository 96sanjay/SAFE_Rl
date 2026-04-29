import pandas as pd
import numpy as np

print("="*80)
print("LAMBDA=40 TRAINING RESULTS ANALYSIS")
print("="*80)

# Load training KPI log
kpi_file = "runs/kpi_logs/SafePPOLag_Lambda40_HighExplore_100ep.csv"
print(f"\n📂 Loading: {kpi_file}")

df = pd.read_csv(kpi_file)
print(f"✅ Loaded {len(df)} timesteps")
print()

# Total costs
print("="*80)
print("CUMULATIVE COSTS")
print("="*80)

total_cost = df['cost'].sum()
ev_cost = df['cost_ev_departure'].sum()
peak_cost = df['cost_grid_peak'].sum()
ramp_cost = df['cost_grid_ramp'].sum()

print(f"Total CMDP Cost:     {total_cost:.2f}")
print(f"  EV Cost:           {ev_cost:.2f} ({ev_cost/total_cost*100:.1f}%)")
print(f"  Peak Cost:         {peak_cost:.2f} ({peak_cost/total_cost*100:.1f}%)")
print(f"  Ramp Cost:         {ramp_cost:.2f} ({ramp_cost/total_cost*100:.1f}%)")
print()

# Violation statistics
print("="*80)
print("VIOLATION COUNTS & PERCENTAGES")
print("="*80)

total_steps = len(df)
peak_violations = (df['grid_peak_violation'] > 0).sum()
ramp_violations = (df['grid_ramp_violation'] > 0).sum()
ev_violations = (df['ev_controllable_deficit_kwh'] > 0).sum() if 'ev_controllable_deficit_kwh' in df.columns else 0

print(f"Total Steps:         {total_steps}")
print(f"Peak Violations:     {peak_violations:4d} steps ({peak_violations/total_steps*100:5.2f}%)")
print(f"Ramp Violations:     {ramp_violations:4d} steps ({ramp_violations/total_steps*100:5.2f}%)")
print(f"EV Violations:       {ev_violations:4d} steps ({ev_violations/total_steps*100:5.2f}%)")
print()

# Violation magnitudes
print("="*80)
print("VIOLATION MAGNITUDES")
print("="*80)

peak_raw = df[df['cost_grid_peak_raw'] > 0]['cost_grid_peak_raw']
if len(peak_raw) > 0:
    print(f"\nPeak Violations ({len(peak_raw)} occurrences):")
    print(f"  Mean:    {peak_raw.mean():.2f} kW over threshold")
    print(f"  Max:     {peak_raw.max():.2f} kW over threshold")

ramp_raw = df[df['cost_grid_ramp_raw'] > 0]['cost_grid_ramp_raw']
if len(ramp_raw) > 0:
    print(f"\nRamp Violations ({len(ramp_raw)} occurrences):")
    print(f"  Mean:    {ramp_raw.mean():.2f} kW/h over threshold")
    print(f"  Max:     {ramp_raw.max():.2f} kW/h over threshold")

print()

# Comparison
print("="*80)
print("COMPARISON TO BASELINES")
print("="*80)

rbc_cost = 304.38
lambda35_cost = 422.9
budget = 350.0

print(f"\n{'Agent':<25} {'Cost':<15} {'vs Budget':<15}")
print("-"*60)
print(f"{'RBC':<25} {rbc_cost:<15.2f} {rbc_cost-budget:<+15.2f}")
print(f"{'Lambda=35':<25} {lambda35_cost:<15.2f} {lambda35_cost-budget:<+15.2f}")
print(f"{'Lambda=40':<25} {total_cost:<15.2f} {total_cost-budget:<+15.2f}")
print()

# Status
if total_cost < budget:
    print("✅ SUCCESS! Under budget")
elif total_cost < budget * 1.05:
    print("⚠️  NEAR SUCCESS! Within 5% of budget")
else:
    print("❌ Over budget")

print(f"\nImprovement over Lambda=35: {lambda35_cost - total_cost:.2f} ({(lambda35_cost - total_cost)/lambda35_cost*100:.1f}%)")
print("="*80)
