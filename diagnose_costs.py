import pandas as pd
import numpy as np
from glob import glob

print("="*80)
print("COMPREHENSIVE COST DIAGNOSTICS")
print("="*80)

# ===========================
# 1. FIND AND LOAD KPI LOG
# ===========================
kpi_pattern = "runs/kpi_logs/EvaluateFinalAgent_Lambda40_Epoch100*.csv"
kpi_files = glob(kpi_pattern)

if not kpi_files:
    print(f"❌ No KPI log found matching: {kpi_pattern}")
    print("\nTrying alternative patterns:")
    alt_patterns = [
        "runs/kpi_logs/*.csv",
        "runs/kpi_logs/EvaluateFinalAgent*.csv"
    ]
    for pattern in alt_patterns:
        files = glob(pattern)
        if files:
            print(f"  Found: {files}")
    exit(1)

kpi_file = kpi_files[0]
print(f"📂 Loading: {kpi_file}\n")

df = pd.read_csv(kpi_file)
print(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns\n")

# ===========================
# 2. CHECK COST COLUMNS EXIST
# ===========================
print("="*80)
print("COST COLUMNS PRESENT IN CSV")
print("="*80)

cost_columns = {
    'cost': 'Total CMDP cost',
    'cost_ev_departure': 'EV departure cost',
    'cost_grid_peak': 'Grid peak cost (weighted)',
    'cost_grid_peak_raw': 'Grid peak cost (raw)',
    'cost_grid_ramp': 'Grid ramp cost (weighted)',
    'cost_grid_ramp_raw': 'Grid ramp cost (raw)',
    'cost_building_soc': 'Building battery SOC cost',
}

for col, desc in cost_columns.items():
    if col in df.columns:
        print(f"✅ {col:<25} - {desc}")
    else:
        print(f"❌ {col:<25} - MISSING!")

print()

# ===========================
# 3. CHECK VIOLATION COLUMNS
# ===========================
print("="*80)
print("VIOLATION COLUMNS PRESENT IN CSV")
print("="*80)

violation_columns = {
    'grid_peak_violation': 'Peak violation flag (0 or 1)',
    'grid_ramp_violation': 'Ramp violation flag (0 or 1)',
    'ev_controllable_deficit_kwh': 'EV controllable deficit',
}

for col, desc in violation_columns.items():
    if col in df.columns:
        print(f"✅ {col:<35} - {desc}")
    else:
        print(f"❌ {col:<35} - MISSING!")

print()

# ===========================
# 4. ANALYZE COST VALUES
# ===========================
print("="*80)
print("COST VALUE ANALYSIS (Are they actually zero?)")
print("="*80)

for col in ['cost', 'cost_ev_departure', 'cost_grid_peak', 'cost_grid_ramp']:
    if col not in df.columns:
        print(f"❌ {col} - Column missing!")
        continue
    
    values = df[col]
    nonzero = values[values > 0]
    
    print(f"\n{col}:")
    print(f"  Total sum:           {values.sum():.4f}")
    print(f"  Mean:                {values.mean():.4f}")
    print(f"  Max:                 {values.max():.4f}")
    print(f"  Non-zero count:      {len(nonzero)} / {len(values)} ({len(nonzero)/len(values)*100:.2f}%)")
    
    if len(nonzero) > 0:
        print(f"  Non-zero mean:       {nonzero.mean():.4f}")
        print(f"  Sample non-zero:     {nonzero.head(3).values}")
    else:
        print(f"  ⚠️  ALL VALUES ARE ZERO!")

# ===========================
# 5. CHECK RAW VIOLATION DATA
# ===========================
print("\n" + "="*80)
print("RAW VIOLATION DATA ANALYSIS")
print("="*80)

# Peak violations
if 'grid_peak_violation' in df.columns:
    peak_viol = df['grid_peak_violation'].sum()
    print(f"\nPeak Violations (from flag):")
    print(f"  Count: {peak_viol}")
    print(f"  Percentage: {peak_viol/len(df)*100:.2f}%")
else:
    print(f"\n❌ grid_peak_violation column missing!")

# Ramp violations
if 'grid_ramp_violation' in df.columns:
    ramp_viol = df['grid_ramp_violation'].sum()
    print(f"\nRamp Violations (from flag):")
    print(f"  Count: {ramp_viol}")
    print(f"  Percentage: {ramp_viol/len(df)*100:.2f}%")
else:
    print(f"\n❌ grid_ramp_violation column missing!")

# EV violations
if 'ev_controllable_deficit_kwh' in df.columns:
    ev_deficit = df['ev_controllable_deficit_kwh']
    ev_viol = (ev_deficit > 0).sum()
    print(f"\nEV Violations (controllable deficit > 0):")
    print(f"  Count: {ev_viol}")
    print(f"  Percentage: {ev_viol/len(df)*100:.2f}%")
    print(f"  Total deficit: {ev_deficit.sum():.2f} kWh")
else:
    print(f"\n❌ ev_controllable_deficit_kwh column missing!")

# ===========================
# 6. CHECK UNDERLYING DATA
# ===========================
print("\n" + "="*80)
print("UNDERLYING GRID DATA (to verify environment is running)")
print("="*80)

grid_cols = {
    'grid_import_kwh': 'Grid import',
    'step_net_consumption_kwh': 'Net consumption',
    'grid_ramp_delta': 'Ramp delta',
}

for col, desc in grid_cols.items():
    if col in df.columns:
        values = df[col]
        print(f"\n{col} ({desc}):")
        print(f"  Mean: {values.mean():.2f}")
        print(f"  Max:  {values.max():.2f}")
        print(f"  Min:  {values.min():.2f}")
        print(f"  Sample: {values.head(3).values}")
    else:
        print(f"\n❌ {col} - Missing!")

# ===========================
# 7. DIAGNOSTIC CONCLUSION
# ===========================
print("\n" + "="*80)
print("DIAGNOSTIC CONCLUSION")
print("="*80)

total_cost = df['cost'].sum() if 'cost' in df.columns else 0
peak_cost = df['cost_grid_peak'].sum() if 'cost_grid_peak' in df.columns else 0
ramp_cost = df['cost_grid_ramp'].sum() if 'cost_grid_ramp' in df.columns else 0

if total_cost == 0:
    print("\n❌ CRITICAL PROBLEM: Total cost is ZERO!")
    print("\nPossible causes:")
    print("  1. Safety wrapper (CityLearnSafetyEnvV3) was NOT used during evaluation")
    print("  2. Cost weights are set to zero in the environment")
    print("  3. KPI logger is not logging costs properly")
    print("\nAction needed:")
    print("  → Check if evaluation script uses CityLearnSafetyEnvV3 wrapper")
    print("  → Verify environment variables (CITYLEARN_EV_COST_SCALE, etc.)")
    print("  → Check safety_env_v3.py for w_grid_peak and w_grid_ramp values")
elif peak_cost == 0 and ramp_cost == 0:
    print("\n⚠️  WARNING: Peak and Ramp costs are ZERO!")
    print("  → Agent may have perfect constraint satisfaction (unlikely)")
    print("  → OR cost weights are zero (check w_grid_peak, w_grid_ramp)")
else:
    print("\n✅ Costs are non-zero - evaluation appears to be working!")
    print(f"  Total cost: {total_cost:.2f}")
    print(f"  Peak cost: {peak_cost:.2f}")
    print(f"  Ramp cost: {ramp_cost:.2f}")

print("="*80)
