#!/usr/bin/env python3
"""
Analysis script for aggregate constraint test.
Works with actual KPI columns from safety_env_v3.py
"""

import pandas as pd
import numpy as np
import sys
import os

print("=" * 80)
print("AGGREGATE CONSTRAINT TEST RESULTS")
print("=" * 80)

# Read KPI data
kpi_file = "runs/kpi_logs/TEST_AGGREGATE_1EP.csv"

if not os.path.exists(kpi_file):
    print(f"ERROR: KPI file not found: {kpi_file}")
    sys.exit(1)

df = pd.read_csv(kpi_file)
print(f"\n✓ Loaded {len(df)} timesteps from {kpi_file}")

# Filter to episode 0 only
ep0 = df[df['episode'] == 0].copy()
print(f"✓ Episode 0: {len(ep0)} steps")

if len(ep0) == 0:
    print("ERROR: No data for episode 0!")
    sys.exit(1)

print("\n" + "=" * 80)
print("1. BATTERY SOC CONSTRAINT VIOLATIONS")
print("=" * 80)

# Main violation metric: binary violation per timestep
viol_steps = (ep0['battery_soc_violation'] > 0).sum()
viol_pct = 100.0 * viol_steps / len(ep0)

print(f"\nTimesteps with violations: {viol_steps}/{len(ep0)} ({viol_pct:.1f}%)")

# Battery cost analysis
battery_cost_total = ep0['cost_stems_battery'].sum()
battery_cost_mean = ep0['cost_stems_battery'].mean()
battery_cost_max = ep0['cost_stems_battery'].max()

print(f"\nBattery Constraint Cost:")
print(f"  Total:  {battery_cost_total:.2f}")
print(f"  Mean:   {battery_cost_mean:.4f}")
print(f"  Max:    {battery_cost_max:.4f}")

# Comparison to targets
print("\n" + "-" * 80)
print("COMPARISON TO TARGETS:")
print("-" * 80)
print(f"{'Metric':<40} {'Target':<15} {'Actual':<15} {'Status':<10}")
print("-" * 80)

target_viol_pct = 10.0  # Target: <10% of timesteps violate

status_viol = "✅ PASS" if viol_pct < target_viol_pct else "❌ FAIL"

print(f"{'Violation % (timesteps)':<40} {'<10.0%':<15} {viol_pct:<15.1f} {status_viol:<10}")

# If using per-building (old method), violations should be ~100%
# If using aggregate, violations should be <50%
if viol_pct > 90.0:
    print("\n⚠️  WARNING: >90% violations suggests PER-BUILDING constraints!")
    print("   Expected: Aggregate constraints should have <50% violations")
elif viol_pct < 50.0:
    print(f"\n✅ GOOD! <50% violations suggests AGGREGATE is working!")
    if viol_pct < 20.0:
        print(f"   Excellent! Only {viol_pct:.1f}% violations")

print("\n" + "=" * 80)
print("2. DISTRICT-LEVEL SOC STATISTICS")
print("=" * 80)

soc_mean_avg = ep0['soc_mean'].mean()
soc_mean_min = ep0['soc_mean'].min()
soc_mean_max = ep0['soc_mean'].max()
soc_mean_std = ep0['soc_mean'].std()

print(f"\nDistrict Average SOC:")
print(f"  Mean: {soc_mean_avg:.3f}")
print(f"  Min:  {soc_mean_min:.3f}")
print(f"  Max:  {soc_mean_max:.3f}")
print(f"  Std:  {soc_mean_std:.3f}")

# Check SOC bounds (0.20 - 0.80 target)
soc_low = 0.20
soc_high = 0.80

district_low_viol = (ep0['soc_mean'] < soc_low).sum()
district_high_viol = (ep0['soc_mean'] > soc_high).sum()
district_total_viol = district_low_viol + district_high_viol
district_viol_pct = 100.0 * district_total_viol / len(ep0)

print(f"\nDistrict Average Bound Violations:")
print(f"  Below {soc_low}: {district_low_viol} steps")
print(f"  Above {soc_high}: {district_high_viol} steps")
print(f"  Total: {district_total_viol}/{len(ep0)} ({district_viol_pct:.1f}%)")

if district_viol_pct < 5.0:
    print("  ✅ District average stays in bounds! Aggregate constraint works!")
elif district_viol_pct < 20.0:
    print("  ⚠️  Some district violations, but much better than per-building")
else:
    print("  ❌ High district violations - may still be using per-building logic")

# Individual building extremes
soc_min_min = ep0['soc_min'].min()
soc_max_max = ep0['soc_max'].max()

print(f"\nIndividual Building Extremes:")
print(f"  Min SOC across all buildings: {soc_min_min:.3f}")
print(f"  Max SOC across all buildings: {soc_max_max:.3f}")

if soc_min_min < 0.05 or soc_max_max > 0.95:
    print("  → Individual buildings still hitting hard limits (expected with aggregate)")

print("\n" + "=" * 80)
print("3. COST ANALYSIS")
print("=" * 80)

# Total CMDP cost
total_cost = ep0['cost'].sum()
cost_soc = ep0['cost_building_soc'].sum()
cost_ev = ep0['cost_ev_departure'].sum()
cost_battery = ep0['cost_stems_battery'].sum()

print(f"\nCMDP Cost Breakdown:")
print(f"  Total:        {total_cost:.2f}")
print(f"  SOC (old):    {cost_soc:.2f}")
print(f"  EV:           {cost_ev:.2f}")
print(f"  Battery:      {cost_battery:.2f}")

# Weighted battery cost (w_soc = 50)
w_soc = 50.0
battery_weighted = cost_battery * w_soc
battery_pct = 100.0 * battery_weighted / max(total_cost, 1.0)

print(f"\nBattery Cost (weighted ×{w_soc}):")
print(f"  Weighted:     {battery_weighted:.2f}")
print(f"  % of total:   {battery_pct:.1f}%")

if battery_pct > 30.0:
    print("  ✅ Battery cost is significant (Lagrangian should respond)")
else:
    print("  ⚠️  Battery cost is low (may need higher weight)")

print("\n" + "=" * 80)
print("4. REWARD ANALYSIS")
print("=" * 80)

reward_mean = ep0['reward'].mean()
reward_sum = ep0['reward'].sum()
reward_type = ep0['reward_type'].iloc[0] if 'reward_type' in ep0.columns else 'unknown'

print(f"\nReward Type: {reward_type}")
print(f"Reward per step (avg): {reward_mean:.3f}")
print(f"Total reward:          {reward_sum:.2f}")

print("\n" + "=" * 80)
print("5. FINAL VERDICT")
print("=" * 80)

# Determine success
aggregate_working = (viol_pct < 90.0 and district_viol_pct < 20.0)

if aggregate_working and viol_pct < 10.0:
    print("\n🎉 SUCCESS! Aggregate constraints are working perfectly!")
    print(f"   - Violations: {viol_pct:.1f}% (target: <10%)")
    print(f"   - District avg violations: {district_viol_pct:.1f}%")
    print("   - Ready for full training (10-100 epochs)")
    
elif aggregate_working:
    print("\n✅ PROMISING! Aggregate approach is working!")
    print(f"   - Violations: {viol_pct:.1f}% (better than per-building's 100%)")
    print(f"   - District avg violations: {district_viol_pct:.1f}%")
    print("   - Needs more training to reach <10% target")
    print("\nRecommendations:")
    print("   - Run 10 epochs to see improvement")
    print("   - Monitor Lagrangian multiplier growth")
    print("   - Consider increasing w_soc if cost % is low")
    
else:
    print("\n❌ ISSUE! Still behaving like per-building constraints")
    print(f"   - Violations: {viol_pct:.1f}% (should be <50% for aggregate)")
    print(f"   - District violations: {district_viol_pct:.1f}%")
    print("\nPossible causes:")
    print("   - Wrong environment file being used")
    print("   - Aggregate constraint logic not implemented")
    print("   - Cost weight w_soc still too low")

print("\n" + "=" * 80)
print("DETAILED STATISTICS")
print("=" * 80)

print("\nKey Metrics Summary:")
summary_cols = ['soc_mean', 'battery_soc_violation', 'cost_stems_battery', 
                'cost', 'reward']
available_cols = [c for c in summary_cols if c in ep0.columns]
print(ep0[available_cols].describe())

print("\n" + "=" * 80)
