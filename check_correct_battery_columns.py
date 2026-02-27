#!/usr/bin/env python3
"""Check the ACTUAL battery violation columns"""

import pandas as pd

df = pd.read_csv("./runs/kpi_logs/EVAL_RCPO_Aggressive.csv")

print("="*80)
print("CHECKING ALL BATTERY-RELATED COLUMNS")
print("="*80)

# Check all battery violation columns
battery_metrics = {
    'battery_soc_violation_count': 'Violation count per step',
    'battery_soc_violation': 'Boolean violation flag',
    'battery_soc_violation_any': 'Any violation this step',
    'battery_soc_violation_rate_%': 'Violation rate %',
    'cost_stems_battery': 'STEMS battery cost',
    'cost_building_soc': 'Building SOC cost',
    'battery_abuse_kwh': 'Battery abuse kWh',
    'battery_abuse_hours': 'Battery abuse hours',
}

for col, desc in battery_metrics.items():
    if col in df.columns:
        values = df[col].dropna()
        if len(values) > 0:
            viol_steps = (values > 0).sum() if col not in ['battery_soc_violation_rate_%'] else None
            print(f"\n{col} ({desc}):")
            print(f"  Sum: {values.sum():.2f}")
            print(f"  Mean: {values.mean():.6f}")
            print(f"  Max: {values.max():.2f}")
            if viol_steps is not None:
                print(f"  Steps > 0: {viol_steps}/{len(values)} ({100*viol_steps/len(values):.2f}%)")

# Check actual battery SOC values
print(f"\n" + "="*80)
print("BATTERY SOC VALUES (First 5 Buildings)")
print("="*80)

for i in range(1, 6):
    col = f'battery_soc_b{i}'
    if col in df.columns:
        soc = df[col].dropna()
        print(f"\n{col}:")
        print(f"  Mean: {soc.mean():.4f}")
        print(f"  Min: {soc.min():.4f}")
        print(f"  Max: {soc.max():.4f}")
        
        # Check constraint violations (SOC < 0.0 or SOC > 0.95)
        below_min = (soc < 0.0).sum()
        above_max = (soc > 0.95).sum()
        print(f"  SOC < 0.0: {below_min} ({100*below_min/len(soc):.2f}%)")
        print(f"  SOC > 0.95: {above_max} ({100*above_max/len(soc):.2f}%)")

# Check the evaluator's cost calculation
print(f"\n" + "="*80)
print("CHECKING EVALUATOR COST AGGREGATION")
print("="*80)

if 'step_cost' in df.columns:
    print(f"Total step_cost: {df['step_cost'].sum():.2f}")
if 'cost' in df.columns:
    print(f"Total cost (evaluator): {df['cost'].sum():.2f}")

# Break down by cost components
cost_components = ['cost_building_soc', 'cost_stems_battery', 'cost_ev_departure_avoidable', 
                   'cost_stems_grid_power', 'cost_stems_building_power']
print("\nCost breakdown:")
for comp in cost_components:
    if comp in df.columns:
        print(f"  {comp}: {df[comp].sum():.2f}")

print("="*80)
