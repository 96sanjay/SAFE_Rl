#!/usr/bin/env python3
"""Create CSV comparison of all 3 policies"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

print("="*80)
print("LOADING RESULTS")
print("="*80)

results_dir = Path("runs/evaluations")

rbc_file = results_dir / "2026-02-06_seed42" / "rbc_detailed.json"
ppolag_file = results_dir / "ppolag_v2_single" / "results.json"
focops_file = results_dir / "focops_v2_single" / "results.json"

def load_result(filepath):
    with open(filepath) as f:
        return json.load(f)

rbc = load_result(rbc_file)
ppolag = load_result(ppolag_file)
focops = load_result(focops_file)

print(f"✓ Loaded all 3 policies")

def extract_metrics(result, name):
    pol = result['policy']
    orc = result['oracle']
    viol = pol['violations']
    costs = pol['cost_breakdown']
    kpis = pol.get('citylearn_kpis', {})
    
    return {
        'Agent': name,
        'Reward': pol['reward'],
        'Cost': pol['cost'],
        'Steps': pol['steps'],
        'EV_Viol_Count': viol['ev']['count'],
        'EV_Total_Deps': viol['ev']['total'],
        'EV_Viol_%': viol['ev']['rate_%'],
        'Grid_Viol_%': viol['grid']['rate_%'],
        'Battery_Viol_%': viol['battery']['rate_%'],
        'Building_Viol_%': viol['building']['rate_%'],
        'EV_Deficit_kWh': pol['ev_deficit_kwh'],
        'Oracle_Deficit_kWh': orc['ev_deficit_kwh'],
        'Avoidable_kWh': orc['avoidable_kwh'],
        'Avoidable_%': orc['avoidable_percent'],
        'Cost_EV': costs['ev'],
        'Cost_Grid': costs['grid'],
        'Cost_Battery': costs['battery'],
        'Cost_Building': costs['building'],
        'CL_Consumption': kpis.get('citylearn_electricity_consumption_total', np.nan),
        'CL_Carbon': kpis.get('citylearn_carbon_emissions_total', np.nan),
        'CL_Cost': kpis.get('citylearn_cost_total', np.nan),
        'CL_Peak_Daily': kpis.get('citylearn_daily_peak_average', np.nan),
        'CL_Peak_AllTime': kpis.get('citylearn_all_time_peak_average', np.nan),
        'CL_Ramping': kpis.get('citylearn_ramping_average', np.nan),
    }

data = [
    extract_metrics(rbc, 'RBC'),
    extract_metrics(ppolag, 'PPOLag_V2'),
    extract_metrics(focops, 'FOCOPS_V2'),
]

df = pd.DataFrame(data)

output_dir = Path("runs/evaluations")
output_dir.mkdir(exist_ok=True)

# Save complete comparison
df.to_csv(output_dir / "complete_comparison.csv", index=False)
print(f"\n✓ Saved: {output_dir / 'complete_comparison.csv'}")

# Display
print("\n" + "="*80)
print("POLICY COMPARISON (Seed 42)")
print("="*80)

print("\n1. PERFORMANCE:")
perf = df[['Agent', 'Reward', 'Cost']].copy()
print(perf.to_string(index=False))

print("\n2. VIOLATIONS (100% Strict Threshold):")
viol = df[['Agent', 'EV_Viol_%', 'Grid_Viol_%', 'Battery_Viol_%', 'Building_Viol_%']].copy()
print(viol.to_string(index=False))

print("\n3. EV DEFICIT ANALYSIS:")
ev = df[['Agent', 'EV_Deficit_kWh', 'Avoidable_kWh', 'Avoidable_%']].copy()
print(ev.to_string(index=False))

print("\n4. CITYLEARN KPIs:")
kpi = df[['Agent', 'CL_Consumption', 'CL_Carbon', 'CL_Cost', 'CL_Ramping']].copy()
print(kpi.to_string(index=False))

print("\n" + "="*80)
