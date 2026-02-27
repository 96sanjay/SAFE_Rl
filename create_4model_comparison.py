#!/usr/bin/env python3
"""Create CSV comparison of all 4 policies"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

print("="*80)
print("LOADING ALL 4 MODEL RESULTS")
print("="*80)

results_dir = Path("runs/evaluations")

# Load all saved results
rbc_file = results_dir / "2026-02-06_seed42" / "rbc_detailed.json"
ppolag_file = results_dir / "ppolag_v2_single" / "results.json"
focops_file = results_dir / "focops_v2_single" / "results.json"
trpo_file = results_dir / "trpo_baseline_single" / "results.json"

def load_result(filepath):
    with open(filepath) as f:
        return json.load(f)

rbc = load_result(rbc_file)
ppolag = load_result(ppolag_file)
focops = load_result(focops_file)
trpo = load_result(trpo_file)

print(f"✓ Loaded RBC")
print(f"✓ Loaded PPOLag_V2")
print(f"✓ Loaded FOCOPS_V2")
print(f"✓ Loaded TRPO_Baseline")

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
        'CL_ZeroNetEnergy': kpis.get('citylearn_zero_net_energy', np.nan),
    }

data = [
    extract_metrics(rbc, 'RBC'),
    extract_metrics(ppolag, 'PPOLag_V2'),
    extract_metrics(focops, 'FOCOPS_V2'),
    extract_metrics(trpo, 'TRPO_Baseline'),
]

df = pd.DataFrame(data)

# Save complete comparison
output_file = results_dir / "4model_complete_comparison.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# Also save separate focused tables
print("\n" + "="*80)
print("1. PERFORMANCE OVERVIEW")
print("="*80)
perf = df[['Agent', 'Reward', 'Cost', 'Steps']].copy()
print(perf.to_string(index=False))
perf.to_csv(results_dir / "comparison_performance.csv", index=False)

print("\n" + "="*80)
print("2. CONSTRAINT VIOLATIONS (100% Strict)")
print("="*80)
viol = df[['Agent', 'EV_Viol_%', 'EV_Viol_Count', 'EV_Total_Deps', 
           'Grid_Viol_%', 'Battery_Viol_%', 'Building_Viol_%']].copy()
print(viol.to_string(index=False))
viol.to_csv(results_dir / "comparison_violations.csv", index=False)

print("\n" + "="*80)
print("3. EV DEFICIT ANALYSIS")
print("="*80)
ev = df[['Agent', 'EV_Deficit_kWh', 'Oracle_Deficit_kWh', 
         'Avoidable_kWh', 'Avoidable_%']].copy()
print(ev.to_string(index=False))
ev.to_csv(results_dir / "comparison_ev_deficits.csv", index=False)

print("\n" + "="*80)
print("4. COST BREAKDOWN")
print("="*80)
cost_df = df[['Agent', 'Cost', 'Cost_EV', 'Cost_Grid', 
              'Cost_Battery', 'Cost_Building']].copy()
print(cost_df.to_string(index=False))
cost_df.to_csv(results_dir / "comparison_costs.csv", index=False)

print("\n" + "="*80)
print("5. CITYLEARN KPIs")
print("="*80)
kpi = df[['Agent', 'CL_Consumption', 'CL_Carbon', 'CL_Cost', 
          'CL_Peak_Daily', 'CL_Ramping']].copy()
print(kpi.to_string(index=False))
kpi.to_csv(results_dir / "comparison_citylearn_kpis.csv", index=False)

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print("\nEV Violation Ranking (Lower is Better):")
print(df[['Agent', 'EV_Viol_%']].sort_values('EV_Viol_%').to_string(index=False))

print("\nCost Ranking (Lower is Better):")
print(df[['Agent', 'Cost']].sort_values('Cost').to_string(index=False))

print("\nReward Ranking (Higher is Better):")
print(df[['Agent', 'Reward']].sort_values('Reward', ascending=False).to_string(index=False))

print("\n" + "="*80)
print("FILES CREATED:")
print("="*80)
print(f"  • {output_file}")
print(f"  • {results_dir / 'comparison_performance.csv'}")
print(f"  • {results_dir / 'comparison_violations.csv'}")
print(f"  • {results_dir / 'comparison_ev_deficits.csv'}")
print(f"  • {results_dir / 'comparison_costs.csv'}")
print(f"  • {results_dir / 'comparison_citylearn_kpis.csv'}")
print("="*80)
