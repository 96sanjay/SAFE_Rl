#!/usr/bin/env python3
"""Evaluate all 5 policies with dual thresholds (100% + 80%)"""

import os
import sys
import pandas as pd
from pathlib import Path
import glob

sys.path.insert(0, os.getcwd())

from evaluation.agents.rbc import RBCAgent
from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent
from evaluation.core.evaluator import SafeRLEvaluator

print("="*80)
print("5-MODEL COMPARISON (100% Strict + 80% Tolerance)")
print("="*80)

# Find checkpoints
trpo_pattern = "runs/trpo_baseline_v2/TRPO-*/seed-042-*/torch_save/epoch-50.pt"
rcpo_pattern = "runs/*rcpo*/RCPO*/seed-*/torch_save/epoch-*.pt"

trpo_matches = glob.glob(trpo_pattern)
rcpo_matches = sorted(glob.glob(rcpo_pattern))

if not trpo_matches:
    print(f"ERROR: TRPO checkpoint not found")
    sys.exit(1)
if not rcpo_matches:
    print(f"ERROR: RCPO checkpoint not found")
    sys.exit(1)

TRPO_CKPT = trpo_matches[0]
RCPO_CKPT = rcpo_matches[-1]  # Use latest epoch

print(f"\nFound TRPO: {TRPO_CKPT}")
print(f"Found RCPO: {RCPO_CKPT}")

# Modified evaluator to track both thresholds
class DualThresholdEvaluator(SafeRLEvaluator):
    def _run_policy(self, agent, seed, run_name):
        result = super()._run_policy(agent, seed, run_name)
        
        # Add 80% tolerance violations
        os.environ["CITYLEARN_KPI_RUN_NAME"] = run_name + "_80pct"
        
        from evaluation.core.evaluator import make_env
        env = make_env()
        obs, info = env.reset(seed=seed)
        agent.reset(env)
        
        ev_departures_80 = 0
        ev_violations_80 = 0
        
        while True:
            action = agent.act(obs, info)
            obs, reward, term, trunc, info = env.step(action)
            
            ev_departures_80 += int(info.get("ev_departure_departures", 0))
            ev_violations_80 += int(info.get("ev_departure_violation_count_80pct", 0))
            
            if term or trunc:
                break
        
        env.close()
        
        result["violations"]["ev_80pct"] = {
            "count": ev_violations_80,
            "total": ev_departures_80,
            "rate_%": 100 * ev_violations_80 / max(1, ev_departures_80)
        }
        
        return result

evaluator = DualThresholdEvaluator(verify_env_vars=False)

agents = [
    ("RBC", RBCAgent(ev_mode="greedy")),
    ("PPOLag_V2", OmniSafeCheckpointAgent("runs/ppolag_P95_V2/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-05-02-08-01/torch_save/epoch-100.pt", "ppolag_v2")),
    ("FOCOPS_V2", OmniSafeCheckpointAgent("runs/focops_P95_V2/FOCOPS-{CityLearnSafety-SoC-v0}/seed-000-2026-02-05-03-18-10/torch_save/epoch-100.pt", "focops_v2")),
    ("RCPO_Aggressive", OmniSafeCheckpointAgent(RCPO_CKPT, "rcpo_aggressive")),
    ("TRPO_Baseline", OmniSafeCheckpointAgent(TRPO_CKPT, "trpo_baseline")),
]

results = []

for name, agent in agents:
    print(f"\nEvaluating {name}...")
    result = evaluator.evaluate(agent, seed=42, run_name=f"EVAL_{name}_dual")
    
    pol = result["policy"]
    viol = pol["violations"]
    orc = result["oracle"]
    costs = pol["cost_breakdown"]
    kpis = pol.get("citylearn_kpis", {})
    
    results.append({
        "Agent": name,
        "Reward": pol["reward"],
        "Cost": pol["cost"],
        "EV_Viol_100%": viol["ev"]["rate_%"],
        "EV_Count_100%": viol["ev"]["count"],
        "EV_Viol_80%": viol["ev_80pct"]["rate_%"],
        "EV_Count_80%": viol["ev_80pct"]["count"],
        "Total_Deps": viol["ev"]["total"],
        "Grid_Viol%": viol["grid"]["rate_%"],
        "Battery_Viol%": viol["battery"]["rate_%"],
        "Building_Viol%": viol["building"]["rate_%"],
        "Avoidable_kWh": orc["avoidable_kwh"],
        "Avoidable_%": orc["avoidable_percent"],
        "Cost_EV": costs["ev"],
        "Cost_Grid": costs["grid"],
        "Cost_Battery": costs["battery"],
        "Cost_Building": costs["building"],
        "CL_Consumption": kpis.get("citylearn_electricity_consumption_total", 0),
        "CL_Carbon": kpis.get("citylearn_carbon_emissions_total", 0),
        "CL_Cost": kpis.get("citylearn_cost_total", 0),
        "CL_Peak_Daily": kpis.get("citylearn_daily_peak_average", 0),
        "CL_Ramping": kpis.get("citylearn_ramping_average", 0),
    })
    
    print(f"  100% Strict:    {viol['ev']['rate_%']:6.2f}%  ({viol['ev']['count']}/{viol['ev']['total']})")
    print(f"  80% Tolerance:  {viol['ev_80pct']['rate_%']:6.2f}%  ({viol['ev_80pct']['count']}/{viol['ev']['total']})")
    print(f"  Reward: {pol['reward']:8.2f}, Cost: {pol['cost']:8.2f}")

df = pd.DataFrame(results)

# Save results
output_dir = Path("runs/evaluations")
output_dir.mkdir(exist_ok=True)

df.to_csv(output_dir / "5model_dual_threshold.csv", index=False)

print("\n" + "="*80)
print("VIOLATION COMPARISON (BOTH THRESHOLDS)")
print("="*80)
viol_df = df[["Agent", "EV_Viol_100%", "EV_Viol_80%", "Grid_Viol%", "Battery_Viol%", "Building_Viol%"]].copy()
print(viol_df.to_string(index=False))

print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)
perf_df = df[["Agent", "Reward", "Cost", "Avoidable_kWh", "Avoidable_%"]].copy()
print(perf_df.to_string(index=False))

print("\n" + "="*80)
print("CITYLEARN KPI COMPARISON")
print("="*80)
kpi_df = df[["Agent", "CL_Consumption", "CL_Carbon", "CL_Cost", "CL_Peak_Daily", "CL_Ramping"]].copy()
print(kpi_df.to_string(index=False))

print(f"\n✓ Saved: {output_dir / '5model_dual_threshold.csv'}")
print("="*80)
