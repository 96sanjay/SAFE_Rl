#!/usr/bin/env python3
"""Evaluate all 3 policies with both 100% and 80% tolerance"""

import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.getcwd())

from evaluation.agents.rbc import RBCAgent
from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent
from evaluation.core.evaluator import SafeRLEvaluator

print("="*80)
print("EVALUATION WITH DUAL THRESHOLDS (100% Strict + 80% Tolerance)")
print("="*80)

# Modified evaluator to track both thresholds
class DualThresholdEvaluator(SafeRLEvaluator):
    def _run_policy(self, agent, seed, run_name):
        result = super()._run_policy(agent, seed, run_name)
        
        # Add 80% tolerance violations (tracked separately in env)
        # We need to re-run and count from info dict
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
        
        # Add 80% tolerance metrics
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
]

results = []

for name, agent in agents:
    print(f"\nEvaluating {name}...")
    result = evaluator.evaluate(agent, seed=42, run_name=f"EVAL_{name}_dual")
    
    pol = result["policy"]
    viol = pol["violations"]
    
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
    })
    
    print(f"  100% Strict:    {viol['ev']['rate_%']:6.2f}%  ({viol['ev']['count']}/{viol['ev']['total']})")
    print(f"  80% Tolerance:  {viol['ev_80pct']['rate_%']:6.2f}%  ({viol['ev_80pct']['count']}/{viol['ev']['total']})")
    print(f"  Reduction:      {viol['ev']['rate_%'] - viol['ev_80pct']['rate_%']:6.2f}%")

df = pd.DataFrame(results)

# Save results
output_file = "runs/evaluations/dual_threshold_comparison.csv"
df.to_csv(output_file, index=False)

print("\n" + "="*80)
print("COMPARISON TABLE")
print("="*80)
print(df.to_string(index=False))

print(f"\n✓ Saved: {output_file}")
print("="*80)
