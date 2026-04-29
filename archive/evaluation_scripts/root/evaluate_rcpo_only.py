#!/usr/bin/env python3
"""Evaluate RCPO Aggressive with full metrics"""

import os
import sys
import json
import glob
from pathlib import Path

sys.path.insert(0, os.getcwd())

from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent
from evaluation.core.evaluator import SafeRLEvaluator

# Find RCPO checkpoint
rcpo_pattern = "runs/*rcpo*/RCPO*/seed-*/torch_save/epoch-*.pt"
rcpo_matches = sorted(glob.glob(rcpo_pattern))

if not rcpo_matches:
    print(f"ERROR: RCPO checkpoint not found!")
    print(f"Pattern: {rcpo_pattern}")
    sys.exit(1)

# Use the latest epoch
RCPO_CKPT = rcpo_matches[-1]
print(f"Found RCPO checkpoint: {RCPO_CKPT}")

print("\n" + "="*80)
print("RCPO AGGRESSIVE EVALUATION")
print("="*80)

# Create agent
print(f"\nLoading checkpoint: {RCPO_CKPT}")
agent = OmniSafeCheckpointAgent(RCPO_CKPT, name="rcpo_aggressive")

# Create evaluator
evaluator = SafeRLEvaluator(verify_env_vars=False)

# Run evaluation
print(f"\nRunning evaluation (seed=42)...")
result = evaluator.evaluate(agent, seed=42, run_name="EVAL_RCPO_Aggressive")

# Display results
pol = result["policy"]
orc = result["oracle"]
viol = pol["violations"]

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nPerformance:")
print(f"  Reward:  {pol['reward']:>10.2f}")
print(f"  Cost:    {pol['cost']:>10.2f}")
print(f"  Steps:   {pol['steps']:>10d}")

print(f"\nViolations:")
print(f"  EV:       {viol['ev']['rate_%']:>6.2f}%  ({viol['ev']['count']}/{viol['ev']['total']})")
print(f"  Grid:     {viol['grid']['rate_%']:>6.2f}%")
print(f"  Battery:  {viol['battery']['rate_%']:>6.2f}%")
print(f"  Building: {viol['building']['rate_%']:>6.2f}%")

print(f"\nEV Analysis:")
print(f"  Total deficit:     {pol['ev_deficit_kwh']:>8.2f} kWh")
print(f"  Oracle deficit:    {orc['ev_deficit_kwh']:>8.2f} kWh")
print(f"  Avoidable gap:     {orc['avoidable_kwh']:>8.2f} kWh ({orc['avoidable_percent']:.1f}%)")

print(f"\nCost Breakdown:")
costs = pol['cost_breakdown']
print(f"  EV:       {costs['ev']:>10.2f}")
print(f"  Grid:     {costs['grid']:>10.2f}")
print(f"  Battery:  {costs['battery']:>10.2f}")
print(f"  Building: {costs['building']:>10.2f}")

kpis = pol.get('citylearn_kpis', {})
if kpis:
    print(f"\nCityLearn KPIs:")
    for key in sorted(kpis.keys()):
        if not key.startswith('citylearn_'):
            continue
        print(f"  {key[10:]}: {kpis[key]:.4f}")

# Save results
output_dir = Path("runs/evaluations/rcpo_aggressive_single")
output_dir.mkdir(parents=True, exist_ok=True)
json_path = output_dir / "results.json"
with open(json_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)

print(f"\n✓ Saved: {json_path}")
print("="*80 + "\n")
