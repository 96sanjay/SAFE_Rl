#!/usr/bin/env python3
"""
Compare multiple Safe RL policies with complete metrics.

Usage:
  python evaluation/scripts/compare_policies.py \
    --agents rbc \
             ppolag_v2 \
             focops_v2 \
    --seed 42
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.getcwd())

from evaluation.agents.rbc import RBCAgent
from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent
from evaluation.core.evaluator import SafeRLEvaluator


# Checkpoint registry (update paths as needed)
CHECKPOINTS = {
    "ppolag_v2": "runs/ppolag_P95_V2/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-05-02-08-01/torch_save/epoch-100.pt",
    "focops_v2": "runs/focops_P95_V2/FOCOPS-{CityLearnSafety-SoC-v0}/seed-000-2026-02-05-03-18-10/torch_save/epoch-100.pt",
}


def parse_agent_spec(spec: str):
    """
    Parse agent specification:
    - "rbc" → RBCAgent
    - "ppolag_v2" → Load from registry
    - "omnisafe:/path/to/ckpt.pt" → Custom checkpoint
    """
    spec = spec.strip()
    
    if spec == "rbc":
        return RBCAgent(ev_mode="greedy")
    
    elif spec in CHECKPOINTS:
        ckpt_path = CHECKPOINTS[spec]
        return OmniSafeCheckpointAgent(ckpt_path, name=spec)
    
    elif spec.startswith("omnisafe:"):
        path = spec.split(":", 1)[1]
        if not path:
            raise ValueError("Empty checkpoint path")
        name = Path(path).stem
        return OmniSafeCheckpointAgent(path, name=name)
    
    else:
        raise ValueError(f"Unknown agent spec: {spec}. Use 'rbc', 'ppolag_v2', 'focops_v2', or 'omnisafe:/path'")


def main():
    parser = argparse.ArgumentParser(description="Compare Safe RL policies")
    parser.add_argument("--agents", nargs="+", required=True, 
                       help="Agent specs: rbc, ppolag_v2, focops_v2, omnisafe:/path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: runs/evaluations/YYYY-MM-DD_seedXX)")
    parser.add_argument("--no-verify-env", action="store_true",
                       help="Skip environment variable verification")
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        output_dir = Path(f"runs/evaluations/{timestamp}_seed{args.seed}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SAFE RL POLICY COMPARISON")
    print("="*80)
    print(f"Agents: {', '.join(args.agents)}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    # Create evaluator
    evaluator = SafeRLEvaluator(verify_env_vars=not args.no_verify_env)
    
    # Evaluate all agents
    results = []
    detailed_results = {}
    
    for spec in args.agents:
        try:
            agent = parse_agent_spec(spec)
            
            print(f"\n{'='*60}")
            print(f"Evaluating: {agent.name}")
            print(f"{'='*60}")
            
            result = evaluator.evaluate(agent, seed=args.seed)
            results.append(result)
            detailed_results[agent.name] = result
            
            # Print summary
            pol = result["policy"]
            orc = result["oracle"]
            viol = pol["violations"]
            
            print(f"\n✓ {agent.name} Results:")
            print(f"  Reward: {pol['reward']:.2f}")
            print(f"  Cost:   {pol['cost']:.2f}")
            print(f"  Violations:")
            print(f"    EV:       {viol['ev']['rate_%']:>6.2f}%")
            print(f"    Grid:     {viol['grid']['rate_%']:>6.2f}%")
            print(f"    Battery:  {viol['battery']['rate_%']:>6.2f}%")
            print(f"    Building: {viol['building']['rate_%']:>6.2f}%")
            print(f"  EV Oracle Gap: {orc['avoidable_kwh']:.2f} kWh ({orc['avoidable_percent']:.1f}%)")
            
        except Exception as e:
            print(f"ERROR evaluating {spec}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("\nNo agents evaluated successfully!")
        return
    
    # Save detailed JSON
    for agent_name, data in detailed_results.items():
        json_path = output_dir / f"{agent_name}_detailed.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n✓ Saved: {json_path}")
    
    # Create comparison summary CSV
    summary_rows = []
    for r in results:
        pol = r["policy"]
        orc = r["oracle"]
        viol = pol["violations"]
        costs = pol["cost_breakdown"]
        
        summary_rows.append({
            "agent": r["agent"],
            "seed": r["seed"],
            "reward": pol["reward"],
            "cost": pol["cost"],
            "ev_viol_%": viol["ev"]["rate_%"],
            "grid_viol_%": viol["grid"]["rate_%"],
            "battery_viol_%": viol["battery"]["rate_%"],
            "building_viol_%": viol["building"]["rate_%"],
            "cost_ev": costs["ev"],
            "cost_grid": costs["grid"],
            "cost_battery": costs["battery"],
            "cost_building": costs["building"],
            "oracle_avoidable_kwh": orc["avoidable_kwh"],
            "oracle_avoidable_%": orc["avoidable_percent"],
        })
    
    df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Saved: {csv_path}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
