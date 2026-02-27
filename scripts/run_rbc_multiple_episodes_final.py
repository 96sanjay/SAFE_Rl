#!/usr/bin/env python3
"""
Run Intelligent RBC for multiple episodes - FINAL VERIFIED VERSION.

Key features:
- Accumulates EV deficits across ALL steps (not just end)
- Uses correct keys: cost_ev_departure_avoidable, cost_ev_departure_unavoidable
- Verifies CMDP cost matches avoidable deficits
- Saves comprehensive results for comparison with PPO-Lag
"""
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv
from scripts.run_intelligent_rbc_v2 import IntelligentRBC

def run_single_episode(episode_num, seed):
    """Run one episode and return comprehensive summary."""
    print(f"\n{'='*70}")
    print(f"RBC Episode {episode_num} (seed={seed})")
    print(f"{'='*70}")
    
    # Setup environment
    base_env = make_base_env(central_agent=True)
    env = CityLearnSafetyEnv(base_env, soc_min=0.0, soc_max=0.95)
    agent = IntelligentRBC(env)
    
    obs, info = env.reset(seed=seed)
    
    # Initialize accumulators
    total_cost = 0.0
    total_building_cost = 0.0
    total_ev_cost_cmdp = 0.0
    total_ev_avoidable = 0.0
    total_ev_unavoidable = 0.0
    total_reward = 0.0
    violation_steps = 0
    total_steps = 0
    
    # Run episode
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Accumulate CMDP costs (per-step values)
        step_cost = float(info.get("cost", 0.0))
        step_building = float(info.get("cost_building_soc", 0.0))
        step_ev_cmdp = float(info.get("cost_ev_departure", 0.0))
        step_ev_avoidable = float(info.get("cost_ev_departure_avoidable", 0.0))
        step_ev_unavoidable = float(info.get("cost_ev_departure_unavoidable", 0.0))
        
        total_cost += step_cost
        total_building_cost += step_building
        total_ev_cost_cmdp += step_ev_cmdp
        total_ev_avoidable += step_ev_avoidable
        total_ev_unavoidable += step_ev_unavoidable
        
        # Other metrics
        total_reward += reward
        violation_steps += int(info.get("soc_max", 0.0) > 0.95)
        total_steps += 1
        
        # Progress indicator (every 1000 steps)
        if total_steps % 1000 == 0:
            avg_cost = total_cost / total_steps
            viol_pct = 100 * violation_steps / total_steps
            print(f"  Step {total_steps:4d} | "
                  f"cost={avg_cost:.4f} | "
                  f"violations={viol_pct:.1f}% | "
                  f"ev_avoid={total_ev_avoidable:.2f} kWh")
    
    env.close()
    
    # Episode-end metrics (only available after completion)
    electricity_cost = float(info.get("citylearn_cost_total", 0.0))
    carbon_emissions = float(info.get("citylearn_carbon_emissions_total", 0.0))
    
    # Compile results
    result = {
        "episode": episode_num,
        "seed": seed,
        "steps": total_steps,
        
        # CMDP costs (what matters for training)
        "total_cost": total_cost,
        "building_cost": total_building_cost,
        "ev_cost_cmdp": total_ev_cost_cmdp,
        
        # EV deficit breakdown
        "ev_deficit_avoidable": total_ev_avoidable,
        "ev_deficit_unavoidable": total_ev_unavoidable,
        "ev_deficit_total": total_ev_avoidable + total_ev_unavoidable,
        
        # Safety metrics
        "violation_steps": violation_steps,
        "violation_rate": 100 * violation_steps / total_steps,
        
        # Performance metrics
        "total_reward": total_reward,
        "avg_reward": total_reward / total_steps,
        "electricity_cost": electricity_cost,
        "carbon_emissions": carbon_emissions,
    }
    
    # Print episode summary
    print(f"\n{'='*70}")
    print(f"Episode {episode_num} Results:")
    print(f"{'='*70}")
    print(f"Total CMDP Cost:           {result['total_cost']:.2f} kWh")
    print(f"  - Building SoC Cost:     {result['building_cost']:.2f} kWh ({100*result['building_cost']/result['total_cost']:.1f}%)")
    print(f"  - EV Cost (CMDP):        {result['ev_cost_cmdp']:.2f} kWh ({100*result['ev_cost_cmdp']/result['total_cost']:.1f}%)")
    print(f"")
    print(f"EV Deficit Breakdown:")
    print(f"  - Avoidable:             {result['ev_deficit_avoidable']:.2f} kWh")
    print(f"  - Unavoidable:           {result['ev_deficit_unavoidable']:.2f} kWh (NOT penalized)")
    print(f"  - Total (real-world):    {result['ev_deficit_total']:.2f} kWh")
    print(f"")
    print(f"Safety:")
    print(f"  - Violation Rate:        {result['violation_rate']:.2f}%")
    print(f"  - Violation Steps:       {result['violation_steps']} / {result['steps']}")
    print(f"")
    print(f"Economics:")
    print(f"  - Electricity Cost:      ${result['electricity_cost']:.2f}")
    print(f"  - Carbon Emissions:      {result['carbon_emissions']:.2f} kg")
    print(f"  - Avg Reward:            {result['avg_reward']:.2f}")
    
    # Verification check
    diff = abs(result['ev_cost_cmdp'] - result['ev_deficit_avoidable'])
    print(f"")
    print(f"Verification:")
    if diff < 0.01:
        print(f"  ✅ PASS: EV CMDP matches Avoidable (diff={diff:.6f})")
        print(f"     Agent correctly sees avoidable deficits only!")
    else:
        print(f"  ❌ FAIL: Mismatch detected!")
        print(f"     EV CMDP: {result['ev_cost_cmdp']:.6f}")
        print(f"     Avoidable: {result['ev_deficit_avoidable']:.6f}")
        print(f"     Difference: {diff:.6f}")
    
    return result

def main():
    """Run multiple episodes and aggregate statistics."""
    
    # Configuration
    n_episodes = 3
    seeds = [42, 123, 456]
    
    print(f"\n{'='*70}")
    print(f"INTELLIGENT RBC BASELINE - MULTI-EPISODE EVALUATION")
    print(f"{'='*70}")
    print(f"Episodes: {n_episodes}")
    print(f"Seeds: {seeds}")
    print(f"{'='*70}")
    
    # Run all episodes
    results = []
    for i, seed in enumerate(seeds, start=1):
        result = run_single_episode(i, seed)
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path("runs/baselines/intelligent_rbc")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "multi_episode_summary.csv"
    df.to_csv(csv_path, index=False)
    
    # Print aggregate statistics
    print(f"\n{'='*70}")
    print(f"AGGREGATE STATISTICS (n={n_episodes})")
    print(f"{'='*70}")
    
    print(f"\nCMDP Costs (what agent is penalized for):")
    print(f"  Total Cost:              {df['total_cost'].mean():.2f} ± {df['total_cost'].std():.2f} kWh")
    print(f"    - Building SoC:        {df['building_cost'].mean():.2f} ± {df['building_cost'].std():.2f} kWh")
    print(f"    - EV (Avoidable):      {df['ev_cost_cmdp'].mean():.2f} ± {df['ev_cost_cmdp'].std():.2f} kWh")
    
    print(f"\nEV Deficits:")
    print(f"  Avoidable:               {df['ev_deficit_avoidable'].mean():.2f} ± {df['ev_deficit_avoidable'].std():.2f} kWh")
    print(f"  Unavoidable (noise):     {df['ev_deficit_unavoidable'].mean():.2f} ± {df['ev_deficit_unavoidable'].std():.2f} kWh")
    print(f"  Total (real-world):      {df['ev_deficit_total'].mean():.2f} ± {df['ev_deficit_total'].std():.2f} kWh")
    
    unavoid_pct = 100 * df['ev_deficit_unavoidable'].mean() / df['ev_deficit_total'].mean()
    print(f"  Noise removed:           {unavoid_pct:.1f}% of total EV deficit")
    
    print(f"\nSafety Violations:")
    print(f"  Violation Rate:          {df['violation_rate'].mean():.2f}% ± {df['violation_rate'].std():.2f}%")
    
    print(f"\nEconomics:")
    print(f"  Electricity Cost:        ${df['electricity_cost'].mean():.2f} ± ${df['electricity_cost'].std():.2f}")
    print(f"  Carbon Emissions:        {df['carbon_emissions'].mean():.2f} ± {df['carbon_emissions'].std():.2f} kg")
    print(f"  Avg Reward:              {df['avg_reward'].mean():.2f} ± {df['avg_reward'].std():.2f}")
    
    # Final verification
    print(f"\n{'='*70}")
    print(f"FINAL VERIFICATION")
    print(f"{'='*70}")
    avg_cmdp = df['ev_cost_cmdp'].mean()
    avg_avoid = df['ev_deficit_avoidable'].mean()
    avg_diff = abs(avg_cmdp - avg_avoid)
    
    print(f"Avg EV CMDP Cost:          {avg_cmdp:.4f} kWh")
    print(f"Avg EV Avoidable Deficit:  {avg_avoid:.4f} kWh")
    print(f"Difference:                {avg_diff:.6f} kWh")
    
    if avg_diff < 0.01:
        print(f"\n✅ VERIFICATION PASSED!")
        print(f"   Agent correctly sees AVOIDABLE deficits only")
        print(f"   Unavoidable noise ({df['ev_deficit_unavoidable'].mean():.2f} kWh) removed from training signal")
    else:
        print(f"\n❌ VERIFICATION FAILED!")
        print(f"   Please check safety_env.py line 236")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*70}\n")
    
    return df

if __name__ == "__main__":
    main()
