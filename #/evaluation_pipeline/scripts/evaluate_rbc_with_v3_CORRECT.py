"""
RBC Baseline with CORRECT V3 EV Classification
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.getcwd())

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3


def get_hour(env):
    """Extract hour from environment"""
    try:
        raw = env._get_citylearn_env()
        if raw is not None:
            t = int(getattr(raw, 'time_step', 0))
            return t % 24
    except:
        pass
    return 12  # default


def rbc_policy(env, hour):
    """
    RBC Policy:
    - Battery: Charge 10-16h, Discharge 17-21h
    - EV: Greedy charging (always 1.0)
    """
    action_dim = env.action_space.shape[0]
    action = np.zeros(action_dim, dtype=np.float32)
    
    # Battery (first 17 actions)
    if 10 <= hour < 16:
        action[:17] = -0.5  # Charge
    elif 17 <= hour < 21:
        action[:17] = 0.5   # Discharge
    
    # EV: Get correct indices from environment
    ev_indices = env._ev_charger_action_indices
    for idx in ev_indices:
        action[idx] = 1.0  # Greedy charging
    
    return action


def run_rbc_evaluation():
    """Run RBC with V3 classification"""
    
    print("="*80)
    print("RBC BASELINE EVALUATION (V3 Classification)")
    print("="*80)
    
    # Setup environment
    os.environ["CITYLEARN_SCHEMA"] = "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
    os.environ["CITYLEARN_KPI_RUN_NAME"] = "RBC_Greedy_V3"
    os.environ["CITYLEARN_EXPORT_FACTOR"] = "0.7"
    os.environ["CITYLEARN_REWARD_SCALE"] = "1.0"
    
    base_env = make_base_env(central_agent=True)
    env = CityLearnSafetyEnvV3(
        base_env,
        soc_min=0.0,
        soc_max=0.95,
        cost_mode="hinge",
        include_ev_in_cost=True,
    )
    
    print(f"\n✅ Environment created")
    print(f"   EV action indices (V3): {env._ev_charger_action_indices}")
    
    # Run episode
    obs, info = env.reset(seed=42)
    
    total_cost = 0.0
    total_reward = 0.0
    violations = 0
    steps = 0
    
    # Tracking
    ev_v3_controllable = []
    ev_v3_uncontrollable = []
    ev_missing_actions = []
    
    done = False
    
    print("\n🚀 Running evaluation (8759 steps, ~5 mins)...")
    
    while not done:
        hour = get_hour(env)
        action = rbc_policy(env, hour)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track metrics
        cost = float(info.get('cost', 0.0))
        total_cost += cost
        total_reward += reward
        if cost > 0:
            violations += 1
        
        ev_v3_controllable.append(float(info.get('cost_ev_departure_agent_controllable_v3', 0.0)))
        ev_v3_uncontrollable.append(float(info.get('cost_ev_departure_uncontrollable_v3', 0.0)))
        ev_missing_actions.append(float(info.get('ev_missing_action_samples', 0.0)))
        
        steps += 1
        
        if steps % 1000 == 0:
            print(f"   Step {steps}/8759")
    
    print(f"\n✅ Evaluation complete!")
    
    # V3 verification
    total_missing = sum(ev_missing_actions)
    total_v3_controllable = sum(ev_v3_controllable)
    total_v3_uncontrollable = sum(ev_v3_uncontrollable)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\n📊 Constraint Performance:")
    print(f"   Total Cost:             {total_cost:.2f}")
    print(f"   Violation Steps:        {violations}")
    print(f"   Violation %:            {violations/steps*100:.2f}%")
    
    print(f"\n🚗 V3 EV Classification:")
    print(f"   Controllable Deficit:   {total_v3_controllable:.2f} kWh")
    print(f"   Uncontrollable Deficit: {total_v3_uncontrollable:.2f} kWh")
    print(f"   Missing Actions:        {int(total_missing)}")
    
    if total_missing == 0:
        print(f"\n   ✅ V3 WORKING! Zero missing actions")
    else:
        print(f"\n   ❌ V3 ISSUE! {int(total_missing)} missing actions")
    
    print(f"\n💰 Reward:")
    print(f"   Total Reward:           {total_reward:.2f}")
    
    # CityLearn KPIs
    print(f"\n🏢 CityLearn KPIs:")
    kpi_keys = [
        'citylearn_electricity_consumption_total',
        'citylearn_carbon_emissions_total',
        'citylearn_cost_total',
        'citylearn_daily_peak_average',
    ]
    for key in kpi_keys:
        if key in info:
            print(f"   {key:45s}: {info[key]:.4f}")
    
    # Save results
    results = {
        'controller': 'RBC_Greedy',
        'total_cost': total_cost,
        'violations': violations,
        'violation_pct': violations/steps*100,
        'total_reward': total_reward,
        'ev_v3_controllable': total_v3_controllable,
        'ev_v3_uncontrollable': total_v3_uncontrollable,
        'ev_missing_actions': total_missing,
        **{k: info.get(k, 0) for k in kpi_keys}
    }
    
    df = pd.DataFrame([results])
    df.to_csv('rbc_greedy_v3_results.csv', index=False)
    
    print(f"\n✅ Saved: rbc_greedy_v3_results.csv")
    print(f"✅ Full KPIs: runs/kpi_logs/RBC_Greedy_V3.csv")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    run_rbc_evaluation()
