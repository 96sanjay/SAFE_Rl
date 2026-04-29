#!/usr/bin/env python3
"""
Quick single-episode evaluation of trained PPOLag checkpoint.

Shows:
- Per-constraint violation counts and rates
- CityLearn default KPIs
- Episode reward and cost

Usage:
  python eval_single_episode.py \
    --checkpoint runs/ppolag_APD_InvLin/PPOLag-{...}/torch_save/epoch-100.pt \
    --seed 42
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.getcwd())

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.adapters import SingleAgentListAdapter
from citylearn.wrappers import NormalizedObservationWrapper


class GaussianActor(nn.Module):
    """Actor network (must match training config)."""
    def __init__(self, obs_dim, act_dim, hidden_sizes=[512, 512, 256]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.mean = nn.Sequential(*layers, nn.Linear(prev, act_dim))
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs):
        return torch.tanh(self.mean(obs))


def make_env():
    schema = os.environ.get("CITYLEARN_SCHEMA", "")
    base = CityLearnEnv(schema=schema, central_agent=True)
    base = NormalizedObservationWrapper(base)
    env = SingleAgentListAdapter(base)
    return CityLearnSafetyEnvV3(env)


def evaluate(checkpoint_path, seed):
    print("\n" + "="*80)
    print("SINGLE EPISODE EVALUATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Seed: {seed}")
    print("="*80 + "\n")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Create env
    env = make_env()
    obs, _ = env.reset(seed=seed)
    
    # Create actor
    actor = GaussianActor(len(obs), env.action_space.shape[0])
    actor.load_state_dict(ckpt['pi'])
    actor.eval()
    
    # Tracking
    ep_reward = 0.0
    ep_cost = 0.0
    steps = 0
    
    # Constraint violation tracking
    ev_departures = 0
    ev_violations = 0
    
    grid_violations = 0
    battery_violations = 0
    building_violations = 0
    
    # Cost breakdown
    cost_ev = 0.0
    cost_grid = 0.0
    cost_battery = 0.0
    cost_building = 0.0
    
    # CityLearn KPIs
    total_carbon = 0.0
    total_grid_import = 0.0
    total_grid_export = 0.0
    total_solar = 0.0
    total_elec_cost = 0.0
    
    # Run episode
    while True:
        with torch.no_grad():
            action = actor(torch.FloatTensor(obs).unsqueeze(0)).squeeze(0).numpy()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        steps += 1
        ep_reward += reward
        ep_cost += info.get('cost', 0)
        
        # ==== CONSTRAINT VIOLATIONS ====
        
        # EV: Count departures with deficit > 0.01 kWh
        ev_dep = int(info.get('ev_departure_departures', 0))
        ev_deficit = float(info.get('ev_departure_deficit_kwh', 0))
        if ev_dep > 0:
            ev_departures += ev_dep
            if ev_deficit > 0.01:
                ev_violations += 1
        
        # Grid: cost > 0 means violation
        if float(info.get('cost_stems_grid_power', 0)) > 0:
            grid_violations += 1
        
        # Battery: per-building SoC violations
        battery_violations += int(info.get('battery_soc_violation_count', 0))
        
        # Building: per-building power violations
        building_violations += int(info.get('building_power_violation_count', 0))
        
        # ==== COST BREAKDOWN ====
        cost_ev += float(info.get('cost_ev_departure', 0))
        cost_grid += float(info.get('cost_stems_grid_power', 0))
        cost_battery += float(info.get('cost_stems_battery', 0))
        cost_building += float(info.get('cost_stems_building_power', 0))
        
        # ==== CITYLEARN KPIS ====
        # Access base environment for CityLearn metrics
        base_env = env.unwrapped
        if hasattr(base_env, 'buildings'):
            for b in base_env.buildings:
                total_carbon += b.carbon_emission[-1] if b.carbon_emission else 0
                total_elec_cost += b.net_electricity_consumption_cost[-1] if b.net_electricity_consumption_cost else 0
                
                # Grid import/export
                net = b.net_electricity_consumption[-1] if b.net_electricity_consumption else 0
                if net > 0:
                    total_grid_import += net
                else:
                    total_grid_export += abs(net)
                
                # Solar
                total_solar += b.solar_generation[-1] if b.solar_generation else 0
        
        if terminated or truncated:
            break
    
    env.close()
    
    # ==== RESULTS ====
    print("\n" + "="*80)
    print("EPISODE SUMMARY")
    print("="*80)
    print(f"Steps:          {steps}")
    print(f"Episode Reward: {ep_reward:.2f}")
    print(f"Episode Cost:   {ep_cost:.2f}")
    print()
    
    print("="*80)
    print("CONSTRAINT VIOLATIONS")
    print("="*80)
    
    # EV violations
    ev_rate = 100 * ev_violations / max(ev_departures, 1)
    print(f"EV Departure:")
    print(f"  Departures with deficit > 0.01 kWh: {ev_violations} / {ev_departures}")
    print(f"  Violation Rate: {ev_rate:.2f}%")
    print()
    
    # Grid violations
    grid_rate = 100 * grid_violations / steps
    print(f"Grid Power:")
    print(f"  Timesteps exceeding limit: {grid_violations} / {steps}")
    print(f"  Violation Rate: {grid_rate:.2f}%")
    print()
    
    # Battery violations (17 buildings × steps)
    battery_rate = 100 * battery_violations / (steps * 17)
    print(f"Battery SoC:")
    print(f"  Building-timesteps violating: {battery_violations} / {steps * 17}")
    print(f"  Violation Rate: {battery_rate:.2f}%")
    print()
    
    # Building violations
    building_rate = 100 * building_violations / (steps * 17)
    print(f"Building Power:")
    print(f"  Building-timesteps violating: {building_violations} / {steps * 17}")
    print(f"  Violation Rate: {building_rate:.2f}%")
    print()
    
    print("="*80)
    print("COST BREAKDOWN")
    print("="*80)
    print(f"EV Departure:    {cost_ev:>12.2f}  ({100*cost_ev/ep_cost:>5.1f}%)")
    print(f"Grid Power:      {cost_grid:>12.2f}  ({100*cost_grid/ep_cost:>5.1f}%)")
    print(f"Battery SoC:     {cost_battery:>12.2f}  ({100*cost_battery/ep_cost:>5.1f}%)")
    print(f"Building Power:  {cost_building:>12.2f}  ({100*cost_building/ep_cost:>5.1f}%)")
    print(f"{'-'*80}")
    print(f"TOTAL:           {ep_cost:>12.2f}")
    print()
    
    print("="*80)
    print("CITYLEARN DEFAULT KPIS")
    print("="*80)
    print(f"Total Carbon Emission:    {total_carbon:.2f} kg CO2")
    print(f"Total Electricity Cost:   ${total_elec_cost:.2f}")
    print(f"Total Grid Import:        {total_grid_import:.2f} kWh")
    print(f"Total Grid Export:        {total_grid_export:.2f} kWh")
    print(f"Total Solar Generation:   {total_solar:.2f} kWh")
    print(f"Net Grid Consumption:     {total_grid_import - total_grid_export:.2f} kWh")
    print()
    
    # Self-consumption rate
    if total_solar > 0:
        self_consumed = total_solar - total_grid_export
        self_consumption_rate = 100 * self_consumed / total_solar
        print(f"Solar Self-Consumption:   {self_consumption_rate:.1f}%")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.seed)
