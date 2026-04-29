#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['CITYLEARN_SCHEMA'] = str(PROJECT_ROOT / "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json")
os.environ['CITYLEARN_KPI_RUN_NAME'] = "EvaluateFinalAgent_Lambda35_Epoch100"
os.environ['CITYLEARN_EV_COST_SCALE'] = "3.0"

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

print("="*80)
print("EVALUATING LAMBDA=35 FINAL AGENT")
print("="*80)

# Load checkpoint
checkpoint_path = PROJECT_ROOT / "runs/ppo_lag_3constraints_lambda35_evweight3_100ep/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-01-14-08-00-59/torch_save/epoch-100.pt"
print(f"\n📦 Loading: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

pi_state = checkpoint['pi']
print(f"✅ Pi keys: {list(pi_state.keys())}")

# Create environment
print(f"\n🏗️  Creating environment...")
base_env = make_base_env(central_agent=True)
env = CityLearnSafetyEnvV3(base_env)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

# Match OmniSafe's Gaussian policy structure
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Mean network (what checkpoint has)
        self.mean = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
            nn.Tanh(),
        )
        # Log std (learned parameter)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs, deterministic=True):
        mean = self.mean(obs)
        if deterministic:
            return mean
        else:
            std = torch.exp(self.log_std)
            return mean + torch.randn_like(mean) * std

policy = GaussianPolicy(obs_dim, act_dim)
policy.load_state_dict(pi_state)
policy.eval()
print(f"✅ Policy loaded successfully")

# Evaluate
print(f"\n🎮 Running evaluation (deterministic actions)...")
obs, info = env.reset(seed=42)
total_cost = 0
total_reward = 0
step = 0

with torch.no_grad():
    while step < 8760:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = policy(obs_tensor, deterministic=True).squeeze(0).cpu().numpy()
        action = np.clip(action, -1, 1)
        
        obs, reward, done, truncated, info = env.step(action)
        
        total_cost += info.get('cost', 0)
        total_reward += reward
        step += 1
        
        if step % 1000 == 0:
            print(f"Step {step}: Cost={total_cost:.2f}, Reward={total_reward:.2f}")
        
        if done or truncated:
            break

print(f"\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Total Cost:   {total_cost:.2f}")
print(f"  Target:     < 350")
print(f"  RBC:        304")
print(f"  Training:   422 (from progress.csv)")
print(f"\nTotal Reward: {total_reward:.2f}")
print(f"Steps:        {step}")

if total_cost < 350:
    print(f"\n✅ SUCCESS! Under budget!")
elif total_cost < 450:
    print(f"\n⚠️  Close but over budget by {total_cost - 350:.1f}")
else:
    print(f"\n❌ High cost - check if policy loaded correctly")

print(f"\n✅ Full KPIs: runs/kpi_logs/EvaluateFinalAgent_Lambda35_Epoch100.csv")
print("="*80)
