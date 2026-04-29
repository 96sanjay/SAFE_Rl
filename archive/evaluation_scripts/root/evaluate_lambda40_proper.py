import os
import sys
import torch
import torch.nn as nn
import numpy as np
from glob import glob

sys.path.insert(0, '/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork')
from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim=153, act_dim=26):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs, deterministic=True):
        mean = self.mean(obs)
        if deterministic:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(self.log_std)

checkpoint_path = glob("runs/ppo_lag_3constraints_lambda40_highexplore_100ep/PPOLag-*/seed-*/torch_save/epoch-100.pt")[0]
print(f"Loading: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check structure
print(f"Checkpoint keys: {list(checkpoint.keys())}")
if 'obs_normalizer' in checkpoint:
    print(f"Normalizer keys: {list(checkpoint['obs_normalizer'].keys())}")

policy = GaussianPolicy()
policy.load_state_dict(checkpoint['pi'])
policy.eval()

# Fix normalizer access
obs_normalizer = checkpoint['obs_normalizer']

def normalize_obs(obs):
    if isinstance(obs_normalizer, dict):
        if 'mean' in obs_normalizer:
            mean = obs_normalizer['mean'].numpy()
            var = obs_normalizer['var'].numpy()
        elif '_mean' in obs_normalizer:
            mean = obs_normalizer['_mean'].numpy()
            var = obs_normalizer['_var'].numpy()
        else:
            print("No normalization available")
            return obs
    else:
        mean = obs_normalizer.mean.numpy()
        var = obs_normalizer.var.numpy()
    return (obs - mean) / np.sqrt(var + 1e-8)

os.environ['CITYLEARN_KPI_RUN_NAME'] = 'Lambda40_FreshEval'
base_env = make_base_env(central_agent=True)
env = CityLearnSafetyEnvV3(base_env)

print("\n" + "="*80)
print("FRESH EVALUATION: Lambda=40 Trained Model")
print("="*80)

obs, info = env.reset(seed=42)
done = False
step = 0

total_cost = 0.0
total_reward = 0.0
ev_cost = 0.0
peak_cost = 0.0
ramp_cost = 0.0
peak_viols = 0
ramp_viols = 0
ev_viols = 0

while not done:
    obs_norm = normalize_obs(obs)
    with torch.no_grad():
        action = policy(torch.FloatTensor(obs_norm).unsqueeze(0), deterministic=True).squeeze(0).numpy()
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    total_reward += reward
    total_cost += info.get('cost', 0.0)
    ev_cost += info.get('cost_ev_departure', 0.0)
    peak_cost += info.get('cost_grid_peak', 0.0)
    ramp_cost += info.get('cost_grid_ramp', 0.0)
    
    if info.get('grid_peak_violation', 0.0) > 0:
        peak_viols += 1
    if info.get('grid_ramp_violation', 0.0) > 0:
        ramp_viols += 1
    if info.get('ev_controllable_deficit_kwh', 0.0) > 0:
        ev_viols += 1
    
    step += 1
    if step % 1000 == 0:
        print(f"  Step {step}/8759")

print(f"\nTotal Steps:         {step}")
print(f"Total Reward:        {total_reward:.2f}")
print(f"Total Cost:          {total_cost:.2f}")
print(f"  EV Cost:           {ev_cost:.2f} ({ev_cost/total_cost*100:.1f}%)")
print(f"  Peak Cost:         {peak_cost:.2f} ({peak_cost/total_cost*100:.1f}%)")
print(f"  Ramp Cost:         {ramp_cost:.2f} ({ramp_cost/total_cost*100:.1f}%)")
print()
print(f"EV Violations:       {ev_viols} ({ev_viols/step*100:.2f}%)")
print(f"Peak Violations:     {peak_viols} ({peak_viols/step*100:.2f}%)")
print(f"Ramp Violations:     {ramp_viols} ({ramp_viols/step*100:.2f}%)")
print()

budget = 350.0
rbc_cost = 304.38
print(f"Budget:              {budget:.2f}")
print(f"RBC:                 {rbc_cost:.2f}")
print(f"Lambda=40 (eval):    {total_cost:.2f}")
print()

if total_cost < budget:
    print("✅ UNDER BUDGET")
if total_cost < rbc_cost:
    print(f"✅ Beats RBC by {rbc_cost - total_cost:.2f}")

print("="*80)
