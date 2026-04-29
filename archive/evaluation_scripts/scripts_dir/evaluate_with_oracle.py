import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.extractors_v3 import run_policy_and_oracle_rollouts, summarize_oracle_gap
import os
import glob

os.environ['CITYLEARN_SCHEMA'] = "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"

# Load checkpoint
checkpoint_pattern = str(PROJECT_ROOT / "runs/ppo_lag_greedy_baseline/*/seed-*/torch_save/epoch-100.pt")
checkpoints = glob.glob(checkpoint_pattern)
CHECKPOINT_PATH = Path(checkpoints[0])

ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')

# Reconstruct model
from torch import nn

class GaussianActor(nn.Module):
    def __init__(self, obs_dim=153, act_dim=26, hidden_sizes=[256, 256]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        self.mean = nn.Sequential(*layers, nn.Linear(in_dim, act_dim), nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs):
        return self.mean(obs)

actor = GaussianActor()
actor.load_state_dict(ckpt['pi'])
actor.eval()

obs_mean = torch.FloatTensor(ckpt['obs_normalizer']['_mean'])
obs_std = torch.FloatTensor(ckpt['obs_normalizer']['_std'])

@torch.no_grad()
def policy_fn(obs, info, env):
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
    action = actor(obs_t).cpu().numpy().squeeze()
    return np.clip(action, -1.0, 1.0)

def make_env():
    base = make_base_env(central_agent=True)
    return CityLearnSafetyEnvV3(base, soc_min=0.0, soc_max=0.95)

print("\n" + "="*80)
print("  ORACLE EVALUATION")
print("="*80)

print("\n🎮 Running POLICY rollout...")
print("🎮 Running ORACLE rollout (EV actions forced to 1.0)...")

policy_summary, oracle_summary = run_policy_and_oracle_rollouts(
    make_env=make_env,
    policy_action_fn=policy_fn,
    seed=42
)

print("\n📊 RESULTS:")
print(f"\nPolicy Rollout:")
print(f"  Total EV deficit: {policy_summary.deficit_kwh_total:.2f} kWh")
print(f"  Departures: {policy_summary.departures_total}")
print(f"  V3 Controllable: {policy_summary.cost_ev_controllable:.2f}")
print(f"  V3 Uncontrollable: {policy_summary.cost_ev_uncontrollable:.2f}")

print(f"\nOracle Rollout (EV=1.0 when plugged):")
print(f"  Total EV deficit: {oracle_summary.deficit_kwh_total:.2f} kWh")
print(f"  Departures: {oracle_summary.departures_total}")
print(f"  This is SIMULATOR MINIMUM (physics limit)")

gap = summarize_oracle_gap(policy_summary, oracle_summary)
print(f"\n🎯 GAP ANALYSIS:")
print(f"  Policy deficit: {gap['policy_deficit_kwh']:.2f} kWh")
print(f"  Oracle deficit: {gap['oracle_deficit_kwh']:.2f} kWh")
print(f"  Avoidable gap: {gap['avoidable_wrt_oracle_kwh']:.2f} kWh ({gap['avoidable_wrt_oracle_fraction']*100:.1f}%)")

if gap['avoidable_wrt_oracle_fraction'] < 0.05:
    print("\n✅ EXCELLENT! Policy is within 5% of oracle (theoretical best)")
elif gap['avoidable_wrt_oracle_fraction'] < 0.20:
    print("\n👍 GOOD! Policy is within 20% of oracle")
else:
    print("\n⚠️  Policy has significant room for improvement")

print("\n" + "="*80 + "\n")
