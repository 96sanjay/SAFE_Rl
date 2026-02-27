#!/usr/bin/env python3
import os, sys, numpy as np, torch, glob
from pathlib import Path
from torch import nn

PROJECT_ROOT = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, str(PROJECT_ROOT))

# Load NEW checkpoint
checkpoints = glob.glob(str(PROJECT_ROOT / "runs/ppo_lag_cost_based_reward/*/seed-*/torch_save/epoch-100.pt"))
if not checkpoints: raise FileNotFoundError("No NEW checkpoint found")
CHECKPOINT_PATH = Path(checkpoints[0])

os.environ['CITYLEARN_SCHEMA'] = "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

print("\n" + "="*80)
print("ORACLE EVALUATION - V3 Classification Verification")
print("="*80)

# Setup
base_env = make_base_env(central_agent=True)
env = CityLearnSafetyEnvV3(base_env, soc_min=0.0, soc_max=0.95)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Load actor
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64]):
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

actor = GaussianActor(obs_dim, act_dim, hidden_sizes=[64, 64])
actor.load_state_dict(ckpt['pi'])
actor.eval()

norm = ckpt['obs_normalizer']
obs_mean = torch.FloatTensor(norm['_mean'])
obs_std = torch.FloatTensor(norm['_std'])

@torch.no_grad()
def policy(obs):
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
    return actor(obs_t).cpu().numpy().squeeze()

# EV action indices (17 batteries, then 8 EVs)
EV_START_IDX = 17
EV_END_IDX = 25

print(f"\n🔍 EV chargers: indices {EV_START_IDX} to {EV_END_IDX-1}")

# ============================================================================
# ROLLOUT 1: POLICY (Agent's learned behavior)
# ============================================================================
print("\n" + "="*80)
print("ROLLOUT 1: Policy (Agent's Decisions)")
print("="*80)

obs, _ = env.reset(seed=42)
policy_deficit = 0
policy_departures = 0
done = truncated = False
step = 0

while not (done or truncated):
    action = np.clip(policy(obs), -1.0, 1.0)
    obs, r, done, truncated, info = env.step(action)
    
    policy_deficit += info.get('ev_departure_deficit_kwh', 0)
    policy_departures += info.get('ev_departure_departures', 0)
    step += 1
    
    if step % 2000 == 0:
        print(f"  Step {step}: Deficit={policy_deficit:.2f} kWh, Departures={policy_departures}")

print(f"\n✅ Policy Complete:")
print(f"   Total EV Deficit: {policy_deficit:.2f} kWh")
print(f"   Departures: {policy_departures}")
print(f"   Avg deficit/departure: {policy_deficit/max(1,policy_departures):.2f} kWh")

# ============================================================================
# ROLLOUT 2: ORACLE (Force max charging when plugged)
# ============================================================================
print("\n" + "="*80)
print("ROLLOUT 2: Oracle (Forced Max Charging)")
print("="*80)

obs, _ = env.reset(seed=42)
oracle_deficit = 0
oracle_departures = 0
oracle_forced_actions = 0
done = truncated = False
step = 0

while not (done or truncated):
    # Get agent's action as starting point
    action = np.clip(policy(obs), -1.0, 1.0)
    
    # ORACLE: Force EV actions to 1.0 if EV is plugged in
    # Check observation to see if EV is connected
    # In CityLearn schema, EV connection is in observations
    for ev_idx in range(EV_START_IDX, EV_END_IDX):
        # Force max charge for all EV chargers
        action[ev_idx] = 1.0
        oracle_forced_actions += 1
    
    obs, r, done, truncated, info = env.step(action)
    
    oracle_deficit += info.get('ev_departure_deficit_kwh', 0)
    oracle_departures += info.get('ev_departure_departures', 0)
    step += 1
    
    if step % 2000 == 0:
        print(f"  Step {step}: Deficit={oracle_deficit:.2f} kWh, Departures={oracle_departures}")

print(f"\n✅ Oracle Complete:")
print(f"   Total EV Deficit: {oracle_deficit:.2f} kWh")
print(f"   Departures: {oracle_departures}")
print(f"   Avg deficit/departure: {oracle_deficit/max(1,oracle_departures):.2f} kWh")
print(f"   Forced actions: {oracle_forced_actions}")

# ============================================================================
# GAP ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("GAP ANALYSIS")
print("="*80)

avoidable_gap = policy_deficit - oracle_deficit
avoidable_pct = 100 * avoidable_gap / max(1, policy_deficit)

print(f"\n📊 Deficit Comparison:")
print(f"   Policy Deficit:  {policy_deficit:.2f} kWh")
print(f"   Oracle Deficit:  {oracle_deficit:.2f} kWh")
print(f"   Avoidable Gap:   {avoidable_gap:.2f} kWh ({avoidable_pct:.1f}%)")

print(f"\n🔍 V3 Classification Check:")
print(f"   V3 said controllable:    1.00 kWh")
print(f"   Oracle reveals avoidable: {avoidable_gap:.2f} kWh")

if abs(avoidable_gap - 1.0) < 50:
    print(f"\n✅ V3 CLASSIFICATION ACCURATE!")
    print(f"   Gap matches V3 prediction (~1 kWh)")
    print(f"   Agent is near-optimal")
elif avoidable_gap > 500:
    print(f"\n❌ V3 CLASSIFICATION BROKEN!")
    print(f"   V3 said only 1 kWh controllable")
    print(f"   Oracle reveals {avoidable_gap:.2f} kWh avoidable")
    print(f"   V3 misclassified {avoidable_gap-1:.2f} kWh as 'uncontrollable'")
else:
    print(f"\n⚠️  MODERATE DISCREPANCY")
    print(f"   V3 slightly underestimated controllable deficit")

print("\n" + "="*80 + "\n")
