#!/usr/bin/env python3
import os, sys, numpy as np, pandas as pd, torch, glob
from pathlib import Path
from torch import nn

PROJECT_ROOT = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, str(PROJECT_ROOT))

checkpoints = glob.glob(str(PROJECT_ROOT / "runs/ppo_lag_cost_based_reward/*/seed-*/torch_save/epoch-100.pt"))
if not checkpoints: raise FileNotFoundError("No NEW checkpoint found")
CHECKPOINT_PATH = Path(checkpoints[0])
OUTPUT_CSV = PROJECT_ROOT / "runs/ppo_lag_cost_based_reward/kpis_ppolag_NEW_TRAINED_MODEL.csv"

print(f"✅ Loading NEW model from: {CHECKPOINT_PATH}")

os.environ['CITYLEARN_SCHEMA'] = "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

print("\n" + "="*90)
print("  NEW TRAINED MODEL EVALUATION (Cost-Based Reward)")
print("="*90)

base_env = make_base_env(central_agent=True)
env = CityLearnSafetyEnvV3(base_env, soc_min=0.0, soc_max=0.95)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(f"✅ Environment: obs_dim={obs_dim}, act_dim={act_dim}")

print(f"\n📦 Loading checkpoint...")
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64]):  # ✅ FIXED
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

actor = GaussianActor(obs_dim, act_dim, hidden_sizes=[64, 64])  # ✅ FIXED
actor.load_state_dict(ckpt['pi'])
actor.eval()
print("✅ Actor loaded")

norm = ckpt['obs_normalizer']
obs_mean = torch.FloatTensor(norm['_mean'])
obs_std = torch.FloatTensor(norm['_std'])
print(f"✅ Normalizer loaded (mean shape: {obs_mean.shape}, std shape: {obs_std.shape})")

@torch.no_grad()
def policy(obs):
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
    return actor(obs_t).cpu().numpy().squeeze()

print(f"\n🎮 Evaluating...")
obs, _ = env.reset(seed=42)
data, cost, reward, step = [], 0, 0, 0
done = truncated = False

while not (done or truncated):
    action = np.clip(policy(obs), -1.0, 1.0)
    obs, r, done, truncated, info = env.step(action)
    row = {'step': step, 'reward': r}
    for i in range(act_dim):
        row[f'action_{i}'] = action[i]
        if i < 17: row[f'action_battery_b{i}'] = action[i]
        elif i < 25: row[f'action_ev_{i-17}'] = action[i]
    if act_dim == 26: row['action_washing_machine'] = action[25]
    row.update(info)
    data.append(row)
    cost += info.get('cost', 0)
    reward += r
    step += 1
    if step % 1000 == 0: print(f"   Step {step}: Cost={cost:.2f}, Reward={reward:.2f}")

print(f"\n✅ Complete! Cost={cost:.2f}, Reward={reward:.2f}")

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"💾 Saved: {OUTPUT_CSV} ({len(df)} rows × {len(df.columns)} cols)")

# Battery analysis
soc_cols = [c for c in df.columns if 'soc' in c.lower() and 'battery' in c.lower()]
if soc_cols:
    soc_mean = df[soc_cols].mean().mean()
    soc_max = df[soc_cols].max().max()
    print(f"\n🔋 Battery Usage:")
    print(f"   Mean SOC: {soc_mean:.4f}")
    print(f"   Max SOC:  {soc_max:.4f}")
    if soc_mean > 0.1:
        print("   ✅ Batteries ARE being used!")
    else:
        print("   ❌ Batteries NOT used")
else:
    print("\n⚠️  Battery SOC columns not in output")

print(f"\n📊 Comparison:")
print(f"   OLD (broken):    Cost=96.78,  Reward=-10,349")
print(f"   NEW (expected):  Cost=2.01,   Reward=-11,634")
print(f"   THIS eval:       Cost={cost:.2f}, Reward={reward:.2f}")

if abs(reward - (-11634)) < 1000:
    print(f"\n✅ REWARD FIX WORKING!")
elif abs(reward - (-10349)) < 500:
    print(f"\n❌ Still using OLD reward")
else:
    print(f"\n⚠️  Unexpected value")

print("\n" + "="*90 + "\n")
