#!/usr/bin/env python3
"""Evaluate FOCOPS v2-shield checkpoint with full KPI breakdown."""
import os, sys, numpy as np, pandas as pd, torch
from pathlib import Path
from torch import nn

PROJECT_ROOT = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_PATH = PROJECT_ROOT / "runs/focops_v2_shield_100ep_v6style/FOCOPS-{CityLearnSafety-V2G-v2-shield}/seed-000-2026-02-19-19-02-27/torch_save/epoch-100.pt"
OUTPUT_CSV = PROJECT_ROOT / "runs/focops_v2_shield_c1c2_50ep_v6/eval_kpis_epoch50.csv"

# --- Environment setup (same stack as training) ---
os.environ['CITYLEARN_SCHEMA'] = "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
os.environ['CITYLEARN_USE_SHIELD'] = "1"
os.environ['SHIELD_ENABLE_C3'] = "0"
os.environ['SHIELD_ENABLE_C4'] = "0"
os.environ['SHIELD_VERBOSE'] = "1"

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.action_projection import ActionProjectionWrapper

print("\n" + "="*90)
print("  SHIELD v6 EVALUATION — FOCOPS epoch-50")
print("="*90)

base_env = make_base_env(central_agent=True)
safety = CityLearnSafetyEnvV3(base_env, soc_min=0.0, soc_max=0.95)
shielded = ActionProjectionWrapper(safety, verbose=1)
env = ForecastObsWrapper(shielded)

obs_space = env.observation_space
act_space = env.action_space
if isinstance(obs_space, list):
    obs_dim = sum(int(np.prod(sp.shape)) for sp in obs_space)
else:
    obs_dim = obs_space.shape[0]
if isinstance(act_space, list):
    act_dim = sum(int(np.prod(sp.shape)) for sp in act_space)
else:
    act_dim = act_space.shape[0]

print(f"Environment: obs_dim={obs_dim}, act_dim={act_dim}")

# --- Actor (must match training: [512, 256] + Tanh) ---
class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[512, 256]):
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

ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
actor = GaussianActor(obs_dim, act_dim, hidden_sizes=[512, 256])
actor.load_state_dict(ckpt['pi'])
actor.eval()
print(f"Actor loaded from {CHECKPOINT_PATH.name}")

norm = ckpt['obs_normalizer']
obs_mean = torch.FloatTensor(norm['_mean'])
obs_std = torch.FloatTensor(norm['_std'])
print(f"Normalizer loaded (obs dim={obs_mean.shape[0]})")

def _flat(obs):
    if isinstance(obs, list):
        return np.concatenate([np.asarray(o).ravel() for o in obs]).astype(np.float32)
    return np.asarray(obs, dtype=np.float32).ravel()

@torch.no_grad()
def policy(obs):
    obs_t = torch.FloatTensor(_flat(obs)).unsqueeze(0)
    obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
    return actor(obs_t).cpu().numpy().squeeze()

# --- Rollout ---
print(f"\nRunning evaluation episode...")
obs, _ = env.reset(seed=42)
data = []
step = 0
done = truncated = False

# KPI accumulators
c1_violations = 0  # EV departure SoC too low
c2_violations = 0  # Battery SoC out of bounds
c3_violations = 0  # Building power > threshold
c4_violations = 0  # Grid power > threshold
c3_total_checks = 0
c4_total_checks = 0
total_reward = 0.0
total_cost = 0.0
shield_interventions = 0

while not (done or truncated):
    action = np.clip(policy(obs), -1.0, 1.0)
    obs, r, done, truncated, info = env.step(action)
    info = dict(info) if info else {}

    # Accumulate KPIs
    total_reward += float(r)

    # C1: EV departure violation
    c1 = float(info.get("cost_ev_departure", 0.0))
    if c1 > 0: c1_violations += 1

    # C2: Battery SoC violation
    c2 = float(info.get("battery_soc_violation_frac", 0.0))
    if c2 > 0: c2_violations += 1

    # C3: Building power violations (per building)
    c3_count = float(info.get("building_power_violation_count", 0.0))
    if c3_count == 0:
        c3_count = float(info.get("cost_stems_building_power", 0.0))
    c3_violations += c3_count
    c3_total_checks += 17  # 17 buildings

    # C4: Grid power violation
    c4 = float(info.get("cost_stems_grid_power", 0.0))
    if c4 == 0:
        c4 = float(info.get("grid_power_violation", 0.0))
    if c4 > 0: c4_violations += 1
    c4_total_checks += 1

    # Shield
    sd = float(info.get("ap_action_delta_l2", 0.0))
    if sd > 0.001: shield_interventions += 1

    row = {'step': step, 'reward': float(r)}
    row.update(info)
    data.append(row)
    step += 1
    if step % 2000 == 0:
        print(f"  Step {step}: C1={c1_violations} C2={c2_violations} C3={c3_violations:.0f}/{c3_total_checks} C4={c4_violations}/{c4_total_checks}")

# --- Results ---
print("\n" + "="*90)
print("  FINAL RESULTS — Run A v6 (Shield C1+C2, CMDP C4)")
print("="*90)

c1_pct = 100.0 * c1_violations / max(1, step)
c2_pct = 100.0 * c2_violations / max(1, step)
c3_pct = 100.0 * c3_violations / max(1, c3_total_checks)
c4_pct = 100.0 * c4_violations / max(1, c4_total_checks)
shield_pct = 100.0 * shield_interventions / max(1, step)

print(f"  C1 (EV departure):     {c1_pct:6.2f}%  ({c1_violations}/{step} steps)")
print(f"  C2 (Battery SoC):      {c2_pct:6.2f}%  ({c2_violations}/{step} steps)")
print(f"  C3 (Building power):   {c3_pct:6.2f}%  ({c3_violations:.0f}/{c3_total_checks} building-steps)")
print(f"  C4 (Grid power):       {c4_pct:6.2f}%  ({c4_violations}/{c4_total_checks} steps)")
print(f"  Shield interventions:  {shield_pct:6.2f}%  ({shield_interventions}/{step} steps)")
print(f"  Total reward (STEMS):  {total_reward:10.2f}")
print(f"  Steps:                 {step}")
print("="*90)

# Save CSV
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV} ({len(df)} rows)")

# Compare with V2 baseline
print("\n  COMPARISON:")
print(f"  {'Metric':<25} {'V2 (no shield)':<20} {'Run A v6 (shield)':<20}")
print(f"  {'-'*65}")
print(f"  {'C1 (EV departure)':<25} {'100.00%':<20} {c1_pct:.2f}%")
print(f"  {'C2 (Battery SoC)':<25} {'0.52%':<20} {c2_pct:.2f}%")
print(f"  {'C3 (Building power)':<25} {'23.76%':<20} {c3_pct:.2f}%")
print(f"  {'C4 (Grid power)':<25} {'0.80%':<20} {c4_pct:.2f}%")
print(f"  {'Reward':<25} {'11,543':<20} {total_reward:.0f}")
