#!/usr/bin/env python3
"""Evaluate all 3 FOCOPS checkpoints: Free vs PSF. Handles different architectures."""
import os, time
import numpy as np
import torch
import torch.nn as nn

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper

SEED = 7
MAX_STEPS = 8760

class GaussianActor(nn.Module):
    def __init__(self, obs_dim=281, act_dim=26, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.mean = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    def forward(self, obs):
        return torch.tanh(self.mean(obs))
    def predict(self, obs, deterministic=True):
        return self.forward(obs), None

class ObsNormalizer:
    def __init__(self, mean, std, clip):
        self.mean = mean; self.std = std; self.clip = clip
    def normalize(self, obs):
        return torch.clamp((obs - self.mean) / (self.std + 1e-8), -self.clip, self.clip)

def infer_hidden_sizes(state_dict):
    """Auto-detect hidden layer sizes from state_dict keys."""
    sizes = []
    i = 0
    while f"mean.{i}.weight" in state_dict:
        w = state_dict[f"mean.{i}.weight"]
        sizes.append(w.shape[0])
        i += 2  # skip ReLU (no params but Sequential indexes it)
    # Last one is the output layer, don't include
    return tuple(sizes[:-1]) if len(sizes) > 1 else (256,)

def load_actor(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    pi = data["pi"]
    hidden = infer_hidden_sizes(pi)
    print(f"  Architecture: 281 -> {' -> '.join(map(str,hidden))} -> 26")
    actor = GaussianActor(hidden_sizes=hidden)
    actor.load_state_dict(pi)
    actor.eval()
    nd = data["obs_normalizer"]
    return actor, ObsNormalizer(nd["_mean"], nd["_std"], nd["_clip"])

CONFIGS = [
    {
        "name": "FOCOPS-100ep (no shield)",
        "ckpt": "runs/focops_v2_100ep/FOCOPS-{CityLearnSafety-V2G-v2}/seed-000-2026-02-17-00-19-14/torch_save/epoch-100.pt",
    },
    {
        "name": "FOCOPS-v3 (shield, 20ep)",
        "ckpt": "runs/focops_v2_shield_c1c2_50ep_v3/FOCOPS-{CityLearnSafety-V2G-v2-shield}/seed-000-2026-02-18-11-45-50/torch_save/epoch-20.pt",
    },
    {
        "name": "FOCOPS-v6 (shield, 50ep)",
        "ckpt": "runs/focops_v2_shield_c1c2_50ep_v6/FOCOPS-{CityLearnSafety-V2G-v2-shield}/seed-000-2026-02-18-16-11-48/torch_save/epoch-50.pt",
    },
]

for c in CONFIGS:
    assert os.path.isfile(c["ckpt"]), f"Missing: {c['ckpt']}"
print(f"All checkpoints found. Seed={SEED}\n")

def build_env_free():
    base = make_base_env()
    env = CityLearnSafetyEnvV3(base, soc_min=0.0, soc_max=0.95)
    return ForecastObsWrapper(env)

def build_env_psf():
    base = make_base_env()
    env = CityLearnSafetyEnvV3(base, soc_min=0.0, soc_max=0.95)
    env = ForecastObsWrapper(env)
    return LookaheadPSFWrapper(env, horizon=24, verbose=0)

def rollout(env, actor, normalizer, seed, max_steps, label=""):
    obs, info = env.reset(seed=seed)
    obs = np.concatenate(obs) if isinstance(obs, list) else np.asarray(obs, dtype=np.float32)
    total_reward = 0.0; total_cost = 0.0; step_count = 0
    c1_dep = 0; c1_viol = 0; c1_deficit = 0.0
    c2_viol = 0; c3_viol = 0; c4_viol = 0
    psf_interv = 0; psf_infeas = 0; psf_times = []; psf_deltas = []
    t0 = time.time()
    for step in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_t = normalizer.normalize(obs_t)
        with torch.no_grad():
            action, _ = actor.predict(obs_t, deterministic=True)
        action_np = action.squeeze(0).numpy()
        obs, reward, terminated, truncated, info = env.step(action_np)
        obs = np.concatenate(obs) if isinstance(obs, list) else np.asarray(obs, dtype=np.float32)
        total_reward += reward; total_cost += info.get("cost", 0.0); step_count += 1
        n_dep = info.get("ev_departure_departures", 0)
        if n_dep > 0:
            c1_dep += n_dep; deficit = info.get("ev_avoidable_deficit_kwh", 0.0)
            c1_deficit += deficit
            if deficit > 0.01: c1_viol += 1
        if info.get("battery_soc_violation_any", 0.0) > 0.5: c2_viol += 1
        if info.get("building_power_violation", 0.0) > 0.5: c3_viol += 1
        if info.get("grid_power_violation", 0.0) > 0.5: c4_viol += 1
        if "psf_any_intervention" in info:
            if info["psf_any_intervention"]: psf_interv += 1
            if info.get("psf_infeasible", 0.0) > 0.5: psf_infeas += 1
            psf_times.append(info.get("psf_solve_ms", 0.0))
            psf_deltas.append(info.get("psf_action_delta_l2", 0.0))
        if (step+1) % 2000 == 0:
            print(f"  [{label}] {step+1}/{max_steps} R={total_reward:.0f} C1v={c1_viol} C2v={c2_viol} C3v={c3_viol} C4v={c4_viol} ({time.time()-t0:.0f}s)")
        if terminated or truncated: break
    elapsed = time.time() - t0
    r = {"label": label, "steps": step_count, "reward": total_reward, "cost": total_cost,
         "c1_dep": c1_dep, "c1_viol": c1_viol, "c1_%": 100.0*c1_viol/max(c1_dep,1), "c1_kwh": c1_deficit,
         "c2_viol": c2_viol, "c2_%": 100.0*c2_viol/max(step_count,1),
         "c3_viol": c3_viol, "c3_%": 100.0*c3_viol/max(step_count,1),
         "c4_viol": c4_viol, "c4_%": 100.0*c4_viol/max(step_count,1), "time_s": elapsed}
    if psf_times:
        r.update({"psf_int": psf_interv, "psf_int_%": 100.0*psf_interv/max(step_count,1),
                   "psf_infeas": psf_infeas, "psf_ms": np.mean(psf_times), "psf_delta": np.mean(psf_deltas)})
    return r

np.random.seed(SEED); torch.manual_seed(SEED)
all_results = []

for cfg in CONFIGS:
    print(f"\nLoading: {cfg['name']}")
    actor, normalizer = load_actor(cfg["ckpt"])
    for mode, builder, suffix in [("Free", build_env_free, " | Free"), ("PSF", build_env_psf, " | +PSF")]:
        label = f"{cfg['name']}{suffix}"
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        env = builder()
        res = rollout(env, actor, normalizer, seed=SEED, max_steps=MAX_STEPS, label=label)
        env.close()
        all_results.append(res)
        print(f"  -> R={res['reward']:.0f} C1={res['c1_%']:.1f}% C2={res['c2_%']:.1f}% C3={res['c3_%']:.1f}% C4={res['c4_%']:.1f}% ({res['time_s']:.0f}s)")

print(f"\n\n{'='*110}")
print(f"  FULL COMPARISON TABLE  (seed={SEED}, 1 episode = {MAX_STEPS} steps)")
print(f"{'='*110}")
hdr = f"  {'Config':<45} {'Reward':>10} {'C1%':>7} {'C2%':>7} {'C3%':>7} {'C4%':>7} {'C1kwh':>8} {'PSF%':>7} {'Time':>6}"
print(hdr); print("  " + "-"*(len(hdr)-2))
for r in all_results:
    psf_s = f"{r.get('psf_int_%',0):.1f}" if "psf_int" in r else "  n/a"
    print(f"  {r['label']:<45} {r['reward']:>10.0f} {r['c1_%']:>6.1f}% {r['c2_%']:>6.1f}% {r['c3_%']:>6.1f}% {r['c4_%']:>6.1f}% {r['c1_kwh']:>7.2f} {psf_s:>7} {r['time_s']:>5.0f}s")
print(f"\n  Legend: C1=EV departure deficit, C2=Battery SoC bounds, C3=Building power, C4=Grid power")
print(f"  PSF% = steps where PSF modified the action\n")
