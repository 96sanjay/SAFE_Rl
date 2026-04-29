#!/usr/bin/env python3
"""Quick 1000-step eval: 4 configs. Per-building violation rates."""
import os, time
import numpy as np
import torch
import torch.nn as nn

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper

SEED = 7
MAX_STEPS = 1000
NB = 17

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
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
    def forward(self, obs): return torch.tanh(self.mean(obs))
    def predict(self, obs, deterministic=True): return self.forward(obs), None

class ObsNormalizer:
    def __init__(self, sd):
        self.mean = sd["_mean"]; self.std = sd["_std"]
        self.clip = sd.get("_clip", torch.full_like(self.mean, 10.0))
    def normalize(self, obs):
        return torch.clamp((obs - self.mean) / (self.std + 1e-8), -self.clip, self.clip)

def infer_hidden(pi):
    s = []; i = 0
    while f"mean.{i}.weight" in pi: s.append(pi[f"mean.{i}.weight"].shape[0]); i += 2
    return tuple(s[:-1])

def load_actor(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    pi = data["pi"]; h = infer_hidden(pi)
    od = pi["mean.0.weight"].shape[1]
    ad = pi[f"mean.{max(int(k.split('.')[1]) for k in pi if k.startswith('mean.') and 'weight' in k)}.weight"].shape[0]
    print(f"  Arch: {od} -> {' -> '.join(map(str,h))} -> {ad}")
    actor = GaussianActor(od, ad, h); actor.load_state_dict(pi); actor.eval()
    return actor, ObsNormalizer(data["obs_normalizer"])

def build_free():
    base = make_base_env()
    env = CityLearnSafetyEnvV3(base, soc_min=0.0, soc_max=0.95)
    return ForecastObsWrapper(env)

def build_psf():
    base = make_base_env()
    env = CityLearnSafetyEnvV3(base, soc_min=0.0, soc_max=0.95)
    env = ForecastObsWrapper(env)
    return LookaheadPSFWrapper(env, horizon=24, verbose=0)

def rollout(env, actor, norm, seed, max_steps, label=""):
    obs, info = env.reset(seed=seed)
    obs = np.concatenate(obs) if isinstance(obs, list) else np.asarray(obs, dtype=np.float32)
    total_reward = 0.0; steps = 0
    c1_dep = 0; c1_viol = 0; c1_kwh = 0.0
    c2_bldg_sum = 0; c3_bldg_sum = 0; c4_steps = 0
    psf_int = 0; psf_inf = 0
    t0 = time.time()
    for step in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_t = norm.normalize(obs_t)
        with torch.no_grad(): action, _ = actor.predict(obs_t, deterministic=True)
        action_np = action.squeeze(0).numpy()
        obs, reward, term, trunc, info = env.step(action_np)
        obs = np.concatenate(obs) if isinstance(obs, list) else np.asarray(obs, dtype=np.float32)
        total_reward += reward; steps += 1
        nd = info.get("ev_departure_departures", 0)
        if nd > 0:
            c1_dep += nd; d = info.get("ev_avoidable_deficit_kwh", 0.0); c1_kwh += d
            if d > 0.01: c1_viol += 1
        c2_bldg_sum += int(round(info.get("battery_soc_violation_frac", 0.0) * NB))
        c3_bldg_sum += int(round(info.get("building_power_violation_frac", 0.0) * NB))
        if info.get("grid_power_violation", 0.0) > 0.5: c4_steps += 1
        if "psf_any_intervention" in info:
            if info["psf_any_intervention"]: psf_int += 1
            if info.get("psf_infeasible", 0.0) > 0.5: psf_inf += 1
        if term or trunc: break
    elapsed = time.time() - t0
    tb = steps * NB
    return {
        "label": label, "steps": steps, "reward": total_reward,
        "c1_dep": c1_dep, "c1_viol": c1_viol,
        "c1_%": 100.0*c1_viol/max(c1_dep,1), "c1_kwh": c1_kwh,
        "c2_%": 100.0*c2_bldg_sum/max(tb,1),
        "c3_%": 100.0*c3_bldg_sum/max(tb,1),
        "c4_%": 100.0*c4_steps/max(steps,1),
        "psf_int": psf_int, "psf_inf": psf_inf,
        "time_s": elapsed,
    }

CONFIGS = [
    ("Plain FOCOPS",
     "runs/focops_v2_100ep/FOCOPS-{CityLearnSafety-V2G-v2}/seed-000-2026-02-17-09-24-21/torch_save/epoch-100.pt"),
    ("Shielded FOCOPS v6",
     "runs/focops_v2_shield_c1c2_50ep_v6/FOCOPS-{CityLearnSafety-V2G-v2-shield}/seed-000-2026-02-18-16-11-48/torch_save/epoch-50.pt"),
]

for name, ckpt in CONFIGS:
    assert os.path.isfile(ckpt), f"Missing: {ckpt}"

np.random.seed(SEED); torch.manual_seed(SEED)
all_results = []

for name, ckpt in CONFIGS:
    print(f"\n{'#'*60}")
    print(f"  Loading: {name}")
    print(f"  {ckpt}")
    print(f"{'#'*60}")
    actor, norm = load_actor(ckpt)
    for mode_name, builder in [("Free", build_free), ("+PSF", build_psf)]:
        label = f"{name} | {mode_name}"
        print(f"\n  --- {label} ---")
        env = builder()
        r = rollout(env, actor, norm, seed=SEED, max_steps=MAX_STEPS, label=label)
        env.close()
        all_results.append(r)
        print(f"  R={r['reward']:.0f} C1={r['c1_%']:.1f}% C2={r['c2_%']:.1f}% C3={r['c3_%']:.1f}% C4={r['c4_%']:.1f}% PSFint={r['psf_int']} PSFinf={r['psf_inf']} ({r['time_s']:.0f}s)")

print(f"\n\n{'='*120}")
print(f"  QUICK EVAL RESULTS  (seed={SEED}, {MAX_STEPS} steps, per-building-step rates)")
print(f"{'='*120}")
hdr = f"  {'Config':<42} {'Reward':>8} {'C1%':>7} {'C2%':>7} {'C3%':>7} {'C4%':>7} {'C1dep':>6} {'C1kwh':>7} {'PSFi':>5} {'PSFx':>5} {'Time':>5}"
print(hdr); print("  " + "-"*(len(hdr)-2))
for r in all_results:
    print(f"  {r['label']:<42} {r['reward']:>8.0f} {r['c1_%']:>6.1f}% {r['c2_%']:>6.1f}% {r['c3_%']:>6.1f}% {r['c4_%']:>6.1f}% {r['c1_dep']:>6} {r['c1_kwh']:>6.1f} {r['psf_int']:>5} {r['psf_inf']:>5} {r['time_s']:>4.0f}s")
print(f"\n  C1% = violating departures / total departures")
print(f"  C2%, C3% = per-building-step (denom = steps × 17)")
print(f"  C4% = per-step")
print(f"  PSFi = PSF interventions, PSFx = PSF infeasible")
print(f"\n  Old eval ref (Gen1, seed=42): C1=59.3% C2=11.8% C3=20.8% C4=8.8%\n")
