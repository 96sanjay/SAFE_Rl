#!/usr/bin/env python
"""Diagnostic: trace Sauté budget dynamics with trained R12a agent."""
import numpy as np, torch, sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

ckpt_path = 'runs/r12a_saute_c1/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-10-20-35-24/torch_save/epoch-50.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

from citylearn_safe.omni_env_v2 import CityLearnCMDPv2
env = CityLearnCMDPv2('CityLearnSafety-V2G-v2')
obs, info = env.reset()
act_dim = env.action_space.shape[-1]

pi_state = ckpt['pi']
norm_state = ckpt['obs_normalizer']
ckpt_obs_dim = pi_state['mean.0.weight'].shape[1]

import torch.nn as nn
net = nn.Sequential(nn.Linear(ckpt_obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, act_dim))
for i, m in enumerate(net):
    if isinstance(m, nn.Linear):
        m.weight.data = pi_state[f'mean.{i}.weight']
        m.bias.data = pi_state[f'mean.{i}.bias']
net.eval()
obs_mean = norm_state['_mean'].numpy()
obs_std = np.clip(norm_state['_std'].numpy(), 1e-6, None)

safe_rewards = []; unsafe_rewards = []; budget_trace = []; raw_costs = []
departures = 0; deficit_viols = 0; budget_depleted_step = None
ev_actions_when_safe = []; ev_actions_when_unsafe = []

obs_np = obs.numpy().ravel()
for t in range(8760):
    obs_n = np.clip((obs_np - obs_mean) / obs_std, -10, 10)
    with torch.no_grad():
        act = net(torch.tensor(obs_n, dtype=torch.float32)).numpy()
    act = np.clip(act, -1, 1)
    obs, reward, cost, term, trunc, info = env.step(torch.tensor(act))
    obs_np = obs.numpy().ravel()
    r = float(reward)
    unsafe = float(info.get('ev_saute_unsafe', 0)) > 0.5
    budget = float(info.get('ev_saute_budget', 1.0))
    raw_cost = float(info.get('ev_saute_raw_cost', 0.0))
    budget_trace.append(budget)
    raw_costs.append(raw_cost)
    departures += int(info.get('ev_departure_departures', 0))
    deficit_viols += int(info.get('ev_departure_violation_count_deficit', 0))
    # Track EV actions (indices 5,6,7 for 5-building schema with 3 EVs)
    ev_act_vals = [float(act[i]) for i in [5, 6, 7] if i < len(act)]
    if unsafe:
        unsafe_rewards.append(r)
        ev_actions_when_unsafe.extend(ev_act_vals)
        if budget_depleted_step is None: budget_depleted_step = t
    else:
        safe_rewards.append(r)
        ev_actions_when_safe.extend(ev_act_vals)
    if term or trunc: break

total = len(safe_rewards) + len(unsafe_rewards)
rc = np.array(raw_costs)

print("=" * 70)
print("  SAUTÉ MDP DIAGNOSTIC — R12a Trained Agent (epoch-50)")
print("=" * 70)
print(f"\n--- Budget Dynamics ---")
print(f"  Budget d = {float(os.environ.get('CITYLEARN_EV_SAUTE_BUDGET', 1500))}")
print(f"  Penalty  = {float(os.environ.get('CITYLEARN_EV_SAUTE_PENALTY', 5.0))}")
print(f"  Budget depleted at step: {budget_depleted_step} of {total}")
print(f"  Final budget: {budget_trace[-1]:.4f}")
print(f"  Safe steps: {len(safe_rewards)}, Unsafe steps: {len(unsafe_rewards)}")
print(f"  Pct unsafe: {100*len(unsafe_rewards)/total:.1f}%")

print(f"\n--- Raw EV Dense Cost (before Sauté zeros it) ---")
print(f"  Non-zero steps: {(rc>0).sum()}/{len(rc)} = {100*(rc>0).mean():.1f}%")
print(f"  Mean (all steps): {rc.mean():.4f}")
if (rc > 0).any():
    print(f"  Mean (when >0):   {rc[rc>0].mean():.4f}")
print(f"  Episode sum:      {rc.sum():.1f}")

print(f"\n--- Reward Comparison (THE KEY QUESTION) ---")
if safe_rewards:
    print(f"  When SAFE (budget > 0):")
    print(f"    Actual reward mean: {np.mean(safe_rewards):.4f}")
    print(f"    Reward std:         {np.std(safe_rewards):.4f}")
    print(f"    Reward range:       [{np.min(safe_rewards):.4f}, {np.max(safe_rewards):.4f}]")
if unsafe_rewards:
    print(f"  When UNSAFE (budget <= 0):")
    print(f"    Actual reward mean: {np.mean(unsafe_rewards):.4f}")
    print(f"    All exactly -5.0?   {all(abs(r-(-5.0))<0.01 for r in unsafe_rewards)}")
    n_not_5 = sum(1 for r in unsafe_rewards if abs(r - (-5.0)) > 0.01)
    if n_not_5 > 0:
        print(f"    WARNING: {n_not_5} unsafe steps NOT -5.0!")
        not5 = [r for r in unsafe_rewards if abs(r - (-5.0)) > 0.01]
        print(f"    Examples: {not5[:5]}")

if safe_rewards and unsafe_rewards:
    safe_mean = np.mean(safe_rewards)
    unsafe_mean = np.mean(unsafe_rewards)
    gap = safe_mean - unsafe_mean
    print(f"\n  >>> SIGNAL GAP: safe_mean - unsafe_mean = {gap:+.4f} <<<")
    print(f"      Safe mean:   {safe_mean:.4f}")
    print(f"      Unsafe mean: {unsafe_mean:.4f}")
    if gap < 0:
        print(f"  *** CRITICAL: Safe reward is WORSE than penalty! ***")
        print(f"  *** Agent is INCENTIVIZED to deplete budget (get -5.0 instead of {safe_mean:.4f}) ***")
    elif gap < 1.0:
        print(f"  *** WARNING: Gap < 1.0 — signal is very weak ***")
    else:
        print(f"  Gap looks adequate ({gap:.2f})")

print(f"\n--- EV Action Behavior ---")
if ev_actions_when_safe:
    print(f"  Mean EV action when SAFE:   {np.mean(ev_actions_when_safe):+.4f}")
if ev_actions_when_unsafe:
    print(f"  Mean EV action when UNSAFE: {np.mean(ev_actions_when_unsafe):+.4f}")
    print(f"  (Should be higher when unsafe if agent learned from Sauté)")

print(f"\n--- C0 Departure Result ---")
print(f"  Total departures: {departures}")
print(f"  Deficit violations: {deficit_viols}")
if departures > 0:
    print(f"  C0 violation rate: {100*deficit_viols/departures:.1f}%")
print("=" * 70)
