#!/usr/bin/env python3
"""Evaluate FOCOPS checkpoint — v3 fixed."""
import os, sys
import numpy as np
import torch
import torch.nn as nn

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=[512, 256]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.mean = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    def forward(self, obs):
        return self.mean(obs)

class ObsNormalizer:
    def __init__(self, sd):
        self._mean = sd['_mean'].float().numpy()
        self._std = np.clip(sd['_std'].float().numpy(), 1e-6, None)
        self._clip = float(sd.get('_clip', torch.tensor(5.0)).float().mean())
    def normalize(self, obs):
        return np.clip((obs - self._mean) / self._std, -self._clip, self._clip)

def make_eval_env():
    base = make_base_env()
    safety = CityLearnSafetyEnvV3(base, soc_min=0.0, soc_max=0.95)
    return ForecastObsWrapper(safety)

def flatten_obs(obs):
    if isinstance(obs, (list, tuple)):
        return np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs])
    return np.asarray(obs, dtype=np.float32).ravel()

def find_citylearn(env):
    cur = env
    for _ in range(20):
        if hasattr(cur, 'buildings') and hasattr(cur, 'time_step'):
            return cur
        for attr in ['env', '_env', 'base', 'unwrapped', 'raw_env']:
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None

def run_eval(env, actor=None, normalizer=None, mode="agent"):
    obs, info = env.reset()
    obs_flat = flatten_obs(obs)
    act_space = env.action_space
    if isinstance(act_space, list):
        act_low = np.concatenate([np.asarray(sp.low).ravel() for sp in act_space])
        act_high = np.concatenate([np.asarray(sp.high).ravel() for sp in act_space])
    else:
        act_low, act_high = np.asarray(act_space.low).ravel(), np.asarray(act_space.high).ravel()
    action_dim = len(act_low)

    P_BMAX, P_GMAX = 2.273834, 27.127751
    SOC_LOW, SOC_HIGH = 0.0, 0.95

    steps = 0
    # C1
    c1_total_departures = 0
    c1_total_violations = 0
    c1_deficit_kwh = 0.0
    # C2
    c2_checks = 0; c2_violations = 0
    # C3
    c3_step_violations = 0; c3_building_violations = 0; c3_total_buildings_checked = 0
    # C4
    c4_violations = 0
    total_reward = 0.0
    grid_trace = []

    city = find_citylearn(env)
    if city is None:
        print("ERROR: Cannot find CityLearn env!")
        return
    print(f"Found CityLearn: {len(city.buildings)} buildings")

    terminated = False
    while not terminated:
        if mode == "zero":
            action = np.zeros(action_dim)
        elif mode == "random":
            action = np.random.uniform(act_low, act_high)
        else:
            obs_norm = normalizer.normalize(obs_flat)
            obs_t = torch.as_tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor(obs_t).squeeze(0).numpy()
            action = np.clip(action, act_low, act_high)

        obs, reward, terminated, truncated, info = env.step(action)
        obs_flat = flatten_obs(obs)
        steps += 1
        total_reward += float(reward) if np.isscalar(reward) else float(np.sum(reward))

        # ── C1: EV departures ──
        dep_count = float(info.get("ev_departure_departures", 0.0))
        viol_count = float(info.get("ev_departure_violation_count_deficit", 0.0))
        deficit = float(info.get("ev_departure_deficit_kwh", 0.0))
        c1_total_departures += int(dep_count)
        c1_total_violations += int(viol_count)
        c1_deficit_kwh += deficit

        # ── C2: Battery SoC ──
        t_idx = max(0, int(city.time_step) - 1)
        for b in city.buildings:
            es = getattr(b, 'electrical_storage', None)
            if es is None: continue
            soc_arr = getattr(es, 'soc', None)
            if soc_arr is None: continue
            soc_np = np.asarray(soc_arr, dtype=float).ravel()
            if t_idx < len(soc_np):
                soc = float(soc_np[t_idx])
                c2_checks += 1
                if soc < SOC_LOW - 0.001 or soc > SOC_HIGH + 0.001:
                    c2_violations += 1

        # ── C3: Building power (from info) ──
        bpv_count = float(info.get("building_power_violation_count", 0.0))
        c3_building_violations += int(bpv_count)
        c3_total_buildings_checked += len(city.buildings)
        if bpv_count > 0:
            c3_step_violations += 1

        # ── C4: Grid power (from info) ──
        gpv = float(info.get("grid_power_violation", 0.0))
        if gpv > 0:
            c4_violations += 1

        # Grid trace
        grid_total = 0.0
        for b in city.buildings:
            nec = getattr(b, 'net_electricity_consumption', None)
            if nec is not None and len(nec) >= 2:
                grid_total += max(0.0, float(nec[-2]))
        grid_trace.append(grid_total)

        if steps % 2000 == 0:
            c1r = (c1_total_violations / c1_total_departures * 100) if c1_total_departures > 0 else 0
            c3r = (c3_building_violations / c3_total_buildings_checked * 100) if c3_total_buildings_checked > 0 else 0
            c4r = (c4_violations / steps * 100)
            print(f"  Step {steps} | C1={c1_total_violations}/{c1_total_departures}({c1r:.0f}%) C2={c2_violations}/{c2_checks} C3_bldg={c3_building_violations}/{c3_total_buildings_checked}({c3r:.1f}%) C4={c4_violations}/{steps}({c4r:.1f}%)")

    # Final results
    c1r = (c1_total_violations / c1_total_departures * 100) if c1_total_departures > 0 else -1
    c2r = (c2_violations / c2_checks * 100) if c2_checks > 0 else 0
    c3r = (c3_building_violations / c3_total_buildings_checked * 100) if c3_total_buildings_checked > 0 else 0
    c4r = (c4_violations / steps * 100)

    print(f"\n{'='*70}")
    print(f"RESULTS — {mode.upper()} — {steps} steps")
    print(f"{'='*70}")
    print(f"\n  C1 (EV departure SoC):")
    print(f"    Total departures:  {c1_total_departures}")
    print(f"    Violations:        {c1_total_violations}")
    print(f"    Violation rate:    {c1r:.2f}%")
    print(f"    Total deficit:     {c1_deficit_kwh:.2f} kWh")

    print(f"\n  C2 (battery SoC [{SOC_LOW}, {SOC_HIGH}]):")
    print(f"    Checks:     {c2_checks}")
    print(f"    Violations: {c2_violations}")
    print(f"    Rate:       {c2r:.2f}%")

    print(f"\n  C3 (building power <= {P_BMAX} kW):")
    print(f"    Building-step violations: {c3_building_violations}/{c3_total_buildings_checked} ({c3r:.2f}%)")
    print(f"    Steps with any violation: {c3_step_violations}/{steps} ({c3_step_violations/steps*100:.2f}%)")

    print(f"\n  C4 (grid power <= {P_GMAX} kW):")
    print(f"    Violations: {c4_violations}/{steps} ({c4r:.2f}%)")

    if grid_trace:
        gp = np.array(grid_trace)
        print(f"    Grid — mean:{gp.mean():.1f} max:{gp.max():.1f} 95th:{np.percentile(gp,95):.1f}")

    print(f"\n  Reward: {total_reward:.1f}")
    print(f"\n  SUMMARY: C1={c1r:.1f}% C2={c2r:.1f}% C3={c3r:.1f}% C4={c4r:.1f}% Rew={total_reward:.1f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--zero", action="store_true")
    p.add_argument("--random", action="store_true")
    args = p.parse_args()

    env = make_eval_env()

    if args.zero:
        run_eval(env, mode="zero")
    elif args.random:
        run_eval(env, mode="random")
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        wkeys = sorted([k for k in ckpt['pi'] if 'weight' in k and 'mean' in k])
        obs_dim = ckpt['pi'][wkeys[0]].shape[1]
        act_dim = ckpt['pi'][wkeys[-1]].shape[0]
        hidden = [ckpt['pi'][k].shape[0] for k in wkeys[:-1]]
        print(f"Actor: obs={obs_dim} hidden={hidden} act={act_dim}")
        actor = GaussianActor(obs_dim, act_dim, hidden)
        actor.load_state_dict(ckpt['pi'])
        actor.eval()
        normalizer = ObsNormalizer(ckpt['obs_normalizer'])
        run_eval(env, actor, normalizer, mode="agent")
    else:
        print("=== ZERO BASELINE ===")
        run_eval(env, mode="zero")
        print("\n\n=== RANDOM BASELINE ===")
        env2 = make_eval_env()
        run_eval(env2, mode="random")
