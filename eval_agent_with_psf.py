#!/usr/bin/env python3
"""
Evaluate FOCOPS agent: bare agent vs agent+PSF.
Uses different random seed (99) from training (42).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

# Ensure env vars
for k, v in {
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "2.273834",
    "CITYLEARN_STEMS_P_GRID_MAX": "27.127751",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_zero",
    "CITYLEARN_SCHEMA": "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json",
}.items():
    os.environ.setdefault(k, v)

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper

# ── Actor ──
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

# ── Env factory ──
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
            if nxt is not None and nxt is cur:
                continue
            if nxt is not None:
                cur = nxt
                break
        else:
            break
    return None

# ── Evaluation loop ──
def run_eval(env, actor=None, normalizer=None, mode="agent", label=""):
    np.random.seed(99)  # Different from training seed=42
    torch.manual_seed(99)

    obs, info = env.reset()
    obs_flat = flatten_obs(obs)
    act_space = env.action_space
    if isinstance(act_space, list):
        act_low = np.concatenate([np.asarray(sp.low).ravel() for sp in act_space])
        act_high = np.concatenate([np.asarray(sp.high).ravel() for sp in act_space])
    else:
        act_low, act_high = np.asarray(act_space.low).ravel(), np.asarray(act_space.high).ravel()
    action_dim = len(act_low)

    steps = 0
    c1_total_departures = 0; c1_total_violations = 0; c1_deficit_kwh = 0.0
    c2_checks = 0; c2_violations = 0
    c3_building_violations = 0; c3_total_checks = 0; c3_step_violations = 0
    c4_violations = 0
    total_reward = 0.0
    grid_trace = []
    psf_interventions = 0

    city = find_citylearn(env)
    if city is None:
        print("ERROR: Cannot find CityLearn env!")
        return None
    print(f"  CityLearn found: {len(city.buildings)} buildings")

    t0 = time.time()
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

        # C1
        dep_count = int(float(info.get("ev_departure_departures", 0)))
        viol_count = int(float(info.get("ev_departure_violation_count_deficit", 0)))
        deficit = float(info.get("ev_departure_deficit_kwh", 0.0))
        c1_total_departures += dep_count
        c1_total_violations += viol_count
        c1_deficit_kwh += deficit

        # C2
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
                if soc < -0.001 or soc > 0.951:
                    c2_violations += 1

        # C3
        bpv = int(float(info.get("building_power_violation_count", 0)))
        c3_building_violations += bpv
        c3_total_checks += len(city.buildings)
        if bpv > 0:
            c3_step_violations += 1

        # C4
        gpv = float(info.get("grid_power_violation", 0.0))
        cost_c4 = float(info.get("cost_stems_grid_power", 0.0))
        if gpv > 0 or cost_c4 > 0:
            c4_violations += 1

        # Grid trace
        grid_total = 0.0
        for b in city.buildings:
            nec = getattr(b, 'net_electricity_consumption', None)
            if nec is not None and len(nec) >= 2:
                grid_total += max(0.0, float(nec[-2]))
        grid_trace.append(grid_total)

        # PSF stats
        if float(info.get("psf_any_intervention", 0)) > 0:
            psf_interventions += 1

        if steps % 2000 == 0:
            elapsed = time.time() - t0
            c1r = (c1_total_violations / c1_total_departures * 100) if c1_total_departures > 0 else 0
            c3r = (c3_building_violations / c3_total_checks * 100) if c3_total_checks > 0 else 0
            c4r = (c4_violations / steps * 100)
            print(f"  Step {steps} | C1={c1_total_violations}/{c1_total_departures}({c1r:.0f}%) "
                  f"C2={c2_violations}/{c2_checks} C3={c3r:.1f}% C4={c4r:.1f}% "
                  f"PSF_int={psf_interventions} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    c1r = (c1_total_violations / c1_total_departures * 100) if c1_total_departures > 0 else -1
    c2r = (c2_violations / c2_checks * 100) if c2_checks > 0 else 0
    c3r = (c3_building_violations / c3_total_checks * 100) if c3_total_checks > 0 else 0
    c4r = (c4_violations / steps * 100)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  C1 (EV departure):  {c1_total_violations}/{c1_total_departures} = {c1r:.2f}%  deficit={c1_deficit_kwh:.1f} kWh")
    print(f"  C2 (battery SoC):   {c2_violations}/{c2_checks} = {c2r:.2f}%")
    print(f"  C3 (building pwr):  {c3_building_violations}/{c3_total_checks} = {c3r:.2f}%  (steps: {c3_step_violations}/{steps})")
    print(f"  C4 (grid power):    {c4_violations}/{steps} = {c4r:.2f}%")
    if grid_trace:
        gp = np.array(grid_trace)
        print(f"  Grid stats:         mean={gp.mean():.1f} max={gp.max():.1f} 95th={np.percentile(gp,95):.1f}")
    print(f"  Reward:             {total_reward:.1f}")
    print(f"  PSF interventions:  {psf_interventions}/{steps}")
    print(f"  Time:               {elapsed:.0f}s")
    print(f"\n  SUMMARY: C1={c1r:.1f}% C2={c2r:.1f}% C3={c3r:.1f}% C4={c4r:.1f}% Rew={total_reward:.1f}")

    return {"c1": c1r, "c2": c2r, "c3": c3r, "c4": c4r, "reward": total_reward,
            "c1_deps": c1_total_departures, "c1_viols": c1_total_violations}


def load_actor(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    wkeys = sorted([k for k in ckpt['pi'] if 'weight' in k and 'mean' in k])
    obs_dim = ckpt['pi'][wkeys[0]].shape[1]
    act_dim = ckpt['pi'][wkeys[-1]].shape[0]
    hidden = [ckpt['pi'][k].shape[0] for k in wkeys[:-1]]
    print(f"Actor: obs={obs_dim} hidden={hidden} act={act_dim}")
    actor = GaussianActor(obs_dim, act_dim, hidden)
    actor.load_state_dict(ckpt['pi'])
    actor.eval()
    normalizer = ObsNormalizer(ckpt['obs_normalizer'])
    return actor, normalizer


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--psf-mode", choices=["none", "c1only", "all"], default="all",
                   help="none=bare agent, c1only=PSF fixes C1+C2, all=PSF fixes all")
    p.add_argument("--compare", action="store_true", help="Run all 3: zero, agent, agent+PSF")
    args = p.parse_args()

    actor, normalizer = load_actor(args.checkpoint)
    results = {}

    if args.compare:
        # 1. Zero baseline
        print("\n" + "="*70)
        print("  [1/4] ZERO BASELINE")
        print("="*70)
        env = make_eval_env()
        results["Zero"] = run_eval(env, mode="zero", label="ZERO BASELINE")

        # 2. Agent only
        print("\n" + "="*70)
        print("  [2/4] AGENT ONLY (no PSF)")
        print("="*70)
        env = make_eval_env()
        results["Agent"] = run_eval(env, actor, normalizer, mode="agent", label="FOCOPS AGENT ONLY")

        # 3. Agent + PSF C1 only
        print("\n" + "="*70)
        print("  [3/4] AGENT + PSF (C1+C2 only)")
        print("="*70)
        from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper
        env = make_eval_env()
        psf_env = LookaheadPSFWrapper(env, horizon=24,
            w_track=100.0, w_slack_c1=1000.0, w_slack_c3=0.0, w_slack_c4=0.0,
            freeze_ev=False, verbose=1)
        results["Agent+PSF(C1)"] = run_eval(psf_env, actor, normalizer, mode="agent",
                                             label="AGENT + PSF (C1+C2)")

        # 4. Agent + PSF all constraints
        print("\n" + "="*70)
        print("  [4/4] AGENT + PSF (ALL)")
        print("="*70)
        env = make_eval_env()
        psf_env = LookaheadPSFWrapper(env, horizon=24,
            w_track=100.0, w_slack_c1=1000.0, w_slack_c3=10.0, w_slack_c4=10.0,
            freeze_ev=False, verbose=1)
        results["Agent+PSF(All)"] = run_eval(psf_env, actor, normalizer, mode="agent",
                                              label="AGENT + PSF (ALL)")

        # Comparison table
        print("\n\n" + "="*80)
        print(f"{'Config':<25} {'C1':>8} {'C2':>8} {'C3':>8} {'C4':>8} {'Reward':>10}")
        print("-"*80)
        for name, r in results.items():
            if r:
                print(f"{name:<25} {r['c1']:>7.1f}% {r['c2']:>7.1f}% "
                      f"{r['c3']:>7.1f}% {r['c4']:>7.1f}% {r['reward']:>10.1f}")
        print("-"*80)
        print(f"{'Structural floor':<25} {'N/A':>8} {'0.0':>8} {'7.35':>8} {'~0':>8}")
        print(f"{'Old PPOLag baseline':<25} {'90.2':>8} {'4.1':>8} {'24.5':>8} {'1.9':>8} {'52.6':>10}")
        print("="*80)

    else:
        # Single run
        if args.psf_mode == "none":
            env = make_eval_env()
            run_eval(env, actor, normalizer, mode="agent", label="FOCOPS AGENT ONLY")
        elif args.psf_mode == "c1only":
            from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper
            env = make_eval_env()
            psf_env = LookaheadPSFWrapper(env, horizon=24,
                w_track=100.0, w_slack_c1=1000.0, w_slack_c3=0.0, w_slack_c4=0.0,
                freeze_ev=False, verbose=1)
            run_eval(psf_env, actor, normalizer, mode="agent", label="AGENT + PSF (C1+C2)")
        else:
            from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper
            env = make_eval_env()
            psf_env = LookaheadPSFWrapper(env, horizon=24,
                w_track=100.0, w_slack_c1=1000.0, w_slack_c3=10.0, w_slack_c4=10.0,
                freeze_ev=False, verbose=1)
            run_eval(psf_env, actor, normalizer, mode="agent", label="AGENT + PSF (ALL)")
