#!/usr/bin/env python3
"""Test A: Is C4 blow-up caused by EV enforcement or battery clamping?
3 configs x 1000 steps. NO files modified."""
import sys, os, time
import numpy as np
os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, ".")
for k, v in {"CITYLEARN_STEMS_SOC_LOW": "0.0", "CITYLEARN_STEMS_SOC_HIGH": "0.95", "CITYLEARN_STEMS_P_BUILDING_MAX": "2.273834", "CITYLEARN_STEMS_P_GRID_MAX": "27.127751", "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_zero", "CITYLEARN_SCHEMA": "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json", "CITYLEARN_EV_DENSE_COST_SCALE": "1.0"}.items():
    os.environ[k] = v
from evaluation.core.evaluator import make_env
from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent
from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper
from citylearn_safe.psf.psf_extract import unwrap_to_citylearn
CKPT = "runs/ppolag_P95_V2/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-05-02-08-01/torch_save/epoch-100.pt"
N_STEPS = 1000; P_GRID_MAX = 27.127751

def run_config(label, w_c1, w_c3, w_c4, freeze_ev):
    env = make_env()
    psf = LookaheadPSFWrapper(env, horizon=24, w_track=1.0, w_slack_c1=w_c1, w_slack_c3=w_c3, w_slack_c4=w_c4, freeze_ev=freeze_ev, verbose=0)
    psf.p_building_max = 2.273834; psf.p_grid_max = 27.127751
    agent = OmniSafeCheckpointAgent(CKPT, name=f"ppolag_{label}")
    agent.reset(psf)
    obs, info = psf.reset()
    c1_viol = 0; c1_deps = 0; c4_viol = 0; c4_total = 0
    sum_reward = 0.0; worst_grid = 0.0; batt_delta_sum = 0.0
    t0 = time.time()
    for step_i in range(N_STEPS):
        action = agent.act(obs, info)
        action_before = np.array(action, dtype=float).ravel().copy()
        obs, reward, term, trunc, info = psf.step(action)
        sum_reward += reward
        city = unwrap_to_citylearn(psf.env)
        t_idx = max(0, city.time_step - 1)
        grid = sum(b.net_electricity_consumption[t_idx] for b in city.buildings if t_idx < len(b.net_electricity_consumption))
        c4_total += 1; worst_grid = max(worst_grid, abs(grid))
        if abs(grid) > P_GRID_MAX + 1e-6: c4_viol += 1
        cost_ev = float(info.get("cost_ev_departure", 0.0))
        if cost_ev > 0: c1_viol += 1; c1_deps += 1
        elif "cost_ev_departure" in info: c1_deps += 0
        if (step_i + 1) % 200 == 0:
            print(f"  [{label}] {step_i+1}/{N_STEPS}  C4={c4_viol}/{c4_total}({100*c4_viol/max(1,c4_total):.1f}%)  worst_grid={worst_grid:.1f}kW  {time.time()-t0:.0f}s")
        if term or trunc: obs, info = psf.reset()
    el = time.time() - t0
    r = {"c4_rate": 100*c4_viol/max(1,c4_total), "c4_viol": c4_viol, "c4_total": c4_total, "worst_grid": worst_grid, "reward": sum_reward, "elapsed": el}
    print(f"\n{'='*55}\n  {label}  ({N_STEPS} steps, {el:.0f}s)")
    print(f"  C4: {c4_viol}/{c4_total} = {r['c4_rate']:.2f}%  worst_grid={worst_grid:.1f} kW")
    print(f"  Reward: {sum_reward:.1f}\n{'='*55}\n")
    return r

print("="*65)
print(f"TEST A: C4 SOURCE DIAGNOSIS  |  {N_STEPS} steps per config")
print("="*65)
results = {}

print("\n>>> RUN 1: FULL PSF (C1+C2+C3+C4 enforced, freeze_ev=True)")
results["full"] = run_config("FULL_PSF", 1000.0, 1000.0, 1000.0, True)

print("\n>>> RUN 2: NO EV ENFORCEMENT (C2+C3+C4 only, freeze_ev=False, w_c1=0)")
results["no_ev"] = run_config("NO_EV", 0.0, 1000.0, 1000.0, False)

print("\n>>> RUN 3: EV ONLY (C1 enforced, no battery/grid correction)")
results["ev_only"] = run_config("EV_ONLY", 1000.0, 0.0, 0.0, True)

f, ne, eo = results["full"], results["no_ev"], results["ev_only"]
print("\n" + "="*65); print("COMPARISON TABLE"); print("="*65)
print(f"{'Metric':<25} {'Full PSF':>12} {'No EV enf':>12} {'EV only':>12}"); print("-"*65)
print(f"{'C4 viol rate':<25} {f['c4_rate']:>11.2f}% {ne['c4_rate']:>11.2f}% {eo['c4_rate']:>11.2f}%")
print(f"{'Worst grid (kW)':<25} {f['worst_grid']:>12.1f} {ne['worst_grid']:>12.1f} {eo['worst_grid']:>12.1f}")
print(f"{'Reward':<25} {f['reward']:>12.1f} {ne['reward']:>12.1f} {eo['reward']:>12.1f}")
print(f"{'Time (s)':<25} {f['elapsed']:>12.0f} {ne['elapsed']:>12.0f} {eo['elapsed']:>12.0f}")
print("-"*65)

if ne['c4_rate'] < f['c4_rate'] - 5.0:
    print(f"\n>>> VERDICT: EV enforcement is the C4 culprit.")
    print(f"    Full PSF C4={f['c4_rate']:.1f}% vs No-EV C4={ne['c4_rate']:.1f}% (delta={ne['c4_rate']-f['c4_rate']:+.1f}pp)")
    print(f"    Fix: make EV actions a QP decision variable (unfreeze + optimize charging schedule)")
elif eo['c4_rate'] > 10.0:
    print(f"\n>>> VERDICT: EV-only mode already causes C4={eo['c4_rate']:.1f}%. EV charging load drives grid violations.")
    print(f"    Fix: QP must co-optimize EV timing + battery dispatch jointly")
else:
    print(f"\n>>> VERDICT: C4 is driven by battery C2 clamping, not EV enforcement.")
    print(f"    Full={f['c4_rate']:.1f}% No-EV={ne['c4_rate']:.1f}% EV-only={eo['c4_rate']:.1f}%")
