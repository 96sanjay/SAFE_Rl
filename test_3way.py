#!/usr/bin/env python3
"""3-way comparison: PPOLag vs IntelligentRBC(greedy) vs PPOLag+PSF(full). 1000 steps."""
import sys, os, time
import numpy as np
os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, ".")
for k, v in {"CITYLEARN_STEMS_SOC_LOW": "0.0", "CITYLEARN_STEMS_SOC_HIGH": "0.95", "CITYLEARN_STEMS_P_BUILDING_MAX": "2.273834", "CITYLEARN_STEMS_P_GRID_MAX": "27.127751", "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_zero", "CITYLEARN_SCHEMA": "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json", "CITYLEARN_EV_DENSE_COST_SCALE": "1.0"}.items():
    os.environ[k] = v
from evaluation.core.evaluator import make_env
from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent
from evaluation.agents.rbc import RBCAgent
from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper
from citylearn_safe.psf.psf_extract import unwrap_to_citylearn
CKPT = "runs/ppolag_P95_V2/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-05-02-08-01/torch_save/epoch-100.pt"
N_STEPS = 1000; P_GRID_MAX = 27.127751; P_BLD_MAX = 2.273834
REAL_SOC_LOW = 0.0; REAL_SOC_HIGH = 0.95

def run_eval(env, agent, label):
    agent.reset(env)
    obs, info = env.reset()
    c1_viol = 0; c2_viol = 0; c2_total = 0; c3_viol = 0; c3_total = 0
    c4_viol = 0; c4_total = 0; sum_reward = 0.0; worst_grid = 0.0; worst_soc = 1.0
    t0 = time.time()
    for step_i in range(N_STEPS):
        action = agent.act(obs, info)
        obs, reward, term, trunc, info = env.step(action)
        sum_reward += reward
        city = unwrap_to_citylearn(env)
        t_idx = max(0, city.time_step - 1)
        c1_viol += 1 if float(info.get("cost_ev_departure", 0.0)) > 0 else 0
        for b in city.buildings:
            es = getattr(b, "electrical_storage", None)
            if es is not None:
                soc_arr = np.asarray(getattr(es, "soc", []), dtype=float)
                if 0 <= t_idx < len(soc_arr):
                    s = float(soc_arr[t_idx]); c2_total += 1
                    worst_soc = min(worst_soc, s)
                    if s < REAL_SOC_LOW - 1e-6 or s > REAL_SOC_HIGH + 1e-6: c2_viol += 1
            nec = getattr(b, "net_electricity_consumption", None)
            if nec is not None and t_idx < len(nec):
                c3_total += 1
                if abs(float(nec[t_idx])) > P_BLD_MAX + 1e-6: c3_viol += 1
        grid = sum(b.net_electricity_consumption[t_idx] for b in city.buildings if t_idx < len(b.net_electricity_consumption))
        c4_total += 1; worst_grid = max(worst_grid, abs(grid))
        if abs(grid) > P_GRID_MAX + 1e-6: c4_viol += 1
        if (step_i + 1) % 500 == 0:
            print(f"  [{label}] {step_i+1}/{N_STEPS}  C1={c1_viol} C2={100*c2_viol/max(1,c2_total):.1f}% C3={100*c3_viol/max(1,c3_total):.1f}% C4={100*c4_viol/max(1,c4_total):.1f}%  {time.time()-t0:.0f}s")
        if term or trunc: obs, info = env.reset()
    el = time.time() - t0
    r = {"c1": c1_viol, "c2": 100*c2_viol/max(1,c2_total), "c3": 100*c3_viol/max(1,c3_total), "c4": 100*c4_viol/max(1,c4_total), "c4_viol": c4_viol, "worst_grid": worst_grid, "worst_soc": worst_soc, "reward": sum_reward, "elapsed": el}
    print(f"\n{'='*55}\n  {label}  ({N_STEPS} steps, {el:.0f}s)")
    print(f"  C1: {c1_viol} violations")
    print(f"  C2: {100*c2_viol/max(1,c2_total):.2f}%  worst_soc={worst_soc:.4f}")
    print(f"  C3: {100*c3_viol/max(1,c3_total):.2f}%")
    print(f"  C4: {100*c4_viol/max(1,c4_total):.2f}%  worst_grid={worst_grid:.1f} kW")
    print(f"  Reward: {sum_reward:.1f}\n{'='*55}\n")
    return r

print("="*65)
print(f"3-WAY COMPARISON  |  {N_STEPS} steps each")
print("="*65)
results = {}

print("\n>>> RUN 1: PPOLag (no PSF)")
env1 = make_env()
agent1 = OmniSafeCheckpointAgent(CKPT, name="PPOLag")
results["ppolag"] = run_eval(env1, agent1, "PPOLag")

print("\n>>> RUN 2: IntelligentRBC (greedy EV)")
env2 = make_env()
agent2 = RBCAgent(ev_mode="greedy")
results["rbc"] = run_eval(env2, agent2, "RBC_greedy")

print("\n>>> RUN 3: PPOLag + Full PSF")
env3 = make_env()
psf = LookaheadPSFWrapper(env3, horizon=24, w_track=1.0, w_slack_c1=1000.0, w_slack_c3=1000.0, w_slack_c4=1000.0, freeze_ev=True, verbose=0)
psf.p_building_max = 2.273834; psf.p_grid_max = 27.127751
agent3 = OmniSafeCheckpointAgent(CKPT, name="PPOLag_PSF")
results["psf"] = run_eval(psf, agent3, "PPOLag+PSF")

p, r, s = results["ppolag"], results["rbc"], results["psf"]
print("\n" + "="*70); print("3-WAY COMPARISON TABLE"); print("="*70)
print(f"{'Metric':<25} {'PPOLag':>14} {'RBC(greedy)':>14} {'PPOLag+PSF':>14}"); print("-"*70)
print(f"{'C1 violations':<25} {p['c1']:>14} {r['c1']:>14} {s['c1']:>14}")
print(f"{'C2 viol rate':<25} {p['c2']:>13.2f}% {r['c2']:>13.2f}% {s['c2']:>13.2f}%")
print(f"{'C3 viol rate':<25} {p['c3']:>13.2f}% {r['c3']:>13.2f}% {s['c3']:>13.2f}%")
print(f"{'C4 viol rate':<25} {p['c4']:>13.2f}% {r['c4']:>13.2f}% {s['c4']:>13.2f}%")
print(f"{'Worst grid (kW)':<25} {p['worst_grid']:>14.1f} {r['worst_grid']:>14.1f} {s['worst_grid']:>14.1f}")
print(f"{'Worst SoC':<25} {p['worst_soc']:>14.4f} {r['worst_soc']:>14.4f} {s['worst_soc']:>14.4f}")
print(f"{'Reward':<25} {p['reward']:>14.1f} {r['reward']:>14.1f} {s['reward']:>14.1f}")
print(f"{'Time (s)':<25} {p['elapsed']:>14.0f} {r['elapsed']:>14.0f} {s['elapsed']:>14.0f}")
print("-"*70)
print(f"\nBest C1: {'PPOLag+PSF' if s['c1']<=min(p['c1'],r['c1']) else 'PPOLag' if p['c1']<=r['c1'] else 'RBC'}")
print(f"Best C4: {'PPOLag+PSF' if s['c4']<=min(p['c4'],r['c4']) else 'PPOLag' if p['c4']<=r['c4'] else 'RBC'}")
print(f"Best Reward: {'PPOLag+PSF' if s['reward']>=max(p['reward'],r['reward']) else 'PPOLag' if p['reward']>=r['reward'] else 'RBC'}")
