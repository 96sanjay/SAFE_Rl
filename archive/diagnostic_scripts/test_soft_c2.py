#!/usr/bin/env python3
import sys, os, time
import numpy as np
os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, ".")
for k, v in {"CITYLEARN_STEMS_P_BUILDING_MAX": "2.273834", "CITYLEARN_STEMS_P_GRID_MAX": "27.127751", "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_zero", "CITYLEARN_SCHEMA": "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json", "CITYLEARN_EV_DENSE_COST_SCALE": "1.0"}.items():
    os.environ[k] = v
from evaluation.core.evaluator import make_env
from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent
from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper
from citylearn_safe.psf.psf_extract import unwrap_to_citylearn
CKPT = "runs/ppolag_P95_V2/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-05-02-08-01/torch_save/epoch-100.pt"
N_STEPS = 1000
REAL_SOC_LOW = 0.0; REAL_SOC_HIGH = 0.95; P_GRID_MAX = 27.127751

def run_config(soc_low, soc_high, label):
    os.environ["CITYLEARN_STEMS_SOC_LOW"] = str(soc_low)
    os.environ["CITYLEARN_STEMS_SOC_HIGH"] = str(soc_high)
    env = make_env()
    psf = LookaheadPSFWrapper(env, horizon=24, w_track=1.0, w_slack_c1=1000.0, w_slack_c3=1000.0, w_slack_c4=1000.0, soc_low=soc_low, soc_high=soc_high, freeze_ev=True, verbose=0)
    psf.p_building_max = 2.273834; psf.p_grid_max = 27.127751
    agent = OmniSafeCheckpointAgent(CKPT, name=f"ppolag_{label}")
    agent.reset(psf)
    obs, info = psf.reset()
    c2_viol = 0; c2_total = 0; c4_viol = 0; c4_total = 0
    sum_reward = 0.0; worst_soc_lo = 1.0; worst_grid = 0.0; c2_mag_sum = 0.0
    t0 = time.time()
    for step_i in range(N_STEPS):
        action = agent.act(obs, info)
        obs, reward, term, trunc, info = psf.step(action)
        sum_reward += reward
        city = unwrap_to_citylearn(psf.env)
        t_idx = max(0, city.time_step - 1)
        for b in city.buildings:
            es = getattr(b, "electrical_storage", None)
            if es is None: continue
            soc_arr = np.asarray(getattr(es, "soc", []), dtype=float)
            if 0 <= t_idx < len(soc_arr):
                s = float(soc_arr[t_idx]); c2_total += 1
                worst_soc_lo = min(worst_soc_lo, s)
                if s < REAL_SOC_LOW - 1e-6 or s > REAL_SOC_HIGH + 1e-6:
                    c2_viol += 1; c2_mag_sum += max(REAL_SOC_LOW - s, s - REAL_SOC_HIGH, 0.0)
        grid = sum(b.net_electricity_consumption[t_idx] for b in city.buildings if t_idx < len(b.net_electricity_consumption))
        c4_total += 1; worst_grid = max(worst_grid, abs(grid))
        if abs(grid) > P_GRID_MAX + 1e-6: c4_viol += 1
        if (step_i + 1) % 200 == 0:
            print(f"  [{label}] {step_i+1}/{N_STEPS}  C2={c2_viol}/{c2_total}({100*c2_viol/max(1,c2_total):.1f}%)  C4={c4_viol}/{c4_total}({100*c4_viol/max(1,c4_total):.1f}%)  worst_soc={worst_soc_lo:.4f}  {time.time()-t0:.0f}s")
        if term or trunc: obs, info = psf.reset()
    el = time.time() - t0
    r = {"c2_rate": 100*c2_viol/max(1,c2_total), "c4_rate": 100*c4_viol/max(1,c4_total), "c2_viol": c2_viol, "c2_total": c2_total, "c4_viol": c4_viol, "c4_total": c4_total, "reward": sum_reward, "worst_soc": worst_soc_lo, "worst_grid": worst_grid, "c2_avg_mag": c2_mag_sum/max(1,c2_viol), "elapsed": el}
    print(f"\n{'='*55}\n  {label}  ({N_STEPS} steps, {el:.0f}s)")
    print(f"  C2: {c2_viol}/{c2_total} = {r['c2_rate']:.2f}%  avg_mag={r['c2_avg_mag']:.4f}  worst_soc={worst_soc_lo:.4f}")
    print(f"  C4: {c4_viol}/{c4_total} = {r['c4_rate']:.2f}%  worst_grid={worst_grid:.1f} kW")
    print(f"  Reward: {sum_reward:.1f}\n{'='*55}\n")
    return r

print("="*60); print(f"SOFT-C2 HYPOTHESIS TEST  |  {N_STEPS} steps per config"); print("="*60)
results = {}
print("\n>>> RUN 1: HARD C2 (soc bounds = [0.0, 0.95])")
results["hard"] = run_config(0.0, 0.95, "HARD_C2")
print("\n>>> RUN 2: SOFT C2 (soc bounds = [-0.05, 1.0])")
results["soft"] = run_config(-0.05, 1.0, "SOFT_C2")
h, s = results["hard"], results["soft"]
print("\n" + "="*65); print("COMPARISON TABLE"); print("="*65)
print(f"{'Metric':<25} {'Hard C2':>14} {'Soft C2':>14} {'Delta':>14}"); print("-"*65)
print(f"{'C2 viol rate':<25} {h['c2_rate']:>13.2f}% {s['c2_rate']:>13.2f}% {s['c2_rate']-h['c2_rate']:>+13.2f}%")
print(f"{'C2 avg magnitude':<25} {h['c2_avg_mag']:>14.4f} {s['c2_avg_mag']:>14.4f} {s['c2_avg_mag']-h['c2_avg_mag']:>+14.4f}")
print(f"{'C4 viol rate':<25} {h['c4_rate']:>13.2f}% {s['c4_rate']:>13.2f}% {s['c4_rate']-h['c4_rate']:>+13.2f}%")
print(f"{'Worst SoC':<25} {h['worst_soc']:>14.4f} {s['worst_soc']:>14.4f} {s['worst_soc']-h['worst_soc']:>+14.4f}")
print(f"{'Worst grid (kW)':<25} {h['worst_grid']:>14.1f} {s['worst_grid']:>14.1f} {s['worst_grid']-h['worst_grid']:>+14.1f}")
print(f"{'Reward':<25} {h['reward']:>14.1f} {s['reward']:>14.1f} {s['reward']-h['reward']:>+14.1f}")
print(f"{'Time (s)':<25} {h['elapsed']:>14.0f} {s['elapsed']:>14.0f}"); print("-"*65)
dc4 = s['c4_rate'] - h['c4_rate']
if dc4 < -2.0: print(f"\n>>> VERDICT: SOFT C2 REDUCES C4 by {abs(dc4):.1f}pp. Worth implementing proper slack.")
elif dc4 < 0: print(f"\n>>> VERDICT: SOFT C2 helps C4 slightly ({abs(dc4):.1f}pp). Marginal benefit.")
else: print(f"\n>>> VERDICT: SOFT C2 does NOT help C4. Coupling is deeper than SoC bounds.")
