#!/usr/bin/env python3
"""Raw agent, no PSF. 1000 steps. How bad is C4 without any filter?"""
import sys, os, time
import numpy as np
os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, ".")
for k, v in {"CITYLEARN_STEMS_SOC_LOW": "0.0", "CITYLEARN_STEMS_SOC_HIGH": "0.95", "CITYLEARN_STEMS_P_BUILDING_MAX": "2.273834", "CITYLEARN_STEMS_P_GRID_MAX": "27.127751", "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_zero", "CITYLEARN_SCHEMA": "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json", "CITYLEARN_EV_DENSE_COST_SCALE": "1.0"}.items():
    os.environ[k] = v
from evaluation.core.evaluator import make_env
from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent
from citylearn_safe.psf.psf_extract import unwrap_to_citylearn
CKPT = "runs/ppolag_P95_V2/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-05-02-08-01/torch_save/epoch-100.pt"
N_STEPS = 1000; P_GRID_MAX = 27.127751; REAL_SOC_LOW = 0.0; REAL_SOC_HIGH = 0.95
env = make_env()
agent = OmniSafeCheckpointAgent(CKPT, name="ppolag_RAW")
agent.reset(env)
obs, info = env.reset()
c2_viol = 0; c2_total = 0; c4_viol = 0; c4_total = 0
sum_reward = 0.0; worst_grid = 0.0; worst_soc = 1.0
t0 = time.time()
for step_i in range(N_STEPS):
    action = agent.act(obs, info)
    obs, reward, term, trunc, info = env.step(action)
    sum_reward += reward
    city = unwrap_to_citylearn(env)
    t_idx = max(0, city.time_step - 1)
    for b in city.buildings:
        es = getattr(b, "electrical_storage", None)
        if es is None: continue
        soc_arr = np.asarray(getattr(es, "soc", []), dtype=float)
        if 0 <= t_idx < len(soc_arr):
            s = float(soc_arr[t_idx]); c2_total += 1
            worst_soc = min(worst_soc, s)
            if s < REAL_SOC_LOW - 1e-6 or s > REAL_SOC_HIGH + 1e-6: c2_viol += 1
    grid = sum(b.net_electricity_consumption[t_idx] for b in city.buildings if t_idx < len(b.net_electricity_consumption))
    c4_total += 1; worst_grid = max(worst_grid, abs(grid))
    if abs(grid) > P_GRID_MAX + 1e-6: c4_viol += 1
    if (step_i + 1) % 200 == 0:
        print(f"  [RAW] {step_i+1}/{N_STEPS}  C2={100*c2_viol/max(1,c2_total):.1f}%  C4={c4_viol}/{c4_total}({100*c4_viol/max(1,c4_total):.1f}%)  worst_grid={worst_grid:.1f}kW  worst_soc={worst_soc:.4f}  {time.time()-t0:.0f}s")
    if term or trunc: obs, info = env.reset()
el = time.time() - t0
print(f"\n{'='*55}\n  RAW AGENT (no PSF)  ({N_STEPS} steps, {el:.0f}s)")
print(f"  C2: {c2_viol}/{c2_total} = {100*c2_viol/max(1,c2_total):.2f}%  worst_soc={worst_soc:.4f}")
print(f"  C4: {c4_viol}/{c4_total} = {100*c4_viol/max(1,c4_total):.2f}%  worst_grid={worst_grid:.1f} kW")
print(f"  Reward: {sum_reward:.1f}\n{'='*55}")
