#!/usr/bin/env python3
"""Comprehensive PSF config sweep. Find the best possible PSF result.
6 configs x 1000 steps. Shows what the filter CAN and CANNOT do."""
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
    c1_v = 0; c2_v = 0; c2_t = 0; c3_v = 0; c3_t = 0; c4_v = 0; c4_t = 0
    rew = 0.0; wg = 0.0; ws = 1.0
    t0 = time.time()
    for si in range(N_STEPS):
        action = agent.act(obs, info)
        obs, reward, term, trunc, info = env.step(action)
        rew += reward
        city = unwrap_to_citylearn(env)
        ti = max(0, city.time_step - 1)
        c1_v += 1 if float(info.get("cost_ev_departure", 0.0)) > 0 else 0
        for b in city.buildings:
            es = getattr(b, "electrical_storage", None)
            if es is not None:
                sa = np.asarray(getattr(es, "soc", []), dtype=float)
                if 0 <= ti < len(sa):
                    s = float(sa[ti]); c2_t += 1; ws = min(ws, s)
                    if s < REAL_SOC_LOW - 1e-6 or s > REAL_SOC_HIGH + 1e-6: c2_v += 1
            nec = getattr(b, "net_electricity_consumption", None)
            if nec is not None and ti < len(nec):
                c3_t += 1
                if abs(float(nec[ti])) > P_BLD_MAX + 1e-6: c3_v += 1
        grid = sum(b.net_electricity_consumption[ti] for b in city.buildings if ti < len(b.net_electricity_consumption))
        c4_t += 1; wg = max(wg, abs(grid))
        if abs(grid) > P_GRID_MAX + 1e-6: c4_v += 1
        if (si + 1) % 500 == 0:
            print(f"  [{label}] {si+1}/{N_STEPS}  C1={c1_v} C2={100*c2_v/max(1,c2_t):.1f}% C3={100*c3_v/max(1,c3_t):.1f}% C4={100*c4_v/max(1,c4_t):.1f}%  {time.time()-t0:.0f}s")
        if term or trunc: obs, info = env.reset()
    el = time.time() - t0
    r = {"c1": c1_v, "c2": 100*c2_v/max(1,c2_t), "c3": 100*c3_v/max(1,c3_t), "c4": 100*c4_v/max(1,c4_t), "wg": wg, "ws": ws, "rew": rew, "el": el}
    print(f"  {label}: C1={c1_v} C2={r['c2']:.1f}% C3={r['c3']:.1f}% C4={r['c4']:.1f}% grid={wg:.0f}kW rew={rew:.0f} ({el:.0f}s)\n")
    return r

print("="*70)
print(f"COMPREHENSIVE PSF SWEEP  |  {N_STEPS} steps each")
print("="*70)
R = {}

# 1. Raw PPOLag baseline
print("\n>>> 1. PPOLag (no PSF)")
e = make_env(); a = OmniSafeCheckpointAgent(CKPT, name="PPOLag")
R["PPOLag"] = run_eval(e, a, "PPOLag")

# 2. RBC greedy baseline
print("\n>>> 2. RBC (greedy EV)")
e = make_env(); a = RBCAgent(ev_mode="greedy")
R["RBC"] = run_eval(e, a, "RBC")

# 3. PSF C1-only (only fix EV, don't touch batteries at all)
print("\n>>> 3. PSF C1-only (w_c1=1000, w_c3=0, w_c4=0)")
e = make_env()
p = LookaheadPSFWrapper(e, horizon=24, w_track=1.0, w_slack_c1=1000.0, w_slack_c3=0.0, w_slack_c4=0.0, freeze_ev=True, verbose=0)
p.p_building_max = P_BLD_MAX; p.p_grid_max = P_GRID_MAX
a = OmniSafeCheckpointAgent(CKPT, name="PSF_C1only")
R["PSF_C1"] = run_eval(p, a, "PSF_C1only")

# 4. PSF Full (current best — all constraints)
print("\n>>> 4. PSF Full (w_c1=1000, w_c3=1000, w_c4=1000)")
e = make_env()
p = LookaheadPSFWrapper(e, horizon=24, w_track=1.0, w_slack_c1=1000.0, w_slack_c3=1000.0, w_slack_c4=1000.0, freeze_ev=True, verbose=0)
p.p_building_max = P_BLD_MAX; p.p_grid_max = P_GRID_MAX
a = OmniSafeCheckpointAgent(CKPT, name="PSF_Full")
R["PSF_Full"] = run_eval(p, a, "PSF_Full")

# 5. PSF C1+C4 heavy (prioritize grid over building power)
print("\n>>> 5. PSF C1+C4 heavy (w_c1=1000, w_c3=100, w_c4=50000)")
e = make_env()
p = LookaheadPSFWrapper(e, horizon=24, w_track=1.0, w_slack_c1=1000.0, w_slack_c3=100.0, w_slack_c4=50000.0, freeze_ev=True, verbose=0)
p.p_building_max = P_BLD_MAX; p.p_grid_max = P_GRID_MAX
a = OmniSafeCheckpointAgent(CKPT, name="PSF_C4heavy")
R["PSF_C4h"] = run_eval(p, a, "PSF_C4heavy")

# 6. PSF low tracking (let QP deviate freely from agent)
print("\n>>> 6. PSF low track (w_track=0.01, w_c1=5000, w_c3=5000, w_c4=5000)")
e = make_env()
p = LookaheadPSFWrapper(e, horizon=24, w_track=0.01, w_slack_c1=5000.0, w_slack_c3=5000.0, w_slack_c4=5000.0, freeze_ev=True, verbose=0)
p.p_building_max = P_BLD_MAX; p.p_grid_max = P_GRID_MAX
a = OmniSafeCheckpointAgent(CKPT, name="PSF_lowtrack")
R["PSF_loT"] = run_eval(p, a, "PSF_lowtrack")

# FINAL TABLE
print("\n" + "="*95)
print("COMPREHENSIVE COMPARISON TABLE")
print("="*95)
keys = ["PPOLag", "RBC", "PSF_C1", "PSF_Full", "PSF_C4h", "PSF_loT"]
labels = ["PPOLag", "RBC(grdy)", "PSF C1only", "PSF Full", "PSF C4hvy", "PSF loTrk"]
print(f"{'Metric':<18}", end="")
for l in labels: print(f"{l:>12}", end="")
print(); print("-"*95)
for metric, fmt in [("c1", "{:>11}"), ("c2", "{:>10.1f}%"), ("c3", "{:>10.1f}%"), ("c4", "{:>10.1f}%"), ("wg", "{:>10.0f}kW"), ("rew", "{:>12.0f}"), ("el", "{:>11.0f}s")]:
    mname = {"c1": "C1 violations", "c2": "C2 rate", "c3": "C3 rate", "c4": "C4 rate", "wg": "Worst grid", "rew": "Reward", "el": "Time"}[metric]
    print(f"{mname:<18}", end="")
    for k in keys:
        v = R[k][metric]
        if metric == "c1": print(f"{v:>12}", end="")
        elif metric in ("c2","c3","c4"): print(f"{v:>11.1f}%", end="")
        elif metric == "wg": print(f"{v:>10.0f}kW", end="")
        elif metric == "rew": print(f"{v:>12.0f}", end="")
        elif metric == "el": print(f"{v:>11.0f}s", end="")
    print()
print("-"*95)

# Find best per metric
print("\nBEST PER METRIC:")
for metric, mname, lower_better in [("c1","C1",True),("c2","C2",True),("c3","C3",True),("c4","C4",True),("rew","Reward",False)]:
    vals = {k: R[k][metric] for k in keys}
    if lower_better: best_k = min(vals, key=vals.get)
    else: best_k = max(vals, key=vals.get)
    best_l = labels[keys.index(best_k)]
    print(f"  {mname}: {best_l} = {vals[best_k]:.1f}" if isinstance(vals[best_k], float) else f"  {mname}: {best_l} = {vals[best_k]}")

print("\nPHYSICS CEILING NOTE:")
print("  C3 minimum ~17-25% (base loads exceed building power limit — no filter can fix this)")
print("  C4 minimum with hard C2: ~30% (battery SoC clamp prevents agent's discharge strategy)")
print("  C4 with agent's own strategy: ~7% (agent accepts 3.4% C2 violations to keep C4 low)")
