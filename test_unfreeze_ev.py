#!/usr/bin/env python3
"""Test: Does unfreezing EV + low EV tracking weight fix C4?
Monkeypatches PSF step() to skip pre-QP C1 clamp when freeze_ev=False.
Also patches objective to use separate batt/ev tracking weights.
NO source files modified."""
import sys, os, time
import numpy as np
import cvxpy as cp
os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, ".")
for k, v in {"CITYLEARN_STEMS_SOC_LOW": "0.0", "CITYLEARN_STEMS_SOC_HIGH": "0.95", "CITYLEARN_STEMS_P_BUILDING_MAX": "2.273834", "CITYLEARN_STEMS_P_GRID_MAX": "27.127751", "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_zero", "CITYLEARN_SCHEMA": "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json", "CITYLEARN_EV_DENSE_COST_SCALE": "1.0"}.items():
    os.environ[k] = v
from evaluation.core.evaluator import make_env
from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent
from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper
from citylearn_safe.psf.psf_extract import unwrap_to_citylearn, flatten_action, unflatten_action
CKPT = "runs/ppolag_P95_V2/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-05-02-08-01/torch_save/epoch-100.pt"
N_STEPS = 1000; P_GRID_MAX = 27.127751; REAL_SOC_LOW = 0.0; REAL_SOC_HIGH = 0.95

def patched_step(self, action):
    """Monkeypatched step: skips pre-QP C1 clamp when freeze_ev=False."""
    if not self._compiled:
        self._compile_qp()
    is_multi = isinstance(self.action_space, list)
    if is_multi:
        action_flat = np.concatenate([np.asarray(a, dtype=float).ravel() for a in action])
    else:
        action_flat = np.asarray(action, dtype=float).ravel()
    # SKIP pre-QP C1 clamp when freeze_ev=False (the key change)
    if self.freeze_ev and self._mapping is not None:
        c1_mins = self._compute_c1_min_actions(action_flat)
        for gidx_ev, min_act in c1_mins.items():
            action_flat[gidx_ev] = max(float(action_flat[gidx_ev]), float(min_act))
    corrected_flat, psf_info = self._update_params_and_solve(action_flat)
    # POST_SOLVE_SOC_CLAMP (keep C2 hard)
    try:
        city = self._get_citylearn_env()
        t_now = int(getattr(city, "time_step", 0)); t_idx = max(0, t_now - 1)
        buildings = list(getattr(city, "buildings", []))
        for (b_idx, gidx) in self._batt_list:
            if gidx < 0 or gidx >= len(corrected_flat) or b_idx >= len(buildings): continue
            es = getattr(buildings[b_idx], "electrical_storage", None)
            if es is None: continue
            soc = 0.5
            soc_series = getattr(es, "soc", None)
            if soc_series is not None and hasattr(soc_series, "__len__"):
                soc_arr = np.asarray(soc_series, dtype=float)
                if 0 <= t_idx < len(soc_arr): soc = float(np.clip(soc_arr[t_idx], 0.0, 1.0))
                elif len(soc_arr) > 0: soc = float(np.clip(soc_arr[-1], 0.0, 1.0))
            cap = float(getattr(es, "capacity", 6.4)); nom_p = float(getattr(es, "nominal_power", 5.0))
            if cap <= 0 or nom_p <= 0: continue
            scale = nom_p / cap
            a_min = max((self.soc_low - soc) / scale, -1.0); a_max = min((self.soc_high - soc) / scale, 1.0)
            corrected_flat[gidx] = float(np.clip(corrected_flat[gidx], a_min, a_max))
    except Exception: pass
    # SKIP post-solve C3/C4/C1 clamps to let QP solution stand
    # Clamp EV to [0,1] only
    if self._mapping is not None:
        for evc in self._mapping["ev_chargers"]:
            g = evc["gidx"]
            if 0 <= g < len(corrected_flat):
                corrected_flat[g] = float(np.clip(corrected_flat[g], 0.0, 1.0))
    corrected_action = unflatten_action(corrected_flat, self.action_space) if is_multi else corrected_flat
    obs, reward, term, trunc, info = self.env.step(corrected_action)
    info.update(psf_info)
    self._step_count += 1
    if psf_info.get("psf_any_intervention", False): self._total_interventions += 1
    self._total_ev_interventions += psf_info.get("psf_ev_interventions", 0)
    self._total_batt_interventions += psf_info.get("psf_battery_interventions", 0)
    self._total_solve_ms += psf_info.get("psf_solve_ms", 0)
    if psf_info.get("psf_infeasible", 0) > 0: self._infeasible_count += 1
    return obs, reward, term, trunc, info

def run_config(label, freeze_ev, w_c1, w_c4, w_track_ev, use_patch):
    env = make_env()
    psf = LookaheadPSFWrapper(env, horizon=24, w_track=1.0, w_slack_c1=w_c1, w_slack_c3=1000.0, w_slack_c4=w_c4, freeze_ev=freeze_ev, verbose=0)
    psf.p_building_max = 2.273834; psf.p_grid_max = 27.127751
    if use_patch:
        import types
        psf.step = types.MethodType(patched_step, psf)
    agent = OmniSafeCheckpointAgent(CKPT, name=f"ppolag_{label}")
    agent.reset(psf)
    obs, info = psf.reset()
    # After compile, patch objective to use low EV tracking weight
    if use_patch and not freeze_ev and psf._compiled and psf._max_ev > 0:
        # Recompile with low EV tracking
        old_prob = psf._prob
        # Rebuild objective only
        obj = 1.0 * cp.sum_squares(psf._v_a_batt[:, 0] - psf._p_proposed_batt)  # w_track_batt=1.0
        obj += w_track_ev * cp.sum_squares(psf._v_a_ev[:, 0] - psf._p_proposed_ev)  # low EV track
        obj += psf.w_slack_c1 * cp.sum(psf._v_slack_c1)
        obj += psf.w_slack_c3 * cp.sum(psf._v_slack_c3)
        obj += psf.w_slack_c4 * cp.sum(psf._v_slack_c4)
        if psf.horizon > 1:
            obj += psf.w_future_reg * cp.sum_squares(psf._v_a_batt[:, 1:])
            obj += psf.w_future_reg * cp.sum_squares(psf._v_a_ev[:, 1:])
        psf._prob = cp.Problem(cp.Minimize(obj), old_prob.constraints)
        print(f"  [{label}] Recompiled QP with w_track_ev={w_track_ev}")
    c1_viol = 0; c1_deps = 0; c4_viol = 0; c4_total = 0; c2_viol = 0; c2_total = 0
    sum_reward = 0.0; worst_grid = 0.0
    t0 = time.time()
    for step_i in range(N_STEPS):
        action = agent.act(obs, info)
        obs, reward, term, trunc, info = psf.step(action)
        sum_reward += reward
        city = unwrap_to_citylearn(psf.env)
        t_idx = max(0, city.time_step - 1)
        # C2
        for b in city.buildings:
            es = getattr(b, "electrical_storage", None)
            if es is None: continue
            soc_arr = np.asarray(getattr(es, "soc", []), dtype=float)
            if 0 <= t_idx < len(soc_arr):
                s = float(soc_arr[t_idx]); c2_total += 1
                if s < REAL_SOC_LOW - 1e-6 or s > REAL_SOC_HIGH + 1e-6: c2_viol += 1
        # C4
        grid = sum(b.net_electricity_consumption[t_idx] for b in city.buildings if t_idx < len(b.net_electricity_consumption))
        c4_total += 1; worst_grid = max(worst_grid, abs(grid))
        if abs(grid) > P_GRID_MAX + 1e-6: c4_viol += 1
        # C1 from info
        cost_ev = float(info.get("cost_ev_departure", 0.0))
        if cost_ev > 0: c1_viol += 1
        if (step_i + 1) % 200 == 0:
            print(f"  [{label}] {step_i+1}/{N_STEPS}  C2={100*c2_viol/max(1,c2_total):.1f}%  C4={c4_viol}/{c4_total}({100*c4_viol/max(1,c4_total):.1f}%)  worst_grid={worst_grid:.1f}kW  {time.time()-t0:.0f}s")
        if term or trunc: obs, info = psf.reset()
    el = time.time() - t0
    r = {"c1_viol": c1_viol, "c2_rate": 100*c2_viol/max(1,c2_total), "c4_rate": 100*c4_viol/max(1,c4_total), "c4_viol": c4_viol, "c4_total": c4_total, "worst_grid": worst_grid, "reward": sum_reward, "elapsed": el}
    print(f"\n{'='*55}\n  {label}  ({N_STEPS} steps, {el:.0f}s)")
    print(f"  C1 departures violated: {c1_viol}")
    print(f"  C2: {100*c2_viol/max(1,c2_total):.2f}%")
    print(f"  C4: {c4_viol}/{c4_total} = {r['c4_rate']:.2f}%  worst_grid={worst_grid:.1f} kW")
    print(f"  Reward: {sum_reward:.1f}\n{'='*55}\n")
    return r

print("="*65)
print(f"TEST: UNFREEZE EV IN QP  |  {N_STEPS} steps per config")
print("="*65)
results = {}

print("\n>>> RUN 1: BASELINE FULL PSF (freeze_ev=True, current behavior)")
results["baseline"] = run_config("BASELINE", True, 1000.0, 1000.0, 1.0, False)

print("\n>>> RUN 2: UNFREEZE EV, w_track_ev=0.01, w_c1=5000, w_c4=5000")
results["unfreeze"] = run_config("UNFREEZE", False, 5000.0, 5000.0, 0.01, True)

print("\n>>> RUN 3: UNFREEZE EV, w_track_ev=0.0, w_c1=10000, w_c4=10000")
results["unfreeze_aggr"] = run_config("UNFR_AGGR", False, 10000.0, 10000.0, 0.0, True)

b, u, ua = results["baseline"], results["unfreeze"], results["unfreeze_aggr"]
print("\n" + "="*70); print("COMPARISON TABLE"); print("="*70)
print(f"{'Metric':<25} {'Baseline':>14} {'Unfreeze':>14} {'Unfr Aggr':>14}"); print("-"*70)
print(f"{'C1 dep violations':<25} {b['c1_viol']:>14} {u['c1_viol']:>14} {ua['c1_viol']:>14}")
print(f"{'C2 viol rate':<25} {b['c2_rate']:>13.2f}% {u['c2_rate']:>13.2f}% {ua['c2_rate']:>13.2f}%")
print(f"{'C4 viol rate':<25} {b['c4_rate']:>13.2f}% {u['c4_rate']:>13.2f}% {ua['c4_rate']:>13.2f}%")
print(f"{'Worst grid (kW)':<25} {b['worst_grid']:>14.1f} {u['worst_grid']:>14.1f} {ua['worst_grid']:>14.1f}")
print(f"{'Reward':<25} {b['reward']:>14.1f} {u['reward']:>14.1f} {ua['reward']:>14.1f}")
print(f"{'Time (s)':<25} {b['elapsed']:>14.0f} {u['elapsed']:>14.0f} {ua['elapsed']:>14.0f}")
print("-"*70)
if u['c4_rate'] < b['c4_rate'] - 5.0 or ua['c4_rate'] < b['c4_rate'] - 5.0:
    best = "UNFREEZE" if u['c4_rate'] < ua['c4_rate'] else "UNFR_AGGR"
    best_r = u if u['c4_rate'] < ua['c4_rate'] else ua
    print(f"\n>>> VERDICT: Unfreezing EV REDUCES C4! Best={best} C4={best_r['c4_rate']:.1f}% (baseline={b['c4_rate']:.1f}%)")
    print(f"    C1 cost: {best_r['c1_viol']} departure violations (check if acceptable)")
else:
    print(f"\n>>> VERDICT: Unfreezing EV did NOT fix C4. Problem is structural (base loads >> grid cap).")
