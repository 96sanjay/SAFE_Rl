#!/usr/bin/env python3
"""Deterministic 1-episode evaluation for CSAC-LB V2G (70-dim, no forecast-obs).

Uses the CSAC-LB-native env pipeline (CityLearnEnv -> NormalizedObservationWrapper
-> SingleAgentListAdapter -> CityLearnSafetyEnvV3, no ForecastObsWrapper) so obs
shape matches the 70-dim checkpoint. Reports the same metrics as
eval_r28b_vs_sac.py (EpRet, per-constraint violations, CityLearn KPIs).
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse metric constants + actor classes from the unified eval
from scripts.eval_r28b_vs_sac import (
    COST_KEYS, COST_LABELS, COST_LIMITS,
    SACActor, normalize_obs, get_citylearn_env,
)

CKPT = "runs/csac_lb_v2g/CSACLBV2G-{CityLearnSafety-V2G-v2}/seed-042-2026-04-17-05-58-55/torch_save/epoch-100.pt"

# Env vars matching the R28 9-term STEMS reward (same as run_r28_benchmarks.sh /
# run_r28_pposaute.sh / /tmp/r28_extra_launch.sh) so CSAC-LB is scored on the
# EXACT same reward and cost aggregation as every other R28 algorithm.
# Even though this CSAC-LB checkpoint was trained on a different reward set
# (csac_lb_v2g.yaml env_overrides), evaluating under the 9-term reward gives
# an apples-to-apples EpRet/EpCost that lines up with the rest of the bench.
CSAC_ENV = {
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_BATT_CLAMP": "0",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_ACTION_MASK": "0",
    "CITYLEARN_POLICY_ACTION_MASK": "0",
    "CITYLEARN_KPI_FLUSH_EVERY_STEP": "0",
    "CITYLEARN_DEBUG_ACTION_CLIP": "0",
    # 9-term STEMS reward (identical to R28 benchmark launch scripts)
    "STEMS_LAMBDA_EV": "2.0",
    "STEMS_ALPHA_EV_SMART": "1.5",
    "STEMS_EV_SLACK_ARB_SCALE": "1.0",
    "STEMS_ALPHA_V2G_CONTEXT": "1.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_GRID": "0.5",
    "STEMS_SG_THRESHOLD": "0.5",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BUILD": "0.3",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_XI_RENEWABLE": "0.2",
    # Disabled terms (R28 stock)
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_EV_GUARD": "0.0",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "STEMS_ALPHA_LOAD_SHIFT": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.0",
    # R28 stock cost aggregation
    "COST_W_C2": "5.0",
    "COST_W_C3": "5.0",
    "COST_W_C1_DENSE": "0.0",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    "CITYLEARN_EV_COST_SCALE": "3.0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
}


def build_csac_env():
    """CSAC-LB native env pipeline: 70-dim obs, no forecast wrapper."""
    from citylearn.citylearn import CityLearnEnv
    from citylearn.wrappers import NormalizedObservationWrapper
    from citylearn_safe.adapters import SingleAgentListAdapter
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

    schema = os.environ["CITYLEARN_SCHEMA"]
    base = CityLearnEnv(schema=schema, central_agent=True)
    base = NormalizedObservationWrapper(base)
    base = SingleAgentListAdapter(base)
    env = CityLearnSafetyEnvV3(
        base,
        soc_min=float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0")),
        soc_max=float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95")),
        cost_mode="hinge",
    )
    return env


def evaluate():
    for k, v in CSAC_ENV.items():
        os.environ[k] = v
    os.environ["CITYLEARN_SCHEMA"] = os.path.join(
        PROJECT_ROOT,
        "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    )

    device = torch.device("cpu")
    env = build_csac_env()
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    print(f"  env obs_dim={obs_dim}  act_dim={act_dim}")

    ckpt_path = os.path.join(PROJECT_ROOT, CKPT)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pi_sd = ckpt["pi"]

    model = SACActor(obs_dim, act_dim)
    net_sd = {k: v for k, v in pi_sd.items() if k.startswith("net.")}
    missing, unexpected = model.load_state_dict(net_sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        print(f"  WARNING unexpected: {unexpected}")
    obs_norm = ckpt.get("obs_normalizer", None)
    model = model.to(device).eval()

    obs, info = env.reset()
    step_rewards = []
    step_costs = defaultdict(list)
    step_violations = defaultdict(list)
    ev_departure_events = []
    c3_per_building_violations = []

    t0 = time.time()
    for step in range(8760):
        obs_normed = normalize_obs(obs, obs_norm, device)
        with torch.no_grad():
            action = model(obs_normed)
        action_np = action.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action_np)
        step_rewards.append(reward)

        for key in COST_KEYS:
            cost_val = float(info.get(key, 0.0))
            step_costs[key].append(cost_val)
            step_violations[key].append(1.0 if cost_val > 0.0 else 0.0)

        ev_departure_events.append(int(info.get("ev_departure_departures", 0)))
        c3_per_building_violations.append(float(info.get("building_power_violation_count", 0.0)))

        if terminated or truncated:
            break
    ep_len = len(step_rewards)
    ep_ret = sum(step_rewards)

    print(f"\n  Ran {ep_len} steps in {time.time()-t0:.1f}s")
    print(f"  EpRet = {ep_ret:.1f}")
    print()
    print("  --- Per-constraint ---")
    for i, (key, label, lim) in enumerate(zip(COST_KEYS, COST_LABELS, COST_LIMITS)):
        total = sum(step_costs[key])
        viol_pct = 100.0 * np.mean(step_violations[key])
        passed = total <= lim
        print(f"    {label:28s}  total={total:10.1f}  limit={lim:>8d}  {'PASS' if passed else 'FAIL'}   viol%={viol_pct:5.1f}%")

    total_deps = sum(ev_departure_events)
    c0_deficit = sum(
        1 for dep, cost in zip(ev_departure_events, step_costs[COST_KEYS[0]])
        if dep > 0 and cost > 0
    )
    print()
    print("  --- C0 per-departure ---")
    print(f"    departures       : {total_deps}")
    print(f"    with deficit     : {c0_deficit}")
    print(f"    rate             : {100.0 * c0_deficit / max(total_deps, 1):.1f}%")

    c3_arr = np.array(c3_per_building_violations)
    print()
    print("  --- C3 per-building ---")
    print(f"    steps with any bld violating : {100.0 * np.sum(c3_arr > 0) / ep_len:.1f}%")
    print(f"    per-building rate            : {100.0 * np.sum(c3_arr) / (ep_len * 5):.1f}%")

    citylearn_env = get_citylearn_env(env)
    if citylearn_env is not None and hasattr(citylearn_env, "evaluate"):
        try:
            kpi_df = citylearn_env.evaluate()
            district = kpi_df[kpi_df["level"] == "district"]
            print()
            print("  --- District KPIs (<1 = better than RBC baseline) ---")
            for _, row in district.iterrows():
                cf = str(row["cost_function"])
                v = row.get("value")
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    print(f"    {cf:44s}  {float(v):.4f}")
        except Exception as e:
            print(f"  WARNING: KPI eval failed: {e}")


if __name__ == "__main__":
    evaluate()
