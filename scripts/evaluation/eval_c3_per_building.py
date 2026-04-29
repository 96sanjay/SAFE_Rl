#!/usr/bin/env python3
"""
Per-Building C3 (Building Power) Violation Analysis
====================================================
Runs deterministic rollouts for both Softmax and GradS checkpoints,
recording per-building net_electricity_consumption at each timestep.

Reports:
  - Per-building violation rate (% of 8759 steps where |power| > P_BUILDING_MAX)
  - Per-timestep violation rate (% of steps where ANY building exceeds)
  - Mean violations per violating timestep (how many buildings exceed simultaneously)
  - Per-building max/P95/mean power
  - Structural vs agent-caused violations (NSL+solar baseline)

Usage:
    cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
    conda activate citylearn
    PYTHONPATH=$PWD python scripts/eval_c3_per_building.py
"""
from __future__ import annotations

import os
import sys
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

P_BUILDING_MAX = 4.6083
P_GRID_MAX = 10.2352
SOC_UPPER_CLAMP = 0.94
BATT_DT = 1.0

# =========================================================================
# Environment variables (matching r25b ablation configs)
# =========================================================================
COMMON_ENV = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_ACTION_MASK": "0",
    "CITYLEARN_BATT_CLAMP": "1",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_KPI_RUN_NAME": "__eval__",
    "STEMS_ALPHA_GRID": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.3",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_XI_RENEWABLE": "0.2",
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_LAMBDA_EV": "5.0",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_EV_GUARD": "1.0",
    "STEMS_ALPHA_V2G_CONTEXT": "3.0",
}


def set_env(overrides=None):
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_", "MLAG_",
                         "SAUTE_", "SHAPING_", "EV_CLIP_", "FRAME_STACK_")):
            os.environ.pop(k, None)
    for k, v in COMMON_ENV.items():
        os.environ[k] = v
    if overrides:
        for k, v in overrides.items():
            os.environ[k] = v


set_env()

from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.extractors import unwrap_to_raw_citylearn_env
import citylearn_safe.schema_index as si


# =========================================================================
# Actor loading
# =========================================================================
class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.mean = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.tanh(self.mean(obs))


def load_actor(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi_state = ckpt["pi"]
    h1 = pi_state["mean.0.weight"].shape[0]
    h2 = pi_state["mean.2.weight"].shape[0]
    obs_dim = pi_state["mean.0.weight"].shape[1]
    act_dim = pi_state["mean.4.weight"].shape[0]
    actor = MLPActor(obs_dim, act_dim, (h1, h2))
    filtered = {k: v for k, v in pi_state.items() if not k.startswith("log_std")}
    result = actor.load_state_dict(filtered, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Missing keys: {result.missing_keys}")
    actor.eval()
    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        clip_t = norm["_clip"].float()
        obs_clip = float(clip_t.mean())
    return actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim


# =========================================================================
# Battery clamping
# =========================================================================
def discover_battery_actions(raw_env):
    buildings = list(getattr(raw_env, 'buildings', []))
    names_raw = getattr(raw_env, 'action_names', [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        flat_names = names_raw[0]
    else:
        flat_names = list(names_raw)
    batt_pos = [i for i, n in enumerate(flat_names) if str(n).lower() == 'electrical_storage']
    result = []
    for b_idx, b in enumerate(buildings):
        es = getattr(b, 'electrical_storage', None)
        if es is None:
            continue
        if b_idx < len(batt_pos):
            act_idx = batt_pos[b_idx]
        else:
            continue
        cap = float(getattr(es, 'capacity', 6.4) or 6.4)
        p_max = float(getattr(es, 'nominal_power', 5.0) or 5.0)
        eta = float(getattr(es, 'efficiency', 0.9) or 0.9)
        result.append((act_idx, b_idx, cap, p_max, eta))
    return result


def clamp_battery_actions(a, batt_action_map, raw_env):
    if not batt_action_map:
        return a
    buildings = list(getattr(raw_env, 'buildings', []))
    t_idx = max(0, int(getattr(raw_env, 'time_step', 0)) - 1)
    a_clamped = a.copy()
    for act_idx, bld_idx, cap, p_max, eta in batt_action_map:
        if act_idx >= len(a_clamped) or bld_idx >= len(buildings):
            continue
        es = getattr(buildings[bld_idx], 'electrical_storage', None)
        if es is None:
            continue
        soc_arr = getattr(es, 'soc', None)
        if soc_arr is None or not hasattr(soc_arr, '__len__') or len(soc_arr) <= t_idx:
            continue
        soc = float(np.clip(soc_arr[t_idx], 0.0, 1.0))
        denom_charge = p_max * BATT_DT * eta
        max_charge = (SOC_UPPER_CLAMP - soc) * cap / denom_charge if denom_charge > 0 else 1.0
        denom_discharge = p_max * BATT_DT
        max_discharge = soc * cap * eta / denom_discharge if denom_discharge > 0 else 1.0
        a_clamped[act_idx] = float(np.clip(a_clamped[act_idx], -max_discharge, max_charge))
    return a_clamped


# =========================================================================
# Rollout with per-building tracking
# =========================================================================
def rollout_per_building(agent_name, ckpt_path, seed=42):
    """Run full rollout, return per-building power arrays and violation data."""
    print(f"\n{'='*70}")
    print(f"  ROLLING OUT: {agent_name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*70}")

    set_env()
    si._CACHE = None

    actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(ckpt_path)
    print(f"  Actor: obs_dim={obs_dim}, act_dim={act_dim}")

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnv(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)

    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))
    n_buildings = len(buildings)
    print(f"  Buildings: {n_buildings}")

    batt_action_map = discover_battery_actions(raw)

    env_obs_dim = env.observation_space.shape[0]
    need_pad = obs_dim > env_obs_dim
    pad_dim = obs_dim - env_obs_dim if need_pad else 0

    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    # Per-building tracking arrays (pre-allocate for 8760)
    max_steps = 8760
    building_power = np.zeros((max_steps, n_buildings), dtype=np.float64)
    building_nsl = np.zeros((max_steps, n_buildings), dtype=np.float64)  # non-shiftable load + solar
    building_batt_power = np.zeros((max_steps, n_buildings), dtype=np.float64)
    per_step_c3_cost = np.zeros(max_steps, dtype=np.float64)  # aggregate C3 from info
    per_step_c4_cost = np.zeros(max_steps, dtype=np.float64)
    grid_import = np.zeros(max_steps, dtype=np.float64)

    while not done:
        if need_pad:
            obs_padded = np.concatenate([obs, np.ones(pad_dim)])
        else:
            obs_padded = obs

        obs_t = torch.as_tensor(obs_padded, dtype=torch.float32).unsqueeze(0)
        if obs_mean is not None:
            obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
            if obs_clip is not None:
                obs_t = obs_t.clamp(-obs_clip, obs_clip)
        with torch.no_grad():
            action_t = actor(obs_t).squeeze(0).numpy()
        action = np.clip(action_t, -1.0, 1.0)
        action = clamp_battery_actions(action, batt_action_map, raw)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        t_now = int(getattr(raw, "time_step", 0))
        t_idx = max(0, t_now - 1)

        # Record per-building power
        for b_idx, bld in enumerate(buildings):
            try:
                nec = getattr(bld, "net_electricity_consumption", None)
                if nec is not None and len(nec) > t_idx:
                    building_power[step, b_idx] = float(nec[t_idx])
            except Exception:
                pass

            # Record exogenous load (NSL + solar)
            try:
                nsl_arr = np.asarray(
                    getattr(bld, "_Building__energy_to_non_shiftable_load", []),
                    dtype=float,
                )
                sg_arr = np.asarray(
                    getattr(bld, "_Building__solar_generation", []),
                    dtype=float,
                )
                exo = 0.0
                if len(nsl_arr) > t_idx:
                    exo = float(nsl_arr[t_idx])
                if len(sg_arr) > t_idx:
                    exo += float(sg_arr[t_idx])
                building_nsl[step, b_idx] = exo
            except Exception:
                pass

            # Battery contribution = NEC - exogenous
            building_batt_power[step, b_idx] = building_power[step, b_idx] - building_nsl[step, b_idx]

        per_step_c3_cost[step] = info.get("cost_stems_building_power", 0.0)
        per_step_c4_cost[step] = info.get("cost_stems_grid_power", 0.0)
        grid_import[step] = info.get("grid_import_kwh", 0.0)

        step += 1
        if step % 2000 == 0:
            print(f"    Step {step}: C3_cum={per_step_c3_cost[:step].sum():.0f}")

    # Trim to actual steps
    n = step
    building_power = building_power[:n]
    building_nsl = building_nsl[:n]
    building_batt_power = building_batt_power[:n]
    per_step_c3_cost = per_step_c3_cost[:n]
    per_step_c4_cost = per_step_c4_cost[:n]
    grid_import = grid_import[:n]

    print(f"  Completed {n} steps.")
    return {
        "name": agent_name,
        "n_steps": n,
        "n_buildings": n_buildings,
        "building_power": building_power,
        "building_nsl": building_nsl,
        "building_batt_power": building_batt_power,
        "per_step_c3_cost": per_step_c3_cost,
        "per_step_c4_cost": per_step_c4_cost,
        "grid_import": grid_import,
    }


# =========================================================================
# Analysis
# =========================================================================
def analyze_violations(result):
    """Compute all violation metrics from rollout data."""
    name = result["name"]
    n = result["n_steps"]
    n_b = result["n_buildings"]
    bp = result["building_power"]      # (n, n_b)
    nsl = result["building_nsl"]       # (n, n_b)
    batt = result["building_batt_power"]  # (n, n_b)
    c3 = result["per_step_c3_cost"]
    c4 = result["per_step_c4_cost"]
    gi = result["grid_import"]

    # Per-building violation matrix: True where |power| > threshold
    viol_matrix = np.abs(bp) > P_BUILDING_MAX   # (n, n_b)
    nsl_viol_matrix = np.abs(nsl) > P_BUILDING_MAX   # structural violations

    print(f"\n{'#'*80}")
    print(f"  DETAILED C3 ANALYSIS: {name}")
    print(f"{'#'*80}")

    # ---- Per-building violation rates ----
    print(f"\n  Per-Building Violation Rates (threshold = {P_BUILDING_MAX} kW)")
    print(f"  {'Building':<12} {'Viols':>8} {'Rate(%)':>10} {'MaxPwr':>10} {'P95Pwr':>10} {'MeanPwr':>10} {'StructViols':>12} {'AgentViols':>11}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*11}")

    per_bld_viols = []
    per_bld_rates = []
    per_bld_structural = []
    per_bld_agent_caused = []

    for b in range(n_b):
        viols = int(viol_matrix[:, b].sum())
        rate = 100.0 * viols / n
        max_pwr = float(np.abs(bp[:, b]).max())
        p95_pwr = float(np.percentile(np.abs(bp[:, b]), 95))
        mean_pwr = float(np.abs(bp[:, b]).mean())

        # Structural vs agent-caused
        struct_viols = int(nsl_viol_matrix[:, b].sum())
        # Agent-caused: violation present in NEC but NOT in NSL
        agent_viols = int((viol_matrix[:, b] & ~nsl_viol_matrix[:, b]).sum())

        per_bld_viols.append(viols)
        per_bld_rates.append(rate)
        per_bld_structural.append(struct_viols)
        per_bld_agent_caused.append(agent_viols)

        print(f"  Bld {b:<7} {viols:>8} {rate:>9.2f}% {max_pwr:>9.2f} {p95_pwr:>9.2f} {mean_pwr:>9.2f} {struct_viols:>12} {agent_viols:>11}")

    # ---- Aggregate per-timestep ----
    any_viol_per_step = viol_matrix.any(axis=1)  # True if ANY building violates
    n_viol_per_step = viol_matrix.sum(axis=1)    # count of violating buildings per step
    timestep_viol_count = int(any_viol_per_step.sum())
    timestep_viol_rate = 100.0 * timestep_viol_count / n

    # How many buildings violate per violating timestep
    violating_counts = n_viol_per_step[any_viol_per_step]
    mean_viols_per_violating = float(violating_counts.mean()) if len(violating_counts) > 0 else 0.0
    max_viols_per_step = int(n_viol_per_step.max())

    print(f"\n  Aggregate Per-Timestep Statistics:")
    print(f"    Timesteps with ANY violation: {timestep_viol_count}/{n} ({timestep_viol_rate:.2f}%)")
    print(f"    Mean buildings violating per violating timestep: {mean_viols_per_violating:.2f}")
    print(f"    Max buildings violating simultaneously: {max_viols_per_step}")

    # Distribution of simultaneous violations
    print(f"\n  Distribution of simultaneous violations per step:")
    for k in range(1, n_b + 1):
        count = int((n_viol_per_step == k).sum())
        if count > 0:
            print(f"    Exactly {k} building(s): {count} steps ({100.0*count/n:.2f}%)")

    # ---- Structural vs agent-caused summary ----
    total_viols = sum(per_bld_viols)
    total_structural = sum(per_bld_structural)
    total_agent = sum(per_bld_agent_caused)
    # Some violations could be both structural AND agent -- agent made it worse
    both = total_viols - total_agent - total_structural + sum(
        int((viol_matrix[:, b] & nsl_viol_matrix[:, b]).sum()) for b in range(n_b)
    )

    print(f"\n  Structural vs Agent-Caused (per building-step pairs):")
    print(f"    Total building-step violations:   {total_viols}")
    print(f"    Structural (NSL+solar > limit):   {total_structural} ({100.0*total_structural/max(1,total_viols):.1f}%)")
    print(f"    Agent-caused (NEC > limit, NSL ok): {total_agent} ({100.0*total_agent/max(1,total_viols):.1f}%)")

    # Violations where both structural AND NEC violate (agent may worsen or improve)
    both_count = sum(int((viol_matrix[:, b] & nsl_viol_matrix[:, b]).sum()) for b in range(n_b))
    print(f"    Both structural AND NEC violate:  {both_count}")

    # ---- C3 cost breakdown ----
    c3_viol_steps = int((c3 > 0).sum())
    print(f"\n  C3 Cost from Environment Info:")
    print(f"    Steps with C3 > 0: {c3_viol_steps}/{n} ({100.0*c3_viol_steps/n:.2f}%)")
    print(f"    Cumulative C3:     {c3.sum():.2f}")
    print(f"    Mean C3 (all):     {c3.mean():.4f}")
    print(f"    Mean C3 (viols):   {c3[c3>0].mean():.4f}" if c3_viol_steps > 0 else "    Mean C3 (viols):   N/A")
    print(f"    Max C3 (step):     {c3.max():.4f}")

    # ---- C4 summary ----
    c4_viol_steps = int((c4 > 0).sum())
    print(f"\n  C4 (Grid Power) Summary:")
    print(f"    Steps with C4 > 0: {c4_viol_steps}/{n} ({100.0*c4_viol_steps/n:.2f}%)")
    print(f"    Cumulative C4:     {c4.sum():.2f}")
    print(f"    Grid import max:   {gi.max():.2f} kW")
    print(f"    Grid import P95:   {np.percentile(gi, 95):.2f} kW")

    # ---- Per-building violation magnitudes ----
    print(f"\n  Per-Building Violation Magnitudes (excess above {P_BUILDING_MAX} kW):")
    print(f"  {'Building':<12} {'MeanExcess':>12} {'MaxExcess':>12} {'P95Excess':>12} {'SumExcess':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for b in range(n_b):
        excess = np.maximum(0, np.abs(bp[:, b]) - P_BUILDING_MAX)
        viol_mask = excess > 0
        if viol_mask.sum() > 0:
            mean_ex = float(excess[viol_mask].mean())
            max_ex = float(excess[viol_mask].max())
            p95_ex = float(np.percentile(excess[viol_mask], 95))
            sum_ex = float(excess.sum())
        else:
            mean_ex = max_ex = p95_ex = sum_ex = 0.0
        print(f"  Bld {b:<7} {mean_ex:>11.4f} {max_ex:>11.4f} {p95_ex:>11.4f} {sum_ex:>11.2f}")

    # ---- Battery contribution analysis ----
    print(f"\n  Battery Contribution at Violation Moments:")
    print(f"  {'Building':<12} {'MeanBattPwr':>13} {'AtViolMean':>13} {'Charging%':>11} {'Dischg%':>9}")
    print(f"  {'-'*12} {'-'*13} {'-'*13} {'-'*11} {'-'*9}")
    for b in range(n_b):
        mean_batt = float(batt[:, b].mean())
        viol_mask = viol_matrix[:, b]
        if viol_mask.sum() > 0:
            mean_batt_at_viol = float(batt[viol_mask, b].mean())
            charge_pct = 100.0 * (batt[viol_mask, b] > 0.01).sum() / viol_mask.sum()
            dischg_pct = 100.0 * (batt[viol_mask, b] < -0.01).sum() / viol_mask.sum()
        else:
            mean_batt_at_viol = 0.0
            charge_pct = 0.0
            dischg_pct = 0.0
        print(f"  Bld {b:<7} {mean_batt:>12.4f} {mean_batt_at_viol:>12.4f} {charge_pct:>10.1f}% {dischg_pct:>8.1f}%")

    return {
        "name": name,
        "n_steps": n,
        "n_buildings": n_b,
        "per_bld_viols": per_bld_viols,
        "per_bld_rates": per_bld_rates,
        "per_bld_structural": per_bld_structural,
        "per_bld_agent_caused": per_bld_agent_caused,
        "timestep_viol_count": timestep_viol_count,
        "timestep_viol_rate": timestep_viol_rate,
        "mean_viols_per_violating_step": mean_viols_per_violating,
        "max_viols_per_step": max_viols_per_step,
        "c3_cumulative": float(c3.sum()),
        "c4_cumulative": float(c4.sum()),
    }


def print_comparison(results_list):
    """Side-by-side comparison table."""
    W = 100
    print(f"\n\n{'='*W}")
    print(f"  SIDE-BY-SIDE COMPARISON: C3 PER-BUILDING VIOLATIONS")
    print(f"{'='*W}")

    names = [r["name"] for r in results_list]
    col_w = 20

    def header():
        h = f"  {'Metric':<45}"
        for n in names:
            h += f"{n:>{col_w}}"
        print(h)
        print(f"  {'='*45}" + f"{'='*col_w}" * len(names))

    header()

    def row(label, key_or_fn, fmt=".2f"):
        line = f"  {label:<45}"
        for r in results_list:
            if callable(key_or_fn):
                v = key_or_fn(r)
            else:
                v = r.get(key_or_fn, float("nan"))
            val_str = f"{v:{fmt}}"
            line += f"{val_str:>{col_w}}"
        print(line)

    print(f"\n  --- Aggregate ---")
    row("Timestep violation rate (%)", "timestep_viol_rate", ".2f")
    row("Mean bldgs violating per viol step", "mean_viols_per_violating_step", ".2f")
    row("Max bldgs violating simultaneously", "max_viols_per_step", ".0f")
    row("C3 cumulative cost", "c3_cumulative", ".1f")
    row("C4 cumulative cost", "c4_cumulative", ".1f")

    n_b = results_list[0]["n_buildings"]
    print(f"\n  --- Per-Building Violation Rates (%) ---")
    for b in range(n_b):
        row(f"Bld {b} violation rate (%)", lambda r, b=b: r["per_bld_rates"][b], ".2f")

    print(f"\n  --- Per-Building Violation Counts ---")
    for b in range(n_b):
        row(f"Bld {b} violations (of {results_list[0]['n_steps']})", lambda r, b=b: r["per_bld_viols"][b], ".0f")

    print(f"\n  --- Structural vs Agent-Caused ---")
    for b in range(n_b):
        row(f"Bld {b} structural", lambda r, b=b: r["per_bld_structural"][b], ".0f")
        row(f"Bld {b} agent-caused", lambda r, b=b: r["per_bld_agent_caused"][b], ".0f")

    print(f"\n{'='*W}")


# =========================================================================
# Main
# =========================================================================
AGENTS = [
    {
        "name": "Softmax",
        "ckpt": f"{PROJECT}/runs/r25b_softmax_ablation/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-04-15-01-37-26/torch_save/epoch-40.pt",
    },
    {
        "name": "GradS",
        "ckpt": f"{PROJECT}/runs/r25b_grads_ablation/5bld/PPOLagGradS-{{CityLearnSafety-V2G-v2}}/seed-042-2026-04-15-01-46-09/torch_save/epoch-40.pt",
    },
]


if __name__ == "__main__":
    all_rollouts = []
    all_analyses = []

    for agent in AGENTS:
        rollout = rollout_per_building(agent["name"], agent["ckpt"], seed=42)
        analysis = analyze_violations(rollout)
        all_rollouts.append(rollout)
        all_analyses.append(analysis)

    print_comparison(all_analyses)

    # Save per-building data for potential plotting later
    out_dir = os.path.join(PROJECT, "docs/temperature_case_study/ablation_eval")
    for rollout in all_rollouts:
        fname = f"c3_perbuilding_{rollout['name'].lower()}.npz"
        np.savez_compressed(
            os.path.join(out_dir, fname),
            building_power=rollout["building_power"],
            building_nsl=rollout["building_nsl"],
            building_batt_power=rollout["building_batt_power"],
            per_step_c3_cost=rollout["per_step_c3_cost"],
            per_step_c4_cost=rollout["per_step_c4_cost"],
            grid_import=rollout["grid_import"],
        )
        print(f"  Saved: {os.path.join(out_dir, fname)}")

    print("\nDone.")
