#!/usr/bin/env python3
"""
Deterministic evaluation: PPO-Lag (R25b, epoch 80) vs SAC-Lag (100ep, epoch 100).

Guarantees identical results across runs by:
  1. Fixed seeds at 3 levels (torch, numpy, gym env)
  2. Deterministic policy inference (mean actions, no sampling)
  3. Fresh env per model (no env-var leakage)
  4. All KPIs computed from deterministic arrays

Usage:
    conda activate citylearn
    python scripts/eval_ppo_vs_sac_deterministic.py
"""
from __future__ import annotations

import contextlib
import csv
import importlib
import io
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)

SCHEMA_PATH = os.path.join(
    PROJECT,
    "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
)
TOTAL_STEPS = 8759
SEED = 42

# Thresholds
DEFAULT_P_BUILDING_MAX = 4.6083
DEFAULT_P_GRID_MAX = 10.2352

# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
PPO_CKPT = os.path.join(
    PROJECT,
    "runs/r25b_report_stable/5bld/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-27-04-04-26/torch_save/epoch-80.pt",
)
SAC_CKPT = os.path.join(
    PROJECT,
    "runs/r25b_report_stable_sac_full_100ep_seed42/sac_epoch_100.pt",
)
SCRIPT_PATH = os.path.join(PROJECT, "run_r25b_report_stable.sh")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def suppress_stdout():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def log(msg: str = ""):
    print(msg, flush=True)


def parse_exports(script_path: str) -> dict[str, str]:
    exports = {}
    with open(script_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("export "):
                continue
            rest = line[len("export "):]
            eq_idx = rest.find("=")
            if eq_idx < 0:
                continue
            key = rest[:eq_idx].strip()
            val_raw = rest[eq_idx + 1:].strip()
            if val_raw.startswith('"'):
                end_q = val_raw.find('"', 1)
                if end_q > 0:
                    val_raw = val_raw[1:end_q]
            else:
                for sep in ["  #", " #", "\t#"]:
                    ci = val_raw.find(sep)
                    if ci >= 0:
                        val_raw = val_raw[:ci]
                val_raw = val_raw.strip().strip('"').strip("'")
            val_raw = val_raw.replace("$PROJECT", PROJECT)
            val_raw = val_raw.replace("${PYTHONPATH:-}", "")
            exports[key] = val_raw
    return exports


def get_citylearn_env(wrapper):
    cur = wrapper
    seen = set()
    for _ in range(40):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "buildings") and hasattr(cur, "time_step"):
            blds = getattr(cur, "buildings", None)
            if blds is not None and len(blds) > 0:
                return cur
        for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None


def get_action_names(env) -> list:
    _e = env
    for _ in range(40):
        act_names_raw = getattr(_e, "action_names", None)
        if act_names_raw is not None:
            if isinstance(act_names_raw, list) and len(act_names_raw) > 0:
                if isinstance(act_names_raw[0], list):
                    return [n for sub in act_names_raw for n in sub]
                return list(act_names_raw)
        nxt = None
        for attr in ("env", "base", "_env", "unwrapped", "raw_env"):
            candidate = getattr(_e, attr, None)
            if candidate is not None and candidate is not _e:
                nxt = candidate
                break
        if nxt is None:
            break
        _e = nxt
    return []


def normalize_obs(obs, norm_mean, norm_std, norm_clip):
    if norm_mean is None:
        return obs
    obs_flat = obs.ravel()
    n = min(len(obs_flat), len(norm_mean))
    normed = np.zeros_like(obs_flat)
    normed[:n] = np.clip(
        (obs_flat[:n] - norm_mean[:n]) / norm_std[:n],
        -norm_clip[:n], norm_clip[:n],
    )
    return normed


def build_env(env_vars: dict):
    """Build env from scratch with clean env var state."""
    # Clear ALL STEMS/CITYLEARN env vars first to prevent leakage
    keys_to_clear = [k for k in os.environ if k.startswith(("STEMS_", "CITYLEARN_", "COST_W_"))]
    for k in keys_to_clear:
        del os.environ[k]

    # Set this run's env vars
    for k, v in env_vars.items():
        os.environ[k] = v

    # Force schema
    os.environ["CITYLEARN_SCHEMA"] = SCHEMA_PATH
    os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
    os.environ["CITYLEARN_REWARD_TYPE"] = "stems"

    # Reload modules to pick up new env vars
    mods_to_reload = [m for name, m in sys.modules.items()
                      if m is not None and any(s in name for s in
                          ("safety_env", "make_env", "forecast_obs", "saute_ev",
                           "extractors"))
                      and "omni_env" not in name]
    for m in mods_to_reload:
        try:
            importlib.reload(m)
        except Exception:
            pass

    import citylearn_safe.omni_env_v2
    from citylearn_safe.omni_env_v2 import CityLearnCMDPv2
    env = CityLearnCMDPv2('CityLearnSafety-V2G-v2')
    return env


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class EvalMetrics:
    run_name: str = ""

    # C0: EV departure SoC
    c0_total_departures: int = 0
    c0_violated: int = 0
    c0_violation_pct: float = 0.0
    c0_mean_dep_soc: float = 0.0
    c0_mean_req_soc: float = 0.0
    c0_mean_deficit: float = 0.0

    # C1: (Saute handles — report saute budget remaining)
    c1_info: str = "N/A"

    # C2: Battery SoC bounds
    c2_violations: int = 0
    c2_total: int = 0
    c2_violation_pct: float = 0.0

    # C3: Building power
    c3_violations: int = 0
    c3_total: int = 0
    c3_violation_pct: float = 0.0

    # C4: Grid power
    c4_violations: int = 0
    c4_total: int = 0
    c4_violation_pct: float = 0.0

    # Battery behavior
    batt_charge_pct: float = 0.0
    batt_discharge_pct: float = 0.0
    batt_idle_pct: float = 0.0
    batt_solar_avg_action: float = 0.0
    batt_peak_avg_action: float = 0.0
    batt_daily_cycling: int = 0

    # EV behavior
    ev_charge_pct: float = 0.0
    ev_v2g_pct: float = 0.0
    ev_v2g_peak_pct: float = 0.0
    ev_total_v2g_kwh: float = 0.0

    # CityLearn KPIs
    total_import_kwh: float = 0.0
    total_export_kwh: float = 0.0
    peak_nec_kw: float = 0.0
    ramping_kwh: float = 0.0
    electricity_cost: float = 0.0
    load_factor: float = 0.0

    # Timing
    eval_seconds: float = 0.0
    steps_completed: int = 0


# ---------------------------------------------------------------------------
# Core evaluation (self-contained, no imports from eval_all_proper)
# ---------------------------------------------------------------------------
def evaluate(run_name: str, env, policy_fn, env_vars: dict) -> EvalMetrics:
    """Run full-year deterministic evaluation."""
    metrics = EvalMetrics(run_name=run_name)

    city = get_citylearn_env(env)
    if city is None:
        log(f"  ERROR: Could not find CityLearnEnv for {run_name}")
        return metrics

    n_buildings = len(city.buildings)
    act_names = get_action_names(env)
    batt_indices = [i for i, n in enumerate(act_names)
                    if str(n).strip().lower() == "electrical_storage"]
    ev_indices = [i for i, n in enumerate(act_names)
                  if "electric_vehicle" in str(n).lower()]

    P_building_max = float(env_vars.get("CITYLEARN_STEMS_P_BUILDING_MAX",
                                         str(DEFAULT_P_BUILDING_MAX)))
    P_grid_max = float(env_vars.get("CITYLEARN_STEMS_P_GRID_MAX",
                                     str(DEFAULT_P_GRID_MAX)))
    c3_controllable = env_vars.get("CITYLEARN_C3_CONTROLLABLE", "0") == "1"

    # Discover EV chargers
    charger_info = []
    for bi, b in enumerate(city.buildings):
        for ch in getattr(b, "electric_vehicle_chargers", []):
            cid = getattr(ch, "charger_id", f"charger_{bi}")
            charger_info.append((bi, ch, cid))

    # ---- Deterministic reset ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with suppress_stdout():
        obs_raw, info = env.reset(seed=SEED)
    obs = obs_raw.numpy() if isinstance(obs_raw, torch.Tensor) else np.asarray(obs_raw, dtype=np.float32)

    # Tracking variables
    departures = []
    nec_prev = 0.0
    ramping_total = 0.0
    total_import = 0.0
    total_export = 0.0
    electricity_cost_total = 0.0
    peak_nec = 0.0

    c2_violations = 0; c2_total = 0
    c3_violations = 0; c3_total = 0
    c4_violations = 0; c4_total = 0

    batt_charge_steps = 0; batt_discharge_steps = 0; batt_total_steps = 0
    batt_solar_actions = []; batt_peak_actions = []
    batt_daily_charge = defaultdict(bool)
    batt_daily_discharge = defaultdict(bool)

    ev_charge_steps = 0; ev_v2g_steps = 0
    ev_v2g_peak_steps = 0; ev_connected_steps = 0
    ev_v2g_peak_connected = 0; ev_total_v2g_kwh = 0.0

    t0 = time.time()

    for step_i in range(TOTAL_STEPS):
        action = policy_fn(obs)
        with suppress_stdout():
            step_result = env.step(action)
        if len(step_result) == 6:
            obs_raw, reward_raw, _cost, terminated_raw, truncated_raw, info = step_result
        else:
            obs_raw, reward_raw, terminated_raw, truncated_raw, info = step_result
        obs = obs_raw.numpy() if isinstance(obs_raw, torch.Tensor) else np.asarray(obs_raw, dtype=np.float32)
        terminated = bool(terminated_raw.item()) if isinstance(terminated_raw, torch.Tensor) else bool(terminated_raw)
        truncated = bool(truncated_raw.item()) if isinstance(truncated_raw, torch.Tensor) else bool(truncated_raw)

        t_after = int(getattr(city, "time_step", 0))
        hour = t_after % 24
        day = t_after // 24
        is_solar = 10 <= hour <= 15
        is_peak = 17 <= hour <= 21

        # Grid metrics from info dict (most reliable)
        step_nec_total = float(info.get("step_net_consumption_kwh", 0.0))
        grid_import = float(info.get("grid_import_kwh", max(0.0, step_nec_total)))
        grid_export_step = float(info.get("grid_export_kwh", max(0.0, -step_nec_total)))

        t_nec = max(0, t_after - 1)

        # ---- C3: Building power violations ----
        for bi, b in enumerate(city.buildings):
            c3_total += 1
            nec_arr = getattr(b, "net_electricity_consumption", None)
            if nec_arr is not None and hasattr(nec_arr, "__len__") and t_nec < len(nec_arr):
                nec_b = float(nec_arr[t_nec])
            else:
                nec_b = 0.0

            if c3_controllable:
                nsl_arr = getattr(b, "_Building__energy_to_non_shiftable_load", [])
                sg_arr = getattr(b, "_Building__solar_generation", [])
                nsl_i = float(nsl_arr[t_nec]) if hasattr(nsl_arr, "__len__") and t_nec < len(nsl_arr) else 0.0
                sg_i = float(sg_arr[t_nec]) if hasattr(sg_arr, "__len__") and t_nec < len(sg_arr) else 0.0
                exog = nsl_i + sg_i
                if abs(exog) <= P_building_max:
                    if abs(nec_b) > P_building_max:
                        c3_violations += 1
                else:
                    if abs(nec_b) > abs(exog):
                        c3_violations += 1
            else:
                if abs(nec_b) > P_building_max:
                    c3_violations += 1

        # ---- C4: Grid power violation ----
        c4_total += 1
        if grid_import > P_grid_max:
            c4_violations += 1

        total_import += grid_import
        total_export += grid_export_step
        peak_nec = max(peak_nec, abs(step_nec_total))
        ramping_total += abs(step_nec_total - nec_prev)
        nec_prev = step_nec_total

        # Electricity cost
        price = 0.0
        for b in city.buildings:
            pricing_obj = getattr(b, "pricing", None)
            if pricing_obj is not None:
                pa = getattr(pricing_obj, "electricity_pricing", None)
                if pa is not None and hasattr(pa, "__len__") and t_nec < len(pa):
                    price = float(pa[t_nec])
                    break
        if grid_import > 0:
            electricity_cost_total += grid_import * price

        # ---- C2: Battery SoC violations (FIXED — use b.electrical_storage) ----
        for bi_idx in range(min(len(batt_indices), n_buildings)):
            bld = city.buildings[bi_idx]
            es = getattr(bld, "electrical_storage", None)
            if es is None:
                continue
            soc_arr = getattr(es, "soc", None)
            if soc_arr is not None and hasattr(soc_arr, "__len__") and t_nec < len(soc_arr):
                soc_val = float(soc_arr[t_nec])
                c2_total += 1
                if soc_val > 0.95 or soc_val < 0.0:
                    c2_violations += 1

        # ---- Battery action stats ----
        for bi in batt_indices:
            if bi < len(action):
                a = float(action[bi])
                batt_total_steps += 1
                if a > 0.1:
                    batt_charge_steps += 1
                    batt_daily_charge[day] = True
                elif a < -0.1:
                    batt_discharge_steps += 1
                    batt_daily_discharge[day] = True
                if is_solar:
                    batt_solar_actions.append(a)
                if is_peak:
                    batt_peak_actions.append(a)

        # ---- EV action stats ----
        for ei in ev_indices:
            if ei < len(action):
                a = float(action[ei])
                ev_connected_steps += 1
                if a > 0.1:
                    ev_charge_steps += 1
                elif a < -0.1:
                    ev_v2g_steps += 1
                    ev_total_v2g_kwh += abs(a) * 6.0
                    if 17 <= hour <= 23:
                        ev_v2g_peak_steps += 1
                if 17 <= hour <= 23:
                    ev_v2g_peak_connected += 1

        # ---- C0: Track EV departures ----
        t_soc = max(0, t_after - 1)
        for bi, ch, cid in charger_info:
            sim = getattr(ch, "charger_simulation", None)
            if sim is None:
                continue
            state_arr = getattr(sim, "_electric_vehicle_charger_state", None)
            dep_time_arr = getattr(sim, "_electric_vehicle_departure_time", None)
            req_soc_arr = getattr(sim, "_electric_vehicle_required_soc_departure", None)
            if state_arr is None or dep_time_arr is None or req_soc_arr is None:
                continue
            if t_after >= len(state_arr):
                continue
            s = float(state_arr[t_after])
            d = float(dep_time_arr[t_after])
            r = float(req_soc_arr[t_after])
            if s == 1.0 and d == 0.0:
                ev_obj = getattr(ch, "connected_electric_vehicle", None)
                actual_soc = 0.0
                if ev_obj is not None:
                    batt = getattr(ev_obj, "battery", None)
                    if batt is not None:
                        soc_series = getattr(batt, "soc", None)
                        if soc_series is not None and hasattr(soc_series, "__len__"):
                            if 0 <= t_soc < len(soc_series):
                                actual_soc = float(np.clip(soc_series[t_soc], 0.0, 1.0))
                departures.append((cid, actual_soc, r))

        if terminated or truncated:
            break

        if (step_i + 1) % 2000 == 0:
            dep_so_far = len(departures)
            viol_so_far = sum(1 for _, ds, dr in departures if ds < dr)
            log(f"    Step {step_i+1}/{TOTAL_STEPS}: deps={dep_so_far}, "
                f"c0_viol={viol_so_far}, c3={c3_violations}/{c3_total}, c4={c4_violations}/{c4_total}")

    elapsed = time.time() - t0
    metrics.eval_seconds = elapsed
    metrics.steps_completed = step_i + 1

    # ---- Final C0 ----
    n_dep = len(departures)
    n_viol = sum(1 for _, ds, dr in departures if ds < dr)
    metrics.c0_total_departures = n_dep
    metrics.c0_violated = n_viol
    metrics.c0_violation_pct = 100.0 * n_viol / max(n_dep, 1)
    metrics.c0_mean_dep_soc = float(np.mean([ds for _, ds, _ in departures])) if departures else 0.0
    metrics.c0_mean_req_soc = float(np.mean([dr for _, _, dr in departures])) if departures else 0.0
    violated_deficits = [max(0, dr - ds) for _, ds, dr in departures if ds < dr]
    metrics.c0_mean_deficit = float(np.mean(violated_deficits)) if violated_deficits else 0.0

    # ---- Final C2/C3/C4 ----
    metrics.c2_violations = c2_violations
    metrics.c2_total = c2_total
    metrics.c2_violation_pct = 100.0 * c2_violations / max(c2_total, 1)
    metrics.c3_violations = c3_violations
    metrics.c3_total = c3_total
    metrics.c3_violation_pct = 100.0 * c3_violations / max(c3_total, 1)
    metrics.c4_violations = c4_violations
    metrics.c4_total = c4_total
    metrics.c4_violation_pct = 100.0 * c4_violations / max(c4_total, 1)

    # ---- Battery ----
    bt = max(batt_total_steps, 1)
    metrics.batt_charge_pct = 100.0 * batt_charge_steps / bt
    metrics.batt_discharge_pct = 100.0 * batt_discharge_steps / bt
    metrics.batt_idle_pct = 100.0 * (bt - batt_charge_steps - batt_discharge_steps) / bt
    metrics.batt_solar_avg_action = float(np.mean(batt_solar_actions)) if batt_solar_actions else 0.0
    metrics.batt_peak_avg_action = float(np.mean(batt_peak_actions)) if batt_peak_actions else 0.0
    all_days = set(list(batt_daily_charge.keys()) + list(batt_daily_discharge.keys()))
    metrics.batt_daily_cycling = sum(
        1 for d in all_days
        if batt_daily_charge.get(d, False) and batt_daily_discharge.get(d, False)
    )

    # ---- EV ----
    et = max(ev_connected_steps, 1)
    metrics.ev_charge_pct = 100.0 * ev_charge_steps / et
    metrics.ev_v2g_pct = 100.0 * ev_v2g_steps / et
    metrics.ev_v2g_peak_pct = 100.0 * ev_v2g_peak_steps / max(ev_v2g_peak_connected, 1)
    metrics.ev_total_v2g_kwh = ev_total_v2g_kwh

    # ---- CityLearn KPIs ----
    metrics.total_import_kwh = total_import
    metrics.total_export_kwh = total_export
    metrics.peak_nec_kw = peak_nec
    metrics.ramping_kwh = ramping_total
    metrics.electricity_cost = electricity_cost_total
    mean_import = total_import / max(metrics.steps_completed, 1)
    metrics.load_factor = mean_import / max(peak_nec, 1e-8)

    return metrics


# ---------------------------------------------------------------------------
# Actor loaders
# ---------------------------------------------------------------------------
def _load_normalizer(ckpt):
    nd = ckpt.get("obs_normalizer")
    if nd is None:
        return None, None, None
    mean = nd["_mean"].numpy()
    std = np.maximum(nd["_std"].numpy(), 1e-8)
    clip = nd["_clip"].numpy()
    return mean, std, clip


def load_ppo_actor(ckpt_path: str, obs_dim: int = 199, act_dim: int = 9):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi = ckpt["pi"]
    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, act_dim),
    )
    key_map = {
        "mean.0.weight": "0.weight", "mean.0.bias": "0.bias",
        "mean.2.weight": "2.weight", "mean.2.bias": "2.bias",
        "mean.4.weight": "4.weight", "mean.4.bias": "4.bias",
    }
    sd = {seq_k: pi[omni_k] for omni_k, seq_k in key_map.items()}
    actor.load_state_dict(sd)
    actor.eval()
    norm_mean, norm_std, norm_clip = _load_normalizer(ckpt)
    return actor, norm_mean, norm_std, norm_clip


def load_sac_actor(ckpt_path: str, obs_dim: int = 199, act_dim: int = 9):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi = ckpt["pi"]
    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, act_dim * 2),
    )
    key_map = {
        "net.0.weight": "0.weight", "net.0.bias": "0.bias",
        "net.2.weight": "2.weight", "net.2.bias": "2.bias",
        "net.4.weight": "4.weight", "net.4.bias": "4.bias",
    }
    sd = {seq_k: pi[omni_k] for omni_k, seq_k in key_map.items()}
    actor.load_state_dict(sd)
    actor.eval()
    norm_mean, norm_std, norm_clip = _load_normalizer(ckpt)
    return actor, norm_mean, norm_std, norm_clip


def make_ppo_policy(actor, norm_mean, norm_std, norm_clip):
    def policy(obs):
        obs_n = normalize_obs(obs, norm_mean, norm_std, norm_clip)
        t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = actor(t)
            return torch.tanh(out).squeeze(0).numpy()
    return policy


def make_sac_policy(actor, norm_mean, norm_std, norm_clip):
    def policy(obs):
        obs_n = normalize_obs(obs, norm_mean, norm_std, norm_clip)
        t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = actor(t).squeeze(0)
            mean = out[:9]
            return torch.tanh(mean).numpy()
    return policy


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_comparison(ppo: EvalMetrics, sac: EvalMetrics):
    def row(label, pv, sv, fmt=".1f", unit=""):
        log(f"  {label:<35} {pv:{fmt}}{unit:>6} {sv:{fmt}}{unit:>6}")

    def row_int(label, pv, sv, unit=""):
        log(f"  {label:<35} {str(pv)+unit:>12} {str(sv)+unit:>12}")

    log()
    log("=" * 72)
    log("  PPO-Lag (ep80) vs SAC-Lag (ep100)  |  Deterministic Evaluation")
    log("=" * 72)
    log(f"  {'Metric':<35} {'PPO-Lag':>12} {'SAC-Lag':>12}")
    log(f"  {'-'*35} {'-'*12} {'-'*12}")

    log()
    log("  ---- Constraint Violations ----")
    row_int("C0 total departures", ppo.c0_total_departures, sac.c0_total_departures)
    row_int("C0 violated departures", ppo.c0_violated, sac.c0_violated)
    row("C0 violation %", ppo.c0_violation_pct, sac.c0_violation_pct, unit="%")
    row("C0 mean departure SoC", ppo.c0_mean_dep_soc, sac.c0_mean_dep_soc, fmt=".3f")
    row("C0 mean required SoC", ppo.c0_mean_req_soc, sac.c0_mean_req_soc, fmt=".3f")
    row("C0 mean deficit (violated)", ppo.c0_mean_deficit, sac.c0_mean_deficit, fmt=".3f")
    log()
    row_int("C2 violations / total", f"{ppo.c2_violations}/{ppo.c2_total}", f"{sac.c2_violations}/{sac.c2_total}")
    row("C2 violation %", ppo.c2_violation_pct, sac.c2_violation_pct, unit="%")
    log()
    row_int("C3 violations / total", f"{ppo.c3_violations}/{ppo.c3_total}", f"{sac.c3_violations}/{sac.c3_total}")
    row("C3 violation %", ppo.c3_violation_pct, sac.c3_violation_pct, unit="%")
    log()
    row_int("C4 violations / total", f"{ppo.c4_violations}/{ppo.c4_total}", f"{sac.c4_violations}/{sac.c4_total}")
    row("C4 violation %", ppo.c4_violation_pct, sac.c4_violation_pct, unit="%")

    log()
    log("  ---- Battery Behavior ----")
    row("Charge %", ppo.batt_charge_pct, sac.batt_charge_pct, unit="%")
    row("Discharge %", ppo.batt_discharge_pct, sac.batt_discharge_pct, unit="%")
    row("Idle %", ppo.batt_idle_pct, sac.batt_idle_pct, unit="%")
    row("Solar-hour avg action", ppo.batt_solar_avg_action, sac.batt_solar_avg_action, fmt=".3f")
    row("Peak-hour avg action", ppo.batt_peak_avg_action, sac.batt_peak_avg_action, fmt=".3f")
    row_int("Daily cycling days", ppo.batt_daily_cycling, sac.batt_daily_cycling)

    log()
    log("  ---- EV / V2G Behavior ----")
    row("EV charge %", ppo.ev_charge_pct, sac.ev_charge_pct, unit="%")
    row("EV V2G discharge %", ppo.ev_v2g_pct, sac.ev_v2g_pct, unit="%")
    row("V2G during peak %", ppo.ev_v2g_peak_pct, sac.ev_v2g_peak_pct, unit="%")
    row("Total V2G energy (kWh)", ppo.ev_total_v2g_kwh, sac.ev_total_v2g_kwh, fmt=".0f")

    log()
    log("  ---- CityLearn KPIs ----")
    row("Total import (kWh)", ppo.total_import_kwh, sac.total_import_kwh, fmt=".0f")
    row("Total export (kWh)", ppo.total_export_kwh, sac.total_export_kwh, fmt=".0f")
    row("Peak NEC (kW)", ppo.peak_nec_kw, sac.peak_nec_kw, fmt=".2f")
    row("Ramping (kWh)", ppo.ramping_kwh, sac.ramping_kwh, fmt=".0f")
    row("Electricity cost ($)", ppo.electricity_cost, sac.electricity_cost, fmt=".0f")
    row("Load factor", ppo.load_factor, sac.load_factor, fmt=".4f")

    log()
    row_int("Steps completed", ppo.steps_completed, sac.steps_completed)
    row("Eval time (s)", ppo.eval_seconds, sac.eval_seconds, fmt=".1f")
    log("=" * 72)


def save_csv(ppo: EvalMetrics, sac: EvalMetrics, path: str):
    fields = ["metric", "PPO-Lag (ep80)", "SAC-Lag (ep100)"]
    rows = [
        ("c0_total_departures", ppo.c0_total_departures, sac.c0_total_departures),
        ("c0_violated", ppo.c0_violated, sac.c0_violated),
        ("c0_violation_pct", f"{ppo.c0_violation_pct:.2f}", f"{sac.c0_violation_pct:.2f}"),
        ("c0_mean_dep_soc", f"{ppo.c0_mean_dep_soc:.4f}", f"{sac.c0_mean_dep_soc:.4f}"),
        ("c0_mean_req_soc", f"{ppo.c0_mean_req_soc:.4f}", f"{sac.c0_mean_req_soc:.4f}"),
        ("c0_mean_deficit", f"{ppo.c0_mean_deficit:.4f}", f"{sac.c0_mean_deficit:.4f}"),
        ("c2_violations", ppo.c2_violations, sac.c2_violations),
        ("c2_total", ppo.c2_total, sac.c2_total),
        ("c2_violation_pct", f"{ppo.c2_violation_pct:.2f}", f"{sac.c2_violation_pct:.2f}"),
        ("c3_violations", ppo.c3_violations, sac.c3_violations),
        ("c3_total", ppo.c3_total, sac.c3_total),
        ("c3_violation_pct", f"{ppo.c3_violation_pct:.2f}", f"{sac.c3_violation_pct:.2f}"),
        ("c4_violations", ppo.c4_violations, sac.c4_violations),
        ("c4_total", ppo.c4_total, sac.c4_total),
        ("c4_violation_pct", f"{ppo.c4_violation_pct:.2f}", f"{sac.c4_violation_pct:.2f}"),
        ("batt_charge_pct", f"{ppo.batt_charge_pct:.2f}", f"{sac.batt_charge_pct:.2f}"),
        ("batt_discharge_pct", f"{ppo.batt_discharge_pct:.2f}", f"{sac.batt_discharge_pct:.2f}"),
        ("batt_idle_pct", f"{ppo.batt_idle_pct:.2f}", f"{sac.batt_idle_pct:.2f}"),
        ("batt_solar_avg_action", f"{ppo.batt_solar_avg_action:.4f}", f"{sac.batt_solar_avg_action:.4f}"),
        ("batt_peak_avg_action", f"{ppo.batt_peak_avg_action:.4f}", f"{sac.batt_peak_avg_action:.4f}"),
        ("batt_daily_cycling", ppo.batt_daily_cycling, sac.batt_daily_cycling),
        ("ev_charge_pct", f"{ppo.ev_charge_pct:.2f}", f"{sac.ev_charge_pct:.2f}"),
        ("ev_v2g_pct", f"{ppo.ev_v2g_pct:.2f}", f"{sac.ev_v2g_pct:.2f}"),
        ("ev_v2g_peak_pct", f"{ppo.ev_v2g_peak_pct:.2f}", f"{sac.ev_v2g_peak_pct:.2f}"),
        ("ev_total_v2g_kwh", f"{ppo.ev_total_v2g_kwh:.1f}", f"{sac.ev_total_v2g_kwh:.1f}"),
        ("total_import_kwh", f"{ppo.total_import_kwh:.1f}", f"{sac.total_import_kwh:.1f}"),
        ("total_export_kwh", f"{ppo.total_export_kwh:.1f}", f"{sac.total_export_kwh:.1f}"),
        ("peak_nec_kw", f"{ppo.peak_nec_kw:.3f}", f"{sac.peak_nec_kw:.3f}"),
        ("ramping_kwh", f"{ppo.ramping_kwh:.1f}", f"{sac.ramping_kwh:.1f}"),
        ("electricity_cost", f"{ppo.electricity_cost:.1f}", f"{sac.electricity_cost:.1f}"),
        ("load_factor", f"{ppo.load_factor:.5f}", f"{sac.load_factor:.5f}"),
        ("steps_completed", ppo.steps_completed, sac.steps_completed),
        ("eval_seconds", f"{ppo.eval_seconds:.1f}", f"{sac.eval_seconds:.1f}"),
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow(r)
    log(f"\nCSV saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 72)
    log("  Deterministic PPO-Lag vs SAC-Lag Evaluation")
    log(f"  Schema: {SCHEMA_PATH}")
    log(f"  Seed: {SEED} | Steps: {TOTAL_STEPS}")
    log(f"  PPO: epoch-80 | SAC: epoch-100")
    log("=" * 72)
    log()

    for name, path in [("PPO", PPO_CKPT), ("SAC", SAC_CKPT)]:
        if not os.path.exists(path):
            log(f"ERROR: {name} checkpoint not found: {path}")
            sys.exit(1)
        log(f"  {name}: {path}")
    log()

    env_vars = parse_exports(SCRIPT_PATH)

    # --- PPO ---
    log(">>> Evaluating PPO-Lag (epoch 80) ...")
    with suppress_stdout():
        env_ppo = build_env(env_vars)
    obs_dim = env_ppo.observation_space.shape[0]
    act_dim = env_ppo.action_space.shape[0]
    log(f"  Env: obs_dim={obs_dim}, act_dim={act_dim}")

    actor_ppo, nm, ns, nc = load_ppo_actor(PPO_CKPT, obs_dim, act_dim)
    log(f"  Actor loaded, normalizer={'YES' if nm is not None else 'NO'}")
    policy_ppo = make_ppo_policy(actor_ppo, nm, ns, nc)
    result_ppo = evaluate("PPO-Lag_ep80", env_ppo, policy_ppo, env_vars)
    log(f"  Done: C0={result_ppo.c0_violation_pct:.1f}%, C3={result_ppo.c3_violation_pct:.1f}%, "
        f"C4={result_ppo.c4_violation_pct:.1f}% in {result_ppo.eval_seconds:.0f}s")
    log()

    # --- SAC ---
    log(">>> Evaluating SAC-Lag (epoch 100) ...")
    with suppress_stdout():
        env_sac = build_env(env_vars)
    obs_dim_s = env_sac.observation_space.shape[0]
    act_dim_s = env_sac.action_space.shape[0]
    log(f"  Env: obs_dim={obs_dim_s}, act_dim={act_dim_s}")

    actor_sac, nm_s, ns_s, nc_s = load_sac_actor(SAC_CKPT, obs_dim_s, act_dim_s)
    log(f"  Actor loaded, normalizer={'YES' if nm_s is not None else 'NO'}")
    policy_sac = make_sac_policy(actor_sac, nm_s, ns_s, nc_s)
    result_sac = evaluate("SAC-Lag_ep100", env_sac, policy_sac, env_vars)
    log(f"  Done: C0={result_sac.c0_violation_pct:.1f}%, C3={result_sac.c3_violation_pct:.1f}%, "
        f"C4={result_sac.c4_violation_pct:.1f}% in {result_sac.eval_seconds:.0f}s")
    log()

    # --- Output ---
    print_comparison(result_ppo, result_sac)
    csv_path = os.path.join(PROJECT, "docs", "ppo_vs_sac_deterministic.csv")
    save_csv(result_ppo, result_sac, csv_path)

    # --- Determinism check hash ---
    import hashlib
    vals = (
        result_ppo.c0_violation_pct, result_ppo.c3_violation_pct, result_ppo.c4_violation_pct,
        result_ppo.total_import_kwh, result_ppo.peak_nec_kw,
        result_sac.c0_violation_pct, result_sac.c3_violation_pct, result_sac.c4_violation_pct,
        result_sac.total_import_kwh, result_sac.peak_nec_kw,
    )
    h = hashlib.md5(str(vals).encode()).hexdigest()[:12]
    log(f"\nDeterminism hash: {h}")
    log("(If this hash differs between runs, something is non-deterministic)")
    log("\nDone.")


if __name__ == "__main__":
    main()
