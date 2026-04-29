#!/usr/bin/env python3
"""
Comprehensive evaluation of ALL 5-building PPOLag runs.

For each run:
  1. Parses env vars from its ORIGINAL run script (.sh)
  2. Builds the EXACT wrapper chain (Base -> Safety -> Forecast -> Saute [-> ActionMask])
  3. Loads checkpoint + obs normalizer
  4. Runs full year (8759 steps) deterministically
  5. Computes all metrics: C0/C2/C3/C4 violations, battery/EV behavior, CityLearn KPIs

Also evaluates three baselines (no checkpoint): Zero, Greedy EV, SmartV2GRBC.

Results saved to docs/master_eval_table.md
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@contextlib.contextmanager
def suppress_stdout():
    """Temporarily suppress stdout (for noisy env debug prints)."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def log(msg: str = ""):
    """Print and flush immediately."""
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)

SCHEMA_PATH = os.path.join(
    PROJECT,
    "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
)
TOTAL_STEPS = 8759
SEED = 42

# Power thresholds (defaults, overridden per-run from env vars)
DEFAULT_P_BUILDING_MAX = 4.6083
DEFAULT_P_GRID_MAX = 10.2352

# Run definitions: (name, script_filename, run_dir_pattern, epoch_override)
# epoch_override=None means "use latest"
RUN_DEFS = [
    ("r19_ablation",     "run_r19_ablation.sh",     "runs/r19_ablation",     80),
    ("r21_ppo",          "run_r21_ppo.sh",          "runs/r21_ppo",          89),
    ("r22_ppo",          "run_r22_ppo.sh",          "runs/r22_ppo",          95),
    ("r23_ppo",          "run_r23_ppo.sh",          "runs/r23_ppo",          89),
    ("r24_solar_store",  "run_r24_solar_store.sh",  "runs/r24_solar_store",  80),
    ("r25a_batt_solar",  "run_r25a_batt_solar.sh",  "runs/r25a_batt_solar",  80),
    ("r25b_ev_slack_arb","run_r25b_ev_slack_arb.sh","runs/r25b_ev_slack_arb",80),
    ("r25c_v2g_reduce",  "run_r25c_v2g_reduce.sh",  "runs/r25c_v2g_reduce",  80),
    ("r26_headroom",     "run_r26_headroom.sh",     "runs/r26_headroom",     80),
    ("r27_price_arb",    "run_r27_price_arb.sh",    "runs/r27_price_arb",    80),
    ("r28_cycling",      "run_r28_cycling.sh",      "runs/r28_cycling",      80),
    ("r28b_bc_nocurr",   "run_r28b_bc_nocurr.sh",   "runs/r28b_bc_nocurr",   80),
    ("r29_mask_simple",  "run_r29_mask_simple.sh",  "runs/r29_mask_simple",  95),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_exports(script_path: str) -> dict[str, str]:
    """Extract all 'export KEY=VALUE' from a bash script."""
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


def find_checkpoint(run_dir: str, epoch: Optional[int] = None) -> Optional[str]:
    """Find checkpoint .pt file in run_dir (searches subdirectories)."""
    import glob
    pattern = os.path.join(run_dir, "**", "torch_save", "epoch-*.pt")
    pts = glob.glob(pattern, recursive=True)
    if not pts:
        return None
    if epoch is not None:
        target = f"epoch-{epoch}.pt"
        matches = [p for p in pts if p.endswith(target)]
        if matches:
            return matches[0]
    # Fall back to latest epoch
    def epoch_num(path):
        base = os.path.basename(path)
        return int(base.replace("epoch-", "").replace(".pt", ""))
    pts.sort(key=epoch_num)
    return pts[-1]


def get_citylearn_env(wrapper):
    """Walk wrapper chain to find the CityLearnEnv."""
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
    """Walk wrapper chain to find action_names."""
    _e = env
    for _ in range(20):
        act_names_raw = getattr(_e, "action_names", None)
        if act_names_raw is not None:
            break
        _e = getattr(_e, "env", getattr(_e, "base", None))
        if _e is None:
            break
    if act_names_raw is None:
        return []
    if isinstance(act_names_raw, list) and len(act_names_raw) > 0 and isinstance(act_names_raw[0], list):
        return [n for sub in act_names_raw for n in sub]
    return list(act_names_raw)


def build_env(env_vars: dict, use_action_mask: bool = False):
    """Build CityLearnCMDPv2 env with the given env vars (matches training exactly)."""
    import importlib

    # Set env vars
    for k, v in env_vars.items():
        os.environ[k] = v

    # Force schema
    os.environ["CITYLEARN_SCHEMA"] = SCHEMA_PATH
    os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
    os.environ["CITYLEARN_REWARD_TYPE"] = "stems"

    if use_action_mask:
        os.environ["CITYLEARN_ACTION_MASK"] = "1"

    # Re-import to pick up new env vars.
    # Do NOT reload omni_env_v2 — OmniSafe's @env_register raises on re-register.
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

    # Use CityLearnCMDPv2 — same env class as training.
    import citylearn_safe.omni_env_v2  # noqa: registers once
    from citylearn_safe.omni_env_v2 import CityLearnCMDPv2
    env = CityLearnCMDPv2('CityLearnSafety-V2G-v2')
    return env


def load_actor(ckpt_path: str, obs_dim: int, act_dim: int):
    """Load actor MLP [256,256] from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi_state = ckpt["pi"]

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
    actor_sd = {}
    for omnisafe_key, seq_key in key_map.items():
        if omnisafe_key not in pi_state:
            raise KeyError(f"Missing key {omnisafe_key} in checkpoint")
        actor_sd[seq_key] = pi_state[omnisafe_key]
    actor.load_state_dict(actor_sd)
    actor.eval()

    # Obs normalizer
    norm_data = ckpt.get("obs_normalizer")
    norm_mean = norm_std = norm_clip = None
    if norm_data is not None:
        norm_mean = norm_data["_mean"].numpy()
        norm_std = np.maximum(norm_data["_std"].numpy(), 1e-8)
        norm_clip = norm_data["_clip"].numpy()

    return actor, norm_mean, norm_std, norm_clip


def normalize_obs(obs, norm_mean, norm_std, norm_clip):
    """Apply OmniSafe-style obs normalization."""
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


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------
@dataclass
class EvalMetrics:
    run_name: str = ""

    # Run metadata
    saute: bool = False
    action_mask: bool = False
    batt_clamp: bool = False
    bc_warmstart: bool = False
    pid_lagrange: bool = False
    curriculum: bool = True  # default ON

    # Key reward weights
    lambda_ev: float = 0.0
    ev_guard: float = 0.0
    v2g_context: float = 0.0
    load_shift: float = 0.0
    price_arb: float = 0.0
    solar_store: float = 0.0
    headroom: float = 0.0
    ev_solar: float = 0.0
    ev_slack_arb: float = 0.0
    grid_penalty: float = 0.0

    # C0: EV departure
    c0_total_departures: int = 0
    c0_violated: int = 0
    c0_violation_pct: float = 0.0
    c0_mean_dep_soc: float = 0.0
    c0_mean_req_soc: float = 0.0
    c0_mean_deficit: float = 0.0

    # C2: Battery SoC
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

    # Battery metrics
    batt_charge_pct: float = 0.0
    batt_discharge_pct: float = 0.0
    batt_idle_pct: float = 0.0
    batt_solar_avg_action: float = 0.0
    batt_peak_avg_action: float = 0.0
    batt_daily_cycling: float = 0.0

    # EV metrics
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


def extract_metadata(env_vars: dict, script_path: str) -> dict:
    """Extract run metadata from env vars and script content."""
    meta = {}
    meta["saute"] = env_vars.get("CITYLEARN_EV_SAUTE", "0") == "1"
    meta["action_mask"] = env_vars.get("CITYLEARN_ACTION_MASK", "0") == "1"
    meta["batt_clamp"] = env_vars.get("CITYLEARN_BATT_CLAMP", "0") == "1"
    meta["pid_lagrange"] = env_vars.get("CITYLEARN_PID_LAGRANGE", "0") == "1"

    # BC warmstart: check if --bc in the python command line
    with open(script_path) as f:
        content = f.read()
    meta["bc_warmstart"] = "--bc" in content
    meta["curriculum"] = "--no-curriculum" not in content

    # Reward weights
    meta["lambda_ev"] = float(env_vars.get("STEMS_LAMBDA_EV", "0"))
    meta["ev_guard"] = float(env_vars.get("STEMS_ALPHA_EV_GUARD", "0"))
    meta["v2g_context"] = float(env_vars.get("STEMS_ALPHA_V2G_CONTEXT", "0"))
    meta["load_shift"] = float(env_vars.get("STEMS_ALPHA_LOAD_SHIFT", "0"))
    meta["price_arb"] = float(env_vars.get("STEMS_ALPHA_PRICE_ARB", "0"))
    meta["solar_store"] = float(env_vars.get("STEMS_ALPHA_SOLAR_STORE", "0"))
    meta["headroom"] = float(env_vars.get("STEMS_ALPHA_HEADROOM", "0"))
    meta["ev_solar"] = float(env_vars.get("STEMS_ALPHA_EV_SOLAR", "0"))
    meta["ev_slack_arb"] = float(env_vars.get("STEMS_EV_SLACK_ARB_SCALE", "0"))
    meta["grid_penalty"] = float(env_vars.get("STEMS_ALPHA_GRID_PENALTY", "0"))
    return meta


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------
def evaluate_run(
    run_name: str,
    env,
    get_action_fn,  # callable(obs) -> np.ndarray
    env_vars: dict,
    metadata: dict,
) -> EvalMetrics:
    """Run full-year eval and compute all metrics."""

    metrics = EvalMetrics(run_name=run_name)

    # Copy metadata
    for k, v in metadata.items():
        if hasattr(metrics, k):
            setattr(metrics, k, v)

    city = get_citylearn_env(env)
    if city is None:
        log(f"  ERROR: Could not find CityLearnEnv for {run_name}")
        return metrics

    n_buildings = len(city.buildings)
    act_names = get_action_names(env)
    batt_indices = [i for i, n in enumerate(act_names) if str(n).strip().lower() == "electrical_storage"]
    ev_indices = [i for i, n in enumerate(act_names) if "electric_vehicle" in str(n).lower()]

    P_building_max = float(env_vars.get("CITYLEARN_STEMS_P_BUILDING_MAX", str(DEFAULT_P_BUILDING_MAX)))
    P_grid_max = float(env_vars.get("CITYLEARN_STEMS_P_GRID_MAX", str(DEFAULT_P_GRID_MAX)))
    c3_controllable = env_vars.get("CITYLEARN_C3_CONTROLLABLE", "0") == "1"

    # Discover chargers
    charger_info = []
    for bi, b in enumerate(city.buildings):
        for ch in getattr(b, "electric_vehicle_chargers", []):
            cid = getattr(ch, "charger_id", f"charger_{bi}")
            charger_info.append((bi, ch, cid))

    # Reset env (suppress noisy debug)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with suppress_stdout():
        obs_raw, info = env.reset(seed=SEED)
    obs = obs_raw.numpy() if isinstance(obs_raw, torch.Tensor) else np.asarray(obs_raw, dtype=np.float32)

    # Tracking
    departures = []
    nec_per_step = []  # total grid NEC each step
    nec_prev = 0.0
    ramping_total = 0.0
    total_import = 0.0
    total_export = 0.0
    electricity_cost_total = 0.0
    peak_nec = 0.0

    c2_violations = 0
    c2_total = 0
    c3_violations = 0
    c3_total = 0
    c4_violations = 0
    c4_total = 0

    batt_charge_steps = 0
    batt_discharge_steps = 0
    batt_total_steps = 0
    batt_solar_actions = []  # actions during solar hours
    batt_peak_actions = []   # actions during peak hours
    batt_daily_charge = defaultdict(bool)
    batt_daily_discharge = defaultdict(bool)

    ev_charge_steps = 0
    ev_v2g_steps = 0
    ev_v2g_peak_steps = 0
    ev_connected_steps = 0
    ev_v2g_peak_connected = 0
    ev_total_v2g_kwh = 0.0

    t0 = time.time()

    for step_i in range(TOTAL_STEPS):
        action = get_action_fn(obs)
        with suppress_stdout():
            step_result = env.step(action)
        if len(step_result) == 6:
            obs_raw, reward_raw, _cost, terminated_raw, truncated_raw, info = step_result
        else:
            obs_raw, reward_raw, terminated_raw, truncated_raw, info = step_result
        obs = obs_raw.numpy() if isinstance(obs_raw, torch.Tensor) else np.asarray(obs_raw, dtype=np.float32)
        reward = float(reward_raw.item()) if isinstance(reward_raw, torch.Tensor) else float(reward_raw)
        terminated = bool(terminated_raw.item()) if isinstance(terminated_raw, torch.Tensor) else bool(terminated_raw)
        truncated = bool(truncated_raw.item()) if isinstance(truncated_raw, torch.Tensor) else bool(truncated_raw)

        t_after = int(getattr(city, "time_step", 0))
        hour = t_after % 24
        day = t_after // 24
        is_solar = 10 <= hour <= 15
        is_peak = 17 <= hour <= 21

        # --- Use info dict for grid-level metrics (most reliable) ---
        step_nec_total = float(info.get("step_net_consumption_kwh", 0.0))
        grid_import = float(info.get("grid_import_kwh", max(0.0, step_nec_total)))
        grid_export_step = float(info.get("grid_export_kwh", max(0.0, -step_nec_total)))

        # --- NEC per building (use t_after-1 since CityLearn writes NEC there) ---
        t_nec = max(0, t_after - 1)
        for bi, b in enumerate(city.buildings):
            c3_total += 1
            nec_arr = getattr(b, "net_electricity_consumption", None)
            if nec_arr is not None and hasattr(nec_arr, "__len__") and t_nec < len(nec_arr):
                nec_b = float(nec_arr[t_nec])
            else:
                nec_b = 0.0

            # C3: building power violation
            if c3_controllable:
                nsl_arr = getattr(b, "_Building__energy_to_non_shiftable_load", [])
                sg_arr = getattr(b, "_Building__solar_generation", [])
                nsl_i = 0.0
                if hasattr(nsl_arr, "__len__") and t_nec < len(nsl_arr):
                    nsl_i = float(nsl_arr[t_nec])
                sg_i = 0.0
                if hasattr(sg_arr, "__len__") and t_nec < len(sg_arr):
                    sg_i = float(sg_arr[t_nec])
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

        # C4: grid power (use total grid import from info)
        c4_total += 1
        if grid_import > P_grid_max:
            c4_violations += 1

        total_import += grid_import
        total_export += grid_export_step
        peak_nec = max(peak_nec, abs(step_nec_total))

        # Ramping
        ramping_total += abs(step_nec_total - nec_prev)
        nec_prev = step_nec_total

        # Electricity cost (NEC * price)
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

        # C2: Battery SoC violations (use t_nec = t_after - 1)
        batt_bld_map = []  # pre-built on first step
        if step_i == 0:
            cnt = 0
            for i, n in enumerate(act_names):
                if str(n).strip().lower() == "electrical_storage":
                    batt_bld_map.append(cnt)
                    cnt += 1
        for bi_idx, bi in enumerate(batt_indices):
            bld_idx = bi_idx if bi_idx < n_buildings else 0
            if bld_idx < len(city.buildings):
                bld = city.buildings[bld_idx]
                for es in getattr(bld, "electrical_storage_devices", []):
                    soc_arr = getattr(es, "soc", None)
                    if soc_arr is not None and hasattr(soc_arr, "__len__") and t_nec < len(soc_arr):
                        soc_val = float(soc_arr[t_nec])
                        c2_total += 1
                        if soc_val > 0.95 or soc_val < 0.0:
                            c2_violations += 1
                    break

        # Battery action stats
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

        # EV action stats
        for ei in ev_indices:
            if ei < len(action):
                a = float(action[ei])
                # Check if EV is connected at this charger
                # Walk charger_info to find matching
                ev_connected_steps += 1
                if a > 0.1:
                    ev_charge_steps += 1
                elif a < -0.1:
                    ev_v2g_steps += 1
                    # Estimate V2G energy (rough: action * max_power * dt)
                    ev_total_v2g_kwh += abs(a) * 6.0  # ~6 kW max charger
                    if is_peak or (17 <= hour <= 23):
                        ev_v2g_peak_steps += 1
                if is_peak or (17 <= hour <= 23):
                    ev_v2g_peak_connected += 1

        # Track departures
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
                ev = getattr(ch, "connected_electric_vehicle", None)
                actual_soc = 0.0
                if ev is not None:
                    batt = getattr(ev, "battery", None)
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
                  f"c0_viol={viol_so_far}, c3={c3_violations}, c4={c4_violations}")

    elapsed = time.time() - t0
    metrics.eval_seconds = elapsed
    metrics.steps_completed = step_i + 1

    # --- Compute final metrics ---

    # C0: EV departure
    n_dep = len(departures)
    n_viol = sum(1 for _, ds, dr in departures if ds < dr)
    metrics.c0_total_departures = n_dep
    metrics.c0_violated = n_viol
    metrics.c0_violation_pct = 100.0 * n_viol / max(n_dep, 1)
    metrics.c0_mean_dep_soc = float(np.mean([ds for _, ds, _ in departures])) if departures else 0.0
    metrics.c0_mean_req_soc = float(np.mean([dr for _, _, dr in departures])) if departures else 0.0
    violated_deficits = [max(0, dr - ds) for _, ds, dr in departures if ds < dr]
    metrics.c0_mean_deficit = float(np.mean(violated_deficits)) if violated_deficits else 0.0

    # C2
    metrics.c2_violations = c2_violations
    metrics.c2_total = c2_total
    metrics.c2_violation_pct = 100.0 * c2_violations / max(c2_total, 1)

    # C3
    metrics.c3_violations = c3_violations
    metrics.c3_total = c3_total
    metrics.c3_violation_pct = 100.0 * c3_violations / max(c3_total, 1)

    # C4
    metrics.c4_violations = c4_violations
    metrics.c4_total = c4_total
    metrics.c4_violation_pct = 100.0 * c4_violations / max(c4_total, 1)

    # Battery
    bt = max(batt_total_steps, 1)
    metrics.batt_charge_pct = 100.0 * batt_charge_steps / bt
    metrics.batt_discharge_pct = 100.0 * batt_discharge_steps / bt
    metrics.batt_idle_pct = 100.0 * (bt - batt_charge_steps - batt_discharge_steps) / bt
    metrics.batt_solar_avg_action = float(np.mean(batt_solar_actions)) if batt_solar_actions else 0.0
    metrics.batt_peak_avg_action = float(np.mean(batt_peak_actions)) if batt_peak_actions else 0.0

    # Daily cycling: count days with both charge AND discharge
    all_days = set(list(batt_daily_charge.keys()) + list(batt_daily_discharge.keys()))
    cycling_days = sum(1 for d in all_days if batt_daily_charge.get(d, False) and batt_daily_discharge.get(d, False))
    total_days = max(len(all_days), 1)
    metrics.batt_daily_cycling = cycling_days

    # EV
    et = max(ev_connected_steps, 1)
    metrics.ev_charge_pct = 100.0 * ev_charge_steps / et
    metrics.ev_v2g_pct = 100.0 * ev_v2g_steps / et
    metrics.ev_v2g_peak_pct = 100.0 * ev_v2g_peak_steps / max(ev_v2g_peak_connected, 1)
    metrics.ev_total_v2g_kwh = ev_total_v2g_kwh

    # CityLearn KPIs
    metrics.total_import_kwh = total_import
    metrics.total_export_kwh = total_export
    metrics.peak_nec_kw = peak_nec
    metrics.ramping_kwh = ramping_total
    metrics.electricity_cost = electricity_cost_total
    mean_import = total_import / max(metrics.steps_completed, 1)
    metrics.load_factor = mean_import / max(peak_nec, 1e-8)

    return metrics


# ---------------------------------------------------------------------------
# Policy functions for baselines
# ---------------------------------------------------------------------------
def zero_action_policy(obs, act_dim=9):
    return np.zeros(act_dim)


def greedy_ev_policy(obs, act_names=None, act_dim=9):
    """Dumb greedy baseline: charge EVs at max + charge batteries at fixed rate.

    Represents the absolute worst active controller — blindly charges everything
    from the grid regardless of price, solar, SoC, or EV departure state.
    No V2G, no peak shaving, no solar awareness.
    """
    action = np.zeros(act_dim)
    if act_names:
        for i, n in enumerate(act_names):
            n_lower = str(n).strip().lower()
            if "electric_vehicle" in n_lower:
                action[i] = 1.0   # full-rate EV charge always
            elif n_lower == "electrical_storage":
                action[i] = 0.5   # constant battery charge from grid
    return action


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    all_results: List[EvalMetrics] = []

    log("=" * 70)
    log("  COMPREHENSIVE EVALUATION - ALL 5-BUILDING RUNS")
    log(f"  Schema: {SCHEMA_PATH}")
    log(f"  Seed: {SEED}, Steps: {TOTAL_STEPS}")
    log("=" * 70)
    log()

    # -----------------------------------------------------------------------
    # Evaluate trained runs
    # -----------------------------------------------------------------------
    for run_name, script_file, run_dir, epoch in RUN_DEFS:
        script_path = os.path.join(PROJECT, script_file)
        run_dir_full = os.path.join(PROJECT, run_dir)

        log(f"--- {run_name} ---")

        # Check script exists
        if not os.path.exists(script_path):
            log(f"  SKIP: script not found: {script_path}")
            continue

        # Find checkpoint
        ckpt_path = find_checkpoint(run_dir_full, epoch)
        if ckpt_path is None:
            log(f"  SKIP: no checkpoint in {run_dir_full}")
            continue

        epoch_num = int(os.path.basename(ckpt_path).replace("epoch-", "").replace(".pt", ""))
        log(f"  Script: {script_file}")
        log(f"  Checkpoint: epoch-{epoch_num}")

        # Parse env vars
        env_vars = parse_exports(script_path)
        metadata = extract_metadata(env_vars, script_path)
        use_action_mask = metadata["action_mask"]

        # Print key settings
        saute_str = "ON" if metadata["saute"] else "OFF"
        mask_str = "ON" if metadata["action_mask"] else "OFF"
        bclamp_str = "ON" if metadata["batt_clamp"] else "OFF"
        log(f"  Saute={saute_str}, ActionMask={mask_str}, BattClamp={bclamp_str}")

        try:
            # Build env (suppress noisy init)
            with suppress_stdout():
                env = build_env(env_vars, use_action_mask=use_action_mask)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            log(f"  Env: obs_dim={obs_dim}, act_dim={act_dim}")

            # Load actor
            actor, norm_mean, norm_std, norm_clip = load_actor(ckpt_path, obs_dim, act_dim)
            norm_str = "YES" if norm_mean is not None else "NO"
            log(f"  Actor loaded, normalizer={norm_str}")

            # Define action function
            def make_policy(actor, nm, ns, nc):
                def policy_fn(obs):
                    obs_n = normalize_obs(obs, nm, ns, nc)
                    obs_t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        mean = actor(obs_t)
                        return torch.tanh(mean).squeeze(0).numpy()
                return policy_fn

            policy_fn = make_policy(actor, norm_mean, norm_std, norm_clip)

            # Run eval
            result = evaluate_run(run_name, env, policy_fn, env_vars, metadata)
            all_results.append(result)

            log(f"  C0: {result.c0_violated}/{result.c0_total_departures} "
                  f"({result.c0_violation_pct:.1f}%), "
                  f"C3: {result.c3_violation_pct:.1f}%, C4: {result.c4_violation_pct:.1f}%")
            log(f"  Import: {result.total_import_kwh:.0f} kWh, "
                  f"Export: {result.total_export_kwh:.0f} kWh, "
                  f"Time: {result.eval_seconds:.1f}s")
            log()

        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()
            log()
            continue

    # -----------------------------------------------------------------------
    # Baselines
    # -----------------------------------------------------------------------
    # Use R19-like env vars as default (most common)
    baseline_env_vars = parse_exports(os.path.join(PROJECT, "run_r19_ablation.sh"))
    baseline_meta = {
        "saute": True, "action_mask": False, "batt_clamp": False,
        "bc_warmstart": False, "pid_lagrange": True, "curriculum": True,
        "lambda_ev": 0, "ev_guard": 0, "v2g_context": 0, "load_shift": 0,
        "price_arb": 0, "solar_store": 0, "headroom": 0, "ev_solar": 0,
        "ev_slack_arb": 0, "grid_penalty": 0,
    }

    # --- Zero action baseline ---
    log("--- zero_action (baseline) ---")
    try:
        with suppress_stdout():
            env = build_env(baseline_env_vars, use_action_mask=False)
        act_dim_base = env.action_space.shape[0]
        zero_fn = lambda obs: np.zeros(act_dim_base)
        result = evaluate_run("zero_action", env, zero_fn, baseline_env_vars, baseline_meta)
        all_results.append(result)
        log(f"  C0: {result.c0_violated}/{result.c0_total_departures} "
              f"({result.c0_violation_pct:.1f}%), "
              f"C3: {result.c3_violation_pct:.1f}%, C4: {result.c4_violation_pct:.1f}%")
        log(f"  Import: {result.total_import_kwh:.0f} kWh, Time: {result.eval_seconds:.1f}s")
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()
    log()

    # --- Greedy EV baseline ---
    log("--- greedy_ev (baseline) ---")
    try:
        with suppress_stdout():
            env = build_env(baseline_env_vars, use_action_mask=False)
        act_dim_base = env.action_space.shape[0]
        act_names_base = get_action_names(env)

        def greedy_fn(obs):
            return greedy_ev_policy(obs, act_names_base, act_dim_base)

        result = evaluate_run("greedy_ev", env, greedy_fn, baseline_env_vars, baseline_meta)
        all_results.append(result)
        log(f"  C0: {result.c0_violated}/{result.c0_total_departures} "
              f"({result.c0_violation_pct:.1f}%), "
              f"C3: {result.c3_violation_pct:.1f}%, C4: {result.c4_violation_pct:.1f}%")
        log(f"  Import: {result.total_import_kwh:.0f} kWh, Time: {result.eval_seconds:.1f}s")
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()
    log()

    # --- SmartV2GRBC baseline ---
    log("--- smart_v2g_rbc (baseline) ---")
    try:
        with suppress_stdout():
            env = build_env(baseline_env_vars, use_action_mask=False)
        from scripts.rbc_policy import SmartV2GRBC
        rbc = SmartV2GRBC(env)

        def rbc_fn(obs):
            return rbc.predict(obs)

        result = evaluate_run("smart_v2g_rbc", env, rbc_fn, baseline_env_vars, baseline_meta)
        all_results.append(result)
        log(f"  C0: {result.c0_violated}/{result.c0_total_departures} "
              f"({result.c0_violation_pct:.1f}%), "
              f"C3: {result.c3_violation_pct:.1f}%, C4: {result.c4_violation_pct:.1f}%")
        log(f"  Import: {result.total_import_kwh:.0f} kWh, Time: {result.eval_seconds:.1f}s")
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()
    log()

    # -----------------------------------------------------------------------
    # Generate markdown table
    # -----------------------------------------------------------------------
    log("=" * 70)
    log("  GENERATING RESULTS TABLE")
    log("=" * 70)

    md = generate_markdown(all_results)

    out_dir = os.path.join(PROJECT, "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "master_eval_table.md")
    with open(out_path, "w") as f:
        f.write(md)
    log(f"\nResults saved to: {out_path}")

    # Also print summary
    log("\n" + "=" * 70)
    log("  SUMMARY")
    log("=" * 70)
    log(f"{'Run':<22} {'C0%':>6} {'C2%':>6} {'C3%':>6} {'C4%':>6} "
          f"{'Import':>8} {'Export':>8} {'Peak':>6} {'Time':>6}")
    log("-" * 86)
    for r in all_results:
        log(f"{r.run_name:<22} {r.c0_violation_pct:>5.1f}% {r.c2_violation_pct:>5.1f}% "
              f"{r.c3_violation_pct:>5.1f}% {r.c4_violation_pct:>5.1f}% "
              f"{r.total_import_kwh:>7.0f} {r.total_export_kwh:>7.0f} "
              f"{r.peak_nec_kw:>5.1f} {r.eval_seconds:>5.0f}s")


def generate_markdown(results: List[EvalMetrics]) -> str:
    """Generate comprehensive markdown eval table."""
    lines = []
    lines.append("# Master Evaluation Table")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Schema: 5-building, full year (8759 steps), seed={SEED}")
    lines.append("")

    # --- Settings table ---
    lines.append("## Run Settings")
    lines.append("")
    lines.append("| Run | Saute | Mask | BClamp | BC | PID | Curric | lambda_ev | ev_guard | v2g_ctx | load_shift | price_arb | solar_store | headroom | ev_solar | ev_slack | grid_pen |")
    lines.append("|-----|-------|------|--------|----|-----|--------|-----------|----------|---------|------------|-----------|-------------|----------|----------|----------|----------|")
    for r in results:
        lines.append(
            f"| {r.run_name} "
            f"| {'Y' if r.saute else 'N'} "
            f"| {'Y' if r.action_mask else 'N'} "
            f"| {'Y' if r.batt_clamp else 'N'} "
            f"| {'Y' if r.bc_warmstart else 'N'} "
            f"| {'Y' if r.pid_lagrange else 'N'} "
            f"| {'Y' if r.curriculum else 'N'} "
            f"| {r.lambda_ev:.1f} "
            f"| {r.ev_guard:.1f} "
            f"| {r.v2g_context:.1f} "
            f"| {r.load_shift:.1f} "
            f"| {r.price_arb:.1f} "
            f"| {r.solar_store:.1f} "
            f"| {r.headroom:.1f} "
            f"| {r.ev_solar:.1f} "
            f"| {r.ev_slack_arb:.1f} "
            f"| {r.grid_penalty:.1f} |"
        )
    lines.append("")

    # --- Constraint violations table ---
    lines.append("## Constraint Violations")
    lines.append("")
    lines.append("| Run | C0 Deps | C0 Viol | C0% | Mean DepSoC | Mean ReqSoC | Mean Deficit | C2 Viol | C2% | C3 Viol | C3% | C4 Viol | C4% |")
    lines.append("|-----|---------|---------|-----|-------------|-------------|--------------|---------|-----|---------|-----|---------|-----|")
    for r in results:
        lines.append(
            f"| {r.run_name} "
            f"| {r.c0_total_departures} "
            f"| {r.c0_violated} "
            f"| {r.c0_violation_pct:.1f}% "
            f"| {r.c0_mean_dep_soc:.3f} "
            f"| {r.c0_mean_req_soc:.3f} "
            f"| {r.c0_mean_deficit:.3f} "
            f"| {r.c2_violations} "
            f"| {r.c2_violation_pct:.1f}% "
            f"| {r.c3_violations} "
            f"| {r.c3_violation_pct:.1f}% "
            f"| {r.c4_violations} "
            f"| {r.c4_violation_pct:.1f}% |"
        )
    lines.append("")

    # --- Battery metrics table ---
    lines.append("## Battery Metrics")
    lines.append("")
    lines.append("| Run | Charge% | Discharge% | Idle% | Solar Avg Act | Peak Avg Act | Daily Cycling |")
    lines.append("|-----|---------|------------|-------|---------------|--------------|---------------|")
    for r in results:
        lines.append(
            f"| {r.run_name} "
            f"| {r.batt_charge_pct:.1f}% "
            f"| {r.batt_discharge_pct:.1f}% "
            f"| {r.batt_idle_pct:.1f}% "
            f"| {r.batt_solar_avg_action:.3f} "
            f"| {r.batt_peak_avg_action:.3f} "
            f"| {r.batt_daily_cycling} |"
        )
    lines.append("")

    # --- EV metrics table ---
    lines.append("## EV Metrics")
    lines.append("")
    lines.append("| Run | Charge% | V2G% | V2G Peak% | V2G Energy (kWh) |")
    lines.append("|-----|---------|------|-----------|------------------|")
    for r in results:
        lines.append(
            f"| {r.run_name} "
            f"| {r.ev_charge_pct:.1f}% "
            f"| {r.ev_v2g_pct:.1f}% "
            f"| {r.ev_v2g_peak_pct:.1f}% "
            f"| {r.ev_total_v2g_kwh:.0f} |"
        )
    lines.append("")

    # --- CityLearn KPIs table ---
    lines.append("## CityLearn KPIs")
    lines.append("")
    lines.append("| Run | Import (kWh) | Export (kWh) | Peak NEC (kW) | Ramping (kWh) | Elec Cost | Load Factor |")
    lines.append("|-----|--------------|--------------|---------------|---------------|-----------|-------------|")
    for r in results:
        lines.append(
            f"| {r.run_name} "
            f"| {r.total_import_kwh:.0f} "
            f"| {r.total_export_kwh:.0f} "
            f"| {r.peak_nec_kw:.2f} "
            f"| {r.ramping_kwh:.0f} "
            f"| {r.electricity_cost:.0f} "
            f"| {r.load_factor:.3f} |"
        )
    lines.append("")

    # --- Compact summary ---
    lines.append("## Compact Summary")
    lines.append("")
    lines.append("| Run | C0% | C3% | C4% | Import | Export | Peak | Steps | Time |")
    lines.append("|-----|-----|-----|-----|--------|--------|------|-------|------|")
    for r in results:
        lines.append(
            f"| {r.run_name} "
            f"| {r.c0_violation_pct:.1f}% "
            f"| {r.c3_violation_pct:.1f}% "
            f"| {r.c4_violation_pct:.1f}% "
            f"| {r.total_import_kwh:.0f} "
            f"| {r.total_export_kwh:.0f} "
            f"| {r.peak_nec_kw:.1f} "
            f"| {r.steps_completed} "
            f"| {r.eval_seconds:.0f}s |"
        )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
