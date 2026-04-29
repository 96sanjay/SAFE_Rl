#!/usr/bin/env python3
"""
Comprehensive ablation evaluation: 5 trained runs + 3 baselines.
Each run uses its OWN env vars from its training shell script.
Produces a master table with all metrics.
"""
from __future__ import annotations

import copy
import gc
import os
import re
import sys
import glob
import importlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)


# ─────────────────────────────────────────────────────────────────
# Run definitions
# ─────────────────────────────────────────────────────────────────
@dataclass
class RunDef:
    name: str
    short: str
    script_path: str                  # .sh file for env vars
    ckpt_pattern: str                 # glob for torch_save dir
    is_multi_lambda: bool = False     # PPOLagMulti vs PPOLag
    is_baseline: bool = False
    baseline_type: str = ""           # "zero", "greedy_ev", "smart_rbc"


RUNS = [
    RunDef("Abl-0: Single-lam OLD",   "Abl0",
           os.path.join(PROJECT, "run_ablation_0_single_old.sh"),
           os.path.join(PROJECT, "runs/ablation_0_single_old/PPOLag*/seed-*/torch_save")),
    RunDef("Abl-1: Single-lam Reform", "Abl1",
           os.path.join(PROJECT, "run_ablation_1_single_reformed.sh"),
           os.path.join(PROJECT, "runs/ablation_1_single_reformed/PPOLag*/seed-*/torch_save")),
    RunDef("Abl-C: Multi-lam noSaute", "AblC",
           os.path.join(PROJECT, "run_ablation_C_nosaute.sh"),
           os.path.join(PROJECT, "runs/ablation_C_nosaute/PPO*/seed-*/torch_save"),
           is_multi_lambda=True),
    RunDef("Abl-B: Multi-lam C2tight", "AblB",
           os.path.join(PROJECT, "run_ablation_B_c2tight.sh"),
           os.path.join(PROJECT, "runs/ablation_B_c2tight/PPO*/seed-*/torch_save"),
           is_multi_lambda=True),
    RunDef("Run-4A: r25b Saute ON",    "r25b",
           os.path.join(PROJECT, "run_r25b_ev_slack_arb.sh"),
           os.path.join(PROJECT, "runs/r25b_ev_slack_arb/5bld/PPO*/seed-*/torch_save"),
           is_multi_lambda=True),
    # Baselines (use r25b env vars as default — only the policy differs)
    RunDef("Baseline: Zero",           "Zero",
           os.path.join(PROJECT, "run_r25b_ev_slack_arb.sh"), "",
           is_baseline=True, baseline_type="zero"),
    RunDef("Baseline: GreedyEV",       "GrdyEV",
           os.path.join(PROJECT, "run_r25b_ev_slack_arb.sh"), "",
           is_baseline=True, baseline_type="greedy_ev"),
    RunDef("Baseline: SmartV2GRBC",    "SmRBC",
           os.path.join(PROJECT, "run_r25b_ev_slack_arb.sh"), "",
           is_baseline=True, baseline_type="smart_rbc"),
]


# ─────────────────────────────────────────────────────────────────
# Metrics container
# ─────────────────────────────────────────────────────────────────
@dataclass
class Metrics:
    # C0 EV departure
    c0_departures: int = 0
    c0_violated: int = 0
    c0_viol_pct: float = 0.0
    c0_mean_dep_soc: float = 0.0
    c0_per_charger: dict = field(default_factory=dict)  # cid -> (departures, violated, mean_soc)

    # C2 Battery SoC
    c2_over_steps: int = 0
    c2_under_steps: int = 0
    c2_total_checks: int = 0
    c2_viol_pct: float = 0.0

    # C3 Building power
    c3_violations: int = 0
    c3_total: int = 0
    c3_viol_pct: float = 0.0

    # C4 Grid power
    c4_violations: int = 0
    c4_total: int = 0
    c4_viol_pct: float = 0.0

    # Battery metrics
    batt_charge_pct: float = 0.0
    batt_discharge_pct: float = 0.0
    batt_solar_avg: float = 0.0   # avg action hours 10-15
    batt_peak_avg: float = 0.0    # avg action hours 17-21

    # EV metrics
    ev_charge_pct: float = 0.0
    ev_v2g_pct: float = 0.0
    ev_v2g_peak_pct: float = 0.0

    # CityLearn KPIs
    total_import_kwh: float = 0.0
    total_export_kwh: float = 0.0
    peak_nec_kw: float = 0.0
    ramping: float = 0.0
    electricity_cost: float = 0.0

    # Summary
    total_reward: float = 0.0
    steps: int = 0


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def parse_exports(script_path: str) -> dict[str, str]:
    """Extract 'export KEY=VALUE' from bash script."""
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
            val_raw = val_raw.replace("${PYTHONPATH:-}", os.environ.get("PYTHONPATH", ""))
            exports[key] = val_raw
    return exports


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


def find_latest_ckpt(pattern: str) -> str:
    """Find latest epoch checkpoint in the torch_save dir matching glob pattern."""
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"No torch_save found: {pattern}")
    ckpt_dir = dirs[-1]
    epoch_files = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch-") and f.endswith(".pt")]
    if not epoch_files:
        raise FileNotFoundError(f"No epoch files in {ckpt_dir}")
    epochs = sorted([int(f.replace("epoch-", "").replace(".pt", "")) for f in epoch_files])
    latest = epochs[-1]
    return os.path.join(ckpt_dir, f"epoch-{latest}.pt")


def build_actor(ckpt_path: str, obs_dim: int, act_dim: int):
    """Load actor MLP and obs normalizer from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pi_state = ckpt["pi"]

    # Infer architecture from weight shapes
    w0 = pi_state["mean.0.weight"]
    in_dim = w0.shape[1]
    h0 = w0.shape[0]
    w2 = pi_state["mean.2.weight"]
    h1 = w2.shape[0]
    w4 = pi_state["mean.4.weight"]
    out_dim = w4.shape[0]

    if in_dim != obs_dim:
        raise ValueError(f"Checkpoint input dim ({in_dim}) != env obs dim ({obs_dim})")
    if out_dim != act_dim:
        raise ValueError(f"Checkpoint output dim ({out_dim}) != env act dim ({act_dim})")

    actor = torch.nn.Sequential(
        torch.nn.Linear(in_dim, h0),
        torch.nn.Tanh(),
        torch.nn.Linear(h0, h1),
        torch.nn.Tanh(),
        torch.nn.Linear(h1, out_dim),
    )

    key_map = {
        "mean.0.weight": "0.weight", "mean.0.bias": "0.bias",
        "mean.2.weight": "2.weight", "mean.2.bias": "2.bias",
        "mean.4.weight": "4.weight", "mean.4.bias": "4.bias",
    }
    actor_sd = {seq_key: pi_state[omni_key] for omni_key, seq_key in key_map.items()}
    actor.load_state_dict(actor_sd)
    actor.eval()

    # Obs normalizer
    norm_data = ckpt.get("obs_normalizer")
    norm_fn = None
    if norm_data is not None:
        nm = norm_data["_mean"].numpy().astype(np.float64)
        ns = np.maximum(norm_data["_std"].numpy().astype(np.float64), 0.01)
        nc = norm_data["_clip"].numpy().astype(np.float64)
        if len(nm) != obs_dim:
            raise ValueError(f"Normalizer dim ({len(nm)}) != obs dim ({obs_dim})")

        def norm_fn(obs, _m=nm, _s=ns, _c=nc):
            return np.clip((obs - _m) / _s, -_c, _c).astype(np.float32)

    return actor, norm_fn


def build_env_for_run(env_vars: dict):
    """Build CityLearnCMDPv2 env using the given env vars (matches training exactly)."""
    # Clear any stale env vars from previous run
    for k in list(os.environ.keys()):
        if k.startswith("STEMS_") or k.startswith("CITYLEARN_") or k.startswith("COST_"):
            del os.environ[k]

    # Apply this run's env vars
    for k, v in env_vars.items():
        os.environ[k] = v

    # Ensure critical defaults
    os.environ.setdefault("CITYLEARN_CENTRAL_AGENT", "1")
    os.environ.setdefault("CITYLEARN_REWARD_TYPE", "stems")

    # Force full-year schema
    os.environ["CITYLEARN_SCHEMA"] = os.path.join(
        PROJECT, "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
    )

    # Re-import to pick up new env vars (safety_env reads at import time).
    # Do NOT reload omni_env_v2 — OmniSafe's @env_register raises on re-register.
    # CityLearnCMDPv2's constructor calls make_base_env/wrappers fresh each time.
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
    # This guarantees identical obs dims, action preprocessing (WM disable,
    # battery clamp, EV clamp), and wrapper chain.
    import citylearn_safe.omni_env_v2  # noqa: registers once
    from citylearn_safe.omni_env_v2 import CityLearnCMDPv2
    env = CityLearnCMDPv2('CityLearnSafety-V2G-v2')
    return env


def get_action_indices(env):
    """Get battery, EV, and WM action indices."""
    city = get_citylearn_env(env)
    names_raw = getattr(city, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) > 0 and isinstance(names_raw[0], list):
        flat_names = [n for sub in names_raw for n in sub]
    else:
        flat_names = list(names_raw)

    batt_idx = [i for i, n in enumerate(flat_names) if n == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(flat_names) if "electric_vehicle" in n.lower()]
    wm_idx = [i for i, n in enumerate(flat_names) if "washing_machine" in str(n).lower()]
    return flat_names, batt_idx, ev_idx, wm_idx


def action_scale_fn(env):
    """Build ActionScale transform: [-1,1] -> [env.low, env.high]."""
    low = env.action_space.low.astype(np.float32)
    high = env.action_space.high.astype(np.float32)

    def scale(a):
        return low + (high - low) * (a - (-1.0)) / 2.0

    return scale


# ─────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────
def evaluate_run(rdef: RunDef) -> Metrics:
    """Run full-year evaluation for a single run definition."""
    print(f"\n{'='*70}")
    print(f"  Evaluating: {rdef.name}")
    print(f"{'='*70}")

    # 1. Parse env vars and build env
    env_vars = parse_exports(rdef.script_path)

    # For baselines, force Saute OFF (baselines don't have Saute-trained policy)
    if rdef.is_baseline:
        env_vars["CITYLEARN_EV_SAUTE"] = "0"

    env = build_env_for_run(env_vars)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")

    city = get_citylearn_env(env)
    flat_names, batt_idx, ev_idx, wm_idx = get_action_indices(env)
    wm_disable = os.environ.get("CITYLEARN_WM_DISABLE", "0") == "1"
    a_scale = action_scale_fn(env)

    # Build charger info for departure tracking
    charger_info = []
    for bi, b in enumerate(city.buildings):
        for ch in getattr(b, "electric_vehicle_chargers", []):
            cid = getattr(ch, "charger_id", f"charger_{bi}")
            charger_info.append((bi, ch, cid))

    # 2. Build policy
    actor = None
    norm_fn = None
    rbc = None

    if not rdef.is_baseline:
        ckpt_path = find_latest_ckpt(rdef.ckpt_pattern)
        print(f"  Checkpoint: {ckpt_path}")
        actor, norm_fn = build_actor(ckpt_path, obs_dim, act_dim)
        print(f"  Actor loaded: {sum(p.numel() for p in actor.parameters())} params")
    elif rdef.baseline_type == "smart_rbc":
        from scripts.rbc_policy import SmartV2GRBC
        rbc = SmartV2GRBC(env)

    # 3. Power thresholds
    P_building_max = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
    P_grid_max = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "10.2352"))

    # 4. Run episode
    torch.manual_seed(42)
    np.random.seed(42)
    obs_raw, info = env.reset(seed=42)
    obs = obs_raw.numpy() if isinstance(obs_raw, torch.Tensor) else np.asarray(obs_raw, dtype=np.float32)

    # Tracking
    departures = []  # (cid, actual_soc, required_soc, step)
    total_reward = 0.0
    step_count = 0

    # Battery action accumulators
    batt_charge_count = 0
    batt_discharge_count = 0
    batt_total_count = 0
    batt_solar_actions = []  # actions during hours 10-15
    batt_peak_actions = []   # actions during hours 17-21

    # EV accumulators
    ev_charge_count = 0
    ev_v2g_count = 0
    ev_v2g_peak_count = 0
    ev_connected_count = 0

    # C2 accumulators
    c2_over = 0
    c2_under = 0
    c2_total = 0

    # C3/C4
    c3_violations = 0
    c3_total = 0
    c4_violations = 0
    c4_total = 0

    # CityLearn KPIs
    nec_series = []  # total grid NEC per step
    price_series = []
    prev_nec = None
    ramping = 0.0

    max_steps = 8760

    for step_i in range(max_steps):
        # Get action
        if rdef.is_baseline:
            if rdef.baseline_type == "zero":
                action = np.zeros(act_dim, dtype=np.float32)
            elif rdef.baseline_type == "greedy_ev":
                action = np.zeros(act_dim, dtype=np.float32)
                for ei in ev_idx:
                    action[ei] = 1.0
            elif rdef.baseline_type == "smart_rbc":
                action = rbc.predict(obs)
            else:
                action = np.zeros(act_dim, dtype=np.float32)
        else:
            # Normalize obs
            obs_n = norm_fn(obs) if norm_fn is not None else obs.astype(np.float32)
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                raw_action = actor(obs_t).squeeze(0).numpy()
            # ActionScale
            action = a_scale(raw_action)
            # WM disable
            if wm_disable:
                for wi in wm_idx:
                    if wi < len(action):
                        action[wi] = 0.0

        # Hour before step
        t_before = int(getattr(city, "time_step", 0))
        hour = t_before % 24

        # Track battery actions
        for bi_idx in batt_idx:
            if bi_idx < len(action):
                a_val = float(action[bi_idx])
                batt_total_count += 1
                if a_val > 0.1:
                    batt_charge_count += 1
                elif a_val < -0.1:
                    batt_discharge_count += 1
                if 10 <= hour <= 15:
                    batt_solar_actions.append(a_val)
                if 17 <= hour <= 21:
                    batt_peak_actions.append(a_val)

        # Track EV actions (only when connected)
        for ei_idx in ev_idx:
            if ei_idx < len(action):
                a_val = float(action[ei_idx])
                # Check if EV connected
                connected = False
                aname = flat_names[ei_idx] if ei_idx < len(flat_names) else ""
                aname_l = str(aname).strip().lower()
                if "electric_vehicle_storage_charger_" in aname_l:
                    suffix = aname_l.split("electric_vehicle_storage_charger_", 1)[1]
                    charger_id = f"charger_{suffix}"
                    t_state = t_before + 1
                    for b in city.buildings:
                        for ch in getattr(b, "electric_vehicle_chargers", []):
                            cid = getattr(ch, "charger_id", "")
                            if str(cid).strip() == charger_id:
                                sim = getattr(ch, "charger_simulation", None)
                                if sim is not None:
                                    state_arr = getattr(sim, "_electric_vehicle_charger_state", None)
                                    if state_arr is not None and t_state < len(state_arr):
                                        if float(state_arr[t_state]) == 1.0:
                                            connected = True

                if connected:
                    ev_connected_count += 1
                    if a_val > 0.1:
                        ev_charge_count += 1
                    elif a_val < -0.1:
                        ev_v2g_count += 1
                        if 17 <= hour <= 23:
                            ev_v2g_peak_count += 1

        # Step — CityLearnCMDPv2 returns 6 values (obs, reward, cost, term, trunc, info)
        step_result = env.step(action)
        if len(step_result) == 6:
            obs_raw, reward_raw, _cost, terminated_raw, truncated_raw, info = step_result
        else:
            obs_raw, reward_raw, terminated_raw, truncated_raw, info = step_result
        obs = obs_raw.numpy() if isinstance(obs_raw, torch.Tensor) else np.asarray(obs_raw, dtype=np.float32)
        reward = float(reward_raw.item()) if isinstance(reward_raw, torch.Tensor) else float(reward_raw)
        terminated = bool(terminated_raw.item()) if isinstance(terminated_raw, torch.Tensor) else bool(terminated_raw)
        truncated = bool(truncated_raw.item()) if isinstance(truncated_raw, torch.Tensor) else bool(truncated_raw)
        step_count += 1
        total_reward += reward

        # Post-step state
        t_after = int(getattr(city, "time_step", 0))
        t_soc = max(0, t_after - 1)

        # --- C2: Battery SoC bounds ---
        for b in city.buildings:
            es = getattr(b, "electrical_storage", None)
            if es is not None:
                soc_arr = getattr(es, "soc", None)
                if soc_arr is not None and hasattr(soc_arr, "__len__") and t_soc < len(soc_arr):
                    soc_val = float(soc_arr[t_soc])
                    c2_total += 1
                    if soc_val > 0.95:
                        c2_over += 1
                    if soc_val < 0.0:
                        c2_under += 1

        # --- C3: Building NEC ---
        building_necs = []
        for b in city.buildings:
            c3_total += 1
            nec_arr = getattr(b, "net_electricity_consumption", None)
            nec_val = 0.0
            if nec_arr is not None and hasattr(nec_arr, "__len__") and t_soc < len(nec_arr):
                nec_val = float(nec_arr[t_soc])
            building_necs.append(nec_val)
            if abs(nec_val) > P_building_max:
                c3_violations += 1

        # --- C4: Grid NEC ---
        total_nec = sum(building_necs)
        c4_total += 1
        if abs(total_nec) > P_grid_max:
            c4_violations += 1

        # --- CityLearn KPIs ---
        nec_series.append(total_nec)
        if prev_nec is not None:
            ramping += abs(total_nec - prev_nec)
        prev_nec = total_nec

        # Price
        price = 0.17  # default
        try:
            pr = city.buildings[0].pricing.electricity_pricing
            if hasattr(pr, "__len__") and t_soc < len(pr):
                price = float(pr[t_soc])
        except Exception:
            pass
        price_series.append(price)

        # --- Departures ---
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
                departures.append((cid, actual_soc, r, step_i))

        if terminated or truncated:
            print(f"  Episode ended at step {step_i + 1}")
            break

        if (step_i + 1) % 2000 == 0:
            dep_so_far = len(departures)
            viol_so_far = sum(1 for _, ds, dr, _ in departures if ds < dr)
            print(f"    Step {step_i+1}: deps={dep_so_far}, violated={viol_so_far}")

    # ── Compute metrics ──
    m = Metrics()
    m.steps = step_count
    m.total_reward = total_reward

    # C0
    m.c0_departures = len(departures)
    m.c0_violated = sum(1 for _, ds, dr, _ in departures if ds < dr - 0.001)
    m.c0_viol_pct = 100.0 * m.c0_violated / max(m.c0_departures, 1)
    m.c0_mean_dep_soc = float(np.mean([ds for _, ds, _, _ in departures])) if departures else 0.0

    # Per-charger
    for cname in sorted(set(cn for cn, _, _, _ in departures)):
        ch_deps = [(ds, dr) for cn, ds, dr, _ in departures if cn == cname]
        ch_viols = sum(1 for ds, dr in ch_deps if ds < dr - 0.001)
        ch_mean_soc = float(np.mean([ds for ds, _ in ch_deps]))
        m.c0_per_charger[cname] = (len(ch_deps), ch_viols, ch_mean_soc)

    # C2
    m.c2_over_steps = c2_over
    m.c2_under_steps = c2_under
    m.c2_total_checks = c2_total
    m.c2_viol_pct = 100.0 * (c2_over + c2_under) / max(c2_total, 1)

    # C3
    m.c3_violations = c3_violations
    m.c3_total = c3_total
    m.c3_viol_pct = 100.0 * c3_violations / max(c3_total, 1)

    # C4
    m.c4_violations = c4_violations
    m.c4_total = c4_total
    m.c4_viol_pct = 100.0 * c4_violations / max(c4_total, 1)

    # Battery
    bt = max(batt_total_count, 1)
    m.batt_charge_pct = 100.0 * batt_charge_count / bt
    m.batt_discharge_pct = 100.0 * batt_discharge_count / bt
    m.batt_solar_avg = float(np.mean(batt_solar_actions)) if batt_solar_actions else 0.0
    m.batt_peak_avg = float(np.mean(batt_peak_actions)) if batt_peak_actions else 0.0

    # EV
    ec = max(ev_connected_count, 1)
    m.ev_charge_pct = 100.0 * ev_charge_count / ec
    m.ev_v2g_pct = 100.0 * ev_v2g_count / ec
    # V2G at peak as % of connected-at-peak
    peak_ev_connected = ev_v2g_peak_count  # already filtered to 17-23 connected
    m.ev_v2g_peak_pct = 100.0 * ev_v2g_peak_count / max(ev_connected_count, 1)

    # CityLearn KPIs
    nec_arr = np.array(nec_series)
    m.total_import_kwh = float(np.sum(np.maximum(nec_arr, 0)))
    m.total_export_kwh = float(np.sum(np.maximum(-nec_arr, 0)))
    m.peak_nec_kw = float(np.max(np.abs(nec_arr))) if len(nec_arr) > 0 else 0.0
    m.ramping = ramping
    price_arr = np.array(price_series)
    m.electricity_cost = float(np.sum(np.maximum(nec_arr, 0) * price_arr[:len(nec_arr)]))

    # Print summary
    print(f"\n  Results for {rdef.short}:")
    print(f"    C0: {m.c0_violated}/{m.c0_departures} ({m.c0_viol_pct:.1f}%), mean SoC={m.c0_mean_dep_soc:.3f}")
    print(f"    C2: {m.c2_viol_pct:.1f}% (over={c2_over}, under={c2_under})")
    print(f"    C3: {m.c3_viol_pct:.1f}%, C4: {m.c4_viol_pct:.1f}%")
    print(f"    Import={m.total_import_kwh:.0f} kWh, Export={m.total_export_kwh:.0f} kWh")
    print(f"    Peak={m.peak_nec_kw:.2f} kW, Cost=${m.electricity_cost:.0f}")

    # Clean up
    del env
    gc.collect()

    return m


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    results = OrderedDict()
    for rdef in RUNS:
        try:
            m = evaluate_run(rdef)
            results[rdef.short] = (rdef.name, m)
        except Exception as e:
            print(f"\n  ERROR evaluating {rdef.name}: {e}")
            import traceback
            traceback.print_exc()
            results[rdef.short] = (rdef.name, None)

    # ── Print master table ──
    print("\n\n")
    print("=" * 120)
    print("  MASTER ABLATION TABLE")
    print("=" * 120)

    # Column headers
    shorts = list(results.keys())

    def val(short, attr, fmt=".1f"):
        name, m = results[short]
        if m is None:
            return "ERR"
        v = getattr(m, attr)
        return f"{v:{fmt}}"

    def ival(short, attr):
        name, m = results[short]
        if m is None:
            return "ERR"
        return str(getattr(m, attr))

    # Build markdown table
    lines = []
    lines.append("# Ablation Final Evaluation Table")
    lines.append("")
    lines.append(f"Schema: 5-building, full year (8759 steps), seed=42")
    lines.append("")

    # Header
    hdr = "| Metric |"
    sep = "|--------|"
    for s in shorts:
        hdr += f" {s} |"
        sep += "------|"
    lines.append(hdr)
    lines.append(sep)

    def row(label, attr, fmt=".1f"):
        r = f"| {label} |"
        for s in shorts:
            r += f" {val(s, attr, fmt)} |"
        lines.append(r)

    def irow(label, attr):
        r = f"| {label} |"
        for s in shorts:
            r += f" {ival(s, attr)} |"
        lines.append(r)

    # Full name row
    r = "| **Run** |"
    for s in shorts:
        name, m = results[s]
        r += f" {name} |"
    lines.append(r)

    lines.append("| | | | | | | | | |")

    # C0
    lines.append("| **C0: EV Departure** | | | | | | | | |")
    irow("Departures", "c0_departures")
    irow("Violated", "c0_violated")
    row("Violation %", "c0_viol_pct", ".1f")
    row("Mean dep SoC", "c0_mean_dep_soc", ".3f")

    lines.append("| | | | | | | | | |")
    lines.append("| **C2: Battery SoC** | | | | | | | | |")
    irow("Over 0.95", "c2_over_steps")
    irow("Under 0.0", "c2_under_steps")
    row("Violation %", "c2_viol_pct", ".2f")

    lines.append("| | | | | | | | | |")
    lines.append("| **C3: Building Power** | | | | | | | | |")
    irow("Violations", "c3_violations")
    row("Violation %", "c3_viol_pct", ".1f")

    lines.append("| | | | | | | | | |")
    lines.append("| **C4: Grid Power** | | | | | | | | |")
    irow("Violations", "c4_violations")
    row("Violation %", "c4_viol_pct", ".1f")

    lines.append("| | | | | | | | | |")
    lines.append("| **Battery Behavior** | | | | | | | | |")
    row("Charge %", "batt_charge_pct", ".1f")
    row("Discharge %", "batt_discharge_pct", ".1f")
    row("Solar avg act", "batt_solar_avg", ".3f")
    row("Peak avg act", "batt_peak_avg", ".3f")

    lines.append("| | | | | | | | | |")
    lines.append("| **EV Behavior** | | | | | | | | |")
    row("Charge % (conn)", "ev_charge_pct", ".1f")
    row("V2G % (conn)", "ev_v2g_pct", ".1f")
    row("V2G peak %", "ev_v2g_peak_pct", ".1f")

    lines.append("| | | | | | | | | |")
    lines.append("| **CityLearn KPIs** | | | | | | | | |")
    row("Import (kWh)", "total_import_kwh", ".0f")
    row("Export (kWh)", "total_export_kwh", ".0f")
    row("Peak |NEC| (kW)", "peak_nec_kw", ".2f")
    row("Ramping", "ramping", ".0f")
    row("Elec cost ($)", "electricity_cost", ".0f")

    lines.append("| | | | | | | | | |")
    row("Total reward", "total_reward", ".1f")
    irow("Steps", "steps")

    # Per-charger C0 details
    lines.append("")
    lines.append("## Per-Charger C0 Breakdown")
    lines.append("")
    for s in shorts:
        name, m = results[s]
        if m is None:
            continue
        if not m.c0_per_charger:
            continue
        lines.append(f"### {name}")
        lines.append("| Charger | Departures | Violated | Viol % | Mean SoC |")
        lines.append("|---------|-----------|----------|--------|----------|")
        for cid, (deps, viols, msoc) in sorted(m.c0_per_charger.items()):
            vpct = 100.0 * viols / max(deps, 1)
            lines.append(f"| {cid} | {deps} | {viols} | {vpct:.1f}% | {msoc:.3f} |")
        lines.append("")

    # Write file
    table_text = "\n".join(lines)
    print(table_text)

    out_path = os.path.join(PROJECT, "docs", "ablation_final_table.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table_text + "\n")
    print(f"\nTable saved to: {out_path}")


if __name__ == "__main__":
    main()
