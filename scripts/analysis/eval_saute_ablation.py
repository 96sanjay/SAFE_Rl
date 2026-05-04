#!/usr/bin/env python3
"""
Deterministic 3-way evaluation:
  PPO Saute ON (r25b_stable) vs PPO Saute OFF (nosaute_no_c1_clean) vs SAC-Lag (100ep).

FIX from eval_ppo_vs_sac_deterministic.py:
  - PPO policy uses torch.clamp(-1,1) NOT torch.tanh()
    (training clips via safety_env.py:712, not tanh)
  - SAC policy correctly uses tanh (matches GaussianSACActor.predict())

Usage:
    conda activate citylearn
    python scripts/eval_saute_ablation.py
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------------------
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT)

SCHEMA_PATH = os.path.join(
    PROJECT,
    "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
)
TOTAL_STEPS = 8759
SEED = 42

DEFAULT_P_BUILDING_MAX = 4.6083
DEFAULT_P_GRID_MAX = 10.2352

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
SAUTE_ON_CKPT = os.path.join(
    PROJECT,
    "runs/r25b_report_stable/5bld/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-27-04-04-26/torch_save/epoch-80.pt",
)
SAUTE_OFF_CKPT = "/tmp/nosaute_epoch-80.pt"
SAC_CKPT = os.path.join(
    PROJECT,
    "runs/r25b_report_stable_sac_full_100ep_seed42/sac_epoch_100.pt",
)

SAUTE_ON_SCRIPT = os.path.join(PROJECT, "run_r25b_report_stable.sh")
SAUTE_OFF_SCRIPT = os.path.join(PROJECT, "run_r25b_report_stable_nosaute_no_c1_clean.sh")
SAC_SCRIPT = os.path.join(PROJECT, "experiments/r25b_report_stable_sac/run.sh")


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
    keys_to_clear = [k for k in os.environ if k.startswith(("STEMS_", "CITYLEARN_", "COST_W_"))]
    for k in keys_to_clear:
        del os.environ[k]

    for k, v in env_vars.items():
        os.environ[k] = v

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

    import citylearn_safe.cmdp_env
    from citylearn_safe.cmdp_env import CityLearnCMDP
    env = CityLearnCMDP('CityLearnSafety-V2G-v2')
    return env


# ---------------------------------------------------------------------------
# Actor loader
# ---------------------------------------------------------------------------
def load_ppo_actor(ckpt_path: str, obs_dim: int, act_dim: int = 9):
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

    nd = ckpt.get("obs_normalizer")
    if nd is None:
        return actor, None, None, None
    mean = nd["_mean"].numpy()
    std = np.maximum(nd["_std"].numpy(), 1e-8)
    clip = nd["_clip"].numpy()
    return actor, mean, std, clip


def make_ppo_policy(actor, norm_mean, norm_std, norm_clip):
    """Deterministic PPO policy — uses clamp, NOT tanh (matches training)."""
    def policy(obs):
        obs_n = normalize_obs(obs, norm_mean, norm_std, norm_clip)
        t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = actor(t)
            # FIXED: clamp to [-1,1] — same as safety_env.py:712 during training
            return torch.clamp(out, -1.0, 1.0).squeeze(0).numpy()
    return policy


def load_sac_actor(ckpt_path: str, obs_dim: int, act_dim: int = 9):
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

    nd = ckpt.get("obs_normalizer")
    if nd is None:
        return actor, None, None, None
    mean = nd["_mean"].numpy()
    std = np.maximum(nd["_std"].numpy(), 1e-8)
    clip = nd["_clip"].numpy()
    return actor, mean, std, clip


def make_sac_policy(actor, norm_mean, norm_std, norm_clip, act_dim=9):
    """Deterministic SAC policy — uses tanh on mean (matches GaussianSACActor.predict)."""
    def policy(obs):
        obs_n = normalize_obs(obs, norm_mean, norm_std, norm_clip)
        t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = actor(t).squeeze(0)
            mean = out[:act_dim]
            return torch.tanh(mean).numpy()
    return policy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(run_name: str, env, policy_fn, env_vars: dict) -> dict:
    """Run full-year deterministic evaluation. Returns dict of metrics."""
    city = get_citylearn_env(env)
    if city is None:
        log(f"  ERROR: Could not find CityLearnEnv for {run_name}")
        return {}

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

    # Tracking
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
    ev_charge_steps = 0; ev_v2g_steps = 0
    ev_connected_steps = 0

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

        # ---- C2: Battery SoC violations ----
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
                elif a < -0.1:
                    batt_discharge_steps += 1

        # ---- EV action stats ----
        for ei in ev_indices:
            if ei < len(action):
                a = float(action[ei])
                ev_connected_steps += 1
                if a > 0.1:
                    ev_charge_steps += 1
                elif a < -0.1:
                    ev_v2g_steps += 1

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

    # ---- Compile metrics ----
    n_dep = len(departures)
    n_viol = sum(1 for _, ds, dr in departures if ds < dr)
    violated_deficits = [max(0, dr - ds) for _, ds, dr in departures if ds < dr]

    bt = max(batt_total_steps, 1)
    et = max(ev_connected_steps, 1)

    mean_import = total_import / max(step_i + 1, 1)

    return {
        "name": run_name,
        "steps": step_i + 1,
        "eval_time_s": elapsed,
        # C0
        "c0_departures": n_dep,
        "c0_violated": n_viol,
        "c0_viol_pct": 100.0 * n_viol / max(n_dep, 1),
        "c0_mean_dep_soc": float(np.mean([ds for _, ds, _ in departures])) if departures else 0.0,
        "c0_mean_req_soc": float(np.mean([dr for _, _, dr in departures])) if departures else 0.0,
        "c0_mean_deficit": float(np.mean(violated_deficits)) if violated_deficits else 0.0,
        # C2
        "c2_violations": c2_violations,
        "c2_total": c2_total,
        "c2_viol_pct": 100.0 * c2_violations / max(c2_total, 1),
        # C3
        "c3_violations": c3_violations,
        "c3_total": c3_total,
        "c3_viol_pct": 100.0 * c3_violations / max(c3_total, 1),
        # C4
        "c4_violations": c4_violations,
        "c4_total": c4_total,
        "c4_viol_pct": 100.0 * c4_violations / max(c4_total, 1),
        # Battery
        "batt_charge_pct": 100.0 * batt_charge_steps / bt,
        "batt_discharge_pct": 100.0 * batt_discharge_steps / bt,
        "batt_idle_pct": 100.0 * (bt - batt_charge_steps - batt_discharge_steps) / bt,
        # EV
        "ev_charge_pct": 100.0 * ev_charge_steps / et,
        "ev_v2g_pct": 100.0 * ev_v2g_steps / et,
        # Grid KPIs
        "total_import_kwh": total_import,
        "total_export_kwh": total_export,
        "peak_nec_kw": peak_nec,
        "ramping_kwh": ramping_total,
        "elec_cost": electricity_cost_total,
        "load_factor": mean_import / max(peak_nec, 1e-8),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_comparison(results: list[dict]):
    """Print side-by-side comparison table for N models."""
    names = [r.get("name", "?") for r in results]
    col_w = 18
    W = 36 + col_w * len(results)

    def row(label, key, fmt=".1f", unit=""):
        line = f"  {label:<34}"
        for r in results:
            v = r.get(key, 0)
            line += f" {v:{fmt}}{unit:>{col_w - len(f'{v:{fmt}}') - len(unit)}}"
        log(line)

    def row_frac(label, viol_k, total_k, pct_k):
        line = f"  {label:<34}"
        for r in results:
            v = r.get(viol_k, 0); t = r.get(total_k, 0); p = r.get(pct_k, 0)
            s = f"{v}/{t} ({p:.1f}%)"
            line += f" {s:>{col_w}}"
        log(line)

    log()
    log("=" * W)
    log(f"  {'3-Way Deterministic Eval (FIXED: PPO=clamp, SAC=tanh)':^{W-2}}")
    log("=" * W)
    header = f"  {'Metric':<34}"
    for n in names:
        header += f" {n:>{col_w}}"
    log(header)
    log(f"  {'='*34}" + f" {'='*col_w}" * len(results))

    log("\n  ---- C0: EV Departure ----")
    row("Total departures", "c0_departures", fmt=".0f")
    row("Violated departures", "c0_violated", fmt=".0f")
    row("Violation %", "c0_viol_pct", unit="%")
    row("Mean departure SoC", "c0_mean_dep_soc", fmt=".4f")
    row("Mean required SoC", "c0_mean_req_soc", fmt=".4f")
    row("Mean deficit (violated)", "c0_mean_deficit", fmt=".4f")

    log("\n  ---- C2: Battery SoC ----")
    row_frac("C2 violations", "c2_violations", "c2_total", "c2_viol_pct")

    log("\n  ---- C3: Building Power ----")
    row_frac("C3 violations", "c3_violations", "c3_total", "c3_viol_pct")

    log("\n  ---- C4: Grid Power ----")
    row_frac("C4 violations", "c4_violations", "c4_total", "c4_viol_pct")

    log("\n  ---- Battery Behavior ----")
    row("Charge %", "batt_charge_pct", unit="%")
    row("Discharge %", "batt_discharge_pct", unit="%")
    row("Idle %", "batt_idle_pct", unit="%")

    log("\n  ---- EV Behavior ----")
    row("Charge %", "ev_charge_pct", unit="%")
    row("V2G discharge %", "ev_v2g_pct", unit="%")

    log("\n  ---- Grid KPIs ----")
    row("Total import (kWh)", "total_import_kwh", fmt=".0f")
    row("Total export (kWh)", "total_export_kwh", fmt=".0f")
    row("Peak NEC (kW)", "peak_nec_kw", fmt=".2f")
    row("Ramping (kWh)", "ramping_kwh", fmt=".0f")
    row("Electricity cost ($)", "elec_cost", fmt=".0f")
    row("Load factor", "load_factor", fmt=".4f")

    steps_line = "  Steps:  " + " / ".join(str(r.get("steps", 0)) for r in results)
    time_line = "  Eval time:  " + " / ".join(f"{r.get('eval_time_s', 0):.0f}s" for r in results)
    log(f"\n{steps_line}")
    log(time_line)
    log("=" * W)

    import hashlib
    vals = tuple(r.get(k, 0) for r in results for k in ["c0_viol_pct", "c3_viol_pct", "c4_viol_pct", "total_import_kwh"])
    h = hashlib.md5(str(vals).encode()).hexdigest()[:12]
    log(f"\nDeterminism hash: {h}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 70)
    log("  3-Way Eval: PPO Saute ON vs PPO Saute OFF vs SAC-Lag")
    log(f"  Seed: {SEED} | Steps: {TOTAL_STEPS}")
    log(f"  FIX: PPO uses clamp(-1,1), SAC uses tanh (matches training)")
    log("=" * 70)

    for name, path in [("PPO Saute ON", SAUTE_ON_CKPT),
                        ("PPO Saute OFF", SAUTE_OFF_CKPT),
                        ("SAC-Lag", SAC_CKPT)]:
        if not os.path.exists(path):
            log(f"ERROR: {name} checkpoint not found: {path}")
            sys.exit(1)
        log(f"  {name}: {path}")
    log()

    all_results = []

    # --- PPO Saute ON ---
    log(">>> [1/3] Evaluating PPO Saute ON (r25b_stable, epoch 80) ...")
    env_vars_on = parse_exports(SAUTE_ON_SCRIPT)
    with suppress_stdout():
        env_on = build_env(env_vars_on)
    obs_dim_on = env_on.observation_space.shape[0]
    act_dim_on = env_on.action_space.shape[0]
    log(f"  Env: obs_dim={obs_dim_on}, act_dim={act_dim_on}")

    actor_on, nm, ns, nc = load_ppo_actor(SAUTE_ON_CKPT, obs_dim_on, act_dim_on)
    log(f"  Actor loaded, normalizer={'YES' if nm is not None else 'NO'}")
    policy_on = make_ppo_policy(actor_on, nm, ns, nc)
    result_on = evaluate("PPO_Saute_ON", env_on, policy_on, env_vars_on)
    log(f"  Done: C0={result_on['c0_viol_pct']:.1f}%, C3={result_on['c3_viol_pct']:.1f}%, "
        f"C4={result_on['c4_viol_pct']:.1f}% in {result_on['eval_time_s']:.0f}s")
    all_results.append(result_on)
    log()

    # --- PPO Saute OFF ---
    log(">>> [2/3] Evaluating PPO Saute OFF (nosaute_no_c1_clean, epoch 80) ...")
    env_vars_off = parse_exports(SAUTE_OFF_SCRIPT)
    with suppress_stdout():
        env_off = build_env(env_vars_off)
    obs_dim_off = env_off.observation_space.shape[0]
    act_dim_off = env_off.action_space.shape[0]
    log(f"  Env: obs_dim={obs_dim_off}, act_dim={act_dim_off}")

    actor_off, nm2, ns2, nc2 = load_ppo_actor(SAUTE_OFF_CKPT, obs_dim_off, act_dim_off)
    log(f"  Actor loaded, normalizer={'YES' if nm2 is not None else 'NO'}")
    policy_off = make_ppo_policy(actor_off, nm2, ns2, nc2)
    result_off = evaluate("PPO_Saute_OFF", env_off, policy_off, env_vars_off)
    log(f"  Done: C0={result_off['c0_viol_pct']:.1f}%, C3={result_off['c3_viol_pct']:.1f}%, "
        f"C4={result_off['c4_viol_pct']:.1f}% in {result_off['eval_time_s']:.0f}s")
    all_results.append(result_off)
    log()

    # --- SAC-Lag ---
    log(">>> [3/3] Evaluating SAC-Lag (100ep, epoch 100) ...")
    env_vars_sac = parse_exports(SAC_SCRIPT)
    with suppress_stdout():
        env_sac = build_env(env_vars_sac)
    obs_dim_sac = env_sac.observation_space.shape[0]
    act_dim_sac = env_sac.action_space.shape[0]
    log(f"  Env: obs_dim={obs_dim_sac}, act_dim={act_dim_sac}")

    actor_sac, nm3, ns3, nc3 = load_sac_actor(SAC_CKPT, obs_dim_sac, act_dim_sac)
    log(f"  Actor loaded, normalizer={'YES' if nm3 is not None else 'NO'}")
    policy_sac = make_sac_policy(actor_sac, nm3, ns3, nc3, act_dim_sac)
    result_sac = evaluate("SAC-Lag_100ep", env_sac, policy_sac, env_vars_sac)
    log(f"  Done: C0={result_sac['c0_viol_pct']:.1f}%, C3={result_sac['c3_viol_pct']:.1f}%, "
        f"C4={result_sac['c4_viol_pct']:.1f}% in {result_sac['eval_time_s']:.0f}s")
    all_results.append(result_sac)
    log()

    # --- Output ---
    print_comparison(all_results)


if __name__ == "__main__":
    main()
