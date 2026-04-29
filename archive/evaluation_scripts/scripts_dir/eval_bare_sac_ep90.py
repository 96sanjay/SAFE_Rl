#!/usr/bin/env python3
"""
Quick eval: Bare SAC-Lag epoch-90 checkpoint.
Deterministic, single-model evaluation with full KPI + constraint violation report.
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

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

SCHEMA_PATH = os.path.join(
    PROJECT,
    "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
)
TOTAL_STEPS = 8759
SEED = 42

CKPT_PATH = os.path.join(
    PROJECT,
    "runs/r25b_report_stable_sac_bare_sac_100ep_seed42/5bld/"
    "SACLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-04-07-01-30-01/torch_save/epoch-90.pt",
)

DEFAULT_P_BUILDING_MAX = 4.6083
DEFAULT_P_GRID_MAX = 10.2352

# ---- Env vars: base (from run_r25b_report_stable.sh) + bare overrides ----
ENV_VARS = {
    "CITYLEARN_SCHEMA": SCHEMA_PATH,
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    "CITYLEARN_PID_LAGRANGE": "1",
    # Bare: no saute, no clamp, no mask
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_BATT_CLAMP": "0",
    "CITYLEARN_ACTION_MASK": "0",
    "CITYLEARN_POLICY_ACTION_MASK": "0",
    "CITYLEARN_EV_DENSE_COST_SCALE": "0.0",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_SPATIAL_OBS": "0",
    # Reward weights (trial #15 + bare overrides)
    "STEMS_ALPHA_GRID": "0.0",
    "STEMS_SG_THRESHOLD": "0.5",
    "STEMS_ALPHA_LOAD_SHIFT": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.8",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_XI_RENEWABLE": "0.2",
    "STEMS_BETA_RAMP": "0.3",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "STEMS_LAMBDA_EV": "3.0",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "STEMS_ALPHA_EV_GUARD": "0.5",
    "STEMS_ALPHA_V2G_CONTEXT": "4.5",
    "STEMS_EV_SLACK_ARB_SCALE": "2.0",
    # Cost weights
    "COST_W_C2": "0.0",
    "COST_W_C3": "3.0",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.03",
    "CITYLEARN_EV_COST_SCALE": "3.0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_C0_SAFETY_FLOOR": "0.5",
}


def log(msg=""):
    print(msg, flush=True)


@contextlib.contextmanager
def suppress_stdout():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def build_env(env_vars):
    keys_to_clear = [k for k in os.environ if k.startswith(("STEMS_", "CITYLEARN_", "COST_W_"))]
    for k in keys_to_clear:
        del os.environ[k]
    for k, v in env_vars.items():
        os.environ[k] = v
    mods_to_reload = [m for name, m in sys.modules.items()
                      if m is not None and any(s in name for s in
                          ("safety_env", "make_env", "forecast_obs", "saute_ev", "extractors"))
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


def get_action_names(env):
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


def load_sac_actor(ckpt_path, obs_dim=199, act_dim=9):
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


def main():
    log("=" * 72)
    log("  Bare SAC-Lag Evaluation (epoch-90)")
    log(f"  Checkpoint: {CKPT_PATH}")
    log(f"  Seed: {SEED} | Steps: {TOTAL_STEPS}")
    log("=" * 72)

    if not os.path.exists(CKPT_PATH):
        log(f"ERROR: Checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log("\nBuilding env...")
    with suppress_stdout():
        env = build_env(ENV_VARS)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    log(f"  obs_dim={obs_dim}, act_dim={act_dim}")

    log("Loading actor...")
    actor, nm, ns, nc = load_sac_actor(CKPT_PATH, obs_dim, act_dim)
    log(f"  Normalizer: {'YES' if nm is not None else 'NO'}")

    def policy(obs):
        obs_n = normalize_obs(obs, nm, ns, nc)
        t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = actor(t).squeeze(0)
            mean = out[:act_dim]
            return torch.tanh(mean).numpy()

    city = get_citylearn_env(env)
    if city is None:
        log("ERROR: Could not find CityLearnEnv")
        sys.exit(1)

    n_buildings = len(city.buildings)
    act_names = get_action_names(env)
    batt_indices = [i for i, n in enumerate(act_names) if str(n).strip().lower() == "electrical_storage"]
    ev_indices = [i for i, n in enumerate(act_names) if "electric_vehicle" in str(n).lower()]
    log(f"  Buildings: {n_buildings}, Batt indices: {batt_indices}, EV indices: {ev_indices}")

    P_bld_max = DEFAULT_P_BUILDING_MAX
    P_grid_max = DEFAULT_P_GRID_MAX

    charger_info = []
    for bi, b in enumerate(city.buildings):
        for ch in getattr(b, "electric_vehicle_chargers", []):
            cid = getattr(ch, "charger_id", f"charger_{bi}")
            charger_info.append((bi, ch, cid))

    # Reset
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
    elec_cost = 0.0
    peak_nec = 0.0

    c2_violations = 0; c2_total = 0
    c3_violations = 0; c3_total = 0
    c4_violations = 0; c4_total = 0

    batt_charge = 0; batt_discharge = 0; batt_total = 0
    batt_solar_acts = []; batt_peak_acts = []
    batt_daily_c = defaultdict(bool); batt_daily_d = defaultdict(bool)

    ev_charge = 0; ev_v2g = 0; ev_v2g_peak = 0; ev_connected = 0; ev_v2g_peak_conn = 0
    ev_total_v2g_kwh = 0.0

    log("\nRunning evaluation...")
    t0 = time.time()

    for step_i in range(TOTAL_STEPS):
        action = policy(obs)
        with suppress_stdout():
            step_result = env.step(action)
        if len(step_result) == 6:
            obs_raw, reward_raw, _cost, term, trunc, info = step_result
        else:
            obs_raw, reward_raw, term, trunc, info = step_result
        obs = obs_raw.numpy() if isinstance(obs_raw, torch.Tensor) else np.asarray(obs_raw, dtype=np.float32)
        term = bool(term.item()) if isinstance(term, torch.Tensor) else bool(term)
        trunc = bool(trunc.item()) if isinstance(trunc, torch.Tensor) else bool(trunc)

        t_after = int(getattr(city, "time_step", 0))
        hour = t_after % 24
        day = t_after // 24
        is_solar = 10 <= hour <= 15
        is_peak = 17 <= hour <= 21

        step_nec = float(info.get("step_net_consumption_kwh", 0.0))
        grid_imp = float(info.get("grid_import_kwh", max(0.0, step_nec)))
        grid_exp = float(info.get("grid_export_kwh", max(0.0, -step_nec)))
        t_nec = max(0, t_after - 1)

        # C3
        for bi, b in enumerate(city.buildings):
            c3_total += 1
            nec_arr = getattr(b, "net_electricity_consumption", None)
            nec_b = float(nec_arr[t_nec]) if nec_arr is not None and t_nec < len(nec_arr) else 0.0
            nsl_arr = getattr(b, "_Building__energy_to_non_shiftable_load", [])
            sg_arr = getattr(b, "_Building__solar_generation", [])
            nsl_i = float(nsl_arr[t_nec]) if hasattr(nsl_arr, "__len__") and t_nec < len(nsl_arr) else 0.0
            sg_i = float(sg_arr[t_nec]) if hasattr(sg_arr, "__len__") and t_nec < len(sg_arr) else 0.0
            exog = nsl_i + sg_i
            if abs(exog) <= P_bld_max:
                if abs(nec_b) > P_bld_max:
                    c3_violations += 1
            else:
                if abs(nec_b) > abs(exog):
                    c3_violations += 1

        # C4
        c4_total += 1
        if grid_imp > P_grid_max:
            c4_violations += 1

        total_import += grid_imp
        total_export += grid_exp
        peak_nec = max(peak_nec, abs(step_nec))
        ramping_total += abs(step_nec - nec_prev)
        nec_prev = step_nec

        # Electricity cost
        price = 0.0
        for b in city.buildings:
            pricing_obj = getattr(b, "pricing", None)
            if pricing_obj is not None:
                pa = getattr(pricing_obj, "electricity_pricing", None)
                if pa is not None and hasattr(pa, "__len__") and t_nec < len(pa):
                    price = float(pa[t_nec])
                    break
        if grid_imp > 0:
            elec_cost += grid_imp * price

        # C2
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

        # Battery actions
        for bi in batt_indices:
            if bi < len(action):
                a = float(action[bi])
                batt_total += 1
                if a > 0.1: batt_charge += 1; batt_daily_c[day] = True
                elif a < -0.1: batt_discharge += 1; batt_daily_d[day] = True
                if is_solar: batt_solar_acts.append(a)
                if is_peak: batt_peak_acts.append(a)

        # EV actions
        for ei in ev_indices:
            if ei < len(action):
                a = float(action[ei])
                ev_connected += 1
                if a > 0.1: ev_charge += 1
                elif a < -0.1:
                    ev_v2g += 1
                    ev_total_v2g_kwh += abs(a) * 6.0
                    if 17 <= hour <= 23: ev_v2g_peak += 1
                if 17 <= hour <= 23: ev_v2g_peak_conn += 1

        # C0: departures
        t_soc = max(0, t_after - 1)
        for bi, ch, cid in charger_info:
            sim = getattr(ch, "charger_simulation", None)
            if sim is None: continue
            state_arr = getattr(sim, "_electric_vehicle_charger_state", None)
            dep_arr = getattr(sim, "_electric_vehicle_departure_time", None)
            req_arr = getattr(sim, "_electric_vehicle_required_soc_departure", None)
            if state_arr is None or dep_arr is None or req_arr is None: continue
            if t_after >= len(state_arr): continue
            s = float(state_arr[t_after])
            d = float(dep_arr[t_after])
            r = float(req_arr[t_after])
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

        if term or trunc:
            break

        if (step_i + 1) % 2000 == 0:
            ndep = len(departures)
            nviol = sum(1 for _, ds, dr in departures if ds < dr)
            log(f"  Step {step_i+1}/{TOTAL_STEPS}: deps={ndep}, c0_viol={nviol}, "
                f"c3={c3_violations}/{c3_total}, c4={c4_violations}/{c4_total}")

    elapsed = time.time() - t0
    steps = step_i + 1

    # ---- Results ----
    ndep = len(departures)
    nviol_100 = sum(1 for _, ds, dr in departures if ds < dr)
    nviol_50 = sum(1 for _, ds, dr in departures if ds < 0.5 * dr)

    log()
    log("=" * 72)
    log(f"  BARE SAC-Lag (epoch-90) — Deterministic Evaluation")
    log(f"  {steps} steps in {elapsed:.1f}s")
    log("=" * 72)

    log()
    log("  ---- Constraint Violations ----")
    log(f"  C0 (EV departure, 100% threshold):")
    log(f"    Total departures:    {ndep}")
    log(f"    Violated (<100%):    {nviol_100}  ({100*nviol_100/max(ndep,1):.1f}%)")
    log(f"    Violated (<50%):     {nviol_50}  ({100*nviol_50/max(ndep,1):.1f}%)")
    if departures:
        mean_dep = np.mean([ds for _, ds, _ in departures])
        mean_req = np.mean([dr for _, _, dr in departures])
        log(f"    Mean departure SoC:  {mean_dep:.3f}")
        log(f"    Mean required SoC:   {mean_req:.3f}")
        viol_deficits = [max(0, dr - ds) for _, ds, dr in departures if ds < dr]
        if viol_deficits:
            log(f"    Mean deficit (viol): {np.mean(viol_deficits):.3f}")

    log()
    log(f"  C2 (Battery SoC bounds):")
    log(f"    {c2_violations}/{c2_total}  ({100*c2_violations/max(c2_total,1):.1f}%)")

    log()
    log(f"  C3 (Building power, {P_bld_max} kW):")
    log(f"    {c3_violations}/{c3_total}  ({100*c3_violations/max(c3_total,1):.1f}%)")

    log()
    log(f"  C4 (Grid power, {P_grid_max} kW):")
    log(f"    {c4_violations}/{c4_total}  ({100*c4_violations/max(c4_total,1):.1f}%)")

    log()
    log("  ---- Battery Behavior ----")
    bt = max(batt_total, 1)
    log(f"    Charge:     {100*batt_charge/bt:.1f}%")
    log(f"    Discharge:  {100*batt_discharge/bt:.1f}%")
    log(f"    Idle:       {100*(bt-batt_charge-batt_discharge)/bt:.1f}%")
    log(f"    Solar avg:  {np.mean(batt_solar_acts):.3f}" if batt_solar_acts else "    Solar avg:  N/A")
    log(f"    Peak avg:   {np.mean(batt_peak_acts):.3f}" if batt_peak_acts else "    Peak avg:   N/A")
    all_days = set(list(batt_daily_c.keys()) + list(batt_daily_d.keys()))
    cycling = sum(1 for d in all_days if batt_daily_c.get(d) and batt_daily_d.get(d))
    log(f"    Daily cycling days: {cycling}")

    log()
    log("  ---- EV / V2G Behavior ----")
    et = max(ev_connected, 1)
    log(f"    Charge:     {100*ev_charge/et:.1f}%")
    log(f"    V2G:        {100*ev_v2g/et:.1f}%")
    log(f"    V2G peak:   {100*ev_v2g_peak/max(ev_v2g_peak_conn,1):.1f}%")
    log(f"    Total V2G:  {ev_total_v2g_kwh:.0f} kWh")

    log()
    log("  ---- CityLearn KPIs ----")
    mean_imp = total_import / max(steps, 1)
    lf = mean_imp / max(peak_nec, 1e-8)
    log(f"    Total import:     {total_import:.0f} kWh")
    log(f"    Total export:     {total_export:.0f} kWh")
    log(f"    Peak NEC:         {peak_nec:.2f} kW")
    log(f"    Ramping:          {ramping_total:.0f} kWh")
    log(f"    Electricity cost: ${elec_cost:.0f}")
    log(f"    Load factor:      {lf:.4f}")
    log()
    log("=" * 72)


if __name__ == "__main__":
    main()
