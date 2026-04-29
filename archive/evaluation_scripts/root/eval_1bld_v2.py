#!/usr/bin/env python3
"""
1-Building Cycling Test V2 -- Full Year Evaluation
===================================================
Loads the best 1bld v2 checkpoint (epoch-50) and runs 1 full year (8759 steps)
with deterministic actions (Gaussian mean). Tests whether battery cycling
and V2G behaviour emerged during the 3-month training.

Usage:
    python eval_1bld_v2.py [--epoch 50]
"""
import os

# --- ALL env vars for v2 run (must be set BEFORE any citylearn imports) ---
os.environ["STEMS_MU_ECONOMIC"] = "0.0"
os.environ["STEMS_ALPHA_GRID"] = "0.0"
os.environ["STEMS_ALPHA_BUILD"] = "0.0"
os.environ["STEMS_BETA_RAMP"] = "0.0"
os.environ["STEMS_XI_RENEWABLE"] = "0.0"
os.environ["STEMS_ALPHA_BARRIER"] = "0.0"
os.environ["STEMS_ALPHA_EV_GUARD"] = "0.0"
os.environ["STEMS_ALPHA_V2G_CONTEXT"] = "0.0"
os.environ["STEMS_ALPHA_PEAK_SHAVE"] = "0.0"
os.environ["STEMS_ALPHA_EV_SOLAR"] = "0.0"
os.environ["STEMS_ALPHA_SOLAR_STORE"] = "0.0"
os.environ["STEMS_ALPHA_HEADROOM"] = "0.0"
os.environ["STEMS_ALPHA_LOAD_SHIFT"] = "0.0"
os.environ["STEMS_ALPHA_GRID_MILD"] = "0.0"
os.environ["STEMS_LAMBDA_EV"] = "15.0"
os.environ["STEMS_ALPHA_NEC_SIGN"] = "3.0"
os.environ["STEMS_ALPHA_PRICE_ARB"] = "1.0"
os.environ["STEMS_EV_SLACK_ARB_SCALE"] = "2.5"
os.environ["STEMS_ALPHA_GRID_PENALTY"] = "1.5"
os.environ["CITYLEARN_STEMS_BATTERY_COST_SCALE"] = "1.0"
os.environ["CITYLEARN_EXPORT_FACTOR"] = "0.7"
os.environ["CITYLEARN_EV_SAUTE"] = "1"
os.environ["CITYLEARN_EV_SAUTE_BUDGET"] = "25000"
os.environ["CITYLEARN_EV_SAUTE_SHAPED_ALPHA"] = "2.0"
os.environ["CITYLEARN_EV_SAUTE_PENALTY"] = "5.0"
os.environ["CITYLEARN_EV_SAUTE_GAMMA"] = "1.0"

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# =====================================================================
# ENV VARS -- match run_1bld_test.sh but with FULL YEAR schema
# =====================================================================
def set_env():
    """Clear all CITYLEARN/STEMS env vars and set 1bld eval vars."""
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k, None)

    env_vars = {
        # --- Schema (FULL YEAR, not 3month) ---
        "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_1building.json",
        "CITYLEARN_CENTRAL_AGENT": "1",

        # --- Action mask ---
        "CITYLEARN_ACTION_MASK": "1",

        # --- KL regularization (same as training, though doesn't affect eval) ---
        "CITYLEARN_KL_BETA": "0.1",
        "CITYLEARN_KL_BETA_DECAY": "0.995",

        # --- NEC-sign reward (v2 weights exactly) ---
        "STEMS_ALPHA_NEC_SIGN": "3.0",
        "STEMS_ALPHA_PRICE_ARB": "1.0",
        "STEMS_LAMBDA_EV": "15.0",
        "STEMS_EV_SLACK_ARB_SCALE": "2.5",
        "STEMS_ALPHA_GRID_PENALTY": "1.5",
        "STEMS_ALPHA_GRID_PENALTY_SOLAR": "0.0",
        "CITYLEARN_STEMS_BATTERY_COST_SCALE": "1.0",
        "CITYLEARN_EXPORT_FACTOR": "0.7",

        # --- All other STEMS terms = 0.0 ---
        "STEMS_MU_ECONOMIC": "0.0",
        "STEMS_ALPHA_GRID": "0.0",
        "STEMS_ALPHA_BUILD": "0.0",
        "STEMS_BETA_RAMP": "0.0",
        "STEMS_XI_RENEWABLE": "0.0",
        "STEMS_ALPHA_BARRIER": "0.0",
        "STEMS_ALPHA_EV_GUARD": "0.0",
        "STEMS_ALPHA_V2G_CONTEXT": "0.0",
        "STEMS_ALPHA_PEAK_SHAVE": "0.0",
        "STEMS_ALPHA_EV_SOLAR": "0.0",
        "STEMS_ALPHA_SOLAR_STORE": "0.0",
        "STEMS_ALPHA_HEADROOM": "0.0",
        "STEMS_ALPHA_LOAD_SHIFT": "0.0",
        "STEMS_ALPHA_GRID_MILD": "0.0",
        "STEMS_ALPHA_NEC": "0.0",
        "STEMS_ALPHA_PEAK": "0.0",
        "STEMS_ALPHA_RAMP": "0.0",
        "STEMS_ALPHA_CARBON": "0.0",
        "STEMS_ALPHA_LOAD": "0.0",

        # --- C1 tolerance ---
        "CITYLEARN_C1_SOC_TOLERANCE": "0.20",

        # --- EV Saute ---
        "CITYLEARN_EV_SAUTE": "1",
        "CITYLEARN_EV_SAUTE_BUDGET": "25000",
        "CITYLEARN_EV_SAUTE_SHAPED_ALPHA": "2.0",
        "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
        "CITYLEARN_EV_SAUTE_GAMMA": "1.0",

        # --- Disable features not used in 1bld training ---
        "CITYLEARN_SPATIAL_OBS": "0",
        "CITYLEARN_TEMPORAL_WINDOW": "0",
    }

    for k, v in env_vars.items():
        os.environ[k] = v


# =====================================================================
# Model
# =====================================================================
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

    # Print log_std for info
    log_std = pi_state.get("log_std", None)
    if log_std is not None:
        print(f"  Policy log_std: {log_std.numpy()}")
        print(f"  Policy std:     {log_std.exp().numpy()}")

    return actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim


def pct(a, b):
    return 100.0 * a / b if b > 0 else 0.0


# =====================================================================
# Main evaluation
# =====================================================================
CKPT_DIR = (
    f"{PROJECT}/runs/test_1bld_v2/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-19-22-56-55/torch_save"
)

OUT_DIR = f"{PROJECT}/runs/test_1bld_v2"


def evaluate(epoch=50, seed=42):
    ckpt_path = os.path.join(CKPT_DIR, f"epoch-{epoch}.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"\n{'='*72}")
    print(f"  1-BUILDING V2 CYCLING TEST -- FULL YEAR EVALUATION")
    print(f"  Checkpoint: epoch-{epoch}.pt")
    print(f"  Seed: {seed}")
    print(f"{'='*72}")

    # --- Load model ---
    actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(ckpt_path)
    print(f"  Actor: obs_dim={obs_dim}, act_dim={act_dim}, hidden=[256,256]")
    print(f"  Obs normalizer: {'YES' if obs_mean is not None else 'NO'}")

    # --- Create environment (same wrapper chain as training) ---
    set_env()

    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.action_mask_wrapper import ActionMaskWrapper
    from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    forecast = ForecastObsWrapper(safety, forecast_horizon=24)
    env = ActionMaskWrapper(forecast)

    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))
    n_buildings = len(buildings)

    # --- Identify action indices ---
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)

    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

    print(f"\n  Buildings: {n_buildings}")
    print(f"  Obs dim (env): {env.observation_space.shape[0]}  (model expects: {obs_dim})")
    print(f"  Act dim: {act_dim}")
    print(f"  Action layout:")
    for i, n in enumerate(names):
        tag = ""
        if i in batt_idx:
            tag = "  [BATTERY]"
        elif i in ev_idx:
            tag = "  [EV]"
        print(f"    [{i}] {n}{tag}")

    # Handle obs dim mismatch
    env_obs_dim = env.observation_space.shape[0]
    need_pad = obs_dim > env_obs_dim
    pad_dim = obs_dim - env_obs_dim if need_pad else 0
    need_trim = obs_dim < env_obs_dim
    if need_pad:
        print(f"  WARNING: Model expects {obs_dim} obs, env gives {env_obs_dim}. Padding {pad_dim} dims.")
    if need_trim:
        print(f"  WARNING: Model expects {obs_dim} obs, env gives {env_obs_dim}. Trimming.")

    # --- Reset ---
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    # =====================================================================
    # TRACKING VARIABLES
    # =====================================================================
    rewards = []
    costs = []
    actions_all = []

    # Per-constraint costs
    c0_vals, c1_vals, c2_vals, c3_vals, c4_vals = [], [], [], [], []

    # Battery tracking
    batt_actions_by_hour = [[] for _ in range(24)]
    batt_soc_all = []

    # EV tracking
    ev_actions_by_hour = [[] for _ in range(24)]
    ev_connected_mask = []
    ev_soc_all = []

    # Grid tracking
    nec_per_building = [[] for _ in range(n_buildings)]
    grid_nec_all = []
    price_all = []
    hour_all = []

    # C3 violations
    P_BUILDING_MAX = 4.6083  # default threshold
    c3_violations = 0
    c3_total_checks = 0

    # EV departure tracking
    total_departures = 0
    violated_departures = 0
    departure_deficits = []
    ev_tracker = {}

    # =====================================================================
    # ROLLOUT
    # =====================================================================
    print(f"\n  Running evaluation rollout (8759 steps)...")

    while not done:
        # --- Prepare observation for model ---
        obs_np = np.asarray(obs, dtype=np.float32).ravel()
        if need_pad:
            obs_np = np.concatenate([obs_np, np.zeros(pad_dim, dtype=np.float32)])
        elif need_trim:
            obs_np = obs_np[:obs_dim]

        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        if obs_mean is not None:
            obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
            if obs_clip is not None:
                obs_t = obs_t.clamp(-obs_clip, obs_clip)

        with torch.no_grad():
            action_t = actor(obs_t).squeeze(0).numpy()
        action = np.clip(action_t, -1.0, 1.0)

        actions_all.append(action.copy())

        t_now = int(getattr(raw, "time_step", 0))
        t_idx = max(0, t_now - 1)
        hour = t_now % 24
        hour_all.append(hour)

        # --- Get price ---
        try:
            pr = buildings[0].pricing.electricity_pricing
            price = float(pr[t_idx]) if t_idx < len(pr) else 0.0
        except Exception:
            price = 0.0
        price_all.append(price)

        # --- Battery SoC ---
        batt_soc_step = []
        for b_idx, bld in enumerate(buildings):
            try:
                es = getattr(bld, "electrical_storage", None)
                soc_arr = getattr(es, "soc", None) if es else None
                if soc_arr is not None and len(soc_arr) > t_idx:
                    s = float(np.clip(soc_arr[t_idx], 0, 1))
                else:
                    s = 0.5
            except Exception:
                s = 0.5
            batt_soc_step.append(s)
        batt_soc_all.append(batt_soc_step)

        # --- Battery actions by hour ---
        for bi in batt_idx:
            if bi < len(action):
                batt_actions_by_hour[hour].append(float(action[bi]))

        # --- EV tracking ---
        ev_soc_step = []
        ev_connected_step = []
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, "electric_vehicle_chargers", None) or []
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                sim = getattr(ch, 'charger_simulation',
                              getattr(ch, '_Charger__charger_simulation', None))
                if sim is None:
                    ev_connected_step.append(False)
                    ev_soc_step.append(0.0)
                    continue
                try:
                    sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                    current_connected = t_now < len(sa) and float(sa[t_now]) == 1.0
                    ev_connected_step.append(current_connected)

                    # Get EV SoC
                    ev_soc = 0.0
                    if current_connected:
                        ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                        if ev_obj is not None:
                            bt = getattr(ev_obj, 'battery', None)
                            if bt is not None:
                                soc_arr = getattr(bt, 'soc', None)
                                if soc_arr is not None:
                                    sn = np.asarray(soc_arr, dtype=float)
                                    ev_soc = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0
                    ev_soc_step.append(ev_soc)

                    # Departure tracking
                    ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                    if current_connected:
                        rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                        if not np.isfinite(rs):
                            rs = 1.0
                        ev_tracker[key] = {'was_connected': True, 'last_soc': ev_soc, 'required_soc': rs}
                    else:
                        prev = ev_tracker.get(key, {})
                        if prev.get('was_connected', False):
                            total_departures += 1
                            last_soc = prev['last_soc']
                            rs = prev['required_soc']
                            deficit = max(0.0, rs - last_soc)
                            if deficit > 0.01:
                                violated_departures += 1
                            departure_deficits.append(deficit)
                        ev_tracker[key] = {'was_connected': False}
                except Exception:
                    ev_connected_step.append(False)
                    ev_soc_step.append(0.0)

        ev_connected_mask.append(ev_connected_step)
        ev_soc_all.append(ev_soc_step)

        # EV actions by hour (only for connected EVs)
        ev_charger_i = 0
        for ei in ev_idx:
            if ei < len(action):
                if ev_charger_i < len(ev_connected_step) and ev_connected_step[ev_charger_i]:
                    ev_actions_by_hour[hour].append(float(action[ei]))
                ev_charger_i += 1

        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        rewards.append(reward)
        costs.append(info.get("cost", 0.0))

        # Per-constraint costs from info
        c0_vals.append(info.get("cost_ev_departure", 0.0))
        c1_vals.append(info.get("cost_ev_dense", 0.0))
        c2_vals.append(info.get("cost_stems_battery", 0.0))
        c3_vals.append(info.get("cost_stems_building_power", 0.0))
        c4_vals.append(info.get("cost_stems_grid_power", 0.0))

        # --- NEC per building (after step) ---
        t_after = max(0, int(getattr(raw, "time_step", 0)) - 1)
        for b_idx, bld in enumerate(buildings):
            try:
                nec = getattr(bld, 'net_electricity_consumption', None)
                if nec is not None and len(nec) > t_after:
                    p = float(nec[t_after])
                else:
                    p = 0.0
            except Exception:
                p = 0.0
            nec_per_building[b_idx].append(p)
            c3_total_checks += 1
            if abs(p) > P_BUILDING_MAX:
                c3_violations += 1

        # Grid NEC
        total_nec = sum(nec_per_building[b][-1] for b in range(n_buildings))
        grid_nec_all.append(total_nec)

        if step % 2000 == 0:
            print(f"    Step {step}/{8759}...")

    print(f"  Rollout complete: {step} steps")

    # =====================================================================
    # COMPUTE METRICS
    # =====================================================================
    actions_arr = np.array(actions_all)  # (T, act_dim)

    # --- Battery metrics ---
    batt_actions = actions_arr[:, batt_idx] if batt_idx else np.array([])
    if batt_actions.size > 0:
        batt_flat = batt_actions.ravel()
        n_batt_total = len(batt_flat)
        batt_charge_pct = pct(np.sum(batt_flat > 0.1), n_batt_total)
        batt_discharge_pct = pct(np.sum(batt_flat < -0.1), n_batt_total)
        batt_idle_pct = pct(np.sum(np.abs(batt_flat) <= 0.1), n_batt_total)

        # Solar hours (10-15) and peak hours (17-21)
        solar_hours = set(range(10, 16))
        peak_hours = set(range(17, 22))
        solar_batt = [a for h, acts in enumerate(batt_actions_by_hour) if h in solar_hours for a in acts]
        peak_batt = [a for h, acts in enumerate(batt_actions_by_hour) if h in peak_hours for a in acts]
        avg_solar_batt = np.mean(solar_batt) if solar_batt else 0.0
        avg_peak_batt = np.mean(peak_batt) if peak_batt else 0.0

        cycling = batt_charge_pct > 20.0 and batt_discharge_pct > 20.0
    else:
        batt_charge_pct = batt_discharge_pct = batt_idle_pct = 0.0
        avg_solar_batt = avg_peak_batt = 0.0
        batt_flat = np.array([])
        cycling = False

    # --- EV metrics ---
    ev_actions = actions_arr[:, ev_idx] if ev_idx else np.array([])
    if ev_actions.size > 0:
        ev_connected_total = 0
        ev_charge_connected = 0
        ev_discharge_connected = 0
        ev_idle_connected = 0
        ev_peak_connected = 0
        ev_peak_discharge = 0

        for t in range(len(actions_all)):
            h = hour_all[t]
            for ei_local, ei in enumerate(ev_idx):
                if ei_local < len(ev_connected_mask[t]) and ev_connected_mask[t][ei_local]:
                    a = float(actions_all[t][ei])
                    ev_connected_total += 1
                    if a > 0.1:
                        ev_charge_connected += 1
                    elif a < -0.1:
                        ev_discharge_connected += 1
                    else:
                        ev_idle_connected += 1
                    if 17 <= h <= 23:
                        ev_peak_connected += 1
                        if a < -0.1:
                            ev_peak_discharge += 1

        ev_charge_pct = pct(ev_charge_connected, ev_connected_total)
        ev_discharge_pct = pct(ev_discharge_connected, ev_connected_total)
        ev_idle_pct = pct(ev_idle_connected, ev_connected_total)
        v2g_peak_pct = pct(ev_peak_discharge, ev_peak_connected)
    else:
        ev_charge_pct = ev_discharge_pct = ev_idle_pct = v2g_peak_pct = 0.0
        ev_connected_total = 0

    # C0
    c0_violation_rate = pct(violated_departures, total_departures)

    # C3
    c3_violation_rate = pct(c3_violations, c3_total_checks)

    # Battery SoC stats
    batt_soc_arr = np.array(batt_soc_all)
    soc_means = batt_soc_arr.mean(axis=0)
    soc_mins = batt_soc_arr.min(axis=0)
    soc_maxs = batt_soc_arr.max(axis=0)
    soc_ranges = soc_maxs - soc_mins

    # Manual energy metrics
    total_import = sum(max(0, nec) for necs in nec_per_building for nec in necs)
    total_export = sum(abs(min(0, nec)) for necs in nec_per_building for nec in necs)

    # =====================================================================
    # PRINT RESULTS
    # =====================================================================
    print(f"\n{'='*72}")
    print(f"  1-BUILDING V2 EVALUATION RESULTS (epoch-{epoch}, full year)")
    print(f"{'='*72}")

    print(f"\n  --- Episode Summary ---")
    print(f"  Steps:          {step}")
    print(f"  Total reward:   {sum(rewards):.2f}")
    print(f"  Mean reward:    {np.mean(rewards):.4f}")
    print(f"  Total cost:     {sum(costs):.2f}")

    print(f"\n  --- Per-Constraint Cumulative Costs ---")
    print(f"  C0 (EV departure):     {sum(c0_vals):.1f}")
    print(f"  C1 (EV dense):         {sum(c1_vals):.1f}")
    print(f"  C2 (battery SoC):      {sum(c2_vals):.1f}")
    print(f"  C3 (building power):   {sum(c3_vals):.1f}")
    print(f"  C4 (grid power):       {sum(c4_vals):.1f}")

    print(f"\n  {'='*60}")
    print(f"  BATTERY METRICS")
    print(f"  {'='*60}")
    print(f"  Charge   (action > 0.1):    {batt_charge_pct:.1f}%")
    print(f"  Discharge(action < -0.1):   {batt_discharge_pct:.1f}%")
    print(f"  Idle     (|action| <= 0.1): {batt_idle_pct:.1f}%")
    print(f"  Avg action during solar hours (10-15):  {avg_solar_batt:+.4f}")
    print(f"  Avg action during peak hours  (17-21):  {avg_peak_batt:+.4f}")
    if batt_flat.size > 0:
        print(f"  Action mean: {batt_flat.mean():.4f}, std: {batt_flat.std():.4f}")
        print(f"  Action min:  {batt_flat.min():.4f}, max: {batt_flat.max():.4f}")
    print(f"  *** CYCLING EMERGED: {'YES' if cycling else 'NO'} ***")
    print(f"      (requires charge > 20% AND discharge > 20%)")

    print(f"\n  Battery SoC:")
    for b in range(n_buildings):
        print(f"    Bld {b}: mean={soc_means[b]:.3f}, "
              f"min={soc_mins[b]:.3f}, max={soc_maxs[b]:.3f}, "
              f"range={soc_ranges[b]:.3f}")

    print(f"\n  {'='*60}")
    print(f"  EV METRICS")
    print(f"  {'='*60}")
    print(f"  Connected steps:          {ev_connected_total}")
    print(f"  Charge (action > 0.1):    {ev_charge_pct:.1f}%  ({ev_charge_connected} steps)")
    print(f"  V2G/discharge (< -0.1):   {ev_discharge_pct:.1f}%  ({ev_discharge_connected} steps)")
    print(f"  Idle (|action| <= 0.1):   {ev_idle_pct:.1f}%  ({ev_idle_connected} steps)")
    print(f"  V2G during peak (17-23):  {v2g_peak_pct:.1f}%  ({ev_peak_discharge}/{ev_peak_connected})")
    print(f"  C0 departure violations:  {violated_departures}/{total_departures} "
          f"({c0_violation_rate:.1f}%)")
    if departure_deficits:
        print(f"  Mean deficit at departure: {np.mean(departure_deficits):.4f}")
        print(f"  Max deficit at departure:  {np.max(departure_deficits):.4f}")

    print(f"\n  {'='*60}")
    print(f"  CONSTRAINT METRICS")
    print(f"  {'='*60}")
    print(f"  C3 building power violations: {c3_violations}/{c3_total_checks} "
          f"({c3_violation_rate:.2f}%)")

    print(f"\n  {'='*60}")
    print(f"  ENERGY METRICS")
    print(f"  {'='*60}")
    print(f"  Total import: {total_import:.1f} kWh")
    print(f"  Total export: {total_export:.1f} kWh")

    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    print(f"\n  Generating visualization...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"1-Building V2 Cycling Test (epoch-{epoch}, full year)\n"
        f"Cycling={'YES' if cycling else 'NO'} | "
        f"Charge={batt_charge_pct:.1f}% | Discharge={batt_discharge_pct:.1f}% | "
        f"V2G={ev_discharge_pct:.1f}%",
        fontsize=12, fontweight='bold'
    )

    # --- Panel 1: Battery action by hour of day (24-bar chart) ---
    ax = axes[0]
    hours = list(range(24))
    batt_hourly_mean = [np.mean(batt_actions_by_hour[h]) if batt_actions_by_hour[h] else 0.0 for h in hours]
    batt_hourly_std = [np.std(batt_actions_by_hour[h]) if batt_actions_by_hour[h] else 0.0 for h in hours]
    bm = np.array(batt_hourly_mean)
    bs = np.array(batt_hourly_std)
    colors = ['green' if v > 0 else 'red' for v in bm]
    ax.bar(hours, bm, color=colors, alpha=0.7, edgecolor='none')
    ax.fill_between(hours, bm - bs, bm + bs, alpha=0.15, color='gray', label='1-sigma')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvspan(10, 15, alpha=0.08, color='gold', label='Solar hours (10-15)')
    ax.axvspan(17, 21, alpha=0.08, color='salmon', label='Peak hours (17-21)')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Battery Action")
    ax.set_title(
        f"Battery Action by Hour\n"
        f"Solar avg={avg_solar_batt:+.3f}, Peak avg={avg_peak_batt:+.3f}"
    )
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xticks(hours)
    ax.set_xlim(-0.5, 23.5)

    # --- Panel 2: Battery action distribution (histogram) ---
    ax = axes[1]
    if batt_flat.size > 0:
        ax.hist(batt_flat, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
        ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.6, label='Charge threshold')
        ax.axvline(x=-0.1, color='red', linestyle='--', alpha=0.6, label='Discharge threshold')
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel("Battery Action Value")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Battery Action Distribution\n"
        f"Charge={batt_charge_pct:.1f}%, Idle={batt_idle_pct:.1f}%, "
        f"Discharge={batt_discharge_pct:.1f}%"
    )
    ax.legend(fontsize=7)

    # --- Panel 3: Battery action + price overlay for 1 week ---
    ax = axes[2]
    week_len = min(168, step)
    t_range = np.arange(week_len)

    batt_avg_per_step = batt_actions[:week_len].mean(axis=1) if batt_actions.ndim > 1 else batt_actions[:week_len].ravel()
    prices_week = np.array(price_all[:week_len])

    ax2 = ax.twinx()
    ax.plot(t_range, prices_week, color='darkgoldenrod', alpha=0.6, linewidth=0.8, label='Price')
    ax2.plot(t_range, batt_avg_per_step, color='steelblue', alpha=0.7, linewidth=0.8, label='Batt action')
    ax2.axhline(y=0, color='gray', linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Hour (first week)")
    ax.set_ylabel("Electricity Price", color='darkgoldenrod')
    ax2.set_ylabel("Battery Action", color='steelblue')
    ax.set_title("Price vs Battery Action (Week 1)")

    # Correlation
    if len(prices_week) > 10:
        valid = np.isfinite(prices_week) & np.isfinite(batt_avg_per_step)
        if valid.sum() > 10:
            corr = np.corrcoef(prices_week[valid], batt_avg_per_step[valid])[0, 1]
            ax.text(0.02, 0.95, f"corr={corr:.3f}", transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "eval_v2.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"  Saved visualization: {out_path}")
    plt.close()

    print(f"\n{'='*72}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*72}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    evaluate(epoch=args.epoch, seed=args.seed)
