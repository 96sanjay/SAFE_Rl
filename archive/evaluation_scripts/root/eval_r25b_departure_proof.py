#!/usr/bin/env python3
"""
EXTENSIVE PROOF: R25b C0 (EV departure) violation rate.

Three independent methods:
  Method 1: Env info dict (cost_ev_departure per step, summed)
  Method 2: Charger simulation arrays (state/departure_time/required_soc)
  Method 3: Raw EV battery SoC at departure timestep

Outputs:
  - Full departure table CSV (every single departure)
  - Summary text file
  - Cross-validation against training progress.csv

Env chain (same as training):
  make_base_env -> CityLearnSafetyEnvV3 -> ForecastObsWrapper -> SauteEVBudgetWrapper
"""
from __future__ import annotations

import csv
import os
import re
import sys
import time
from io import StringIO

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────
# 1) Set env vars from run_r25b_ev_slack_arb.sh
# ──────────────────────────────────────────────────────────────────
PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)

RUN_SCRIPT = os.path.join(PROJECT, "run_r25b_ev_slack_arb.sh")


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
            val_raw = val_raw.replace("${PYTHONPATH:-}", os.environ.get("PYTHONPATH", ""))
            exports[key] = val_raw
    return exports


env_vars = parse_exports(RUN_SCRIPT)
for k, v in env_vars.items():
    os.environ[k] = v

# Ensure critical vars
schema_path = os.path.join(
    PROJECT,
    "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
)
os.environ["CITYLEARN_SCHEMA"] = schema_path
os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"

print("=" * 70)
print("  R25b DEPARTURE PROOF EVALUATION (epoch-80)")
print("=" * 70)
print(f"Schema: {schema_path}")
print(f"Parsed {len(env_vars)} env vars from run script")
print()
print("Key env vars:")
for k in sorted(env_vars.keys()):
    if k.startswith("STEMS_") or k.startswith("CITYLEARN_") or k.startswith("COST_"):
        print(f"  {k}={env_vars[k]}")
print()

# ──────────────────────────────────────────────────────────────────
# 2) Build env chain (EXACT same as training)
# ──────────────────────────────────────────────────────────────────
from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.saute_ev_wrapper import SauteEVBudgetWrapper

base = make_base_env(central_agent=True)
safety = CityLearnSafetyEnvV3(base)
forecast = ForecastObsWrapper(safety, forecast_horizon=24)
env = SauteEVBudgetWrapper(forecast)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(f"Env obs_dim={obs_dim}, act_dim={act_dim}")

# ──────────────────────────────────────────────────────────────────
# 3) Load checkpoint
# ──────────────────────────────────────────────────────────────────
CKPT_PATH = os.path.join(
    PROJECT,
    "runs/r25b_ev_slack_arb/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-17-17-01-55/torch_save/epoch-80.pt",
)
print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

# Build actor
pi_state = ckpt["pi"]
actor = torch.nn.Sequential(
    torch.nn.Linear(obs_dim, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, act_dim),
)

key_map = {
    "mean.0.weight": "0.weight",
    "mean.0.bias": "0.bias",
    "mean.2.weight": "2.weight",
    "mean.2.bias": "2.bias",
    "mean.4.weight": "4.weight",
    "mean.4.bias": "4.bias",
}
actor_sd = {}
for omnisafe_key, seq_key in key_map.items():
    if omnisafe_key not in pi_state:
        raise KeyError(f"Missing key {omnisafe_key} in checkpoint pi state")
    actor_sd[seq_key] = pi_state[omnisafe_key]

actor.load_state_dict(actor_sd)
actor.eval()
print(f"Actor loaded: {sum(p.numel() for p in actor.parameters())} params")

# Load obs normalizer
norm_data = ckpt.get("obs_normalizer")
has_normalizer = norm_data is not None
if has_normalizer:
    norm_mean = norm_data["_mean"].numpy()
    norm_std = norm_data["_std"].numpy()
    norm_clip = norm_data["_clip"].numpy()
    norm_std = np.maximum(norm_std, 1e-8)
    print(f"Obs normalizer loaded: dim={len(norm_mean)}")
else:
    print("WARNING: No obs normalizer in checkpoint")


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    if not has_normalizer:
        return obs
    obs_flat = obs.ravel()
    n = min(len(obs_flat), len(norm_mean))
    normed = np.zeros_like(obs_flat)
    normed[:n] = np.clip(
        (obs_flat[:n] - norm_mean[:n]) / norm_std[:n],
        -norm_clip[:n],
        norm_clip[:n],
    )
    return normed


# ──────────────────────────────────────────────────────────────────
# 4) Walk wrappers to find CityLearnEnv
# ──────────────────────────────────────────────────────────────────
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


city = get_citylearn_env(env)
if city is None:
    print("ERROR: Could not find CityLearnEnv in wrapper chain")
    sys.exit(1)

print(f"CityLearnEnv found: {len(city.buildings)} buildings")

# Discover chargers
charger_info = []  # (building_idx, charger_obj, charger_name)
for bi, b in enumerate(city.buildings):
    for ci, ch in enumerate(getattr(b, "electric_vehicle_chargers", [])):
        cname = f"B{bi}_Ch{ci}"
        charger_info.append((bi, ch, cname))
        print(f"  Charger: {cname} in {b.name}")

print(f"\nTotal chargers discovered: {len(charger_info)}")

# Action names
act_names = []
_e = env
for _ in range(20):
    act_names_raw = getattr(_e, "action_names", None)
    if act_names_raw is not None:
        if isinstance(act_names_raw, list) and len(act_names_raw) > 0 and isinstance(act_names_raw[0], list):
            act_names = [n for sub in act_names_raw for n in sub]
        else:
            act_names = list(act_names_raw)
        break
    _e = getattr(_e, "env", getattr(_e, "base", None))
    if _e is None:
        break

batt_indices = [i for i, n in enumerate(act_names) if n == "electrical_storage"]
ev_indices = [i for i, n in enumerate(act_names) if "electric_vehicle" in n.lower()]
print(f"Action names: {act_names}")
print(f"Battery indices: {batt_indices}, EV indices: {ev_indices}")

# ──────────────────────────────────────────────────────────────────
# 5) Run full year
# ──────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

obs, info = env.reset(seed=42)
print(f"\nReset obs shape: {obs.shape}")

total_steps = 8759

# METHOD 1: Info dict tracking
method1_cost_ev_departure_sum = 0.0
method1_cost_ev_dense_sum = 0.0
method1_ev_departure_count = 0
method1_ev_violation_count_deficit = 0
method1_ev_violation_count_80pct = 0
method1_ev_departure_deficit_kwh = 0.0

# METHOD 2 & 3: Direct charger state reading
# We track every departure in a list of dicts
all_departures = []
departure_counter = 0

# EV action stats
ev_charge_steps = 0
ev_v2g_steps = 0
ev_idle_steps = 0
ev_total_steps = 0

# Previous charger states for detecting transitions
prev_charger_connected = {}  # (bi, ci) -> bool

start_time = time.time()

for step_i in range(total_steps):
    # Normalize obs and get action
    obs_normed = normalize_obs(obs)
    obs_t = torch.as_tensor(obs_normed, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        mean = actor(obs_t)
        action = torch.tanh(mean).squeeze(0).numpy()

    # Step the env
    obs, reward, terminated, truncated, info = env.step(action)

    t_after = int(getattr(city, "time_step", 0))

    # ── METHOD 1: Accumulate from info dict ──
    method1_cost_ev_departure_sum += float(info.get("cost_ev_departure", 0.0))
    method1_cost_ev_dense_sum += float(info.get("cost_ev_dense", 0.0))
    method1_ev_departure_count += int(info.get("ev_departure_departures", 0))
    method1_ev_violation_count_deficit += int(info.get("ev_departure_violation_count_deficit", 0))
    method1_ev_violation_count_80pct += int(info.get("ev_departure_violation_count_80pct", 0))
    method1_ev_departure_deficit_kwh += float(info.get("ev_departure_deficit_kwh", 0.0))

    # ── METHOD 2 & 3: Read charger simulation arrays directly ──
    t_soc = max(0, t_after - 1)

    for bi, ch, cname in charger_info:
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
            # DEPARTURE DETECTED
            departure_counter += 1

            # METHOD 3: Read actual SoC from EV battery
            actual_soc = 0.0
            capacity_kwh = 0.0

            ev = getattr(ch, "connected_electric_vehicle", None)
            if ev is not None:
                batt = getattr(ev, "battery", None)
                if batt is not None:
                    capacity_kwh = float(getattr(batt, "capacity", 0.0))
                    soc_series = getattr(batt, "soc", None)
                    if soc_series is not None and hasattr(soc_series, "__len__"):
                        if 0 <= t_soc < len(soc_series):
                            actual_soc = float(np.clip(soc_series[t_soc], 0.0, 1.0))

            # Also try reading SoC from the simulation's own soc array
            actual_soc_sim = 0.0
            soc_arr_sim = getattr(sim, "_electric_vehicle_soc", None)
            if soc_arr_sim is not None and hasattr(soc_arr_sim, "__len__"):
                if 0 <= t_soc < len(soc_arr_sim):
                    actual_soc_sim = float(np.clip(soc_arr_sim[t_soc], 0.0, 1.0))

            deficit = max(0.0, r - actual_soc)
            deficit_strict = r - actual_soc  # can be negative (over-charged)
            violated_strict = actual_soc < r
            violated_tolerant = actual_soc < (r - 0.001)

            all_departures.append({
                "departure_num": departure_counter,
                "step": step_i + 1,
                "timestep": t_after,
                "building": bi,
                "charger": cname,
                "actual_soc": actual_soc,
                "actual_soc_sim": actual_soc_sim,
                "required_soc": r,
                "deficit": deficit,
                "deficit_signed": deficit_strict,
                "violated_strict": violated_strict,
                "violated_tolerant": violated_tolerant,
                "capacity_kwh": capacity_kwh,
            })

    # EV action stats
    for ei in ev_indices:
        if ei < len(action):
            a = float(action[ei])
            ev_total_steps += 1
            if a > 0.05:
                ev_charge_steps += 1
            elif a < -0.05:
                ev_v2g_steps += 1
            else:
                ev_idle_steps += 1

    if terminated or truncated:
        print(f"Episode ended at step {step_i + 1}")
        break

    if (step_i + 1) % 1000 == 0:
        dep_so_far = len(all_departures)
        viol_so_far = sum(1 for d in all_departures if d["violated_strict"])
        elapsed = time.time() - start_time
        print(f"  Step {step_i + 1}/{total_steps}: "
              f"departures={dep_so_far}, violated_strict={viol_so_far}, "
              f"info_cost_ev={method1_cost_ev_departure_sum:.2f}, "
              f"elapsed={elapsed:.0f}s")

elapsed_total = time.time() - start_time

# ──────────────────────────────────────────────────────────────────
# 6) Save full departure table CSV
# ──────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(PROJECT, "eval_results/r25b_departure_proof.csv")
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "departure_num", "step", "timestep", "building", "charger",
        "actual_soc", "actual_soc_sim", "required_soc", "deficit",
        "deficit_signed", "violated_strict", "violated_tolerant", "capacity_kwh",
    ])
    writer.writeheader()
    for dep in all_departures:
        writer.writerow(dep)

print(f"\nFull departure table saved to: {CSV_PATH}")
print(f"Total rows: {len(all_departures)}")

# ──────────────────────────────────────────────────────────────────
# 7) Compute and display results
# ──────────────────────────────────────────────────────────────────
n_departures = len(all_departures)
n_violated_strict = sum(1 for d in all_departures if d["violated_strict"])
n_violated_tolerant = sum(1 for d in all_departures if d["violated_tolerant"])

if n_departures > 0:
    actual_socs = np.array([d["actual_soc"] for d in all_departures])
    required_socs = np.array([d["required_soc"] for d in all_departures])
    deficits = np.array([d["deficit"] for d in all_departures])
    deficits_signed = np.array([d["deficit_signed"] for d in all_departures])
else:
    actual_socs = np.array([])
    required_socs = np.array([])
    deficits = np.array([])
    deficits_signed = np.array([])

# SoC distribution
soc_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
soc_dist = {}
for lo, hi in soc_bins:
    count = int(np.sum((actual_socs >= lo) & (actual_socs < hi))) if n_departures > 0 else 0
    soc_dist[f"{lo:.1f}-{hi:.1f}"] = count
# Handle exact 1.0
if n_departures > 0:
    soc_dist["0.8-1.0"] += int(np.sum(actual_socs == 1.0))

# Per-charger breakdown
charger_stats = {}
for d in all_departures:
    cn = d["charger"]
    if cn not in charger_stats:
        charger_stats[cn] = {"deps": 0, "viol_strict": 0, "viol_tolerant": 0, "socs": [], "reqs": []}
    charger_stats[cn]["deps"] += 1
    if d["violated_strict"]:
        charger_stats[cn]["viol_strict"] += 1
    if d["violated_tolerant"]:
        charger_stats[cn]["viol_tolerant"] += 1
    charger_stats[cn]["socs"].append(d["actual_soc"])
    charger_stats[cn]["reqs"].append(d["required_soc"])

# Violated departures detail
violated_list = [d for d in all_departures if d["violated_strict"]]

# ──────────────────────────────────────────────────────────────────
# 8) Cross-validation with training progress.csv
# ──────────────────────────────────────────────────────────────────
import pandas as pd
PROGRESS_PATH = os.path.join(
    PROJECT,
    "runs/r25b_ev_slack_arb/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-17-17-01-55/progress.csv",
)
progress = pd.read_csv(PROGRESS_PATH)
final_row = progress.iloc[-1]
train_epcost_0 = float(final_row["Metrics/EpCost_0"])
train_epcost_1 = float(final_row["Metrics/EpCost_1"])
train_epoch = int(final_row["Train/Epoch"])

# ──────────────────────────────────────────────────────────────────
# 9) Build summary
# ──────────────────────────────────────────────────────────────────
summary_lines = []


def p(line=""):
    summary_lines.append(line)
    print(line)


p("=" * 70)
p("  R25b DEPARTURE PROOF - FULL RESULTS")
p("=" * 70)
p(f"  Checkpoint: epoch-80")
p(f"  Eval steps: {total_steps}")
p(f"  Elapsed: {elapsed_total:.1f}s")
p()

p("=" * 70)
p("  METHOD 2 & 3: DIRECT CHARGER STATE ANALYSIS (GROUND TRUTH)")
p("=" * 70)
p(f"  Total departures detected:     {n_departures}")
p(f"  Violated (strict, SoC < req):  {n_violated_strict}  ({100.0 * n_violated_strict / max(n_departures, 1):.2f}%)")
p(f"  Violated (tolerant, SoC < req-0.001): {n_violated_tolerant}  ({100.0 * n_violated_tolerant / max(n_departures, 1):.2f}%)")
p()
if n_departures > 0:
    p(f"  Mean actual SoC at departure:  {np.mean(actual_socs):.6f}")
    p(f"  Mean required SoC:             {np.mean(required_socs):.6f}")
    p(f"  Median actual SoC:             {np.median(actual_socs):.6f}")
    p(f"  Min actual SoC:                {np.min(actual_socs):.6f}")
    p(f"  Max actual SoC:                {np.max(actual_socs):.6f}")
    p(f"  Std actual SoC:                {np.std(actual_socs):.6f}")
    p()
    p(f"  Mean deficit (when violated):  {np.mean(deficits[deficits > 0]):.6f}" if np.any(deficits > 0) else "  Mean deficit (when violated):  N/A (no violations)")
    p(f"  Max deficit:                   {np.max(deficits):.6f}")
    p(f"  Sum of all deficits:           {np.sum(deficits):.6f}")
p()

p("  SoC distribution at departure:")
for bucket, count in soc_dist.items():
    bar = "#" * (count // 5)
    p(f"    [{bucket}): {count:5d}  {bar}")
p()

p("  Per-charger breakdown:")
for cn in sorted(charger_stats.keys()):
    cs = charger_stats[cn]
    mean_soc = np.mean(cs["socs"]) if cs["socs"] else 0.0
    mean_req = np.mean(cs["reqs"]) if cs["reqs"] else 0.0
    p(f"    {cn}: {cs['deps']:4d} departures, "
      f"{cs['viol_strict']:3d} violated_strict ({100 * cs['viol_strict'] / max(cs['deps'], 1):.1f}%), "
      f"mean_soc={mean_soc:.4f}, mean_req={mean_req:.4f}")
p()

if violated_list:
    p(f"  EVERY VIOLATED DEPARTURE (strict):")
    p(f"    {'#':>4}  {'step':>5}  {'bld':>3}  {'charger':>8}  {'actual':>8}  {'required':>8}  {'deficit':>8}")
    for d in violated_list:
        p(f"    {d['departure_num']:4d}  {d['step']:5d}  {d['building']:3d}  {d['charger']:>8}  "
          f"{d['actual_soc']:8.6f}  {d['required_soc']:8.6f}  {d['deficit']:8.6f}")
    p()

p("=" * 70)
p("  METHOD 1: INFO DICT ACCUMULATION")
p("=" * 70)
p(f"  Sum cost_ev_departure:             {method1_cost_ev_departure_sum:.4f}")
p(f"  Sum cost_ev_dense:                 {method1_cost_ev_dense_sum:.4f}")
p(f"  Combined (C1_ev for OmniSafe):     {method1_cost_ev_departure_sum + method1_cost_ev_dense_sum:.4f}")
p(f"  Departure count (from info):       {method1_ev_departure_count}")
p(f"  Deficit violation count (info):    {method1_ev_violation_count_deficit}")
p(f"  80pct violation count (info):      {method1_ev_violation_count_80pct}")
p(f"  Departure deficit kWh (info):      {method1_ev_departure_deficit_kwh:.4f}")
p()
p(f"  NOTE: cost_ev_departure = EV_COST_SCALE({float(os.environ.get('CITYLEARN_EV_COST_SCALE', '1.0'))}) * V3_agent_controllable_deficit")
p(f"  NOTE: cost_ev_dense = Saute dense signal (every step)")
p()

p("=" * 70)
p("  CROSS-VALIDATION: TRAINING progress.csv")
p("=" * 70)
p(f"  Final training epoch: {train_epoch}")
p(f"  Training EpCost_0 (cost_ev_departure): {train_epcost_0:.2f}")
p(f"  Training EpCost_1 (cost_ev_dense):     {train_epcost_1:.2f}")
p(f"  Eval cost_ev_departure sum:            {method1_cost_ev_departure_sum:.2f}")
p(f"  Eval cost_ev_dense sum:                {method1_cost_ev_dense_sum:.2f}")
p()
p(f"  EpCost_0 trajectory (last 10 epochs):")
for i in range(max(0, len(progress) - 10), len(progress)):
    row = progress.iloc[i]
    p(f"    Epoch {int(row['Train/Epoch']):3d}: EpCost_0={row['Metrics/EpCost_0']:8.2f}")
p()

p("=" * 70)
p("  EV ACTION STATISTICS")
p("=" * 70)
et = max(ev_total_steps, 1)
p(f"  EV charge steps (a > 0.05):    {ev_charge_steps:6d}  ({100.0 * ev_charge_steps / et:.1f}%)")
p(f"  EV V2G steps (a < -0.05):      {ev_v2g_steps:6d}  ({100.0 * ev_v2g_steps / et:.1f}%)")
p(f"  EV idle steps:                  {ev_idle_steps:6d}  ({100.0 * ev_idle_steps / et:.1f}%)")
p(f"  EV total steps:                 {ev_total_steps:6d}")
p()

p("=" * 70)
p("  CONCLUSION")
p("=" * 70)
viol_rate = 100.0 * n_violated_strict / max(n_departures, 1)
p(f"  C0 violation rate (strict):  {n_violated_strict}/{n_departures} = {viol_rate:.2f}%")
p(f"  C0 violation rate (tolerant): {n_violated_tolerant}/{n_departures} = {100.0 * n_violated_tolerant / max(n_departures, 1):.2f}%")
p()
if n_departures > 0 and n_violated_strict > 0:
    max_deficit = max(d["deficit"] for d in violated_list)
    mean_deficit = np.mean([d["deficit"] for d in violated_list])
    p(f"  Worst-case deficit:          {max_deficit:.6f} SoC ({max_deficit * 100:.2f}%)")
    p(f"  Mean violated deficit:       {mean_deficit:.6f} SoC ({mean_deficit * 100:.2f}%)")
elif n_departures > 0:
    p(f"  NO VIOLATIONS DETECTED. Agent meets all departure requirements.")
p()

# ──────────────────────────────────────────────────────────────────
# 10) Save summary
# ──────────────────────────────────────────────────────────────────
SUMMARY_PATH = os.path.join(PROJECT, "eval_results/r25b_departure_proof.txt")
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(summary_lines))
    f.write("\n")

print(f"\nSummary saved to: {SUMMARY_PATH}")
print(f"CSV saved to: {CSV_PATH}")
print(f"\nDone.")
