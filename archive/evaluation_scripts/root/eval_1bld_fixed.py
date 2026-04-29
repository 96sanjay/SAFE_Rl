#!/usr/bin/env python3
"""
Fixed 1-Building Evaluation Script
====================================
Replicates the EXACT OmniSafe training action/obs pipeline:

  Training wrapper chain (inside OmniSafe adapter):
    CityLearnCMDPv2 (contains: make_base_env -> SafetyEnvV3 -> ForecastObs -> [Saute])
      -> AutoReset -> ObsNormalize -> [RewardNormalize] -> [CostNormalize] -> ActionScale -> Unsqueeze

  Key transformations this script replicates:
    1. Obs normalization: (obs - mean) / std, clipped to [-clip, clip]
       (frozen from checkpoint, no online updates during eval)
    2. ActionScale: maps agent's [-1,1] output to env's actual action bounds
       For actions with bounds [-1,1]: identity
       For actions with bounds [0,1] (e.g., washing_machine): (a+1)/2
    3. WM disable: clamp washing_machine actions to 0 (CITYLEARN_WM_DISABLE=1)
    4. Battery/EV clamp: optional per env vars (usually OFF)
    5. NO tanh on actor output (OmniSafe GaussianLearningActor uses raw MLP mean)

Usage:
    python eval_1bld_fixed.py --run-script run_r25b_c2_tight.sh [--epoch 50] [--schema-override path]
    python eval_1bld_fixed.py --run-script run_r25b_loadshift_test.sh --epoch 50
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────
# 0) Parse arguments
# ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fixed 1-building eval")
parser.add_argument("--run-script", required=True,
                    help="Path to the .sh training script (for env var parsing)")
parser.add_argument("--ckpt-dir", default=None,
                    help="Override checkpoint directory (default: auto-detect from run log dir)")
parser.add_argument("--epoch", type=int, default=None,
                    help="Epoch to load (default: latest)")
parser.add_argument("--schema-override", default=None,
                    help="Override schema path (e.g., full year instead of 3-month)")
parser.add_argument("--steps", type=int, default=None,
                    help="Max steps (default: from schema)")
args = parser.parse_args()

PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)


# ──────────────────────────────────────────────────────────────────
# 1) Parse env vars from run script
# ──────────────────────────────────────────────────────────────────
def parse_exports(script_path: str) -> dict[str, str]:
    """Extract 'export KEY=VALUE' and 'export KEY="VALUE"' from bash script."""
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
            # Handle quoted values
            if val_raw.startswith('"'):
                end_q = val_raw.find('"', 1)
                if end_q > 0:
                    val_raw = val_raw[1:end_q]
            else:
                # Remove trailing comments
                for sep in ["  #", " #", "\t#"]:
                    ci = val_raw.find(sep)
                    if ci >= 0:
                        val_raw = val_raw[:ci]
                val_raw = val_raw.strip().strip('"').strip("'")
            # Expand $PROJECT and ${PYTHONPATH:-}
            val_raw = val_raw.replace("$PROJECT", PROJECT)
            val_raw = val_raw.replace("${PYTHONPATH:-}", os.environ.get("PYTHONPATH", ""))
            exports[key] = val_raw
    return exports


run_script_path = os.path.abspath(args.run_script)
env_vars = parse_exports(run_script_path)

# Apply ALL parsed env vars
for k, v in env_vars.items():
    os.environ[k] = v

if args.schema_override:
    os.environ["CITYLEARN_SCHEMA"] = os.path.abspath(args.schema_override)

# Ensure critical vars
os.environ.setdefault("CITYLEARN_CENTRAL_AGENT", "1")
os.environ.setdefault("CITYLEARN_REWARD_TYPE", "stems")

schema_path = os.environ["CITYLEARN_SCHEMA"]

print("=" * 70)
print("  Fixed 1-Building Evaluation")
print("=" * 70)
print(f"Run script: {run_script_path}")
print(f"Schema: {schema_path}")
print(f"Parsed {len(env_vars)} env vars")
print()
print("Key env vars:")
for k in sorted(env_vars.keys()):
    if any(k.startswith(p) for p in ("STEMS_", "CITYLEARN_", "COST_")):
        print(f"  {k}={env_vars[k]}")
print()

# ──────────────────────────────────────────────────────────────────
# 2) Build env chain (EXACT same as CityLearnCMDPv2.__init__)
# ──────────────────────────────────────────────────────────────────
from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper

base = make_base_env(central_agent=True)
safety = CityLearnSafetyEnvV3(base)
forecast = ForecastObsWrapper(safety, forecast_horizon=24)
env_final = forecast

# Saute wrapper (same logic as CityLearnCMDPv2.__init__)
if os.environ.get("CITYLEARN_EV_SAUTE", "0") == "1":
    from citylearn_safe.saute_ev_wrapper import SauteEVBudgetWrapper
    env_final = SauteEVBudgetWrapper(env_final)

env = env_final
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(f"Env obs_dim={obs_dim}, act_dim={act_dim}")
print(f"Action space low={env.action_space.low}, high={env.action_space.high}")

# ──────────────────────────────────────────────────────────────────
# 3) Discover action indices (same as CityLearnCMDPv2)
# ──────────────────────────────────────────────────────────────────
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


city = get_citylearn_env(env)
if city is None:
    print("ERROR: Could not find CityLearnEnv in wrapper chain")
    sys.exit(1)
print(f"CityLearnEnv found: {len(city.buildings)} buildings")

# Get flattened action names
names_raw = getattr(city, "action_names", [])
if isinstance(names_raw, list) and len(names_raw) > 0 and isinstance(names_raw[0], list):
    flat_names = [n for sub in names_raw for n in sub]
else:
    flat_names = list(names_raw)
print(f"Action names: {flat_names}")

# WM action indices
wm_disable = os.environ.get("CITYLEARN_WM_DISABLE", "0") == "1"
wm_indices = [i for i, n in enumerate(flat_names) if "washing_machine" in str(n).lower()]
print(f"WM disable={wm_disable}, indices={wm_indices}")

# Battery/EV action indices (for reporting only)
batt_indices = [i for i, n in enumerate(flat_names) if n == "electrical_storage"]
ev_indices = [i for i, n in enumerate(flat_names) if "electric_vehicle" in n.lower()]
print(f"Battery indices: {batt_indices}, EV indices: {ev_indices}")

# Battery clamp config
batt_clamp_enabled = os.environ.get("CITYLEARN_BATT_CLAMP", "1") == "1"
ev_clamp_enabled = os.environ.get("CITYLEARN_EV_ACTION_CLAMP", "0") == "1"
print(f"Battery clamp={batt_clamp_enabled}, EV clamp={ev_clamp_enabled}")

# ActionScale parameters (replicating OmniSafe ActionScale wrapper)
# Maps from [-1, 1] (agent output space) to env's actual action bounds
act_low = env.action_space.low.astype(np.float32)
act_high = env.action_space.high.astype(np.float32)


def action_scale(a: np.ndarray) -> np.ndarray:
    """Replicate OmniSafe's ActionScale: map [-1, 1] -> [act_low, act_high]."""
    return act_low + (act_high - act_low) * (a - (-1.0)) / 2.0


# Charger info for departure tracking
charger_info = []
for bi, b in enumerate(city.buildings):
    for ch in getattr(b, "electric_vehicle_chargers", []):
        cid = getattr(ch, "charger_id", f"charger_{bi}")
        charger_info.append((bi, ch, cid))
        print(f"  Charger: {cid} in {b.name}")

if not charger_info:
    print("WARNING: No EV chargers found")

# ──────────────────────────────────────────────────────────────────
# 4) Find and load checkpoint
# ──────────────────────────────────────────────────────────────────
if args.ckpt_dir:
    ckpt_dir = args.ckpt_dir
else:
    # Auto-detect: parse --cfg from run script to find log_dir, then find latest run
    cfg_path = None
    with open(run_script_path) as f:
        for line in f:
            m = re.search(r'--cfg\s+(\S+)', line)
            if m:
                cfg_path = os.path.join(PROJECT, m.group(1))
                break

    if cfg_path and os.path.exists(cfg_path):
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        log_dir_raw = cfg.get("logger_cfgs", {}).get("log_dir", "./runs")
        log_dir = os.path.join(PROJECT, log_dir_raw.lstrip("./"))
    else:
        # Fallback: guess from run script name
        script_name = os.path.basename(run_script_path).replace("run_", "").replace(".sh", "")
        log_dir = os.path.join(PROJECT, "runs", script_name)

    # Find the PPOLagMulti run directory
    algo_dirs = []
    for root, dirs, files in os.walk(log_dir):
        if "torch_save" in dirs:
            algo_dirs.append(os.path.join(root, "torch_save"))

    if not algo_dirs:
        print(f"ERROR: No torch_save found under {log_dir}")
        sys.exit(1)

    # Use the most recent one (by directory name which contains timestamp)
    ckpt_dir = sorted(algo_dirs)[-1]

print(f"Checkpoint dir: {ckpt_dir}")

# Find epoch
epoch_files = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch-") and f.endswith(".pt")]
epochs = sorted([int(f.replace("epoch-", "").replace(".pt", "")) for f in epoch_files])

if args.epoch is not None:
    if args.epoch not in epochs:
        print(f"ERROR: epoch-{args.epoch}.pt not found. Available: {epochs}")
        sys.exit(1)
    load_epoch = args.epoch
else:
    load_epoch = epochs[-1]

ckpt_path = os.path.join(ckpt_dir, f"epoch-{load_epoch}.pt")
print(f"Loading: epoch-{load_epoch}")

ckpt = torch.load(ckpt_path, map_location="cpu")

# ──────────────────────────────────────────────────────────────────
# 5) Build actor MLP (same architecture as OmniSafe default)
# ──────────────────────────────────────────────────────────────────
pi_state = ckpt["pi"]

# Infer hidden sizes from weight shapes
w0 = pi_state["mean.0.weight"]
in_dim = w0.shape[1]
h0 = w0.shape[0]
w2 = pi_state["mean.2.weight"]
h1 = w2.shape[0]
w4 = pi_state["mean.4.weight"]
out_dim = w4.shape[0]

if in_dim != obs_dim:
    print(f"WARNING: Checkpoint input dim ({in_dim}) != env obs dim ({obs_dim})")
    print(f"  This likely means env vars don't match training configuration!")
    sys.exit(1)

if out_dim != act_dim:
    print(f"WARNING: Checkpoint output dim ({out_dim}) != env act dim ({act_dim})")
    sys.exit(1)

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
actor_sd = {}
for omnisafe_key, seq_key in key_map.items():
    if omnisafe_key not in pi_state:
        raise KeyError(f"Missing key {omnisafe_key} in checkpoint")
    actor_sd[seq_key] = pi_state[omnisafe_key]

actor.load_state_dict(actor_sd)
actor.eval()
print(f"Actor: [{in_dim}] -> [{h0}] -> [{h1}] -> [{out_dim}], "
      f"{sum(p.numel() for p in actor.parameters())} params")

# Also extract log_std for reporting
log_std = pi_state.get("log_std", None)
if log_std is not None:
    print(f"Policy std: {torch.exp(log_std).numpy()}")

# ──────────────────────────────────────────────────────────────────
# 6) Load obs normalizer (frozen -- NO online updates during eval)
# ──────────────────────────────────────────────────────────────────
norm_data = ckpt.get("obs_normalizer")
has_normalizer = norm_data is not None
if has_normalizer:
    if isinstance(norm_data, dict):
        norm_mean = norm_data["_mean"].numpy().astype(np.float64)
        norm_std = norm_data["_std"].numpy().astype(np.float64)
        norm_clip = norm_data["_clip"].numpy().astype(np.float64)
    else:
        # It's an OrderedDict from state_dict
        norm_mean = norm_data["_mean"].numpy().astype(np.float64)
        norm_std = norm_data["_std"].numpy().astype(np.float64)
        norm_clip = norm_data["_clip"].numpy().astype(np.float64)

    # OmniSafe Normalizer uses min_std=0.01 (line 139 of normalizer.py)
    norm_std = np.maximum(norm_std, 0.01)

    if len(norm_mean) != obs_dim:
        print(f"WARNING: Normalizer dim ({len(norm_mean)}) != obs dim ({obs_dim})")
        sys.exit(1)

    print(f"Obs normalizer loaded: dim={len(norm_mean)}, count={int(norm_data['_count'])}")
else:
    print("WARNING: No obs normalizer in checkpoint!")


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """Replicate OmniSafe ObsNormalize (frozen, no push)."""
    if not has_normalizer:
        return obs
    return np.clip(
        (obs - norm_mean) / norm_std,
        -norm_clip,
        norm_clip,
    ).astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# 7) Run evaluation (deterministic: use mean, NO tanh, NO sampling)
# ──────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

obs, info = env.reset(seed=42)
print(f"Reset obs shape: {obs.shape}")

# Determine max steps
if args.steps:
    max_steps = args.steps
else:
    # 3-month = 2190, full year = 8759
    max_steps = 8760  # will stop on truncation anyway

# Tracking
departures = []  # (charger_id, actual_soc, required_soc, step)
action_stats = {"raw_min": [], "raw_max": [], "raw_mean": []}
total_reward = 0.0
total_cost_ev = 0.0
step_count = 0

for step_i in range(max_steps):
    # 1. Normalize obs (frozen normalizer, no push)
    obs_normed = normalize_obs(obs)
    obs_t = torch.as_tensor(obs_normed, dtype=torch.float32).unsqueeze(0)

    # 2. Get deterministic action: raw MLP mean (NO tanh, NO sampling)
    with torch.no_grad():
        raw_action = actor(obs_t).squeeze(0).numpy()

    # 3. Apply ActionScale: map from [-1, 1] to env's action bounds
    action = action_scale(raw_action)

    # 4. Apply WM disable (same as CityLearnCMDPv2.step)
    if wm_disable:
        for wm_idx in wm_indices:
            if wm_idx < len(action):
                action[wm_idx] = 0.0

    # Track action stats
    action_stats["raw_min"].append(raw_action.min())
    action_stats["raw_max"].append(raw_action.max())
    action_stats["raw_mean"].append(raw_action.mean())

    # 5. Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1
    total_reward += float(reward) if np.isscalar(reward) else float(reward.sum())

    # Track cost_ev_departure from info
    cev = float(info.get("cost_ev_departure", 0.0))
    total_cost_ev += cev

    # 6. Detect departures (same logic as eval_r19_proper.py)
    t_after = int(getattr(city, "time_step", 0))
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
            departures.append((cid, actual_soc, r, step_i))

    if terminated or truncated:
        print(f"Episode ended at step {step_i + 1} (terminated={terminated}, truncated={truncated})")
        break

    if (step_i + 1) % 500 == 0:
        dep_so_far = len(departures)
        viol_so_far = sum(1 for _, ds, dr, _ in departures if ds < dr)
        print(f"  Step {step_i + 1}/{max_steps}: departures={dep_so_far}, "
              f"violated={viol_so_far}, cost_ev_cum={total_cost_ev:.2f}")

# ──────────────────────────────────────────────────────────────────
# 8) Results
# ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"  EVALUATION RESULTS (epoch-{load_epoch}, {step_count} steps)")
print("=" * 70)

# C0: EV departure
n_departures = len(departures)
n_violated = sum(1 for _, ds, dr, _ in departures if ds < dr)
viol_pct = 100.0 * n_violated / max(n_departures, 1)
mean_dep_soc = np.mean([ds for _, ds, _, _ in departures]) if departures else 0.0
mean_req_soc = np.mean([dr for _, _, dr, _ in departures]) if departures else 0.0
mean_deficit = np.mean([max(0, dr - ds) for _, ds, dr, _ in departures]) if departures else 0.0

print(f"\n--- C0: EV Departure Constraint ---")
print(f"  Total departures:    {n_departures}")
print(f"  Violated:            {n_violated}")
print(f"  Violation %:         {viol_pct:.1f}%")
print(f"  Mean departure SoC:  {mean_dep_soc:.4f}")
print(f"  Mean required SoC:   {mean_req_soc:.4f}")
print(f"  Mean deficit:        {mean_deficit:.4f}")
print(f"  Cumulative cost_ev:  {total_cost_ev:.2f}")

# Per-charger breakdown
if departures:
    print(f"\n  Per-charger breakdown:")
    for cname in sorted(set(cn for cn, _, _, _ in departures)):
        ch_deps = [(ds, dr) for cn, ds, dr, _ in departures if cn == cname]
        ch_viols = sum(1 for ds, dr in ch_deps if ds < dr)
        ch_mean_soc = np.mean([ds for ds, _ in ch_deps])
        ch_mean_req = np.mean([dr for _, dr in ch_deps])
        print(f"    {cname}: {len(ch_deps)} departures, {ch_viols} violated "
              f"({100*ch_viols/max(len(ch_deps),1):.1f}%), "
              f"mean_soc={ch_mean_soc:.4f}, mean_req={ch_mean_req:.4f}")

# Action stats
raw_mins = np.array(action_stats["raw_min"])
raw_maxs = np.array(action_stats["raw_max"])
print(f"\n--- Action Stats (raw MLP output, before ActionScale) ---")
print(f"  Min across steps:    {raw_mins.min():.3f}")
print(f"  Max across steps:    {raw_maxs.max():.3f}")
print(f"  Mean of means:       {np.mean(action_stats['raw_mean']):.3f}")

# Summary
print(f"\n--- Summary ---")
print(f"  Total steps:         {step_count}")
print(f"  Total reward:        {total_reward:.1f}")
print(f"  Avg reward/step:     {total_reward / max(step_count, 1):.4f}")

# Verify pipeline
print(f"\n--- Pipeline Verification ---")
print(f"  Actor input dim:     {in_dim} (checkpoint) == {obs_dim} (env)")
print(f"  Actor output dim:    {out_dim} (checkpoint) == {act_dim} (env)")
print(f"  Normalizer dim:      {len(norm_mean) if has_normalizer else 'N/A'}")
print(f"  ActionScale:         [-1,1] -> {list(zip(act_low, act_high))}")
print(f"  WM disable:          {wm_disable}, indices={wm_indices}")
print(f"  Battery clamp:       {batt_clamp_enabled}")
print(f"  EV clamp:            {ev_clamp_enabled}")
print(f"  Tanh applied:        NO (correct -- OmniSafe uses raw mean)")
