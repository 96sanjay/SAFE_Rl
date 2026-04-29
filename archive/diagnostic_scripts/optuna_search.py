#!/usr/bin/env python3
"""
Optuna hyperparameter search for 1-building CityLearn V2G Safe RL.

Searches reward weights, constraint limits, and key hyperparams.
Each trial: 20 epochs training on 1-building 3-month schema (~5 min),
then 1 eval episode to compute a composite score.

Usage:
    /home/christmas/miniconda3/envs/citylearn/bin/python optuna_search.py

Settings:
    n_trials = 50
    ~5 min per trial => ~4 hours total
    Pruning: if cycling_score=0 after 10 epochs, prune early.

Storage: SQLite DB at ./optuna_search.db (resumable).
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

PYTHON = "/home/christmas/miniconda3/envs/citylearn/bin/python"
SCHEMA = f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_1building_3month.json"
BASE_CFG = f"{PROJECT}/configs/on-policy/test_1bld.yaml"
TRAIN_SCRIPT = f"{PROJECT}/scripts/train_multi_lag.py"

# Where Optuna trial runs go
OPTUNA_RUNS_DIR = f"{PROJECT}/runs/optuna_search"

# ---------------------------------------------------------------------------
# Base environment variables (from run_1bld_test.sh)
# These are the defaults; trial params override specific ones.
# ---------------------------------------------------------------------------
BASE_ENV_VARS = {
    "CITYLEARN_SCHEMA": SCHEMA,
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_ACTION_MASK": "1",

    # KL regularization
    "CITYLEARN_KL_BETA": "0.1",
    "CITYLEARN_KL_BETA_DECAY": "0.995",

    # NEC-sign reward (defaults from R30c)
    "STEMS_ALPHA_NEC_SIGN": "3.0",
    "STEMS_ALPHA_PRICE_ARB": "1.0",
    "STEMS_LAMBDA_EV": "4.0",
    "STEMS_EV_SLACK_ARB_SCALE": "2.5",
    "STEMS_ALPHA_GRID_PENALTY": "1.5",
    "STEMS_ALPHA_GRID_PENALTY_SOLAR": "0.0",
    "CITYLEARN_STEMS_BATTERY_COST_SCALE": "1.0",

    # All other STEMS terms off
    "STEMS_ALPHA_NEC": "0.0",
    "STEMS_ALPHA_PEAK": "0.0",
    "STEMS_ALPHA_RAMP": "0.0",
    "STEMS_ALPHA_CARBON": "0.0",
    "STEMS_ALPHA_LOAD": "0.0",

    # C1 tolerance
    "CITYLEARN_C1_SOC_TOLERANCE": "0.20",

    # Disabled features
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_EV_SAUTE": "0",
}


def clear_citylearn_env_vars():
    """Remove all CITYLEARN/STEMS/COST_W env vars to prevent leakage."""
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k, None)


def set_env_vars(overrides: dict[str, str] | None = None):
    """Set base env vars with optional overrides."""
    clear_citylearn_env_vars()
    for k, v in BASE_ENV_VARS.items():
        os.environ[k] = v
    if overrides:
        for k, v in overrides.items():
            os.environ[k] = str(v)


def make_trial_config(
    trial_dir: str,
    cost_limit_0: float,
    epochs: int = 20,
) -> str:
    """Create a trial-specific YAML config from the base config.

    Returns path to the generated config file.
    """
    with open(BASE_CFG) as f:
        cfg = yaml.safe_load(f)

    # Override for fast search
    steps_per_epoch = cfg["algo_cfgs"]["steps_per_epoch"]  # 2190
    cfg["train_cfgs"]["total_steps"] = epochs * steps_per_epoch

    # Override cost_limit_0 (the one we search over)
    cfg["multi_cfgs"]["cost_limit_0"] = cost_limit_0

    # Point logs to trial-specific directory
    cfg["logger_cfgs"]["log_dir"] = trial_dir
    cfg["logger_cfgs"]["save_model_freq"] = 10

    cfg_path = os.path.join(trial_dir, "config.yaml")
    os.makedirs(trial_dir, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return cfg_path


# ---------------------------------------------------------------------------
# Training (in-process, not subprocess)
# ---------------------------------------------------------------------------
def run_training(cfg_path: str, use_bc: bool = True) -> str:
    """Run training in-process and return the run directory containing checkpoints.

    Returns the directory path like:
        <trial_dir>/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-.../
    """
    # Import here to avoid polluting module scope
    from scripts.train_multi_lag import main as train_main
    train_main(cfg_path, use_bc=use_bc)

    # Find the run directory (OmniSafe creates a timestamped subdir)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    log_dir = cfg["logger_cfgs"]["log_dir"]

    # Pattern: <log_dir>/PPOLagMulti-{...}/seed-*
    pattern = os.path.join(log_dir, "PPOLagMulti-*", "seed-*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise RuntimeError(f"No run directory found matching {pattern}")
    return matches[-1]  # latest run


# ---------------------------------------------------------------------------
# Evaluation (adapted from eval_1bld.py)
# ---------------------------------------------------------------------------
def evaluate_checkpoint(
    run_dir: str,
    epoch: int = 20,
) -> dict:
    """Load checkpoint and run 1 eval episode. Return metrics dict.

    Metrics returned:
        cycling_score, v2g_score, c0_score, c3_score, price_score,
        combined_score, plus raw values for logging.
    """
    import torch
    import torch.nn as nn

    ckpt_dir = os.path.join(run_dir, "torch_save")
    ckpt_path = os.path.join(ckpt_dir, f"epoch-{epoch}.pt")

    # If exact epoch not found, try the highest available
    if not os.path.exists(ckpt_path):
        available = sorted(glob.glob(os.path.join(ckpt_dir, "epoch-*.pt")))
        if not available:
            return {"combined_score": 0.0, "error": "no_checkpoint"}
        ckpt_path = available[-1]
        actual_epoch = int(Path(ckpt_path).stem.split("-")[1])
        print(f"[Eval] Exact epoch-{epoch} not found, using epoch-{actual_epoch}")

    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi_state = ckpt["pi"]
    h1 = pi_state["mean.0.weight"].shape[0]
    h2 = pi_state["mean.2.weight"].shape[0]
    obs_dim = pi_state["mean.0.weight"].shape[1]
    act_dim = pi_state["mean.4.weight"].shape[0]

    # Build simple MLP actor
    class MLPActor(nn.Module):
        def __init__(self, obs_d, act_d, hidden):
            super().__init__()
            layers = []
            in_d = obs_d
            for h in hidden:
                layers.append(nn.Linear(in_d, h))
                layers.append(nn.Tanh())
                in_d = h
            layers.append(nn.Linear(in_d, act_d))
            self.mean = nn.Sequential(*layers)

        def forward(self, obs):
            return torch.tanh(self.mean(obs))

    actor = MLPActor(obs_dim, act_dim, (h1, h2))
    filtered = {k: v for k, v in pi_state.items() if not k.startswith("log_std")}
    actor.load_state_dict(filtered, strict=False)
    actor.eval()

    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        clip_t = norm["_clip"].float()
        obs_clip = float(clip_t.mean())

    # Create environment (same wrapper chain as training)
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

    # Identify action indices
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)

    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names)
              if "electric_vehicle_storage_charger_" in str(n).lower()]

    env_obs_dim = env.observation_space.shape[0]
    need_pad = obs_dim > env_obs_dim
    pad_dim = obs_dim - env_obs_dim if need_pad else 0
    need_trim = obs_dim < env_obs_dim

    # Reset and run eval episode
    obs, _ = env.reset(seed=42)
    done = False
    step = 0

    # Tracking
    actions_all = []
    hour_all = []
    price_all = []
    ev_connected_mask = []
    c3_vals = []
    P_BUILDING_MAX = 4.6083

    # C3 violation tracking
    c3_violations = 0
    c3_total_checks = 0

    # EV departure tracking
    total_departures = 0
    violated_departures = 0
    ev_tracker = {}

    max_steps = 2190  # 3 months

    while not done and step < max_steps:
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

        # Price
        try:
            pr = buildings[0].pricing.electricity_pricing
            price = float(pr[t_idx]) if t_idx < len(pr) else 0.0
        except Exception:
            price = 0.0
        price_all.append(price)

        # EV connection tracking
        ev_connected_step = []
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, "electric_vehicle_chargers", None) or []
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                sim = getattr(ch, 'charger_simulation',
                              getattr(ch, '_Charger__charger_simulation', None))
                if sim is None:
                    ev_connected_step.append(False)
                    continue
                try:
                    sa = np.asarray(
                        getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                    current_connected = t_now < len(sa) and float(sa[t_now]) == 1.0
                    ev_connected_step.append(current_connected)

                    # EV SoC for departure tracking
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

                    # Departure tracking
                    ra = np.asarray(
                        getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                    if current_connected:
                        rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                        if not np.isfinite(rs):
                            rs = 1.0
                        ev_tracker[key] = {
                            'was_connected': True, 'last_soc': ev_soc, 'required_soc': rs}
                    else:
                        prev = ev_tracker.get(key, {})
                        if prev.get('was_connected', False):
                            total_departures += 1
                            deficit = max(0.0, prev['required_soc'] - prev['last_soc'])
                            if deficit > 0.01:
                                violated_departures += 1
                        ev_tracker[key] = {'was_connected': False}
                except Exception:
                    ev_connected_step.append(False)

        ev_connected_mask.append(ev_connected_step)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        c3_vals.append(info.get("cost_stems_building_power", 0.0))

        # NEC for C3 violation counting
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
            c3_total_checks += 1
            if abs(p) > P_BUILDING_MAX:
                c3_violations += 1

    # --- Compute metrics ---
    actions_arr = np.array(actions_all)

    # Battery metrics
    batt_actions = actions_arr[:, batt_idx] if batt_idx else np.array([])
    if batt_actions.size > 0:
        batt_flat = batt_actions.ravel()
        charge_pct = float(np.mean(batt_flat > 0.1))
        discharge_pct = float(np.mean(batt_flat < -0.1))
    else:
        charge_pct = 0.0
        discharge_pct = 0.0

    # Cycling score (0-1)
    cycling_score = min(charge_pct, discharge_pct) / 0.3
    cycling_score = min(cycling_score, 1.0)

    # V2G score (0-1): EV discharge during peak (17-23)
    ev_peak_connected = 0
    ev_peak_discharge = 0
    for t in range(len(actions_all)):
        h = hour_all[t]
        for ei_local, ei in enumerate(ev_idx):
            if ei_local < len(ev_connected_mask[t]) and ev_connected_mask[t][ei_local]:
                a = float(actions_all[t][ei])
                if 17 <= h <= 23:
                    ev_peak_connected += 1
                    if a < -0.1:
                        ev_peak_discharge += 1

    v2g_peak_pct = ev_peak_discharge / ev_peak_connected if ev_peak_connected > 0 else 0.0
    v2g_score = min(v2g_peak_pct / 0.2, 1.0)

    # C0 score (0-1): EV departure compliance
    c0_violation_pct = violated_departures / total_departures if total_departures > 0 else 0.0
    c0_score = max(0.0, 1.0 - c0_violation_pct / 0.10)

    # C3 score (0-1): building power
    c3_violation_pct = c3_violations / c3_total_checks if c3_total_checks > 0 else 0.0
    c3_score = max(0.0, 1.0 - c3_violation_pct / 0.10)

    # Price awareness (0-1)
    price_score = 0.0
    if batt_actions.size > 0 and len(price_all) > 10:
        prices_np = np.array(price_all[:len(batt_actions)])
        batt_avg = batt_actions.mean(axis=1) if batt_actions.ndim > 1 else batt_actions.ravel()
        batt_avg = batt_avg[:len(prices_np)]
        valid = np.isfinite(prices_np) & np.isfinite(batt_avg)
        if valid.sum() > 10:
            corr = np.corrcoef(prices_np[valid], batt_avg[valid])[0, 1]
            # Negative correlation is GOOD (discharge at high price)
            price_score = max(0.0, -corr)

    # Combined score
    combined = (
        cycling_score * 0.25
        + v2g_score * 0.20
        + c0_score * 0.25
        + c3_score * 0.15
        + price_score * 0.15
    )

    metrics = {
        "combined_score": combined,
        "cycling_score": cycling_score,
        "v2g_score": v2g_score,
        "c0_score": c0_score,
        "c3_score": c3_score,
        "price_score": price_score,
        "charge_pct": charge_pct,
        "discharge_pct": discharge_pct,
        "v2g_peak_pct": v2g_peak_pct,
        "c0_violation_pct": c0_violation_pct,
        "c3_violation_pct": c3_violation_pct,
        "total_departures": total_departures,
        "violated_departures": violated_departures,
        "eval_steps": step,
    }
    return metrics


# ---------------------------------------------------------------------------
# Mid-training check for pruning (read progress.csv after N epochs)
# ---------------------------------------------------------------------------
def mid_training_check(run_dir: str, min_epochs: int = 10) -> dict | None:
    """Parse progress.csv to check if training is worth continuing.

    Returns metrics dict or None if not enough data yet.
    """
    progress_path = os.path.join(run_dir, "progress.csv")
    if not os.path.exists(progress_path):
        return None

    try:
        with open(progress_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return None

    if len(rows) < min_epochs:
        return None

    # Check last few epochs for reward trend
    last = rows[-1]
    ep_ret = float(last.get("Metrics/EpRet", 0))
    ep_cost = float(last.get("Metrics/EpCost", 0))

    return {
        "epochs_completed": len(rows),
        "ep_ret": ep_ret,
        "ep_cost": ep_cost,
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial):
    """Optuna objective: train + eval, return combined score."""
    import optuna

    # --- Suggest hyperparameters ---
    alpha_nec_sign = trial.suggest_float("alpha_nec_sign", 1.0, 5.0)
    alpha_price_arb = trial.suggest_float("alpha_price_arb", 0.5, 3.0)
    lambda_ev = trial.suggest_float("lambda_ev", 5.0, 20.0)
    ev_arb_scale = trial.suggest_float("ev_arb_scale", 1.0, 5.0)
    alpha_grid_penalty = trial.suggest_float("alpha_grid_penalty", 0.5, 3.0)
    cost_limit_0 = trial.suggest_float("cost_limit_0", 200.0, 1500.0)
    kl_beta = trial.suggest_float("kl_beta", 0.01, 0.3)
    use_saute = trial.suggest_categorical("use_saute", [0, 1])

    trial_name = f"trial_{trial.number:04d}"
    trial_dir = os.path.join(OPTUNA_RUNS_DIR, trial_name)
    os.makedirs(trial_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  OPTUNA TRIAL {trial.number}")
    print(f"  alpha_nec_sign={alpha_nec_sign:.2f}")
    print(f"  alpha_price_arb={alpha_price_arb:.2f}")
    print(f"  lambda_ev={lambda_ev:.1f}")
    print(f"  ev_arb_scale={ev_arb_scale:.2f}")
    print(f"  alpha_grid_penalty={alpha_grid_penalty:.2f}")
    print(f"  cost_limit_0={cost_limit_0:.0f}")
    print(f"  kl_beta={kl_beta:.3f}")
    print(f"  use_saute={use_saute}")
    print(f"{'=' * 60}\n")

    # --- Set env vars ---
    env_overrides = {
        "STEMS_ALPHA_NEC_SIGN": str(alpha_nec_sign),
        "STEMS_ALPHA_PRICE_ARB": str(alpha_price_arb),
        "STEMS_LAMBDA_EV": str(lambda_ev),
        "STEMS_EV_SLACK_ARB_SCALE": str(ev_arb_scale),
        "STEMS_ALPHA_GRID_PENALTY": str(alpha_grid_penalty),
        "CITYLEARN_KL_BETA": str(kl_beta),
        "CITYLEARN_EV_SAUTE": str(use_saute),
    }
    set_env_vars(env_overrides)

    # --- Create trial config ---
    epochs = 20
    cfg_path = make_trial_config(trial_dir, cost_limit_0=cost_limit_0, epochs=epochs)

    # --- Train ---
    t0 = time.time()
    try:
        run_dir = run_training(cfg_path, use_bc=True)
    except Exception as e:
        print(f"[Trial {trial.number}] Training FAILED: {e}")
        traceback.print_exc()
        return 0.0

    train_time = time.time() - t0
    print(f"[Trial {trial.number}] Training took {train_time:.1f}s")

    # --- Mid-training pruning check ---
    # Read progress.csv to see if there is any cycling signal
    mid = mid_training_check(run_dir, min_epochs=10)
    if mid is not None:
        print(f"[Trial {trial.number}] Mid-check: "
              f"EpRet={mid['ep_ret']:.1f}, EpCost={mid['ep_cost']:.1f}")

    # --- Evaluate ---
    try:
        # Re-set env vars for eval (same overrides)
        set_env_vars(env_overrides)
        metrics = evaluate_checkpoint(run_dir, epoch=epochs)
    except Exception as e:
        print(f"[Trial {trial.number}] Eval FAILED: {e}")
        traceback.print_exc()
        return 0.0

    score = metrics.get("combined_score", 0.0)

    # Report intermediate value for pruning
    trial.report(score, step=epochs)
    if trial.should_prune():
        print(f"[Trial {trial.number}] PRUNED at epoch {epochs}")
        raise optuna.TrialPruned()

    # Also check cycling for early pruning at epoch 10
    cycling = metrics.get("cycling_score", 0.0)
    if cycling == 0.0:
        # Report zero at step 10 to trigger median pruner
        trial.report(0.0, step=10)
        if trial.should_prune():
            print(f"[Trial {trial.number}] PRUNED: zero cycling")
            raise optuna.TrialPruned()

    # Log all metrics as trial user attrs
    for k, v in metrics.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("train_time_s", train_time)

    print(f"\n{'=' * 60}")
    print(f"  TRIAL {trial.number} RESULT")
    print(f"  Combined score: {score:.4f}")
    print(f"  Cycling: {metrics['cycling_score']:.3f} "
          f"(charge={metrics['charge_pct']:.1%}, "
          f"discharge={metrics['discharge_pct']:.1%})")
    print(f"  V2G:     {metrics['v2g_score']:.3f} "
          f"(peak_v2g={metrics['v2g_peak_pct']:.1%})")
    print(f"  C0:      {metrics['c0_score']:.3f} "
          f"(violations={metrics['violated_departures']}/{metrics['total_departures']})")
    print(f"  C3:      {metrics['c3_score']:.3f} "
          f"(violation_rate={metrics['c3_violation_pct']:.2%})")
    print(f"  Price:   {metrics['price_score']:.3f}")
    print(f"{'=' * 60}\n")

    # Clean up large files to save disk (keep config + progress, delete checkpoints)
    torch_save_dir = os.path.join(run_dir, "torch_save")
    if os.path.exists(torch_save_dir):
        # Keep only best checkpoint
        best_ckpt = os.path.join(torch_save_dir, f"epoch-{epochs}.pt")
        for f in glob.glob(os.path.join(torch_save_dir, "epoch-*.pt")):
            if f != best_ckpt:
                try:
                    os.remove(f)
                except OSError:
                    pass

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import optuna

    parser = argparse.ArgumentParser(description="Optuna HPO for CityLearn V2G")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", type=str, default="citylearn_v2g_hpo")
    parser.add_argument("--db", type=str,
                        default=f"{PROJECT}/optuna_search.db")
    parser.add_argument("--resume", action="store_true",
                        help="Resume existing study instead of creating new")
    args = parser.parse_args()

    os.makedirs(OPTUNA_RUNS_DIR, exist_ok=True)

    storage = f"sqlite:///{args.db}"

    # Create or load study
    if args.resume:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=storage,
        )
        print(f"Resumed study '{args.study_name}' with "
              f"{len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            ),
            load_if_exists=True,
        )
        print(f"Study '{args.study_name}': "
              f"{len(study.trials)} existing trials")

    print(f"\nStarting {args.n_trials} trials...")
    print(f"DB: {args.db}")
    print(f"Runs: {OPTUNA_RUNS_DIR}")
    print(f"Estimated time: {args.n_trials * 5 / 60:.1f} hours\n")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        catch=(Exception,),  # Don't crash on individual trial failures
    )

    # Print results
    print(f"\n{'=' * 72}")
    print(f"  OPTUNA SEARCH COMPLETE")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print(f"{'=' * 72}")

    if study.best_trial:
        best = study.best_trial
        print(f"\n  BEST TRIAL: {best.number}")
        print(f"  Score: {best.value:.4f}")
        print(f"  Params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
        print(f"  Metrics:")
        for k, v in best.user_attrs.items():
            print(f"    {k}: {v}")

        # Save best params to YAML
        best_path = os.path.join(OPTUNA_RUNS_DIR, "best_params.yaml")
        with open(best_path, "w") as f:
            yaml.dump({
                "trial": best.number,
                "score": best.value,
                "params": best.params,
                "metrics": {k: v for k, v in best.user_attrs.items()
                           if isinstance(v, (int, float))},
            }, f, default_flow_style=False)
        print(f"\n  Best params saved to: {best_path}")

    # Also dump top-5 trials
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        completed.sort(key=lambda t: t.value or 0, reverse=True)
        print(f"\n  TOP 5 TRIALS:")
        for i, t in enumerate(completed[:5]):
            print(f"    #{i+1} Trial {t.number}: score={t.value:.4f} "
                  f"cycling={t.user_attrs.get('cycling_score', '?'):.3f} "
                  f"v2g={t.user_attrs.get('v2g_score', '?'):.3f} "
                  f"c0={t.user_attrs.get('c0_score', '?'):.3f}")
            for k, v in t.params.items():
                print(f"      {k}={v}")


if __name__ == "__main__":
    main()
