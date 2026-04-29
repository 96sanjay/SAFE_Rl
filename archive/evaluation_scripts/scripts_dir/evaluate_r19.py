#!/usr/bin/env python3
"""
R19 Ablation Evaluation Script
===============================
Deterministic evaluation of R19 trained actor on the EXACT same environment
pipeline used during training. Computes all constraint metrics from RAW
per-step environment data (not from training logs).

Wrapper chain (matches training):
  make_base_env(central_agent=True)
    -> CityLearnSafetyEnvV3
      -> ForecastObsWrapper(horizon=24)  [+128 dims]
        -> SauteEVBudgetWrapper           [+1 dim]

obs_dim = 70 (base) + 128 (forecast) + 1 (saute) = 199
act_dim = 26

Usage:
  conda activate citylearn
  python scripts/evaluate_r19.py [--checkpoint PATH] [--epoch 80]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)

# ---------------------------------------------------------------------------
# Set ALL env vars IDENTICAL to run_r19_ablation.sh
# (Must happen BEFORE any citylearn import)
# ---------------------------------------------------------------------------
ENV_VARS = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    "CITYLEARN_PID_LAGRANGE": "1",
    "CITYLEARN_EV_SAUTE": "1",
    "CITYLEARN_EV_SAUTE_BUDGET": "25000",
    "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
    "CITYLEARN_EV_SAUTE_GAMMA": "1.0",
    "CITYLEARN_EV_SAUTE_SHAPED_ALPHA": "10.0",
    "STEMS_ALPHA_GRID": "0.0",
    "STEMS_SG_THRESHOLD": "0.5",
    "STEMS_ALPHA_LOAD_SHIFT": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.3",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_XI_RENEWABLE": "0.2",
    "STEMS_BETA_RAMP": "0.3",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "STEMS_LAMBDA_EV": "5.0",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "STEMS_ALPHA_EV_GUARD": "1.0",
    "STEMS_ALPHA_V2G_CONTEXT": "3.0",
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    "CITYLEARN_EV_COST_SCALE": "3.0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_BATT_CLAMP": "0",
    # Disable KPI logging to avoid polluting run directories
    "CITYLEARN_KPI_RUN_NAME": "__eval_disabled__",
}

for key, val in ENV_VARS.items():
    os.environ[key] = val

# ---------------------------------------------------------------------------
# Imports (after env var setup)
# ---------------------------------------------------------------------------
from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.saute_ev_wrapper import SauteEVBudgetWrapper


# ---------------------------------------------------------------------------
# Actor network (matches training architecture exactly)
# ---------------------------------------------------------------------------
class GaussianActor(nn.Module):
    """MLP Gaussian actor with Tanh activations and Tanh output squashing."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        self.mean = nn.Sequential(*layers, nn.Linear(in_dim, act_dim), nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(obs)


# ---------------------------------------------------------------------------
# Environment builder (manual wrapper chain, NO OmniSafe CMDP)
# ---------------------------------------------------------------------------
def build_eval_env():
    """Build the exact wrapper chain used in training, minus OmniSafe CMDP."""
    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    forecast = ForecastObsWrapper(safety, forecast_horizon=24)
    saute = SauteEVBudgetWrapper(forecast)
    return saute


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def find_checkpoint(epoch: int = 80) -> Path:
    """Find the R19 checkpoint for the specified epoch."""
    pattern = str(
        Path(PROJECT)
        / "runs/r19_ablation/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}"
        / "seed-*/torch_save"
        / f"epoch-{epoch}.pt"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for epoch {epoch}. Pattern: {pattern}"
        )
    # Use the LAST match (latest seed directory by timestamp)
    return Path(matches[-1])


def load_actor(ckpt_path: Path, obs_dim: int, act_dim: int):
    """Load actor weights and observation normalizer from checkpoint."""
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    actor = GaussianActor(obs_dim, act_dim)
    actor.load_state_dict(ckpt["pi"])
    actor.eval()

    norm = ckpt["obs_normalizer"]
    obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
    obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)

    print(f"  Actor loaded: obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  Normalizer: mean shape={obs_mean.shape}, std shape={obs_std.shape}")
    return actor, obs_mean, obs_std


# ---------------------------------------------------------------------------
# Deterministic policy
# ---------------------------------------------------------------------------
@torch.no_grad()
def deterministic_action(
    actor: GaussianActor,
    obs: np.ndarray,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
) -> np.ndarray:
    """Compute deterministic action from normalized observation."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    obs_norm = (obs_t - obs_mean) / (obs_std + 1e-8)
    action = actor(obs_norm).cpu().numpy().squeeze(0)
    return np.clip(action, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------
def run_evaluation(actor, obs_mean, obs_std, seed=42, verbose=True):
    """Run one full-year deterministic evaluation, returning raw metrics."""
    env = build_eval_env()
    obs, info = env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if verbose:
        print(f"  Environment: obs_dim={obs_dim}, act_dim={act_dim}")

    # -------------------------------------------------------------------
    # Accumulators for raw per-step data
    # -------------------------------------------------------------------
    total_reward = 0.0
    total_cost = 0.0

    # Per-constraint cost accumulators
    total_c0 = 0.0  # cost_ev_departure (weighted)
    total_c1 = 0.0  # cost_ev_dense
    total_c2 = 0.0  # cost_stems_battery
    total_c3 = 0.0  # cost_stems_building_power
    total_c4 = 0.0  # cost_stems_grid_power

    # Raw cost accumulators (unweighted)
    raw_c0 = 0.0
    raw_c2 = 0.0
    raw_c3 = 0.0
    raw_c4 = 0.0

    # Violation counters
    c2_violation_steps = 0  # battery SoC out of band
    c3_violation_steps = 0  # building power over threshold
    c4_violation_steps = 0  # grid power over threshold

    # EV departure tracking
    total_departures = 0
    departures_with_deficit = 0
    ev_deficit_kwh_total = 0.0
    ev_avoidable_deficit_kwh = 0.0
    ev_unavoidable_deficit_kwh = 0.0

    # V2G tracking
    v2g_discharge_steps = 0

    # Electricity cost
    total_import_kwh = 0.0
    total_export_kwh = 0.0
    total_bill = 0.0

    # Building power stats
    building_power_max_seen = 0.0
    grid_power_max_seen = 0.0

    # Per-step raw data storage for detailed analysis
    step_data = []

    step = 0
    done = truncated = False
    t_start = time.time()

    while not (done or truncated):
        action = deterministic_action(actor, obs, obs_mean, obs_std)
        obs, reward, done, truncated, info = env.step(action)
        step += 1

        # ------ Extract raw metrics from info dict ------

        # Reward
        r = float(reward)
        total_reward += r

        # Total CMDP cost (as computed by safety_env_v3)
        step_cost = float(info.get("cost", 0.0))
        total_cost += step_cost

        # Per-constraint costs (from safety_env_v3 info dict)
        c0_val = float(info.get("cost_ev_departure", 0.0))
        c1_val = float(info.get("cost_ev_dense", 0.0))
        c2_val = float(info.get("cost_stems_battery", 0.0))
        c3_val = float(info.get("cost_stems_building_power", 0.0))
        c4_val = float(info.get("cost_stems_grid_power", 0.0))

        total_c0 += c0_val
        total_c1 += c1_val
        total_c2 += c2_val
        total_c3 += c3_val
        total_c4 += c4_val

        # Raw violation indicators (binary from safety_env_v3)
        if float(info.get("battery_soc_violation", 0.0)) > 0.5:
            c2_violation_steps += 1

        if float(info.get("building_power_violation", 0.0)) > 0.5:
            c3_violation_steps += 1

        if float(info.get("grid_power_violation", 0.0)) > 0.5:
            c4_violation_steps += 1

        # EV departure events (only fires when a departure actually occurs)
        ev_deps_this_step = int(info.get("ev_departure_departures", 0))
        if ev_deps_this_step > 0:
            total_departures += ev_deps_this_step
            # A departure has a deficit if the agent-controllable cost > 0
            ev_deficit_v3 = float(info.get("cost_ev_departure_agent_controllable_v3", 0.0))
            ev_unavoid = float(info.get("cost_ev_departure_uncontrollable_v3", 0.0))
            ev_total_deficit = float(info.get("ev_departure_deficit_kwh", 0.0))

            # Count departures with any deficit
            deficit_count = int(info.get("ev_departure_violation_count_deficit", 0))
            departures_with_deficit += deficit_count

            ev_deficit_kwh_total += ev_total_deficit
            ev_avoidable_deficit_kwh += ev_deficit_v3
            ev_unavoidable_deficit_kwh += ev_unavoid

        # V2G discharge tracking: count steps where any EV action < -0.1
        for i in range(8):
            ev_act = float(info.get(f"action_ev_{i}", 0.0))
            if ev_act < -0.1:
                v2g_discharge_steps += 1
                break  # count step once even if multiple EVs discharge

        # Electricity tracking
        import_kwh = float(info.get("grid_import_kwh", 0.0))
        export_kwh = float(info.get("grid_export_kwh", 0.0))
        total_import_kwh += import_kwh
        total_export_kwh += export_kwh
        bill = float(info.get("reward_bill_raw", 0.0))
        total_bill += bill

        # Power stats
        bld_pmax = float(info.get("building_power_max", 0.0))
        if bld_pmax > building_power_max_seen:
            building_power_max_seen = bld_pmax

        grid_kwh = float(info.get("grid_import_kwh", 0.0))
        if grid_kwh > grid_power_max_seen:
            grid_power_max_seen = grid_kwh

        # Store per-step record for optional detailed analysis
        step_record = {
            "step": step,
            "reward": r,
            "cost": step_cost,
            "c0_ev_departure": c0_val,
            "c1_ev_dense": c1_val,
            "c2_battery_soc": c2_val,
            "c3_building_power": c3_val,
            "c4_grid_power": c4_val,
            "c2_violation": 1 if float(info.get("battery_soc_violation", 0)) > 0.5 else 0,
            "c3_violation": 1 if float(info.get("building_power_violation", 0)) > 0.5 else 0,
            "c4_violation": 1 if float(info.get("grid_power_violation", 0)) > 0.5 else 0,
            "grid_import_kwh": import_kwh,
            "grid_export_kwh": export_kwh,
            "ev_departures": ev_deps_this_step,
            "ev_saute_budget": float(info.get("ev_saute_budget", 0.0)),
        }
        step_data.append(step_record)

        if verbose and step % 2000 == 0:
            elapsed = time.time() - t_start
            print(
                f"  Step {step:5d}/{8759}: reward={total_reward:+.1f}  "
                f"cost={total_cost:.1f}  C0={total_c0:.1f}  "
                f"C3={total_c3:.1f}  C4={total_c4:.1f}  "
                f"[{elapsed:.0f}s]"
            )

    elapsed = time.time() - t_start
    total_steps = step

    if verbose:
        print(f"  Completed {total_steps} steps in {elapsed:.1f}s")

    # -------------------------------------------------------------------
    # Compute violation RATES
    # -------------------------------------------------------------------
    c0_rate = (
        100.0 * departures_with_deficit / total_departures
        if total_departures > 0
        else 0.0
    )
    c2_rate = 100.0 * c2_violation_steps / total_steps
    c3_rate = 100.0 * c3_violation_steps / total_steps
    c4_rate = 100.0 * c4_violation_steps / total_steps

    results = {
        # Episode summary
        "total_steps": total_steps,
        "total_reward": total_reward,
        "mean_reward_per_step": total_reward / total_steps,
        "total_cost": total_cost,
        "elapsed_seconds": elapsed,
        # Per-constraint cumulative costs
        "C0_ev_departure_cost": total_c0,
        "C1_ev_dense_cost": total_c1,
        "C2_battery_soc_cost": total_c2,
        "C3_building_power_cost": total_c3,
        "C4_grid_power_cost": total_c4,
        # Violation rates (%)
        "C0_departure_violation_rate_pct": c0_rate,
        "C0_total_departures": total_departures,
        "C0_departures_with_deficit": departures_with_deficit,
        "C2_violation_rate_pct": c2_rate,
        "C2_violation_steps": c2_violation_steps,
        "C3_violation_rate_pct": c3_rate,
        "C3_violation_steps": c3_violation_steps,
        "C4_violation_rate_pct": c4_rate,
        "C4_violation_steps": c4_violation_steps,
        # EV departure details
        "ev_deficit_kwh_total": ev_deficit_kwh_total,
        "ev_avoidable_deficit_kwh": ev_avoidable_deficit_kwh,
        "ev_unavoidable_deficit_kwh": ev_unavoidable_deficit_kwh,
        # V2G
        "v2g_discharge_steps": v2g_discharge_steps,
        "v2g_discharge_rate_pct": 100.0 * v2g_discharge_steps / total_steps,
        # Electricity
        "total_import_kwh": total_import_kwh,
        "total_export_kwh": total_export_kwh,
        "total_bill_usd": total_bill,
        # Power peaks
        "building_power_max_kw": building_power_max_seen,
        "grid_power_max_kwh": grid_power_max_seen,
    }

    return results, step_data


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(results: dict, run_label: str = ""):
    """Print a clean summary table."""
    label = f" ({run_label})" if run_label else ""
    width = 70
    print()
    print("=" * width)
    print(f"  R19 ABLATION EVALUATION RESULTS{label}")
    print("=" * width)
    print()

    # Episode summary
    print("  EPISODE SUMMARY")
    print("  " + "-" * (width - 4))
    print(f"  Total steps:          {results['total_steps']}")
    print(f"  Total reward:         {results['total_reward']:+.2f}")
    print(f"  Mean reward/step:     {results['mean_reward_per_step']:+.4f}")
    print(f"  Total CMDP cost:      {results['total_cost']:.2f}")
    print(f"  Elapsed time:         {results['elapsed_seconds']:.1f}s")
    print()

    # Constraint violations
    print("  CONSTRAINT VIOLATIONS")
    print("  " + "-" * (width - 4))
    print(
        f"  C0 (EV departure):    {results['C0_departures_with_deficit']}"
        f" / {results['C0_total_departures']} departures"
        f" = {results['C0_departure_violation_rate_pct']:.1f}%"
        f"   (cost: {results['C0_ev_departure_cost']:.2f})"
    )
    print(
        f"  C1 (EV dense SoC):    cost = {results['C1_ev_dense_cost']:.2f}"
        f"   (handled by Saute MDP, not in CMDP)"
    )
    print(
        f"  C2 (Battery SoC):     {results['C2_violation_steps']}"
        f" / {results['total_steps']} steps"
        f" = {results['C2_violation_rate_pct']:.1f}%"
        f"   (cost: {results['C2_battery_soc_cost']:.2f})"
    )
    print(
        f"  C3 (Building power):  {results['C3_violation_steps']}"
        f" / {results['total_steps']} steps"
        f" = {results['C3_violation_rate_pct']:.1f}%"
        f"   (cost: {results['C3_building_power_cost']:.2f})"
    )
    print(
        f"  C4 (Grid power):      {results['C4_violation_steps']}"
        f" / {results['total_steps']} steps"
        f" = {results['C4_violation_rate_pct']:.1f}%"
        f"   (cost: {results['C4_grid_power_cost']:.2f})"
    )
    print()

    # EV details
    print("  EV DEPARTURE DETAILS")
    print("  " + "-" * (width - 4))
    print(f"  Total deficit (kWh):      {results['ev_deficit_kwh_total']:.2f}")
    print(f"  Avoidable deficit (kWh):  {results['ev_avoidable_deficit_kwh']:.2f}")
    print(f"  Unavoidable deficit (kWh):{results['ev_unavoidable_deficit_kwh']:.2f}")
    print()

    # V2G
    print("  V2G ACTIVITY")
    print("  " + "-" * (width - 4))
    print(
        f"  V2G discharge steps:  {results['v2g_discharge_steps']}"
        f" ({results['v2g_discharge_rate_pct']:.1f}% of episode)"
    )
    print()

    # Electricity
    print("  ELECTRICITY")
    print("  " + "-" * (width - 4))
    print(f"  Total import (kWh):   {results['total_import_kwh']:.1f}")
    print(f"  Total export (kWh):   {results['total_export_kwh']:.1f}")
    print(f"  Total bill (USD):     {results['total_bill_usd']:.2f}")
    print(f"  Grid peak (kWh/step): {results['grid_power_max_kwh']:.2f}")
    print()
    print("=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="R19 Ablation Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt file (auto-detected if not given)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=80,
        help="Epoch number to load (default: 80)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: auto-generated in runs/r19_ablation/)",
    )
    parser.add_argument(
        "--skip-determinism-check",
        action="store_true",
        help="Skip the second evaluation run for determinism verification",
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  R19 ABLATION -- DETERMINISTIC EVALUATION")
    print("=" * 70)
    print()

    # --- Find and load checkpoint ---
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        ckpt_path = find_checkpoint(epoch=args.epoch)

    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Epoch: {args.epoch}")
    print()

    # --- Build a temporary env to get dimensions ---
    print("  Building environment to determine dimensions...")
    tmp_env = build_eval_env()
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    del tmp_env

    if obs_dim != 199:
        print(
            f"  WARNING: Expected obs_dim=199 (70 base + 128 forecast + 1 saute), "
            f"got {obs_dim}"
        )

    # --- Load actor ---
    actor, obs_mean, obs_std = load_actor(ckpt_path, obs_dim, act_dim)

    # Verify normalizer dimensions match
    if obs_mean.shape[0] != obs_dim:
        raise ValueError(
            f"Normalizer mean dim ({obs_mean.shape[0]}) != env obs_dim ({obs_dim}). "
            f"Architecture mismatch."
        )

    # --- Run 1: Primary evaluation ---
    print()
    print("-" * 70)
    print("  RUN 1: Primary evaluation (seed=42)")
    print("-" * 70)
    results_1, step_data_1 = run_evaluation(actor, obs_mean, obs_std, seed=42)
    print_summary(results_1, "Run 1")

    # --- Run 2: Determinism check ---
    if not args.skip_determinism_check:
        print()
        print("-" * 70)
        print("  RUN 2: Determinism check (seed=42)")
        print("-" * 70)
        results_2, step_data_2 = run_evaluation(
            actor, obs_mean, obs_std, seed=42, verbose=False
        )

        # Compare key metrics
        deterministic = True
        fields_to_check = [
            "total_reward",
            "total_cost",
            "C0_ev_departure_cost",
            "C3_building_power_cost",
            "C4_grid_power_cost",
            "C0_departures_with_deficit",
            "C3_violation_steps",
            "C4_violation_steps",
            "total_import_kwh",
        ]
        print()
        print("  Determinism verification:")
        for field in fields_to_check:
            v1 = results_1[field]
            v2 = results_2[field]
            match = abs(v1 - v2) < 1e-6
            status = "PASS" if match else "FAIL"
            if not match:
                deterministic = False
            print(f"    {field:40s}: {v1:>12.4f} vs {v2:>12.4f}  [{status}]")

        if deterministic:
            print()
            print("  DETERMINISM CHECK: PASSED -- Both runs produced identical results.")
        else:
            print()
            print(
                "  DETERMINISM CHECK: FAILED -- Results differ between runs. "
                "This may indicate stochastic components in the environment."
            )
    else:
        print()
        print("  Determinism check skipped (--skip-determinism-check)")

    # --- Save results to JSON ---
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(PROJECT) / "runs" / "r19_ablation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_epoch{args.epoch}_results.json"

    # Make results JSON-serializable
    json_results = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                    for k, v in results_1.items()}
    json_results["checkpoint"] = str(ckpt_path)
    json_results["epoch"] = args.epoch
    json_results["env_vars"] = ENV_VARS

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    # --- Save per-step data as CSV for detailed analysis ---
    csv_path = output_path.with_suffix(".csv")
    try:
        import pandas as pd
        df = pd.DataFrame(step_data_1)
        df.to_csv(csv_path, index=False)
        print(f"  Per-step data saved to: {csv_path} ({len(df)} rows)")
    except ImportError:
        # Fallback without pandas
        with open(csv_path, "w") as f:
            if step_data_1:
                header = ",".join(step_data_1[0].keys())
                f.write(header + "\n")
                for row in step_data_1:
                    f.write(",".join(str(v) for v in row.values()) + "\n")
        print(f"  Per-step data saved to: {csv_path} ({len(step_data_1)} rows)")

    print()
    print("=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
