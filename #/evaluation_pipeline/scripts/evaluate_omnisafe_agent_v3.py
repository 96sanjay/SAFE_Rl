
"""
Evaluate a trained OmniSafe agent (epoch-XX.pt) in CityLearnSafetyEnvV3 and write a KPI CSV.

This version:
- Loads OmniSafe checkpoint actor with keys: mean.* and log_std (matches your ckpt).
- Runs one full episode (8759 steps).
- Calls raw CityLearn evaluate() at episode end.
- Injects CityLearn episode KPIs into the LAST ROW of the KPI CSV (so comparisons never show NaN).

Usage:
  python evaluation_pipeline/scripts/evaluate_omnisafe_agent_v3.py \
    --checkpoint "runs/.../torch_save/epoch-35.pt" \
    --kpi_name "Evaluated_PPOLag_bill_evscale2_ep35_seed000" \
    --seed 42
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# Ensure repo root is importable
sys.path.insert(0, os.getcwd())

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3


# ---------------------------
# OmniSafe checkpoint helpers
# ---------------------------

class OmniSafeGaussianActor(nn.Module):
    """
    Actor that matches OmniSafe checkpoint structure:
      - self.mean: MLP -> Linear -> Tanh
      - optional log_std in state dict (we ignore it for deterministic actions)
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.mean = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # deterministic mean action, squashed to [-1, 1]
        return torch.tanh(self.mean(obs))


def _infer_hidden_sizes_from_state_dict(pi_state: dict) -> tuple:
    """
    Tries to infer hidden sizes from keys mean.0.weight, mean.2.weight, ...
    Falls back to (64,64) if inference fails.
    """
    try:
        # mean.0 is first Linear: shape (h1, obs_dim)
        h1 = int(pi_state["mean.0.weight"].shape[0])
        # mean.2 is second Linear: shape (h2, h1)
        h2 = int(pi_state["mean.2.weight"].shape[0])
        return (h1, h2)
    except Exception:
        return (64, 64)


def load_policy_from_checkpoint(checkpoint_path: str, obs_dim: int, act_dim: int):
    """
    Loads OmniSafe checkpoint and returns a callable policy(obs)->action in numpy.
    Handles obs normalizer if present in the checkpoint.
    Loads actor weights from ckpt['pi'] where keys are mean.* (and log_std).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "pi" not in ckpt:
        raise KeyError("Checkpoint missing key 'pi'. Cannot load policy weights.")

    pi_state = ckpt["pi"]
    hidden_sizes = _infer_hidden_sizes_from_state_dict(pi_state)

    actor = OmniSafeGaussianActor(obs_dim, act_dim, hidden_sizes=hidden_sizes)

    # Filter out log_std if present (we do deterministic mean actions)
    filtered_state = {k: v for k, v in pi_state.items() if not k.startswith("log_std")}
    missing, unexpected = actor.load_state_dict(filtered_state, strict=False)

    # If something is still off, fail loudly with good info
    if unexpected:
        print("[EVAL] Unexpected keys when loading actor:", unexpected)
    if missing:
        print("[EVAL] Missing keys when loading actor:", missing)

    actor.eval()

    # obs normalizer
    obs_mean = None
    obs_std = None
    if "obs_normalizer" in ckpt and isinstance(ckpt["obs_normalizer"], dict):
        norm = ckpt["obs_normalizer"]
        if "_mean" in norm and "_std" in norm:
            obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
            obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)

    @torch.no_grad()
    def policy(obs_np: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        if obs_mean is not None and obs_std is not None:
            obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
        act_t = actor(obs_t).squeeze(0)
        act = act_t.cpu().numpy()
        return np.clip(act, -1.0, 1.0)

    return policy


# ---------------------------------------
# CityLearn KPI injection (episode end)
# ---------------------------------------

def _extract_citylearn_kpis_from_evaluate(env: CityLearnSafetyEnvV3) -> dict:
    """
    Calls CityLearn evaluate() on the raw env and extracts District KPI values.
    """
    raw = None
    try:
        raw = env._get_citylearn_env()
    except Exception:
        raw = None

    if raw is None or not hasattr(raw, "evaluate"):
        return {}

    try:
        df = raw.evaluate()
    except Exception as e:
        print(f"[EVAL] WARNING: raw.evaluate() failed: {e}")
        return {}

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}

    # District row if present
    if "name" in df.columns and (df["name"] == "District").any():
        district_name = "District"
    elif "name" in df.columns:
        district_name = df["name"].iloc[0]
    else:
        district_name = None

    def get_value(cost_function: str) -> float:
        try:
            if district_name is not None and "name" in df.columns:
                sub = df[(df["name"] == district_name) & (df["cost_function"] == cost_function)]
            else:
                sub = df[df["cost_function"] == cost_function]
            if sub.empty:
                return float("nan")
            v = sub["value"].iloc[0]
            return float(v) if pd.notna(v) else float("nan")
        except Exception:
            return float("nan")

    return {
        "citylearn_electricity_consumption_total": get_value("electricity_consumption_total"),
        "citylearn_carbon_emissions_total": get_value("carbon_emissions_total"),
        "citylearn_cost_total": get_value("cost_total"),
        "citylearn_daily_peak_average": get_value("daily_peak_average"),
        "citylearn_all_time_peak_average": get_value("all_time_peak_average"),
        "citylearn_ramping_average": get_value("ramping_average"),
        "citylearn_discomfort_proportion": get_value("discomfort_proportion"),
        "citylearn_zero_net_energy": get_value("zero_net_energy"),
    }


def inject_citylearn_kpis_into_last_row(kpi_csv_path: str, citylearn_kpis: dict) -> None:
    """
    Overwrite the last row in KPI CSV with the provided citylearn_* values.
    """
    if not citylearn_kpis:
        print("[EVAL] No CityLearn KPIs to inject (evaluate() returned empty).")
        return

    if not os.path.exists(kpi_csv_path):
        print(f"[EVAL] WARNING: KPI CSV not found for injection: {kpi_csv_path}")
        return

    df = pd.read_csv(kpi_csv_path)
    if df.empty:
        print(f"[EVAL] WARNING: KPI CSV empty: {kpi_csv_path}")
        return

    for k, v in citylearn_kpis.items():
        if k not in df.columns:
            df[k] = np.nan
        df.loc[df.index[-1], k] = v

    df.to_csv(kpi_csv_path, index=False)
    print(f"[EVAL] Injected CityLearn KPIs into last row: {kpi_csv_path}")


# ---------------------------
# Evaluation main
# ---------------------------

def evaluate(checkpoint_path: str, kpi_name: str, seed: int):
    # Required env var: schema
    os.environ.setdefault(
        "CITYLEARN_SCHEMA",
        "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json",
    )

    # KPI CSV name
    os.environ["CITYLEARN_KPI_RUN_NAME"] = kpi_name

    # Keep consistent with training unless you override
    os.environ.setdefault("CITYLEARN_EXPORT_FACTOR", "0.7")
    os.environ.setdefault("CITYLEARN_REWARD_SCALE", "1.0")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create env
    base_env = make_base_env(central_agent=True)
    env = CityLearnSafetyEnvV3(
        base_env,
        soc_min=0.0,
        soc_max=0.95,
        cost_mode="hinge",
        include_ev_in_cost=True,
    )

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    print("=" * 90)
    print(f"EVALUATING CHECKPOINT: {checkpoint_path}")
    print(f"KPI name: {kpi_name}")
    print("=" * 90)

    policy = load_policy_from_checkpoint(checkpoint_path, obs_dim, act_dim)

    # Rollout
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    total_cost = 0.0
    cost_pos_steps = 0

    steps = 0
    done = False

    while not done:
        action = policy(obs)
        obs, reward, term, trunc, info = env.step(action)
        done = bool(term) or bool(trunc)

        steps += 1
        total_reward += float(reward)
        c = float(info.get("cost", 0.0))
        total_cost += c
        if c > 0:
            cost_pos_steps += 1

        if steps % 1000 == 0:
            print(f"[EVAL] step {steps}/8759")

    # Extract and inject CityLearn evaluate() KPIs
    citylearn_kpis = _extract_citylearn_kpis_from_evaluate(env)

    kpi_csv_path = os.path.join(os.getcwd(), "runs", "kpi_logs", f"{kpi_name}.csv")
    inject_citylearn_kpis_into_last_row(kpi_csv_path, citylearn_kpis)

    # Print summary + normalized EV metrics from KPI CSV
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"steps: {steps}")
    print(f"total_reward: {total_reward:.3f}")
    print(f"total_cost:   {total_cost:.3f}")
    print(f"cost>0 steps: {cost_pos_steps} ({100.0*cost_pos_steps/max(1,steps):.2f}%)")

    try:
        df = pd.read_csv(kpi_csv_path)

        dep = (df["ev_departure_departures"].astype(float).fillna(0) > 0)
        deficit = df["ev_departure_deficit_kwh"].astype(float).fillna(0)
        avoid = df["ev_avoidable_deficit_kwh"].astype(float).fillna(0)

        dep_steps = int(dep.sum())
        miss = int(((dep) & (deficit > 0)).sum())

        missing_actions = 0
        if "ev_missing_action_samples" in df.columns:
            missing_actions = int(df["ev_missing_action_samples"].astype(float).fillna(0).sum())

        print("\nEV V3 totals (from KPI CSV):")
        print(f"  controllable (avoidable): {avoid.sum():.3f} kWh")
        print(f"  total deficit:           {deficit.sum():.3f} kWh")
        print(f"  missing action samples:  {missing_actions}")

        print("\nNormalized EV metrics:")
        print(f"  miss_rate:        {miss/max(1,dep_steps)}")
        print(f"  avoidable_per_dep:{avoid.sum()/max(1,dep_steps)}")
        print(f"  deficit_per_dep:  {deficit.sum()/max(1,dep_steps)}")

        print("\nCityLearn episode-end KPIs (from last row):")
        last = df.tail(1)
        for k in [
            "citylearn_electricity_consumption_total",
            "citylearn_carbon_emissions_total",
            "citylearn_cost_total",
            "citylearn_daily_peak_average",
            "citylearn_all_time_peak_average",
            "citylearn_ramping_average",
            "citylearn_discomfort_proportion",
            "citylearn_zero_net_energy",
        ]:
            if k in df.columns:
                print(f"  {k:40s}: {last[k].iloc[0]}")

    except Exception as e:
        print(f"[EVAL] WARNING: could not read/compute from KPI CSV: {e}")

    print(f"\nKPI CSV: {kpi_csv_path}")
    print("=" * 90)

    try:
        env.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to epoch-XX.pt (or glob pattern)")
    parser.add_argument("--kpi_name", required=True, help="CSV will be runs/kpi_logs/<kpi_name>.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt = args.checkpoint
    if "*" in ckpt:
        matches = sorted(glob.glob(ckpt))
        if not matches:
            raise FileNotFoundError(f"No checkpoints matched glob: {ckpt}")
        ckpt = matches[-1]

    evaluate(ckpt, args.kpi_name, args.seed)


if __name__ == "__main__":
    main()
