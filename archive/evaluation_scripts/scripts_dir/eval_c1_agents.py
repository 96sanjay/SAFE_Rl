#!/usr/bin/env python3
"""
eval_c1_agents.py -- Evaluate two c1 agents (SoC-v0 and Forecast-v0) trained with
TRPOLag on the 17-building CityLearn environment.

Collects per-constraint violation percentages, CityLearn KPIs, reward statistics,
and total CMDP cost. Prints a comparison table and saves results.

Usage:
    python scripts/eval_c1_agents.py
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Project root and path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Environment variables (must be set BEFORE importing citylearn modules)
# ---------------------------------------------------------------------------
os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_STEMS_SOC_LOW"] = "0.0"
os.environ["CITYLEARN_STEMS_SOC_HIGH"] = "0.95"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_full"
os.environ["CITYLEARN_COST_MODE"] = "hinge"

# Suppress excessive debug printing from safety env
os.environ["CITYLEARN_KPI_FLUSH_EVERY_STEP"] = "0"
os.environ["CITYLEARN_DEBUG_ACTION_CLIP"] = "0"
os.environ["CITYLEARN_DEBUG_OBS_VS_STATE"] = "0"

# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------
SCHEMA_CANDIDATES = [
    PROJECT_ROOT / "data" / "citylearn_challenge_2022_phase_all" / "schema.json",
    PROJECT_ROOT / "data" / "citylearn_challenge_2022_phase_all_plus_evs" / "schema.json",
]

SCHEMA_PATH = None
for candidate in SCHEMA_CANDIDATES:
    if candidate.exists():
        SCHEMA_PATH = str(candidate)
        break

if SCHEMA_PATH is None:
    print("ERROR: No schema file found. Tried:")
    for c in SCHEMA_CANDIDATES:
        print(f"  {c}")
    sys.exit(1)

os.environ["CITYLEARN_SCHEMA"] = SCHEMA_PATH
print(f"[eval_c1] Using schema: {SCHEMA_PATH}")

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------
AGENTS = {
    "c1_B_shaped": {
        "label": "c1_B_shaped (TRPOLag, SoC-v0)",
        "checkpoint": str(
            PROJECT_ROOT / "runs" / "c1_B_shaped"
            / "TRPOLag-{CityLearnSafety-SoC-v0}"
            / "seed-042-2026-02-25-05-06-47"
            / "torch_save" / "epoch-70.pt"
        ),
        "obs_dim": 153,
        "act_dim": 26,
        "env_type": "soc",
    },
    "c1_C_forecast": {
        "label": "c1_C_forecast (TRPOLag, Forecast-v0)",
        "checkpoint": str(
            PROJECT_ROOT / "runs" / "c1_C_forecast"
            / "TRPOLag-{CityLearnSafety-Forecast-v0}"
            / "seed-042-2026-02-25-05-06-48"
            / "torch_save" / "epoch-70.pt"
        ),
        "obs_dim": 281,
        "act_dim": 26,
        "env_type": "forecast",
    },
}

MAX_STEPS = 8759

# ---------------------------------------------------------------------------
# Actor MLP builder
# ---------------------------------------------------------------------------

def build_actor_mlp(checkpoint_path: str):
    """Load checkpoint and build a deterministic actor MLP (mean network).

    Returns:
        actor: nn.Sequential -- the mean network
        obs_mean: np.ndarray
        obs_std: np.ndarray
        obs_clip: np.ndarray
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    pi = ckpt["pi"]
    obs_norm = ckpt["obs_normalizer"]

    # Detect layer structure from checkpoint keys: mean.0, mean.2, mean.4, mean.6
    layer_indices = sorted(set(
        int(k.split(".")[1])
        for k in pi.keys()
        if k.startswith("mean.") and k.split(".")[1].isdigit()
    ))

    layers = []
    for i, idx in enumerate(layer_indices):
        w = pi[f"mean.{idx}.weight"]
        b = pi[f"mean.{idx}.bias"]
        linear = nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data.copy_(w)
        linear.bias.data.copy_(b)
        layers.append(linear)
        # Add activation after every layer EXCEPT the last (output layer uses Identity)
        if i < len(layer_indices) - 1:
            layers.append(nn.ReLU())

    actor = nn.Sequential(*layers)
    actor.eval()

    obs_mean = obs_norm["_mean"].numpy()
    obs_std = obs_norm["_std"].numpy()
    obs_clip = obs_norm["_clip"].numpy()

    return actor, obs_mean, obs_std, obs_clip


def normalize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Apply running-mean normalization identical to OmniSafe's obs_normalizer."""
    return np.clip((obs - mean) / (std + 1e-8), -clip, clip).astype(np.float32)


def select_action(actor: nn.Module, obs_norm: np.ndarray) -> np.ndarray:
    """Deterministic forward pass (mean of Gaussian policy)."""
    with torch.no_grad():
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        action = actor(obs_t).squeeze(0).numpy()
    return action


# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------

def build_soc_env():
    """Build SoC-v0 environment (no forecast wrapper)."""
    from citylearn.citylearn import CityLearnEnv
    from citylearn.wrappers import NormalizedObservationWrapper
    from citylearn_safe.adapters import SingleAgentListAdapter
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

    base = CityLearnEnv(schema=SCHEMA_PATH, central_agent=True)
    base = NormalizedObservationWrapper(base)
    base = SingleAgentListAdapter(base)
    env = CityLearnSafetyEnvV3(base, soc_min=0.0, soc_max=0.95, cost_mode="hinge")
    return env


def build_forecast_env():
    """Build Forecast-v0 environment (with NormalizedForecastObsWrapper)."""
    from citylearn.citylearn import CityLearnEnv
    from citylearn.wrappers import NormalizedObservationWrapper
    from citylearn_safe.adapters import SingleAgentListAdapter
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.omni_env_forecast import NormalizedForecastObsWrapper

    base = CityLearnEnv(schema=SCHEMA_PATH, central_agent=True)
    base = NormalizedObservationWrapper(base)
    base = SingleAgentListAdapter(base)
    env = CityLearnSafetyEnvV3(base, soc_min=0.0, soc_max=0.95, cost_mode="hinge")
    env = NormalizedForecastObsWrapper(env, forecast_horizon=24)
    return env


def get_citylearn_base(env):
    """Unwrap through wrapper chain to find the underlying CityLearnEnv.

    Walks through .base, .env, .unwrapped, ._env attributes until finding
    an object with .buildings and .time_step.
    """
    cur = env
    seen = set()
    for _ in range(50):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if (hasattr(cur, "buildings") and hasattr(cur, "time_step")
                and hasattr(cur.buildings, "__len__") and len(cur.buildings) > 0):
            return cur
        for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None


# ---------------------------------------------------------------------------
# Cost key extraction (handles possible name variations)
# ---------------------------------------------------------------------------

# Mapping from our canonical name to possible info dict keys
COST_KEY_MAP = {
    "C1_ev_departure": ["cost_ev_departure", "cost_ev_departure_avoidable"],
    "C2_battery_soc": ["cost_stems_battery", "cost_building_soc"],
    "C3_building_power": ["cost_stems_building_power"],
    "C4_grid_power": ["cost_stems_grid_power"],
}

# Violation indicator keys (1.0 if violated at this step)
VIOLATION_KEY_MAP = {
    "C1_ev_departure": ["cost_ev_departure"],  # >0 means violation
    "C2_battery_soc": ["battery_soc_violation", "battery_soc_violation_any"],
    "C3_building_power": ["building_power_violation"],
    "C4_grid_power": ["grid_power_violation"],
}


def get_cost_value(info: dict, canonical_name: str) -> float:
    """Extract cost value from info dict, trying multiple key variants."""
    for key in COST_KEY_MAP.get(canonical_name, []):
        if key in info:
            return float(info[key])
    return 0.0


def get_violation_indicator(info: dict, canonical_name: str) -> float:
    """Return 1.0 if constraint is violated this step, 0.0 otherwise."""
    for key in VIOLATION_KEY_MAP.get(canonical_name, []):
        if key in info:
            val = float(info[key])
            if val > 0.0:
                return 1.0
    # Fallback: check if cost > 0
    cost = get_cost_value(info, canonical_name)
    return 1.0 if cost > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_agent(agent_name: str, agent_cfg: dict) -> dict:
    """Run one full episode and collect all metrics."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {agent_cfg['label']}")
    print(f"Checkpoint: {agent_cfg['checkpoint']}")
    print(f"{'='*70}")

    # Load actor
    if not os.path.exists(agent_cfg["checkpoint"]):
        print(f"ERROR: Checkpoint not found: {agent_cfg['checkpoint']}")
        return None

    actor, obs_mean, obs_std, obs_clip = build_actor_mlp(agent_cfg["checkpoint"])
    print(f"  Actor loaded: obs_dim={agent_cfg['obs_dim']}, act_dim={agent_cfg['act_dim']}")
    print(f"  Actor architecture: {actor}")

    # Build environment
    if agent_cfg["env_type"] == "soc":
        env = build_soc_env()
    elif agent_cfg["env_type"] == "forecast":
        env = build_forecast_env()
    else:
        raise ValueError(f"Unknown env_type: {agent_cfg['env_type']}")

    obs_space_dim = env.observation_space.shape[0]
    act_space_dim = env.action_space.shape[0]
    print(f"  Env obs_dim={obs_space_dim}, act_dim={act_space_dim}")

    # Verify dimensions match
    if obs_space_dim != agent_cfg["obs_dim"]:
        print(f"  WARNING: obs dim mismatch! env={obs_space_dim}, checkpoint={agent_cfg['obs_dim']}")
    if act_space_dim != agent_cfg["act_dim"]:
        print(f"  WARNING: act dim mismatch! env={act_space_dim}, checkpoint={agent_cfg['act_dim']}")

    # Reset
    obs, info = env.reset()
    obs = np.asarray(obs, dtype=np.float32).ravel()

    # Tracking
    total_reward = 0.0
    total_cmdp_cost = 0.0
    step_rewards = []
    constraint_costs = {name: [] for name in COST_KEY_MAP}
    constraint_violations = {name: 0 for name in VIOLATION_KEY_MAP}
    all_infos = []

    t_start = time.time()

    for step_i in range(MAX_STEPS):
        # Normalize obs and select action
        obs_n = normalize_obs(obs, obs_mean, obs_std, obs_clip)
        action = select_action(actor, obs_n)

        # Clip action to env bounds
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Step
        obs_next, reward, terminated, truncated, info = env.step(action)
        obs_next = np.asarray(obs_next, dtype=np.float32).ravel()

        # Accumulate reward
        reward_f = float(reward)
        total_reward += reward_f
        step_rewards.append(reward_f)

        # Accumulate CMDP cost
        cmdp_cost = float(info.get("cost", 0.0))
        total_cmdp_cost += cmdp_cost

        # Per-constraint costs and violations
        for cname in COST_KEY_MAP:
            cost_val = get_cost_value(info, cname)
            constraint_costs[cname].append(cost_val)
            viol = get_violation_indicator(info, cname)
            if viol > 0.0:
                constraint_violations[cname] += 1

        # Store subset of info for analysis (not all, to save memory)
        all_infos.append({
            "step": step_i,
            "reward": reward_f,
            "cost": cmdp_cost,
            "C1": get_cost_value(info, "C1_ev_departure"),
            "C2": get_cost_value(info, "C2_battery_soc"),
            "C3": get_cost_value(info, "C3_building_power"),
            "C4": get_cost_value(info, "C4_grid_power"),
        })

        obs = obs_next

        if terminated or truncated:
            print(f"  Episode ended at step {step_i} (terminated={terminated}, truncated={truncated})")
            break

        # Progress
        if (step_i + 1) % 2000 == 0:
            elapsed = time.time() - t_start
            print(f"  Step {step_i+1}/{MAX_STEPS} "
                  f"({elapsed:.1f}s, reward_so_far={total_reward:.1f}, "
                  f"cost_so_far={total_cmdp_cost:.1f})")

    total_steps = len(step_rewards)
    elapsed = time.time() - t_start
    print(f"  Episode complete: {total_steps} steps in {elapsed:.1f}s")

    # CityLearn KPIs
    # evaluate() returns DataFrame with columns: cost_function, value, name, level
    # We want district-level rows: filter by level == 'district'
    citylearn_kpis = {}
    city_env = get_citylearn_base(env)
    if city_env is not None:
        try:
            kpi_df = city_env.evaluate()
            # Filter to district-level metrics
            if "level" in kpi_df.columns:
                district_df = kpi_df[kpi_df["level"] == "district"]
            elif "name" in kpi_df.columns:
                district_df = kpi_df[kpi_df["name"].str.lower() == "district"]
            else:
                district_df = kpi_df

            for _, row in district_df.iterrows():
                kpi_name = str(row.get("cost_function", ""))
                kpi_val = row.get("value", float("nan"))
                try:
                    citylearn_kpis[kpi_name] = float(kpi_val)
                except (ValueError, TypeError):
                    citylearn_kpis[kpi_name] = float("nan")

            print(f"  CityLearn KPIs collected: {len(citylearn_kpis)} metrics")
        except Exception as e:
            print(f"  WARNING: Failed to get CityLearn KPIs: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  WARNING: Could not unwrap to CityLearnEnv for KPIs")

    # Compute violation percentages
    violation_pcts = {}
    for cname in constraint_violations:
        violation_pcts[cname] = (constraint_violations[cname] / max(1, total_steps)) * 100.0

    # Compute total costs per constraint
    total_costs = {}
    for cname in constraint_costs:
        total_costs[cname] = sum(constraint_costs[cname])

    # Build results dict
    results = {
        "agent": agent_name,
        "label": agent_cfg["label"],
        "checkpoint": agent_cfg["checkpoint"],
        "total_steps": total_steps,
        "total_reward": total_reward,
        "avg_reward_per_step": total_reward / max(1, total_steps),
        "total_cmdp_cost": total_cmdp_cost,
        "violation_pcts": violation_pcts,
        "total_costs_per_constraint": total_costs,
        "citylearn_kpis": citylearn_kpis,
        "elapsed_seconds": elapsed,
    }

    env.close()
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(all_results: dict) -> str:
    """Format a human-readable comparison table."""
    lines = []
    lines.append("=" * 80)
    lines.append("  C1 AGENT EVALUATION REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Schema: {SCHEMA_PATH}")
    lines.append("=" * 80)

    agent_names = list(all_results.keys())
    if len(agent_names) == 0:
        lines.append("  No agents evaluated.")
        return "\n".join(lines)

    # Header
    col_width = 30
    header = f"  {'Metric':<35}"
    for name in agent_names:
        header += f"  {name:>{col_width}}"
    lines.append("")
    lines.append(header)
    lines.append("  " + "-" * (35 + (col_width + 2) * len(agent_names)))

    # Reward section
    lines.append("")
    lines.append("  --- Reward ---")
    for metric_name, key in [
        ("Total Episode Reward", "total_reward"),
        ("Avg Reward/Step", "avg_reward_per_step"),
    ]:
        row = f"  {metric_name:<35}"
        for name in agent_names:
            r = all_results[name]
            if r is None:
                row += f"  {'N/A':>{col_width}}"
            else:
                row += f"  {r[key]:>{col_width}.4f}"
        lines.append(row)

    # CMDP cost
    lines.append("")
    lines.append("  --- CMDP Cost ---")
    row = f"  {'Total CMDP Cost':<35}"
    for name in agent_names:
        r = all_results[name]
        if r is None:
            row += f"  {'N/A':>{col_width}}"
        else:
            row += f"  {r['total_cmdp_cost']:>{col_width}.2f}"
    lines.append(row)

    # Per-constraint violation %
    lines.append("")
    lines.append("  --- Constraint Violation Rates (% of steps) ---")
    constraint_labels = {
        "C1_ev_departure": "C1: EV Departure SoC",
        "C2_battery_soc": "C2: Battery SoC Band",
        "C3_building_power": "C3: Building Power Cap",
        "C4_grid_power": "C4: Grid Power Cap",
    }
    for cname, clabel in constraint_labels.items():
        row = f"  {clabel:<35}"
        for name in agent_names:
            r = all_results[name]
            if r is None:
                row += f"  {'N/A':>{col_width}}"
            else:
                pct = r["violation_pcts"].get(cname, 0.0)
                row += f"  {pct:>{col_width}.2f}%"
        lines.append(row)

    # Per-constraint total costs
    lines.append("")
    lines.append("  --- Total Constraint Costs (sum over episode) ---")
    for cname, clabel in constraint_labels.items():
        row = f"  {clabel:<35}"
        for name in agent_names:
            r = all_results[name]
            if r is None:
                row += f"  {'N/A':>{col_width}}"
            else:
                total_c = r["total_costs_per_constraint"].get(cname, 0.0)
                row += f"  {total_c:>{col_width}.2f}"
        lines.append(row)

    # CityLearn KPIs
    lines.append("")
    lines.append("  --- CityLearn KPIs (district-level, normalized to no-control) ---")
    # Collect all unique KPI names across all agents
    all_kpi_names = set()
    for name in agent_names:
        r = all_results[name]
        if r is not None:
            all_kpi_names.update(r.get("citylearn_kpis", {}).keys())

    for kpi_name in sorted(all_kpi_names):
        # Use wider column for long KPI names
        display_name = kpi_name[:50] if len(kpi_name) > 50 else kpi_name
        row = f"  {display_name:<52}"
        for name in agent_names:
            r = all_results[name]
            if r is None:
                row += f"  {'N/A':>{col_width-5}}"
            else:
                val = r.get("citylearn_kpis", {}).get(kpi_name, float("nan"))
                if isinstance(val, float) and not np.isnan(val):
                    row += f"  {val:>{col_width-5}.6f}"
                else:
                    row += f"  {'N/A':>{col_width-5}}"
        lines.append(row)

    # Runtime
    lines.append("")
    lines.append("  --- Runtime ---")
    row = f"  {'Total Steps':<35}"
    for name in agent_names:
        r = all_results[name]
        if r is None:
            row += f"  {'N/A':>{col_width}}"
        else:
            row += f"  {r['total_steps']:>{col_width}d}"
    lines.append(row)

    row = f"  {'Elapsed (seconds)':<35}"
    for name in agent_names:
        r = all_results[name]
        if r is None:
            row += f"  {'N/A':>{col_width}}"
        else:
            row += f"  {r['elapsed_seconds']:>{col_width}.1f}"
    lines.append(row)

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def make_json_serializable(obj):
    """Recursively convert numpy types to Python builtins for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[eval_c1] Starting evaluation at {datetime.now()}")
    print(f"[eval_c1] Project root: {PROJECT_ROOT}")

    all_results = {}

    for agent_name, agent_cfg in AGENTS.items():
        try:
            result = evaluate_agent(agent_name, agent_cfg)
            all_results[agent_name] = result
        except Exception as e:
            print(f"\nERROR evaluating {agent_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[agent_name] = None

    # Generate report
    report = format_report(all_results)
    print("\n" + report)

    # Save outputs
    output_dir = PROJECT_ROOT / "runs" / "c1_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "eval_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n[eval_c1] Report saved to: {report_path}")

    json_path = output_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(make_json_serializable(all_results), f, indent=2)
    print(f"[eval_c1] JSON results saved to: {json_path}")

    print(f"\n[eval_c1] Evaluation complete at {datetime.now()}")


if __name__ == "__main__":
    main()
