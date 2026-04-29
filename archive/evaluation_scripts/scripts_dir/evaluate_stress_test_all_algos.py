"""72-hour heat wave stress test for all 7 algorithms.

Evaluates CSAC-LB, CPO, FOCOPS, CUP, PPO-Lag, SAC-Lag, and PPO (unconstrained)
on the maximum contiguous heat wave spell with amplified cooling demand and
reduced solar generation. Also evaluates RBC and Zero baselines.

For each algorithm, evaluates both the "best" checkpoint (min EpCost from
progress.csv) and the "final" checkpoint (highest epoch number).

Outputs a summary table to stdout and saves comprehensive JSON to
runs/stress_test_72h_all_algos.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from gymnasium.spaces import Box

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENDORED_OMNISAFE = PROJECT_ROOT / "vendor_deps"
if str(VENDORED_OMNISAFE) not in sys.path:
    sys.path.insert(0, str(VENDORED_OMNISAFE))

from citylearn_safe.omni_env_temp_cooling_only import (
    CityLearnTempCoolingOnlyMaskedRewardCMDP,
)
from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBS_DIM = 54
ACT_DIM = 3
TEMP_THRESHOLD = 25.0
COOLING_DEMAND_MULTIPLIER = 2.5
SOLAR_GENERATION_MULTIPLIER = 1.2
SEED = 0

DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "runs" / "stress_test_72h_all_algos.json"

ALGORITHMS: list[dict[str, Any]] = [
    {
        "name": "CSAC-LB (seed 1)",
        "run_dir": "runs/csac_lb_multi_seed/seed_1/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-09-11-15-54",
        "actor_type": "sac",
        "best_epoch": None,
    },
    {
        "name": "CPO",
        "run_dir": "runs/cpo_temp_cooling_only/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-20-54-15",
        "actor_type": "ppo",
        "best_epoch": None,
    },
    {
        "name": "FOCOPS",
        "run_dir": "runs/focops_temp_cooling_only/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-21-17-52",
        "actor_type": "ppo",
        "best_epoch": None,
    },
    {
        "name": "CUP",
        "run_dir": "runs/cup_temp_cooling_only/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-21-43-13",
        "actor_type": "ppo",
        "best_epoch": None,
    },
    {
        "name": "PPO-Lag (tight v2)",
        "run_dir": "runs/ppolag_temp_cooling_only_tight_v2/PPOLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-11-44-33",
        "actor_type": "ppo",
        "best_epoch": None,
    },
    {
        "name": "SAC-Lag",
        "run_dir": "runs/saclag_temp_cooling_only_v1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-15-33-32",
        "actor_type": "sac",
        "best_epoch": None,
    },
    {
        "name": "PPO (unconstrained BC)",
        "run_dir": "runs/ppo_temp_cooling_only_masked_reward_bc_40ep/PPOTempMasked-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-03-28-19-39-49",
        "actor_type": "ppo",
        "best_epoch": None,
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class WindowResult:
    """Result of a single rollout on one heat wave window."""

    algo_name: str
    checkpoint_label: str  # "best", "final", "rbc", "zero"
    window_start: int
    window_end: int
    window_hours: int
    total_reward: float
    total_cost: float
    violation_rate: float
    discomfort_rate: float
    mean_cooling_action: float
    total_steps: int
    active_steps: int
    comfort_violations: int
    discomfort_count: float
    occupied_count: float
    reward_economic: float
    reward_stability: float
    reward_renewable: float
    reward_comfort: float
    district_import_kwh: float
    district_net_kwh: float
    kpis: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config / checkpoint helpers
# ---------------------------------------------------------------------------
def _load_run_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _apply_env_overrides(run_dir: Path) -> None:
    """Set environment variables from config.json env_overrides."""
    cfg = _load_run_config(run_dir)
    overrides = cfg.get("env_overrides") or {}
    for key, value in overrides.items():
        os.environ[str(key)] = str(value)


def _clear_env_overrides(run_dir: Path) -> None:
    """Remove environment variables set by _apply_env_overrides."""
    cfg = _load_run_config(run_dir)
    overrides = cfg.get("env_overrides") or {}
    for key in overrides:
        os.environ.pop(str(key), None)


def _best_epoch_from_progress(progress_path: Path) -> int:
    """Find the epoch with the minimum cost in progress.csv.

    Prefers Metrics/TestEpCost if available (SAC-style runs with eval episodes),
    otherwise falls back to Metrics/EpCost (PPO-style runs without eval).
    """
    df = pd.read_csv(progress_path)
    if "Metrics/TestEpCost" in df.columns:
        cost_col = "Metrics/TestEpCost"
    elif "Metrics/EpCost" in df.columns:
        cost_col = "Metrics/EpCost"
    else:
        raise ValueError(
            f"Neither Metrics/TestEpCost nor Metrics/EpCost found in {progress_path}. "
            f"Available columns: {list(df.columns)}"
        )
    idx = df[cost_col].astype(float).idxmin()
    return int(float(df.loc[idx, "Train/Epoch"]))


def _latest_checkpoint_epoch(run_dir: Path) -> int:
    """Return the highest epoch number from saved checkpoints."""
    torch_dir = run_dir / "torch_save"
    epochs: list[int] = []
    for path in torch_dir.glob("epoch-*.pt"):
        try:
            epochs.append(int(path.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    if not epochs:
        raise FileNotFoundError(f"No checkpoints found under {torch_dir}")
    return max(epochs)


def _checkpoint_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / "torch_save" / f"epoch-{epoch}.pt"


# ---------------------------------------------------------------------------
# Actor loading
# ---------------------------------------------------------------------------
def _infer_sac_hidden_sizes(pi_state: dict[str, Any]) -> list[int]:
    """Infer hidden layer sizes from SAC actor state dict."""
    hidden = []
    layer_idx = 0
    while f"net.{layer_idx}.weight" in pi_state:
        weight = pi_state[f"net.{layer_idx}.weight"]
        # All layers except the final output layer are hidden
        if layer_idx + 2 < 5:
            hidden.append(int(weight.shape[0]))
        layer_idx += 2
        if f"net.{layer_idx}.weight" not in pi_state:
            break
    return hidden if len(hidden) >= 2 else [256, 256]


def _load_sac_actor(
    checkpoint_path: Path,
    activation: str = "relu",
) -> GaussianSACActor:
    """Load a GaussianSACActor from a SAC-style checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pi_state = state["pi"]
    obs_dim = int(pi_state["net.0.weight"].shape[1])
    act_dim = int(pi_state["net.4.weight"].shape[0] // 2)
    hidden_sizes = _infer_sac_hidden_sizes(pi_state)

    actor = GaussianSACActor(
        obs_space=Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        act_space=Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
        hidden_sizes=hidden_sizes,
        activation=activation,
        weight_initialization_mode="kaiming_uniform",
    )
    actor.load_state_dict(pi_state, strict=True)
    actor.eval()
    return actor


def _load_ppo_actor(
    checkpoint_path: Path,
    activation: str = "tanh",
) -> tuple[GaussianLearningActor, dict[str, Any] | None]:
    """Load a GaussianLearningActor from a PPO-style checkpoint.

    Returns:
        (actor, obs_normalizer) where obs_normalizer may be None.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pi_state = state["pi"]
    obs_dim = int(pi_state["mean.0.weight"].shape[1])
    act_dim = int(pi_state["mean.4.weight"].shape[0])

    actor = GaussianLearningActor(
        obs_space=Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
        act_space=Box(low=-1.0, high=1.0, shape=(act_dim,)),
        hidden_sizes=[256, 256],
        activation=activation,
        weight_initialization_mode="kaiming_uniform",
    )
    actor.load_state_dict(pi_state, strict=True)
    actor.eval()

    obs_normalizer = state.get("obs_normalizer")
    return actor, obs_normalizer


def _normalize_obs(obs: np.ndarray, obs_normalizer: dict[str, Any]) -> np.ndarray:
    """Apply observation normalization using saved normalizer statistics."""
    mean = obs_normalizer["_mean"].numpy()
    var = obs_normalizer["_var"].numpy()
    clip = obs_normalizer.get("_clip")
    normalized = (obs - mean) / (np.sqrt(var) + 1e-8)
    if clip is not None:
        clip_val = clip.numpy()
        normalized = np.clip(normalized, -clip_val, clip_val)
    return normalized


# ---------------------------------------------------------------------------
# Heat wave window detection
# ---------------------------------------------------------------------------
def _find_max_heatwave_spell(
    temp: np.ndarray, threshold: float
) -> list[tuple[int, int]]:
    """Find the single longest contiguous spell above threshold."""
    hot = np.asarray(temp >= threshold, dtype=bool)
    best: tuple[int, int] | None = None
    best_len = 0
    i = 0
    n = len(hot)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < n and hot[j]:
            j += 1
        span_len = j - i
        if span_len > best_len:
            best = (i, j - 1)
            best_len = span_len
        i = j
    return [] if best is None else [best]


# ---------------------------------------------------------------------------
# Environment creation and perturbation
# ---------------------------------------------------------------------------
def _make_window_env(
    start: int,
    end: int,
    env_id: str = "CityLearnTemp-CoolingOnly-Masked-Reward-v0",
) -> CityLearnTempCoolingOnlyMaskedRewardCMDP:
    """Create an environment restricted to [start, end] timestep window."""
    env = CityLearnTempCoolingOnlyMaskedRewardCMDP(
        env_id,
        citylearn_env_kwargs={
            "episode_time_steps": [[int(start), int(end)]],
            "rolling_episode_split": False,
            "random_episode_split": False,
            "simulation_power_outage": 0,
        },
    )
    return env


def _make_full_env(
    env_id: str = "CityLearnTemp-CoolingOnly-Masked-Reward-v0",
) -> CityLearnTempCoolingOnlyMaskedRewardCMDP:
    """Create a full-year environment (used for temperature probing)."""
    return CityLearnTempCoolingOnlyMaskedRewardCMDP(env_id)


def _apply_heatwave_perturbation(
    env: CityLearnTempCoolingOnlyMaskedRewardCMDP,
    start: int,
    end: int,
    cooling_multiplier: float,
    solar_multiplier: float,
) -> None:
    """Amplify cooling demand and scale solar generation in the window."""
    city = env._get_citylearn()
    assert city is not None, "CityLearn env not initialized"
    for b in city.buildings:
        es = getattr(b, "energy_simulation", None)
        if es is None:
            continue
        # Amplify cooling demand
        cooling = np.asarray(
            getattr(es, "_cooling_demand"), dtype=np.float32
        ).copy()
        cooling[start : end + 1] *= np.float32(cooling_multiplier)
        setattr(es, "_cooling_demand", cooling)

        cooling_wo = np.asarray(
            getattr(es, "_cooling_demand_without_control"), dtype=np.float32
        ).copy()
        cooling_wo[start : end + 1] *= np.float32(cooling_multiplier)
        setattr(es, "_cooling_demand_without_control", cooling_wo)

        # Scale solar generation (>1 = more solar during heat wave)
        solar = np.asarray(
            getattr(es, "_solar_generation"), dtype=np.float32
        ).copy()
        solar[start : end + 1] *= np.float32(solar_multiplier)
        setattr(es, "_solar_generation", solar)


# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------
def _selected_kpis(eval_df: pd.DataFrame) -> dict[str, float]:
    if eval_df is None or eval_df.empty:
        return {}
    cols = {str(c).lower(): c for c in eval_df.columns}
    key_col = cols.get("cost_function")
    value_col = cols.get("value")
    if key_col is None or value_col is None:
        return {}
    out: dict[str, float] = {}
    for _, row in eval_df.iterrows():
        try:
            out[str(row[key_col])] = float(row[value_col])
        except Exception:
            continue
    return out


def _evaluate_citylearn_kpis(
    env: CityLearnTempCoolingOnlyMaskedRewardCMDP,
) -> dict[str, float]:
    city = env._get_citylearn()
    if city is None or not hasattr(city, "evaluate"):
        return {}
    try:
        eval_df = city.evaluate()
    except Exception:
        return {}
    return _selected_kpis(eval_df)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def _rollout_window(
    algo_name: str,
    checkpoint_label: str,
    window: tuple[int, int],
    actor: GaussianSACActor | GaussianLearningActor | None,
    obs_normalizer: dict[str, Any] | None,
    policy_kind: str,
    env_id: str,
    seed: int,
    cooling_multiplier: float,
    solar_multiplier: float,
) -> WindowResult:
    """Run a single rollout on one heat wave window.

    Args:
        policy_kind: one of "sac", "ppo", "rbc", "zero"
    """
    start, end = window
    env = _make_window_env(start, end, env_id)
    _apply_heatwave_perturbation(env, start, end, cooling_multiplier, solar_multiplier)
    obs, _ = env.reset(seed=seed)

    terminated = truncated = False
    total_reward = 0.0
    total_cost = 0.0
    total_steps = 0
    active_steps = 0
    comfort_violations = 0
    occupied_count = 0.0
    discomfort_count = 0.0
    action_sum = 0.0
    action_count = 0
    reward_economic = 0.0
    reward_stability = 0.0
    reward_renewable = 0.0
    reward_comfort = 0.0
    district_import_kwh = 0.0
    district_net_kwh = 0.0

    while not (bool(terminated) or bool(truncated)):
        if policy_kind == "zero":
            action = np.zeros(ACT_DIM, dtype=np.float32)
        elif policy_kind == "rbc":
            action = env._rbc_cooling_action()
        elif policy_kind in ("sac", "ppo"):
            assert actor is not None
            obs_input = obs
            if obs_normalizer is not None:
                obs_input = _normalize_obs(obs_input, obs_normalizer)
            obs_t = torch.as_tensor(obs_input, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                raw_action = (
                    actor.predict(obs_t, deterministic=True).cpu().numpy().reshape(-1)
                )
            action = np.clip((raw_action + 1.0) * 0.5, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown policy_kind: {policy_kind}")

        obs, reward, cost, terminated, truncated, info = env.step(action)
        total_steps += 1
        total_reward += float(reward)
        total_cost += float(cost)
        if float(info.get("comfort_in_warmup", 0.0)) < 0.5:
            active_steps += 1
            comfort_violations += int(
                float(info.get("comfort_violation", 0.0)) > 0.5
            )
        occupied_count += float(info.get("occupied_count", 0.0))
        discomfort_count += float(info.get("discomfort_count", 0.0))
        action_sum += float(np.sum(action))
        action_count += int(np.size(action))
        reward_economic += float(info.get("reward_economic", 0.0))
        reward_stability += float(info.get("reward_stability", 0.0))
        reward_renewable += float(info.get("reward_renewable", 0.0))
        reward_comfort += float(info.get("reward_comfort", 0.0))
        district_import_kwh += float(info.get("district_import_kwh", 0.0))
        district_net_kwh += float(info.get("district_net_kwh", 0.0))

    kpis = _evaluate_citylearn_kpis(env)

    return WindowResult(
        algo_name=algo_name,
        checkpoint_label=checkpoint_label,
        window_start=start,
        window_end=end,
        window_hours=end - start + 1,
        total_reward=total_reward,
        total_cost=total_cost,
        violation_rate=(comfort_violations / active_steps) if active_steps > 0 else 0.0,
        discomfort_rate=(discomfort_count / occupied_count)
        if occupied_count > 0
        else 0.0,
        mean_cooling_action=(action_sum / action_count) if action_count > 0 else 0.0,
        total_steps=total_steps,
        active_steps=active_steps,
        comfort_violations=comfort_violations,
        discomfort_count=discomfort_count,
        occupied_count=occupied_count,
        reward_economic=reward_economic,
        reward_stability=reward_stability,
        reward_renewable=reward_renewable,
        reward_comfort=reward_comfort,
        district_import_kwh=district_import_kwh,
        district_net_kwh=district_net_kwh,
        kpis=kpis,
    )


# ---------------------------------------------------------------------------
# Per-algorithm evaluation
# ---------------------------------------------------------------------------
def _evaluate_algorithm(
    algo_cfg: dict[str, Any],
    windows: list[tuple[int, int]],
    seed: int,
) -> list[WindowResult]:
    """Evaluate a single algorithm on all windows (best + final checkpoints)."""
    name = algo_cfg["name"]
    run_dir = PROJECT_ROOT / algo_cfg["run_dir"]
    actor_type = algo_cfg["actor_type"]

    if not run_dir.exists():
        print(f"  WARNING: run_dir does not exist: {run_dir}")
        print(f"  Skipping {name}.")
        return []

    # Apply env overrides for this algorithm
    _apply_env_overrides(run_dir)

    # Load config for activation and env_id
    cfg = _load_run_config(run_dir)
    env_id = cfg.get("env_id", "CityLearnTemp-CoolingOnly-Masked-Reward-v0")
    actor_activation = (
        cfg.get("model_cfgs", {}).get("actor", {}).get("activation", "relu")
    )

    # Determine best and final epochs
    progress_path = run_dir / "progress.csv"
    best_epoch = algo_cfg["best_epoch"]
    if best_epoch is None:
        if progress_path.exists():
            best_epoch = _best_epoch_from_progress(progress_path)
        else:
            print(f"  WARNING: progress.csv not found for {name}, using epoch 0")
            best_epoch = 0

    final_epoch = _latest_checkpoint_epoch(run_dir)

    print(f"  best_epoch={best_epoch}, final_epoch={final_epoch}")

    results: list[WindowResult] = []

    # Evaluate both best and final, skip duplicate if they are the same
    epochs_to_eval = [("best", best_epoch)]
    if final_epoch != best_epoch:
        epochs_to_eval.append(("final", final_epoch))
    else:
        epochs_to_eval.append(("final (=best)", final_epoch))

    for label, epoch in epochs_to_eval:
        ckpt_path = _checkpoint_path(run_dir, epoch)
        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint not found: {ckpt_path}, skipping {label}")
            continue

        # Load actor
        obs_normalizer = None
        if actor_type == "sac":
            actor = _load_sac_actor(ckpt_path, activation=actor_activation)
        elif actor_type == "ppo":
            actor, obs_normalizer = _load_ppo_actor(
                ckpt_path, activation=actor_activation
            )
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}")

        print(f"  Rolling out {label} (epoch {epoch}) on {len(windows)} window(s)...")
        for w_idx, window in enumerate(windows):
            t0 = time.time()
            result = _rollout_window(
                algo_name=name,
                checkpoint_label=f"{label} (ep {epoch})",
                window=window,
                actor=actor,
                obs_normalizer=obs_normalizer,
                policy_kind=actor_type,
                env_id=env_id,
                seed=seed,
                cooling_multiplier=COOLING_DEMAND_MULTIPLIER,
                solar_multiplier=SOLAR_GENERATION_MULTIPLIER,
            )
            elapsed = time.time() - t0
            print(
                f"    window {w_idx}: [{window[0]}-{window[1]}] "
                f"reward={result.total_reward:.1f} cost={result.total_cost:.1f} "
                f"viol={result.violation_rate:.3f} discomf={result.discomfort_rate:.3f} "
                f"({elapsed:.1f}s)"
            )
            results.append(result)

    # Clean up env overrides
    _clear_env_overrides(run_dir)

    return results


def _evaluate_baseline(
    policy_kind: str,
    baseline_name: str,
    windows: list[tuple[int, int]],
    env_id: str,
    seed: int,
) -> list[WindowResult]:
    """Evaluate a baseline (RBC or Zero) on all windows."""
    results: list[WindowResult] = []
    for w_idx, window in enumerate(windows):
        t0 = time.time()
        result = _rollout_window(
            algo_name=baseline_name,
            checkpoint_label=policy_kind,
            window=window,
            actor=None,
            obs_normalizer=None,
            policy_kind=policy_kind,
            env_id=env_id,
            seed=seed,
            cooling_multiplier=COOLING_DEMAND_MULTIPLIER,
            solar_multiplier=SOLAR_GENERATION_MULTIPLIER,
        )
        elapsed = time.time() - t0
        print(
            f"    window {w_idx}: [{window[0]}-{window[1]}] "
            f"reward={result.total_reward:.1f} cost={result.total_cost:.1f} "
            f"viol={result.violation_rate:.3f} discomf={result.discomfort_rate:.3f} "
            f"({elapsed:.1f}s)"
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------
def _aggregate_results(results: list[WindowResult]) -> dict[str, float]:
    """Compute mean metrics across windows."""
    if not results:
        return {}
    metrics = {
        "total_reward": np.mean([r.total_reward for r in results]),
        "total_cost": np.mean([r.total_cost for r in results]),
        "violation_rate": np.mean([r.violation_rate for r in results]),
        "discomfort_rate": np.mean([r.discomfort_rate for r in results]),
        "mean_cooling_action": np.mean([r.mean_cooling_action for r in results]),
        "reward_economic": np.mean([r.reward_economic for r in results]),
        "reward_stability": np.mean([r.reward_stability for r in results]),
        "reward_renewable": np.mean([r.reward_renewable for r in results]),
        "reward_comfort": np.mean([r.reward_comfort for r in results]),
        "district_import_kwh": np.mean([r.district_import_kwh for r in results]),
        "district_net_kwh": np.mean([r.district_net_kwh for r in results]),
        "n_windows": len(results),
    }
    # Aggregate KPIs
    all_kpi_keys: set[str] = set()
    for r in results:
        all_kpi_keys.update(r.kpis.keys())
    for k in sorted(all_kpi_keys):
        vals = [r.kpis[k] for r in results if k in r.kpis]
        if vals:
            metrics[f"kpi::{k}"] = float(np.mean(vals))
    return {k: float(v) for k, v in metrics.items()}


def _build_summary_table(
    all_results: dict[str, list[WindowResult]],
) -> pd.DataFrame:
    """Build a summary DataFrame with one row per (algo, checkpoint_label)."""
    rows = []
    for key, results in all_results.items():
        if not results:
            continue
        agg = _aggregate_results(results)
        row = {
            "algorithm": key,
            "n_windows": int(agg.get("n_windows", 0)),
            "avg_reward": agg.get("total_reward", float("nan")),
            "avg_cost": agg.get("total_cost", float("nan")),
            "avg_violation_rate": agg.get("violation_rate", float("nan")),
            "avg_discomfort_rate": agg.get("discomfort_rate", float("nan")),
            "avg_cooling_action": agg.get("mean_cooling_action", float("nan")),
            "avg_r_economic": agg.get("reward_economic", float("nan")),
            "avg_r_stability": agg.get("reward_stability", float("nan")),
            "avg_r_renewable": agg.get("reward_renewable", float("nan")),
            "avg_r_comfort": agg.get("reward_comfort", float("nan")),
            "avg_import_kwh": agg.get("district_import_kwh", float("nan")),
            "avg_net_kwh": agg.get("district_net_kwh", float("nan")),
        }
        # Add KPIs
        for k, v in agg.items():
            if k.startswith("kpi::"):
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _build_json_payload(
    windows: list[tuple[int, int]],
    all_results: dict[str, list[WindowResult]],
) -> dict[str, Any]:
    """Build the comprehensive JSON output."""
    payload: dict[str, Any] = {
        "stress_test_config": {
            "temp_threshold_c": TEMP_THRESHOLD,
            "cooling_demand_multiplier": COOLING_DEMAND_MULTIPLIER,
            "solar_generation_multiplier": SOLAR_GENERATION_MULTIPLIER,
            "window_selection": "max_available_spell",
            "seed": SEED,
        },
        "windows": [
            {"start": int(s), "end": int(e), "hours": int(e - s + 1)}
            for s, e in windows
        ],
        "n_windows": len(windows),
        "summary": {},
        "per_algorithm": {},
    }

    for key, results in all_results.items():
        if not results:
            continue
        agg = _aggregate_results(results)
        payload["summary"][key] = agg
        payload["per_algorithm"][key] = [asdict(r) for r in results]

    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("72-HOUR HEAT WAVE STRESS TEST -- ALL ALGORITHMS")
    print("=" * 80)
    print(f"Temp threshold: {TEMP_THRESHOLD} C")
    print(f"Cooling demand multiplier: {COOLING_DEMAND_MULTIPLIER}x")
    print(f"Solar generation multiplier: {SOLAR_GENERATION_MULTIPLIER}x")
    print(f"Seed: {SEED}")
    print()

    # Step 1: Find heat wave windows using a probe environment
    print("Step 1: Finding heat wave windows...")
    probe_env = _make_full_env()
    city = probe_env._get_citylearn()
    assert city is not None, "Failed to initialize CityLearn"
    temp = np.asarray(
        city.buildings[0].weather.outdoor_dry_bulb_temperature, dtype=np.float32
    )
    windows = _find_max_heatwave_spell(temp, TEMP_THRESHOLD)
    if not windows:
        print(
            f"ERROR: No heat wave spell found at threshold {TEMP_THRESHOLD} C. "
            "Try lowering the threshold."
        )
        sys.exit(1)

    for i, (s, e) in enumerate(windows):
        hours = e - s + 1
        max_temp = float(temp[s : e + 1].max())
        mean_temp = float(temp[s : e + 1].mean())
        print(
            f"  Window {i}: steps [{s}, {e}] = {hours} hours, "
            f"max_temp={max_temp:.1f} C, mean_temp={mean_temp:.1f} C"
        )
    print()

    # Step 2: Evaluate all algorithms
    all_results: dict[str, list[WindowResult]] = {}

    for algo_cfg in ALGORITHMS:
        name = algo_cfg["name"]
        print(f"--- Evaluating: {name} ---")
        results = _evaluate_algorithm(algo_cfg, windows, SEED)

        # Group results by checkpoint label
        label_groups: dict[str, list[WindowResult]] = {}
        for r in results:
            label_groups.setdefault(r.checkpoint_label, []).append(r)

        for label, group in label_groups.items():
            key = f"{name} [{label}]"
            all_results[key] = group

        print()

    # Step 3: Evaluate baselines
    # Use a generic env_id for baselines
    baseline_env_id = "CityLearnTemp-CoolingOnly-Masked-Reward-v0"

    print("--- Evaluating: Cooling RBC ---")
    rbc_results = _evaluate_baseline("rbc", "Cooling RBC", windows, baseline_env_id, SEED)
    all_results["Cooling RBC"] = rbc_results
    print()

    print("--- Evaluating: Zero (no cooling) ---")
    zero_results = _evaluate_baseline("zero", "Zero", windows, baseline_env_id, SEED)
    all_results["Zero (no cooling)"] = zero_results
    print()

    # Step 4: Print summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    summary_df = _build_summary_table(all_results)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 40,
        "display.float_format", "{:.4f}".format,
    ):
        print(summary_df.to_string(index=False))
    print()

    # Step 5: Print focused comparison
    print("=" * 80)
    print("FOCUSED COMPARISON (avg across windows)")
    print("=" * 80)
    print(
        f"{'Algorithm':<40s} {'Reward':>10s} {'Cost':>10s} "
        f"{'ViolRate':>10s} {'Discomf':>10s} {'Action':>10s}"
    )
    print("-" * 90)
    for key in all_results:
        results = all_results[key]
        if not results:
            continue
        agg = _aggregate_results(results)
        print(
            f"{key:<40s} {agg.get('total_reward', 0):>10.1f} "
            f"{agg.get('total_cost', 0):>10.1f} "
            f"{agg.get('violation_rate', 0):>10.4f} "
            f"{agg.get('discomfort_rate', 0):>10.4f} "
            f"{agg.get('mean_cooling_action', 0):>10.4f}"
        )
    print()

    # Step 6: Save JSON
    payload = _build_json_payload(windows, all_results)
    output_path = DEFAULT_OUTPUT_JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved comprehensive JSON to {output_path}")

    # Also print the JSON path at the very end
    print(f"\nDone. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
