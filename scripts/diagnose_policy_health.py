#!/usr/bin/env python3
"""
Policy Health Diagnostic Suite
===============================

Runs 8 mathematical tests on RL checkpoints to identify what the policy
fails to learn. Each test produces a scalar health score in [0, 1] plus
structured diagnostics that feed into a final report.

Tests:
  1. Value function accuracy (test_value_function)
  2. Feature-action mutual information (test_feature_action_mi)
  3. Conditional entropy of actions (test_conditional_entropy)
  4. Inter-building action correlation (test_action_correlation)
  5. Gradient attribution (test_gradient_attribution)
  6. Temporal planning horizon (test_temporal_planning)
  7. Constraint decomposition (test_constraint_decomposition)
  8. Safety headroom (test_headroom)

Run:
  cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
  conda run -n citylearn python scripts/diagnose_policy_health.py \\
      --checkpoint path/to/checkpoint --output-dir /tmp/diag_out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_BUILDINGS = 5
TEMPORAL_WINDOW = 12
TEMPORAL_FEATURES_PER_STEP = 11
CURRENT_OBS_DIM = 198
OBS_DIM = 330  # 198 current + 12*11 history
ACT_DIM = 9
GAMMA = 0.99

# ---------------------------------------------------------------------------
# Feature index groups
# ---------------------------------------------------------------------------
PRICE_IDX = 22
SOC_INDICES = list(range(23, 28))
HOUR_COS_IDX = 4
HOUR_SIN_IDX = 5
HISTORY_START = 198
HISTORY_END = 330


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the diagnostic suite."""
    parser = argparse.ArgumentParser(
        description="Policy Health Diagnostic Suite — "
        "runs 8 tests on RL checkpoints to identify learning failures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the OmniSafe checkpoint directory or torch_save folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write JSON report and plots. "
        "Defaults to <checkpoint>/diagnostics/.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training YAML config (used to reconstruct env if needed).",
    )
    parser.add_argument(
        "--skip-env",
        action="store_true",
        default=False,
        help="Skip tests that require a live environment rollout.",
    )
    parser.add_argument(
        "--rollout-data",
        type=str,
        default=None,
        help="Path to a pre-saved rollout .npz file. "
        "If provided, skips live rollout collection.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def collect_rollout(
    actor: Any,
    env: Any,
    deterministic: bool = True,
) -> Dict[str, np.ndarray]:
    """Run one full episode and collect per-step data.

    Parameters
    ----------
    actor : torch.nn.Module
        The policy network (must accept obs tensor, return action tensor).
    env : CMDP environment
        OmniSafe-wrapped CityLearn env. Step returns
        (obs, reward, cost, terminated, truncated, info).
    deterministic : bool
        If True, use the mean action (no sampling).

    Returns
    -------
    dict of np.ndarray
        Keys:
          - obs:       (T, OBS_DIM)
          - actions:   (T, ACT_DIM)
          - rewards:   (T,)
          - costs:     (T,)
          - reward_economic:         (T,)
          - reward_stability_grid:   (T,)
          - reward_stability_building: (T,)
          - reward_ramp:             (T,)
          - reward_renewable:        (T,)
          - cost_C1:   (T,)  — cost_ev_departure
          - cost_C2:   (T,)  — cost_stems_battery
          - cost_C3:   (T,)  — cost_stems_building_power
          - cost_C4:   (T,)  — cost_stems_grid_power
    """
    import torch

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    rew_list: List[float] = []
    cost_list: List[float] = []

    # Reward components
    rew_economic: List[float] = []
    rew_stability_grid: List[float] = []
    rew_stability_building: List[float] = []
    rew_ramp: List[float] = []
    rew_renewable: List[float] = []

    # Cost components
    cost_c1: List[float] = []
    cost_c2: List[float] = []
    cost_c3: List[float] = []
    cost_c4: List[float] = []

    obs, info = env.reset()
    if isinstance(obs, torch.Tensor):
        obs_np = obs.detach().cpu().numpy().flatten()
    else:
        obs_np = np.asarray(obs).flatten()

    done = False
    while not done:
        obs_list.append(obs_np.copy())

        # Get action from policy
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            if deterministic:
                action = actor.predict(obs_t, deterministic=True)
            else:
                action = actor.predict(obs_t, deterministic=False)
            if isinstance(action, torch.Tensor):
                action_np = action.detach().cpu().numpy().flatten()
            else:
                action_np = np.asarray(action).flatten()

        act_list.append(action_np.copy())

        # Step environment — OmniSafe CMDP signature
        obs, reward, cost, terminated, truncated, info = env.step(
            torch.as_tensor(action_np, dtype=torch.float32)
        )

        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy().flatten()
        else:
            obs_np = np.asarray(obs).flatten()

        rew_list.append(float(reward))
        cost_list.append(float(cost))

        # Extract reward components from info
        rew_economic.append(float(info.get("reward_economic", 0.0)))
        rew_stability_grid.append(float(info.get("reward_stability_grid", 0.0)))
        rew_stability_building.append(
            float(info.get("reward_stability_building", 0.0))
        )
        rew_ramp.append(float(info.get("reward_ramp", 0.0)))
        rew_renewable.append(float(info.get("reward_renewable", 0.0)))

        # Extract cost components from info
        cost_c1.append(float(info.get("cost_ev_departure", 0.0)))
        cost_c2.append(float(info.get("cost_stems_battery", 0.0)))
        cost_c3.append(float(info.get("cost_stems_building_power", 0.0)))
        cost_c4.append(float(info.get("cost_stems_grid_power", 0.0)))

        done = bool(terminated) or bool(truncated)

    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.float32),
        "rewards": np.array(rew_list, dtype=np.float32),
        "costs": np.array(cost_list, dtype=np.float32),
        "reward_economic": np.array(rew_economic, dtype=np.float32),
        "reward_stability_grid": np.array(rew_stability_grid, dtype=np.float32),
        "reward_stability_building": np.array(
            rew_stability_building, dtype=np.float32
        ),
        "reward_ramp": np.array(rew_ramp, dtype=np.float32),
        "reward_renewable": np.array(rew_renewable, dtype=np.float32),
        "cost_C1": np.array(cost_c1, dtype=np.float32),
        "cost_C2": np.array(cost_c2, dtype=np.float32),
        "cost_C3": np.array(cost_c3, dtype=np.float32),
        "cost_C4": np.array(cost_c4, dtype=np.float32),
    }


def collect_zero_action_rollout(env: Any) -> Dict[str, np.ndarray]:
    """Run one full episode with zero actions (do-nothing baseline).

    Parameters
    ----------
    env : CMDP environment
        OmniSafe-wrapped CityLearn env.

    Returns
    -------
    dict of np.ndarray
        Same keys as collect_rollout.
    """
    import torch

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    rew_list: List[float] = []
    cost_list: List[float] = []

    rew_economic: List[float] = []
    rew_stability_grid: List[float] = []
    rew_stability_building: List[float] = []
    rew_ramp: List[float] = []
    rew_renewable: List[float] = []

    cost_c1: List[float] = []
    cost_c2: List[float] = []
    cost_c3: List[float] = []
    cost_c4: List[float] = []

    obs, info = env.reset()
    if isinstance(obs, torch.Tensor):
        obs_np = obs.detach().cpu().numpy().flatten()
    else:
        obs_np = np.asarray(obs).flatten()

    zero_action = np.zeros(ACT_DIM, dtype=np.float32)
    done = False

    while not done:
        obs_list.append(obs_np.copy())
        act_list.append(zero_action.copy())

        obs, reward, cost, terminated, truncated, info = env.step(
            torch.as_tensor(zero_action, dtype=torch.float32)
        )

        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy().flatten()
        else:
            obs_np = np.asarray(obs).flatten()

        rew_list.append(float(reward))
        cost_list.append(float(cost))

        rew_economic.append(float(info.get("reward_economic", 0.0)))
        rew_stability_grid.append(float(info.get("reward_stability_grid", 0.0)))
        rew_stability_building.append(
            float(info.get("reward_stability_building", 0.0))
        )
        rew_ramp.append(float(info.get("reward_ramp", 0.0)))
        rew_renewable.append(float(info.get("reward_renewable", 0.0)))

        cost_c1.append(float(info.get("cost_ev_departure", 0.0)))
        cost_c2.append(float(info.get("cost_stems_battery", 0.0)))
        cost_c3.append(float(info.get("cost_stems_building_power", 0.0)))
        cost_c4.append(float(info.get("cost_stems_grid_power", 0.0)))

        done = bool(terminated) or bool(truncated)

    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.float32),
        "rewards": np.array(rew_list, dtype=np.float32),
        "costs": np.array(cost_list, dtype=np.float32),
        "reward_economic": np.array(rew_economic, dtype=np.float32),
        "reward_stability_grid": np.array(rew_stability_grid, dtype=np.float32),
        "reward_stability_building": np.array(
            rew_stability_building, dtype=np.float32
        ),
        "reward_ramp": np.array(rew_ramp, dtype=np.float32),
        "reward_renewable": np.array(rew_renewable, dtype=np.float32),
        "cost_C1": np.array(cost_c1, dtype=np.float32),
        "cost_C2": np.array(cost_c2, dtype=np.float32),
        "cost_C3": np.array(cost_c3, dtype=np.float32),
        "cost_C4": np.array(cost_c4, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Diagnostic tests (Task 2+)
# ---------------------------------------------------------------------------
def test_value_function(data, v_reward=None, v_cost=None, checkpoint_path=None, gamma=GAMMA):
    """Test 1: Value Function Accuracy."""
    rewards = data['rewards']
    costs = data['costs']
    T = len(rewards)

    # Compute actual discounted returns (backward pass)
    returns_r = np.zeros(T, dtype=np.float64)
    returns_c = np.zeros(T, dtype=np.float64)
    returns_r[T - 1] = rewards[T - 1]
    returns_c[T - 1] = costs[T - 1]
    for t in range(T - 2, -1, -1):
        returns_r[t] = rewards[t] + gamma * returns_r[t + 1]
        returns_c[t] = costs[t] + gamma * returns_c[t + 1]

    # Explained variance
    var_r = np.var(returns_r)
    var_c = np.var(returns_c)
    ev_r = 1.0 - np.var(returns_r - v_reward) / (var_r + 1e-8) if v_reward is not None else float('nan')
    ev_c = 1.0 - np.var(returns_c - v_cost) / (var_c + 1e-8) if v_cost is not None else float('nan')

    # TD errors and autocorrelation
    def autocorr_lag1(x):
        if len(x) < 3 or np.std(x) < 1e-10:
            return 0.0
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    td_ac_r, td_ac_c = 0.0, 0.0
    if v_reward is not None:
        td_r = rewards[:-1] + gamma * v_reward[1:] - v_reward[:-1]
        td_ac_r = autocorr_lag1(td_r)
    if v_cost is not None:
        td_c = costs[:-1] + gamma * v_cost[1:] - v_cost[:-1]
        td_ac_c = autocorr_lag1(td_c)

    ev_min = min(ev_r if not np.isnan(ev_r) else 1.0, ev_c if not np.isnan(ev_c) else 1.0)
    status = 'broken' if ev_min < 0.1 else ('warning' if ev_min < 0.5 else 'healthy')

    return {
        'ev_reward': float(ev_r), 'ev_cost': float(ev_c),
        'td_autocorr_reward': float(td_ac_r), 'td_autocorr_cost': float(td_ac_c),
        'returns_reward_mean': float(np.mean(returns_r)),
        'returns_cost_mean': float(np.mean(returns_c)),
        'status': status,
    }


def test_feature_action_mi(data, n_neighbors=5):
    """Test 2: Feature-action mutual information (KSG estimator)."""
    from sklearn.feature_selection import mutual_info_regression

    obs = data['obs']       # (T, 330)
    actions = data['actions']  # (T, 9)
    T, n_features = obs.shape
    n_actions = actions.shape[1]

    # Compute full MI matrix
    mi_matrix = np.zeros((n_features, n_actions), dtype=np.float64)
    for j in range(n_actions):
        mi_matrix[:, j] = mutual_info_regression(
            obs, actions[:, j], n_neighbors=n_neighbors, random_state=42
        )

    # Noise floor via shuffled obs
    rng = np.random.RandomState(123)
    obs_shuffled = obs.copy()
    for col in range(n_features):
        rng.shuffle(obs_shuffled[:, col])
    noise_mis = []
    for j in range(n_actions):
        noise_mis.append(np.mean(mutual_info_regression(
            obs_shuffled, actions[:, j], n_neighbors=n_neighbors, random_state=42
        )))
    noise_floor = float(np.mean(noise_mis))

    # Top-5 mean MI
    flat_mi = mi_matrix.flatten()
    top5_indices = np.argsort(flat_mi)[-5:]
    mean_mi_top5 = float(np.mean(flat_mi[top5_indices]))

    # Status
    if mean_mi_top5 < 2 * noise_floor:
        status = 'broken'
    elif mean_mi_top5 < 0.1:
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'mi_matrix': mi_matrix,
        'mean_mi_top5': mean_mi_top5,
        'noise_floor': noise_floor,
        'status': status,
    }


def test_conditional_entropy(data, n_bins=20):
    """Test 3: Conditional entropy reduction of actions given key features."""
    obs = data['obs']       # (T, 330)
    actions = data['actions']  # (T, 9)
    T = obs.shape[0]

    # Key feature indices and names
    key_features = {
        'price': PRICE_IDX,
        'hour_cos': HOUR_COS_IDX,
        'hour_sin': HOUR_SIN_IDX,
    }
    for i, idx in enumerate(SOC_INDICES):
        key_features[f'soc_{i}'] = idx

    # Battery action columns (0..4)
    battery_cols = list(range(min(NUM_BUILDINGS, actions.shape[1])))

    entropy_reduction = {}

    for feat_name, feat_idx in key_features.items():
        feat_vals = obs[:, feat_idx]

        # Quantile binning — handle constant features
        try:
            bins = np.percentile(feat_vals, np.linspace(0, 100, n_bins + 1))
            # Remove duplicate bin edges
            bins = np.unique(bins)
            if len(bins) < 2:
                entropy_reduction[feat_name] = 0.0
                continue
            bin_idx = np.digitize(feat_vals, bins[1:-1])
        except Exception:
            entropy_reduction[feat_name] = 0.0
            continue

        # Compute conditional variance reduction (rho) averaged over battery actions
        rho_per_action = []
        for act_col in battery_cols:
            act_vals = actions[:, act_col]
            total_var = np.var(act_vals)
            if total_var < 1e-12:
                rho_per_action.append(0.0)
                continue

            # Weighted conditional variance
            cond_var = 0.0
            for b in np.unique(bin_idx):
                mask = bin_idx == b
                n_b = np.sum(mask)
                if n_b < 2:
                    cond_var += (n_b / T) * total_var
                else:
                    cond_var += (n_b / T) * np.var(act_vals[mask])

            rho = 1.0 - cond_var / (total_var + 1e-12)
            rho_per_action.append(float(rho))

        entropy_reduction[feat_name] = float(np.mean(rho_per_action))

    # Top-5 mean rho
    rho_vals = sorted(entropy_reduction.values(), reverse=True)
    mean_rho_top5 = float(np.mean(rho_vals[:5])) if len(rho_vals) >= 5 else float(np.mean(rho_vals))

    # Status
    if mean_rho_top5 < 0.01:
        status = 'broken'
    elif mean_rho_top5 < 0.05:
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'entropy_reduction': entropy_reduction,
        'mean_rho_top5': mean_rho_top5,
        'status': status,
    }


def test_action_correlation(data):
    """Test 4: Inter-building action correlation analysis."""
    actions = data['actions']  # (T, 9)
    T = actions.shape[0]

    # Battery actions: columns 0..4
    n_batt = min(NUM_BUILDINGS, actions.shape[1])
    batt_actions = actions[:, :n_batt]  # (T, 5)

    # 5x5 Pearson correlation matrix
    battery_corr_matrix = np.corrcoef(batt_actions.T)  # (5, 5)

    # Mean absolute off-diagonal correlation
    mask = ~np.eye(n_batt, dtype=bool)
    mean_abs_corr = float(np.mean(np.abs(battery_corr_matrix[mask])))

    # Spatial variance ratio: eta = mean(Var_across_buildings_per_timestep) / Var(all_battery_actions)
    var_across_buildings_per_t = np.var(batt_actions, axis=1)  # (T,) variance across 5 buildings each step
    total_var = np.var(batt_actions)
    spatial_variance_ratio = float(np.mean(var_across_buildings_per_t) / (total_var + 1e-12))

    # ACF for each battery action up to lag 48
    max_lag = min(48, T - 1)
    acf = np.zeros((n_batt, max_lag), dtype=np.float64)
    for j in range(n_batt):
        x = batt_actions[:, j]
        x_centered = x - np.mean(x)
        var_x = np.var(x)
        if var_x < 1e-12:
            continue
        for lag in range(max_lag):
            acf[j, lag] = float(np.mean(x_centered[:T - lag - 1] * x_centered[lag + 1:])) / (var_x + 1e-12)

    # Status
    if mean_abs_corr > 0.9 or spatial_variance_ratio < 0.05:
        status = 'broken'
    elif mean_abs_corr > 0.7 or spatial_variance_ratio < 0.1:
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'battery_corr_matrix': battery_corr_matrix,
        'mean_abs_corr': mean_abs_corr,
        'spatial_variance_ratio': spatial_variance_ratio,
        'acf': acf,
        'status': status,
    }


def test_gradient_attribution(
    rollout: Dict[str, np.ndarray], actor: Any
) -> Dict[str, Any]:
    """Test 5: Gradient-based feature attribution."""
    raise NotImplementedError(
        "test_gradient_attribution — to be implemented in Task 6"
    )


def test_temporal_planning(rollout: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Test 6: Temporal planning horizon analysis."""
    raise NotImplementedError(
        "test_temporal_planning — to be implemented in Task 7"
    )


def test_constraint_decomposition(rollout: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Test 7: Per-constraint cost decomposition and diagnosis."""
    raise NotImplementedError(
        "test_constraint_decomposition — to be implemented in Task 8"
    )


def test_headroom(
    rollout: Dict[str, np.ndarray],
    zero_rollout: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Test 8: Safety headroom relative to do-nothing baseline."""
    raise NotImplementedError("test_headroom — to be implemented in Task 9")


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------
def compute_phi(test_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the overall health score phi from individual test results."""
    raise NotImplementedError("compute_phi — to be implemented in Task 10")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def diagnose(
    checkpoint_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    skip_env: bool = False,
    rollout_data_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full diagnostic suite and return structured results."""
    raise NotImplementedError("diagnose — to be implemented in Task 10")


def generate_report(
    results: Dict[str, Any], output_dir: str
) -> str:
    """Write JSON report and summary to output_dir. Returns report path."""
    raise NotImplementedError("generate_report — to be implemented in Task 10")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Resolve output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.checkpoint, "diagnostics")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Policy Health Diagnostic Suite")
    print("=" * 60)
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Config:       {args.config or '(none)'}")
    print(f"  Skip env:     {args.skip_env}")
    print(f"  Rollout data: {args.rollout_data or '(none)'}")
    print()
    print("  Scaffold loaded successfully.")
    print("  Individual tests are not yet implemented (NotImplementedError).")
    print()
    print(f"  Constants:")
    print(f"    NUM_BUILDINGS          = {NUM_BUILDINGS}")
    print(f"    OBS_DIM                = {OBS_DIM}")
    print(f"    ACT_DIM                = {ACT_DIM}")
    print(f"    CURRENT_OBS_DIM        = {CURRENT_OBS_DIM}")
    print(f"    TEMPORAL_WINDOW        = {TEMPORAL_WINDOW}")
    print(f"    TEMPORAL_FEATURES/STEP = {TEMPORAL_FEATURES_PER_STEP}")
    print(f"    GAMMA                  = {GAMMA}")
    print(f"    PRICE_IDX              = {PRICE_IDX}")
    print(f"    SOC_INDICES            = {SOC_INDICES}")
    print(f"    HISTORY_START          = {HISTORY_START}")
    print(f"    HISTORY_END            = {HISTORY_END}")
    print("=" * 60)

    # Write a marker file so tests can verify the output dir was created
    marker = os.path.join(output_dir, "scaffold_ok.json")
    with open(marker, "w") as f:
        json.dump(
            {
                "status": "scaffold",
                "checkpoint": args.checkpoint,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            f,
            indent=2,
        )
    print(f"  Wrote marker: {marker}")


if __name__ == "__main__":
    main()
