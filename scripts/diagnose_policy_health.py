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
    data: Dict[str, np.ndarray], actor: Any, n_samples: int = 200
) -> Dict[str, Any]:
    """Test 5: Gradient-based feature attribution.

    Computes mean |d action_j / d obs_i| across samples and action dims,
    then groups by pathway (temporal, current, global, price, building).

    Parameters
    ----------
    data : dict
        Rollout data with 'obs' key of shape (T, OBS_DIM).
    actor : torch.nn.Module
        Policy network that maps obs tensor -> action tensor (T, ACT_DIM).
    n_samples : int
        Number of observations to use for gradient estimation.

    Returns
    -------
    dict with temporal_fraction, current_fraction, global_fraction,
         price_gradient, building_gradient, total_gradient, status.
    """
    import torch

    obs = data['obs']
    n_samples = min(n_samples, len(obs))
    obs_np = obs[:n_samples]

    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    grad_accum = torch.zeros(obs_t.shape[1])

    for j in range(ACT_DIM):
        obs_t_j = obs_t.detach().clone().requires_grad_(True)
        out_j = actor(obs_t_j)[:, j].sum()
        out_j.backward()
        grad_accum += obs_t_j.grad.abs().mean(dim=0)

    grad_accum /= ACT_DIM
    grad_np = grad_accum.detach().numpy()

    total_gradient = float(grad_np.sum())

    # Pathway groupings
    temporal_grad = float(grad_np[HISTORY_START:HISTORY_END].sum())
    current_grad = float(grad_np[:HISTORY_START].sum())
    global_indices = [PRICE_IDX, HOUR_COS_IDX, HOUR_SIN_IDX]
    global_grad = float(grad_np[global_indices].sum())
    price_gradient = float(grad_np[PRICE_IDX])
    building_gradient = float(grad_np[SOC_INDICES].sum())

    # Fractions
    temporal_fraction = temporal_grad / (total_gradient + 1e-12)
    current_fraction = current_grad / (total_gradient + 1e-12)
    global_fraction = global_grad / (total_gradient + 1e-12)

    # Status
    if temporal_fraction < 0.01 or price_gradient < 0.001:
        status = 'broken'
    elif temporal_fraction < 0.10 or price_gradient < 0.01:
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'temporal_fraction': float(temporal_fraction),
        'current_fraction': float(current_fraction),
        'global_fraction': float(global_fraction),
        'price_gradient': float(price_gradient),
        'building_gradient': float(building_gradient),
        'total_gradient': float(total_gradient),
        'status': status,
    }


def test_temporal_planning(
    data: Dict[str, np.ndarray], actor: Any = None, max_lag: int = 24
) -> Dict[str, Any]:
    """Test 6: Temporal planning horizon analysis.

    Sub-tests:
      6a. Cross-temporal correlation between battery actions and shifted price.
      6b. TPS (Temporal Planning Score) — exponentially weighted future corr.
      6c. Perturbation tests (only if actor is provided).

    Parameters
    ----------
    data : dict
        Rollout data with 'obs' (T, OBS_DIM) and 'actions' (T, ACT_DIM).
    actor : torch.nn.Module or None
        If provided, run perturbation tests (zero/shuffle/reverse history).
    max_lag : int
        Maximum forward lag for cross-temporal correlation.

    Returns
    -------
    dict with tps, cross_temporal_corr, perturbation_effects, lag0_corr, status.
    """
    obs = data['obs']
    actions = data['actions']
    T = len(obs)
    price = obs[:, PRICE_IDX]
    n_batt = min(NUM_BUILDINGS, actions.shape[1])

    # 6a. Cross-temporal correlation: tau from -6 to +max_lag
    cross_temporal_corr = {}
    for tau in range(-6, max_lag + 1):
        corrs = []
        for j in range(n_batt):
            shifted_price = np.roll(price, -tau)
            # Trim edges to avoid wrap-around artifacts
            margin = abs(tau) + 1 if tau != 0 else 0
            if margin > 0 and margin < T:
                c = np.corrcoef(actions[margin:T - margin, j],
                                shifted_price[margin:T - margin])[0, 1]
            else:
                c = np.corrcoef(actions[:, j], shifted_price)[0, 1]
            if np.isnan(c):
                c = 0.0
            corrs.append(c)
        cross_temporal_corr[tau] = float(np.mean(corrs))

    lag0_corr = cross_temporal_corr.get(0, 0.0)

    # 6b. TPS: sum(|corr[tau]| * exp(-tau/6) for tau=1..max_lag) / sum(exp(-tau/6))
    weights_sum = 0.0
    weighted_corr_sum = 0.0
    for tau in range(1, max_lag + 1):
        w = np.exp(-tau / 6.0)
        weights_sum += w
        corr_val = cross_temporal_corr.get(tau, 0.0)
        weighted_corr_sum += abs(corr_val) * w
    tps = weighted_corr_sum / (weights_sum + 1e-12)

    # 6c. Perturbation tests (only if actor is provided)
    perturbation_effects = {}
    if actor is not None:
        import torch

        n_test = min(500, T)
        obs_t = torch.as_tensor(obs[:n_test], dtype=torch.float32)

        with torch.no_grad():
            base_actions = actor(obs_t).detach().numpy()

            # Zero history
            obs_zero = obs_t.clone()
            obs_zero[:, HISTORY_START:HISTORY_END] = 0.0
            zero_actions = actor(obs_zero).detach().numpy()
            perturbation_effects['zero_history'] = float(
                np.mean(np.abs(base_actions - zero_actions))
            )

            # Shuffle history: permute TEMPORAL_WINDOW timesteps randomly
            obs_shuffle = obs_t.clone().numpy()
            rng = np.random.RandomState(99)
            for i in range(n_test):
                # Reshape history into (TEMPORAL_WINDOW, TEMPORAL_FEATURES_PER_STEP)
                hist = obs_shuffle[i, HISTORY_START:HISTORY_END].reshape(
                    TEMPORAL_WINDOW, TEMPORAL_FEATURES_PER_STEP
                )
                perm = rng.permutation(TEMPORAL_WINDOW)
                obs_shuffle[i, HISTORY_START:HISTORY_END] = hist[perm].flatten()
            shuffle_actions = actor(
                torch.as_tensor(obs_shuffle, dtype=torch.float32)
            ).detach().numpy()
            perturbation_effects['shuffle_history'] = float(
                np.mean(np.abs(base_actions - shuffle_actions))
            )

            # Reverse history: flip timestep order
            obs_reverse = obs_t.clone().numpy()
            for i in range(n_test):
                hist = obs_reverse[i, HISTORY_START:HISTORY_END].reshape(
                    TEMPORAL_WINDOW, TEMPORAL_FEATURES_PER_STEP
                )
                obs_reverse[i, HISTORY_START:HISTORY_END] = hist[::-1].flatten()
            reverse_actions = actor(
                torch.as_tensor(obs_reverse, dtype=torch.float32)
            ).detach().numpy()
            perturbation_effects['reverse_history'] = float(
                np.mean(np.abs(base_actions - reverse_actions))
            )

    # Status
    zero_hist_effect = perturbation_effects.get('zero_history', None)
    if tps < 0.02 and (zero_hist_effect is not None and zero_hist_effect < 0.01):
        status = 'broken'
    elif tps < 0.02 and zero_hist_effect is None:
        # No actor provided but TPS is very low
        status = 'broken'
    elif tps < 0.05:
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'tps': float(tps),
        'cross_temporal_corr': cross_temporal_corr,
        'perturbation_effects': perturbation_effects,
        'lag0_corr': float(lag0_corr),
        'status': status,
    }


def test_constraint_decomposition(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Test 7: Per-constraint cost decomposition and diagnosis.

    For each constraint (C1..C4), computes:
      - violation_rate: fraction of steps with cost > 0
      - violation_magnitude: mean cost when cost > 0
      - total_cost: sum of costs
      - hourly_violation_rate: 24-element array (violation rate per hour)
      - concentration_top10pct: fraction of total cost from top 10% violating steps

    C1 (EV departure) and C4 (grid power) are behavioral constraints.
    C2 (battery) and C3 (building power) have structural floors.

    Parameters
    ----------
    data : dict
        Rollout data with cost_C1, cost_C2, cost_C3, cost_C4 keys.

    Returns
    -------
    dict with per_constraint, total_behavioral_vr, status.
    """
    T = len(data['costs'])
    constraint_names = ['C1', 'C2', 'C3', 'C4']
    per_constraint = {}

    for cname in constraint_names:
        costs = data[f'cost_{cname}']

        # Violation rate
        violating_mask = costs > 0
        violation_rate = float(np.mean(violating_mask))

        # Violation magnitude (mean cost when violating)
        if np.any(violating_mask):
            violation_magnitude = float(np.mean(costs[violating_mask]))
        else:
            violation_magnitude = 0.0

        # Total cost
        total_cost = float(np.sum(costs))

        # Hourly violation rate (24 bins, using step_idx % 24)
        hourly_vr = np.zeros(24, dtype=np.float64)
        for h in range(24):
            hour_mask = (np.arange(T) % 24) == h
            if np.any(hour_mask):
                hourly_vr[h] = float(np.mean(violating_mask[hour_mask]))

        # Concentration: fraction of total cost from top 10% of violating steps
        if np.any(violating_mask) and total_cost > 0:
            violating_costs = costs[violating_mask]
            n_violating = len(violating_costs)
            n_top10 = max(1, int(np.ceil(n_violating * 0.1)))
            sorted_costs = np.sort(violating_costs)[::-1]
            concentration_top10pct = float(np.sum(sorted_costs[:n_top10]) / total_cost)
        else:
            concentration_top10pct = 0.0

        per_constraint[cname] = {
            'violation_rate': violation_rate,
            'violation_magnitude': violation_magnitude,
            'total_cost': total_cost,
            'hourly_violation_rate': hourly_vr,
            'concentration_top10pct': concentration_top10pct,
        }

    # Behavioral violation rate: mean of C1 and C4
    total_behavioral_vr = float(
        np.mean([per_constraint['C1']['violation_rate'],
                 per_constraint['C4']['violation_rate']])
    )

    # Status
    if total_behavioral_vr > 0.5:
        status = 'broken'
    elif total_behavioral_vr > 0.3:
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'per_constraint': per_constraint,
        'total_behavioral_vr': total_behavioral_vr,
        'status': status,
    }


def test_headroom(
    data: Dict[str, np.ndarray],
    baseline_data: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Test 8: Safety headroom relative to do-nothing baseline.

    Compares per-component reward and cost between agent and baseline.
    For rewards, higher is better. For costs, lower is better.

    Parameters
    ----------
    data : dict
        Agent rollout data.
    baseline_data : dict
        Baseline (e.g. zero-action) rollout data. Must have same keys.

    Returns
    -------
    dict with reward_headroom, cost_headroom, total_reward_vs_baseline,
         total_cost_vs_baseline, agent/baseline totals, status.
    """
    if baseline_data is None:
        return {
            'reward_headroom': {},
            'cost_headroom': {},
            'total_reward_vs_baseline': 0.0,
            'total_cost_vs_baseline': 0.0,
            'agent_total_reward': float(np.sum(data['rewards'])),
            'baseline_total_reward': 0.0,
            'agent_total_cost': float(np.sum(data['costs'])),
            'baseline_total_cost': 0.0,
            'status': 'warning',
        }

    # Reward components
    reward_components = ['economic', 'stability_grid', 'stability_building', 'ramp', 'renewable']
    reward_headroom = {}
    for comp in reward_components:
        key = f'reward_{comp}'
        agent_val = float(np.sum(data[key]))
        baseline_val = float(np.sum(baseline_data[key]))
        delta = agent_val - baseline_val
        reward_headroom[comp] = {
            'agent_val': agent_val,
            'baseline_val': baseline_val,
            'delta': delta,
            'better_than_baseline': agent_val > baseline_val,
        }

    # Cost components
    cost_components = ['C1', 'C2', 'C3', 'C4']
    cost_headroom = {}
    for comp in cost_components:
        key = f'cost_{comp}'
        agent_val = float(np.sum(data[key]))
        baseline_val = float(np.sum(baseline_data[key]))
        delta = agent_val - baseline_val
        cost_headroom[comp] = {
            'agent_val': agent_val,
            'baseline_val': baseline_val,
            'delta': delta,
            'better_than_baseline': agent_val < baseline_val,  # lower cost is better
        }

    # Totals
    agent_total_reward = float(np.sum(data['rewards']))
    baseline_total_reward = float(np.sum(baseline_data['rewards']))
    total_reward_vs_baseline = agent_total_reward - baseline_total_reward

    agent_total_cost = float(np.sum(data['costs']))
    baseline_total_cost = float(np.sum(baseline_data['costs']))
    total_cost_vs_baseline = agent_total_cost - baseline_total_cost

    # Status: based on reward improvement
    if total_reward_vs_baseline < 0:
        status = 'broken'
    elif baseline_total_reward != 0:
        improvement_frac = total_reward_vs_baseline / (abs(baseline_total_reward) + 1e-12)
        if improvement_frac < 0.05:
            status = 'warning'
        else:
            status = 'healthy'
    else:
        # Baseline total reward is zero
        if total_reward_vs_baseline > 0:
            status = 'healthy'
        else:
            status = 'warning'

    return {
        'reward_headroom': reward_headroom,
        'cost_headroom': cost_headroom,
        'total_reward_vs_baseline': total_reward_vs_baseline,
        'total_cost_vs_baseline': total_cost_vs_baseline,
        'agent_total_reward': agent_total_reward,
        'baseline_total_reward': baseline_total_reward,
        'agent_total_cost': agent_total_cost,
        'baseline_total_cost': baseline_total_cost,
        'status': status,
    }


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------
def compute_phi(results: Dict[str, Dict[str, Any]]) -> float:
    """Compute the overall Policy Health Index (PHI) from test results.

    Takes a results dict with keys 'test1' through 'test8'. Returns a float
    in [0, 1] representing overall policy health.

    Parameters
    ----------
    results : dict
        Keys 'test1'..'test8', each a dict of test-specific metrics.

    Returns
    -------
    float
        PHI score in [0.0, 1.0].
    """
    def clip01(x):
        return max(0.0, min(1.0, x))

    t1 = results.get('test1', {})
    t2 = results.get('test2', {})
    t3 = results.get('test3', {})
    t4 = results.get('test4', {})
    t5 = results.get('test5', {})
    t6 = results.get('test6', {})
    t7 = results.get('test7', {})

    ev_r = t1.get('ev_reward', 0)
    ev_c = t1.get('ev_cost', 0)
    s_value = clip01(np.mean([
        ev_r if not np.isnan(ev_r) else 0,
        ev_c if not np.isnan(ev_c) else 0,
    ]))

    s_mi = clip01(t2.get('mean_mi_top5', 0) / 0.3)

    s_entropy = clip01(t3.get('mean_rho_top5', 0) / 0.2)

    s_corr = (
        clip01(1.0 - t4.get('mean_abs_corr', 1.0) / 0.9)
        * clip01(t4.get('spatial_variance_ratio', 0) / 0.1)
    )

    s_gradient = (
        clip01(t5.get('temporal_fraction', 0) / 0.2)
        * clip01(t5.get('price_gradient', 0) / 0.01)
    )

    tps_s = clip01(t6.get('tps', 0) / 0.1)
    perturb = t6.get('perturbation_effects', {}).get('zero_history', 0)
    if isinstance(perturb, float) and np.isnan(perturb):
        perturb = 0.0
    s_temporal = (tps_s * clip01(perturb / 0.05)) ** 0.5

    s_constraints = clip01(1.0 - t7.get('total_behavioral_vr', 1.0) / 0.5)

    return float(
        0.20 * s_value
        + 0.20 * s_mi
        + 0.10 * s_entropy
        + 0.10 * s_corr
        + 0.10 * s_gradient
        + 0.20 * s_temporal
        + 0.10 * s_constraints
    )


# ---------------------------------------------------------------------------
# Decision-tree diagnosis
# ---------------------------------------------------------------------------
def diagnose(results: Dict[str, Dict[str, Any]]) -> str:
    """Produce an actionable diagnosis string from test results.

    Walks a decision tree over the test metrics, returning the first
    matching failure mode or an 'ARCHITECTURE WORKS' message.

    Parameters
    ----------
    results : dict
        Keys 'test1'..'test8', each a dict of test-specific metrics.

    Returns
    -------
    str
        Human-readable diagnosis with fix suggestions.
    """
    t1 = results.get('test1', {})
    t2 = results.get('test2', {})
    t4 = results.get('test4', {})
    t5 = results.get('test5', {})
    t6 = results.get('test6', {})
    t7 = results.get('test7', {})
    t8 = results.get('test8', {})

    ev_r = t1.get('ev_reward', 0)
    ev_c = t1.get('ev_cost', 0)
    if (ev_r if not np.isnan(ev_r) else 1) < 0.1 or (ev_c if not np.isnan(ev_c) else 1) < 0.1:
        return (
            f"CRITIC BROKEN: Value function cannot predict returns "
            f"(EV_r={ev_r:.3f}, EV_c={ev_c:.3f}). "
            f"Fix: increase critic LR, add critic update epochs."
        )

    mi = t2.get('mean_mi_top5', 0)
    if mi < 0.02:
        pg = t5.get('price_gradient', 0)
        if (pg if not np.isnan(pg) else 0) < 0.001:
            return (
                f"ENCODER DEAD: Price->action gradient near zero "
                f"(price_grad={pg:.5f}). "
                f"Fix: check global encoder -> broadcast -> per-node integration."
            )
        return (
            f"POLICY TOO NOISY: Gradients exist but MI low "
            f"(MI={mi:.4f}). "
            f"Fix: reduce PolicyStd, more training epochs."
        )

    if t4.get('mean_abs_corr', 0) > 0.9:
        return (
            f"GCN DEAD: All buildings identical "
            f"(mean_corr={t4['mean_abs_corr']:.3f}). "
            f"Fix: check adjacency, gated fusion."
        )

    if t8.get('total_reward_vs_baseline', 0) < 0:
        return (
            f"WORSE THAN ZERO-ACTION: Lambda erasing reward "
            f"(delta={t8['total_reward_vs_baseline']:.1f}). "
            f"Fix: enable standardized_cost_adv, raise cost_limit."
        )

    tf = t5.get('temporal_fraction', 0)
    tps = t6.get('tps', 0)
    if (tf if not np.isnan(tf) else 0) < 0.01 and tps < 0.02:
        return (
            f"TEMPORAL DEAD: History no effect "
            f"(temporal_frac={tf:.4f}, TPS={tps:.4f}). "
            f"Fix: check concat projection weights."
        )

    if t7.get('total_behavioral_vr', 0) > 0.5:
        return (
            f"CONSTRAINT FAILURE: Avoidable violations "
            f"(behavioral_VR={t7['total_behavioral_vr']:.3f}). "
            f"Fix: dense cost shaping."
        )

    return "ARCHITECTURE WORKS: Policy shows learning signals. Needs more training epochs with lambda cap."


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    results: Dict[str, Any],
    phi: float,
    diagnosis: str,
    output_dir: str,
) -> str:
    """Generate diagnostic report with figures, markdown, and JSON.

    Creates:
      - figures/mi_heatmap.png
      - figures/cross_temporal_corr.png
      - figures/building_correlation.png
      - figures/action_autocorrelation.png
      - figures/violation_timing.png
      - report.md
      - phi_score.json

    Parameters
    ----------
    results : dict
        Keys 'test1'..'test8' with per-test metric dicts.
    phi : float
        Computed PHI score.
    diagnosis : str
        Diagnosis string from diagnose().
    output_dir : str
        Directory to write all outputs.

    Returns
    -------
    str
        Path to the generated report.md file.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # --- Figure 1: MI heatmap ---
    t2 = results.get('test2', {})
    mi_matrix = t2.get('mi_matrix', None)
    if mi_matrix is not None:
        mi_matrix = np.asarray(mi_matrix)
        # Top 30 features by max MI across actions
        max_mi_per_feature = mi_matrix.max(axis=1)
        top30_idx = np.argsort(max_mi_per_feature)[-30:][::-1]
        mi_top30 = mi_matrix[top30_idx, :]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mi_top30, aspect='auto', cmap='viridis')
        ax.set_xlabel('Action dimension')
        ax.set_ylabel('Feature index (top 30 by max MI)')
        ax.set_yticks(range(len(top30_idx)))
        ax.set_yticklabels([str(i) for i in top30_idx], fontsize=7)
        ax.set_title('Feature-Action Mutual Information (top 30)')
        plt.colorbar(im, ax=ax, label='MI (nats)')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'mi_heatmap.png'), dpi=150)
        plt.close(fig)

    # --- Figure 2: Cross-temporal correlation bar chart ---
    t6 = results.get('test6', {})
    ctc = t6.get('cross_temporal_corr', None)
    if ctc is not None:
        lags = sorted(ctc.keys(), key=lambda x: int(x))
        corr_vals = [ctc[lag] for lag in lags]
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#d62728' if int(lag) < 0 else '#1f77b4' for lag in lags]
        ax.bar([int(lag) for lag in lags], corr_vals, color=colors, width=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Lag (hours)')
        ax.set_ylabel('Mean correlation with price')
        ax.set_title(f'Cross-temporal correlation (TPS={t6.get("tps", 0):.4f})')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'cross_temporal_corr.png'), dpi=150)
        plt.close(fig)

    # --- Figure 3: Building correlation 5x5 heatmap ---
    t4 = results.get('test4', {})
    corr_matrix = t4.get('battery_corr_matrix', None)
    if corr_matrix is not None:
        corr_matrix = np.asarray(corr_matrix)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        n = corr_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', fontsize=9)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f'B{i}' for i in range(n)])
        ax.set_yticklabels([f'B{i}' for i in range(n)])
        ax.set_title(f'Battery Action Correlation (mean_abs={t4.get("mean_abs_corr", 0):.3f})')
        plt.colorbar(im, ax=ax, label='Pearson r')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'building_correlation.png'), dpi=150)
        plt.close(fig)

    # --- Figure 4: Action autocorrelation curves ---
    acf_data = t4.get('acf', None)
    if acf_data is not None:
        if isinstance(acf_data, dict):
            # dict keyed by building index
            fig, ax = plt.subplots(figsize=(10, 4))
            for j in sorted(acf_data.keys(), key=lambda x: int(x)):
                acf_vals = np.asarray(acf_data[j])
                ax.plot(range(len(acf_vals)), acf_vals, label=f'B{j}', alpha=0.7)
            ax.set_xlabel('Lag (hours)')
            ax.set_ylabel('Autocorrelation')
            ax.set_title('Battery Action Autocorrelation')
            ax.legend(fontsize=8)
            ax.axhline(0, color='black', linewidth=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'action_autocorrelation.png'), dpi=150)
            plt.close(fig)
        else:
            # numpy array (n_batt, max_lag)
            acf_arr = np.asarray(acf_data)
            fig, ax = plt.subplots(figsize=(10, 4))
            for j in range(acf_arr.shape[0]):
                ax.plot(range(acf_arr.shape[1]), acf_arr[j], label=f'B{j}', alpha=0.7)
            ax.set_xlabel('Lag (hours)')
            ax.set_ylabel('Autocorrelation')
            ax.set_title('Battery Action Autocorrelation')
            ax.legend(fontsize=8)
            ax.axhline(0, color='black', linewidth=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'action_autocorrelation.png'), dpi=150)
            plt.close(fig)

    # --- Figure 5: Violation timing 24-hour heatmap ---
    t7 = results.get('test7', {})
    per_constraint = t7.get('per_constraint', None)
    if per_constraint is not None and len(per_constraint) > 0:
        constraint_names = sorted(per_constraint.keys())
        n_constraints = len(constraint_names)
        hourly_matrix = np.zeros((n_constraints, 24))
        for i, cname in enumerate(constraint_names):
            hvr = per_constraint[cname].get('hourly_violation_rate', np.zeros(24))
            hvr = np.asarray(hvr)
            hourly_matrix[i, :len(hvr)] = hvr[:24]

        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(hourly_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_yticks(range(n_constraints))
        ax.set_yticklabels(constraint_names)
        ax.set_xlabel('Hour of day')
        ax.set_title('Constraint Violation Rate by Hour')
        plt.colorbar(im, ax=ax, label='Violation rate')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'violation_timing.png'), dpi=150)
        plt.close(fig)

    # --- report.md ---
    status_icon = {
        'healthy': '[OK]',
        'warning': '[WARN]',
        'broken': '[FAIL]',
    }
    lines = []
    lines.append('# Policy Health Diagnostic Report')
    lines.append('')
    lines.append(f'**PHI Score: {phi:.3f}**')
    lines.append('')
    lines.append(f'**Diagnosis:** {diagnosis}')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## Per-Test Summary')
    lines.append('')

    test_names = {
        'test1': 'Value Function Accuracy',
        'test2': 'Feature-Action Mutual Information',
        'test3': 'Conditional Entropy',
        'test4': 'Inter-Building Action Correlation',
        'test5': 'Gradient Attribution',
        'test6': 'Temporal Planning Horizon',
        'test7': 'Constraint Decomposition',
        'test8': 'Safety Headroom',
    }

    for tkey in [f'test{i}' for i in range(1, 9)]:
        tdata = results.get(tkey, {})
        tname = test_names.get(tkey, tkey)
        st = tdata.get('status', 'unknown')
        icon = status_icon.get(st, '[??]')
        lines.append(f'### {icon} {tname}')
        lines.append('')
        # Print key scalar metrics
        for k, v in sorted(tdata.items()):
            if k in ('mi_matrix', 'battery_corr_matrix', 'acf',
                      'cross_temporal_corr', 'perturbation_effects',
                      'per_constraint', 'reward_headroom', 'cost_headroom',
                      'entropy_reduction', 'hourly_violation_rate'):
                continue
            if isinstance(v, float):
                lines.append(f'- {k}: {v:.4f}')
            elif isinstance(v, (int, str, bool)):
                lines.append(f'- {k}: {v}')
        lines.append('')

    report_md = '\n'.join(lines)
    report_path = os.path.join(output_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write(report_md)

    # --- phi_score.json ---
    phi_json = {'phi': phi, 'diagnosis': diagnosis}
    with open(os.path.join(output_dir, 'phi_score.json'), 'w') as f:
        json.dump(phi_json, f, indent=2)

    return report_path


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
def run_all_tests(
    data: Dict[str, np.ndarray],
    baseline_data: Optional[Dict[str, np.ndarray]] = None,
    actor: Any = None,
    v_reward: Optional[np.ndarray] = None,
    v_cost: Optional[np.ndarray] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run all 8 diagnostic tests and return structured results.

    Parameters
    ----------
    data : dict
        Agent rollout data (from collect_rollout or synthetic).
    baseline_data : dict or None
        Zero-action baseline rollout.
    actor : torch.nn.Module or None
        Policy network for gradient/perturbation tests.
    v_reward : np.ndarray or None
        Value predictions for reward critic.
    v_cost : np.ndarray or None
        Value predictions for cost critic.
    checkpoint_path : str or None
        Path to checkpoint (passed to test_value_function).

    Returns
    -------
    dict
        Keys 'test1'..'test8', each a dict of test-specific metrics.
    """
    results = {}

    # Test 1: Value Function Accuracy
    print("[1/8] Value Function Accuracy...", end=" ", flush=True)
    try:
        results['test1'] = test_value_function(
            data, v_reward=v_reward, v_cost=v_cost,
            checkpoint_path=checkpoint_path, gamma=GAMMA,
        )
        print(f"status={results['test1'].get('status', '?')}")
    except Exception as e:
        print(f"SKIPPED ({e})")
        results['test1'] = {'status': 'skipped', 'error': str(e)}

    # Test 2: Feature-Action MI
    print("[2/8] Feature-Action Mutual Information...", end=" ", flush=True)
    try:
        results['test2'] = test_feature_action_mi(data)
        print(f"status={results['test2'].get('status', '?')}")
    except Exception as e:
        print(f"SKIPPED ({e})")
        results['test2'] = {'status': 'skipped', 'error': str(e)}

    # Test 3: Conditional Entropy
    print("[3/8] Conditional Entropy...", end=" ", flush=True)
    try:
        results['test3'] = test_conditional_entropy(data)
        print(f"status={results['test3'].get('status', '?')}")
    except Exception as e:
        print(f"SKIPPED ({e})")
        results['test3'] = {'status': 'skipped', 'error': str(e)}

    # Test 4: Action Correlation
    print("[4/8] Inter-Building Action Correlation...", end=" ", flush=True)
    try:
        results['test4'] = test_action_correlation(data)
        print(f"status={results['test4'].get('status', '?')}")
    except Exception as e:
        print(f"SKIPPED ({e})")
        results['test4'] = {'status': 'skipped', 'error': str(e)}

    # Test 5: Gradient Attribution
    print("[5/8] Gradient Attribution...", end=" ", flush=True)
    if actor is not None:
        try:
            results['test5'] = test_gradient_attribution(data, actor)
            print(f"status={results['test5'].get('status', '?')}")
        except Exception as e:
            print(f"SKIPPED ({e})")
            results['test5'] = {'status': 'skipped', 'error': str(e)}
    else:
        print("SKIPPED (no actor)")
        results['test5'] = {'status': 'skipped', 'error': 'no actor provided'}

    # Test 6: Temporal Planning
    print("[6/8] Temporal Planning Horizon...", end=" ", flush=True)
    try:
        results['test6'] = test_temporal_planning(data, actor=actor)
        print(f"status={results['test6'].get('status', '?')}")
    except Exception as e:
        print(f"SKIPPED ({e})")
        results['test6'] = {'status': 'skipped', 'error': str(e)}

    # Test 7: Constraint Decomposition
    print("[7/8] Constraint Decomposition...", end=" ", flush=True)
    try:
        results['test7'] = test_constraint_decomposition(data)
        print(f"status={results['test7'].get('status', '?')}")
    except Exception as e:
        print(f"SKIPPED ({e})")
        results['test7'] = {'status': 'skipped', 'error': str(e)}

    # Test 8: Safety Headroom
    print("[8/8] Safety Headroom...", end=" ", flush=True)
    try:
        results['test8'] = test_headroom(data, baseline_data=baseline_data)
        print(f"status={results['test8'].get('status', '?')}")
    except Exception as e:
        print(f"SKIPPED ({e})")
        results['test8'] = {'status': 'skipped', 'error': str(e)}

    return results


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_actor_from_checkpoint(
    checkpoint_path: str,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Load the policy actor from a training checkpoint.

    Detects STEMS vs MLP architecture by checking for 'mean.encoder.' keys
    in the state dict.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file (.pt or directory containing one).

    Returns
    -------
    tuple of (actor, obs_normalizer_dict_or_None)
        actor is a torch.nn.Module, obs_normalizer is a dict or None.
    """
    import torch

    # Find the checkpoint file
    ckpt_file = checkpoint_path
    if os.path.isdir(checkpoint_path):
        # Look for common checkpoint filenames
        candidates = ['model.pt', 'actor.pt', 'checkpoint.pt']
        for c in candidates:
            p = os.path.join(checkpoint_path, c)
            if os.path.exists(p):
                ckpt_file = p
                break
        else:
            # Try torch_save subdirectory
            ts_dir = os.path.join(checkpoint_path, 'torch_save')
            if os.path.isdir(ts_dir):
                for c in candidates:
                    p = os.path.join(ts_dir, c)
                    if os.path.exists(p):
                        ckpt_file = p
                        break

    ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)

    # Get the actor state dict
    if 'pi' in ckpt:
        actor_sd = ckpt['pi']
    elif 'actor' in ckpt:
        actor_sd = ckpt['actor']
    else:
        actor_sd = ckpt

    # Detect STEMS vs MLP by checking for encoder keys
    is_stems = any('encoder.' in k or 'mean.encoder.' in k for k in actor_sd.keys())

    # Strip 'mean.' prefix if present
    cleaned_sd = {}
    for k, v in actor_sd.items():
        if k.startswith('mean.'):
            cleaned_sd[k[5:]] = v
        else:
            cleaned_sd[k] = v

    if is_stems:
        # Reconstruct STEMS architecture from checkpoint
        from scripts.stems_v3 import STEMSEncoderV3, STEMSMeanNet

        # Extract node_info from checkpoint buffers
        node_info = {}
        buffer_keys = ['bld_idx', 'ev_idx', 'ev_mask', 'global_idx']
        for bk in buffer_keys:
            for k, v in cleaned_sd.items():
                if bk in k:
                    node_info[bk] = v
                    break

        # Infer base_obs_dim from global encoder weight shape
        base_obs_dim = CURRENT_OBS_DIM
        for k, v in cleaned_sd.items():
            if 'global_enc' in k and 'weight' in k and v.dim() == 2:
                base_obs_dim = v.shape[1]
                break

        encoder = STEMSEncoderV3(
            obs_dim=OBS_DIM,
            base_obs_dim=base_obs_dim,
            node_info=node_info,
        )
        actor = STEMSMeanNet(encoder=encoder, action_dim=ACT_DIM)
        actor.load_state_dict(cleaned_sd, strict=False)
    else:
        # MLP architecture: infer dimensions from weight shapes
        layer_dims = []
        weight_keys = sorted([k for k in cleaned_sd.keys() if 'weight' in k])
        for k in weight_keys:
            w = cleaned_sd[k]
            if w.dim() == 2:
                if not layer_dims:
                    layer_dims.append(w.shape[1])  # input dim
                layer_dims.append(w.shape[0])  # output dim

        if not layer_dims:
            raise ValueError("Cannot infer MLP dimensions from checkpoint")

        import torch.nn as nn
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.Tanh())
        # Final Tanh for action output
        layers.append(nn.Tanh())
        actor = nn.Sequential(*layers)
        actor.load_state_dict(cleaned_sd, strict=False)

    actor.eval()

    # Load obs normalizer if present
    obs_norm = ckpt.get('obs_normalizer', None)

    return actor, obs_norm


def load_critics_from_checkpoint(
    checkpoint_path: str,
) -> Tuple[Any, Any]:
    """Load reward and cost critics from a training checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.

    Returns
    -------
    tuple of (v_r_critic, v_c_critic)
        Both are torch.nn.Module instances, or None if not found.
    """
    import torch
    import torch.nn as nn

    # Find the checkpoint file
    ckpt_file = checkpoint_path
    if os.path.isdir(checkpoint_path):
        candidates = ['model.pt', 'actor.pt', 'checkpoint.pt']
        for c in candidates:
            p = os.path.join(checkpoint_path, c)
            if os.path.exists(p):
                ckpt_file = p
                break
        else:
            ts_dir = os.path.join(checkpoint_path, 'torch_save')
            if os.path.isdir(ts_dir):
                for c in candidates:
                    p = os.path.join(ts_dir, c)
                    if os.path.exists(p):
                        ckpt_file = p
                        break

    ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)

    def _build_critic(state_dict):
        """Build a critic MLP from a state dict by inferring layer shapes."""
        if state_dict is None:
            return None
        weight_keys = sorted([k for k in state_dict.keys() if 'weight' in k])
        layer_dims = []
        for k in weight_keys:
            w = state_dict[k]
            if w.dim() == 2:
                if not layer_dims:
                    layer_dims.append(w.shape[1])
                layer_dims.append(w.shape[0])

        if not layer_dims:
            return None

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.Tanh())
        critic = nn.Sequential(*layers)
        critic.load_state_dict(state_dict, strict=False)
        critic.eval()
        return critic

    v_r = _build_critic(ckpt.get('vr', None))
    v_c = _build_critic(ckpt.get('vc', None))

    return v_r, v_c


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Set environment variables for CityLearn
    os.environ.setdefault('CITYLEARN_CENTRAL_AGENT', '1')
    os.environ.setdefault('CITYLEARN_TEMPORAL_WINDOW', str(TEMPORAL_WINDOW))
    os.environ.setdefault('CITYLEARN_NUM_BUILDINGS', str(NUM_BUILDINGS))

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

    # ---------------------------------------------------------------
    # Step 1: Load actor + critics from checkpoint
    # ---------------------------------------------------------------
    print("Loading checkpoint...")
    try:
        actor, obs_norm = load_actor_from_checkpoint(args.checkpoint)
        print(f"  Actor loaded: {type(actor).__name__}")
    except Exception as e:
        print(f"  WARNING: Could not load actor: {e}")
        actor, obs_norm = None, None

    v_r_critic, v_c_critic = None, None
    try:
        v_r_critic, v_c_critic = load_critics_from_checkpoint(args.checkpoint)
        if v_r_critic is not None:
            print(f"  Reward critic loaded: {type(v_r_critic).__name__}")
        if v_c_critic is not None:
            print(f"  Cost critic loaded: {type(v_c_critic).__name__}")
    except Exception as e:
        print(f"  WARNING: Could not load critics: {e}")

    # ---------------------------------------------------------------
    # Step 2: Collect or load rollout data
    # ---------------------------------------------------------------
    import torch

    data = None
    baseline_data = None

    if args.rollout_data is not None:
        print(f"Loading pre-saved rollout from {args.rollout_data}...")
        npz = np.load(args.rollout_data, allow_pickle=True)
        data = {k: npz[k] for k in npz.files}
        print(f"  Loaded {len(data['rewards'])} timesteps")

        # Check for baseline in same directory
        base_dir = os.path.dirname(args.rollout_data)
        baseline_path = os.path.join(base_dir, 'baseline_data.npz')
        if os.path.exists(baseline_path):
            npz_b = np.load(baseline_path, allow_pickle=True)
            baseline_data = {k: npz_b[k] for k in npz_b.files}
            print(f"  Loaded baseline: {len(baseline_data['rewards'])} timesteps")

    elif not args.skip_env:
        print("Instantiating environment and collecting rollouts...")
        try:
            # Import environment setup
            from omnisafe.common.env import make as omnisafe_make

            env = omnisafe_make('CityLearnEnv-v0', config=args.config)

            print("  Collecting policy rollout...")
            data = collect_rollout(actor, env, deterministic=True)
            print(f"  Policy rollout: {len(data['rewards'])} timesteps")

            print("  Collecting zero-action baseline...")
            baseline_data = collect_zero_action_rollout(env)
            print(f"  Baseline rollout: {len(baseline_data['rewards'])} timesteps")

            # Save rollout data
            rollout_path = os.path.join(output_dir, 'rollout_data.npz')
            np.savez_compressed(rollout_path, **data)
            baseline_path = os.path.join(output_dir, 'baseline_data.npz')
            np.savez_compressed(baseline_path, **baseline_data)
            print(f"  Saved rollout data to {rollout_path}")
        except Exception as e:
            print(f"  ERROR collecting rollouts: {e}")
            print("  Falling back to skip-env mode.")

    if data is None:
        print("ERROR: No rollout data available. Use --rollout-data or remove --skip-env.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 3: Compute value predictions using critics
    # ---------------------------------------------------------------
    v_reward = None
    v_cost = None
    if v_r_critic is not None:
        print("Computing reward value predictions...")
        obs_t = torch.as_tensor(data['obs'], dtype=torch.float32)
        with torch.no_grad():
            v_reward = v_r_critic(obs_t).squeeze(-1).numpy()

    if v_c_critic is not None:
        print("Computing cost value predictions...")
        obs_t = torch.as_tensor(data['obs'], dtype=torch.float32)
        with torch.no_grad():
            v_cost = v_c_critic(obs_t).squeeze(-1).numpy()

    # ---------------------------------------------------------------
    # Step 4: Run all tests
    # ---------------------------------------------------------------
    print()
    print("Running diagnostic tests...")
    print("-" * 40)
    results = run_all_tests(
        data, baseline_data,
        actor=actor, v_reward=v_reward, v_cost=v_cost,
        checkpoint_path=args.checkpoint,
    )

    # ---------------------------------------------------------------
    # Step 5: Compute PHI + diagnosis
    # ---------------------------------------------------------------
    print()
    print("-" * 40)
    phi = compute_phi(results)
    diagnosis_str = diagnose(results)
    print(f"PHI Score: {phi:.3f}")
    print(f"Diagnosis: {diagnosis_str}")

    # ---------------------------------------------------------------
    # Step 6: Generate report
    # ---------------------------------------------------------------
    print()
    print("Generating report...")
    report_path = generate_report(results, phi, diagnosis_str, output_dir)
    print(f"  Report: {report_path}")

    # Save MI matrix if available
    mi_matrix = results.get('test2', {}).get('mi_matrix', None)
    if mi_matrix is not None:
        mi_path = os.path.join(output_dir, 'mi_matrix.npy')
        np.save(mi_path, mi_matrix)
        print(f"  MI matrix: {mi_path}")

    # ---------------------------------------------------------------
    # Step 7: Summary
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  PHI Score:  {phi:.3f}")
    print(f"  Diagnosis:  {diagnosis_str}")
    print()
    for tkey in [f'test{i}' for i in range(1, 9)]:
        tdata = results.get(tkey, {})
        st = tdata.get('status', 'unknown')
        print(f"  {tkey}: {st}")
    print()
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
