# Policy Diagnostic Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `scripts/diagnose_policy_health.py` — a single script that runs 8 mathematical diagnostic tests on a trained RL checkpoint and outputs an actionable report identifying exactly what the policy fails to learn.

**Architecture:** Single-file script with helper functions per test. Loads checkpoint + env, collects rollout data, runs all 8 tests, computes composite PHI score, outputs report + figures. Follows existing patterns from `diagnose_stems_v3.py` and `eval_intelligence_grads_v3.py`.

**Tech Stack:** Python 3.10, PyTorch, NumPy, scikit-learn (MI estimator), Matplotlib, OmniSafe (env)

**Design doc:** `docs/plans/2026-03-09-policy-diagnostic-suite-design.md`

---

## Task 1: Script Scaffold — CLI, Imports, Data Collection

**Files:**
- Create: `scripts/diagnose_policy_health.py`
- Create: `tests/test_diagnose_policy_health.py`

**Step 1: Write the test for data collection helpers**

```python
# tests/test_diagnose_policy_health.py
"""Tests for diagnose_policy_health.py using synthetic data."""
import sys, os
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def make_synthetic_rollout(T=500, obs_dim=330, act_dim=9):
    """Generate fake rollout data for testing diagnostic functions."""
    np.random.seed(42)
    obs = np.random.randn(T, obs_dim).astype(np.float32)
    # Make actions correlated with some obs features (price=index 22, SOC=indices 23-27)
    actions = np.zeros((T, act_dim), dtype=np.float32)
    for j in range(5):  # battery actions correlated with price
        actions[:, j] = -0.3 * obs[:, 22] + 0.2 * obs[:, 23 + j] + 0.5 * np.random.randn(T)
    actions[:, 5:8] = np.random.randn(T, 3).astype(np.float32)  # EV random
    actions[:, 8] = np.random.randn(T).astype(np.float32)  # washer random
    actions = np.clip(actions, -1, 1)
    rewards = np.random.randn(T).astype(np.float32) * 3 + 3
    costs = np.abs(np.random.randn(T).astype(np.float32)) * 5
    # Per-component costs
    cost_components = {
        'C1': np.abs(np.random.randn(T).astype(np.float32)),
        'C2': np.abs(np.random.randn(T).astype(np.float32)) * 3,
        'C3': np.abs(np.random.randn(T).astype(np.float32)) * 2,
        'C4': np.zeros(T, dtype=np.float32),
    }
    reward_components = {
        'economic': np.random.randn(T).astype(np.float32),
        'stability_grid': np.random.randn(T).astype(np.float32),
        'stability_building': np.random.randn(T).astype(np.float32),
        'ramp': -np.abs(np.random.randn(T).astype(np.float32)),
        'renewable': np.random.rand(T).astype(np.float32) * 0.2,
    }
    return dict(
        obs=obs, actions=actions, rewards=rewards, costs=costs,
        cost_components=cost_components, reward_components=reward_components,
    )


class TestScaffold:
    def test_synthetic_rollout_shapes(self):
        data = make_synthetic_rollout()
        assert data['obs'].shape == (500, 330)
        assert data['actions'].shape == (500, 9)
        assert data['rewards'].shape == (500,)
        assert data['costs'].shape == (500,)
```

**Step 2: Run test to verify it passes**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestScaffold -v`
Expected: PASS

**Step 3: Write the script scaffold**

```python
#!/usr/bin/env python3
"""
Policy Health Diagnostic Suite
===============================
Runs 8 mathematical diagnostic tests on a trained RL checkpoint to identify
exactly what the policy fails to learn.

Usage:
    cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
    conda run -n citylearn python scripts/diagnose_policy_health.py \
        --checkpoint runs/r10c_stems_v3/.../torch_save/epoch-1.pt \
        --output-dir diagnostics/r10c_ep1/

Tests:
    1. Value Function Accuracy (explained variance, TD errors)
    2. Feature-Action Mutual Information (KSG estimator)
    3. Conditional Entropy Decomposition
    4. Inter-Building Action Correlation
    5. Gradient Attribution by Pathway
    6. Temporal Planning Assessment
    7. Constraint Decomposition (structural vs behavioral)
    8. Headroom Analysis (vs zero-action baseline)
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

# ── Constants ──────────────────────────────────────────────────────────────
NUM_BUILDINGS = 5
TEMPORAL_WINDOW = 12
TEMPORAL_FEATURES_PER_STEP = 11
CURRENT_OBS_DIM = 198
OBS_DIM = CURRENT_OBS_DIM + TEMPORAL_WINDOW * TEMPORAL_FEATURES_PER_STEP  # 330
ACT_DIM = 9
GAMMA = 0.99

# Feature index groups (from ObsIndex for schema_5bld)
# These are set during env setup; defaults for standalone use
PRICE_IDX = 22
SOC_INDICES = list(range(23, 28))  # 5 buildings
NET_INDICES = list(range(40, 45))  # approximate; set precisely at runtime
HOUR_COS_IDX = 4
HOUR_SIN_IDX = 5
HISTORY_START = 198
HISTORY_END = 330


def parse_args():
    p = argparse.ArgumentParser(description="Policy Health Diagnostic Suite")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to epoch-N.pt checkpoint")
    p.add_argument("--output-dir", type=str, default="diagnostics/default",
                   help="Output directory for report and figures")
    p.add_argument("--config", type=str, default=None,
                   help="Path to training YAML config (optional)")
    p.add_argument("--skip-env", action="store_true",
                   help="Skip environment rollout, use pre-saved data")
    p.add_argument("--rollout-data", type=str, default=None,
                   help="Path to pre-saved rollout_data.npz")
    return p.parse_args()


# ── Data Collection ────────────────────────────────────────────────────────
def collect_rollout(actor, env, deterministic: bool = True) -> Dict[str, np.ndarray]:
    """Run one full episode, collecting obs/actions/rewards/costs per step."""
    obs_list, act_list, rew_list, cost_list = [], [], [], []
    reward_comp_lists = {k: [] for k in
        ['economic', 'stability_grid', 'stability_building', 'ramp', 'renewable']}
    cost_comp_lists = {k: [] for k in ['C1', 'C2', 'C3', 'C4']}

    obs, info = env.reset()
    done = False
    step = 0

    while not done:
        obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
        obs_list.append(obs_np.flatten())

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            action = actor(obs_t).squeeze(0).numpy()
        action = np.clip(action, -1.0, 1.0)
        act_list.append(action)

        obs, reward, cost, terminated, truncated, info = env.step(
            torch.as_tensor(action, dtype=torch.float32))

        rew_list.append(float(reward))
        cost_list.append(float(cost))

        # Extract per-component metrics from info
        for k in reward_comp_lists:
            reward_comp_lists[k].append(float(info.get(f'reward_{k}', 0.0)))
        cost_comp_lists['C1'].append(float(info.get('cost_ev_departure', 0.0)))
        cost_comp_lists['C2'].append(float(info.get('cost_stems_battery', 0.0)))
        cost_comp_lists['C3'].append(float(info.get('cost_stems_building_power', 0.0)))
        cost_comp_lists['C4'].append(float(info.get('cost_stems_grid_power', 0.0)))

        done = bool(terminated) or bool(truncated)
        step += 1

    return dict(
        obs=np.array(obs_list, dtype=np.float32),
        actions=np.array(act_list, dtype=np.float32),
        rewards=np.array(rew_list, dtype=np.float32),
        costs=np.array(cost_list, dtype=np.float32),
        cost_components={k: np.array(v, dtype=np.float32) for k, v in cost_comp_lists.items()},
        reward_components={k: np.array(v, dtype=np.float32) for k, v in reward_comp_lists.items()},
    )


def collect_zero_action_rollout(env) -> Dict[str, np.ndarray]:
    """Run one full episode with zero actions (do-nothing baseline)."""
    obs_list, rew_list, cost_list = [], [], []
    reward_comp_lists = {k: [] for k in
        ['economic', 'stability_grid', 'stability_building', 'ramp', 'renewable']}
    cost_comp_lists = {k: [] for k in ['C1', 'C2', 'C3', 'C4']}

    obs, info = env.reset()
    done = False
    zero_action = torch.zeros(ACT_DIM, dtype=torch.float32)

    while not done:
        obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
        obs_list.append(obs_np.flatten())
        obs, reward, cost, terminated, truncated, info = env.step(zero_action)

        rew_list.append(float(reward))
        cost_list.append(float(cost))

        for k in reward_comp_lists:
            reward_comp_lists[k].append(float(info.get(f'reward_{k}', 0.0)))
        cost_comp_lists['C1'].append(float(info.get('cost_ev_departure', 0.0)))
        cost_comp_lists['C2'].append(float(info.get('cost_stems_battery', 0.0)))
        cost_comp_lists['C3'].append(float(info.get('cost_stems_building_power', 0.0)))
        cost_comp_lists['C4'].append(float(info.get('cost_stems_grid_power', 0.0)))

        done = bool(terminated) or bool(truncated)

    return dict(
        obs=np.array(obs_list, dtype=np.float32),
        rewards=np.array(rew_list, dtype=np.float32),
        costs=np.array(cost_list, dtype=np.float32),
        cost_components={k: np.array(v, dtype=np.float32) for k, v in cost_comp_lists.items()},
        reward_components={k: np.array(v, dtype=np.float32) for k, v in reward_comp_lists.items()},
    )


# ── Placeholder test functions (implemented in subsequent tasks) ───────────
def test_value_function(data, checkpoint_path, gamma=GAMMA):
    """Test 1: Value Function Accuracy."""
    raise NotImplementedError("Task 2")

def test_feature_action_mi(data):
    """Test 2: Feature-Action Mutual Information."""
    raise NotImplementedError("Task 3")

def test_conditional_entropy(data):
    """Test 3: Conditional Entropy Decomposition."""
    raise NotImplementedError("Task 3")

def test_action_correlation(data):
    """Test 4: Inter-Building Action Correlation."""
    raise NotImplementedError("Task 4")

def test_gradient_attribution(data, actor):
    """Test 5: Gradient Attribution by Pathway."""
    raise NotImplementedError("Task 5")

def test_temporal_planning(data, actor=None):
    """Test 6: Temporal Planning Assessment."""
    raise NotImplementedError("Task 6")

def test_constraint_decomposition(data):
    """Test 7: Constraint Decomposition."""
    raise NotImplementedError("Task 7")

def test_headroom(data, baseline_data):
    """Test 8: Headroom Analysis."""
    raise NotImplementedError("Task 8")

def compute_phi(results):
    """Compute composite Policy Health Index."""
    raise NotImplementedError("Task 9")

def diagnose(results):
    """Automated decision tree diagnosis."""
    raise NotImplementedError("Task 9")

def generate_report(results, output_dir):
    """Generate markdown report + figures."""
    raise NotImplementedError("Task 9")


if __name__ == "__main__":
    args = parse_args()
    print(f"Policy Health Diagnostic Suite")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    # TODO: Load actor, run rollout, execute tests (Task 2+)
    print("Scaffold loaded. Tests not yet implemented.")
```

**Step 4: Run scaffold to verify it loads**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && conda run -n citylearn python scripts/diagnose_policy_health.py --checkpoint dummy --output-dir /tmp/test_diag`
Expected: Prints "Scaffold loaded" without import errors

**Step 5: Commit**

```bash
cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
git add scripts/diagnose_policy_health.py tests/test_diagnose_policy_health.py
git commit -m "feat: scaffold policy health diagnostic suite with CLI and data collection"
```

---

## Task 2: Test 1 — Value Function Accuracy

**Files:**
- Modify: `scripts/diagnose_policy_health.py` — implement `test_value_function()`
- Modify: `tests/test_diagnose_policy_health.py` — add test

**Step 1: Write the failing test**

```python
# In tests/test_diagnose_policy_health.py
from scripts.diagnose_policy_health import test_value_function

class TestValueFunction:
    def test_returns_expected_keys(self):
        data = make_synthetic_rollout(T=200)
        # Create mock value predictions (slightly noisy version of actual returns)
        T = len(data['rewards'])
        returns_r = np.zeros(T, dtype=np.float32)
        returns_r[-1] = data['rewards'][-1]
        for t in range(T - 2, -1, -1):
            returns_r[t] = data['rewards'][t] + 0.99 * returns_r[t + 1]
        # Good critic: actual returns + small noise
        v_pred_r = returns_r + np.random.randn(T).astype(np.float32) * 0.5
        v_pred_c = np.cumsum(data['costs'][::-1])[::-1].astype(np.float32) * 0.01
        v_pred_c += np.random.randn(T).astype(np.float32) * 0.5

        result = test_value_function(data, v_reward=v_pred_r, v_cost=v_pred_c, gamma=0.99)
        assert 'ev_reward' in result
        assert 'ev_cost' in result
        assert 'td_autocorr_reward' in result
        assert 'td_autocorr_cost' in result
        assert 'status' in result  # 'healthy', 'warning', or 'broken'
        # Good critic should have high explained variance
        assert result['ev_reward'] > 0.3

    def test_random_critic_is_broken(self):
        data = make_synthetic_rollout(T=200)
        T = len(data['rewards'])
        # Bad critic: random predictions
        v_pred_r = np.random.randn(T).astype(np.float32) * 100
        v_pred_c = np.random.randn(T).astype(np.float32) * 100
        result = test_value_function(data, v_reward=v_pred_r, v_cost=v_pred_c, gamma=0.99)
        assert result['ev_reward'] < 0.1
        assert result['status'] == 'broken'
```

**Step 2: Run test to verify it fails**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestValueFunction -v`
Expected: FAIL (NotImplementedError)

**Step 3: Implement `test_value_function()`**

```python
# In scripts/diagnose_policy_health.py, replace the placeholder
def test_value_function(data, v_reward=None, v_cost=None,
                        checkpoint_path=None, gamma=GAMMA):
    """Test 1: Value Function Accuracy.

    Args:
        data: rollout dict with 'rewards', 'costs'
        v_reward: (T,) reward critic predictions (or loaded from checkpoint)
        v_cost: (T,) cost critic predictions (or loaded from checkpoint)
    Returns:
        dict with ev_reward, ev_cost, td_autocorr_reward, td_autocorr_cost, status
    """
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

    # TD errors
    td_r = np.zeros(T - 1, dtype=np.float64)
    td_c = np.zeros(T - 1, dtype=np.float64)
    if v_reward is not None:
        td_r = rewards[:-1] + gamma * v_reward[1:] - v_reward[:-1]
    if v_cost is not None:
        td_c = costs[:-1] + gamma * v_cost[1:] - v_cost[:-1]

    # TD autocorrelation (lag 1)
    def autocorr_lag1(x):
        if len(x) < 3 or np.std(x) < 1e-10:
            return 0.0
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    td_ac_r = autocorr_lag1(td_r)
    td_ac_c = autocorr_lag1(td_c)

    # Status classification
    ev_min = min(ev_r if not np.isnan(ev_r) else 1.0,
                 ev_c if not np.isnan(ev_c) else 1.0)
    if ev_min < 0.1:
        status = 'broken'
    elif ev_min < 0.5:
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'ev_reward': float(ev_r),
        'ev_cost': float(ev_c),
        'td_autocorr_reward': float(td_ac_r),
        'td_autocorr_cost': float(td_ac_c),
        'returns_reward_mean': float(np.mean(returns_r)),
        'returns_cost_mean': float(np.mean(returns_c)),
        'status': status,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestValueFunction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/diagnose_policy_health.py tests/test_diagnose_policy_health.py
git commit -m "feat: implement Test 1 — value function accuracy (explained variance + TD autocorr)"
```

---

## Task 3: Tests 2 & 3 — Mutual Information + Conditional Entropy

**Files:**
- Modify: `scripts/diagnose_policy_health.py` — implement `test_feature_action_mi()` and `test_conditional_entropy()`
- Modify: `tests/test_diagnose_policy_health.py`

**Step 1: Write failing tests**

```python
from scripts.diagnose_policy_health import test_feature_action_mi, test_conditional_entropy

class TestMutualInformation:
    def test_detects_correlated_features(self):
        data = make_synthetic_rollout(T=2000)
        result = test_feature_action_mi(data)
        assert 'mi_matrix' in result  # (330, 9)
        assert result['mi_matrix'].shape == (330, 9)
        # Price (idx 22) should have higher MI with battery actions (0-4) than EV (5-7)
        price_battery_mi = np.mean(result['mi_matrix'][22, :5])
        price_ev_mi = np.mean(result['mi_matrix'][22, 5:8])
        assert price_battery_mi > price_ev_mi
        assert 'mean_mi_top5' in result
        assert 'status' in result

    def test_random_actions_low_mi(self):
        data = make_synthetic_rollout(T=2000)
        # Override actions with pure random
        data['actions'] = np.random.randn(2000, 9).astype(np.float32)
        data['actions'] = np.clip(data['actions'], -1, 1)
        result = test_feature_action_mi(data)
        assert result['mean_mi_top5'] < 0.1  # Should be near noise floor


class TestConditionalEntropy:
    def test_returns_expected_structure(self):
        data = make_synthetic_rollout(T=2000)
        result = test_conditional_entropy(data)
        assert 'entropy_reduction' in result  # dict of feature_name -> rho
        assert 'status' in result
        assert isinstance(result['entropy_reduction'], dict)
```

**Step 2: Run to verify failure**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestMutualInformation -v`
Expected: FAIL

**Step 3: Implement both functions**

```python
def test_feature_action_mi(data, n_neighbors=5):
    """Test 2: Feature-Action Mutual Information using KSG estimator."""
    from sklearn.feature_selection import mutual_info_regression

    obs = data['obs']  # (T, 330)
    actions = data['actions']  # (T, 9)
    T, obs_dim = obs.shape
    _, act_dim = actions.shape

    mi_matrix = np.zeros((obs_dim, act_dim), dtype=np.float32)
    for j in range(act_dim):
        mi_matrix[:, j] = mutual_info_regression(
            obs, actions[:, j], n_neighbors=n_neighbors, random_state=42
        )

    # Expected high-MI pairs (using global feature indices)
    # Price → battery actions (0-4)
    expected_pairs = []
    for j in range(5):  # battery actions
        expected_pairs.append((PRICE_IDX, j))
        if j < len(SOC_INDICES):
            expected_pairs.append((SOC_INDICES[j], j))

    top5_mi = sorted([mi_matrix[i, j] for i, j in expected_pairs], reverse=True)[:5]
    mean_mi_top5 = float(np.mean(top5_mi)) if top5_mi else 0.0

    # Noise floor: MI of random permutation
    obs_shuffled = obs.copy()
    np.random.shuffle(obs_shuffled)
    noise_mi = mutual_info_regression(
        obs_shuffled[:, :10], actions[:, 0], n_neighbors=n_neighbors, random_state=42
    )
    noise_floor = float(np.mean(noise_mi))

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
        'top5_pairs': [(int(i), int(j), float(mi_matrix[i, j])) for i, j in expected_pairs[:5]],
        'status': status,
    }


def test_conditional_entropy(data, n_bins=20):
    """Test 3: Conditional Entropy Decomposition."""
    obs = data['obs']
    actions = data['actions']
    T = obs.shape[0]

    # Key features to check
    feature_map = {
        'price': PRICE_IDX,
        'hour_cos': HOUR_COS_IDX,
        'hour_sin': HOUR_SIN_IDX,
    }
    for i, idx in enumerate(SOC_INDICES):
        feature_map[f'soc_{i}'] = idx

    entropy_reduction = {}
    for feat_name, feat_idx in feature_map.items():
        x = obs[:, feat_idx]
        # Skip constant features
        if np.std(x) < 1e-10:
            entropy_reduction[feat_name] = 0.0
            continue

        # Quantile binning
        try:
            bin_edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
            bin_edges[-1] += 1e-6  # Avoid edge case
            bins = np.digitize(x, bin_edges[1:-1])
        except Exception:
            entropy_reduction[feat_name] = 0.0
            continue

        # For each battery action, compute entropy reduction
        rhos = []
        for j in range(min(5, actions.shape[1])):
            a = actions[:, j]
            total_var = np.var(a)
            if total_var < 1e-10:
                rhos.append(0.0)
                continue

            # Conditional variance
            cond_var = 0.0
            for b in range(n_bins):
                mask = bins == b
                n_b = mask.sum()
                if n_b < 5:
                    continue
                p_b = n_b / T
                cond_var += p_b * np.var(a[mask])

            # rho = 1 - H(a|X)/H(a) ≈ 1 - cond_var/total_var (for Gaussian approx)
            rho = max(0.0, 1.0 - cond_var / (total_var + 1e-10))
            rhos.append(rho)

        entropy_reduction[feat_name] = float(np.mean(rhos)) if rhos else 0.0

    mean_rho = float(np.mean(list(entropy_reduction.values())))
    if mean_rho < 0.01:
        status = 'broken'
    elif mean_rho < 0.05:
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'entropy_reduction': entropy_reduction,
        'mean_rho_top5': float(np.mean(sorted(entropy_reduction.values(), reverse=True)[:5])),
        'status': status,
    }
```

**Step 4: Run tests**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestMutualInformation tests/test_diagnose_policy_health.py::TestConditionalEntropy -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/diagnose_policy_health.py tests/test_diagnose_policy_health.py
git commit -m "feat: implement Tests 2+3 — feature-action MI and conditional entropy"
```

---

## Task 4: Test 4 — Inter-Building Action Correlation

**Files:**
- Modify: `scripts/diagnose_policy_health.py` — implement `test_action_correlation()`
- Modify: `tests/test_diagnose_policy_health.py`

**Step 1: Write failing test**

```python
from scripts.diagnose_policy_health import test_action_correlation

class TestActionCorrelation:
    def test_detects_differentiated_buildings(self):
        data = make_synthetic_rollout(T=1000)
        result = test_action_correlation(data)
        assert 'battery_corr_matrix' in result  # (5, 5)
        assert result['battery_corr_matrix'].shape == (5, 5)
        assert 'spatial_variance_ratio' in result
        assert 'acf' in result  # dict of action_idx -> array of ACF values
        assert 'mean_abs_corr' in result
        assert 'status' in result

    def test_identical_actions_detected(self):
        data = make_synthetic_rollout(T=1000)
        # Make all battery actions identical
        for j in range(1, 5):
            data['actions'][:, j] = data['actions'][:, 0]
        result = test_action_correlation(data)
        assert result['mean_abs_corr'] > 0.95
        assert result['status'] == 'broken'
```

**Step 2: Run to verify failure**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestActionCorrelation -v`
Expected: FAIL

**Step 3: Implement**

```python
def test_action_correlation(data, max_lag=48):
    """Test 4: Inter-Building Action Correlation + Autocorrelation."""
    actions = data['actions']  # (T, 9)
    T = actions.shape[0]

    # Battery actions: columns 0-4
    battery_actions = actions[:, :5]

    # 5x5 Pearson correlation matrix
    battery_corr = np.corrcoef(battery_actions.T)  # (5, 5)

    # Mean absolute off-diagonal correlation
    mask = ~np.eye(5, dtype=bool)
    mean_abs_corr = float(np.mean(np.abs(battery_corr[mask])))

    # Spatial variance ratio
    # temporal variance = Var of cross-building mean over time
    cross_bld_mean = np.mean(battery_actions, axis=1)  # (T,)
    temporal_var = np.var(cross_bld_mean)
    # spatial variance = mean of Var across buildings at each timestep
    spatial_var_per_t = np.var(battery_actions, axis=1)  # (T,)
    spatial_var = float(np.mean(spatial_var_per_t))
    total_var = float(np.var(battery_actions))
    eta = spatial_var / (total_var + 1e-10)

    # Autocorrelation function for each battery action
    acf = {}
    for j in range(5):
        a = battery_actions[:, j]
        a_centered = a - np.mean(a)
        var_a = np.var(a)
        if var_a < 1e-10:
            acf[j] = np.zeros(max_lag + 1)
            continue
        acf_vals = np.zeros(max_lag + 1)
        for tau in range(max_lag + 1):
            if tau >= T:
                break
            acf_vals[tau] = np.mean(a_centered[:T - tau] * a_centered[tau:]) / (var_a + 1e-10)
        acf[j] = acf_vals

    # Mean ACF at lag 24 (daily cycle)
    mean_acf_24 = float(np.mean([acf[j][24] for j in range(5)])) if T > 24 else 0.0

    # Status
    if mean_abs_corr > 0.9 or eta < 0.05:
        status = 'broken'
    elif mean_abs_corr > 0.7 or eta < 0.1:
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'battery_corr_matrix': battery_corr,
        'mean_abs_corr': mean_abs_corr,
        'spatial_variance_ratio': eta,
        'temporal_var': temporal_var,
        'spatial_var': spatial_var,
        'acf': acf,
        'mean_acf_lag24': mean_acf_24,
        'status': status,
    }
```

**Step 4: Run tests**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestActionCorrelation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/diagnose_policy_health.py tests/test_diagnose_policy_health.py
git commit -m "feat: implement Test 4 — inter-building action correlation + ACF"
```

---

## Task 5: Test 5 — Gradient Attribution by Pathway

**Files:**
- Modify: `scripts/diagnose_policy_health.py` — implement `test_gradient_attribution()`
- Modify: `tests/test_diagnose_policy_health.py`

**Step 1: Write failing test**

```python
from scripts.diagnose_policy_health import test_gradient_attribution
import torch
import torch.nn as nn

class TestGradientAttribution:
    def _make_mock_actor(self):
        """Simple MLP that mimics the actor interface."""
        class MockActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(330, 64), nn.ReLU(),
                    nn.Linear(64, 9), nn.Tanh(),
                )
            def forward(self, x):
                return self.net(x)
        return MockActor()

    def test_returns_pathway_fractions(self):
        data = make_synthetic_rollout(T=100)
        actor = self._make_mock_actor()
        result = test_gradient_attribution(data, actor)
        assert 'temporal_fraction' in result
        assert 'current_fraction' in result
        assert 'global_fraction' in result
        assert 'status' in result
        # Fractions should sum to ~1
        total = result['temporal_fraction'] + result['current_fraction']
        assert 0.5 < total < 1.5  # Approximate due to overlap
```

**Step 2: Run to verify failure**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestGradientAttribution -v`
Expected: FAIL

**Step 3: Implement**

```python
def test_gradient_attribution(data, actor, n_samples=200):
    """Test 5: Gradient Attribution by Pathway.

    Computes mean |∂action/∂obs| grouped by observation pathway.
    """
    obs = data['obs'][:n_samples]
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    obs_t.requires_grad_(True)

    # Forward pass
    actions = actor(obs_t)  # (N, 9)

    # Compute gradient of each action w.r.t. all obs dims
    grad_abs = torch.zeros(obs_t.shape[1], dtype=torch.float32)
    for j in range(actions.shape[1]):
        actor.zero_grad()
        if obs_t.grad is not None:
            obs_t.grad.zero_()
        actions_j = actor(obs_t)[:, j].sum()
        actions_j.backward(retain_graph=False)
        if obs_t.grad is not None:
            grad_abs += obs_t.grad.abs().mean(dim=0).detach()
        obs_t = obs_t.detach().requires_grad_(True)

    grad_abs /= actions.shape[1]  # average over actions
    total_grad = grad_abs.sum().item()

    if total_grad < 1e-12:
        return {
            'temporal_fraction': 0.0, 'current_fraction': 0.0,
            'global_fraction': 0.0, 'price_gradient': 0.0,
            'mean_grad_per_dim': {}, 'status': 'broken',
        }

    # Pathway groupings
    temporal_grad = grad_abs[HISTORY_START:HISTORY_END].sum().item()
    current_grad = grad_abs[:HISTORY_START].sum().item()

    # Within current: global features (price, time, forecasts)
    global_dims = [PRICE_IDX, HOUR_COS_IDX, HOUR_SIN_IDX]
    global_grad = grad_abs[global_dims].sum().item()

    # Price specifically
    price_grad = grad_abs[PRICE_IDX].item()

    # Per-building features
    building_grad = sum(grad_abs[idx].item() for idx in SOC_INDICES)

    temporal_frac = temporal_grad / (total_grad + 1e-10)
    current_frac = current_grad / (total_grad + 1e-10)
    global_frac = global_grad / (total_grad + 1e-10)

    # Status
    if temporal_frac < 0.01 or price_grad < 0.001:
        status = 'broken'
    elif temporal_frac < 0.10 or price_grad < 0.01:
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'temporal_fraction': float(temporal_frac),
        'current_fraction': float(current_frac),
        'global_fraction': float(global_frac),
        'price_gradient': float(price_grad),
        'building_gradient': float(building_grad),
        'total_gradient': float(total_grad),
        'status': status,
    }
```

**Step 4: Run tests**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestGradientAttribution -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/diagnose_policy_health.py tests/test_diagnose_policy_health.py
git commit -m "feat: implement Test 5 — gradient attribution by encoder pathway"
```

---

## Task 6: Test 6 — Temporal Planning Assessment

**Files:**
- Modify: `scripts/diagnose_policy_health.py` — implement `test_temporal_planning()`
- Modify: `tests/test_diagnose_policy_health.py`

**Step 1: Write failing test**

```python
from scripts.diagnose_policy_health import test_temporal_planning

class TestTemporalPlanning:
    def test_detects_temporal_correlation(self):
        np.random.seed(42)
        T = 2000
        data = make_synthetic_rollout(T=T)
        # Inject temporal planning signal: action correlates with price 3 steps ahead
        price = data['obs'][:, PRICE_IDX]
        for j in range(5):
            # Shift price forward by 3 steps to simulate planning
            future_price = np.roll(price, -3)
            data['actions'][:, j] = -0.5 * future_price + 0.3 * np.random.randn(T)
            data['actions'][:, j] = np.clip(data['actions'][:, j], -1, 1)

        result = test_temporal_planning(data)
        assert 'tps' in result  # Temporal Planning Score
        assert 'cross_temporal_corr' in result  # dict: lag -> correlation
        assert 'perturbation_effects' in result
        assert 'status' in result
        # Should detect the forward correlation
        assert result['tps'] > 0.02

    def test_myopic_agent_low_tps(self):
        data = make_synthetic_rollout(T=2000)
        # Actions only correlated with current price (lag 0), not future
        price = data['obs'][:, PRICE_IDX]
        for j in range(5):
            data['actions'][:, j] = -0.5 * price + 0.5 * np.random.randn(2000)
            data['actions'][:, j] = np.clip(data['actions'][:, j], -1, 1)
        result = test_temporal_planning(data)
        # TPS should be low since no forward correlation beyond lag 0
        assert result['tps'] < 0.15
```

**Step 2: Run to verify failure**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestTemporalPlanning -v`
Expected: FAIL

**Step 3: Implement**

```python
def test_temporal_planning(data, actor=None, max_lag=24):
    """Test 6: Temporal Planning Assessment.

    Sub-tests:
      6a. Cross-temporal correlation (action vs future price)
      6b. Temporal Planning Score (TPS)
      6c. Perturbation tests (zero/shuffle history)
    """
    obs = data['obs']
    actions = data['actions']
    T = obs.shape[0]

    price = obs[:, PRICE_IDX]

    # ── 6a. Cross-temporal correlation ──────────────────────────────────
    cross_corr = {}
    for tau in range(-6, max_lag + 1):
        if abs(tau) >= T:
            cross_corr[tau] = 0.0
            continue
        if tau >= 0:
            a_slice = actions[:T - tau, :5]  # battery actions
            p_slice = price[tau:]
        else:
            a_slice = actions[-tau:, :5]
            p_slice = price[:T + tau]
        # Mean correlation across battery actions
        corrs = []
        for j in range(5):
            if np.std(a_slice[:, j]) < 1e-10 or np.std(p_slice) < 1e-10:
                corrs.append(0.0)
            else:
                corrs.append(float(np.corrcoef(a_slice[:, j], p_slice)[0, 1]))
        cross_corr[tau] = float(np.mean(corrs))

    # ── 6b. Temporal Planning Score ─────────────────────────────────────
    # TPS = weighted avg of |corr| for positive lags, with exponential decay
    tps_num = 0.0
    tps_den = 0.0
    for tau in range(1, max_lag + 1):
        decay = np.exp(-tau / 6.0)
        tps_num += abs(cross_corr.get(tau, 0.0)) * decay
        tps_den += decay
    tps = tps_num / (tps_den + 1e-10)

    # ── 6c. Perturbation tests ──────────────────────────────────────────
    perturbation_effects = {}
    if actor is not None:
        n_test = min(500, T)
        obs_sample = torch.as_tensor(obs[:n_test], dtype=torch.float32)
        with torch.no_grad():
            actions_orig = actor(obs_sample).numpy()

        # Zero history
        obs_zeroed = obs_sample.clone()
        obs_zeroed[:, HISTORY_START:HISTORY_END] = 0.0
        with torch.no_grad():
            actions_zeroed = actor(obs_zeroed).numpy()
        perturbation_effects['zero_history'] = float(
            np.mean(np.abs(actions_orig - actions_zeroed)))

        # Shuffle history timesteps
        obs_shuffled = obs_sample.clone()
        history = obs_shuffled[:, HISTORY_START:HISTORY_END].reshape(
            n_test, TEMPORAL_WINDOW, TEMPORAL_FEATURES_PER_STEP)
        # Random permutation of time dimension
        perm = torch.randperm(TEMPORAL_WINDOW)
        history = history[:, perm, :]
        obs_shuffled[:, HISTORY_START:HISTORY_END] = history.reshape(
            n_test, TEMPORAL_WINDOW * TEMPORAL_FEATURES_PER_STEP)
        with torch.no_grad():
            actions_shuffled = actor(obs_shuffled).numpy()
        perturbation_effects['shuffle_history'] = float(
            np.mean(np.abs(actions_orig - actions_shuffled)))

        # Reverse history
        obs_reversed = obs_sample.clone()
        history_rev = obs_reversed[:, HISTORY_START:HISTORY_END].reshape(
            n_test, TEMPORAL_WINDOW, TEMPORAL_FEATURES_PER_STEP)
        history_rev = history_rev.flip(dims=[1])
        obs_reversed[:, HISTORY_START:HISTORY_END] = history_rev.reshape(
            n_test, TEMPORAL_WINDOW * TEMPORAL_FEATURES_PER_STEP)
        with torch.no_grad():
            actions_reversed = actor(obs_reversed).numpy()
        perturbation_effects['reverse_history'] = float(
            np.mean(np.abs(actions_orig - actions_reversed)))
    else:
        perturbation_effects = {'zero_history': float('nan'),
                                 'shuffle_history': float('nan'),
                                 'reverse_history': float('nan')}

    # ── Status ──────────────────────────────────────────────────────────
    perturb_effect = perturbation_effects.get('zero_history', 0.0)
    if np.isnan(perturb_effect):
        perturb_effect = 0.0

    if tps < 0.02 and perturb_effect < 0.01:
        status = 'broken'
    elif tps < 0.05:
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'tps': float(tps),
        'cross_temporal_corr': cross_corr,
        'perturbation_effects': perturbation_effects,
        'lag0_corr': float(cross_corr.get(0, 0.0)),
        'status': status,
    }
```

**Step 4: Run tests**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestTemporalPlanning -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/diagnose_policy_health.py tests/test_diagnose_policy_health.py
git commit -m "feat: implement Test 6 — temporal planning assessment (TPS + perturbation)"
```

---

## Task 7: Test 7 — Constraint Decomposition

**Files:**
- Modify: `scripts/diagnose_policy_health.py` — implement `test_constraint_decomposition()`
- Modify: `tests/test_diagnose_policy_health.py`

**Step 1: Write failing test**

```python
from scripts.diagnose_policy_health import test_constraint_decomposition

class TestConstraintDecomposition:
    def test_computes_violation_metrics(self):
        data = make_synthetic_rollout(T=500)
        result = test_constraint_decomposition(data)
        assert 'per_constraint' in result
        for c in ['C1', 'C2', 'C3', 'C4']:
            assert c in result['per_constraint']
            m = result['per_constraint'][c]
            assert 'violation_rate' in m
            assert 'violation_magnitude' in m
            assert 0.0 <= m['violation_rate'] <= 1.0
        assert 'total_behavioral_vr' in result
        assert 'status' in result
```

**Step 2: Run to verify failure**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestConstraintDecomposition -v`
Expected: FAIL

**Step 3: Implement**

```python
def test_constraint_decomposition(data):
    """Test 7: Constraint Decomposition — structural vs behavioral violations."""
    cost_components = data.get('cost_components', {})
    T = len(data['costs'])

    per_constraint = {}
    for c_name, c_values in cost_components.items():
        violations = c_values > 0
        vr = float(np.mean(violations))
        vm = float(np.mean(c_values[violations])) if np.any(violations) else 0.0
        total_cost = float(np.sum(c_values))

        # Violation timing by hour (assuming hourly steps, 8759 = 1 year)
        hours = np.arange(T) % 24
        hourly_vr = np.zeros(24)
        for h in range(24):
            mask_h = hours == h
            if mask_h.sum() > 0:
                hourly_vr[h] = float(np.mean(violations[mask_h]))

        # Concentration: what fraction of total cost comes from top 10% of violating steps
        if np.any(violations):
            sorted_costs = np.sort(c_values[violations])[::-1]
            top_10_pct = max(1, int(0.1 * len(sorted_costs)))
            concentration = float(np.sum(sorted_costs[:top_10_pct]) / (np.sum(sorted_costs) + 1e-10))
        else:
            concentration = 0.0

        per_constraint[c_name] = {
            'violation_rate': vr,
            'violation_magnitude': vm,
            'total_cost': total_cost,
            'hourly_violation_rate': hourly_vr.tolist(),
            'concentration_top10pct': concentration,
        }

    # Total behavioral VR (approximate: assume C1 and C4 are largely behavioral,
    # C2 and C3 have large structural components)
    behavioral_vrs = []
    if 'C1' in per_constraint:
        behavioral_vrs.append(per_constraint['C1']['violation_rate'])
    if 'C4' in per_constraint:
        behavioral_vrs.append(per_constraint['C4']['violation_rate'])
    total_behavioral_vr = float(np.mean(behavioral_vrs)) if behavioral_vrs else 0.0

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
```

**Step 4: Run tests**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestConstraintDecomposition -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/diagnose_policy_health.py tests/test_diagnose_policy_health.py
git commit -m "feat: implement Test 7 — constraint decomposition with violation timing"
```

---

## Task 8: Test 8 — Headroom Analysis

**Files:**
- Modify: `scripts/diagnose_policy_health.py` — implement `test_headroom()`
- Modify: `tests/test_diagnose_policy_health.py`

**Step 1: Write failing test**

```python
from scripts.diagnose_policy_health import test_headroom

class TestHeadroom:
    def test_computes_headroom_per_component(self):
        data = make_synthetic_rollout(T=500)
        baseline = make_synthetic_rollout(T=500)
        # Make baseline slightly better on rewards
        baseline['rewards'] = data['rewards'] + 2.0
        result = test_headroom(data, baseline)
        assert 'reward_headroom' in result
        assert 'cost_headroom' in result
        assert 'total_reward_vs_baseline' in result
        assert 'status' in result
        # Per component
        for k in ['economic', 'stability_grid', 'stability_building', 'ramp', 'renewable']:
            assert k in result['reward_headroom']
        for k in ['C1', 'C2', 'C3', 'C4']:
            assert k in result['cost_headroom']

    def test_worse_than_baseline_detected(self):
        data = make_synthetic_rollout(T=500)
        baseline = make_synthetic_rollout(T=500)
        # Agent worse than baseline
        data['rewards'] = baseline['rewards'] - 10.0
        result = test_headroom(data, baseline)
        assert result['total_reward_vs_baseline'] < 0  # worse
        assert result['status'] in ['broken', 'warning']
```

**Step 2: Run to verify failure**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestHeadroom -v`
Expected: FAIL

**Step 3: Implement**

```python
def test_headroom(data, baseline_data):
    """Test 8: Headroom Analysis — compare policy vs zero-action baseline."""
    # Reward headroom per component
    reward_headroom = {}
    for k in ['economic', 'stability_grid', 'stability_building', 'ramp', 'renewable']:
        agent_val = float(np.sum(data['reward_components'].get(k, [0])))
        base_val = float(np.sum(baseline_data['reward_components'].get(k, [0])))
        reward_headroom[k] = {
            'agent': agent_val,
            'baseline': base_val,
            'delta': agent_val - base_val,
            'better_than_baseline': agent_val > base_val,
        }

    # Cost headroom per component
    cost_headroom = {}
    for k in ['C1', 'C2', 'C3', 'C4']:
        agent_val = float(np.sum(data['cost_components'].get(k, [0])))
        base_val = float(np.sum(baseline_data['cost_components'].get(k, [0])))
        cost_headroom[k] = {
            'agent': agent_val,
            'baseline': base_val,
            'delta': agent_val - base_val,  # negative = agent has less cost = better
            'better_than_baseline': agent_val < base_val,
        }

    # Totals
    agent_total_reward = float(np.sum(data['rewards']))
    base_total_reward = float(np.sum(baseline_data['rewards']))
    agent_total_cost = float(np.sum(data['costs']))
    base_total_cost = float(np.sum(baseline_data['costs']))

    reward_vs_baseline = agent_total_reward - base_total_reward
    cost_vs_baseline = agent_total_cost - base_total_cost

    # Status: worse than zero-action on reward = broken
    if reward_vs_baseline < 0:
        status = 'broken'
    elif reward_vs_baseline < base_total_reward * 0.05:  # <5% improvement
        status = 'warning'
    else:
        status = 'healthy'

    return {
        'reward_headroom': reward_headroom,
        'cost_headroom': cost_headroom,
        'total_reward_vs_baseline': float(reward_vs_baseline),
        'total_cost_vs_baseline': float(cost_vs_baseline),
        'agent_total_reward': agent_total_reward,
        'baseline_total_reward': base_total_reward,
        'agent_total_cost': agent_total_cost,
        'baseline_total_cost': base_total_cost,
        'status': status,
    }
```

**Step 4: Run tests**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestHeadroom -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/diagnose_policy_health.py tests/test_diagnose_policy_health.py
git commit -m "feat: implement Test 8 — headroom analysis vs zero-action baseline"
```

---

## Task 9: PHI Score, Decision Tree, Report Generation

**Files:**
- Modify: `scripts/diagnose_policy_health.py` — implement `compute_phi()`, `diagnose()`, `generate_report()`
- Modify: `tests/test_diagnose_policy_health.py`

**Step 1: Write failing test**

```python
from scripts.diagnose_policy_health import compute_phi, diagnose

class TestPHIAndDiagnosis:
    def test_phi_returns_float_0_to_1(self):
        results = {
            'test1': {'ev_reward': 0.7, 'ev_cost': 0.6, 'status': 'healthy'},
            'test2': {'mean_mi_top5': 0.15, 'status': 'healthy'},
            'test3': {'mean_rho_top5': 0.1, 'status': 'healthy'},
            'test4': {'mean_abs_corr': 0.3, 'spatial_variance_ratio': 0.15, 'status': 'healthy'},
            'test5': {'temporal_fraction': 0.15, 'price_gradient': 0.02, 'status': 'healthy'},
            'test6': {'tps': 0.08, 'perturbation_effects': {'zero_history': 0.06}, 'status': 'healthy'},
            'test7': {'total_behavioral_vr': 0.2, 'status': 'healthy'},
            'test8': {'total_reward_vs_baseline': 100.0, 'status': 'healthy'},
        }
        phi = compute_phi(results)
        assert 0.0 <= phi <= 1.0
        assert phi > 0.3  # Healthy inputs → decent score

    def test_broken_critic_diagnosed(self):
        results = {
            'test1': {'ev_reward': 0.05, 'ev_cost': 0.02, 'status': 'broken'},
            'test2': {'mean_mi_top5': 0.01, 'status': 'broken'},
            'test3': {'mean_rho_top5': 0.0, 'status': 'broken'},
            'test4': {'mean_abs_corr': 0.95, 'spatial_variance_ratio': 0.02, 'status': 'broken'},
            'test5': {'temporal_fraction': 0.005, 'price_gradient': 0.0005, 'status': 'broken'},
            'test6': {'tps': 0.01, 'perturbation_effects': {'zero_history': 0.005}, 'status': 'broken'},
            'test7': {'total_behavioral_vr': 0.6, 'status': 'broken'},
            'test8': {'total_reward_vs_baseline': -500.0, 'status': 'broken'},
        }
        diagnosis = diagnose(results)
        assert 'CRITIC BROKEN' in diagnosis
```

**Step 2: Run to verify failure**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestPHIAndDiagnosis -v`
Expected: FAIL

**Step 3: Implement all three functions**

```python
def compute_phi(results):
    """Compute composite Policy Health Index (0-1)."""
    def clip01(x):
        return max(0.0, min(1.0, x))

    t1 = results.get('test1', {})
    t2 = results.get('test2', {})
    t3 = results.get('test3', {})
    t4 = results.get('test4', {})
    t5 = results.get('test5', {})
    t6 = results.get('test6', {})
    t7 = results.get('test7', {})

    s_value = clip01(np.mean([
        t1.get('ev_reward', 0), t1.get('ev_cost', 0)
    ]))
    s_mi = clip01(t2.get('mean_mi_top5', 0) / 0.3)
    s_entropy = clip01(t3.get('mean_rho_top5', 0) / 0.2)

    corr_score = clip01(1.0 - t4.get('mean_abs_corr', 1.0) / 0.9)
    eta_score = clip01(t4.get('spatial_variance_ratio', 0) / 0.1)
    s_corr = corr_score * eta_score

    temp_frac_score = clip01(t5.get('temporal_fraction', 0) / 0.2)
    price_score = clip01(t5.get('price_gradient', 0) / 0.01)
    s_gradient = temp_frac_score * price_score

    tps_score = clip01(t6.get('tps', 0) / 0.1)
    perturb = t6.get('perturbation_effects', {}).get('zero_history', 0)
    if isinstance(perturb, float) and np.isnan(perturb):
        perturb = 0.0
    perturb_score = clip01(perturb / 0.05)
    s_temporal = (tps_score * perturb_score) ** 0.5  # geometric mean of 2

    s_constraints = clip01(1.0 - t7.get('total_behavioral_vr', 1.0) / 0.5)

    phi = (0.20 * s_value + 0.20 * s_mi + 0.10 * s_entropy +
           0.10 * s_corr + 0.10 * s_gradient + 0.20 * s_temporal +
           0.10 * s_constraints)
    return float(phi)


def diagnose(results):
    """Automated decision tree → actionable diagnosis string."""
    t1 = results.get('test1', {})
    t2 = results.get('test2', {})
    t4 = results.get('test4', {})
    t5 = results.get('test5', {})
    t6 = results.get('test6', {})
    t7 = results.get('test7', {})
    t8 = results.get('test8', {})

    ev_r = t1.get('ev_reward', 0)
    ev_c = t1.get('ev_cost', 0)
    if ev_r < 0.1 or ev_c < 0.1:
        return ("CRITIC BROKEN: Value function cannot predict returns "
                f"(EV_r={ev_r:.3f}, EV_c={ev_c:.3f}). "
                "Fix: increase critic LR, add critic update epochs, "
                "check if obs_normalize is working properly.")

    mi = t2.get('mean_mi_top5', 0)
    if mi < 0.02:
        price_grad = t5.get('price_gradient', 0)
        if price_grad < 0.001:
            return ("ENCODER DEAD: Price→action gradient is near zero "
                    f"(price_grad={price_grad:.5f}). "
                    "Fix: check global encoder → broadcast → per-node integration.")
        else:
            return ("POLICY TOO NOISY: Gradients exist but MI is low "
                    f"(MI_top5={mi:.4f}, price_grad={price_grad:.4f}). "
                    "Fix: reduce PolicyStd, increase training epochs.")

    corr = t4.get('mean_abs_corr', 0)
    if corr > 0.9:
        return ("GCN DEAD: All buildings take identical actions "
                f"(mean_corr={corr:.3f}). "
                "Fix: check adjacency learning, gated fusion weights, "
                "per-building feature differentiation.")

    reward_vs = t8.get('total_reward_vs_baseline', 0)
    if reward_vs < 0:
        return ("WORSE THAN ZERO-ACTION: Lambda erasing reward signal "
                f"(reward_delta={reward_vs:.1f}). "
                "Fix: enable standardized_cost_adv: true, raise cost_limit "
                "to match structural floor, drop non-binding constraints.")

    temporal = t5.get('temporal_fraction', 0)
    tps = t6.get('tps', 0)
    if temporal < 0.01 and tps < 0.02:
        return ("TEMPORAL DEAD: History has no effect on actions "
                f"(temporal_frac={temporal:.4f}, TPS={tps:.4f}). "
                "Fix: check temporal transformer → concat projection weights.")

    behavioral_vr = t7.get('total_behavioral_vr', 0)
    if behavioral_vr > 0.5:
        return ("CONSTRAINT FAILURE: Agent causes avoidable violations "
                f"(behavioral_VR={behavioral_vr:.3f}). "
                "Fix: enable dense cost shaping (CITYLEARN_EV_DENSE_COST_SCALE), "
                "increase cost critic capacity.")

    return ("ARCHITECTURE WORKS: Policy shows learning signals across all tests. "
            "Needs more training epochs with lambda cap to converge.")


def generate_report(results, phi, diagnosis, output_dir):
    """Generate markdown report and figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Figure 1: MI Heatmap (top 30 features × 9 actions) ────────────
    mi_matrix = results.get('test2', {}).get('mi_matrix', None)
    if mi_matrix is not None:
        # Find top 30 features by max MI across actions
        max_mi_per_feat = np.max(mi_matrix, axis=1)
        top30 = np.argsort(max_mi_per_feat)[-30:][::-1]
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mi_matrix[top30, :], aspect='auto', cmap='YlOrRd')
        ax.set_xlabel('Action dimension')
        ax.set_ylabel('Observation feature (top 30)')
        ax.set_yticks(range(len(top30)))
        ax.set_yticklabels([str(i) for i in top30], fontsize=7)
        ax.set_xticks(range(9))
        ax.set_xticklabels([f'a{i}' for i in range(9)])
        plt.colorbar(im, label='MI (nats)')
        ax.set_title('Feature-Action Mutual Information (Top 30 Features)')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'mi_heatmap.png'), dpi=150)
        plt.close()

    # ── Figure 2: Cross-temporal correlation ───────────────────────────
    ctc = results.get('test6', {}).get('cross_temporal_corr', {})
    if ctc:
        lags = sorted(ctc.keys())
        vals = [ctc[l] for l in lags]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(lags, vals, color=['steelblue' if l >= 0 else 'salmon' for l in lags])
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='red', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Lag τ (hours)')
        ax.set_ylabel('corr(a_battery, price_{t+τ})')
        ax.set_title(f'Temporal Planning Fingerprint (TPS={results.get("test6", {}).get("tps", 0):.4f})')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'cross_temporal_corr.png'), dpi=150)
        plt.close()

    # ── Figure 3: Building correlation matrix ──────────────────────────
    bcm = results.get('test4', {}).get('battery_corr_matrix', None)
    if bcm is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(bcm, cmap='RdBu_r', vmin=-1, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{bcm[i,j]:.2f}', ha='center', va='center', fontsize=9)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([f'Bld {i}' for i in range(5)])
        ax.set_yticklabels([f'Bld {i}' for i in range(5)])
        plt.colorbar(im, label='Pearson r')
        ax.set_title(f'Battery Action Correlation (η={results.get("test4", {}).get("spatial_variance_ratio", 0):.3f})')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'building_correlation.png'), dpi=150)
        plt.close()

    # ── Figure 4: Action autocorrelation ───────────────────────────────
    acf_data = results.get('test4', {}).get('acf', {})
    if acf_data:
        fig, ax = plt.subplots(figsize=(10, 4))
        for j in range(min(5, len(acf_data))):
            ax.plot(acf_data[j][:49], label=f'Bld {j}', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(24, color='red', linewidth=0.5, linestyle='--', label='24h cycle')
        ax.set_xlabel('Lag (hours)')
        ax.set_ylabel('ACF')
        ax.set_title('Battery Action Autocorrelation')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'action_autocorrelation.png'), dpi=150)
        plt.close()

    # ── Figure 5: Violation timing heatmap ─────────────────────────────
    pc = results.get('test7', {}).get('per_constraint', {})
    if pc:
        constraint_names = [k for k in ['C1', 'C2', 'C3', 'C4'] if k in pc]
        if constraint_names:
            hourly_data = np.array([pc[c]['hourly_violation_rate'] for c in constraint_names])
            fig, ax = plt.subplots(figsize=(10, 3))
            im = ax.imshow(hourly_data, aspect='auto', cmap='Reds')
            ax.set_xticks(range(24))
            ax.set_yticks(range(len(constraint_names)))
            ax.set_yticklabels(constraint_names)
            ax.set_xlabel('Hour of Day')
            ax.set_title('Constraint Violation Rate by Hour')
            plt.colorbar(im, label='Violation Rate')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'violation_timing.png'), dpi=150)
            plt.close()

    # ── Markdown Report ────────────────────────────────────────────────
    lines = [
        f"# Policy Health Diagnostic Report",
        f"",
        f"**Policy Health Index (PHI): {phi:.3f}**",
        f"",
        f"**Diagnosis:** {diagnosis}",
        f"",
        f"## PHI Interpretation",
        f"- < 0.1: Dead policy | 0.1-0.3: Failing | 0.3-0.5: Learning | 0.5-0.7: Functional | > 0.7: Intelligent",
        f"",
        f"---",
        f"",
    ]

    # Test summaries
    test_names = {
        'test1': 'Value Function Accuracy',
        'test2': 'Feature-Action Mutual Information',
        'test3': 'Conditional Entropy',
        'test4': 'Action Correlation',
        'test5': 'Gradient Attribution',
        'test6': 'Temporal Planning',
        'test7': 'Constraint Decomposition',
        'test8': 'Headroom Analysis',
    }
    for key, name in test_names.items():
        r = results.get(key, {})
        status = r.get('status', 'N/A')
        status_icon = {'healthy': 'PASS', 'warning': 'WARN', 'broken': 'FAIL'}.get(status, '???')
        lines.append(f"## Test: {name} [{status_icon}]")
        lines.append(f"")
        # Dump key metrics
        for k, v in r.items():
            if k in ('mi_matrix', 'acf', 'battery_corr_matrix', 'cross_temporal_corr',
                      'hourly_violation_rate', 'entropy_reduction', 'perturbation_effects',
                      'reward_headroom', 'cost_headroom', 'per_constraint'):
                if isinstance(v, dict):
                    lines.append(f"**{k}:**")
                    for sk, sv in v.items():
                        if isinstance(sv, dict):
                            lines.append(f"  - {sk}: {json.dumps(sv, default=str)}")
                        elif isinstance(sv, (list, np.ndarray)):
                            continue  # skip large arrays
                        else:
                            lines.append(f"  - {sk}: {sv}")
                continue
            if isinstance(v, (np.ndarray,)):
                continue
            lines.append(f"- **{k}:** {v}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    # Save PHI score
    phi_path = os.path.join(output_dir, "phi_score.json")
    with open(phi_path, 'w') as f:
        json.dump({'phi': phi, 'diagnosis': diagnosis}, f, indent=2)

    return report_path
```

**Step 4: Run tests**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestPHIAndDiagnosis -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/diagnose_policy_health.py tests/test_diagnose_policy_health.py
git commit -m "feat: implement PHI composite score, decision tree diagnosis, and report generation"
```

---

## Task 10: Wire Up Main — Checkpoint Loading + Full Pipeline

**Files:**
- Modify: `scripts/diagnose_policy_health.py` — complete `__main__` block with checkpoint loading, env setup, full pipeline
- Modify: `tests/test_diagnose_policy_health.py` — add integration test with synthetic data

**Step 1: Write failing integration test**

```python
class TestFullPipeline:
    def test_synthetic_pipeline(self, tmp_path):
        """Test full pipeline with synthetic data (no env, no checkpoint)."""
        from scripts.diagnose_policy_health import run_all_tests, compute_phi, diagnose, generate_report

        data = make_synthetic_rollout(T=500)
        baseline = make_synthetic_rollout(T=500)

        results = run_all_tests(data, baseline, actor=None)
        assert len(results) == 8

        phi = compute_phi(results)
        assert 0.0 <= phi <= 1.0

        diag = diagnose(results)
        assert isinstance(diag, str) and len(diag) > 10

        report_path = generate_report(results, phi, diag, str(tmp_path))
        assert os.path.exists(report_path)
```

**Step 2: Run to verify failure**

Run: `conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestFullPipeline -v`
Expected: FAIL (run_all_tests not defined)

**Step 3: Implement `run_all_tests()` and complete `__main__`**

```python
def run_all_tests(data, baseline_data, actor=None,
                  v_reward=None, v_cost=None,
                  checkpoint_path=None):
    """Execute all 8 diagnostic tests and return results dict."""
    results = {}

    print("  [1/8] Value Function Accuracy...")
    if v_reward is not None and v_cost is not None:
        results['test1'] = test_value_function(data, v_reward=v_reward, v_cost=v_cost)
    else:
        results['test1'] = {'ev_reward': float('nan'), 'ev_cost': float('nan'),
                            'td_autocorr_reward': float('nan'), 'td_autocorr_cost': float('nan'),
                            'status': 'skipped'}

    print("  [2/8] Feature-Action MI...")
    results['test2'] = test_feature_action_mi(data)

    print("  [3/8] Conditional Entropy...")
    results['test3'] = test_conditional_entropy(data)

    print("  [4/8] Action Correlation...")
    results['test4'] = test_action_correlation(data)

    print("  [5/8] Gradient Attribution...")
    if actor is not None:
        results['test5'] = test_gradient_attribution(data, actor)
    else:
        results['test5'] = {'temporal_fraction': float('nan'), 'current_fraction': float('nan'),
                            'global_fraction': float('nan'), 'price_gradient': float('nan'),
                            'status': 'skipped'}

    print("  [6/8] Temporal Planning...")
    results['test6'] = test_temporal_planning(data, actor=actor)

    print("  [7/8] Constraint Decomposition...")
    results['test7'] = test_constraint_decomposition(data)

    print("  [8/8] Headroom Analysis...")
    if baseline_data is not None:
        results['test8'] = test_headroom(data, baseline_data)
    else:
        results['test8'] = {'total_reward_vs_baseline': float('nan'), 'status': 'skipped'}

    return results


# ── Checkpoint & Environment Loading ───────────────────────────────────
def load_actor_from_checkpoint(checkpoint_path):
    """Load STEMSMeanNet or MLP actor from OmniSafe checkpoint.

    Returns: (actor, obs_normalizer_dict_or_None)
    """
    from citylearn_safe.stems_encoder_v3 import STEMSEncoderV3
    from citylearn_safe.stems_encoder_5bld import build_node_indices

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("pi", ckpt)

    # Detect STEMS vs MLP: STEMS has "mean.encoder." keys
    is_stems = any(k.startswith("mean.encoder.") for k in state.keys())

    if is_stems:
        # Reconstruct encoder from buffer tensors
        bld_idx = state["mean.encoder.bld_idx"]
        ev_idx = state["mean.encoder.ev_idx"]
        ev_mask = state["mean.encoder.ev_mask"]
        global_idx = state["mean.encoder.global_idx"]

        # Infer dimensions
        global_enc_weight = state["mean.encoder.global_enc.0.weight"]
        global_raw_dim = global_enc_weight.shape[1]
        n_global = global_idx.shape[0]
        forecast_dim = global_raw_dim - n_global
        base_obs_dim = CURRENT_OBS_DIM - forecast_dim

        node_info = {
            'building_indices': bld_idx.tolist(),
            'ev_indices': ev_idx.tolist(),
            'global_indices': global_idx.tolist(),
            'base_obs_dim': int(base_obs_dim),
        }

        encoder = STEMSEncoderV3(
            obs_dim=OBS_DIM,
            node_info=node_info,
            num_buildings=NUM_BUILDINGS,
            hidden_dim=64,
            global_hidden=32,
            temporal_window=TEMPORAL_WINDOW,
            temporal_features_per_step=TEMPORAL_FEATURES_PER_STEP,
            temporal_hidden=32,
            temporal_heads=4,
            num_gcn_layers=3,
            dropout=0.1,
            output_dim=256,
        )

        class STEMSMeanNet(nn.Module):
            def __init__(self, enc, act_dim=ACT_DIM):
                super().__init__()
                self.encoder = enc
                self.action_head = nn.Sequential(
                    nn.Linear(enc.output_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, act_dim),
                )
            def forward(self, obs):
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                return torch.tanh(self.action_head(self.encoder(obs)))

        actor = STEMSMeanNet(encoder)
        # Load weights
        mean_state = {}
        for k, v in state.items():
            if k == "log_std":
                continue
            if k.startswith("mean."):
                mean_state[k[5:]] = v  # strip "mean." prefix
            else:
                mean_state[k] = v
        actor.load_state_dict(mean_state, strict=False)
    else:
        # MLP actor
        h1 = state["mean.0.weight"].shape[0]
        h2 = state["mean.2.weight"].shape[0]
        in_dim = state["mean.0.weight"].shape[1]
        out_dim = state["mean.4.weight"].shape[0]

        class MLPActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.mean = nn.Sequential(
                    nn.Linear(in_dim, h1), nn.Tanh(),
                    nn.Linear(h1, h2), nn.Tanh(),
                    nn.Linear(h2, out_dim), nn.Tanh(),
                )
            def forward(self, obs):
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                return self.mean(obs)

        actor = MLPActor()
        mean_state = {k: v for k, v in state.items() if not k.startswith("log_std")}
        actor.load_state_dict(mean_state, strict=False)

    actor.eval()

    # Load obs normalizer if present
    obs_norm = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_norm = {
            'mean': np.array(norm["_mean"], dtype=np.float32),
            'std': np.array(norm["_std"], dtype=np.float32),
            'clip': float(norm.get("_clip", torch.tensor(5.0)).float().mean()),
        }

    return actor, obs_norm


def load_critics_from_checkpoint(checkpoint_path):
    """Load reward and cost critics from checkpoint. Returns (v_r_fn, v_c_fn) or (None, None)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    critics = {}
    for key_prefix, name in [("vr", "reward"), ("vc", "cost")]:
        if key_prefix not in ckpt:
            continue
        state = ckpt[key_prefix]
        # Infer MLP shape from weight keys
        layer_keys = sorted([k for k in state.keys() if "weight" in k])
        if not layer_keys:
            continue
        in_dim = state[layer_keys[0]].shape[1]
        layers = []
        for k in layer_keys:
            out_d = state[k].shape[0]
            in_d = state[k].shape[1]
            layers.append(nn.Linear(in_d, out_d))
            if k != layer_keys[-1]:
                layers.append(nn.Tanh())
        critic = nn.Sequential(*layers)
        critic.load_state_dict(state, strict=False)
        critic.eval()
        critics[name] = critic

    return critics.get("reward"), critics.get("cost")


if __name__ == "__main__":
    args = parse_args()
    print(f"Policy Health Diagnostic Suite")
    print(f"{'=' * 50}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    if args.rollout_data and os.path.exists(args.rollout_data):
        print(f"\nLoading pre-saved rollout from {args.rollout_data}")
        loaded = np.load(args.rollout_data, allow_pickle=True)
        data = {k: loaded[k] for k in loaded.files}
        if 'cost_components' in loaded:
            data['cost_components'] = loaded['cost_components'].item()
        if 'reward_components' in loaded:
            data['reward_components'] = loaded['reward_components'].item()
        actor, obs_norm = None, None
        baseline_data = None
    else:
        # Set environment variables
        os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
        os.environ["CITYLEARN_TEMPORAL_WINDOW"] = str(TEMPORAL_WINDOW)
        os.environ["STEMS_ENCODER_VERSION"] = "v3"
        SCHEMA_PATH = os.path.join(PROJECT_ROOT, "data", "schemas", "schema_5bld.json")
        if os.path.exists(SCHEMA_PATH):
            os.environ["CITYLEARN_SCHEMA"] = SCHEMA_PATH

        print(f"\nLoading actor from checkpoint...")
        actor, obs_norm = load_actor_from_checkpoint(args.checkpoint)
        print(f"  Actor type: {'STEMS' if hasattr(actor, 'encoder') else 'MLP'}")

        print(f"\nLoading critics...")
        v_r_critic, v_c_critic = load_critics_from_checkpoint(args.checkpoint)
        print(f"  Reward critic: {'loaded' if v_r_critic else 'not found'}")
        print(f"  Cost critic: {'loaded' if v_c_critic else 'not found'}")

        if not args.skip_env:
            # Import env registration
            import citylearn_safe.omni_env_v2  # noqa: F401 (registers env)
            from omnisafe.envs.core import make as omnisafe_make
            print(f"\nCreating environment...")
            env = omnisafe_make("CityLearnSafety-V2G-v2")

            print(f"Collecting policy rollout (8,759 steps)...")
            t0 = time.time()
            if obs_norm:
                # Wrap actor with normalization
                class NormalizedActor(nn.Module):
                    def __init__(self, base_actor, mean, std, clip_val):
                        super().__init__()
                        self.base = base_actor
                        self.register_buffer('obs_mean', torch.as_tensor(mean))
                        self.register_buffer('obs_std', torch.as_tensor(std))
                        self.clip_val = clip_val
                    def forward(self, obs):
                        obs_n = (obs - self.obs_mean) / (self.obs_std + 1e-8)
                        obs_n = obs_n.clamp(-self.clip_val, self.clip_val)
                        return self.base(obs_n)
                wrapped_actor = NormalizedActor(actor, obs_norm['mean'], obs_norm['std'], obs_norm['clip'])
            else:
                wrapped_actor = actor
            data = collect_rollout(wrapped_actor, env)
            print(f"  Done in {time.time() - t0:.1f}s ({len(data['obs'])} steps)")

            print(f"Collecting zero-action baseline...")
            t0 = time.time()
            env2 = omnisafe_make("CityLearnSafety-V2G-v2")
            baseline_data = collect_zero_action_rollout(env2)
            print(f"  Done in {time.time() - t0:.1f}s")

            # Save rollout data
            np.savez_compressed(
                os.path.join(args.output_dir, "rollout_data.npz"),
                obs=data['obs'], actions=data['actions'],
                rewards=data['rewards'], costs=data['costs'],
                cost_components=data['cost_components'],
                reward_components=data['reward_components'],
            )
        else:
            data = None
            baseline_data = None

    if data is None:
        print("No rollout data available. Exiting.")
        sys.exit(1)

    # Compute value predictions if critics available
    v_reward, v_cost = None, None
    if v_r_critic is not None:
        obs_t = torch.as_tensor(data['obs'], dtype=torch.float32)
        if obs_norm:
            obs_t = (obs_t - torch.as_tensor(obs_norm['mean'])) / (torch.as_tensor(obs_norm['std']) + 1e-8)
            obs_t = obs_t.clamp(-obs_norm['clip'], obs_norm['clip'])
        with torch.no_grad():
            v_reward = v_r_critic(obs_t).squeeze(-1).numpy()
    if v_c_critic is not None:
        obs_t = torch.as_tensor(data['obs'], dtype=torch.float32)
        if obs_norm:
            obs_t = (obs_t - torch.as_tensor(obs_norm['mean'])) / (torch.as_tensor(obs_norm['std']) + 1e-8)
            obs_t = obs_t.clamp(-obs_norm['clip'], obs_norm['clip'])
        with torch.no_grad():
            v_cost = v_c_critic(obs_t).squeeze(-1).numpy()

    # Run all tests
    print(f"\nRunning diagnostic tests...")
    results = run_all_tests(
        data, baseline_data, actor=actor,
        v_reward=v_reward, v_cost=v_cost,
        checkpoint_path=args.checkpoint,
    )

    # Compute PHI + diagnosis
    phi = compute_phi(results)
    diagnosis = diagnose(results)

    print(f"\n{'=' * 50}")
    print(f"POLICY HEALTH INDEX (PHI): {phi:.3f}")
    print(f"")
    for key in sorted(results.keys()):
        status = results[key].get('status', '???')
        icon = {'healthy': 'PASS', 'warning': 'WARN', 'broken': 'FAIL', 'skipped': 'SKIP'}.get(status, '???')
        print(f"  [{icon}] {key}")
    print(f"")
    print(f"DIAGNOSIS: {diagnosis}")
    print(f"{'=' * 50}")

    # Generate report
    report_path = generate_report(results, phi, diagnosis, args.output_dir)
    print(f"\nReport saved to: {report_path}")

    # Save MI matrix
    mi_matrix = results.get('test2', {}).get('mi_matrix', None)
    if mi_matrix is not None:
        np.save(os.path.join(args.output_dir, "mi_matrix.npy"), mi_matrix)
```

**Step 4: Run integration test**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py::TestFullPipeline -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/diagnose_policy_health.py tests/test_diagnose_policy_health.py
git commit -m "feat: complete diagnostic pipeline — checkpoint loading, env setup, full report generation"
```

---

## Task 11: Run on R10b Checkpoint (Real Validation)

**Files:**
- No new files — runs existing script on real data

**Step 1: Run on R10b epoch 5 checkpoint (best available with diagnostics)**

Run:
```bash
cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
conda run -n citylearn python scripts/diagnose_policy_health.py \
    --checkpoint runs/r10b_stems_v3/r10b_5bld/PPOLag-\{CityLearnSafety-V2G-v2\}/seed-000-2026-03-08-20-41-14/torch_save/epoch-5.pt \
    --output-dir diagnostics/r10b_ep5/
```

Expected: Script runs all 8 tests, outputs report.md, figures, PHI score.

**Step 2: Review report**

Read: `diagnostics/r10b_ep5/report.md`
Check: PHI score, per-test status, diagnosis string.
Verify: The diagnosis matches what we already know (lambda death spiral, temporal attention working at 74% entropy).

**Step 3: Commit diagnostics output**

```bash
git add diagnostics/r10b_ep5/report.md diagnostics/r10b_ep5/phi_score.json
git commit -m "results: R10b epoch 5 diagnostic report — first real validation"
```

---

## Task 12: Run on R10c Checkpoint (Lambda-Capped Validation)

**Step 1: Run on R10c epoch 1 (lambda cap working)**

Run:
```bash
cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
conda run -n citylearn python scripts/diagnose_policy_health.py \
    --checkpoint runs/r10c_stems_v3/r10c_5bld/PPOLag-\{CityLearnSafety-V2G-v2\}/seed-000-2026-03-09-01-42-20/torch_save/epoch-1.pt \
    --output-dir diagnostics/r10c_ep1/
```

**Step 2: Compare R10b vs R10c reports**

Key comparisons:
- PHI score: should be similar (both early epochs)
- Headroom: R10c should NOT be worse than zero-action (lambda capped)
- Temporal attention: both should show learning
- Building correlation: check if differentiation improves with stable training

**Step 3: Commit**

```bash
git add diagnostics/r10c_ep1/report.md diagnostics/r10c_ep1/phi_score.json
git commit -m "results: R10c epoch 1 diagnostic — lambda cap validation"
```
