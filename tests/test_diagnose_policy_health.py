#!/usr/bin/env python3
"""
Tests for the Policy Health Diagnostic Suite scaffold.

Run:
  cd <project_root>
  conda run -n citylearn python -m pytest tests/test_diagnose_policy_health.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup (same pattern as the main script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# The diagnose_policy_health module is an advanced diagnostic tool that
# requires a trained checkpoint. Skip the entire test module if unavailable.
try:
    from scripts.diagnose_policy_health import (
        ACT_DIM,
        CURRENT_OBS_DIM,
        HISTORY_END,
        HISTORY_START,
        NUM_BUILDINGS,
        OBS_DIM,
        PRICE_IDX,
        SOC_INDICES,
        TEMPORAL_FEATURES_PER_STEP,
        TEMPORAL_WINDOW,
    )
except ImportError:
    pytest.skip(
        "scripts/diagnose_policy_health.py not available (advanced diagnostic, not part of core)",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def make_synthetic_rollout(
    T: int = 500,
    obs_dim: int = OBS_DIM,
    act_dim: int = ACT_DIM,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a synthetic rollout with structured correlations.

    The fake data has the following properties:
      - Battery actions (dims 0..4) are negatively correlated with the
        electricity price (obs[:, PRICE_IDX]): when the price is high the
        agent should discharge (negative action).
      - Battery actions also respond to SOC (obs[:, 23:28]): higher SOC
        leads to more discharge (negative action).
      - Remaining actions (dims 5..8) are noisy but weakly correlated with
        the hour-of-day signal.
      - Reward components have plausible signs (economic negative,
        stability negative, ramp negative, renewable positive).
      - Cost components are non-negative.

    Parameters
    ----------
    T : int
        Number of timesteps.
    obs_dim : int
        Observation dimensionality (default 330).
    act_dim : int
        Action dimensionality (default 9).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict of np.ndarray
        Same schema as collect_rollout output.
    """
    rng = np.random.RandomState(seed)

    # --- Observations ---
    obs = rng.randn(T, obs_dim).astype(np.float32)

    # Make the price signal oscillate (day/night pattern)
    hours = np.linspace(0, T / 24 * 2 * np.pi, T)
    price = 0.5 + 0.4 * np.sin(hours) + 0.1 * rng.randn(T)
    obs[:, PRICE_IDX] = price.astype(np.float32)

    # SOC starts at 0.5, drifts randomly but stays in [0, 1]
    soc = np.clip(
        0.5 + np.cumsum(0.01 * rng.randn(T, len(SOC_INDICES)), axis=0),
        0.0,
        1.0,
    ).astype(np.float32)
    obs[:, SOC_INDICES] = soc

    # Hour encoding
    hour_angle = 2 * np.pi * (np.arange(T) % 24) / 24.0
    obs[:, 4] = np.cos(hour_angle).astype(np.float32)  # HOUR_COS_IDX
    obs[:, 5] = np.sin(hour_angle).astype(np.float32)  # HOUR_SIN_IDX

    # --- Actions ---
    actions = np.zeros((T, act_dim), dtype=np.float32)

    # Battery actions (0..4): discharge when price high or SOC high
    for i in range(min(NUM_BUILDINGS, act_dim)):
        actions[:, i] = (
            -0.3 * price
            - 0.2 * soc[:, min(i, soc.shape[1] - 1)]
            + 0.05 * rng.randn(T)
        ).astype(np.float32)

    # Remaining actions: weak hour correlation + noise
    for i in range(NUM_BUILDINGS, act_dim):
        actions[:, i] = (
            0.1 * np.sin(hour_angle + i) + 0.1 * rng.randn(T)
        ).astype(np.float32)

    # Clip to [-1, 1]
    actions = np.clip(actions, -1.0, 1.0)

    # --- Rewards ---
    base_reward = -50.0 + 10.0 * rng.randn(T)
    rewards = base_reward.astype(np.float32)

    rew_economic = (-30.0 + 5.0 * rng.randn(T)).astype(np.float32)
    rew_stability_grid = (-10.0 + 3.0 * rng.randn(T)).astype(np.float32)
    rew_stability_building = (-5.0 + 2.0 * rng.randn(T)).astype(np.float32)
    rew_ramp = (-3.0 + 1.0 * rng.randn(T)).astype(np.float32)
    rew_renewable = (2.0 + 1.0 * rng.randn(T)).astype(np.float32)

    # --- Costs (non-negative) ---
    costs = np.abs(5.0 * rng.randn(T)).astype(np.float32)
    cost_c1 = np.abs(2.0 * rng.randn(T)).astype(np.float32)
    cost_c2 = np.abs(1.0 * rng.randn(T)).astype(np.float32)
    cost_c3 = np.abs(1.0 * rng.randn(T)).astype(np.float32)
    cost_c4 = np.abs(1.0 * rng.randn(T)).astype(np.float32)

    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "costs": costs,
        "reward_economic": rew_economic,
        "reward_stability_grid": rew_stability_grid,
        "reward_stability_building": rew_stability_building,
        "reward_ramp": rew_ramp,
        "reward_renewable": rew_renewable,
        "cost_C1": cost_c1,
        "cost_C2": cost_c2,
        "cost_C3": cost_c3,
        "cost_C4": cost_c4,
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestScaffold:
    """Verify the scaffold: synthetic data shapes, constants, and imports."""

    def test_synthetic_rollout_shapes(self) -> None:
        """make_synthetic_rollout returns arrays with correct shapes."""
        T = 500
        rollout = make_synthetic_rollout(T=T, obs_dim=OBS_DIM, act_dim=ACT_DIM)

        assert rollout["obs"].shape == (T, OBS_DIM), (
            f"obs shape {rollout['obs'].shape} != ({T}, {OBS_DIM})"
        )
        assert rollout["actions"].shape == (T, ACT_DIM), (
            f"actions shape {rollout['actions'].shape} != ({T}, {ACT_DIM})"
        )
        assert rollout["rewards"].shape == (T,), (
            f"rewards shape {rollout['rewards'].shape} != ({T},)"
        )
        assert rollout["costs"].shape == (T,), (
            f"costs shape {rollout['costs'].shape} != ({T},)"
        )

        # Reward components
        for key in [
            "reward_economic",
            "reward_stability_grid",
            "reward_stability_building",
            "reward_ramp",
            "reward_renewable",
        ]:
            assert rollout[key].shape == (T,), (
                f"{key} shape {rollout[key].shape} != ({T},)"
            )

        # Cost components
        for key in ["cost_C1", "cost_C2", "cost_C3", "cost_C4"]:
            assert rollout[key].shape == (T,), (
                f"{key} shape {rollout[key].shape} != ({T},)"
            )

    def test_synthetic_rollout_dtypes(self) -> None:
        """All arrays should be float32."""
        rollout = make_synthetic_rollout(T=100)
        for key, arr in rollout.items():
            assert arr.dtype == np.float32, (
                f"{key} dtype {arr.dtype} != float32"
            )

    def test_synthetic_rollout_action_range(self) -> None:
        """Actions should be clipped to [-1, 1]."""
        rollout = make_synthetic_rollout(T=500)
        assert rollout["actions"].min() >= -1.0, "Actions below -1.0"
        assert rollout["actions"].max() <= 1.0, "Actions above 1.0"

    def test_synthetic_rollout_costs_nonnegative(self) -> None:
        """Cost components should be non-negative."""
        rollout = make_synthetic_rollout(T=500)
        for key in ["costs", "cost_C1", "cost_C2", "cost_C3", "cost_C4"]:
            assert rollout[key].min() >= 0.0, f"{key} has negative values"

    def test_synthetic_rollout_price_action_correlation(self) -> None:
        """Battery actions should be negatively correlated with price."""
        rollout = make_synthetic_rollout(T=1000, seed=42)
        price = rollout["obs"][:, PRICE_IDX]
        for i in range(NUM_BUILDINGS):
            corr = np.corrcoef(price, rollout["actions"][:, i])[0, 1]
            assert corr < -0.1, (
                f"Battery action {i} has insufficient negative "
                f"correlation with price: {corr:.3f}"
            )

    def test_constants_consistency(self) -> None:
        """Verify constant relationships are self-consistent."""
        assert OBS_DIM == CURRENT_OBS_DIM + TEMPORAL_WINDOW * TEMPORAL_FEATURES_PER_STEP
        assert HISTORY_END - HISTORY_START == TEMPORAL_WINDOW * TEMPORAL_FEATURES_PER_STEP
        assert HISTORY_START == CURRENT_OBS_DIM
        assert HISTORY_END == OBS_DIM
        assert len(SOC_INDICES) == NUM_BUILDINGS

    def test_core_functions_importable(self) -> None:
        """compute_phi, diagnose, generate_report should be importable and callable."""
        from scripts.diagnose_policy_health import (
            compute_phi,
            diagnose,
            generate_report,
            run_all_tests,
        )
        assert callable(compute_phi)
        assert callable(diagnose)
        assert callable(generate_report)
        assert callable(run_all_tests)


# ---------------------------------------------------------------------------
# Task 2: Value Function tests
# ---------------------------------------------------------------------------
class TestValueFunction:
    def test_returns_expected_keys(self):
        data = make_synthetic_rollout(T=200)
        T = len(data['rewards'])
        returns_r = np.zeros(T, dtype=np.float32)
        returns_r[-1] = data['rewards'][-1]
        for t in range(T - 2, -1, -1):
            returns_r[t] = data['rewards'][t] + 0.99 * returns_r[t + 1]
        v_pred_r = returns_r + np.random.randn(T).astype(np.float32) * 0.5
        v_pred_c = np.cumsum(data['costs'][::-1])[::-1].astype(np.float32) * 0.01 + np.random.randn(T).astype(np.float32) * 0.5
        from scripts.diagnose_policy_health import test_value_function
        result = test_value_function(data, v_reward=v_pred_r, v_cost=v_pred_c, gamma=0.99)
        assert 'ev_reward' in result and 'ev_cost' in result and 'status' in result
        assert result['ev_reward'] > 0.3  # good critic

    def test_random_critic_is_broken(self):
        data = make_synthetic_rollout(T=200)
        T = len(data['rewards'])
        from scripts.diagnose_policy_health import test_value_function
        result = test_value_function(data, v_reward=np.random.randn(T).astype(np.float32)*100,
                                     v_cost=np.random.randn(T).astype(np.float32)*100, gamma=0.99)
        assert result['ev_reward'] < 0.1
        assert result['status'] == 'broken'


# ---------------------------------------------------------------------------
# Task 3: Mutual Information tests
# ---------------------------------------------------------------------------
class TestMutualInformation:
    def test_detects_correlated_features(self):
        from scripts.diagnose_policy_health import test_feature_action_mi
        data = make_synthetic_rollout(T=2000)
        result = test_feature_action_mi(data)
        assert result['mi_matrix'].shape == (330, 9)
        # Price should have higher MI with battery than EV actions
        assert np.mean(result['mi_matrix'][22, :5]) > np.mean(result['mi_matrix'][22, 5:8])
        assert 'mean_mi_top5' in result and 'status' in result

    def test_random_actions_low_mi(self):
        from scripts.diagnose_policy_health import test_feature_action_mi
        data = make_synthetic_rollout(T=2000)
        data['actions'] = np.clip(np.random.randn(2000, 9).astype(np.float32), -1, 1)
        result = test_feature_action_mi(data)
        assert result['mean_mi_top5'] < 0.1


# ---------------------------------------------------------------------------
# Task 3: Conditional Entropy tests
# ---------------------------------------------------------------------------
class TestConditionalEntropy:
    def test_returns_expected_structure(self):
        from scripts.diagnose_policy_health import test_conditional_entropy
        data = make_synthetic_rollout(T=2000)
        result = test_conditional_entropy(data)
        assert 'entropy_reduction' in result and isinstance(result['entropy_reduction'], dict)
        assert 'status' in result


# ---------------------------------------------------------------------------
# Task 4: Action Correlation tests
# ---------------------------------------------------------------------------
class TestActionCorrelation:
    def test_detects_differentiated_buildings(self):
        from scripts.diagnose_policy_health import test_action_correlation
        data = make_synthetic_rollout(T=1000)
        result = test_action_correlation(data)
        assert result['battery_corr_matrix'].shape == (5, 5)
        assert 'spatial_variance_ratio' in result
        assert 'acf' in result and 'status' in result

    def test_identical_actions_detected(self):
        from scripts.diagnose_policy_health import test_action_correlation
        data = make_synthetic_rollout(T=1000)
        for j in range(1, 5):
            data['actions'][:, j] = data['actions'][:, 0]
        result = test_action_correlation(data)
        assert result['mean_abs_corr'] > 0.95
        assert result['status'] == 'broken'


# ---------------------------------------------------------------------------
# Task 5: Gradient Attribution tests
# ---------------------------------------------------------------------------
class TestGradientAttribution:
    def _make_mock_actor(self):
        import torch.nn as nn

        class MockActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(330, 64), nn.ReLU(), nn.Linear(64, 9), nn.Tanh()
                )

            def forward(self, x):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                return self.net(x)

        return MockActor()

    def test_returns_pathway_fractions(self):
        from scripts.diagnose_policy_health import test_gradient_attribution
        data = make_synthetic_rollout(T=100)
        actor = self._make_mock_actor()
        result = test_gradient_attribution(data, actor)
        assert 'temporal_fraction' in result and 'current_fraction' in result
        assert 'price_gradient' in result and 'status' in result
        assert 0.0 <= result['temporal_fraction'] <= 1.0


# ---------------------------------------------------------------------------
# Task 6: Temporal Planning tests
# ---------------------------------------------------------------------------
class TestTemporalPlanning:
    def test_detects_forward_correlation(self):
        from scripts.diagnose_policy_health import test_temporal_planning
        np.random.seed(42)
        data = make_synthetic_rollout(T=2000)
        price = data['obs'][:, 22]  # PRICE_IDX
        for j in range(5):
            future_price = np.roll(price, -3)
            data['actions'][:, j] = np.clip(
                -0.5 * future_price + 0.3 * np.random.randn(2000), -1, 1
            ).astype(np.float32)
        result = test_temporal_planning(data)
        assert 'tps' in result and 'cross_temporal_corr' in result
        assert result['tps'] > 0.02

    def test_myopic_low_tps(self):
        from scripts.diagnose_policy_health import test_temporal_planning
        data = make_synthetic_rollout(T=2000)
        price = data['obs'][:, 22]
        for j in range(5):
            data['actions'][:, j] = np.clip(
                -0.5 * price + 0.5 * np.random.randn(2000), -1, 1
            ).astype(np.float32)
        result = test_temporal_planning(data)
        # Myopic agent still shows some TPS due to price autocorrelation,
        # but should be well below a truly forward-looking agent.
        assert result['tps'] < 0.20


# ---------------------------------------------------------------------------
# Task 7: Constraint Decomposition tests
# ---------------------------------------------------------------------------
class TestConstraintDecomposition:
    def test_computes_violation_metrics(self):
        from scripts.diagnose_policy_health import test_constraint_decomposition
        data = make_synthetic_rollout(T=500)
        result = test_constraint_decomposition(data)
        assert 'per_constraint' in result
        for c in ['C1', 'C2', 'C3', 'C4']:
            assert c in result['per_constraint']
            assert 0.0 <= result['per_constraint'][c]['violation_rate'] <= 1.0
        assert 'total_behavioral_vr' in result and 'status' in result


# ---------------------------------------------------------------------------
# Task 8: Headroom tests
# ---------------------------------------------------------------------------
class TestHeadroom:
    def test_computes_per_component(self):
        from scripts.diagnose_policy_health import test_headroom
        data = make_synthetic_rollout(T=500)
        baseline = make_synthetic_rollout(T=500)
        baseline['rewards'] = data['rewards'] + 2.0
        result = test_headroom(data, baseline)
        assert 'reward_headroom' in result and 'cost_headroom' in result
        for k in ['economic', 'stability_grid', 'stability_building', 'ramp', 'renewable']:
            assert k in result['reward_headroom']
        assert 'status' in result

    def test_worse_than_baseline_detected(self):
        from scripts.diagnose_policy_health import test_headroom
        data = make_synthetic_rollout(T=500)
        baseline = make_synthetic_rollout(T=500)
        data['rewards'] = baseline['rewards'] - 10.0
        result = test_headroom(data, baseline)
        assert result['total_reward_vs_baseline'] < 0
        assert result['status'] in ['broken', 'warning']


# ---------------------------------------------------------------------------
# Task 9: PHI Score, Diagnosis, Report Generation tests
# ---------------------------------------------------------------------------
class TestPHIAndDiagnosis:
    def test_phi_returns_0_to_1(self):
        from scripts.diagnose_policy_health import compute_phi
        results = {f'test{i}': {} for i in range(1, 9)}
        results['test1'] = {'ev_reward': 0.7, 'ev_cost': 0.6}
        results['test2'] = {'mean_mi_top5': 0.15}
        results['test3'] = {'mean_rho_top5': 0.1}
        results['test4'] = {'mean_abs_corr': 0.3, 'spatial_variance_ratio': 0.15}
        results['test5'] = {'temporal_fraction': 0.15, 'price_gradient': 0.02}
        results['test6'] = {'tps': 0.08, 'perturbation_effects': {'zero_history': 0.06}}
        results['test7'] = {'total_behavioral_vr': 0.2}
        phi = compute_phi(results)
        assert 0.0 <= phi <= 1.0 and phi > 0.3

    def test_phi_empty_results_returns_valid(self):
        from scripts.diagnose_policy_health import compute_phi
        results = {f'test{i}': {} for i in range(1, 9)}
        phi = compute_phi(results)
        assert 0.0 <= phi <= 1.0

    def test_broken_critic_diagnosed(self):
        from scripts.diagnose_policy_health import diagnose
        results = {f'test{i}': {} for i in range(1, 9)}
        results['test1'] = {'ev_reward': 0.05, 'ev_cost': 0.02}
        results['test8'] = {'total_reward_vs_baseline': -500}
        d = diagnose(results)
        assert 'CRITIC BROKEN' in d

    def test_gcn_dead_diagnosed(self):
        from scripts.diagnose_policy_health import diagnose
        results = {f'test{i}': {} for i in range(1, 9)}
        results['test1'] = {'ev_reward': 0.5, 'ev_cost': 0.5}
        results['test2'] = {'mean_mi_top5': 0.1}
        results['test4'] = {'mean_abs_corr': 0.95}
        d = diagnose(results)
        assert 'GCN DEAD' in d

    def test_architecture_works_diagnosed(self):
        from scripts.diagnose_policy_health import diagnose
        results = {f'test{i}': {} for i in range(1, 9)}
        results['test1'] = {'ev_reward': 0.7, 'ev_cost': 0.6}
        results['test2'] = {'mean_mi_top5': 0.15}
        results['test4'] = {'mean_abs_corr': 0.3}
        results['test5'] = {'temporal_fraction': 0.15, 'price_gradient': 0.02}
        results['test6'] = {'tps': 0.08}
        results['test7'] = {'total_behavioral_vr': 0.1}
        results['test8'] = {'total_reward_vs_baseline': 100}
        d = diagnose(results)
        assert 'ARCHITECTURE WORKS' in d

    def test_report_generation(self, tmp_path):
        from scripts.diagnose_policy_health import generate_report, compute_phi, diagnose
        results = {f'test{i}': {'status': 'healthy'} for i in range(1, 9)}
        results['test2']['mi_matrix'] = np.random.rand(330, 9).astype(np.float32)
        results['test4']['battery_corr_matrix'] = np.eye(5)
        results['test4']['acf'] = {j: np.random.rand(49) for j in range(5)}
        results['test6']['cross_temporal_corr'] = {t: 0.01 * t for t in range(-6, 25)}
        results['test6']['tps'] = 0.05
        results['test7']['per_constraint'] = {
            c: {'violation_rate': 0.1, 'violation_magnitude': 1.0, 'total_cost': 100,
                'hourly_violation_rate': [0.1] * 24, 'concentration_top10pct': 0.5}
            for c in ['C1', 'C2', 'C3', 'C4']
        }
        phi = 0.5
        report = generate_report(results, phi, "TEST DIAGNOSIS", str(tmp_path))
        assert os.path.exists(report)
        assert os.path.exists(os.path.join(str(tmp_path), 'figures', 'mi_heatmap.png'))
        assert os.path.exists(os.path.join(str(tmp_path), 'phi_score.json'))

    def test_report_contains_phi_and_diagnosis(self, tmp_path):
        from scripts.diagnose_policy_health import generate_report
        results = {f'test{i}': {'status': 'healthy'} for i in range(1, 9)}
        report_path = generate_report(results, 0.42, "TEST DIAG", str(tmp_path))
        with open(report_path) as f:
            content = f.read()
        assert '0.42' in content
        assert 'TEST DIAG' in content


# ---------------------------------------------------------------------------
# Task 10: Full Pipeline tests
# ---------------------------------------------------------------------------
class TestFullPipeline:
    def test_synthetic_pipeline(self, tmp_path):
        from scripts.diagnose_policy_health import run_all_tests, compute_phi, diagnose, generate_report
        data = make_synthetic_rollout(T=500)
        baseline = make_synthetic_rollout(T=500)
        results = run_all_tests(data, baseline, actor=None)
        assert len(results) == 8
        phi = compute_phi(results)
        assert 0.0 <= phi <= 1.0
        diag = diagnose(results)
        assert isinstance(diag, str) and len(diag) > 10
        report = generate_report(results, phi, diag, str(tmp_path))
        assert os.path.exists(report)

    def test_run_all_tests_returns_8_keys(self):
        from scripts.diagnose_policy_health import run_all_tests
        data = make_synthetic_rollout(T=200)
        results = run_all_tests(data, baseline_data=None, actor=None)
        for i in range(1, 9):
            assert f'test{i}' in results
            assert 'status' in results[f'test{i}']

    def test_run_all_tests_skips_gradient_without_actor(self):
        from scripts.diagnose_policy_health import run_all_tests
        data = make_synthetic_rollout(T=200)
        results = run_all_tests(data, baseline_data=None, actor=None)
        assert results['test5']['status'] == 'skipped'
