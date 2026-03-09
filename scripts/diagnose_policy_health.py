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
def test_value_function(
    rollout: Dict[str, np.ndarray], critic: Any
) -> Dict[str, Any]:
    """Test 1: Value function accuracy vs. Monte Carlo returns."""
    raise NotImplementedError("test_value_function — to be implemented in Task 2")


def test_feature_action_mi(rollout: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Test 2: Mutual information between key features and actions."""
    raise NotImplementedError("test_feature_action_mi — to be implemented in Task 3")


def test_conditional_entropy(rollout: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Test 3: Conditional entropy of actions given state context."""
    raise NotImplementedError(
        "test_conditional_entropy — to be implemented in Task 4"
    )


def test_action_correlation(rollout: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Test 4: Inter-building action correlation analysis."""
    raise NotImplementedError(
        "test_action_correlation — to be implemented in Task 5"
    )


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
