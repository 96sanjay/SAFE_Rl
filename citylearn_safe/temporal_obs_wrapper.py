"""
temporal_obs_wrapper.py -- Selective Temporal History Observation Wrapper
========================================================================

Appends a sliding window of *selected* observation features to the obs vector.
Unlike SpatialTemporalHistoryWrapper (which tracks the entire obs and computes
summary statistics), this wrapper:

  1. Tracks only user-specified indices (e.g., SoC, price, load) -- keeps the
     augmented obs small and focused on features that actually benefit from
     temporal context.
  2. Appends raw history values in chronological order (oldest first):
       [original_obs | hist_t-T | hist_t-T+1 | ... | hist_t-1]
     Note: t-1 is the PREVIOUS timestep, not the current one (current values
     are already in original_obs).
  3. Fills with zeros on reset -- no warm-up artifacts.

This is PPO-compatible because all temporal information is embedded in the
observation vector itself. No hidden state, no recurrence, works with
mini-batch shuffling during updates.

Augmented obs size = original_obs_dim + len(history_indices) * window_size
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import List, Sequence


class TemporalHistoryWrapper(gym.ObservationWrapper):
    """
    Gymnasium ObservationWrapper that appends a sliding window of selected
    observation features to the flat observation vector.

    Args:
        env:              The environment to wrap.
        history_indices:  List of integer indices into the flat obs vector
                          specifying which features to track over time.
        window_size:      Number of past timesteps to retain (default 12).
    """

    def __init__(
        self,
        env: gym.Env,
        history_indices: Sequence[int],
        window_size: int = 12,
    ):
        super().__init__(env)

        if len(history_indices) == 0:
            raise ValueError("history_indices must be non-empty.")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")

        self.history_indices = list(history_indices)
        self.window_size = window_size
        self.n_tracked = len(self.history_indices)
        self.n_extra = self.n_tracked * self.window_size

        # Resolve base observation space bounds
        orig = env.observation_space
        if isinstance(orig, spaces.Box):
            lo = orig.low.ravel()
            hi = orig.high.ravel()
        else:
            first = orig[0] if isinstance(orig, list) else orig
            lo = np.asarray(first.low).ravel()
            hi = np.asarray(first.high).ravel()

        self._base_obs_dim = len(lo)

        # Validate that all history_indices are within bounds
        max_idx = max(self.history_indices)
        if max_idx >= self._base_obs_dim:
            raise ValueError(
                f"history_indices contains index {max_idx} but obs dim is "
                f"{self._base_obs_dim}. Indices must be in [0, {self._base_obs_dim - 1}]."
            )
        if min(self.history_indices) < 0:
            raise ValueError("history_indices must contain non-negative integers.")

        # Build expanded observation space
        # History features are bounded generously since they mirror obs values
        hist_lo = np.tile(lo[self.history_indices], self.window_size)
        hist_hi = np.tile(hi[self.history_indices], self.window_size)

        self.observation_space = spaces.Box(
            low=np.concatenate([lo, hist_lo]).astype(np.float32),
            high=np.concatenate([hi, hist_hi]).astype(np.float32),
            dtype=np.float32,
        )

        # History buffer: each entry is a 1-D array of shape (n_tracked,)
        self._history: deque = deque(maxlen=window_size)

        print(
            f"[TemporalHistoryObs] tracking {self.n_tracked} features, "
            f"window={self.window_size}, "
            f"{self._base_obs_dim} + {self.n_extra} = "
            f"{self._base_obs_dim + self.n_extra} dims"
        )

    def _flatten_obs(self, obs) -> np.ndarray:
        """Convert any obs format to a flat float32 array."""
        if isinstance(obs, (list, tuple)):
            return np.concatenate(
                [np.asarray(o, dtype=np.float32).ravel() for o in obs]
            )
        return np.asarray(obs, dtype=np.float32).ravel()

    def _get_history_vector(self) -> np.ndarray:
        """
        Flatten the history deque into a 1-D array of shape (n_extra,).

        Layout: [tracked_features_t-T, tracked_features_t-T+1, ..., tracked_features_t-1]
        If the deque has fewer than window_size entries, the oldest slots are zeros.
        """
        if len(self._history) == 0:
            return np.zeros(self.n_extra, dtype=np.float32)

        # Stack available history entries: shape (len(deque), n_tracked)
        available = np.stack(list(self._history), axis=0)

        # If deque is not full yet, pad with zeros at the front (oldest slots)
        if len(self._history) < self.window_size:
            padding = np.zeros(
                (self.window_size - len(self._history), self.n_tracked),
                dtype=np.float32,
            )
            available = np.concatenate([padding, available], axis=0)

        return available.ravel().astype(np.float32)

    def observation(self, obs):
        flat = self._flatten_obs(obs)

        # Extract tracked features from the CURRENT observation
        tracked = flat[self.history_indices].copy()

        # Build augmented obs BEFORE pushing current to history
        # (current timestep's values are already in flat, so history contains t-T..t-1)
        hist_vec = self._get_history_vector()
        augmented = np.concatenate([flat, hist_vec]).astype(np.float32)

        # Push current tracked features into history for future steps
        self._history.append(tracked)

        return augmented

    def reset(self, **kwargs):
        self._history.clear()
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# =============================================================================
# Rich temporal history config builder (for STEMS V3 with expanded features)
# =============================================================================

def build_rich_history_config(obs_index, num_buildings):
    """
    Build expanded temporal history config for STEMS V3.

    Tracks per-building: battery_soc, net_consumption, solar, non_shiftable_load
    Tracks per-EV charger: soc, departure_time, connected_state
    Tracks global: electricity_pricing, hour_cos, hour_sin

    Returns dict with:
        history_indices:     flat list of obs indices to track (len = features_per_step)
        per_node_map:        [N, features_per_node] within-step indices for each building
        history_ev_mask:     [N, features_per_node] mask (0.0 for dummy EV slots)
        features_per_node:   int (10)
        features_per_step:   int (total tracked features per timestep)
    """
    import re

    N = num_buildings
    indices = []

    # Per-building features (4 per building) -- contiguous blocks of N
    # Block 0: battery_soc[0..N-1]
    for i in range(N):
        indices.append(obs_index.electrical_storage_soc[i])
    # Block 1: net_consumption[0..N-1]
    for i in range(N):
        indices.append(obs_index.net_electricity_consumption[i])
    # Block 2: solar_generation[0..N-1]
    for i in range(N):
        indices.append(obs_index.solar_generation[i])
    # Block 3: non_shiftable_load[0..N-1]
    for i in range(N):
        indices.append(obs_index.non_shiftable_load[i])

    # EV features: soc, departure_time, connected_state per charger
    # Map charger_id -> building index
    charger_to_bld = {}
    for charger_id in sorted(obs_index.ev.keys()):
        match = re.match(r"charger_(\d+)_(\d+)", charger_id)
        if match:
            bld_idx = int(match.group(1)) - 1
            if bld_idx < N:
                charger_to_bld[charger_id] = bld_idx

    ev_feature_names = ["soc", "departure_time", "connected_state"]
    bld_to_ev_offsets = {}  # bld_idx -> [offset_soc, offset_dep, offset_conn]
    for charger_id, bld_idx in sorted(charger_to_bld.items(), key=lambda x: x[1]):
        if bld_idx not in bld_to_ev_offsets:  # first charger per building
            offsets = []
            for feat in ev_feature_names:
                offsets.append(len(indices))
                indices.append(obs_index.ev[charger_id][feat])
            bld_to_ev_offsets[bld_idx] = offsets

    # Global features
    price_offset = len(indices)
    indices.append(obs_index.electricity_pricing)
    hour_cos_offset = len(indices)
    indices.append(obs_index.hour_cos)
    hour_sin_offset = len(indices)
    indices.append(obs_index.hour_sin)

    features_per_node = 4 + 3 + 3  # building(4) + ev(3) + global(3) = 10

    per_node_map = []
    ev_mask = []
    for i in range(N):
        # Building features: soc_i, net_i, solar_i, load_i
        node_idx = [i, N + i, 2 * N + i, 3 * N + i]
        node_mask = [1.0, 1.0, 1.0, 1.0]

        # EV features (dummy index 0 + mask=0 for buildings without charger)
        if i in bld_to_ev_offsets:
            node_idx.extend(bld_to_ev_offsets[i])
            node_mask.extend([1.0, 1.0, 1.0])
        else:
            node_idx.extend([0, 0, 0])  # dummy, will be masked
            node_mask.extend([0.0, 0.0, 0.0])

        # Global features (shared across all nodes)
        node_idx.extend([price_offset, hour_cos_offset, hour_sin_offset])
        node_mask.extend([1.0, 1.0, 1.0])

        per_node_map.append(node_idx)
        ev_mask.append(node_mask)

    return {
        'history_indices': indices,
        'per_node_map': per_node_map,
        'history_ev_mask': ev_mask,
        'features_per_node': features_per_node,
        'features_per_step': len(indices),
    }


def build_basic_history_indices(obs_index, num_buildings):
    """Original 11-feature history: battery_soc + net_consumption + price.

    Layout is CONTIGUOUS BLOCKS (not interleaved):
        [soc_0, soc_1, ..., soc_N-1, net_0, net_1, ..., net_N-1, price]
    This matches the encoder's _build_default_history_mapping which expects
    building i's features at within-step indices [i, N+i, 2*N].
    """
    indices = []
    for i in range(num_buildings):
        indices.append(obs_index.electrical_storage_soc[i])
    for i in range(num_buildings):
        indices.append(obs_index.net_electricity_consumption[i])
    indices.append(obs_index.electricity_pricing)
    return indices


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TemporalHistoryWrapper -- Smoke Test")
    print("=" * 70)

    OBS_DIM = 10

    class DummyEnv(gym.Env):
        """Minimal env with deterministic observations for testing."""

        def __init__(self, obs_dim: int = OBS_DIM):
            super().__init__()
            self.obs_dim = obs_dim
            self.observation_space = spaces.Box(
                low=-np.ones(obs_dim, dtype=np.float32) * 10.0,
                high=np.ones(obs_dim, dtype=np.float32) * 10.0,
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(2)
            self._step_count = 0

        def reset(self, **kwargs):
            self._step_count = 0
            # Return obs where feature i = 0.0 (so we can verify zero-init)
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            return obs, {}

        def step(self, action):
            self._step_count += 1
            # Deterministic obs: feature i = step_count * (i + 1)
            obs = np.array(
                [self._step_count * (i + 1) for i in range(self.obs_dim)],
                dtype=np.float32,
            )
            terminated = self._step_count >= 20
            return obs, 0.0, terminated, False, {}

    # -- Setup --
    history_indices = [0, 1, 2]
    window_size = 3
    n_tracked = len(history_indices)
    expected_obs_dim = OBS_DIM + n_tracked * window_size  # 10 + 9 = 19

    env = TemporalHistoryWrapper(
        DummyEnv(), history_indices=history_indices, window_size=window_size
    )

    errors = []

    # -- Test 1: Observation space shape --
    print(f"\nTest 1: Observation space shape")
    actual_shape = env.observation_space.shape
    print(f"  Expected: ({expected_obs_dim},)")
    print(f"  Actual:   {actual_shape}")
    if actual_shape != (expected_obs_dim,):
        errors.append(
            f"Shape mismatch: expected ({expected_obs_dim},), got {actual_shape}"
        )
    else:
        print("  PASS")

    # -- Test 2: Reset returns correct shape, history is zeros --
    print(f"\nTest 2: Reset -- obs shape and history zeros")
    obs, info = env.reset()
    print(f"  obs shape: {obs.shape}")
    print(f"  obs[:10] (base):    {obs[:OBS_DIM]}")
    print(f"  obs[10:] (history): {obs[OBS_DIM:]}")

    if obs.shape != (expected_obs_dim,):
        errors.append(f"Reset obs shape: expected ({expected_obs_dim},), got {obs.shape}")
    elif not np.allclose(obs[OBS_DIM:], 0.0):
        errors.append(
            f"History should be zeros after reset, got {obs[OBS_DIM:]}"
        )
    else:
        print("  PASS (history is all zeros after reset)")

    # -- Test 3: Step 1 -- history should still reflect reset's obs (all zeros)
    print(f"\nTest 3: After step 1 -- history reflects previous obs (zeros from reset)")
    obs, _, _, _, _ = env.step(0)
    base = obs[:OBS_DIM]
    hist = obs[OBS_DIM:]
    print(f"  base obs: {base}")
    print(f"  history:  {hist}")
    # After step 1:
    #   current obs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #   history deque has 1 entry (from reset): [0, 0, 0]
    #   history vector = [0,0,0 | 0,0,0 | 0,0,0]  (2 padding slots + 1 entry)
    expected_hist = np.zeros(n_tracked * window_size, dtype=np.float32)
    if not np.allclose(hist, expected_hist):
        errors.append(
            f"Step 1 history mismatch:\n  expected {expected_hist}\n  got      {hist}"
        )
    else:
        print("  PASS (history is [0,0,0 | 0,0,0 | 0,0,0])")

    # -- Test 4: Step 2 -- history should contain reset obs + step1 obs
    print(f"\nTest 4: After step 2 -- history contains reset and step-1 obs")
    obs, _, _, _, _ = env.step(0)
    base = obs[:OBS_DIM]
    hist = obs[OBS_DIM:]
    print(f"  base obs: {base}")
    print(f"  history:  {hist}")
    # After step 2:
    #   current obs = [2, 4, 6, ...]
    #   history deque has 2 entries: [0,0,0] from reset, [1,2,3] from step 1
    #   history vector = [0,0,0 | 0,0,0 | 1,2,3]  (1 padding + 2 entries)
    expected_hist = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3], dtype=np.float32)
    if not np.allclose(hist, expected_hist):
        errors.append(
            f"Step 2 history mismatch:\n  expected {expected_hist}\n  got      {hist}"
        )
    else:
        print("  PASS (history is [0,0,0 | 0,0,0 | 1,2,3])")

    # -- Test 5: Step 3 -- history deque is now full (3 entries)
    print(f"\nTest 5: After step 3 -- full history window")
    obs, _, _, _, _ = env.step(0)
    base = obs[:OBS_DIM]
    hist = obs[OBS_DIM:]
    print(f"  base obs: {base}")
    print(f"  history:  {hist}")
    # After step 3:
    #   current obs = [3, 6, 9, ...]
    #   history deque has 3 entries: [0,0,0], [1,2,3], [2,4,6]
    #   history vector = [0,0,0 | 1,2,3 | 2,4,6]  (no padding, full window)
    expected_hist = np.array([0, 0, 0, 1, 2, 3, 2, 4, 6], dtype=np.float32)
    if not np.allclose(hist, expected_hist):
        errors.append(
            f"Step 3 history mismatch:\n  expected {expected_hist}\n  got      {hist}"
        )
    else:
        print("  PASS (history is [0,0,0 | 1,2,3 | 2,4,6])")

    # -- Test 6: Step 4 -- oldest entry should be evicted (deque rolls)
    print(f"\nTest 6: After step 4 -- deque rolls, oldest evicted")
    obs, _, _, _, _ = env.step(0)
    base = obs[:OBS_DIM]
    hist = obs[OBS_DIM:]
    print(f"  base obs: {base}")
    print(f"  history:  {hist}")
    # After step 4:
    #   current obs = [4, 8, 12, ...]
    #   history deque (maxlen=3): [1,2,3], [2,4,6], [3,6,9]
    #   (the [0,0,0] from reset was evicted)
    #   history vector = [1,2,3 | 2,4,6 | 3,6,9]
    expected_hist = np.array([1, 2, 3, 2, 4, 6, 3, 6, 9], dtype=np.float32)
    if not np.allclose(hist, expected_hist):
        errors.append(
            f"Step 4 history mismatch:\n  expected {expected_hist}\n  got      {hist}"
        )
    else:
        print("  PASS (oldest entry evicted, history is [1,2,3 | 2,4,6 | 3,6,9])")

    # -- Test 7: Validation -- bad indices should raise
    print(f"\nTest 7: Validation -- out-of-bounds index raises ValueError")
    try:
        bad_env = TemporalHistoryWrapper(
            DummyEnv(), history_indices=[0, 1, 99], window_size=3
        )
        errors.append("Should have raised ValueError for out-of-bounds index 99")
    except ValueError as e:
        print(f"  Caught expected error: {e}")
        print("  PASS")

    print(f"\nTest 8: Validation -- empty indices raises ValueError")
    try:
        bad_env = TemporalHistoryWrapper(
            DummyEnv(), history_indices=[], window_size=3
        )
        errors.append("Should have raised ValueError for empty history_indices")
    except ValueError as e:
        print(f"  Caught expected error: {e}")
        print("  PASS")

    # -- Summary --
    print("\n" + "=" * 70)
    if errors:
        print(f"FAILED -- {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("ALL TESTS PASSED")
    print("=" * 70)
