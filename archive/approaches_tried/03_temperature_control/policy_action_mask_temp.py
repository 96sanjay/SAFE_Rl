"""Temperature-specific policy-side action masking support.

This module is intentionally separate from :mod:`citylearn_safe.policy_action_mask`
so the temperature case study can evolve independently from the EV / battery
masking logic.

The support surface is deliberately small:
* a temperature-safe bounds provider for the 3-building LSTM schema
* the latent-Gaussian -> bounded-action transform used by masked PPO
* the inverse log-prob transform needed for PPO updates

The provider is strict about the schema layout because this code is meant for a
single, controlled case study rather than a generic CityLearn action masker.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


_EXPECTED_ACTION_ROLES = ("dhw_storage", "electrical_storage", "cooling_device")


def _unwrap_citylearn(env: Any):
    """Walk the wrapper chain to find the live CityLearn env."""
    inner = env
    for _ in range(20):
        if hasattr(inner, "buildings") and len(getattr(inner, "buildings", [])) > 0:
            return inner
        inner = getattr(inner, "_env", getattr(inner, "env", getattr(inner, "base", None)))
        if inner is None:
            break
    return None


def _flatten_action_names(action_names: Any) -> list[str]:
    """Return a flat list of action names from CityLearn."""
    if isinstance(action_names, list) and len(action_names) == 1 and isinstance(action_names[0], list):
        action_names = action_names[0]
    if action_names is None:
        return []
    return [str(name).strip() for name in list(action_names)]


def _canonical_role(name: str) -> str:
    """Map an action name to the case-study role name."""
    name_l = name.strip().lower()
    if "dhw_storage" in name_l:
        return "dhw_storage"
    if "electrical_storage" in name_l:
        return "electrical_storage"
    if "cooling_device" in name_l:
        return "cooling_device"
    return name_l


class CityLearnTempActionBoundsProvider:
    """Compute safe action bounds for the temperature case study.

    The provider mirrors the true action-space bounds exposed by the live
    3-building LSTM schema, while explicitly clamping cooling actions to the
    physically meaningful interval ``[0, 1]``.
    """

    def __init__(self, env: Any, strict: bool = True) -> None:
        self._env = env
        self._city = _unwrap_citylearn(env)
        if self._city is None:
            raise RuntimeError(
                "[TempPolicyMask] Cannot find a CityLearn env with .buildings in the wrapper chain",
            )

        action_space = getattr(env, "action_space", None)
        if action_space is None:
            action_space = getattr(self._city, "action_space", None)
        if action_space is None:
            raise RuntimeError("[TempPolicyMask] Environment does not expose an action space")

        if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
            raise RuntimeError("[TempPolicyMask] Action space must expose low/high bounds")

        self._action_low = np.asarray(action_space.low, dtype=np.float32).copy().reshape(-1)
        self._action_high = np.asarray(action_space.high, dtype=np.float32).copy().reshape(-1)
        if self._action_low.shape != self._action_high.shape:
            raise ValueError("[TempPolicyMask] Action-space low/high shapes do not match")

        self._action_names = _flatten_action_names(getattr(self._city, "action_names", None))
        if not self._action_names:
            raise RuntimeError("[TempPolicyMask] CityLearn action_names are missing")
        if len(self._action_names) != int(self._action_low.shape[0]):
            raise ValueError(
                "[TempPolicyMask] action_names length does not match action-space dimensionality",
            )

        self._strict = bool(strict)
        self._roles = [_canonical_role(name) for name in self._action_names]
        self._cooling_action_indices = [
            idx for idx, role in enumerate(self._roles) if role == "cooling_device"
        ]
        self._comfort_tmin = float(
            getattr(env, "comfort_tmin", getattr(env, "tmin", 20.0)),
        )
        self._comfort_tmax = float(
            getattr(env, "comfort_tmax", getattr(env, "tmax", 26.0)),
        )
        self._warmup_steps = int(
            getattr(env, "_lstm_warmup_steps", getattr(env, "lstm_warmup_steps", 13)),
        )
        # Heuristic temperature-aware cooling bands. These are intentionally
        # configurable because they shape the feasible interval seen by PPO.
        self._cooling_disable_margin = float(
            getattr(env, "policy_mask_cooling_disable_margin", 0.0),
        )
        self._cooling_full_range = float(
            getattr(env, "policy_mask_cooling_full_range", 6.0),
        )
        self._cooling_force_range = float(
            getattr(env, "policy_mask_cooling_force_range", 4.0),
        )
        self._cooling_setpoint_deadband = float(
            getattr(env, "policy_mask_cooling_setpoint_deadband", 0.5),
        )
        self._cooling_min_cap = float(
            getattr(env, "policy_mask_cooling_min_cap", 0.6),
        )
        self._cooling_force_trigger = float(
            getattr(env, "policy_mask_cooling_force_trigger", 28.0),
        )
        self._cooling_rbc_divisor = float(
            getattr(env, "policy_mask_cooling_rbc_divisor", 3.0),
        )
        self._freeze_dhw_storage = bool(
            int(getattr(env, "policy_mask_freeze_dhw_storage", 1)),
        )
        self._battery_follow_rbc = bool(
            int(getattr(env, "policy_mask_battery_follow_rbc", 1)),
        )
        self._validate_layout()

    def _validate_layout(self) -> None:
        """Validate the case-study layout and fail fast on mismatches."""
        if len(self._roles) % len(_EXPECTED_ACTION_ROLES) != 0:
            raise ValueError(
                f"[TempPolicyMask] Expected a triplet action layout, got {len(self._roles)} dims",
            )

        for offset in range(0, len(self._roles), len(_EXPECTED_ACTION_ROLES)):
            block = tuple(self._roles[offset : offset + len(_EXPECTED_ACTION_ROLES)])
            if block != _EXPECTED_ACTION_ROLES:
                if self._strict:
                    raise ValueError(
                        "[TempPolicyMask] Unexpected action layout block at "
                        f"indices {offset}:{offset + len(_EXPECTED_ACTION_ROLES)}: {block}",
                    )

        cooling_count = sum(role == "cooling_device" for role in self._roles)
        if cooling_count == 0:
            raise ValueError("[TempPolicyMask] No cooling_device actions found in the schema")

    def _state_time_index(self) -> int:
        """Return the current 0-based CityLearn timestep index."""
        return max(0, int(getattr(self._city, "time_step", 0)) - 1)

    def _as_scalar_at(self, x: Any, t_idx: int) -> float | None:
        """Convert a scalar or time series to a finite float at timestep ``t_idx``."""
        if x is None:
            return None
        try:
            if np.isscalar(x):
                value = float(x)
                return value if np.isfinite(value) else None
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 0:
                value = float(arr)
                return value if np.isfinite(value) else None
            if len(arr) <= t_idx:
                return None
            value = float(arr[t_idx])
            return value if np.isfinite(value) else None
        except Exception:
            return None

    def _cooling_state(self, building_idx: int) -> tuple[float | None, float | None]:
        """Return ``(Tin, Tset_cool)`` for the specified building."""
        try:
            building = self._city.buildings[building_idx]
        except Exception:
            return None, None

        t_idx = self._state_time_index()
        es = getattr(building, "energy_simulation", None)
        tin = self._as_scalar_at(
            getattr(es, "indoor_dry_bulb_temperature", None) if es is not None else None,
            t_idx,
        )
        tset = self._as_scalar_at(
            getattr(es, "indoor_dry_bulb_temperature_cooling_set_point", None)
            if es is not None
            else None,
            t_idx,
        )

        if tin is not None:
            return tin, tset

        try:
            data = building._get_observations_data()
        except Exception:
            data = {}

        tin = self._as_scalar_at(data.get("indoor_dry_bulb_temperature"), t_idx)
        tset = self._as_scalar_at(
            data.get("indoor_dry_bulb_temperature_cooling_set_point"),
            t_idx,
        )
        return tin, tset

    def _cooling_bounds_for_building(self, building_idx: int) -> tuple[float, float]:
        """Return temperature-aware ``(safe_min, safe_max)`` for cooling action."""
        low = 0.0
        high = 1.0
        if self._state_time_index() < self._warmup_steps:
            return low, high

        tin, tset = self._cooling_state(building_idx)
        if tin is None:
            return low, high

        disable_temp = self._comfort_tmin + self._cooling_disable_margin
        if tset is not None:
            disable_temp = max(disable_temp, float(tset) + self._cooling_setpoint_deadband)

        if tin <= disable_temp:
            # Never block cooling — let the learned policy cool preemptively
            # if the cost critic says a violation is coming.
            return 0.0, 1.0

        force_start = max(self._cooling_force_trigger, disable_temp)
        # Keep the upper bound at least as permissive as the RBC thermostat
        # response. The previous 6.0 divisor systematically capped cooling below
        # the controller's proportional action on hot steps.
        rbc_like_max = max(0.0, (tin - disable_temp) / max(self._cooling_rbc_divisor, 1e-6))
        cool_max = min(
            1.0,
            max(
                0.0,
                max(
                    (tin - disable_temp) / max(self._cooling_full_range, 1e-6),
                    rbc_like_max,
                ),
            ),
        )

        if tin <= force_start:
            cool_min = 0.0
        else:
            cool_min = min(
                self._cooling_min_cap,
                max(0.0, (tin - force_start) / max(self._cooling_force_range, 1e-6)),
            )

        return cool_min, max(cool_min, cool_max)

    def _hour_of_day(self) -> int:
        """Return the current hour-of-day for schedule-based storage control."""
        return self._state_time_index() % 24

    def _battery_bounds(self) -> tuple[float, float] | None:
        """Return an RBC-like battery action interval for the current hour."""
        if not self._battery_follow_rbc:
            return None

        hour = self._hour_of_day()
        if 10 <= hour <= 16:
            return 0.8, 0.8
        if 17 <= hour <= 21:
            return -0.6, -0.6
        return 0.0, 0.0

    def _storage_bounds_for_role(self, role: str) -> tuple[float, float] | None:
        """Return a temperature-study-specific storage interval for a role."""
        if role == "dhw_storage" and self._freeze_dhw_storage:
            return 0.0, 0.0
        if role == "electrical_storage":
            return self._battery_bounds()
        return None

    def current_safe_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-dimension safe bounds for the live temperature schema."""
        safe_min = self._action_low.copy()
        safe_max = self._action_high.copy()

        for idx, role in enumerate(self._roles):
            if role != "cooling_device":
                storage_bounds = self._storage_bounds_for_role(role)
                if storage_bounds is None:
                    safe_min[idx] = float(self._action_low[idx])
                    safe_max[idx] = float(self._action_high[idx])
                else:
                    s_min, s_max = storage_bounds
                    safe_min[idx] = max(float(self._action_low[idx]), float(s_min))
                    safe_max[idx] = min(float(self._action_high[idx]), float(s_max))

        for building_idx, act_idx in enumerate(self._cooling_action_indices):
            cooling_min, cooling_max = self._cooling_bounds_for_building(building_idx)
            safe_min[act_idx] = max(float(self._action_low[act_idx]), cooling_min)
            safe_max[act_idx] = min(float(self._action_high[act_idx]), cooling_max)
            safe_max[act_idx] = max(float(safe_min[act_idx]), float(safe_max[act_idx]))

        if np.any(~np.isfinite(safe_min)) or np.any(~np.isfinite(safe_max)):
            raise ValueError("[TempPolicyMask] Non-finite safe bounds computed")
        if np.any(safe_max < safe_min):
            raise ValueError("[TempPolicyMask] Invalid safe bounds: max < min")

        return safe_min.astype(np.float32, copy=False), safe_max.astype(np.float32, copy=False)


def masked_action_from_pretanh(
    pre_tanh: torch.Tensor,
    safe_min: torch.Tensor,
    safe_max: torch.Tensor,
) -> torch.Tensor:
    """Map latent Gaussian samples into the current safe action interval."""
    z = torch.tanh(pre_tanh)
    out = torch.zeros_like(z)
    eps = 1e-6
    collapsed = (safe_max - safe_min) <= eps
    pos_only = (safe_min >= 0.0) & (safe_max > safe_min + eps)
    neg_only = (safe_max <= 0.0) & (safe_max > safe_min + eps)
    cross_zero = ~(collapsed | pos_only | neg_only)

    affine = safe_min + 0.5 * (z + 1.0) * (safe_max - safe_min)
    out = torch.where(pos_only | neg_only, affine, out)

    neg = z < 0
    pos = z > 0
    cross_val = torch.zeros_like(z)
    cross_val = torch.where(neg, z * torch.abs(torch.clamp_max(safe_min, 0.0)), cross_val)
    cross_val = torch.where(pos, z * torch.clamp_min(safe_max, 0.0), cross_val)
    out = torch.where(cross_zero, cross_val, out)
    out = torch.where(collapsed, safe_min, out)
    return out


def masked_log_prob_from_action(
    dist: torch.distributions.Normal,
    action: torch.Tensor,
    safe_min: torch.Tensor,
    safe_max: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the masked-action log-probability with Jacobian correction."""
    active = (safe_max - safe_min) > eps
    pos_only = (safe_min >= 0.0) & active
    neg_only = (safe_max <= 0.0) & active
    same_sign = pos_only | neg_only
    cross_zero = active & ~same_sign

    z = torch.zeros_like(action)

    same_span = torch.clamp(safe_max - safe_min, min=eps)
    z_same = 2.0 * (action - safe_min) / same_span - 1.0
    z = torch.where(same_sign, z_same, z)

    scale_neg = torch.clamp(torch.abs(torch.clamp_max(safe_min, 0.0)), min=eps)
    scale_pos = torch.clamp(torch.clamp_min(safe_max, 0.0), min=eps)
    neg = action < 0
    pos = action > 0
    z_cross = torch.zeros_like(action)
    z_cross = torch.where(neg & cross_zero, action / scale_neg, z_cross)
    z_cross = torch.where(pos & cross_zero, action / scale_pos, z_cross)
    z = torch.where(cross_zero, z_cross, z)
    z = torch.clamp(z, -1.0 + eps, 1.0 - eps)

    pre_tanh = 0.5 * (torch.log1p(z) - torch.log1p(-z))
    logp_u = dist.log_prob(pre_tanh)
    log_det_tanh = torch.log(torch.clamp(1.0 - z.pow(2), min=eps))

    log_det_mask = torch.zeros_like(action)
    log_det_mask = torch.where(same_sign, torch.log(0.5 * same_span), log_det_mask)

    cross_scale = torch.ones_like(action)
    cross_scale = torch.where(neg & cross_zero, scale_neg, cross_scale)
    cross_scale = torch.where(pos & cross_zero, scale_pos, cross_scale)
    log_det_mask = torch.where(cross_zero, torch.log(torch.clamp(cross_scale, min=eps)), log_det_mask)

    return (logp_u - log_det_tanh - log_det_mask).sum(axis=-1)


def build_policy_action_bounds_provider(env: Any) -> CityLearnTempActionBoundsProvider:
    """Build the temperature action-bounds provider from an env or wrapper."""
    for hook_name in (
        "make_policy_action_bounds_provider",
        "get_policy_action_bounds_provider",
        "policy_action_bounds_provider",
        "masked_action_bounds_provider",
    ):
        factory = getattr(env, hook_name, None)
        if callable(factory):
            provider = factory()
            if provider is None:
                raise RuntimeError("[TempPolicyMask] Env hook returned no bounds provider")
            return provider
    return CityLearnTempActionBoundsProvider(env)


__all__ = [
    "CityLearnTempActionBoundsProvider",
    "build_policy_action_bounds_provider",
    "masked_action_from_pretanh",
    "masked_log_prob_from_action",
]
