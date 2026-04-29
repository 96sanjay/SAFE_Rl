"""Structured action wrapper for building-level coupled budgets.

This wrapper replaces flat device-level control with a compact action space:

- one net controllable budget per building
- one EV-share parameter only for buildings that have both a battery and an EV
- raw passthrough controls for any remaining action dimensions

The wrapper decodes those structured actions into the original CityLearn action
vector expected by the environment. It uses the same CityLearn-specific state
inspection logic as the policy-side mask utilities, but does not rely on PPO-
side action transforms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from citylearn_safe.policy_action_mask import CityLearnActionBoundsProvider


@dataclass(frozen=True)
class _BuildingDecoderSpec:
    """Decoder metadata for a single building."""

    building_idx: int
    batt_act_idx: int | None
    ev_act_idx: int | None
    has_battery: bool
    has_ev: bool
    has_share: bool


class CoupledBudgetActionWrapper(gym.ActionWrapper):
    """Expose building-level budget actions instead of flat device controls."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._provider = CityLearnActionBoundsProvider(env)
        self._n_buildings = self._provider._n_buildings

        self._building_specs: list[_BuildingDecoderSpec] = []
        self._share_buildings: list[int] = []
        for b_idx in range(self._n_buildings):
            has_batt = b_idx in self._provider._building_batt_act
            has_ev = b_idx in self._provider._building_ev_act
            has_share = has_batt and has_ev
            if has_share:
                self._share_buildings.append(b_idx)
            self._building_specs.append(
                _BuildingDecoderSpec(
                    building_idx=b_idx,
                    batt_act_idx=self._provider._building_batt_act.get(b_idx),
                    ev_act_idx=self._provider._building_ev_act.get(b_idx),
                    has_battery=has_batt,
                    has_ev=has_ev,
                    has_share=has_share,
                ),
            )

        self._passthrough_indices = list(self._provider._passthrough_indices)
        self._budget_dim = self._n_buildings
        self._share_dim = len(self._share_buildings)
        self._passthrough_dim = len(self._passthrough_indices)
        self._action_dim = self._budget_dim + self._share_dim + self._passthrough_dim

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._action_dim,),
            dtype=np.float32,
        )

        self._share_offset = self._budget_dim
        self._passthrough_offset = self._budget_dim + self._share_dim
        self._share_lookup = {b_idx: i for i, b_idx in enumerate(self._share_buildings)}

        print(
            "[CoupledBudget] ENABLED "
            f"(budgets={self._budget_dim}, ev_shares={self._share_dim}, "
            f"passthrough={self._passthrough_dim}, total_dim={self._action_dim})",
        )

    @staticmethod
    def _spill_split(total_kw: float, primary_cap: float, secondary_cap: float, primary_share: float):
        """Allocate total power to two channels with spillover."""
        total_kw = max(0.0, float(total_kw))
        primary_cap = max(0.0, float(primary_cap))
        secondary_cap = max(0.0, float(secondary_cap))
        primary_share = float(np.clip(primary_share, 0.0, 1.0))

        primary = min(primary_cap, total_kw * primary_share)
        remaining = total_kw - primary
        secondary = min(secondary_cap, remaining)
        remaining -= secondary
        if remaining > 1e-9 and primary < primary_cap:
            extra = min(primary_cap - primary, remaining)
            primary += extra
            remaining -= extra
        if remaining > 1e-9 and secondary < secondary_cap:
            extra = min(secondary_cap - secondary, remaining)
            secondary += extra
        return primary, secondary

    def action(self, act: np.ndarray) -> np.ndarray:
        """Decode structured action into the original flat CityLearn action vector."""
        act = np.asarray(act, dtype=np.float32).reshape(-1)
        if act.shape[0] != self._action_dim:
            raise ValueError(
                f"[CoupledBudget] Expected structured action dim {self._action_dim}, got {act.shape[0]}",
            )

        safe_min, safe_max = self._provider.current_safe_bounds()
        decoded = np.zeros(self.env.action_space.shape, dtype=np.float32)

        for i, action_idx in enumerate(self._passthrough_indices):
            decoded[action_idx] = float(np.clip(act[self._passthrough_offset + i], -1.0, 1.0))

        for spec in self._building_specs:
            b_idx = spec.building_idx
            budget = float(np.clip(act[b_idx], -1.0, 1.0))
            share = 0.0
            if spec.has_share:
                share_raw = float(np.clip(act[self._share_offset + self._share_lookup[b_idx]], -1.0, 1.0))
                share = 0.5 * (share_raw + 1.0)

            batt_charge_cap_kw = 0.0
            batt_discharge_cap_kw = 0.0
            batt_power = 0.0
            if spec.has_battery and spec.batt_act_idx is not None:
                batt_power = self._provider._batt_powers.get(b_idx, 0.0)
                batt_charge_cap_kw = max(0.0, float(safe_max[spec.batt_act_idx])) * batt_power
                batt_discharge_cap_kw = max(0.0, -float(safe_min[spec.batt_act_idx])) * batt_power

            ev_charge_floor_kw = 0.0
            ev_charge_cap_kw = 0.0
            ev_discharge_cap_kw = 0.0
            ev_max_charge = 0.0
            ev_max_discharge = 0.0
            if spec.has_ev and spec.ev_act_idx is not None:
                ev_max_charge = self._provider._ev_max_charge.get(b_idx, 0.0)
                ev_max_discharge = self._provider._ev_max_discharge.get(b_idx, 0.0)
                ev_min_norm = float(safe_min[spec.ev_act_idx])
                ev_max_norm = float(safe_max[spec.ev_act_idx])
                if ev_min_norm > 0.0 and ev_max_charge > 0.0:
                    ev_charge_floor_kw = ev_min_norm * ev_max_charge
                if ev_max_norm > 0.0 and ev_max_charge > 0.0:
                    ev_charge_cap_kw = ev_max_norm * ev_max_charge
                if ev_min_norm < 0.0 and ev_max_discharge > 0.0:
                    ev_discharge_cap_kw = -ev_min_norm * ev_max_discharge
                ev_charge_cap_kw = max(ev_charge_floor_kw, ev_charge_cap_kw)

            optional_charge_total_kw = batt_charge_cap_kw + max(0.0, ev_charge_cap_kw - ev_charge_floor_kw)
            discharge_total_kw = batt_discharge_cap_kw + ev_discharge_cap_kw

            batt_action = 0.0
            ev_action = 0.0

            if budget >= 0.0:
                total_charge_kw = ev_charge_floor_kw + budget * optional_charge_total_kw
                optional_charge_kw = max(0.0, total_charge_kw - ev_charge_floor_kw)
                ev_optional_cap = max(0.0, ev_charge_cap_kw - ev_charge_floor_kw)

                if spec.has_share:
                    ev_optional_kw, batt_charge_kw = self._spill_split(
                        optional_charge_kw,
                        ev_optional_cap,
                        batt_charge_cap_kw,
                        share,
                    )
                elif spec.has_ev and not spec.has_battery:
                    ev_optional_kw = min(ev_optional_cap, optional_charge_kw)
                    batt_charge_kw = 0.0
                else:
                    ev_optional_kw = 0.0
                    batt_charge_kw = min(batt_charge_cap_kw, optional_charge_kw)

                if spec.has_battery and batt_power > 0.0:
                    batt_action = batt_charge_kw / max(batt_power, 1e-6)
                if spec.has_ev and ev_max_charge > 0.0:
                    ev_action = (ev_charge_floor_kw + ev_optional_kw) / max(ev_max_charge, 1e-6)
            else:
                total_discharge_kw = (-budget) * discharge_total_kw
                if spec.has_share:
                    ev_discharge_kw, batt_discharge_kw = self._spill_split(
                        total_discharge_kw,
                        ev_discharge_cap_kw,
                        batt_discharge_cap_kw,
                        share,
                    )
                elif spec.has_ev and not spec.has_battery:
                    ev_discharge_kw = min(ev_discharge_cap_kw, total_discharge_kw)
                    batt_discharge_kw = 0.0
                else:
                    ev_discharge_kw = 0.0
                    batt_discharge_kw = min(batt_discharge_cap_kw, total_discharge_kw)

                if spec.has_battery and batt_power > 0.0:
                    batt_action = -batt_discharge_kw / max(batt_power, 1e-6)
                if spec.has_ev and ev_max_discharge > 0.0:
                    ev_action = -ev_discharge_kw / max(ev_max_discharge, 1e-6)

            if spec.has_battery and spec.batt_act_idx is not None:
                decoded[spec.batt_act_idx] = float(np.clip(batt_action, safe_min[spec.batt_act_idx], safe_max[spec.batt_act_idx]))
            if spec.has_ev and spec.ev_act_idx is not None:
                decoded[spec.ev_act_idx] = float(np.clip(ev_action, safe_min[spec.ev_act_idx], safe_max[spec.ev_act_idx]))

        return decoded
