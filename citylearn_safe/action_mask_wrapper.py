"""
Action masking wrapper for C2 (battery SoC), C3 (per-building power), C4 (grid aggregate).

Rescales policy actions from [-1, 1] to per-device safe bounds that satisfy:
    |NEC_b| < P_building_max   for each building b

where NEC_b = non_shiftable_load_b + solar_generation_b + battery_power_b + ev_power_b.
    sum(max(0, NEC_b)) < P_grid_max   grid aggregate import (C4)

The wrapper computes exogenous NEC (load + solar) each timestep, derives the
remaining headroom for controllable devices (battery, EV), and linearly maps
the policy's [-1, 1] output into the feasible range. This is RESCALING, not
clipping -- the full [-1, 1] range always maps onto the full safe range, so
policy gradients remain meaningful everywhere.

For buildings with both battery AND EV, headroom is split proportionally by
each device's nominal power rating. The EV charger's min_charging_power and
min_discharging_power are respected: if the headroom is less than the minimum,
the EV action is forced to zero.

Also enforces C2 (battery SoC bounds): prevents discharge below SOC_low and
charge above SOC_high via safe_min/safe_max overrides.

The action layout is discovered dynamically from CityLearn's action_names.

Gated by CITYLEARN_ACTION_MASK="1" env var. When disabled, pass through.

Env vars:
    CITYLEARN_ACTION_MASK         - "1" to enable (default "0")
    CITYLEARN_STEMS_P_BUILDING_MAX - per-building power limit kW (default 4.6083)
    CITYLEARN_STEMS_SOC_LOW       - battery SoC lower bound (default 0.0)
    CITYLEARN_STEMS_SOC_HIGH      - battery SoC upper bound (default 0.95)
    CITYLEARN_STEMS_P_GRID_MAX    - grid aggregate power limit kW (default 10.2352)
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


def _unwrap_citylearn(env):
    """Walk the wrapper chain to find the CityLearn env with .buildings."""
    inner = env
    for _ in range(20):
        if hasattr(inner, 'buildings') and len(getattr(inner, 'buildings', [])) > 0:
            return inner
        inner = getattr(inner, '_env',
                 getattr(inner, 'env',
                 getattr(inner, 'base', None)))
        if inner is None:
            break
    return None


_EV_RE = re.compile(
    r"^electric_vehicle_storage_charger_(?P<suffix>.+)$", re.IGNORECASE)


class ActionMaskWrapper(gym.Wrapper):
    """Rescales actions to satisfy per-building power constraints (C3) and
    battery SoC bounds (C2).

    Action layout is discovered dynamically from CityLearn action_names.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Beta actor mode: x ∈ (0,1) → affine transform to [safe_min, safe_max]
        self._beta_mode = os.environ.get("CITYLEARN_BETA_ACTOR", "0") == "1"

        self._p_bmax = float(os.environ.get(
            "CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
        self._soc_low = float(os.environ.get(
            "CITYLEARN_STEMS_SOC_LOW", "0.0"))
        self._soc_high = float(os.environ.get(
            "CITYLEARN_STEMS_SOC_HIGH", "0.95"))
        self._p_gmax = float(os.environ.get(
            "CITYLEARN_STEMS_P_GRID_MAX", "10.2352"))

        # SoC margin for C2 enforcement
        self._soc_margin = 0.02

        # Markgraf et al. 2025 Eq. 24: SE-RL penalty weight (cached, not per-step)
        self._penalty_w = float(os.environ.get("SE_RL_PENALTY_WEIGHT", "0.0"))

        # Discover CityLearn env and read device specs
        self._city = _unwrap_citylearn(env)
        if self._city is None:
            raise RuntimeError(
                "[ActionMask] Cannot find CityLearn env with .buildings "
                "in wrapper chain")

        self._n_buildings = len(self._city.buildings)

        # --- Discover action layout from action_names ---
        action_names = self._city.action_names
        if isinstance(action_names, list) and len(action_names) == 1 \
                and isinstance(action_names[0], list):
            action_names_flat = action_names[0]
        else:
            action_names_flat = list(action_names)

        self._n_actions = len(action_names_flat)

        # Parse action_names to build mapping
        self._building_batt_act = {}  # b_idx -> action_idx
        self._building_ev_act = {}    # b_idx -> action_idx
        self._passthrough_indices = []

        b_idx = 0
        batt_count_for_building = {}
        for act_idx, name in enumerate(action_names_flat):
            name_lower = name.lower()
            if _EV_RE.match(name):
                m = _EV_RE.match(name)
                suffix = m.group('suffix')
                parts = suffix.split('_')
                bldg_num = int(parts[0])
                ev_b_idx = bldg_num - 1
                self._building_ev_act[ev_b_idx] = act_idx
            elif 'washing_machine' in name_lower:
                self._passthrough_indices.append(act_idx)
            elif name_lower == 'electrical_storage':
                while b_idx in batt_count_for_building:
                    b_idx += 1
                self._building_batt_act[b_idx] = act_idx
                batt_count_for_building[b_idx] = True
                b_idx += 1
            else:
                self._passthrough_indices.append(act_idx)

        self._ev_building_indices = sorted(self._building_ev_act.keys())
        self._batt_act_indices = sorted(self._building_batt_act.values())
        self._ev_act_indices = [self._building_ev_act[b] for b in self._ev_building_indices]

        # Read battery nominal powers (kW)
        self._batt_powers = {}
        for bi in self._building_batt_act:
            b = self._city.buildings[bi]
            es = getattr(b, 'electrical_storage', None)
            if es is not None:
                p = getattr(es, 'nominal_power', None)
                self._batt_powers[bi] = float(p) if p and p > 0 else 5.0
            else:
                self._batt_powers[bi] = 5.0

        # Read EV charger powers (max and min for charge/discharge)
        self._ev_max_charge = {}    # b_idx -> kW
        self._ev_min_charge = {}    # b_idx -> kW
        self._ev_max_discharge = {} # b_idx -> kW
        self._ev_min_discharge = {} # b_idx -> kW
        for bi in self._ev_building_indices:
            b = self._city.buildings[bi]
            chargers = getattr(b, 'electric_vehicle_chargers',
                       getattr(b, 'chargers',
                       getattr(b, 'ev_chargers', [])))
            if chargers and len(chargers) > 0:
                c = chargers[0]
                self._ev_max_charge[bi] = float(getattr(c, 'max_charging_power', 0) or 0)
                self._ev_min_charge[bi] = float(getattr(c, 'min_charging_power', 0) or 0)
                self._ev_max_discharge[bi] = float(getattr(c, 'max_discharging_power', 0) or 0)
                self._ev_min_discharge[bi] = float(getattr(c, 'min_discharging_power', 0) or 0)
            else:
                self._ev_max_charge[bi] = 0
                self._ev_min_charge[bi] = 0
                self._ev_max_discharge[bi] = 0
                self._ev_min_discharge[bi] = 0

        # For backwards compatibility, keep _ev_powers as max_charging_power
        self._ev_powers = {bi: self._ev_max_charge[bi] for bi in self._ev_building_indices}

        # Diagnostics
        self._step_count = 0
        self._total_interventions = 0
        self._last_safe_min = np.full(self._n_actions, -1.0, dtype=np.float32)
        self._last_safe_max = np.full(self._n_actions, 1.0, dtype=np.float32)
        self._last_interventions = 0
        self._last_total_exo_import = 0.0

        print(f"[ActionMask] ENABLED: P_bmax={self._p_bmax:.2f} kW, "
              f"P_gmax={self._p_gmax:.2f} kW, "
              f"N_buildings={self._n_buildings}")
        print(f"[ActionMask] Action layout ({self._n_actions} dims):")
        print(f"  Battery actions: {self._building_batt_act}")
        print(f"  EV actions: {self._building_ev_act}")
        print(f"  Passthrough actions: {self._passthrough_indices}")
        print(f"[ActionMask] Battery powers: {self._batt_powers}")
        print(f"[ActionMask] EV max charge: {self._ev_max_charge}")
        print(f"[ActionMask] EV min charge: {self._ev_min_charge}")
        print(f"[ActionMask] EV max discharge: {self._ev_max_discharge}")
        print(f"[ActionMask] EV min discharge: {self._ev_min_discharge}")
        print(f"[ActionMask] SoC bounds: [{self._soc_low}, {self._soc_high}], "
              f"margin={self._soc_margin}")

    def reset(self, **kwargs):
        self._step_count = 0
        self._total_interventions = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32)
        safe_action = self._apply_mask(raw_action)

        # Markgraf et al. 2025 Eq. 24: penalty for action correction
        # h = 0 if u ∈ safe set, else w × ||u - clip(u, safe_min, safe_max)||²
        # We measure the CLIPPING distance (how much the raw action exceeds bounds),
        # NOT the rescaling distance. Rescaling maps [-1,1] to [safe_min, safe_max]
        # but doesn't indicate a safety violation. Only exceeding bounds does.
        if self._penalty_w > 0:
            clipped = np.clip(raw_action, self._last_safe_min, self._last_safe_max)
            delta = raw_action - clipped
            mask_penalty = self._penalty_w * float(np.sum(delta ** 2))
        else:
            mask_penalty = 0.0

        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        info["action_mask_enabled"] = 1.0
        info["action_mask_safe_min"] = self._last_safe_min.tolist()
        info["action_mask_safe_max"] = self._last_safe_max.tolist()
        info["action_mask_raw_action"] = raw_action.tolist()
        info["action_mask_interventions"] = float(self._last_interventions)
        info["mask_penalty"] = mask_penalty
        info["mask_delta_l2"] = float(np.sqrt(np.sum((raw_action - safe_action) ** 2)))

        batt_ranges = np.array([
            self._last_safe_max[ai] - self._last_safe_min[ai]
            for ai in self._batt_act_indices])
        ev_ranges = np.array([
            self._last_safe_max[ai] - self._last_safe_min[ai]
            for ai in self._ev_act_indices]) if self._ev_act_indices else np.array([0.0])
        info["action_mask_batt_range_mean"] = float(np.mean(batt_ranges))
        info["action_mask_ev_range_mean"] = float(np.mean(ev_ranges))
        self._step_count += 1

        return obs, reward, terminated, truncated, info

    def _get_exogenous_nec(self) -> list[float]:
        """Read exogenous NEC (non-shiftable load + solar) for each building.

        Uses energy_to_non_shiftable_load (thermal demand = electricity for
        this schema since HVAC is not actively controlled) plus solar_generation.
        """
        city = self._city
        t_idx = int(getattr(city, 'time_step', 0))

        exo_nec = []
        for b in city.buildings:
            nsl = getattr(b, '_Building__energy_to_non_shiftable_load', None)
            sg = getattr(b, '_Building__solar_generation', None)

            load_val = 0.0
            if nsl is not None and len(nsl) > t_idx:
                load_val = float(nsl[t_idx])

            solar_val = 0.0
            if sg is not None and len(sg) > t_idx:
                solar_val = float(sg[t_idx])

            exo_nec.append(load_val + solar_val)

        return exo_nec

    def _get_battery_socs(self) -> list[float]:
        """Read current battery SoC for each building."""
        socs = []
        for b in self._city.buildings:
            es = getattr(b, 'electrical_storage', None)
            if es is not None:
                soc = getattr(es, 'soc', None)
                if soc is not None:
                    if hasattr(soc, '__len__') and len(soc) > 0:
                        # Read actual current SoC, not end of pre-allocated array
                        t = int(getattr(self._city, 'time_step', 0))
                        idx = max(0, t - 1)
                        if idx < len(soc):
                            socs.append(float(soc[idx]))
                        else:
                            socs.append(float(soc[-1]))
                    else:
                        socs.append(float(soc))
                else:
                    socs.append(0.5)
            else:
                socs.append(0.5)
        return socs

    def _compute_safe_bounds(
        self,
        exo_nec: list[float],
        socs: list[float],
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Compute per-action [safe_min, safe_max] bounds.

        For EV chargers, respects min_charging_power / min_discharging_power:
        - If remaining headroom < min_charging_power, EV action is forced to 0
          (since any positive action would trigger at least min_charge kW).
        - Otherwise, EV safe range allows actions that map to [min_charge, headroom].

        Returns:
            safe_min, safe_max: arrays of shape (n_actions,)
            interventions: count of collapsed ranges
        """
        safe_min = np.full(self._n_actions, -1.0, dtype=np.float32)
        safe_max = np.full(self._n_actions, 1.0, dtype=np.float32)
        interventions = 0

        for b_idx in range(self._n_buildings):
            if b_idx >= len(exo_nec) or b_idx not in self._building_batt_act:
                continue

            exo = exo_nec[b_idx]
            batt_act_idx = self._building_batt_act[b_idx]
            p_batt = self._batt_powers[b_idx]

            # Total headroom for all controllable devices
            # Import: NEC < P_bmax => total_device_power < P_bmax - exo
            # Export: -NEC < P_bmax => total_device_power > -(P_bmax + exo)
            import_headroom = self._p_bmax - exo  # max total device power (charge)
            export_headroom = self._p_bmax + exo   # max total device power magnitude (discharge)

            has_ev = b_idx in self._ev_max_charge
            if has_ev:
                ev_max_ch = self._ev_max_charge[b_idx]
                ev_min_ch = self._ev_min_charge[b_idx]
                ev_max_dis = self._ev_max_discharge[b_idx]
                ev_min_dis = self._ev_min_discharge[b_idx]
                ev_act_idx = self._building_ev_act[b_idx]

                # --- Allocate headroom between battery and EV ---
                # Battery gets first priority since it's more controllable
                # (linear action-to-power, no min_power issue).
                # EV gets the remainder.

                # CHARGE direction (import): both battery and EV want to charge
                # Battery can use up to min(p_batt, import_headroom)
                batt_charge_limit = min(p_batt, max(0, import_headroom))
                ev_charge_headroom = max(0, import_headroom - batt_charge_limit)

                # DISCHARGE direction (export): both want to discharge
                batt_discharge_limit = min(p_batt, max(0, export_headroom))
                ev_discharge_headroom = max(0, export_headroom - batt_discharge_limit)

                # But we also need to be conservative: if battery uses LESS than
                # its limit (e.g., action=0.5 not 1.0), EV could use MORE.
                # The worst case is: battery at max charge + EV at max charge.
                # So we need BOTH to fit within headroom simultaneously.
                # Use proportional split based on max device powers.
                total_power = p_batt + ev_max_ch if ev_max_ch > 0 else p_batt + 1.0
                batt_frac = p_batt / total_power
                ev_frac = 1.0 - batt_frac

                # Battery bounds
                batt_import = import_headroom * batt_frac
                batt_export = export_headroom * batt_frac
                b_smin = max(-1.0, -batt_export / p_batt)
                b_smax = min(1.0, batt_import / p_batt)

                # EV bounds -- but with min power enforcement
                ev_import = import_headroom * ev_frac
                ev_export = export_headroom * ev_frac

                # CityLearn clamps: any positive action -> at least min_charge kW
                # any negative action -> at least min_discharge kW
                # So if ev_import < min_charge, we can't allow ANY positive action
                if ev_import >= ev_min_ch and ev_min_ch > 0:
                    # Positive actions map to [min_charge, ev_import] kW
                    # In action space: [min_charge/max_charge, ev_import/max_charge]
                    # But we rescale [-1,1] -> [safe_min, safe_max], so safe_max
                    # should limit the max import
                    ev_smax = min(1.0, ev_import / ev_max_ch) if ev_max_ch > 0 else 0.0
                elif ev_min_ch == 0 and ev_max_ch > 0:
                    ev_smax = min(1.0, ev_import / ev_max_ch)
                else:
                    # Can't charge at all without exceeding limit
                    ev_smax = 0.0

                # Discharge side
                if ev_export >= ev_min_dis and ev_min_dis > 0:
                    ev_smin = max(-1.0, -ev_export / ev_max_dis) if ev_max_dis > 0 else 0.0
                elif ev_min_dis == 0 and ev_max_dis > 0:
                    ev_smin = max(-1.0, -ev_export / ev_max_dis)
                else:
                    ev_smin = 0.0

                if ev_smax < ev_smin:
                    interventions += 1
                safe_min[ev_act_idx] = ev_smin
                safe_max[ev_act_idx] = ev_smax
            else:
                # Battery-only: all headroom goes to battery
                b_smin = max(-1.0, -export_headroom / p_batt)
                b_smax = min(1.0, import_headroom / p_batt)

            if b_smax < b_smin:
                interventions += 1
            safe_min[batt_act_idx] = b_smin
            safe_max[batt_act_idx] = b_smax

            # C2: Magnitude-limited SoC bounds
            # Instead of binary block, limit charge/discharge AMOUNT to stay within SoC bounds
            # SoC change per step = action * power * efficiency / capacity
            # So max safe action = remaining_soc_room / (power * efficiency / capacity)
            soc = socs[b_idx] if b_idx < len(socs) else 0.5
            eta = 0.95  # battery efficiency
            soc_per_unit_action = (p_batt * eta) / 6.4  # SoC change per unit action (assuming 6.4kWh capacity)

            # Read actual capacity if available
            try:
                es = self._city.buildings[b_idx].electrical_storage
                cap = float(getattr(es, 'capacity', 6.4) or 6.4)
                if cap > 0:
                    soc_per_unit_action = (p_batt * eta) / cap
            except Exception:
                pass

            # Max charge: don't push SoC above soc_high
            remaining_charge = max(0.0, self._soc_high - soc)
            max_charge_action = remaining_charge / max(soc_per_unit_action, 1e-9)
            safe_max[batt_act_idx] = min(safe_max[batt_act_idx], max_charge_action)

            # Max discharge: don't push SoC below soc_low
            remaining_discharge = max(0.0, soc - self._soc_low)
            max_discharge_action = remaining_discharge / max(soc_per_unit_action, 1e-9)
            safe_min[batt_act_idx] = max(safe_min[batt_act_idx], -max_discharge_action)

        # Passthrough indices keep [-1, 1]
        for idx in self._passthrough_indices:
            safe_min[idx] = -1.0
            safe_max[idx] = 1.0

        # ── C4: Grid-level aggregate import constraint (BATTERY ONLY) ──
        # EVs are exempt to preserve C1 (EV departure SoC) priority.
        # Only tighten battery safe_max (charge side), leave EV bounds unchanged.
        c4_enabled = os.environ.get("MASK_C4_ENABLED", "0") == "1"
        self._last_total_exo_import = sum(max(0.0, e) for e in exo_nec)

        if c4_enabled:
            total_exo_import = self._last_total_exo_import
            grid_headroom = max(0.0, self._p_gmax - total_exo_import)

            # Total battery charge power at current safe_max
            total_batt_charge = sum(
                max(0.0, safe_max[self._building_batt_act[b]]) * self._batt_powers[b]
                for b in self._building_batt_act
            )

            # Also count EV charge (but don't restrict it)
            total_ev_charge = sum(
                max(0.0, safe_max[self._building_ev_act[b]]) * self._ev_max_charge.get(b, 0.0)
                for b in self._building_ev_act
            ) if self._building_ev_act else 0.0

            remaining_for_batt = max(0.0, grid_headroom - total_ev_charge)

            if total_batt_charge > remaining_for_batt and total_batt_charge > 0:
                # Scale down ALL battery charge proportionally
                scale = max(0.0, remaining_for_batt / total_batt_charge)
                for b in self._building_batt_act:
                    batt_act_idx = self._building_batt_act[b]
                    safe_max[batt_act_idx] = max(safe_min[batt_act_idx],
                                                  safe_max[batt_act_idx] * scale)

        # Ensure no inverted ranges after C4 scaling
        inverted = safe_max < safe_min
        if np.any(inverted):
            safe_max = np.maximum(safe_max, safe_min)
            interventions += int(inverted.sum())

        return safe_min, safe_max, interventions

    @staticmethod
    def _rescale(
        raw: np.ndarray,
        safe_min: np.ndarray,
        safe_max: np.ndarray,
    ) -> np.ndarray:
        """Piecewise linear rescale: raw=0 always maps to executed=0.

        raw in [-1, 0) → executed in [safe_min, 0)  (discharge side)
        raw = 0        → executed = 0                (idle)
        raw in (0, +1] → executed in (0, safe_max]   (charge side)

        This prevents the discharge bias that occurs with linear rescaling
        when |safe_min| > safe_max (more room for discharge than charge).
        """
        result = np.zeros_like(raw)

        # Discharge side: raw < 0 maps to [safe_min, 0]
        neg = raw < 0
        if np.any(neg):
            # raw=-1 → safe_min, raw=0 → 0
            result = np.where(neg, raw * np.abs(safe_min), result)

        # Charge side: raw > 0 maps to [0, safe_max]
        pos = raw > 0
        if np.any(pos):
            # raw=0 → 0, raw=+1 → safe_max
            result = np.where(pos, raw * np.maximum(safe_max, 0.0), result)

        # Collapsed ranges: use 0 (idle)
        collapsed = safe_max <= safe_min
        if np.any(collapsed):
            result = np.where(collapsed, 0.0, result)

        return result

    def _apply_mask(self, action: np.ndarray) -> np.ndarray:
        """Apply action masking: compute safe bounds and rescale."""
        exo_nec = self._get_exogenous_nec()
        socs = self._get_battery_socs()
        safe_min, safe_max, interventions = self._compute_safe_bounds(
            exo_nec, socs)

        if self._beta_mode:
            # Beta mode: x ∈ (0,1) → affine transform to [safe_min, safe_max]
            x = np.clip(action, 1e-6, 1 - 1e-6)
            safe_action = safe_min + x * (safe_max - safe_min)
            safe_action = np.clip(safe_action, safe_min, safe_max)
        else:
            safe_action = self._rescale(action, safe_min, safe_max)

        self._last_safe_min = safe_min.copy()
        self._last_safe_max = safe_max.copy()
        self._last_interventions = interventions

        if self._city is not None:
            self._city._action_mask_safe_min = safe_min.copy()
            self._city._action_mask_safe_max = safe_max.copy()

        if self._step_count % 1000 == 0 and self._step_count > 0:
            batt_ranges = np.array([
                safe_max[ai] - safe_min[ai]
                for ai in self._batt_act_indices])
            ev_ranges = np.array([
                safe_max[ai] - safe_min[ai]
                for ai in self._ev_act_indices]) if self._ev_act_indices else np.array([0.0])
            print(f"[ActionMask] step={self._step_count} "
                  f"batt_range_mean={np.mean(batt_ranges):.3f} "
                  f"ev_range_mean={np.mean(ev_ranges):.3f} "
                  f"interventions={interventions}")
            total_charge_kw = sum(
                max(0.0, safe_max[self._building_batt_act[b]]) * self._batt_powers[b]
                + (max(0.0, safe_max[self._building_ev_act[b]]) * self._ev_max_charge.get(b, 0.0)
                   if b in self._building_ev_act else 0.0)
                for b in self._building_batt_act)
            c4_util = (self._last_total_exo_import + total_charge_kw) / max(self._p_gmax, 1e-6)
            print(f"[ActionMask] C4: grid_headroom={max(0, self._p_gmax - self._last_total_exo_import):.1f}kW "
                  f"max_charge={total_charge_kw:.1f}kW "
                  f"c4_util={c4_util:.2f}")

        self._total_interventions += interventions
        return safe_action
