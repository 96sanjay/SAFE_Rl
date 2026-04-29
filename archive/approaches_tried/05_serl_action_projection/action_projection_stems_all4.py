"""All-4 action projector for the isolated STEMS SP-RL path.

This module keeps legacy trainers untouched. It combines:
  - EV departure-aware bounds from CityLearnActionBoundsProvider (C0/C2/C3/C4)
  - closest-point QP projection across the full action vector
  - replayable projector state tensors for projected training

The intended use is:
  1. actor proposes raw action u
  2. projector computes safe action u_safe
  3. env executes u_safe
  4. training stores both u and u_safe plus projector state

`C1` remains a dense learning cost and is not a hard projection constraint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from citylearn_safe.policy_action_mask import CityLearnActionBoundsProvider, _unwrap_citylearn

try:
    import cvxpy as cp

    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


@dataclass
class ProjectorState:
    safe_min: np.ndarray
    safe_max: np.ndarray
    exo_nec: np.ndarray
    socs: np.ndarray


class StemsAll4Projector:
    """Closest-point all-4 projector for the isolated STEMS branch.

    The EV departure logic enters as current-step EV bounds produced by the
    existing CityLearnActionBoundsProvider. The QP then solves the nearest
    feasible action subject to:
      - box bounds from the provider (contains C0/C2/C3/C4 local logic)
      - battery SoC constraints
      - per-building power limits
      - district import limit
    """

    def __init__(self, env: Any):
        self._env = env
        self._city = _unwrap_citylearn(env)
        if self._city is None:
            raise RuntimeError(
                "[StemsAll4Projector] Cannot find CityLearn env with .buildings in wrapper chain.",
            )

        self._bounds_provider = CityLearnActionBoundsProvider(env)
        self._n_buildings = len(self._city.buildings)
        self._n_actions = int(env.action_space.shape[0])
        self._p_bmax = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
        self._soc_low = float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0"))
        self._soc_high = float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95"))
        self._p_gmax = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "10.2352"))
        self._c4_enabled = os.environ.get("MASK_C4_ENABLED", "0") == "1"

        self._building_batt_act = dict(self._bounds_provider._building_batt_act)
        self._building_ev_act = dict(self._bounds_provider._building_ev_act)
        self._passthrough_indices = list(self._bounds_provider._passthrough_indices)
        self._batt_powers = dict(self._bounds_provider._batt_powers)
        self._ev_max_charge = dict(self._bounds_provider._ev_max_charge)

        self.state_tensor_dim = (2 * self._n_actions) + (2 * self._n_buildings)
        self._last_qp_path = "unknown"
        self._last_qp_ms = 0.0

    def _capture_state(self) -> ProjectorState:
        exo_nec = np.asarray(self._bounds_provider._get_exogenous_nec(), dtype=np.float32)
        socs = np.asarray(self._bounds_provider._get_battery_socs(), dtype=np.float32)
        safe_min, safe_max = self._bounds_provider._compute_safe_bounds(
            exo_nec.tolist(),
            socs.tolist(),
        )
        return ProjectorState(
            safe_min=np.asarray(safe_min, dtype=np.float32),
            safe_max=np.asarray(safe_max, dtype=np.float32),
            exo_nec=exo_nec,
            socs=socs,
        )

    def extract_state_tensor(self) -> torch.Tensor:
        state = self._capture_state()
        arr = np.concatenate([state.safe_min, state.safe_max, state.exo_nec, state.socs], axis=0)
        return torch.as_tensor(arr, dtype=torch.float32)

    def _state_from_tensor(self, state_tensor: torch.Tensor) -> ProjectorState:
        if state_tensor.dim() != 1:
            raise ValueError("Projector state tensor must be rank-1 per sample.")
        n_act = self._n_actions
        n_bld = self._n_buildings
        expected = (2 * n_act) + (2 * n_bld)
        if int(state_tensor.numel()) != expected:
            raise ValueError(
                f"Projector state tensor size mismatch: expected {expected}, got {int(state_tensor.numel())}.",
            )
        safe_min = state_tensor[:n_act].detach().cpu().numpy().astype(np.float32, copy=False)
        safe_max = state_tensor[n_act : 2 * n_act].detach().cpu().numpy().astype(np.float32, copy=False)
        exo_nec = state_tensor[2 * n_act : 2 * n_act + n_bld].detach().cpu().numpy().astype(np.float32, copy=False)
        socs = state_tensor[2 * n_act + n_bld :].detach().cpu().numpy().astype(np.float32, copy=False)
        return ProjectorState(safe_min=safe_min, safe_max=safe_max, exo_nec=exo_nec, socs=socs)

    def _battery_soc_scales(self, b_idx: int) -> tuple[float, float]:
        try:
            es = self._city.buildings[b_idx].electrical_storage
            cap = float(getattr(es, "capacity", 6.4) or 6.4)
            charge_eta, discharge_eta = self._bounds_provider._battery_charge_discharge_etas(b_idx)
            p_batt = float(self._batt_powers.get(b_idx, 0.0))
            dt = float(self._bounds_provider._batt_step_hours)
            scale_ch = (p_batt * dt * charge_eta) / max(cap, 1e-9)
            scale_dis = (p_batt * dt) / max(cap * discharge_eta, 1e-9)
            return float(scale_ch), float(scale_dis)
        except Exception:
            return 0.0, 0.0

    def _qp_project_np(self, raw_action: np.ndarray, state: ProjectorState) -> np.ndarray:
        import time as _time

        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        safe_min = state.safe_min
        safe_max = state.safe_max
        exo_nec = state.exo_nec
        socs = state.socs

        _t0 = _time.monotonic()

        if not _HAS_CVXPY:
            self._last_qp_path = "no_cvxpy"
            self._last_qp_ms = 0.0
            return np.clip(raw_action, safe_min, safe_max)

        clipped = np.clip(raw_action, safe_min, safe_max)

        all_feasible = True
        for b_idx, act_idx in self._building_batt_act.items():
            scale_ch, scale_dis = self._battery_soc_scales(b_idx)
            soc = float(socs[b_idx]) if b_idx < len(socs) else 0.5
            u = float(clipped[act_idx])
            if u >= 0.0:
                next_soc = soc + (u * scale_ch)
            else:
                next_soc = soc + (u * scale_dis)
            if next_soc < self._soc_low - 1e-6 or next_soc > self._soc_high + 1e-6:
                all_feasible = False
                break

        if all_feasible:
            grid_total = 0.0
            for b_idx in range(self._n_buildings):
                exo = float(exo_nec[b_idx]) if b_idx < len(exo_nec) else 0.0
                net_b = exo
                if b_idx in self._building_batt_act:
                    net_b += float(clipped[self._building_batt_act[b_idx]]) * float(self._batt_powers.get(b_idx, 0.0))
                if b_idx in self._building_ev_act:
                    net_b += float(clipped[self._building_ev_act[b_idx]]) * float(
                        self._ev_max_charge.get(b_idx, 0.0),
                    )
                if abs(net_b) > self._p_bmax + 1e-6:
                    all_feasible = False
                    break
                grid_total += net_b
            if all_feasible and self._c4_enabled and grid_total > self._p_gmax + 1e-6:
                all_feasible = False

        if all_feasible:
            self._last_qp_path = "fastpath"
            self._last_qp_ms = (_time.monotonic() - _t0) * 1000.0
            return clipped

        z = cp.Variable(self._n_actions)
        objective = cp.Minimize(0.5 * cp.sum_squares(z - raw_action))
        constraints = [
            z >= safe_min,
            z <= safe_max,
            z >= -1.0,
            z <= 1.0,
        ]

        for b_idx, act_idx in self._building_batt_act.items():
            soc = float(socs[b_idx]) if b_idx < len(socs) else 0.5
            scale_ch, scale_dis = self._battery_soc_scales(b_idx)
            if scale_ch > 0.0:
                constraints.append(soc + cp.pos(z[act_idx]) * scale_ch <= self._soc_high)
            if scale_dis > 0.0:
                constraints.append(soc - cp.neg(z[act_idx]) * scale_dis >= self._soc_low)

        net_exprs = []
        for b_idx in range(self._n_buildings):
            exo = float(exo_nec[b_idx]) if b_idx < len(exo_nec) else 0.0
            net_b = exo
            if b_idx in self._building_batt_act:
                act_idx = self._building_batt_act[b_idx]
                net_b = net_b + float(self._batt_powers.get(b_idx, 0.0)) * z[act_idx]
            if b_idx in self._building_ev_act:
                act_idx = self._building_ev_act[b_idx]
                net_b = net_b + float(self._ev_max_charge.get(b_idx, 0.0)) * z[act_idx]
            constraints.append(net_b <= self._p_bmax)
            constraints.append(net_b >= -self._p_bmax)
            net_exprs.append(net_b)

        if self._c4_enabled and net_exprs:
            constraints.append(sum(net_exprs) <= self._p_gmax)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000, eps=1e-4)
            if prob.status in ("optimal", "optimal_inaccurate") and z.value is not None:
                result = np.asarray(z.value, dtype=np.float32).reshape(-1)
                self._last_qp_path = "qp_solved"
                self._last_qp_ms = (_time.monotonic() - _t0) * 1000.0
                return np.clip(result, safe_min, safe_max)
        except Exception:
            pass

        self._last_qp_path = "fallback"
        self._last_qp_ms = (_time.monotonic() - _t0) * 1000.0
        return np.clip(raw_action, safe_min, safe_max)

    def project(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        state_tensors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del obs  # projection depends on state_tensors/live env, not encoded obs
        if action.dim() == 1:
            action = action.unsqueeze(0)
        batch = int(action.size(0))
        if state_tensors is not None and state_tensors.dim() == 1:
            state_tensors = state_tensors.unsqueeze(0)
        if state_tensors is not None and int(state_tensors.size(0)) != batch:
            raise ValueError("Projector state batch size must match action batch size.")

        safe_actions: list[torch.Tensor] = []
        safe_min_list: list[torch.Tensor] = []
        safe_max_list: list[torch.Tensor] = []
        delta_norms: list[torch.Tensor] = []

        for i in range(batch):
            state = self._capture_state() if state_tensors is None else self._state_from_tensor(state_tensors[i])
            raw_np = action[i].detach().cpu().numpy().astype(np.float32, copy=False)
            safe_np = self._qp_project_np(raw_np, state)
            safe_actions.append(torch.as_tensor(safe_np, dtype=action.dtype, device=action.device))
            safe_min_list.append(torch.as_tensor(state.safe_min, dtype=action.dtype, device=action.device))
            safe_max_list.append(torch.as_tensor(state.safe_max, dtype=action.dtype, device=action.device))
            delta = raw_np - safe_np
            delta_norms.append(
                torch.as_tensor(float(np.linalg.norm(delta)), dtype=action.dtype, device=action.device),
            )

        safe_action_t = torch.stack(safe_actions, dim=0)
        info = {
            "safe_min": torch.stack(safe_min_list, dim=0),
            "safe_max": torch.stack(safe_max_list, dim=0),
            "delta_l2": torch.stack(delta_norms, dim=0),
        }
        return safe_action_t, info
