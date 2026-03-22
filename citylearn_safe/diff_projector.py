"""
DiffProjector: Differentiable action projection layer for SP-RL (Markgraf et al. 2025).

Solves a QP via cvxpylayers to project unsafe RL actions onto the safe set,
yielding gradients through the projection for end-to-end policy optimization.

Hard constraints only (no slack variables):
  C2: Battery SoC bounds after action
  C3: Per-building net power magnitude bound
  C4: Grid-level import bound (linear, one-sided)
  Action bounds: z in [-1, 1]^n

On infeasibility (e.g. base load already violates limits), falls back to
clamping actions to [-1, 1] and returns feasible=False. NaN gradients are
blocked via detach-and-replace.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import gymnasium as gym


# ---------------------------------------------------------------------------
#  Env unwrapping
# ---------------------------------------------------------------------------

def _unwrap_to_citylearn(env: Any) -> Any:
    """Walk wrapper chain to find the CityLearn env (has .buildings + .time_step)."""
    cur = env
    seen: set = set()
    for _ in range(80):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        blds = getattr(cur, "buildings", None)
        ts = getattr(cur, "time_step", None)
        if (blds is not None and hasattr(blds, "__len__")
                and len(blds) > 0 and ts is not None):
            return cur
        for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None


# ---------------------------------------------------------------------------
#  Device mapping: discover battery / EV action indices per building
# ---------------------------------------------------------------------------

def _build_device_mapping(env: Any) -> Dict[str, Any]:
    """
    Parse action_names from the CityLearn env to locate battery and EV charger
    action indices in the flat action vector.

    Returns dict with keys:
        nb         - number of buildings
        total_dim  - total flat action dimension
        batt_gidx  - np array of global indices for battery actions
        batt_bidx  - np array of building index for each battery
        ev_gidx    - np array of global indices for EV charger actions
        ev_bidx    - np array of building index for each EV charger
        ev_local   - np array of local EV index within each building
    """
    city = _unwrap_to_citylearn(env)
    if city is None:
        raise RuntimeError("[DiffProjector] Cannot unwrap env to CityLearn")

    names_raw = getattr(city, "action_names", None)
    buildings = list(getattr(city, "buildings", []))
    nb = len(buildings)

    # Handle central-agent flat action_names: single list wrapping all buildings.
    # Reconstruct per-building sublists using "electrical_storage" as delimiters.
    is_central = (
        isinstance(names_raw, list) and len(names_raw) == 1
        and isinstance(names_raw[0], list) and nb > 1
        and len(names_raw[0]) > nb
    )
    if is_central:
        flat = names_raw[0]
        batt_positions = [
            i for i, n in enumerate(flat)
            if str(n).lower() == "electrical_storage"
        ]
        if len(batt_positions) == nb:
            reconstructed = []
            for b in range(nb):
                start = batt_positions[b]
                end = batt_positions[b + 1] if b + 1 < nb else len(flat)
                reconstructed.append(list(flat[start:end]))
            names_raw = reconstructed

    # Flatten list-of-lists if still nested
    if (isinstance(names_raw, list) and len(names_raw) > 0
            and not isinstance(names_raw[0], list)):
        # Already flat -- wrap to single building (shouldn't happen in practice)
        names_raw = [names_raw]

    g = 0
    batt_gidx, batt_bidx = [], []
    ev_gidx, ev_bidx, ev_local = [], [], []

    for b_idx, sub in enumerate(names_raw):
        if not isinstance(sub, list):
            sub = [sub]
        local_ev = 0
        batt_seen = False
        for n in sub:
            nl = str(n).lower()
            if nl == "electrical_storage" and not batt_seen:
                batt_gidx.append(g)
                batt_bidx.append(b_idx)
                batt_seen = True
            if "electric_vehicle_storage_charger_" in nl:
                ev_gidx.append(g)
                ev_bidx.append(b_idx)
                ev_local.append(local_ev)
                local_ev += 1
            g += 1

    return {
        "nb": nb,
        "total_dim": g,
        "batt_gidx": np.asarray(batt_gidx, dtype=int),
        "batt_bidx": np.asarray(batt_bidx, dtype=int),
        "ev_gidx": np.asarray(ev_gidx, dtype=int),
        "ev_bidx": np.asarray(ev_bidx, dtype=int),
        "ev_local": np.asarray(ev_local, dtype=int),
    }


# ---------------------------------------------------------------------------
#  DiffProjector
# ---------------------------------------------------------------------------

class DiffProjector:
    """
    Differentiable safety projection using cvxpylayers.

    Solves:
        min  ||z - u||^2
        s.t. -1 <= z <= 1                                   (action bounds)
             soc_low <= soc0 + z_batt * soc_scale <= soc_high  (C2)
             -P_bmax <= base_b + gains_b . z_b <= P_bmax       (C3)
             sum(base_b + gains_b . z_b) <= P_grid_max         (C4, import-only)

    All constraints are hard (no slack). If the QP is infeasible (e.g. base
    load already exceeds limits with zero controllable action), falls back to
    simple clamping.
    """

    def __init__(
        self,
        env: gym.Env,
        soc_low: Optional[float] = None,
        soc_high: Optional[float] = None,
        p_build_max: Optional[float] = None,
        p_grid_max: Optional[float] = None,
        solver_eps: float = 1e-4,
        solver_max_iters: int = 5000,
    ):
        self.env = env

        # Read from env vars with constructor args as fallback defaults
        self.soc_low = float(os.environ.get(
            "CITYLEARN_STEMS_SOC_LOW",
            str(soc_low if soc_low is not None else 0.0),
        ))
        self.soc_high = float(os.environ.get(
            "CITYLEARN_STEMS_SOC_HIGH",
            str(soc_high if soc_high is not None else 0.95),
        ))
        self.p_build_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_BUILDING_MAX",
            str(p_build_max if p_build_max is not None else 4.6083),
        ))
        self.p_grid_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_GRID_MAX",
            str(p_grid_max if p_grid_max is not None else 10.2352),
        ))

        self.solver_eps = float(solver_eps)
        self.solver_max_iters = int(solver_max_iters)

        self._built = False
        self._mapping: Optional[Dict[str, Any]] = None
        self._layer: Optional[CvxpyLayer] = None

        # Precomputed constant matrices (set in build)
        self._A_batt: Optional[np.ndarray] = None
        self._A_ev: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    #  Build: construct the cvxpylayers problem (call once after env reset)
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the cvxpylayers QP. Must be called once after the env is created/reset."""
        self._mapping = _build_device_mapping(self.env)
        m = self._mapping
        nb = m["nb"]
        n_total = m["total_dim"]
        n_batt = len(m["batt_gidx"])
        n_ev = len(m["ev_gidx"])

        # ---- Building-to-device mapping matrices ----
        # A_batt[b, j] = 1  iff battery j belongs to building b
        A_batt = np.zeros((nb, n_batt), dtype=float)
        for j, b in enumerate(m["batt_bidx"]):
            A_batt[int(b), j] = 1.0
        self._A_batt = A_batt

        # A_ev[b, j] = 1  iff EV charger j belongs to building b
        A_ev = np.zeros((nb, n_ev), dtype=float)
        for j, b in enumerate(m["ev_bidx"]):
            A_ev[int(b), j] = 1.0
        self._A_ev = A_ev

        # ---- CVXPY variables ----
        z = cp.Variable(n_total, name="z")

        # ---- CVXPY parameters (updated each call) ----
        u = cp.Parameter(n_total, name="u")           # unsafe action
        base = cp.Parameter(nb, name="base")           # base NEC per building (kW)
        soc0 = cp.Parameter(n_batt, name="soc0")       # current battery SoC
        soc_scale = cp.Parameter(n_batt, name="soc_scale", nonneg=True)  # nom_power / capacity
        batt_gain = cp.Parameter(n_batt, name="batt_gain", nonneg=True)  # kW per unit action
        ev_gain = cp.Parameter(n_ev, name="ev_gain", nonneg=True)        # kW per unit action

        # ---- Constraints ----
        constraints = []

        # Action bounds: z in [-1, 1]
        constraints += [z >= -1.0, z <= 1.0]

        # C2: Battery SoC bounds
        if n_batt > 0:
            z_batt = z[m["batt_gidx"].tolist()]
            soc_next = soc0 + cp.multiply(z_batt, soc_scale)
            constraints += [
                soc_next >= self.soc_low,
                soc_next <= self.soc_high,
            ]

        # Build net power per building:
        #   net_b = base_b + sum_j(batt_gain_j * z_batt_j) [for batts in b]
        #                   + sum_k(ev_gain_k * z_ev_k)    [for EVs in b]
        net = base  # shape (nb,)
        if n_batt > 0:
            z_batt_expr = z[m["batt_gidx"].tolist()]
            net = net + A_batt @ cp.multiply(z_batt_expr, batt_gain)
        if n_ev > 0:
            z_ev_expr = z[m["ev_gidx"].tolist()]
            net = net + A_ev @ cp.multiply(z_ev_expr, ev_gain)

        # C3: Per-building power magnitude bound
        constraints += [
            net <= self.p_build_max,
            net >= -self.p_build_max,
        ]

        # C4: Grid-level import bound (linear, one-sided)
        grid_total = cp.sum(net)
        constraints += [grid_total <= self.p_grid_max]

        # ---- Objective ----
        objective = cp.Minimize(cp.sum_squares(z - u))
        prob = cp.Problem(objective, constraints)

        # ---- Wrap as differentiable layer ----
        param_list = [u, base, soc0, soc_scale, batt_gain, ev_gain]
        self._layer = CvxpyLayer(
            prob,
            parameters=param_list,
            variables=[z],
        )

        # Store parameter order for reference
        self._param_names = ["u", "base", "soc0", "soc_scale", "batt_gain", "ev_gain"]
        self._built = True

        # ---- Diagnostics ----
        print(f"[DiffProjector] Built QP for {nb} buildings, "
              f"{n_batt} batteries, {n_ev} EV chargers, "
              f"{n_total} total action dims")
        print(f"[DiffProjector] Constraint bounds: "
              f"SoC=[{self.soc_low}, {self.soc_high}], "
              f"P_build_max={self.p_build_max:.4f} kW, "
              f"P_grid_max={self.p_grid_max:.4f} kW")
        print(f"[DiffProjector] Solver: SCS eps={self.solver_eps}, "
              f"max_iters={self.solver_max_iters}")

        if n_batt > 0:
            batt_buildings = sorted(set(m["batt_bidx"].tolist()))
            print(f"[DiffProjector] Battery actions at global idx "
                  f"{m['batt_gidx'].tolist()}, buildings {batt_buildings}")
        if n_ev > 0:
            ev_buildings = sorted(set(m["ev_bidx"].tolist()))
            print(f"[DiffProjector] EV actions at global idx "
                  f"{m['ev_gidx'].tolist()}, buildings {ev_buildings}")

    # ------------------------------------------------------------------
    #  Extract state from the live CityLearn environment
    # ------------------------------------------------------------------

    def _extract_state(self) -> Dict[str, np.ndarray]:
        """
        Read current state from CityLearn env.

        Returns dict with:
            base_nec   : (nb,)    base net electricity consumption per building (kW)
            soc0       : (n_batt,) current battery SoC in [0, 1]
            soc_scale  : (n_batt,) nominal_power / capacity (SoC change per unit action)
            batt_gain  : (n_batt,) nominal power in kW (power per unit action)
            ev_gain    : (n_ev,)   max charging power in kW (power per unit action)
        """
        m = self._mapping
        city = _unwrap_to_citylearn(self.env)
        if city is None:
            raise RuntimeError("[DiffProjector] Cannot unwrap to CityLearn env")

        buildings = list(getattr(city, "buildings", []))
        nb = m["nb"]
        t_now = int(getattr(city, "time_step", 0))
        t_idx = max(0, t_now - 1)

        # ---- Base net electricity consumption per building ----
        # base_nec = non_shiftable_load + solar_generation  (solar is negative)
        base_nec = np.zeros(nb, dtype=np.float64)
        for b_idx, b in enumerate(buildings):
            nsl = np.asarray(
                getattr(b, "_Building__energy_to_non_shiftable_load", []),
                dtype=np.float64,
            )
            sg = np.asarray(
                getattr(b, "_Building__solar_generation", []),
                dtype=np.float64,
            )
            T_max = min(len(nsl), len(sg))
            if T_max > 0:
                idx = t_now if t_now < T_max else T_max - 1
                base_nec[b_idx] = float(nsl[idx] + sg[idx])

        # ---- Battery parameters ----
        n_batt = len(m["batt_gidx"])
        soc0 = np.full(n_batt, 0.5, dtype=np.float64)
        soc_scale = np.ones(n_batt, dtype=np.float64)
        batt_gain = np.ones(n_batt, dtype=np.float64)

        for i in range(n_batt):
            b_idx = int(m["batt_bidx"][i])
            es = getattr(buildings[b_idx], "electrical_storage", None)
            if es is None:
                continue
            cap = float(getattr(es, "capacity", 6.4) or 6.4)
            nom = float(getattr(es, "nominal_power", 5.0) or 5.0)
            batt_gain[i] = max(1e-6, nom)
            soc_scale[i] = max(1e-6, nom / max(1e-6, cap))
            soc_arr = np.asarray(getattr(es, "soc", []), dtype=np.float64).ravel()
            if 0 <= t_idx < len(soc_arr):
                soc0[i] = float(np.clip(soc_arr[t_idx], 0.0, 1.0))

        # ---- EV charger parameters ----
        n_ev = len(m["ev_gidx"])
        ev_gain = np.zeros(n_ev, dtype=np.float64)

        for i in range(n_ev):
            b_idx = int(m["ev_bidx"][i])
            local = int(m["ev_local"][i])
            chargers = getattr(buildings[b_idx], "electric_vehicle_chargers", None) or []
            if local >= len(chargers):
                continue
            ch = chargers[local]
            mp = getattr(
                ch, "max_charging_power",
                getattr(ch, "_Charger__max_charging_power", 0),
            )
            if isinstance(mp, np.ndarray):
                mp = float(mp.ravel()[0])
            ev_gain[i] = max(0.0, float(mp or 0.0))

        return {
            "base_nec": base_nec,
            "soc0": soc0,
            "soc_scale": soc_scale,
            "batt_gain": batt_gain,
            "ev_gain": ev_gain,
        }

    # ------------------------------------------------------------------
    #  Project: differentiable action projection
    # ------------------------------------------------------------------

    def project(
        self,
        obs_tensor: torch.Tensor,
        unsafe_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Project an unsafe action onto the safe set. Differentiable w.r.t. unsafe_action.

        Args:
            obs_tensor:    Observation tensor (not used directly; state is read from env).
            unsafe_action: Shape (batch, act_dim) or (act_dim,), values in [-1, 1].

        Returns:
            safe_action: Same shape as unsafe_action, in the safe set.
            info: Dict with 'projection_delta', 'solve_time_ms', 'feasible'.
        """
        if not self._built:
            self.build()

        # Handle shape: support both batched and unbatched
        squeeze_output = False
        if unsafe_action.dim() == 1:
            unsafe_action = unsafe_action.unsqueeze(0)
            squeeze_output = True

        batch_size = unsafe_action.shape[0]
        device = unsafe_action.device
        dtype = torch.float32

        # Extract physical state from the environment
        state = self._extract_state()

        # Convert state to tensors (no grad -- these are environment constants)
        base_t = torch.tensor(state["base_nec"], dtype=dtype, device=device)
        soc0_t = torch.tensor(state["soc0"], dtype=dtype, device=device)
        soc_scale_t = torch.tensor(state["soc_scale"], dtype=dtype, device=device)
        batt_gain_t = torch.tensor(state["batt_gain"], dtype=dtype, device=device)
        ev_gain_t = torch.tensor(state["ev_gain"], dtype=dtype, device=device)

        # Solve for each sample in the batch
        safe_actions = []
        total_delta = 0.0
        all_feasible = True

        t0 = time.monotonic()

        for i in range(batch_size):
            u_i = unsafe_action[i]  # shape (act_dim,), keeps grad

            try:
                (z_opt,) = self._layer(
                    u_i, base_t, soc0_t, soc_scale_t, batt_gain_t, ev_gain_t,
                    solver_args={
                        "eps": self.solver_eps,
                        "max_iters": self.solver_max_iters,
                    },
                )

                # Guard against NaN from solver failure
                if torch.isnan(z_opt).any() or torch.isinf(z_opt).any():
                    raise RuntimeError("Solver returned NaN/Inf")

                safe_actions.append(z_opt)
                with torch.no_grad():
                    total_delta += torch.norm(z_opt - u_i).item()

            except Exception:
                # Infeasible or solver error: fall back to clamping
                all_feasible = False
                clamped = self._fallback_clamp(u_i, state)
                safe_actions.append(clamped)
                with torch.no_grad():
                    total_delta += torch.norm(clamped - u_i).item()

        solve_ms = (time.monotonic() - t0) * 1000.0

        result = torch.stack(safe_actions, dim=0)
        if squeeze_output:
            result = result.squeeze(0)

        info = {
            "projection_delta": total_delta / max(1, batch_size),
            "solve_time_ms": solve_ms,
            "feasible": all_feasible,
        }
        return result, info

    # ------------------------------------------------------------------
    #  Fallback: simple clamping when QP is infeasible
    # ------------------------------------------------------------------

    def _fallback_clamp(
        self,
        unsafe_action: torch.Tensor,
        state: Dict[str, np.ndarray],
    ) -> torch.Tensor:
        """
        Fallback when QP is infeasible. Clamp to [-1, 1] and apply C2 bounds
        analytically. Returns a tensor detached from the original graph to
        prevent NaN gradients, then re-attached via straight-through estimator.
        """
        m = self._mapping
        n_batt = len(m["batt_gidx"])

        # Detach to block NaN gradients, then use straight-through
        z = unsafe_action.detach().clone()
        z = torch.clamp(z, -1.0, 1.0)

        # Apply C2 analytically: soc_low <= soc0 + z_b * scale <= soc_high
        for i in range(n_batt):
            gidx = int(m["batt_gidx"][i])
            s = state["soc_scale"][i]
            s0 = state["soc0"][i]
            if s > 1e-9:
                a_lo = max((self.soc_low - s0) / s, -1.0)
                a_hi = min((self.soc_high - s0) / s, 1.0)
                z[gidx] = torch.clamp(z[gidx], min=a_lo, max=a_hi)

        # Straight-through estimator: gradient flows through unsafe_action
        # but forward value is the clamped z.
        return unsafe_action + (z - unsafe_action.detach())
