"""
DiffProjector: Differentiable action projection layer for SP-RL (Markgraf et al. 2025).

Solves a QP via cvxpylayers to project unsafe RL actions onto the safe set,
yielding gradients through the projection for end-to-end policy optimization.

Hard constraints only (no slack variables):
  C2: Battery SoC bounds after action (asymmetric charge/discharge via rte)
  C3: Per-building net power magnitude bound
  C4: Grid-level import bound (linear, one-sided)
  Action bounds: z in [-1, 1]^n

On infeasibility (e.g. base load already violates limits), falls back to
clamping actions to [-1, 1] with C2/C3/C4 enforcement and returns
feasible=False. NaN gradients are blocked via detach-and-replace.

Supports per-sample state for replay buffer updates: pass state_tensors
(from extract_state_tensor()) to project() instead of reading from env.
"""
from __future__ import annotations

import math
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
        s.t. -1 <= z <= 1                                          (action bounds)
             soc0 + z_batt * soc_scale_ch <= soc_high              (C2 upper, charge rate)
             soc0 + z_batt * soc_scale_dis >= soc_low              (C2 lower, discharge rate)
             -P_bmax <= base_b + gains_b . z_b <= P_bmax           (C3)
             sum(base_b + gains_b . z_b) <= P_grid_max             (C4, import-only)

    C2 uses asymmetric SoC scales to account for round-trip efficiency:
      - Charge:    delta_SoC = z * power * sqrt(rte) / capacity
      - Discharge: delta_SoC = z * power / sqrt(rte) / capacity

    All constraints are hard (no slack). If the QP is infeasible (e.g. base
    load already exceeds limits with zero controllable action), falls back to
    clamping with analytical C2/C3/C4 enforcement.

    Supports per-sample state for replay buffer updates via state_tensors arg.
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

        # State tensor layout (set in build)
        self._state_tensor_dim: int = 0

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

        # State tensor dimension: base_nec(nb) + soc0(n_batt)
        #   + soc_scale_ch(n_batt) + soc_scale_dis(n_batt)
        #   + batt_gain(n_batt) + ev_gain(n_ev)
        self._state_tensor_dim = nb + n_batt * 4 + n_ev

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
        u = cp.Parameter(n_total, name="u")                # unsafe action
        base = cp.Parameter(nb, name="base")                # base NEC per building (kW)
        soc0 = cp.Parameter(n_batt, name="soc0")            # current battery SoC
        soc_scale_ch = cp.Parameter(n_batt, name="soc_scale_ch", nonneg=True)   # charge rate
        soc_scale_dis = cp.Parameter(n_batt, name="soc_scale_dis", nonneg=True)  # discharge rate
        batt_gain = cp.Parameter(n_batt, name="batt_gain", nonneg=True)  # kW per unit action
        ev_gain = cp.Parameter(n_ev, name="ev_gain", nonneg=True)        # kW per unit action

        # ---- Constraints ----
        constraints = []

        # Action bounds: z in [-1, 1]
        constraints += [z >= -1.0, z <= 1.0]

        # C2: Battery SoC bounds (asymmetric charge/discharge)
        # Upper bound uses charge scale (binding when z > 0):
        #   soc0 + z_batt * soc_scale_ch <= soc_high
        # Lower bound uses discharge scale (binding when z < 0):
        #   soc0 + z_batt * soc_scale_dis >= soc_low
        if n_batt > 0:
            z_batt = z[m["batt_gidx"].tolist()]
            soc_next_ch = soc0 + cp.multiply(z_batt, soc_scale_ch)
            soc_next_dis = soc0 + cp.multiply(z_batt, soc_scale_dis)
            constraints += [
                soc_next_dis >= self.soc_low,
                soc_next_ch <= self.soc_high,
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
        param_list = [u, base, soc0, soc_scale_ch, soc_scale_dis, batt_gain, ev_gain]
        self._layer = CvxpyLayer(
            prob,
            parameters=param_list,
            variables=[z],
        )

        # Store parameter order for reference
        self._param_names = [
            "u", "base", "soc0", "soc_scale_ch", "soc_scale_dis",
            "batt_gain", "ev_gain",
        ]
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
        print(f"[DiffProjector] State tensor dim: {self._state_tensor_dim} "
              f"(nb={nb}, n_batt={n_batt}, n_ev={n_ev})")

        if n_batt > 0:
            batt_buildings = sorted(set(m["batt_bidx"].tolist()))
            print(f"[DiffProjector] Battery actions at global idx "
                  f"{m['batt_gidx'].tolist()}, buildings {batt_buildings}")
        if n_ev > 0:
            ev_buildings = sorted(set(m["ev_bidx"].tolist()))
            print(f"[DiffProjector] EV actions at global idx "
                  f"{m['ev_gidx'].tolist()}, buildings {ev_buildings}")

    # ------------------------------------------------------------------
    #  State tensor dimension (for replay buffer allocation)
    # ------------------------------------------------------------------

    @property
    def state_tensor_dim(self) -> int:
        """Return the dimension of the flat state tensor for replay buffer storage."""
        if not self._built:
            raise RuntimeError(
                "[DiffProjector] Must call build() before accessing state_tensor_dim"
            )
        return self._state_tensor_dim

    # ------------------------------------------------------------------
    #  Extract state from the live CityLearn environment
    # ------------------------------------------------------------------

    def _extract_state(self) -> Dict[str, np.ndarray]:
        """
        Read current state from CityLearn env.

        Returns dict with:
            base_nec       : (nb,)     base net electricity consumption per building (kW)
            soc0           : (n_batt,) current battery SoC in [0, 1]
            soc_scale_ch   : (n_batt,) charge SoC rate: power * sqrt(rte) / capacity
            soc_scale_dis  : (n_batt,) discharge SoC rate: power / sqrt(rte) / capacity
            batt_gain      : (n_batt,) nominal power in kW (power per unit action)
            ev_gain        : (n_ev,)   max charging power in kW (power per unit action)
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

        # ---- Battery parameters (asymmetric C2) ----
        n_batt = len(m["batt_gidx"])
        soc0 = np.full(n_batt, 0.5, dtype=np.float64)
        soc_scale_ch = np.ones(n_batt, dtype=np.float64)
        soc_scale_dis = np.ones(n_batt, dtype=np.float64)
        batt_gain = np.ones(n_batt, dtype=np.float64)

        for i in range(n_batt):
            b_idx = int(m["batt_bidx"][i])
            es = getattr(buildings[b_idx], "electrical_storage", None)
            if es is None:
                continue
            cap = float(getattr(es, "capacity", 6.4) or 6.4)
            nom = float(getattr(es, "nominal_power", 5.0) or 5.0)
            rte = float(getattr(es, "round_trip_efficiency",
                        getattr(es, "efficiency", 0.9)) or 0.9)
            rte = max(0.01, min(1.0, rte))  # clamp to valid range
            sqrt_rte = math.sqrt(rte)

            batt_gain[i] = max(1e-6, nom)
            # Charge: delta_SoC = z * power * sqrt(rte) / capacity
            soc_scale_ch[i] = max(1e-6, nom * sqrt_rte / max(1e-6, cap))
            # Discharge: delta_SoC = z * power / sqrt(rte) / capacity
            soc_scale_dis[i] = max(1e-6, nom / sqrt_rte / max(1e-6, cap))

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
            "soc_scale_ch": soc_scale_ch,
            "soc_scale_dis": soc_scale_dis,
            "batt_gain": batt_gain,
            "ev_gain": ev_gain,
        }

    # ------------------------------------------------------------------
    #  State tensor: pack/unpack for replay buffer storage
    # ------------------------------------------------------------------

    def extract_state_tensor(self) -> torch.Tensor:
        """Extract current env state as a flat tensor for replay buffer storage.

        Layout: [base_nec(nb), soc0(n_batt), soc_scale_ch(n_batt),
                 soc_scale_dis(n_batt), batt_gain(n_batt), ev_gain(n_ev)]

        Returns:
            Flat tensor of shape (state_tensor_dim,), float32.
        """
        state = self._extract_state()
        parts = [
            state["base_nec"],
            state["soc0"],
            state["soc_scale_ch"],
            state["soc_scale_dis"],
            state["batt_gain"],
            state["ev_gain"],
        ]
        flat = np.concatenate(parts).astype(np.float32)
        return torch.from_numpy(flat)

    def _unpack_state_tensor(
        self,
        state_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """Unpack a flat state tensor (or batch of them) into named components.

        Args:
            state_tensor: shape (state_dim,) or (batch, state_dim)

        Returns:
            base_nec, soc0, soc_scale_ch, soc_scale_dis, batt_gain, ev_gain
            Each has shape (dim,) or (batch, dim) matching input.
        """
        m = self._mapping
        nb = m["nb"]
        n_batt = len(m["batt_gidx"])
        n_ev = len(m["ev_gidx"])

        # Offsets in the flat tensor
        o1 = nb
        o2 = o1 + n_batt
        o3 = o2 + n_batt
        o4 = o3 + n_batt
        o5 = o4 + n_batt
        # o6 = o5 + n_ev  (end)

        base_nec = state_tensor[..., :o1]
        soc0 = state_tensor[..., o1:o2]
        soc_scale_ch = state_tensor[..., o2:o3]
        soc_scale_dis = state_tensor[..., o3:o4]
        batt_gain = state_tensor[..., o4:o5]
        ev_gain = state_tensor[..., o5:]

        return base_nec, soc0, soc_scale_ch, soc_scale_dis, batt_gain, ev_gain

    # ------------------------------------------------------------------
    #  Project: differentiable action projection
    # ------------------------------------------------------------------

    def project(
        self,
        obs_tensor: torch.Tensor,
        unsafe_action: torch.Tensor,
        state_tensors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Project an unsafe action onto the safe set. Differentiable w.r.t. unsafe_action.

        Args:
            obs_tensor:    Observation tensor (not used directly).
            unsafe_action: Shape (batch, act_dim) or (act_dim,), values in [-1, 1].
            state_tensors: Optional. Shape (batch, state_dim) or (state_dim,).
                           If provided, uses these per-sample states instead of
                           reading from the live env. Required for replay buffer
                           updates where each sample was collected at a different
                           env state.

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
        if state_tensors is not None and state_tensors.dim() == 1:
            state_tensors = state_tensors.unsqueeze(0)

        batch_size = unsafe_action.shape[0]
        device = unsafe_action.device
        dtype = torch.float32

        # Get state: either from stored tensors or live env
        if state_tensors is not None:
            # Per-sample state from replay buffer
            base_batch, soc0_batch, ssc_batch, ssd_batch, bg_batch, eg_batch = (
                self._unpack_state_tensor(state_tensors.to(device=device, dtype=dtype))
            )
        else:
            # Live env state (broadcast to batch)
            state = self._extract_state()
            base_t = torch.tensor(state["base_nec"], dtype=dtype, device=device)
            soc0_t = torch.tensor(state["soc0"], dtype=dtype, device=device)
            ssc_t = torch.tensor(state["soc_scale_ch"], dtype=dtype, device=device)
            ssd_t = torch.tensor(state["soc_scale_dis"], dtype=dtype, device=device)
            bg_t = torch.tensor(state["batt_gain"], dtype=dtype, device=device)
            eg_t = torch.tensor(state["ev_gain"], dtype=dtype, device=device)

            base_batch = base_t.unsqueeze(0).expand(batch_size, -1)
            soc0_batch = soc0_t.unsqueeze(0).expand(batch_size, -1)
            ssc_batch = ssc_t.unsqueeze(0).expand(batch_size, -1)
            ssd_batch = ssd_t.unsqueeze(0).expand(batch_size, -1)
            bg_batch = bg_t.unsqueeze(0).expand(batch_size, -1)
            eg_batch = eg_t.unsqueeze(0).expand(batch_size, -1)

        t0 = time.monotonic()
        all_feasible = True

        try:
            # Batched solve: unsafe_action is already (batch, act_dim)
            (z_opt,) = self._layer(
                unsafe_action, base_batch, soc0_batch,
                ssc_batch, ssd_batch, bg_batch, eg_batch,
                solver_args={
                    "eps": self.solver_eps,
                    "max_iters": self.solver_max_iters,
                },
            )

            # Guard against NaN/Inf
            if torch.isnan(z_opt).any() or torch.isinf(z_opt).any():
                raise RuntimeError("Batched solver returned NaN/Inf")

            with torch.no_grad():
                total_delta = torch.norm(z_opt - unsafe_action, dim=-1).mean().item()

            result = z_opt

        except Exception:
            # Fallback: solve one at a time (slower but handles infeasibility)
            all_feasible = False
            safe_actions = []
            total_delta = 0.0
            for i in range(batch_size):
                u_i = unsafe_action[i]
                base_i = base_batch[i]
                soc0_i = soc0_batch[i]
                ssc_i = ssc_batch[i]
                ssd_i = ssd_batch[i]
                bg_i = bg_batch[i]
                eg_i = eg_batch[i]
                try:
                    (z_i,) = self._layer(
                        u_i, base_i, soc0_i, ssc_i, ssd_i, bg_i, eg_i,
                        solver_args={
                            "eps": self.solver_eps,
                            "max_iters": self.solver_max_iters,
                        },
                    )
                    if torch.isnan(z_i).any() or torch.isinf(z_i).any():
                        raise RuntimeError("NaN")
                    safe_actions.append(z_i)
                except Exception:
                    state_np = {
                        "base_nec": base_i.detach().cpu().numpy(),
                        "soc0": soc0_i.detach().cpu().numpy(),
                        "soc_scale_ch": ssc_i.detach().cpu().numpy(),
                        "soc_scale_dis": ssd_i.detach().cpu().numpy(),
                        "batt_gain": bg_i.detach().cpu().numpy(),
                        "ev_gain": eg_i.detach().cpu().numpy(),
                    }
                    safe_actions.append(self._fallback_clamp(u_i, state_np))
                with torch.no_grad():
                    total_delta += torch.norm(safe_actions[-1] - u_i).item()
            total_delta /= max(1, batch_size)
            result = torch.stack(safe_actions, dim=0)

        solve_ms = (time.monotonic() - t0) * 1000.0

        if squeeze_output:
            result = result.squeeze(0)

        info = {
            "projection_delta": total_delta / max(1, batch_size),
            "solve_time_ms": solve_ms,
            "feasible": all_feasible,
        }
        return result, info

    # ------------------------------------------------------------------
    #  Fallback: clamping with C2/C3/C4 enforcement when QP is infeasible
    # ------------------------------------------------------------------

    def _fallback_clamp(
        self,
        unsafe_action: torch.Tensor,
        state: Dict[str, np.ndarray],
    ) -> torch.Tensor:
        """
        Fallback when QP is infeasible. Clamp to [-1, 1] and enforce C2/C3/C4
        analytically. Returns a tensor detached from the original graph to
        prevent NaN gradients, then re-attached via straight-through estimator.
        """
        m = self._mapping
        nb = m["nb"]
        n_batt = len(m["batt_gidx"])
        n_ev = len(m["ev_gidx"])

        # Detach to block NaN gradients, then use straight-through
        z = unsafe_action.detach().clone()
        z = torch.clamp(z, -1.0, 1.0)

        # --- C2: Battery SoC bounds (asymmetric) ---
        # Upper: soc0 + z * scale_ch <= soc_high  =>  z <= (soc_high - soc0) / scale_ch
        # Lower: soc0 + z * scale_dis >= soc_low  =>  z >= (soc_low - soc0) / scale_dis
        for i in range(n_batt):
            gidx = int(m["batt_gidx"][i])
            s0 = float(state["soc0"][i])
            sch = float(state["soc_scale_ch"][i])
            sdis = float(state["soc_scale_dis"][i])
            if sch > 1e-9:
                a_hi = min((self.soc_high - s0) / sch, 1.0)
                z[gidx] = torch.clamp(z[gidx], max=a_hi)
            if sdis > 1e-9:
                a_lo = max((self.soc_low - s0) / sdis, -1.0)
                z[gidx] = torch.clamp(z[gidx], min=a_lo)

        # --- C3: Per-building net power magnitude bound ---
        # net_b = base_b + sum(batt_gain_j * z_j) + sum(ev_gain_k * z_k)
        # Enforce -p_build_max <= net_b <= p_build_max by scaling actions down
        base_nec = state["base_nec"]
        batt_gain = state["batt_gain"]
        ev_gain_arr = state["ev_gain"]

        for b_idx in range(nb):
            exo = float(base_nec[b_idx])
            import_headroom = self.p_build_max - exo   # room for positive power
            export_headroom = self.p_build_max + exo   # room for negative power

            # Collect controllable device indices and gains for this building
            batt_indices = [
                (int(m["batt_gidx"][j]), float(batt_gain[j]))
                for j in range(n_batt) if int(m["batt_bidx"][j]) == b_idx
            ]
            ev_indices = [
                (int(m["ev_gidx"][j]), float(ev_gain_arr[j]))
                for j in range(n_ev) if int(m["ev_bidx"][j]) == b_idx
            ]

            all_devices = batt_indices + ev_indices
            if not all_devices:
                continue

            # Compute current controllable power contribution
            ctrl_power = sum(
                float(z[gidx].item()) * gain for gidx, gain in all_devices
            )

            # If net exceeds p_build_max, scale down positive contributions
            net = exo + ctrl_power
            if net > self.p_build_max and ctrl_power > 1e-9:
                allowed = max(0.0, import_headroom)
                scale = allowed / ctrl_power
                for gidx, gain in all_devices:
                    val = float(z[gidx].item())
                    if val > 0:
                        z[gidx] = torch.tensor(val * scale, dtype=z.dtype)

            # If net below -p_build_max, scale down negative contributions
            net = exo + sum(
                float(z[gidx].item()) * gain for gidx, gain in all_devices
            )
            if net < -self.p_build_max:
                neg_power = sum(
                    float(z[gidx].item()) * gain
                    for gidx, gain in all_devices
                    if float(z[gidx].item()) < 0
                )
                if neg_power < -1e-9:
                    allowed = min(0.0, -export_headroom)
                    scale = allowed / neg_power
                    for gidx, gain in all_devices:
                        val = float(z[gidx].item())
                        if val < 0:
                            z[gidx] = torch.tensor(val * scale, dtype=z.dtype)

        # --- C4: Grid-level import bound ---
        # sum(net_b) <= p_grid_max
        grid_total = 0.0
        for b_idx in range(nb):
            exo = float(base_nec[b_idx])
            ctrl = 0.0
            for j in range(n_batt):
                if int(m["batt_bidx"][j]) == b_idx:
                    ctrl += float(z[int(m["batt_gidx"][j])].item()) * float(batt_gain[j])
            for j in range(n_ev):
                if int(m["ev_bidx"][j]) == b_idx:
                    ctrl += float(z[int(m["ev_gidx"][j])].item()) * float(ev_gain_arr[j])
            grid_total += exo + ctrl

        if grid_total > self.p_grid_max:
            # Scale down all positive controllable contributions proportionally
            total_pos_ctrl = 0.0
            pos_devices = []
            for j in range(n_batt):
                gidx = int(m["batt_gidx"][j])
                val = float(z[gidx].item())
                gain = float(batt_gain[j])
                if val > 0 and gain > 0:
                    total_pos_ctrl += val * gain
                    pos_devices.append((gidx, val, gain))
            for j in range(n_ev):
                gidx = int(m["ev_gidx"][j])
                val = float(z[gidx].item())
                gain = float(ev_gain_arr[j])
                if val > 0 and gain > 0:
                    total_pos_ctrl += val * gain
                    pos_devices.append((gidx, val, gain))

            if total_pos_ctrl > 1e-9:
                overshoot = grid_total - self.p_grid_max
                reduction_needed = min(overshoot, total_pos_ctrl)
                scale = max(0.0, 1.0 - reduction_needed / total_pos_ctrl)
                for gidx, val, gain in pos_devices:
                    z[gidx] = torch.tensor(val * scale, dtype=z.dtype)

        # --- Verification pass: re-enforce C2 after C3/C4 may have changed actions ---
        # C3/C4 scaling can push battery actions beyond SoC limits set in C2.
        # Re-clamp to ensure C2 is never violated.
        for i in range(n_batt):
            gidx = int(m["batt_gidx"][i])
            s0 = float(state["soc0"][i])
            sch = float(state["soc_scale_ch"][i])
            sdis = float(state["soc_scale_dis"][i])
            if sch > 1e-9:
                a_hi = min((self.soc_high - s0) / sch, 1.0)
                z[gidx] = torch.clamp(z[gidx], max=a_hi)
            if sdis > 1e-9:
                a_lo = max((self.soc_low - s0) / sdis, -1.0)
                z[gidx] = torch.clamp(z[gidx], min=a_lo)

        # Straight-through estimator: gradient flows through unsafe_action
        # but forward value is the clamped z.
        return unsafe_action + (z - unsafe_action.detach())
