from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import time
import numpy as np
import cvxpy as cp

from .psf_extract import (
    unwrap_to_citylearn, action_name_list, build_ev_action_map,
    get_ev_states, flatten_action, unflatten_action
)

@dataclass
class PSFConfig:
    horizon: int = 24
    soc_low: float = 0.0
    soc_high: float = 0.95
    p_building_max: float = 2.273834
    p_grid_max: float = 27.127751
    eta_ev: float = 0.95
    w_energy: float = 1.0
    w_track: float = 1000.0

class PredictiveSafetyFilter:
    def __init__(self, env: Any, cfg: PSFConfig):
        self.env = env
        self.cfg = cfg
        raw = unwrap_to_citylearn(env)
        names = action_name_list(raw)
        self.ev_map = build_ev_action_map(names)
        self.last_safe_action = None

    def reset(self):
        self.last_safe_action = None

    def filter(self, action_rl) -> Tuple[Any, Dict[str, float]]:
        t0 = time.time()
        a_rl = flatten_action(action_rl, self.env.action_space).astype(float, copy=True)

        raw = unwrap_to_citylearn(self.env)
        buildings = getattr(raw, "buildings", []) or []
        nb = len(buildings)
        N = int(self.cfg.horizon)

        # EV states only (MVP): safe EV charging w/ deadline reachability
        evs = get_ev_states(raw, self.ev_map)
        ne = len(evs)

        if ne == 0:
            return action_rl, {
                "psf_active": 0.0, "psf_status": "no_ev", "psf_delta_l2": 0.0,
                "psf_solve_ms": 0.0, "psf_used_fallback": 0.0, "psf_infeasible": 0.0,
                "psf_ev_active": 0.0, "psf_horizon": float(N),
            }

        # Variables
        P_ev = cp.Variable((ne, N), nonneg=True)
        Edef = cp.Variable((ne, N+1))

        constraints = []
        Edef0 = np.array([e.E_def_kwh for e in evs], dtype=float)
        constraints += [Edef[:, 0] == Edef0]

        dt = 1.0
        Pmax = np.array([e.Pmax_kw for e in evs], dtype=float)

        for k in range(N):
            constraints += [
                Edef[:, k+1] == cp.maximum(0, Edef[:, k] - self.cfg.eta_ev * P_ev[:, k] * dt),
                P_ev[:, k] <= Pmax
            ]

        # Deadline constraints
        for j, e in enumerate(evs):
            tau = int(e.tau_steps)
            if tau <= N:
                constraints += [Edef[j, tau] <= 1e-3]
            else:
                remaining = (tau - N) * float(e.Pmax_kw) * self.cfg.eta_ev * dt
                constraints += [Edef[j, N] <= remaining + 1e-3]

        # Objective: small energy + small control (MVP)
        obj = self.cfg.w_track * cp.sum_squares(P_ev) + self.cfg.w_energy * cp.sum(P_ev)

        prob = cp.Problem(cp.Minimize(obj), constraints)

        status = "unknown"
        infeasible = 0.0
        used_fallback = 0.0

        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=20000)
            status = str(prob.status)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                infeasible = 1.0
        except Exception:
            status = "solve_error"
            infeasible = 1.0

        if infeasible > 0.0:
            used_fallback = 1.0
            if self.last_safe_action is not None:
                a_safe = self.last_safe_action.copy()
            else:
                a_safe = a_rl.copy()
        else:
            a_safe = a_rl.copy()
            pev0 = np.array(P_ev.value[:, 0]).reshape(-1)
            frac = np.clip(pev0 / np.maximum(1e-6, Pmax), 0.0, 1.0)
            for j, e in enumerate(evs):
                a_safe[int(e.action_idx)] = float(frac[j])
            self.last_safe_action = a_safe.copy()

        delta = float(np.linalg.norm(a_safe - a_rl))
        ms = (time.time() - t0) * 1000.0

        info = {
            "psf_active": 1.0,
            "psf_status": status,
            "psf_delta_l2": delta,
            "psf_solve_ms": float(ms),
            "psf_used_fallback": used_fallback,
            "psf_infeasible": infeasible,
            "psf_ev_active": float(ne),
            "psf_horizon": float(N),
        }
        return unflatten_action(a_safe, self.env.action_space), info
