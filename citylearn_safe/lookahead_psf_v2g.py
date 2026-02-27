"""
LookaheadPSF V2G: Predictive Safety Filter with V2G support.
  - EV action bounds [-1, 1] (V2G discharge allowed)
  - Multi-step C1 constraint: sum of energy over horizon >= deficit
    (QP plans charge/discharge schedule that meets departure deadline)
  - No pre-QP C1 clamps (QP is sole decision-maker for EVs)
  - No freeze_ev (QP actively controls EVs)
  - SE-RL compatible: reports action delta for penalty computation

Based on LookaheadPSFWrapper, modified for V2G thesis.
"""
from __future__ import annotations
import os, time
from typing import Any, Dict, Tuple
import numpy as np
import cvxpy as cp
import gymnasium as gym


def _unwrap_to_citylearn(env):
    cur = env
    seen = set()
    for _ in range(60):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        blds = getattr(cur, "buildings", None)
        ts = getattr(cur, "time_step", None)
        if blds is not None and hasattr(blds, "__len__") and len(blds) > 0 and ts is not None:
            return cur
        for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None


def _flatten_action(action, action_space):
    if isinstance(action, (list, tuple)):
        return np.concatenate([np.asarray(a, dtype=float).ravel() for a in action])
    return np.asarray(action, dtype=float).ravel()


def _unflatten_action(flat, action_space):
    flat = np.asarray(flat, dtype=np.float32).ravel()
    if not isinstance(action_space, list):
        return flat
    out, off = [], 0
    for sp in action_space:
        dim = int(np.prod(sp.shape))
        out.append(flat[off:off + dim].reshape(sp.shape))
        off += dim
    return out


class LookaheadPSFv2G(gym.Wrapper):
    """
    Predictive Safety Filter with V2G support using parametric MPC-style QP.

    Key differences from original LookaheadPSF:
      1. EV bounds [-1, 1] — allows V2G discharge
      2. Multi-step C1: QP plans charge/discharge schedule over H steps
         that meets departure energy requirement
      3. No pre-QP C1 clamps — QP is sole authority on EV actions
      4. No freeze_ev — QP actively optimises EV actions
      5. C3 can exclude EV power (exempt_ev_from_c3)
      6. C4 import-only option (only penalise grid import, not export)

    SE-RL integration:
      Reports 'psf_action_delta_l2' in info dict.
      The CMDP wrapper computes: reward -= w * delta^2
    """

    def __init__(self, env, horizon: int = 24,
                 soc_low=None, soc_high=None,
                 p_building_max=None, p_grid_max=None,
                 ev_efficiency: float = 0.95,
                 w_track: float = 100.0,
                 w_slack_c1: float = 5000.0,
                 w_slack_c3: float = 500.0,
                 w_slack_c4: float = 1000.0,
                 w_future_reg: float = 0.01,
                 exempt_ev_from_c3: bool = True,
                 c4_one_sided: bool = True,
                 verbose: int = 0):
        super().__init__(env)
        self.horizon = horizon
        self.soc_low = soc_low if soc_low is not None else \
            float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0"))
        self.soc_high = soc_high if soc_high is not None else \
            float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95"))
        self.p_building_max = p_building_max if p_building_max is not None else \
            float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "2.273834"))
        self.p_grid_max = p_grid_max if p_grid_max is not None else \
            float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "27.127751"))
        self.ev_efficiency = ev_efficiency
        # PSF_W_TRACK_ENVVAR_PATCH: allow env var override
        _w_track_override = os.environ.get("PSF_W_TRACK", "")
        self.w_track = float(_w_track_override) if _w_track_override else w_track
        self.w_slack_c1 = w_slack_c1
        self.w_slack_c3 = w_slack_c3
        self.w_slack_c4 = w_slack_c4
        self.w_future_reg = w_future_reg
        self.exempt_ev_from_c3 = exempt_ev_from_c3
        self.c4_one_sided = c4_one_sided
        self.verbose = verbose

        self._prob = None
        self._compiled = False
        self._mapping = None
        self._step_count = 0
        self._episode_count = 0
        self._total_interventions = 0
        self._total_ev_interventions = 0
        self._total_batt_interventions = 0
        self._total_solve_ms = 0.0
        self._infeasible_count = 0

        if self.verbose >= 1:
            print(f"[PSF-V2G] Init: H={horizon} soc=[{self.soc_low},{self.soc_high}] "
                  f"p_bld={self.p_building_max:.2f} p_grid={self.p_grid_max:.2f} "
                  f"ev_exempt_c3={self.exempt_ev_from_c3} c4_one_sided={self.c4_one_sided}")
            print(f"[PSF-V2G] Weights: track={w_track} c1={w_slack_c1} "
                  f"c3={w_slack_c3} c4={w_slack_c4} reg={w_future_reg}")

    def _build_mapping(self):
        city = _unwrap_to_citylearn(self.env)
        if city is None:
            raise RuntimeError("[PSF-V2G] Cannot find CityLearn env")
        names_raw = getattr(city, "action_names", [])
        buildings = getattr(city, "buildings", [])
        nb = len(buildings)
        m = {"agent_dims": [], "batt_gidx": [], "ev_chargers": [],
             "gidx_to_building": {}}

        is_central = (len(names_raw) == 1 and isinstance(names_raw[0], list)
                      and nb > 1 and len(names_raw[0]) > nb)
        if is_central:
            flat_names = names_raw[0]
            batt_positions = [i for i, n in enumerate(flat_names)
                              if str(n).lower() == "electrical_storage"]
            if len(batt_positions) == nb:
                reconstructed = []
                for b_idx in range(nb):
                    start = batt_positions[b_idx]
                    end = batt_positions[b_idx + 1] if b_idx + 1 < nb else len(flat_names)
                    reconstructed.append(list(flat_names[start:end]))
                names_raw = reconstructed

        g = 0
        for b_idx, sub in enumerate(names_raw):
            if not isinstance(sub, list):
                sub = [sub]
            m["agent_dims"].append(len(sub))
            batt_found = False
            for n in sub:
                n_str = str(n).lower()
                m["gidx_to_building"][g] = b_idx
                if n_str == "electrical_storage" and not batt_found:
                    m["batt_gidx"].append(g)
                    batt_found = True
                if "electric_vehicle_storage_charger_" in n_str:
                    m["ev_chargers"].append({"gidx": g, "b_idx": b_idx})
                g += 1
            if not batt_found:
                m["batt_gidx"].append(-1)
        m["total_dim"] = g
        return m

    def _compile_qp(self):
        city = _unwrap_to_citylearn(self.env)
        self._mapping = self._build_mapping()
        self._nb = len(city.buildings)
        nb = self._nb
        H = self.horizon

        self._batt_list = [(b_idx, g)
                           for b_idx, g in enumerate(self._mapping["batt_gidx"])
                           if g >= 0]
        n_batt = len(self._batt_list)
        self._n_batt = n_batt
        self._max_ev = len(self._mapping["ev_chargers"])
        n_ev = self._max_ev

        self._batt_by_bldg = {}
        for bi, (b_idx, g) in enumerate(self._batt_list):
            self._batt_by_bldg[b_idx] = bi
        self._ev_by_bldg = {}
        for ei, evc in enumerate(self._mapping["ev_chargers"]):
            self._ev_by_bldg.setdefault(evc["b_idx"], []).append(ei)

        self._v_a_batt = cp.Variable((n_batt, H))
        self._v_a_ev = cp.Variable((n_ev, H)) if n_ev > 0 else None
        self._v_slack_c3 = cp.Variable((nb, H), nonneg=True)
        self._v_slack_c4 = cp.Variable(H, nonneg=True)
        self._v_slack_c1 = cp.Variable(n_ev, nonneg=True) if n_ev > 0 else None

        self._p_proposed_batt = cp.Parameter(n_batt, value=np.zeros(n_batt))
        self._p_proposed_ev = cp.Parameter(n_ev, value=np.zeros(n_ev)) \
            if n_ev > 0 else None
        self._p_soc0 = cp.Parameter(n_batt, value=np.zeros(n_batt))
        self._p_soc_scale = cp.Parameter(n_batt, value=np.ones(n_batt), pos=True)
        self._p_batt_gain = cp.Parameter(n_batt, value=np.ones(n_batt), pos=True)
        self._p_base_net = cp.Parameter((nb, H), value=np.zeros((nb, H)))
        self._p_grid_limit = cp.Parameter(value=float(self.p_grid_max), nonneg=True)

        if n_ev > 0:
            self._p_ev_gain = cp.Parameter((n_ev, H),
                                           value=np.zeros((n_ev, H)), nonneg=True)
            self._p_ev_c1_coeff = cp.Parameter((n_ev, H),
                                               value=np.zeros((n_ev, H)), nonneg=True)
            self._p_ev_deficit = cp.Parameter(n_ev,
                                              value=np.zeros(n_ev), nonneg=True)

        constraints = []
        constraints += [self._v_a_batt >= -1.0, self._v_a_batt <= 1.0]
        if n_ev > 0:
            constraints += [self._v_a_ev >= -1.0, self._v_a_ev <= 1.0]

        for bi in range(n_batt):
            for k in range(H):
                soc_k = self._p_soc0[bi] + \
                    cp.sum(self._v_a_batt[bi, :k+1]) * self._p_soc_scale[bi]
                constraints += [soc_k >= self.soc_low, soc_k <= self.soc_high]

        _net_full_by_k = {k: [] for k in range(H)}

        for b_idx in range(nb):
            for k in range(H):
                net_batt_only = self._p_base_net[b_idx, k]
                if b_idx in self._batt_by_bldg:
                    bi = self._batt_by_bldg[b_idx]
                    net_batt_only = net_batt_only + \
                        self._v_a_batt[bi, k] * self._p_batt_gain[bi]

                net_full = net_batt_only
                if n_ev > 0:
                    for ei in self._ev_by_bldg.get(b_idx, []):
                        net_full = net_full + \
                            self._v_a_ev[ei, k] * self._p_ev_gain[ei, k]

                if self.exempt_ev_from_c3:
                    net_c3 = net_batt_only
                else:
                    net_c3 = net_full
                constraints += [
                    net_c3 <= self.p_building_max + self._v_slack_c3[b_idx, k],
                    net_c3 >= -self.p_building_max - self._v_slack_c3[b_idx, k],
                ]

                _net_full_by_k[k].append(net_full)

        for k in range(H):
            grid_sum_k = sum(_net_full_by_k[k])
            if self.c4_one_sided:
                constraints += [grid_sum_k <= self._p_grid_limit + self._v_slack_c4[k]]
            else:
                constraints += [
                    grid_sum_k <= self._p_grid_limit + self._v_slack_c4[k],
                    grid_sum_k >= -self._p_grid_limit - self._v_slack_c4[k],
                ]

        if n_ev > 0:
            for ei in range(n_ev):
                energy = cp.sum(
                    cp.multiply(self._v_a_ev[ei, :], self._p_ev_c1_coeff[ei, :]))
                constraints += [
                    energy >= self._p_ev_deficit[ei] - self._v_slack_c1[ei]
                ]

        obj = self.w_track * cp.sum_squares(
            self._v_a_batt[:, 0] - self._p_proposed_batt)
        if n_ev > 0:
            obj += self.w_track * cp.sum_squares(
                self._v_a_ev[:, 0] - self._p_proposed_ev)

        if n_ev > 0:
            obj += self.w_slack_c1 * cp.sum(self._v_slack_c1)
        obj += self.w_slack_c3 * cp.sum(self._v_slack_c3)
        obj += self.w_slack_c4 * cp.sum(self._v_slack_c4)

        if H > 1:
            obj += self.w_future_reg * cp.sum_squares(self._v_a_batt[:, 1:])
            if n_ev > 0:
                obj += self.w_future_reg * cp.sum_squares(self._v_a_ev[:, 1:])

        self._prob = cp.Problem(cp.Minimize(obj), constraints)
        assert self._prob.is_dpp(), \
            "[PSF-V2G] Problem must be DPP for fast parametric solves!"
        self._compiled = True

        if self.verbose >= 1:
            n_vars = sum(v.size for v in self._prob.variables())
            print(f"[PSF-V2G] Compiled DPP: {n_batt} batt, {n_ev} ev, "
                  f"H={H}, {n_vars} vars, {len(constraints)} constraints, "
                  f"EV bounds=[-1,1]")

    def _update_params_and_solve(self, proposed_flat):
        t0 = time.time()
        city = _unwrap_to_citylearn(self.env)
        buildings = list(getattr(city, "buildings", []))
        H = self.horizon
        t_now = int(getattr(city, "time_step", 0))
        t_idx = max(0, t_now - 1)

        soc0 = np.zeros(self._n_batt)
        soc_scale = np.ones(self._n_batt)
        batt_gain = np.ones(self._n_batt)
        proposed_batt = np.zeros(self._n_batt)

        for bi, (b_idx, gidx) in enumerate(self._batt_list):
            es = getattr(buildings[b_idx], "electrical_storage", None)
            if es is None:
                continue
            cap = float(getattr(es, "capacity", 6.4))
            nom_p = float(getattr(es, "nominal_power", 5.0))
            soc_arr = np.asarray(getattr(es, "soc", []), dtype=float)
            soc0[bi] = float(np.clip(soc_arr[t_idx], 0.0, 1.0)) \
                if 0 <= t_idx < len(soc_arr) else 0.5
            soc_scale[bi] = max(1e-6, nom_p / max(1e-6, cap))
            batt_gain[bi] = max(1e-6, nom_p)
            proposed_batt[bi] = float(proposed_flat[gidx])

        self._p_soc0.value = soc0
        self._p_soc_scale.value = soc_scale
        self._p_batt_gain.value = batt_gain
        self._p_proposed_batt.value = proposed_batt

        base_net = np.zeros((self._nb, H))
        for b_idx, b in enumerate(buildings):
            raw_nsl = np.asarray(
                getattr(b, "_Building__energy_to_non_shiftable_load", []),
                dtype=float)
            raw_sg = np.asarray(
                getattr(b, "_Building__solar_generation", []), dtype=float)
            T_max = min(len(raw_nsl), len(raw_sg))
            for k in range(H):
                idx = t_now + k
                if idx < T_max:
                    base_net[b_idx, k] = raw_nsl[idx] + raw_sg[idx]
                elif T_max > 0:
                    base_net[b_idx, k] = raw_nsl[T_max - 1] + raw_sg[T_max - 1]
        self._p_base_net.value = base_net

        if self._max_ev > 0:
            ev_gain = np.zeros((self._max_ev, H))
            ev_c1_coeff = np.zeros((self._max_ev, H))
            ev_deficit = np.zeros(self._max_ev)
            proposed_ev = np.zeros(self._max_ev)

            for ei, evc in enumerate(self._mapping["ev_chargers"]):
                b_idx = evc["b_idx"]
                gidx = evc["gidx"]
                proposed_ev[ei] = float(proposed_flat[gidx]) \
                    if gidx < len(proposed_flat) else 0.0

                b = buildings[b_idx]
                chargers = getattr(b, "electric_vehicle_chargers", None) or []
                ev_in_bldg = [e for e in self._mapping["ev_chargers"]
                              if e["b_idx"] == b_idx]
                local_idx = next(
                    (li for li, e in enumerate(ev_in_bldg)
                     if e["gidx"] == gidx), None)
                if local_idx is None or local_idx >= len(chargers):
                    continue
                ch = chargers[local_idx]

                max_p = getattr(ch, "max_charging_power",
                                getattr(ch, "_Charger__max_charging_power", 0))
                if isinstance(max_p, np.ndarray):
                    max_p = float(max_p.ravel()[0])
                elif max_p is not None:
                    max_p = float(max_p)
                else:
                    max_p = 0.0

                sim = getattr(ch, "charger_simulation",
                              getattr(ch, "_Charger__charger_simulation", None))
                if sim is None:
                    continue
                try:
                    state_arr = np.asarray(
                        getattr(sim, "_electric_vehicle_charger_state"),
                        dtype=float)
                    dep_arr = np.asarray(
                        getattr(sim, "_electric_vehicle_departure_time"),
                        dtype=float)
                    req_arr = np.asarray(
                        getattr(sim, "_electric_vehicle_required_soc_departure"),
                        dtype=float)
                    ev_id_arr = getattr(sim, "_electric_vehicle_id")
                except (AttributeError, TypeError):
                    continue

                if t_now >= len(state_arr):
                    continue
                if float(state_arr[t_now]) != 1.0:
                    continue
                if ev_id_arr[t_now] is None:
                    continue

                dep_raw = float(dep_arr[t_now])
                tau = 999
                if np.isfinite(dep_raw) and dep_raw > 0:
                    tau = max(1, int(dep_raw))

                req_soc = float(req_arr[t_now])
                if not np.isfinite(req_soc):
                    req_soc = 1.0
                req_soc = np.clip(req_soc, 0.0, 1.0)

                ev_soc = 0.0
                ev_cap = 0.0
                ev_obj = getattr(ch, "connected_electric_vehicle", None)
                if ev_obj is not None:
                    batt = getattr(ev_obj, "battery", None)
                    if batt is not None:
                        ev_cap = float(getattr(batt, "capacity", 0) or 0)
                        soc_data = getattr(batt, "soc", None)
                        if soc_data is not None:
                            soc_np = np.asarray(soc_data, dtype=float)
                            if 0 <= t_idx < len(soc_np):
                                ev_soc = float(np.clip(soc_np[t_idx], 0.0, 1.0))

                ev_deficit[ei] = max(0.0, (req_soc - ev_soc) * ev_cap)

                for k in range(H):
                    if k < tau:
                        ev_gain[ei, k] = max_p
                        ev_c1_coeff[ei, k] = max_p * self.ev_efficiency

            self._p_ev_gain.value = ev_gain
            self._p_ev_c1_coeff.value = ev_c1_coeff
            self._p_ev_deficit.value = ev_deficit
            self._p_proposed_ev.value = proposed_ev

        self._p_grid_limit.value = float(self.p_grid_max)

        status = "unknown"
        try:
            self._prob.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                             max_iter=5000, eps_abs=1e-3, eps_rel=1e-3)
            status = str(self._prob.status)
        except Exception as ex:
            status = f"solve_error: {str(ex)[:80]}"

        solve_ms = (time.time() - t0) * 1000.0
        is_optimal = status in ("optimal", "optimal_inaccurate")

        corrected = proposed_flat.copy()
        ev_interv = 0
        batt_interv = 0

        if is_optimal and self._v_a_batt.value is not None:
            for bi, (b_idx, gidx) in enumerate(self._batt_list):
                new_val = float(np.clip(self._v_a_batt.value[bi, 0], -1.0, 1.0))
                old_val = corrected[gidx]
                corrected[gidx] = new_val
                if abs(new_val - old_val) > 1e-4:
                    batt_interv += 1

            if self._max_ev > 0 and self._v_a_ev is not None \
                    and self._v_a_ev.value is not None:
                for ei, evc in enumerate(self._mapping["ev_chargers"]):
                    g = evc["gidx"]
                    if 0 <= g < len(corrected):
                        new_val = float(np.clip(
                            self._v_a_ev.value[ei, 0], -1.0, 1.0))
                        old_val = corrected[g]
                        corrected[g] = new_val
                        if abs(new_val - old_val) > 1e-4:
                            ev_interv += 1
        else:
            for bi, (b_idx, gidx) in enumerate(self._batt_list):
                a_min = max((self.soc_low - soc0[bi]) /
                            max(1e-9, soc_scale[bi]), -1.0)
                a_max = min((self.soc_high - soc0[bi]) /
                            max(1e-9, soc_scale[bi]), 1.0)
                old_val = corrected[gidx]
                corrected[gidx] = float(np.clip(corrected[gidx], a_min, a_max))
                if abs(corrected[gidx] - old_val) > 1e-4:
                    batt_interv += 1
            self._infeasible_count += 1

        for bi, (b_idx, gidx) in enumerate(self._batt_list):
            es = getattr(buildings[b_idx], "electrical_storage", None)
            if es is None:
                continue
            soc_arr = np.asarray(getattr(es, "soc", []), dtype=float)
            soc = float(np.clip(soc_arr[t_idx], 0.0, 1.0)) \
                if 0 <= t_idx < len(soc_arr) else 0.5
            cap = float(getattr(es, "capacity", 6.4))
            nom = float(getattr(es, "nominal_power", 5.0))
            sc = max(1e-9, nom / max(1e-6, cap))
            a_lo = max((self.soc_low - soc) / sc, -1.0)
            a_hi = min((self.soc_high - soc) / sc, 1.0)
            corrected[gidx] = float(np.clip(corrected[gidx], a_lo, a_hi))

        delta_l2 = float(np.linalg.norm(corrected - proposed_flat))
        any_intervention = delta_l2 > 1e-4

        info = {
            "psf_active": 1.0,
            "psf_any_intervention": float(any_intervention),
            "psf_action_delta_l2": delta_l2,
            "psf_ev_interventions": float(ev_interv),
            "psf_batt_interventions": float(batt_interv),
            "psf_solve_ms": solve_ms,
            "psf_status": status,
            "psf_infeasible": 0.0 if is_optimal else 1.0,
        }

        if self.verbose >= 2 and self._max_ev > 0 and is_optimal \
                and self._v_a_ev.value is not None:
            ev_acts = [float(self._v_a_ev.value[ei, 0])
                       for ei in range(self._max_ev)]
            n_discharge = sum(1 for a in ev_acts if a < -0.01)
            n_charge = sum(1 for a in ev_acts if a > 0.01)
            if n_discharge > 0 or self._step_count % 500 == 0:
                print(f"[PSF-V2G] step={self._step_count} "
                      f"EV actions: {[f'{a:.2f}' for a in ev_acts]} "
                      f"(charge={n_charge} discharge={n_discharge})")

        info["shielded_action"] = corrected.copy()

        return corrected, info

    def step(self, action):
        if not self._compiled:
            self._compile_qp()

        is_multi = isinstance(self.action_space, list)
        if is_multi:
            action_flat = np.concatenate(
                [np.asarray(a, dtype=float).ravel() for a in action])
            lo = np.concatenate(
                [np.asarray(sp.low).ravel() for sp in self.action_space])
            hi = np.concatenate(
                [np.asarray(sp.high).ravel() for sp in self.action_space])
        else:
            action_flat = np.asarray(action, dtype=float).ravel()
            lo = np.asarray(self.action_space.low).ravel()
            hi = np.asarray(self.action_space.high).ravel()

        action_flat = np.clip(action_flat, lo, hi)

        corrected_flat, psf_info = self._update_params_and_solve(action_flat)
        corrected_flat = np.clip(corrected_flat, lo, hi)

        corrected_action = _unflatten_action(corrected_flat, self.action_space) \
            if is_multi else corrected_flat

        obs, reward, term, trunc, info = self.env.step(corrected_action)
        info = dict(info) if info else {}
        info.update(psf_info)

        self._step_count += 1
        self._total_solve_ms += psf_info.get("psf_solve_ms", 0)
        if psf_info.get("psf_any_intervention", False):
            self._total_interventions += 1
        self._total_ev_interventions += int(
            psf_info.get("psf_ev_interventions", 0))
        self._total_batt_interventions += int(
            psf_info.get("psf_batt_interventions", 0))

        if self.verbose >= 1 and self._step_count % 1000 == 0:
            avg_ms = self._total_solve_ms / max(1, self._step_count)
            pct = 100.0 * self._total_interventions / max(1, self._step_count)
            print(f"[PSF-V2G] step={self._step_count} "
                  f"interventions={self._total_interventions} ({pct:.1f}%) "
                  f"ev={self._total_ev_interventions} "
                  f"batt={self._total_batt_interventions} "
                  f"infeasible={self._infeasible_count} "
                  f"avg_solve={avg_ms:.1f}ms")

        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        if self._step_count > 0 and self.verbose >= 1:
            avg_ms = self._total_solve_ms / max(1, self._step_count)
            pct = 100.0 * self._total_interventions / max(1, self._step_count)
            print(f"[PSF-V2G] Ep{self._episode_count} done | "
                  f"steps={self._step_count} "
                  f"interventions={self._total_interventions} ({pct:.1f}%) "
                  f"(ev={self._total_ev_interventions} "
                  f"batt={self._total_batt_interventions}) "
                  f"infeasible={self._infeasible_count} "
                  f"avg_solve={avg_ms:.1f}ms")

        self._step_count = 0
        self._episode_count += 1
        self._total_interventions = 0
        self._total_ev_interventions = 0
        self._total_batt_interventions = 0
        self._total_solve_ms = 0.0
        self._infeasible_count = 0
        return self.env.reset(**kwargs)
