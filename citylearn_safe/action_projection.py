"""
ActionProjectionWrapper: Training-time safety shield using lexicographic QP.
3-stage: Stage0=C1+C2 hard, Stage1=min C3/C4 slack, Stage2=closest to RL.
Toggle: CITYLEARN_USE_SHIELD=1, SHIELD_ENABLE_C3=1, SHIELD_ENABLE_C4=1

V2G-enabled version (v2):
  - EV action bounds [-1, 1] to allow V2G discharge
  - Urgency-aware C1 minimum actions (slack-based, not greedy)
  - C4 import-only (no penalty for grid export)
  - C3 optionally excludes EV power (EV exempt from building limit)
"""
from __future__ import annotations
import os, time, warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
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


class ActionProjectionWrapper(gym.Wrapper):
    """
    Training-time safety shield with V2G support.

    3-stage lexicographic QP:
      Stage 0: Satisfy C1 (EV deadline) + C2 (battery SoC) as hard constraints,
               closest to RL proposal.
      Stage 1: Minimise C3 (building power) + C4 (grid power) slack,
               subject to C1+C2.
      Stage 2: Find action closest to RL proposal,
               subject to C1+C2 and C3/C4 slack ≤ Stage1 optimum.

    V2G changes vs original:
      1. EV bounds [-1, 1] instead of [0, 1]
      2. C1 min_action is urgency-aware (allows V2G when slack > 2h)
      3. C4 is one-sided (only penalises import, not export)
      4. C3 can exclude EV power (SHIELD_C3_EV_EXEMPT=1)
    """

    def __init__(self, env, soc_low=None, soc_high=None,
                 p_building_max=None, p_grid_max=None,
                 enable_c3=None, enable_c4=None,
                 c3_ev_exempt=None, c4_one_sided=None,
                 solver=None, slack_tol=1e-5, verbose=None):
        super().__init__(env)
        self.soc_low = soc_low if soc_low is not None else float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0"))
        self.soc_high = soc_high if soc_high is not None else float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95"))
        self.p_building_max = p_building_max if p_building_max is not None else float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "2.273834"))
        self.p_grid_max = p_grid_max if p_grid_max is not None else float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "27.127751"))
        self.enable_c3 = enable_c3 if enable_c3 is not None else (os.environ.get("SHIELD_ENABLE_C3", "1") == "1")
        self.enable_c4 = enable_c4 if enable_c4 is not None else (os.environ.get("SHIELD_ENABLE_C4", "1") == "1")
        # --- V2G options ---
        self.c3_ev_exempt = c3_ev_exempt if c3_ev_exempt is not None else (os.environ.get("SHIELD_C3_EV_EXEMPT", "1") == "1")
        self.c4_one_sided = c4_one_sided if c4_one_sided is not None else (os.environ.get("SHIELD_C4_ONE_SIDED", "1") == "1")
        self.solver = solver or os.environ.get("SHIELD_SOLVER", "OSQP")
        self.slack_tol = float(slack_tol)
        self.verbose = verbose if verbose is not None else int(os.environ.get("SHIELD_VERBOSE", "0"))
        self._compiled = False
        self._mapping = None
        self._cvx = None
        self._step_count = 0
        self._episode_count = 0
        self._total_interventions = 0
        self._total_solve_ms = 0.0
        self._total_infeasible = 0
        self._c1_interventions = 0
        self._c2_interventions = 0
        if self.verbose >= 1:
            print(f"[Shield-V2G] Init: soc=[{self.soc_low},{self.soc_high}] "
                  f"p_bld={self.p_building_max} p_grid={self.p_grid_max} "
                  f"C3={'ON' if self.enable_c3 else 'OFF'} "
                  f"C4={'ON' if self.enable_c4 else 'OFF'} "
                  f"C3_ev_exempt={self.c3_ev_exempt} "
                  f"C4_one_sided={self.c4_one_sided}")

    # ------------------------------------------------------------------ #
    #  Mapping: discover battery / EV positions in flat action vector     #
    # ------------------------------------------------------------------ #
    def _build_mapping(self):
        city = _unwrap_to_citylearn(self.env)
        if city is None:
            raise RuntimeError("[Shield] Cannot find CityLearn env")
        names_raw = getattr(city, "action_names", None)
        buildings = list(getattr(city, "buildings", []))
        nb = len(buildings)
        is_central = (isinstance(names_raw, list) and len(names_raw) == 1
                      and isinstance(names_raw[0], list) and nb > 1 and len(names_raw[0]) > nb)
        if is_central:
            flat_names = names_raw[0]
            batt_positions = [i for i, n in enumerate(flat_names) if str(n).lower() == "electrical_storage"]
            if len(batt_positions) == nb:
                reconstructed = []
                for b in range(nb):
                    start = batt_positions[b]
                    end = batt_positions[b + 1] if b + 1 < nb else len(flat_names)
                    reconstructed.append(list(flat_names[start:end]))
                names_raw = reconstructed
        g = 0
        batt, ev = [], []
        for b_idx, sub in enumerate(names_raw):
            if not isinstance(sub, list): sub = [sub]
            local_ev, batt_found = 0, False
            for n in sub:
                nl = str(n).lower()
                if nl == "electrical_storage" and not batt_found:
                    batt.append({"b_idx": b_idx, "gidx": g})
                    batt_found = True
                if "electric_vehicle_storage_charger_" in nl:
                    ev.append({"b_idx": b_idx, "gidx": g, "local_ev_idx": local_ev})
                    local_ev += 1
                g += 1
        if self.verbose >= 1:
            print(f"[Shield-V2G] Mapping: {nb} bldg, {len(batt)} batt, {len(ev)} ev, {g} total")
        return {"nb": nb, "batt": batt, "ev": ev, "total_dim": g}

    # ------------------------------------------------------------------ #
    #  C1: Urgency-aware EV minimum actions (V2G-compatible)             #
    #                                                                     #
    #  Instead of greedy deficit/T, uses slack = T_remaining - T_needed:  #
    #    slack > 2h  → no minimum (V2G allowed)                          #
    #    0 < slack ≤ 2h → gentle ramp proportional to urgency            #
    #    slack ≤ 0   → must charge now                                   #
    #    dep_hours ≤ 0 → departing now, skip (charge won't help)         #
    # ------------------------------------------------------------------ #
    def _compute_c1_min_actions(self, city):
        ev_list = self._mapping["ev"]
        mins = np.zeros(len(ev_list), dtype=float)
        buildings = list(getattr(city, "buildings", []))
        t_now = int(getattr(city, "time_step", 0))
        t_idx = max(0, t_now - 1)

        for ei, einfo in enumerate(ev_list):
            b_idx, local = einfo["b_idx"], einfo["local_ev_idx"]
            if b_idx >= len(buildings):
                continue
            chargers = getattr(buildings[b_idx], "electric_vehicle_chargers", None) or []
            if local >= len(chargers):
                continue
            ch = chargers[local]
            sim = getattr(ch, "charger_simulation",
                          getattr(ch, "_Charger__charger_simulation", None))
            if sim is None:
                continue

            # --- Read EV state ---
            try:
                state_arr = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                dep_arr = np.asarray(getattr(sim, "_electric_vehicle_departure_time"), dtype=float)
                req_arr = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
                ev_id_arr = getattr(sim, "_electric_vehicle_id")
            except Exception:
                continue
            if t_now >= len(state_arr) or float(state_arr[t_now]) != 1.0:
                continue
            if ev_id_arr[t_now] is None:
                continue

            dep_hours = float(dep_arr[t_now]) if t_now < len(dep_arr) else float("nan")
            req_soc = float(req_arr[t_now]) if t_now < len(req_arr) else 1.0
            if not np.isfinite(req_soc):
                req_soc = 1.0
            req_soc = np.clip(req_soc, 0.0, 1.0)

            # FIX: dep_hours <= 0 means departing NOW — charging this step won't help
            if not np.isfinite(dep_hours) or dep_hours <= 0:
                continue

            # --- Read EV SoC and capacity ---
            ev_soc, ev_cap = 0.0, 0.0
            ev_obj = getattr(ch, "connected_electric_vehicle", None)
            if ev_obj is not None:
                batt_obj = getattr(ev_obj, "battery", None)
                if batt_obj is not None:
                    ev_cap = float(getattr(batt_obj, "capacity", 0) or 0.0)
                    soc_data = getattr(batt_obj, "soc", None)
                    if soc_data is not None:
                        soc_np = np.asarray(soc_data, dtype=float).ravel()
                        if 0 <= t_idx < len(soc_np):
                            ev_soc = float(np.clip(soc_np[t_idx], 0.0, 1.0))
            if ev_cap <= 0:
                continue

            # --- Read charger specs ---
            max_p = getattr(ch, "max_charging_power",
                            getattr(ch, "_Charger__max_charging_power", 0))
            if isinstance(max_p, np.ndarray):
                max_p = float(max_p.ravel()[0])
            max_p = float(max_p or 0.0)
            if max_p <= 0:
                continue
            eff = 0.95
            try:
                eff = float(ch.get_efficiency(1.0, True))
            except Exception:
                pass
            if not np.isfinite(eff) or eff <= 0:
                eff = 0.95
            rte = float(getattr(
                getattr(ev_obj, "battery", None), "round_trip_efficiency", 1.0) or 1.0)

            # --- Urgency calculation ---
            energy_needed_kwh = max(0.0, (req_soc - ev_soc) * ev_cap)
            charge_rate_kwh_per_h = max_p * eff * rte

            # Already charged enough → no minimum, V2G fully allowed
            if energy_needed_kwh < 0.01:
                continue
            if charge_rate_kwh_per_h <= 0:
                mins[ei] = 0.01
                continue

            hours_needed = (energy_needed_kwh / charge_rate_kwh_per_h) * 1.3  # 30% safety margin
            slack = dep_hours - hours_needed

            if slack > 2.0:
                # Plenty of time → no minimum, V2G allowed
                continue
            elif slack > 0:
                # Getting tight → gentle ramp
                urgency = 1.0 - (slack / 2.0)  # 0 at slack=2, 1 at slack=0
                uniform_rate = energy_needed_kwh / (charge_rate_kwh_per_h * dep_hours)
                mins[ei] = float(np.clip(urgency * uniform_rate * 1.2, 0.01, 0.6))
            else:
                # No slack → must charge now
                mins[ei] = float(np.clip(
                    (energy_needed_kwh / (charge_rate_kwh_per_h * max(0.5, dep_hours))) * 1.3,
                    0.2, 1.0))

        return mins

    # ------------------------------------------------------------------ #
    #  Compile 3-stage CVXPY problem (V2G-enabled)                       #
    # ------------------------------------------------------------------ #
    def _compile(self):
        import cvxpy as cp
        self._mapping = self._build_mapping()
        nb = self._mapping["nb"]
        batt, ev = self._mapping["batt"], self._mapping["ev"]
        n_batt, n_ev = len(batt), len(ev)

        # Building-to-actuator mapping matrices
        A_batt = np.zeros((nb, n_batt))
        for bi, binfo in enumerate(batt):
            A_batt[binfo["b_idx"], bi] = 1.0
        A_ev = np.zeros((nb, n_ev))
        for ei, einfo in enumerate(ev):
            A_ev[einfo["b_idx"], ei] = 1.0

        # Parameters (updated each step)
        p_base = cp.Parameter(nb, value=np.zeros(nb))
        p_soc0 = cp.Parameter(n_batt, value=np.zeros(n_batt))
        p_soc_scale = cp.Parameter(n_batt, value=np.ones(n_batt), pos=True)
        p_batt_gain = cp.Parameter(n_batt, value=np.ones(n_batt), pos=True)
        p_a_batt_rl = cp.Parameter(n_batt, value=np.zeros(n_batt))
        p_build_max = cp.Parameter(nonneg=True, value=float(self.p_building_max))
        p_grid_max = cp.Parameter(nonneg=True, value=float(self.p_grid_max))

        ev_p = {}
        if n_ev > 0:
            ev_p["p_ev_gain"] = cp.Parameter(n_ev, value=np.zeros(n_ev), nonneg=True)
            ev_p["p_a_ev_rl"] = cp.Parameter(n_ev, value=np.zeros(n_ev))
            ev_p["p_c1_min"] = cp.Parameter(n_ev, value=np.zeros(n_ev))  # removed nonneg: can be 0

        # ============================================================== #
        #  STAGE 0: Hard C1 + C2, closest to RL                         #
        # ============================================================== #
        a_b0 = cp.Variable(n_batt)
        cons0 = [a_b0 >= -1.0, a_b0 <= 1.0]
        soc0_ = p_soc0 + cp.multiply(a_b0, p_soc_scale)
        cons0 += [soc0_ >= self.soc_low, soc0_ <= self.soc_high]
        obj0 = cp.sum_squares(a_b0 - p_a_batt_rl)
        a_e0 = None
        if n_ev > 0:
            a_e0 = cp.Variable(n_ev)
            # FIX 1: EV bounds [-1, 1] for V2G
            cons0 += [a_e0 >= -1.0, a_e0 <= 1.0]
            # C1: EV action >= minimum required (0 when V2G-safe, >0 when urgent)
            cons0 += [a_e0 >= ev_p["p_c1_min"]]
            obj0 += cp.sum_squares(a_e0 - ev_p["p_a_ev_rl"])
        prob0 = cp.Problem(cp.Minimize(obj0), cons0)

        # ============================================================== #
        #  STAGE 1: Minimise C3 + C4 slack, subject to C1+C2            #
        # ============================================================== #
        a_b1 = cp.Variable(n_batt)
        s_bld = cp.Variable(nb, nonneg=True)
        s_grid = cp.Variable(1, nonneg=True)
        cons1 = [a_b1 >= -1.0, a_b1 <= 1.0]
        soc1_ = p_soc0 + cp.multiply(a_b1, p_soc_scale)
        cons1 += [soc1_ >= self.soc_low, soc1_ <= self.soc_high]
        a_e1 = None
        if n_ev > 0:
            a_e1 = cp.Variable(n_ev)
            # FIX 1: EV bounds [-1, 1]
            cons1 += [a_e1 >= -1.0, a_e1 <= 1.0]
            cons1 += [a_e1 >= ev_p["p_c1_min"]]

        # Net power per building (battery + EV contributions)
        # For C3: optionally exclude EV power (EV charges/discharges don't count
        # towards building power limit since EV charger is behind a separate meter)
        net_batt_only = p_base + A_batt @ cp.multiply(a_b1, p_batt_gain)
        net_full = net_batt_only
        if n_ev > 0:
            net_full = net_batt_only + A_ev @ cp.multiply(a_e1, ev_p["p_ev_gain"])

        if self.enable_c3:
            # FIX 4: C3 uses battery-only net if EV exempt, else full net
            net_c3 = net_batt_only if (self.c3_ev_exempt and n_ev > 0) else net_full
            cons1 += [net_c3 <= p_build_max + s_bld,
                      net_c3 >= -p_build_max - s_bld]
        else:
            cons1 += [s_bld == 0.0]

        # C4: grid-level constraint
        grid1 = cp.sum(net_full)
        if self.enable_c4:
            # FIX 3: Import-only — only penalise grid import, not export
            if self.c4_one_sided:
                cons1 += [grid1 <= p_grid_max + s_grid[0]]
            else:
                cons1 += [grid1 <= p_grid_max + s_grid[0],
                          grid1 >= -p_grid_max - s_grid[0]]
        else:
            cons1 += [s_grid[0] == 0.0]

        obj1 = s_grid[0] + (1.0 / max(1, nb)) * cp.sum(s_bld)
        prob1 = cp.Problem(cp.Minimize(obj1), cons1)

        # ============================================================== #
        #  STAGE 2: Closest to RL, subject to C1+C2+C3/C4 slack bound   #
        # ============================================================== #
        p_s_grid_star = cp.Parameter(nonneg=True, value=0.0)
        p_s_bld_star = cp.Parameter(nb, value=np.zeros(nb), nonneg=True)

        a_b2 = cp.Variable(n_batt)
        cons2 = [a_b2 >= -1.0, a_b2 <= 1.0]
        soc2_ = p_soc0 + cp.multiply(a_b2, p_soc_scale)
        cons2 += [soc2_ >= self.soc_low, soc2_ <= self.soc_high]
        a_e2 = None
        if n_ev > 0:
            a_e2 = cp.Variable(n_ev)
            # FIX 1: EV bounds [-1, 1]
            cons2 += [a_e2 >= -1.0, a_e2 <= 1.0]
            cons2 += [a_e2 >= ev_p["p_c1_min"]]

        net_batt_only2 = p_base + A_batt @ cp.multiply(a_b2, p_batt_gain)
        net_full2 = net_batt_only2
        if n_ev > 0:
            net_full2 = net_batt_only2 + A_ev @ cp.multiply(a_e2, ev_p["p_ev_gain"])

        if self.enable_c3:
            net_c3_2 = net_batt_only2 if (self.c3_ev_exempt and n_ev > 0) else net_full2
            cons2 += [net_c3_2 <= p_build_max + p_s_bld_star,
                      net_c3_2 >= -p_build_max - p_s_bld_star]

        grid2 = cp.sum(net_full2)
        if self.enable_c4:
            if self.c4_one_sided:
                cons2 += [grid2 <= p_grid_max + p_s_grid_star]
            else:
                cons2 += [grid2 <= p_grid_max + p_s_grid_star,
                          grid2 >= -p_grid_max - p_s_grid_star]

        obj2 = cp.sum_squares(a_b2 - p_a_batt_rl)
        if n_ev > 0:
            obj2 += cp.sum_squares(a_e2 - ev_p["p_a_ev_rl"])
        prob2 = cp.Problem(cp.Minimize(obj2), cons2)

        self._cvx = {
            "cp": cp, "A_batt": A_batt, "A_ev": A_ev,
            "p_base": p_base, "p_soc0": p_soc0, "p_soc_scale": p_soc_scale,
            "p_batt_gain": p_batt_gain, "p_a_batt_rl": p_a_batt_rl,
            "p_build_max": p_build_max, "p_grid_max": p_grid_max,
            "prob0": prob0, "a_b0": a_b0, "a_e0": a_e0,
            "prob1": prob1, "a_b1": a_b1, "a_e1": a_e1, "s_bld": s_bld, "s_grid": s_grid,
            "prob2": prob2, "a_b2": a_b2, "a_e2": a_e2,
            "p_s_grid_star": p_s_grid_star, "p_s_bld_star": p_s_bld_star,
        }
        self._cvx.update(ev_p)
        self._compiled = True
        if self.verbose >= 1:
            print(f"[Shield-V2G] Compiled: {n_batt} batt, {n_ev} ev, "
                  f"EV bounds=[-1,1], C3_ev_exempt={self.c3_ev_exempt}, "
                  f"C4_one_sided={self.c4_one_sided}")

    # ------------------------------------------------------------------ #
    #  QP solver helper                                                   #
    # ------------------------------------------------------------------ #
    def _solve_qp(self, prob):
        cp = self._cvx["cp"]
        try:
            prob.solve(solver=getattr(cp, self.solver, cp.OSQP),
                       warm_start=True, verbose=False,
                       max_iter=4000, eps_abs=1e-3, eps_rel=1e-3)
            return str(prob.status)
        except Exception as e:
            return f"error:{str(e)[:60]}"

    # ------------------------------------------------------------------ #
    #  Project RL action through 3-stage QP                              #
    # ------------------------------------------------------------------ #
    def _project_action(self, a_rl_flat):
        t0 = time.time()
        city = _unwrap_to_citylearn(self.env)
        buildings = list(getattr(city, "buildings", []))
        t_now = int(getattr(city, "time_step", 0))
        t_idx = max(0, t_now - 1)
        batt, ev, nb = self._mapping["batt"], self._mapping["ev"], self._mapping["nb"]
        n_batt, n_ev = len(batt), len(ev)
        cvx = self._cvx

        # --- Update battery parameters ---
        soc0 = np.zeros(n_batt)
        soc_scale = np.ones(n_batt)
        batt_gain = np.ones(n_batt)
        a_batt_rl = np.zeros(n_batt)
        for bi, binfo in enumerate(batt):
            es = getattr(buildings[binfo["b_idx"]], "electrical_storage", None)
            if es is None:
                continue
            cap = float(getattr(es, "capacity", 6.4) or 6.4)
            nom = float(getattr(es, "nominal_power", 5.0) or 5.0)
            batt_gain[bi] = max(1e-6, nom)
            soc_scale[bi] = max(1e-6, nom / max(1e-6, cap))
            soc_arr = np.asarray(getattr(es, "soc", []), dtype=float).ravel()
            soc0[bi] = float(np.clip(soc_arr[t_idx], 0.0, 1.0)) if 0 <= t_idx < len(soc_arr) else 0.5
            a_batt_rl[bi] = float(a_rl_flat[binfo["gidx"]]) if binfo["gidx"] < len(a_rl_flat) else 0.0
        cvx["p_soc0"].value = soc0
        cvx["p_soc_scale"].value = soc_scale
        cvx["p_batt_gain"].value = batt_gain
        cvx["p_a_batt_rl"].value = a_batt_rl

        # --- Update base net power ---
        base_net = np.zeros(nb)
        for b_idx, b in enumerate(buildings):
            nsl = np.asarray(getattr(b, "_Building__energy_to_non_shiftable_load", []), dtype=float)
            sg = np.asarray(getattr(b, "_Building__solar_generation", []), dtype=float)
            T_max = min(len(nsl), len(sg))
            if T_max > 0:
                idx = t_now if t_now < T_max else T_max - 1
                base_net[b_idx] = nsl[idx] + sg[idx]
        cvx["p_base"].value = base_net

        # --- Update EV parameters ---
        if n_ev > 0:
            ev_gain = np.zeros(n_ev)
            a_ev_rl = np.zeros(n_ev)
            for ei, einfo in enumerate(ev):
                a_ev_rl[ei] = float(a_rl_flat[einfo["gidx"]]) if einfo["gidx"] < len(a_rl_flat) else 0.0
                chargers = getattr(buildings[einfo["b_idx"]], "electric_vehicle_chargers", None) or []
                if einfo["local_ev_idx"] < len(chargers):
                    mp = getattr(chargers[einfo["local_ev_idx"]], "max_charging_power",
                                 getattr(chargers[einfo["local_ev_idx"]], "_Charger__max_charging_power", 0))
                    if isinstance(mp, np.ndarray):
                        mp = float(mp.ravel()[0])
                    ev_gain[ei] = float(mp or 0.0)
            c1_min = self._compute_c1_min_actions(city)
            cvx["p_ev_gain"].value = ev_gain
            cvx["p_a_ev_rl"].value = a_ev_rl
            cvx["p_c1_min"].value = c1_min

        # --- Solve 3 stages ---
        st0 = self._solve_qp(cvx["prob0"])
        st1 = self._solve_qp(cvx["prob1"])

        # Extract Stage 1 optimal slack
        sg_star = float(max(0.0, cvx["s_grid"].value[0])) if cvx["s_grid"].value is not None else 0.0
        sb_star = np.maximum(0.0, np.asarray(cvx["s_bld"].value).ravel()) if cvx["s_bld"].value is not None else np.zeros(nb)
        cvx["p_s_grid_star"].value = float(sg_star + self.slack_tol)
        cvx["p_s_bld_star"].value = sb_star + self.slack_tol

        st2 = self._solve_qp(cvx["prob2"])

        # --- Extract safe action ---
        a_safe = a_rl_flat.copy()
        is_opt = st2 in ("optimal", "optimal_inaccurate")
        c1c, c2c = 0, 0

        if is_opt and cvx["a_b2"].value is not None:
            for bi, binfo in enumerate(batt):
                old = a_safe[binfo["gidx"]]
                new = float(np.clip(cvx["a_b2"].value[bi], -1.0, 1.0))
                a_safe[binfo["gidx"]] = new
                if abs(new - old) > 1e-4:
                    c2c += 1
            if n_ev > 0 and cvx["a_e2"] is not None and cvx["a_e2"].value is not None:
                for ei, einfo in enumerate(ev):
                    old = a_safe[einfo["gidx"]]
                    # FIX 1: clip to [-1, 1] not [0, 1]
                    new = float(np.clip(cvx["a_e2"].value[ei], -1.0, 1.0))
                    a_safe[einfo["gidx"]] = new
                    if abs(new - old) > 1e-4:
                        c1c += 1
        else:
            # Fallback: simple clamping if QP failed
            for bi, binfo in enumerate(batt):
                g = binfo["gidx"]
                a_min = max((self.soc_low - soc0[bi]) / max(1e-9, soc_scale[bi]), -1.0)
                a_max = min((self.soc_high - soc0[bi]) / max(1e-9, soc_scale[bi]), 1.0)
                old = a_safe[g]
                a_safe[g] = float(np.clip(a_safe[g], a_min, a_max))
                if abs(a_safe[g] - old) > 1e-4:
                    c2c += 1
            if n_ev > 0:
                c1m = self._compute_c1_min_actions(city)
                for ei, einfo in enumerate(ev):
                    g = einfo["gidx"]
                    old = a_safe[g]
                    # Fallback: enforce C1 minimum, allow V2G otherwise
                    a_safe[g] = float(np.clip(a_safe[g], c1m[ei], 1.0))
                    if abs(a_safe[g] - old) > 1e-4:
                        c1c += 1
            self._total_infeasible += 1

        delta = float(np.linalg.norm(a_safe - a_rl_flat))
        info = {
            "ap_active": 1.0,
            "ap_any_intervention": float(delta > 1e-4),
            "ap_action_delta_l2": delta,
            "ap_c1_clamps": float(c1c),
            "ap_c2_clamps": float(c2c),
            "ap_s_grid_star": float(sg_star),
            "ap_s_bld_star_sum": float(sb_star.sum()),
            "ap_solve_ms": (time.time() - t0) * 1000.0,
            "ap_infeasible": 0.0 if is_opt else 1.0,
            "ap_st0": st0, "ap_st1": st1, "ap_st2": st2,
        }
        return a_safe, info

    # ------------------------------------------------------------------ #
    #  Step: project → env.step → annotate info                          #
    # ------------------------------------------------------------------ #
    def step(self, action):
        if not self._compiled:
            self._compile()
        is_multi = isinstance(self.action_space, list)
        if is_multi:
            lo = np.concatenate([np.asarray(sp.low).ravel() for sp in self.action_space])
            hi = np.concatenate([np.asarray(sp.high).ravel() for sp in self.action_space])
        else:
            lo = np.asarray(self.action_space.low).ravel()
            hi = np.asarray(self.action_space.high).ravel()
        a_rl = np.clip(_flatten_action(action, self.action_space), lo, hi)
        a_safe, si = self._project_action(a_rl)
        a_safe = np.clip(a_safe, lo, hi)
        a_out = _unflatten_action(a_safe, self.action_space) if is_multi else a_safe
        obs, rew, term, trunc, info = self.env.step(a_out)
        info = dict(info) if info else {}
        info.update(si)
        self._step_count += 1
        self._total_solve_ms += si.get("ap_solve_ms", 0)
        if si.get("ap_any_intervention", 0) > 0:
            self._total_interventions += 1
        self._c1_interventions += int(si.get("ap_c1_clamps", 0))
        self._c2_interventions += int(si.get("ap_c2_clamps", 0))
        # Store projected action for NFWPO distillation
        info["shielded_action"] = a_safe.copy() if hasattr(a_safe, "copy") else a_safe
        return obs, rew, term, trunc, info

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #
    def reset(self, **kwargs):
        if self._step_count > 0 and self.verbose >= 1:
            avg = self._total_solve_ms / max(1, self._step_count)
            print(f"[Shield-V2G] Ep{self._episode_count} | "
                  f"steps={self._step_count} "
                  f"interv={self._total_interventions} "
                  f"(c1={self._c1_interventions} c2={self._c2_interventions}) "
                  f"infeas={self._total_infeasible} avg={avg:.1f}ms")
        self._compiled = False
        self._mapping = None
        self._step_count = 0
        self._episode_count += 1
        self._total_interventions = 0
        self._total_solve_ms = 0
        self._total_infeasible = 0
        self._c1_interventions = 0
        self._c2_interventions = 0
        return self.env.reset(**kwargs)
