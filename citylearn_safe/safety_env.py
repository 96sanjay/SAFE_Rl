
# citylearn_safe/safety_env.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import gymnasium as gym

from .schema_index import build_index  # deterministic obs-name based indexer
from .kpi_logger import log_kpis, log_episode_end, init_kpi_logger
from citylearn_safe.extractors import ev_departure_cost_components


class CityLearnSafetyEnv(gym.Env):
    """
    Wrapper around a CityLearn-based env that:

    - Computes safety cost info['cost'] based on INTERNAL building battery SoC band.
      (Source of truth: citylearn_env.buildings[*].electrical_storage.soc[t_idx])
    - Optionally adds EV departure deficit cost (toggle via env var).
    - Computes an energy-based reward:
          reward = - sum_b net_electricity_consumption_b(t_idx)
      while still logging CityLearn's original reward as info['citylearn_reward'].
    - Logs KPIs for analysis, including robust thermal discomfort and EV fairness split.

    Typical wrapper chain:
        CityLearnSafetyEnv
          -> base = SingleAgentListAdapter
                -> base = NormalizedObservationWrapper
                      -> env = CityLearnEnv
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: Any,
        *,
        soc_min: float = 0.0,
        soc_max: float = 0.95,
        cost_mode: str = "hinge",
    ):
        super().__init__()
        self.base = base_env
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)

        # From the agent's POV, look exactly like base_env.
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

        override_mode = os.environ.get("CITYLEARN_COST_MODE")
        if override_mode:
            cost_mode = override_mode

        cost_mode_normalized = cost_mode.lower()
        if cost_mode_normalized not in {"hinge", "binary"}:
            raise ValueError(
                f"Unsupported cost_mode '{cost_mode}'. Choose between 'hinge' or 'binary'."
            )
        self.cost_mode = cost_mode_normalized

        # Toggle: should EV deficit contribute to CMDP "cost" for OmniSafe?
        self.include_ev_in_cost = bool(int(os.environ.get("CITYLEARN_INCLUDE_EV_COST", "1")))

        # 1 hour per step (CityLearn challenge is hourly)
        self._dt_h = 1.0

        # --- Deterministic obs indices (ONLY for KPI/debug; may be dead/zero under wrappers) ---
        self._obs_index = None
        self._soc_idx_obs: List[int] = []
        self._idx_net_consumption_obs = None
        self._idx_non_shiftable_load_obs = None
        self._idx_month_cos = None
        self._idx_month_sin = None
        self._idx_day_type_cos = None
        self._idx_day_type_sin = None
        self._idx_hour_cos = None
        self._idx_hour_sin = None

        try:
            # NOTE: build_index wants names that match the *flat* observation vector.
            # Prefer pulling names from the adapter chain (self.base) rather than raw CityLearnEnv.
            self._obs_index = build_index(self.base, expected_buildings=17)
            self._soc_idx_obs = list(self._obs_index.electrical_storage_soc)
            self._idx_net_consumption_obs = self._obs_index.net_electricity_consumption[0]  # building_1
            self._idx_non_shiftable_load_obs = self._obs_index.non_shiftable_load[0]
            self._idx_month_cos = self._obs_index.month_cos
            self._idx_month_sin = self._obs_index.month_sin
            self._idx_day_type_cos = self._obs_index.day_type_cos
            self._idx_day_type_sin = self._obs_index.day_type_sin
            self._idx_hour_cos = self._obs_index.hour_cos
            self._idx_hour_sin = self._obs_index.hour_sin
        except Exception:
            # We can still run perfectly fine using INTERNAL STATE only.
            self._obs_index = None

        # --- KPI + episode tracking ---
        self._step_count = 0
        self._episode_count = 0
        self._kpi_logger_initialized = False

        # Debug toggle: include obs-vs-state comparisons in info
        self._debug_obs_vs_state = bool(int(os.environ.get("CITYLEARN_DEBUG_OBS_VS_STATE", "0")))

    # -------------------------------------------------------------------------
    # Robust unwrapping helper: get object that exposes buildings + time_step.
    # (In your stack, deepest stable object is often NormalizedObservationWrapper.)
    # -------------------------------------------------------------------------
    def _get_citylearn_env(self):
        """Best-effort unwrap down to an object that exposes .buildings and .time_step."""
        cur = self.base
        seen = set()

        for _ in range(40):
            if cur is None:
                break
            obj_id = id(cur)
            if obj_id in seen:
                break
            seen.add(obj_id)

            # Accept any layer that exposes buildings + time_step (wrapper or raw env).
            try:
                blds = getattr(cur, "buildings", None)
                ts = getattr(cur, "time_step", None)
                if blds is not None and hasattr(blds, "__len__") and len(blds) > 0 and ts is not None:
                    return cur
            except Exception:
                pass

            # Common wrapper attributes (ordered)
            advanced = False
            for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
                if hasattr(cur, attr):
                    nxt = getattr(cur, attr, None)
                    # gymnasium's .unwrapped can point to itself; guard it
                    if nxt is not None and nxt is not cur:
                        cur = nxt
                        advanced = True
                        break
            if not advanced:
                break

        return None

    # -------------------------------------------------------------------------
    # Expose unwrapped env (CityLearn agents often use env.unwrapped)
    # -------------------------------------------------------------------------
    @property
    def unwrapped(self):
        env = self.base
        if hasattr(env, "unwrapped"):
            try:
                return env.unwrapped
            except Exception:
                return env
        return env

    # -------------------------------------------------------------------------
    # Expose properties used by CityLearn agents (e.g., RBC)
    # Prefer adapter's names (aligned with action/obs vectors).
    # -------------------------------------------------------------------------
    @property
    def observation_names(self):
        if hasattr(self.base, "observation_names"):
            return self.base.observation_names
        citylearn_env = self._get_citylearn_env()
        if citylearn_env is not None and hasattr(citylearn_env, "observation_names"):
            return citylearn_env.observation_names
        dim = int(self.observation_space.shape[0])
        return [f"obs_{i}" for i in range(dim)]

    @property
    def action_names(self):
        if hasattr(self.base, "action_names"):
            return self.base.action_names
        citylearn_env = self._get_citylearn_env()
        if citylearn_env is not None and hasattr(citylearn_env, "action_names"):
            return citylearn_env.action_names
        dim = int(self.action_space.shape[0])
        return [f"action_{i}" for i in range(dim)]

    @property
    def time_steps(self):
        citylearn_env = self._get_citylearn_env()
        if citylearn_env is not None:
            return getattr(citylearn_env, "time_steps", 0)
        return 0

    # -------------------------------------------------------------------------
    # KPI logger init
    # -------------------------------------------------------------------------
    def _ensure_kpi_logger_initialized(self):
        if self._kpi_logger_initialized:
            return

        runs_dir = os.path.join(os.getcwd(), "runs")
        if os.path.exists(runs_dir):
            most_recent_dir = None
            most_recent_time = 0.0

            import time

            for item in os.listdir(runs_dir):
                item_path = os.path.join(runs_dir, item)
                if not os.path.isdir(item_path):
                    continue
                if "CityLearnSafety" not in item:
                    continue

                for seed_item in os.listdir(item_path):
                    seed_path = os.path.join(item_path, seed_item)
                    if os.path.isdir(seed_path) and seed_item.startswith("seed-"):
                        dir_ctime = os.path.getctime(seed_path)
                        if time.time() - dir_ctime < 180:
                            if dir_ctime > most_recent_time:
                                most_recent_time = dir_ctime
                                most_recent_dir = seed_path

            if most_recent_dir:
                init_kpi_logger(most_recent_dir, "kpis")
                self._kpi_logger_initialized = True
                print(f"[CityLearnSafetyEnv] KPI logger initialized in: {most_recent_dir}")
                return

        # Fallback
        temp_dir = os.path.join(os.getcwd(), "runs", "kpi_logs")
        os.makedirs(temp_dir, exist_ok=True)
        init_kpi_logger(temp_dir, "CityLearnSafety_kpis")
        self._kpi_logger_initialized = True
        print("[CityLearnSafetyEnv] KPI logger initialized in fallback location:", temp_dir)

    # -------------------------------------------------------------------------
    # Gym API: reset / step
    # -------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        obs, info = self.base.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)

        self._step_count = 0

        citylearn_env = self._get_citylearn_env()

        # INTERNAL SoC values (truth)
        soc_state_vals = self._soc_values_from_state(citylearn_env)
        metrics = self._soc_metrics(soc_state_vals)
        building_cost = self._soc_band_cost(metrics)

        dummy_action = np.zeros_like(np.asarray(self.action_space.sample(), dtype=float))
        kpis = self._compute_basic_kpis(obs, dummy_action)

        info = dict(info)
        info["metrics"] = metrics

        # CMDP cost at reset: only building SoC band (EV deficit is event-based anyway)
        info["cost"] = float(building_cost)
        info["cost_building_soc"] = float(building_cost)
        info["cost_ev_departure"] = 0.0
        info["cost_ev_departure_avoidable"] = 0.0
        info["cost_ev_departure_unavoidable"] = 0.0
        info["ev_departure_departures"] = 0

        info.update(kpis)

        # Advanced KPI defaults
        info["ev_departure_deficit_kwh"] = 0.0
        info["ev_avoidable_deficit_kwh"] = 0.0
        info["ev_unavoidable_deficit_kwh"] = 0.0
        info["ev_impossible_request_count"] = 0.0

        info["battery_abuse_kwh"] = 0.0
        info["battery_abuse_excess_kwh_equiv"] = 0.0
        info["battery_abuse_hours"] = 0.0

        info["solar_waste_kwh"] = 0.0

        info["reward"] = 0.0
        info["citylearn_reward"] = 0.0
        info["used_energy_reward"] = False

        # Optional debug: compare obs SoC channel vs internal state SoC
        if self._debug_obs_vs_state:
            info.update(self._debug_soc_obs_vs_state(obs, citylearn_env))

        log_kpis(info, self._step_count, self._episode_count)
        return obs, info

    def step(self, action):
        self._ensure_kpi_logger_initialized()

        obs, r_base, term, trunc, info = self.base.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        self._step_count += 1

        citylearn_env = self._get_citylearn_env()

        # --- Safety cost from INTERNAL building battery SoC ---
        soc_state_vals = self._soc_values_from_state(citylearn_env)
        metrics = self._soc_metrics(soc_state_vals)
        building_cost = self._soc_band_cost(metrics)

        # --- KPIs (district-level scalars) ---
        action_arr = np.asarray(action, dtype=float)
        kpis = self._compute_basic_kpis(obs, action_arr)

        # --- Energy-based reward (fallback to CityLearn reward) ---
        reward = None
        used_energy_reward = False
        try:
            if citylearn_env is not None:
                idx = self._state_time_index(citylearn_env)
                total_kw = 0.0
                for b in getattr(citylearn_env, "buildings", []):
                    nec = getattr(b, "net_electricity_consumption", None)
                    if nec is not None and hasattr(nec, "__len__") and len(nec) > idx:
                        total_kw += float(nec[idx])
                reward = -total_kw
                used_energy_reward = True
        except Exception:
            reward = None

        if reward is None:
            if isinstance(r_base, (list, tuple, np.ndarray)):
                reward = float(np.sum(r_base))
            else:
                reward = float(r_base)

        # --- EV departure deficit (total + avoidable/unavoidable) ---
        ev_cost_components = {
            "total": 0.0,
            "avoidable": 0.0,
            "unavoidable": 0.0,
            "departures": 0,
        }
        try:
            if citylearn_env is not None:
                ev_cost_components = ev_departure_cost_components(citylearn_env) or ev_cost_components
        except Exception:
            pass

        ev_total = float(ev_cost_components.get("total", 0.0))
        ev_avoid = float(ev_cost_components.get("avoidable", 0.0))
        ev_unavoid = float(ev_cost_components.get("unavoidable", 0.0))
        ev_deps = int(ev_cost_components.get("departures", 0) or 0)

        # --- Battery abuse + solar waste (district totals) ---
        adv = self._compute_battery_solar_kpis(citylearn_env)
        battery_abuse_kwh = float(adv.get("battery_abuse_kwh", 0.0))
        solar_waste_kwh = float(adv.get("solar_waste_kwh", 0.0))
        battery_abuse_hours = float(adv.get("battery_abuse_hours", 0.0))
        battery_abuse_excess = float(adv.get("battery_abuse_excess_kwh_equiv", 0.0))

        # --- CMDP cost (what OmniSafe sees) ---
        ev_cost_for_cmdp = ev_avoid if self.include_ev_in_cost else 0.0
        total_cost = float(building_cost + ev_cost_for_cmdp)

        # --- Pack info ---
        info = dict(info)
        info["metrics"] = metrics

        info["cost"] = float(total_cost)
        info["cost_building_soc"] = float(building_cost)
        info["cost_ev_departure"] = float(ev_cost_for_cmdp)

        # Always log EV breakdown even if not included in CMDP cost
        info["cost_ev_departure_avoidable"] = float(ev_avoid)
        info["cost_ev_departure_unavoidable"] = float(ev_unavoid)
        info["ev_departure_departures"] = int(ev_deps)

        info.update(kpis)

        # Advanced KPIs
        info["ev_departure_deficit_kwh"] = float(ev_total)
        info["ev_avoidable_deficit_kwh"] = float(ev_avoid)
        info["ev_unavoidable_deficit_kwh"] = float(ev_unavoid)
        info["ev_impossible_request_count"] = 1.0 if ev_unavoid > 0 else 0.0

        info["battery_abuse_kwh"] = float(battery_abuse_kwh)
        info["battery_abuse_excess_kwh_equiv"] = float(battery_abuse_excess)
        info["battery_abuse_hours"] = float(battery_abuse_hours)

        info["solar_waste_kwh"] = float(solar_waste_kwh)

        info["reward"] = float(reward)
        info["citylearn_reward"] = (
            float(np.sum(r_base)) if isinstance(r_base, (list, tuple, np.ndarray)) else float(r_base)
        )
        info["used_energy_reward"] = bool(used_energy_reward)

        # Optional debug: compare obs SoC channel vs internal state SoC
        if self._debug_obs_vs_state:
            info.update(self._debug_soc_obs_vs_state(obs, citylearn_env))

        # log step
        log_kpis(info, self._step_count, self._episode_count)

        # episode end
        if term or trunc:
            self._episode_count += 1
            citylearn_kpis = self._extract_citylearn_kpis()
            info.update(citylearn_kpis)
            log_episode_end(info, self._episode_count)

        return obs, float(reward), bool(term), bool(trunc), info

    # -------------------------------------------------------------------------
    # INTERNAL SoC helpers (truth)
    # -------------------------------------------------------------------------
    def _state_time_index(self, env) -> int:
        """
        Robust timestep index into CityLearn series.

        IMPORTANT:
        CityLearn writes many per-timestep series such that, after env.step(),
        the "current realized" values live at index (time_step - 1).
        """
        if env is None:
            return 0
        t = getattr(env, "time_step", 0)
        return max(0, int(t) - 1)

    def _soc_values_from_state(self, env) -> List[float]:
        """
        Read battery SoC from internal CityLearn state:
          b.electrical_storage.soc[t_idx]
        Returns list length == number of buildings (17).
        """
        if env is None or not getattr(env, "buildings", None):
            return []

        t_idx = self._state_time_index(env)
        out: List[float] = []

        for b in env.buildings:
            soc_val = 0.0
            try:
                es = getattr(b, "electrical_storage", None)
                soc = getattr(es, "soc", None) if es is not None else None

                if soc is None:
                    soc_val = 0.0
                elif hasattr(soc, "__len__") and len(soc) > t_idx:
                    soc_val = float(soc[t_idx])
                elif np.isscalar(soc):
                    soc_val = float(soc)
                else:
                    soc_val = 0.0

                if not np.isfinite(soc_val):
                    soc_val = 0.0

                # CityLearn SoC is usually [0, 1]. Clamp for safety.
                soc_val = float(np.clip(soc_val, 0.0, 1.0))
            except Exception:
                soc_val = 0.0

            out.append(soc_val)

        return out

    # -------------------------------------------------------------------------
    # OBS SoC helpers (debug only — may be dead/zero)
    # -------------------------------------------------------------------------
    def _soc_values_from_obs(self, obs: np.ndarray) -> List[float]:
        vals: List[float] = []
        if not self._soc_idx_obs:
            return vals
        for idx in self._soc_idx_obs:
            if 0 <= idx < obs.shape[0]:
                vals.append(float(obs[idx]))
        return vals

    def _soc_metrics(self, vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {
                "soc_mean": 0.5,
                "soc_min_obs": 0.5,
                "soc_max_obs": 0.5,
                "num_storages": 0.0,
            }
        return {
            "soc_mean": float(np.mean(vals)),
            "soc_min_obs": float(np.min(vals)),
            "soc_max_obs": float(np.max(vals)),
            "num_storages": float(len(vals)),
        }

    def _soc_band_cost(self, stats: Dict[str, float]) -> float:
        if stats.get("num_storages", 0.0) <= 0.0:
            return 0.0

        low_violation = max(0.0, self.soc_min - stats["soc_min_obs"])
        high_violation = max(0.0, stats["soc_max_obs"] - self.soc_max)

        if self.cost_mode == "binary":
            return 1.0 if (low_violation > 0.0 or high_violation > 0.0) else 0.0

        band = max(1e-6, (self.soc_max - self.soc_min))
        return (low_violation + high_violation) / band

    def _debug_soc_obs_vs_state(self, obs: np.ndarray, env) -> Dict[str, float]:
        """Attach quick sanity checks to info dict."""
        obs_vals = self._soc_values_from_obs(obs)
        state_vals = self._soc_values_from_state(env)

        d: Dict[str, float] = {}
        d["debug_obs_soc_min"] = float(np.min(obs_vals)) if obs_vals else 0.0
        d["debug_obs_soc_max"] = float(np.max(obs_vals)) if obs_vals else 0.0
        d["debug_obs_soc_mean"] = float(np.mean(obs_vals)) if obs_vals else 0.0

        d["debug_state_soc_min"] = float(np.min(state_vals)) if state_vals else 0.0
        d["debug_state_soc_max"] = float(np.max(state_vals)) if state_vals else 0.0
        d["debug_state_soc_mean"] = float(np.mean(state_vals)) if state_vals else 0.0
        return d

    # -------------------------------------------------------------------------
    # Advanced KPIs: Battery abuse + Solar waste (district totals)
    # -------------------------------------------------------------------------
    def _compute_battery_solar_kpis(self, env) -> Dict[str, float]:
        kpis = {
            "battery_abuse_kwh": 0.0,
            "battery_abuse_excess_kwh_equiv": 0.0,
            "battery_abuse_hours": 0.0,
            "solar_waste_kwh": 0.0,
        }
        if env is None or not getattr(env, "buildings", None):
            return kpis

        t_idx = self._state_time_index(env)
        any_abuse = False

        for b in env.buildings:
            es = getattr(b, "electrical_storage", None)

            # Battery abuse: SoC > 0.95 -> excess * capacity
            if es is not None:
                try:
                    soc_data = getattr(es, "soc", None)
                    cap = float(getattr(es, "capacity", 0.0) or 0.0)

                    if soc_data is None:
                        soc = 0.0
                    elif hasattr(soc_data, "__len__") and len(soc_data) > t_idx:
                        soc = float(soc_data[t_idx])
                    elif np.isscalar(soc_data):
                        soc = float(soc_data)
                    else:
                        soc = 0.0

                    if np.isfinite(soc) and soc > 0.95 and cap > 0.0:
                        excess = (soc - 0.95) * cap
                        kpis["battery_abuse_kwh"] += excess
                        kpis["battery_abuse_excess_kwh_equiv"] += excess
                        any_abuse = True
                except Exception:
                    pass

            # Solar waste heuristic
            try:
                nec = getattr(b, "net_electricity_consumption", None)
                if nec is None:
                    net_grid = 0.0
                elif hasattr(nec, "__len__") and len(nec) > t_idx:
                    net_grid = float(nec[t_idx])
                elif np.isscalar(nec):
                    net_grid = float(nec)
                else:
                    net_grid = 0.0

                batt_soc = None
                if es is not None:
                    soc_data = getattr(es, "soc", None)
                    if soc_data is not None and hasattr(soc_data, "__len__") and len(soc_data) > t_idx:
                        batt_soc = float(soc_data[t_idx])
                    elif np.isscalar(soc_data):
                        batt_soc = float(soc_data)

                if net_grid < -0.01 and (batt_soc is not None) and (batt_soc < 0.9):
                    kpis["solar_waste_kwh"] += abs(net_grid) * self._dt_h
            except Exception:
                pass

        kpis["battery_abuse_hours"] = 1.0 if any_abuse else 0.0
        return kpis

    # -------------------------------------------------------------------------
    # Basic KPI helpers (district-level)
    # -------------------------------------------------------------------------
    def _compute_basic_kpis(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        kpis: Dict[str, float] = {}

        # 1) Battery SoC from INTERNAL STATE (district aggregation)
        citylearn_env = self._get_citylearn_env()
        soc_vals = self._soc_values_from_state(citylearn_env)
        kpis["soc_mean"] = float(np.mean(soc_vals)) if soc_vals else 0.0
        kpis["soc_min"] = float(np.min(soc_vals)) if soc_vals else 0.0
        kpis["soc_max"] = float(np.max(soc_vals)) if soc_vals else 0.0
        kpis["soc_std"] = float(np.std(soc_vals)) if soc_vals else 0.0

        # 2) Action stats
        action = np.asarray(action, dtype=float).ravel()
        kpis["action_mean"] = float(np.mean(action)) if action.size else 0.0
        kpis["action_std"] = float(np.std(action)) if action.size else 0.0
        kpis["action_min"] = float(np.min(action)) if action.size else 0.0
        kpis["action_max"] = float(np.max(action)) if action.size else 0.0

        # 3) Step tracking
        kpis["step_count"] = float(self._step_count)

        # 4) Time-related obs features (cos/sin) — only if indices exist
        def safe_obs(i: Optional[int]) -> float:
            if i is None:
                return 0.0
            return float(obs[i]) if 0 <= i < len(obs) else 0.0

        kpis["month_cos"] = safe_obs(self._idx_month_cos)
        kpis["month_sin"] = safe_obs(self._idx_month_sin)
        kpis["day_type_cos"] = safe_obs(self._idx_day_type_cos)
        kpis["day_type_sin"] = safe_obs(self._idx_day_type_sin)
        kpis["hour_cos"] = safe_obs(self._idx_hour_cos)
        kpis["hour_sin"] = safe_obs(self._idx_hour_sin)

        kpis["non_shiftable_load_obs_b1"] = safe_obs(self._idx_non_shiftable_load_obs)

        # 5) District net electricity + price + carbon (robust timestep indexing)
        step_net_kwh = 0.0
        current_price = 0.0
        carbon_intensity = 0.0

        if citylearn_env is not None and getattr(citylearn_env, "buildings", None):
            idx = self._state_time_index(citylearn_env)

            total_kw = 0.0
            for b in citylearn_env.buildings:
                nec = getattr(b, "net_electricity_consumption", None)
                if nec is not None and hasattr(nec, "__len__") and len(nec) > idx:
                    total_kw += float(nec[idx])
            step_net_kwh = total_kw * self._dt_h

            b0 = citylearn_env.buildings[0]
            try:
                ep = b0.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > idx:
                    current_price = float(ep[idx])
            except Exception:
                pass
            try:
                ci = b0.carbon_intensity.carbon_intensity
                if hasattr(ci, "__len__") and len(ci) > idx:
                    carbon_intensity = float(ci[idx])
            except Exception:
                pass
        else:
            # fallback to obs (best-effort)
            if self._idx_net_consumption_obs is not None and self._idx_net_consumption_obs < len(obs):
                step_net_kwh = float(obs[self._idx_net_consumption_obs]) * self._dt_h

        kpis["step_net_consumption_kwh"] = float(step_net_kwh)
        kpis["electricity_price"] = float(current_price)

        import_kwh = max(step_net_kwh, 0.0)
        export_kwh = max(-step_net_kwh, 0.0)
        kpis["grid_import_kwh"] = float(import_kwh)
        kpis["grid_export_kwh"] = float(export_kwh)

        kpis["step_cost"] = float(import_kwh * current_price)
        kpis["carbon_intensity"] = float(carbon_intensity)
        kpis["step_carbon_kg"] = float(import_kwh * carbon_intensity)

        # 6) Robust thermal discomfort (average across buildings)
        thermal_discomfort_vals: List[float] = []
        indoor_temp_b0 = 0.0
        outdoor_temp_b0 = 0.0

        if citylearn_env is not None and getattr(citylearn_env, "buildings", None):
            idx = self._state_time_index(citylearn_env)

            for bi, b in enumerate(citylearn_env.buildings):
                try:
                    indoor = (
                        b.indoor_dry_bulb_temperature[idx]
                        if len(b.indoor_dry_bulb_temperature) > idx
                        else float(b.indoor_dry_bulb_temperature[-1])
                    )

                    # Cooling setpoint
                    if hasattr(b, "indoor_dry_bulb_temperature_cooling_set_point"):
                        csp_arr = b.indoor_dry_bulb_temperature_cooling_set_point
                        csp = csp_arr[idx] if len(csp_arr) > idx else float(csp_arr[-1])
                    elif hasattr(b, "indoor_dry_bulb_temperature_set_point"):
                        sp_arr = b.indoor_dry_bulb_temperature_set_point
                        csp = sp_arr[idx] if len(sp_arr) > idx else float(sp_arr[-1])
                    else:
                        csp = 100.0

                    # Heating setpoint
                    if hasattr(b, "indoor_dry_bulb_temperature_heating_set_point"):
                        hsp_arr = b.indoor_dry_bulb_temperature_heating_set_point
                        hsp = hsp_arr[idx] if len(hsp_arr) > idx else float(hsp_arr[-1])
                    elif hasattr(b, "indoor_dry_bulb_temperature_set_point"):
                        sp_arr = b.indoor_dry_bulb_temperature_set_point
                        hsp = sp_arr[idx] if len(sp_arr) > idx else float(sp_arr[-1])
                    else:
                        hsp = -100.0

                    diff_hot = max(0.0, float(indoor) - float(csp))
                    diff_cold = max(0.0, float(hsp) - float(indoor))
                    thermal_discomfort_vals.append(diff_hot + diff_cold)

                    if bi == 0:
                        indoor_temp_b0 = float(indoor)
                        if hasattr(b, "weather") and hasattr(b.weather, "outdoor_dry_bulb_temperature"):
                            out_arr = b.weather.outdoor_dry_bulb_temperature
                            if hasattr(out_arr, "__len__") and len(out_arr) > idx:
                                outdoor_temp_b0 = float(out_arr[idx])
                except Exception:
                    continue

        kpis["thermal_discomfort"] = float(np.mean(thermal_discomfort_vals)) if thermal_discomfort_vals else 0.0
        kpis["indoor_temperature"] = float(indoor_temp_b0)
        kpis["outdoor_temperature"] = float(outdoor_temp_b0)

        # 7) Solar generation (building 0, indexed)
        solar_gen = 0.0
        if citylearn_env is not None and getattr(citylearn_env, "buildings", None):
            idx = self._state_time_index(citylearn_env)
            b0 = citylearn_env.buildings[0]
            try:
                sg = getattr(b0, "solar_generation", None)
                if sg is not None and hasattr(sg, "__len__") and len(sg) > idx:
                    solar_gen = float(sg[idx]) * self._dt_h
            except Exception:
                pass
        kpis["solar_generation_kwh"] = float(solar_gen)

        # 8) Non-shiftable load (building 0, indexed)
        nsl = 0.0
        if citylearn_env is not None and getattr(citylearn_env, "buildings", None):
            idx = self._state_time_index(citylearn_env)
            b0 = citylearn_env.buildings[0]
            try:
                load = getattr(b0, "non_shiftable_load", None)
                if load is not None and hasattr(load, "__len__") and len(load) > idx:
                    nsl = float(load[idx]) * self._dt_h
            except Exception:
                pass
        kpis["non_shiftable_load_kwh"] = float(nsl)

        # 9) Binary SoC band violation (district) — INTERNAL STATE
        violation = False
        if soc_vals:
            violation = any(s < self.soc_min or s > self.soc_max for s in soc_vals)
        kpis["constraint_violation"] = 1.0 if violation else 0.0

        return kpis

    # -------------------------------------------------------------------------
    # CityLearn KPI extraction at episode end (District-normalized)
    # -------------------------------------------------------------------------
    def _extract_citylearn_kpis(self) -> Dict[str, float]:
        citylearn_kpis: Dict[str, float] = {}
        try:
            citylearn_env = self._get_citylearn_env()
            if citylearn_env is None:
                raise RuntimeError("CityLearnEnv not found in wrapper stack.")

            # Some wrappers forward evaluate(); if not, try env.env.evaluate()
            kpis_df = None
            if hasattr(citylearn_env, "evaluate"):
                kpis_df = citylearn_env.evaluate()
            elif hasattr(citylearn_env, "env") and hasattr(citylearn_env.env, "evaluate"):
                kpis_df = citylearn_env.env.evaluate()
            else:
                raise RuntimeError("evaluate() not found on unwrapped CityLearn object.")

            building_name = "District"
            if "name" in kpis_df.columns:
                if (kpis_df["name"] == "District").any():
                    building_name = "District"
                else:
                    unique = kpis_df["name"].unique()
                    building_name = unique[0] if len(unique) else "District"

            def extract_kpi(cost_function_name: str) -> float:
                result = kpis_df[
                    (kpis_df["name"] == building_name)
                    & (kpis_df["cost_function"] == cost_function_name)
                ]
                if not result.empty:
                    value = result["value"].iloc[0]
                    if pd.notna(value):
                        return float(value)
                return 0.0

            citylearn_kpis["citylearn_electricity_consumption_total"] = extract_kpi("electricity_consumption_total")
            citylearn_kpis["citylearn_carbon_emissions_total"] = extract_kpi("carbon_emissions_total")
            citylearn_kpis["citylearn_cost_total"] = extract_kpi("cost_total")
            citylearn_kpis["citylearn_zero_net_energy"] = extract_kpi("zero_net_energy")

            citylearn_kpis["citylearn_daily_peak_average"] = extract_kpi("daily_peak_average")
            citylearn_kpis["citylearn_all_time_peak_average"] = extract_kpi("all_time_peak_average")
            citylearn_kpis["citylearn_ramping_average"] = extract_kpi("ramping_average")

            citylearn_kpis["citylearn_discomfort_proportion"] = extract_kpi("discomfort_proportion")

        except Exception as e:
            import traceback
            print(f"[CityLearnSafetyEnv] Warning: Could not extract CityLearn KPIs: {e}")
            print(f"[CityLearnSafetyEnv] Traceback: {traceback.format_exc()}")
            citylearn_kpis = {
                "citylearn_electricity_consumption_total": 0.0,
                "citylearn_carbon_emissions_total": 0.0,
                "citylearn_cost_total": 0.0,
                "citylearn_daily_peak_average": 0.0,
                "citylearn_all_time_peak_average": 0.0,
                "citylearn_ramping_average": 0.0,
                "citylearn_discomfort_proportion": 0.0,
                "citylearn_zero_net_energy": 0.0,
            }

        return citylearn_kpis

# Alias for backwards compatibility
SafetyEnv = CityLearnSafetyEnv
