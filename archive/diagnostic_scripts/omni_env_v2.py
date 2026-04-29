"""
OmniSafe-compatible CityLearn environment for FOCOPS/PPOLag training.

Key changes:
  1. ForecastObsWrapper adds 24h lookahead (price, load, solar, EV urgency)
  2. EV charging reward component added to STEMS
  3. Cost signal rebalanced (C1 boosted, C3 dampened)
  4. STEMS reward weights tuned for constraint satisfaction
"""
from __future__ import annotations

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def make_citylearn_base(schema_path: str | None = None):
    from citylearn.citylearn import CityLearnEnv
    from citylearn.wrappers import NormalizedObservationWrapper

    if schema_path is None:
        schema_path = os.environ.get(
            "CITYLEARN_SCHEMA",
            "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json")

    env = CityLearnEnv(schema=schema_path, central_agent=True)
    env = NormalizedObservationWrapper(env)
    return env


class CityLearnSafetyEnvV2Omni(gym.Env):
    """
    OmniSafe-compatible wrapper with:
      - ForecastObsWrapper for 24h lookahead
      - Rebalanced STEMS reward (with EV component)
      - Rebalanced CMDP cost signal
    """

    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()

        from citylearn_safe.adapters import SingleAgentListAdapter
        from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
        from forecast_obs_wrapper import ForecastObsWrapper

        base = make_citylearn_base()
        adapter = SingleAgentListAdapter(base)
        safety = CityLearnSafetyEnvV3(adapter)
        forecast = ForecastObsWrapper(safety, forecast_horizon=24)

        self._inner = forecast
        self.observation_space = forecast.observation_space
        self.action_space = forecast.action_space

        # STEMS reward weights
        self.mu_economic = float(os.environ.get("STEMS_MU_ECONOMIC", "0.3"))
        self.alpha_grid = float(os.environ.get("STEMS_ALPHA_GRID", "3.0"))
        self.alpha_build = float(os.environ.get("STEMS_ALPHA_BUILD", "2.0"))
        self.beta_ramp = float(os.environ.get("STEMS_BETA_RAMP", "0.5"))
        self.xi_renewable = float(os.environ.get("STEMS_XI_RENEWABLE", "0.2"))
        self.lambda_ev = float(os.environ.get("STEMS_LAMBDA_EV", "5.0"))

        # CMDP cost weights
        self.w_c1 = float(os.environ.get("COST_W_C1", "10.0"))
        self.w_c1_dense = float(os.environ.get("COST_W_C1_DENSE", "5.0"))
        self.w_c2 = float(os.environ.get("COST_W_C2", "1.0"))
        self.w_c3 = float(os.environ.get("COST_W_C3", "0.1"))
        self.w_c4 = float(os.environ.get("COST_W_C4", "5.0"))

        self.P_building_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_BUILDING_MAX", "2.273834"))
        self.P_grid_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_GRID_MAX", "27.127751"))

        self._prev_net_consumption = None
        self._step_count = 0

        print(f"[OmniEnvV2] STEMS weights: eco={self.mu_economic} "
              f"grid={self.alpha_grid} build={self.alpha_build} "
              f"ramp={self.beta_ramp} renew={self.xi_renewable} "
              f"ev={self.lambda_ev}")
        print(f"[OmniEnvV2] Cost weights: C1={self.w_c1} C1d={self.w_c1_dense} "
              f"C2={self.w_c2} C3={self.w_c3} C4={self.w_c4}")
        print(f"[OmniEnvV2] Obs dim={self.observation_space.shape} "
              f"Act dim={self.action_space.shape}")

    def _get_citylearn(self):
        cur = self._inner
        seen = set()
        for _ in range(40):
            if cur is None or id(cur) in seen:
                break
            seen.add(id(cur))
            if (hasattr(cur, 'buildings') and hasattr(cur, 'time_step')
                    and hasattr(cur.buildings, '__len__') and len(cur.buildings) > 0):
                return cur
            for attr in ('base', 'env', 'unwrapped', '_env', 'raw_env'):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
            else:
                break
        return None

    def _compute_ev_reward(self, action_flat: np.ndarray, info: dict) -> float:
        """
        Penalize under-charging of connected EVs proportional to urgency.
        R_ev = -lambda_ev * sum(urgency_i * max(0, min_action_i - actual_action_i))
        """
        city = self._get_citylearn()
        if city is None:
            return 0.0

        t_now = int(getattr(city, 'time_step', 0))
        t_idx = max(0, t_now - 1)
        buildings = list(getattr(city, 'buildings', []))
        penalty = 0.0

        names_raw = getattr(city, 'action_names', [])
        if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
            flat_names = names_raw[0]
        elif isinstance(names_raw, list):
            flat_names = []
            for sub in names_raw:
                if isinstance(sub, list):
                    flat_names.extend(sub)
                else:
                    flat_names.append(sub)
        else:
            return 0.0

        ev_key = "electric_vehicle_storage_charger_"
        batt_positions = [i for i, n in enumerate(flat_names)
                          if str(n).lower() == "electrical_storage"]
        if len(batt_positions) != len(buildings):
            return 0.0

        for b_idx in range(len(buildings)):
            start = batt_positions[b_idx]
            end = batt_positions[b_idx + 1] if b_idx + 1 < len(buildings) else len(flat_names)
            b_names = flat_names[start:end]

            chargers = getattr(buildings[b_idx], 'electric_vehicle_chargers', None) or []
            local_ev_idx = 0
            for i, n in enumerate(b_names):
                if ev_key in str(n).lower():
                    if local_ev_idx >= len(chargers):
                        local_ev_idx += 1
                        continue
                    gidx = start + i
                    if gidx >= len(action_flat):
                        local_ev_idx += 1
                        continue

                    ch = chargers[local_ev_idx]
                    sim = getattr(ch, 'charger_simulation',
                                  getattr(ch, '_Charger__charger_simulation', None))
                    if sim is None:
                        local_ev_idx += 1
                        continue

                    try:
                        state_arr = np.asarray(
                            getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                        dep_arr = np.asarray(
                            getattr(sim, '_electric_vehicle_departure_time'), dtype=float)
                        req_arr = np.asarray(
                            getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)

                        if t_now >= len(state_arr) or float(state_arr[t_now]) != 1.0:
                            local_ev_idx += 1
                            continue

                        dep_hours = float(dep_arr[t_now])
                        req_soc = float(req_arr[t_now])
                        if not np.isfinite(req_soc):
                            req_soc = 1.0

                        ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                        ev_soc = 0.0
                        ev_cap = 0.0
                        if ev_obj is not None:
                            batt = getattr(ev_obj, 'battery', None)
                            if batt is not None:
                                ev_cap = float(getattr(batt, 'capacity', 0) or 0)
                                soc_data = getattr(batt, 'soc', None)
                                if soc_data is not None:
                                    soc_np = np.asarray(soc_data, dtype=float)
                                    if 0 <= t_idx < len(soc_np):
                                        ev_soc = float(np.clip(soc_np[t_idx], 0, 1))

                        max_p = float(getattr(ch, 'max_charging_power', 0) or 0)
                        if isinstance(max_p, np.ndarray):
                            max_p = float(max_p.ravel()[0])

                        deficit = max(0.0, req_soc - ev_soc)
                        if deficit <= 1e-6:
                            local_ev_idx += 1
                            continue

                        tau = max(1, int(dep_hours)) if np.isfinite(dep_hours) and dep_hours > 0 else 999
                        if ev_cap > 0 and max_p > 0:
                            max_soc_per_step = (max_p * 0.95) / ev_cap
                            steps_needed = deficit / max(max_soc_per_step, 1e-9)
                            urgency = min(1.0, steps_needed / tau)
                        else:
                            urgency = 1.0

                        if ev_cap > 0 and max_p > 0 and tau > 0:
                            max_soc_per_step = (max_p * 0.95) / ev_cap
                            min_action = min(1.0, deficit / (tau * max_soc_per_step))
                        else:
                            min_action = 1.0

                        actual_action = float(action_flat[gidx])
                        shortfall = max(0.0, min_action - actual_action)
                        penalty += urgency * shortfall

                    except Exception:
                        pass
                    local_ev_idx += 1

        return -self.lambda_ev * penalty

    def _compute_custom_stems_reward(self, info: dict, action_flat: np.ndarray) -> float:
        """STEMS reward with tuned weights + EV component."""
        city = self._get_citylearn()
        if city is None:
            return 0.0

        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)
        buildings = list(getattr(city, 'buildings', []))

        # Economic
        try:
            prices = buildings[0].pricing.electricity_pricing
            price = float(prices[t_idx]) if hasattr(prices, '__len__') and len(prices) > t_idx else 0.17
        except Exception:
            price = 0.17

        total_net = 0.0
        for b in buildings:
            try:
                nec = getattr(b, 'net_electricity_consumption', None)
                if nec is not None and hasattr(nec, '__len__') and len(nec) > t_idx:
                    total_net += float(nec[t_idx])
            except Exception:
                pass

        import_kwh = max(0.0, total_net)
        export_kwh = max(0.0, -total_net)
        export_factor = float(os.environ.get("CITYLEARN_EXPORT_FACTOR", "1.0"))
        r_economic = -self.mu_economic * price * (import_kwh - export_factor * export_kwh)

        # Stability Grid
        grid_ratio = import_kwh / max(1e-6, self.P_grid_max)
        r_stability_grid = self.alpha_grid * (1.0 - min(grid_ratio ** 2, 4.0))

        # Stability Building
        building_sum = 0.0
        b_count = 0
        for b in buildings:
            try:
                nec = getattr(b, 'net_electricity_consumption', None)
                if nec is not None and hasattr(nec, '__len__') and len(nec) > t_idx:
                    p_i = float(nec[t_idx])
                    ratio = abs(p_i) / max(1e-6, self.P_building_max)
                    building_sum += 1.0 - min(ratio, 4.0)
                    b_count += 1
            except Exception:
                pass
        r_stability_build = self.alpha_build * (building_sum / max(1, b_count)) if b_count > 0 else 0.0

        # Ramp
        if self._prev_net_consumption is None:
            ramp_delta = 0.0
        else:
            ramp_delta = abs(total_net - self._prev_net_consumption)
        self._prev_net_consumption = total_net
        r_stability_ramp = -self.beta_ramp * (ramp_delta / max(1e-6, self.P_grid_max))

        # Renewable
        solar_gen = 0.0
        for b in buildings:
            try:
                sg = getattr(b, 'solar_generation', None)
                if sg is not None and hasattr(sg, '__len__') and len(sg) > t_idx:
                    solar_gen += abs(float(sg[t_idx]))
            except Exception:
                pass
        if solar_gen > 0 or import_kwh > 0:
            solar_ratio = solar_gen / (solar_gen + import_kwh)
            r_renewable = self.xi_renewable * min(solar_ratio, 1.0)
        else:
            r_renewable = 0.0

        # EV Charging (NEW)
        r_ev = self._compute_ev_reward(action_flat, info)

        return float(r_economic + r_stability_grid + r_stability_build +
                     r_stability_ramp + r_renewable + r_ev)

    def _compute_rebalanced_cost(self, info: dict) -> float:
        """Rebalanced cost: C1×10 + C1_dense×5 + C2×1 + C3×0.1 + C4×5"""
        c1 = float(info.get('cost_ev_departure', 0.0))
        c1_dense = float(info.get('cost_ev_dense', 0.0))
        c2 = float(info.get('cost_stems_battery', 0.0))
        c3 = float(info.get('cost_stems_building_power', 0.0))
        c4 = float(info.get('cost_stems_grid_power', 0.0))

        return float(self.w_c1 * c1 +
                     self.w_c1_dense * c1_dense +
                     self.w_c2 * c2 +
                     self.w_c3 * c3 +
                     self.w_c4 * c4)

    def reset(self, **kwargs):
        obs, info = self._inner.reset(**kwargs)
        self._prev_net_consumption = None
        self._step_count = 0
        info = dict(info)
        info['cost'] = 0.0
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).ravel()
        obs, reward_base, term, trunc, info = self._inner.step(action)
        self._step_count += 1
        info = dict(info)

        reward = self._compute_custom_stems_reward(info, action)
        cost = self._compute_rebalanced_cost(info)
        info['cost'] = float(cost)

        if self._step_count <= 10 or self._step_count % 1000 == 0:
            c1 = info.get('cost_ev_departure', 0)
            c2 = info.get('cost_stems_battery', 0)
            c3 = info.get('cost_stems_building_power', 0)
            c4 = info.get('cost_stems_grid_power', 0)
            print(f"  [OmniV2] t={self._step_count} r={reward:.3f} "
                  f"cost={cost:.3f} C1={c1:.3f} C2={c2:.3f} "
                  f"C3={c3:.3f} C4={c4:.3f}")

        return np.asarray(obs, dtype=np.float32), float(reward), bool(term), bool(trunc), info

    def render(self):
        pass

    def close(self):
        self._inner.close()
