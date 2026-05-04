"""
OmniSafe CMDP env with ActionProjection shield. Registers: CityLearnSafety-V2G-v2-shield
Same STEMS reward + rebalanced cost as V2, plus shield cost passthrough.
"""
from __future__ import annotations
import os
from typing import Any, Dict
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from omnisafe.envs.core import CMDP, env_register


def _unwrap_to_citylearn(env):
    cur = env; seen = set()
    for _ in range(60):
        if cur is None or id(cur) in seen: break
        seen.add(id(cur))
        blds = getattr(cur, "buildings", None); ts = getattr(cur, "time_step", None)
        if blds is not None and hasattr(blds, "__len__") and len(blds) > 0 and ts is not None: return cur
        for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur: cur = nxt; break
        else: break
    return None


@env_register
class CityLearnCMDPv2Shield(CMDP):
    _support_envs = ["CityLearnSafety-V2G-v2-shield"]
    need_time_limit_wrapper = False
    need_auto_reset_wrapper = True

    def __init__(self, env_id: str, **kwargs):
        super().__init__(env_id, **kwargs)
        from scripts.make_env import make_base_env
        from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
        from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
        from citylearn_safe.action_projection import ActionProjectionWrapper

        base = make_base_env()
        safety = CityLearnSafetyEnvV3(base,
            soc_min=float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0")),
            soc_max=float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95")))

        use_shield = os.environ.get("CITYLEARN_USE_SHIELD", "1") == "1"
        if use_shield:
            inner = ActionProjectionWrapper(safety, verbose=int(os.environ.get("SHIELD_VERBOSE", "1")))
            print("[V2-Shield] ActionProjectionWrapper ENABLED")
        else:
            inner = safety
            print("[V2-Shield] ActionProjectionWrapper DISABLED")

        forecast = ForecastObsWrapper(inner)
        self._env = forecast

        obs_space = forecast.observation_space
        act_space = forecast.action_space
        if isinstance(act_space, list):
            lows = np.concatenate([np.asarray(sp.low).ravel() for sp in act_space])
            highs = np.concatenate([np.asarray(sp.high).ravel() for sp in act_space])
            act_space = spaces.Box(low=lows, high=highs, dtype=np.float32)
        if isinstance(obs_space, list):
            total_dim = sum(int(np.prod(sp.shape)) for sp in obs_space)
            obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
        self._observation_space = obs_space
        self._action_space = act_space
        self._num_envs = 1
        self._max_episode_steps = 8759

        self.mu_economic = float(os.environ.get("STEMS_MU_ECONOMIC", "0.3"))
        self.alpha_grid = float(os.environ.get("STEMS_ALPHA_GRID", "3.0"))
        self.alpha_build = float(os.environ.get("STEMS_ALPHA_BUILD", "2.0"))
        self.beta_ramp = float(os.environ.get("STEMS_BETA_RAMP", "0.5"))
        self.xi_renewable = float(os.environ.get("STEMS_XI_RENEWABLE", "0.2"))
        self.lambda_ev = float(os.environ.get("STEMS_LAMBDA_EV", "1.0"))
        self.export_factor = float(os.environ.get("CITYLEARN_EXPORT_FACTOR", "1.0"))
        self.P_building_max = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "2.273834"))
        self.P_grid_max = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "27.127751"))
        self.w_c1 = float(os.environ.get("COST_W_C1", "10.0"))
        self.w_c1_dense = float(os.environ.get("COST_W_C1_DENSE", "5.0"))
        self.w_c2 = float(os.environ.get("COST_W_C2", "1.0"))
        self.w_c3 = float(os.environ.get("COST_W_C3", "0.3"))
        self.w_c4 = float(os.environ.get("COST_W_C4", "5.0"))
        self.shield_penalty_weight = float(os.environ.get("SHIELD_PENALTY_WEIGHT", "5.0"))
        self._prev_net_consumption = None
        print(f"[V2-Shield] STEMS: mu={self.mu_economic} ag={self.alpha_grid} ab={self.alpha_build} br={self.beta_ramp} xi={self.xi_renewable} lev={self.lambda_ev}")
        print(f"[V2-Shield] Cost: C1={self.w_c1} C1d={self.w_c1_dense} C2={self.w_c2} C3={self.w_c3} C4={self.w_c4} shield_pen={self.shield_penalty_weight}")

    def _get_citylearn(self):
        return _unwrap_to_citylearn(self._env)

    def _stems_reward(self, info, action_np):
        price = float(info.get("electricity_price", 0.17))
        imp = float(info.get("grid_import_kwh", 0.0))
        exp = float(info.get("grid_export_kwh", 0.0))
        r_econ = -self.mu_economic * price * (imp - self.export_factor * exp)

        r_grid = self.alpha_grid * (1.0 - (imp / max(1e-6, self.P_grid_max)) ** 2)

        r_bld = 0.0
        city = self._get_citylearn()
        if city is not None and getattr(city, "buildings", None):
            t_idx = max(0, int(getattr(city, "time_step", 0)) - 1)
            s, c = 0.0, 0
            for b in city.buildings:
                try:
                    nec = getattr(b, "net_electricity_consumption", None)
                    if nec is not None and hasattr(nec, "__len__") and len(nec) > t_idx:
                        s += 1.0 - abs(float(nec[t_idx])) / max(1e-6, self.P_building_max); c += 1
                except: continue
            if c > 0: r_bld = self.alpha_build * (s / c)

        net = float(info.get("step_net_consumption_kwh", 0.0))
        ramp = abs(net - self._prev_net_consumption) if self._prev_net_consumption is not None else 0.0
        self._prev_net_consumption = net
        r_ramp = -self.beta_ramp * ramp / max(1e-6, self.P_grid_max)

        solar = float(info.get("solar_generation_kwh", 0.0))
        denom = solar + max(0.0, net)
        r_renew = self.xi_renewable * (solar / denom) if denom > 1e-6 else 0.0

        r_ev = self._ev_reward(action_np)
        return float(r_econ + r_grid + r_bld + r_ramp + r_renew + r_ev)

    def _ev_reward(self, action_np):
        if self.lambda_ev <= 0: return 0.0
        city = self._get_citylearn()
        if city is None: return 0.0
        t_now = int(getattr(city, "time_step", 0)); t_idx = max(0, t_now - 1)
        pen = 0.0
        for b in city.buildings:
            for ch in (getattr(b, "electric_vehicle_chargers", None) or []):
                sim = getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))
                if sim is None: continue
                try:
                    st = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                    dep = np.asarray(getattr(sim, "_electric_vehicle_departure_time"), dtype=float)
                    req = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
                    eid = getattr(sim, "_electric_vehicle_id")
                except: continue
                if t_now >= len(st) or float(st[t_now]) != 1.0 or eid[t_now] is None: continue
                d = float(dep[t_now]) if t_now < len(dep) else float("nan")
                T = max(1.0, d) if np.isfinite(d) and d > 0 else 999.0
                rs = float(req[t_now]) if t_now < len(req) else 1.0
                if not np.isfinite(rs): rs = 1.0
                ev_soc, ev_cap = 0.0, 0.0
                evo = getattr(ch, "connected_electric_vehicle", None)
                if evo:
                    bt = getattr(evo, "battery", None)
                    if bt:
                        ev_cap = float(getattr(bt, "capacity", 0) or 0)
                        sd = getattr(bt, "soc", None)
                        if sd is not None:
                            sn = np.asarray(sd, dtype=float).ravel()
                            if 0 <= t_idx < len(sn): ev_soc = float(np.clip(sn[t_idx], 0, 1))
                deficit = max(0.0, rs - ev_soc)
                if deficit <= 1e-6 or ev_cap <= 0: continue
                mp = getattr(ch, "max_charging_power", getattr(ch, "_Charger__max_charging_power", 0))
                if isinstance(mp, np.ndarray): mp = float(mp.ravel()[0])
                mp = float(mp or 0)
                msps = (mp * 0.95) / ev_cap if ev_cap > 0 else 0
                urg = min(1.0, (deficit / max(1e-9, msps)) / T) if msps > 0 else 1.0
                pen += urg * deficit
        return -self.lambda_ev * pen

    def _rebalanced_cost(self, info):
        c1 = float(info.get("cost_ev_departure", 0.0))
        c1d = float(info.get("ev_v3_missed_charge_soc", 0.0))
        c2 = float(info.get("battery_soc_violation_frac", 0.0))
        c3 = float(info.get("building_power_violation_count", 0.0))
        if c3 == 0: c3 = float(info.get("cost_stems_building_power", 0.0))
        c4 = float(info.get("cost_stems_grid_power", 0.0))
        if c4 == 0: c4 = float(info.get("grid_power_violation", 0.0))
        return float(self.w_c1 * c1 + self.w_c1_dense * c1d + self.w_c2 * c2 + self.w_c3 * c3 + self.w_c4 * c4)

    def reset(self, seed=None, options=None):
        self._prev_net_consumption = None
        obs, info = self._env.reset()
        return torch.as_tensor(self._flat(obs), dtype=torch.float32), dict(info) if info else {}

    def step(self, action):
        a_np = action.detach().cpu().numpy().ravel()
        a_np = np.clip(a_np, self._action_space.low, self._action_space.high)
        obs, _, term, trunc, info = self._env.step(a_np)
        info = dict(info) if info else {}
        rew = self._stems_reward(info, a_np)
        cost = self._rebalanced_cost(info)
        shield_delta = float(info.get("ap_action_delta_l2", 0.0))
        shield_pen = self.shield_penalty_weight * shield_delta
        rew -= shield_pen  # SE-RL: penalty in reward (Markgraf 2025, Eq.23)
        info["v2_rew"] = float(rew); info["v2_cost"] = float(cost); info["v2_shield_pen"] = float(shield_pen)
        return (torch.as_tensor(self._flat(obs), dtype=torch.float32),
                torch.as_tensor(rew, dtype=torch.float32),
                torch.as_tensor(cost, dtype=torch.float32),
                torch.as_tensor(term, dtype=torch.float32),
                torch.as_tensor(trunc, dtype=torch.float32), info)

    def close(self):
        try: self._env.close()
        except: pass
    def render(self): return None
    def set_seed(self, seed): pass

    def _flat(self, obs):
        if isinstance(obs, (list, tuple)):
            return np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs])
        return np.asarray(obs, dtype=np.float32).ravel()
