"""
OmniSafe CMDP registration for V2 training (FOCOPS / PPOLag).

Uses @env_register + CMDP base class. Returns torch tensors.
step() returns 6 values: (obs, reward, cost, terminated, truncated, info)

Changes from previous omni_env:
  1. ForecastObsWrapper for 24h lookahead
  2. EV charging reward in STEMS (fixes 90% C1 violations)
  3. Rebalanced CMDP cost signal (C1×10, C3×0.1, C4×5)
  4. Tuned STEMS weights (safety > economics)
"""
from __future__ import annotations
import os
import sys
from typing import Any, ClassVar
import numpy as np
import torch
import gymnasium as gym

from omnisafe.envs.core import CMDP, env_register

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper


@env_register
class CityLearnCMDPv2(CMDP):
    """
    OmniSafe CMDP with forecast obs + EV reward + rebalanced cost.
    Environment ID: CityLearnSafety-V2G-v2
    """
    _support_envs: ClassVar[list[str]] = ['CityLearnSafety-V2G-v2']

    need_time_limit_wrapper: bool = False
    need_auto_reset_wrapper: bool = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)

        base: gym.Env = make_base_env(central_agent=True)
        safety = CityLearnSafetyEnvV3(base)
        forecast = ForecastObsWrapper(safety, forecast_horizon=24)

        # P0: Add spatial observations (per-building C3 headroom, SoC spread, etc.)
        if os.environ.get("CITYLEARN_SPATIAL_OBS", "0") == "1":
            p_bmax = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
            env_final = SpatialGraphFeaturesWrapper(forecast, num_buildings=17, p_building_max=p_bmax)
            print(f"[CMDPv2] Spatial obs ENABLED (+68 dims, P_building_max={p_bmax})")
        else:
            env_final = forecast

        self._env = env_final
        self._observation_space = env_final.observation_space
        self._action_space = env_final.action_space
        self._num_envs = 1
        self._max_episode_steps = 8759

        # STEMS reward weights (safety-first)
        self.mu_economic = float(os.environ.get("STEMS_MU_ECONOMIC", "0.3"))
        self.alpha_grid = float(os.environ.get("STEMS_ALPHA_GRID", "3.0"))
        self.alpha_build = float(os.environ.get("STEMS_ALPHA_BUILD", "2.0"))
        self.beta_ramp = float(os.environ.get("STEMS_BETA_RAMP", "0.5"))
        self.xi_renewable = float(os.environ.get("STEMS_XI_RENEWABLE", "0.2"))
        self.lambda_ev = float(os.environ.get("STEMS_LAMBDA_EV", "5.0"))

        # CMDP cost weights (C1 boosted, C3 dampened)
        self.w_c1 = float(os.environ.get("COST_W_C1", "10.0"))
        self.w_c1_dense = float(os.environ.get("COST_W_C1_DENSE", "5.0"))
        self.w_c2 = float(os.environ.get("COST_W_C2", "1.0"))
        self.w_c3 = float(os.environ.get("COST_W_C3", "0.1"))
        self.w_c4 = float(os.environ.get("COST_W_C4", "5.0"))

        self.P_building_max = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "2.273834"))
        self.P_grid_max = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "27.127751"))

        self._prev_net = None
        self._step_count = 0

        print(f"[CMDPv2] obs={self._observation_space.shape} act={self._action_space.shape}")
        print(f"[CMDPv2] STEMS: eco={self.mu_economic} grid={self.alpha_grid} "
              f"build={self.alpha_build} ramp={self.beta_ramp} renew={self.xi_renewable} "
              f"ev={self.lambda_ev}")
        print(f"[CMDPv2] Cost: C1={self.w_c1} C1d={self.w_c1_dense} "
              f"C2={self.w_c2} C3={self.w_c3} C4={self.w_c4}")

    def _get_citylearn(self):
        cur = self._env
        seen = set()
        for _ in range(40):
            if cur is None or id(cur) in seen:
                break
            seen.add(id(cur))
            if hasattr(cur, 'buildings') and hasattr(cur, 'time_step') and \
               hasattr(cur.buildings, '__len__') and len(cur.buildings) > 0:
                return cur
            for attr in ('base', 'env', 'unwrapped', '_env', 'raw_env'):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
            else:
                break
        return None

    def _ev_reward(self, action_np: np.ndarray) -> float:
        """Penalize under-charging of connected EVs proportional to urgency."""
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
                flat_names.extend(sub) if isinstance(sub, list) else flat_names.append(sub)
        else:
            return 0.0

        ev_key = "electric_vehicle_storage_charger_"
        batt_pos = [i for i, n in enumerate(flat_names) if str(n).lower() == "electrical_storage"]
        if len(batt_pos) != len(buildings):
            return 0.0

        for b_idx in range(len(buildings)):
            start = batt_pos[b_idx]
            end = batt_pos[b_idx + 1] if b_idx + 1 < len(buildings) else len(flat_names)
            chargers = getattr(buildings[b_idx], 'electric_vehicle_chargers', None) or []
            li = 0
            for i, n in enumerate(flat_names[start:end]):
                if ev_key not in str(n).lower():
                    continue
                if li >= len(chargers):
                    li += 1; continue
                gidx = start + i
                if gidx >= len(action_np):
                    li += 1; continue
                ch = chargers[li]
                sim = getattr(ch, 'charger_simulation', getattr(ch, '_Charger__charger_simulation', None))
                if sim is None:
                    li += 1; continue
                try:
                    sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                    da = np.asarray(getattr(sim, '_electric_vehicle_departure_time'), dtype=float)
                    ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                    if t_now >= len(sa) or float(sa[t_now]) != 1.0:
                        li += 1; continue
                    dh = float(da[t_now]); rs = float(ra[t_now])
                    if not np.isfinite(rs): rs = 1.0
                    ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                    es, ec = 0.0, 0.0
                    if ev_obj:
                        bt = getattr(ev_obj, 'battery', None)
                        if bt:
                            ec = float(getattr(bt, 'capacity', 0) or 0)
                            sd = getattr(bt, 'soc', None)
                            if sd is not None:
                                sn = np.asarray(sd, dtype=float)
                                if 0 <= t_idx < len(sn): es = float(np.clip(sn[t_idx], 0, 1))
                    mp = float(getattr(ch, 'max_charging_power', 0) or 0)
                    if isinstance(mp, np.ndarray): mp = float(mp.ravel()[0])
                    deficit = max(0.0, rs - es)
                    if deficit <= 1e-6:
                        li += 1; continue
                    tau = max(1, int(dh)) if np.isfinite(dh) and dh > 0 else 999
                    if ec > 0 and mp > 0:
                        mps = (mp * 0.95) / ec
                        urgency = min(1.0, (deficit / max(mps, 1e-9)) / tau)
                        min_act = min(1.0, deficit / (tau * mps))
                    else:
                        urgency, min_act = 1.0, 1.0
                    shortfall = max(0.0, min_act - float(action_np[gidx]))
                    penalty += urgency * shortfall
                except Exception:
                    pass
                li += 1
        return -self.lambda_ev * penalty

    def _stems_reward(self, info: dict, action_np: np.ndarray) -> float:
        """Custom STEMS with tuned weights + EV component."""
        city = self._get_citylearn()
        if city is None:
            return 0.0
        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)
        buildings = list(getattr(city, 'buildings', []))

        # Economic
        try:
            pr = buildings[0].pricing.electricity_pricing
            price = float(pr[t_idx]) if hasattr(pr, '__len__') and len(pr) > t_idx else 0.17
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
        imp = max(0.0, total_net)
        exp = max(0.0, -total_net)
        ef = float(os.environ.get("CITYLEARN_EXPORT_FACTOR", "1.0"))
        r_eco = -self.mu_economic * price * (imp - ef * exp)

        # Grid stability
        r_sg = self.alpha_grid * (1.0 - min((imp / max(1e-6, self.P_grid_max)) ** 2, 4.0))

        # Building stability
        bs, bc = 0.0, 0
        for b in buildings:
            try:
                nec = getattr(b, 'net_electricity_consumption', None)
                if nec is not None and hasattr(nec, '__len__') and len(nec) > t_idx:
                    ratio = abs(float(nec[t_idx])) / max(1e-6, self.P_building_max)
                    bs += 1.0 - min(ratio, 4.0); bc += 1
            except Exception:
                pass
        r_sb = self.alpha_build * (bs / max(1, bc)) if bc > 0 else 0.0

        # Ramp
        rd = abs(total_net - self._prev_net) if self._prev_net is not None else 0.0
        self._prev_net = total_net
        r_ramp = -self.beta_ramp * (rd / max(1e-6, self.P_grid_max))

        # Renewable
        sg = 0.0
        for b in buildings:
            try:
                s = getattr(b, 'solar_generation', None)
                if s is not None and hasattr(s, '__len__') and len(s) > t_idx:
                    sg += abs(float(s[t_idx]))
            except Exception:
                pass
        r_ren = self.xi_renewable * min(sg / (sg + imp), 1.0) if (sg + imp) > 0 else 0.0

        # EV (NEW)
        r_ev = self._ev_reward(action_np)

        return float(r_eco + r_sg + r_sb + r_ramp + r_ren + r_ev)

    def _rebalanced_cost(self, info: dict) -> float:
        """Rebalanced: C1×10 + C1_dense×5 + C2×1 + C3×0.1 + C4×5"""
        return float(
            self.w_c1 * float(info.get('cost_ev_departure', 0.0)) +
            self.w_c1_dense * float(info.get('cost_ev_dense', 0.0)) +
            self.w_c2 * float(info.get('cost_stems_battery', 0.0)) +
            self.w_c3 * float(info.get('cost_stems_building_power', 0.0)) +
            self.w_c4 * float(info.get('cost_stems_grid_power', 0.0)))

    # ── OmniSafe API ──

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._prev_net = None
        self._step_count = 0
        return torch.as_tensor(obs, dtype=torch.float32), info

    def step(self, action):
        """Returns 6 values: (obs, reward, cost, terminated, truncated, info)"""
        if isinstance(action, torch.Tensor):
            a = action.detach().cpu().numpy().ravel()
        else:
            a = np.asarray(action, dtype=np.float32).ravel()

        obs, _reward_base, terminated, truncated, info = self._env.step(a)
        self._step_count += 1

        reward = self._stems_reward(info, a)
        cost = self._rebalanced_cost(info)

        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        reward_t = torch.as_tensor(reward, dtype=torch.float32)
        cost_t = torch.as_tensor(cost, dtype=torch.float32)
        terminated_t = torch.as_tensor(terminated, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool)

        # Metrics for OmniSafe logger
        for key, value in list(info.items()):
            if isinstance(value, (int, float)):
                info[f'Metrics/{key}'] = float(value)

        if self._step_count <= 5 or self._step_count % 2000 == 0:
            print(f"  [CMDPv2] t={self._step_count} r={reward:.3f} cost={cost:.3f} "
                  f"C1={info.get('cost_ev_departure',0):.2f} "
                  f"C4={info.get('cost_stems_grid_power',0):.2f}")

        return obs_t, reward_t, cost_t, terminated_t, truncated_t, info

    def close(self) -> None:
        if hasattr(self._env, 'close'):
            self._env.close()

    def render(self):
        return None

    def set_seed(self, seed: int) -> None:
        pass
