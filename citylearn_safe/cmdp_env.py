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
from citylearn_safe.safety_env import CityLearnSafetyEnv
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper


@env_register
class CityLearnCMDP(CMDP):
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
        safety = CityLearnSafetyEnv(base)
        forecast = ForecastObsWrapper(safety, forecast_horizon=24)

        # P0.5: Add temporal history if STEMS v3 requested
        temporal_window = int(os.environ.get("CITYLEARN_TEMPORAL_WINDOW", "0"))
        if temporal_window > 0:
            from citylearn_safe.temporal_obs_wrapper import TemporalHistoryWrapper
            from citylearn_safe.schema_index import build_index
            # Auto-detect building count
            _e = safety
            _n_bld = 0
            for _ in range(20):
                if hasattr(_e, 'buildings') and len(getattr(_e, 'buildings', [])) > 0:
                    _n_bld = len(_e.buildings)
                    break
                _e = getattr(_e, 'env', getattr(_e, 'base', None))
                if _e is None:
                    break
            if _n_bld == 0:
                _n_bld = 5
            obs_idx = build_index(safety, expected_buildings=_n_bld)

            # Rich temporal: includes solar, EV SoC/departure, hour encoding
            if os.environ.get("CITYLEARN_TEMPORAL_RICH", "0") == "1":
                from citylearn_safe.temporal_obs_wrapper import build_rich_history_config
                rich_cfg = build_rich_history_config(obs_idx, _n_bld)
                history_indices = rich_cfg['history_indices']
                print(f"[CMDPv2] Rich temporal history: {rich_cfg['features_per_step']} "
                      f"features/step, {rich_cfg['features_per_node']} features/node")
            else:
                from citylearn_safe.temporal_obs_wrapper import build_basic_history_indices
                history_indices = build_basic_history_indices(obs_idx, _n_bld)

            forecast = TemporalHistoryWrapper(forecast, history_indices, temporal_window)
            print(f"[CMDPv2] Temporal history ENABLED "
                  f"(T={temporal_window}, +{len(history_indices)*temporal_window} dims)")

        # P0: Add spatial observations (per-building C3 headroom, SoC spread, etc.)
        if os.environ.get("CITYLEARN_SPATIAL_OBS", "0") == "1":
            p_bmax = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
            # Auto-detect building count from the underlying CityLearn env
            n_buildings = int(os.environ.get("CITYLEARN_NUM_BUILDINGS", "0"))
            if n_buildings == 0:
                # Walk wrapper chain to find CityLearnEnv.buildings
                _e = safety
                for _ in range(20):
                    if hasattr(_e, 'buildings') and len(getattr(_e, 'buildings', [])) > 0:
                        n_buildings = len(_e.buildings)
                        break
                    _e = getattr(_e, 'env', getattr(_e, 'base', None))
                    if _e is None:
                        break
                if n_buildings == 0:
                    n_buildings = 17  # fallback
            n_extra = n_buildings * 4
            env_final = SpatialGraphFeaturesWrapper(forecast, num_buildings=n_buildings, p_building_max=p_bmax)
            print(f"[CMDPv2] Spatial obs ENABLED (+{n_extra} dims, {n_buildings} buildings, P_building_max={p_bmax})")
        else:
            env_final = forecast

        # Sauté MDP: budget-aware obs augmentation for C1 (EV charging)
        if os.environ.get("CITYLEARN_EV_SAUTE", "0") == "1":
            from citylearn_safe.saute_ev_wrapper import SauteEVBudgetWrapper
            env_final = SauteEVBudgetWrapper(env_final)

        # Sauté MDP: budget-aware obs augmentation for C4 (grid power)
        if os.environ.get("SAUTE_C4_ENABLED", "0") == "1":
            from citylearn_safe.saute_constraint_wrapper import SauteConstraintWrapper
            env_final = SauteConstraintWrapper(
                env_final,
                cost_key="cost_stems_grid_power",
                budget_d=float(os.environ.get("SAUTE_C4_BUDGET", "1500")),
                penalty=float(os.environ.get("SAUTE_C4_PENALTY", "5.0")),
                gamma=float(os.environ.get("SAUTE_C4_GAMMA", "1.0")),
                label="c4",
            )

        # Structured building-level coupled budget actions.
        # Exposes one controllable budget per building plus EV-share actions only
        # for buildings that actually have both a battery and an EV.
        if os.environ.get("CITYLEARN_COUPLED_BUDGET_ACTION", "0") == "1":
            from citylearn_safe.coupled_budget_action_wrapper import CoupledBudgetActionWrapper
            env_final = CoupledBudgetActionWrapper(env_final)

        # R29: Action mask/projection wrapper (satisfies C2/C3/C4 power constraints)
        # SE-RL projection (paper-exact clip + penalty) takes priority over rescaling mask
        if os.environ.get("CITYLEARN_SERL_PROJECTION", "0") == "1":
            from citylearn_safe.action_projection_serl import ActionProjectionSERL
            env_final = ActionProjectionSERL(env_final)
            print("[CMDPv2] ActionProjectionSERL ENABLED (SE-RL clip + penalty)")
        elif os.environ.get("CITYLEARN_ACTION_MASK", "0") == "1":
            from citylearn_safe.action_mask_wrapper import ActionMaskWrapper
            env_final = ActionMaskWrapper(env_final)
            print("[CMDPv2] ActionMaskWrapper ENABLED")

        self._env = env_final
        self._observation_space = env_final.observation_space
        self._action_space = env_final.action_space
        self._num_envs = 1
        self._max_episode_steps = 8759

        # Beta actor mode: action space is (0, 1) not (-1, 1)
        self._beta_mode = os.environ.get("CITYLEARN_BETA_ACTOR", "0") == "1"
        if self._beta_mode:
            act_dim = self._action_space.shape[0]
            self._action_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32
            )
            print(f"[CMDPv2] Beta actor mode — action_space=(0,1) dim={act_dim}")

        # STEMS reward weights (safety-first)
        self.mu_economic = float(os.environ.get("STEMS_MU_ECONOMIC", "0.3"))
        self.alpha_grid = float(os.environ.get("STEMS_ALPHA_GRID", "3.0"))
        self.alpha_build = float(os.environ.get("STEMS_ALPHA_BUILD", "2.0"))
        self.beta_ramp = float(os.environ.get("STEMS_BETA_RAMP", "0.5"))
        self.xi_renewable = float(os.environ.get("STEMS_XI_RENEWABLE", "0.2"))
        self.lambda_ev = float(os.environ.get("STEMS_LAMBDA_EV", "0.0"))
        self.alpha_barrier = float(os.environ.get("STEMS_ALPHA_BARRIER", "0.5"))

        # R16: New reward components (battery-only price arbitrage + gentle grid awareness)
        self.alpha_load_shift = float(os.environ.get("STEMS_ALPHA_LOAD_SHIFT", "0.0"))
        self.alpha_grid_mild = float(os.environ.get("STEMS_ALPHA_GRID_MILD", "0.0"))

        # Simple battery price arbitrage: R_batt = -action * price (AL-SAC style)
        self.alpha_price_arb = float(os.environ.get("STEMS_ALPHA_PRICE_ARB", "0.0"))

        # R30: NEC-sign reward — align storage actions with exogenous load direction
        self.alpha_nec_sign = float(os.environ.get("STEMS_ALPHA_NEC_SIGN", "0.0"))

        # R17: Threshold r_sg — only penalize imports above fraction of P_grid_max
        # 0.0 = original (penalize all imports), 0.5 = penalize above 50% P_grid_max
        self.sg_threshold_frac = float(os.environ.get("STEMS_SG_THRESHOLD", "0.0"))

        # Compute mean_price from actual pricing data (adapts to any schema)
        self.mean_price = self._compute_mean_price(safety)

        # V2G-aware reward flags (R13+)
        # Fix 1: r_sb only penalizes imports, not exports (enables V2G)
        self.sb_asymmetric = os.environ.get("STEMS_SB_ASYMMETRIC", "0") == "1"
        # Fix 4: r_sg gives partial credit for net exports
        self.sg_export_credit = float(os.environ.get("STEMS_SG_EXPORT_CREDIT", "0.0"))

        # DEPRECATED: These weights only affect _rebalanced_cost() which feeds
        # OmniSafe's single cost critic. Multi-constraint algorithms (SACLagMulti,
        # PPOLagMulti) bypass this — use cost_weight_* in multi_cfgs instead.
        self.w_c1 = float(os.environ.get("COST_W_C1", "10.0"))
        self.w_c1_dense = float(os.environ.get("COST_W_C1_DENSE", "5.0"))
        self.w_c2 = float(os.environ.get("COST_W_C2", "0.0"))  # SoC clamp + barrier make C2 redundant
        self.w_c3 = float(os.environ.get("COST_W_C3", "0.1"))
        self.w_c4 = float(os.environ.get("COST_W_C4", "5.0"))

        # Use safety env's auto-calibrated values (adapts to any building count)
        self.P_building_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_BUILDING_MAX", str(safety.P_building_max)))
        self.P_grid_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_GRID_MAX", str(safety.P_grid_max)))

        self._prev_net = None
        self._step_count = 0
        self._last_obs = None
        self._execution_projector = None
        self._execution_shield_enabled = os.environ.get("CITYLEARN_EXECUTION_SHIELD", "1") == "1"

        # Discover battery action indices and pair with buildings
        self._batt_action_map = self._discover_battery_actions(safety)

        # EV charger action map (R15a+)
        self._ev_action_map = self._discover_ev_charger_actions(safety)
        self._ev_clamp_enabled = os.environ.get("CITYLEARN_EV_ACTION_CLAMP", "0") == "1"
        self._ev_disconnect_mask_enabled = os.environ.get("CITYLEARN_EV_DISCONNECT_MASK", "0") == "1"
        self._ev_clamp_margin = float(os.environ.get("CITYLEARN_EV_CLAMP_MARGIN", "0.1"))
        self.alpha_ev_guard = float(os.environ.get("STEMS_ALPHA_EV_GUARD", "0.0"))
        self.alpha_v2g_context = float(os.environ.get("STEMS_ALPHA_V2G_CONTEXT", "0.0"))
        self.alpha_peak_shave = float(os.environ.get("STEMS_ALPHA_PEAK_SHAVE", "0.0"))
        self.alpha_ev_solar = float(os.environ.get("STEMS_ALPHA_EV_SOLAR", "0.0"))
        self.alpha_solar_store = float(os.environ.get("STEMS_ALPHA_SOLAR_STORE", "0.0"))
        self.alpha_ev_smart = float(os.environ.get("STEMS_ALPHA_EV_SMART", "0.0"))
        self._solar_store_batt_only = os.environ.get("STEMS_SOLAR_STORE_BATT_ONLY", "0") == "1"
        self.ev_slack_arb_scale = float(os.environ.get("STEMS_EV_SLACK_ARB_SCALE", "0.0"))
        self.alpha_headroom = float(os.environ.get("STEMS_ALPHA_HEADROOM", "0.0"))
        self.alpha_grid_penalty = float(os.environ.get("STEMS_ALPHA_GRID_PENALTY", "0.0"))
        self._ev_saute_shaped_alpha = float(os.environ.get("CITYLEARN_EV_SAUTE_SHAPED_ALPHA", "0.0"))
        self._saute_c4_shaped_alpha = float(os.environ.get("SAUTE_C4_SHAPED_ALPHA", "0.0"))
        self._ev_clamp_count = 0
        self._ev_disconnect_mask_count = 0

        # R23: Solar capacity for r_ev_solar normalization
        self._solar_capacity = 0.0
        _city_init = self._get_citylearn()
        if _city_init is not None:
            for b in _city_init.buildings:
                pv = getattr(b, 'pv', None)
                if pv is not None:
                    self._solar_capacity += abs(float(getattr(pv, 'nominal_power', 0.0) or 0.0))
        if self._solar_capacity <= 0:
            self._solar_capacity = 20.0  # fallback for 5-building schema

        # R18: Washing machine disable (clamp WM action to 0)
        # Diagnostic showed WM draws 46 kW (10x C3 threshold), causing top 15 worst violations
        self._wm_disable = os.environ.get("CITYLEARN_WM_DISABLE", "0") == "1"
        self._wm_action_indices = self._discover_wm_actions(safety)

        # R18: Battery clamp optional (CityLearn handles physical SoC bounds)
        self._batt_clamp_enabled = os.environ.get("CITYLEARN_BATT_CLAMP", "1") == "1"

        # Configurable SoC upper clamp (default 0.94 = legacy, set to 0.88 for R12b)
        self._SOC_UPPER = float(os.environ.get(
            "CITYLEARN_BATT_SOC_UPPER_CLAMP", str(self._SOC_UPPER_DEFAULT)))

        # R18: V2G discharge tracking (proves agent is exploring V2G)
        self._v2g_discharge_count = 0

        # ── Forecast-aware arbitrage (R26h+) ──
        self.alpha_trajectory = float(os.environ.get("STEMS_ALPHA_TRAJECTORY", "0.0"))
        self.traj_ev_w = float(os.environ.get("STEMS_TRAJ_EV_WEIGHT", "1.0"))
        self._traj_forecast_hours = int(os.environ.get("STEMS_TRAJ_FORECAST_HOURS", "24"))
        # Cache annual pricing array for forecast lookups
        self._pricing_arr = None
        _city_fc = self._get_citylearn()
        if _city_fc is not None:
            try:
                _blds = list(getattr(_city_fc, 'buildings', []))
                if _blds:
                    _pr = _blds[0].pricing.electricity_pricing
                    self._pricing_arr = np.asarray(_pr, dtype=float)
            except Exception:
                pass

        print(f"[CMDPv2] obs={self._observation_space.shape} act={self._action_space.shape}")
        print(f"[CMDPv2] Battery clamp: {len(self._batt_action_map)} batteries, "
              f"enabled={self._batt_clamp_enabled}, SOC_UPPER={self._SOC_UPPER:.2f}")
        print(f"[CMDPv2] EV clamp: {len(self._ev_action_map)} chargers, "
              f"enabled={self._ev_clamp_enabled}, margin={self._ev_clamp_margin}, "
              f"guard={self.alpha_ev_guard}, v2g_ctx={self.alpha_v2g_context}, "
              f"peak_shave={self.alpha_peak_shave}, "
              f"ev_solar={self.alpha_ev_solar} (PV_cap={self._solar_capacity:.1f}kW)")
        print(f"[CMDPv2] EV disconnect mask: enabled={self._ev_disconnect_mask_enabled}")
        print(f"[CMDPv2] Sauté shaped_alpha={self._ev_saute_shaped_alpha} "
              f"(0=binary penalty, >0=smooth gradient)")
        print(f"[CMDPv2] STEMS: eco={self.mu_economic} grid={self.alpha_grid} "
              f"build={self.alpha_build} ramp={self.beta_ramp} renew={self.xi_renewable} "
              f"ev={self.lambda_ev} barrier={self.alpha_barrier}")
        print(f"[CMDPv2] WM disable: {self._wm_disable}, indices={self._wm_action_indices}")
        print(f"[CMDPv2] R16: load_shift={self.alpha_load_shift} grid_mild={self.alpha_grid_mild} "
              f"mean_price={self.mean_price:.4f} sg_threshold={self.sg_threshold_frac}")
        print(f"[CMDPv2] Cost: C1={self.w_c1} C1d={self.w_c1_dense} "
              f"C2={self.w_c2} C3={self.w_c3} C4={self.w_c4}")
        # ── Auxiliary temporal prediction targets (R26k) ──
        self._aux_targets_enabled = int(os.environ.get("STEMS_AUX_TARGETS", "0")) > 0

        print(f"[CMDPv2] ForecastArb: alpha={self.alpha_trajectory} "
              f"ev_w={self.traj_ev_w} horizon={self._traj_forecast_hours}h "
              f"pricing_cached={self._pricing_arr is not None}")
        if self._aux_targets_enabled:
            print(f"[CMDPv2] AuxTargets: enabled (3 targets: price, solar, net_load)")
        # Warn about overlapping price-arbitrage signals
        _price_signals = []
        if self.alpha_trajectory > 0:
            _price_signals.append(f"forecast_arb={self.alpha_trajectory}")
        if self.alpha_load_shift > 0:
            _price_signals.append(f"load_shift={self.alpha_load_shift}")
        if self.alpha_price_arb > 0:
            _price_signals.append(f"price_arb={self.alpha_price_arb}")
        if len(_price_signals) > 1:
            print(f"[CMDPv2] WARNING: {len(_price_signals)} overlapping price-arbitrage "
                  f"rewards active: {', '.join(_price_signals)}. "
                  f"Consider using only forecast_arb (replaces load_shift/price_arb).")

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

    def _get_action_mask_max(self, action_idx: int):
        """Get the action mask upper bound for a given action index.

        The ActionMaskWrapper stores bounds on the CityLearn env object
        as _action_mask_safe_max. Returns None if no mask is active.
        """
        city = self._get_citylearn()
        if city is not None:
            safe_max = getattr(city, '_action_mask_safe_max', None)
            if safe_max is not None and action_idx < len(safe_max):
                return float(safe_max[action_idx])
        return None

    def _compute_mean_price(self, safety_env) -> float:
        """Compute mean electricity price from actual data at init (adapts to any schema)."""
        city = None
        cur = safety_env
        for _ in range(20):
            if cur is None:
                break
            if hasattr(cur, 'buildings') and hasattr(cur, 'action_names'):
                city = cur
                break
            cur = getattr(cur, 'env', getattr(cur, 'base', None))
        if city is None:
            return 0.17  # fallback
        try:
            buildings = list(city.buildings)
            if not buildings:
                return 0.17
            pr = buildings[0].pricing.electricity_pricing
            prices = np.asarray(pr, dtype=float)
            mean_p = float(np.mean(prices[prices > 0])) if np.any(prices > 0) else 0.17
            return mean_p
        except Exception:
            return 0.17

    def _load_shift_reward(self, action_np: np.ndarray, price: float) -> float:
        """R27: Solar-aware battery price arbitrage.

        Per-building effective cost = price - solar_credit.
        Solar credit = max(0, solar_gen - load) / P_bmax (EXOGENOUS).

        During solar surplus: effective cost is deeply negative → charge rewarded.
        During expensive + no solar: effective cost is positive → discharge rewarded.
        Signal clipped to [-1, 1] to prevent solar hours dominating 30:1.

        Linear SoC gate: discharge reward scales with SoC (not saturating).
        This forces the critic to learn SoC-dependent value → lookahead emerges.
        """
        if self.alpha_load_shift <= 0 or not self._batt_action_map:
            return 0.0

        city = self._get_citylearn()
        if city is None:
            return 0.0
        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)
        buildings = list(getattr(city, 'buildings', []))

        n_batt = len(self._batt_action_map)
        r_ls = 0.0

        for act_idx, bld_idx, cap, p_max, eta in self._batt_action_map:
            if act_idx >= len(action_np) or bld_idx >= len(buildings):
                continue
            b = buildings[bld_idx]
            act = float(action_np[act_idx])

            # Per-building solar credit (EXOGENOUS — not affected by agent actions)
            solar_credit = 0.0
            try:
                sg = getattr(b, 'solar_generation', None)
                nsl = getattr(b, '_Building__energy_to_non_shiftable_load', None)
                if sg is not None and nsl is not None:
                    sg_val = abs(float(sg[t_idx])) if hasattr(sg, '__len__') and len(sg) > t_idx else 0.0
                    nsl_val = float(nsl[t_idx]) if hasattr(nsl, '__len__') and len(nsl) > t_idx else 0.0
                    solar_surplus = max(0.0, sg_val - nsl_val)
                    solar_credit = solar_surplus / max(1e-6, self.P_building_max)
            except Exception:
                pass

            # Effective cost: price deviation minus solar credit
            eff_cost = price / max(1e-8, self.mean_price) - 1.0 - solar_credit
            eff_cost_signal = max(-1.0, min(1.0, eff_cost))  # CLIP to [-1, 1]

            # Linear SoC gate (forces SoC-dependent value learning)
            soc = 0.5
            try:
                es = getattr(b, 'electrical_storage', None)
                if es is not None and hasattr(es, 'soc') and hasattr(es.soc, '__len__') and len(es.soc) > t_idx:
                    soc = float(np.clip(es.soc[t_idx], 0.01, 0.99))
            except Exception:
                pass

            if act < 0:  # discharge
                gate = max(0.2, soc)  # R28: floor at 0.2 — always penalizes wrong-time discharge
            else:  # charge
                gate = max(0.2, 1.0 - soc)  # R28: floor at 0.2 — always rewards right-time charge

            r_ls += -act * gate * eff_cost_signal

        return self.alpha_load_shift * r_ls / max(1, n_batt)

    def _simple_price_reward(self, action_np: np.ndarray, price: float) -> float:
        """Simple battery price arbitrage: R_batt = -action * normalized_price.

        charge (act>0) at low price  -> positive reward (good)
        discharge (act<0) at high price -> positive reward (good)
        charge (act>0) at high price -> negative reward (bad)
        discharge (act<0) at low price -> negative reward (bad)
        """
        if self.alpha_price_arb <= 0 or not self._batt_action_map:
            return 0.0

        norm_price = price / max(1e-8, self.mean_price)
        n_batt = len(self._batt_action_map)
        total = 0.0

        for act_idx, _bld_idx, _cap, _p_max, _eta in self._batt_action_map:
            if act_idx >= len(action_np):
                continue
            act = float(action_np[act_idx])
            total += -act * (norm_price - 1.0)  # centered: <0 when cheap, >0 when expensive

        return self.alpha_price_arb * total / max(1, n_batt)

    def _nec_sign_reward(self, action_np: np.ndarray) -> float:
        """R30: NEC-sign reward — align storage actions with exogenous NEC direction.

        Uses EXOGENOUS NEC (load + solar only, before storage actions) to determine
        whether a building is importing or exporting:
          - Importing (exo_nec > 0): discharge helps (reward = -act), charge hurts
          - Exporting (exo_nec < 0): charge helps (reward = +act), discharge hurts
        Actions in deadzone (|act| < 0.05) are skipped.
        Averaged over all active devices, scaled by alpha_nec_sign.
        """
        if self.alpha_nec_sign <= 0:
            return 0.0
        city = self._get_citylearn()
        if city is None:
            return 0.0

        t_now = int(getattr(city, 'time_step', 0))
        t_idx = max(0, t_now - 1)
        buildings = list(getattr(city, 'buildings', []))
        total = 0.0
        count = 0

        # Battery actions
        n_devices = 0
        for act_idx, bld_idx, _cap, _p_max, _eta in self._batt_action_map:
            if act_idx >= len(action_np) or bld_idx >= len(buildings):
                continue
            act = float(action_np[act_idx])
            if abs(act) < 0.05:
                n_devices += 1  # count idle devices in denominator to prevent exploit
                continue  # deadzone
            b = buildings[bld_idx]
            try:
                nsl = getattr(b, '_Building__energy_to_non_shiftable_load', None)
                sg = getattr(b, '_Building__solar_generation', None)
                if nsl is None or sg is None:
                    continue
                if not hasattr(nsl, '__len__') or len(nsl) <= t_idx:
                    continue
                if not hasattr(sg, '__len__') or len(sg) <= t_idx:
                    continue
                exo_nec = float(nsl[t_idx]) + float(sg[t_idx])
                if exo_nec > 0:
                    # Building importing: discharge helps, charge hurts
                    total += -act
                elif exo_nec < 0:
                    # Building exporting: charge helps, discharge hurts
                    total += act
                count += 1
            except Exception:
                pass

        # R30d: NEC-sign does NOT apply to EVs.
        # EVs are driven by urgency + V2G arb only.
        # Reason: NEC-sign penalizes EV charging during peak (import hours),
        # but EVs may NEED to charge then to meet departure SoC.
        # Applying NEC-sign to EVs creates an unsolvable conflict.

        denom = count + n_devices
        if denom == 0:
            return 0.0
        return self.alpha_nec_sign * total / denom

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
                        urgency = max(0.3, min(1.0, (deficit / max(mps, 1e-9)) / tau))
                        min_act = min(1.0, deficit / (tau * mps))
                    else:
                        urgency, min_act = 1.0, 1.0
                    # R29: Cap min_act at mask upper bound if action masking is active.
                    # Without this, the agent is penalized for not charging enough
                    # when the mask physically prevents the required charge level.
                    mask_max = self._get_action_mask_max(gidx)
                    if mask_max is not None:
                        min_act = min(min_act, max(0.0, mask_max))
                    shortfall = max(0.0, min_act - float(action_np[gidx]))
                    penalty += urgency * shortfall
                except Exception:
                    pass
                li += 1
        return -self.lambda_ev * penalty

    def _ev_guard_penalty(self, action_np: np.ndarray) -> float:
        """Penalize discharge when EV SoC < required (uses pre-clamp action for gradient).

        Returns negative penalty proportional to urgency * |discharge_action|.
        """
        if self.alpha_ev_guard <= 0 or not self._ev_action_map:
            return 0.0
        city = self._get_citylearn()
        if city is None:
            return 0.0

        t_now = int(getattr(city, 'time_step', 0))
        t_idx = max(0, t_now - 1)
        buildings = list(getattr(city, 'buildings', []))
        penalty = 0.0

        for gidx, b_idx, ch_idx in self._ev_action_map:
            if gidx >= len(action_np) or b_idx >= len(buildings):
                continue
            act = float(action_np[gidx])
            if act >= 0:
                continue  # Charging — no penalty
            chargers = getattr(buildings[b_idx], 'electric_vehicle_chargers', None) or []
            if ch_idx >= len(chargers):
                continue
            ch = chargers[ch_idx]
            sim = getattr(ch, 'charger_simulation',
                          getattr(ch, '_Charger__charger_simulation', None))
            if sim is None:
                continue
            try:
                sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                if t_now >= len(sa) or float(sa[t_now]) != 1.0:
                    continue
                ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                if not np.isfinite(rs):
                    rs = 1.0
                ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                if ev_obj is None:
                    continue
                bt = getattr(ev_obj, 'battery', None)
                if bt is None:
                    continue
                soc_arr = getattr(bt, 'soc', None)
                if soc_arr is None:
                    continue
                sn = np.asarray(soc_arr, dtype=float)
                current_soc = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0
                if current_soc >= rs + self._ev_clamp_margin:
                    continue  # Surplus — discharge OK
                deficit = rs - current_soc
                urgency = min(1.0, deficit * 2.0)  # 0.5 deficit → urgency 1.0
                penalty += urgency * abs(act)
            except Exception:
                pass
        return -self.alpha_ev_guard * penalty

    def _ev_v2g_context_reward(self, action_np: np.ndarray,
                                total_net: float, imp: float, solar: float) -> float:
        """Context-aware V2G reward for EVs with surplus SoC (R15b+).

        Only fires for EVs with soc >= required + margin.
        Rewards: V2G during grid import, charging during solar.
        Penalizes: discharge during solar abundance.
        """
        if self.alpha_v2g_context <= 0 or not self._ev_action_map:
            return 0.0
        city = self._get_citylearn()
        if city is None:
            return 0.0

        t_now = int(getattr(city, 'time_step', 0))
        t_idx = max(0, t_now - 1)
        buildings = list(getattr(city, 'buildings', []))
        r_v2g = 0.0

        for gidx, b_idx, ch_idx in self._ev_action_map:
            if gidx >= len(action_np) or b_idx >= len(buildings):
                continue
            chargers = getattr(buildings[b_idx], 'electric_vehicle_chargers', None) or []
            if ch_idx >= len(chargers):
                continue
            ch = chargers[ch_idx]
            sim = getattr(ch, 'charger_simulation',
                          getattr(ch, '_Charger__charger_simulation', None))
            if sim is None:
                continue
            try:
                sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                if t_now >= len(sa) or float(sa[t_now]) != 1.0:
                    continue
                ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                if not np.isfinite(rs):
                    rs = 1.0
                ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                if ev_obj is None:
                    continue
                bt = getattr(ev_obj, 'battery', None)
                if bt is None:
                    continue
                soc_arr = getattr(bt, 'soc', None)
                if soc_arr is None:
                    continue
                sn = np.asarray(soc_arr, dtype=float)
                current_soc = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0

                # ONLY surplus EVs get V2G context rewards
                if current_soc < rs + self._ev_clamp_margin:
                    continue

                act = float(action_np[gidx])

                # V2G during grid import: discharge helps reduce grid stress
                if total_net > 0 and act < 0:
                    grid_need = min(total_net / max(1e-6, self.P_grid_max), 1.0)
                    r_v2g += abs(act) * grid_need

                # Charging during solar: use clean cheap energy
                if solar > 0 and act > 0:
                    solar_frac = min(solar / (solar + imp + 1e-6), 1.0)
                    r_v2g += abs(act) * solar_frac

                # Penalize discharge during solar (should charge instead)
                if solar > 0 and act < 0:
                    solar_frac = min(solar / (solar + imp + 1e-6), 1.0)
                    r_v2g -= 0.5 * abs(act) * solar_frac
            except Exception:
                pass
        return self.alpha_v2g_context * r_v2g

    def _ev_solar_reward(self, action_np: np.ndarray, solar: float) -> float:
        """R23: Reward EV charging during solar abundance.

        r = alpha * mean_over_EVs(max(0, action_i) * solar_norm)

        Only rewards positive actions (charging), scaled by normalized solar.
        Fires for ALL connected EVs, not just surplus.
        """
        if not self._ev_action_map or solar <= 0:
            return 0.0

        solar_norm = min(solar / self._solar_capacity, 1.0) if self._solar_capacity > 0 else 0.0

        total = 0.0
        count = 0
        for act_idx, bld_idx, _ in self._ev_action_map:
            if act_idx < len(action_np):
                charge = max(0.0, float(action_np[act_idx]))  # only reward charging
                total += charge * solar_norm
                count += 1

        if count == 0:
            return 0.0

        return self.alpha_ev_solar * (total / count)

    def _solar_store_reward(self, action_np: np.ndarray) -> float:
        """R24: Reward charging storage devices during REAL solar surplus.

        r = alpha * mean_over_surplus_devices(max(0, action_i) * surplus_norm_i)

        Key design choices (each addressing a specific bug or failure mode):
        1. Requires actual solar generation > 0 at the building (prevents fake
           surplus from battery discharge at night — Bug 1 fix)
        2. Surplus capped at solar generation (cannot exceed what sun provides)
        3. Only counts devices at surplus buildings in the mean (prevents
           dilution from non-surplus buildings — Bug 3 fix)
        4. EV chargers require connected EV (prevents phantom reward for
           actions CityLearn ignores — Bug 2 fix)
        5. Only rewards charging (action > 0), NEVER discharge (Rule 5)
        6. Uses POST-action NEC: self-correcting if charge overshoots surplus
        """
        if self.alpha_solar_store <= 0:
            return 0.0
        city = self._get_citylearn()
        if city is None:
            return 0.0

        t_now = int(getattr(city, 'time_step', 0))
        t_idx = max(0, t_now - 1)
        buildings = list(getattr(city, 'buildings', []))
        total = 0.0
        count = 0

        # Battery actions
        for act_idx, bld_idx, cap, p_max, eta in self._batt_action_map:
            if act_idx >= len(action_np) or bld_idx >= len(buildings):
                continue
            b = buildings[bld_idx]
            try:
                # Gate 1: building must have actual solar generation
                sg = getattr(b, 'solar_generation', None)
                if sg is None or not hasattr(sg, '__len__') or len(sg) <= t_idx:
                    continue
                solar_gen = abs(float(sg[t_idx]))
                if solar_gen <= 0:
                    continue  # no solar at this building right now

                nec = getattr(b, 'net_electricity_consumption', None)
                if nec is None or not hasattr(nec, '__len__') or len(nec) <= t_idx:
                    continue
                nec_val = float(nec[t_idx])

                # Surplus = negative NEC, but capped at actual solar generation
                # This prevents discharge-created fake surplus
                surplus = min(max(0.0, -nec_val), solar_gen)
                if surplus <= 0:
                    continue  # no real solar surplus

                surplus_norm = min(surplus / max(1e-6, self.P_building_max), 2.0)
                charge = max(0.0, float(action_np[act_idx]))
                total += charge * surplus_norm
                count += 1
            except Exception:
                pass  # don't inflate count on error

        # EV charger actions (only connected EVs, skip if battery-only mode)
        if self._solar_store_batt_only:
            if count == 0:
                return 0.0
            return self.alpha_solar_store * total / count
        for gidx, b_idx, ch_idx in self._ev_action_map:
            if gidx >= len(action_np) or b_idx >= len(buildings):
                continue
            b = buildings[b_idx]
            try:
                # Gate 1: building must have actual solar generation
                sg = getattr(b, 'solar_generation', None)
                if sg is None or not hasattr(sg, '__len__') or len(sg) <= t_idx:
                    continue
                solar_gen = abs(float(sg[t_idx]))
                if solar_gen <= 0:
                    continue

                # Gate 2: EV must be connected (prevents phantom reward)
                chargers = getattr(b, 'electric_vehicle_chargers', None) or []
                if ch_idx >= len(chargers):
                    continue
                ch = chargers[ch_idx]
                sim = getattr(ch, 'charger_simulation',
                              getattr(ch, '_Charger__charger_simulation', None))
                if sim is None:
                    continue
                sa = np.asarray(
                    getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                if t_now >= len(sa) or float(sa[t_now]) != 1.0:
                    continue  # EV not connected — action has no effect

                nec = getattr(b, 'net_electricity_consumption', None)
                if nec is None or not hasattr(nec, '__len__') or len(nec) <= t_idx:
                    continue
                nec_val = float(nec[t_idx])

                surplus = min(max(0.0, -nec_val), solar_gen)
                if surplus <= 0:
                    continue

                surplus_norm = min(surplus / max(1e-6, self.P_building_max), 2.0)
                charge = max(0.0, float(action_np[gidx]))
                total += charge * surplus_norm
                count += 1
            except Exception:
                pass  # don't inflate count on error

        if count == 0:
            return 0.0
        return self.alpha_solar_store * total / count

    def _ev_slack_arbitrage_reward(self, action_np: np.ndarray, price: float) -> float:
        """R25: Slack-gated EV price arbitrage — EV as mobile battery.

        When slack is HIGH (plenty of time to charge before departure):
          → Full price signal: charge cheap, V2G during expensive
        When slack is LOW (must charge soon or miss departure):
          → Gate → 0: price signal fades, only r_ev urgency drives charging

        For V2G discharge, requires BOTH slack AND surplus SoC (above required + margin).
        This prevents the R4 failure where EV arbitrage penalized evening charging.

        slack = hours_until_departure - hours_needed_to_full_charge
        hours_needed = deficit / (max_charge_power * efficiency / battery_capacity)
        gate = clip(slack / hours_until_departure, 0, 1)

        Charge reward: scale * gate * action * cheapness
        V2G reward:    scale * gate * surplus_gate * |action| * expensiveness
        """
        if self.ev_slack_arb_scale <= 0 or not self._ev_action_map:
            return 0.0
        city = self._get_citylearn()
        if city is None:
            return 0.0

        t_now = int(getattr(city, 'time_step', 0))
        t_idx = max(0, t_now - 1)
        buildings = list(getattr(city, 'buildings', []))

        # R27: Solar-aware price signal computed per-building inside the loop
        total = 0.0
        count = 0

        for gidx, b_idx, ch_idx in self._ev_action_map:
            if gidx >= len(action_np) or b_idx >= len(buildings):
                continue
            chargers = getattr(buildings[b_idx], 'electric_vehicle_chargers', None) or []
            if ch_idx >= len(chargers):
                continue
            ch = chargers[ch_idx]
            sim = getattr(ch, 'charger_simulation',
                          getattr(ch, '_Charger__charger_simulation', None))
            if sim is None:
                continue
            try:
                # Check EV connected
                sa = np.asarray(
                    getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                if t_now >= len(sa) or float(sa[t_now]) != 1.0:
                    continue  # not connected

                # Get departure time, required SoC, current SoC
                da = np.asarray(
                    getattr(sim, '_electric_vehicle_departure_time'), dtype=float)
                ra = np.asarray(
                    getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                dep_hours = float(da[t_now]) if t_now < len(da) else 1.0
                req_soc = float(ra[t_now]) if t_now < len(ra) else 1.0
                if not np.isfinite(dep_hours) or dep_hours <= 0:
                    dep_hours = 1.0
                if not np.isfinite(req_soc):
                    req_soc = 1.0

                # Get current SoC
                ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                if ev_obj is None:
                    continue
                bt = getattr(ev_obj, 'battery', None)
                if bt is None:
                    continue
                soc_arr = getattr(bt, 'soc', None)
                if soc_arr is None:
                    continue
                sn = np.asarray(soc_arr, dtype=float)
                current_soc = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0

                # Compute max charge rate in SoC/hour (charger-specific)
                mp = float(getattr(ch, 'max_charging_power', 0) or 0)
                if isinstance(mp, np.ndarray):
                    mp = float(mp.ravel()[0])
                ec = float(getattr(bt, 'capacity', 0) or 0)
                eff = float(getattr(ch, 'efficiency', 0.95) or 0.95)
                if ec <= 0 or mp <= 0:
                    continue
                soc_per_hour = (mp * eff) / ec  # charger-specific charge rate

                # Compute slack
                deficit = max(0.0, req_soc - current_soc)
                hours_needed = deficit / max(soc_per_hour, 1e-9)
                slack = max(0.0, dep_hours - hours_needed)
                gate = min(1.0, slack / max(dep_hours, 1e-9))

                # R27: Per-building solar-aware effective cost (same as battery arb)
                solar_credit = 0.0
                try:
                    b = buildings[b_idx]
                    sg = getattr(b, 'solar_generation', None)
                    nsl = getattr(b, '_Building__energy_to_non_shiftable_load', None)
                    if sg is not None and nsl is not None:
                        sg_val = abs(float(sg[t_idx])) if hasattr(sg, '__len__') and len(sg) > t_idx else 0.0
                        nsl_val = float(nsl[t_idx]) if hasattr(nsl, '__len__') and len(nsl) > t_idx else 0.0
                        solar_surplus = max(0.0, sg_val - nsl_val)
                        solar_credit = solar_surplus / max(1e-6, self.P_building_max)
                except Exception:
                    pass
                eff_cost = price / max(1e-8, self.mean_price) - 1.0 - solar_credit
                price_signal = max(-1.0, min(1.0, eff_cost))  # CLIP [-1, 1]

                act = float(action_np[gidx])
                hour = t_now % 24

                if act >= 0:
                    # CHARGING: reward when effective cost is negative (cheap/solar)
                    total += gate * act * max(0.0, -price_signal)
                    count += 1
                else:
                    # C0 fix: no V2G arb reward for deficit EVs
                    # Charging branch (line 904) still active for price-timing
                    if current_soc < req_soc:
                        count += 1
                        continue
                    # R29: V2G DISCHARGE — time-gated to peak hours (17-23) ONLY.
                    # Root cause: during solar hours price_signal is weakly positive,
                    # so V2G gives small positive reward with zero urgency penalty.
                    # Agent V2Gs "for free" at solar, depleting SoC for peak hours.
                    # Fix: zero V2G arb reward outside peak → agent charges during solar.
                    if hour < 17:
                        # Non-peak: no V2G arb reward (agent should charge or idle)
                        count += 1
                        continue

                    # Peak hours (17-23): time-slack gate
                    soc_after_v2g = current_soc - soc_per_hour
                    new_deficit = max(0.0, req_soc - soc_after_v2g)
                    hours_to_recharge = new_deficit / max(soc_per_hour, 1e-9)
                    v2g_slack = dep_hours * 0.7 - hours_to_recharge

                    if dep_hours < 2.0:
                        v2g_gate = 0.0  # hard block: no V2G in last 2 hours
                    else:
                        v2g_gate = float(np.clip(v2g_slack / 2.0, 0.0, 1.0))

                    # RAW price_signal: rewards peak V2G (price_signal > 0 at peak)
                    total += gate * v2g_gate * abs(act) * price_signal
                    count += 1
            except Exception:
                pass

        if count == 0:
            return 0.0
        return self.ev_slack_arb_scale * total / count

    def _ev_smart_reward(self, action_np: np.ndarray, price: float) -> float:
        """R27a: Headroom-gated EV economic reward (departure-aware).

        Computes a feasibility corridor for each connected EV:
          soc_min_feasible = max(0, required_soc - hours_left * charge_rate)
          headroom = current_soc - soc_min_feasible

        When headroom > 0 (agent has slack time):
          → Price signal ACTIVE: charge during cheap/solar, V2G during expensive
        When headroom ≤ 0 (must charge NOW to meet departure):
          → Price signal OFF: no conflict with C0 Lagrangian

        This replaces r_trajectory which was departure-blind and conflicted
        with C0 by penalizing mandatory charging during high-price hours.
        """
        if self.alpha_ev_smart <= 0 or not self._ev_action_map:
            return 0.0
        city = self._get_citylearn()
        if city is None:
            return 0.0

        t_now = int(getattr(city, 'time_step', 0))
        t_idx = max(0, t_now - 1)
        buildings = list(getattr(city, 'buildings', []))

        total = 0.0
        count = 0

        for gidx, b_idx, ch_idx in self._ev_action_map:
            if gidx >= len(action_np) or b_idx >= len(buildings):
                continue
            chargers = getattr(buildings[b_idx], 'electric_vehicle_chargers', None) or []
            if ch_idx >= len(chargers):
                continue
            ch = chargers[ch_idx]
            sim = getattr(ch, 'charger_simulation',
                          getattr(ch, '_Charger__charger_simulation', None))
            if sim is None:
                continue
            try:
                # Check EV connected
                sa = np.asarray(
                    getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                if t_now >= len(sa) or float(sa[t_now]) != 1.0:
                    continue

                # Departure time & required SoC
                da = np.asarray(
                    getattr(sim, '_electric_vehicle_departure_time'), dtype=float)
                ra = np.asarray(
                    getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                dep_hours = float(da[t_now]) if t_now < len(da) else 1.0
                req_soc = float(ra[t_now]) if t_now < len(ra) else 1.0
                if not np.isfinite(dep_hours) or dep_hours <= 0:
                    dep_hours = 1.0
                if not np.isfinite(req_soc):
                    req_soc = 1.0

                # Current SoC
                ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                if ev_obj is None:
                    continue
                bt = getattr(ev_obj, 'battery', None)
                if bt is None:
                    continue
                soc_arr = getattr(bt, 'soc', None)
                if soc_arr is None:
                    continue
                sn = np.asarray(soc_arr, dtype=float)
                current_soc = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0

                # Charger properties
                mp = getattr(ch, 'max_charging_power', 0)
                if isinstance(mp, np.ndarray):
                    mp = float(mp.ravel()[0])
                else:
                    mp = float(mp or 0)
                ec = float(getattr(bt, 'capacity', 0) or 0)
                eff = float(getattr(ch, 'efficiency', 0.95) or 0.95)
                if not (np.isfinite(ec) and ec > 0 and np.isfinite(mp) and mp > 0):
                    continue
                soc_per_hour = (mp * eff) / ec

                # ── Feasibility corridor ──
                # Minimum SoC needed NOW to still meet departure at max charge rate
                soc_min_feasible = max(0.0, req_soc - dep_hours * soc_per_hour)
                headroom = current_soc - soc_min_feasible

                # Gate: 0 when urgent (headroom ≤ 0), opens smoothly as headroom grows
                gate = float(np.clip(headroom / max(req_soc, 0.01), 0.0, 1.0))

                # Price signal: positive = expensive now, negative = cheap now
                price_signal = price / max(1e-8, self.mean_price) - 1.0
                price_signal = max(-1.0, min(1.0, price_signal))

                act = float(action_np[gidx])

                if act >= 0:
                    # CHARGING: reward when gate open AND price is cheap (signal < 0)
                    total += gate * act * max(0.0, -price_signal)
                    count += 1
                else:
                    # V2G DISCHARGE: only when surplus SoC above required
                    if current_soc < req_soc + self._ev_clamp_margin:
                        count += 1
                        continue
                    # Reward V2G when gate open AND price is expensive (signal > 0)
                    total += gate * abs(act) * max(0.0, price_signal)
                    count += 1
            except Exception:
                pass

        if count == 0:
            return 0.0
        result = self.alpha_ev_smart * total / count
        return result if np.isfinite(result) else 0.0

    def _headroom_penalty(self, action_np: np.ndarray) -> float:
        """R26: Direction-aware headroom penalty for charging actions.

        C3 uses abs(NEC), so BOTH excess import AND excess export violate.
        The penalty must understand the DIRECTION:

        IMPORT violation (NEC > +P_bmax): charging WORSENS it → PENALIZE
        EXPORT violation (NEC < -P_bmax): charging HELPS it → DO NOT penalize
        Within safe zone (|NEC| < P_bmax):  no violation → no penalty

        For discharge (action < 0):
        IMPORT violation: discharge helps → no penalty
        EXPORT violation: discharge worsens → PENALIZE
        Within safe zone: no penalty

        This correctly handles the solar absorption case:
        NEC = -11 kW (export violation), agent charges 5 kW → NEC = -6 → BETTER
        → no penalty (charging reduces export violation)

        NEC = +2 kW (within limit), agent charges 5 kW → NEC = +7 → VIOLATION
        → penalty (charging caused import violation)
        """
        if self.alpha_headroom <= 0:
            return 0.0
        city = self._get_citylearn()
        if city is None:
            return 0.0

        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)
        buildings = list(getattr(city, 'buildings', []))
        total_penalty = 0.0
        count = 0

        def _compute_penalty(nec_val, act):
            """Compute directional headroom penalty for one device."""
            abs_nec = abs(nec_val)
            overshoot = max(0.0, abs_nec - self.P_building_max)
            if overshoot <= 0:
                return 0.0  # within safe zone

            if nec_val > 0:
                # IMPORT violation: charge worsens, discharge helps
                if act > 0:
                    return -(overshoot / self.P_building_max) ** 2
                # discharge → no penalty (it helps)
                return 0.0
            else:
                # EXPORT violation: charge helps, discharge worsens
                if act < 0:
                    return -(overshoot / self.P_building_max) ** 2
                # charge → no penalty (it helps by absorbing solar)
                return 0.0

        # Battery actions
        for act_idx, bld_idx, cap, p_max, eta in self._batt_action_map:
            if act_idx >= len(action_np) or bld_idx >= len(buildings):
                continue
            act = float(action_np[act_idx])
            if act == 0:
                count += 1
                continue
            b = buildings[bld_idx]
            try:
                nec = getattr(b, 'net_electricity_consumption', None)
                if nec is None or not hasattr(nec, '__len__') or len(nec) <= t_idx:
                    count += 1
                    continue
                nec_val = float(nec[t_idx])
                total_penalty += _compute_penalty(nec_val, act)
                count += 1
            except Exception:
                count += 1

        # EV charger actions (only connected EVs)
        t_now = int(getattr(city, 'time_step', 0))
        for gidx, b_idx, ch_idx in self._ev_action_map:
            if gidx >= len(action_np) or b_idx >= len(buildings):
                continue
            act = float(action_np[gidx])
            if act == 0:
                count += 1
                continue
            b = buildings[b_idx]
            try:
                chargers = getattr(b, 'electric_vehicle_chargers', None) or []
                if ch_idx >= len(chargers):
                    count += 1
                    continue
                ch = chargers[ch_idx]
                sim = getattr(ch, 'charger_simulation',
                              getattr(ch, '_Charger__charger_simulation', None))
                if sim is None:
                    count += 1
                    continue
                sa = np.asarray(
                    getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                if t_now >= len(sa) or float(sa[t_now]) != 1.0:
                    count += 1
                    continue  # not connected

                nec = getattr(b, 'net_electricity_consumption', None)
                if nec is None or not hasattr(nec, '__len__') or len(nec) <= t_idx:
                    count += 1
                    continue
                nec_val = float(nec[t_idx])
                total_penalty += _compute_penalty(nec_val, act)
                count += 1
            except Exception:
                count += 1

        if count == 0:
            return 0.0
        return self.alpha_headroom * total_penalty / count

    def _peak_shave_reward(self, action_np: np.ndarray) -> float:
        """Reward battery discharge during building power peaks (C3 correlation).

        Explicitly connects battery actions (C2) with building power (C3):
        - Building near peak + battery discharging → positive (helping reduce C3)
        - Building near peak + battery charging → negative (worsening C3)
        - Building off-peak + battery charging → small positive (storing for later)
        """
        if self.alpha_peak_shave <= 0 or not self._batt_action_map:
            return 0.0
        city = self._get_citylearn()
        if city is None:
            return 0.0

        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)
        buildings = list(getattr(city, 'buildings', []))
        r_ps = 0.0

        for act_idx, bld_idx, cap, p_max, eta in self._batt_action_map:
            if act_idx >= len(action_np) or bld_idx >= len(buildings):
                continue
            b = buildings[bld_idx]
            try:
                nec = getattr(b, 'net_electricity_consumption', None)
                if nec is None or not hasattr(nec, '__len__') or len(nec) <= t_idx:
                    continue
                net_power = float(nec[t_idx])
                ratio = net_power / max(1e-6, self.P_building_max)
                act = float(action_np[act_idx])

                if ratio > 0.7:  # Building approaching/exceeding C3 threshold
                    urgency = min(1.0, (ratio - 0.7) / 0.3)  # 0→1 over 70-100%
                    if act < 0:  # Discharging — reducing peak
                        r_ps += urgency * abs(act)
                    elif act > 0:  # Charging — worsening peak
                        r_ps -= 0.5 * urgency * act
                elif ratio < 0.3 and act > 0:  # Off-peak charging — storing for later
                    r_ps += 0.2 * act
            except Exception:
                pass

        n_batt = len(self._batt_action_map)
        return self.alpha_peak_shave * r_ps / max(1, n_batt)

    def _grid_penalty(self, total_net: float) -> float:
        """Quadratic grid penalty. Penalizes high grid IMPORT only (not export).

        Changed from abs(total_net) to max(0, total_net) because abs() penalizes
        solar export and V2G discharge, creating a severe gradient conflict with
        r_ren (cosine = -0.82 in diagnostic). Import-only preserves the grid peak
        penalty without fighting renewable self-consumption.
        """
        if self.alpha_grid_penalty <= 0:
            return 0.0
        ratio = max(0.0, total_net) / max(1e-6, self.P_grid_max)
        return -self.alpha_grid_penalty * ratio * ratio

    def _populate_aux_targets(self, info: dict) -> None:
        """Populate ground-truth next-hour targets for auxiliary prediction loss (R26k).

        Targets (all normalized):
          [0] next-hour price / mean_price
          [1] next-hour district solar / solar_capacity
          [2] next-hour district net_load / P_grid_max
        """
        city = self._get_citylearn()
        if city is None:
            return
        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)
        t_next = t_idx + 1
        buildings = list(getattr(city, 'buildings', []))

        # Target 0: next-hour price (normalized by annual mean)
        price_target = 0.0
        if self._pricing_arr is not None and t_next < len(self._pricing_arr):
            price_target = float(self._pricing_arr[t_next]) / max(1e-6, self.mean_price)

        # Target 1: next-hour district solar (normalized by total PV capacity)
        solar_target = 0.0
        for b in buildings:
            try:
                sg = getattr(b, 'solar_generation', None)
                if sg is not None and hasattr(sg, '__len__') and len(sg) > t_next:
                    solar_target += abs(float(sg[t_next]))
            except Exception:
                pass
        solar_target /= max(1e-6, self._solar_capacity)

        # Target 2: next-hour district net load (normalized by P_grid_max)
        net_load_target = 0.0
        for b in buildings:
            try:
                nsl = getattr(b, '_Building__energy_to_non_shiftable_load', None)
                if nsl is not None and hasattr(nsl, '__len__') and len(nsl) > t_next:
                    net_load_target += float(nsl[t_next])
            except Exception:
                pass
        net_load_target /= max(1e-6, self.P_grid_max)

        info['aux_targets'] = [price_target, solar_target, net_load_target]

    def _trajectory_reward(self, info: dict, action_np: np.ndarray) -> float:
        """Forecast-aware price arbitrage reward (R26h+).

        Compares spot price to the mean of the next N hours (observable via
        the forecast already in the agent's observation).  This creates direct
        gradient pressure on the temporal transformer's forecast attention
        heads: the reward is only predictable if the policy attends to future
        price tokens.

        Unlike r_load_shift / r_price_arb (which compare to a STATIC annual
        mean), this compares to a DYNAMIC future mean that changes every hour.
        Two hours with the same spot price get different rewards if their
        24h-ahead outlooks differ.

        Fixes vs original WACE design:
          #1 No hidden state — uses only observable quantities
          #2 Forward-looking — rewards forecast use, not past-action history
          #3 Proper magnitude — scaled to ~0.3–1.0/step (comparable to r_sg)
          #4 No bootstrap — pricing array available from step 0
          #5 Distinct signal — dynamic future mean ≠ static annual mean
          #6 EV surplus gate — skips mandatory charging (no C0 conflict)
          #7 No dead code
        """
        city = self._get_citylearn()
        if city is None:
            return 0.0
        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)
        buildings = list(getattr(city, 'buildings', []))

        # ── Spot price & future mean ──
        pricing = self._pricing_arr
        if pricing is None or t_idx >= len(pricing):
            return 0.0
        price = float(pricing[t_idx])

        # Future window: next N hours (clamped to episode end, no wrap)
        h = self._traj_forecast_hours
        future_start = t_idx + 1
        future_end = min(future_start + h, len(pricing))
        if future_start >= len(pricing) or future_end - future_start < 6:
            return 0.0  # too few future prices for a meaningful signal
        future_slice = pricing[future_start:future_end]
        future_mean = float(np.mean(future_slice))

        # Forecast signal: positive = current price ABOVE future mean (sell now)
        #                  negative = current price BELOW future mean (buy now)
        mean_ref = max(1e-6, self.mean_price)
        forecast_signal = (price - future_mean) / mean_ref
        forecast_signal = max(-2.0, min(2.0, forecast_signal))

        # ── A. Battery forecast arbitrage ──
        r_batt = 0.0
        n_batt = len(self._batt_action_map)
        for act_idx, bld_idx, cap, p_max, eta in self._batt_action_map:
            if act_idx >= len(action_np) or bld_idx >= len(buildings):
                continue
            b = buildings[bld_idx]
            act = float(action_np[act_idx])

            # SoC gate: forces SoC-dependent value learning.
            # Discharge reward scales with available energy (high SoC → more to sell).
            # Charge reward scales with available headroom (low SoC → more room to buy).
            soc = 0.5
            try:
                es = getattr(b, 'electrical_storage', None)
                if es is not None and hasattr(es, 'soc') and hasattr(es.soc, '__len__') and len(es.soc) > t_idx:
                    soc = float(np.clip(es.soc[t_idx], 0.01, 0.99))
            except Exception:
                pass
            if act < 0:  # discharge
                gate = max(0.2, soc)
            else:  # charge
                gate = max(0.2, 1.0 - soc)

            # -act * gate * signal:
            #   discharge (act<0) when signal>0 (price high vs future) → +reward
            #   charge    (act>0) when signal<0 (price low vs future)  → +reward
            r_batt += -act * gate * forecast_signal
        if n_batt > 0:
            r_batt /= n_batt

        # ── B. EV forecast arbitrage (surplus SoC only) ──
        r_ev = 0.0
        n_ev_active = 0
        t_now = int(getattr(city, 'time_step', 0))
        for gidx, b_idx, ch_idx in self._ev_action_map:
            if gidx >= len(action_np) or b_idx >= len(buildings):
                continue
            b = buildings[b_idx]
            chargers = getattr(b, 'electric_vehicle_chargers', None) or []
            if ch_idx >= len(chargers):
                continue
            ch = chargers[ch_idx]

            # Check connection
            sim = getattr(ch, 'charger_simulation',
                          getattr(ch, '_Charger__charger_simulation', None))
            if sim is None:
                continue
            try:
                sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                if t_now >= len(sa) or float(sa[t_now]) != 1.0:
                    continue  # Not connected — skip entirely
            except Exception:
                continue

            # Only reward EVs with surplus SoC (agent has a real choice).
            # When SoC < required + margin, the agent MUST charge for C0
            # compliance — penalizing that would fight the safety constraint.
            try:
                ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                if not np.isfinite(rs):
                    rs = 1.0
                ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                if ev_obj is None:
                    continue
                bt = getattr(ev_obj, 'battery', None)
                if bt is None:
                    continue
                soc_arr = getattr(bt, 'soc', None)
                if soc_arr is None:
                    continue
                sn = np.asarray(soc_arr, dtype=float)
                current_soc = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0
                if current_soc < rs + self._ev_clamp_margin:
                    continue  # Mandatory charging — no arbitrage reward/penalty
            except Exception:
                continue

            n_ev_active += 1
            act = float(action_np[gidx])
            r_ev += -act * forecast_signal

        if n_ev_active > 0:
            r_ev /= n_ev_active

        # ── C. Combine and log ──
        r_total = r_batt + self.traj_ev_w * r_ev

        info['r_traj_batt'] = float(r_batt)
        info['r_traj_ev'] = float(r_ev)
        info['r_traj_total'] = float(r_total)
        info['traj_forecast_mean'] = float(future_mean)
        info['traj_forecast_signal'] = float(forecast_signal)

        return self.alpha_trajectory * r_total

    def _stems_reward(self, info: dict, action_np: np.ndarray,
                      action_pre_ev_clamp: np.ndarray = None) -> float:
        """Custom STEMS with tuned weights + EV component."""
        city = self._get_citylearn()
        if city is None:
            return 0.0
        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)
        buildings = list(getattr(city, 'buildings', []))

        # Treat both wrapper-side and policy-side masking as active for reward shaping.
        mask_active = (
            os.environ.get("CITYLEARN_ACTION_MASK", "0") == "1"
            or os.environ.get("CITYLEARN_POLICY_ACTION_MASK", "0") == "1"
        )

        # Beta actor fix: convert x ∈ (0,1) to physical action for reward computation.
        # The agent outputs x, the mask transforms x → physical via affine mapping.
        # Reward functions need physical actions to correctly detect charge vs discharge.
        # This does NOT affect the gradient path (reward is a scalar, not differentiable).
        beta_mode = os.environ.get("CITYLEARN_BETA_ACTOR", "0") == "1"
        if beta_mode and mask_active:
            _city = self._get_citylearn()
            _smin = getattr(_city, '_action_mask_safe_min', None) if _city else None
            _smax = getattr(_city, '_action_mask_safe_max', None) if _city else None
            if _smin is not None and _smax is not None:
                action_np = _smin + action_np * (_smax - _smin)
                if action_pre_ev_clamp is not None:
                    action_pre_ev_clamp = _smin + action_pre_ev_clamp * (_smax - _smin)

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
        # Fix 4 (STEMS_SG_EXPORT_CREDIT>0): partial credit for net exports.
        # Reduces the quadratic asymmetry that punishes charging more than
        # V2G benefits, enabling grid-based price arbitrage.
        # imp_eff = max(0, net) - credit * max(0, -net)
        imp_eff = imp
        if self.sg_export_credit > 0 and exp > 0:
            imp_eff = max(0.0, imp - self.sg_export_credit * exp)
        # R17: threshold r_sg — only penalize imports above threshold
        if self.sg_threshold_frac > 0:
            sg_thresh = self.P_grid_max * self.sg_threshold_frac
            imp_above = max(0.0, imp_eff - sg_thresh)
        else:
            imp_above = imp_eff
        r_sg = self.alpha_grid * (1.0 - min((imp_above / max(1e-6, self.P_grid_max)) ** 2, 4.0))

        # Building stability
        # Fix 1 (STEMS_SB_ASYMMETRIC=1): only penalize imports, not exports.
        # This enables V2G: exporting power from a building is not penalized.
        bs, bc = 0.0, 0
        for b in buildings:
            try:
                nec = getattr(b, 'net_electricity_consumption', None)
                if nec is not None and hasattr(nec, '__len__') and len(nec) > t_idx:
                    nec_val = float(nec[t_idx])
                    if self.sb_asymmetric:
                        ratio = max(0.0, nec_val) / max(1e-6, self.P_building_max)
                    else:
                        ratio = abs(nec_val) / max(1e-6, self.P_building_max)
                    bs += 1.0 - min(ratio, 4.0); bc += 1
            except Exception:
                pass
        r_sb = self.alpha_build * (bs / max(1, bc)) if bc > 0 else 0.0

        # Ramp
        rd = abs(total_net - self._prev_net) if self._prev_net is not None else 0.0
        self._prev_net = total_net
        # R29: Disable r_ramp when action mask is active. The mask changes
        # executed actions based on state-dependent bounds, creating NEC ramps
        # the agent didn't intend. Penalizing these spurious ramps hurts learning.
        if mask_active:
            r_ramp = 0.0
        else:
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

        # EV shaping: dense urgency×shortfall reward (complementary to Sauté budget)
        # STEMS_LAMBDA_EV=0: disabled (R11b/R12a). >0: provides dense gradient for EV charging.
        r_ev = self._ev_reward(action_np) if self.lambda_ev > 0 else 0.0

        # R15a: Anti-discharge penalty (uses pre-clamp action for gradient signal)
        r_ev_guard = self._ev_guard_penalty(
            action_pre_ev_clamp if action_pre_ev_clamp is not None else action_np)

        # R15b: Context-aware V2G signal (smart discharge timing)
        r_v2g_ctx = self._ev_v2g_context_reward(action_np, total_net, imp, sg)

        # R15c: Peak-shaving reward (battery C2 ↔ building C3 correlation)
        r_peak_shave = self._peak_shave_reward(action_np)

        # R16: Battery-only price arbitrage (replaces r_eco for battery intelligence)
        r_load_shift = self._load_shift_reward(action_np, price)

        # R16: Gentle linear grid awareness (replaces quadratic r_sg)
        r_grid_mild = -self.alpha_grid_mild * (imp / max(1e-6, self.P_grid_max)) if self.alpha_grid_mild > 0 else 0.0

        # R23: Solar-aligned EV charging reward
        r_ev_solar = self._ev_solar_reward(action_np, solar=sg) if self.alpha_ev_solar > 0 else 0.0

        # R24: Solar storage — charge batteries+EVs during solar surplus
        r_solar_store = self._solar_store_reward(action_np) if self.alpha_solar_store > 0 else 0.0

        # R25: Slack-gated EV price arbitrage — EV as mobile battery
        r_ev_slack_arb = self._ev_slack_arbitrage_reward(action_np, price) if self.ev_slack_arb_scale > 0 else 0.0

        # R26: Headroom penalty — penalize charging that exceeds building power limit
        # R29: Redundant when action mask is active (mask prevents violations by construction)
        if mask_active:
            r_headroom = 0.0
        else:
            r_headroom = self._headroom_penalty(action_np) if self.alpha_headroom > 0 else 0.0

        # SoC barrier reward (smooth penalty near battery SoC boundaries)
        r_barrier = 0.0
        n_batt = len(self._batt_action_map)
        if self.alpha_barrier > 0 and n_batt > 0:
            for b_i in range(1, n_batt + 1):
                soc = float(info.get(f'battery_soc_b{b_i}', 0.5))
                # Safe zone [0.10, 0.85]: zero penalty
                # Transition [0.05, 0.10] and [0.85, 0.90]: linear
                # Outside [0.05] or [0.90]: max penalty = -1
                if soc < 0.05:
                    pen = -1.0
                elif soc < 0.10:
                    pen = -(0.10 - soc) / 0.05  # linear 0→-1
                elif soc > 0.90:
                    pen = -1.0
                elif soc > 0.85:
                    pen = -(soc - 0.85) / 0.05  # linear 0→-1
                else:
                    pen = 0.0
                r_barrier += pen
            r_barrier *= self.alpha_barrier / n_batt  # normalize by num buildings

        # R29: Simple quadratic grid penalty
        r_grid_penalty = self._grid_penalty(total_net) if self.alpha_grid_penalty > 0 else 0.0

        # Simple battery price arbitrage (AL-SAC style): R = -action * norm_price
        r_price_arb = self._simple_price_reward(action_np, price) if self.alpha_price_arb > 0 else 0.0

        # R30: NEC-sign reward — align storage with exogenous load direction
        r_nec_sign = self._nec_sign_reward(action_np) if self.alpha_nec_sign > 0 else 0.0

        # R26h: Forecast-aware price arbitrage
        r_trajectory = self._trajectory_reward(info, action_np) if self.alpha_trajectory > 0 else 0.0

        # R27a: Headroom-gated EV economic reward (departure-aware)
        r_ev_smart = self._ev_smart_reward(action_np, price) if self.alpha_ev_smart > 0 else 0.0

        # Log individual reward components for ablation analysis
        info['r_eco'] = float(r_eco)
        info['r_sg'] = float(r_sg)
        info['r_sb'] = float(r_sb)
        info['r_ramp'] = float(r_ramp)
        info['r_ren'] = float(r_ren)
        info['r_ev'] = float(r_ev)
        info['r_ev_guard'] = float(r_ev_guard)
        info['r_v2g_ctx'] = float(r_v2g_ctx)
        info['r_peak_shave'] = float(r_peak_shave)
        info['r_load_shift'] = float(r_load_shift)
        info['r_grid_mild'] = float(r_grid_mild)
        info['r_barrier'] = float(r_barrier)
        info['r_ev_solar'] = float(r_ev_solar)
        info['r_solar_store'] = float(r_solar_store)
        info['r_ev_slack_arb'] = float(r_ev_slack_arb)
        info['r_headroom'] = float(r_headroom)
        info['r_grid_penalty'] = float(r_grid_penalty)
        info['r_price_arb'] = float(r_price_arb)
        info['r_nec_sign'] = float(r_nec_sign)
        info['r_trajectory'] = float(r_trajectory)
        info['r_ev_smart'] = float(r_ev_smart)

        return float(r_eco + r_sg + r_sb + r_ramp + r_ren + r_ev + r_ev_guard
                     + r_v2g_ctx + r_peak_shave + r_load_shift + r_grid_mild + r_barrier
                     + r_ev_solar + r_solar_store + r_ev_slack_arb + r_headroom
                     + r_grid_penalty + r_price_arb + r_nec_sign + r_trajectory
                     + r_ev_smart)

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
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        self._last_obs = obs_t
        return obs_t, info

    def attach_execution_projector(self, projector: Any) -> None:
        """Attach the final execution-time shield used immediately before env.step."""
        self._execution_projector = projector

    def _apply_execution_shield(self, a: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Run the final hard shield on the exact flat action sent to the env."""
        if not self._execution_shield_enabled or self._execution_projector is None:
            return a, {}

        obs_t = self._last_obs
        if obs_t is None:
            obs_t = torch.zeros(
                self._observation_space.shape,
                dtype=torch.float32,
            )
        act_t = torch.as_tensor(a, dtype=torch.float32)

        try:
            safe_t, pinfo = self._execution_projector.project(obs_t, act_t)
            safe_np = safe_t.detach().cpu().numpy().ravel().astype(np.float32, copy=False)
            info = {
                "execution_projection_delta": float(pinfo.get("projection_delta", 0.0)),
                "execution_projection_feasible": 1.0 if bool(pinfo.get("feasible", False)) else 0.0,
                "execution_projection_solve_ms": float(pinfo.get("solve_time_ms", 0.0)),
            }
            if "hybrid_used_psf_c0" in pinfo:
                info["execution_hybrid_used_psf_c0"] = 1.0 if bool(pinfo.get("hybrid_used_psf_c0")) else 0.0
            if "hybrid_psf_infeasible" in pinfo:
                info["execution_hybrid_psf_infeasible"] = float(pinfo.get("hybrid_psf_infeasible", 0.0))
            for k, v in pinfo.items():
                if k.startswith("lex_") or k.startswith("lex_dbg_"):
                    info[k] = v
            return safe_np, info
        except Exception:
            return a, {"execution_projection_error": 1.0}

    _SOC_UPPER_DEFAULT = 0.94   # legacy default (validated MAE=0.009, true limit 0.95)
    _BATT_DT = 1.0              # hours per step

    def _discover_battery_actions(self, safety_env) -> list:
        """Discover (action_index, building_index, cap, p_max, eta) for each battery.

        Returns list of tuples: (act_idx, bld_idx, capacity, nominal_power, efficiency)
        """
        city = None
        cur = safety_env
        for _ in range(20):
            if cur is None:
                break
            if hasattr(cur, 'buildings') and hasattr(cur, 'action_names'):
                city = cur
                break
            cur = getattr(cur, 'env', getattr(cur, 'base', None))
        if city is None:
            return []

        buildings = list(city.buildings)
        names_raw = getattr(city, 'action_names', [])
        if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
            flat_names = names_raw[0]
        elif isinstance(names_raw, list):
            flat_names = []
            for sub in names_raw:
                flat_names.extend(sub) if isinstance(sub, list) else flat_names.append(sub)
        else:
            return []

        # Map each battery action to its building
        batt_key = 'electrical_storage'
        batt_pos = [i for i, n in enumerate(flat_names) if str(n).lower() == batt_key]

        result = []
        for b_idx, b in enumerate(buildings):
            es = getattr(b, 'electrical_storage', None)
            if es is None:
                continue
            # Find this building's battery action index
            # Battery actions appear in building order, one per building
            if b_idx < len(batt_pos):
                act_idx = batt_pos[b_idx]
            else:
                continue
            cap = float(getattr(es, 'capacity', 6.4) or 6.4)
            p_max = float(getattr(es, 'nominal_power', 5.0) or 5.0)
            eta = float(getattr(es, 'efficiency', 0.9) or 0.9)
            result.append((act_idx, b_idx, cap, p_max, eta))

        return result

    def _discover_ev_charger_actions(self, safety_env) -> list:
        """Discover (global_action_idx, building_idx, charger_local_idx) for each EV charger."""
        city = None
        cur = safety_env
        for _ in range(20):
            if cur is None:
                break
            if hasattr(cur, 'buildings') and hasattr(cur, 'action_names'):
                city = cur
                break
            cur = getattr(cur, 'env', getattr(cur, 'base', None))
        if city is None:
            return []

        buildings = list(city.buildings)
        names_raw = getattr(city, 'action_names', [])
        if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
            flat_names = names_raw[0]
        elif isinstance(names_raw, list):
            flat_names = []
            for sub in names_raw:
                flat_names.extend(sub) if isinstance(sub, list) else flat_names.append(sub)
        else:
            return []

        ev_key = "electric_vehicle_storage_charger_"
        batt_key = "electrical_storage"
        batt_pos = [i for i, n in enumerate(flat_names) if str(n).lower() == batt_key]
        if len(batt_pos) != len(buildings):
            print(f"[CMDPv2] WARNING: battery count ({len(batt_pos)}) != building count "
                  f"({len(buildings)}). EV charger discovery disabled.")
            return []

        result = []
        for b_idx in range(len(buildings)):
            start = batt_pos[b_idx]
            end = batt_pos[b_idx + 1] if b_idx + 1 < len(buildings) else len(flat_names)
            chargers = getattr(buildings[b_idx], 'electric_vehicle_chargers', None) or []
            li = 0
            for i, n in enumerate(flat_names[start:end]):
                if ev_key not in str(n).lower():
                    continue
                if li < len(chargers):
                    result.append((start + i, b_idx, li))
                li += 1
        return result

    def _discover_wm_actions(self, safety_env) -> list:
        """Discover action indices for washing machines."""
        city = None
        cur = safety_env
        for _ in range(20):
            if cur is None:
                break
            if hasattr(cur, 'buildings') and hasattr(cur, 'action_names'):
                city = cur
                break
            cur = getattr(cur, 'env', getattr(cur, 'base', None))
        if city is None:
            return []

        names_raw = getattr(city, 'action_names', [])
        if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
            flat_names = names_raw[0]
        elif isinstance(names_raw, list):
            flat_names = []
            for sub in names_raw:
                flat_names.extend(sub) if isinstance(sub, list) else flat_names.append(sub)
        else:
            return []

        return [i for i, n in enumerate(flat_names) if "washing_machine" in str(n).lower()]

    def _clamp_battery_actions(self, a: np.ndarray) -> np.ndarray:
        """Clamp battery actions to prevent SoC violations."""
        if not self._batt_action_map:
            return a
        city = self._get_citylearn()
        if city is None:
            return a
        buildings = list(getattr(city, 'buildings', []))
        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)

        a_clamped = a.copy()
        for act_idx, bld_idx, cap, p_max, eta in self._batt_action_map:
            if act_idx >= len(a_clamped) or bld_idx >= len(buildings):
                continue
            es = getattr(buildings[bld_idx], 'electrical_storage', None)
            if es is None:
                continue
            soc_arr = getattr(es, 'soc', None)
            if soc_arr is None or not hasattr(soc_arr, '__len__') or len(soc_arr) <= t_idx:
                continue
            soc = float(np.clip(soc_arr[t_idx], 0.0, 1.0))

            # Max safe charge: positive action = charge
            denom_charge = p_max * self._BATT_DT * eta
            max_charge = (self._SOC_UPPER - soc) * cap / denom_charge if denom_charge > 0 else 1.0

            # Max safe discharge: negative action = discharge
            denom_discharge = p_max * self._BATT_DT
            max_discharge = soc * cap * eta / denom_discharge if denom_discharge > 0 else 1.0

            a_clamped[act_idx] = float(np.clip(a_clamped[act_idx], -max_discharge, max_charge))

        return a_clamped

    def _clamp_ev_actions(self, a: np.ndarray) -> np.ndarray:
        """Prevent EV discharge when SoC < required_soc + margin.

        When SoC >= required + margin: full [-1, 1] range (V2G of surplus OK).
        When SoC < required + margin: clamp to [0, 1] (charge only).
        """
        if not self._ev_action_map:
            return a
        city = self._get_citylearn()
        if city is None:
            return a

        t_now = int(getattr(city, 'time_step', 0))
        t_idx = max(0, t_now - 1)
        buildings = list(getattr(city, 'buildings', []))
        a_clamped = a.copy()

        for gidx, b_idx, ch_idx in self._ev_action_map:
            if gidx >= len(a_clamped) or b_idx >= len(buildings):
                continue
            chargers = getattr(buildings[b_idx], 'electric_vehicle_chargers', None) or []
            if ch_idx >= len(chargers):
                continue
            ch = chargers[ch_idx]
            sim = getattr(ch, 'charger_simulation',
                          getattr(ch, '_Charger__charger_simulation', None))
            if sim is None:
                continue
            try:
                sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                if t_now >= len(sa) or float(sa[t_now]) != 1.0:
                    continue  # No EV connected
                ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                if not np.isfinite(rs):
                    rs = 1.0
                ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                if ev_obj is None:
                    continue
                bt = getattr(ev_obj, 'battery', None)
                if bt is None:
                    continue
                soc_arr = getattr(bt, 'soc', None)
                if soc_arr is None:
                    continue
                sn = np.asarray(soc_arr, dtype=float)
                current_soc = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0
                if current_soc < rs + self._ev_clamp_margin:
                    if a_clamped[gidx] < 0:
                        self._ev_clamp_count += 1
                    a_clamped[gidx] = float(np.clip(a_clamped[gidx], 0.0, 1.0))
            except Exception:
                pass
        return a_clamped

    def _mask_disconnected_ev_actions(self, a: np.ndarray) -> np.ndarray:
        """Force EV actions to zero when no EV is connected.

        This is a narrow ablation switch: it removes impossible EV commands
        without changing the rest of the control stack.
        """
        if not self._ev_action_map:
            return a
        city = self._get_citylearn()
        if city is None:
            return a

        t_now = int(getattr(city, 'time_step', 0))
        buildings = list(getattr(city, 'buildings', []))
        a_masked = a.copy()

        for gidx, b_idx, ch_idx in self._ev_action_map:
            if gidx >= len(a_masked) or b_idx >= len(buildings):
                continue
            chargers = getattr(buildings[b_idx], 'electric_vehicle_chargers', None) or []
            if ch_idx >= len(chargers):
                continue
            ch = chargers[ch_idx]
            sim = getattr(ch, 'charger_simulation',
                          getattr(ch, '_Charger__charger_simulation', None))
            if sim is None:
                continue
            try:
                sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                connected = t_now < len(sa) and float(sa[t_now]) == 1.0
                if not connected and abs(float(a_masked[gidx])) > 1e-9:
                    self._ev_disconnect_mask_count += 1
                    a_masked[gidx] = 0.0
            except Exception:
                pass
        return a_masked

    def step(self, action):
        """Returns 6 values: (obs, reward, cost, terminated, truncated, info)"""
        if isinstance(action, torch.Tensor):
            a = action.detach().cpu().numpy().ravel()
        else:
            a = np.asarray(action, dtype=np.float32).ravel()
        shield_info = {}

        # R18: Disable washing machine (clamp action to 0)
        if self._wm_disable:
            for wm_idx in self._wm_action_indices:
                if wm_idx < len(a):
                    a[wm_idx] = 0.0

        # Safety clamp: prevent battery SoC violations (optional in R18)
        if self._batt_clamp_enabled:
            a = self._clamp_battery_actions(a)

        # Optional EV disconnect gate: remove impossible EV commands early.
        if self._ev_disconnect_mask_enabled:
            a = self._mask_disconnected_ev_actions(a)

        # R15a: EV action clamp (prevent discharge when under-charged)
        a_pre_ev_clamp = a.copy() if self._ev_clamp_enabled else a
        if self._ev_clamp_enabled:
            a = self._clamp_ev_actions(a)

        # Final execution-time shield: repair the exact action vector that will
        # be executed by the base env. Pre-shield clamps above define the
        # executable coordinates; after this point, do not mutate `a` again or
        # the certified action and executed action diverge.
        a, shield_info = self._apply_execution_shield(a)

        # R18: Track V2G discharge attempts (proves agent is exploring V2G)
        for gidx, b_idx, ch_idx in self._ev_action_map:
            if gidx < len(a) and float(a[gidx]) < -0.1:
                self._v2g_discharge_count += 1

        obs, _reward_base, terminated, truncated, info = self._env.step(a)
        self._step_count += 1

        reward = self._stems_reward(info, a, a_pre_ev_clamp)

        if self._ev_disconnect_mask_enabled:
            info["ev_disconnect_mask_count"] = float(self._ev_disconnect_mask_count)
        info.update(shield_info)

        # ---- Auxiliary temporal targets (R26k) ----
        # Ground truth next-hour: price, solar, net_load for transformer supervision
        if self._aux_targets_enabled:
            self._populate_aux_targets(info)

        # Markgraf et al. 2025 Eq. 23: SE-RL augmented reward
        mask_penalty = float(info.get("mask_penalty", 0.0))
        reward = reward - mask_penalty

        # Sauté MDP reward reshaping (per Sootla et al., ICML 2022):
        # When safety budget is exhausted, penalize the agent.
        # shaped_alpha > 0: smooth gradient (reward - alpha * |deficit|)
        #   Keeps ALL reward signals alive — agent can still learn from
        #   r_eco, r_sb, r_ev_guard, r_v2g_ctx even after budget depletion.
        # shaped_alpha = 0: legacy binary penalty (reward = -5.0)
        #   Kills all gradient for 93%+ of episode when budget is too small.
        if info.get("ev_saute_unsafe", 0.0) > 0.5:
            if self._ev_saute_shaped_alpha > 0:
                ev_deficit = min(abs(float(info.get("ev_saute_budget", 0.0))), 2.0)
                reward = reward - self._ev_saute_shaped_alpha * ev_deficit
            else:
                reward = -float(info.get("ev_saute_penalty", 5.0)) - mask_penalty

        # Sauté C4 (grid power): same mechanism, separate budget
        if info.get("saute_c4_unsafe", 0.0) > 0.5:
            if self._saute_c4_shaped_alpha > 0:
                c4_deficit = min(abs(float(info.get("saute_c4_budget", 0.0))), 2.0)
                reward = reward - self._saute_c4_shaped_alpha * c4_deficit
            else:
                reward = -float(info.get("saute_c4_penalty", 5.0))

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
                  f"C4={info.get('cost_stems_grid_power',0):.2f} "
                  f"ev_clamp={self._ev_clamp_count} "
                  f"v2g_discharge={self._v2g_discharge_count}")

        self._last_obs = obs_t
        return obs_t, reward_t, cost_t, terminated_t, truncated_t, info

    def close(self) -> None:
        if hasattr(self._env, 'close'):
            self._env.close()

    def render(self):
        return None

    def set_seed(self, seed: int) -> None:
        pass
