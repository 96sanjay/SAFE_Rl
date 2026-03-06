
"""
V3: CityLearnSafetyEnv with Action-Based EV Deficit Calculation (UPDATED, SAFE VERSION)

Key fixes included in this version:
✅ Action clipping to env.action_space.low/high (per-dimension; action_2 is [0,1])
✅ NO double-step bug (self.base.step called exactly once)
✅ EV actions stored aligned with extractor tau = (pre-step time_step + 1)
✅ EV logging stable: action_ev_0..7 always present even if fewer chargers exist
✅ Reset logging no longer crashes if < 8 EV chargers
✅ Keeps your bill-based reward + KPI logging logic intact

Env vars:
- CITYLEARN_EV_MISSING_ACTION_MODE: "assume_full" | "assume_zero" | "error"
- CITYLEARN_EXPORT_FACTOR: float (default 1.0)
- CITYLEARN_REWARD_SCALE: float (default 1.0)
- CITYLEARN_EV_COST_SCALE: float (default 1.0)
- CITYLEARN_KPI_FLUSH_EVERY_STEP: 1 to flush every step (debug)
- CITYLEARN_KPI_RUN_NAME: override KPI filename prefix (default CityLearnSafety_kpis_v3)
- CITYLEARN_DEBUG_ACTION_CLIP: 1 to print clip diagnostics every N steps
- CITYLEARN_DEBUG_ACTION_CLIP_EVERY: integer (default 500)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import gymnasium as gym

from .schema_index import build_index
from .kpi_logger import log_kpis, log_episode_end, init_kpi_logger
from citylearn_safe.extractors_v3 import ev_departure_cost_components_v3



# === AUTO-GENERATED ActionMap (module-level) ===
class ActionMap:
    __slots__ = ['battery_gidx', 'charger_id_to_gidx', 'ev_gidx', 'flat_names', 'global_to_meta', 'name_to_global']
    def __init__(self, battery_gidx, charger_id_to_gidx, ev_gidx, flat_names, global_to_meta, name_to_global):
        self.battery_gidx = battery_gidx
        self.charger_id_to_gidx = charger_id_to_gidx
        self.ev_gidx = ev_gidx
        self.flat_names = flat_names
        self.global_to_meta = global_to_meta
        self.name_to_global = name_to_global

class CityLearnSafetyEnvV3(gym.Env):

    # Regex for EV charger action names -> suffix (e.g., '15_2')
    _EV_RE = re.compile(r"^electric_vehicle_storage_charger_(?P<suffix>.+)$", re.IGNORECASE)
    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: Any,
        *,
        soc_min: float = 0.0,
        soc_max: float = 0.95,
        cost_mode: str = "hinge",
        include_ev_in_cost: bool = True,
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
        self.include_ev_in_cost = (
            bool(include_ev_in_cost)
            if include_ev_in_cost is not None
            else bool(int(os.environ.get("CITYLEARN_INCLUDE_EV_COST", "1")))
        )

        # 1 hour per step (CityLearn challenge is hourly)
        self._dt_h = 1.0

        # --- Grid Peak Constraint (operational safety) ---
        self.peak_threshold = 96.10  # kW (calibrated from RBC 97th percentile)
        self.w_grid_peak = 0.05      # Weight for peak cost

        # --- Grid Ramp Constraint (smoothness / grid stability) ---
        self.ramp_threshold = 48.41  # kW/hour (calibrated from RBC 97th percentile)
        self.w_grid_ramp = 0.05      # Weight for ramp cost
        self._prev_grid_signal = None

        self._prev_net_consumption = None  # For ramping penalty in STEMS reward

        # --- STEMS Reward Hyperparameters (economic + stability + renewable) ---
        # Economic component weight
        self.mu_economic = 1.0
        
        # Stability component weights
        self.alpha_grid = 0.5      # Grid-level coordination weight
        self.alpha_build = 0.3     # Building-level smoothness weight
        self.beta_ramp = 0.2       # Power ramping penalty weight
        
        # Renewable component weight
        self.xi_renewable = 0.6    # Solar utilization weight
        
        # Comfort component weight (STEMS Equation 8)
        self.lambda_indoor = float(os.environ.get("CITYLEARN_STEMS_LAMBDA_INDOOR", "0.4"))
        
        # Building/grid limits (for normalization in stability reward)
        self.P_building_max = 50.0  # kW per building (approximate)
        self.P_grid_max = 500.0     # kW for entire district (17 buildings)
        
        # Reward type: "bill" (old) or "stems" (new)
        self.reward_type = os.environ.get("CITYLEARN_REWARD_TYPE", "bill").strip().lower()
        # --- Comfort (temperature) constraint (auto-enabled for LSTM datasets) ---
        self.comfort_mode = os.environ.get("CITYLEARN_ENABLE_COMFORT", "auto").strip().lower()
        self.w_comfort = float(os.environ.get("CITYLEARN_W_COST_COMFORT", "0.05"))
        self.comfort_deadband = float(os.environ.get("CITYLEARN_COMFORT_DEADBAND", "0.5"))
        self.comfort_tmin = float(os.environ.get("CITYLEARN_COMFORT_TMIN", "20.0"))
        self.comfort_tmax = float(os.environ.get("CITYLEARN_COMFORT_TMAX", "26.0"))
        self._comfort_enabled = False
        
        # LSTM warmup tracking - temperature control only active after warmup
        # During warmup (steps 0-12), LSTM replays CSV data, agent has no control
        self._lstm_warmup_steps = int(os.environ.get("CITYLEARN_LSTM_WARMUP_STEPS", "13"))
        self._comfort_warmup_complete = False


        # Track actions per timestep per charger-action-index
        # NOTE: keys MUST match the "tau" indices used by extractor (CityLearn time_step index)
        self._actions_history: Dict[int, Dict[int, float]] = {}

        # Cached schema-agnostic mapping from charger_id/name -> flat action index
        self._action_map = None

        # Discover EV charger action indices
        self._ev_charger_action_indices: List[int] = self._discover_ev_action_indices()

        # How to treat missing action samples in extractor
        self._missing_action_mode = os.environ.get(
            "CITYLEARN_EV_MISSING_ACTION_MODE", "assume_full"
        ).strip().lower()
        if self._missing_action_mode not in {"assume_full", "assume_zero", "error"}:
            print(
                f"[CityLearnSafetyEnvV3] Warning: invalid CITYLEARN_EV_MISSING_ACTION_MODE={self._missing_action_mode}, using assume_full"
            )
            self._missing_action_mode = "assume_full"

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
            self._obs_index = None

        # --- KPI + episode tracking ---
        self._step_count = 0
        self._episode_count = 0
        self._kpi_logger_initialized = False

        # Debug toggle: include obs-vs-state comparisons in info
        self._debug_obs_vs_state = bool(int(os.environ.get("CITYLEARN_DEBUG_OBS_VS_STATE", "0")))

        # Debug action clipping
        self._debug_action_clip = bool(int(os.environ.get("CITYLEARN_DEBUG_ACTION_CLIP", "0")))
        self._debug_action_clip_every = int(os.environ.get("CITYLEARN_DEBUG_ACTION_CLIP_EVERY", "500"))

        print("[CityLearnSafetyEnvV3] Initialized (V3 action-based EV deficits)")
        print(f"[CityLearnSafetyEnvV3] EV action indices (from action_names): {self._ev_charger_action_indices}")
        print(f"[CityLearnSafetyEnvV3] Missing action mode: {self._missing_action_mode}")

    # -------------------------------------------------------------------------
    # Discover EV indices robustly from action_names
    # -------------------------------------------------------------------------
    def _discover_ev_action_indices(self) -> List[int]:
        names = None
        try:
            names = getattr(self.base, "action_names", None)
        except Exception:
            names = None

        if names is None:
            citylearn_env = self._get_citylearn_env()
            names = getattr(citylearn_env, "action_names", None) if citylearn_env is not None else None

        # FLATTEN_ALL_SUBLISTS: robust multi-agent action_names -> flat list
        if isinstance(names, list) and len(names) > 0 and isinstance(names[0], list):
            flat = []
            for sub in names:
                if isinstance(sub, list):
                    flat.extend(sub)
                else:
                    flat.append(sub)
            names = flat

        if not isinstance(names, list) or len(names) == 0:
            raise RuntimeError("[CityLearnSafetyEnvV3] Could not read action_names to discover EV indices.")

        names_l = [str(n).lower() for n in names]
        key = "electric_vehicle_storage_charger_"
        ev_idx = [i for i, n in enumerate(names_l) if key in n]

        if len(ev_idx) == 0:
            print("[CityLearnSafetyEnvV3] Note: No EV chargers found. EV costs disabled.")
            return []
        return ev_idx

    def _build_action_map(self) -> "ActionMap":
        """
        Build a schema-agnostic mapping for action indices.
        Works for:
          - flat action_names
          - list-of-lists action_names (multi-building / multi-agent)
        """
        names_raw = getattr(self.base, "action_names", None)
        if names_raw is None:
            city = self._get_citylearn_env()
            names_raw = getattr(city, "action_names", None) if city is not None else None
        if names_raw is None:
            raise RuntimeError("[ActionMap] action_names not found.")

        global_to_meta: Dict[int, tuple] = {}
        flat: List[str] = []

        # Flatten list-of-lists
        if isinstance(names_raw, list) and len(names_raw) > 0 and isinstance(names_raw[0], list):
            g = 0
            for b_i, sub in enumerate(names_raw):
                if not isinstance(sub, list):
                    sub = [sub]
                for l_i, n in enumerate(sub):
                    n = str(n)
                    flat.append(n)
                    global_to_meta[g] = (b_i, l_i, n)
                    g += 1
        else:
            for g, n in enumerate(names_raw):
                n = str(n)
                flat.append(n)
                global_to_meta[g] = (-1, g, n)

        name_to_global: Dict[str, List[int]] = {}
        battery_gidx: List[int] = []
        ev_gidx: List[int] = []
        charger_id_to_gidx: Dict[str, int] = {}

        for g, n in enumerate(flat):
            name_to_global.setdefault(n, []).append(g)

            # Battery action dims (common in CityLearn)
            if n == "electrical_storage":
                battery_gidx.append(g)

            # EV charger dims
            m = self._EV_RE.match(n)
            if m:
                suffix = m.group(1)       # e.g. "15_2"
                cid = f"charger_{suffix}" # matches CityLearn charger_id
                ev_gidx.append(g)
                charger_id_to_gidx[cid] = g

        return ActionMap(
            flat_names=flat,
            global_to_meta=global_to_meta,
            name_to_global=name_to_global,
            battery_gidx=battery_gidx,
            ev_gidx=ev_gidx,
            charger_id_to_gidx=charger_id_to_gidx,
        )

    def _validate_action_map(self, am: "ActionMap") -> None:
        """
        Backward-safe validation:
        - if something is missing, we WARN (do not crash old training/eval)
        - PSF code can choose to hard-fail if it requires this mapping.
        """
        city = self._get_citylearn_env()
        if city is None:
            print("[ActionMap] Warning: could not unwrap CityLearn env for validation.")
            return

        missing = []
        for b in getattr(city, "buildings", []) or []:
            for ch in getattr(b, "electric_vehicle_chargers", []) or []:
                cid = str(getattr(ch, "charger_id", getattr(ch, "name", "")))
                if cid and cid not in am.charger_id_to_gidx:
                    missing.append(cid)

        if missing:
            print("[ActionMap] Warning: missing charger_id -> action index for:",
                  ", ".join(sorted(set(missing))))

    # -------------------------------------------------------------------------
    # Robust unwrapping helper
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

            try:
                blds = getattr(cur, "buildings", None)
                ts = getattr(cur, "time_step", None)
                if blds is not None and hasattr(blds, "__len__") and len(blds) > 0 and ts is not None:
                    return cur
            except Exception:
                pass

            advanced = False
            for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
                if hasattr(cur, attr):
                    nxt = getattr(cur, attr, None)
                    if nxt is not None and nxt is not cur:
                        cur = nxt
                        advanced = True
                        break
            if not advanced:
                break

        return None

    # -------------------------------------------------------------------------
    # Store actions under the extractor's tau index
    # -------------------------------------------------------------------------
    def _store_actions_at(self, tau: int, action):
        # Flatten list or array actions
        if isinstance(action, list):
            action_flat = np.concatenate([np.asarray(a, dtype=float).ravel() for a in action])
        else:
            action_flat = np.asarray(action, dtype=float).ravel()

        if tau not in self._actions_history:
            self._actions_history[tau] = {}

        for a_idx in self._ev_charger_action_indices:
            if 0 <= a_idx < len(action_flat):
                self._actions_history[tau][a_idx] = float(action_flat[a_idx])

    # -------------------------------------------------------------------------
    # KPI logger init
    # -------------------------------------------------------------------------
    def _ensure_kpi_logger_initialized(self):
        if self._kpi_logger_initialized:
            return

        log_dir = os.path.join(os.getcwd(), "runs", "kpi_logs")
        os.makedirs(log_dir, exist_ok=True)

        run_name = os.environ.get("CITYLEARN_KPI_RUN_NAME", "CityLearnSafety_kpis_v3").strip()
        if not run_name:
            run_name = "CityLearnSafety_kpis_v3"

        init_kpi_logger(log_dir, run_name)
        self._kpi_logger_initialized = True
        print(f"[CityLearnSafetyEnvV3] KPI logger initialized in: {log_dir} (run_name={run_name})")

    
    # === AUTO-GENERATED OBS HANDLING (multi-agent safe) ===
    def _obs_to_flat(self, obs):
        """Return a 1D float32 view of obs for internal indexing/KPIs."""
        import numpy as _np
        if isinstance(obs, _np.ndarray):
            return _np.asarray(obs, dtype=_np.float32).ravel()
        if isinstance(obs, (list, tuple)):
            parts = []
            for o in obs:
                parts.append(_np.asarray(o, dtype=_np.float32).ravel())
            return _np.concatenate(parts) if parts else _np.zeros((0,), dtype=_np.float32)
        return _np.asarray(obs, dtype=_np.float32).ravel()

    def _cast_obs_like_base(self, obs):
        """Cast obs to float32 but keep the *same structure* as base_env returns."""
        import numpy as _np
        if isinstance(obs, _np.ndarray):
            return _np.asarray(obs, dtype=_np.float32)
        if isinstance(obs, (list, tuple)):
            return [ _np.asarray(o, dtype=_np.float32) for o in obs ]
        return _np.asarray(obs, dtype=_np.float32)

    # === AUTO-GENERATED ACTION HANDLING (list-of-actions safe) ===
    def _action_to_flat(self, action):
        """Flatten actions into a single 1D array (supports list-of-arrays multi-agent)."""
        import numpy as _np
        if isinstance(action, (list, tuple)):
            parts = []
            for a in action:
                parts.append(_np.asarray(a, dtype=float).ravel())
            return _np.concatenate(parts) if parts else _np.zeros((0,), dtype=float)
        return _np.asarray(action, dtype=float).ravel()
# -------------------------------------------------------------------------
    # Gym API: reset / step
    # -------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        self._prev_net_consumption = None  # Reset STEMS ramp memory
        self._ensure_kpi_logger_initialized()

        obs, info = self.base.reset(seed=seed, options=options)

        # Build/cache schema-agnostic action index map (safe; does not break old code)

        if getattr(self, '_action_map', None) is None:

            try:

                self._action_map = self._build_action_map()

                self._validate_action_map(self._action_map)

            except Exception as _e:

                print('[ActionMap] Warning: could not build action map:', _e)

                self._action_map = None

        obs_out = obs
        obs = self._obs_to_flat(obs_out)
        obs_out = self._cast_obs_like_base(obs_out)

        self._step_count = 0
        self._actions_history.clear()
        self._comfort_warmup_complete = False  # Reset warmup tracking

        citylearn_env = self._get_citylearn_env()

        soc_state_vals = self._soc_values_from_state(citylearn_env, t_idx=self._state_time_index(citylearn_env))

        metrics = self._soc_metrics(soc_state_vals)

        building_cost = self._soc_band_cost(metrics)
        # --- Per-building battery SOC (b1..b17) for analysis ---
        # Log from OBS indices (robust across wrapper stacks).
        if self._soc_idx_obs and len(self._soc_idx_obs) >= 17:
            for i in range(17):
                idx_soc = int(self._soc_idx_obs[i])
                info[f"battery_soc_b{i+1}"] = float(obs[idx_soc]) if 0 <= idx_soc < len(obs) else 0.0
        else:
            # fallback
            for i in range(17):
                info[f"battery_soc_b{i+1}"] = 0.0

        # Handle both Box and list action spaces
        if isinstance(self.action_space, list):
            total_dim = sum(int(np.prod(sp.shape)) for sp in self.action_space)
        else:
            total_dim = int(np.prod(self.action_space.shape))
        dummy_action = np.zeros(total_dim, dtype=float)
        kpis = self._compute_basic_kpis(obs, dummy_action)

        info = dict(info)
        metrics = self._soc_metrics(soc_state_vals)
        metrics = self._soc_metrics(soc_state_vals)
        metrics = self._soc_metrics(soc_state_vals)
        info["metrics"] = metrics
        # --- Per-building battery SOC (b1..b17) for analysis ---
        # Log from OBS indices (robust across wrapper stacks).
        if self._soc_idx_obs and len(self._soc_idx_obs) >= 17:
            for i in range(17):
                idx_soc = int(self._soc_idx_obs[i])
                info[f"battery_soc_b{i+1}"] = float(obs[idx_soc]) if 0 <= idx_soc < len(obs) else 0.0
        else:
            # fallback
            for i in range(17):
                info[f"battery_soc_b{i+1}"] = 0.0

        # CMDP cost at reset: 0.0
        info["cost"] = 0.0
        info["cost_building_soc"] = float(building_cost)
        info["cost_ev_departure"] = 0.0
        info["cost_ev_dense"] = 0.0

        # Grid peak/ramp defaults
        info["cost_grid_peak"] = 0.0
        info["cost_grid_peak_raw"] = 0.0
        info["grid_peak_violation"] = 0.0

        info["cost_grid_ramp"] = 0.0
        info["cost_grid_ramp_raw"] = 0.0
        info["grid_ramp_delta"] = 0.0
        info["grid_ramp_violation"] = 0.0

        self._prev_grid_signal = None


        # Auto-enable comfort only if LSTMDynamics exists
        if not hasattr(self, '_comfort_init_done'):
            has_lstm = False
            try:
                for b in getattr(citylearn_env, "buildings", []):
                    d = getattr(b, "dynamics", None)
                    if isinstance(d, dict):
                        d = d.get("cooling") or next(iter(d.values()), None)
                    if d and "LSTMDynamics" in type(d).__name__:
                        has_lstm = True
                        break
            except Exception:
                pass
            if self.comfort_mode in ("0", "false", "off"):
                self._comfort_enabled = False
            elif self.comfort_mode in ("1", "true", "on"):
                self._comfort_enabled = True
            else:
                self._comfort_enabled = bool(has_lstm)
            self._comfort_init_done = True

        # EV defaults
        info["cost_ev_departure_agent_controllable_v3"] = 0.0
        info["cost_ev_departure_uncontrollable_v3"] = 0.0
        info["cost_ev_departure_agent_controllable_v2"] = 0.0
        info["cost_ev_departure_uncontrollable_v2"] = 0.0
        info["ev_departure_departures"] = 0
        info["ev_missing_action_samples"] = 0.0


        # Comfort defaults
        info["comfort_enabled"] = 1.0 if getattr(self, "_comfort_enabled", False) else 0.0
        info["comfort_warmup_complete"] = 0.0
        info["comfort_in_warmup"] = 1.0
        info["cost_comfort"] = 0.0
        info["cost_comfort_raw"] = 0.0
        info["comfort_violation"] = 0.0
        info["comfort_tin"] = float("nan")
        info["comfort_tset"] = float("nan")
        info.update(kpis)

        # Advanced KPI defaults
        info["ev_departure_deficit_kwh"] = 0.0
        info["ev_avoidable_deficit_kwh"] = 0.0
        info["ev_unavoidable_deficit_kwh"] = 0.0

        info["ev_v3_missed_charge_soc"] = 0.0
        info["ev_v3_discharge_harm_soc"] = 0.0
        info["ev_v3_total_blame_soc"] = 0.0
        info["ev_impossible_request_count"] = 0.0

        info["battery_abuse_kwh"] = 0.0
        info["cost_stems_battery"] = 0.0
        info["cost_stems_building_power"] = 0.0
        info["cost_stems_grid_power"] = 0.0
        info["grid_power_violation"] = 0.0
        info["building_power_violation"] = 0.0
        info["battery_soc_violation"] = 0.0
        info["battery_soc_violation_any"] = 0.0
        info["battery_soc_violation_frac"] = 0.0
        info["battery_soc_violation_rate_%"] = 0.0
        info["battery_soc_violation_count"] = 0.0

        info["cost_soc_mean_hinge"] = 0.0
        info["cost_soc_max_hinge"] = 0.0
        info["cost_soc_pnorm"] = 0.0
        info["battery_abuse_excess_kwh_equiv"] = 0.0
        info["battery_abuse_hours"] = 0.0
        info["solar_waste_kwh"] = 0.0

        info["reward"] = 0.0
        info["citylearn_reward"] = 0.0
        info["used_energy_reward"] = False

        # Bill reward metadata
        info["reward_bill_raw"] = 0.0
        info["reward_bill"] = 0.0
        info["reward_export_factor"] = float(os.environ.get("CITYLEARN_EXPORT_FACTOR", "1.0"))
        info["reward_scale"] = float(os.environ.get("CITYLEARN_REWARD_SCALE", "1.0"))
        
        # STEMS reward defaults
        info["reward_type"] = str(self.reward_type)
        info["reward_stems_total"] = 0.0
        info["reward_economic"] = 0.0
        info["reward_stability"] = 0.0
        info["reward_stability_grid"] = 0.0
        info["reward_stability_building"] = 0.0
        info["reward_stability_ramp"] = 0.0
        info["reward_renewable"] = 0.0
        info["reward_comfort"] = 0.0

        # Stable action_0..25
        for i in range(26):
            info[f"action_{i}"] = float(dummy_action[i]) if i < len(dummy_action) else 0.0

        # Stable action_ev_0..7 (safe even if fewer EVs exist)
        n_ev = len(self._ev_charger_action_indices)
        for j in range(min(8, n_ev)):
            idx = self._ev_charger_action_indices[j]
            info[f"action_ev_{j}"] = float(dummy_action[idx]) if (0 <= idx < len(dummy_action)) else 0.0
        for j in range(min(8, n_ev), 8):
            info[f"action_ev_{j}"] = 0.0

        if self._debug_obs_vs_state:
            info.update(self._debug_soc_obs_vs_state(obs, citylearn_env))

        log_kpis(info, self._step_count, self._episode_count)
        return obs_out, info

    def step(self, action):
        import numpy as np  # ensure np available for entire step method


        self._ensure_kpi_logger_initialized()

        # --- raw -> clipped action (per-dim bounds; action_2 is [0,1]) ---
        if isinstance(self.action_space, list):
            # List action space: clip per-agent, keep list structure for CityLearn
            if isinstance(action, list):
                action_arr = []
                action_raw_parts = []
                for i, a in enumerate(action):
                    a_flat = np.asarray(a, dtype=float).ravel()
                    action_raw_parts.append(a_flat)
                    low = np.asarray(self.action_space[i].low, dtype=float).ravel()
                    high = np.asarray(self.action_space[i].high, dtype=float).ravel()
                    action_arr.append(np.clip(a_flat, low, high))
                action_raw = np.concatenate(action_raw_parts)
            else:
                action_raw = np.asarray(action, dtype=float).ravel()
                action_arr = [action_raw]  # Wrap in list for CityLearn
        else:
            # Box action space
            action_raw = np.asarray(action, dtype=float).ravel()
            low = np.asarray(self.action_space.low, dtype=float).ravel()
            high = np.asarray(self.action_space.high, dtype=float).ravel()
            action_arr = np.clip(action_raw, low, high)

        if self._debug_action_clip and (self._step_count % self._debug_action_clip_every == 0):
            clip_frac = float(np.mean((action_raw < low) | (action_raw > high)))
            print(
                f"[ACTION CLIP] step={self._step_count} clip_frac={clip_frac:.3f} "
                f"raw_min={float(action_raw.min()):.2f} raw_max={float(action_raw.max()):.2f}"
            )

        # --- Align tau with extractor (pre-step time_step + 1) ---
        citylearn_env_before = self._get_citylearn_env()
        t_before = int(getattr(citylearn_env_before, "time_step", 0)) if citylearn_env_before is not None else 0
        tau_store = t_before + 1
        self._store_actions_at(tau_store, action_arr)
        if self._step_count < 15: _af = np.concatenate([np.asarray(a).ravel() for a in action_arr]) if isinstance(action_arr, list) else np.asarray(action_arr).ravel(); _evi = self._ev_charger_action_indices; print("  [SafeEnv] step=%d EV_in=%s" % (self._step_count, [round(float(_af[i]),3) for i in _evi]))



        # --- SINGLE STEP ONLY ONCE ---
        # FIX_8760: Catch IndexError from CityLearn washing machine at end of year
        try:
            obs, r_base, term, trunc, info = self.base.step(action_arr)
        except IndexError as _ie:
            print(f"[FIX_8760] IndexError at step {self._step_count}: {_ie} — ending episode")
            obs = self.base.observation_space.low if not isinstance(self.base.observation_space, list) else [s.low for s in self.base.observation_space]
            r_base = 0.0
            term = True
            trunc = False
            info = {"cost": 0.0, "fix_8760": True}
        obs_out = obs
        obs = self._obs_to_flat(obs_out)
        obs_out = self._cast_obs_like_base(obs_out)
        self._step_count += 1
        info = dict(info)

        # Per-dim action logging (stable 26 dims)
        # Flatten action_arr if it's a list
        if isinstance(action_arr, list):
            action_flat = np.concatenate([np.asarray(a).ravel() for a in action_arr])
        else:
            action_flat = np.asarray(action_arr).ravel()
        for i in range(26):
            info[f"action_{i}"] = float(action_flat[i]) if i < len(action_flat) else 0.0

        # EV action logging (stable 8 columns)
        n_ev = len(self._ev_charger_action_indices)
        for j in range(min(8, n_ev)):
            idx = self._ev_charger_action_indices[j]
            info[f"action_ev_{j}"] = float(action_flat[idx]) if (0 <= idx < len(action_flat)) else 0.0
        for j in range(min(8, n_ev), 8):
            info[f"action_ev_{j}"] = 0.0

        citylearn_env = self._get_citylearn_env()

        # --- Safety cost from INTERNAL building battery SoC (logged only) ---
        idx = self._state_time_index(citylearn_env)
        soc_state_vals = self._soc_values_from_state(citylearn_env, t_idx=idx)
        metrics = self._soc_metrics(soc_state_vals)
        building_cost = self._soc_band_cost(metrics)
        metrics = self._soc_metrics(soc_state_vals)
        building_cost = self._soc_band_cost(metrics)

        # --- STEMS Battery Safety Constraint (Eq. 16) - logged only, not in CMDP cost ---
        soc_low = float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.05"))
        soc_high = float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95"))

        # Soft-max via p-norm (p=4 default; larger => closer to max)
        p = float(os.environ.get("CITYLEARN_STEMS_PNORM_P", "4.0"))
        if p < 1.0:
            p = 1.0

        if soc_state_vals:
            v = np.array([
                max(0.0, soc_low - float(s)) + max(0.0, float(s) - soc_high)
                for s in soc_state_vals
            ], dtype=float)
            viol_ind = (v > 0.0).astype(float)

            # Keep your p-norm cost (smooth magnitude)
            battery_scale = float(os.environ.get("CITYLEARN_STEMS_BATTERY_COST_SCALE", "50.0"))
            cost_stems_battery = float(np.sum(v) * battery_scale)  # SUM + scale

            # Realistic violation metrics
            viol_any = float(viol_ind.max())        # 1 if any of 17 violates (strict)
            viol_frac = float(viol_ind.mean())      # fraction of buildings violating (recommended)
            viol_cnt = float(viol_ind.sum())        # count of buildings violating
            viol_rate_pct = 100.0 * viol_frac       # % buildings violating this step (averaged over time)

            info["battery_soc_violation_any"] = viol_any
            info["battery_soc_violation_frac"] = viol_frac
            info["battery_soc_violation_rate_%"] = viol_rate_pct
            info["battery_soc_violation_count"] = viol_cnt

            # Extra smooth diagnostics (often helpful in training)
            info["cost_soc_mean_hinge"] = float(v.mean())
            info["cost_soc_max_hinge"] = float(v.max())
            info["cost_soc_pnorm"] = float(cost_stems_battery)

            # Backward compatible: your old field (strict OR)
            soc_any_violation = viol_any
        else:
            cost_stems_battery = 0.0
            soc_any_violation = 0.0
            info["battery_soc_violation_any"] = 0.0
            info["battery_soc_violation_frac"] = 0.0
            info["battery_soc_violation_rate_%"] = 0.0
            info["battery_soc_violation_count"] = 0.0
            info["cost_soc_mean_hinge"] = 0.0
            info["cost_soc_max_hinge"] = 0.0
            info["cost_soc_pnorm"] = 0.0

        info["cost_stems_battery"] = float(cost_stems_battery)
        info["battery_soc_violation"] = float(soc_any_violation)

        # --- STEMS Building Power Capacity Constraint (Eq. 17) - logged only ---
        p_building_max = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", str(self.P_building_max)))

        # Soft-max via p-norm (shared with battery; p=4 default)
        p = float(os.environ.get("CITYLEARN_STEMS_PNORM_P", "4.0"))
        if p < 1.0:
            p = 1.0

        b_p_sum = 0.0
        b_any_violation = 0.0
        b_viol_cnt = 0.0
        n_b = 0

        if citylearn_env is not None and getattr(citylearn_env, "buildings", None):
            violations_list = []  # Track violations
            building_powers = []  # Track raw powers for debug logging
            idx_bp = self._state_time_index(citylearn_env)
            c3_controllable = os.environ.get("CITYLEARN_C3_CONTROLLABLE", "0") == "1"

            for b in citylearn_env.buildings:
                try:
                    nec = getattr(b, "net_electricity_consumption", None)
                    if nec is not None and hasattr(nec, "__len__") and len(nec) > idx_bp:
                        p_i = float(nec[idx_bp])
                    else:
                        p_i = 0.0
                except Exception:
                    p_i = 0.0

                building_powers.append(p_i)

                if c3_controllable:
                    # Agent-controllable C3: only penalize agent's contribution
                    nsl_i = 0.0
                    try:
                        nsl_arr = np.asarray(
                            getattr(b, "_Building__energy_to_non_shiftable_load", []),
                            dtype=float,
                        )
                        sg_arr = np.asarray(
                            getattr(b, "_Building__solar_generation", []),
                            dtype=float,
                        )
                        if len(nsl_arr) > idx_bp:
                            nsl_i = float(nsl_arr[idx_bp])
                        if len(sg_arr) > idx_bp:
                            nsl_i += float(sg_arr[idx_bp])  # base = NSL + solar
                    except Exception:
                        nsl_i = p_i  # fallback: treat all as uncontrollable (v_i=0)

                    if abs(nsl_i) <= p_building_max:
                        # Normal case: base load under threshold
                        v_i = max(0.0, abs(p_i) - p_building_max)
                    else:
                        # Structural violation: only penalize agent's marginal excess
                        v_i = max(0.0, abs(p_i) - abs(nsl_i))
                else:
                    # Original C3 (unchanged)
                    v_i = max(0.0, abs(p_i) - p_building_max)

                if v_i > 0.0:
                    b_any_violation = 1.0
                    b_viol_cnt += 1.0
                violations_list.append(v_i)
                n_b += 1

            building_scale = float(os.environ.get("CITYLEARN_STEMS_BUILDING_COST_SCALE", "1.0"))
            cost_stems_building_power = float(sum(violations_list) * building_scale)
        else:
            cost_stems_building_power = 0.0

        info["cost_stems_building_power"] = float(cost_stems_building_power)
        info["building_power_violation"] = float(b_any_violation)
        # --- Debug: log building power summary stats used for violations ---
        try:
            # violations_list holds per-building violation magnitudes; to infer power we need the raw powers
            # If raw building powers were computed in this scope as `building_powers`, log stats:
            if "building_powers" in locals():
                bp = list(map(float, building_powers))
                bp_sorted = sorted(bp)
                n = len(bp_sorted)
                def q(p):
                    if n == 0: return 0.0
                    idx = int(round(p*(n-1)))
                    idx = max(0, min(n-1, idx))
                    return bp_sorted[idx]
                info["p_building_max_used"] = float(p_building_max)
                info["building_power_min"] = float(bp_sorted[0]) if n else 0.0
                info["building_power_mean"] = float(sum(bp_sorted)/n) if n else 0.0
                info["building_power_max"] = float(bp_sorted[-1]) if n else 0.0
                info["building_power_p95"] = float(q(0.95))
                info["building_power_p99"] = float(q(0.99))
        except Exception:
            pass





        b_viol_frac = float(b_viol_cnt / max(1.0, float(len(getattr(self, "buildings", [])) or 17.0)))
        info["building_power_violation_count"] = float(b_viol_cnt)
        info["building_power_violation_frac"]  = float(b_viol_frac)
        info["building_power_violation_rate_%"] = float(100.0 * b_viol_frac)
        # --- KPIs ---
        kpis = self._compute_basic_kpis(obs, action_arr)

        # --- STEMS Grid Power Capacity Constraint (Eq. 18) - logged only ---
        p_grid_max = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", str(self.P_grid_max)))
        grid_import_kwh = float(kpis.get("grid_import_kwh", 0.0))
        grid_scale = float(os.environ.get("CITYLEARN_STEMS_GRID_COST_SCALE", "1.0"))
        cost_stems_grid_power = max(0.0, grid_import_kwh - p_grid_max) * grid_scale
        info["cost_stems_grid_power"] = float(cost_stems_grid_power)
        info["grid_power_violation"] = float(1.0 if cost_stems_grid_power > 0.0 else 0.0)


        # --- Bill-based reward ($) ---
        bill = 0.0
        export_factor = float(os.environ.get("CITYLEARN_EXPORT_FACTOR", "1.0"))
        reward_scale = float(os.environ.get("CITYLEARN_REWARD_SCALE", "1.0"))

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

                b0 = citylearn_env.buildings[0]
                try:
                    ep = b0.pricing.electricity_pricing
                    current_price = float(ep[idx]) if hasattr(ep, "__len__") and len(ep) > idx else 0.17
                except Exception:
                    current_price = 0.17

                if not np.isfinite(current_price) or current_price < 0:
                    current_price = 0.17

                step_net_kwh = float(total_kw) * self._dt_h
                import_kwh = max(0.0, step_net_kwh)
                export_kwh = max(0.0, -step_net_kwh)

                bill = (import_kwh * current_price) - (export_factor * export_kwh * current_price)
                reward = -reward_scale * bill
                used_energy_reward = True

                if not np.isfinite(reward):
                    print(
                        f"[WARNING] Non-finite reward: {reward}, import_kwh={import_kwh}, "
                        f"export_kwh={export_kwh}, price={current_price}, bill={bill}"
                    )
                    reward = 0.0
        except Exception:
            reward = None

        # fallback to base reward if bill reward failed
        if reward is None:
            reward = float(np.sum(r_base)) if isinstance(r_base, (list, tuple, np.ndarray)) else float(r_base)
        
        # Store bill reward for comparison
        reward_bill = float(reward)
        
        # --- Compute STEMS reward if enabled ---
        stems_components = self._compute_stems_reward(kpis)
        reward_stems = float(stems_components["reward_stems_total"])
        
        # Select reward based on reward_type
        if self.reward_type == "stems":
            reward = reward_stems
        else:
            reward = reward_bill
        
        # === EV reward shaping (gated by CITYLEARN_EV_SHAPING_ALPHA) ===
        ev_shaping_reward = self._compute_ev_reward_shaping()
        reward = reward + ev_shaping_reward

        # --- EV departure deficit (V3 + V2 for comparison) ---
        ev_cost_components = {
            "total": 0.0,
            "agent_controllable": 0.0,
            "uncontrollable": 0.0,
            "agent_controllable_v2": 0.0,
            "uncontrollable_v2": 0.0,
            "departures": 0,
            "missing_action_samples": 0.0,
        }
        try:
            if citylearn_env is not None and len(self._ev_charger_action_indices) > 0:
                ev_cost_components = ev_departure_cost_components_v3(
                    citylearn_env,
                    self._actions_history,
                    missing_action_mode=self._missing_action_mode,
                ) or ev_cost_components
        except Exception as e:
            print(f"[CityLearnSafetyEnvV3] Warning: Could not compute EV costs: {e}")

        ev_total = float(ev_cost_components.get("total", 0.0))
        ev_agent_control_v3 = float(ev_cost_components.get("agent_controllable", 0.0))
        ev_uncontrol_v3 = float(ev_cost_components.get("uncontrollable", 0.0))
        ev_agent_control_v2 = float(ev_cost_components.get("agent_controllable_v2", 0.0))
        ev_uncontrol_v2 = float(ev_cost_components.get("uncontrollable_v2", 0.0))
        ev_deps = int(ev_cost_components.get("departures", 0) or 0)
        missing_samples = float(ev_cost_components.get("missing_action_samples", 0.0) or 0.0)

        # V3 diagnostics
        v3_missed_charge_soc = float(ev_cost_components.get("v3_missed_charge_soc", 0.0) or 0.0)
        v3_discharge_harm_soc = float(ev_cost_components.get("v3_discharge_harm_soc", 0.0) or 0.0)
        v3_total_blame_soc = float(ev_cost_components.get("v3_total_blame_soc", 0.0) or 0.0)

        # --- Battery abuse + solar waste ---
        adv = self._compute_battery_solar_kpis(citylearn_env)
        battery_abuse_kwh = float(adv.get("battery_abuse_kwh", 0.0))
        solar_waste_kwh = float(adv.get("solar_waste_kwh", 0.0))
        battery_abuse_hours = float(adv.get("battery_abuse_hours", 0.0))
        battery_abuse_excess = float(adv.get("battery_abuse_excess_kwh_equiv", 0.0))

        # --- CMDP cost: scaled EV controllable component (V3) ---
        ev_cost_scale = float(os.environ.get("CITYLEARN_EV_COST_SCALE", "1.0"))
        ev_cost_for_cmdp = (ev_cost_scale * ev_agent_control_v3) if self.include_ev_in_cost else 0.0

        info["cost_ev_departure_avoidable"] = float(ev_cost_scale * ev_agent_control_v3)
        info["cost_ev_departure_unavoidable"] = float(ev_cost_scale * ev_uncontrol_v3)

        # --- Grid Peak Cost ---
        grid_import = float(kpis.get("grid_import_kwh", 0.0))
        cost_grid_peak_raw = max(0.0, grid_import - self.peak_threshold)
        cost_grid_peak = self.w_grid_peak * cost_grid_peak_raw

        info["cost_grid_peak"] = float(cost_grid_peak)
        info["cost_grid_peak_raw"] = float(cost_grid_peak_raw)
        info["grid_peak_violation"] = 1.0 if cost_grid_peak_raw > 0 else 0.0

        # --- Grid Ramp Cost ---
        p_signal = float(kpis.get("step_net_consumption_kwh", 0.0))
        if self._prev_grid_signal is None:
            ramp_delta = 0.0
            cost_grid_ramp_raw = 0.0
        else:
            ramp_delta = abs(p_signal - float(self._prev_grid_signal))
            cost_grid_ramp_raw = max(0.0, ramp_delta - self.ramp_threshold)

        self._prev_grid_signal = p_signal
        cost_grid_ramp = self.w_grid_ramp * cost_grid_ramp_raw

        info["cost_grid_ramp"] = float(cost_grid_ramp)
        info["cost_grid_ramp_raw"] = float(cost_grid_ramp_raw)
        info["grid_ramp_delta"] = float(ramp_delta)
        info["grid_ramp_violation"] = 1.0 if cost_grid_ramp_raw > 0 else 0.0

        # --- CMDP cost (for OmniSafe / Lagrangian) ---
        # EV departure shortfall + STEMS constraints (battery SOC, building power, grid import)
        w_ev = float(os.environ.get("CITYLEARN_W_COST_EV", "1.0"))
        w_soc = float(os.environ.get("CITYLEARN_W_COST_SOC", "1.0"))
        w_bld = float(os.environ.get("CITYLEARN_W_COST_BUILDING", "1.0"))
        w_grid = float(os.environ.get("CITYLEARN_W_COST_GRID", "1.0"))

        c_soc = float(info.get("cost_stems_battery", 0.0))
        c_bld = float(info.get("cost_stems_building_power", 0.0))
        c_grid = float(info.get("cost_stems_grid_power", 0.0))
        # --- Comfort (temperature) cost ---
        # NOTE: LSTM dynamics require warmup period (default 13 steps)
        # During warmup, temperature follows CSV replay and is NOT controllable
        # We only evaluate comfort constraints AFTER warmup is complete
        cost_comfort = 0.0
        cost_comfort_raw = 0.0
        comfort_viol = 0.0
        tin_out = float("nan")
        tset_out = float("nan")
        in_warmup = False

        if getattr(self, "_comfort_enabled", False):
            # Check if we're past LSTM warmup period
            warmup_steps = getattr(self, "_lstm_warmup_steps", 13)
            in_warmup = self._step_count < warmup_steps
            
            # Also check simulate_dynamics flag if available
            if not in_warmup and citylearn_env is not None:
                try:
                    b0 = citylearn_env.buildings[0]
                    sim_dyn = getattr(b0, "simulate_dynamics", True)
                    if not sim_dyn:
                        in_warmup = True
                except Exception:
                    pass
            
            if not in_warmup:
                self._comfort_warmup_complete = True
            
            # Get temperature from building
            try:
                b0 = citylearn_env.buildings[0] if citylearn_env and getattr(citylearn_env, "buildings", None) else None
                if b0:
                    t_idx = self._state_time_index(citylearn_env)
                    
                    # Read from energy_simulation (ground truth after LSTM prediction)
                    es_temps = getattr(b0.energy_simulation, "indoor_dry_bulb_temperature", None)
                    if es_temps is not None and hasattr(es_temps, "__len__") and len(es_temps) > t_idx:
                        tin_out = float(es_temps[t_idx])
                    
                    # Get setpoint if available
                    try:
                        sp_series = getattr(b0.energy_simulation, "indoor_dry_bulb_temperature_cooling_set_point", None)
                        if sp_series is not None and hasattr(sp_series, "__len__") and len(sp_series) > t_idx:
                            tset_out = float(sp_series[t_idx])
                    except Exception:
                        pass
                    
                    # Compute cost ONLY if warmup is complete
                    # DISTRICT-LEVEL: Check all 3 buildings
                    total_cost_raw = 0.0
                    any_violation = False
                    
                    for b_idx in range(len(citylearn_env.buildings)):
                        try:
                            es = getattr(citylearn_env.buildings[b_idx], "energy_simulation", None)
                            if es is None:
                                continue
                            temps = getattr(es, "indoor_dry_bulb_temperature", None)
                            if temps is None or not hasattr(temps, "__len__") or len(temps) <= t_idx:
                                continue
                            
                            t_in = float(temps[t_idx])
                            if not np.isfinite(t_in) or in_warmup:
                                continue
                            
                            # Check [20, 26] bounds for this building
                            hi = max(0.0, t_in - self.comfort_tmax)
                            lo = max(0.0, self.comfort_tmin - t_in)
                            building_cost_raw = hi + lo
                            
                            total_cost_raw += building_cost_raw
                            if building_cost_raw > 0:
                                any_violation = True
                        except Exception:
                            continue
                    
                    cost_comfort_raw = total_cost_raw
                    cost_comfort = self.w_comfort * cost_comfort_raw
                    comfort_viol = 1.0 if any_violation else 0.0
                    
                    # Keep single building values for backward compatibility
                    if np.isfinite(tin_out):
                        pass  # tin_out already set above
            except Exception:
                pass

        info["comfort_enabled"] = 1.0 if getattr(self, "_comfort_enabled", False) else 0.0
        info["comfort_warmup_complete"] = 1.0 if getattr(self, "_comfort_warmup_complete", False) else 0.0
        info["comfort_in_warmup"] = 1.0 if in_warmup else 0.0
        info["cost_comfort"] = float(cost_comfort)
        info["cost_comfort_raw"] = float(cost_comfort_raw)
        info["comfort_violation"] = float(comfort_viol)
        info["comfort_tin"] = float(tin_out)
        info["comfort_tset"] = float(tset_out)

        # --- Dense EV signal (per-step penalty for not charging when needed) ---
        ev_dense_scale = float(os.environ.get("CITYLEARN_EV_DENSE_COST_SCALE", "0.0"))
        ev_dense_cost = ev_dense_scale * v3_missed_charge_soc  # v3_missed_charge_soc is non-zero ~36% of steps
        info["cost_ev_dense"] = float(ev_dense_cost)
        
        total_cost = float(w_ev * ev_cost_for_cmdp + ev_dense_cost + info.get("cost_comfort", 0.0) + w_soc * c_soc + w_bld * c_bld + w_grid * c_grid)

        info["ev_cost_scale"] = float(ev_cost_scale)
        info["metrics"] = metrics

        info["cost"] = float(total_cost)
        info["cost_building_soc"] = float(building_cost)
        info["cost_ev_departure"] = float(ev_cost_for_cmdp)

        info["cost_ev_departure_agent_controllable_v3"] = float(ev_agent_control_v3)
        info["cost_ev_departure_uncontrollable_v3"] = float(ev_uncontrol_v3)
        info["cost_ev_departure_agent_controllable_v2"] = float(ev_agent_control_v2)
        info["cost_ev_departure_uncontrollable_v2"] = float(ev_uncontrol_v2)
        info["ev_departure_departures"] = int(ev_deps)
        info["ev_missing_action_samples"] = float(missing_samples)
        info["ev_departure_violation_count_deficit"] = int(ev_cost_components.get("deficit_violation_count", 0))
        info["ev_departure_violation_count_80pct"] = int(ev_cost_components.get("deficit_violation_count_80pct", 0))

        info.update(kpis)

        info["ev_departure_deficit_kwh"] = float(ev_total)
        info["ev_avoidable_deficit_kwh"] = float(ev_agent_control_v3)
        info["ev_unavoidable_deficit_kwh"] = float(ev_uncontrol_v3)

        info["ev_v3_missed_charge_soc"] = float(v3_missed_charge_soc)
        info["ev_v3_discharge_harm_soc"] = float(v3_discharge_harm_soc)
        info["ev_v3_total_blame_soc"] = float(v3_total_blame_soc)
        info["ev_impossible_request_count"] = 1.0 if ev_uncontrol_v3 > 0 else 0.0

        info["battery_abuse_kwh"] = float(battery_abuse_kwh)
        info["battery_abuse_excess_kwh_equiv"] = float(battery_abuse_excess)
        info["battery_abuse_hours"] = float(battery_abuse_hours)
        info["solar_waste_kwh"] = float(solar_waste_kwh)

        info["reward"] = float(reward)
        info["citylearn_reward"] = (
            float(np.sum(r_base)) if isinstance(r_base, (list, tuple, np.ndarray)) else float(r_base)
        )
        info["used_energy_reward"] = bool(used_energy_reward)

        # bill reward metadata always defined
        info["reward_bill_raw"] = float(bill)
        info["reward_bill"] = float(reward_bill)
        info["reward_export_factor"] = float(export_factor)
        info["reward_scale"] = float(reward_scale)
        
        # STEMS reward components
        info["reward_type"] = str(self.reward_type)
        info["reward_stems_total"] = float(stems_components["reward_stems_total"])
        info["reward_economic"] = float(stems_components["reward_economic"])
        info["reward_stability"] = float(stems_components["reward_stability"])
        info["reward_stability_grid"] = float(stems_components["reward_stability_grid"])
        info["reward_stability_building"] = float(stems_components["reward_stability_building"])
        info["reward_stability_ramp"] = float(stems_components["reward_stability_ramp"])
        info["reward_renewable"] = float(stems_components["reward_renewable"])
        info["reward_comfort"] = float(stems_components["reward_comfort"])
        info["reward_ev_shaping"] = float(ev_shaping_reward)

        if self._debug_obs_vs_state:
            info.update(self._debug_soc_obs_vs_state(obs, citylearn_env))

        log_kpis(info, self._step_count, self._episode_count)

        # Episode end KPIs
        if term or trunc:
            citylearn_kpis = self._extract_citylearn_kpis()
            info.update(citylearn_kpis)
            log_episode_end(info, self._episode_count)
            self._episode_count += 1

        return obs_out, float(reward), bool(term), bool(trunc), info

    # -------------------------------------------------------------------------
    # INTERNAL SoC helpers (truth)
    # -------------------------------------------------------------------------
    def _state_time_index(self, env) -> int:
        if env is None:
            return 0
        t = getattr(env, "time_step", 0)
        return max(0, int(t) - 1)

    def _soc_values_from_state(self, env, t_idx: int | None = None) -> List[float]:
        if env is None or not getattr(env, "buildings", None):
            return []
        if t_idx is None:
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

                soc_val = float(np.clip(soc_val, 0.0, 1.0))
            except Exception:
                soc_val = 0.0

            out.append(soc_val)

        return out

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
            return {"soc_mean": 0.5, "soc_min_obs": 0.5, "soc_max_obs": 0.5, "num_storages": 0.0}
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

    # === EV_REWARD_SHAPING_PATCH ===
    def _compute_ev_reward_shaping(self) -> float:
        """
        Per-step EV reward shaping: reward charging urgent EVs, penalize discharging them.
        
        For each connected EV:
          shaping += urgency * soc_progress
        
        urgency = (deficit / max_charge_per_step) / hours_until_departure
          - 0 when EV has plenty of time
          - ~1 when EV is critical
        
        soc_progress = current_soc - previous_soc (positive = charged)
        
        Gated by env var CITYLEARN_EV_SHAPING_ALPHA (default 0.0 = off).
        """
        alpha = float(os.environ.get("CITYLEARN_EV_SHAPING_ALPHA", "0.0"))
        if alpha <= 0.0:
            return 0.0
        
        citylearn_env = self._get_citylearn_env()
        if citylearn_env is None:
            return 0.0
        
        t_now = int(getattr(citylearn_env, "time_step", 0))
        t_idx = max(0, t_now - 1)
        t_prev = max(0, t_idx - 1)
        
        total_shaping = 0.0
        
        for b in getattr(citylearn_env, "buildings", []):
            for ch in (getattr(b, "electric_vehicle_chargers", None) or []):
                try:
                    sim = getattr(ch, "charger_simulation",
                                  getattr(ch, "_Charger__charger_simulation", None))
                    if sim is None:
                        continue
                    
                    state_arr = np.asarray(
                        getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                    if t_now >= len(state_arr) or float(state_arr[t_now]) != 1.0:
                        continue  # no EV connected
                    
                    dep_arr = np.asarray(
                        getattr(sim, "_electric_vehicle_departure_time"), dtype=float)
                    req_arr = np.asarray(
                        getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
                    
                    dep_hours = float(dep_arr[t_now])
                    req_soc = float(req_arr[t_now])
                    if not np.isfinite(dep_hours) or dep_hours <= 0:
                        continue
                    if not np.isfinite(req_soc):
                        req_soc = 1.0
                    req_soc = np.clip(req_soc, 0.0, 1.0)
                    
                    # Get EV SoC now and previous step
                    ev_obj = getattr(ch, "connected_electric_vehicle", None)
                    if ev_obj is None:
                        continue
                    batt = getattr(ev_obj, "battery", None)
                    if batt is None:
                        continue
                    
                    ev_cap = float(getattr(batt, "capacity", 0) or 0)
                    if ev_cap <= 0:
                        continue
                    
                    soc_data = getattr(batt, "soc", None)
                    if soc_data is None:
                        continue
                    soc_np = np.asarray(soc_data, dtype=float)
                    
                    ev_soc_now = float(np.clip(soc_np[t_idx], 0, 1)) if t_idx < len(soc_np) else 0.0
                    ev_soc_prev = float(np.clip(soc_np[t_prev], 0, 1)) if t_prev < len(soc_np) else ev_soc_now
                    
                    # SoC progress this step
                    soc_progress = ev_soc_now - ev_soc_prev
                    
                    # Urgency: how critical is charging right now?
                    deficit = max(0.0, req_soc - ev_soc_now)
                    if deficit <= 1e-6:
                        continue  # already met requirement
                    
                    max_p = getattr(ch, "max_charging_power",
                                    getattr(ch, "_Charger__max_charging_power", 0))
                    if isinstance(max_p, np.ndarray):
                        max_p = float(max_p.ravel()[0])
                    else:
                        max_p = float(max_p or 0)
                    
                    if max_p <= 0:
                        continue
                    
                    max_soc_per_step = (max_p * 0.95) / ev_cap
                    steps_needed = deficit / max(max_soc_per_step, 1e-9)
                    urgency = min(1.0, steps_needed / max(1.0, dep_hours))
                    
                    # Shaping: urgency * soc_progress
                    # When urgency high + agent charges: positive reward
                    # When urgency high + agent discharges: negative reward
                    total_shaping += urgency * soc_progress
                    
                except Exception:
                    continue
        
        return alpha * total_shaping

    # === END EV_REWARD_SHAPING_PATCH ===

    def _compute_stems_reward(self, kpis: Dict[str, float]) -> Dict[str, float]:
        """
        Compute STEMS-style reward with 3 components:
        R = R_economic + R_stability + R_renewable
        
        Based on STEMS paper equations (3)-(9), adapted for no temperature control.
        """
        result = {
            "reward_stems_total": 0.0,
            "reward_economic": 0.0,
            "reward_stability": 0.0,
            "reward_stability_grid": 0.0,
            "reward_stability_building": 0.0,
            "reward_stability_ramp": 0.0,
            "reward_renewable": 0.0,
            "reward_comfort": 0.0,
        }
        
        # --- Economic Component (Equation 5) ---
        # R_economic = -μ × price × (import - export_factor × export)
        # Export paid at CITYLEARN_EXPORT_FACTOR of import price (env var)
        price = float(kpis.get("electricity_price", 0.17))
        net_consumption = float(kpis.get("step_net_consumption_kwh", 0.0))
        import_kwh = float(kpis.get("grid_import_kwh", 0.0))
        export_kwh = float(kpis.get("grid_export_kwh", 0.0))
        export_factor = float(os.environ.get("CITYLEARN_EXPORT_FACTOR", "1.0"))
        
        reward_economic = -self.mu_economic * price * (import_kwh - export_factor * export_kwh)
        
        # --- Stability Component (Equations 6-7) ---
        # Grid-level: penalize total district import relative to grid capacity
        grid_import = float(kpis.get("grid_import_kwh", 0.0))
        grid_import_ratio = grid_import / max(1e-6, self.P_grid_max)
        reward_stability_grid = self.alpha_grid * (1.0 - grid_import_ratio**2)
        
        # Building-level: evaluate EACH building individually (STEMS Eq. 6-7)
        # R_stability_building = α_build × Σ_i(1 - |P_i|/P_max)
        reward_stability_building = 0.0
        try:
            citylearn_env = self._get_citylearn_env()
            if citylearn_env and getattr(citylearn_env, "buildings", None):
                t_idx = self._state_time_index(citylearn_env)
                building_sum = 0.0
                building_count = 0
                
                for b in citylearn_env.buildings:
                    try:
                        # Get net electricity consumption for this building
                        nec = getattr(b, "net_electricity_consumption", None)
                        if nec is None:
                            continue
                        
                        if hasattr(nec, "__len__") and len(nec) > t_idx:
                            p_i = float(nec[t_idx])
                        elif np.isscalar(nec):
                            p_i = float(nec)
                        else:
                            continue
                        
                        if not np.isfinite(p_i):
                            continue
                        
                        # Compute (1 - |P_i|/P_max) for this building
                        power_ratio = abs(p_i) / max(1e-6, self.P_building_max)
                        building_component = 1.0 - power_ratio
                        building_sum += building_component
                        building_count += 1
                        
                    except Exception:
                        continue
                
                # Average across all buildings
                if building_count > 0:
                    reward_stability_building = self.alpha_build * (building_sum / building_count)
                    
        except Exception:
            # Fallback to old approximation if loop fails
            building_consumption = abs(net_consumption) / 17.0
            building_ratio = building_consumption / max(1e-6, self.P_building_max)
            reward_stability_building = self.alpha_build * (1.0 - building_ratio)
        
        # Ramp penalty: penalize power fluctuations (Equation 7)
        # Compute internally (do NOT rely on kpis["grid_ramp_delta"])
        net_consumption = float(kpis.get("step_net_consumption_kwh", 0.0))
        if self._prev_net_consumption is None:
            ramp_delta = 0.0
        else:
            ramp_delta = abs(net_consumption - float(self._prev_net_consumption))
        self._prev_net_consumption = float(net_consumption)
        
        # Normalize by P_grid_max (district-level) not P_building_max
        ramp_ratio = ramp_delta / max(1e-6, self.P_grid_max)
        reward_stability_ramp = -self.beta_ramp * ramp_ratio
        
        reward_stability = (
            reward_stability_grid + 
            reward_stability_building + 
            reward_stability_ramp
        )
        
        # --- Renewable Component (Equation 9) ---
        # R_renewable = ξ × min(solar / (solar + import), 1.0)
        solar_gen = float(kpis.get("solar_generation_kwh", 0.0))
        import_kwh = max(0.0, net_consumption)
        
        if solar_gen > 0.0 or import_kwh > 0.0:
            solar_ratio = solar_gen / (solar_gen + import_kwh)
            reward_renewable = self.xi_renewable * min(solar_ratio, 1.0)
        else:
            reward_renewable = 0.0
        
        # --- Comfort Component (Equation 8) ---
        # R_comfort = Σ_i(-λ_indoor · |T_in,i - T_ref,i|²) for all buildings
        # Sum of individual building penalties (district-level comfort)
        reward_comfort = 0.0
        if getattr(self, "_comfort_enabled", False):
            try:
                citylearn_env = self._get_citylearn_env()
                if citylearn_env and getattr(citylearn_env, "buildings", None):
                    t_idx = self._state_time_index(citylearn_env)
                    
                    # Aggregate comfort penalty across all buildings
                    total_penalty = 0.0
                    building_count = 0
                    
                    for building_idx, b in enumerate(citylearn_env.buildings):
                        try:
                            # Get indoor temperature from energy_simulation (ground truth)
                            es = getattr(b, "energy_simulation", None)
                            if es is None:
                                continue
                            
                            temps = getattr(es, "indoor_dry_bulb_temperature", None)
                            if temps is None or not hasattr(temps, "__len__") or len(temps) <= t_idx:
                                continue
                            
                            tin = float(temps[t_idx])
                            if not np.isfinite(tin) or tin <= 0.0:
                                continue
                            
                            # Get setpoint for this building
                            tref = (self.comfort_tmin + self.comfort_tmax) / 2.0  # Default: 23°C
                            sp_series = getattr(es, "indoor_dry_bulb_temperature_cooling_set_point", None)
                            if sp_series is not None and hasattr(sp_series, "__len__") and len(sp_series) > t_idx:
                                tref_candidate = float(sp_series[t_idx])
                                if np.isfinite(tref_candidate):
                                    tref = tref_candidate
                            
                            # Compute penalty for this building: -λ_indoor · |T_in - T_ref|²
                            temp_deviation = abs(tin - tref)
                            building_penalty = -self.lambda_indoor * (temp_deviation ** 2)
                            total_penalty += building_penalty
                            building_count += 1
                            
                        except Exception:
                            continue
                    
                    # Use sum of penalties (district-level)
                    reward_comfort = float(total_penalty)
                    
            except Exception:
                reward_comfort = 0.0
        
        # Store component before computing total
        result["reward_comfort"] = float(reward_comfort)
        
        # Total STEMS reward (4 components: economic + stability + renewable + comfort)
        reward_stems_total = reward_economic + reward_stability + reward_renewable + reward_comfort
        
        # Store components
        result["reward_stems_total"] = float(reward_stems_total)
        result["reward_economic"] = float(reward_economic)
        result["reward_stability"] = float(reward_stability)
        result["reward_stability_grid"] = float(reward_stability_grid)
        result["reward_stability_building"] = float(reward_stability_building)
        result["reward_stability_ramp"] = float(reward_stability_ramp)
        result["reward_renewable"] = float(reward_renewable)
        
        return result

    def _compute_basic_kpis(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        kpis: Dict[str, float] = {}

        citylearn_env = self._get_citylearn_env()
        soc_vals = self._soc_values_from_state(citylearn_env)
        kpis["soc_mean"] = float(np.mean(soc_vals)) if soc_vals else 0.0
        kpis["soc_min"] = float(np.min(soc_vals)) if soc_vals else 0.0
        kpis["soc_max"] = float(np.max(soc_vals)) if soc_vals else 0.0
        kpis["soc_std"] = float(np.std(soc_vals)) if soc_vals else 0.0

        # --- Per-building battery SOC (b1..b17) from STATE soc_vals ---
        # This is what you need for: sum(violated buildings)/(17*T)
        for i in range(17):
            kpis[f"battery_soc_b{i+1}"] = float(soc_vals[i]) if i < len(soc_vals) else 0.0

        action = self._action_to_flat(action)
        kpis["action_mean"] = float(np.mean(action)) if action.size else 0.0
        kpis["action_std"] = float(np.std(action)) if action.size else 0.0
        kpis["action_min"] = float(np.min(action)) if action.size else 0.0
        kpis["action_max"] = float(np.max(action)) if action.size else 0.0

        kpis["step_count"] = float(self._step_count)

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

        kpis["thermal_discomfort"] = 0.0
        
        # Indoor temperature: average across all buildings with temp data
        indoor_temps = []
        if citylearn_env is not None and getattr(citylearn_env, "buildings", None):
            idx = self._state_time_index(citylearn_env)
            for b in citylearn_env.buildings:
                try:
                    es = getattr(b, "energy_simulation", None)
                    if es is not None:
                        temps = getattr(es, "indoor_dry_bulb_temperature", None)
                        if temps is not None and hasattr(temps, "__len__") and len(temps) > idx:
                            t = float(temps[idx])
                            if np.isfinite(t):
                                indoor_temps.append(t)
                except Exception:
                    pass
        kpis["indoor_temperature"] = float(np.mean(indoor_temps)) if indoor_temps else 0.0
        
        # Outdoor temperature (from building 0)
        kpis["outdoor_temperature"] = 0.0
        if citylearn_env is not None and getattr(citylearn_env, "buildings", None):
            idx = self._state_time_index(citylearn_env)
            b0 = citylearn_env.buildings[0]
            try:
                weather = getattr(b0, "weather", None)
                if weather is not None:
                    outdoor = getattr(weather, "outdoor_dry_bulb_temperature", None)
                    if outdoor is not None and hasattr(outdoor, "__len__") and len(outdoor) > idx:
                        kpis["outdoor_temperature"] = float(outdoor[idx])
            except Exception:
                pass

        # Sum solar generation across ALL buildings (not just building 0)
        # Note: CityLearn stores solar as negative, so we take absolute value
        solar_gen = 0.0
        if citylearn_env is not None and getattr(citylearn_env, "buildings", None):
            idx = self._state_time_index(citylearn_env)
            for b in citylearn_env.buildings:
                try:
                    sg = getattr(b, "solar_generation", None)
                    if sg is not None and hasattr(sg, "__len__") and len(sg) > idx:
                        solar_gen += abs(float(sg[idx])) * self._dt_h
                except Exception:
                    pass
        kpis["solar_generation_kwh"] = float(solar_gen)

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

        violation = False
        if soc_vals:
            viol_ind = [1.0 if (s < self.soc_min or s > self.soc_max) else 0.0 for s in soc_vals]
            violation = bool(max(viol_ind))
            kpis["constraint_violation_any"] = float(max(viol_ind))
            kpis["constraint_violation_frac"] = float(np.mean(viol_ind))
            kpis["constraint_violation_rate_%"] = 100.0 * float(np.mean(viol_ind))
        else:
            kpis["constraint_violation_any"] = 0.0
            kpis["constraint_violation_frac"] = 0.0
            kpis["constraint_violation_rate_%"] = 0.0
        kpis["constraint_violation"] = 1.0 if violation else 0.0

        return kpis

    # -------------------------------------------------------------------------
    # CityLearn KPI extraction at episode end
    # -------------------------------------------------------------------------
    def _extract_citylearn_kpis(self) -> Dict[str, float]:
        citylearn_kpis: Dict[str, float] = {}
        try:
            citylearn_env = self._get_citylearn_env()
            if citylearn_env is None:
                raise RuntimeError("CityLearnEnv not found in wrapper stack.")

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
                return float("nan")

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
            print(f"[CityLearnSafetyEnvV3] Warning: Could not extract CityLearn KPIs: {e}")
            print(f"[CityLearnSafetyEnvV3] Traceback: {traceback.format_exc()}")
            citylearn_kpis = {
                "citylearn_electricity_consumption_total": float("nan"),
                "citylearn_carbon_emissions_total": float("nan"),
                "citylearn_cost_total": float("nan"),
                "citylearn_daily_peak_average": float("nan"),
                "citylearn_all_time_peak_average": float("nan"),
                "citylearn_ramping_average": float("nan"),
                "citylearn_discomfort_proportion": float("nan"),
                "citylearn_zero_net_energy": float("nan"),
            }

        return citylearn_kpis
