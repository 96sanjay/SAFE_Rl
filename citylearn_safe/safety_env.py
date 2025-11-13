# citylearn_safe/safety_env.py
from __future__ import annotations
import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import gymnasium as gym

from .schema_index import soc_indices_from_schema_and_obs_dim, obs_feature_index
from .kpi_logger import log_kpis, log_episode_end, init_kpi_logger


class CityLearnSafetyEnv(gym.Env):
    """Adds a CMDP-style safety cost via info['cost'] (SoC band from observations)."""
    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: Any,
        *,
        #soc_min: float = 0.1,
        #soc_max: float = 0.9
        soc_min: float = 0.0,
        soc_max: float = 0.95,
        soc_obs_name: str = "electrical_storage_soc",
        cost_mode: str = "hinge",
    ):
        super().__init__()
        self.base = base_env
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        override_mode = os.environ.get("CITYLEARN_COST_MODE")
        if override_mode:
            cost_mode = override_mode

        cost_mode_normalized = cost_mode.lower()
        if cost_mode_normalized not in {"hinge", "binary"}:
            raise ValueError(
                f"Unsupported cost_mode '{cost_mode}'. "
                "Choose between 'hinge' (default) or 'binary'."
            )
        self.cost_mode = cost_mode_normalized

        schema_path = os.environ.get("CITYLEARN_SCHEMA")
        if not schema_path or not os.path.exists(schema_path):
            raise RuntimeError("CITYLEARN_SCHEMA must point to your schema.json.")

        obs_dim = int(self.observation_space.shape[0])
        self._soc_idx, self._per_b_names = soc_indices_from_schema_and_obs_dim(
            schema_path,
            obs_dim,
            soc_name=soc_obs_name,
        )
        if len(self._soc_idx) == 0:
            print(
                f"[CityLearnSafetyEnv] WARNING: No '{soc_obs_name}' indices found "
                f"(per-building active names: {self._per_b_names})."
            )

        # --- KPI setup ---
        self._dt_h = 1.0  # 1 hour per step (from CityLearn schema)
        self._idx_net_consumption = None  # obs index for net_electricity_consumption
        self._step_count = 0
        self._episode_count = 0
        self._kpi_logger_initialized = False
        
        # Find net_electricity_consumption index
        net_consumption_idx, _ = soc_indices_from_schema_and_obs_dim(
            schema_path,
            obs_dim,
            soc_name="net_electricity_consumption",
        )
        if net_consumption_idx:
            self._idx_net_consumption = net_consumption_idx[0]  # Take first building's index
        
        # Find indices for specific observation features
        self._idx_month = obs_feature_index(schema_path, obs_dim, "month")
        self._idx_day_type = obs_feature_index(schema_path, obs_dim, "day_type")
        self._idx_non_shiftable_load = obs_feature_index(schema_path, obs_dim, "non_shiftable_load")

    def _ensure_kpi_logger_initialized(self):
        """Initialize KPI logger if not already done."""
        if not self._kpi_logger_initialized:
            # Try to find the current run directory by looking for the most recently created directory
            # that matches the expected pattern and has a progress.csv file
            runs_dir = os.path.join(os.getcwd(), "runs")
            if os.path.exists(runs_dir):
                most_recent_dir = None
                most_recent_time = 0
                
                for item in os.listdir(runs_dir):
                    item_path = os.path.join(runs_dir, item)
                    if os.path.isdir(item_path) and "CityLearnSafety-SoC-v0" in item:
                        # Check for seed directories within this run
                        for seed_item in os.listdir(item_path):
                            seed_path = os.path.join(item_path, seed_item)
                            if os.path.isdir(seed_path) and seed_item.startswith("seed-"):
                                # Check if this directory was created very recently (within last 2 minutes)
                                import time
                                dir_creation_time = os.path.getctime(seed_path)
                                current_time = time.time()
                                
                                # Look for directories created within the last 2 minutes
                                if current_time - dir_creation_time < 120:  # 2 minutes
                                    if dir_creation_time > most_recent_time:
                                        most_recent_time = dir_creation_time
                                        most_recent_dir = seed_path
                
                if most_recent_dir:
                    # Initialize KPI logger in this directory
                    init_kpi_logger(most_recent_dir, "kpis")
                    self._kpi_logger_initialized = True
                    print(f"[CityLearnSafetyEnv] KPI logger initialized in: {most_recent_dir}")
                else:
                    # Fallback: create a temporary KPI file in the runs directory
                    temp_dir = os.path.join(runs_dir, "kpi_logs")
                    os.makedirs(temp_dir, exist_ok=True)
                    init_kpi_logger(temp_dir, "CityLearnSafety-SoC-v0_kpis")
                    self._kpi_logger_initialized = True
                    print(f"[CityLearnSafetyEnv] KPI logger initialized in fallback location: {temp_dir}")

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        obs, info = self.base.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)

        # KPI logger will be initialized in the first step call

        # Reset KPI tracking
        self._step_count = 0
        # NOTE: Episode counter should NOT increment on reset
        # It will increment when episode ends (term or trunc in step())
        # self._episode_count += 1  # REMOVED - causes double counting

        # Compute safety metrics
        soc_vals = self._soc_values_from_obs(obs)
        metrics = self._soc_metrics(soc_vals)
        cost = self._soc_band_cost(metrics)

        # Compute basic KPIs
        kpis = self._compute_basic_kpis(obs, np.zeros_like(self.action_space.sample()))

        info = dict(info)
        info["metrics"] = metrics
        info["cost"] = float(cost)
        info.update(kpis)  # Add KPIs to info
        info["reward"] = 0.0
        
        # Log KPIs using custom logger
        log_kpis(info, self._step_count, self._episode_count)
        
        return obs, info

    def step(self, action):
        # Ensure KPI logger is initialized before logging
        self._ensure_kpi_logger_initialized()
        
        obs, r, term, trunc, info = self.base.step(action)
        obs = np.asarray(obs, dtype=np.float32)

        # Update step count
        self._step_count += 1

        # Compute safety metrics
        soc_vals = self._soc_values_from_obs(obs)
        metrics = self._soc_metrics(soc_vals)
        cost = self._soc_band_cost(metrics)

        # Compute basic KPIs
        kpis = self._compute_basic_kpis(obs, action)

        info = dict(info)
        info["metrics"] = metrics
        info["cost"] = float(cost)
        info.update(kpis)  # Add KPIs to info
        info["reward"] = float(r)
        
        # Log KPIs using custom logger
        log_kpis(info, self._step_count, self._episode_count)
        
        # Log episode end if episode is finished
        if term or trunc:
            # Increment episode counter when episode actually ends
            self._episode_count += 1
            # Extract CityLearn KPIs at episode end
            citylearn_kpis = self._extract_citylearn_kpis()
            info.update(citylearn_kpis)
            log_episode_end(info, self._episode_count)
        
        return obs, float(r), bool(term), bool(trunc), info

    # ---- SoC helpers
    def _soc_values_from_obs(self, obs: np.ndarray) -> List[float]:
        vals: List[float] = []
        for idx in self._soc_idx:
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

    # ---- Basic KPI helpers
    def _compute_basic_kpis(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        """Compute basic KPIs from current observation and action."""
        kpis = {}
        
        # 1. Battery SoC tracking (already computed in metrics)
        soc_vals = self._soc_values_from_obs(obs)
        if soc_vals:
            kpis["soc_mean"] = float(np.mean(soc_vals))
            kpis["soc_min"] = float(np.min(soc_vals))
            kpis["soc_max"] = float(np.max(soc_vals))
            kpis["soc_std"] = float(np.std(soc_vals))
        else:
            kpis["soc_mean"] = 0.0 #!!!
            kpis["soc_min"] = 0.0 #!!!
            kpis["soc_max"] = 0.0 #!!!
            kpis["soc_std"] = 0.0 #!!!
        
        # 3. Action statistics
        kpis["action_mean"] = float(np.mean(action))
        kpis["action_std"] = float(np.std(action))
        kpis["action_min"] = float(np.min(action))
        kpis["action_max"] = float(np.max(action))
        
        # 4. Step tracking
        kpis["step_count"] = float(self._step_count)
        
        # 5. Specific observation features
        if self._idx_month is not None and self._idx_month < len(obs):
            kpis["month"] = float(obs[self._idx_month])
        else:
            kpis["month"] = 0.0
        
        if self._idx_day_type is not None and self._idx_day_type < len(obs):
            kpis["day_type"] = float(obs[self._idx_day_type])
        else:
            kpis["day_type"] = 0.0
        
        if self._idx_non_shiftable_load is not None and self._idx_non_shiftable_load < len(obs):
            # Note: This is normalized, so value is in [0, 1]
            kpis["non_shiftable_load"] = float(obs[self._idx_non_shiftable_load])
        else:
            kpis["non_shiftable_load"] = 0.0
        
        # 6. Net electricity consumption tracking (step-by-step)
        # Positive = import from grid, Negative = export to grid
        # Read RAW value from building (not normalized observation) to match reward calculation
        try:
            citylearn_env = self.base.base
            building = citylearn_env.buildings[0]  # First building
            if hasattr(building, 'net_electricity_consumption') and len(building.net_electricity_consumption) > 0:
                # Get raw net consumption in kW (not normalized)
                net_consumption_kw = building.net_electricity_consumption[-1]
                # Convert to kWh for this step
                step_consumption_kwh = net_consumption_kw * self._dt_h
            else:
                step_consumption_kwh = 0.0
        except Exception:
            # Fallback to normalized observation if building access fails
            if self._idx_net_consumption is not None and self._idx_net_consumption < len(obs):
                # Note: This is normalized, so won't match reward exactly
                net_consumption_kw = obs[self._idx_net_consumption]
                step_consumption_kwh = net_consumption_kw * self._dt_h
            else:
                step_consumption_kwh = 0.0
        kpis["step_net_consumption_kwh"] = float(step_consumption_kwh)
        
        # 6. Electricity pricing (from CityLearn building)
        try:
            # Access the underlying CityLearn environment
            citylearn_env = self.base.base
            building = citylearn_env.buildings[0]  # First building
            
            # Get current electricity price ($/kWh)
            if hasattr(building.pricing, 'electricity_pricing') and len(building.pricing.electricity_pricing) > 0:
                current_price = building.pricing.electricity_pricing[-1]
                kpis["electricity_price"] = float(current_price)
                
                # Calculate step-wise cost ($ for this timestep)
                step_cost = step_consumption_kwh * current_price
                kpis["step_cost"] = float(step_cost)
            else:
                kpis["electricity_price"] = 0.0
                kpis["step_cost"] = 0.0
        except Exception as e:
            kpis["electricity_price"] = 0.0
            kpis["step_cost"] = 0.0
        
        # 7. Temperature tracking (outdoor and indoor if available)
        try:
            citylearn_env = self.base.base
            building = citylearn_env.buildings[0]
            
            # Outdoor temperature
            if hasattr(building.weather, 'outdoor_dry_bulb_temperature') and len(building.weather.outdoor_dry_bulb_temperature) > 0:
                kpis["outdoor_temperature"] = float(building.weather.outdoor_dry_bulb_temperature[-1])
            else:
                kpis["outdoor_temperature"] = 0.0
            
            # Indoor temperature (if available)
            if hasattr(building, 'indoor_dry_bulb_temperature') and len(building.indoor_dry_bulb_temperature) > 0:
                kpis["indoor_temperature"] = float(building.indoor_dry_bulb_temperature[-1])
            else:
                kpis["indoor_temperature"] = 0.0
        except Exception as e:
            kpis["outdoor_temperature"] = 0.0
            kpis["indoor_temperature"] = 0.0
        
        # 8. Solar generation (if available)
        try:
            citylearn_env = self.base.base
            building = citylearn_env.buildings[0]
            
            if hasattr(building, 'solar_generation') and len(building.solar_generation) > 0:
                # Solar generation is typically negative in net consumption
                solar_kwh = building.solar_generation[-1] * self._dt_h
                kpis["solar_generation_kwh"] = float(solar_kwh)
            else:
                kpis["solar_generation_kwh"] = 0.0
        except Exception as e:
            kpis["solar_generation_kwh"] = 0.0
        
        # 8b. Raw non-shiftable load (if available)
        try:
            citylearn_env = self.base.base
            building = citylearn_env.buildings[0]
            
            # Get raw load from building's non_shiftable_load attribute
            if hasattr(building, 'non_shiftable_load') and len(building.non_shiftable_load) > 0:
                # Raw load in kW, convert to kWh
                load_kwh = building.non_shiftable_load[-1] * self._dt_h
                kpis["non_shiftable_load_kwh"] = float(load_kwh)
            else:
                kpis["non_shiftable_load_kwh"] = 0.0
        except Exception as e:
            kpis["non_shiftable_load_kwh"] = 0.0
        
        # 9. Import/Export breakdown
        # Positive net consumption = importing, negative = exporting
        if step_consumption_kwh > 0:
            kpis["grid_import_kwh"] = float(step_consumption_kwh)
            kpis["grid_export_kwh"] = 0.0
        else:
            kpis["grid_import_kwh"] = 0.0
            kpis["grid_export_kwh"] = float(abs(step_consumption_kwh))
        
        # 10. Constraint violations (binary)
        soc_vals = self._soc_values_from_obs(obs)
        violation = False
        if soc_vals:
            violation = any(soc < self.soc_min or soc > self.soc_max for soc in soc_vals)
        kpis["constraint_violation"] = 1.0 if violation else 0.0
        
        return kpis
    
    def _extract_citylearn_kpis(self) -> Dict[str, float]:
        """Extract CityLearn KPIs from the underlying environment."""
        citylearn_kpis = {}
        try:
            # Access the underlying CityLearn environment
            # self.base is SingleAgentListAdapter -> self.base.base is the CityLearn environment (with wrappers)
            citylearn_env = self.base.base
            
            # Get CityLearn KPIs
            kpis_df = citylearn_env.evaluate()
            
            # Dynamically detect the building name from the results
            # For single building setup, there should be only one unique building name
            if 'name' in kpis_df.columns:
                unique_buildings = kpis_df['name'].unique()
                if len(unique_buildings) > 0:
                    building_name = unique_buildings[0]  # Use the first (and likely only) building
                    if len(unique_buildings) > 1:
                        print(f"[CityLearnSafetyEnv] Warning: Multiple buildings found, using {building_name}")
                else:
                    building_name = "Building_1"  # Fallback
                    print(f"[CityLearnSafetyEnv] Warning: No building names found in KPI results, using fallback")
            else:
                building_name = "Building_1"  # Fallback
                print(f"[CityLearnSafetyEnv] Warning: 'name' column not found in KPI results, using fallback")
            
            # Helper function to safely extract KPI value
            def extract_kpi(cost_function_name: str) -> float:
                result = kpis_df[
                    (kpis_df['name'] == building_name) & 
                    (kpis_df['cost_function'] == cost_function_name)
                ]
                if not result.empty:
                    value = result['value'].iloc[0]
                    # Handle NaN values
                    if pd.notna(value):
                        return float(value)
                return 0.0
            
            # Extract all KPIs
            citylearn_kpis['citylearn_electricity_consumption_total'] = extract_kpi('electricity_consumption_total')
            citylearn_kpis['citylearn_carbon_emissions_total'] = extract_kpi('carbon_emissions_total')
            citylearn_kpis['citylearn_cost_total'] = extract_kpi('cost_total')
            citylearn_kpis['citylearn_zero_net_energy'] = extract_kpi('zero_net_energy')
            
            # Note: daily_peak_average and discomfort_proportion are not available for single building setups
            # Set them to 0.0 as defaults
            citylearn_kpis['citylearn_daily_peak_average'] = 0.0
            citylearn_kpis['citylearn_discomfort_proportion'] = 0.0
                
        except Exception as e:
            import traceback
            print(f"[CityLearnSafetyEnv] Warning: Could not extract CityLearn KPIs: {e}")
            print(f"[CityLearnSafetyEnv] Traceback: {traceback.format_exc()}")
            # Provide default values
            citylearn_kpis = {
                'citylearn_electricity_consumption_total': 0.0,
                'citylearn_carbon_emissions_total': 0.0,
                'citylearn_cost_total': 0.0,
                'citylearn_daily_peak_average': 0.0,
                'citylearn_discomfort_proportion': 0.0,
                'citylearn_zero_net_energy': 0.0,
            }
        
        return citylearn_kpis


#How to Add New Constraints:
#Option A: Extend the existing cost function


#def _power_constraint_cost(self, stats: Dict[str, float]) -> float:
#    """Penalize power consumption above threshold."""
#    max_power = 50.0  # kW threshold
#    power_violation = max(0.0, stats.get("max_power_obs", 0) - max_power)
#    return power_violation / max_power

#def _temperature_constraint_cost(self, stats: Dict[str, float]) -> float:
#    """Penalize battery temperature outside safe range."""
#    temp_min, temp_max = 15.0, 35.0  # Celsius
#    temp = stats.get("battery_temp_obs", 25.0)
#    low_violation = max(0.0, temp_min - temp)
#    high_violation = max(0.0, temp - temp_max)
#    return (low_violation + high_violation) / 20.0



#Observation Extraction: citylearn_safe/schema_index.py
#they need to extract new observations from the CityLearn environment.

#citylearn_safe/schema_index.py

#Safety Environment Initialization: citylearn_safe/safety_env.py
#Update the __init__ method to handle new constraints:

#Environment Registration: citylearn_safe/omni_env.py
#Update the CMDP environment to pass new constraint parameters:


# Example alternative cost functions (commented out for reference):
#
# def _soc_band_cost(self, stats: Dict[str, float]) -> float:
#     """Binary cost with different weights for different violations."""
#     if stats.get("num_storages", 0.0) <= 0.0:
#         return 0.0
#     
#     soc_min_obs = stats["soc_min_obs"]
#     soc_max_obs = stats["soc_max_obs"]
#     
#     cost = 0.0
#     
#     # Different penalties for different violations
#     if soc_min_obs < self.soc_min:
#         cost += 1.0  # Undercharge violation
#     if soc_max_obs > self.soc_max:
#         cost += 1.5  # Overcharge violation (more dangerous)
#     
#     return cost
#
# def _soc_band_cost_per_building(self, obs: np.ndarray) -> float:
#     """Binary cost counting violations per building."""
#     soc_vals = self._soc_values_from_obs(obs)
#     
#     if not soc_vals:
#         return 0.0
#     
#     violations = 0
#     for soc in soc_vals:
#         if soc < self.soc_min or soc > self.soc_max:
#             violations += 1
#     
#     return float(violations)  # 0 to N_buildings