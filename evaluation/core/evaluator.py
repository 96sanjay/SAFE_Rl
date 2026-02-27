from __future__ import annotations
import os
from typing import Any, Dict
import numpy as np


def make_env() -> Any:
    """Create standardized environment with all wrappers"""
    from citylearn.citylearn import CityLearnEnv
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.adapters import SingleAgentListAdapter
    
    schema = os.environ.get("CITYLEARN_SCHEMA", "").strip()
    if not schema:
        raise RuntimeError("CITYLEARN_SCHEMA env var not set")
    
    base = CityLearnEnv(schema=schema, central_agent=True)
    
    try:
        from citylearn.wrappers import NormalizedObservationWrapper
        base = NormalizedObservationWrapper(base)
    except Exception:
        pass
    
    env = SingleAgentListAdapter(base)
    env = CityLearnSafetyEnvV3(env)
    return env


def _is_charger_connected(raw_env, charger_idx: int, action_names) -> bool:
    """Check if EV connected at current timestep"""
    if charger_idx >= len(action_names):
        return False
    
    name = str(action_names[charger_idx]).strip().lower()
    if "electric_vehicle_storage_charger_" not in name:
        return False
    
    suffix = name.split("electric_vehicle_storage_charger_", 1)[1]
    charger_id = f"charger_{suffix}"
    
    # Use time_step + 1 (aligns with wrapper tau indexing)
    t_state = int(getattr(raw_env, "time_step", 0)) + 1
    
    for b in getattr(raw_env, "buildings", []) or []:
        for ch in getattr(b, "electric_vehicle_chargers", []) or []:
            cid = str(getattr(ch, "charger_id", "")).strip()
            if cid.startswith("b'") and cid.endswith("'"):
                cid = cid[2:-1]
            
            if cid != charger_id:
                continue
            
            sim = getattr(ch, "charger_simulation", None)
            if sim is None:
                return False
            
            try:
                state = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                ev_id = np.asarray(getattr(sim, "_electric_vehicle_id"))
            except Exception:
                return False
            
            if state.ndim != 1 or t_state >= len(state):
                return False
            
            st = float(state[t_state])
            eid_str = str(ev_id[t_state]).strip()
            if eid_str.startswith("b'") and eid_str.endswith("'"):
                eid_str = eid_str[2:-1]
            
            is_valid = eid_str != "" and eid_str.lower() not in ("nan", "none")
            return (st == 1.0) and is_valid
    
    return False


class SafeRLEvaluator:
    """Complete Safe RL evaluation: policy + oracle + violations + costs"""
    
    def __init__(self, verify_env_vars: bool = True):
        self.verify_env_vars = verify_env_vars
        self.standard_env_vars = {
            "CITYLEARN_EV_DENSE_COST_SCALE": "1.0",
            "CITYLEARN_STEMS_P_GRID_MAX": "27.127751",
            "CITYLEARN_STEMS_P_BUILDING_MAX": "2.273834",
            "CITYLEARN_STEMS_SOC_LOW": "0.0",
            "CITYLEARN_STEMS_SOC_HIGH": "0.95",
        }
    
    def _check_env_vars(self) -> Dict[str, tuple]:
        """Check if current env vars match standard"""
        mismatches = {}
        for key, expected in self.standard_env_vars.items():
            actual = os.environ.get(key, "NOT_SET")
            if actual != expected:
                mismatches[key] = (expected, actual)
        return mismatches
    
    def evaluate(self, agent, seed: int = 42, run_name: str = None) -> Dict[str, Any]:
        """Complete evaluation: policy + oracle + all metrics"""
        if run_name is None:
            run_name = f"EVAL_{agent.name}_seed{seed}"
        
        # Check environment variables
        mismatches = self._check_env_vars()
        if mismatches and self.verify_env_vars:
            print("\n⚠️  WARNING: Non-standard environment variables:")
            for key, (expected, actual) in mismatches.items():
                print(f"  {key}: {actual} (standard: {expected})")
            print("  → Cost values may not be comparable across models!")
            print("  → Use VIOLATION RATES for fair comparison.\n")
        
        # Run policy rollout
        policy_metrics = self._run_policy(agent, seed, run_name)
        
        # Run oracle rollout (EV only)
        oracle_metrics = self._run_oracle(agent, seed, f"{run_name}_ORACLE")
        
        # Compute EV gap
        policy_deficit = policy_metrics["ev_deficit_kwh"]
        oracle_deficit = oracle_metrics["ev_deficit_kwh"]
        avoidable_kwh = max(0.0, policy_deficit - oracle_deficit)
        avoidable_pct = (avoidable_kwh / policy_deficit * 100.0) if policy_deficit > 0 else 0.0
        
        return {
            "agent": agent.name,
            "seed": seed,
            "policy": policy_metrics,
            "oracle": {
                "ev_deficit_kwh": oracle_deficit,
                "avoidable_kwh": avoidable_kwh,
                "avoidable_percent": avoidable_pct,
            },
            "env_vars": {k: os.environ.get(k, "NOT_SET") for k in self.standard_env_vars},
            "warnings": mismatches,
        }
    
    def _run_policy(self, agent, seed: int, run_name: str) -> Dict[str, Any]:
        """Run normal policy rollout with full metrics"""
        os.environ["CITYLEARN_KPI_RUN_NAME"] = run_name
        
        env = make_env()
        obs, info = env.reset(seed=seed)
        agent.reset(env)
        
        steps = 0
        ep_reward = 0.0
        ep_cost = 0.0
        
        # EV tracking
        ev_departures = 0
        ev_violations = 0
        ev_deficit_kwh = 0.0
        
        # Other constraints
        grid_violations = 0
        battery_violations = 0
        building_violations = 0
        
        # Cost breakdown
        cost_ev = 0.0
        cost_grid = 0.0
        cost_battery = 0.0
        cost_building = 0.0
        
        last_info = info
        
        while True:
            action = agent.act(obs, info)
            obs, reward, term, trunc, info = env.step(action)
            
            steps += 1
            ep_reward += reward
            ep_cost += info.get("cost", 0)
            
            # ✅ FIX: Use violation_count from V3 wrapper
            ev_departures += int(info.get("ev_departure_departures", 0))
            ev_violations += int(info.get("ev_departure_violation_count_deficit", 0))
            ev_deficit_kwh += float(info.get("ev_departure_deficit_kwh", 0))
            
            # Grid
            if float(info.get("cost_stems_grid_power", 0)) > 0:
                grid_violations += 1
            
            # Battery & Building
            battery_violations += int(info.get("battery_soc_violation_count", 0))
            building_violations += int(info.get("building_power_violation_count", 0))
            
            # Costs
            cost_ev += float(info.get("cost_ev_departure", 0))
            cost_grid += float(info.get("cost_stems_grid_power", 0))
            cost_battery += float(info.get("cost_stems_battery", 0))
            cost_building += float(info.get("cost_stems_building_power", 0))
            
            last_info = info
            
            if term or trunc:
                break
        
        env.close()
        
        # CityLearn KPIs
        city_kpis = {k: last_info.get(k, np.nan) for k in last_info if str(k).startswith("citylearn_")}
        
        return {
            "steps": steps,
            "reward": ep_reward,
            "cost": ep_cost,
            "violations": {
                "ev": {"count": ev_violations, "total": ev_departures, "rate_%": 100 * ev_violations / max(1, ev_departures)},
                "grid": {"count": grid_violations, "total": steps, "rate_%": 100 * grid_violations / steps},
                "battery": {"count": battery_violations, "total": steps * 17, "rate_%": 100 * battery_violations / (steps * 17)},
                "building": {"count": building_violations, "total": steps * 17, "rate_%": 100 * building_violations / (steps * 17)},
            },
            "cost_breakdown": {
                "ev": cost_ev,
                "grid": cost_grid,
                "battery": cost_battery,
                "building": cost_building,
            },
            "ev_deficit_kwh": ev_deficit_kwh,
            "citylearn_kpis": city_kpis,
        }
    
    def _run_oracle(self, agent, seed: int, run_name: str) -> Dict[str, Any]:
        """Run oracle: force EV=1.0 when connected"""
        os.environ["CITYLEARN_KPI_RUN_NAME"] = run_name
        
        env = make_env()
        obs, info = env.reset(seed=seed)
        agent.reset(env)
        
        # Get EV indices
        ev_indices = env._ev_charger_action_indices if hasattr(env, '_ev_charger_action_indices') else []
        raw_env = env._get_citylearn_env() if hasattr(env, '_get_citylearn_env') else env
        action_names = getattr(raw_env, "action_names", [])
        if isinstance(action_names, list) and len(action_names) == 1:
            action_names = action_names[0]
        
        ev_deficit_kwh = 0.0
        
        while True:
            action = agent.act(obs, info)
            action = np.asarray(action, dtype=float).ravel()
            
            # Force EV=1.0 when connected
            raw_env = env._get_citylearn_env() if hasattr(env, '_get_citylearn_env') else env
            for i in ev_indices:
                if _is_charger_connected(raw_env, i, action_names):
                    action[i] = 1.0
            
            obs, reward, term, trunc, info = env.step(action)
            
            ev_deficit_kwh += float(info.get("ev_departure_deficit_kwh", 0))
            
            if term or trunc:
                break
        
        env.close()
        
        return {"ev_deficit_kwh": ev_deficit_kwh}
