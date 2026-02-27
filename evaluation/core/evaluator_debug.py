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
    """Check if EV connected - FIXED TIMING"""
    if charger_idx >= len(action_names):
        return False
    
    name = str(action_names[charger_idx]).strip().lower()
    if "electric_vehicle_storage_charger_" not in name:
        return False
    
    suffix = name.split("electric_vehicle_storage_charger_", 1)[1]
    charger_id = f"charger_{suffix}"
    
    # FIX: Use CURRENT timestep, not +1
    t_state = int(getattr(raw_env, "time_step", 0))
    
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
            connected = (st == 1.0) and is_valid
            
            return connected
    
    return False


def debug_oracle_rbc(seed=42):
    """Debug oracle to see why there are violations"""
    from evaluation.agents.rbc import RBCAgent
    
    env = make_env()
    obs, info = env.reset(seed=seed)
    agent = RBCAgent(ev_mode="greedy")
    agent.reset(env)
    
    ev_indices = env._ev_charger_action_indices if hasattr(env, '_ev_charger_action_indices') else []
    raw_env = env._get_citylearn_env() if hasattr(env, '_get_citylearn_env') else env
    action_names = getattr(raw_env, "action_names", [])
    if isinstance(action_names, list) and len(action_names) == 1:
        action_names = action_names[0]
    
    print(f"\nEV charger indices: {ev_indices}")
    
    step_count = 0
    forced_count = 0
    departure_count = 0
    deficit_count = 0
    
    deficits_log = []
    
    while True:
        # Get RBC action
        rbc_action = agent.act(obs, info)
        action = np.asarray(rbc_action, dtype=float).ravel()
        
        # Check and force EV actions
        raw_env = env._get_citylearn_env() if hasattr(env, '_get_citylearn_env') else env
        for i in ev_indices:
            if _is_charger_connected(raw_env, i, action_names):
                if action[i] < 1.0:
                    print(f"  [Step {step_count}] Charger {i}: RBC={action[i]:.2f}, forcing to 1.0")
                action[i] = 1.0
                forced_count += 1
        
        obs, reward, term, trunc, info = env.step(action)
        step_count += 1
        
        # Check for deficits
        ev_dep = int(info.get("ev_departure_departures", 0))
        ev_def = float(info.get("ev_departure_deficit_kwh", 0))
        
        if ev_dep > 0:
            departure_count += ev_dep
            if ev_def > 0.01:
                deficit_count += 1
                msg = f"[Step {step_count}] ⚠️ DEFICIT: {ev_def:.3f} kWh"
                print(f"  {msg}")
                deficits_log.append((step_count, ev_def))
        
        if term or trunc:
            break
    
    env.close()
    
    print(f"\n" + "="*60)
    print("DEBUG RESULTS:")
    print(f"  Total steps: {step_count}")
    print(f"  EV actions forced to 1.0: {forced_count}")
    print(f"  Total departures: {departure_count}")
    print(f"  Departures with deficit: {deficit_count}")
    print(f"  Violation rate: {100 * deficit_count / max(1, departure_count):.2f}%")
    
    if deficits_log:
        print(f"\nFirst 10 deficits:")
        for step, deficit in deficits_log[:10]:
            print(f"    Step {step}: {deficit:.3f} kWh")
    
    print("="*60)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.getcwd())
    debug_oracle_rbc()
