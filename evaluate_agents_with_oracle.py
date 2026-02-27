from __future__ import annotations

import os
import sys
import argparse
from typing import Any, Dict, List
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.adapters import SingleAgentListAdapter

from evaluation.agents.rbc import RBCAgent  # ✅ FIXED TYPO
from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent


def make_env() -> Any:
    schema = os.environ.get("CITYLEARN_SCHEMA", "").strip()
    if not schema:
        raise RuntimeError("CITYLEARN_SCHEMA env var is not set.")
    base = CityLearnEnv(schema=schema, central_agent=True)
    
    # Apply NormalizedObservationWrapper (same as training!)
    try:
        from citylearn.wrappers import NormalizedObservationWrapper
        base = NormalizedObservationWrapper(base)
    except Exception:
        pass
    
    env = SingleAgentListAdapter(base)
    env = CityLearnSafetyEnvV3(env)
    return env


def _is_charger_connected(raw_env, charger_idx: int, action_names: List[str]) -> bool:
    """Check if EV is connected to charger at current timestep"""
    if charger_idx >= len(action_names):
        return False
    
    action_name = str(action_names[charger_idx]).strip().lower()
    if "electric_vehicle_storage_charger_" not in action_name:
        return False
    
    suffix = action_name.split("electric_vehicle_storage_charger_", 1)[1]
    charger_id = f"charger_{suffix}"
    
    # Align with wrapper tau indexing
    t_state = int(getattr(raw_env, "time_step", 0)) + 1
    if t_state < 0:
        t_state = 0
    
    for b in getattr(raw_env, "buildings", []) or []:
        for ch in getattr(b, "electric_vehicle_chargers", []) or []:
            cid_raw = getattr(ch, "charger_id", getattr(ch, "name", None))
            if cid_raw is None:
                continue
            
            # Normalize ID
            cid = str(cid_raw).strip()
            if cid.startswith("b'") and cid.endswith("'"):
                cid = cid[2:-1]
            
            if cid != charger_id:
                continue
            
            sim = getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))
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
            eid = ev_id[t_state]
            
            # Validate EV ID
            eid_str = str(eid).strip()
            if eid_str.startswith("b'") and eid_str.endswith("'"):
                eid_str = eid_str[2:-1]
            
            is_valid = eid_str != "" and eid_str.lower() not in ("nan", "none")
            
            return (st == 1.0) and is_valid
    
    return False


def run_policy_rollout(agent, *, seed: int, run_name: str) -> Dict[str, Any]:
    """Run normal policy rollout"""
    os.environ["CITYLEARN_KPI_RUN_NAME"] = run_name
    
    env = make_env()
    obs, info = env.reset(seed=seed)
    agent.reset(env)
    
    done = False
    steps = 0
    ep_reward = 0.0
    
    sum_policy_deficit = 0.0
    policy_departures = 0
    
    last_info = info
    
    while not done:
        action = agent.act(obs, info)
        obs, reward, term, trunc, info = env.step(action)
        
        done = bool(term or trunc)
        last_info = info
        
        steps += 1
        ep_reward += float(reward)
        
        # Accumulate V3 deficit (actual policy actions)
        sum_policy_deficit += float(info.get("ev_departure_deficit_kwh", 0.0))
        policy_departures += int(info.get("ev_departure_departures", 0))
    
    # Collect CityLearn KPIs
    city_kpis = {k: last_info.get(k, np.nan) for k in last_info.keys() if str(k).startswith("citylearn_")}
    
    result = {
        "agent": getattr(agent, "name", agent.__class__.__name__),
        "seed": seed,
        "steps": steps,
        "episode_reward": ep_reward,
        "policy_deficit_kwh": sum_policy_deficit,
        "policy_departures": policy_departures,
        **city_kpis,
    }
    
    env.close()
    return result


def run_oracle_rollout(agent, *, seed: int, run_name: str) -> Dict[str, Any]:
    """Run oracle rollout: force EV actions to 1.0 when connected"""
    os.environ["CITYLEARN_KPI_RUN_NAME"] = run_name
    
    env = make_env()
    obs, info = env.reset(seed=seed)
    agent.reset(env)
    
    # Get EV indices and action names
    ev_indices = env._ev_charger_action_indices if hasattr(env, '_ev_charger_action_indices') else []
    
    raw_env = env._get_citylearn_env() if hasattr(env, '_get_citylearn_env') else env
    action_names = getattr(raw_env, "action_names", [])
    if action_names and isinstance(action_names, list) and len(action_names) == 1:
        action_names = action_names[0]
    
    done = False
    steps = 0
    sum_oracle_deficit = 0.0
    oracle_departures = 0
    
    last_info = info
    
    while not done:
        # Get policy action
        action = agent.act(obs, info)
        action = np.asarray(action, dtype=float).ravel()
        
        # Override: force EV actions to 1.0 when connected
        raw_env = env._get_citylearn_env() if hasattr(env, '_get_citylearn_env') else env
        for i in ev_indices:
            if _is_charger_connected(raw_env, i, action_names):
                action[i] = 1.0  # Force max charge when connected
        
        obs, reward, term, trunc, info = env.step(action)
        
        done = bool(term or trunc)
        last_info = info
        
        steps += 1
        
        # Accumulate oracle deficit
        sum_oracle_deficit += float(info.get("ev_departure_deficit_kwh", 0.0))
        oracle_departures += int(info.get("ev_departure_departures", 0))
    
    result = {
        "oracle_deficit_kwh": sum_oracle_deficit,
        "oracle_departures": oracle_departures,
    }
    
    env.close()
    return result


def parse_agents(agent_specs: List[str]):
    agents = []
    for spec in agent_specs:
        if spec == "rbc":
            agents.append(RBCAgent())
        elif spec.startswith("omnisafe:"):
            path = spec.split(":", 1)[1]
            if not path:
                raise ValueError("Empty omnisafe checkpoint path.")
            agents.append(OmniSafeCheckpointAgent(path))
        else:
            raise ValueError(f"Unknown agent spec: {spec}")
    return agents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", required=True, help="e.g. rbc omnisafe:/path/to/epoch-100.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", type=str, default="ORACLE")
    args = p.parse_args()
    
    agents = parse_agents(args.agents)
    results = []
    
    for agent in agents:
        print(f"\n{'='*60}")
        print(f"Agent: {agent.name}")
        print(f"{'='*60}")
        
        # Rollout 1: Policy
        policy_run_name = f"{args.prefix}_{agent.name}_POLICY_seed{args.seed}"
        print(f"\n[1/2] Running POLICY rollout (run_name={policy_run_name})...")
        policy_result = run_policy_rollout(agent, seed=args.seed, run_name=policy_run_name)
        
        # Rollout 2: Oracle
        oracle_run_name = f"{args.prefix}_{agent.name}_ORACLE_seed{args.seed}"
        print(f"[2/2] Running ORACLE rollout (run_name={oracle_run_name})...")
        oracle_result = run_oracle_rollout(agent, seed=args.seed, run_name=oracle_run_name)
        
        # Merge results
        combined = {**policy_result, **oracle_result}
        
        # Compute gap
        policy_deficit = float(combined["policy_deficit_kwh"])
        oracle_deficit = float(combined["oracle_deficit_kwh"])
        
        avoidable_kwh = max(0.0, policy_deficit - oracle_deficit)
        avoidable_percent = (avoidable_kwh / policy_deficit * 100.0) if policy_deficit > 0 else 0.0
        
        combined["avoidable_wrt_oracle_kwh"] = avoidable_kwh
        combined["avoidable_wrt_oracle_percent"] = avoidable_percent
        
        results.append(combined)
        
        print(f"\n--- Results for {agent.name} ---")
        print(f"Policy deficit:  {policy_deficit:.2f} kWh")
        print(f"Oracle deficit:  {oracle_deficit:.2f} kWh")
        print(f"Avoidable gap:   {avoidable_kwh:.2f} kWh ({avoidable_percent:.1f}%)")
    
    # Save comparison CSV
    df = pd.DataFrame(results)
    
    out_dir = os.path.join("runs", "kpi_logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.prefix}_oracle_comparison_seed{args.seed}.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Saved oracle comparison CSV: {out_path}")
    print(f"{'='*60}\n")
    
    # Display summary
    cols_show = [c for c in [
        "agent", "steps", "episode_reward",
        "policy_deficit_kwh", "oracle_deficit_kwh",
        "avoidable_wrt_oracle_kwh", "avoidable_wrt_oracle_percent",
        "citylearn_cost_total", "citylearn_ramping_average"
    ] if c in df.columns]
    
    print(df[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
