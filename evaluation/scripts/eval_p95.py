
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.getcwd())

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.adapters import SingleAgentListAdapter

from evaluation.agents.rbc import RBCAgent

# ---------------------------------------------------------------------
# Configuration: set p95 thresholds and SOC bounds before running.
# You can also export these in your shell before launching the script.
# ---------------------------------------------------------------------
os.environ["CITYLEARN_STEMS_P_GRID_MAX"]     = os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "27.127751")
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "2.273834")
os.environ["CITYLEARN_STEMS_SOC_LOW"]        = os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0")
os.environ["CITYLEARN_STEMS_SOC_HIGH"]       = os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95")
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = os.environ.get("CITYLEARN_EV_MISSING_ACTION_MODE", "assume_zero")

# Schema must point to your dataset (adjust this path if needed).
schema = os.environ.get("CITYLEARN_SCHEMA", "")
if not schema:
    raise RuntimeError("CITYLEARN_SCHEMA env var must be set")

def make_env() -> CityLearnSafetyEnvV3:
    base = CityLearnEnv(schema=schema, central_agent=True)
    # Apply normalization wrapper if you used it during training
    try:
        from citylearn.wrappers import NormalizedObservationWrapper
        base = NormalizedObservationWrapper(base)
    except Exception:
        pass
    env = SingleAgentListAdapter(base)
    env = CityLearnSafetyEnvV3(env)
    return env

def roll(agent: RBCAgent, *, oracle: bool = False) -> dict:
    env = make_env()
    obs, info = env.reset(seed=42)
    agent.reset(env)

    # For the oracle run we force missing EV actions to be treated as full charge
    if oracle:
        os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_full"
    else:
        os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"

    # Identify EV charger action indices once (from the wrapper)
    ev_indices = getattr(env, "_ev_charger_action_indices", [])
    raw_env   = env._get_citylearn_env() if hasattr(env, "_get_citylearn_env") else env
    action_names = getattr(raw_env, "action_names", [])
    if action_names and isinstance(action_names, list) and len(action_names) == 1:
        action_names = action_names[0]

    def is_charger_connected(idx: int) -> bool:
        # Align with wrapper tau indexing (t_state = time_step + 1)
        cur_env = env._get_citylearn_env() if hasattr(env, "_get_citylearn_env") else env
        t_state = int(getattr(cur_env, "time_step", 0)) + 1
        if idx >= len(action_names): 
            return False
        name = str(action_names[idx]).strip().lower()
        if "electric_vehicle_storage_charger_" not in name:
            return False
        suffix    = name.split("electric_vehicle_storage_charger_", 1)[1]
        charger_id = f"charger_{suffix}"
        for b in getattr(cur_env, "buildings", []) or []:
            for ch in getattr(b, "electric_vehicle_chargers", []) or []:
                cid = str(getattr(ch, "charger_id", getattr(ch, "name", ""))).strip()
                if cid.startswith("b'") and cid.endswith("'"):
                    cid = cid[2:-1]
                if cid != charger_id:
                    continue
                sim = getattr(ch, "charger_simulation",
                              getattr(ch, "_Charger__charger_simulation", None))
                if sim is None:
                    return False
                try:
                    state  = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                    ev_id  = np.asarray(getattr(sim, "_electric_vehicle_id"))
                except Exception:
                    return False
                if state.ndim != 1 or t_state >= len(state):
                    return False
                st  = float(state[t_state])
                eid = ev_id[t_state]
                eid_s = str(eid).strip()
                if eid_s.startswith("b'") and eid_s.endswith("'"):
                    eid_s = eid_s[2:-1]
                return (st == 1.0) and (eid_s != "" and eid_s.lower() not in ("nan", "none"))
        return False

    steps = 0
    n_bld = len(getattr(raw_env, "buildings", []) or [])

    # Accumulators for the 4 CMDP constraints
    ev_viol = 0
    ev_dep  = 0
    grid_viol_steps = 0
    soc_viol_bld   = 0
    soc_denom      = 0
    bld_viol_bld   = 0
    bld_denom      = 0

    last_info = info

    while True:
        action = agent.act(obs, info)
        action = np.asarray(action, dtype=float).ravel()

        # In oracle mode, force EV chargers to 1.0 when connected
        if oracle:
            for idx in ev_indices:
                if is_charger_connected(idx):
                    action[idx] = 1.0

        obs, reward, term, trunc, info = env.step(action)
        last_info = info
        steps += 1

        # EV constraint: count violations per departure
        ev_dep  += int(info.get("ev_departure_departures", 0))
        ev_viol += int(info.get("ev_departure_violation_count", 0))

        # Grid constraint: use cost_stems_grid_power (already thresholded by P_GRID_MAX)
        if float(info.get("cost_stems_grid_power", 0.0)) > 0.0:
            grid_viol_steps += 1

        # Battery SOC constraint: per-building violation count
        soc_viol_bld += int(info.get("battery_soc_violation_count", 0))
        soc_denom    += n_bld

        # Building power constraint: per-building violation count
        bld_viol_bld += int(info.get("building_power_violation_count", 0))
        bld_denom    += n_bld

        if term or trunc:
            break

    env.close()

    return {
        "steps": steps,
        "buildings": n_bld,
        "ev_viol": ev_viol,
        "ev_dep": ev_dep,
        "grid_viol_steps": grid_viol_steps,
        "soc_viol_bld": soc_viol_bld,
        "soc_denom": soc_denom,
        "bld_viol_bld": bld_viol_bld,
        "bld_denom": bld_denom,
        "city_kpis": {k: last_info.get(k, np.nan) for k in last_info if str(k).startswith("citylearn_")},
    }

def pct(n: int, d: int) -> float:
    return 100.0 * (n / d) if d > 0 else 0.0

if __name__ == "__main__":
    agent = RBCAgent(ev_mode="greedy")

    # Policy run (greedy EV)
    policy = roll(agent, oracle=False)

    # Oracle run (EV chargers forced to 1.0; missing actions interpreted as full)
    oracle = roll(agent, oracle=True)

    print("\n================== EVALUATION WITH P95 THRESHOLDS ==================")
    print(f"Grid threshold (P_GRID_MAX)     : {os.environ['CITYLEARN_STEMS_P_GRID_MAX']}")
    print(f"Building threshold (P_BUILD_MAX): {os.environ['CITYLEARN_STEMS_P_BUILDING_MAX']}")
    print(f"Steps: {policy['steps']} | Buildings: {policy['buildings']}\n")

    print("---- Policy rollout (greedy RBC) ----")
    print(f"EV departures: {policy['ev_dep']}")
    print(f"EV violations: {policy['ev_viol']}  "
          f"({pct(policy['ev_viol'], policy['ev_dep']):.2f}% per-departure)")
    print(f"Grid violations: {policy['grid_viol_steps']}  "
          f"({pct(policy['grid_viol_steps'], policy['steps']):.2f}% per-step)")
    print(f"Battery SOC violations: {policy['soc_viol_bld']}/{policy['soc_denom']}  "
          f"({pct(policy['soc_viol_bld'], policy['soc_denom']):.2f}% per-bld-step)")
    print(f"Building power violations: {policy['bld_viol_bld']}/{policy['bld_denom']}  "
          f"({pct(policy['bld_viol_bld'], policy['bld_denom']):.2f}% per-bld-step)")

    print("\n---- Oracle EV rollout ----")
    print(f"EV departures: {oracle['ev_dep']}")
    print(f"EV violations: {oracle['ev_viol']}  "
          f"({pct(oracle['ev_viol'], oracle['ev_dep']):.2f}% per-departure)")

    # Compute avoidable deficit gap (optional)
    avoidable = policy['ev_viol'] - oracle['ev_viol']
    avoidable_pct = pct(avoidable, policy['ev_dep'])
    print(f"Avoidable EV violations: {avoidable}  "
          f"({avoidable_pct:.2f}% of departures)\n")

    # CityLearn KPIs (from last step of policy run)
    print("---- CityLearn KPIs (last info) ----")
    for key, val in sorted(policy["city_kpis"].items()):
        print(f"{key}: {val}")
