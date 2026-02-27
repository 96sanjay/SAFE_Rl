from __future__ import annotations

import os
import sys
import argparse
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Ensure repo root is on path
sys.path.insert(0, os.getcwd())

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.adapters import SingleAgentListAdapter

from evalaution.agents.rbc import RBCAgent
from evalaution.agents.omnisafe_gaussian import OmniSafeCheckpointAgent


def make_env() -> Any:
    schema = os.environ.get("CITYLEARN_SCHEMA", "").strip()
    if not schema:
        raise RuntimeError("CITYLEARN_SCHEMA env var is not set.")
    base = CityLearnEnv(schema=schema, central_agent=True)
    env = SingleAgentListAdapter(base)
    env = CityLearnSafetyEnvV3(env)
    return env


def run_one_agent(agent, *, seed: int, run_name: str, max_steps: int | None = None) -> Dict[str, Any]:
    # Force KPI output filenames per agent
    os.environ["CITYLEARN_KPI_RUN_NAME"] = run_name

    env = make_env()
    obs, info = env.reset(seed=seed)
    agent.reset(env)

    done = False
    steps = 0
    ep_reward = 0.0

    # Sums (CMDP + logged-only constraints)
    sum_cost_cmdp = 0.0
    sum_ev_cost = 0.0
    sum_peak_cost = 0.0
    sum_ramp_cost = 0.0

    sum_batt_soc = 0.0
    sum_bld_pow = 0.0

    viol_batt_steps = 0
    viol_batt_frac_sum = 0.0
    viol_bld_steps = 0

    last_info = info

    while not done:
        action = agent.act(obs, info)
        obs, reward, term, trunc, info = env.step(action)

        done = bool(term or trunc)
        last_info = info

        steps += 1
        ep_reward += float(reward)

        sum_cost_cmdp += float(info.get("cost", 0.0))
        sum_ev_cost += float(info.get("cost_ev_departure", 0.0))

        sum_peak_cost += float(info.get("cost_grid_peak", 0.0))
        sum_ramp_cost += float(info.get("cost_grid_ramp", 0.0))

        sum_batt_soc += float(info.get("cost_stems_battery", 0.0))
        sum_bld_pow += float(info.get("cost_stems_building_power", 0.0))
        if "battery_soc_violation_frac" in info:
            viol_batt_frac_sum += float(info["battery_soc_violation_frac"])
        elif "battery_soc_violation_rate_%" in info:
            viol_batt_frac_sum += float(info["battery_soc_violation_rate_%"]) / 100.0
        else:
            viol_batt_frac_sum += float(info.get("battery_soc_violation", 0.0))
        viol_bld_steps += 1 if float(info.get("building_power_violation", 0.0)) > 0 else 0

        if max_steps is not None and steps >= int(max_steps):
            # hard stop for quick smoke tests
            break

    # Collect CityLearn KPIs if present at episode end
    city_kpis = {k: last_info.get(k, np.nan) for k in last_info.keys() if str(k).startswith("citylearn_")}

    out = {
        "agent": getattr(agent, "name", agent.__class__.__name__),
        "seed": seed,
        "steps": steps,
        "episode_reward": ep_reward,

        "cmdp_cost_total": sum_cost_cmdp,
        "ev_cost_total": sum_ev_cost,

        # These may be logged-only now (depending on your current design)
        "grid_peak_cost_sum": sum_peak_cost,
        "grid_ramp_cost_sum": sum_ramp_cost,

        "stems_battery_cost_sum": sum_batt_soc,
        "stems_building_power_cost_sum": sum_bld_pow,

        "battery_violation_rate_%": (viol_batt_frac_sum / max(1, steps)) * 100.0,
        "building_power_violation_rate_%": (viol_bld_steps / max(1, steps)) * 100.0,

        **city_kpis,
    }

    env.close()
    return out


def parse_agents(agent_specs: List[str]):
    agents = []
    for spec in agent_specs:
        # Supported formats:
        #   rbc
        #   omnisafe:/abs/or/rel/path/to/epoch-XXX.pt
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
    p.add_argument("--prefix", type=str, default="EVAL")
    p.add_argument("--max_steps", type=int, default=None, help="Optional quick smoke-test cap.")
    args = p.parse_args()

    agents = parse_agents(args.agents)
    results = []

    for agent in agents:
        run_name = f"{args.prefix}_{agent.name}_seed{args.seed}"
        print(f"\n=== Running {agent.name} (run_name={run_name}) ===")
        res = run_one_agent(agent, seed=args.seed, run_name=run_name, max_steps=args.max_steps)
        results.append(res)

    df = pd.DataFrame(results)

    out_dir = os.path.join("runs", "kpi_logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.prefix}_comparison_seed{args.seed}.csv")
    df.to_csv(out_path, index=False)

    print("\nSaved comparison CSV:", out_path)
    cols_show = [c for c in [
        "agent","steps","episode_reward","cmdp_cost_total","ev_cost_total",
        "battery_violation_rate_%","building_power_violation_rate_%",
        "citylearn_cost_total","citylearn_ramping_average","citylearn_daily_peak_average"
    ] if c in df.columns]
    print(df[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
