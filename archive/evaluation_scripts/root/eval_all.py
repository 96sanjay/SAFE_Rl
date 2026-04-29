"""
eval_all.py — Evaluate RBC, TRPOLag, CPO, PPOLag + zero-action
All using same env (SoC-v0), same STEMS reward, same cost weights.
"""
import os, sys, json
import numpy as np
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
sys.path.insert(0, os.getcwd())

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn_safe.adapters import SingleAgentListAdapter
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent
from evaluation.agents.rbc import RBCAgent

schema = os.environ.get("CITYLEARN_SCHEMA", "")
assert schema and os.path.exists(schema), "Run: source set_env_v2g_psf.sh first"

AGENTS = {
    "Zero-Action": None,
    "RBC": "rbc",
    "TRPOLag": "runs/trpolag_v2g_stems/TRPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-23-16-07-24/torch_save/epoch-100.pt",
    "CPO": "runs/cpo_v2g_stems/CPO-{CityLearnSafety-SoC-v0}/seed-000-2026-02-23-16-08-15/torch_save/epoch-100.pt",
    "PPOLag": "runs/ppolag_v2g_stems/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-23-19-08-07/torch_save/epoch-100.pt",
}

def make_env():
    base = CityLearnEnv(schema=schema, central_agent=True)
    base = NormalizedObservationWrapper(base)
    base = SingleAgentListAdapter(base)
    return CityLearnSafetyEnvV3(
        base,
        soc_min=float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0")),
        soc_max=float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95")),
    )

def evaluate(name, agent_spec):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"{'='*60}")

    env = make_env()
    obs, info = env.reset()
    obs_np = np.asarray(obs, dtype=np.float32).ravel()
    act_dim = 26

    # Build agent
    if agent_spec is None:
        agent = None  # zero-action
    elif agent_spec == "rbc":
        agent = RBCAgent(ev_mode="greedy")
        agent.reset(env)
    else:
        agent = OmniSafeCheckpointAgent(agent_spec, name=name)
        agent.reset(env)

    # Accumulators
    total_reward = 0.0
    total_cost = 0.0
    c1_avoidable_viol = 0
    c1_unavoidable_viol = 0
    c1_total_deps = 0
    c2_sum = 0.0
    c3_sum = 0.0
    c4_sum = 0.0
    n = 0

    # Per-step for CityLearn KPIs
    rewards = []

    for step in range(8759):
        if agent is None:
            action = np.zeros(act_dim, dtype=np.float32)
        else:
            action = agent.act(obs_np, info)

        obs, reward, term, trunc, info = env.step(action)
        obs_np = np.asarray(obs, dtype=np.float32).ravel()

        total_reward += float(reward)
        total_cost += float(info.get("cost", 0))
        rewards.append(float(reward))

        # C1: avoidable only
        deps = int(info.get("ev_departure_departures", 0))
        if deps > 0:
            c1_total_deps += deps
            avoid = float(info.get("ev_avoidable_deficit_kwh", 0))
            unavoid = float(info.get("ev_unavoidable_deficit_kwh", 0))
            if avoid > 0.01:
                c1_avoidable_viol += deps
            if unavoid > 0.01:
                c1_unavoidable_viol += 1

        # C2: mean battery SoC violation fraction
        c2_sum += float(info.get("battery_soc_violation_frac", 0))

        # C3: building power violations (per building per step)
        c3_sum += float(info.get("building_power_violation_count", 0))

        # C4: grid power violation
        c4_sum += float(info.get("grid_power_violation", 0))

        n += 1

        if step % 2000 == 0:
            print(f"  step {step}/{8759} | reward={total_reward:.0f} | cost={total_cost:.0f}")

        if term or trunc:
            break

    # Compute percentages
    c1_pct = (c1_avoidable_viol / max(1, c1_total_deps)) * 100
    c2_pct = (c2_sum / max(1, n)) * 100
    c3_pct = (c3_sum / (17.0 * max(1, n))) * 100
    c4_pct = (c4_sum / max(1, n)) * 100

    # CityLearn KPIs from last info
    cl_kpis = {k: info.get(k, float('nan')) for k in [
        "citylearn_cost_total", "citylearn_carbon_emissions_total",
        "citylearn_daily_peak_average", "citylearn_ramping_average",
        "citylearn_electricity_consumption_total", "citylearn_discomfort_proportion",
    ]}

    result = {
        "method": name,
        "reward": total_reward,
        "cost": total_cost,
        "c1_avoidable_pct": c1_pct,
        "c1_departures": c1_total_deps,
        "c1_avoidable_viol": c1_avoidable_viol,
        "c2_pct": c2_pct,
        "c3_pct": c3_pct,
        "c4_pct": c4_pct,
    }
    result.update(cl_kpis)

    print(f"\n  {'='*40}")
    print(f"  {name} RESULTS:")
    print(f"    Reward:              {total_reward:.1f}")
    print(f"    Total Cost:          {total_cost:.1f}")
    print(f"    C1 EV (avoidable):   {c1_pct:.2f}% ({c1_avoidable_viol}/{c1_total_deps})")
    print(f"    C2 Battery SoC:      {c2_pct:.2f}%")
    print(f"    C3 Building Power:   {c3_pct:.2f}%")
    print(f"    C4 Grid Power:       {c4_pct:.2f}%")
    print(f"  {'='*40}")

    env.close()
    return result

# ═══════════════ RUN ALL ═══════════════
all_results = []
for name, spec in AGENTS.items():
    try:
        r = evaluate(name, spec)
        all_results.append(r)
    except Exception as e:
        print(f"\n  ERROR evaluating {name}: {e}")
        import traceback; traceback.print_exc()

# ═══════════════ FINAL TABLE ═══════════════
print(f"\n{'='*90}")
print(f"  FINAL EVALUATION TABLE (STEMS reward, SoC-v0 env, epoch-100)")
print(f"{'='*90}")
print(f"  {'Method':<15} {'Reward':>10} {'Cost':>10} {'C1%':>8} {'C2%':>8} {'C3%':>8} {'C4%':>8}")
print(f"  {'-'*75}")
for r in all_results:
    print(f"  {r['method']:<15} {r['reward']:>10.1f} {r['cost']:>10.1f} "
          f"{r['c1_avoidable_pct']:>7.2f}% {r['c2_pct']:>7.2f}% {r['c3_pct']:>7.2f}% {r['c4_pct']:>7.2f}%")
print(f"{'='*90}")

# Save as JSON
with open("runs/plots/eval_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: runs/plots/eval_results.json")
