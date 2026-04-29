#!/usr/bin/env python3
"""
Evaluate R5a (MLP) and R8 (STEMS) trained agents with full KPI metrics.
Compares against Intelligent RBC baseline.

Handles both MLP and STEMS checkpoint architectures correctly,
including ForecastObsWrapper for obs_dim=198.

Usage:
  python scripts/evaluate_r8_comparison.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# ---- Environment variables (same as training) ----
os.environ["CITYLEARN_SCHEMA"] = f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_EXPORT_FACTOR"] = "0.7"
os.environ["CITYLEARN_C3_CONTROLLABLE"] = "0"
os.environ["CITYLEARN_SPATIAL_OBS"] = "0"
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "10.2352"
os.environ["CITYLEARN_STEMS_SOC_LOW"] = "0.0"
os.environ["CITYLEARN_STEMS_SOC_HIGH"] = "0.95"
os.environ["CITYLEARN_STEMS_PNORM_P"] = "4.0"
os.environ["CITYLEARN_W_COST_EV"] = "1.0"
os.environ["CITYLEARN_W_COST_SOC"] = "10.0"
os.environ["CITYLEARN_W_COST_BUILDING"] = "0.5"
os.environ["CITYLEARN_W_COST_GRID"] = "0.05"
os.environ["CITYLEARN_EV_COST_SCALE"] = "3.0"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_full"
os.environ["CITYLEARN_INCLUDE_EV_COST"] = "1"

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.schema_index import build_index
from citylearn_safe.stems_encoder_5bld import STEMSEncoder5Bld, build_node_indices
import citylearn_safe.schema_index as si


# ---- Actor classes ----
class MLPActor(nn.Module):
    """Standard OmniSafe MLP actor."""
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.mean = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.tanh(self.mean(obs))


class STEMSActor(nn.Module):
    """STEMS GCN-Transformer actor (matches train_stems_5bld.py STEMSMeanNet)."""
    def __init__(self, encoder, act_dim):
        super().__init__()
        self.encoder = encoder
        self.action_head = nn.Sequential(
            nn.Linear(encoder.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, obs):
        features = self.encoder(obs)
        return torch.tanh(self.action_head(features))


def load_actor(checkpoint_path, obs_dim, act_dim, actor_type="mlp", node_info=None, num_buildings=5):
    """Load actor from checkpoint. Returns (actor, obs_normalizer)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    pi_state = ckpt["pi"]

    if actor_type == "stems":
        encoder = STEMSEncoder5Bld(
            obs_dim=obs_dim,
            node_info=node_info,
            num_buildings=num_buildings,
            hidden_dim=64,
            global_hidden=32,
            num_gcn_layers=3,
            num_heads=4,
            output_dim=256,
            dropout=0.1,
        )
        actor = STEMSActor(encoder, act_dim)
        # Wrap with 'mean' prefix to match checkpoint keys
        wrapper = nn.Module()
        wrapper.mean = actor
        filtered = {k: v for k, v in pi_state.items() if not k.startswith("log_std")}
        missing, unexpected = wrapper.load_state_dict(filtered, strict=False)
        if missing:
            print(f"  WARNING missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  WARNING unexpected keys: {unexpected[:5]}...")
        actor = wrapper.mean
    else:
        # Infer hidden sizes
        h1 = pi_state["mean.0.weight"].shape[0]
        h2 = pi_state["mean.2.weight"].shape[0]
        actor = MLPActor(obs_dim, act_dim, hidden_sizes=(h1, h2))
        filtered = {k: v for k, v in pi_state.items() if not k.startswith("log_std")}
        actor.load_state_dict(filtered, strict=False)

    actor.eval()

    # Obs normalizer
    obs_mean = obs_std = None
    if "obs_normalizer" in ckpt and isinstance(ckpt["obs_normalizer"], dict):
        norm = ckpt["obs_normalizer"]
        if "_mean" in norm and "_std" in norm:
            obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
            obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)

    return actor, obs_mean, obs_std


def evaluate_agent(actor, obs_mean, obs_std, env, label="Agent", seed=42):
    """Run 1 full episode and collect all metrics."""
    obs, info = env.reset(seed=seed)
    done = False
    steps = 0

    total_reward = 0.0
    total_cost = 0.0
    rewards = []
    costs = []
    cost_components = {"c1_ev": [], "c1d_ev_dense": [], "c2_soc": [], "c3_building": [], "c4_grid": []}
    actions_all = []

    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if obs_mean is not None:
            obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
        with torch.no_grad():
            action_t = actor(obs_t).squeeze(0).numpy()
        action = np.clip(action_t, -1.0, 1.0)
        actions_all.append(action.copy())

        obs, reward, term, trunc, info = env.step(action)
        done = bool(term) or bool(trunc)

        steps += 1
        r = float(reward)
        c = float(info.get("cost", 0.0))
        total_reward += r
        total_cost += c
        rewards.append(r)
        costs.append(c)

        # Extract cost components if available
        for key, store_key in [("cost_ev_departure", "c1_ev"),
                                ("cost_ev_dense", "c1d_ev_dense"),
                                ("cost_stems_battery", "c2_soc"),
                                ("cost_stems_building_power", "c3_building"),
                                ("cost_stems_grid_power", "c4_grid")]:
            cost_components[store_key].append(float(info.get(key, 0.0)))

        if steps % 2000 == 0:
            print(f"  [{label}] step {steps}/8759, cum_reward={total_reward:.0f}, cum_cost={total_cost:.0f}")

    # CityLearn evaluate()
    citylearn_kpis = {}
    try:
        raw_env = env
        for _ in range(20):
            if hasattr(raw_env, 'buildings') and len(getattr(raw_env, 'buildings', [])) > 0:
                break
            raw_env = getattr(raw_env, 'env', getattr(raw_env, 'base', getattr(raw_env, 'unwrapped', None)))
            if raw_env is None:
                break

        if raw_env and hasattr(raw_env, 'evaluate'):
            eval_df = raw_env.evaluate()
            if eval_df is not None and not eval_df.empty:
                district = eval_df[eval_df["name"] == "District"] if "name" in eval_df.columns else eval_df
                for _, row in district.iterrows():
                    cf = row.get("cost_function", "")
                    val = row.get("value", np.nan)
                    if pd.notna(val) and cf:
                        citylearn_kpis[cf] = float(val)
    except Exception as e:
        print(f"  [{label}] CityLearn evaluate() failed: {e}")

    # Action statistics
    actions_arr = np.array(actions_all)

    results = {
        "label": label,
        "steps": steps,
        "total_reward": total_reward,
        "avg_reward": total_reward / steps,
        "total_cost": total_cost,
        "avg_cost": total_cost / steps,
        "cost_c1_ev": sum(cost_components["c1_ev"]),
        "cost_c1d_ev_dense": sum(cost_components["c1d_ev_dense"]),
        "cost_c2_soc": sum(cost_components["c2_soc"]),
        "cost_c3_building": sum(cost_components["c3_building"]),
        "cost_c4_grid": sum(cost_components["c4_grid"]),
        "cost_positive_steps": sum(1 for c in costs if c > 0),
        "cost_positive_pct": 100 * sum(1 for c in costs if c > 0) / steps,
        "action_mean": float(actions_arr.mean()),
        "action_std": float(actions_arr.std()),
        "action_abs_mean": float(np.abs(actions_arr).mean()),
        **{f"citylearn_{k}": v for k, v in citylearn_kpis.items()},
    }

    return results


def print_comparison(results_list, rbc_path=None):
    """Print formatted comparison table."""
    rbc = None
    if rbc_path and os.path.exists(rbc_path):
        with open(rbc_path) as f:
            rbc = json.load(f)

    print("\n" + "=" * 100)
    print("FULL EVALUATION COMPARISON")
    print("=" * 100)

    # Header
    agents = [r["label"] for r in results_list]
    if rbc:
        agents.append("Intelligent RBC")

    header = f"{'Metric':<45}" + "".join(f"{a:>18}" for a in agents)
    print(header)
    print("-" * len(header))

    def row(metric, key, rbc_key=None, fmt=".1f", higher_better=None):
        vals = []
        for r in results_list:
            v = r.get(key, float('nan'))
            vals.append(v)
        if rbc and rbc_key:
            vals.append(rbc.get(rbc_key, float('nan')))
        elif rbc:
            vals.append(float('nan'))

        val_strs = [f"{v:{fmt}}" if not np.isnan(v) else "N/A" for v in vals]

        # Mark best
        valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
        if valid and higher_better is not None:
            if higher_better:
                best_i = max(valid, key=lambda x: x[1])[0]
            else:
                best_i = min(valid, key=lambda x: x[1])[0]
            val_strs[best_i] = f"*{val_strs[best_i]}*"

        line = f"{metric:<45}" + "".join(f"{s:>18}" for s in val_strs)
        print(line)

    print("\n--- Reward & Cost ---")
    row("Total Reward", "total_reward", "total_reward", ".0f", higher_better=True)
    row("Avg Reward/Step", "avg_reward", "avg_reward_per_step", ".3f", higher_better=True)
    row("Total CMDP Cost", "total_cost", "cmdp_cost_total", ".0f", higher_better=False)
    row("Avg Cost/Step", "avg_cost", "cmdp_cost_per_step", ".4f", higher_better=False)
    row("Cost>0 Steps (%)", "cost_positive_pct", None, ".1f", higher_better=False)

    print("\n--- Cost Breakdown (weighted) ---")
    row("C1: EV Departure (w=1.0)", "cost_c1_ev", "cmdp_cost_ev_departure", ".1f", higher_better=False)
    row("C1d: EV Dense (w=5.0)", "cost_c1d_ev_dense", None, ".1f", higher_better=False)
    row("C2: Battery SoC (w=10.0)", "cost_c2_soc", "cmdp_cost_building_soc", ".1f", higher_better=False)
    row("C3: Building Power (w=0.5)", "cost_c3_building", None, ".1f", higher_better=False)
    row("C4: Grid Power (w=0.05)", "cost_c4_grid", None, ".1f", higher_better=False)

    print("\n--- CityLearn KPIs ---")
    row("Electricity Cost ($)", "citylearn_cost_total", "electricity_cost_total", ".0f", higher_better=False)
    row("Electricity Consumption", "citylearn_electricity_consumption_total", None, ".4f", higher_better=False)
    row("Carbon Emissions", "citylearn_carbon_emissions_total", None, ".4f")
    row("Daily Peak Average", "citylearn_daily_peak_average", None, ".4f", higher_better=False)
    row("All-Time Peak Average", "citylearn_all_time_peak_average", None, ".4f", higher_better=False)
    row("Ramping Average", "citylearn_ramping_average", None, ".4f", higher_better=False)
    row("Zero Net Energy", "citylearn_zero_net_energy", None, ".4f", higher_better=True)

    print("\n--- Action Statistics ---")
    row("Action Mean", "action_mean", None, ".4f")
    row("Action Std", "action_std", None, ".4f")
    row("Action |Mean|", "action_abs_mean", None, ".4f")

    # EV-specific
    print("\n--- EV Metrics ---")
    if rbc:
        rbc_ev = rbc.get("ev_deficit_total_kwh", float('nan'))
        rbc_avoid = rbc.get("ev_deficit_avoidable_kwh", float('nan'))
        print(f"{'RBC EV Deficit (total kWh)':<45}{rbc_ev:>18.1f}")
        print(f"{'RBC EV Deficit (avoidable kWh)':<45}{rbc_avoid:>18.1f}")
        print(f"{'RBC SoC Violation %':<45}{rbc.get('soc_violation_pct', 0):>18.1f}%")

    print("\n" + "=" * 100)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build ObsIndex
    print("=== Building environment and ObsIndex ===")
    si._CACHE = None
    base_env = make_base_env(central_agent=True)
    safety_env = CityLearnSafetyEnvV3(base_env)

    # Count buildings
    city = base_env
    for _ in range(20):
        if hasattr(city, 'buildings') and len(getattr(city, 'buildings', [])) > 0:
            break
        city = getattr(city, 'env', getattr(city, 'base', getattr(city, 'unwrapped', None)))
        if city is None:
            break
    num_buildings = len(city.buildings) if city else 5

    obs_index = build_index(safety_env, expected_buildings=num_buildings)
    node_info = build_node_indices(obs_index, num_buildings)

    # Create eval env with forecast wrapper (same as training)
    forecast_env = ForecastObsWrapper(safety_env, forecast_horizon=24)
    obs_space = forecast_env.observation_space
    obs_dim = int(obs_space.shape[0]) if not isinstance(obs_space, (list, tuple)) else int(obs_space[0].shape[0])
    act_space = forecast_env.action_space
    act_dim = int(act_space.shape[0]) if not isinstance(act_space, (list, tuple)) else int(act_space[0].shape[0])
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, buildings={num_buildings}")

    # Checkpoints
    r5a_ckpt = f"{PROJECT}/runs/r6_compare/r5a_5bld/PPOLag-{{CityLearnSafety-V2G-v2}}/seed-000-2026-03-06-11-04-24/torch_save/epoch-50.pt"
    r8_ckpt = f"{PROJECT}/runs/r8_stems/r8_5bld/PPOLag-{{CityLearnSafety-V2G-v2}}/seed-000-2026-03-07-05-06-59/torch_save/epoch-50.pt"
    rbc_path = f"{PROJECT}/runs/baselines/evaluation/intelligent-rbc_eval.json"

    results = []

    # ---- Evaluate R5a ----
    print("\n=== Evaluating R5a (MLP baseline) ===")
    actor_r5a, norm_mean_r5a, norm_std_r5a = load_actor(r5a_ckpt, obs_dim, act_dim, "mlp")

    # Need fresh env for each evaluation
    si._CACHE = None
    base1 = make_base_env(central_agent=True)
    env1 = ForecastObsWrapper(CityLearnSafetyEnvV3(base1), forecast_horizon=24)
    r5a_results = evaluate_agent(actor_r5a, norm_mean_r5a, norm_std_r5a, env1, "R5a (MLP)", seed=42)
    results.append(r5a_results)

    # ---- Evaluate R8 ----
    print("\n=== Evaluating R8 (STEMS GCN-Transformer) ===")
    actor_r8, norm_mean_r8, norm_std_r8 = load_actor(
        r8_ckpt, obs_dim, act_dim, "stems",
        node_info=node_info, num_buildings=num_buildings
    )

    si._CACHE = None
    base2 = make_base_env(central_agent=True)
    env2 = ForecastObsWrapper(CityLearnSafetyEnvV3(base2), forecast_horizon=24)
    r8_results = evaluate_agent(actor_r8, norm_mean_r8, norm_std_r8, env2, "R8 (STEMS)", seed=42)
    results.append(r8_results)

    # Print comparison
    print_comparison(results, rbc_path)

    # Save results
    out_dir = f"{PROJECT}/runs/r8_stems/evaluation"
    os.makedirs(out_dir, exist_ok=True)

    results_json = {r["label"]: r for r in results}
    with open(f"{out_dir}/evaluation_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir}/evaluation_results.json")

    # ---- Plot cost breakdown comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Cost breakdown bar chart
    ax = axes[0]
    components = ["C1\nEV Depart", "C1d\nEV Dense", "C2\nSoC", "C3\nBuilding", "C4\nGrid"]
    keys = ["cost_c1_ev", "cost_c1d_ev_dense", "cost_c2_soc", "cost_c3_building", "cost_c4_grid"]
    x = np.arange(len(components))
    width = 0.35
    for i, r in enumerate(results):
        vals = [r.get(k, 0) for k in keys]
        ax.bar(x + i*width, vals, width, label=r["label"], alpha=0.8)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(components, fontsize=9)
    ax.set_ylabel("Total Cost")
    ax.set_title("CMDP Cost Breakdown")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: CityLearn KPIs comparison
    ax = axes[1]
    cl_metrics = [
        ("Elec Cost", "citylearn_cost_total"),
        ("Daily Peak", "citylearn_daily_peak_average"),
        ("Ramping", "citylearn_ramping_average"),
        ("Zero Net E", "citylearn_zero_net_energy"),
    ]
    x = np.arange(len(cl_metrics))
    for i, r in enumerate(results):
        vals = [r.get(k, 0) for _, k in cl_metrics]
        # Normalize to R5a for comparison
        ax.bar(x + i*width, vals, width, label=r["label"], alpha=0.8)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([m[0] for m in cl_metrics], fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("CityLearn KPIs")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Reward vs Cost scatter
    ax = axes[2]
    for r in results:
        ax.scatter(r["total_cost"], r["total_reward"], s=200, label=r["label"],
                   edgecolors='black', linewidth=1.5, zorder=5)
    if os.path.exists(rbc_path):
        with open(rbc_path) as f:
            rbc = json.load(f)
        ax.scatter(rbc["cmdp_cost_total"], rbc["total_reward"], s=200,
                   label="Intelligent RBC", marker='D', edgecolors='black', zorder=5)
    ax.set_xlabel("Total CMDP Cost (Lower = Safer)")
    ax.set_ylabel("Total Reward (Higher = Better)")
    ax.set_title("Reward vs Cost Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("R5a MLP vs R8 STEMS: Full Evaluation Comparison", fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{out_dir}/evaluation_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_dir}/evaluation_comparison.png")


if __name__ == "__main__":
    main()
