#!/usr/bin/env python3
"""Ablation evaluation: R21 baseline (no mask) vs R21 + C2/C3/C4 mask."""
import argparse, os, sys, numpy as np, torch, torch.nn as nn

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# Common env vars (R21 config)
COMMON_ENV = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    "CITYLEARN_EV_COST_SCALE": "3.0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_KPI_RUN_NAME": "__eval_disabled__",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    # Disable training-only wrappers
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_BATT_CLAMP": "1",
    "CITYLEARN_WM_DISABLE": "1",
    # R21 reward weights (for info logging, not training)
    "STEMS_ALPHA_GRID": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.3",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_XI_RENEWABLE": "0.2",
    "STEMS_BETA_RAMP": "0.0",
    "STEMS_LAMBDA_EV": "5.0",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_EV_GUARD": "1.0",
    "STEMS_ALPHA_V2G_CONTEXT": "3.0",
}

def set_env(mask_on: bool):
    for k, v in COMMON_ENV.items():
        os.environ[k] = v
    os.environ["CITYLEARN_ACTION_MASK"] = "1" if mask_on else "0"

def build_env():
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)
    if os.environ.get("CITYLEARN_ACTION_MASK", "0") == "1":
        from citylearn_safe.action_mask_wrapper import ActionMaskWrapper
        env = ActionMaskWrapper(env)
    return env

def load_policy(ckpt_path, obs_dim, act_dim):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi = ckpt["pi"]
    ckpt_dim = pi["mean.0.weight"].shape[1]
    layers = []
    in_d = ckpt_dim
    for h in [256, 256]:
        layers += [nn.Linear(in_d, h), nn.Tanh()]
        in_d = h
    layers.append(nn.Linear(in_d, act_dim))
    net = nn.Sequential(*layers)
    for i, m in enumerate(net):
        if isinstance(m, nn.Linear):
            m.weight.data = pi[f"mean.{i}.weight"]
            m.bias.data = pi[f"mean.{i}.bias"]
    net.eval()
    obs_mean = ckpt["obs_normalizer"]["_mean"].numpy()
    obs_std = np.clip(ckpt["obs_normalizer"]["_std"].numpy(), 1e-6, None)
    return net, obs_mean, obs_std, ckpt_dim

def run_episode(env, net, obs_mean, obs_std, ckpt_dim):
    from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(raw.buildings)
    names = getattr(raw, "action_names", [[]])[0] if isinstance(getattr(raw, "action_names", []), list) and len(getattr(raw, "action_names", [])) == 1 else list(getattr(raw, "action_names", []))
    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

    obs, _ = env.reset()
    infos, batt_acts, ev_acts, hours_list = [], [], [], []
    for t in range(8760):
        obs_np = np.asarray(obs, dtype=np.float32)
        if ckpt_dim > len(obs_np):
            obs_np = np.concatenate([obs_np, np.ones(ckpt_dim - len(obs_np))])
        obs_n = (obs_np - obs_mean[:ckpt_dim]) / obs_std[:ckpt_dim]
        with torch.no_grad():
            action = net(torch.tensor(obs_n, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(action, -1, 1)
        t_now = int(getattr(raw, "time_step", 0))
        hours_list.append(t_now % 24)
        batt_acts.append([float(action[i]) for i in batt_idx])
        ev_acts.append([float(action[i]) for i in ev_idx])
        obs, r, term, trunc, info = env.step(action)
        infos.append(dict(info))
        if term or trunc: break
    return infos, np.array(batt_acts), np.array(ev_acts), np.array(hours_list), batt_idx, ev_idx, len(buildings)

def report(label, infos, batt_acts, ev_acts, hours, batt_idx, ev_idx, n_bld):
    N = len(infos)
    bld_steps = N * n_bld
    # C0
    ev_dep = sum(int(i.get("ev_departure_departures", 0)) for i in infos)
    ev_def = sum(int(i.get("ev_departure_violation_count_deficit", 0)) for i in infos)
    ev_def80 = sum(int(i.get("ev_departure_violation_count_80pct", 0)) for i in infos)
    # C2
    c2_viols = sum(int(i.get("battery_soc_violation_count", 0)) for i in infos)
    # C3
    c3_viols = sum(int(i.get("building_power_violation_count", 0)) for i in infos)
    c3_any = sum(1 for i in infos if float(i.get("building_power_violation", 0)) > 0.5)
    # C4
    c4_viols = sum(1 for i in infos if float(i.get("cost_stems_grid_power", 0)) > 1e-6)

    mean_batt = batt_acts.mean(axis=1)
    mean_ev = ev_acts.mean(axis=1)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"\n  CONSTRAINTS (proper denominators)")
    print(f"  C0 EV departure:  {ev_def}/{ev_dep} dep = {100*ev_def/max(1,ev_dep):.1f}%")
    print(f"     Severe (>20%): {ev_def80}/{ev_dep} = {100*ev_def80/max(1,ev_dep):.1f}%")
    print(f"  C2 Battery SoC:   {c2_viols}/{bld_steps} bld-steps = {100*c2_viols/max(1,bld_steps):.1f}%")
    print(f"  C3 Building pwr:  {c3_viols}/{bld_steps} bld-steps = {100*c3_viols/max(1,bld_steps):.1f}%")
    print(f"     Any building:  {c3_any}/{N} steps = {100*c3_any/N:.1f}%")
    print(f"  C4 Grid power:    {c4_viols}/{N} steps = {100*c4_viols/N:.1f}%")

    # Per-building battery
    print(f"\n  PER-BUILDING BATTERY")
    for bi, idx in enumerate(batt_idx):
        ba = batt_acts[:, bi]
        d = 100*(ba < -0.01).sum()/N
        c = 100*(ba > 0.01).sum()/N
        print(f"  Bldg {bi}: discharge={d:.1f}% charge={c:.1f}% idle={100-d-c:.1f}% mean={ba.mean():+.4f}")

    # Per-charger EV
    print(f"\n  PER-CHARGER EV")
    for ei, idx in enumerate(ev_idx):
        ea = ev_acts[:, ei]
        d = 100*(ea < -0.01).sum()/N
        c = 100*(ea > 0.01).sum()/N
        print(f"  EV {ei}: V2G={d:.1f}% charge={c:.1f}% idle={100-d-c:.1f}% mean={ea.mean():+.4f}")

    # CityLearn KPIs
    fi = infos[-1]
    print(f"\n  CITYLEARN KPIs")
    for k in ["citylearn_electricity_consumption_total", "citylearn_carbon_emissions_total",
              "citylearn_cost_total", "citylearn_daily_peak_average",
              "citylearn_all_time_peak_average", "citylearn_ramping_average",
              "citylearn_zero_net_energy"]:
        v = fi.get(k, "N/A")
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Hourly profile
    print(f"\n  HOURLY PROFILE (cycling check)")
    hourly_b = np.zeros(24)
    hourly_e = np.zeros(24)
    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            hourly_b[h] = mean_batt[mask].mean()
            hourly_e[h] = mean_ev[mask].mean()
    charge_hrs = sum(1 for h in range(24) if hourly_b[h] > 0.05)
    discharge_hrs = sum(1 for h in range(24) if hourly_b[h] < -0.05)
    print(f"  Batt charge hours: {charge_hrs}/24, discharge hours: {discharge_hrs}/24")
    for h in range(24):
        tag = "CHARGE" if hourly_b[h] > 0.05 else "DISCH" if hourly_b[h] < -0.05 else "idle"
        print(f"  {h:02d}:00 Batt={hourly_b[h]:+.3f} EV={hourly_e[h]:+.3f} [{tag}]")

    # Totals
    total_rew = sum(float(i.get("reward_stems_total", 0)) for i in infos)
    grid_imp = sum(float(i.get("grid_import_kwh", 0)) for i in infos)
    grid_exp = sum(float(i.get("grid_export_kwh", 0)) for i in infos)
    print(f"\n  Total reward: {total_rew:,.1f}")
    print(f"  Grid: import={grid_imp:,.0f} export={grid_exp:,.0f} net={grid_imp-grid_exp:,.0f} kWh")

    return {
        "c0_pct": 100*ev_def/max(1,ev_dep), "c2_pct": 100*c2_viols/max(1,bld_steps),
        "c3_pct": 100*c3_viols/max(1,bld_steps), "c4_pct": 100*c4_viols/N,
        "reward": total_rew, "charge_hrs": charge_hrs, "discharge_hrs": discharge_hrs,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Baseline (no mask)
    set_env(mask_on=False)
    env = build_env()
    obs, _ = env.reset()
    net, om, os_, cd = load_policy(args.baseline, len(obs), env.action_space.shape[0])
    env.close()
    env = build_env()
    infos, ba, ea, hrs, bi, ei, nb = run_episode(env, net, om, os_, cd)
    r_base = report("BASELINE (no mask)", infos, ba, ea, hrs, bi, ei, nb)
    env.close()

    # Treatment (with mask)
    set_env(mask_on=True)
    env = build_env()
    obs, _ = env.reset()
    net, om, os_, cd = load_policy(args.treatment, len(obs), env.action_space.shape[0])
    env.close()
    env = build_env()
    infos, ba, ea, hrs, bi, ei, nb = run_episode(env, net, om, os_, cd)
    r_treat = report("TREATMENT (C2+C3+C4 mask)", infos, ba, ea, hrs, bi, ei, nb)
    env.close()

    # Side-by-side
    print(f"\n{'='*70}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Treatment':>12} {'Delta':>12}")
    print(f"  {'-'*61}")
    for k in ["c0_pct", "c2_pct", "c3_pct", "c4_pct", "reward", "charge_hrs", "discharge_hrs"]:
        b, t = r_base[k], r_treat[k]
        d = t - b
        print(f"  {k:<25} {b:>12.1f} {t:>12.1f} {d:>+12.1f}")

if __name__ == "__main__":
    main()
