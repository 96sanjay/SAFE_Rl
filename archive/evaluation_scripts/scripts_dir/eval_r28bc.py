#!/usr/bin/env python3
"""Evaluation of R28b and R28c (100-epoch PPOLag runs).

R28b: 9-term reward, feasible limits (C3=18000, C4=13000), Saute OFF
R28c: R28b + r_load_shift=0.5 (battery price arbitrage)

Usage:
    cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
    conda activate citylearn
    python scripts/eval_r28bc.py [--runs R28b R28c NoControl]
"""
import os, sys, warnings, json, argparse
import numpy as np
warnings.filterwarnings('ignore')

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import torch

# ── Base env vars shared by R28b and R28c ──
BASE_ENV = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    "CITYLEARN_PID_LAGRANGE": "1",
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_KPI_RUN_NAME": "__eval__",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_BATT_CLAMP": "0",
    "CITYLEARN_ACTION_MASK": "0",
    "CITYLEARN_POLICY_ACTION_MASK": "0",
    # EV reward
    "STEMS_LAMBDA_EV": "2.0",
    "STEMS_ALPHA_EV_SMART": "1.5",
    "STEMS_EV_SLACK_ARB_SCALE": "1.0",
    "STEMS_ALPHA_V2G_CONTEXT": "1.5",
    # Battery
    "STEMS_ALPHA_BARRIER": "0.5",
    # Grid/Building
    "STEMS_ALPHA_GRID": "0.5",
    "STEMS_SG_THRESHOLD": "0.5",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BUILD": "0.3",
    "STEMS_SB_ASYMMETRIC": "1",
    # Grid quality
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_XI_RENEWABLE": "0.2",
    # Disabled
    "STEMS_ALPHA_EV_GUARD": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.0",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "STEMS_ALPHA_LOAD_SHIFT": "0.0",
    "STEMS_ALPHA_EV_SOLAR": "0.0",
    "STEMS_ALPHA_SOLAR_STORE": "0.0",
    "STEMS_SOLAR_STORE_BATT_ONLY": "0",
    "STEMS_ALPHA_HEADROOM": "0.0",
    # Cost weights
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    "CITYLEARN_EV_COST_SCALE": "3.0",
}

R28B_CKPT = f"{PROJECT}/runs/r28b/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-04-15-09-32-30/torch_save/epoch-100.pt"
R28C_CKPT = f"{PROJECT}/runs/r28c/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-04-15-10-33-26/torch_save/epoch-100.pt"

RUN_CONFIGS = {
    "R28b": {
        "checkpoint": R28B_CKPT,
        "env_overrides": {},
        "desc": "9-term reward, feasible limits (C3=18k, C4=13k)",
    },
    "R28c": {
        "checkpoint": R28C_CKPT,
        "env_overrides": {"STEMS_ALPHA_LOAD_SHIFT": "0.5"},
        "desc": "R28b + battery price arbitrage (r_load_shift=0.5)",
    },
    "NoControl": {
        "checkpoint": None,
        "env_overrides": {},
        "desc": "Zero-action baseline",
    },
}

P_BMAX = 4.6083
P_GMAX = 10.2352


def make_env():
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)
    return env, safety


class MLPActor(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, h))
            layers.append(torch.nn.Tanh())
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, act_dim))
        self.mean = torch.nn.Sequential(*layers)

    def forward(self, obs):
        return torch.tanh(self.mean(obs))


def load_policy(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    pi_state = ckpt.get("pi", {})
    if not pi_state:
        return None, None, None, None
    h1 = pi_state["mean.0.weight"].shape[0]
    h2 = pi_state["mean.2.weight"].shape[0]
    obs_dim = pi_state["mean.0.weight"].shape[1]
    act_dim = pi_state["mean.4.weight"].shape[0]
    actor = MLPActor(obs_dim, act_dim, (h1, h2))
    filtered = {k: v for k, v in pi_state.items() if not k.startswith("log_std")}
    actor.load_state_dict(filtered, strict=False)
    actor.eval()
    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        obs_clip = float(torch.as_tensor(norm["_clip"]).float().mean())
    return actor, obs_mean, obs_std, obs_clip


def evaluate_run(run_name, config):
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {run_name}  —  {config.get('desc','')}")
    print(f"{'='*60}")

    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k)
    os.environ.update(BASE_ENV)
    os.environ.update(config["env_overrides"])

    env, safety = make_env()
    from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(raw.buildings)
    n_b = len(buildings)

    actor = obs_mean = obs_std = obs_clip = None
    ckpt_path = config["checkpoint"]
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            actor, obs_mean, obs_std, obs_clip = load_policy(ckpt_path)
            print(f"  Loaded: {os.path.basename(ckpt_path)}")
            print(f"  Obs normalizer: {'YES' if obs_mean is not None else 'NO'}")
        except Exception as e:
            print(f"  WARNING: {e}")
            import traceback; traceback.print_exc()
    elif ckpt_path:
        print(f"  WARNING: checkpoint not found: {ckpt_path}")

    obs, _ = env.reset(seed=42)

    total_steps = 0
    total_reward = 0.0
    ev_departures = ev_violated_departures = 0
    ev_deficit_kwh = 0.0
    c2_violations = c2_total_checks = 0
    c3_per_building = [0] * n_b
    c3_total_per_building = [0] * n_b
    c4_violations = 0
    v2g_discharge_steps = 0
    solar_charge_steps = 0
    hourly_charge = [[] for _ in range(24)]
    hourly_discharge = [[] for _ in range(24)]
    nec_history = []
    nec_baseline_history = []
    ev_tracker = {}

    done = False
    while not done:
        if actor is not None:
            obs_flat = np.asarray(obs).ravel()
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32).unsqueeze(0)
            if obs_mean is not None:
                obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
                if obs_clip is not None:
                    obs_t = obs_t.clamp(-obs_clip, obs_clip)
            with torch.no_grad():
                action = actor(obs_t).squeeze(0).numpy()
            action = np.clip(action, -1.0, 1.0)
        else:
            action = np.zeros(9)

        action[2] = 0.0  # disable washing machine

        hour = total_steps % 24
        t_now = int(getattr(raw, 'time_step', 0))

        # Track EV departures BEFORE step
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, 'electric_vehicle_chargers', None) or []
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                sim = getattr(ch, 'charger_simulation', getattr(ch, '_Charger__charger_simulation', None))
                if sim is None:
                    continue
                try:
                    sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                    connected = t_now < len(sa) and float(sa[t_now]) == 1.0
                    if connected:
                        ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                        rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                        ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                        soc = 0.0
                        if ev_obj and ev_obj.battery:
                            soc_arr = np.asarray(ev_obj.battery.soc, dtype=float)
                            t_idx = max(0, t_now - 1)
                            soc = float(np.clip(soc_arr[t_idx], 0, 1)) if 0 <= t_idx < len(soc_arr) else 0.0
                        ev_tracker[key] = {'was': True, 'soc': soc, 'req': rs}
                    else:
                        prev = ev_tracker.get(key, {})
                        if prev.get('was', False):
                            ev_departures += 1
                            deficit = max(0.0, prev['req'] - prev['soc'])
                            if deficit > 0.01:
                                ev_violated_departures += 1
                                ev_deficit_kwh += deficit
                        ev_tracker[key] = {'was': False}
                except:
                    pass

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        total_steps += 1
        total_reward += float(reward)

        t_idx = max(0, int(getattr(raw, 'time_step', 0)) - 1)

        # C2: Battery SoC
        for bld in buildings:
            es = getattr(bld, 'electrical_storage', None)
            if es and hasattr(es, 'soc') and len(es.soc) > t_idx:
                soc = float(es.soc[t_idx])
                c2_total_checks += 1
                if soc < 0.0 or soc > 0.95:
                    c2_violations += 1

        # C3: Per-building power
        district_nec = 0.0
        district_baseline = 0.0
        for b_idx, bld in enumerate(buildings):
            nec = getattr(bld, 'net_electricity_consumption', None)
            if nec is not None and hasattr(nec, '__len__') and len(nec) > t_idx:
                p_i = float(nec[t_idx])
                district_nec += p_i
                c3_total_per_building[b_idx] += 1
                if abs(p_i) > P_BMAX:
                    c3_per_building[b_idx] += 1
            try:
                nsl = getattr(bld, '_Building__energy_to_non_shiftable_load', [])
                sg = getattr(bld, '_Building__solar_generation', [])
                base_nec = 0.0
                if len(nsl) > t_idx:
                    base_nec += float(nsl[t_idx])
                if len(sg) > t_idx:
                    base_nec += float(sg[t_idx])
                district_baseline += base_nec
            except:
                pass

        nec_history.append(district_nec)
        nec_baseline_history.append(district_baseline)

        # C4: Grid power
        if max(0.0, district_nec) > P_GMAX:
            c4_violations += 1

        # V2G tracking
        ev_indices = [1, 6, 8]
        for ei in ev_indices:
            if ei < len(action) and action[ei] < -0.05:
                v2g_discharge_steps += 1

        # Solar charge tracking
        total_solar = sum(abs(float(getattr(b, 'solar_generation', [0])[t_idx]))
                         for b in buildings if hasattr(getattr(b, 'solar_generation', None), '__len__')
                         and len(b.solar_generation) > t_idx)
        if total_solar > 1.0:
            for ei in ev_indices:
                if ei < len(action) and action[ei] > 0.05:
                    solar_charge_steps += 1
                    break

        batt_indices = [0, 3, 4, 5, 7]
        mean_batt = np.mean([action[i] for i in batt_indices if i < len(action)])
        if mean_batt > 0.05:
            hourly_charge[hour].append(mean_batt)
        elif mean_batt < -0.05:
            hourly_discharge[hour].append(abs(mean_batt))

    # ── Compute KPIs ──
    nec_arr = np.array(nec_history)
    base_arr = np.array(nec_baseline_history)

    agent_import = np.sum(np.maximum(0, nec_arr))
    base_import = np.sum(np.maximum(0, base_arr))
    kpi_consumption = agent_import / max(base_import, 1e-6)

    agent_ramp = np.sum(np.maximum(0, np.diff(nec_arr)))
    base_ramp = np.sum(np.maximum(0, np.diff(base_arr)))
    kpi_ramping = agent_ramp / max(base_ramp, 1e-6)

    n_days = total_steps // 24
    agent_peaks = [np.max(nec_arr[d*24:(d+1)*24]) for d in range(n_days)]
    base_peaks = [np.max(base_arr[d*24:(d+1)*24]) for d in range(n_days)]
    kpi_daily_peak = np.mean(agent_peaks) / max(np.mean(base_peaks), 1e-6)

    solar_score = sum(1 for h in range(7, 18) if len(hourly_charge[h]) > 0) / 11.0
    v2g_score = sum(1 for h in range(19, 24) if len(hourly_discharge[h]) > 0) / 5.0
    offpeak_charge = sum(len(hourly_charge[h]) for h in range(7))
    total_charge = sum(len(hourly_charge[h]) for h in range(24))
    offpeak_score = offpeak_charge / max(total_charge, 1)
    ev_compliance = 1.0 - (ev_violated_departures / max(ev_departures, 1))

    c3_total = sum(c3_per_building)
    c3_total_checks = sum(c3_total_per_building)

    print(f"\n  CONSTRAINT VIOLATIONS:")
    print(f"    C0 (EV departure): {ev_violated_departures}/{ev_departures} = {100*ev_violated_departures/max(ev_departures,1):.1f}%  deficit={ev_deficit_kwh:.2f} kWh")
    print(f"    C2 (battery SoC):  {c2_violations}/{c2_total_checks} = {100*c2_violations/max(c2_total_checks,1):.1f}%")
    print(f"    C3 (building power, per-building):")
    for b_idx in range(n_b):
        pct = 100 * c3_per_building[b_idx] / max(c3_total_per_building[b_idx], 1)
        print(f"       Building_{b_idx+1}: {c3_per_building[b_idx]}/{c3_total_per_building[b_idx]} = {pct:.1f}%")
    print(f"       TOTAL: {c3_total}/{c3_total_checks} = {100*c3_total/max(c3_total_checks,1):.1f}%")
    print(f"    C4 (grid power):   {c4_violations}/{total_steps} = {100*c4_violations/max(total_steps,1):.1f}%")

    print(f"\n  CITYLEARN KPIs (< 1.0 = better than no-control):")
    print(f"    Electricity consumption: {kpi_consumption:.3f}")
    print(f"    Ramping:                 {kpi_ramping:.3f}")
    print(f"    Daily peak:              {kpi_daily_peak:.3f}")

    print(f"\n  INTELLIGENCE MATRIX:")
    print(f"    Solar charging:    {solar_score:.2f}")
    print(f"    V2G timing:        {v2g_score:.2f}")
    print(f"    Off-peak charging: {offpeak_score:.2f}")
    print(f"    EV compliance:     {ev_compliance:.2f}")

    print(f"\n  SUMMARY:")
    print(f"    Total reward: {total_reward:.0f}")
    print(f"    V2G discharge steps: {v2g_discharge_steps}")
    print(f"    Solar charge steps:  {solar_charge_steps}")

    return {
        'run': run_name,
        'desc': config.get('desc', ''),
        'reward': total_reward,
        'c0_violations': ev_violated_departures,
        'c0_departures': ev_departures,
        'c0_pct': 100*ev_violated_departures/max(ev_departures, 1),
        'c0_deficit_kwh': ev_deficit_kwh,
        'c2_pct': 100*c2_violations/max(c2_total_checks, 1),
        'c3_per_building': [100*c3_per_building[i]/max(c3_total_per_building[i], 1) for i in range(n_b)],
        'c3_total_pct': 100*c3_total/max(c3_total_checks, 1),
        'c4_pct': 100*c4_violations/max(total_steps, 1),
        'kpi_consumption': kpi_consumption,
        'kpi_ramping': kpi_ramping,
        'kpi_daily_peak': kpi_daily_peak,
        'solar_score': solar_score,
        'v2g_score': v2g_score,
        'offpeak_score': offpeak_score,
        'ev_compliance': ev_compliance,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', default=['R28b', 'R28c'])
    parser.add_argument('--save', default='eval_results/batch_5bld/r28bc_ep100.json')
    args = parser.parse_args()

    results = []
    for run_name in args.runs:
        if run_name not in RUN_CONFIGS:
            print(f"Unknown run: {run_name}. Available: {list(RUN_CONFIGS.keys())}")
            continue
        try:
            r = evaluate_run(run_name, RUN_CONFIGS[run_name])
            results.append(r)
        except Exception as e:
            print(f"ERROR evaluating {run_name}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*110}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*110}")
    print(f"{'Run':>10s} | {'Reward':>8s} | {'C0%':>6s} | {'C2%':>6s} | {'C3%':>6s} | {'C4%':>6s} | {'KPI_elec':>8s} | {'KPI_ramp':>8s} | {'KPI_peak':>8s} | {'Solar':>5s} | {'V2G':>5s} | {'EV_comp':>7s}")
    print("-" * 110)
    for r in results:
        print(f"{r['run']:>10s} | {r['reward']:>+8.0f} | {r['c0_pct']:>5.1f}% | {r['c2_pct']:>5.1f}% | {r['c3_total_pct']:>5.1f}% | {r['c4_pct']:>5.1f}% | {r['kpi_consumption']:>8.3f} | {r['kpi_ramping']:>8.3f} | {r['kpi_daily_peak']:>8.3f} | {r['solar_score']:>5.2f} | {r['v2g_score']:>5.2f} | {r['ev_compliance']:>7.2f}")

    if results and args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {args.save}")
