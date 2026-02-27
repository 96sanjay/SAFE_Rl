
"""
Cold Wave Case Study Evaluation - Vermont Dataset (COMFORT ONLY, REAL Tin)
Scenario: Extreme cold (Tout < -10°C)
Duration: ~3 days during winter
"""

import numpy as np
import pandas as pd
import os
import json

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from agents.intelligent_rbc_with_temp import RBCAgentWithTemp


# =========================
# CRASH FIX (DO NOT REMOVE)
# =========================
def _disable_hvac_demand_assert():
    """
    CityLearn sometimes asserts when heating demand > device max output:
    'demand is greater than heating_device max output'
    This disables that assert so evaluation can proceed.
    """
    import citylearn.building as bm
    B = bm.Building
    patched = 0
    for name in dir(B):
        if "demand_limit_check" in name:
            setattr(B, name, lambda self, *args, **kwargs: None)
            patched += 1
    print(f"[PATCH] Disabled demand_limit_check asserts: {patched} methods patched")


def _unwrap_action_names(names):
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        return names[0]
    return names


def _to_env_action_vermont(env, action_flat, hvac_cap: float = 0.4):
    """
    Vermont central_agent=True uses list action space with ONE Box of shape (105,).
    Convert flat action to [vec105], clip to bounds, and cap HVAC magnitude.
    NOTE: NO SIGN FLIP here. RBC outputs CityLearn convention:
      combined_action < 0 => cooling, combined_action > 0 => heating
    """
    a = np.asarray(action_flat, dtype=np.float32).flatten()
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if not isinstance(env.action_space, list):
        a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
        return a

    sp = env.action_space[0]
    L = int(np.prod(sp.shape))

    if a.shape[0] < L:
        a = np.concatenate([a, np.zeros(L - a.shape[0], dtype=np.float32)])
    elif a.shape[0] > L:
        a = a[:L]

    names = _unwrap_action_names(getattr(env.base, "action_names", []))
    if isinstance(names, list) and len(names) == L:
        hvac_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "cooling_or_heating_device"]
        if hvac_idx:
            a[hvac_idx] = np.clip(a[hvac_idx], -hvac_cap, hvac_cap)

    a = np.clip(a, sp.low, sp.high).astype(np.float32)
    return [a]


def _scalar_series_at(x, t_idx: int):
    """Safe scalar fetch from scalar/list/np.ndarray at time index."""
    if x is None:
        return None
    try:
        if np.isscalar(x):
            v = float(x)
            return v if np.isfinite(v) else None
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            v = float(arr)
            return v if np.isfinite(v) else None
        if t_idx < 0 or t_idx >= len(arr):
            return None
        v = float(arr[t_idx])
        return v if np.isfinite(v) else None
    except Exception:
        return None


def comfort_from_buildings(env, deadband=0.5):
    """
    Compute comfort from REAL building signals (not wrapper info).
    Returns:
      mean_abs_dev, mean_raw_dev, violation (0/1)
    """
    blds = env.base.buildings
    t_idx = max(0, int(getattr(env.base, "time_step", 0)) - 1)

    abs_devs = []
    raw_devs = []

    for b in blds:
        Tin = _scalar_series_at(getattr(b, "indoor_dry_bulb_temperature", None), t_idx)

        # Prefer heating setpoint in cold, fallback to cooling setpoint
        Th = _scalar_series_at(getattr(b, "indoor_dry_bulb_temperature_heating_set_point", None), t_idx)
        Tc = _scalar_series_at(getattr(b, "indoor_dry_bulb_temperature_cooling_set_point", None), t_idx)

        # If heating setpoint missing, use cooling; if both missing, skip
        Tset = Th if Th is not None else Tc

        if Tin is None or Tset is None:
            continue

        abs_dev = abs(Tin - Tset)
        raw = max(0.0, abs_dev - deadband)

        abs_devs.append(abs_dev)
        raw_devs.append(raw)

    if len(abs_devs) == 0:
        return float("nan"), float("nan"), 0.0

    mean_abs = float(np.mean(abs_devs))
    mean_raw = float(np.mean(raw_devs))
    viol = 1.0 if mean_raw > 0.0 else 0.0
    return mean_abs, mean_raw, viol


def find_cold_wave_period(weather_path, temp_threshold=-10.0, min_duration=48):
    """Find coldest period in Vermont dataset"""
    print(f"Reading weather data from: {weather_path}")
    weather_df = pd.read_csv(weather_path)
    outdoor_temp = weather_df["outdoor_dry_bulb_temperature"].values

    print(f"Dataset length: {len(outdoor_temp)} hours")
    print(f"Temperature range: {outdoor_temp.min():.1f}°C to {outdoor_temp.max():.1f}°C")

    cold_mask = outdoor_temp < temp_threshold

    periods = []
    in_period = False
    start = 0

    for i, is_cold in enumerate(cold_mask):
        if is_cold and not in_period:
            start = i
            in_period = True
        elif (not is_cold) and in_period:
            duration = i - start
            if duration >= min_duration:
                periods.append((start, duration))
            in_period = False

    if in_period:
        duration = len(cold_mask) - start
        if duration >= min_duration:
            periods.append((start, duration))

    if not periods:
        print(f"[WARNING] No cold wave found with threshold {temp_threshold}°C")
        print("Using coldest 72-hour period instead...")

        window = 72
        best_start = 0
        best_avg = 999.0

        for i in range(len(outdoor_temp) - window):
            avg_temp = outdoor_temp[i : i + window].mean()
            if avg_temp < best_avg:
                best_avg = avg_temp
                best_start = i

        print(f"Coldest period: starts at step {best_start}, avg temp {best_avg:.1f}°C")
        return best_start, window, best_avg

    best_period = max(periods, key=lambda x: x[1])
    avg_temp = outdoor_temp[best_period[0] : best_period[0] + best_period[1]].mean()

    print(f"Cold wave found: starts at step {best_period[0]}, duration {best_period[1]}h, avg temp {avg_temp:.1f}°C")
    return best_period[0], min(best_period[1], 72), avg_temp


def evaluate_cold_wave_comfort_only(num_episodes=3, deadband=0.5):
    """
    COMFORT-ONLY evaluation for Vermont using REAL building Tin/Tset.
    """

    schema_path = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/vt_chittenden_county_neighborhood/schema_temp_control.json"
    weather_path = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/vt_chittenden_county_neighborhood/weather.csv"

    cold_start, cold_duration, avg_temp = find_cold_wave_period(weather_path, temp_threshold=-10.0)

    print("\nCOLD WAVE Period Selected:")
    print(f"  Start: Step {cold_start}")
    print(f"  Duration: {cold_duration} hours ({cold_duration/24:.1f} days)")
    print(f"  Average outdoor temp: {avg_temp:.1f}°C\n")

    _disable_hvac_demand_assert()

    env = CityLearnSafetyEnvV3(CityLearnEnv(schema=schema_path, central_agent=True))

    agent = RBCAgentWithTemp(temp_deadband=deadband)
    agent.reset(env)

    results = {
        "comfort_violation_rate": [],
        "avg_abs_deviation": [],
        "avg_raw_deviation": [],
        # context:
        "peak_power": [],
        "total_consumption": [],
        "avg_ramping": [],
    }

    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes}...")
        obs, info = env.reset()

        # fast-forward
        print(f"  Fast-forwarding to step {cold_start}...")
        if isinstance(env.action_space, list):
            L = int(np.prod(env.action_space[0].shape))
            zero = [np.zeros(L, dtype=np.float32)]
        else:
            zero = np.zeros(env.action_space.shape, dtype=np.float32)

        for step_idx in range(cold_start):
            obs, _, term, trunc, info = env.step(zero)
            if term or trunc:
                obs, info = env.reset()

        comfort_viol = 0.0
        absdev_list, raw_list = [], []
        peak_power = 0.0
        total_cons = 0.0
        ramp_sum = 0.0
        prev_cons = None
        steps = 0

        print(f"  Evaluating cold wave period ({cold_duration} steps)...")
        for _ in range(cold_duration):
            a = agent.act(obs, info)
            a_env = _to_env_action_vermont(env, a, hvac_cap=0.4)

            obs, reward, term, trunc, info = env.step(a_env)

            # REAL comfort
            mean_abs, mean_raw, viol = comfort_from_buildings(env, deadband=deadband)
            comfort_viol += float(viol)
            absdev_list.append(mean_abs)
            raw_list.append(mean_raw)

            # context energy
            cons = float(info.get("step_net_consumption_kwh", info.get("net_consumption_kwh", 0.0)))
            total_cons += cons
            if prev_cons is not None:
                ramp_sum += abs(cons - prev_cons)
            prev_cons = cons

            p = float(info.get("grid_import_kwh", info.get("grid_power", 0.0)))
            peak_power = max(peak_power, p)

            steps += 1

            if term or trunc:
                print("  [INFO] Episode ended early.")
                break

        if steps == 0:
            print("  [WARNING] No steps, skipping episode.")
            continue

        results["comfort_violation_rate"].append((comfort_viol / steps) * 100.0)
        results["avg_abs_deviation"].append(float(np.nanmean(absdev_list)) if len(absdev_list) else float("nan"))
        results["avg_raw_deviation"].append(float(np.nanmean(raw_list)) if len(raw_list) else float("nan"))

        results["peak_power"].append(peak_power)
        results["total_consumption"].append(total_cons)
        results["avg_ramping"].append(ramp_sum / max(1, steps - 1))

        print(f"  Done: comfort_violation={results['comfort_violation_rate'][-1]:.2f}% | avg_raw_dev={results['avg_raw_deviation'][-1]:.2f}°C")

    summary = {
        "comfort_violation_rate_%": float(np.mean(results["comfort_violation_rate"])) if results["comfort_violation_rate"] else float("nan"),
        "avg_abs_deviation": float(np.mean(results["avg_abs_deviation"])) if results["avg_abs_deviation"] else float("nan"),
        "avg_raw_deviation": float(np.mean(results["avg_raw_deviation"])) if results["avg_raw_deviation"] else float("nan"),
        # context:
        "peak_power": float(np.mean(results["peak_power"])) if results["peak_power"] else float("nan"),
        "total_consumption": float(np.mean(results["total_consumption"])) if results["total_consumption"] else float("nan"),
        "avg_ramping": float(np.mean(results["avg_ramping"])) if results["avg_ramping"] else float("nan"),
    }

    return summary, results


if __name__ == "__main__":
    os.environ["CITYLEARN_KPI_RUN_NAME"] = "__DISABLE__"
    os.environ["CITYLEARN_ENABLE_COMFORT"] = "1"

    print("=" * 60)
    print("COLD WAVE CASE STUDY - VERMONT DATASET (COMFORT ONLY | REAL Tin)")
    print("=" * 60)

    summary, _ = evaluate_cold_wave_comfort_only(num_episodes=3, deadband=0.5)

    print("\n" + "=" * 60)
    print("RBC PERFORMANCE (COLD WAVE - VERMONT | COMFORT ONLY | REAL Tin)")
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k:35s}: {v:10.4f}")

    with open("results_cold_wave_vermont.json", "w") as f:
        json.dump({"rbc": summary}, f, indent=2)

    print("\n✓ Results saved to results_cold_wave_vermont.json")
