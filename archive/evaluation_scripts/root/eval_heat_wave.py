"""
Heat Wave Case Study Evaluation
Dataset: 2023 LSTM (temperature control)
Scenario: Extreme heat (Tout > 28°C)
Duration: ~3 days during peak summer
"""

import numpy as np
import pandas as pd
import json
import os
from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from agents.intelligent_rbc_with_temp import RBCAgentWithTemp

def find_heat_wave_period(weather_path, temp_threshold=28.0, min_duration=48):
    """
    Find heat wave period in dataset
    
    Args:
        weather_path: Path to weather.csv file
        temp_threshold: Minimum outdoor temp for heat wave (°C)
        min_duration: Minimum consecutive hours
    
    Returns:
        (start_idx, duration, avg_temp) of longest heat wave period
    """
    
    print(f"Reading weather data from: {weather_path}")
    
    # Read weather data
    weather_df = pd.read_csv(weather_path)
    outdoor_temp = weather_df['outdoor_dry_bulb_temperature'].values
    
    print(f"Dataset length: {len(outdoor_temp)} hours")
    print(f"Temperature range: {outdoor_temp.min():.1f}°C to {outdoor_temp.max():.1f}°C")
    
    # Find consecutive hot periods
    hot_mask = outdoor_temp > temp_threshold
    
    # Find runs of True values
    periods = []
    in_period = False
    start = 0
    
    for i, is_hot in enumerate(hot_mask):
        if is_hot and not in_period:
            start = i
            in_period = True
        elif not is_hot and in_period:
            duration = i - start
            if duration >= min_duration:
                periods.append((start, duration))
            in_period = False
    
    # Check if period extends to end
    if in_period:
        duration = len(hot_mask) - start
        if duration >= min_duration:
            periods.append((start, duration))
    
    if not periods:
        print(f"[WARNING] No heat wave found with threshold {temp_threshold}°C")
        print(f"Using warmest 72-hour period instead...")
        
        # Find warmest 72-hour period
        window = 72
        best_start = 0
        best_avg = -999
        for i in range(len(outdoor_temp) - window):
            avg_temp = outdoor_temp[i:i+window].mean()
            if avg_temp > best_avg:
                best_avg = avg_temp
                best_start = i
        
        print(f"Warmest period: starts at step {best_start}, avg temp {best_avg:.1f}°C")
        return best_start, window, best_avg
    
    # Return longest period
    best_period = max(periods, key=lambda x: x[1])
    avg_temp = outdoor_temp[best_period[0]:best_period[0]+best_period[1]].mean()
    
    print(f"Heat wave found: starts at step {best_period[0]}, duration {best_period[1]}h, avg temp {avg_temp:.1f}°C")
    
    return best_period[0], min(best_period[1], 72), avg_temp  # Cap at 3 days

def find_cold_wave_period(weather_path, temp_threshold=5.0, min_duration=48):
    """
    Find cold wave period in dataset
    
    Args:
        weather_path: Path to weather.csv file
        temp_threshold: Maximum outdoor temp for cold wave (°C)
        min_duration: Minimum consecutive hours
    
    Returns:
        (start_idx, duration, avg_temp) of longest cold wave period
    """
    
    print(f"Reading weather data from: {weather_path}")
    
    weather_df = pd.read_csv(weather_path)
    outdoor_temp = weather_df['outdoor_dry_bulb_temperature'].values
    
    print(f"Temperature range: {outdoor_temp.min():.1f}°C to {outdoor_temp.max():.1f}°C")
    
    # Find consecutive cold periods
    cold_mask = outdoor_temp < temp_threshold
    
    periods = []
    in_period = False
    start = 0
    
    for i, is_cold in enumerate(cold_mask):
        if is_cold and not in_period:
            start = i
            in_period = True
        elif not is_cold and in_period:
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
        print(f"Using coldest 72-hour period instead...")
        
        window = 72
        best_start = 0
        best_avg = 999
        for i in range(len(outdoor_temp) - window):
            avg_temp = outdoor_temp[i:i+window].mean()
            if avg_temp < best_avg:
                best_avg = avg_temp
                best_start = i
        
        print(f"Coldest period: starts at step {best_start}, avg temp {best_avg:.1f}°C")
        return best_start, window, best_avg
    
    # Return longest period
    best_period = max(periods, key=lambda x: x[1])
    avg_temp = outdoor_temp[best_period[0]:best_period[0]+best_period[1]].mean()
    
    print(f"Cold wave found: starts at step {best_period[0]}, duration {best_period[1]}h, avg temp {avg_temp:.1f}°C")
    
    return best_period[0], min(best_period[1], 72), avg_temp

def evaluate_scenario(scenario_type="heat_wave", agent_type="rbc", num_episodes=1):
    """Evaluate agent during extreme weather scenario"""
    
    schema_path = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/citylearn_challenge_2023_phase_2_online_evaluation_3/schema.json"
    weather_path = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/citylearn_challenge_2023_phase_2_online_evaluation_3/weather.csv"
    
    # Find extreme weather period
    if scenario_type == "heat_wave":
        period_start, period_duration, avg_temp = find_heat_wave_period(weather_path, temp_threshold=25.0)
    else:  # cold_wave
        period_start, period_duration, avg_temp = find_cold_wave_period(weather_path, temp_threshold=10.0)
    
    print(f"\n{scenario_type.upper().replace('_', ' ')} Period Selected:")
    print(f"  Start: Step {period_start}")
    print(f"  Duration: {period_duration} hours ({period_duration/24:.1f} days)")
    print(f"  Average outdoor temp: {avg_temp:.1f}°C\n")
    
    env = CityLearnSafetyEnvV3(
        CityLearnEnv(schema=schema_path, central_agent=True)
    )
    
    # Initialize agent
    if agent_type == "rbc":
        agent = RBCAgentWithTemp(temp_deadband=0.5)
        agent.reset(env)
    else:
        raise NotImplementedError(f"Agent type {agent_type} not implemented yet")
    
    results = {
        "cost": [],
        "emission": [],
        "peak": [],
        "consumption": [],
        "ramping": [],
        "comfort_violation_rate": [],
        "safety_violations": [],
        "tin_values": [],
        "tset_values": [],
        "comfort_cost_raw": [],
    }
    
    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes}...")
        obs, info = env.reset()
        
        # Fast-forward to period
        print(f"  Fast-forwarding to step {period_start}...")
        for step_idx in range(period_start):
            if isinstance(env.action_space, list):
                action = [np.zeros(sp.shape) for sp in env.action_space]
            else:
                action = np.zeros(env.action_space.shape)
            
            obs, _, term, trunc, info = env.step(action)
            
            if term or trunc:
                print(f"  [WARNING] Episode ended at step {step_idx}, resetting...")
                obs, info = env.reset()
        
        # Evaluate during extreme weather
        episode_data = {
            "total_cost": 0.0,
            "total_emission": 0.0,
            "peak_power": 0.0,
            "total_consumption": 0.0,
            "ramping_sum": 0.0,
            "comfort_violations": 0,
            "safety_violations": 0,
            "steps": 0,
            "tin_list": [],
            "tset_list": [],
            "comfort_raw_list": [],
        }
        
        prev_consumption = None
        
        print(f"  Evaluating {scenario_type} period ({period_duration} steps)...")
        for step in range(period_duration):
            action = agent.act(obs, info)
            obs, reward, term, trunc, info = env.step(action)
            
            # Accumulate metrics
            episode_data["total_cost"] += info.get("step_cost", 0.0)
            episode_data["total_emission"] += info.get("step_carbon_kg", 0.0)
            episode_data["total_consumption"] += info.get("step_net_consumption_kwh", 0.0)
            
            current_power = info.get("grid_import_kwh", 0.0)
            episode_data["peak_power"] = max(episode_data["peak_power"], current_power)
            
            if prev_consumption is not None:
                ramp = abs(info.get("step_net_consumption_kwh", 0.0) - prev_consumption)
                episode_data["ramping_sum"] += ramp
            prev_consumption = info.get("step_net_consumption_kwh", 0.0)
            
            # Comfort tracking
            if info.get("comfort_enabled", 0.0) > 0.5:
                episode_data["comfort_violations"] += info.get("comfort_violation", 0.0)
                episode_data["tin_list"].append(info.get("comfort_tin", float("nan")))
                episode_data["tset_list"].append(info.get("comfort_tset", float("nan")))
                episode_data["comfort_raw_list"].append(info.get("cost_comfort_raw", 0.0))
            
            # Count any constraint violation
            has_violation = (
                info.get("battery_soc_violation_any", 0.0) > 0 or
                info.get("building_power_violation", 0.0) > 0 or
                info.get("grid_power_violation", 0.0) > 0 or
                info.get("comfort_violation", 0.0) > 0
            )
            episode_data["safety_violations"] += float(has_violation)
            
            episode_data["steps"] += 1
            
            if term or trunc:
                print(f"  [INFO] Episode ended early at step {step}")
                break
        
        # Normalize metrics
        steps = episode_data["steps"]
        if steps == 0:
            print("  [WARNING] No steps recorded, skipping episode")
            continue
        
        print(f"  Completed {steps} steps")
            
        results["cost"].append(episode_data["total_cost"])
        results["emission"].append(episode_data["total_emission"])
        results["peak"].append(episode_data["peak_power"])
        results["consumption"].append(episode_data["total_consumption"])
        results["ramping"].append(episode_data["ramping_sum"] / max(1, steps - 1))
        
        comfort_viol_rate = (episode_data["comfort_violations"] / steps) * 100.0
        results["comfort_violation_rate"].append(comfort_viol_rate)
        
        safety_viol_rate = (episode_data["safety_violations"] / steps) * 100.0
        results["safety_violations"].append(safety_viol_rate)
        
        results["tin_values"].append(episode_data["tin_list"])
        results["tset_values"].append(episode_data["tset_list"])
        results["comfort_cost_raw"].append(episode_data["comfort_raw_list"])
    
    # Compute averages
    summary = {
        "cost": np.mean(results["cost"]),
        "emission": np.mean(results["emission"]),
        "peak": np.mean(results["peak"]),
        "consumption": np.mean(results["consumption"]),
        "ramping": np.mean(results["ramping"]),
        "comfort_violation_rate_%": np.mean(results["comfort_violation_rate"]),
        "safety_violations_%": np.mean(results["safety_violations"]),
        "avg_tin": np.mean([np.nanmean(t) for t in results["tin_values"] if len(t) > 0]),
        "avg_comfort_deviation": np.mean([np.nanmean(c) for c in results["comfort_cost_raw"] if len(c) > 0]),
    }
    
    return summary, results

if __name__ == "__main__":
    import os
    os.environ["CITYLEARN_KPI_RUN_NAME"] = "__DISABLE__"
    
    print("=" * 60)
    print("EXTREME WEATHER CASE STUDIES")
    print("=" * 60)
    
    # Heat Wave
    print("\n" + "=" * 60)
    print("HEAT WAVE SCENARIO")
    print("=" * 60)
    heat_summary, heat_results = evaluate_scenario(scenario_type="heat_wave", agent_type="rbc", num_episodes=3)
    
    print("\n--- RBC Performance (Heat Wave) ---")
    for metric, value in heat_summary.items():
        print(f"{metric:35s}: {value:8.4f}")
    
    # Cold Wave
    print("\n" + "=" * 60)
    print("COLD WAVE SCENARIO")
    print("=" * 60)
    cold_summary, cold_results = evaluate_scenario(scenario_type="cold_wave", agent_type="rbc", num_episodes=3)
    
    print("\n--- RBC Performance (Cold Wave) ---")
    for metric, value in cold_summary.items():
        print(f"{metric:35s}: {value:8.4f}")
    
    # Save results
    import json
    with open("results_extreme_weather.json", "w") as f:
        json.dump({
            "heat_wave": {
                "rbc": heat_summary,
            },
            "cold_wave": {
                "rbc": cold_summary,
            }
        }, f, indent=2)
    
    print("\n✓ Results saved to results_extreme_weather.json")
