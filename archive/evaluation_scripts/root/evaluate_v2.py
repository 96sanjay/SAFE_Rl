#!/usr/bin/env python3
"""
Evaluate trained Safe RL model and compare with baselines.

Usage:
    python evaluate_v2.py --checkpoint PATH_TO_EPOCH.pt
    python evaluate_v2.py --checkpoint PATH_TO_EPOCH.pt --with-psf
"""
from __future__ import annotations

import os
import sys
import argparse
import time
import numpy as np

PROJECT_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

for k, v in {
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "2.273834",
    "CITYLEARN_STEMS_P_GRID_MAX": "27.127751",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_zero",
    "CITYLEARN_SCHEMA": "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json",
    "CITYLEARN_EV_DENSE_COST_SCALE": "1.0",
    "CITYLEARN_REWARD_TYPE": "stems",
}.items():
    os.environ.setdefault(k, v)


def make_eval_env(use_forecast=False):
    from evaluation.core.evaluator import make_env
    env = make_env()
    if use_forecast:
        from forecast_obs_wrapper import ForecastObsWrapper
        env = ForecastObsWrapper(env, forecast_horizon=24)
    return env


def load_agent(checkpoint_path, obs_dim):
    from evaluation.agents.omnisafe_gaussian import OmniSafeCheckpointAgent
    return OmniSafeCheckpointAgent(checkpoint_path, name="safe_rl_v2")


def run_evaluation(env, agent, max_steps=8760, verbose=True):
    obs, info = env.reset()
    total_reward = 0.0
    metrics = {
        'ev_violations': 0, 'ev_departures': 0,
        'battery_violations': 0, 'battery_checks': 0,
        'building_violations': 0, 'building_checks': 0,
        'grid_violations': 0, 'grid_checks': 0,
    }

    t0 = time.time()
    for step in range(max_steps):
        action = agent.predict(obs)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward

        deps = int(info.get('ev_departure_departures', 0))
        if deps > 0:
            metrics['ev_departures'] += deps
            deficit = float(info.get('ev_departure_deficit_kwh', 0.0))
            if deficit > 0:
                viol_cnt = int(info.get('ev_departure_violation_count_deficit', 0))
                metrics['ev_violations'] += max(viol_cnt, 1 if deficit > 0 else 0)

        metrics['battery_violations'] += int(float(info.get('battery_soc_violation_count', 0.0)))
        metrics['battery_checks'] += 17
        metrics['building_violations'] += int(float(info.get('building_power_violation_count', 0.0)))
        metrics['building_checks'] += 17
        if float(info.get('cost_stems_grid_power', 0.0)) > 0:
            metrics['grid_violations'] += 1
        metrics['grid_checks'] += 1

        if verbose and (step + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  Step {step+1}/{max_steps} | "
                  f"Reward: {total_reward:.1f} | "
                  f"C1: {metrics['ev_violations']}/{metrics['ev_departures']} | "
                  f"C4: {metrics['grid_violations']}/{metrics['grid_checks']} | "
                  f"Time: {elapsed:.0f}s")

        if term or trunc:
            break

    elapsed = time.time() - t0
    return {
        'reward': total_reward, 'steps': step + 1, 'time_s': elapsed,
        'C1_violations': metrics['ev_violations'],
        'C1_total': metrics['ev_departures'],
        'C1_rate': (metrics['ev_violations'] / max(1, metrics['ev_departures'])) * 100,
        'C2_violations': metrics['battery_violations'],
        'C2_total': metrics['battery_checks'],
        'C2_rate': (metrics['battery_violations'] / max(1, metrics['battery_checks'])) * 100,
        'C3_violations': metrics['building_violations'],
        'C3_total': metrics['building_checks'],
        'C3_rate': (metrics['building_violations'] / max(1, metrics['building_checks'])) * 100,
        'C4_violations': metrics['grid_violations'],
        'C4_total': metrics['grid_checks'],
        'C4_rate': (metrics['grid_violations'] / max(1, metrics['grid_checks'])) * 100,
    }


def print_results(name, r):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Reward:  {r['reward']:.1f}")
    print(f"  C1 (EV):       {r['C1_violations']}/{r['C1_total']} = {r['C1_rate']:.2f}%")
    print(f"  C2 (batt SoC): {r['C2_violations']}/{r['C2_total']} = {r['C2_rate']:.2f}%")
    print(f"  C3 (bldg pwr): {r['C3_violations']}/{r['C3_total']} = {r['C3_rate']:.2f}%")
    print(f"  C4 (grid pwr): {r['C4_violations']}/{r['C4_total']} = {r['C4_rate']:.2f}%")


def print_comparison(results):
    print(f"\n{'='*80}")
    print(f"{'Config':<25} {'C1':>8} {'C2':>8} {'C3':>8} {'C4':>8} {'Reward':>10}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<25} {r['C1_rate']:>7.2f}% {r['C2_rate']:>7.2f}% "
              f"{r['C3_rate']:>7.2f}% {r['C4_rate']:>7.2f}% {r['reward']:>10.1f}")
    print(f"{'='*80}")
    print("  Target:                   <5.00%   <5.00%  <15.00%   <5.00%")
    print("  Old PPOLag:              90.20%    4.08%   24.50%    1.86%    52.6")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--with-psf', action='store_true')
    parser.add_argument('--steps', type=int, default=8760)
    args = parser.parse_args()

    all_results = {}

    print("\n[1] Evaluating new model (agent only)...")
    env = make_eval_env(use_forecast=True)
    agent = load_agent(args.checkpoint, env.observation_space.shape[0])
    r = run_evaluation(env, agent, max_steps=args.steps)
    all_results['New Agent Only'] = r
    print_results('New Agent Only', r)
    env.close()

    if args.with_psf:
        print("\n[2] Evaluating new model + PSF C1-only...")
        env = make_eval_env(use_forecast=True)
        from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper
        psf_env = LookaheadPSFWrapper(
            env, horizon=24, w_track=1.0,
            w_slack_c1=1000.0, w_slack_c3=0.0, w_slack_c4=0.0,
            freeze_ev=True, verbose=0)
        r = run_evaluation(psf_env, agent, max_steps=args.steps)
        all_results['New + PSF(C1)'] = r
        print_results('New + PSF(C1)', r)
        psf_env.close()

    print_comparison(all_results)


if __name__ == '__main__':
    main()
