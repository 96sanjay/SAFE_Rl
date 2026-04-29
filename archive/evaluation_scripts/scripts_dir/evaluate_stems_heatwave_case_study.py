from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from gymnasium.spaces import Box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENDORED_OMNISAFE = PROJECT_ROOT / "vendor_deps"
if str(VENDORED_OMNISAFE) not in sys.path:
    sys.path.insert(0, str(VENDORED_OMNISAFE))

from citylearn_safe.omni_env_temp_cooling_only import CityLearnTempCoolingOnlyMaskedRewardCMDP
from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor


@dataclass
class WindowResult:
    start: int
    end: int
    name: str
    total_reward: float
    total_cost: float
    violation_rate: float
    discomfort_rate: float
    kpis: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STEMS-style Heat Wave evaluation for the temperature case study.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--temp-threshold", type=float, default=25.7)
    parser.add_argument("--min-hours", type=int, default=24)
    parser.add_argument("--max-hours", type=int, default=24)
    parser.add_argument(
        "--use-max-available-spell",
        action="store_true",
        help="Adapt the paper's 2-3 day Heat Wave concept to the maximum contiguous hot spell available in this schema.",
    )
    parser.add_argument("--cooling-demand-multiplier", type=float, default=2.5)
    parser.add_argument("--solar-generation-multiplier", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _apply_env_overrides(run_dir: Path) -> None:
    cfg = _load_run_config(run_dir)
    for key, value in (cfg.get("env_overrides") or {}).items():
        os.environ[str(key)] = str(value)


def _latest_checkpoint(run_dir: Path) -> Path:
    torch_dir = run_dir / "torch_save"
    candidates = sorted(torch_dir.glob("epoch-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {torch_dir}")
    return candidates[-1]


def _infer_hidden_sizes(pi_state: dict[str, Any]) -> list[int]:
    hidden = []
    layer_idx = 0
    while f"net.{layer_idx}.weight" in pi_state:
        weight = pi_state[f"net.{layer_idx}.weight"]
        if layer_idx + 2 < 5:
            hidden.append(int(weight.shape[0]))
        layer_idx += 2
        if f"net.{layer_idx}.weight" not in pi_state:
            break
    return hidden if len(hidden) >= 2 else [256, 256]


def _load_actor(checkpoint: Path) -> GaussianSACActor:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pi_state = state["pi"]
    obs_dim = int(pi_state["net.0.weight"].shape[1])
    act_dim = int(pi_state["net.4.weight"].shape[0] // 2)
    actor = GaussianSACActor(
        obs_space=Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        act_space=Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
        hidden_sizes=_infer_hidden_sizes(pi_state),
        activation="relu",
        weight_initialization_mode="kaiming_uniform",
    )
    actor.load_state_dict(pi_state, strict=True)
    actor.eval()
    return actor


def _selected_kpis(eval_df: pd.DataFrame) -> dict[str, float]:
    if eval_df is None or eval_df.empty:
        return {}
    cols = {str(c).lower(): c for c in eval_df.columns}
    key_col = cols.get("cost_function")
    value_col = cols.get("value")
    if key_col is None or value_col is None:
        return {}
    out: dict[str, float] = {}
    for _, row in eval_df.iterrows():
        try:
            out[str(row[key_col])] = float(row[value_col])
        except Exception:
            continue
    return out


def _find_heatwave_windows(temp: np.ndarray, threshold: float, min_hours: int, max_hours: int) -> list[tuple[int, int]]:
    hot = np.asarray(temp >= threshold, dtype=bool)
    windows: list[tuple[int, int]] = []
    i = 0
    n = len(hot)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < n and hot[j]:
            j += 1
        span_len = j - i
        if span_len >= min_hours:
            start = i
            while start + min_hours <= j:
                end = min(start + max_hours - 1, j - 1)
                if end - start + 1 >= min_hours:
                    windows.append((start, end))
                start = end + 1
        i = j
    return windows


def _find_max_heatwave_spell(temp: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    hot = np.asarray(temp >= threshold, dtype=bool)
    best: tuple[int, int] | None = None
    best_len = 0
    i = 0
    n = len(hot)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < n and hot[j]:
            j += 1
        span_len = j - i
        if span_len > best_len:
            best = (i, j - 1)
            best_len = span_len
        i = j
    return [] if best is None else [best]


def _make_window_env(run_dir: Path, start: int, end: int) -> CityLearnTempCoolingOnlyMaskedRewardCMDP:
    _apply_env_overrides(run_dir)
    cfg = _load_run_config(run_dir)
    env_id = cfg.get("env_id", "CityLearnTemp-CoolingOnly-CSACLB-v0")
    env = CityLearnTempCoolingOnlyMaskedRewardCMDP(
        env_id,
        citylearn_env_kwargs={
            "episode_time_steps": [[int(start), int(end)]],
            "rolling_episode_split": False,
            "random_episode_split": False,
        },
    )
    return env


def _make_full_env(run_dir: Path) -> CityLearnTempCoolingOnlyMaskedRewardCMDP:
    _apply_env_overrides(run_dir)
    cfg = _load_run_config(run_dir)
    env_id = cfg.get("env_id", "CityLearnTemp-CoolingOnly-CSACLB-v0")
    return CityLearnTempCoolingOnlyMaskedRewardCMDP(env_id)


def _apply_heatwave_perturbation(
    env: CityLearnTempCoolingOnlyMaskedRewardCMDP,
    start: int,
    end: int,
    cooling_multiplier: float,
    solar_multiplier: float,
) -> None:
    city = env._get_citylearn()
    assert city is not None
    for b in city.buildings:
        es = getattr(b, "energy_simulation", None)
        if es is not None:
            cooling = np.asarray(getattr(es, "_cooling_demand"), dtype=np.float32).copy()
            cooling[start : end + 1] *= np.float32(cooling_multiplier)
            setattr(es, "_cooling_demand", cooling)
            cooling_wo = np.asarray(getattr(es, "_cooling_demand_without_control"), dtype=np.float32).copy()
            cooling_wo[start : end + 1] *= np.float32(cooling_multiplier)
            setattr(es, "_cooling_demand_without_control", cooling_wo)

            solar = np.asarray(getattr(es, "_solar_generation"), dtype=np.float32).copy()
            solar[start : end + 1] *= np.float32(solar_multiplier)
            setattr(es, "_solar_generation", solar)


def _evaluate_citylearn_kpis(env: CityLearnTempCoolingOnlyMaskedRewardCMDP) -> dict[str, float]:
    city = env._get_citylearn()
    if city is None or not hasattr(city, "evaluate"):
        return {}
    try:
        eval_df = city.evaluate()
    except Exception:
        return {}
    return _selected_kpis(eval_df)


def _rollout_window(
    run_dir: Path,
    window: tuple[int, int],
    name: str,
    actor: GaussianSACActor | None,
    seed: int,
    cooling_multiplier: float,
    solar_multiplier: float,
) -> WindowResult:
    start, end = window
    env = _make_window_env(run_dir, start, end)
    _apply_heatwave_perturbation(env, start, end, cooling_multiplier, solar_multiplier)
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    total_reward = 0.0
    total_cost = 0.0
    comfort_violations = 0
    active_steps = 0
    discomfort_count = 0.0
    occupied_count = 0.0

    while not (bool(terminated) or bool(truncated)):
        if actor is None:
            action = env._rbc_cooling_action()
        elif name == "RBC 2":
            action = env._rbc2_cooling_action()
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                raw_action = actor.predict(obs_t, deterministic=True).cpu().numpy().reshape(-1)
            action = np.clip((raw_action + 1.0) * 0.5, 0.0, 1.0)

        obs, reward, cost, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_cost += float(cost)
        if float(info.get("comfort_in_warmup", 0.0)) < 0.5:
            active_steps += 1
            comfort_violations += int(float(info.get("comfort_violation", 0.0)) > 0.5)
        discomfort_count += float(info.get("discomfort_count", 0.0))
        occupied_count += float(info.get("occupied_count", 0.0))

    return WindowResult(
        start=start,
        end=end,
        name=name,
        total_reward=total_reward,
        total_cost=total_cost,
        violation_rate=(comfort_violations / active_steps) if active_steps > 0 else 0.0,
        discomfort_rate=(discomfort_count / occupied_count) if occupied_count > 0 else 0.0,
        kpis=_evaluate_citylearn_kpis(env),
    )


def _aggregate(results: list[WindowResult]) -> dict[str, Any]:
    if not results:
        return {}
    df = pd.DataFrame(
        [
            {
                "total_reward": r.total_reward,
                "total_cost": r.total_cost,
                "violation_rate": r.violation_rate,
                "discomfort_rate": r.discomfort_rate,
                **{f"kpi::{k}": v for k, v in r.kpis.items()},
            }
            for r in results
        ]
    )
    out = {col: float(df[col].mean()) for col in df.columns}
    out["n_windows"] = int(len(results))
    return out


def _improvement(model: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return 100.0 * (baseline - model) / baseline


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or _latest_checkpoint(args.run_dir)
    actor = _load_actor(checkpoint)

    probe_env = _make_full_env(args.run_dir)
    city = probe_env._get_citylearn()
    assert city is not None
    temp = np.asarray(city.buildings[0].weather.outdoor_dry_bulb_temperature, dtype=np.float32)
    if args.use_max_available_spell:
        windows = _find_max_heatwave_spell(temp, args.temp_threshold)
    else:
        windows = _find_heatwave_windows(temp, args.temp_threshold, args.min_hours, args.max_hours)
    if not windows:
        raise RuntimeError("No Heat Wave windows found for the requested threshold/min-hours settings.")

    model_results = [
        _rollout_window(
            args.run_dir,
            window,
            "CSAC-LB",
            actor,
            args.seed,
            args.cooling_demand_multiplier,
            args.solar_generation_multiplier,
        )
        for window in windows
    ]
    baseline_results = [
        _rollout_window(
            args.run_dir,
            window,
            "Cooling RBC",
            None,
            args.seed,
            args.cooling_demand_multiplier,
            args.solar_generation_multiplier,
        )
        for window in windows
    ]
    baseline2_results = [
        _rollout_window(
            args.run_dir,
            window,
            "RBC 2",
            "rbc2",
            args.seed,
            args.cooling_demand_multiplier,
            args.solar_generation_multiplier,
        )
        for window in windows
    ]

    model_avg = _aggregate(model_results)
    baseline_avg = _aggregate(baseline_results)
    baseline2_avg = _aggregate(baseline2_results)

    tech_pairs = {
        "cost_reduction_pct_vs_rbc": ("kpi::cost_total", baseline_avg.get("kpi::cost_total")),
        "emission_reduction_pct_vs_rbc": ("kpi::carbon_emissions_total", baseline_avg.get("kpi::carbon_emissions_total")),
        "peak_reduction_pct_vs_rbc": ("kpi::daily_peak_average", baseline_avg.get("kpi::daily_peak_average")),
        "consumption_reduction_pct_vs_rbc": ("kpi::electricity_consumption_total", baseline_avg.get("kpi::electricity_consumption_total")),
        "ramping_reduction_pct_vs_rbc": ("kpi::ramping_average", baseline_avg.get("kpi::ramping_average")),
    }
    improvement = {}
    for key, (metric, baseline_val) in tech_pairs.items():
        model_val = model_avg.get(metric)
        improvement[key] = None if model_val is None or baseline_val is None else _improvement(model_val, baseline_val)

    payload = {
        "paper_concept": {
            "window_hours": f"{args.min_hours}-{args.max_hours}",
            "heatwave_threshold_c": args.temp_threshold,
            "cooling_demand_multiplier": args.cooling_demand_multiplier,
            "solar_generation_multiplier": args.solar_generation_multiplier,
            "adaptation": "max_available_spell" if args.use_max_available_spell else "exact_2_to_3_day_windows",
        },
        "checkpoint": str(checkpoint),
        "n_windows": len(windows),
        "windows": [{"start": int(s), "end": int(e), "hours": int(e - s + 1)} for s, e in windows],
        "csac_lb_avg": model_avg,
        "cooling_rbc_avg": baseline_avg,
        "rbc_2_avg": baseline2_avg,
        "improvement_vs_rbc_pct": improvement,
        "per_window": {
            "csac_lb": [r.__dict__ for r in model_results],
            "cooling_rbc": [r.__dict__ for r in baseline_results],
            "rbc_2": [r.__dict__ for r in baseline2_results],
        },
    }

    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
