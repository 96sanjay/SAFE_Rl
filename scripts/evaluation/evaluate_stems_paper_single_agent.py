#!/usr/bin/env python3
"""Evaluate isolated paper-style single-agent STEMS checkpoints.

This intentionally does not depend on PPO loaders. It reconstructs the
paper-style STEMS actor directly from the checkpoint and replays the policy on
the 5-building schema using the same raw count logic used elsewhere in the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import citylearn_safe.cmdp_env  # noqa: F401  (registers env)

from citylearn_safe.feasibility_obs_wrapper import FeasibilityObsWrapper
from citylearn_safe.cmdp_env import CityLearnCMDP
from citylearn_safe.policy_action_mask import CityLearnActionBoundsProvider
from citylearn_safe.stems_5bld_factory import build_obs_index_5bld_spec, build_stems_encoder_5bld
from citylearn_safe.stems_paper_single_agent import build_paper_single_agent_stems_ac
from scripts.train_stems_paper_single_agent import DEFAULT_COST_KEYS


@contextmanager
def temporary_env_overrides(overrides: dict[str, str]):
    old_values = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def build_model_from_checkpoint(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["cfg"]
    cost_keys = list(cfg.get("resolved_cost_keys", DEFAULT_COST_KEYS))
    schema_path = cfg["schema_path"]
    encoder_cfgs = cfg["encoder_cfgs"]
    observation_cfgs = cfg.get("observation_cfgs", {})
    actor_cfgs = cfg.get("actor_cfgs", {})
    overrides = fair_r25_recipe_overrides(
        schema_path=schema_path,
        temporal_window=int(encoder_cfgs["temporal_window"]),
    )
    with temporary_env_overrides(overrides):
        spec = build_obs_index_5bld_spec(
            schema_path=schema_path,
            temporal_window=int(encoder_cfgs["temporal_window"]),
        )
        env = CityLearnCMDP(cfg["env_id"])
        if bool(observation_cfgs.get("append_feasibility_bounds", False)):
            env = FeasibilityObsWrapper(env)
        extra_obs_dim = int(env.observation_space.shape[0]) - spec.encoder_obs_dim
        actor_in_features = ckpt["model"]["actor.mean_net.0.weight"].shape[1]
        inferred_extra_obs_dim = int(actor_in_features) - int(encoder_cfgs["output_dim"])
        if inferred_extra_obs_dim >= 0:
            extra_obs_dim = inferred_extra_obs_dim
        env.close()
        encoder = build_stems_encoder_5bld(
            spec,
            hidden_dim=int(encoder_cfgs["hidden_dim"]),
            global_hidden=int(encoder_cfgs["global_hidden"]),
            temporal_hidden=int(encoder_cfgs["temporal_hidden"]),
            temporal_heads=int(encoder_cfgs["temporal_heads"]),
            num_gcn_layers=int(encoder_cfgs["num_gcn_layers"]),
            output_dim=int(encoder_cfgs["output_dim"]),
            dropout=float(encoder_cfgs["dropout"]),
        )
        model = build_paper_single_agent_stems_ac(
            spec=spec,
            encoder=encoder,
            num_costs=len(cost_keys),
            actor_hidden_sizes=tuple(int(x) for x in encoder_cfgs["actor_hidden_sizes"]),
            critic_hidden_sizes=tuple(int(x) for x in encoder_cfgs["critic_hidden_sizes"]),
            extra_obs_dim=extra_obs_dim,
            use_action_bounds_mask=bool(actor_cfgs.get("use_action_bounds_mask", False)),
            action_bounds_dim=(2 * spec.act_dim if bool(observation_cfgs.get("append_feasibility_bounds", False)) else 0),
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
    return model, cfg


def fair_r25_recipe_overrides(schema_path: str, temporal_window: int) -> dict[str, str]:
    return {
        "CITYLEARN_SCHEMA": schema_path,
        "CITYLEARN_TEMPORAL_WINDOW": str(temporal_window),
        "STEMS_ENCODER_VERSION": "v3",
        "CITYLEARN_CENTRAL_AGENT": "1",
        "CITYLEARN_REWARD_TYPE": "stems",
        "CITYLEARN_EXPORT_FACTOR": "0.7",
        "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
        "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
        "CITYLEARN_STEMS_SOC_LOW": "0.0",
        "CITYLEARN_STEMS_SOC_HIGH": "0.95",
        "CITYLEARN_STEMS_PNORM_P": "4.0",
        "CITYLEARN_PID_LAGRANGE": "1",
        "CITYLEARN_EV_SAUTE": "1",
        "CITYLEARN_EV_SAUTE_BUDGET": "25000",
        "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
        "CITYLEARN_EV_SAUTE_GAMMA": "1.0",
        "CITYLEARN_EV_SAUTE_SHAPED_ALPHA": "10.0",
        "STEMS_ALPHA_GRID": "0.0",
        "STEMS_SG_THRESHOLD": "0.5",
        "STEMS_ALPHA_LOAD_SHIFT": "0.0",
        "STEMS_ALPHA_GRID_MILD": "0.3",
        "STEMS_MU_ECONOMIC": "0.0",
        "STEMS_ALPHA_BUILD": "0.0",
        "STEMS_XI_RENEWABLE": "0.2",
        "STEMS_BETA_RAMP": "0.3",
        "CITYLEARN_C3_CONTROLLABLE": "1",
        "STEMS_LAMBDA_EV": "5.0",
        "STEMS_SB_ASYMMETRIC": "1",
        "STEMS_SG_EXPORT_CREDIT": "0.5",
        "STEMS_ALPHA_BARRIER": "0.5",
        "STEMS_ALPHA_PEAK_SHAVE": "0.0",
        "STEMS_ALPHA_EV_GUARD": "1.0",
        "STEMS_ALPHA_V2G_CONTEXT": "3.0",
        "STEMS_EV_SLACK_ARB_SCALE": "2.0",
        "COST_W_C2": "0.0",
        "COST_W_C3": "5.0",
        "CITYLEARN_W_COST_EV": "1.0",
        "CITYLEARN_W_COST_SOC": "10.0",
        "CITYLEARN_W_COST_BUILDING": "0.5",
        "CITYLEARN_W_COST_GRID": "0.05",
        "CITYLEARN_EV_COST_SCALE": "3.0",
        "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
        "CITYLEARN_INCLUDE_EV_COST": "1",
        "CITYLEARN_SPATIAL_OBS": "0",
        "CITYLEARN_WM_DISABLE": "1",
        "CITYLEARN_EV_ACTION_CLAMP": "0",
        "CITYLEARN_BATT_CLAMP": "0",
        "CITYLEARN_ACTION_MASK": "0",
        "CITYLEARN_POLICY_ACTION_MASK": "0",
    }


def unwrap_to_raw_citylearn_env(env):
    cur = env
    for _ in range(30):
        if hasattr(cur, "buildings"):
            return cur
        cur = getattr(cur, "env", getattr(cur, "base", getattr(cur, "_env", None)))
        if cur is None:
            break
    raise RuntimeError("Could not unwrap to raw CityLearn env.")


def unwrap_to_safety_env(env):
    cur = env
    for _ in range(30):
        if hasattr(cur, "P_building_max") and hasattr(cur, "P_grid_max"):
            return cur
        cur = getattr(cur, "env", getattr(cur, "base", getattr(cur, "_env", None)))
        if cur is None:
            break
    raise RuntimeError("Could not unwrap to safety env.")


def evaluate(
    checkpoint_path: str,
    output_json: str | None = None,
    project_actions: bool = False,
) -> dict[str, Any]:
    model, cfg = build_model_from_checkpoint(checkpoint_path)
    schema_path = cfg["schema_path"]
    encoder_cfgs = cfg["encoder_cfgs"]

    overrides = fair_r25_recipe_overrides(
        schema_path=schema_path,
        temporal_window=int(encoder_cfgs["temporal_window"]),
    )

    with temporary_env_overrides(overrides):
        env = CityLearnCMDP(cfg["env_id"])
        if bool(cfg.get("observation_cfgs", {}).get("append_feasibility_bounds", False)):
            env = FeasibilityObsWrapper(env)
        bounds_provider = CityLearnActionBoundsProvider(env) if project_actions else None
        raw = unwrap_to_raw_citylearn_env(env)
        safety = unwrap_to_safety_env(env)
        buildings = list(raw.buildings)
        p_bmax = float(getattr(safety, "P_building_max", 4.6083))
        p_gmax = float(getattr(safety, "P_grid_max", 10.2352))

        obs, _ = env.reset(seed=int(cfg.get("seed", 42)))

        total_reward = 0.0
        total_steps = 0

        c0_events = 0
        c0_violations = 0
        c1_total = 0
        c1_violations = 0
        c2_total = 0
        c2_violations = 0
        c3_total = 0
        c3_violations = 0
        c4_violations = 0
        projection_steps = 0
        projection_changed_steps = 0
        projection_l2_sum = 0.0
        projection_linf_max = 0.0

        ev_tracker: dict[tuple[int, int], float] = {}

        done = False
        while not done:
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = model.act(obs_t, deterministic=True)
            action = out.action.squeeze(0).cpu().numpy()
            if bounds_provider is not None:
                safe_min, safe_max = bounds_provider.current_safe_bounds()
                projected_action = np.clip(action, safe_min, safe_max)
                delta = action - projected_action
                projection_steps += 1
                if np.any(np.abs(delta) > 1e-6):
                    projection_changed_steps += 1
                projection_l2_sum += float(np.linalg.norm(delta))
                projection_linf_max = max(projection_linf_max, float(np.max(np.abs(delta))))
                action = projected_action

            t_now = int(getattr(raw, "time_step", 0))

            for b_idx, bld in enumerate(buildings):
                chargers = getattr(bld, "electric_vehicle_chargers", None) or []
                for ch_idx, ch in enumerate(chargers):
                    key = (b_idx, ch_idx)
                    sim = getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))
                    if sim is None:
                        continue
                    try:
                        state_arr = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                        connected_now = t_now < len(state_arr) and float(state_arr[t_now]) == 1.0
                        if connected_now:
                            req_arr = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
                            req_soc = float(req_arr[t_now]) if t_now < len(req_arr) else 1.0
                            ev = getattr(ch, "connected_electric_vehicle", None)
                            bt = getattr(ev, "battery", None) if ev is not None else None
                            soc_now = 0.0
                            if bt is not None and getattr(bt, "soc", None) is not None:
                                soc_arr = np.asarray(bt.soc, dtype=float)
                                soc_idx = max(0, t_now - 1)
                                if soc_idx < len(soc_arr):
                                    soc_now = float(np.clip(soc_arr[soc_idx], 0.0, 1.0))
                            ev_tracker[key] = {"was": True, "soc": soc_now, "req": req_soc}
                        else:
                            prev = ev_tracker.get(key, {})
                            if prev.get("was", False):
                                c0_events += 1
                                deficit = max(0.0, float(prev["req"]) - float(prev["soc"]))
                                if deficit > 0.01:
                                    c0_violations += 1
                            ev_tracker[key] = {"was": False}
                    except Exception:
                        pass

            next_obs, reward, _, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            total_steps += 1
            t_idx = max(0, int(getattr(raw, "time_step", 0)) - 1)

            c1_total += 1
            c1_violations += int(float(info.get("cost_ev_dense", 0.0)) > 0.0)
            c4_violations += int(float(info.get("cost_stems_grid_power", 0.0)) > 0.0)

            for bld in buildings:
                es = getattr(bld, "electrical_storage", None)
                soc_arr = getattr(es, "soc", None) if es is not None else None
                if soc_arr is not None and len(soc_arr) > t_idx:
                    soc = float(soc_arr[t_idx])
                    c2_total += 1
                    if soc < 0.0 or soc > 0.95:
                        c2_violations += 1

                nec_arr = getattr(bld, "net_electricity_consumption", None)
                if nec_arr is not None and len(nec_arr) > t_idx:
                    c3_total += 1
                    if abs(float(nec_arr[t_idx])) > p_bmax:
                        c3_violations += 1

            obs = next_obs
            done = bool(terminated.item() if hasattr(terminated, "item") else terminated) or bool(
                truncated.item() if hasattr(truncated, "item") else truncated
            )

        results = {
            "checkpoint": checkpoint_path,
            "steps": total_steps,
            "reward_sum": total_reward,
            "c0_violations": c0_violations,
            "c0_events": c0_events,
            "c0_pct": 100.0 * c0_violations / max(c0_events, 1),
            "c1_violations": c1_violations,
            "c1_total": c1_total,
            "c1_pct": 100.0 * c1_violations / max(c1_total, 1),
            "c2_violations": c2_violations,
            "c2_total": c2_total,
            "c2_pct": 100.0 * c2_violations / max(c2_total, 1),
            "c3_violations": c3_violations,
            "c3_total": c3_total,
            "c3_pct": 100.0 * c3_violations / max(c3_total, 1),
            "c4_violations": c4_violations,
            "c4_total": total_steps,
            "c4_pct": 100.0 * c4_violations / max(total_steps, 1),
            "p_building_max": p_bmax,
            "p_grid_max": p_gmax,
            "project_actions": bool(project_actions),
        }
        if project_actions:
            results.update(
                {
                    "projection_steps": projection_steps,
                    "projection_changed_steps": projection_changed_steps,
                    "projection_changed_pct": 100.0 * projection_changed_steps / max(projection_steps, 1),
                    "projection_mean_l2": projection_l2_sum / max(projection_steps, 1),
                    "projection_max_linf": projection_linf_max,
                },
            )

        env.close()

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper-style STEMS single-agent checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--project-actions", action="store_true")
    args = parser.parse_args()

    results = evaluate(
        args.checkpoint,
        output_json=args.output_json,
        project_actions=bool(args.project_actions),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
