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
from citylearn_safe.policy_action_mask_temp import masked_action_from_pretanh
from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor


@dataclass
class RolloutResult:
    name: str
    total_reward: float
    total_cost: float
    violation_rate: float
    discomfort_rate: float
    mean_cooling_action: float
    reward_economic: float
    reward_stability: float
    reward_renewable: float
    reward_comfort: float
    district_import_kwh: float
    district_net_kwh: float
    total_steps: int
    active_steps: int
    comfort_violations: int
    discomfort_count: float
    occupied_count: float
    kpis: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate masked PPO on the cooling-only temperature case study.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--best-epoch", type=int, default=None)
    parser.add_argument("--final-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--policy-kind", type=str, default=None,
                        choices=["ckpt", "ckpt_linear"],
                        help="Override policy kind for checkpoint evaluation (default: auto-detect)")
    return parser.parse_args()


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _best_epoch_from_progress(progress_path: Path) -> int:
    df = pd.read_csv(progress_path)
    idx = df["Metrics/EpCost"].astype(float).idxmin()
    return int(float(df.loc[idx, "Train/Epoch"]))


def _latest_checkpoint_epoch(run_dir: Path) -> int:
    torch_dir = run_dir / "torch_save"
    epochs: list[int] = []
    for path in torch_dir.glob("epoch-*.pt"):
        try:
            epochs.append(int(path.stem.split("-")[1]))
        except Exception:
            continue
    if not epochs:
        raise FileNotFoundError(f"No checkpoints found under {torch_dir}")
    return max(epochs)


def _apply_env_overrides(run_dir: Path) -> None:
    cfg = _load_run_config(run_dir)
    for key, value in (cfg.get("env_overrides") or {}).items():
        os.environ[str(key)] = str(value)


def _is_sac_checkpoint(pi_state: dict[str, Any]) -> bool:
    return any(key.startswith("net.") for key in pi_state)


def _infer_hidden_sizes(pi_state: dict[str, Any], prefix: str = "mean") -> list[int]:
    hidden = []
    layer_ids = sorted(
        int(key.split(".")[1])
        for key in pi_state
        if key.startswith(f"{prefix}.") and key.endswith(".weight")
    )
    for layer_id in layer_ids[:-1]:
        weight = pi_state[f"{prefix}.{layer_id}.weight"]
        hidden.append(int(weight.shape[0]))
    return hidden or [256, 256]


def _load_actor(checkpoint: Path) -> GaussianLearningActor | GaussianSACActor:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pi_state = state["pi"]

    if _is_sac_checkpoint(pi_state):
        obs_dim = int(pi_state["net.0.weight"].shape[1])
        last_layer = max(
            int(key.split(".")[1])
            for key in pi_state
            if key.startswith("net.") and key.endswith(".weight")
        )
        act_dim = int(pi_state[f"net.{last_layer}.weight"].shape[0]) // 2
        hidden_sizes = _infer_hidden_sizes(pi_state, prefix="net")
        actor = GaussianSACActor(
            obs_space=Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            act_space=Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
            hidden_sizes=hidden_sizes,
            activation="relu",
            weight_initialization_mode="kaiming_uniform",
        )
        actor.load_state_dict(pi_state, strict=True)
        actor.eval()
        return actor

    obs_dim = int(pi_state["mean.0.weight"].shape[1])
    last_layer = max(
        int(key.split(".")[1])
        for key in pi_state
        if key.startswith("mean.") and key.endswith(".weight")
    )
    act_dim = int(pi_state[f"mean.{last_layer}.weight"].shape[0])
    hidden_sizes = _infer_hidden_sizes(pi_state, prefix="mean")
    actor = GaussianLearningActor(
        obs_space=Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        act_space=Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
        hidden_sizes=hidden_sizes,
        activation="tanh",
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


def _evaluate_citylearn_kpis(env: CityLearnTempCoolingOnlyMaskedRewardCMDP) -> dict[str, float]:
    city = env._get_citylearn()
    if city is None or not hasattr(city, "evaluate"):
        return {}
    try:
        eval_df = city.evaluate()
    except Exception:
        return {}
    return _selected_kpis(eval_df)


def _make_eval_env(run_dir: Path) -> CityLearnTempCoolingOnlyMaskedRewardCMDP:
    _apply_env_overrides(run_dir)
    cfg = _load_run_config(run_dir)
    env_id = cfg.get("env_id", "CityLearnTemp-CoolingOnly-Masked-Reward-v0")
    env_cfgs = cfg.get("env_cfgs") or {}
    return CityLearnTempCoolingOnlyMaskedRewardCMDP(env_id, **env_cfgs)


def _cooling_only_rbc_action(env: CityLearnTempCoolingOnlyMaskedRewardCMDP) -> np.ndarray:
    return env._rbc_cooling_action()


def _cooling_only_rbc2_action(env: CityLearnTempCoolingOnlyMaskedRewardCMDP) -> np.ndarray:
    return env._rbc2_cooling_action()


def _cooling_only_rbc3_action(env: CityLearnTempCoolingOnlyMaskedRewardCMDP) -> np.ndarray:
    return env._rbc3_cooling_action()


def run_rollout(name: str, policy_kind: str, seed: int, run_dir: Path, checkpoint: Path | None = None) -> RolloutResult:
    env = _make_eval_env(run_dir)
    actor = _load_actor(checkpoint) if checkpoint is not None else None

    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    total_reward = 0.0
    total_cost = 0.0
    total_steps = 0
    active_steps = 0
    comfort_violations = 0
    occupied_count = 0.0
    discomfort_count = 0.0
    action_sum = 0.0
    action_count = 0
    reward_economic = 0.0
    reward_stability = 0.0
    reward_renewable = 0.0
    reward_comfort = 0.0
    district_import_kwh = 0.0
    district_net_kwh = 0.0

    while not (bool(terminated) or bool(truncated)):
        if policy_kind == "rbc":
            action = _cooling_only_rbc_action(env)
        elif policy_kind == "rbc2":
            action = _cooling_only_rbc2_action(env)
        elif policy_kind == "rbc3":
            action = _cooling_only_rbc3_action(env)
        elif policy_kind == "zero":
            action = np.zeros(env.cooling_action_dim, dtype=np.float32)
        elif policy_kind == "ckpt":
            assert actor is not None
            obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                if isinstance(actor, GaussianSACActor):
                    pre_tanh = actor._distribution(obs_t).mean
                else:
                    pre_tanh = actor.predict(obs_t, deterministic=True)
                safe_min_np, safe_max_np = env.make_policy_action_bounds_provider().current_safe_bounds()
                safe_min = torch.as_tensor(safe_min_np, dtype=torch.float32).unsqueeze(0)
                safe_max = torch.as_tensor(safe_max_np, dtype=torch.float32).unsqueeze(0)
                action_t = masked_action_from_pretanh(pre_tanh, safe_min, safe_max)
            action = action_t.cpu().numpy().reshape(-1)
        elif policy_kind == "ckpt_linear":
            assert actor is not None
            obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                raw_action = actor.predict(obs_t, deterministic=True)
            action = ((raw_action.cpu().numpy().reshape(-1) + 1.0) / 2.0).clip(0.0, 1.0)
        else:
            raise ValueError(policy_kind)

        obs, reward, cost, terminated, truncated, info = env.step(action)
        total_steps += 1
        total_reward += float(reward)
        total_cost += float(cost)
        if float(info.get("comfort_in_warmup", 0.0)) < 0.5:
            active_steps += 1
            comfort_violations += int(float(info.get("comfort_violation", 0.0)) > 0.5)
        occupied_count += float(info.get("occupied_count", 0.0))
        discomfort_count += float(info.get("discomfort_count", 0.0))
        action_sum += float(np.sum(action))
        action_count += int(np.size(action))
        reward_economic += float(info.get("reward_economic", 0.0))
        reward_stability += float(info.get("reward_stability", 0.0))
        reward_renewable += float(info.get("reward_renewable", 0.0))
        reward_comfort += float(info.get("reward_comfort", 0.0))
        district_import_kwh += float(info.get("district_import_kwh", 0.0))
        district_net_kwh += float(info.get("district_net_kwh", 0.0))

    return RolloutResult(
        name=name,
        total_reward=total_reward,
        total_cost=total_cost,
        violation_rate=(comfort_violations / active_steps) if active_steps > 0 else 0.0,
        discomfort_rate=(discomfort_count / occupied_count) if occupied_count > 0 else 0.0,
        mean_cooling_action=(action_sum / action_count) if action_count > 0 else 0.0,
        reward_economic=reward_economic,
        reward_stability=reward_stability,
        reward_renewable=reward_renewable,
        reward_comfort=reward_comfort,
        district_import_kwh=district_import_kwh,
        district_net_kwh=district_net_kwh,
        total_steps=total_steps,
        active_steps=active_steps,
        comfort_violations=comfort_violations,
        discomfort_count=discomfort_count,
        occupied_count=occupied_count,
        kpis=_evaluate_citylearn_kpis(env),
    )


def main() -> None:
    args = parse_args()
    progress_path = args.run_dir / "progress.csv"
    best_epoch = args.best_epoch if args.best_epoch is not None else _best_epoch_from_progress(progress_path)
    final_epoch = args.final_epoch if args.final_epoch is not None else _latest_checkpoint_epoch(args.run_dir)
    best_ckpt = args.run_dir / "torch_save" / f"epoch-{best_epoch}.pt"
    final_ckpt = args.run_dir / "torch_save" / f"epoch-{final_epoch}.pt"

    # Auto-detect policy kind: masked (sigmoid) vs linear (ActionScale)
    ckpt_kind = args.policy_kind or "ckpt"

    # Infer algo name from run dir for display
    run_name = args.run_dir.parent.name if args.run_dir.parent else "RL"
    algo_label = run_name.split("-{")[0] if "-{" in run_name else "RL"

    results = [
        run_rollout(f"{algo_label} best", ckpt_kind, args.seed, args.run_dir, best_ckpt),
        run_rollout(f"{algo_label} final", ckpt_kind, args.seed, args.run_dir, final_ckpt),
        run_rollout("Cooling RBC", "rbc", args.seed, args.run_dir),
        run_rollout("RBC 2", "rbc2", args.seed, args.run_dir),
        run_rollout("RBC 3", "rbc3", args.seed, args.run_dir),
        run_rollout("Zero", "zero", args.seed, args.run_dir),
    ]

    rows = []
    for r in results:
        row = {
            "name": r.name,
            "total_reward": r.total_reward,
            "total_cost": r.total_cost,
            "violation_rate": r.violation_rate,
            "discomfort_rate": r.discomfort_rate,
            "mean_cooling_action": r.mean_cooling_action,
            "reward_economic": r.reward_economic,
            "reward_stability": r.reward_stability,
            "reward_renewable": r.reward_renewable,
            "reward_comfort": r.reward_comfort,
            "district_import_kwh": r.district_import_kwh,
            "district_net_kwh": r.district_net_kwh,
            "total_steps": r.total_steps,
            "active_steps": r.active_steps,
            "comfort_violations": r.comfort_violations,
            "discomfort_count": r.discomfort_count,
            "occupied_count": r.occupied_count,
        }
        for key, value in sorted(r.kpis.items()):
            row[f"kpi::{key}"] = value
        rows.append(row)

    payload = {"best_epoch": best_epoch, "final_epoch": final_epoch, "rows": rows}
    out_path = args.output_json or (args.run_dir / "eval_case_study.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"\nSaved JSON to {out_path}")


if __name__ == "__main__":
    main()
