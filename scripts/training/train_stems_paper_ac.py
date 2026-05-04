"""Smoke-first builder for a paper-style single-agent STEMS actor-critic.

This script intentionally validates the implementation in sections before a
full training loop is introduced:

1. exact 5-building STEMS observation contract
2. shared STEMS representation + actor/value heads
3. repo-aligned C0-C4 cost extraction and Lagrangian controller creation
4. one short rollout + one backward pass

The full training loop should be layered on top of this verified slice rather
than written as a monolithic first draft.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Register envs first (cmdp_env registers CityLearnSafety-V2G-v2)
import citylearn_safe.cmdp_env  # noqa: F401  @env_register side-effect

from citylearn_safe.cmdp_env import CityLearnCMDP
from citylearn_safe.pid_lagrange import PIDLagrange
from citylearn_safe.stems_5bld_factory import build_obs_index_5bld_spec, build_stems_encoder_5bld
from citylearn_safe.stems_paper_single_agent import build_paper_single_agent_stems_ac

DEFAULT_5BLD_SCHEMA = os.path.join(
    PROJECT_ROOT,
    "data",
    "citylearn_challenge_2022_phase_all_plus_evs",
    "schema_5buildings.json",
)

COST_KEYS = [
    "cost_ev_departure",
    "cost_ev_dense",
    "cost_stems_battery",
    "cost_stems_building_power",
    "cost_stems_grid_power",
]


@dataclass
class SmokeBatch:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    rew: torch.Tensor
    costs: torch.Tensor
    v_r: torch.Tensor
    v_cs: torch.Tensor


def _to_tensor(x, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def _extract_costs(info: dict) -> np.ndarray:
    return np.asarray([float(info.get(k, 0.0)) for k in COST_KEYS], dtype=np.float32)


def _make_lagranges() -> list[PIDLagrange]:
    # Mirrors current stable defaults and keeps the same five repo costs.
    limits = [5.0, 5.0, 5000.0, 1500.0, 1500.0]
    kp = [5.0, 0.1, 0.1, 0.3, 0.2]
    ki = [0.0, 0.01, 0.01, 0.03, 0.02]
    return [
        PIDLagrange(
            cost_limit=limits[i],
            pid_kp=kp[i],
            pid_ki=ki[i],
            pid_kd=0.0,
            pid_d_delay=10,
            pid_delta_p_ema_alpha=0.95,
            pid_delta_d_ema_alpha=0.95,
            penalty_max=3.0,
            lagrangian_multiplier_init=0.001,
        )
        for i in range(len(COST_KEYS))
    ]


@contextmanager
def _temporary_env_overrides(overrides: dict[str, str]):
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


def collect_smoke_rollout(
    model,
    env: CityLearnCMDP,
    device: torch.device,
    smoke_steps: int,
) -> SmokeBatch:
    obs, _ = env.reset()
    obs_list, act_list, logp_list, rew_list, cost_list, vr_list, vcs_list = ([] for _ in range(7))

    for _ in range(smoke_steps):
        obs_t = _to_tensor(obs, device).unsqueeze(0)
        out = model.act(obs_t, deterministic=False)
        action = out.action.squeeze(0).detach().cpu().numpy()
        next_obs, rew, cost, terminated, truncated, info = env.step(action)

        obs_list.append(obs_t.squeeze(0))
        act_list.append(out.action.squeeze(0))
        logp_list.append(out.log_prob.squeeze(0))
        rew_list.append(torch.tensor(float(rew), dtype=torch.float32, device=device))
        cost_list.append(torch.tensor(_extract_costs(info), dtype=torch.float32, device=device))
        vr_list.append(out.value_r.squeeze(0))
        vcs_list.append(torch.stack([v.squeeze(0) for v in out.values_c], dim=0))

        obs = next_obs
        done = bool(terminated.item() if hasattr(terminated, "item") else terminated) or bool(
            truncated.item() if hasattr(truncated, "item") else truncated
        )
        if done:
            break

    return SmokeBatch(
        obs=torch.stack(obs_list, dim=0),
        act=torch.stack(act_list, dim=0),
        logp=torch.stack(logp_list, dim=0),
        rew=torch.stack(rew_list, dim=0),
        costs=torch.stack(cost_list, dim=0),
        v_r=torch.stack(vr_list, dim=0),
        v_cs=torch.stack(vcs_list, dim=0),
    )


def run_smoke(smoke_steps: int, seed: int, schema_path: str) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    overrides = {
        "CITYLEARN_SCHEMA": schema_path,
        "CITYLEARN_TEMPORAL_WINDOW": "12",
        "STEMS_ENCODER_VERSION": "v3",
    }
    with _temporary_env_overrides(overrides):
        spec = build_obs_index_5bld_spec(schema_path=schema_path, temporal_window=12)
        if spec.num_buildings != 5:
            raise AssertionError(f"Expected 5 buildings, got {spec.num_buildings}")
        encoder = build_stems_encoder_5bld(spec)
        model = build_paper_single_agent_stems_ac(spec=spec, encoder=encoder, num_costs=len(COST_KEYS))
        device = torch.device("cpu")
        model.to(device)

        env = CityLearnCMDP("CityLearnSafety-V2G-v2")
        env_obs_dim = int(env.observation_space.shape[0])
        env_act_dim = int(env.action_space.shape[0])
        if env_obs_dim != spec.encoder_obs_dim:
            raise AssertionError(
                f"Live env obs_dim {env_obs_dim} does not match STEMS spec encoder_obs_dim {spec.encoder_obs_dim}."
            )
        if env_act_dim != spec.act_dim:
            raise AssertionError(f"Live env act_dim {env_act_dim} does not match STEMS spec act_dim {spec.act_dim}.")
        batch = collect_smoke_rollout(model=model, env=env, device=device, smoke_steps=smoke_steps)

        lagranges = _make_lagranges()
        lambdas = torch.tensor(
            [float(lag.lagrangian_multiplier) for lag in lagranges],
            dtype=torch.float32,
            device=device,
        )

        adv_r = batch.rew - batch.v_r.detach()
        adv_c = batch.costs - batch.v_cs.detach()
        constrained_adv = adv_r - (adv_c * lambdas.unsqueeze(0)).sum(dim=-1)

        logp_new, entropy, value_r_new, value_cs_new = model.evaluate_actions(batch.obs, batch.act)
        actor_loss = -(logp_new * constrained_adv.detach()).mean() - 0.001 * entropy.mean()
        reward_value_loss = F.mse_loss(value_r_new, batch.rew)
        cost_value_loss = sum(
            F.mse_loss(value_cs_new[i], batch.costs[:, i]) for i in range(len(COST_KEYS))
        )
        total_loss = actor_loss + reward_value_loss + cost_value_loss

        model.zero_grad(set_to_none=True)
        total_loss.backward()

        grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_sq += float(torch.sum(p.grad.detach() ** 2).cpu().item())
        grad_norm = grad_norm_sq ** 0.5

        print("=" * 60)
        print("STEMS Paper Single-Agent AC Smoke")
        print("=" * 60)
        print(f"schema_path: {schema_path}")
        print(f"obs_dim(current+forecast): {spec.obs_dim}")
        print(f"encoder_obs_dim(with history): {spec.encoder_obs_dim}")
        print(f"act_dim: {spec.act_dim}")
        print(f"num_buildings: {spec.num_buildings}")
        print(f"num_evs: {spec.num_evs}")
        print(f"smoke_steps_collected: {batch.obs.size(0)}")
        print(f"actor_action_shape: {tuple(batch.act.shape)}")
        print(f"reward_value_shape: {tuple(batch.v_r.shape)}")
        print(f"cost_value_shape: {tuple(batch.v_cs.shape)}")
        print(f"mean_reward: {float(batch.rew.mean().cpu().item()):.4f}")
        print(f"mean_costs: {[round(float(x), 4) for x in batch.costs.mean(dim=0).cpu().tolist()]}")
        print(f"actor_loss: {float(actor_loss.detach().cpu().item()):.6f}")
        print(f"reward_value_loss: {float(reward_value_loss.detach().cpu().item()):.6f}")
        print(f"cost_value_loss: {float(cost_value_loss.detach().cpu().item()):.6f}")
        print(f"total_grad_norm: {grad_norm:.6f}")
        print("status: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for paper-style single-agent STEMS actor-critic.")
    parser.add_argument("--smoke-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schema-path", type=str, default=DEFAULT_5BLD_SCHEMA)
    args = parser.parse_args()
    run_smoke(smoke_steps=args.smoke_steps, seed=args.seed, schema_path=args.schema_path)


if __name__ == "__main__":
    main()
