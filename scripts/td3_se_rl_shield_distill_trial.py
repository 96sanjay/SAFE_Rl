#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, argparse, random
from typing import List, Tuple

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _unflatten_action(flat, action_space):
    flat = np.asarray(flat, dtype=np.float32).ravel()
    if not isinstance(action_space, list):
        return flat
    out, off = [], 0
    for sp in action_space:
        dim = int(np.prod(sp.shape))
        out.append(flat[off:off + dim].reshape(sp.shape))
        off += dim
    return out


def mlp(sizes: List[int], act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        a = act if i < len(sizes) - 2 else out_act
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if a is not None:
            layers.append(a())
    return nn.Sequential(*layers)


class SquashedActor(nn.Module):
    """
    Outputs actions with *per-dimension* squashing:
      - battery/cooling: tanh -> [-1, 1]
      - EV dims: sigmoid -> [0, 1]
    This matches your shield behavior (it clamps EV actions to >=0 anyway),
    so it reduces huge corrections early.
    """
    def __init__(self, obs_dim: int, act_dim: int, ev_idx: np.ndarray, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], act=nn.ReLU, out_act=None)
        self.register_buffer("ev_idx_t", torch.as_tensor(ev_idx.astype(np.int64)))
        self.act_dim = act_dim

        # Small init helps avoid wild actions early
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.8)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.net(obs)  # unconstrained
        a = torch.tanh(raw)  # default [-1,1]
        if self.ev_idx_t.numel() > 0:
            a_ev = torch.sigmoid(raw.index_select(-1, self.ev_idx_t))  # [0,1]
            a = a.clone()
            a.index_copy_(-1, self.ev_idx_t, a_ev)
        return a


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1], act=nn.ReLU, out_act=None)
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1], act=nn.ReLU, out_act=None)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, max_size: int, obs_dim: int, act_dim: int, device: torch.device):
        self.max = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        self.obs = torch.zeros((max_size, obs_dim), device=device)
        self.act = torch.zeros((max_size, act_dim), device=device)     # proposed u
        self.u_phi = torch.zeros((max_size, act_dim), device=device)   # projected u_phi (teacher)
        self.rew = torch.zeros((max_size,), device=device)
        self.nxt = torch.zeros((max_size, obs_dim), device=device)
        self.done = torch.zeros((max_size,), device=device)

    def add(self, obs, act, u_phi, rew, nxt, done):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.u_phi[i] = u_phi
        self.rew[i] = rew
        self.nxt[i] = nxt
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch: int):
        idx = torch.randint(0, self.size, (batch,), device=self.device)
        return (self.obs[idx], self.act[idx], self.u_phi[idx],
                self.rew[idx], self.nxt[idx], self.done[idx])


def make_env(env_id: str):
    try:
        return gym.make(env_id)
    except Exception as e:
        print(f"[WARN] gym.make({env_id}) failed: {e}")
        print("[WARN] Falling back to scripts.make_env.make_base_env()")
        from scripts.make_env import make_base_env
        return make_base_env()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="CityLearnSafety-V2G-v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--total_steps", type=int, default=8000)
    ap.add_argument("--start_updates", type=int, default=1500)
    ap.add_argument("--updates_per_step", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--buffer_size", type=int, default=200000)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)

    ap.add_argument("--actor_lr", type=float, default=3e-4)
    ap.add_argument("--critic_lr", type=float, default=1e-3)

    ap.add_argument("--policy_noise", type=float, default=0.2)
    ap.add_argument("--noise_clip", type=float, default=0.5)
    ap.add_argument("--explore_noise", type=float, default=0.05)  # LOWER by default
    ap.add_argument("--policy_delay", type=int, default=2)

    # distillation controls
    ap.add_argument("--w_proj", type=float, default=0.10)
    ap.add_argument("--distill_after", type=int, default=2000)
    ap.add_argument("--distill_ramp", type=int, default=2000)
    ap.add_argument("--distill_only_when_intervened", action="store_true")

    # shield toggles (for this trial, keep C3/C4 OFF unless you intentionally want them)
    ap.add_argument("--shield_enable_c3", type=int, default=int(os.environ.get("SHIELD_ENABLE_C3", "0")))
    ap.add_argument("--shield_enable_c4", type=int, default=int(os.environ.get("SHIELD_ENABLE_C4", "0")))
    ap.add_argument("--shield_verbose", type=int, default=0)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    try:
        from citylearn_safe.action_projection import ActionProjectionWrapper
    except Exception as e:
        print("[ERR] Cannot import ActionProjectionWrapper:", e)
        sys.exit(1)

    env = make_env(args.env_id)
    obs, _ = env.reset()

    obs_dim = int(np.asarray(obs).size)
    if isinstance(env.action_space, list):
        act_dim = int(sum(int(np.prod(sp.shape)) for sp in env.action_space))
        low = np.concatenate([np.asarray(sp.low).ravel() for sp in env.action_space]).astype(np.float32)
        high = np.concatenate([np.asarray(sp.high).ravel() for sp in env.action_space]).astype(np.float32)
    else:
        act_dim = int(np.prod(env.action_space.shape))
        low = np.asarray(env.action_space.low).ravel().astype(np.float32)
        high = np.asarray(env.action_space.high).ravel().astype(np.float32)

    print(f"[RUN] obs_dim={obs_dim} act_dim={act_dim} device={device}")

    shield = ActionProjectionWrapper(
        env,
        enable_c3=(args.shield_enable_c3 == 1),
        enable_c4=(args.shield_enable_c4 == 1),
        verbose=args.shield_verbose
    )
    shield._compile()
    ev_idx = np.array([e["gidx"] for e in shield._mapping["ev"]], dtype=int) if shield._mapping else np.array([], dtype=int)

    actor = SquashedActor(obs_dim, act_dim, ev_idx=ev_idx).to(device)
    actor_t = SquashedActor(obs_dim, act_dim, ev_idx=ev_idx).to(device)
    actor_t.load_state_dict(actor.state_dict())

    critic = Critic(obs_dim, act_dim).to(device)
    critic_t = Critic(obs_dim, act_dim).to(device)
    critic_t.load_state_dict(critic.state_dict())

    aopt = optim.Adam(actor.parameters(), lr=args.actor_lr)
    copt = optim.Adam(critic.parameters(), lr=args.critic_lr)

    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim, device)

    def clamp_to_space(u: np.ndarray) -> np.ndarray:
        return np.clip(u, low, high)

    o = torch.as_tensor(np.asarray(obs, dtype=np.float32).ravel(), device=device)

    ma_rew = 0.0
    ma_delta = 0.0
    ma_proj = 0.0
    t0 = time.time()

    for t in range(1, args.total_steps + 1):
        # propose action
        actor.eval()
        with torch.no_grad():
            u = actor(o.unsqueeze(0)).squeeze(0).cpu().numpy()

        # exploration noise (small)
        u = u + np.random.normal(0.0, args.explore_noise, size=u.shape).astype(np.float32)
        u = clamp_to_space(u)

        # project (teacher)
        u_phi, _si = shield._project_action(u.astype(np.float32))
        u_phi = clamp_to_space(u_phi.astype(np.float32))

        delta = float(np.linalg.norm(u_phi - u))
        a_out = _unflatten_action(u_phi, env.action_space) if isinstance(env.action_space, list) else u_phi

        obs2, rew, term, trunc, _info = env.step(a_out)
        done = float(term or trunc)

        o2 = torch.as_tensor(np.asarray(obs2, dtype=np.float32).ravel(), device=device)
        r = torch.tensor(float(rew), device=device)
        d = torch.tensor(done, device=device)

        buf.add(
            o,
            torch.as_tensor(u, dtype=torch.float32, device=device),
            torch.as_tensor(u_phi, dtype=torch.float32, device=device),
            r,
            o2,
            d
        )

        o = o2
        if done:
            obs, _ = env.reset()
            o = torch.as_tensor(np.asarray(obs, dtype=np.float32).ravel(), device=device)

        # moving averages
        ma_rew = 0.98 * ma_rew + 0.02 * float(rew)
        ma_delta = 0.98 * ma_delta + 0.02 * delta
        ma_proj = 0.98 * ma_proj + 0.02 * (1.0 if delta > 1e-4 else 0.0)

        # updates
        if t >= args.start_updates and buf.size >= args.batch:
            for _ in range(args.updates_per_step):
                obs_b, act_b, u_phi_b, rew_b, nxt_b, done_b = buf.sample(args.batch)

                with torch.no_grad():
                    a2 = actor_t(nxt_b)
                    noise = (torch.randn_like(a2) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                    a2 = (a2 + noise).clamp(-1.0, 1.0)

                    q1_t, q2_t = critic_t(nxt_b, a2)
                    q_t = torch.min(q1_t, q2_t)
                    y = rew_b + (1.0 - done_b) * args.gamma * q_t

                q1, q2 = critic(obs_b, act_b)
                c_loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
                copt.zero_grad(set_to_none=True)
                c_loss.backward()
                copt.step()

                if (t % args.policy_delay) == 0:
                    a_now = actor(obs_b)
                    a_loss_rl = (-critic.q1_only(obs_b, a_now)).mean()

                    # distillation weight schedule
                    if t < args.distill_after:
                        w_eff = 0.0
                    else:
                        prog = min(1.0, (t - args.distill_after) / max(1, args.distill_ramp))
                        w_eff = args.w_proj * prog

                    if w_eff > 0.0:
                        if args.distill_only_when_intervened:
                            mask = (torch.norm(u_phi_b - act_b, dim=-1) > 1e-4).float()
                            if mask.sum() > 0:
                                mse = ((a_now - u_phi_b.detach()) ** 2).mean(dim=-1)
                                a_loss_proj = (mse * (mask / mask.sum())).sum()
                            else:
                                a_loss_proj = torch.zeros((), device=device)
                        else:
                            a_loss_proj = ((a_now - u_phi_b.detach()) ** 2).mean()
                    else:
                        a_loss_proj = torch.zeros((), device=device)

                    a_loss = a_loss_rl + w_eff * a_loss_proj

                    aopt.zero_grad(set_to_none=True)
                    a_loss.backward()
                    aopt.step()

                    with torch.no_grad():
                        for p, tp in zip(critic.parameters(), critic_t.parameters()):
                            tp.data.mul_(1.0 - args.tau).add_(args.tau * p.data)
                        for p, tp in zip(actor.parameters(), actor_t.parameters()):
                            tp.data.mul_(1.0 - args.tau).add_(args.tau * p.data)

        if t % 200 == 0:
            elapsed = time.time() - t0
            print(f"[t={t:6d}] ma_rew={ma_rew:+.3f} ma_delta={ma_delta:.4f} proj_rate~={ma_proj:.3f} buf={buf.size} elapsed={elapsed:.1f}s")

    print("[DONE] Watch for ma_delta and proj_rate to start dropping after distill_after (policy learns to stop triggering the shield).")
    env.close()


if __name__ == "__main__":
    main()
