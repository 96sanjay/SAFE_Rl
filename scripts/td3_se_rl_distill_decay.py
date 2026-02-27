#!/usr/bin/env python3

import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---- use gymnasium (CityLearn default) ----
try:
    import gymnasium as gym
except:
    import gym


# ----------------------------
# Networks
# ----------------------------

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# ----------------------------
# Safe wrappers
# ----------------------------

def safe_reset(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def safe_step(env, action):
    out = env.step(action)

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, reward, done, info = out

    return obs, reward, done, info


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--w_start", type=float, default=5.0)
    parser.add_argument("--decay_tau", type=float, default=3000.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--env_id", type=str, default="CityLearnSafety-V2G-v2")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[INFO] Creating env: {args.env_id}")
    env = gym.make(args.env_id)

    obs = safe_reset(env)
    obs = np.array(obs, dtype=np.float32).flatten()

    obs_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"[RUN] obs_dim={obs_dim} act_dim={act_dim} device={device}")

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim, act_dim).to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=3e-4)

    gamma = 0.99
    buffer = []

    start_time = time.time()
    w_proj = args.w_start

    for t in range(1, args.steps + 1):

        obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)

        with torch.no_grad():
            u = actor(obs_tensor).cpu().numpy()[0]

        next_obs, reward, done, info = safe_step(env, u)
        next_obs = np.array(next_obs, dtype=np.float32).flatten()

        if t == 1:
            print("[DEBUG] info keys:", list(info.keys()))

        u_phi = info.get("shielded_action", u)
        delta = np.linalg.norm(u - u_phi)

        buffer.append((obs, u, u_phi, reward, next_obs, done))
        obs = next_obs

        if done:
            obs = safe_reset(env)
            obs = np.array(obs, dtype=np.float32).flatten()

        if len(buffer) > 256:

            batch_idx = np.random.choice(len(buffer), 64)
            batch = [buffer[i] for i in batch_idx]

            s, a, a_phi, r, s2, d = map(np.array, zip(*batch))

            s = torch.from_numpy(s).float().to(device)
            a_phi = torch.from_numpy(a_phi).float().to(device)
            r = torch.from_numpy(r).float().to(device).unsqueeze(1)
            s2 = torch.from_numpy(s2).float().to(device)
            d = torch.from_numpy(d).float().to(device).unsqueeze(1)

            # Critic
            with torch.no_grad():
                a2 = actor(s2)
                q_target = r + gamma * critic(s2, a2) * (1 - d)

            q = critic(s, actor(s))
            critic_loss = ((q - q_target) ** 2).mean()

            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()

            # Actor
            pred_a = actor(s)
            q_val = critic(s, pred_a)

            w_proj = args.w_start * math.exp(-t / args.decay_tau)

            diff = pred_a - a_phi.detach()
            norm = torch.norm(diff, dim=1, keepdim=True)

            proj_mask = (norm > args.tolerance).float()
            proj_loss = (proj_mask * (diff ** 2).sum(dim=1, keepdim=True)).mean()

            actor_loss = -q_val.mean() + w_proj * proj_loss

            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()

        if t % 200 == 0:
            elapsed = time.time() - start_time
            print(f"[t={t:6d}] delta={delta:.4f} w_proj={w_proj:.3f} elapsed={elapsed:.1f}s")

    print("[DONE]")


if __name__ == "__main__":
    main()

