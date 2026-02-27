#!/usr/bin/env python3
"""
TD3 + NFWPO-Style Shield Distillation for Safe V2G Energy Management

Based on:
- Lin et al. (UAI 2021) NFWPO: decouple constraint satisfaction from policy update
- Markgraf et al. (2025) SE-RL: safe environment RL with action projection

actor_loss = -Q(s, u) + w_proj * ||u - stop_grad(u_phi)||^2
"""

import os, sys, time, math, argparse, copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT: Manual wrapper stack (THE CRITICAL FIX)
# ═══════════════════════════════════════════════════════════════════

def make_shielded_env():
    """
    Build:  CityLearnEnv → SafetyEnvV3 → ActionProjectionWrapper
            → (optional) ShieldInfoWrapper → ForecastObsWrapper
    """
    project_root = os.environ.get("CITYLEARN_PROJECT_ROOT", os.getcwd())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    schema_path = os.environ.get(
        "CITYLEARN_SCHEMA",
        "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
    )
    if not os.path.isabs(schema_path):
        schema_path = os.path.join(project_root, schema_path)

    print(f"[ENV] Schema: {schema_path}")
    assert os.path.exists(schema_path), f"Schema not found: {schema_path}"

    # --- Base CityLearn ---
    from citylearn.citylearn import CityLearnEnv
    base_env = CityLearnEnv(schema=schema_path, central_agent=True)
    print(f"[ENV] CityLearnEnv: {len(base_env.buildings)} buildings")

    # --- SafetyEnvV3 ---
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    safety_env = CityLearnSafetyEnvV3(
        base_env,
        soc_min=float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0")),
        soc_max=float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95")),
    )
    print(f"[ENV] SafetyEnvV3 wrapped")

    # --- ActionProjectionWrapper (SHIELD) ---
    use_shield = os.environ.get("CITYLEARN_USE_SHIELD", "1") == "1"
    if use_shield:
        from citylearn_safe.action_projection import ActionProjectionWrapper
        shield_env = ActionProjectionWrapper(safety_env)
        print(f"[ENV] ActionProjectionWrapper ACTIVE")
        print(f"[ENV]   C3={os.environ.get('SHIELD_ENABLE_C3','0')} "
              f"C4={os.environ.get('SHIELD_ENABLE_C4','0')}")

        # Wrap with ShieldInfoWrapper to guarantee shielded_action in info
        try:
            from citylearn_safe.shield_info_wrapper import ShieldInfoWrapper
            shield_env = ShieldInfoWrapper(shield_env)
            print(f"[ENV] ShieldInfoWrapper added (guarantees info key)")
        except ImportError:
            print(f"[ENV] ShieldInfoWrapper not found, relying on AP patch")
    else:
        shield_env = safety_env
        print(f"[ENV] WARNING: Shield DISABLED")

    # --- ForecastObsWrapper ---
    try:
        from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
        final_env = ForecastObsWrapper(shield_env)
        print(f"[ENV] ForecastObsWrapper added")
    except ImportError:
        print(f"[ENV] ForecastObsWrapper not found, skipping")
        final_env = shield_env

    return final_env


# ═══════════════════════════════════════════════════════════════════
# NETWORKS
# ═══════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(512, 256)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(512, 256)):
        super().__init__()
        self.q1 = self._build(obs_dim, act_dim, hidden)
        self.q2 = self._build(obs_dim, act_dim, hidden)

    def _build(self, obs_dim, act_dim, hidden):
        layers = []
        prev = obs_dim + act_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, s, a):
        return self.q1(torch.cat([s, a], dim=-1))


# ═══════════════════════════════════════════════════════════════════
# REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs     = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act     = np.zeros((capacity, act_dim), dtype=np.float32)
        self.act_s   = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew     = np.zeros((capacity, 1),       dtype=np.float32)
        self.nxt     = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done    = np.zeros((capacity, 1),       dtype=np.float32)

    def add(self, obs, act, act_safe, rew, nxt, done):
        i = self.ptr
        self.obs[i]   = obs
        self.act[i]   = act
        self.act_s[i] = act_safe
        self.rew[i]   = rew
        self.nxt[i]   = nxt
        self.done[i]  = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n, dev):
        idx = np.random.randint(0, self.size, size=n)
        return (torch.from_numpy(self.obs[idx]).to(dev),
                torch.from_numpy(self.act[idx]).to(dev),
                torch.from_numpy(self.act_s[idx]).to(dev),
                torch.from_numpy(self.rew[idx]).to(dev),
                torch.from_numpy(self.nxt[idx]).to(dev),
                torch.from_numpy(self.done[idx]).to(dev))


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def safe_reset(env):
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    info = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
    return np.asarray(obs, dtype=np.float32).flatten(), info


def safe_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, rew, term, trunc, info = out
        done = term or trunc
    else:
        obs, rew, done, info = out
    obs = np.asarray(obs, dtype=np.float32).flatten()
    rew = float(np.sum(rew)) if isinstance(rew, (list, np.ndarray)) else float(rew)
    return obs, rew, done, info


def get_shielded_action(info, raw):
    """Extract shield-projected action from info dict."""
    for key in ["shielded_action", "projected_action", "ap_projected_action", "safe_action"]:
        if key in info:
            sa = info[key]
            if sa is not None:
                sa = np.asarray(sa, dtype=np.float32).flatten()
                if sa.shape == raw.shape:
                    return sa
    return raw


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--steps",       type=int,   default=100000)
    pa.add_argument("--warmup",      type=int,   default=3000)
    pa.add_argument("--batch_size",  type=int,   default=256)
    pa.add_argument("--buffer_size", type=int,   default=200000)
    pa.add_argument("--gamma",       type=float, default=0.995)
    pa.add_argument("--tau_target",  type=float, default=0.005)
    pa.add_argument("--policy_delay",type=int,   default=2)
    pa.add_argument("--lr_actor",    type=float, default=3e-4)
    pa.add_argument("--lr_critic",   type=float, default=3e-4)
    pa.add_argument("--expl_noise",  type=float, default=0.1)
    pa.add_argument("--target_noise",type=float, default=0.2)
    pa.add_argument("--noise_clip",  type=float, default=0.5)
    pa.add_argument("--w_start",     type=float, default=10.0)
    pa.add_argument("--w_end",       type=float, default=0.5)
    pa.add_argument("--decay_tau",   type=float, default=30000.0)
    pa.add_argument("--tolerance",   type=float, default=0.02)
    pa.add_argument("--log_every",   type=int,   default=500)
    pa.add_argument("--device",      type=str,   default="cpu")
    pa.add_argument("--seed",        type=int,   default=42)
    args = pa.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- Create env ---
    env = make_shielded_env()
    obs, info = safe_reset(env)
    obs_dim = obs.shape[0]
    act_dim = (sum(int(np.prod(s.shape)) for s in env.action_space)
               if isinstance(env.action_space, list)
               else int(np.prod(env.action_space.shape)))

    print(f"\n{'='*60}")
    print(f"  TD3 + NFWPO Shield Distillation")
    print(f"  obs={obs_dim}  act={act_dim}  device={device}")
    print(f"  steps={args.steps}  w=[{args.w_start}→{args.w_end}]  τ={args.decay_tau}")
    print(f"{'='*60}")

    # --- Verify shield ---
    obs_v, _ = safe_reset(env)
    _, _, _, info_v = safe_step(env, np.zeros(act_dim, dtype=np.float32))
    shield_keys = [k for k in info_v if k in
        ("shielded_action","projected_action","ap_action_delta_l2","ap_projected_action")]
    if shield_keys:
        print(f"[✓] Shield keys found: {shield_keys}")
        for k in shield_keys:
            v = info_v[k]
            if isinstance(v, (int, float)):
                print(f"    {k} = {v}")
            elif hasattr(v, 'shape'):
                print(f"    {k} shape={np.asarray(v).shape}")
    else:
        print(f"\n{'!'*60}")
        print(f"  WARNING: No shield keys in info!")
        print(f"  info keys: {sorted(info_v.keys())[:15]}...")
        print(f"  Distillation will be INACTIVE (pure TD3)")
        print(f"{'!'*60}\n")

    obs, _ = safe_reset(env)

    # --- Networks ---
    actor        = Actor(obs_dim, act_dim).to(device)
    actor_tgt    = copy.deepcopy(actor)
    critic       = TwinCritic(obs_dim, act_dim).to(device)
    critic_tgt   = copy.deepcopy(critic)
    opt_a = optim.Adam(actor.parameters(),  lr=args.lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=args.lr_critic)

    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    # --- Tracking ---
    ep_rew, ep_steps, ep_count = 0.0, 0, 0
    ep_interventions, ep_delta_sum = 0, 0.0
    ma_rew   = deque(maxlen=10)
    ma_delta = deque(maxlen=2000)
    ma_proj  = deque(maxlen=2000)
    t0 = time.time()
    n_updates = 0

    print(f"\n[TRAIN] Starting ({args.steps} steps, warmup={args.warmup})...\n")

    for t in range(1, args.steps + 1):

        # --- Action ---
        if t <= args.warmup:
            u = np.random.uniform(-1, 1, size=act_dim).astype(np.float32)
        else:
            with torch.no_grad():
                ot = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                u = actor(ot).cpu().numpy()[0]
            u += np.random.normal(0, args.expl_noise, size=act_dim).astype(np.float32)
            u = np.clip(u, -1.0, 1.0)

        # --- Step (shield projects internally) ---
        nxt, rew, done, info = safe_step(env, u)
        u_phi = get_shielded_action(info, u)
        delta = float(np.linalg.norm(u - u_phi))

        ma_delta.append(delta)
        intervened = delta > args.tolerance
        ma_proj.append(1.0 if intervened else 0.0)
        ep_interventions += int(intervened)
        ep_delta_sum += delta

        buf.add(obs, u, u_phi, rew, nxt, done)
        obs = nxt
        ep_rew += rew
        ep_steps += 1

        if done:
            ma_rew.append(ep_rew)
            avg_r = np.mean(ma_rew)
            pr = ep_interventions / max(1, ep_steps)
            ad = ep_delta_sum / max(1, ep_steps)
            print(f"[EP {ep_count:4d}] steps={ep_steps:5d} R={ep_rew:10.1f} "
                  f"avgR={avg_r:10.1f} proj_rate={pr:.3f} avg_δ={ad:.4f}")
            ep_count += 1
            ep_rew, ep_steps, ep_interventions, ep_delta_sum = 0.0, 0, 0, 0.0
            obs, _ = safe_reset(env)

        # --- Train ---
        if t > args.warmup and buf.size >= args.batch_size:
            s, a, a_phi, r, s2, d = buf.sample(args.batch_size, device)

            # ── Critic ──
            with torch.no_grad():
                noise = (torch.randn_like(a) * args.target_noise
                        ).clamp(-args.noise_clip, args.noise_clip)
                a2 = (actor_tgt(s2) + noise).clamp(-1, 1)
                q1t, q2t = critic_tgt(s2, a2)
                y = r + args.gamma * torch.min(q1t, q2t) * (1 - d)

            q1, q2 = critic(s, a_phi)   # train on SAFE actions
            c_loss = ((q1 - y)**2).mean() + ((q2 - y)**2).mean()
            opt_c.zero_grad(); c_loss.backward(); opt_c.step()

            # ── Actor (delayed) ──
            if t % args.policy_delay == 0:
                n_updates += 1
                pred = actor(s)

                # RL objective
                rl_loss = -critic.q1_only(s, pred).mean()

                # NFWPO distillation (Lin et al. Eq. 14)
                w = args.w_end + (args.w_start - args.w_end) * math.exp(-t / args.decay_tau)
                diff = pred - a_phi
                norm = torch.norm(diff, dim=1, keepdim=True)
                mask = (norm > args.tolerance).float()
                proj_loss = (mask * (diff**2).sum(dim=1, keepdim=True)).mean()

                a_loss = rl_loss + w * proj_loss
                opt_a.zero_grad(); a_loss.backward(); opt_a.step()

                # Soft update targets
                with torch.no_grad():
                    for p, pt in zip(actor.parameters(), actor_tgt.parameters()):
                        pt.data.mul_(1 - args.tau_target).add_(p.data, alpha=args.tau_target)
                    for p, pt in zip(critic.parameters(), critic_tgt.parameters()):
                        pt.data.mul_(1 - args.tau_target).add_(p.data, alpha=args.tau_target)

        # --- Log ---
        if t % args.log_every == 0:
            el = time.time() - t0
            ad = np.mean(ma_delta) if ma_delta else 0
            ap = np.mean(ma_proj)  if ma_proj  else 0
            w  = args.w_end + (args.w_start - args.w_end) * math.exp(-t / args.decay_tau)
            c1 = float(info.get("cost_ev_departure", 0))
            c3 = float(info.get("cost_stems_building_power", 0))
            c4 = float(info.get("cost_stems_grid_power", 0))
            print(f"[t={t:7d}] δ={ad:.4f} proj={ap:.3f} w={w:.2f} "
                  f"C1={c1:.2f} C3={c3:.2f} C4={c4:.2f} {el:.0f}s")

    # --- Save ---
    sd = os.path.join("runs", "td3_nfwpo_distill")
    os.makedirs(sd, exist_ok=True)
    sp = os.path.join(sd, "final_actor.pt")
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "obs_dim": obs_dim, "act_dim": act_dim, "args": vars(args)}, sp)
    print(f"\n[DONE] Saved → {sp}")
    print(f"[DONE] Episodes={ep_count}  Updates={n_updates}")


if __name__ == "__main__":
    main()
