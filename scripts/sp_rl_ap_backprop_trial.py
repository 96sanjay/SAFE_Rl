#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, argparse, random
from typing import List, Tuple, Dict

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _unwrap_to_citylearn(env):
    cur = env
    seen = set()
    for _ in range(80):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        blds = getattr(cur, "buildings", None)
        ts = getattr(cur, "time_step", None)
        if blds is not None and hasattr(blds, "__len__") and len(blds) > 0 and ts is not None:
            return cur
        for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None


def _flatten_action(action, action_space):
    if isinstance(action, (list, tuple)):
        return np.concatenate([np.asarray(a, dtype=float).ravel() for a in action])
    return np.asarray(action, dtype=float).ravel()


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
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if a is not None:
            layers.append(a())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], act=nn.ReLU, out_act=None)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1], act=nn.ReLU, out_act=None)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, max_size: int, obs_dim: int, act_dim: int, device: torch.device):
        self.max = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        self.obs = torch.zeros((max_size, obs_dim), device=device)
        self.act_rl = torch.zeros((max_size, act_dim), device=device)
        self.act_phi = torch.zeros((max_size, act_dim), device=device)
        self.rew = torch.zeros((max_size,), device=device)
        self.nxt = torch.zeros((max_size, obs_dim), device=device)
        self.done = torch.zeros((max_size,), device=device)

    def add(self, obs, act_rl, act_phi, rew, nxt, done):
        i = self.ptr
        self.obs[i] = obs
        self.act_rl[i] = act_rl
        self.act_phi[i] = act_phi
        self.rew[i] = rew
        self.nxt[i] = nxt
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch: int):
        idx = torch.randint(0, self.size, (batch,), device=self.device)
        return (self.obs[idx], self.act_rl[idx], self.act_phi[idx],
                self.rew[idx], self.nxt[idx], self.done[idx])


class DiffProjector:
    """
    Differentiable projection (paper-style):
      minimize ||z - u||^2 + w_slack * (s_grid + mean(s_bld))
      s.t. hard C1/C2, soft C3/C4 via slack
    Uses cvxpylayers with SCS (supported solver).
    """
    def __init__(self, env: gym.Env, w_slack: float, verbose: int = 0):
        self.env = env
        self.w_slack = float(w_slack)
        self.verbose = int(verbose)
        self._built = False

        self.soc_low = float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0"))
        self.soc_high = float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95"))
        self.p_build = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "2.273834"))
        self.p_grid = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "27.127751"))

        self.map = None
        self.layer = None

    def _build_mapping(self):
        city = _unwrap_to_citylearn(self.env)
        if city is None:
            raise RuntimeError("Cannot unwrap to CityLearn env")
        names_raw = getattr(city, "action_names", None)
        buildings = list(getattr(city, "buildings", []))
        nb = len(buildings)

        # central-agent flatten fix (same spirit as your wrapper)
        is_central = (isinstance(names_raw, list) and len(names_raw) == 1
                      and isinstance(names_raw[0], list) and nb > 1 and len(names_raw[0]) > nb)
        if is_central:
            flat = names_raw[0]
            batt_pos = [i for i, n in enumerate(flat) if str(n).lower() == "electrical_storage"]
            if len(batt_pos) == nb:
                rec = []
                for b in range(nb):
                    st = batt_pos[b]
                    en = batt_pos[b + 1] if b + 1 < nb else len(flat)
                    rec.append(list(flat[st:en]))
                names_raw = rec

        g = 0
        batt_g, batt_b = [], []
        ev_g, ev_b, ev_local = [], [], []
        for b_idx, sub in enumerate(names_raw):
            if not isinstance(sub, list):
                sub = [sub]
            local = 0
            batt_seen = False
            for n in sub:
                nl = str(n).lower()
                if nl == "electrical_storage" and not batt_seen:
                    batt_g.append(g); batt_b.append(b_idx)
                    batt_seen = True
                if "electric_vehicle_storage_charger_" in nl:
                    ev_g.append(g); ev_b.append(b_idx); ev_local.append(local)
                    local += 1
                g += 1

        if self.verbose:
            print(f"[DiffProj] nb={nb} total_dim={g} batt={len(batt_g)} ev={len(ev_g)}")
        return {
            "nb": nb, "total": g,
            "batt_g": np.asarray(batt_g, dtype=int), "batt_b": np.asarray(batt_b, dtype=int),
            "ev_g": np.asarray(ev_g, dtype=int), "ev_b": np.asarray(ev_b, dtype=int), "ev_local": np.asarray(ev_local, dtype=int),
        }

    def _c1_min_actions(self, city) -> np.ndarray:
        buildings = list(getattr(city, "buildings", []))
        t_now = int(getattr(city, "time_step", 0))
        t_idx = max(0, t_now - 1)

        ev_b = self.map["ev_b"]
        ev_local = self.map["ev_local"]
        n_ev = len(ev_b)

        mins = np.zeros(n_ev, dtype=float)
        for i in range(n_ev):
            b_idx = int(ev_b[i]); local = int(ev_local[i])
            chargers = getattr(buildings[b_idx], "electric_vehicle_chargers", None) or []
            if local >= len(chargers):
                continue
            ch = chargers[local]
            sim = getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))
            if sim is None:
                continue
            try:
                state_arr = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                dep_arr = np.asarray(getattr(sim, "_electric_vehicle_departure_time"), dtype=float)
                req_arr = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
                ev_id_arr = getattr(sim, "_electric_vehicle_id")
            except Exception:
                continue

            if t_now >= len(state_arr) or float(state_arr[t_now]) != 1.0:
                continue
            if ev_id_arr[t_now] is None:
                continue

            dep = float(dep_arr[t_now]) if t_now < len(dep_arr) else float("nan")
            T = max(1.0, dep) if np.isfinite(dep) and dep > 0 else 1.0
            req_soc = float(req_arr[t_now]) if t_now < len(req_arr) else 1.0
            req_soc = float(np.clip(req_soc if np.isfinite(req_soc) else 1.0, 0.0, 1.0))

            ev_soc, ev_cap = 0.0, 0.0
            ev_obj = getattr(ch, "connected_electric_vehicle", None)
            if ev_obj is not None:
                batt = getattr(ev_obj, "battery", None)
                if batt is not None:
                    ev_cap = float(getattr(batt, "capacity", 0) or 0.0)
                    soc_data = getattr(batt, "soc", None)
                    if soc_data is not None:
                        soc_np = np.asarray(soc_data, dtype=float).ravel()
                        if 0 <= t_idx < len(soc_np):
                            ev_soc = float(np.clip(soc_np[t_idx], 0.0, 1.0))
            if ev_cap <= 0:
                continue

            max_p = getattr(ch, "max_charging_power", getattr(ch, "_Charger__max_charging_power", 0))
            if isinstance(max_p, np.ndarray):
                max_p = float(max_p.ravel()[0])
            max_p = float(max_p or 0.0)
            if max_p <= 0:
                continue

            eff = 0.95
            try:
                eff = float(ch.get_efficiency(1.0, True))
            except Exception:
                pass
            eff = eff if np.isfinite(eff) and eff > 0 else 0.95

            deficit_kwh = max(0.0, (req_soc - ev_soc) * ev_cap)
            if deficit_kwh <= 1e-9:
                continue
            mins[i] = float(np.clip(deficit_kwh / max(1e-9, max_p * eff * T), 0.0, 1.0))

        return mins

    def build(self):
        self.map = self._build_mapping()
        nb = int(self.map["nb"])
        n_total = int(self.map["total"])
        batt_g = self.map["batt_g"]; batt_b = self.map["batt_b"]
        ev_g = self.map["ev_g"]; ev_b = self.map["ev_b"]

        n_batt = len(batt_g)
        n_ev = len(ev_g)

        A_batt = np.zeros((nb, n_batt))
        for j, b in enumerate(batt_b):
            A_batt[int(b), j] = 1.0
        A_ev = np.zeros((nb, n_ev))
        for j, b in enumerate(ev_b):
            A_ev[int(b), j] = 1.0

        z = cp.Variable(n_total)
        s_bld = cp.Variable(nb, nonneg=True)
        s_grid = cp.Variable(1, nonneg=True)

        u = cp.Parameter(n_total)
        base = cp.Parameter(nb)
        soc0 = cp.Parameter(n_batt)
        soc_scale = cp.Parameter(n_batt, nonneg=True)
        batt_gain = cp.Parameter(n_batt, nonneg=True)
        ev_gain = cp.Parameter(n_ev, nonneg=True)
        c1min = cp.Parameter(n_ev, nonneg=True)
        p_build = cp.Parameter(nonneg=True)
        p_grid = cp.Parameter(nonneg=True)
        w_slack = cp.Parameter(nonneg=True)

        cons = [z >= -1.0, z <= 1.0]

        if n_ev > 0:
            z_ev = z[ev_g]
            cons += [z_ev >= 0.0, z_ev <= 1.0, z_ev >= c1min]

        if n_batt > 0:
            z_b = z[batt_g]
            soc_next = soc0 + cp.multiply(z_b, soc_scale)
            cons += [soc_next >= self.soc_low, soc_next <= self.soc_high]

        net = base
        if n_batt > 0:
            net = net + A_batt @ cp.multiply(z[batt_g], batt_gain)
        if n_ev > 0:
            net = net + A_ev @ cp.multiply(z[ev_g], ev_gain)

        cons += [net <= p_build + s_bld, net >= -p_build - s_bld]
        grid = cp.sum(net)
        cons += [grid <= p_grid + s_grid[0], grid >= -p_grid - s_grid[0]]

        obj = cp.sum_squares(z - u) + w_slack * (s_grid[0] + (1.0 / max(1, nb)) * cp.sum(s_bld))
        prob = cp.Problem(cp.Minimize(obj), cons)

        # Use SCS (supported by cvxpylayers); pass solver args at call-time.
        self.layer = CvxpyLayer(
            prob,
            parameters=[u, base, soc0, soc_scale, batt_gain, ev_gain, c1min, p_build, p_grid, w_slack],
            variables=[z, s_bld, s_grid]
        )
        self._built = True

    def project(self, u_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self._built:
            self.build()

        city = _unwrap_to_citylearn(self.env)
        buildings = list(getattr(city, "buildings", []))
        nb = int(self.map["nb"])

        t_now = int(getattr(city, "time_step", 0))
        t_idx = max(0, t_now - 1)

        batt_g = self.map["batt_g"]; batt_b = self.map["batt_b"]
        ev_g = self.map["ev_g"]; ev_b = self.map["ev_b"]; ev_local = self.map["ev_local"]

        # base net
        base_net = np.zeros(nb, dtype=float)
        for b_idx, b in enumerate(buildings):
            nsl = np.asarray(getattr(b, "_Building__energy_to_non_shiftable_load", []), dtype=float)
            sg = np.asarray(getattr(b, "_Building__solar_generation", []), dtype=float)
            T_max = min(len(nsl), len(sg))
            if T_max > 0:
                idx = t_now if t_now < T_max else T_max - 1
                base_net[b_idx] = float(nsl[idx] + sg[idx])

        # battery params
        n_batt = len(batt_g)
        soc0 = np.zeros(n_batt, dtype=float)
        soc_scale = np.ones(n_batt, dtype=float)
        batt_gain = np.ones(n_batt, dtype=float)
        for i in range(n_batt):
            b_idx = int(batt_b[i])
            es = getattr(buildings[b_idx], "electrical_storage", None)
            if es is None:
                continue
            cap = float(getattr(es, "capacity", 6.4) or 6.4)
            nom = float(getattr(es, "nominal_power", 5.0) or 5.0)
            batt_gain[i] = max(1e-6, nom)
            soc_scale[i] = max(1e-6, nom / max(1e-6, cap))
            soc_arr = np.asarray(getattr(es, "soc", []), dtype=float).ravel()
            soc0[i] = float(np.clip(soc_arr[t_idx], 0.0, 1.0)) if 0 <= t_idx < len(soc_arr) else 0.5

        # EV gains + C1 min
        n_ev = len(ev_g)
        ev_gain = np.zeros(n_ev, dtype=float)
        if n_ev > 0:
            for i in range(n_ev):
                b_idx = int(ev_b[i]); local = int(ev_local[i])
                chargers = getattr(buildings[b_idx], "electric_vehicle_chargers", None) or []
                if local < len(chargers):
                    mp = getattr(chargers[local], "max_charging_power", getattr(chargers[local], "_Charger__max_charging_power", 0))
                    if isinstance(mp, np.ndarray):
                        mp = float(mp.ravel()[0])
                    ev_gain[i] = float(mp or 0.0)

        c1min = self._c1_min_actions(city) if n_ev > 0 else np.zeros((0,), dtype=float)

        dev = u_t.device
        params = [
            u_t,
            torch.tensor(base_net, dtype=torch.float32, device=dev),
            torch.tensor(soc0, dtype=torch.float32, device=dev),
            torch.tensor(soc_scale, dtype=torch.float32, device=dev),
            torch.tensor(batt_gain, dtype=torch.float32, device=dev),
            torch.tensor(ev_gain, dtype=torch.float32, device=dev),
            torch.tensor(c1min, dtype=torch.float32, device=dev),
            torch.tensor(self.p_build, dtype=torch.float32, device=dev),
            torch.tensor(self.p_grid, dtype=torch.float32, device=dev),
            torch.tensor(self.w_slack, dtype=torch.float32, device=dev),
        ]

        t0 = time.time()
        # solver_args are passed as per cvxpylayers docs
        z_opt, s_bld_opt, s_grid_opt = self.layer(*params, solver_args={"eps": 1e-4, "max_iters": 5000})
        solve_ms = (time.time() - t0) * 1000.0

        with torch.no_grad():
            delta = torch.norm(z_opt - u_t).item()
            sgrid = float(torch.clamp(s_grid_opt.squeeze(), min=0).cpu().item())
            sbld = float(torch.clamp(s_bld_opt, min=0).sum().cpu().item())

        return z_opt, {"delta_l2": delta, "s_grid": sgrid, "s_bld_sum": sbld, "solve_ms": solve_ms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="CityLearnSafety-V2G-v2")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--w_slack", type=float, default=500.0)
    p.add_argument("--aux_penalty_w", type=float, default=0.1)
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # ensure env is registered
    try:
        import citylearn_safe.omni_env_v2  # registers CityLearnSafety-V2G-v2
    except Exception as e:
        print("[WARN] env registration import failed:", e)

    try:
        env = gym.make(args.env_id)
    except Exception as e:
        print(f"[WARN] gym.make({args.env_id}) failed: {e}")
        print("[WARN] Falling back to scripts.make_env.make_base_env() (repo factory).")
        from scripts.make_env import make_base_env
        env = make_base_env()
    obs, _ = env.reset()
    obs_dim = int(np.asarray(obs).size)
    if isinstance(env.action_space, list):
        act_dim = int(sum(int(np.prod(sp.shape)) for sp in env.action_space))
    else:
        act_dim = int(np.prod(env.action_space.shape))

    print(f"[RUN] env_id={args.env_id} obs_dim={obs_dim} act_dim={act_dim} device={device}")

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim, act_dim).to(device)
    targ = Critic(obs_dim, act_dim).to(device)
    targ.load_state_dict(critic.state_dict())

    aopt = optim.Adam(actor.parameters(), lr=3e-4)
    qopt = optim.Adam(critic.parameters(), lr=1e-3)

    buf = ReplayBuffer(max_size=20000, obs_dim=obs_dim, act_dim=act_dim, device=device)
    proj = DiffProjector(env, w_slack=args.w_slack, verbose=args.verbose)

    gamma = 0.99
    tau = 0.01

    o = torch.tensor(np.asarray(obs, dtype=np.float32).ravel(), device=device)
    ma = 0.0

    # rollout
    for t in range(args.steps):
        with torch.no_grad():
            u_rl = actor(o.unsqueeze(0)).squeeze(0)

        u_phi, pinfo = proj.project(u_rl)

        a_np = u_phi.detach().cpu().numpy()
        if isinstance(env.action_space, list):
            a_out = _unflatten_action(a_np, env.action_space)
        else:
            lo = np.asarray(env.action_space.low).ravel()
            hi = np.asarray(env.action_space.high).ravel()
            a_out = np.clip(a_np, lo, hi)

        obs2, rew, term, trunc, _info = env.step(a_out)
        done = float(term or trunc)

        o2 = torch.tensor(np.asarray(obs2, dtype=np.float32).ravel(), device=device)
        r = torch.tensor(float(rew), device=device)
        d = torch.tensor(done, device=device)

        buf.add(o, u_rl.detach(), u_phi.detach(), r, o2, d)

        ma = 0.95 * ma + 0.05 * pinfo["delta_l2"]
        if (t + 1) % 20 == 0:
            print(f"[STEP {t+1:4d}] rew={float(rew):+.3f} done={done:.0f} proj_delta_ma={ma:.4f} s_grid={pinfo['s_grid']:.4f} solve_ms={pinfo['solve_ms']:.1f}")

        o = o2
        if done:
            obs, _ = env.reset()
            o = torch.tensor(np.asarray(obs, dtype=np.float32).ravel(), device=device)

    if buf.size < args.batch:
        print("[ERR] Not enough samples collected.")
        return

    # updates (key: actor backprops through projection)
    for u in range(args.updates):
        obs_b, act_rl_b, act_phi_b, rew_b, nxt_b, done_b = buf.sample(args.batch)

        with torch.no_grad():
            u_rl_next = actor(nxt_b)
            u_phi_next = []
            for i in range(args.batch):
                up, _ = proj.project(u_rl_next[i])
                u_phi_next.append(up.detach())
            u_phi_next = torch.stack(u_phi_next, dim=0)
            y = rew_b + (1.0 - done_b) * gamma * targ(nxt_b, u_phi_next)

        q = critic(obs_b, act_phi_b)
        q_loss = ((q - y) ** 2).mean()
        qopt.zero_grad(set_to_none=True)
        q_loss.backward()
        qopt.step()

        u_rl_cur = actor(obs_b)
        u_phi_cur = []
        proj_pen_terms = []
        for i in range(args.batch):
            up, _ = proj.project(u_rl_cur[i])  # gradients ON
            u_phi_cur.append(up)
            proj_pen_terms.append(((u_rl_cur[i] - up) ** 2).sum())
        u_phi_cur = torch.stack(u_phi_cur, dim=0)
        proj_pen = torch.stack(proj_pen_terms, dim=0).mean()

        act_loss = (-critic(obs_b, u_phi_cur)).mean() + args.aux_penalty_w * proj_pen
        aopt.zero_grad(set_to_none=True)
        act_loss.backward()

        with torch.no_grad():
            gsum = 0.0
            for p in actor.parameters():
                if p.grad is not None:
                    gsum += float(p.grad.norm().cpu().item())

        aopt.step()

        with torch.no_grad():
            for p, tp in zip(critic.parameters(), targ.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

        if (u + 1) % 10 == 0:
            print(f"[UPD {u+1:4d}] q_loss={q_loss.item():.4f} act_loss={act_loss.item():.4f} actor_grad_norm_sum={gsum:.4f} proj_pen={proj_pen.item():.4f}")

    print("[DONE] If actor_grad_norm_sum stays > 0, SP-style backprop through projection works here.")
    env.close()


if __name__ == "__main__":
    main()
