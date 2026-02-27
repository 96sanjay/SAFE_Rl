# scripts/run_baselines_cmdp.py
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Callable, Dict, Any, Tuple

import numpy as np

from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv, SoCSafetyConfig


def baseline_zero(env):
    return np.zeros(env.action_space.shape, dtype=np.float32)


def baseline_random(env):
    return env.action_space.sample()


def step_unpack(out) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """
    Expect safety-wrapped env returns gymnasium 5-tuple:
      (obs, reward, terminated, truncated, info)
    Also supports old gym 4-tuple.
    """
    if len(out) == 5:
        obs, r, term, trunc, info = out
        return obs, float(r), bool(term), bool(trunc), (info or {})
    if len(out) == 4:
        obs, r, done, info = out
        return obs, float(r), bool(done), False, (info or {})
    raise RuntimeError(f"Unexpected step() return length: {len(out)}")


def run(baseline_fn: Callable, out_dir: str, steps: int = 8760, cost_mode: str = "hinge"):
    os.makedirs(out_dir, exist_ok=True)

    # Build the SAME base env stack used in training (central agent)
    base_env = make_base_env(central_agent=True)

    # Wrap with safety env that injects info["cost"] and kpi_soc_*
    safe_env = CityLearnSafetyEnv(
        base_env,
        soc_cfg=SoCSafetyConfig(lower=0.0, upper=0.95, cost_mode=cost_mode, scale=1.0),
        log_kpis=False,
    )

    # reset (gymnasium reset returns (obs, info) sometimes)
    r = safe_env.reset()
    if isinstance(r, tuple) and len(r) == 2:
        obs, info = r
    else:
        obs, info = r, {}

    rows = []
    total_r = 0.0
    total_c = 0.0

    for t in range(steps):
        a = baseline_fn(safe_env)
        obs, reward, term, trunc, info = step_unpack(safe_env.step(a))

        cost = float(info.get("cost", 0.0))
        total_r += reward
        total_c += cost

        row: Dict[str, Any] = {"t": t, "reward": reward, "cost": cost}
        # store scalar KPIs from info
        for k, v in (info or {}).items():
            if isinstance(v, (int, float, np.floating, np.integer, str, bool)):
                row[k] = v
        rows.append(row)

        if term or trunc:
            break

    csv_path = os.path.join(out_dir, "kpis.csv")
    all_keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {csv_path}")
    print(f"Steps: {len(rows)} | Total reward: {total_r:.4f} | Total cost: {total_c:.4f}")

    try:
        safe_env.close()
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["zero", "random"], default="zero")
    p.add_argument("--out_root", type=str, default="runs/baselines_cmdp")
    p.add_argument("--steps", type=int, default=8760)
    p.add_argument("--cost_mode", choices=["hinge", "binary", "count"], default="hinge")
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, f"{args.baseline}_{args.cost_mode}_{ts}")

    baseline_map = {"zero": baseline_zero, "random": baseline_random}
    run(baseline_map[args.baseline], out_dir, steps=args.steps, cost_mode=args.cost_mode)


if __name__ == "__main__":
    main()
