import os, sys, argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.adapters import SingleAgentListAdapter
from citylearn.wrappers import NormalizedObservationWrapper


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(512, 512, 256), activation="relu"):
        super().__init__()
        act = nn.ReLU if activation.lower() == "relu" else nn.Tanh
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        self.mean = nn.Sequential(*layers, nn.Linear(prev, act_dim))
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        return torch.tanh(self.mean(obs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to OmniSafe epoch-XX.pt (FOCOPS)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8759)
    args = ap.parse_args()

    schema = os.environ.get("CITYLEARN_SCHEMA", "")
    if not schema or not os.path.exists(schema):
        raise FileNotFoundError(f"CITYLEARN_SCHEMA not set or not found: {schema}")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # Build env exactly like your PPOLag eval (important!)
    base = CityLearnEnv(schema=schema, central_agent=True)
    base = NormalizedObservationWrapper(base)
    env = SingleAgentListAdapter(base)
    env = CityLearnSafetyEnvV3(env)

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # OmniSafe checkpoints typically store actor under 'pi'
    if "pi" not in ckpt:
        raise KeyError(f"Checkpoint missing key 'pi'. Keys = {list(ckpt.keys())}")

    obs, _ = env.reset(seed=args.seed)

    actor = GaussianActor(obs_dim=len(obs), act_dim=env.action_space.shape[0])
    actor.load_state_dict(ckpt["pi"], strict=True)
    actor.eval()

    ep_reward, ep_cost, steps = 0.0, 0.0, 0
    ev_departures, ev_violations = 0, 0
    grid_viol, battery_viol, building_viol = 0, 0, 0
    cost_ev, cost_grid, cost_battery, cost_building = 0.0, 0.0, 0.0, 0.0

    print("\n" + "=" * 80)
    print("EVALUATION (FOCOPS) — 1 episode")
    print("=" * 80)
    print(f"SCHEMA: {schema}")
    print(f"CKPT:   {args.ckpt}")
    print(f"seed={args.seed} | steps={args.steps}")
    print("-" * 80)
    print(f"EV_MISSING_ACTION_MODE = {os.environ.get('CITYLEARN_EV_MISSING_ACTION_MODE')}")
    print(f"SOC_LOW/HIGH           = {os.environ.get('CITYLEARN_STEMS_SOC_LOW')}/{os.environ.get('CITYLEARN_STEMS_SOC_HIGH')}")
    print(f"EV_DENSE_COST_SCALE    = {os.environ.get('CITYLEARN_EV_DENSE_COST_SCALE')}")
    print("=" * 80 + "\n")

    while True:
        with torch.no_grad():
            action = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        ep_reward += float(reward)
        step_cost = float(info.get("cost", 0.0))
        ep_cost += step_cost

        # EV departure violation rate (same logic as your PPOLag eval)
        ev_dep = int(info.get("ev_departure_departures", 0))
        ev_deficit = float(info.get("ev_departure_deficit_kwh", 0.0))
        if ev_dep > 0:
            ev_departures += ev_dep
            if ev_deficit > 0.01:
                ev_violations += 1

        if float(info.get("cost_stems_grid_power", 0.0)) > 0:
            grid_viol += 1
        battery_viol += int(info.get("battery_soc_violation_count", 0))
        building_viol += int(info.get("building_power_violation_count", 0))

        cost_ev += float(info.get("cost_ev_departure", 0.0))
        cost_grid += float(info.get("cost_stems_grid_power", 0.0))
        cost_battery += float(info.get("cost_stems_battery", 0.0))
        cost_building += float(info.get("cost_stems_building_power", 0.0))

        if terminated or truncated or steps >= args.steps:
            break

    env.close()

    print("\n" + "=" * 80)
    print("FINAL RESULTS (FOCOPS)")
    print("=" * 80)
    print(f"Steps:          {steps}")
    print(f"Episode Reward: {ep_reward:.2f}")
    print(f"Episode Cost:   {ep_cost:.2f}")
    print()

    print("=" * 80)
    print("CONSTRAINT VIOLATIONS")
    print("=" * 80)
    ev_rate = 100.0 * ev_violations / max(ev_departures, 1)
    print(f"EV Departure:  {ev_violations:>5} / {ev_departures:<5} = {ev_rate:>6.2f}%")
    grid_rate = 100.0 * grid_viol / max(steps, 1)
    print(f"Grid Power:    {grid_viol:>5} / {steps:<5} = {grid_rate:>6.2f}%")
    battery_rate = 100.0 * battery_viol / max(steps * 17, 1)
    print(f"Battery SoC:   {battery_viol:>5} / {steps*17:<5} = {battery_rate:>6.2f}%")
    building_rate = 100.0 * building_viol / max(steps * 17, 1)
    print(f"Building Pwr:  {building_viol:>5} / {steps*17:<5} = {building_rate:>6.2f}%")
    print()

    print("=" * 80)
    print("COST BREAKDOWN")
    print("=" * 80)
    if ep_cost <= 1e-9:
        print("TOTAL COST is ~0, breakdown skipped.")
    else:
        print(f"EV:       {cost_ev:>12.2f}  ({100*cost_ev/ep_cost:>5.1f}%)")
        print(f"Grid:     {cost_grid:>12.2f}  ({100*cost_grid/ep_cost:>5.1f}%)")
        print(f"Battery:  {cost_battery:>12.2f}  ({100*cost_battery/ep_cost:>5.1f}%)")
        print(f"Building: {cost_building:>12.2f}  ({100*cost_building/ep_cost:>5.1f}%)")
        print(f"{'-'*80}")
        print(f"TOTAL:    {ep_cost:>12.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("\n[ERROR] Evaluation crashed:\n")
        traceback.print_exc()
        raise SystemExit(1)
