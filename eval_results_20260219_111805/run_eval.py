"""
Evaluation script: Free FOCOPS vs FOCOPS+PSF
Compares constraint violation rates and reward across 1 full episode (8760 steps)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, PROJECT_ROOT)

# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["free", "psf"], required=True,
                    help="free = no PSF, psf = with LookaheadPSFWrapper")
parser.add_argument("--checkpoint", type=str, default="",
                    help="Path to actor checkpoint (.pt file)")
parser.add_argument("--episodes", type=int, default=1,
                    help="Number of evaluation episodes")
parser.add_argument("--out", type=str, default="eval_result.json",
                    help="Output JSON file for results")
parser.add_argument("--psf_horizon", type=int, default=24,
                    help="PSF lookahead horizon")
parser.add_argument("--psf_p_building", type=float,
                    default=float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "3.47")),
                    help="Building power limit kW")
parser.add_argument("--psf_p_grid", type=float,
                    default=float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "35.76")),
                    help="Grid power limit kW")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

print(f"\n{'='*60}")
print(f"  Mode: {args.mode.upper()}")
print(f"  Checkpoint: {args.checkpoint or 'RANDOM POLICY'}")
print(f"  Episodes: {args.episodes}")
print(f"{'='*60}\n")

# -----------------------------------------------------------------------
# Build environment (reuse your existing wrapper stack)
# -----------------------------------------------------------------------
def make_env(use_psf=False, psf_horizon=24):
    """Build the CityLearn wrapper stack exactly as in your training setup."""

    # Try importing your existing env builder
    env = None

    # Method 1: Try omni_env_v2_shield style
    try:
        from citylearn_safe.omni_env_v2_shield import make_env as _make
        env = _make()
        print("[Env] Built via omni_env_v2_shield.make_env()")
    except Exception as e:
        print(f"[Env] omni_env_v2_shield failed: {e}")

    # Method 2: Try train_omnisafe style
    if env is None:
        try:
            from scripts.train_omnisafe import make_env as _make
            env = _make()
            print("[Env] Built via train_omnisafe.make_env()")
        except Exception as e:
            print(f"[Env] train_omnisafe failed: {e}")

    # Method 3: Direct construction following your known wrapper stack
    if env is None:
        try:
            print("[Env] Building manually...")
            from citylearn.citylearn import CityLearnEnv
            from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
            from citylearn_safe.action_projection import ActionProjectionWrapper
            from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper

            # Find schema
            schema_candidates = [
                "citylearn_challenge_2023_phase_2/schema.json",
                "data/citylearn_challenge_2023_phase_2/schema.json",
                "schema.json",
            ]
            schema = None
            for s in schema_candidates:
                if os.path.exists(s):
                    schema = s
                    break
            if schema is None:
                raise FileNotFoundError("schema.json not found")

            base = CityLearnEnv(schema=schema)
            env = CityLearnSafetyEnvV3(base)

            # Add action projection (C1+C2 shield)
            shield_c1 = bool(int(os.environ.get("CITYLEARN_USE_SHIELD", "1")))
            if shield_c1:
                env = ActionProjectionWrapper(env)
                print("[Env] ActionProjectionWrapper added (C1+C2)")

            # Add forecast obs
            try:
                env = ForecastObsWrapper(env)
                print("[Env] ForecastObsWrapper added")
            except Exception:
                pass

            print(f"[Env] Built manually | obs={env.observation_space.shape} "
                  f"act={env.action_space.shape}")
        except Exception as e:
            raise RuntimeError(f"[Env] All env construction methods failed: {e}")

    # Add PSF wrapper if requested
    if use_psf:
        try:
            from citylearn_safe.lookahead_psf import LookaheadPSFWrapper
            env = LookaheadPSFWrapper(
                env,
                horizon=psf_horizon,
                p_building_max=args.psf_p_building,
                p_grid_max=args.psf_p_grid,
                w_track=100.0,
                w_slack_c1=1000.0,
                w_slack_c3=10.0,
                w_slack_c4=10.0,
                verbose=0,
            )
            print(f"[Env] LookaheadPSFWrapper added (H={psf_horizon})")
        except ImportError:
            # Try alternate import path
            try:
                from citylearn_safe.psf_lookahead_qp import LookaheadPSFWrapper
                env = LookaheadPSFWrapper(env, horizon=psf_horizon, verbose=0)
                print(f"[Env] LookaheadPSFWrapper added from psf_lookahead_qp")
            except ImportError as e2:
                print(f"[!] PSF import failed: {e2} — running WITHOUT PSF")

    return env

# -----------------------------------------------------------------------
# Load actor policy
# -----------------------------------------------------------------------
def load_actor(checkpoint_path, obs_dim, act_dim):
    """Load trained actor. Tries OmniSafe format first, then generic."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("[Policy] No checkpoint — using RANDOM policy")
        return None

    try:
        # Try OmniSafe checkpoint format
        data = torch.load(checkpoint_path, map_location="cpu")

        # OmniSafe saves as dict with 'pi' or 'actor' keys
        if isinstance(data, dict):
            for key in ["pi", "actor", "policy", "model"]:
                if key in data:
                    actor = data[key]
                    print(f"[Policy] Loaded OmniSafe actor from key '{key}'")
                    return actor

        # Direct model
        if hasattr(data, "forward"):
            print("[Policy] Loaded model directly")
            return data

        print(f"[Policy] Unknown checkpoint format, keys: "
              f"{list(data.keys()) if isinstance(data, dict) else type(data)}")
        return None

    except Exception as e:
        print(f"[Policy] Load failed: {e} — using RANDOM policy")
        return None

# -----------------------------------------------------------------------
# Get action from policy or random
# -----------------------------------------------------------------------
def get_action(actor, obs, action_space, deterministic=True):
    if actor is None:
        return action_space.sample()

    try:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            # OmniSafe actor interface
            if hasattr(actor, "predict"):
                act, _ = actor.predict(obs_t, deterministic=deterministic)
                return act.numpy().squeeze()
            elif hasattr(actor, "act"):
                act = actor.act(obs_t)
                return act.numpy().squeeze()
            elif callable(actor):
                act = actor(obs_t)
                if isinstance(act, tuple):
                    act = act[0]
                return act.detach().numpy().squeeze()
            else:
                return action_space.sample()
    except Exception as e:
        return action_space.sample()

# -----------------------------------------------------------------------
# Metric tracking
# -----------------------------------------------------------------------
class MetricTracker:
    def __init__(self):
        self.steps = 0
        self.total_reward = 0.0
        self.total_bill = 0.0

        # C1: EV departure violation
        self.c1_violations = 0          # steps with EV departure deficit
        self.c1_departure_events = 0    # total EV departures

        # C2: Battery SoC violation
        self.c2_violations = 0          # steps with SoC out of [0, 0.95]
        self.c2_violation_kwh = 0.0

        # C3: Building power violation
        self.c3_violations = 0          # steps where any building > P_max
        self.c3_violation_magnitude = 0.0

        # C4: Grid power violation
        self.c4_violations = 0          # steps where grid > P_grid_max
        self.c4_violation_magnitude = 0.0

        # PSF stats
        self.psf_interventions = 0
        self.psf_ev_interventions = 0
        self.psf_batt_interventions = 0
        self.psf_grid_interventions = 0
        self.psf_infeasible = 0
        self.psf_solve_ms_total = 0.0

        # EV stats
        self.ev_avoidable_deficit = 0.0
        self.ev_unavoidable_deficit = 0.0

        # Grid stats
        self.solar_waste_kwh = 0.0
        self.grid_import_kwh = 0.0
        self.grid_export_kwh = 0.0

    def update(self, info, reward):
        self.steps += 1
        self.total_reward += float(reward)
        self.total_bill += float(info.get("reward_bill_raw", 0.0))

        # C1: EV departure violations
        ev_dep = int(info.get("ev_departure_departures", 0))
        ev_deficit = float(info.get("ev_avoidable_deficit_kwh", 0.0))
        if ev_dep > 0:
            self.c1_departure_events += ev_dep
            if ev_deficit > 0.01:
                self.c1_violations += 1

        # C2: Battery SoC violation — use battery_soc_violation_any
        c2_viol = float(info.get("battery_soc_violation_any",
                        info.get("battery_soc_violation", 0.0)))
        if c2_viol > 0:
            self.c2_violations += 1
        self.c2_violation_kwh += float(info.get("cost_soc_mean_hinge", 0.0))

        # C3: Building power violation
        c3_viol = float(info.get("building_power_violation", 0.0))
        if c3_viol > 0:
            self.c3_violations += 1
        self.c3_violation_magnitude += float(info.get("cost_stems_building_power", 0.0))

        # C4: Grid power violation
        c4_viol = float(info.get("grid_power_violation", 0.0))
        if c4_viol > 0:
            self.c4_violations += 1
        self.c4_violation_magnitude += float(info.get("cost_stems_grid_power", 0.0))

        # PSF stats
        if info.get("psf_any_intervention", False):
            self.psf_interventions += 1
        self.psf_ev_interventions += int(info.get("psf_ev_interventions", 0))
        self.psf_batt_interventions += int(info.get("psf_battery_interventions", 0))
        self.psf_grid_interventions += int(info.get("psf_grid_interventions", 0))
        self.psf_infeasible += int(info.get("psf_infeasible", 0) > 0)
        self.psf_solve_ms_total += float(info.get("psf_solve_ms", 0.0))

        # EV
        self.ev_avoidable_deficit += float(info.get("ev_avoidable_deficit_kwh", 0.0))
        self.ev_unavoidable_deficit += float(info.get("ev_unavoidable_deficit_kwh", 0.0))

        # Grid/solar
        self.solar_waste_kwh += float(info.get("solar_waste_kwh", 0.0))
        self.grid_import_kwh += float(info.get("grid_import_kwh", 0.0))
        self.grid_export_kwh += float(info.get("grid_export_kwh", 0.0))

    def summary(self):
        s = self.steps
        if s == 0:
            return {}
        return {
            "total_steps": s,
            "total_reward": round(self.total_reward, 2),
            "total_bill_dollars": round(self.total_bill, 2),
            "avg_reward_per_step": round(self.total_reward / s, 4),

            # C1
            "c1_ev_departure_violation_rate_%": round(
                100.0 * self.c1_violations / max(1, self.c1_departure_events), 2)
                if self.c1_departure_events > 0 else 0.0,
            "c1_departure_events": self.c1_departure_events,
            "c1_violations": self.c1_violations,
            "c1_ev_avoidable_deficit_kwh": round(self.ev_avoidable_deficit, 2),

            # C2
            "c2_battery_soc_violation_rate_%": round(100.0 * self.c2_violations / s, 2),
            "c2_violations_steps": self.c2_violations,

            # C3
            "c3_building_power_violation_rate_%": round(100.0 * self.c3_violations / s, 2),
            "c3_violations_steps": self.c3_violations,
            "c3_violation_magnitude_total": round(self.c3_violation_magnitude, 2),

            # C4
            "c4_grid_power_violation_rate_%": round(100.0 * self.c4_violations / s, 2),
            "c4_violations_steps": self.c4_violations,
            "c4_violation_magnitude_total": round(self.c4_violation_magnitude, 2),

            # PSF
            "psf_intervention_rate_%": round(100.0 * self.psf_interventions / s, 2),
            "psf_ev_interventions": self.psf_ev_interventions,
            "psf_batt_interventions": self.psf_batt_interventions,
            "psf_grid_interventions": self.psf_grid_interventions,
            "psf_infeasible_steps": self.psf_infeasible,
            "psf_avg_solve_ms": round(self.psf_solve_ms_total / max(1, s), 2),

            # Energy
            "solar_waste_kwh": round(self.solar_waste_kwh, 2),
            "grid_import_kwh_total": round(self.grid_import_kwh, 2),
            "grid_export_kwh_total": round(self.grid_export_kwh, 2),
            "ev_unavoidable_deficit_kwh": round(self.ev_unavoidable_deficit, 2),
        }

# -----------------------------------------------------------------------
# Main evaluation loop
# -----------------------------------------------------------------------
def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_psf = (args.mode == "psf")
    env = make_env(use_psf=use_psf, psf_horizon=args.psf_horizon)

    # Flatten obs for policy
    if isinstance(env.observation_space, list):
        obs_dim = sum(sp.shape[0] for sp in env.observation_space)
    else:
        obs_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, list):
        act_dim = sum(int(np.prod(sp.shape)) for sp in env.action_space)
    else:
        act_dim = int(np.prod(env.action_space.shape))

    print(f"[Eval] obs_dim={obs_dim} act_dim={act_dim}")

    actor = load_actor(args.checkpoint, obs_dim, act_dim)

    all_summaries = []
    t_start_total = time.time()

    for ep in range(args.episodes):
        print(f"\n[Episode {ep+1}/{args.episodes}] Starting...")
        tracker = MetricTracker()

        obs, info = env.reset(seed=args.seed + ep)
        if isinstance(obs, (list, tuple)):
            obs_flat = np.concatenate([np.asarray(o).ravel() for o in obs])
        else:
            obs_flat = np.asarray(obs).ravel()

        ep_start = time.time()
        done = False
        step = 0

        while not done:
            # Get action
            if isinstance(env.action_space, list):
                act_flat = get_action(actor, obs_flat, env.action_space[0])
                # Replicate across agents or use single actor
                action = act_flat
            else:
                action = get_action(actor, obs_flat, env.action_space)

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            if isinstance(obs, (list, tuple)):
                obs_flat = np.concatenate([np.asarray(o).ravel() for o in obs])
            else:
                obs_flat = np.asarray(obs).ravel()

            tracker.update(info, reward)
            step += 1

            # Progress print every 1000 steps
            if step % 1000 == 0:
                elapsed = time.time() - ep_start
                print(f"  step={step:5d} | "
                      f"reward={tracker.total_reward:8.1f} | "
                      f"C1={tracker.c1_violations:4d} "
                      f"C2={tracker.c2_violations:4d} "
                      f"C3={tracker.c3_violations:4d} "
                      f"C4={tracker.c4_violations:4d} | "
                      f"{elapsed:.0f}s elapsed")

        ep_time = time.time() - ep_start
        summary = tracker.summary()
        summary["episode"] = ep + 1
        summary["episode_time_s"] = round(ep_time, 1)
        all_summaries.append(summary)

        print(f"\n  [Episode {ep+1} DONE] {step} steps in {ep_time:.0f}s")
        print(f"  Reward: {summary['total_reward']:.1f}")
        print(f"  C1 violation: {summary['c1_ev_departure_violation_rate_%']:.1f}%")
        print(f"  C2 violation: {summary['c2_battery_soc_violation_rate_%']:.1f}%")
        print(f"  C3 violation: {summary['c3_building_power_violation_rate_%']:.1f}%")
        print(f"  C4 violation: {summary['c4_grid_power_violation_rate_%']:.1f}%")
        if use_psf:
            print(f"  PSF interventions: {summary['psf_intervention_rate_%']:.1f}%")
            print(f"  PSF avg solve: {summary['psf_avg_solve_ms']:.1f}ms")

    total_time = time.time() - t_start_total

    # Average across episodes
    keys = all_summaries[0].keys()
    averaged = {}
    for k in keys:
        vals = [s[k] for s in all_summaries if isinstance(s[k], (int, float))]
        averaged[k] = round(float(np.mean(vals)), 4) if vals else all_summaries[0][k]

    result = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "total_eval_time_s": round(total_time, 1),
        "per_episode": all_summaries,
        "averaged": averaged,
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[✓] Results saved to {args.out}")
    return result

if __name__ == "__main__":
    main()

