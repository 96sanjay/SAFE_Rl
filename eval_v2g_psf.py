import argparse, os, numpy as np, torch, torch.nn as nn

os.environ.setdefault("CITYLEARN_SCHEMA", "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json")
os.environ.setdefault("CITYLEARN_CENTRAL_AGENT", "1")

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.lookahead_psf_v2g import LookaheadPSFv2G
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(278, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 26),
        )
        self.log_std = nn.Parameter(torch.zeros(26))

    def forward(self, obs):
        return torch.tanh(self.mean(obs))


def evaluate(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    actor = Actor()
    actor.load_state_dict(ckpt["pi"])
    actor.eval()

    obs_norm = ckpt.get("obs_normalizer", None)
    obs_mean = obs_norm["mean"] if obs_norm and "mean" in obs_norm else None
    obs_var = obs_norm["var"] if obs_norm and "var" in obs_norm else None

    base = CityLearnEnv(
        schema=os.environ["CITYLEARN_SCHEMA"],
        central_agent=os.environ.get("CITYLEARN_CENTRAL_AGENT", "1") == "1",
    )
    safety = CityLearnSafetyEnvV3(base)
    psf = LookaheadPSFv2G(
        safety, horizon=24, w_track=100.0, w_slack_c1=5000.0,
        w_slack_c3=500.0, w_slack_c4=1000.0, w_future_reg=0.01,
        exempt_ev_from_c3=True, c4_one_sided=True, verbose=1,
    )
    env = ForecastObsWrapper(psf, forecast_horizon=24)

    act_space = env.action_space
    if isinstance(act_space, list):
        act_low = np.concatenate([np.asarray(s.low).ravel() for s in act_space])
        act_high = np.concatenate([np.asarray(s.high).ravel() for s in act_space])
    else:
        act_low, act_high = act_space.low, act_space.high

    obs, _ = env.reset()
    if isinstance(obs, (list, tuple)):
        obs = np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs])
    obs = np.asarray(obs, dtype=np.float32).ravel()

    T, total_reward = 0, 0.0
    total_interventions, total_ev_interventions, v2g_events = 0, 0, 0

    # C1: violated_departures / total_departures
    c1_violated_departures = 0
    c1_total_departures = 0

    # C2: mean(battery_soc_violation_frac) over episode
    c2_frac_sum = 0.0

    # C3: sum(building_violation_count) / (17 * T)
    c3_bldg_violation_total = 0.0

    # C4: steps with grid violation / T
    c4_violation_steps = 0

    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if obs_mean is not None and obs_var is not None:
            obs_t = (obs_t - obs_mean) / torch.sqrt(obs_var + 1e-8)
        with torch.no_grad():
            action = actor(obs_t.unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(action, act_low, act_high)

        obs, reward, term, trunc, info = env.step(action)
        if isinstance(obs, (list, tuple)):
            obs = np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs])
        obs = np.asarray(obs, dtype=np.float32).ravel()

        total_reward += float(reward)
        T += 1

        # Shield
        delta = float(info.get("psf_action_delta_l2", 0.0))
        if delta > 1e-4:
            total_interventions += 1
        total_ev_interventions += int(info.get("psf_ev_interventions", 0))
        safe_act = info.get("shielded_action", None)
        if safe_act is not None and len(safe_act) > 24:
            v2g_events += sum(1 for a in safe_act[17:25] if a < -0.01)

        # C1: count departures with deficit
        c1_violated_departures += int(info.get("ev_departure_violation_count_deficit", 0))
        c1_total_departures += int(info.get("ev_departure_departures", 0))

        # C2: per-step fraction of batteries violating SoC
        c2_frac_sum += float(info.get("battery_soc_violation_frac", 0.0))

        # C3: number of buildings violating this step (0-17)
        c3_bldg_violation_total += float(info.get("building_power_violation_count", 0.0))

        # C4: binary - did grid violate this step?
        c4_violation_steps += int(float(info.get("grid_power_violation", 0.0)) > 0.5)

        done = bool(term) or bool(trunc)
        if T % 2000 == 0:
            print(f"  step {T}: rew={total_reward:.1f} "
                  f"c1={c1_violated_departures}/{c1_total_departures} "
                  f"c3={c3_bldg_violation_total:.0f}/{17*T} "
                  f"c4={c4_violation_steps}/{T}")

    sep = "=" * 70
    c1_pct = 100.0 * c1_violated_departures / max(1, c1_total_departures)
    c2_pct = 100.0 * c2_frac_sum / max(1, T)
    c3_pct = 100.0 * c3_bldg_violation_total / (17.0 * max(1, T))
    c4_pct = 100.0 * c4_violation_steps / max(1, T)
    interv_pct = 100.0 * total_interventions / max(1, T)

    print()
    print(sep)
    print("EVALUATION RESULTS")
    print(sep)
    print(f"Steps:                {T}")
    print(f"Total reward:         {total_reward:.1f}")
    print(f"Shield interventions: {total_interventions} ({interv_pct:.1f}%)")
    print(f"EV interventions:     {total_ev_interventions}")
    print(f"V2G discharge events: {v2g_events}")
    print()
    print("CONSTRAINT VIOLATIONS:")
    print(f"  C1 (EV departure):   {c1_violated_departures}/{c1_total_departures} departures = {c1_pct:.2f}%")
    print(f"  C2 (Battery SoC):    mean violation frac = {c2_pct:.2f}%")
    print(f"  C3 (Building power): {c3_bldg_violation_total:.0f}/{17*T} building-steps = {c3_pct:.2f}%")
    print(f"  C4 (Grid power):     {c4_violation_steps}/{T} steps = {c4_pct:.2f}%")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint)
