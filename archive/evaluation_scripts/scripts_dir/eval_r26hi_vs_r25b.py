#!/usr/bin/env python3
"""
Evaluate R26h, R26i, and R25b-stable with physics-based constraint checking.

Uses the SAME physics-based C0 logic as eval_all_runs.py (raw SoC vs required SoC),
NOT the agent-controllable metric from info["cost_ev_departure"].

All three runs use STEMS v3 encoder (GCN + temporal transformer).
Each run gets its EXACT training env vars.
"""
import os, sys, warnings, json, time
import numpy as np
warnings.filterwarnings('ignore')

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import torch
import torch.nn as nn

P_BMAX = 4.6083
P_GMAX = 10.2352

# ── Common env vars shared by all three runs ──
COMMON_ENV = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "12",
    "STEMS_ENCODER_VERSION": "v3",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_BATT_CLAMP": "0",
    "CITYLEARN_ACTION_MASK": "0",
    "CITYLEARN_POLICY_ACTION_MASK": "0",
}

# ── Per-run overrides (EXACT match to training run scripts) ──
RUN_CONFIGS = {
    "R25b-stable": {
        "checkpoint": f"{PROJECT}/runs/r25b_report_stable_stems_v3_100ep/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-03-28-20-06-40/torch_save/epoch-80.pt",
        "env_overrides": {
            # R25b training env vars from run_r25b_report_stable_stems_v3_100ep.sh
            "CITYLEARN_EV_SAUTE": "1",
            "CITYLEARN_EV_SAUTE_BUDGET": "25000",
            "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
            "CITYLEARN_EV_SAUTE_GAMMA": "1.0",
            "CITYLEARN_EV_SAUTE_SHAPED_ALPHA": "10.0",
            "STEMS_ALPHA_GRID": "0.0",
            "STEMS_ALPHA_BUILD": "0.0",
            "STEMS_MU_ECONOMIC": "0.0",
            "STEMS_ALPHA_LOAD_SHIFT": "0.0",
            "STEMS_ALPHA_GRID_MILD": "0.3",
            "STEMS_LAMBDA_EV": "5.0",
            "STEMS_ALPHA_EV_GUARD": "1.0",
            "STEMS_ALPHA_V2G_CONTEXT": "3.0",
            "STEMS_ALPHA_BARRIER": "0.5",
            "STEMS_ALPHA_PEAK_SHAVE": "0.0",
            "STEMS_XI_RENEWABLE": "0.2",
            "STEMS_BETA_RAMP": "0.3",
            "STEMS_SB_ASYMMETRIC": "1",
            "STEMS_SG_EXPORT_CREDIT": "0.5",
            "STEMS_SG_THRESHOLD": "0.5",
            "STEMS_EV_SLACK_ARB_SCALE": "2.0",
            "STEMS_ALPHA_HEADROOM": "0.0",
            "STEMS_ALPHA_PRICE_ARB": "0.0",
            "STEMS_ALPHA_NEC_SIGN": "0.0",
            "STEMS_ALPHA_EV_SOLAR": "0.0",
            "STEMS_ALPHA_SOLAR_STORE": "0.0",
            "STEMS_ALPHA_TRAJECTORY": "0.0",
            "STEMS_ALPHA_GRID_PENALTY": "0.0",
        },
    },
    "R26h-lean": {
        "checkpoint": "/tmp/r26_checkpoints/r26h_epoch80.pt",
        "env_overrides": {
            # R26h training env vars from run_r26h.sh
            "CITYLEARN_TEMPORAL_RICH": "1",
            "CITYLEARN_EV_SAUTE": "1",
            "CITYLEARN_EV_SAUTE_BUDGET": "25000",
            "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
            "CITYLEARN_EV_SAUTE_GAMMA": "1.0",
            "CITYLEARN_EV_SAUTE_SHAPED_ALPHA": "10.0",
            "STEMS_MU_ECONOMIC": "0.0",
            "STEMS_ALPHA_GRID": "2.0",
            "STEMS_ALPHA_BUILD": "3.0",
            "STEMS_BETA_RAMP": "0.0",
            "STEMS_XI_RENEWABLE": "0.3",
            "STEMS_LAMBDA_EV": "0.0",
            "STEMS_ALPHA_BARRIER": "1.0",
            "STEMS_ALPHA_TRAJECTORY": "2.0",
            "STEMS_ALPHA_EV_GUARD": "0.0",
            "STEMS_ALPHA_V2G_CONTEXT": "0.0",
            "STEMS_ALPHA_PEAK_SHAVE": "0.0",
            "STEMS_ALPHA_LOAD_SHIFT": "0.0",
            "STEMS_ALPHA_GRID_MILD": "0.0",
            "STEMS_ALPHA_EV_SOLAR": "0.0",
            "STEMS_ALPHA_SOLAR_STORE": "0.0",
            "STEMS_EV_SLACK_ARB_SCALE": "0.0",
            "STEMS_ALPHA_HEADROOM": "0.0",
            "STEMS_ALPHA_GRID_PENALTY": "0.0",
            "STEMS_ALPHA_PRICE_ARB": "0.0",
            "STEMS_ALPHA_NEC_SIGN": "0.0",
            "STEMS_TRAJ_EV_WEIGHT": "1.0",
            "STEMS_TRAJ_FORECAST_HOURS": "24",
            "CITYLEARN_EV_CLAMP": "0",
            "CITYLEARN_C3_COST": "0.1",
            "CITYLEARN_C4_COST": "5.0",
            "STEMS_SB_ASYMMETRIC": "1",
            "STEMS_SG_EXPORT_CREDIT": "0.5",
            "STEMS_SG_THRESHOLD": "0.5",
        },
    },
    "R26i-full": {
        "checkpoint": "/tmp/r26_checkpoints/r26i_epoch80.pt",
        "env_overrides": {
            # R26i training env vars from run_r26i.sh
            "CITYLEARN_TEMPORAL_RICH": "1",
            "CITYLEARN_EV_SAUTE": "1",
            "CITYLEARN_EV_SAUTE_BUDGET": "25000",
            "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
            "CITYLEARN_EV_SAUTE_GAMMA": "1.0",
            "CITYLEARN_EV_SAUTE_SHAPED_ALPHA": "10.0",
            "STEMS_MU_ECONOMIC": "0.3",
            "STEMS_ALPHA_GRID": "2.0",
            "STEMS_ALPHA_BUILD": "3.0",
            "STEMS_BETA_RAMP": "0.0",
            "STEMS_XI_RENEWABLE": "0.3",
            "STEMS_LAMBDA_EV": "0.5",
            "STEMS_ALPHA_BARRIER": "1.0",
            "STEMS_ALPHA_TRAJECTORY": "2.0",
            "STEMS_ALPHA_EV_GUARD": "1.5",
            "STEMS_ALPHA_V2G_CONTEXT": "0.0",
            "STEMS_ALPHA_PEAK_SHAVE": "0.0",
            "STEMS_ALPHA_LOAD_SHIFT": "0.0",
            "STEMS_ALPHA_GRID_MILD": "0.0",
            "STEMS_ALPHA_EV_SOLAR": "0.0",
            "STEMS_ALPHA_SOLAR_STORE": "0.0",
            "STEMS_EV_SLACK_ARB_SCALE": "0.0",
            "STEMS_ALPHA_HEADROOM": "0.0",
            "STEMS_ALPHA_GRID_PENALTY": "0.0",
            "STEMS_ALPHA_PRICE_ARB": "0.0",
            "STEMS_ALPHA_NEC_SIGN": "0.0",
            "STEMS_TRAJ_EV_WEIGHT": "1.0",
            "STEMS_TRAJ_FORECAST_HOURS": "24",
            "CITYLEARN_EV_CLAMP": "0",
            "CITYLEARN_C3_COST": "0.1",
            "CITYLEARN_C4_COST": "5.0",
            "STEMS_SB_ASYMMETRIC": "1",
            "STEMS_SG_EXPORT_CREDIT": "0.5",
            "STEMS_SG_THRESHOLD": "0.5",
        },
    },
    "R25b-SAC-cap10": {
        "checkpoint": "/tmp/r26_checkpoints/r25b_sac_cap10_epoch80.pt",
        "actor_type": "sac_mlp",
        "env_overrides": {
            # Base R25b reward vars (same as R25b-stable)
            "STEMS_ALPHA_GRID": "0.0",
            "STEMS_ALPHA_BUILD": "0.0",
            "STEMS_MU_ECONOMIC": "0.0",
            "STEMS_ALPHA_LOAD_SHIFT": "0.0",
            "STEMS_ALPHA_PEAK_SHAVE": "0.0",
            "STEMS_XI_RENEWABLE": "0.2",
            "STEMS_BETA_RAMP": "0.3",
            "STEMS_SB_ASYMMETRIC": "1",
            "STEMS_SG_EXPORT_CREDIT": "0.5",
            "STEMS_SG_THRESHOLD": "0.5",
            "STEMS_EV_SLACK_ARB_SCALE": "2.0",
            "STEMS_ALPHA_BARRIER": "0.5",
            "STEMS_ALPHA_HEADROOM": "0.0",
            "STEMS_ALPHA_PRICE_ARB": "0.0",
            "STEMS_ALPHA_NEC_SIGN": "0.0",
            "STEMS_ALPHA_EV_SOLAR": "0.0",
            "STEMS_ALPHA_SOLAR_STORE": "0.0",
            "STEMS_ALPHA_TRAJECTORY": "0.0",
            "STEMS_ALPHA_GRID_PENALTY": "0.0",
            # SAC cap10 overrides (Optuna-tuned + safety wrappers)
            "CITYLEARN_TEMPORAL_WINDOW": "0",       # No temporal history
            "CITYLEARN_EV_SAUTE": "0",              # Saute disabled
            "CITYLEARN_EV_ACTION_CLAMP": "1",       # Clamp enabled
            "CITYLEARN_BATT_CLAMP": "1",            # Clamp enabled
            "CITYLEARN_ACTION_MASK": "1",           # Action mask enabled
            "STEMS_LAMBDA_EV": "3.0",               # Reduced from 5.0
            "STEMS_ALPHA_GRID_MILD": "0.8",         # Optuna (was 0.3)
            "STEMS_ALPHA_EV_GUARD": "0.5",          # Optuna (was 1.0)
            "STEMS_ALPHA_V2G_CONTEXT": "4.5",       # Optuna (was 3.0)
            "COST_W_C3": "3.0",                     # Optuna
            "CITYLEARN_W_COST_GRID": "0.03",        # Optuna
            "CITYLEARN_C0_SAFETY_FLOOR": "0.5",
            "CITYLEARN_EV_DENSE_COST_SCALE": "0.0",
        },
    },
}


# ── Environment (matches training pipeline in omni_env_v2.py) ──
def make_env():
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)

    # Add temporal history wrapper (same as omni_env_v2.py)
    temporal_window = int(os.environ.get("CITYLEARN_TEMPORAL_WINDOW", "0"))
    if temporal_window > 0:
        from citylearn_safe.temporal_obs_wrapper import TemporalHistoryWrapper
        from citylearn_safe.schema_index import build_index
        _e = safety
        _n_bld = 0
        for _ in range(20):
            if hasattr(_e, 'buildings') and len(getattr(_e, 'buildings', [])) > 0:
                _n_bld = len(_e.buildings)
                break
            _e = getattr(_e, 'env', getattr(_e, 'base', getattr(_e, 'unwrapped', None)))
            if _e is None:
                break
        if _n_bld == 0:
            _n_bld = 5
        obs_idx = build_index(safety, expected_buildings=_n_bld)

        if os.environ.get("CITYLEARN_TEMPORAL_RICH", "0") == "1":
            from citylearn_safe.temporal_obs_wrapper import build_rich_history_config
            rich_cfg = build_rich_history_config(obs_idx, _n_bld)
            history_indices = rich_cfg['history_indices']
        else:
            from citylearn_safe.temporal_obs_wrapper import build_basic_history_indices
            history_indices = build_basic_history_indices(obs_idx, _n_bld)

        env = TemporalHistoryWrapper(env, history_indices, temporal_window)
        print(f"  Temporal history: T={temporal_window}, +{len(history_indices)*temporal_window} dims")

    # Add Saute wrapper if enabled
    if os.environ.get("CITYLEARN_EV_SAUTE", "0") == "1":
        from citylearn_safe.saute_ev_wrapper import SauteEVBudgetWrapper
        env = SauteEVBudgetWrapper(env)
    return env, safety


# ── STEMS Actor Loading (reuses training code's build pipeline) ──

# Cache the STEMS build artifacts (env-independent after first call)
_STEMS_CACHE = {}

def _get_stems_build_artifacts(use_rich_temporal: bool):
    """Build STEMS encoder args once, reuse across runs."""
    cache_key = f"rich={use_rich_temporal}"
    if cache_key in _STEMS_CACHE:
        return _STEMS_CACHE[cache_key]

    from scripts.train_multi_lag_stems import build_obs_index_5bld, STEMSMeanNet
    from citylearn_safe.stems_encoder_5bld import build_node_indices

    obs_index, obs_dim, act_dim, num_buildings, num_evs = build_obs_index_5bld()
    node_info = build_node_indices(obs_index, num_buildings)
    temporal_window = 12

    if use_rich_temporal:
        from citylearn_safe.temporal_obs_wrapper import build_rich_history_config
        rich_cfg = build_rich_history_config(obs_index, num_buildings)
        features_per_step = rich_cfg['features_per_step']
        features_per_node = rich_cfg['features_per_node']
        per_node_map = rich_cfg['per_node_map']
        ev_mask = rich_cfg['history_ev_mask']
    else:
        from citylearn_safe.temporal_obs_wrapper import build_basic_history_indices
        history_indices = build_basic_history_indices(obs_index, num_buildings)
        features_per_step = len(history_indices)
        features_per_node = 3
        per_node_map = None
        ev_mask = None

    obs_dim_v3 = obs_dim + features_per_step * temporal_window

    stems_kwargs = dict(
        obs_dim=obs_dim_v3,
        node_info=node_info,
        num_buildings=num_buildings,
        hidden_dim=64,
        global_hidden=32,
        temporal_window=temporal_window,
        temporal_features_per_step=features_per_step,
        temporal_hidden=32,
        temporal_heads=4,
        num_gcn_layers=3,
        dropout=0.1,
        output_dim=256,
        temporal_features_per_node=features_per_node,
        per_node_history_map=per_node_map,
        history_ev_mask=ev_mask,
        temporal_num_layers=2,
        temporal_pool_mode="mean",
    )

    result = {
        'stems_kwargs': stems_kwargs,
        'obs_dim_v3': obs_dim_v3,
        'act_dim': act_dim,
    }
    _STEMS_CACHE[cache_key] = result
    return result


def load_stems_actor(ckpt_path, use_rich_temporal: bool = False):
    """Load STEMS-based actor from checkpoint using training code's build logic."""
    from citylearn_safe.stems_encoder_v3 import STEMSEncoderV3
    from scripts.train_multi_lag_stems import STEMSMeanNet

    artifacts = _get_stems_build_artifacts(use_rich_temporal)
    stems_kwargs = artifacts['stems_kwargs']
    obs_dim_v3 = artifacts['obs_dim_v3']
    act_dim = artifacts['act_dim']

    # Build encoder with EXACT same args as training
    encoder = STEMSEncoderV3(**stems_kwargs)
    mean_net = STEMSMeanNet(encoder, act_dim, encoder_obs_dim=obs_dim_v3)

    # Load checkpoint weights
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    pi_state = ckpt.get("pi", {})
    if not pi_state:
        raise ValueError("No 'pi' key in checkpoint")

    # Strip "mean." prefix and filter out log_std
    filtered = {}
    for k, v in pi_state.items():
        if k.startswith("log_std"):
            continue
        if k.startswith("mean."):
            filtered[k[5:]] = v
        else:
            filtered[k] = v

    result = mean_net.load_state_dict(filtered, strict=False)
    if result.unexpected_keys:
        print(f"  WARNING: unexpected keys: {result.unexpected_keys}")
    mean_net.eval()

    # Load obs normalizer
    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        clip_t = norm["_clip"].float()
        obs_clip = float(clip_t.mean())

    return mean_net, obs_mean, obs_std, obs_clip, obs_dim_v3


# ── SAC MLP Actor Loading ──

class SACDeterministicActor(nn.Module):
    """Wraps SAC MLP net, returns tanh(mean) for deterministic eval."""
    def __init__(self, net, act_dim):
        super().__init__()
        self.net = net
        self.act_dim = act_dim

    def forward(self, obs):
        out = self.net(obs)
        return torch.tanh(out[:, :self.act_dim])


def load_sac_mlp_actor(ckpt_path):
    """Load SAC MLP actor from checkpoint. Deterministic: tanh(mean[:9])."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    pi_state = ckpt.get("pi", {})
    if not pi_state:
        raise ValueError("No 'pi' key in checkpoint")

    # Build MLP from weight shapes
    obs_dim = pi_state['net.0.weight'].shape[1]
    h1 = pi_state['net.0.weight'].shape[0]
    h2 = pi_state['net.2.weight'].shape[0]
    out_dim = pi_state['net.4.weight'].shape[0]
    act_dim = out_dim // 2  # 18 -> 9 (mean + log_std)

    net = nn.Sequential(
        nn.Linear(obs_dim, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, out_dim),
    )

    # Load net.* keys, stripping "net." prefix
    net_state = {k[4:]: v for k, v in pi_state.items() if k.startswith('net.')}
    net.load_state_dict(net_state)

    actor = SACDeterministicActor(net, act_dim)
    actor.eval()

    # Load obs normalizer
    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        clip_t = norm["_clip"].float()
        obs_clip = float(clip_t.mean())

    return actor, obs_mean, obs_std, obs_clip, obs_dim


# ── Evaluation (physics-based, same as eval_all_runs.py) ──
def evaluate_run(run_name, config):
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {run_name}")
    print(f"{'='*60}")

    # Set env vars: clear all, set common, apply overrides
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k)
    os.environ.update(COMMON_ENV)
    os.environ.update(config["env_overrides"])

    ckpt_path = config["checkpoint"]
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: checkpoint not found: {ckpt_path}")
        return None

    # Load actor based on type
    actor_type = config.get("actor_type", "stems")
    if actor_type == "sac_mlp":
        actor, obs_mean, obs_std, obs_clip, actor_obs_dim = load_sac_mlp_actor(ckpt_path)
        print(f"  Loaded SAC MLP actor from: {os.path.basename(ckpt_path)}")
        print(f"  MLP obs dim: {actor_obs_dim}, obs normalizer: {'YES' if obs_mean is not None else 'NO'}")
    else:
        use_rich = config["env_overrides"].get("CITYLEARN_TEMPORAL_RICH", "0") == "1"
        actor, obs_mean, obs_std, obs_clip, actor_obs_dim = load_stems_actor(ckpt_path, use_rich_temporal=use_rich)
        print(f"  Loaded STEMS actor from: {os.path.basename(ckpt_path)}")
        print(f"  Encoder obs dim: {actor_obs_dim}, obs normalizer: {'YES' if obs_mean is not None else 'NO'}")

    # Create env
    env, safety = make_env()
    from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(raw.buildings)
    n_b = len(buildings)

    obs, _ = env.reset(seed=42)

    # Tracking
    total_steps = 0
    total_reward = 0.0

    # C0: EV departure (physics-based)
    ev_departures = 0
    ev_violated_departures = 0
    ev_deficit_kwh = 0.0
    ev_departure_details = []  # (dep_soc, req_soc, deficit)

    # C2: Battery SoC
    c2_violations = 0
    c2_total_checks = 0

    # C3: Per-building power
    c3_per_building = [0] * n_b
    c3_total_per_building = [0] * n_b

    # C4: Grid
    c4_violations = 0

    # EV action tracking
    ev_charge_steps = 0
    ev_v2g_steps = 0
    ev_idle_steps = 0
    ev_connected_steps = 0

    # NEC tracking
    nec_history = []
    nec_baseline_history = []

    # EV departure tracker
    ev_tracker = {}

    t0 = time.time()
    done = False
    while not done:
        # Get action (deterministic)
        obs_flat = np.asarray(obs).ravel()
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32).unsqueeze(0)

        # Apply obs normalization (same as training)
        # obs_normalizer covers base+temporal dims; Saute may add +1 extra dim
        if obs_mean is not None:
            norm_dim = obs_mean.shape[0]
            if obs_t.shape[1] > norm_dim:
                # Normalize base dims, keep extra (Saute budget) as-is
                obs_base = obs_t[:, :norm_dim]
                obs_extra = obs_t[:, norm_dim:]
                obs_base = (obs_base - obs_mean) / (obs_std + 1e-8)
                if obs_clip is not None:
                    obs_base = obs_base.clamp(-obs_clip, obs_clip)
                obs_t = torch.cat([obs_base, obs_extra], dim=1)
            else:
                obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
                if obs_clip is not None:
                    obs_t = obs_t.clamp(-obs_clip, obs_clip)

        with torch.no_grad():
            action = actor(obs_t).squeeze(0).numpy()
        action = np.clip(action, -1.0, 1.0)

        # Disable washing machine
        action[2] = 0.0

        t_now = int(getattr(raw, 'time_step', 0))

        # ── Track EV departures BEFORE step (physics-based) ──
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, 'electric_vehicle_chargers', None) or []
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                sim = getattr(ch, 'charger_simulation', getattr(ch, '_Charger__charger_simulation', None))
                if sim is None:
                    continue
                try:
                    sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                    connected = t_now < len(sa) and float(sa[t_now]) == 1.0
                    if connected:
                        ev_connected_steps += 1
                        ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                        rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                        ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                        soc = 0.0
                        if ev_obj and ev_obj.battery:
                            soc_arr = np.asarray(ev_obj.battery.soc, dtype=float)
                            t_idx = max(0, t_now - 1)
                            soc = float(np.clip(soc_arr[t_idx], 0, 1)) if 0 <= t_idx < len(soc_arr) else 0.0
                        ev_tracker[key] = {'was': True, 'soc': soc, 'req': rs}
                    else:
                        prev = ev_tracker.get(key, {})
                        if prev.get('was', False):
                            ev_departures += 1
                            deficit = max(0.0, prev['req'] - prev['soc'])
                            ev_departure_details.append((prev['soc'], prev['req'], deficit))
                            if deficit > 0.01:
                                ev_violated_departures += 1
                                ev_deficit_kwh += deficit
                        ev_tracker[key] = {'was': False}
                except:
                    pass

        # Step
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        total_steps += 1
        total_reward += float(reward)

        t_idx = max(0, int(getattr(raw, 'time_step', 0)) - 1)

        # C2: Battery SoC
        for bld in buildings:
            es = getattr(bld, 'electrical_storage', None)
            if es and hasattr(es, 'soc') and len(es.soc) > t_idx:
                soc = float(es.soc[t_idx])
                c2_total_checks += 1
                if soc < 0.0 or soc > 0.95:
                    c2_violations += 1

        # C3: Per-building power (abs(NEC) > P_bmax)
        district_nec = 0.0
        district_baseline = 0.0
        for b_idx, bld in enumerate(buildings):
            nec = getattr(bld, 'net_electricity_consumption', None)
            if nec is not None and hasattr(nec, '__len__') and len(nec) > t_idx:
                p_i = float(nec[t_idx])
                district_nec += p_i
                c3_total_per_building[b_idx] += 1
                if abs(p_i) > P_BMAX:
                    c3_per_building[b_idx] += 1
            try:
                nsl = getattr(bld, '_Building__energy_to_non_shiftable_load', [])
                sg = getattr(bld, '_Building__solar_generation', [])
                base_nec = 0.0
                if len(nsl) > t_idx:
                    base_nec += float(nsl[t_idx])
                if len(sg) > t_idx:
                    base_nec += float(sg[t_idx])
                district_baseline += base_nec
            except:
                pass

        nec_history.append(district_nec)
        nec_baseline_history.append(district_baseline)

        # C4: Grid power
        grid_import = max(0.0, district_nec)
        if grid_import > P_GMAX:
            c4_violations += 1

        # EV action tracking (indices 1, 6, 8 are EV actions)
        ev_indices = [1, 6, 8]
        for ei in ev_indices:
            if ei < len(action):
                if action[ei] > 0.05:
                    ev_charge_steps += 1
                elif action[ei] < -0.05:
                    ev_v2g_steps += 1
                else:
                    ev_idle_steps += 1

    elapsed = time.time() - t0

    # Compute KPIs
    nec_arr = np.array(nec_history)
    base_arr = np.array(nec_baseline_history)
    agent_import = np.sum(np.maximum(0, nec_arr))
    base_import = np.sum(np.maximum(0, base_arr))
    kpi_consumption = agent_import / max(base_import, 1e-6)

    agent_ramp = np.sum(np.maximum(0, np.diff(nec_arr)))
    base_ramp = np.sum(np.maximum(0, np.diff(base_arr)))
    kpi_ramping = agent_ramp / max(base_ramp, 1e-6)

    n_days = total_steps // 24
    agent_peaks = [np.max(nec_arr[d*24:(d+1)*24]) for d in range(n_days)]
    base_peaks = [np.max(base_arr[d*24:(d+1)*24]) for d in range(n_days)]
    kpi_daily_peak = np.mean(agent_peaks) / max(np.mean(base_peaks), 1e-6)

    # C3 totals
    c3_total = sum(c3_per_building)
    c3_total_checks = sum(c3_total_per_building)

    # EV action percentages
    ev_total_actions = ev_charge_steps + ev_v2g_steps + ev_idle_steps
    charge_pct = 100 * ev_charge_steps / max(ev_total_actions, 1)
    v2g_pct = 100 * ev_v2g_steps / max(ev_total_actions, 1)

    # Departure SoC statistics
    dep_socs = [d[0] for d in ev_departure_details]
    req_socs = [d[1] for d in ev_departure_details]

    # Print results
    print(f"\n{'─'*60}")
    print(f"  {run_name} RESULTS (physics-based)")
    print(f"{'─'*60}")

    print(f"\n  CONSTRAINT VIOLATIONS:")
    print(f"    C0 (EV departure): {ev_violated_departures}/{ev_departures} = {100*ev_violated_departures/max(ev_departures,1):.1f}%")
    print(f"       Mean dep SoC: {np.mean(dep_socs):.3f}, Mean req SoC: {np.mean(req_socs):.3f}")
    print(f"       Total deficit: {ev_deficit_kwh:.3f} (SoC units)")
    print(f"    C2 (battery SoC): {c2_violations}/{c2_total_checks} = {100*c2_violations/max(c2_total_checks,1):.1f}%")
    print(f"    C3 (building power):")
    for b_idx in range(n_b):
        pct = 100 * c3_per_building[b_idx] / max(c3_total_per_building[b_idx], 1)
        print(f"       Building_{b_idx+1}: {c3_per_building[b_idx]}/{c3_total_per_building[b_idx]} = {pct:.1f}%")
    print(f"       TOTAL: {c3_total}/{c3_total_checks} = {100*c3_total/max(c3_total_checks,1):.1f}%")
    print(f"    C4 (grid power): {c4_violations}/{total_steps} = {100*c4_violations/max(total_steps,1):.1f}%")

    print(f"\n  EV BEHAVIOR:")
    print(f"    Charge: {charge_pct:.1f}%  V2G: {v2g_pct:.1f}%  Connected steps: {ev_connected_steps}")

    print(f"\n  CITYLEARN KPIs:")
    print(f"    Electricity consumption: {kpi_consumption:.3f}")
    print(f"    Ramping:                 {kpi_ramping:.3f}")
    print(f"    Daily peak:              {kpi_daily_peak:.3f}")
    print(f"    Import: {agent_import:.0f} kWh, Export: {abs(np.sum(np.minimum(0, nec_arr))):.0f} kWh")

    print(f"\n  Total reward: {total_reward:.0f}, Steps: {total_steps}, Time: {elapsed:.0f}s")

    return {
        'run': run_name,
        'reward': total_reward,
        'c0_violated': ev_violated_departures,
        'c0_departures': ev_departures,
        'c0_pct': 100 * ev_violated_departures / max(ev_departures, 1),
        'c0_mean_dep_soc': float(np.mean(dep_socs)) if dep_socs else 0,
        'c0_mean_req_soc': float(np.mean(req_socs)) if req_socs else 0,
        'c2_pct': 100 * c2_violations / max(c2_total_checks, 1),
        'c3_per_building': [100*c3_per_building[i]/max(c3_total_per_building[i],1) for i in range(n_b)],
        'c3_total_pct': 100 * c3_total / max(c3_total_checks, 1),
        'c4_pct': 100 * c4_violations / max(total_steps, 1),
        'ev_charge_pct': charge_pct,
        'ev_v2g_pct': v2g_pct,
        'kpi_consumption': kpi_consumption,
        'kpi_ramping': kpi_ramping,
        'kpi_daily_peak': kpi_daily_peak,
        'import_kwh': agent_import,
        'export_kwh': abs(np.sum(np.minimum(0, nec_arr))),
        'elapsed': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', default=['R25b-stable', 'R25b-SAC-cap10', 'R26h-lean', 'R26i-full'])
    args = parser.parse_args()

    results = []
    for run_name in args.runs:
        if run_name not in RUN_CONFIGS:
            print(f"Unknown run: {run_name}")
            continue
        try:
            r = evaluate_run(run_name, RUN_CONFIGS[run_name])
            if r:
                results.append(r)
        except Exception as e:
            print(f"ERROR evaluating {run_name}: {e}")
            import traceback
            traceback.print_exc()

    # Final comparison table
    if results:
        print(f"\n{'='*110}")
        print(f"  COMPARISON TABLE (physics-based C0)")
        print(f"{'='*110}")
        print(f"{'Run':>12s} | {'Reward':>8s} | {'C0':>12s} | {'C2%':>5s} | {'C3%':>5s} | {'C4%':>5s} | {'Charge%':>7s} | {'V2G%':>5s} | {'Import':>8s} | {'KPI_elec':>8s}")
        print("-" * 110)
        for r in results:
            c0_str = f"{r['c0_violated']}/{r['c0_departures']}={r['c0_pct']:.1f}%"
            print(f"{r['run']:>12s} | {r['reward']:>+8.0f} | {c0_str:>12s} | {r['c2_pct']:>4.1f}% | {r['c3_total_pct']:>4.1f}% | {r['c4_pct']:>4.1f}% | {r['ev_charge_pct']:>6.1f}% | {r['ev_v2g_pct']:>4.1f}% | {r['import_kwh']:>7.0f} | {r['kpi_consumption']:>8.3f}")

    # Save results
    out_path = os.path.join(PROJECT, "eval_results", "r26hi_vs_r25b.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
