"""Tests for ActionProjectionSERL — EV-only topology and infeasibility."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set env vars before imports
os.environ.update({
    "CITYLEARN_SCHEMA": os.path.join(
        os.path.dirname(__file__), "..",
        "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
    ),
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_SERL_PROJECTION": "1",
    "SE_RL_PENALTY_WEIGHT": "0.5",
    "MASK_C4_ENABLED": "1",
    "CITYLEARN_BATT_CLAMP": "0",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_KPI_RUN_NAME": "__test__",
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
})


def _build_env():
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.action_projection_serl import ActionProjectionSERL

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    forecast = ForecastObsWrapper(safety, forecast_horizon=24)
    return ActionProjectionSERL(forecast)


def test_ev_only_no_crash():
    """EV-only building must not crash with ZeroDivisionError."""
    env = _build_env()

    # Remove battery for building 0 to simulate EV-only
    env._building_batt_act.pop(0, None)
    env._batt_powers.pop(0, None)

    obs, _ = env.reset()
    for t in range(50):
        action = np.random.uniform(-1, 1, size=env.action_space.shape)
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            break

    assert True, "EV-only building did not crash"


def test_ev_only_bounds():
    """EV-only building: battery bounds untouched, EV bounds restricted."""
    env = _build_env()

    # Read indices BEFORE mutating (dynamic, not hardcoded)
    batt_act_idx = env._building_batt_act.get(0)  # e.g., 0
    ev_act_idx = env._building_ev_act.get(0)      # e.g., 1

    assert batt_act_idx is not None, "Building 0 must have battery to test removal"
    assert ev_act_idx is not None, "Building 0 must have EV"

    # Remove battery to simulate EV-only
    env._building_batt_act.pop(0, None)
    env._batt_powers.pop(0, None)

    obs, _ = env.reset()
    action = np.random.uniform(-1, 1, size=env.action_space.shape)
    obs, r, term, trunc, info = env.step(action)

    # Battery action at its dynamic index: should be untouched [-1, 1]
    assert abs(info["serl_safe_min"][batt_act_idx] - (-1.0)) < 0.01, \
        f"Battery safe_min should be -1.0, got {info['serl_safe_min'][batt_act_idx]}"
    assert abs(info["serl_safe_max"][batt_act_idx] - 1.0) < 0.01, \
        f"Battery safe_max should be 1.0, got {info['serl_safe_max'][batt_act_idx]}"

    # EV action at its dynamic index: should be restricted by C3
    ev_restricted = (info["serl_safe_max"][ev_act_idx] < 1.0 or
                     info["serl_safe_min"][ev_act_idx] > -1.0)
    assert ev_restricted, "EV bounds should be restricted by C3 headroom"


def test_structural_infeasibility_reported():
    """When exogenous load exceeds P_building_max with no device to fix it,
    serl_n_infeasible should be > 0."""
    env = _build_env()

    # Remove ALL devices from building 0 to make it uncontrollable
    env._building_batt_act.pop(0, None)
    env._batt_powers.pop(0, None)
    env._building_ev_act.pop(0, None)
    env._ev_max_charge.pop(0, None)

    obs, _ = env.reset()
    # Run until we hit a high-load step
    found_infeasible = False
    for t in range(500):
        action = np.zeros(env.action_space.shape)
        obs, r, term, trunc, info = env.step(action)
        if info.get("serl_n_infeasible", 0) > 0:
            found_infeasible = True
            break
        if term or trunc:
            break

    # Building 0 has base load up to 8.0 kW (> P_bmax=4.6).
    # With no devices, some steps MUST be structurally infeasible.
    assert found_infeasible, \
        "Expected structural infeasibility when building has no devices and high load"


def test_penalty_zero_inside_bounds():
    """Penalty must be exactly zero when raw action is inside safe bounds."""
    env = _build_env()
    obs, _ = env.reset()

    # Zero action should usually be inside bounds
    action = np.zeros(env.action_space.shape)
    obs, r, term, trunc, info = env.step(action)

    # Check: if action=0 is inside all bounds, penalty should be 0
    safe_min = np.array(info["serl_safe_min"])
    safe_max = np.array(info["serl_safe_max"])
    inside = np.all((action >= safe_min) & (action <= safe_max))
    if inside:
        assert info["serl_penalty"] == 0.0, \
            f"Penalty should be 0 when inside bounds, got {info['serl_penalty']}"


def test_penalty_nonzero_outside_bounds():
    """Penalty must be > 0 when raw action exceeds safe bounds."""
    env = _build_env()
    obs, _ = env.reset()

    # Extreme action: all +1.0 — will exceed safe bounds on constrained dims
    action = np.ones(env.action_space.shape)
    obs, r, term, trunc, info = env.step(action)

    assert info["serl_penalty"] > 0, \
        f"Penalty should be > 0 for extreme actions, got {info['serl_penalty']}"
    assert info["serl_n_clipped"] > 0, \
        f"Some dimensions should be clipped, got {info['serl_n_clipped']}"


if __name__ == "__main__":
    tests = [
        test_ev_only_no_crash,
        test_ev_only_bounds,
        test_structural_infeasibility_reported,
        test_penalty_zero_inside_bounds,
        test_penalty_nonzero_outside_bounds,
    ]
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError as e:
            print(f"FAIL: {test.__name__} — {e}")
        except Exception as e:
            print(f"ERROR: {test.__name__} — {type(e).__name__}: {e}")
