# tests/test_r28_pposaute_config.py
"""Validate that r28_pposaute.yaml is structurally correct and loadable by OmniSafe."""
from __future__ import annotations
import os
import yaml
import pytest

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG_PATH = os.path.join(_PROJECT_ROOT, "configs", "active", "benchmark_pposaute.yaml")


def test_config_file_exists():
    assert os.path.isfile(CFG_PATH), f"{CFG_PATH} not found"


def test_config_parses_as_yaml():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)


def test_config_has_required_top_level_keys():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    for key in ("algo", "env_id", "seed", "train_cfgs", "algo_cfgs", "model_cfgs", "logger_cfgs"):
        assert key in cfg, f"missing top-level key: {key}"


def test_algo_is_pposaute():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert cfg["algo"] == "PPOSaute"


def test_env_id_matches_r28():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert cfg["env_id"] == "CityLearnSafety-V2G-v2"


def test_saute_specific_fields_present():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    for key in ("safety_budget", "saute_gamma", "unsafe_reward", "max_ep_len"):
        assert key in cfg["algo_cfgs"], f"missing Saute field: algo_cfgs.{key}"


def test_saute_specific_values():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    a = cfg["algo_cfgs"]
    assert a["safety_budget"] == 400000.0
    assert a["saute_gamma"] == 0.9999
    assert a["unsafe_reward"] == -1.0
    assert a["max_ep_len"] == 8759


def test_reward_normalize_off_for_saute_adapter():
    """Saute adapter hard-disables reward and cost normalization."""
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert cfg["algo_cfgs"]["reward_normalize"] is False
    assert cfg["algo_cfgs"]["cost_normalize"] is False


def test_no_lagrange_cfgs_block():
    """PPOSaute has no Lagrangian multiplier."""
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert "lagrange_cfgs" not in cfg


def test_steps_per_epoch_matches_r28():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert cfg["algo_cfgs"]["steps_per_epoch"] == 8759


def test_total_steps_is_100_epochs():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert cfg["train_cfgs"]["total_steps"] == 875900


def test_omnisafe_agent_loads_with_saute_adapter(tmp_path, monkeypatch):
    """PPOSaute agent instantiates, env loads, obs space is augmented by +1 dim."""
    # Env vars required by CityLearn omni_env (minimal subset; full set only needed for training)
    monkeypatch.chdir(_PROJECT_ROOT)
    monkeypatch.setenv("CITYLEARN_SCHEMA",
                       os.path.join(_PROJECT_ROOT,
                                    "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"))
    monkeypatch.setenv("CITYLEARN_CENTRAL_AGENT", "1")
    monkeypatch.setenv("CITYLEARN_REWARD_TYPE", "stems")
    monkeypatch.setenv("CITYLEARN_TEMPORAL_WINDOW", "0")
    monkeypatch.setenv("CITYLEARN_PID_LAGRANGE", "0")
    monkeypatch.setenv("CITYLEARN_EV_SAUTE", "0")

    import omnisafe
    import citylearn_safe.cmdp_env     # noqa: F401 - registers CityLearnSafety-V2G-v2

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Point logs at tmp_path so test doesn't pollute runs/
    cfg["logger_cfgs"]["log_dir"] = str(tmp_path)

    allowed = ("train_cfgs", "algo_cfgs", "logger_cfgs",
               "model_cfgs", "save_cfgs", "env_cfgs")
    custom_cfgs = {k: v for k, v in cfg.items() if k in allowed}

    agent = omnisafe.Agent(cfg["algo"], cfg["env_id"], custom_cfgs=custom_cfgs)
    # Adapter wires obs space; +1 for safety state
    env_obs_shape = agent.agent._env.observation_space.shape
    assert len(env_obs_shape) == 1
    assert env_obs_shape[0] >= 2, "obs space too small to be real"
    # The SauteAdapter augments by exactly +1
    # agent.agent._env is the SauteAdapter instance; its ._env is the underlying wrapped env
    # (chain of wrappers: Unsqueeze -> ActionScale -> ObsNormalize -> AutoReset -> raw env).
    # SauteAdapter.observation_space returns self._observation_space (augmented Box),
    # while self._env.observation_space returns the pre-augmentation space.
    underlying = agent.agent._env._env.observation_space.shape
    assert env_obs_shape[0] == underlying[0] + 1, (
        f"Saute should add +1 dim: augmented={env_obs_shape[0]} vs underlying={underlying[0]}"
    )
