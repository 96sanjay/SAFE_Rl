# scripts/make_env.py
from __future__ import annotations
import os, json, hashlib
from typing import Any, Mapping
import numpy as np
from citylearn_safe.adapters import SingleAgentListAdapter


def _valid_split_cache_path(schema_path: str, episode_time_steps: int) -> str:
    validation_version = "v3_full_episode_zero_action"
    with open(schema_path, "rb") as f:
        schema_digest = hashlib.md5(f.read()).hexdigest()
    key = hashlib.md5(
        f"{os.path.abspath(schema_path)}::{schema_digest}::{episode_time_steps}::{validation_version}".encode("utf-8")
    ).hexdigest()
    return os.path.join("/tmp", f"citylearn_valid_splits_{key}.json")


def _compute_valid_episode_splits(
    schema: dict[str, Any],
    schema_path: str,
    central_agent: bool,
    episode_time_steps: int,
) -> list[list[int]]:
    from citylearn.citylearn import CityLearnEnv

    cache_path = _valid_split_cache_path(schema_path, episode_time_steps)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if isinstance(cached, list) and cached:
            return cached

    probe = CityLearnEnv(
        schema=schema,
        central_agent=central_agent,
        episode_time_steps=episode_time_steps,
        rolling_episode_split=True,
        random_episode_split=False,
    )
    probe = SingleAgentListAdapter(probe)
    valid_splits: list[list[int]] = []
    zero_action = np.zeros(probe.action_space.shape, dtype=np.float32)
    tracker = getattr(probe.base, "episode_tracker", None)
    simulation_time_steps = int(getattr(tracker, "simulation_time_steps", 0) or 0)
    candidate_count = max(1, simulation_time_steps - int(episode_time_steps) + 1)

    for _ in range(candidate_count):
        probe.reset()
        start = int(getattr(tracker, "episode_start_time_step", 0))
        end = int(getattr(tracker, "episode_end_time_step", start + int(episode_time_steps) - 1))
        try:
            done = False
            while not done:
                _, _, terminated, truncated, _ = probe.step(zero_action)
                done = bool(terminated) or bool(truncated)
        except Exception:
            continue
        valid_splits.append([start, end])

    probe.close()
    if not valid_splits:
        raise RuntimeError(
            f"No valid {episode_time_steps}-step episode splits found for schema {schema_path}."
        )

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(valid_splits, f)
    return valid_splits

def make_base_env(
    central_agent: bool = True,
    env_kwargs: Mapping[str, Any] | None = None,
):
    from citylearn.citylearn import CityLearnEnv

    schema_path = os.environ.get("CITYLEARN_SCHEMA")
    if not schema_path:
        raise RuntimeError(
            'Set CITYLEARN_SCHEMA to your local schema.json, e.g.:\n'
            'export CITYLEARN_SCHEMA="$PWD/data/citylearn/schema.json"'
        )

    # Load schema and FORCE root_directory to the folder holding schema.json
    with open(schema_path, "r") as f:
        schema = json.load(f)
    schema_dir = os.path.dirname(os.path.abspath(schema_path))
    schema["root_directory"] = schema_dir  # <-- critical

    # Optional: sanity-check a few referenced files exist
    missing = []
    for bname, bconf in schema.get("buildings", {}).items():
        rel = bconf.get("energy_simulation")
        if rel:
            p = os.path.join(schema["root_directory"], rel)
            if not os.path.exists(p):
                missing.append(p)
    if missing:
        raise FileNotFoundError("Missing files:\n- " + "\n- ".join(missing))

    kwargs = dict(env_kwargs or {})
    if bool(kwargs.pop("validate_episode_splits", False)):
        episode_time_steps = kwargs.get("episode_time_steps")
        if not isinstance(episode_time_steps, int):
            raise ValueError("validate_episode_splits requires integer episode_time_steps")
        kwargs["episode_time_steps"] = _compute_valid_episode_splits(
            schema=schema,
            schema_path=schema_path,
            central_agent=central_agent,
            episode_time_steps=episode_time_steps,
        )
        kwargs["rolling_episode_split"] = False
        kwargs["random_episode_split"] = True
    env = CityLearnEnv(schema=schema, central_agent=central_agent, **kwargs)

    # Optional normalization for stability
    try:
        from citylearn.wrappers import NormalizedObservationWrapper
        env = NormalizedObservationWrapper(env)
    except Exception:
        pass
    env = SingleAgentListAdapter(env)
    return env

if __name__ == "__main__":
    env = make_base_env(central_agent=True)
    obs, info = env.reset()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("First obs dtype:", getattr(obs, "dtype", type(obs)))
