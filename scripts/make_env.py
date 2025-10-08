# scripts/make_env.py
from __future__ import annotations
import os, json
from citylearn_safe.adapters import SingleAgentListAdapter

def make_base_env(central_agent: bool = True):
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

    env = CityLearnEnv(schema=schema, central_agent=central_agent)

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
