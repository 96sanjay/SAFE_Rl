# scripts/register_env.py
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv
from citylearn_safe.safety_env import CityLearnSafetyEnv


def _thunk_v1():
    base = make_base_env(central_agent=True)
    env = CityLearnSafetyEnv(base, soc_min=0.1, soc_max=0.9)
    max_steps = getattr(base, "time_steps", None)
    if not isinstance(max_steps, int) or max_steps <= 0:
        max_steps = 35040
    return TimeLimit(env, max_episode_steps=max_steps)


def _thunk_v3():
    base = make_base_env(central_agent=True)
    env = CityLearnSafetyEnv(base, soc_min=0.1, soc_max=0.9)
    max_steps = getattr(base, "time_steps", None)
    if not isinstance(max_steps, int) or max_steps <= 0:
        max_steps = 35040
    return TimeLimit(env, max_episode_steps=max_steps)


def _safe_register(env_id: str, thunk):
    # Overwrite any existing env_id in *this Python process*
    try:
        gym.envs.registry.pop(env_id)
    except Exception:
        pass
    gym.register(id=env_id, entry_point=thunk)


_safe_register("Simple-v0", _thunk_v1)
_safe_register("SimpleV3-v0", _thunk_v3)


if __name__ == "__main__":
    env1 = gym.make("Simple-v0")
    env3 = gym.make("SimpleV3-v0")

    assert isinstance(env1.unwrapped, CityLearnSafetyEnv), f"Simple-v0 failed, got {type(env1.unwrapped)}"
    assert isinstance(env3.unwrapped, CityLearnSafetyEnv), f"SimpleV3-v0 failed, got {type(env3.unwrapped)}"

    print("✅ Simple-v0      ->", type(env1.unwrapped), "Act:", env1.action_space)
    print("✅ SimpleV3-v0    ->", type(env3.unwrapped), "Act:", env3.action_space)
