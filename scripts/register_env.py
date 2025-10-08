# scripts/register_env.py
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv

def _thunk():
    base = make_base_env(central_agent=True)
    env = CityLearnSafetyEnv(base, soc_min=0.1, soc_max=0.9)
    max_steps = getattr(base, "time_steps", None)
    if not isinstance(max_steps, int) or max_steps <= 0:
        max_steps = 35040
    return TimeLimit(env, max_episode_steps=max_steps)

# Overwrite any existing 'Simple-v0' *in this Python process*
try:
    gym.envs.registry.pop("Simple-v0")
except Exception:
    pass
gym.register(id="Simple-v0", entry_point=_thunk)

if __name__ == "__main__":
    env = gym.make("Simple-v0")
    # sanity: verify alias points to our safety wrapper
    from citylearn_safe.safety_env import CityLearnSafetyEnv
    assert isinstance(env.unwrapped, CityLearnSafetyEnv), f"Alias failed, got {type(env.unwrapped)}"
    print("Obs:", env.observation_space, "Act:", env.action_space)
