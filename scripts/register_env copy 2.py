# scripts/register_env.py
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv

def _thunk():
    base = make_base_env(central_agent=True)
    env = CityLearnSafetyEnv(base, soc_min=0.1, soc_max=0.9)

    # Use CityLearn’s horizon if available; else ~1 year @ 15 min steps.
    max_steps = getattr(base, "time_steps", None)
    if not isinstance(max_steps, int) or max_steps <= 0:
        max_steps = 35040
    return TimeLimit(env, max_episode_steps=max_steps)

# re-register
if "CityLearnSafety-SoC-v0" in gym.envs.registry:
    gym.envs.registry.pop("CityLearnSafety-SoC-v0")
gym.register(id="CityLearnSafety-SoC-v0", entry_point=_thunk)

if __name__ == "__main__":
    env = gym.make("CityLearnSafety-SoC-v0")
    print("Obs:", env.observation_space, "Act:", env.action_space)
