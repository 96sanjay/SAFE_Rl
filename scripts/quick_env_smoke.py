# scripts/quick_env_smoke.py
from __future__ import annotations
import gymnasium as gym
from gymnasium.spaces import Box
from scripts.make_env import make_base_env

def main(steps: int = 5):
    env = make_base_env(central_agent=True)

    # sanity: spaces should be plain Box (not lists)
    assert isinstance(env.observation_space, Box), f"obs space not Box: {env.observation_space}"
    assert isinstance(env.action_space, Box), f"action space not Box: {env.action_space}"

    obs, info = env.reset()
    print("obs dtype/shape:", getattr(obs, "dtype", type(obs)), getattr(obs, "shape", None))
    print("action space:", env.action_space)

    for t in range(steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"t={t:02d} r={float(r): .3f} done={term or trunc}")
        if term or trunc:
            obs, info = env.reset()

    print("✅ CityLearn base env smoke test passed.")

if __name__ == "__main__":
    main()
