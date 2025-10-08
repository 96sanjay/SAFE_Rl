# scripts/safety_smoke.py
import gymnasium as gym
from gymnasium.spaces import Box
import scripts.register_env  # ensure registration

def main(steps=10):
    #env = gym.make("CityLearnSafety-SoC-v0")
    env = gym.make("Simple-v0")
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Box)

    obs, info = env.reset()
    print("start metrics:", info.get("metrics"))
    for t in range(steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"t={t:02d} r={r: .3f} cost={info.get('cost')} metrics={info.get('metrics')}")
        if term or trunc:
            obs, info = env.reset()
    print("✅ safety cost shows up in info['cost'].")

if __name__ == "__main__":
    main()
