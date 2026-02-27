from scripts.make_env import make_base_env
import numpy as np

env = make_base_env()

obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

print("Observation shape:", np.array(obs).shape)

action = env.action_space.sample()

out = env.step(action)

if len(out) == 5:
    next_obs, reward, terminated, truncated, info = out
    done = terminated or truncated
else:
    next_obs, reward, done, info = out

print("Info keys:", info.keys())
print("Full info dict:", info)
