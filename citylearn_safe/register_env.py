"""Register CityLearn environments with Gymnasium"""
from gymnasium.envs.registration import register

# Register the main CityLearn Safety environment
register(
    id='CityLearnSafety-SoC-v0',
    entry_point='citylearn_safe.gym_env:CityLearnGymEnv',
    max_episode_steps=8760,
)

print("✓ Registered CityLearnSafety-SoC-v0")
