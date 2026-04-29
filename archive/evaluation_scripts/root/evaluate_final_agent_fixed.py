"""
Evaluate final trained PPO-Lag agent
"""
import os
os.environ['CITYLEARN_SCHEMA'] = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
os.environ['CITYLEARN_KPI_RUN_NAME'] = "EvaluateFinalAgent_Lambda35_Epoch100"

import torch
import numpy as np
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn.citylearn import CityLearnEnv

print("="*80)
print("EVALUATING FINAL TRAINED AGENT (Epoch 100)")
print("="*80)

# Load checkpoint
checkpoint_path = "./runs/ppo_lag_3constraints_lambda35_evweight3_100ep/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-01-14-08-00-59/torch_save/epoch-100.pt"
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Create environment
schema_path = os.environ['CITYLEARN_SCHEMA']
print(f"Creating environment...")
base_env = CityLearnEnv(schema=schema_path)
env = CityLearnSafetyEnvV3(base_env)

# Get observation and action dimensions
obs_space = env.observation_space
act_space = env.action_space
obs_dim = obs_space.shape[0]
act_dim = act_space.shape[0]

print(f"Observation dim: {obs_dim}")
print(f"Action dim: {act_dim}")

# Create actor-critic model (matching training config)
model_cfgs = {
    'actor': {
        'hidden_sizes': [64, 64],
        'activation': 'tanh',
    },
    'critic': {
        'hidden_sizes': [64, 64],
        'activation': 'tanh',
    }
}

# Create the model
actor_critic = ConstraintActorCritic(
    obs_space=obs_space,
    act_space=act_space,
    model_cfgs=model_cfgs,
    epochs=100,
)

# Load weights
actor_critic.load_state_dict(checkpoint)
actor_critic.eval()

print("✅ Model loaded successfully")
print("\n" + "="*80)
print("RUNNING EVALUATION (1 EPISODE)")
print("="*80)

# Run evaluation
obs, info = env.reset(seed=42)
done = False
step = 0
total_reward = 0
total_cost = 0

with torch.no_grad():
    while not done:
        # Convert obs to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # Get action from policy (deterministic)
        action = actor_critic.actor.predict(obs_tensor, deterministic=True)
        action = action.cpu().numpy().flatten()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        total_cost += info.get('cost', 0.0)
        step += 1
        
        if step % 1000 == 0:
            print(f"Step {step}: Reward={reward:.2f}, Cost={info.get('cost', 0.0):.2f}")

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Total Reward: {total_reward:.2f}")
print(f"Total Cost:   {total_cost:.2f}")
print(f"Steps:        {step}")
print(f"\n✅ Detailed KPIs saved to: runs/kpi_logs/EvaluateFinalAgent_Lambda35_Epoch100.csv")
print("="*80)
