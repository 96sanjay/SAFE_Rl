#!/usr/bin/env python3
"""
Evaluate a trained PPO-Lag checkpoint from OmniSafe.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn.citylearn import CityLearnEnv

def evaluate_checkpoint(checkpoint_path, episodes=1):
    print("="*80)
    print("EVALUATING TRAINED SAFE PPO-LAG AGENT")
    print("="*80)
    
    # Load checkpoint
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Create environment
    schema = os.environ.get("CITYLEARN_SCHEMA")
    print(f"🏗️  Creating environment: {schema}")
    
    base_env = CityLearnEnv(schema=schema)
    env = CityLearnSafetyEnvV3(base_env)
    
    # Extract policy (actor network)
    actor_state = checkpoint['actor']
    
    # Create actor network (match training config: 64x64)
    from torch import nn
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    class Actor(nn.Module):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, act_dim),
                nn.Tanh()  # Actions in [-1, 1]
            )
        
        def forward(self, obs):
            return self.net(obs)
    
    actor = Actor(obs_dim, act_dim)
    actor.load_state_dict(actor_state)
    actor.eval()
    
    print(f"✅ Loaded actor network: {obs_dim} → 64 → 64 → {act_dim}")
    
    # Evaluate
    print(f"\n🧪 Running {episodes} evaluation episode(s)...\n")
    
    for ep in range(episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        step = 0
        
        ep_reward = 0.0
        ep_cost = 0.0
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = actor(obs_tensor).squeeze(0).numpy()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_cost += info.get('cost', 0.0)
            step += 1
            
            if step % 1000 == 0:
                print(f"  Step {step}: reward={reward:.3f}, cost={info.get('cost', 0.0):.3f}")
        
        print(f"\n{'='*80}")
        print(f"EPISODE {ep+1} RESULTS:")
        print(f"{'='*80}")
        print(f"Total Reward:  {ep_reward:.2f}")
        print(f"Total Cost:    {ep_cost:.2f}")
        print(f"Steps:         {step}")
        print(f"{'='*80}\n")
    
    return ep_cost

if __name__ == "__main__":
    checkpoint = "./runs/ppo_lag_3constraints_lambda35_evweight3_100ep/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-01-14-08-00-59/torch_save/epoch-100.pt"
    
    cost = evaluate_checkpoint(checkpoint, episodes=1)
    
    print(f"\n🎯 FINAL EVALUATION:")
    print(f"   Cost: {cost:.2f}")
    print(f"   Target: < 350.0")
    print(f"   Status: {'✅ PASS' if cost < 350 else '❌ FAIL'}")
