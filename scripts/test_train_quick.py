#!/usr/bin/env python3
"""Quick training test - runs for ~2 minutes to verify pipeline"""
import sys
import os
sys.path.insert(0, os.getcwd())

print("="*60)
print("QUICK TRAINING TEST")
print("="*60)

# Check CITYLEARN_SCHEMA is set
schema = os.environ.get("CITYLEARN_SCHEMA")
if not schema:
    print("\n❌ CITYLEARN_SCHEMA not set!")
    print("\nSet it with:")
    print('export CITYLEARN_SCHEMA="/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"')
    sys.exit(1)

print(f"\n✅ CITYLEARN_SCHEMA: {schema}")
if not os.path.exists(schema):
    print(f"❌ Schema file not found: {schema}")
    sys.exit(1)

print("\nStarting quick training test...")
print("This will:")
print("  - Train PPO-Lag for 10,000 steps (~5 epochs)")
print("  - Take about 2-3 minutes")
print("  - Verify the complete pipeline works")
print("\n" + "="*60 + "\n")

try:
    # Import and run training
    import citylearn_safe.omni_env  # Register environment
    import omnisafe
    import yaml
    
    # Load config
    with open('configs/on-policy/test_quick.yaml') as f:
        cfg = yaml.safe_load(f)
    
    print(f"Config loaded:")
    print(f"  Algorithm: {cfg['algo']}")
    print(f"  Environment: {cfg['env_id']}")
    print(f"  Total steps: {cfg['train_cfgs']['total_steps']}")
    print(f"  Cost limit: {cfg['lagrange_cfgs']['cost_limit']}")
    print()
    
    # Create agent
    print("Creating OmniSafe agent...")
    allowed_cfgs = (
        'train_cfgs', 'algo_cfgs', 'logger_cfgs', 'lagrange_cfgs',
        'model_cfgs', 'save_cfgs', 'env_cfgs',
    )
    custom_cfgs = {k: v for k, v in cfg.items() if k in allowed_cfgs}
    
    agent = omnisafe.Agent(
        cfg['algo'],
        cfg['env_id'],
        custom_cfgs=custom_cfgs
    )
    print("✅ Agent created!\n")
    
    # Train
    print("Starting training...")
    print("-"*60)
    agent.learn()
    print("-"*60)
    
    print("\n✅ TRAINING TEST PASSED!")
    print(f"\nLogs saved to: {agent.logger.log_dir}")
    print("\nYou can now run full training with:")
    print("  python scripts/train_omnisafe.py --cfg configs/on-policy/ppo_lag_soc.yaml")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
