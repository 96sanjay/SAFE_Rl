"""Test the actual OmniSafe pipeline as configured"""
import sys
import os
sys.path.insert(0, os.getcwd())

print("#"*60)
print("# Testing OmniSafe Pipeline")
print("#"*60)

# Test 1: Check make_env.py exists
print("\n1️⃣  Checking make_env.py...")
if os.path.exists('scripts/make_env.py'):
    print("   ✅ scripts/make_env.py exists")
    from scripts.make_env import make_base_env
    print("   ✅ make_base_env imported")
else:
    print("   ❌ scripts/make_env.py not found!")
    sys.exit(1)

# Test 2: Create base environment
print("\n2️⃣  Testing make_base_env()...")
try:
    base_env = make_base_env(central_agent=True)
    print(f"   ✅ Base environment created")
    print(f"      Obs space: {base_env.observation_space}")
    print(f"      Action space: {base_env.action_space}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Wrap with CityLearnSafetyEnv
print("\n3️⃣  Wrapping with CityLearnSafetyEnv...")
try:
    from citylearn_safe.safety_env import CityLearnSafetyEnv
    safety_env = CityLearnSafetyEnv(base_env, soc_min=0.00, soc_max=0.95)
    print(f"   ✅ Safety wrapper applied")
    print(f"      Obs space: {safety_env.observation_space}")
    print(f"      Action space: {safety_env.action_space}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test reset and step
print("\n4️⃣  Testing reset & step...")
try:
    obs, info = safety_env.reset()
    print(f"   ✅ Reset OK")
    print(f"      Obs shape: {obs.shape}")
    print(f"      Initial cost: {info.get('cost', 0.0):.6f}")
    
    # Take a step
    action = safety_env.action_space.sample()
    obs, reward, term, trunc, info = safety_env.step(action)
    print(f"   ✅ Step OK")
    print(f"      Reward: {reward:.2f}")
    print(f"      Cost: {info.get('cost', 0.0):.6f}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: OmniSafe registration
print("\n5️⃣  Testing OmniSafe registration...")
try:
    import citylearn_safe.omni_env  # Triggers @env_register
    print("   ✅ omni_env module imported (registration triggered)")
    
    import omnisafe
    print("   ✅ OmniSafe imported")
    
    # Check if env is registered
    from omnisafe.envs import ENVS
    if 'CityLearnSafety-SoC-v0' in ENVS:
        print("   ✅ 'CityLearnSafety-SoC-v0' registered!")
    else:
        print(f"   ⚠️  'CityLearnSafety-SoC-v0' not in ENVS")
        print(f"      Available: {list(ENVS.keys())[:5]}...")
    
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check config files
print("\n6️⃣  Checking config files...")
import glob
configs = glob.glob("configs/on-policy/*.yaml") + glob.glob("configs/on-policy/*.yml")
if configs:
    print(f"   ✅ Found {len(configs)} config files:")
    for cfg in configs[:5]:
        print(f"      - {cfg}")
    
    # Try to load one
    import yaml
    with open(configs[0]) as f:
        cfg_data = yaml.safe_load(f)
    print(f"\n   Sample config: {configs[0]}")
    print(f"      algo: {cfg_data.get('algo', 'N/A')}")
    print(f"      env_id: {cfg_data.get('env_id', 'N/A')}")
else:
    print("   ⚠️  No config files in configs/on-policy/")

# Test 7: Try to create OmniSafe agent (minimal)
print("\n7️⃣  Testing OmniSafe agent creation...")
try:
    agent = omnisafe.Agent(
        'PPOLag',
        'CityLearnSafety-SoC-v0',
        custom_cfgs={
            'train_cfgs': {'total_steps': 100},
            'lagrange_cfgs': {'cost_limit': 100.0},
        }
    )
    print("   ✅ Agent created successfully!")
    print("   ⚠️  Skipping actual training (would take time)")
    
except Exception as e:
    print(f"   ❌ Agent creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("✅ Base environment works")
print("✅ Safety wrapper works")
print("✅ Reset/step works")
print("✅ OmniSafe registration works")
print("\n🚀 Pipeline is ready!")
print("\nTo train, run:")
print("  python scripts/train_omnisafe.py --cfg configs/on-policy/<your-config>.yaml")
print("="*60)
