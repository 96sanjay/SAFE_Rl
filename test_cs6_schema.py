#!/usr/bin/env python3
"""
Test cs6 schema compatibility with CityLearn.

This script systematically tests each component to isolate issues.

Usage:
    python test_cs6_schema.py --schema data/cs6/cs6/schema.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    """ANSI colors for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def test_schema_structure(schema_path: str) -> Tuple[bool, Dict]:
    """
    Test 1: Verify schema structure is flat (not grouped).
    """
    print_header("TEST 1: Schema Structure")
    
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        print_success(f"Schema loaded: {schema_path}")
    except Exception as e:
        print_error(f"Failed to load schema: {e}")
        return False, {}
    
    # Check if observations are grouped
    obs = schema.get('observations', {})
    if 'buildings' in obs or 'ev_chargers' in obs:
        print_error("Schema has GROUPED structure (buildings/ev_chargers)")
        print("   Expected: flat structure with observation keys directly")
        print("   Found: grouped structure")
        print(f"   Keys: {list(obs.keys())[:10]}")
        return False, schema
    
    print_success("Schema has FLAT structure ✅")
    print(f"   Total observation keys: {len(obs)}")
    
    # Check if actions are grouped
    actions = schema.get('actions', {})
    if 'buildings' in actions or 'ev_chargers' in actions:
        print_error("Actions have GROUPED structure")
        return False, schema
    
    print_success("Actions have FLAT structure ✅")
    print(f"   Total action keys: {len(actions)}")
    
    return True, schema


def test_ev_observations(schema: Dict) -> bool:
    """
    Test 2: Verify all required EV observations are present.
    """
    print_header("TEST 2: EV Observations")
    
    obs = schema.get('observations', {})
    
    # Required EV observations (with alternatives)
    required = {
        'required_soc_departure': ['electric_vehicle_required_soc_departure'],
        'estimated_soc_arrival': ['electric_vehicle_estimated_soc_arrival'],
        'estimated_arrival_time': ['electric_vehicle_estimated_arrival_time', 'estimated_departure_time'],
        'ev_soc': ['soc', 'electric_vehicle_soc']
    }
    
    all_present = True
    for main_name, alternatives in required.items():
        found = main_name in obs
        found_alt = None
        
        if not found:
            for alt in alternatives:
                if alt in obs:
                    found = True
                    found_alt = alt
                    break
        
        if found:
            name = found_alt if found_alt else main_name
            active = obs[name].get('active', False)
            if active:
                print_success(f"{name}: present and active")
            else:
                print_warning(f"{name}: present but INACTIVE")
                all_present = False
        else:
            print_error(f"{main_name}: MISSING (checked alternatives: {alternatives})")
            all_present = False
    
    if all_present:
        print_success("\nAll required EV observations are present and active ✅")
    else:
        print_error("\nSome EV observations are missing or inactive ❌")
    
    return all_present


def test_temperature_observations(schema: Dict) -> bool:
    """
    Test 3: Verify temperature observations are present.
    """
    print_header("TEST 3: Temperature Observations")
    
    obs = schema.get('observations', {})
    
    # Required temperature observations
    temp_obs = {
        'indoor_dry_bulb_temperature': 'Indoor temperature',
        'cooling_demand': 'Cooling demand',
        'heating_demand': 'Heating demand'
    }
    
    all_present = True
    for obs_name, description in temp_obs.items():
        if obs_name in obs:
            active = obs[obs_name].get('active', False)
            if active:
                print_success(f"{obs_name}: present and active ({description})")
            else:
                print_warning(f"{obs_name}: present but INACTIVE")
                all_present = False
        else:
            print_error(f"{obs_name}: MISSING ({description})")
            all_present = False
    
    if all_present:
        print_success("\nAll temperature observations are present and active ✅")
    else:
        print_warning("\nSome temperature observations are missing or inactive")
        print("   (This is okay if you're testing without temperature control)")
    
    return all_present


def test_csv_files(schema: Dict, schema_path: str) -> bool:
    """
    Test 4: Verify CSV files exist and have correct columns.
    """
    print_header("TEST 4: CSV Files")
    
    schema_dir = Path(schema_path).parent
    
    # Check building CSVs
    buildings = schema.get('buildings', {})
    print(f"Checking {len(buildings)} building CSV files...")
    
    building_ok = True
    for i, (building_name, building_config) in enumerate(buildings.items()):
        if i >= 2:  # Only check first 2 buildings
            print(f"   ... (skipping remaining {len(buildings) - 2} buildings)")
            break
        
        # FIX: Handle different schema structures
        csv_file = building_config.get('include', None)
        if csv_file is None or isinstance(csv_file, bool):
            # Try alternative key
            csv_file = building_config.get('data', None)
        
        if not csv_file or isinstance(csv_file, bool):
            print_warning(f"{building_name}: No CSV file specified in schema")
            continue
            
        csv_path = schema_dir / csv_file
        
        if not csv_path.exists():
            print_error(f"{building_name}: CSV not found at {csv_path}")
            building_ok = False
            continue
        
        # Check for temperature columns
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, nrows=1)
            
            required_cols = ['indoor_dry_bulb_temperature']
            found_cols = [c for c in required_cols if c in df.columns]
            
            if found_cols:
                print_success(f"{building_name}: has temperature data ({', '.join(found_cols)})")
            else:
                print_warning(f"{building_name}: NO temperature columns found")
        
        except Exception as e:
            print_error(f"{building_name}: Failed to read CSV: {e}")
            building_ok = False
    
    # Check EV CSVs
    print(f"\nChecking EV CSV files...")
    
    ev_ok = True
    ev_count = 0
    
    # Find chargers in buildings
    for building_name, building_config in buildings.items():
        chargers = building_config.get('electric_vehicle_chargers', {})
        
        for charger_name, charger_config in chargers.items():
            ev_count += 1
            
            # Try different keys for CSV file
            csv_file = charger_config.get('charger_simulation', None)
            if csv_file is None:
                csv_file = charger_config.get('data', None)
            
            if not csv_file or isinstance(csv_file, bool):
                print_warning(f"{charger_name}: No CSV file specified")
                continue
            
            csv_path = schema_dir / csv_file
            
            if not csv_path.exists():
                print_error(f"{charger_name}: CSV not found at {csv_path}")
                ev_ok = False
                continue
            
            # Check for required EV columns
            try:
                import pandas as pd
                df = pd.read_csv(csv_path, nrows=1)
                
                required_cols = [
                    'electric_vehicle_required_soc_departure',
                    'electric_vehicle_estimated_soc_arrival',
                    'electric_vehicle_charger_state'
                ]
                
                missing = [c for c in required_cols if c not in df.columns]
                
                if not missing:
                    print_success(f"{charger_name}: all required columns present")
                else:
                    print_error(f"{charger_name}: missing columns: {missing}")
                    print(f"   Found columns: {list(df.columns)}")
                    ev_ok = False
            
            except Exception as e:
                print_error(f"{charger_name}: Failed to read CSV: {e}")
                ev_ok = False
            
            if ev_count >= 2:  # Only check first 2 chargers
                print(f"   ... (checked {ev_count} chargers)")
                break
        
        if ev_count >= 2:
            break
    
    if ev_count == 0:
        print_warning("No EV chargers found in schema")
        print("   This means EV control is not available")
    
    return building_ok and (ev_ok or ev_count == 0)


def test_environment_creation(schema_path: str) -> bool:
    """
    Test 5: Try to create CityLearn environment.
    """
    print_header("TEST 5: Environment Creation")
    
    try:
        # Import here to catch import errors
        from citylearn import CityLearnEnv
        
        print("Creating CityLearnEnv...")
        env = CityLearnEnv(schema=schema_path)
        
        print_success("Environment created successfully!")
        print(f"   Buildings: {len(env.buildings)}")
        print(f"   Action space: {env.action_space.shape}")
        print(f"   Observation space: {env.observation_space.shape}")
        
        # Try reset
        print("\nResetting environment...")
        obs = env.reset()
        print_success(f"Reset successful! Observation shape: {obs.shape}")
        
        # Try one step
        print("\nTaking one step...")
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print_success(f"Step successful! Reward: {reward:.2f}")
        
        return True
    
    except Exception as e:
        print_error(f"Environment creation failed: {e}")
        print(f"\n{Colors.RED}Full error:{Colors.END}")
        import traceback
        traceback.print_exc()
        return False


def test_safety_wrapper(schema_path: str) -> bool:
    """
    Test 6: Try to create Safety wrapper.
    """
    print_header("TEST 6: Safety Wrapper (Optional)")
    
    try:
        # Try importing the wrapper
        import sys
        sys.path.insert(0, '/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork')
        
        from scripts.make_env import make_base_env
        import os
        
        # Set environment variable
        os.environ['CITYLEARN_SCHEMA'] = schema_path
        
        print("Creating safety-wrapped environment...")
        env = make_base_env()
        
        print_success("Safety wrapper created successfully!")
        print(f"   Action space: {env.action_space.shape}")
        print(f"   Observation space: {env.observation_space.shape}")
        
        return True
    
    except ImportError as e:
        print_warning(f"Could not import safety wrapper (this is okay if not testing from thesis repo)")
        print(f"   Error: {e}")
        return True  # Not a failure
    
    except Exception as e:
        print_error(f"Safety wrapper creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(schema_path: str) -> bool:
    """Run all tests in sequence."""
    
    print(f"\n{Colors.BOLD}CS6 SCHEMA COMPATIBILITY TEST SUITE{Colors.END}")
    print(f"Schema: {schema_path}\n")
    
    results = {}
    
    # Test 1: Schema structure
    passed, schema = test_schema_structure(schema_path)
    results['structure'] = passed
    if not passed:
        print_error("\n⛔ Schema structure test FAILED - cannot continue")
        print("   Run: python convert_cs6_schema.py to fix")
        return False
    
    # Test 2: EV observations
    results['ev_obs'] = test_ev_observations(schema)
    
    # Test 3: Temperature observations
    results['temp_obs'] = test_temperature_observations(schema)
    
    # Test 4: CSV files
    results['csv_files'] = test_csv_files(schema, schema_path)
    
    # Test 5: Environment creation (critical)
    results['env_creation'] = test_environment_creation(schema_path)
    
    # Test 6: Safety wrapper (optional)
    results['safety_wrapper'] = test_safety_wrapper(schema_path)
    
    # Print summary
    print_header("TEST SUMMARY")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s} {status}")
    
    all_critical = results['structure'] and results['env_creation']
    
    if all_critical:
        print_success("\n🎉 All critical tests PASSED!")
        print("   Your schema is compatible with CityLearn.")
        
        if not results['ev_obs']:
            print_warning("   Note: Some EV observations missing/inactive")
        if not results['temp_obs']:
            print_warning("   Note: Temperature observations not fully configured")
        
        return True
    else:
        print_error("\n⛔ Critical tests FAILED")
        print("   See errors above for details.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test cs6 schema compatibility with CityLearn'
    )
    parser.add_argument(
        '--schema',
        required=True,
        help='Path to cs6 schema.json file'
    )
    
    args = parser.parse_args()
    
    if not Path(args.schema).exists():
        print_error(f"Schema file not found: {args.schema}")
        return 1
    
    success = run_all_tests(args.schema)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
