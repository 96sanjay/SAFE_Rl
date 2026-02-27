#!/usr/bin/env python3
"""
Compare two CityLearn schemas to identify differences.

Usage:
    python compare_schemas.py \
        --working /path/to/working/schema.json \
        --target /path/to/cs6/schema.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Set


def load_schema(path: str) -> Dict:
    """Load schema JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_observation_keys(schema: Dict) -> Set[str]:
    """Extract all observation keys (handling both flat and grouped)."""
    obs = schema.get('observations', {})
    
    # If grouped, flatten
    if 'buildings' in obs or 'ev_chargers' in obs:
        keys = set()
        if 'buildings' in obs:
            keys.update(obs['buildings'].keys())
        if 'ev_chargers' in obs:
            keys.update(obs['ev_chargers'].keys())
        return keys
    
    # Already flat
    return set(obs.keys())


def extract_action_keys(schema: Dict) -> Set[str]:
    """Extract all action keys (handling both flat and grouped)."""
    actions = schema.get('actions', {})
    
    # If grouped, flatten
    if 'buildings' in actions or 'ev_chargers' in actions:
        keys = set()
        if 'buildings' in actions:
            keys.update(actions['buildings'].keys())
        if 'ev_chargers' in actions:
            keys.update(actions['ev_chargers'].keys())
        return keys
    
    # Already flat
    return set(actions.keys())


def compare_observations(working: Dict, target: Dict) -> None:
    """Compare observations between schemas."""
    print("\n" + "="*70)
    print("OBSERVATION COMPARISON")
    print("="*70)
    
    working_obs = extract_observation_keys(working)
    target_obs = extract_observation_keys(target)
    
    print(f"\nWorking schema: {len(working_obs)} observations")
    print(f"Target schema:  {len(target_obs)} observations")
    
    # Observations in working but not target
    missing_in_target = working_obs - target_obs
    if missing_in_target:
        print(f"\n⚠️  Missing in target ({len(missing_in_target)}):")
        for obs in sorted(missing_in_target):
            print(f"   - {obs}")
    else:
        print("\n✅ All working observations present in target")
    
    # Observations in target but not working (new additions)
    new_in_target = target_obs - working_obs
    if new_in_target:
        print(f"\n➕ New in target ({len(new_in_target)}):")
        for obs in sorted(new_in_target):
            print(f"   + {obs}")
            
        # Categorize new observations
        temp_obs = [o for o in new_in_target if 'temperature' in o.lower() or 'hvac' in o.lower() or 'cooling' in o.lower() or 'heating' in o.lower()]
        if temp_obs:
            print(f"\n   Temperature-related ({len(temp_obs)}):")
            for obs in sorted(temp_obs):
                print(f"      + {obs}")
    else:
        print("\n→ No new observations in target")
    
    # Common observations
    common = working_obs & target_obs
    print(f"\n✅ Common observations: {len(common)}")
    
    # Check for EV observations
    ev_obs = [o for o in common if any(x in o.lower() for x in ['ev', 'electric_vehicle', 'charger', 'soc', 'departure', 'arrival'])]
    if ev_obs:
        print(f"   EV observations ({len(ev_obs)}):")
        for obs in sorted(ev_obs)[:5]:
            print(f"      ✅ {obs}")
        if len(ev_obs) > 5:
            print(f"      ... and {len(ev_obs) - 5} more")


def compare_actions(working: Dict, target: Dict) -> None:
    """Compare actions between schemas."""
    print("\n" + "="*70)
    print("ACTION COMPARISON")
    print("="*70)
    
    working_actions = extract_action_keys(working)
    target_actions = extract_action_keys(target)
    
    print(f"\nWorking schema: {len(working_actions)} actions")
    print(f"Target schema:  {len(target_actions)} actions")
    
    # Actions in working but not target
    missing_in_target = working_actions - target_actions
    if missing_in_target:
        print(f"\n⚠️  Missing in target ({len(missing_in_target)}):")
        for action in sorted(missing_in_target):
            print(f"   - {action}")
    else:
        print("\n✅ All working actions present in target")
    
    # Actions in target but not working (new additions)
    new_in_target = target_actions - working_actions
    if new_in_target:
        print(f"\n➕ New in target ({len(new_in_target)}):")
        for action in sorted(new_in_target):
            print(f"   + {action}")
            
        # Categorize new actions
        hvac_actions = [a for a in new_in_target if 'cooling' in a.lower() or 'heating' in a.lower() or 'hvac' in a.lower()]
        if hvac_actions:
            print(f"\n   HVAC-related ({len(hvac_actions)}):")
            for action in sorted(hvac_actions):
                print(f"      + {action}")
    else:
        print("\n→ No new actions in target")
    
    # Common actions
    common = working_actions & target_actions
    print(f"\n✅ Common actions: {len(common)}")


def compare_structure(working: Dict, target: Dict) -> None:
    """Compare schema structure."""
    print("\n" + "="*70)
    print("SCHEMA STRUCTURE")
    print("="*70)
    
    # Check observations structure
    working_obs = working.get('observations', {})
    target_obs = target.get('observations', {})
    
    working_grouped = 'buildings' in working_obs or 'ev_chargers' in working_obs
    target_grouped = 'buildings' in target_obs or 'ev_chargers' in target_obs
    
    print("\nObservations structure:")
    print(f"   Working: {'GROUPED (buildings/ev_chargers)' if working_grouped else 'FLAT ✅'}")
    print(f"   Target:  {'GROUPED (buildings/ev_chargers) ⚠️' if target_grouped else 'FLAT ✅'}")
    
    if target_grouped and not working_grouped:
        print(f"\n   ⚠️  TARGET HAS GROUPED STRUCTURE - NEEDS CONVERSION")
        print(f"   Run: python convert_cs6_schema.py --input <target> --output <output>")
    
    # Check actions structure
    working_actions = working.get('actions', {})
    target_actions = target.get('actions', {})
    
    working_grouped = 'buildings' in working_actions or 'ev_chargers' in working_actions
    target_grouped = 'buildings' in target_actions or 'ev_chargers' in target_actions
    
    print("\nActions structure:")
    print(f"   Working: {'GROUPED (buildings/ev_chargers)' if working_grouped else 'FLAT ✅'}")
    print(f"   Target:  {'GROUPED (buildings/ev_chargers) ⚠️' if target_grouped else 'FLAT ✅'}")


def compare_buildings(working: Dict, target: Dict) -> None:
    """Compare building configurations."""
    print("\n" + "="*70)
    print("BUILDING CONFIGURATION")
    print("="*70)
    
    working_buildings = working.get('buildings', {})
    target_buildings = target.get('buildings', {})
    
    print(f"\nWorking schema: {len(working_buildings)} buildings")
    print(f"Target schema:  {len(target_buildings)} buildings")
    
    # Count EV chargers
    working_chargers = sum(
        len(b.get('electric_vehicle_chargers', {}))
        for b in working_buildings.values()
    )
    target_chargers = sum(
        len(b.get('electric_vehicle_chargers', {}))
        for b in target_buildings.values()
    )
    
    print(f"\nEV chargers:")
    print(f"   Working: {working_chargers} chargers")
    print(f"   Target:  {target_chargers} chargers")
    
    if target_chargers == 0:
        print(f"   ⚠️  WARNING: No EV chargers found in target!")


def main():
    parser = argparse.ArgumentParser(
        description='Compare two CityLearn schemas'
    )
    parser.add_argument(
        '--working',
        required=True,
        help='Path to working schema (reference)'
    )
    parser.add_argument(
        '--target',
        required=True,
        help='Path to target schema (cs6)'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.working).exists():
        print(f"❌ Working schema not found: {args.working}")
        return 1
    
    if not Path(args.target).exists():
        print(f"❌ Target schema not found: {args.target}")
        return 1
    
    # Load schemas
    print("Loading schemas...")
    working = load_schema(args.working)
    target = load_schema(args.target)
    
    print(f"✅ Loaded working: {args.working}")
    print(f"✅ Loaded target:  {args.target}")
    
    # Run comparisons
    compare_structure(working, target)
    compare_buildings(working, target)
    compare_observations(working, target)
    compare_actions(working, target)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    target_obs = target.get('observations', {})
    target_grouped = 'buildings' in target_obs or 'ev_chargers' in target_obs
    
    if target_grouped:
        print("\n🔧 ACTION REQUIRED:")
        print("   Target schema has grouped structure and needs conversion.")
        print("\n   Run this command:")
        print(f"   python convert_cs6_schema.py \\")
        print(f"       --input {args.target} \\")
        print(f"       --output {Path(args.target).parent / 'schema_converted.json'} \\")
        print(f"       --reference {args.working}")
    else:
        print("\n✅ Target schema structure is compatible (flat)")
        print("   You can proceed with testing:")
        print(f"   python test_cs6_schema.py --schema {args.target}")
    
    return 0


if __name__ == "__main__":
    exit(main())
