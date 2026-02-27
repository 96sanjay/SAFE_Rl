#!/usr/bin/env python3
"""
Convert cs6 grouped schema to CityLearn-compatible flat schema.

Usage:
    python convert_cs6_schema.py \
        --input /path/to/cs6/schema_grouped.json \
        --output /path/to/cs6/schema.json \
        --reference /path/to/citylearn_challenge_2022_phase_all_plus_evs/schema.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, path: str) -> None:
    """Save JSON file with pretty formatting."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved to: {path}")


def flatten_observations(grouped_obs: Dict) -> Dict:
    """
    Convert grouped observations to flat format.
    
    Input (grouped):
        {
            "buildings": {
                "indoor_dry_bulb_temperature": {"active": true},
                ...
            },
            "ev_chargers": {
                "required_soc_departure": {"active": true},
                ...
            }
        }
    
    Output (flat):
        {
            "indoor_dry_bulb_temperature": {"active": true},
            "required_soc_departure": {"active": true},
            ...
        }
    """
    flat_obs = {}
    
    # Extract building observations
    if "buildings" in grouped_obs:
        for key, val in grouped_obs["buildings"].items():
            flat_obs[key] = val
    
    # Extract EV charger observations
    if "ev_chargers" in grouped_obs:
        for key, val in grouped_obs["ev_chargers"].items():
            flat_obs[key] = val
    
    # If already flat, return as-is
    if "buildings" not in grouped_obs and "ev_chargers" not in grouped_obs:
        return grouped_obs
    
    return flat_obs


def flatten_actions(grouped_actions: Dict) -> Dict:
    """
    Convert grouped actions to flat format.
    
    Similar to observations but for actions.
    """
    flat_actions = {}
    
    # Extract building actions
    if "buildings" in grouped_actions:
        for key, val in grouped_actions["buildings"].items():
            flat_actions[key] = val
    
    # Extract EV charger actions
    if "ev_chargers" in grouped_actions:
        for key, val in grouped_actions["ev_chargers"].items():
            flat_actions[key] = val
    
    # If already flat, return as-is
    if "buildings" not in grouped_actions and "ev_chargers" not in grouped_actions:
        return grouped_actions
    
    return flat_actions


def validate_ev_observations(schema: Dict, reference: Dict = None) -> bool:
    """
    Validate that EV observations are correctly defined.
    
    Required EV observations:
    - required_soc_departure (or electric_vehicle_required_soc_departure)
    - estimated_soc_arrival
    - estimated_arrival_time (or estimated_departure_time)
    - ev_soc (or soc)
    """
    obs = schema.get("observations", {})
    
    required_ev_obs = [
        "required_soc_departure",
        "estimated_soc_arrival", 
        "estimated_arrival_time",
        "ev_soc"
    ]
    
    # Alternative names
    alternatives = {
        "required_soc_departure": ["electric_vehicle_required_soc_departure"],
        "ev_soc": ["soc"],
        "estimated_arrival_time": ["estimated_departure_time"]
    }
    
    missing = []
    for obs_name in required_ev_obs:
        found = obs_name in obs
        if not found and obs_name in alternatives:
            found = any(alt in obs for alt in alternatives[obs_name])
        if not found:
            missing.append(obs_name)
    
    if missing:
        print(f"⚠️  Missing EV observations: {missing}")
        if reference:
            ref_obs = reference.get("observations", {})
            print(f"📖 Reference schema has: {list(ref_obs.keys())[:10]}...")
        return False
    
    print("✅ All required EV observations present")
    return True


def convert_schema(input_path: str, output_path: str, reference_path: str = None) -> None:
    """
    Convert cs6 grouped schema to flat CityLearn-compatible schema.
    """
    print(f"Loading input schema: {input_path}")
    schema = load_json(input_path)
    
    reference = None
    if reference_path:
        print(f"Loading reference schema: {reference_path}")
        reference = load_json(reference_path)
    
    # Create converted schema
    converted = schema.copy()
    
    # Flatten observations
    if "observations" in schema:
        print("\n🔄 Flattening observations...")
        original_obs = schema["observations"]
        flattened_obs = flatten_observations(original_obs)
        converted["observations"] = flattened_obs
        print(f"   Before: {len(str(original_obs))} chars")
        print(f"   After:  {len(str(flattened_obs))} chars")
        print(f"   Keys:   {len(flattened_obs)} observations")
    
    # Flatten actions
    if "actions" in schema:
        print("\n�� Flattening actions...")
        original_actions = schema["actions"]
        flattened_actions = flatten_actions(original_actions)
        converted["actions"] = flattened_actions
        print(f"   Before: {len(str(original_actions))} chars")
        print(f"   After:  {len(str(flattened_actions))} chars")
        print(f"   Keys:   {len(flattened_actions)} actions")
    
    # Validate EV observations
    print("\n🔍 Validating EV observations...")
    validate_ev_observations(converted, reference)
    
    # Save converted schema
    print(f"\n💾 Saving converted schema...")
    save_json(converted, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Buildings:  {len(schema.get('buildings', {}))}")
    
    if "observations" in converted:
        obs_count = len(converted["observations"])
        print(f"Observations: {obs_count}")
        
        # Count EV-related observations
        ev_obs_count = sum(1 for k in converted["observations"] 
                          if any(x in k.lower() for x in ['ev', 'electric_vehicle', 'charger']))
        print(f"  - EV obs:   {ev_obs_count}")
        print(f"  - Other:    {obs_count - ev_obs_count}")
    
    if "actions" in converted:
        action_count = len(converted["actions"])
        print(f"Actions:      {action_count}")
        
        # Count EV-related actions
        ev_action_count = sum(1 for k in converted["actions"] 
                             if any(x in k.lower() for x in ['ev', 'electric_vehicle', 'charger']))
        print(f"  - EV act:   {ev_action_count}")
        print(f"  - Other:    {action_count - ev_action_count}")
    
    print("="*60)
    print("\n✅ CONVERSION COMPLETE")
    print("\nNext steps:")
    print(f"1. export CITYLEARN_SCHEMA='{output_path}'")
    print("2. Test environment creation:")
    print("   python -c 'from scripts.make_env import make_base_env; env = make_base_env(); print(env)'")


def main():
    parser = argparse.ArgumentParser(
        description='Convert cs6 grouped schema to CityLearn-compatible flat schema'
    )
    parser.add_argument(
        '--input', 
        required=True,
        help='Path to input cs6 schema (grouped format)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output schema (flat format)'
    )
    parser.add_argument(
        '--reference',
        help='Path to reference schema (e.g., working citylearn_challenge schema) for validation'
    )
    
    args = parser.parse_args()
    
    # Validate input exists
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Run conversion
    convert_schema(args.input, args.output, args.reference)
    
    return 0


if __name__ == "__main__":
    exit(main())
