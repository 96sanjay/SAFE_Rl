
"""
Diagnostic EV Scanner - Copy/Paste to CMD
Analyzes the first 500 steps to understand EV behavior.

Usage: python diagnostic_ev_scan.py
"""
import os
import sys
import numpy as np
from citylearn.citylearn import CityLearnEnv

def scan_ev_behavior():
    """Scan first 500 steps and log all EV events."""
    
    schema_path = os.environ.get("CITYLEARN_SCHEMA")
    if not schema_path:
        print("ERROR: CITYLEARN_SCHEMA not set")
        return
    
    print("=" * 80)
    print(" EV BEHAVIOR DIAGNOSTIC SCAN (First 500 Steps)")
    print("=" * 80)
    
    env = CityLearnEnv(schema=schema_path)
    result = env.reset()
    
    # Find EV buildings
    ev_buildings = []
    for b_idx, building in enumerate(env.buildings):
        if hasattr(building, 'electric_vehicle') and building.electric_vehicle:
            ev_buildings.append(b_idx)
    
    print(f"\nFound {len(ev_buildings)} buildings with EVs: {ev_buildings}")
    
    # Get EV properties
    for b_idx in ev_buildings:
        ev = env.buildings[b_idx].electric_vehicle
        print(f"\nBuilding {b_idx} EV properties:")
        print(f"  capacity: {getattr(ev, 'capacity_kwh', getattr(ev, 'capacity', 'N/A'))}")
        print(f"  max_charging_power: {getattr(ev, 'max_charging_power', 'N/A')}")
        print(f"  charging_power: {getattr(ev, 'charging_power', 'N/A')}")
        print(f"  nominal_power: {getattr(ev, 'nominal_power', 'N/A')}")
        
        # Check if arrays exist
        print(f"  Has soc array: {hasattr(ev, 'soc') and ev.soc is not None}")
        print(f"  Has available array: {hasattr(ev, 'available') and ev.available is not None}")
        print(f"  Has required_soc: {hasattr(ev, 'required_soc')}")
    
    print("\n" + "=" * 80)
    print("SCANNING 500 STEPS...")
    print("=" * 80)
    
    # Track state per building
    ev_state = {}
    for b_idx in ev_buildings:
        ev_state[b_idx] = {
            'last_available': False,
            'last_arrival_step': None,
            'last_arrival_soc': None,
            'arrivals': [],
            'departures': []
        }
    
    # Run 500 steps
    for step in range(500):
        # Zero actions
        actions = []
        for space in env.action_space:
            if hasattr(space, 'sample'):
                actions.append(np.zeros_like(space.sample()))
            else:
                actions.append(0.0)
        
        # Step
        step_result = env.step(actions)
        if len(step_result) == 5:
            obs, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            obs, _, done, _ = step_result
        
        if done:
            break
        
        # Check each EV
        for b_idx in ev_buildings:
            ev = env.buildings[b_idx].electric_vehicle
            state = ev_state[b_idx]
            
            if not hasattr(ev, 'available') or not ev.available:
                continue
            
            if len(ev.available) <= step or len(ev.soc) <= step:
                continue
            
            is_available = ev.available[step] > 0.5
            was_available = state['last_available']
            current_soc = float(ev.soc[step])
            
            # Detect arrival
            if not was_available and is_available:
                state['last_arrival_step'] = step
                state['last_arrival_soc'] = current_soc
                state['arrivals'].append({
                    'step': step,
                    'soc': current_soc
                })
                print(f"\n[ARRIVAL] Step {step:4d}, Building {b_idx}: EV arrived at {current_soc:.2%} SoC")
            
            # Detect departure
            if was_available and not is_available:
                arrival_step = state['last_arrival_step']
                arrival_soc = state['last_arrival_soc']
                
                hours_available = (step - arrival_step) if arrival_step is not None else 0
                
                # Get required SoC
                required_soc = 0.8  # Default
                if hasattr(ev, 'required_soc'):
                    if isinstance(ev.required_soc, (list, np.ndarray)):
                        if len(ev.required_soc) > step:
                            required_soc = float(ev.required_soc[step])
                    else:
                        required_soc = float(ev.required_soc)
                
                # Calculate max possible
                capacity = getattr(ev, 'capacity_kwh', getattr(ev, 'capacity', 30.0))
                max_charge_rate = 7.2  # Default
                if hasattr(ev, 'max_charging_power'):
                    max_charge_rate = float(ev.max_charging_power)
                elif hasattr(ev, 'charging_power'):
                    max_charge_rate = float(ev.charging_power)
                
                max_energy = max_charge_rate * hours_available
                max_achievable_soc = (arrival_soc if arrival_soc else 0.2) + (max_energy / capacity)
                max_achievable_soc = min(1.0, max_achievable_soc)
                
                deficit = max(0, required_soc - current_soc)
                
                state['departures'].append({
                    'step': step,
                    'arrival_step': arrival_step,
                    'arrival_soc': arrival_soc,
                    'departure_soc': current_soc,
                    'required_soc': required_soc,
                    'hours_available': hours_available,
                    'max_achievable_soc': max_achievable_soc,
                    'deficit': deficit
                })
                
                classification = "UNAVOIDABLE" if max_achievable_soc < required_soc else "AVOIDABLE"
                
                print(f"\n[DEPARTURE] Step {step:4d}, Building {b_idx}:")
                print(f"  Arrival: Step {arrival_step}, SoC {arrival_soc:.2%}")
                print(f"  Time available: {hours_available:.1f} hours")
                print(f"  Departure: SoC {current_soc:.2%} (required {required_soc:.2%})")
                print(f"  Max achievable: {max_achievable_soc:.2%}")
                print(f"  Deficit: {deficit * capacity:.2f} kWh")
                print(f"  Classification: {classification}")
            
            state['last_available'] = is_available
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for b_idx in ev_buildings:
        state = ev_state[b_idx]
        n_arrivals = len(state['arrivals'])
        n_departures = len(state['departures'])
        
        print(f"\nBuilding {b_idx}:")
        print(f"  Arrivals: {n_arrivals}")
        print(f"  Departures: {n_departures}")
        
        if state['departures']:
            avoidable = sum(1 for d in state['departures'] 
                          if d['max_achievable_soc'] >= d['required_soc'])
            unavoidable = n_departures - avoidable
            
            total_deficit = sum(d['deficit'] for d in state['departures'])
            
            print(f"  Departures with deficit: {sum(1 for d in state['departures'] if d['deficit'] > 0.001)}")
            print(f"  Total deficit: {total_deficit:.2f}")
            print(f"  Avoidable: {avoidable}")
            print(f"  Unavoidable: {unavoidable}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  - With ZERO actions (no charging), ALL deficits should be unavoidable")
    print("  - If seeing avoidable deficits, the logic is incorrect")
    print("  - Arrival SoC is the KEY - it should be LOW (~20-40%) for no-control")
    print("=" * 80)

if __name__ == "__main__":
    scan_ev_behavior()