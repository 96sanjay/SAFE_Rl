import os
from citylearn.citylearn import CityLearnEnv

def get_schema_path():
    path = os.environ.get("CITYLEARN_SCHEMA")
    if not path:
        return "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
    return path

def main():
    print("="*80)
    print("      ACTION SPACE MAPPING (THE TRUTH)")
    print("="*80)
    
    env = CityLearnEnv(schema=get_schema_path())
    
    total_actions = 0
    
    for i, b in enumerate(env.buildings):
        print(f"\n[Building {i}]")
        
        # 1. Get the official metadata
        # This dict tells us True/False for what is controllable
        meta = b.action_metadata
        
        # 2. Count EV Chargers manually
        # Note: In CityLearn 2022, 'electric_vehicle_chargers' might be a list of objects
        num_evs = 0
        if hasattr(b, 'electric_vehicle_chargers'):
             num_evs = len(b.electric_vehicle_chargers)
        
        # 3. Print the Mapping
        print(f"  Actions available: {b.action_space.shape[0]}")
        print(f"  EV Chargers found: {num_evs}")
        print(f"  Action Metadata:   {meta}")
        
        # 4. Interpret
        active_controls = []
        if meta.get('electrical_storage', False): active_controls.append("Battery")
        if meta.get('dhw_storage', False):        active_controls.append("DHW Tank")
        if meta.get('cooling_storage', False):    active_controls.append("Cold Water Tank")
        if meta.get('heating_storage', False):    active_controls.append("Hot Water Tank")
        if meta.get('cooling_device', False):     active_controls.append("AC (Temp Control)")
        if meta.get('heating_device', False):     active_controls.append("Heater (Temp Control)")
        
        # EV logic for 2022 Challenge
        # Usually EVs don't show up in action_metadata dictionary keys in this version,
        # they just add to the action dimension count.
        
        print(f"  -> CONTROLLABLE: {', '.join(active_controls)}")
        
        # Temperature Check
        if meta.get('cooling_device', False) or meta.get('heating_device', False):
            print("  -> STATUS: AGENT CONTROLS TEMPERATURE ✅")
        else:
            print("  -> STATUS: BUILDING CONTROLS TEMPERATURE (PASSIVE) ❌")

if __name__ == "__main__":
    main()
