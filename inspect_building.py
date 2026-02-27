import os
import pprint
from citylearn.citylearn import CityLearnEnv

def get_schema_path():
    path = os.environ.get("CITYLEARN_SCHEMA")
    if not path:
        return "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
    return path

def main():
    print("="*60)
    print("      BUILDING OBJECT INSPECTION")
    print("="*60)
    
    try:
        env = CityLearnEnv(schema=get_schema_path())
        b = env.buildings[0] # Look at the first building
        
        print(f"[INFO] Inspecting Building 0 (Class: {type(b).__name__})")
        
        # 1. Filter relevant attributes (Energy, Storage, EV)
        keywords = ['charg', 'ev', 'vehicle', 'stor', 'batt', 'device', 'action']
        
        found_attrs = {}
        for attr in dir(b):
            # Skip python internals
            if attr.startswith("__"): continue
            
            # Check if value is relevant
            val = getattr(b, attr)
            
            # Store if it matches keyword
            if any(k in attr.lower() for k in keywords):
                found_attrs[attr] = str(type(val))
                
        # 2. Print Findings
        print(f"\n[FOUND CANDIDATES]")
        for k, v in found_attrs.items():
            print(f"  .{k:<35} -> {v}")

        # 3. Check for the 'electric_vehicle_chargers' candidate explicitly
        print("\n[SPECIFIC CHECKS]")
        if hasattr(b, 'electric_vehicle_chargers'):
            print(f"  ✅ b.electric_vehicle_chargers FOUND! Length: {len(b.electric_vehicle_chargers)}")
        else:
            print(f"  ❌ b.electric_vehicle_chargers NOT found.")
            
        if hasattr(b, 'ev_chargers'):
            print(f"  ✅ b.ev_chargers FOUND! Length: {len(b.ev_chargers)}")
            
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")

if __name__ == "__main__":
    main()
