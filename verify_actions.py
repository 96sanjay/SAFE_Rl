import os
from citylearn.citylearn import CityLearnEnv

def get_schema_path():
    path = os.environ.get("CITYLEARN_SCHEMA")
    if not path or not os.path.exists(path):
        # Fallback to the path you were using in logs
        return "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
    return path

def main():
    print("="*80)
    print("                 CITYLEARN COMPONENT AUDIT")
    print("="*80)
    
    env = CityLearnEnv(schema=get_schema_path())
    
    total_actions_district = 0
    total_evs_district = 0
    total_batteries_district = 0
    total_temp_control = 0
    
    print(f"{'ID':<3} | {'Actions':<7} | {'Battery':<8} | {'EV Chargers':<12} | {'DHW/CoolStore':<15} | {'Temp Ctrl?'}")
    print("-" * 80)
    
    for i, b in enumerate(env.buildings):
        # 1. Count Action Dimensions
        # (This is the number of 'knobs' your RL agent turns for this building)
        n_actions = b.action_space.shape[0]
        total_actions_district += n_actions
        
        # 2. Check for Battery (Electrical Storage)
        has_battery = 0
        if b.electrical_storage is not None:
            has_battery = 1
            total_batteries_district += 1
            
        # 3. Check for EV Chargers (The 'charging_stations' list)
        num_evs = len(b.charging_stations)
        total_evs_district += num_evs
        
        # 4. Check for Thermal Storage (DHW / Cooling)
        # These are common in CityLearn and usually take up 1 action each if present
        thermal_count = 0
        if getattr(b, 'dhw_storage', None) is not None: thermal_count += 1
        if getattr(b, 'cooling_storage', None) is not None: thermal_count += 1
        
        # 5. Check for Temperature Control (Thermostat Actuation)
        # If this is False, the agent cannot change the setpoint.
        controls_temp = getattr(b, 'indoor_dry_bulb_temperature_set_point_actuation', False)
        if controls_temp:
            total_temp_control += 1
        
        # Format for table
        batt_str = "YES" if has_battery else "-"
        ev_str = f"YES ({num_evs})" if num_evs > 0 else "-"
        therm_str = f"YES ({thermal_count})" if thermal_count > 0 else "-"
        temp_flag = "YES" if controls_temp else "NO"
        
        print(f"{i:<3} | {n_actions:<7} | {batt_str:<8} | {ev_str:<12} | {therm_str:<15} | {temp_flag}")

    print("-" * 80)
    print(f"DISTRICT TOTALS:")
    print(f"  Total Buildings:      {len(env.buildings)}")
    print(f"  Total Action Dims:    {total_actions_district} (Should match your 28)")
    print(f"  Total Batteries:      {total_batteries_district}")
    print(f"  Total EV Chargers:    {total_evs_district}")
    print(f"  Total Temp Controls:  {total_temp_control}")
    print("="*80)

    # --- VERDICT LOGIC ---
    print("\n[VERDICT]")
    
    # 1. EV Check
    if total_evs_district > 0:
        print(f"✅ SUCCESS: Found {total_evs_district} EV Chargers. Your agent IS controlling EVs.")
    else:
        print("❌ FAILURE: No EVs detected.")

    # 2. Temperature Check
    if total_temp_control == 0:
        print("ℹ️  NOTE: Temperature Control is NO. The building manages its own thermostat.")
        print("          This explains why your 'Discomfort' plots are flat/zero.")
        print("          (The building buys grid power to maintain comfort automatically).")
    else:
        print("⚠️  NOTE: Temperature Control is YES. Your agent is responsible for comfort.")

if __name__ == "__main__":
    main()
