"""
FINAL CORRECTED VERSION: Count EV violations at DEPARTURE level

INTERPRETATION:
- Building SOC violations: TIMESTEPS (e.g., "2190 timesteps had battery SOC > 0.95")
- EV violations: DEPARTURES (e.g., "200 EV departures had deficits")
- EV unavoidable: DEPARTURES (e.g., "50 departures were unavoidable due to physics")

This makes sense because:
1. Building SOC is checked EVERY timestep
2. EV deficit is only relevant AT DEPARTURE  
3. Unavoidable should be a subset of total EV departures

"""

# Replace the PerBuildingMetrics class with this version:

class PerBuildingMetrics:
    def __init__(self, citylearn_env):
        self.env = unwrap_to_raw_citylearn_env(citylearn_env)
        self.n_buildings = len(self.env.buildings)
        self.charger_to_building = self._build_charger_mapping()
        self._seen_departures: Set[Tuple[Any, ...]] = set()
        
        # CORRECTED: Separate timestep vs departure counters
        self.violation_counters = {
            "total_violations_timesteps": 0,       # Timesteps with ANY violation
            "building_soc_violations_timesteps": 0, # Timesteps with building SOC > 0.95
            "ev_departures_with_deficit": 0,        # Number of departures with deficit
            "ev_departures_unavoidable": 0,         # Number of departures that were unavoidable
        }

    def reset_episode(self):
        self._seen_departures.clear()
        self.violation_counters = {
            "total_violations_timesteps": 0,
            "building_soc_violations_timesteps": 0,
            "ev_departures_with_deficit": 0,
            "ev_departures_unavoidable": 0,
        }

    # ... (keep all other methods the same) ...

    def extract(self) -> Dict[int, Dict[str, float]]:
        # ... (existing code until EV departure processing) ...
        
        # Process EV departures (THIS PART CHANGES)
        departures_this_step = 0
        unavoidable_this_step = 0
        
        all_records = _ev_departure_records(self.env) or []
        for record in all_records:
            charger_id = _norm_id(getattr(record, "charger_id", "UNKNOWN"))
            bi = self.charger_to_building.get(charger_id)
            if bi is None:
                continue

            key = self._record_key(record)
            if key in self._seen_departures:
                continue
            self._seen_departures.add(key)

            # ... (existing SOC/kWh calculation) ...

            # COUNT DEPARTURES (NEW)
            if da_soc > 0:  # Departure has deficit
                departures_this_step += 1
                
                if du_soc > 0:  # Deficit was unavoidable
                    unavoidable_this_step += 1

        # Update DEPARTURE counters
        self.violation_counters["ev_departures_with_deficit"] += departures_this_step
        self.violation_counters["ev_departures_unavoidable"] += unavoidable_this_step
        
        # Count TIMESTEP violations for building SOC
        step_has_building_violation = False
        step_has_ev_violation = (departures_this_step > 0)
        
        for bi in range(self.n_buildings):
            bm = metrics.get(bi, {})
            if bm.get("soc_violation_flag", 0.0) > 0:
                step_has_building_violation = True
                break
        
        # Update TIMESTEP counters
        if step_has_building_violation or step_has_ev_violation:
            self.violation_counters["total_violations_timesteps"] += 1
        
        if step_has_building_violation:
            self.violation_counters["building_soc_violations_timesteps"] += 1

        return metrics
