"""
EV Deficit Tracker - FINAL DEBUGGED VERSION
With extensive logging to identify exactly what's happening.
"""
import numpy as np
import pandas as pd

class EVDeficitTracker:
    """
    Tracks EV departures and calculates actual vs required SoC.
    FINAL VERSION with corrected avoidable/unavoidable logic and debug logging.
    """
    
    def __init__(self, env, debug=True):
        self.env = env
        self.debug = debug
        self.ev_buildings = self._find_ev_buildings()
        self.departure_log = []
        
        # CRITICAL: Track state per EV
        self.ev_state = {}  # {building_idx: {'last_arrival_step': int, 'arrival_soc': float}}
        
        if self.debug:
            print(f"\n[EVTracker] Initialized with {len(self.ev_buildings)} EV buildings")
            for b_idx in self.ev_buildings:
                building = env.buildings[b_idx]
                ev = building.electric_vehicle
                cap = getattr(ev, 'capacity_kwh', getattr(ev, 'capacity', 30.0))
                print(f"  Building {b_idx}: Capacity={cap:.1f} kWh")
                
                # Initialize state tracking
                self.ev_state[b_idx] = {
                    'last_arrival_step': None,
                    'arrival_soc': None,
                    'last_was_available': False
                }
    
    def _find_ev_buildings(self):
        """Find which buildings have EVs."""
        ev_buildings = []
        for b_idx, building in enumerate(self.env.buildings):
            if hasattr(building, 'electric_vehicle') and building.electric_vehicle:
                ev_buildings.append(b_idx)
        return ev_buildings
    
    def calculate_step_deficit(self):
        """
        Calculate deficit at current timestep.
        Returns: {'total': float, 'avoidable': float, 'unavoidable': float}
        """
        t_idx = max(0, getattr(self.env, "time_step", 1) - 1)
        
        total_deficit = 0.0
        avoidable_deficit = 0.0
        unavoidable_deficit = 0.0
        
        for b_idx in self.ev_buildings:
            building = self.env.buildings[b_idx]
            ev = building.electric_vehicle
            
            if not hasattr(ev, 'soc') or not ev.soc or len(ev.soc) <= t_idx:
                continue
            
            # Update arrival tracking BEFORE checking departure
            self._track_arrivals(b_idx, ev, t_idx)
            
            # Check for departure
            is_departure = self._is_departure_event(ev, t_idx)
            
            if not is_departure:
                continue
            
            # Get departure details
            current_soc = float(ev.soc[t_idx])
            required_soc = self._get_required_soc(ev, t_idx)
            capacity = getattr(ev, 'capacity_kwh', getattr(ev, 'capacity', 30.0))
            
            # Only count if there's a deficit
            if current_soc >= required_soc:
                if self.debug and t_idx < 200:
                    print(f"  [Step {t_idx}] Building {b_idx}: No deficit (SoC {current_soc:.2%} >= {required_soc:.2%})")
                continue
            
            deficit_kwh = (required_soc - current_soc) * capacity
            total_deficit += deficit_kwh
            
            # CRITICAL: Determine avoidable vs unavoidable
            state = self.ev_state[b_idx]
            arrival_step = state['last_arrival_step']
            arrival_soc = state['arrival_soc']
            
            if arrival_step is None or arrival_soc is None:
                # Unknown arrival - assume it just arrived at 20%
                if self.debug:
                    print(f"  [Step {t_idx}] Building {b_idx}: Unknown arrival, assuming 20% SoC")
                arrival_soc = 0.2
                arrival_step = t_idx - 8  # Assume 8 hours available
            
            # Calculate physics constraints
            hours_available = float(t_idx - arrival_step)
            max_charge_rate_kw = self._get_max_charge_rate(ev)
            
            # Maximum energy that COULD have been added (kWh)
            max_energy_addable = max_charge_rate_kw * hours_available
            
            # Maximum SoC achievable
            max_achievable_soc = arrival_soc + (max_energy_addable / capacity)
            max_achievable_soc = min(1.0, max_achievable_soc)  # Cap at 100%
            
            # Energy needed from arrival to reach requirement
            energy_needed = (required_soc - arrival_soc) * capacity
            
            # CORRECTED LOGIC:
            # Unavoidable = physically impossible to reach required_soc
            # Avoidable = had time/power but agent failed to charge properly
            
            if max_achievable_soc < required_soc:
                # Physics made it impossible
                # Deficit is ONLY the gap between max_achievable and required
                physics_limited_deficit = (required_soc - max_achievable_soc) * capacity
                unavoidable_deficit += physics_limited_deficit
                
                # The rest (if any) is avoidable
                agent_failure_deficit = deficit_kwh - physics_limited_deficit
                if agent_failure_deficit > 0.001:  # Numerical tolerance
                    avoidable_deficit += agent_failure_deficit
                
                classification = "UNAVOIDABLE (physics)"
                
            else:
                # Agent could have reached required_soc but didn't
                avoidable_deficit += deficit_kwh
                classification = "AVOIDABLE (agent fault)"
            
            # Log this departure
            log_entry = {
                'step': t_idx,
                'building': b_idx,
                'current_soc': current_soc,
                'required_soc': required_soc,
                'arrival_soc': arrival_soc,
                'arrival_step': arrival_step,
                'hours_available': hours_available,
                'max_charge_rate_kw': max_charge_rate_kw,
                'energy_needed_kwh': energy_needed,
                'max_energy_addable_kwh': max_energy_addable,
                'max_achievable_soc': max_achievable_soc,
                'deficit_kwh': deficit_kwh,
                'classification': classification
            }
            self.departure_log.append(log_entry)
            
            # Debug output for first 20 departures
            if self.debug and len(self.departure_log) <= 20:
                print(f"\n[DEPARTURE #{len(self.departure_log)}] Step {t_idx}, Building {b_idx}")
                print(f"  Arrival: SoC={arrival_soc:.2%} at step {arrival_step}")
                print(f"  Departure: SoC={current_soc:.2%} (required {required_soc:.2%})")
                print(f"  Time available: {hours_available:.1f} hours")
                print(f"  Max charge rate: {max_charge_rate_kw:.2f} kW")
                print(f"  Energy needed: {energy_needed:.2f} kWh")
                print(f"  Max energy addable: {max_energy_addable:.2f} kWh")
                print(f"  Max achievable SoC: {max_achievable_soc:.2%}")
                print(f"  Actual deficit: {deficit_kwh:.2f} kWh")
                print(f"  Classification: {classification}")
        
        return {
            'total': total_deficit,
            'avoidable': avoidable_deficit,
            'unavoidable': unavoidable_deficit
        }
    
    def _track_arrivals(self, b_idx, ev, t_idx):
        """
        Track when EV arrives (becomes available) and its arrival SoC.
        This is CRITICAL for determining avoidable vs unavoidable.
        """
        if not hasattr(ev, 'available') or len(ev.available) <= t_idx:
            return
        
        state = self.ev_state[b_idx]
        is_available_now = ev.available[t_idx] > 0.5
        was_available = state['last_was_available']
        
        # Detect arrival: transition from unavailable -> available
        if not was_available and is_available_now:
            # EV just arrived!
            state['last_arrival_step'] = t_idx
            
            if len(ev.soc) > t_idx:
                state['arrival_soc'] = float(ev.soc[t_idx])
            else:
                state['arrival_soc'] = 0.2  # Default assumption
            
            if self.debug and t_idx < 200:
                print(f"  [Step {t_idx}] Building {b_idx} ARRIVAL: SoC={state['arrival_soc']:.2%}")
        
        # Update state
        state['last_was_available'] = is_available_now
    
    def _is_departure_event(self, ev, t_idx):
        """Detect if this timestep is a departure."""
        if t_idx == 0:
            return False
        
        # Method 1: Check availability transition (available -> unavailable)
        if hasattr(ev, 'available') and len(ev.available) > t_idx:
            if len(ev.available) > t_idx - 1:
                was_available = ev.available[t_idx - 1] > 0.5
                now_unavailable = ev.available[t_idx] < 0.5
                if was_available and now_unavailable:
                    return True
        
        # Method 2: Check departure time flag
        if hasattr(ev, 'departure_time') and len(ev.departure_time) > t_idx:
            return ev.departure_time[t_idx] > 0
        
        return False
    
    def _get_required_soc(self, ev, t_idx):
        """Get required SoC for this departure."""
        # Try various attribute names
        if hasattr(ev, 'required_soc'):
            if isinstance(ev.required_soc, (list, np.ndarray)):
                if len(ev.required_soc) > t_idx:
                    return float(ev.required_soc[t_idx])
            else:
                return float(ev.required_soc)
        
        if hasattr(ev, 'target_soc'):
            if isinstance(ev.target_soc, (list, np.ndarray)):
                if len(ev.target_soc) > t_idx:
                    return float(ev.target_soc[t_idx])
            else:
                return float(ev.target_soc)
        
        # Default: 80% for commute
        return 0.8
    
    def _get_max_charge_rate(self, ev):
        """Get maximum charging rate in kW."""
        # Try various attribute names
        if hasattr(ev, 'max_charging_power'):
            return float(ev.max_charging_power)
        if hasattr(ev, 'charging_power'):
            return float(ev.charging_power)
        if hasattr(ev, 'nominal_power'):
            return float(ev.nominal_power)
        
        # Default: Level 2 charger (7.2 kW)
        return 7.2
    
    def get_summary(self):
        """Get summary of all departures."""
        if not self.departure_log:
            return {
                'total_departures': 0,
                'total_deficit': 0.0,
                'avoidable_count': 0,
                'unavoidable_count': 0,
                'avg_deficit': 0.0
            }
        
        avoidable_count = sum(1 for d in self.departure_log 
                             if d['classification'] == "AVOIDABLE (agent fault)")
        unavoidable_count = len(self.departure_log) - avoidable_count
        
        return {
            'total_departures': len(self.departure_log),
            'total_deficit': sum(d['deficit_kwh'] for d in self.departure_log),
            'avoidable_count': avoidable_count,
            'unavoidable_count': unavoidable_count,
            'avg_deficit': np.mean([d['deficit_kwh'] for d in self.departure_log])
        }
    
    def print_detailed_analysis(self):
        """Print detailed analysis of departures."""
        if not self.departure_log:
            print("\n[EVTracker] No departures recorded")
            return
        
        print("\n" + "=" * 80)
        print("DETAILED EV DEPARTURE ANALYSIS")
        print("=" * 80)
        
        # Overall stats
        avoidable_entries = [d for d in self.departure_log 
                            if d['classification'] == "AVOIDABLE (agent fault)"]
        unavoidable_entries = [d for d in self.departure_log 
                              if d['classification'] == "UNAVOIDABLE (physics)"]
        
        print(f"\nTotal departures: {len(self.departure_log)}")
        print(f"  Avoidable: {len(avoidable_entries)} ({100*len(avoidable_entries)/len(self.departure_log):.1f}%)")
        print(f"  Unavoidable: {len(unavoidable_entries)} ({100*len(unavoidable_entries)/len(self.departure_log):.1f}%)")
        
        if avoidable_entries:
            print(f"\nAvoidable deficits:")
            print(f"  Total energy: {sum(d['deficit_kwh'] for d in avoidable_entries):.2f} kWh")
            print(f"  Avg per event: {np.mean([d['deficit_kwh'] for d in avoidable_entries]):.2f} kWh")
            
        if unavoidable_entries:
            print(f"\nUnavoidable deficits:")
            print(f"  Total energy: {sum(d['deficit_kwh'] for d in unavoidable_entries):.2f} kWh")
            print(f"  Avg per event: {np.mean([d['deficit_kwh'] for d in unavoidable_entries]):.2f} kWh")
        
        # Show first 10 avoidable
        if avoidable_entries:
            print(f"\nFirst 10 AVOIDABLE departures:")
            for i, d in enumerate(avoidable_entries[:10]):
                print(f"  {i+1}. Step {d['step']}: {d['deficit_kwh']:.2f} kWh deficit")
                print(f"     Arrival: {d['arrival_soc']:.2%}, Required: {d['required_soc']:.2%}, Achieved: {d['current_soc']:.2%}")
                print(f"     Time: {d['hours_available']:.1f}h, Could have reached: {d['max_achievable_soc']:.2%}")
        
        # Show first 10 unavoidable
        if unavoidable_entries:
            print(f"\nFirst 10 UNAVOIDABLE departures:")
            for i, d in enumerate(unavoidable_entries[:10]):
                print(f"  {i+1}. Step {d['step']}: {d['deficit_kwh']:.2f} kWh deficit")
                print(f"     Arrival: {d['arrival_soc']:.2%}, Required: {d['required_soc']:.2%}, Achieved: {d['current_soc']:.2%}")
                print(f"     Time: {d['hours_available']:.1f}h, Max possible: {d['max_achievable_soc']:.2%}")
                print(f"     Physics-limited by {(d['required_soc'] - d['max_achievable_soc'])*100:.1f}% SoC")
        
        print("=" * 80)
    
    def print_detailed_log(self, save_path=None):
        """Print or save detailed departure log."""
        if not self.departure_log:
            print("[EVTracker] No departures recorded")
            return
        
        df = pd.DataFrame(self.departure_log)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\n[EVTracker] Saved {len(df)} departures to {save_path}")
        else:
            print("\n" + "="*100)
            print("DETAILED EV DEPARTURE LOG")
            print("="*100)
            print(df.to_string(max_rows=50))
            print("="*100)
