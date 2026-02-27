
"""
EV Deficit Tracker - CORRECTED LOGIC
Ground truth calculation with proper avoidable/unavoidable classification.
"""
import numpy as np

class EVDeficitTracker:
    """
    Tracks EV departures and calculates actual vs required SoC.
    FIXED: Proper logic for avoidable vs unavoidable deficits.
    """
    
    def __init__(self, env):
        self.env = env
        self.ev_buildings = self._find_ev_buildings()
        self.departure_log = []
        self.last_available = {}  # Track when each EV became available
        self.arrival_soc = {}      # Track SoC at arrival
        
        print(f"[EVTracker] Found {len(self.ev_buildings)} buildings with EVs")
        for b_idx in self.ev_buildings:
            building = env.buildings[b_idx]
            ev = building.electric_vehicle
            cap = getattr(ev, 'capacity_kwh', getattr(ev, 'capacity', 30.0))
            print(f"  Building {b_idx}: EV capacity = {cap:.1f} kWh")
            
            # Initialize tracking
            self.last_available[b_idx] = None
            self.arrival_soc[b_idx] = None
    
    def _find_ev_buildings(self):
        """Find which buildings have EVs."""
        ev_buildings = []
        for b_idx, building in enumerate(self.env.buildings):
            if hasattr(building, 'electric_vehicle') and building.electric_vehicle:
                ev_buildings.append(b_idx)
        return ev_buildings
    
    def calculate_step_deficit(self):
        """
        Calculate deficit at current timestep with CORRECTED logic.
        
        FIXED LOGIC:
        - Unavoidable: Not enough TIME to charge (arrival SoC + max possible charge < required)
        - Avoidable: Had enough time but agent didn't charge properly
        
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
            
            # Track availability changes
            self._update_availability_tracking(b_idx, ev, t_idx)
            
            # Detect departure event
            is_departure = self._is_departure_event(ev, t_idx)
            
            if not is_departure:
                continue
            
            # Get current SoC at departure
            current_soc = float(ev.soc[t_idx])
            
            # Get required SoC
            required_soc = self._get_required_soc(ev, t_idx)
            
            # Get capacity
            capacity = getattr(ev, 'capacity_kwh', getattr(ev, 'capacity', 30.0))
            
            # Calculate deficit
            if current_soc < required_soc:
                deficit_kwh = (required_soc - current_soc) * capacity
                total_deficit += deficit_kwh
                
                # FIXED: Determine if avoidable
                arrival_soc = self.arrival_soc.get(b_idx, 0.2)
                hours_available = self._get_hours_available(b_idx, t_idx)
                
                # Get max charging rate (kW)
                max_charge_rate = self._get_max_charge_rate(ev)
                
                # Calculate max possible charge (kWh)
                max_possible_charge = max_charge_rate * hours_available
                max_achievable_soc = arrival_soc + (max_possible_charge / capacity)
                
                # Unavoidable if physically impossible to reach required SoC
                if max_achievable_soc < required_soc:
                    unavoidable_deficit += deficit_kwh
                    avoidable = False
                else:
                    # Had enough time and energy, agent's fault
                    avoidable_deficit += deficit_kwh
                    avoidable = True
                
                # Log the departure
                self.departure_log.append({
                    'step': t_idx,
                    'building': b_idx,
                    'current_soc': current_soc,
                    'required_soc': required_soc,
                    'arrival_soc': arrival_soc,
                    'hours_available': hours_available,
                    'max_achievable_soc': max_achievable_soc,
                    'deficit_kwh': deficit_kwh,
                    'avoidable': avoidable
                })
                
                if deficit_kwh > 0.1:  # Only print significant deficits
                    print(f"[EV Deficit] Step {t_idx}, Building {b_idx}: "
                          f"{deficit_kwh:.2f} kWh deficit "
                          f"({'AVOIDABLE' if avoidable else 'UNAVOIDABLE'})")
                    print(f"  Arrival: {arrival_soc:.2%}, Current: {current_soc:.2%}, "
                          f"Required: {required_soc:.2%}")
                    print(f"  Time available: {hours_available:.1f}h, "
                          f"Max achievable: {max_achievable_soc:.2%}")
        
        return {
            'total': total_deficit,
            'avoidable': avoidable_deficit,
            'unavoidable': unavoidable_deficit
        }
    
    def _update_availability_tracking(self, b_idx, ev, t_idx):
        """Track when EV becomes available and its arrival SoC."""
        if not hasattr(ev, 'available') or len(ev.available) <= t_idx:
            return
        
        is_available = ev.available[t_idx] > 0.5
        
        # Detect arrival (transition from unavailable to available)
        if t_idx > 0 and len(ev.available) > t_idx - 1:
            was_unavailable = ev.available[t_idx - 1] < 0.5
            if was_unavailable and is_available:
                # EV just arrived
                self.last_available[b_idx] = t_idx
                if len(ev.soc) > t_idx:
                    self.arrival_soc[b_idx] = float(ev.soc[t_idx])
                    print(f"[EV Arrival] Step {t_idx}, Building {b_idx}: "
                          f"SoC = {self.arrival_soc[b_idx]:.2%}")
    
    def _get_hours_available(self, b_idx, departure_step):
        """Calculate hours between arrival and departure."""
        arrival_step = self.last_available.get(b_idx)
        if arrival_step is None:
            # Unknown arrival, assume 8 hours (typical overnight)
            return 8.0
        return float(departure_step - arrival_step)
    
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
    
    def _is_departure_event(self, ev, t_idx):
        """Detect if this timestep is a departure."""
        if t_idx == 0:
            return False
        
        # Method 1: Check availability transition
        if hasattr(ev, 'available') and len(ev.available) > t_idx:
            if len(ev.available) > t_idx - 1:
                was_available = ev.available[t_idx - 1] > 0.5
                now_unavailable = ev.available[t_idx] < 0.5
                if was_available and now_unavailable:
                    return True
        
        # Method 2: Check departure time array
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
        
        return {
            'total_departures': len(self.departure_log),
            'total_deficit': sum(d['deficit_kwh'] for d in self.departure_log),
            'avoidable_count': sum(1 for d in self.departure_log if d['avoidable']),
            'unavoidable_count': sum(1 for d in self.departure_log if not d['avoidable']),
            'avg_deficit': np.mean([d['deficit_kwh'] for d in self.departure_log])
        }
    
    def print_detailed_log(self, save_path=None):
        """Print or save detailed departure log."""
        import pandas as pd
        
        if not self.departure_log:
            print("[EVTracker] No departures recorded")
            return
        
        df = pd.DataFrame(self.departure_log)
        print("\n" + "="*80)
        print("DETAILED EV DEPARTURE LOG")
        print("="*80)
        print(df.to_string())
        print("="*80)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\n[EVTracker] Saved detailed log to {save_path}")