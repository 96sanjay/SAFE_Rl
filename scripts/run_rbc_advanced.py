
# scripts/run_rbc_advanced.py
"""
Advanced RBC baseline (conservative, inline):

- Wraps CityLearn's default RBC with very light safety heuristics:
  * Optionally block battery charging when SoC > 0.95 (to reduce battery_abuse_kwh),
  * Optionally nudge EV charging when departure is soon & SoC is low (if signals exist).

- If the action structure is not a flat numeric list, the wrapper leaves actions unchanged,
  so we NEVER break the shape expected by the environment.

- Logs the SAME KPIs as the other baselines, including:
  ev_departure_deficit_kwh, ev_impossible_request_count,
  battery_abuse_kwh, solar_waste_kwh.

Output: runs/kpi_rbc_advanced_kpis.csv
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np

from citylearn.citylearn import CityLearnEnv
try:
    from citylearn.agents import RBC as CityLearnRBC
except ImportError:
    from citylearn.agents.rbc import RBC as CityLearnRBC


# ---------------------------------------------------------------------
# Optional EV extractor: use your repo's version if available
# ---------------------------------------------------------------------
def _safe_import_ev_extractor():
    try:
        from citylearn_safe.extractors import ev_departure_cost_components
        return ev_departure_cost_components
    except Exception:
        # Fallback: return zeros but keep columns consistent
        def _fallback_ev_departure_cost_components(env):
            return {
                "total": 0.0,
                "avoidable": 0.0,
                "unavoidable": 0.0,
                "departures": 0,
            }
        return _fallback_ev_departure_cost_components


ev_departure_cost_components = _safe_import_ev_extractor()


# ---------------------------------------------------------------------
# Inline Advanced RBC (no numpy ramp limiting, shape-safe)
# ---------------------------------------------------------------------
class AdvancedRBC(CityLearnRBC):
    """
    Conservative 'safe' RBC wrapper.

    - Calls default CityLearn RBC to get actions.
    - If actions are a flat numeric list:
        * Optionally block battery charging when SoC > 0.95
        * Optionally nudge EV charging when departure is soon & SoC low
    - If actions are nested/complex, returns them unchanged
      (so we NEVER crash due to action shape).
    """

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.env = env
        self._battery_charge_idx = []
        self._ev_charge_idx = []

        # Discover indices if action_names are available
        names = getattr(env, "action_names", None) or getattr(
            getattr(env, "unwrapped", env), "action_names", None
        )

        if isinstance(names, list):
            for i, n in enumerate(names):
                low = str(n).lower()
                if (
                    "electrical" in low
                    and "storage" in low
                    and "charge" in low
                    and "discharge" not in low
                ):
                    self._battery_charge_idx.append(i)
                if "electric_vehicle" in low and (
                    "charge" in low or "grid_charging" in low or "ev" in low
                ):
                    self._ev_charge_idx.append(i)

    @staticmethod
    def _is_flat_numeric_list(x):
        """Check if x is a simple 1-D numeric list/tuple."""
        if not isinstance(x, (list, tuple)):
            return False
        for v in x:
            if isinstance(v, (list, tuple, np.ndarray, dict)):
                return False
            if not isinstance(v, (int, float, np.integer, np.floating)):
                return False
        return True

    def predict(self, obs):
        actions = super().predict(obs)

        # If not a flat numeric list, leave as-is (avoid shape issues).
        if not self._is_flat_numeric_list(actions):
            return actions

        t_idx = getattr(self.env, "time_step", 0) - 1
        if t_idx < 0:
            return actions

        buildings = getattr(self.env, "buildings", [])

        # --- Battery abuse guard: block charge if any SoC > 0.95 ---
        any_batt_high = False
        for b in buildings:
            es = getattr(b, "electrical_storage", None)
            if es is None:
                continue
            soc_data = es.soc
            soc = (
                soc_data[t_idx]
                if hasattr(soc_data, "__getitem__") and len(soc_data) > t_idx
                else soc_data
            )
            if soc is not None and soc > 0.95:
                any_batt_high = True
                break

        # --- EV gentle nudge (if signals exist) ---
        ev_departing_soon = False
        for b in buildings:
            evs = getattr(b, "electric_vehicles", None)
            if evs is None:
                continue
            try:
                dep = getattr(evs, "departure_time_step", None)
                socs = getattr(evs, "soc", None)
                dep_t = (
                    dep[t_idx]
                    if dep is not None
                    and hasattr(dep, "__getitem__")
                    and len(dep) > t_idx
                    else None
                )
                soc_t = (
                    socs[t_idx]
                    if socs is not None
                    and hasattr(socs, "__getitem__")
                    and len(socs) > t_idx
                    else None
                )
                if dep_t is not None and soc_t is not None:
                    # If departure in next 2 steps & SoC < 0.8 -> nudge EV charging
                    if 0 <= (dep_t - getattr(self.env, "time_step", 0)) <= 2 and soc_t < 0.8:
                        ev_departing_soon = True
                        break
            except Exception:
                pass

        actions = list(actions)

        # Apply conservative rules only on flat list
        if any_batt_high:
            for i in self._battery_charge_idx:
                if 0 <= i < len(actions):
                    actions[i] = 0.0  # block charging above 95% SoC

        if ev_departing_soon:
            for i in self._ev_charge_idx:
                if 0 <= i < len(actions):
                    actions[i] = float(np.clip(actions[i] + 0.1, -1.0, 1.0))

        return actions


# ---------------------------------------------------------------------
# Helpers for schema + battery/solar KPIs
# ---------------------------------------------------------------------
def _schema():
    p = os.environ.get("CITYLEARN_SCHEMA")
    if not p or not os.path.exists(p):
        raise RuntimeError(
            "CITYLEARN_SCHEMA not set or path missing. "
            "Run: export CITYLEARN_SCHEMA=/path/to/schema.json"
        )
    return p


def _battery_solar_kpis(env, dt_h=1.0):
    out = {"battery_abuse_kwh": 0.0, "solar_waste_kwh": 0.0}
    t_idx = getattr(env, "time_step", 0) - 1
    if t_idx < 0:
        return out

    for b in getattr(env, "buildings", []):
        es = getattr(b, "electrical_storage", None)
        # Battery abuse
        if es is not None:
            try:
                soc_data = es.soc
                capacity = es.capacity
                soc = (
                    soc_data[t_idx]
                    if hasattr(soc_data, "__getitem__") and len(soc_data) > t_idx
                    else soc_data
                )
                if soc is not None and capacity is not None and soc > 0.95:
                    out["battery_abuse_kwh"] += (soc - 0.95) * capacity
            except Exception:
                pass

        # Solar waste
        try:
            nec = getattr(b, "net_electricity_consumption", [])
            net_grid = (
                nec[t_idx] if hasattr(nec, "__getitem__") and len(nec) > t_idx else 0.0
            )
            if es is not None:
                soc_data = es.soc
                batt_soc = (
                    soc_data[t_idx]
                    if hasattr(soc_data, "__getitem__") and len(soc_data) > t_idx
                    else soc_data
                )
                if net_grid < -0.01 and batt_soc is not None and batt_soc < 0.9:
                    out["solar_waste_kwh"] += abs(net_grid) * dt_h
        except Exception:
            pass

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" Running Advanced RBC (Conservative, inline)")
    print("=" * 60)

    schema = _schema()
    print(f"[INFO] Using schema: {schema}")
    env = CityLearnEnv(schema=schema)

    print("[INFO] Using AdvancedRBC (inline).")
    agent = AdvancedRBC(env)

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    buildings = env.buildings
    dt_h = 1.0

    runs_dir = Path.cwd() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    kpi_csv = runs_dir / "kpi_rbc_advanced_kpis.csv"
    summary_csv = runs_dir / "kpi_rbc_advanced_episode_summary.csv"

    rows = []
    done = False
    step = 0

    print("[INFO] Starting simulation...")
    while not done:
        actions = agent.predict(obs)

        res = env.step(actions)
        if len(res) == 5:
            obs, reward, term, trunc, info = res
            done = bool(term) or bool(trunc)
        else:
            obs, reward, done, info = res

        t_idx = env.time_step - 1

        # --- Energy & cost ---
        net_kw = sum(b.net_electricity_consumption[t_idx] for b in buildings)
        import_kwh = max(net_kw, 0.0) * dt_h
        export_kwh = abs(min(net_kw, 0.0)) * dt_h

        b0 = buildings[0]
        price = b0.pricing.electricity_pricing[t_idx]
        carbon_intensity = b0.carbon_intensity.carbon_intensity[t_idx]

        step_cost = import_kwh * price
        step_carbon = import_kwh * carbon_intensity

        # --- SoC & constraint ---
        socs = [b.electrical_storage.soc[t_idx] for b in buildings]
        soc_mean = float(np.mean(socs))
        soc_min = float(np.min(socs))
        soc_max = float(np.max(socs))
        violation = 1.0 if any(s < 0.0 or s > 1.0 for s in socs) else 0.0

        # --- Thermal discomfort (robust) ---
        discomfort_vals = []
        for b in buildings:
            indoor = b.indoor_dry_bulb_temperature[t_idx]
            if hasattr(b, "indoor_dry_bulb_temperature_cooling_set_point"):
                c_sp = b.indoor_dry_bulb_temperature_cooling_set_point[t_idx]
            elif hasattr(b, "indoor_dry_bulb_temperature_set_point"):
                c_sp = b.indoor_dry_bulb_temperature_set_point[t_idx]
            else:
                c_sp = 100.0

            if hasattr(b, "indoor_dry_bulb_temperature_heating_set_point"):
                h_sp = b.indoor_dry_bulb_temperature_heating_set_point[t_idx]
            elif hasattr(b, "indoor_dry_bulb_temperature_set_point"):
                h_sp = b.indoor_dry_bulb_temperature_set_point[t_idx]
            else:
                h_sp = -100.0

            diff_hot = max(0.0, indoor - c_sp)
            diff_cold = max(0.0, h_sp - indoor)
            discomfort_vals.append(diff_hot + diff_cold)

        thermal_discomfort = float(np.mean(discomfort_vals))

        # --- Advanced KPIs (EV + battery/solar) ---
        try:
            ev_comp = ev_departure_cost_components(env)
        except Exception:
            ev_comp = {
                "total": 0.0,
                "avoidable": 0.0,
                "unavoidable": 0.0,
                "departures": 0,
            }

        ev_deficit = float(ev_comp.get("total", 0.0))
        ev_impossible = 1.0 if ev_comp.get("unavoidable", 0.0) > 0 else 0.0

        adv = _battery_solar_kpis(env, dt_h)
        battery_abuse = float(adv["battery_abuse_kwh"])
        solar_waste = float(adv["solar_waste_kwh"])

        rows.append(
            {
                "episode": 1,
                "step": step + 1,
                "step_net_consumption_kwh": float(net_kw * dt_h),
                "grid_import_kwh": float(import_kwh),
                "grid_export_kwh": float(export_kwh),
                "electricity_price": float(price),
                "step_cost": float(step_cost),
                "step_carbon_kg": float(step_carbon),
                "soc_mean": float(soc_mean),
                "soc_min": float(soc_min),
                "soc_max": float(soc_max),
                "constraint_violation": float(violation),
                "thermal_discomfort": float(thermal_discomfort),
                "ev_departure_deficit_kwh": float(ev_deficit),
                "ev_impossible_request_count": float(ev_impossible),
                "battery_abuse_kwh": float(battery_abuse),
                "solar_waste_kwh": float(solar_waste),
                "reward": float(reward) if np.isscalar(reward) else 0.0,
            }
        )

        step += 1
        if step % 1000 == 0:
            print(f" -> Step {step} done...")

    print(f"[INFO] Simulation finished. {step} steps.")
    print(f"[INFO] Writing per-step KPIs to {kpi_csv}")

    fieldnames = [
        "episode",
        "step",
        "step_net_consumption_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "electricity_price",
        "step_cost",
        "step_carbon_kg",
        "soc_mean",
        "soc_min",
        "soc_max",
        "constraint_violation",
        "thermal_discomfort",
        "ev_departure_deficit_kwh",
        "ev_impossible_request_count",
        "battery_abuse_kwh",
        "solar_waste_kwh",
        "reward",
    ]

    with open(kpi_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, 0.0) for k in fieldnames})

    # --- Optional summary ---
    tot_import = float(sum(r["grid_import_kwh"] for r in rows))
    tot_cost = float(sum(r["step_cost"] for r in rows))
    tot_carbon = float(sum(r["step_carbon_kg"] for r in rows))

    print("[INFO] Running CityLearn evaluate()...")
    eval_scores = {}
    try:
        df = env.evaluate()
        if "name" in df.columns:
            df = df[df["name"] == "District"]
        for _, rr in df.iterrows():
            cf = rr.get("cost_function")
            val = rr.get("value")
            if cf is not None:
                eval_scores[cf] = val
    except Exception as e:
        print(f"[WARN] evaluate() failed: {e}")

    summary_row = {
        "episode": 1,
        "total_import_kwh": tot_import,
        "total_electricity_cost": tot_cost,
        "carbon_emissions_total": tot_carbon,
        "electricity_score": eval_scores.get("electricity_consumption_total", 0.0),
        "cost_score": eval_scores.get("cost_total", 0.0),
        "carbon_score": eval_scores.get("carbon_emissions_total", 0.0),
        "ramping_score": eval_scores.get("ramping_average", 0.0),
        "discomfort_proportion": eval_scores.get("discomfort_proportion", 0.0),
    }

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        w.writeheader()
        w.writerow(summary_row)

    print("=" * 60)
    print(" DONE. kpi_rbc_advanced_kpis.csv (full year) is ready for thesis plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
#