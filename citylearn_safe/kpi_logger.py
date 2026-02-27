
"""
Custom KPI logger for CityLearn safety environment - UPDATED VERSION (FULL FILE).

This logger writes:
- Per-step KPI CSV: <run_name>.csv
- Per-step costs CSV: <run_name>_costs.csv
- Episode summary CSV: <run_name>_episode_summary.csv

Key guarantees:
✅ Fieldnames defined ONCE in __init__ (stable schema)
✅ Includes action_0..action_25 and action_ev_0..action_ev_7
✅ Includes bill/reward bill fields:
   - step_bill, export_factor
   - reward_bill_raw, reward_export_factor, reward_scale
✅ Auto-upgrades CSV header if schema changed (rewrites file + preserves old rows)
✅ Optional flush every step with env var:
   CITYLEARN_KPI_FLUSH_EVERY_STEP=1  (debug)
"""

import csv
import os
from typing import Dict, Any, List

import numpy as np


class KPILogger:
    """Custom logger to track KPIs separately from OmniSafe."""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = log_dir
        self.run_name = run_name

        os.makedirs(log_dir, exist_ok=True)

        # File paths
        self.csv_path = os.path.join(log_dir, f"{run_name}.csv")
        self.cost_csv_path = os.path.join(log_dir, f"{run_name}_costs.csv")
        self.episode_summary_csv_path = os.path.join(log_dir, f"{run_name}_episode_summary.csv")

        # Debug: flush behavior
        self.flush_every_step = bool(int(os.environ.get("CITYLEARN_KPI_FLUSH_EVERY_STEP", "0")))

        # =====================================================================
        # MAIN KPI CSV COLUMNS (FIXED HEADER)
        # =====================================================================
        self.fieldnames: List[str] = [
            # -------------------- METADATA --------------------
            "episode",
            "step",
            "hour",

            # -------------------- BATTERY SAFETY --------------------
            "soc_mean",
            "soc_min",
            "soc_max",
            "soc_std",
            "battery_soc_b1",
            "battery_soc_b2",
            "battery_soc_b3",
            "battery_soc_b4",
            "battery_soc_b5",
            "battery_soc_b6",
            "battery_soc_b7",
            "battery_soc_b8",
            "battery_soc_b9",
            "battery_soc_b10",
            "battery_soc_b11",
            "battery_soc_b12",
            "battery_soc_b13",
            "battery_soc_b14",
            "battery_soc_b15",
            "battery_soc_b16",
            "battery_soc_b17",
            "battery_abuse_kwh",
            "battery_abuse_excess_kwh_equiv",
            "battery_abuse_hours",
            "cost_stems_battery",
            "battery_soc_violation",
            "battery_soc_violation_any",
            "battery_soc_violation_frac",
            "battery_soc_violation_rate_%",
            "battery_soc_violation_count",


            # -------------------- ACTION SUMMARY --------------------
            "action_mean",
            "action_std",
            "action_min",
            "action_max",
            "step_count",

            # -------------------- ENERGY/GRID --------------------
            "step_net_consumption_kwh",
            "grid_import_kwh",
            "grid_export_kwh",
            "net_grid_kwh",
            "cost_stems_building_power",
            "building_power_violation",
            "building_power_violation_count",
            "building_power_violation_frac",
            "building_power_violation_rate_%",

            "cost_stems_grid_power",
            "grid_power_violation",
            "solar_generation_kwh",
            "solar_waste_kwh",
            "non_shiftable_load_kwh",
            "non_shiftable_load_obs_b1",

            # -------------------- PRICING & CARBON --------------------
            "electricity_price",
            "step_cost",
            "carbon_intensity",
            "step_carbon_kg",

            # -------------------- BILLING LOGIC --------------------
            "step_bill",
            "export_factor",

            # -------------------- THERMAL COMFORT --------------------
            "outdoor_temperature",
            "indoor_temperature",
            "thermal_discomfort",

            # -------------------- COMFORT CONSTRAINT (LSTM) --------------------
            "comfort_enabled",
            "comfort_warmup_complete",
            "comfort_in_warmup",
            "cost_comfort",
            "cost_comfort_raw",
            "comfort_violation",
            "comfort_tin",
            "comfort_tset",

            # -------------------- TIME FEATURES --------------------
            "month_cos",
            "month_sin",
            "hour_cos",
            "hour_sin",
            "day_type_cos",
            "day_type_sin",

            # -------------------- EV RELIABILITY --------------------
            "ev_departure_deficit_kwh",
            "ev_avoidable_deficit_kwh",
            "ev_unavoidable_deficit_kwh",
            "ev_departure_departures",
            "ev_impossible_request_count",
            "ev_missing_action_samples",

            # --- NEW V3 diagnostics (charge/discharge split) ---
            "ev_v3_missed_charge_soc",
            "ev_v3_discharge_harm_soc",
            "ev_v3_total_blame_soc",

            # Backward compatibility (if some scripts still populate)
            "cost_ev_departure_avoidable",
            "cost_ev_departure_unavoidable",

            # -------------------- CMDP COSTS --------------------
            "cost",
            "cost_building_soc",
            "cost_ev_departure",
            "cost_ev_dense",
            "constraint_violation",

            # -------------------- GRID PEAK CONSTRAINT --------------------
            "cost_grid_peak",
            "cost_grid_peak_raw",
            "grid_peak_violation",

            # -------------------- GRID RAMP CONSTRAINT --------------------
            "cost_grid_ramp",
            "cost_grid_ramp_raw",
            "grid_ramp_delta",
            "grid_ramp_violation",

            # -------------------- REWARDS --------------------
            "reward",
            "citylearn_reward",
            "used_energy_reward",

            # ✅ Bill reward fields
            "reward_bill_raw",
            "reward_bill",
            "reward_export_factor",
            "reward_scale",
            
            # ✅ STEMS reward fields
            "reward_type",
            "reward_stems_total",
            "reward_economic",
            "reward_stability",
            "reward_stability_grid",
            "reward_stability_building",
            "reward_stability_ramp",
            "reward_renewable",
            "reward_comfort",

            # -------------------- CITYLEARN EPISODE-END KPIS --------------------
            "citylearn_electricity_consumption_total",
            "citylearn_carbon_emissions_total",
            "citylearn_cost_total",
            "citylearn_daily_peak_average",
            "citylearn_all_time_peak_average",
            "citylearn_ramping_average",
            "citylearn_discomfort_proportion",
            "citylearn_zero_net_energy",
        ]

        # ✅ Per-dimension actions
        self.fieldnames += [f"action_{i}" for i in range(26)]
        self.fieldnames += [f"action_ev_{j}" for j in range(8)]

        # Writers/handles
        self.csv_file = None
        self.writer = None

        self.cost_csv_file = None
        self.cost_writer = None
        self.cost_fieldnames = [
            "episode",
            "step",
            "cost_total",
            "cost_building_soc",
            "cost_ev_departure",
            "cost_ev_departure_avoidable",
            "cost_ev_departure_unavoidable",
            "ev_departure_departures",
            "cost_grid_peak",
            "cost_grid_peak_raw",
            "cost_grid_ramp",
            "cost_grid_ramp_raw",
        ]

        self.episode_summary_csv_file = None
        self.episode_summary_writer = None
        self.episode_summary_fieldnames = [
            "episode",
            "episode_length",
            "total_cost",
            "violation_percentage",
            "avg_cost_per_step",
            "total_reward",
            "avg_reward_per_step",
            "total_import_kwh",
            "total_export_kwh",
            "total_generation_kwh",
            "total_load_kwh",
            "total_electricity_cost",
            "carbon_emissions_total",
            "avg_soc",
            "min_soc",
            "max_soc",
            "avg_electricity_price",
            "peak_demand_kw",
            "zero_net_energy",
            "discomfort_proportion",
            "lagrangian_multiplier",
            "ramping_score",
        ]

        # Buffers
        self.episode_data = []
        self.current_episode_data = []

    def log_step(self, info: Dict[str, Any], step: int, episode: int):
        """Log KPIs from a single step."""
        info = dict(info) if info is not None else {}

        # Reset episode data if this is a new episode
        if step <= 1 and self.current_episode_data:
            prev_episode = self.current_episode_data[0].get("episode", 0)
            if episode != prev_episode:
                self.current_episode_data = []

        # Track episode data for summary computation
        step_data = {
            "episode": episode,
            "step": step,
            **{k: v for k, v in info.items() if k in self.fieldnames},
        }
        self.current_episode_data.append(step_data)

        # ---- Build per-step KPI row ----
        kpi_data: Dict[str, float] = {
            "episode": float(episode),
            "step": float(step),
        }

        citylearn_kpis = {
            "citylearn_electricity_consumption_total",
            "citylearn_carbon_emissions_total",
            "citylearn_cost_total",
            "citylearn_daily_peak_average",
            "citylearn_all_time_peak_average",
            "citylearn_ramping_average",
            "citylearn_discomfort_proportion",
            "citylearn_zero_net_energy",
        }

        # Fill all fields from info or default
        for key in self.fieldnames:
            if key in ("episode", "step"):
                continue

            if key in citylearn_kpis:
                kpi_data[key] = float("nan")
                continue

            if key in info:
                value = info[key]
                # Special handling for string fields
                if key == "reward_type":
                    kpi_data[key] = str(value)
                elif isinstance(value, (int, float, np.number, bool)):
                    kpi_data[key] = float(value)
                elif isinstance(value, np.ndarray):
                    try:
                        kpi_data[key] = float(value.item() if value.size == 1 else np.mean(value))
                    except Exception:
                        kpi_data[key] = 0.0
                else:
                    kpi_data[key] = 0.0
            else:
                # Default for string fields
                if key == "reward_type":
                    kpi_data[key] = "unknown"
                else:
                    kpi_data[key] = 0.0

        # -------------------- BILLING LOGIC (always compute) --------------------
        export_factor = float(os.environ.get("CITYLEARN_EXPORT_FACTOR", "1.0"))
        reward_scale = float(os.environ.get("CITYLEARN_REWARD_SCALE", "1.0"))

        import_kwh = float(kpi_data.get("grid_import_kwh", 0.0))
        export_kwh = float(kpi_data.get("grid_export_kwh", 0.0))
        price = float(kpi_data.get("electricity_price", 0.0))

        bill = float((import_kwh * price) - (export_factor * export_kwh * price))

        kpi_data["export_factor"] = export_factor
        kpi_data["step_bill"] = bill

        # ✅ Ensure reward bill fields always exist
        kpi_data["reward_bill_raw"] = float(info.get("reward_bill_raw", bill))
        kpi_data["reward_export_factor"] = float(info.get("reward_export_factor", export_factor))
        kpi_data["reward_scale"] = float(info.get("reward_scale", reward_scale))
        # ----------------------------------------------------------------------

        # Special case: compute 'hour' if not provided
        if "hour" not in info:
            kpi_data["hour"] = float(step % 24)

        # Special case: compute 'net_grid_kwh' if not present
        if "net_grid_kwh" not in info:
            kpi_data["net_grid_kwh"] = import_kwh - export_kwh

        # Optional: Console debug
        if step % 100 == 0:
            print(
                f"[KPILogger] Ep {episode}, Step {step}: "
                f"bill={kpi_data.get('step_bill', 0.0):.3f} | "
                f"cost={kpi_data.get('cost', 0.0):.3f} | "
                f"reward={kpi_data.get('reward', 0.0):.3f}"
            )

        self.episode_data.append(kpi_data)

        # ---- Write per-step cost breakdown CSV ----
        self._log_cost(
            step=step,
            episode=episode,
            cost_total=float(kpi_data.get("cost", 0.0)),
            cost_building_soc=float(kpi_data.get("cost_building_soc", 0.0)),
            cost_ev_departure=float(kpi_data.get("cost_ev_departure", 0.0)),
            cost_ev_departure_avoidable=float(kpi_data.get("cost_ev_departure_avoidable", 0.0)),
            cost_ev_departure_unavoidable=float(kpi_data.get("cost_ev_departure_unavoidable", 0.0)),
            ev_departure_departures=int(kpi_data.get("ev_departure_departures", 0)),
            cost_grid_peak=float(kpi_data.get("cost_grid_peak", 0.0)),
            cost_grid_peak_raw=float(kpi_data.get("cost_grid_peak_raw", 0.0)),
            cost_grid_ramp=float(kpi_data.get("cost_grid_ramp", 0.0)),
            cost_grid_ramp_raw=float(kpi_data.get("cost_grid_ramp_raw", 0.0)),
        )

        # Flush behavior
        if self.flush_every_step:
            self._write_to_csv()
        else:
            if step % 100 == 0:
                self._write_to_csv()

    def log_episode_end(self, info: Dict[str, Any], episode: int):
        """Log episode-end KPIs."""
        info = dict(info) if info is not None else {}

        if self.episode_data:
            last_step = self.episode_data[-1]
            for key in [
                "citylearn_electricity_consumption_total",
                "citylearn_carbon_emissions_total",
                "citylearn_cost_total",
                "citylearn_daily_peak_average",
                "citylearn_all_time_peak_average",
                "citylearn_ramping_average",
                "citylearn_discomfort_proportion",
                "citylearn_zero_net_energy",
            ]:
                if key in info:
                    last_step[key] = float(info[key])

            self._write_to_csv()
            self._log_episode_summary(episode, info)
            self.episode_data = []
        else:
            self._write_to_csv()

    def _log_episode_summary(self, episode: int, info: Dict[str, Any]):
        """Compute and log episode-level summary statistics."""
        if not self.current_episode_data:
            return

        step_data = [d for d in self.current_episode_data if d.get("step", 0) > 0]
        if not step_data:
            self.current_episode_data = []
            return

        summary = {"episode": episode, "episode_length": len(step_data)}

        costs = [float(d.get("cost", 0.0)) for d in step_data]
        violations = [float(d.get("constraint_violation", 0.0)) for d in step_data]
        summary["total_cost"] = float(sum(costs))
        summary["violation_percentage"] = float(sum(violations) / len(step_data) * 100.0) if step_data else 0.0
        summary["avg_cost_per_step"] = float(np.mean(costs)) if costs else 0.0

        rewards = [float(d.get("reward", 0.0)) for d in step_data]
        summary["total_reward"] = float(sum(rewards))
        summary["avg_reward_per_step"] = float(np.mean(rewards)) if rewards else 0.0

        imports = [float(d.get("grid_import_kwh", 0.0)) for d in step_data]
        exports = [float(d.get("grid_export_kwh", 0.0)) for d in step_data]
        generation = [float(d.get("solar_generation_kwh", 0.0)) for d in step_data]

        summary["total_import_kwh"] = float(sum(imports))
        summary["total_export_kwh"] = float(sum(exports))
        summary["total_generation_kwh"] = float(sum(abs(g) for g in generation))
        summary["total_load_kwh"] = float(info.get("citylearn_electricity_consumption_total", 0.0))

        summary["total_electricity_cost"] = float(info.get("citylearn_cost_total", 0.0))
        summary["carbon_emissions_total"] = float(info.get("citylearn_carbon_emissions_total", 0.0))

        soc_means = [float(d.get("soc_mean", 0.0)) for d in step_data]
        soc_mins = [float(d.get("soc_min", 0.0)) for d in step_data]
        soc_maxs = [float(d.get("soc_max", 0.0)) for d in step_data]
        summary["avg_soc"] = float(np.mean(soc_means)) if soc_means else 0.0
        summary["min_soc"] = float(np.min(soc_mins)) if soc_mins else 0.0
        summary["max_soc"] = float(np.max(soc_maxs)) if soc_maxs else 0.0

        prices = [float(d.get("electricity_price", 0.0)) for d in step_data]
        summary["avg_electricity_price"] = float(np.mean(prices)) if prices else 0.0

        summary["peak_demand_kw"] = float(info.get("citylearn_daily_peak_average", 0.0))
        summary["zero_net_energy"] = float(info.get("citylearn_zero_net_energy", 0.0))
        summary["discomfort_proportion"] = float(info.get("citylearn_discomfort_proportion", 0.0))
        summary["lagrangian_multiplier"] = 0.0
        summary["ramping_score"] = float(info.get("citylearn_ramping_average", 0.0))

        if self.episode_summary_csv_file is None:
            self.episode_summary_csv_file = open(self.episode_summary_csv_path, "w", newline="")
            self.episode_summary_writer = csv.DictWriter(
                self.episode_summary_csv_file,
                fieldnames=self.episode_summary_fieldnames,
            )
            self.episode_summary_writer.writeheader()

        filtered_summary = {k: summary.get(k, 0.0) for k in self.episode_summary_fieldnames}
        self.episode_summary_writer.writerow(filtered_summary)
        self.episode_summary_csv_file.flush()

        self.current_episode_data = []

    def _write_to_csv(self):
        """
        Write accumulated per-step KPI data to the main CSV file.

        ✅ Auto-upgrades schema:
        If existing header != current self.fieldnames, rewrite file with new header
        and preserve old rows (missing columns filled with 0.0).
        """
        if not self.episode_data:
            return

        def read_existing_rows(path: str):
            rows = []
            try:
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        rows.append(r)
            except Exception:
                return []
            return rows

        rewrite = False
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, "r", encoding="utf-8") as f:
                    existing_header = f.readline().strip().split(",")
                if existing_header != self.fieldnames:
                    rewrite = True
            except Exception:
                rewrite = True
        else:
            rewrite = True

        if rewrite:
            old_rows = read_existing_rows(self.csv_path) if os.path.exists(self.csv_path) else []

            # Close existing handle if open
            if self.csv_file:
                try:
                    self.csv_file.close()
                except Exception:
                    pass
                self.csv_file = None
                self.writer = None

            # Rewrite with new header
            self.csv_file = open(self.csv_path, "w", newline="")
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.writer.writeheader()

            # Re-write old rows into new schema
            for r in old_rows:
                filtered_old = {k: r.get(k, 0.0) for k in self.fieldnames}
                self.writer.writerow(filtered_old)

        # Open normally if still not open
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.writer.writeheader()

        # Write new rows
        for row in self.episode_data:
            filtered_row = {k: row.get(k, 0.0) for k in self.fieldnames}
            self.writer.writerow(filtered_row)

        self.csv_file.flush()
        self.episode_data = []

    def _log_cost(
        self,
        *,
        step: int,
        episode: int,
        cost_total: float,
        cost_building_soc: float,
        cost_ev_departure: float,
        cost_ev_departure_avoidable: float,
        cost_ev_departure_unavoidable: float,
        ev_departure_departures: int,
        cost_grid_peak: float,
        cost_grid_peak_raw: float,
        cost_grid_ramp: float,
        cost_grid_ramp_raw: float,
    ) -> None:
        """Write per-step safety cost breakdown to a dedicated CSV."""
        if self.cost_csv_file is None:
            self.cost_csv_file = open(self.cost_csv_path, "w", newline="")
            self.cost_writer = csv.DictWriter(self.cost_csv_file, fieldnames=self.cost_fieldnames)
            self.cost_writer.writeheader()

        self.cost_writer.writerow(
            {
                "episode": episode,
                "step": step,
                "cost_total": cost_total,
                "cost_building_soc": cost_building_soc,
                "cost_ev_departure": cost_ev_departure,
                "cost_ev_departure_avoidable": cost_ev_departure_avoidable,
                "cost_ev_departure_unavoidable": cost_ev_departure_unavoidable,
                "ev_departure_departures": ev_departure_departures,
                "cost_grid_peak": cost_grid_peak,
                "cost_grid_peak_raw": cost_grid_peak_raw,
                "cost_grid_ramp": cost_grid_ramp,
                "cost_grid_ramp_raw": cost_grid_ramp_raw,
            }
        )
        self.cost_csv_file.flush()

    def close(self):
        """Close all CSV files."""
        if self.csv_file:
            self._write_to_csv()
            try:
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = None
            self.writer = None

        if self.cost_csv_file:
            try:
                self.cost_csv_file.close()
            except Exception:
                pass
            self.cost_csv_file = None
            self.cost_writer = None

        if self.episode_summary_csv_file:
            try:
                self.episode_summary_csv_file.close()
            except Exception:
                pass
            self.episode_summary_csv_file = None
            self.episode_summary_writer = None


# Global logger instance
_kpi_logger = None


def init_kpi_logger(log_dir: str, run_name: str):
    """Initialize the global KPI logger."""
    global _kpi_logger
    _kpi_logger = KPILogger(log_dir, run_name)


def log_kpis(info: Dict[str, Any], step: int, episode: int):
    """Log KPIs using the global logger."""
    global _kpi_logger
    if _kpi_logger:
        _kpi_logger.log_step(info, step, episode)


def log_episode_end(info: Dict[str, Any], episode: int):
    """Log episode-end KPIs."""
    global _kpi_logger
    if _kpi_logger:
        _kpi_logger.log_episode_end(info, episode)


def close_kpi_logger():
    """Close the global KPI logger."""
    global _kpi_logger
    if _kpi_logger:
        _kpi_logger.close()
        _kpi_logger = None
