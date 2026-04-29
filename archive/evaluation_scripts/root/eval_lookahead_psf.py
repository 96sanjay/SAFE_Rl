import os, sys, time, json
os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())
os.environ["CITYLEARN_STEMS_SOC_LOW"] = "0.0"
os.environ["CITYLEARN_STEMS_SOC_HIGH"] = "0.95"
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "3.47"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "35.76"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"
import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper
from evaluation.agents.rbc import IntelligentRBC

schema_path = "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
with open(schema_path) as f:
    schema = json.load(f)
schema["root_directory"] = os.path.dirname(os.path.abspath(schema_path))

# --- Run WITHOUT PSF ---
print("=" * 60)
print("Running WITHOUT PSF (baseline RBC)...")
env_base = CityLearnEnv(schema=schema)
safety_base = CityLearnSafetyEnvV3(env_base)
obs_b, _ = safety_base.reset()
flat_names = [n for sub in env_base.action_names for n in sub]
class _FP:
    def __init__(self, raw, names):
        self._raw = raw
        self.action_names = names
        self.action_space = type("S", (), {"shape": (len(names),)})()
    def __getattr__(self, name):
        return getattr(self._raw, name)
rbc_b = IntelligentRBC(_FP(env_base, flat_names), ev_mode="greedy")
dims_b = [len(sub) for sub in env_base.action_names]
base_rewards = []
base_costs = []
for step in range(8759):
    flat_a = rbc_b.predict(obs_b)
    actions = []
    offset = 0
    for d in dims_b:
        actions.append(flat_a[offset:offset+d].copy())
        offset += d
    obs_b, rew, term, trunc, info_b = safety_base.step(actions)
    base_rewards.append(rew)
    base_costs.append(info_b.get("cost", 0.0))
print("Baseline done. Total reward: %.2f" % sum(base_rewards))

# --- Run WITH PSF ---
print("=" * 60)
print("Running WITH Lookahead PSF (H=12)...")
env_psf = CityLearnEnv(schema=schema)
safety_psf = CityLearnSafetyEnvV3(env_psf)
psf_env = LookaheadPSFWrapper(safety_psf, horizon=12, verbose=1)
obs_p, _ = psf_env.reset()
flat_names_p = [n for sub in env_psf.action_names for n in sub]
rbc_p = IntelligentRBC(_FP(env_psf, flat_names_p), ev_mode="greedy")
dims_p = [len(sub) for sub in env_psf.action_names]
psf_rewards = []
psf_costs = []
psf_solves = []
psf_interventions = 0
psf_infeasible = 0
t0 = time.time()
for step in range(8759):
    flat_a = rbc_p.predict(obs_p)
    actions = []
    offset = 0
    for d in dims_p:
        actions.append(flat_a[offset:offset+d].copy())
        offset += d
    obs_p, rew, term, trunc, info_p = psf_env.step(actions)
    psf_rewards.append(rew)
    psf_costs.append(info_p.get("cost", 0.0))
    psf_solves.append(info_p.get("psf_solve_ms", 0))
    if info_p.get("psf_any_intervention"):
        psf_interventions += 1
    if info_p.get("psf_infeasible", 0) > 0:
        psf_infeasible += 1
elapsed = time.time() - t0
print("PSF done in %.1fs (%.1fms/step avg solve)" % (elapsed, np.mean(psf_solves)))

# --- Compute violations from KPI logs ---
# Read constraint violations from the safety env's internal tracking
print("\n" + "=" * 72)
print("  COMPARISON: No PSF vs Lookahead PSF (H=12)")
print("=" * 72)

# Use the eval script's logic to compute violations
from citylearn_safe.wrappers.eval_psf_effectiveness import compute_stems_kpis
kpi_base = compute_stems_kpis(safety_base)
kpi_psf = compute_stems_kpis(safety_psf)

metrics = [
    ("Total Reward", sum(base_rewards), sum(psf_rewards)),
    ("EV deficit rate", kpi_base.get("ev_deficit_rate", 0), kpi_psf.get("ev_deficit_rate", 0)),
    ("Battery violation rate", kpi_base.get("batt_viol_rate", 0), kpi_psf.get("batt_viol_rate", 0)),
    ("Building power viol rate", kpi_base.get("bld_pwr_viol_rate", 0), kpi_psf.get("bld_pwr_viol_rate", 0)),
    ("Grid import viol rate", kpi_base.get("grid_viol_rate", 0), kpi_psf.get("grid_viol_rate", 0)),
]
print("%-35s %12s %12s" % ("Metric", "No PSF", "With PSF"))
print("-" * 60)
for name, v1, v2 in metrics:
    if "rate" in name:
        print("%-35s %11.2f%% %11.2f%%" % (name, v1*100, v2*100))
    else:
        print("%-35s %12.2f %12.2f" % (name, v1, v2))
print("-" * 60)
print("PSF interventions: %d / %d steps (%.1f%%)" % (
    psf_interventions, 8759, 100*psf_interventions/8759))
print("PSF infeasible: %d" % psf_infeasible)
print("Avg solve time: %.1fms" % np.mean(psf_solves))
