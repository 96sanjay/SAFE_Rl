#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# -----------------------------
# Config
# -----------------------------
REPO_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, str(Path(REPO_ROOT) / "ev"))  # for IntelligentRBC

SEED = 42
OUT_DIR = Path("runs/oracle_eval_full_1seed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# env vars (same as your training/eval)
os.environ["PYTHONPATH"] = f"{REPO_ROOT}:" + os.environ.get("PYTHONPATH", "")
os.environ["CITYLEARN_SCHEMA"] = f"{REPO_ROOT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
os.environ["CITYLEARN_EXPORT_FACTOR"] = "0.7"
os.environ["CITYLEARN_REWARD_SCALE"] = "1.0"
os.environ["CITYLEARN_EV_COST_SCALE"] = "3.0"
os.environ["CITYLEARN_KPI_FLUSH_EVERY_STEP"] = "0"
os.environ["CITYLEARN_KPI_RUN_NAME"] = "OracleEval_FULL_1seed"

# Models to run
MODELS = [
    {"name": "RBC-Greedy", "type": "rbc", "checkpoint_dir": None},
    {"name": "PPO-Lag-Lambda40", "type": "ppo", "checkpoint_dir": "runs/ppo_lag_3constraints_lambda40_highexplore_100ep"},
]

# -----------------------------
# Imports from repo
# -----------------------------
from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env

# fixed RBC implementation (you already patched this file)
from run_rbc_comparison_COMPLETE import IntelligentRBC, _norm_id, _is_valid_ev_id


# -----------------------------
# PPO policy loader (OmniSafe)
# -----------------------------
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim=153, act_dim=26):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs, deterministic=True):
        mean = self.mean(obs)
        if deterministic:
            return mean
        std = torch.exp(self.log_std)
        return mean + torch.randn_like(mean) * std


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    full_path = f"{REPO_ROOT}/{checkpoint_dir}" if not checkpoint_dir.startswith("/") else checkpoint_dir
    cps = glob(f"{full_path}/**/torch_save/epoch-*.pt", recursive=True)
    if not cps:
        raise FileNotFoundError(f"No checkpoints found in: {full_path}")

    best_cp, best_epoch = None, -1
    for cp in cps:
        bn = os.path.basename(cp)
        try:
            ep = int(bn.replace("epoch-", "").replace(".pt", ""))
        except Exception:
            continue
        if ep > best_epoch:
            best_epoch, best_cp = ep, cp

    if best_cp is None:
        raise FileNotFoundError(f"No valid epoch-*.pt found in: {full_path}")
    return best_cp


def load_ppo_policy_fn(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    net = GaussianPolicy(obs_dim=153, act_dim=26)
    net.load_state_dict(ckpt["pi"])
    net.eval()

    obs_norm_raw = ckpt.get("obs_normalizer", {})
    if "mean" in obs_norm_raw:
        mu, var = obs_norm_raw["mean"], obs_norm_raw["var"]
    elif "_mean" in obs_norm_raw:
        mu, var = obs_norm_raw["_mean"], obs_norm_raw["_var"]
    else:
        mu, var = torch.zeros(153), torch.ones(153)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        o = torch.FloatTensor(obs).unsqueeze(0)
        o = (o - mu) / torch.sqrt(var + 1e-8)
        with torch.no_grad():
            a = net(o, deterministic=True).squeeze(0).numpy()
        return a.astype(np.float32)

    return policy_fn


# -----------------------------
# Oracle EV forcing (aligned with your wrapper)
# -----------------------------
def _unwrap_action_names(names):
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        return names[0]
    return names


def build_ev_idx_by_charger(raw_env):
    names = _unwrap_action_names(getattr(raw_env, "action_names", []))
    out = {}
    for idx, n in enumerate(names):
        s = str(n).strip().lower()
        key = "electric_vehicle_storage_charger_"
        if key in s:
            suffix = s.split(key, 1)[1]
            out[f"charger_{suffix}"] = idx
    return out


def oracle_connected_next(raw_env, charger_id: str) -> bool:
    """
    IMPORTANT:
    Your wrapper stores actions at tau_store = time_step_before + 1,
    so check connectivity for the action you are about to apply at time_step+1.
    """
    t_check = int(getattr(raw_env, "time_step", 0)) + 1
    if t_check < 0:
        t_check = 0

    for b in getattr(raw_env, "buildings", []) or []:
        for ch in getattr(b, "electric_vehicle_chargers", []) or []:
            cid = _norm_id(getattr(ch, "charger_id", getattr(ch, "name", "")))
            if cid != charger_id:
                continue

            sim = getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))
            if sim is None:
                return False

            state = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
            ev_id = np.asarray(getattr(sim, "_electric_vehicle_id"))

            if state.ndim != 1:
                return False

            if t_check >= len(state):
                t_check = len(state) - 1
            if t_check < 0:
                return False

            st = float(state[t_check])
            eid = ev_id[t_check]
            return (st == 1.0) and _is_valid_ev_id(eid)

    return False


# -----------------------------
# CityLearn KPI extraction (reliable)
# -----------------------------
def extract_citylearn_kpis(raw_env) -> dict:
    try:
        df = raw_env.evaluate()
    except Exception as e:
        return {"_kpi_error": str(e)}

    building_name = "District"
    if "name" in df.columns:
        if (df["name"] == "District").any():
            building_name = "District"
        else:
            uniq = df["name"].unique()
            building_name = uniq[0] if len(uniq) else "District"

    def get_val(cost_fn: str):
        try:
            sub = df[(df["name"] == building_name) & (df["cost_function"] == cost_fn)]
            if not sub.empty:
                v = sub["value"].iloc[0]
                return float(v) if pd.notna(v) else float("nan")
        except Exception:
            pass
        return float("nan")

    return {
        "citylearn_electricity_consumption_total": get_val("electricity_consumption_total"),
        "citylearn_carbon_emissions_total": get_val("carbon_emissions_total"),
        "citylearn_cost_total": get_val("cost_total"),
        "citylearn_zero_net_energy": get_val("zero_net_energy"),
        "citylearn_discomfort_proportion": get_val("discomfort_proportion"),
        "citylearn_ramping_average": get_val("ramping_average"),
        "citylearn_daily_peak_average": get_val("daily_peak_average"),
        "citylearn_all_time_peak_average": get_val("all_time_peak_average"),
    }


# -----------------------------
# Rollout runner (sums everything from info)
# -----------------------------
SUM_KEYS = [
    # total CMDP cost + split
    "cost",
    "cost_ev_departure",
    "cost_ev_departure_avoidable",
    "cost_ev_departure_unavoidable",
    "cost_grid_peak",
    "cost_grid_peak_raw",
    "cost_grid_ramp",
    "cost_grid_ramp_raw",
    # EV deficits + counts
    "ev_departure_deficit_kwh",
    "ev_avoidable_deficit_kwh",
    "ev_unavoidable_deficit_kwh",
    "ev_departure_departures",
    "ev_missing_action_samples",
    # reward/bill
    "reward",
    "reward_bill_raw",
]


def make_env():
    base = make_base_env(central_agent=True)
    return CityLearnSafetyEnvV3(base)


def run_rollout(model_name: str, model_type: str, policy_fn, seed: int, use_oracle: bool):
    env = make_env()
    obs, info = env.reset(seed=seed)

    raw = unwrap_to_raw_citylearn_env(env)
    ev_map = build_ev_idx_by_charger(raw)

    sums = {k: 0.0 for k in SUM_KEYS}
    steps = 0
    done = False

    t0 = time.time()
    while not done:
        action = np.asarray(policy_fn(obs), dtype=np.float32).copy()

        # Oracle forcing: EV action=1.0 when connected (at time_step+1)
        if use_oracle:
            raw = unwrap_to_raw_citylearn_env(env)
            for charger_id, a_idx in ev_map.items():
                if 0 <= a_idx < len(action) and oracle_connected_next(raw, charger_id):
                    action[a_idx] = 1.0

        obs, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        steps += 1

        for k in SUM_KEYS:
            sums[k] += float(info.get(k, 0.0) or 0.0)

        if steps % 1000 == 0:
            print(f"  [{model_name} | {'ORACLE' if use_oracle else 'POLICY'}] step {steps}/8760")

    elapsed = time.time() - t0

    raw = unwrap_to_raw_citylearn_env(env)
    kpis = extract_citylearn_kpis(raw)

    row = {
        "model_name": model_name,
        "model_type": model_type,
        "rollout_type": "oracle" if use_oracle else "policy",
        "seed": seed,
        "steps": steps,
        "elapsed_sec": elapsed,
        **sums,
        **kpis,
    }
    return row


def build_rbc_policy_for_env(env):
    agent = IntelligentRBC(env, ev_mode="greedy")
    def policy_fn(obs):
        return agent.predict(obs).astype(np.float32)
    return policy_fn


# -----------------------------
# Main
# -----------------------------
rows = []

for model in MODELS:
    print("\n" + "=" * 90)
    print(f"RUNNING MODEL: {model['name']}  (seed={SEED})")
    print("=" * 90)

    if model["type"] == "rbc":
        # Bind agent to the env used in the rollout
        def run_rbc(model_name: str, seed: int, use_oracle: bool):
            env = make_env()
            obs, info = env.reset(seed=seed)
            policy_fn = build_rbc_policy_for_env(env)

            raw = unwrap_to_raw_citylearn_env(env)
            ev_map = build_ev_idx_by_charger(raw)

            sums = {k: 0.0 for k in SUM_KEYS}
            steps = 0
            done = False
            t0 = time.time()

            while not done:
                action = np.asarray(policy_fn(obs), dtype=np.float32).copy()

                if use_oracle:
                    raw = unwrap_to_raw_citylearn_env(env)
                    for charger_id, a_idx in ev_map.items():
                        if 0 <= a_idx < len(action) and oracle_connected_next(raw, charger_id):
                            action[a_idx] = 1.0

                obs, reward, term, trunc, info = env.step(action)
                done = bool(term or trunc)
                steps += 1

                for k in SUM_KEYS:
                    sums[k] += float(info.get(k, 0.0) or 0.0)

                if steps % 1000 == 0:
                    print(f"  [{model_name} | {'ORACLE' if use_oracle else 'POLICY'}] step {steps}/8760")

            elapsed = time.time() - t0
            raw = unwrap_to_raw_citylearn_env(env)
            kpis = extract_citylearn_kpis(raw)

            return {
                "model_name": model_name,
                "model_type": "rbc",
                "rollout_type": "oracle" if use_oracle else "policy",
                "seed": seed,
                "steps": steps,
                "elapsed_sec": elapsed,
                **sums,
                **kpis,
            }

        rows.append(run_rbc(model["name"], SEED, use_oracle=False))
        rows.append(run_rbc(model["name"], SEED, use_oracle=True))

    else:
        ckpt = find_latest_checkpoint(model["checkpoint_dir"])
        print(f"  Loading PPO checkpoint: {ckpt}")
        policy_fn = load_ppo_policy_fn(ckpt)

        rows.append(run_rollout(model["name"], model["type"], policy_fn, SEED, use_oracle=False))
        rows.append(run_rollout(model["name"], model["type"], policy_fn, SEED, use_oracle=True))

df = pd.DataFrame(rows)

cost_csv = OUT_DIR / "rollout_costs_and_kpis.csv"
df.to_csv(cost_csv, index=False)

# compact summary CSV
summary_cols = [
    "model_name","rollout_type","seed",
    "cost","cost_ev_departure","cost_grid_peak","cost_grid_ramp",
    "ev_departure_deficit_kwh","ev_avoidable_deficit_kwh","ev_unavoidable_deficit_kwh",
    "citylearn_cost_total","citylearn_carbon_emissions_total","citylearn_electricity_consumption_total",
    "elapsed_sec"
]
summary_csv = OUT_DIR / "summary.csv"
df[summary_cols].to_csv(summary_csv, index=False)

print("\n" + "=" * 90)
print("DONE")
print(f"Saved: {cost_csv}")
print(f"Saved: {summary_csv}")
print("=" * 90)

# Print oracle gap per model (policy - oracle)
print("\nORACLE GAP (EV deficit):")
for m in df["model_name"].unique():
    pol = df[(df["model_name"]==m) & (df["rollout_type"]=="policy")]["ev_departure_deficit_kwh"].values
    orc = df[(df["model_name"]==m) & (df["rollout_type"]=="oracle")]["ev_departure_deficit_kwh"].values
    if len(pol)==1 and len(orc)==1:
        avoid = max(0.0, float(pol[0]) - float(orc[0]))
        frac = (avoid / float(pol[0])) if float(pol[0]) > 0 else 0.0
        print(f"  {m:20s} policy={pol[0]:8.3f}  oracle={orc[0]:8.3f}  avoidable={avoid:8.3f}  ({frac*100:5.1f}%)")
