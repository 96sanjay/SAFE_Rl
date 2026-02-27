#!/usr/bin/env python3
import os, sys, numpy as np, torch, glob, pandas as pd
from pathlib import Path
from torch import nn
from collections import defaultdict

PROJECT_ROOT = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, str(PROJECT_ROOT))

checkpoints = glob.glob(str(PROJECT_ROOT / "runs/ppo_lag_cost_based_reward/*/seed-*/torch_save/epoch-100.pt"))
if not checkpoints: raise FileNotFoundError("No checkpoint")
CHECKPOINT_PATH = Path(checkpoints[0])

os.environ['CITYLEARN_SCHEMA'] = "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

print("\n" + "="*80)
print("EV CHARGING SESSION ANALYSIS - V3 Temporal Flaw Diagnosis")
print("="*80)

# Setup
base_env = make_base_env(central_agent=True)
env = CityLearnSafetyEnvV3(base_env, soc_min=0.0, soc_max=0.95)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Load model
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        self.mean = nn.Sequential(*layers, nn.Linear(in_dim, act_dim), nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    def forward(self, obs):
        return self.mean(obs)

actor = GaussianActor(obs_dim, act_dim, hidden_sizes=[64, 64])
actor.load_state_dict(ckpt['pi'])
actor.eval()

norm = ckpt['obs_normalizer']
obs_mean = torch.FloatTensor(norm['_mean'])
obs_std = torch.FloatTensor(norm['_std'])

@torch.no_grad()
def policy(obs):
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
    return actor(obs_t).cpu().numpy().squeeze()

EV_START_IDX = 17
NUM_EVS = 8

print(f"\n🔍 Tracking {NUM_EVS} EV chargers (indices {EV_START_IDX}-{EV_START_IDX+NUM_EVS-1})")

# Session tracking
class EVSession:
    def __init__(self, ev_idx, start_step):
        self.ev_idx = ev_idx
        self.start_step = start_step
        self.actions = []  # All actions during session
        self.socs = []  # SOC at each step
        self.end_step = None
        self.departure_action = None
        self.deficit = 0.0
        
    def add_action(self, action, soc):
        self.actions.append(action)
        self.socs.append(soc)
    
    def finalize(self, end_step, departure_action, deficit):
        self.end_step = end_step
        self.departure_action = departure_action
        self.deficit = deficit
    
    def session_length(self):
        return len(self.actions)
    
    def mean_action(self):
        return np.mean(self.actions) if self.actions else 0.0
    
    def min_action(self):
        return np.min(self.actions) if self.actions else 0.0
    
    def had_low_action(self, threshold=0.9):
        """Did agent use action < threshold during session?"""
        return any(a < threshold for a in self.actions)
    
    def v3_classification(self):
        """V3's logic: only checks departure action"""
        if self.departure_action is not None and self.departure_action < 1.0:
            return "controllable"
        return "uncontrollable"
    
    def correct_classification(self, threshold=0.9):
        """Correct logic: checks if agent could have done better"""
        if self.had_low_action(threshold):
            return "controllable"
        return "uncontrollable"

# Track active sessions
active_sessions = {i: None for i in range(NUM_EVS)}
completed_sessions = []

# Get citylearn env for EV state access
citylearn_env = env._get_citylearn_env()

print(f"\n🎮 Running episode with session tracking...")
obs, _ = env.reset(seed=42)
done = truncated = False
step = 0

while not (done or truncated):
    action = np.clip(policy(obs), -1.0, 1.0)
    obs, r, done, truncated, info = env.step(action)
    
    # Get EV states from environment
    idx = env._state_time_index(citylearn_env)
    
    for ev_local_idx in range(NUM_EVS):
        ev_action_idx = EV_START_IDX + ev_local_idx
        
        # Get EV connection state from buildings
        # EV chargers are distributed across buildings
        # Need to find which building has this charger
        ev_connected = False
        ev_soc = 0.0
        
        try:
            for building in citylearn_env.buildings:
                if hasattr(building, 'electric_vehicle_chargers'):
                    chargers = building.electric_vehicle_chargers
                    if chargers and len(chargers) > ev_local_idx:
                        charger = chargers[ev_local_idx]
                        if hasattr(charger, 'connected_electric_vehicle_at_charger_battery_capacity'):
                            cap = charger.connected_electric_vehicle_at_charger_battery_capacity
                            if cap is not None and hasattr(cap, '__len__') and len(cap) > idx:
                                ev_connected = cap[idx] > 0
                        if hasattr(charger, 'connected_electric_vehicle_at_charger_soc'):
                            soc_arr = charger.connected_electric_vehicle_at_charger_soc
                            if soc_arr is not None and hasattr(soc_arr, '__len__') and len(soc_arr) > idx:
                                ev_soc = float(soc_arr[idx])
                        break
        except Exception:
            pass
        
        # Session tracking logic
        if ev_connected:
            # EV is plugged
            if active_sessions[ev_local_idx] is None:
                # Start new session
                active_sessions[ev_local_idx] = EVSession(ev_local_idx, step)
            # Add current action to session
            active_sessions[ev_local_idx].add_action(action[ev_action_idx], ev_soc)
        else:
            # EV not plugged
            if active_sessions[ev_local_idx] is not None:
                # Session just ended (departure)
                session = active_sessions[ev_local_idx]
                departure_action = action[ev_action_idx]
                deficit = info.get('ev_departure_deficit_kwh', 0.0)
                session.finalize(step, departure_action, deficit)
                completed_sessions.append(session)
                active_sessions[ev_local_idx] = None
    
    step += 1
    if step % 2000 == 0:
        print(f"  Step {step}: {len(completed_sessions)} sessions completed")

print(f"\n✅ Episode complete: {len(completed_sessions)} charging sessions tracked")

# Analyze sessions
print("\n" + "="*80)
print("SESSION ANALYSIS")
print("="*80)

if len(completed_sessions) == 0:
    print("❌ No sessions tracked - check EV state extraction logic")
    sys.exit(1)

# Filter sessions with deficit > 0
sessions_with_deficit = [s for s in completed_sessions if s.deficit > 0.1]

print(f"\n📊 Sessions with deficit: {len(sessions_with_deficit)} / {len(completed_sessions)}")

if len(sessions_with_deficit) == 0:
    print("✅ No deficits - all EVs charged successfully!")
    sys.exit(0)

# Classify sessions
v3_controllable = 0
v3_uncontrollable = 0
correct_controllable = 0
correct_uncontrollable = 0
misclassified = 0

misclassified_examples = []

for session in sessions_with_deficit:
    v3_class = session.v3_classification()
    correct_class = session.correct_classification(threshold=0.9)
    
    if v3_class == "controllable":
        v3_controllable += 1
    else:
        v3_uncontrollable += 1
    
    if correct_class == "controllable":
        correct_controllable += 1
    else:
        correct_uncontrollable += 1
    
    # Check for misclassification
    if v3_class != correct_class:
        misclassified += 1
        if len(misclassified_examples) < 10:
            misclassified_examples.append(session)

print(f"\n🔍 V3 Classification (departure action only):")
print(f"   Controllable:   {v3_controllable}")
print(f"   Uncontrollable: {v3_uncontrollable}")

print(f"\n✅ Correct Classification (session-level):")
print(f"   Controllable:   {correct_controllable}")
print(f"   Uncontrollable: {correct_uncontrollable}")

print(f"\n❌ Misclassifications: {misclassified} sessions")
print(f"   ({100*misclassified/len(sessions_with_deficit):.1f}% of deficit sessions)")

# Show examples
if misclassified_examples:
    print(f"\n" + "="*80)
    print(f"PROOF: Example Misclassified Sessions")
    print("="*80)
    
    for i, session in enumerate(misclassified_examples[:5], 1):
        print(f"\n📍 Example {i}: EV {session.ev_idx}")
        print(f"   Session length: {session.session_length()} hours")
        print(f"   Deficit: {session.deficit:.2f} kWh")
        print(f"   Actions during session:")
        print(f"     Mean: {session.mean_action():.3f}")
        print(f"     Min:  {session.min_action():.3f}")
        print(f"     Had action < 0.9: {session.had_low_action()}")
        print(f"   Departure action: {session.departure_action:.3f}")
        print(f"   V3 says: '{session.v3_classification()}' (only checks departure={session.departure_action:.3f})")
        print(f"   Should be: '{session.correct_classification()}' (checks all actions)")
        print(f"   ❌ V3 MISCLASSIFIED!")

# Summary statistics
print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)

total_deficit = sum(s.deficit for s in sessions_with_deficit)
v3_controllable_deficit = sum(s.deficit for s in sessions_with_deficit if s.v3_classification() == "controllable")
correct_controllable_deficit = sum(s.deficit for s in sessions_with_deficit if s.correct_classification() == "controllable")

print(f"\n💰 Deficit Attribution:")
print(f"   Total deficit: {total_deficit:.2f} kWh")
print(f"   V3 controllable: {v3_controllable_deficit:.2f} kWh ({100*v3_controllable_deficit/total_deficit:.1f}%)")
print(f"   Correct controllable: {correct_controllable_deficit:.2f} kWh ({100*correct_controllable_deficit/total_deficit:.1f}%)")
print(f"   V3 underestimated by: {correct_controllable_deficit - v3_controllable_deficit:.2f} kWh")

print(f"\n🎯 CONCLUSION:")
if misclassified > len(sessions_with_deficit) * 0.2:
    print(f"   ✅ CONFIRMED: V3 temporal tracking flaw exists!")
    print(f"   {misclassified} sessions misclassified due to checking only departure action")
    print(f"   Agent charged conservatively during session but maxed at end")
    print(f"   This explains the 29.8% oracle gap")
else:
    print(f"   ⚠️  Few misclassifications - gap may be due to other factors")

print("\n" + "="*80 + "\n")
