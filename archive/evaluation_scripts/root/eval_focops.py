import os, sys, torch, torch.nn as nn
sys.path.insert(0, os.getcwd())
from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.adapters import SingleAgentListAdapter
from citylearn.wrappers import NormalizedObservationWrapper

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[512, 512, 256]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.mean = nn.Sequential(*layers, nn.Linear(prev, act_dim))
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    def forward(self, obs):
        return torch.tanh(self.mean(obs))

schema = os.environ.get("CITYLEARN_SCHEMA")
base = CityLearnEnv(schema=schema, central_agent=True)
base = NormalizedObservationWrapper(base)
env = SingleAgentListAdapter(base)
env = CityLearnSafetyEnvV3(env)

ckpt = torch.load("runs/focops_P95_V2/FOCOPS-{CityLearnSafety-SoC-v0}/seed-000-2026-02-05-03-18-10/torch_save/epoch-100.pt", map_location='cpu')
obs, _ = env.reset(seed=42)

actor = GaussianActor(len(obs), env.action_space.shape[0])
actor.load_state_dict(ckpt['pi'])
actor.eval()

ep_reward, ep_cost, steps = 0.0, 0.0, 0
ev_departures, ev_violations = 0, 0
grid_viol, battery_viol, building_viol = 0, 0, 0
cost_ev, cost_grid, cost_battery, cost_building = 0.0, 0.0, 0.0, 0.0

print("\n" + "="*80)
print("EVALUATING: FOCOPS_P95_V2")
print("="*80 + "\n")

while True:
    with torch.no_grad():
        action = actor(torch.FloatTensor(obs).unsqueeze(0)).squeeze(0).numpy()
    obs, reward, terminated, truncated, info = env.step(action)
    
    steps += 1
    ep_reward += reward
    ep_cost += info.get('cost', 0)
    
    ev_dep = int(info.get('ev_departure_departures', 0))
    ev_deficit = float(info.get('ev_departure_deficit_kwh', 0))
    if ev_dep > 0:
        ev_departures += ev_dep
        if ev_deficit > 0.01:
            ev_violations += 1
    
    if float(info.get('cost_stems_grid_power', 0)) > 0:
        grid_viol += 1
    battery_viol += int(info.get('battery_soc_violation_count', 0))
    building_viol += int(info.get('building_power_violation_count', 0))
    
    cost_ev += float(info.get('cost_ev_departure', 0))
    cost_grid += float(info.get('cost_stems_grid_power', 0))
    cost_battery += float(info.get('cost_stems_battery', 0))
    cost_building += float(info.get('cost_stems_building_power', 0))
    
    if terminated or truncated:
        break

env.close()

print("\n" + "="*80)
print("FOCOPS_P95_V2 RESULTS")
print("="*80)
print(f"Reward: {ep_reward:.2f}")
print(f"Cost:   {ep_cost:.2f} / 45000")
print()
print("VIOLATIONS:")
ev_rate = 100 * ev_violations / max(ev_departures, 1)
print(f"  EV:       {ev_violations:>5}/{ev_departures:<5} = {ev_rate:>6.2f}%")
grid_rate = 100 * grid_viol / steps
print(f"  Grid:     {grid_viol:>5}/{steps:<5} = {grid_rate:>6.2f}%")
battery_rate = 100 * battery_viol / (steps * 17)
print(f"  Battery:  {battery_viol:>5}/{steps*17:<6} = {battery_rate:>6.2f}%")
building_rate = 100 * building_viol / (steps * 17)
print(f"  Building: {building_viol:>5}/{steps*17:<6} = {building_rate:>6.2f}%")
print("="*80 + "\n")
