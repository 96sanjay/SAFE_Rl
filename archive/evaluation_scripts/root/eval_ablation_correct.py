"""Correct eval of all ablation runs using CityLearnCMDPv2 + fixed weight mapping."""
import os, sys, torch, numpy as np, glob, subprocess

runs = [
    ('Abl-0 Single-lam OLD', 'run_ablation_0_single_old.sh', 'runs/ablation_0_single_old'),
    ('Abl-1 Single-lam Reform', 'run_ablation_1_single_reformed.sh', 'runs/ablation_1_single_reformed'),
    ('Abl-C Multi noSaute', 'run_ablation_C_nosaute.sh', 'runs/ablation_C_nosaute'),
    ('Abl-B Multi C2tight', 'run_ablation_B_c2tight.sh', 'runs/ablation_B_c2tight'),
    ('r25b Saute ON', 'run_r25b_ev_slack_arb.sh', 'runs/r25b_ev_slack_arb/5bld'),
]

for name, script, run_dir in runs:
    # Find latest checkpoint
    ckpts = sorted(glob.glob(f'{run_dir}/PPO*/seed-*/torch_save/epoch-*.pt'),
                   key=lambda x: int(x.split('epoch-')[1].split('.')[0]))
    if not ckpts:
        print(f'{name}: NO CHECKPOINT')
        continue
    ckpt_path = ckpts[-1]
    epoch = ckpt_path.split('epoch-')[1].split('.')[0]
    
    print(f'\n=== {name} (epoch-{epoch}) ===')
    
    # Run as separate process to ensure clean env
    code = f'''
import os, torch, numpy as np, glob

for k in list(os.environ.keys()):
    if k.startswith('STEMS_') or k.startswith('CITYLEARN_') or k.startswith('COST_W'):
        del os.environ[k]

for line in open('{script}'):
    line = line.strip()
    if line.startswith('export '):
        parts = line.split('#')[0].strip()[7:].split('=', 1)
        if len(parts) == 2:
            k, v = parts[0].strip(), parts[1].strip('"').strip("'")
            if '$' not in v and '{{' not in v and v:
                os.environ[k] = v

os.environ['CITYLEARN_SCHEMA'] = os.path.join(os.getcwd(), 'data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json')
os.environ['CITYLEARN_CENTRAL_AGENT'] = '1'
os.environ['CITYLEARN_REWARD_TYPE'] = 'stems'

import citylearn_safe.omni_env, citylearn_safe.omni_env_v2
from citylearn_safe.omni_env_v2 import CityLearnCMDPv2

np.random.seed(42); torch.manual_seed(42)
env = CityLearnCMDPv2('CityLearnSafety-V2G-v2')
obs, info = env.reset()
obs = obs.numpy().ravel()

ckpt = torch.load('{ckpt_path}', map_location='cpu')
first_w = [k for k in ckpt['pi'] if '0.weight' in k and 'log_std' not in k][0]
if ckpt['pi'][first_w].shape[1] != len(obs):
    print(f'DIM MISMATCH: actor={{ckpt["pi"][first_w].shape[1]}} env={{len(obs)}}')
    exit(1)

class Actor(torch.nn.Module):
    def __init__(s, od, ad):
        super().__init__()
        s.net = torch.nn.Sequential(torch.nn.Linear(od,256),torch.nn.Tanh(),torch.nn.Linear(256,256),torch.nn.Tanh(),torch.nn.Linear(256,ad))
    def forward(s, x): return s.net(x)

actor = Actor(len(obs), 9)
st = {{k.replace('mean.', 'net.'): v for k, v in ckpt['pi'].items() if 'log_std' not in k}}
actor.load_state_dict(st, strict=True)
actor.eval()

w_sum = actor.net[0].weight.data.sum().item()
ckpt_sum = ckpt['pi'][first_w].sum().item()
assert abs(w_sum - ckpt_sum) < 0.001, f'Weight mismatch: {{w_sum}} vs {{ckpt_sum}}'

norm = ckpt.get('obs_normalizer', {{}})
mn = np.array(norm.get('mean', np.zeros(len(obs)))).ravel()[:len(obs)]
vr = np.array(norm.get('var', np.ones(len(obs)))).ravel()[:len(obs)]
if hasattr(mn,'numpy'): mn=mn.numpy()
if hasattr(vr,'numpy'): vr=vr.numpy()

city = None
inner = env
for _ in range(15):
    for attr in ['_city','_env','env','base']:
        i2 = getattr(inner, attr, None)
        if i2 is not None:
            if hasattr(i2, 'buildings'): city=i2; break
            inner=i2
    if city: break
if city is None and hasattr(env,'_get_citylearn'): city=env._get_citylearn()

chargers = []
for bi, b in enumerate(city.buildings):
    for ch in b.electric_vehicle_chargers:
        sim = getattr(ch,'charger_simulation',getattr(ch,'_Charger__charger_simulation',None))
        chargers.append((bi,ch,sim))

batt_idx=[0,3,4,5,7]; ev_idx=[1,6,8]
obs,_=env.reset(); obs=obs.numpy().ravel()

deps=[]; c2v=0; c3v=0; c4v=0; bc=0; bd=0; bt=0; ec=0; ev2g=0; et=0
imp_t=0; exp_t=0; ramp_t=0; prev_g=0; peak_nec=0

for step in range(8759):
    on=np.clip((obs-mn)/np.sqrt(vr+1e-8),-5,5)
    with torch.no_grad():
        act=actor(torch.as_tensor(on,dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    t_pre=int(city.time_step)
    for bi,ch,sim in chargers:
        sa=np.asarray(sim._electric_vehicle_charger_state,dtype=float)
        da=np.asarray(sim._electric_vehicle_departure_time,dtype=float)
        ra=np.asarray(sim._electric_vehicle_required_soc_departure,dtype=float)
        if t_pre<len(sa) and float(sa[t_pre])==1.0 and float(da[t_pre])==0.0:
            evv=ch.connected_electric_vehicle
            soc=0.0
            if evv is not None and evv.battery is not None:
                sarr=np.asarray(evv.battery.soc,dtype=float)
                ti=max(0,t_pre-1)
                if ti<len(sarr): soc=float(sarr[ti])
            deps.append((soc,float(ra[t_pre]),bi))
    result=env.step(torch.as_tensor(act,dtype=torch.float32))
    obs=result[0].numpy().ravel()
    t=int(city.time_step); ti=max(0,t-1)
    for bi2 in batt_idx:
        bt+=1
        if act[bi2]>0.1: bc+=1
        elif act[bi2]<-0.1: bd+=1
    for ei in ev_idx:
        et+=1
        if act[ei]>0.1: ec+=1
        elif act[ei]<-0.1: ev2g+=1
    gnec=0
    for b in city.buildings:
        es=getattr(b,'electrical_storage',None)
        if es:
            sa2=np.asarray(es.soc,dtype=float)
            if ti<len(sa2) and float(sa2[ti])>0.95: c2v+=1
        nec=getattr(b,'net_electricity_consumption',None)
        if nec is not None and hasattr(nec,'__len__') and len(nec)>ti:
            nv=float(nec[ti])
            if abs(nv)>4.6083: c3v+=1
            gnec+=nv
    if abs(gnec)>10.2352: c4v+=1
    if abs(gnec)>peak_nec: peak_nec=abs(gnec)
    imp_t+=max(0,gnec); exp_t+=max(0,-gnec)
    ramp_t+=abs(gnec-prev_g); prev_g=gnec
    if bool(result[3]) or bool(result[4]): break

nd=len(deps); nv=sum(1 for s,r,_ in deps if s<r-0.001); tbs=8759*5
ms=np.mean([s for s,_,_ in deps]) if deps else 0
print(f'C0={{nv}}/{{nd}} ({{nv/max(1,nd)*100:.1f}}%) dep_soc={{ms:.4f}}')
print(f'C2={{c2v}}/{{tbs}} ({{c2v/tbs*100:.1f}}%)')
print(f'C3={{c3v}}/{{tbs}} ({{c3v/tbs*100:.1f}}%)')
print(f'C4={{c4v}}/8759 ({{c4v/8759*100:.1f}}%)')
print(f'Batt: chg={{bc/bt*100:.1f}}% dis={{bd/bt*100:.1f}}%')
print(f'EV: chg={{ec/et*100:.1f}}% v2g={{ev2g/et*100:.1f}}%')
print(f'Import={{imp_t:.0f}} Export={{exp_t:.0f}} Peak={{peak_nec:.1f}} Ramp={{ramp_t:.0f}}')
'''
    
    result = subprocess.run(
        ['/home/christmas/miniconda3/envs/citylearn/bin/python', '-c', code],
        capture_output=True, text=True, timeout=300, cwd=os.getcwd()
    )
    
    # Extract just the metric lines
    for line in result.stdout.split('\n'):
        if any(x in line for x in ['C0=', 'C2=', 'C3=', 'C4=', 'Batt:', 'EV:', 'Import=', 'DIM MISMATCH']):
            print(f'  {line}')
    
    if result.returncode != 0:
        for line in result.stderr.split('\n')[-5:]:
            if line.strip(): print(f'  ERR: {line}')

