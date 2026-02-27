
✅ Permanent source-code changes in CityLearn repo (what you actually edited in /home/extra-storage/THESIS/CityLearn)

🧪 Runtime-only experimental patches (the monkeypatch snippets you ran to prove controllability; not yet committed as source changes unless you copied them into the repo)

That distinction matters a lot for thesis credibility.

1) Thesis objective and system definition
1.1 Objective

You turned CityLearn from a “replay loads from CSV” benchmark into a true control environment where:

Indoor temperature is a state that evolves over time.

HVAC actions causally affect indoor temperature.

Comfort constraints become meaningful as a safety signal for Safe RL.

EV actions and EV system remain intact and controllable under a central agent.

1.2 Non-negotiable design constraints

Keep EV subsystems intact (actions + observations).

Use central_agent=True and a flat action vector.

Maintain backward compatibility: schemas without "dynamics" should still run unchanged.

2) Repo and execution context
2.1 Environment

Conda env: citylearn

Python: 3.10

2.2 Active edited CityLearn repo

/home/extra-storage/THESIS/CityLearn

2.3 Safe fork used for datasets/schemas

/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork

2.4 Schema selection mechanism

Schemas are selected via an environment variable:

export CITYLEARN_SCHEMA="/path/to/schema.json"

3) The root problems in baseline CityLearn

This is important in your thesis: explain why you modified the environment.

D1) Default CityLearn doesn’t control indoor temperature

By default, CityLearn replays “ideal” temperature and demand series from CSV files. Indoor temperature did not respond to HVAC actions, making comfort constraints meaningless.

D2) DynamicsBuilding existed but wasn’t instantiated automatically

Even if a schema included a dynamics block, buildings were still instantiated as the normal Building class, so the dynamics were unused.

D3) EV observation min/max estimation crashed at init (KeyError)

EV-related observation keys (e.g., required_soc_departure) exist in schema, but the min/max estimator incorrectly assumed these columns existed in Building_*.csv. In reality, those columns live in EV CSVs.

D4) EV actions were not exposed in the central agent action vector

EV chargers actions existed in schema, but the central agent’s flat action vector missed those dimensions.

D5) Cooling action was silently ignored (key mismatch)

Action parsing produced cooling_device_action, but the building expected cooling_device. Result: cooling action remained zero and had no effect.

4) Permanent source-code changes (what you actually changed in CityLearn)

Below is the clean “change log” you can put in your thesis Appendix.

4.1 citylearn/citylearn.py — instantiate DynamicsBuilding when schema has dynamics

File:
/home/extra-storage/THESIS/CityLearn/citylearn/citylearn.py

Problem solved: Buildings with "dynamics" in schema were not using the dynamics.

Change: In CityLearnEnv._load_building() after creating dynamics, you added logic:

If building_kwargs["dynamics"] exists → instantiate DynamicsBuilding.

Effect: Any schema with "dynamics" now produces a Dynamics-enabled building automatically.

✅ Verification: You confirmed buildings became DynamicsBuilding in CS6 RC schema runs.

4.2 citylearn/dynamics.py — added 1R1C thermal model (RCDynamics)

File:
/home/extra-storage/THESIS/CityLearn/citylearn/dynamics.py

Change: Added a new dynamics class implementing 1R1C thermal update:

Parameters in schema: dt_seconds, R_K_per_kW, C_kWh_per_K, hvac_max_kW, initial_indoor_temperature.

Uses outdoor_dry_bulb_temperature from obs.

Adds clamps: min/max indoor temperature.

Effect: Indoor temperature becomes a state updated by physics-inspired dynamics.

✅ Verification: In CS6, u=0 and u=1 produced different Tin trajectories (not CSV replay).

4.3 citylearn/building.py — implement indoor temperature update for DynamicsBuilding

File:
/home/extra-storage/THESIS/CityLearn/citylearn/building.py

Problem: DynamicsBuilding.update_indoor_dry_bulb_temperature() used to raise NotImplementedError.

Change: Replaced stub with:

read Tin and Tout

read stored cooling_device_action

call self.dynamics.step(obs, u_cool)

write result into writable indoor temperature state

✅ Verification: Env could step without NotImplementedError, and Tin changed under action.

4.4 citylearn/building.py — EV observation min/max estimator fix

File:
/home/extra-storage/THESIS/CityLearn/citylearn/building.py

Problem: EV keys in schema caused KeyError during min/max estimation because estimator looked only in Building CSV.

Change: In observation min/max estimation:

If key missing from building CSV → load EV CSV or use placeholder series.

✅ Verification: CS6 env initialization + reset succeeded without EV KeyError.

4.5 citylearn/citylearn.py — enable EV actions in central flat action vector

File:
/home/extra-storage/THESIS/CityLearn/citylearn/citylearn.py

Problem: EV action dims not present for central agent.

Change: In charger loop inside _load_building():

detect EV charger action metadata and mark it active so it’s included in action_metadata.

Effect: Action vector dimension increased (example: 34 → 44).

✅ Verification: _parse_actions produced EV mapping, and EV actions were controllable.

4.6 citylearn/building.py — fix action key mismatch (*_action → base key)

File:
/home/extra-storage/THESIS/CityLearn/citylearn/building.py

Problem: Parser produced cooling_device_action, apply_actions expected cooling_device.

Change: At top of Building.apply_actions(**kwargs):

For each key that ends with _action, also create base key if missing.

Effect: cooling_device_action is no longer ignored.

✅ Verification: Cooling action stored as non-zero, and Tin differs for u=0 vs u=1.

4.7 citylearn/citylearn.py + building.py — normalize / alias SimBuild dataset fields

Problem: SimBuild dataset uses “human readable” CSV columns and different schema dynamics format.

Changes included:

Snake_case rename / alias in loader for EnergySimulation, Weather, CarbonIntensity, Pricing.

Filter unexpected columns to match constructors.

random_seed default when None.

In Building.observations(): changed observations[k] to observations.get(k, 0.0) to avoid KeyErrors for missing predicted weather columns.

✅ Verification: SimBuild schema can load and reset.

4.8 citylearn/citylearn.py — prevent DynamicsBuilding override for SimBuild multi-dynamics format

Problem: You forced LSTMDynamicsBuilding for SimBuild, but later code overwrote constructor back to DynamicsBuilding.

Change: Added simbuild_multi_dyn flag:

If schema uses multi-block dynamics format (dyn['cooling']) → set flag.

Skip the generic “if dynamics exists use DynamicsBuilding” override when flag True.

Ensure SimBuild uses LSTMDynamicsBuilding.

✅ Verification: SimBuild loads with:

Building0 class: LSTMDynamicsBuilding

Building0 dynamics: LSTMDynamics

5) Runtime-only experimental patches (used for proof, not yet committed)

These were critical to prove action → Tin coupling for SimBuild+LSTM.

5.1 Fix: SimBuild “Cooling Load (kWh)” wasn’t mapped into CityLearn cooling demand fields

Observation:
Raw Building_1.csv has Cooling Load (kWh) with non-zero values, but energy_simulation.cooling_demand_without_control was all zeros.

Runtime patch:
Injected Cooling Load (kWh) into energy_simulation.cooling_demand_without_control.

✅ Verified: full cooling series had max ~10.6 kWh at timestep 400.

5.2 Fix: partial-load coupling (action → controlled cooling demand)

You forced:

𝑐
𝑜
𝑜
𝑙
𝑖
𝑛
𝑔
_
𝑑
𝑒
𝑚
𝑎
𝑛
𝑑
[
𝑡
]
=
𝑢
[
𝑡
]
⋅
𝑐
𝑜
𝑜
𝑙
𝑖
𝑛
𝑔
_
𝑑
𝑒
𝑚
𝑎
𝑛
𝑑
_
𝑤
𝑖
𝑡
ℎ
𝑜
𝑢
𝑡
_
𝑐
𝑜
𝑛
𝑡
𝑟
𝑜
𝑙
[
𝑡
]
cooling_demand[t]=u[t]⋅cooling_demand_without_control[t]

This matches the “partial-load coupling” logic the paper implies.

✅ Verified: OFF vs ON creates different controlled demand.

5.3 Fix: LSTM input buffer wasn’t reading the controlled cooling demand

Even after setting cooling_demand[t], LSTM input assembled_last stayed 0.0. So you force-wrote:

LSTM input feature "cooling_demand" in _model_input[ix][-1] using normalized controlled demand.

✅ Verified:

OFF assembled_last = 0.0

ON assembled_last = 0.837...

5.4 Final proof: causal A/B test, same history, different action

You ran two identical envs, warmed to t*=400, then:

OFF: u=0

ON: u=1

Result (from your own output):

ctrl_cool_used: 0.0 vs 10.6085

Tin_used: 27.4686 vs 17.6309

ΔTin = -9.84°C immediately

ΔTin remains non-zero for 10 subsequent steps

✅ This is the strongest possible evidence of controllability.

6) How to explain the “Tin[t] identical but Tin[t-1] differs” effect

This must be explained in thesis to avoid confusion.

What is happening

After env.step():

CityLearn increments time_step to t_after.

Your LSTM update writes the new predicted temperature into the series index corresponding to the just-updated transition, which you observed as t_used = t_after - 1.

So the “fresh updated temperature” right after stepping is:

Tin[t_after - 1]

That’s why:

Tin[t_after] can look identical initially

while Tin[t_after - 1] differs causally due to action

Thesis wording (use this)

“Due to the environment update order, the LSTM-predicted indoor temperature is written to the time-index corresponding to the completed transition. Therefore, immediately after env.step(), the action-dependent temperature update is observed at T_in[t-1] relative to the updated time_step. This is an indexing artifact of the simulation loop rather than a lack of controllability.”

7) How to put this into your thesis (recommended structure)
7.1 Main Chapters
Chapter: Problem Setup / Environment

Describe CityLearn baseline behavior (CSV replay).

Explain why comfort constraints are meaningless without controlled Tin.

State your thesis need: Safe RL requires meaningful safety signals.

Chapter: Environment Modifications

Split into subsections:

Dynamics integration

RC model implementation

LSTM pipeline enabling

EV system preservation

Robust dataset loading + backward compatibility

Chapter: Validation

Use your A/B tests:

show action vector includes HVAC and EV (if EV integrated)

show Tin changes under action (causal test)

show controlled demand changes under action (partial load coupling)

mention indexing artifact and how you measure Tin consistently

7.2 Appendices (very important)
Appendix A: Code Change Log

Include a table like this:

File	Function/Area	Change	Purpose
citylearn.py	_load_building	Instantiate DynamicsBuilding when schema has dynamics	Activate dynamics automatically
dynamics.py	RCDynamics	1R1C model	Controllable Tin via physics
building.py	DynamicsBuilding.update_indoor...	implement step() call	update Tin each step
building.py	apply_actions	map *_action keys	fix cooling action ignored
building.py	obs min/max	EV keys fallback	prevent init crash
citylearn.py	EV action metadata	include EV dims	central agent controls EV
citylearn.py/building.py	SimBuild normalization	alias/fallback	load external datasets
citylearn.py	simbuild_multi_dyn	force LSTMDynamicsBuilding	match paper architecture
Appendix B: Reproducibility

exact schema paths used

env variables

commit hash (if you have it)

commands to run your A/B test script

8) What to do next to “make it thesis-clean”

Right now, some key fixes exist as runtime patches. For thesis, you’ll want them as permanent code or a documented “preprocessing layer”.

Best practice for thesis

Implement those runtime patch concepts as a deterministic preprocessing step:

Map Cooling Load (kWh) → cooling_demand_without_control inside the SimBuild loader normalization.

In LSTMDynamics input update: ensure "cooling_demand" uses controlled demand (not zero).

Document it as “dataset adapter + control coupling layer”.