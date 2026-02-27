
1) What is the problem with your “EV + temperature control” dataset right now?

Even though the folder name says WITH_TEMP_CONTROL, the actual Building_1.csv (and likely the others) shows:

indoor_dry_bulb_temperature = NaN for all 8760 rows

indoor_relative_humidity = NaN for all 8760 rows

average_unmet_cooling_setpoint_difference = NaN for all 8760 rows

cooling_demand = 0 for all 8760 rows

heating_demand = 0 for all 8760 rows

dhw_demand = 0 for all 8760 rows

So the dataset has no indoor temperature state and no HVAC thermal load.

Why that’s fatal for “temperature control”

Temperature control in CityLearn needs two things:

A temperature state that exists and evolves (e.g., indoor_dry_bulb_temperature)

A thermal/HVAC mechanism that can change that state (e.g., nonzero cooling/heating demand served by your HVAC action)

Your dataset has neither:

The temperature columns exist but are empty (NaN) → there is no state to control.

Cooling/heating demands are all zero → even if you take HVAC actions, there is no “cooling/heating job” to do, so actions cannot affect temperature.

So: temperature control cannot work, even if you enable temperature observations and HVAC actions in the schema.

2) What happened in the “non-EV dataset approach” we discussed?

We compared your “bad” dataset with another dataset snippet you pasted (the “good” one). That “good” dataset had:

real numeric indoor_dry_bulb_temperature (not NaN)

setpoints like indoor_dry_bulb_temperature_cooling_set_point

nonzero cooling_demand at many hours

often additional helpful signals like occupant_count, indoor_relative_humidity

That “good” dataset is the kind of dataset where:

indoor temperature can be computed/updated,

comfort/unmet setpoint can be calculated,

HVAC actions can actually change outcomes.

So the “approach we tried” was basically:

If the dataset contains temperature + loads, then enabling temperature control is meaningful.

If the dataset contains NaNs/zeros, enabling temperature control is only “cosmetic” (it changes spaces but not physics).

3) LSTM dynamics model: what data does it require, and what do you have vs not have?

To make indoor temperature update based on actions, CityLearn usually uses a building dynamics model (often an LSTM) that predicts next indoor temperature from recent history.

What an LSTM dynamics model typically needs

It needs a window of past inputs (lookback), including some combination of:

Indoor / comfort state inputs

indoor_dry_bulb_temperature (required; otherwise the model has nothing to predict from)

Weather boundary inputs

outdoor_dry_bulb_temperature, solar irradiance, humidity, etc.

HVAC-related “cause” inputs

cooling_demand and/or heating_demand

often hvac_mode

often setpoints (cooling_set_point, heating_set_point)

Plus: normalization min/max arrays that match those inputs, and the trained model weights file (.pth) if you’re using the built-in LSTM approach.

What you currently have (EV dataset)

You have weather observations configured in schema ✅ (good)

But in the building CSV:

indoor_dry_bulb_temperature is all NaN ❌

cooling_demand and heating_demand are all zero ❌

no setpoint columns ❌

no hvac_mode column ❌

That means:

You cannot train or run an LSTM dynamics model properly, because the key signals (indoor temp and HVAC demand) are missing.

Even if you force-fill indoor temperature with a constant, with HVAC demand = 0, the model has no meaningful relationship between actions and temperature.

In short: LSTM temperature dynamics requires exactly the data your EV dataset is missing.

4) What happens if you add a CMDP cost for temperature control anyway?

A CMDP cost needs a well-defined constraint signal. Example:
“Penalty if indoor temperature > cooling setpoint” or “unmet setpoint difference”.

But in your dataset:

indoor_dry_bulb_temperature = NaN → you can’t compute “too hot / too cold”.

average_unmet_cooling_setpoint_difference = NaN → the cost signal itself is missing.

Setpoints are missing → even if temperature existed, you don’t know the target.

HVAC demand is always zero → actions cannot improve the cost.

So CMDP cost leads to one of these bad outcomes

Cost becomes NaN → training breaks or produces nonsense.

You replace NaNs with 0 → cost becomes always 0 → constraint is meaningless.

You fill indoor temp with a constant → cost becomes constant → agent can’t affect it.

You compute cost from weather instead of indoor temp → you’re not controlling “comfort”, you’re controlling nothing relevant.

So yes, you can add a cost term in code, but it will not represent real “temperature safety/comfort” unless temperature dynamics and setpoints exist.

5) “But I updated action space and observation space—why doesn’t it help?”

This is the most important concept:

Observation space ≠ actual state dynamics

Turning on an observation in the schema only means:

“CityLearn will try to output this value.”

If the underlying data is NaN or missing:

The observation is NaN / invalid, or you end up filling it artificially.

Action space ≠ actual controllability

Turning on an action only means:

“The agent can output a control value.”

But for the control to matter, the environment must have:

a device or mechanism linked to that action, and

a state transition / physics model where that action changes the next state.

In your dataset:

Temperature state is missing (NaN)

HVAC demand is zero
So even if the agent outputs HVAC actions, there is no temperature state transition to respond to those actions.

Bottom line: changing action/observation space is like adding steering wheel + dashboard, but the car has no engine and no road model.

Minimal conclusion and what you must do next

To make temperature-control CMDP meaningful, you need at least one of these:

Option 1 (recommended)

Use a dataset that already includes:

indoor_dry_bulb_temperature (numeric)

setpoints (cooling/heating)

nonzero cooling_demand / heating_demand

(ideally) hvac_mode

Option 2

If you must stay with this EV dataset:

synthesize/fill indoor temperature,

generate realistic cooling/heating demand (or build your own RC thermal model),

define setpoints and hvac_mode,

then implement/attach a real dynamics model so actions actually change indoor temperature.

Only after that does it make sense to:

add CMDP comfort costs,

enforce safety constraints,

evaluate tradeoffs between energy, EV objectives, and comfort.