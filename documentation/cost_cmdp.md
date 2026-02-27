Operational Safety Cost 1: Grid Peak Import (Capacity / Demand-Limit Safety)
1) What it means (plain language)

This cost penalizes the agent only when the district imports “too much” power from the grid in a single timestep.

Think: the grid connection has a safe limit:

transformer / feeder thermal capacity

contractual demand limit (e.g., “don’t exceed 200 kW”)

operational safety margin

If the agent tries to pull more than that, you add a penalty.

2) Which signal to use (IMPORTANT)

Use grid import, not net.

Net can be negative (export), and exporting doesn’t stress import capacity.

Import is the actual load the upstream grid must deliver.

You already compute:

grid_import_kwh = max(step_net_consumption_kwh, 0.0)

Units clarification (dt = 1 hour)

Because CityLearn timestep is 1 hour:

grid_import_kwh is energy over the hour (kWh)

but numerically it’s also the average kW during that hour
So you can treat it as “kW” for demand-limit logic.

3) Why it helps CityLearn KPIs

This directly improves “peak” style KPIs because you’re explicitly discouraging peaks.

Most direct:

citylearn_daily_peak_average

citylearn_all_time_peak_average

Often also improves:

citylearn_cost_total (realistic pricing tends to punish peaks)

sometimes citylearn_ramping_average (less spiky behavior)

4) Cost function (hinge / soft constraint)

Let:

𝑃
imp
(
𝑡
)
=
P
imp
	​

(t)= grid_import_kwh[t]

𝑃
th
=
P
th
	​

= peak threshold (capacity limit)

Raw violation:

𝑐
peak
(
𝑡
)
=
max
⁡
(
0
,
𝑃
imp
(
𝑡
)
−
𝑃
th
)
c
peak
	​

(t)=max(0,P
imp
	​

(t)−P
th
	​

)

Weighted cost:

𝑐
~
peak
(
𝑡
)
=
𝑤
peak
⋅
𝑐
peak
(
𝑡
)
c
~
peak
	​

(t)=w
peak
	​

⋅c
peak
	​

(t)
Interpretation:

If import is below threshold → cost = 0

If import exceeds threshold → cost grows linearly with the amount above threshold

This is exactly the behavior you want for safe RL: allow minor exploration, but punish bigger violations more.

5) Step-by-step wrapper integration

You already have kpis["grid_import_kwh"]. In step() after KPI computation:

p_import = float(kpis.get("grid_import_kwh", 0.0))
cost_grid_peak_raw = max(0.0, p_import - self.peak_threshold)
cost_grid_peak = self.w_grid_peak * cost_grid_peak_raw

6) What to log (so you can debug + plot)

Log both weighted and raw:

cost_grid_peak (weighted)

cost_grid_peak_raw (raw violation)

grid_peak_violation (0/1 flag)

Example:

info["cost_grid_peak"] = float(cost_grid_peak)
info["cost_grid_peak_raw"] = float(cost_grid_peak_raw)
info["grid_peak_violation"] = 1.0 if cost_grid_peak_raw > 0 else 0.0

7) How to set a good peak threshold (this is crucial)

If threshold is too low → violation every step → Lagrange explodes → learning dies.

Best practical method:

use your Greedy RBC baseline CSV

compute percentile of grid_import_kwh

Recommended start:

𝑃
th
=
percentile
(
import
,
95
)
P
th
	​

=percentile(import,95)
or safer (fewer violations):

𝑃
th
=
percentile
(
import
,
97
)
P
th
	​

=percentile(import,97)

This means: “Only punish the worst 3–5% of peaks that RBC already produces.”

Operational Safety Cost 2: Grid Ramping (Smoothness / Grid Stability)
1) What it means (plain language)

This cost penalizes sudden changes in the district’s grid signal between consecutive timesteps.

Real-world meaning:

large swings require balancing reserves

can stress voltage/frequency stability

increase operational difficulty for the grid operator

So we want the district net load to change smoothly.

2) Which signal to ramp on

Recommended: ramp on district net signal (signed):

𝑃
net
(
𝑡
)
=
step_net_consumption_kwh
[
𝑡
]
P
net
	​

(t)=step_net_consumption_kwh[t]

positive = importing

negative = exporting

This matches what the grid “sees” as net load.

Alternative (less ideal but valid):

ramp on import-only (always ≥ 0)

But net is usually better because exports can also cause grid instability if they swing wildly.

3) Why it helps CityLearn KPIs

Directly improves:

citylearn_ramping_average

Often indirectly improves:

peaks (sharp ramps often create spikes)

cost/carbon slightly (smoother schedules often mean less panic importing)

4) Cost function (hinge on delta)

Let:

𝑃
(
𝑡
)
P(t) be chosen ramp signal (net or import)

Δ
𝑃
(
𝑡
)
=
∣
𝑃
(
𝑡
)
−
𝑃
(
𝑡
−
1
)
∣
ΔP(t)=∣P(t)−P(t−1)∣

𝑅
th
R
th
	​

 ramp threshold

Raw violation:

𝑐
ramp
(
𝑡
)
=
max
⁡
(
0
,
 
∣
𝑃
(
𝑡
)
−
𝑃
(
𝑡
−
1
)
∣
−
𝑅
th
)
c
ramp
	​

(t)=max(0, ∣P(t)−P(t−1)∣−R
th
	​

)

Weighted:

𝑐
~
ramp
(
𝑡
)
=
𝑤
ramp
⋅
𝑐
ramp
(
𝑡
)
c
~
ramp
	​

(t)=w
ramp
	​

⋅c
ramp
	​

(t)
Interpretation:

if change is small → cost = 0

if change is too big → pay penalty proportional to how much too big

5) Needs memory (wrapper state)

You must store the previous signal:

self._prev_grid_signal

Reset it at episode start:

self._prev_grid_signal = None

6) Step-by-step wrapper integration

After KPI computation:

p_signal = float(kpis.get("step_net_consumption_kwh", 0.0))  # recommended net

if self._prev_grid_signal is None:
    ramp_delta = 0.0
    cost_grid_ramp_raw = 0.0
else:
    ramp_delta = abs(p_signal - float(self._prev_grid_signal))
    cost_grid_ramp_raw = max(0.0, ramp_delta - self.ramp_threshold)

self._prev_grid_signal = p_signal
cost_grid_ramp = self.w_grid_ramp * cost_grid_ramp_raw

7) What to log

You want to see the actual delta and the violations:

cost_grid_ramp (weighted)

cost_grid_ramp_raw (raw violation)

grid_ramp_delta (absolute delta)

grid_ramp_violation (0/1)

info["cost_grid_ramp"] = float(cost_grid_ramp)
info["cost_grid_ramp_raw"] = float(cost_grid_ramp_raw)
info["grid_ramp_delta"] = float(ramp_delta)
info["grid_ramp_violation"] = 1.0 if cost_grid_ramp_raw > 0 else 0.0

8) How to set a good ramp threshold

Same rule: don’t make it too strict.

From Greedy RBC baseline:

compute net[t] = step_net_consumption_kwh[t]

compute delta[t] = abs(net[t] - net[t-1])

set:

𝑅
th
=
percentile
(
𝛿
,
95
)
R
th
	​

=percentile(δ,95) (start)
or safer:

𝑅
th
=
percentile
(
𝛿
,
97
)
R
th
	​

=percentile(δ,97)

How they combine with your CMDP cost

If you’re keeping the EV “avoidable deficit” cost as your main safety driver:

cost
(
𝑡
)
=
𝑐
EV
(
𝑡
)
+
𝑐
~
peak
(
𝑡
)
+
𝑐
~
ramp
(
𝑡
)
cost(t)=c
EV
	​

(t)+
c
~
peak
	​

(t)+
c
~
ramp
	​

(t)

Implementation:

total_cost = float(ev_cost_for_cmdp + cost_grid_peak + cost_grid_ramp)


(If you keep SoC band too, just add it.)

Practical tuning rules (so training doesn’t collapse)
1) Start with permissive thresholds

Use 97th percentile first run.
Then tighten to 95th later.

2) Keep weights small at first

If your per-step EV cost is typically small, and peak/ramp raw violations can be big numbers, start with:

w_grid_peak = 0.01 to 0.1

w_grid_ramp = 0.01 to 0.1

Then scale up only if:

violations stay high, and

lambda isn’t exploding early.

3) Always log violation rate

You want something like:

peak violations: ~1–5% of timesteps initially

ramp violations: ~1–5% initially

If it’s 50–90%, your thresholds are too strict.

Why these 2 are the “most KPI-impactful”

Because they directly shape the exact grid-level KPIs CityLearn evaluates:

peak KPIs ← peak import constraint

ramp KPI ← ramping constraint
and both influence cost/carbon indirectly by encouraging smart scheduling.