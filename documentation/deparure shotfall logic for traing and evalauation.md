
EV Departure Deficits: V3 vs Oracle (Training vs Evaluation)
Why there are two “unavoidable” notions

There are two different questions you might want to answer:

Responsibility (for training):
“Did the agent cause the deficit by not requesting enough EV charging?”

Feasibility (for evaluation):
“Even with the best possible EV charging behavior, would a deficit still remain?”

These are different questions, so they need different metrics.

Part A — V3 metrics (use for Training)
What V3 measures

V3 splits the observed deficit at EV departures into:

V3 controllable (agent-responsible):
deficit attributed to timesteps where the agent chose action < 1.0 while the EV was connected.

V3 uncontrollable (residual):
the leftover deficit after subtracting the controllable part.

V3 definition (per departure)

Let:

deficit_actual = max(0, required_soc - actual_soc_at_departure)

missed_soc = Σ over connected timesteps of ((1 - action) * pmax * dt / cap) (only when action < 1)

Then:

controllable_v3 = min(missed_soc, deficit_actual)

uncontrollable_v3 = deficit_actual - controllable_v3

Important note

✅ V3 is good for Safe RL training because it gives a clear gradient:

Increase EV action → lower controllable deficit → lower penalty.

⚠️ V3 “uncontrollable” is NOT the simulator minimum deficit.
It is a residual after blame assignment.
So it can change across policies.

What to call it in your thesis

To avoid confusion, don’t call this “physics unavoidable.”

Use these names:

ev_avoidable_deficit_kwh (V3) = agent-responsible deficit (training signal)

ev_unavoidable_deficit_kwh (V3 residual) = residual not blamed on EV actions

What to use in code (training)

Use ev_departure_safe_cost_v3(...) for the safety cost.

Use ev_departure_cost_components_v3(...) for logging and debugging.

Recommendation:

During debugging: set missing_action_mode="error" to catch logging/mapping bugs.

During training: missing_action_mode="assume_full" is ok, but error mode is safer if stable.

Part B — Oracle metrics (use for Evaluation)
What Oracle measures

Oracle answers feasibility:

“What deficit remains if EV charging requests are forced to 1.0 whenever the EV is connected?”

This gives the simulator-unavoidable minimum for EV deficits under your dataset and simulator constraints.

How Oracle is computed

You do a second rollout with the same seed and environment, but you override EV actions:

At each step:

If charger has a connected EV → set its EV action = 1.0

Leave all other actions the same (battery/thermal) or keep policy actions—depending on your evaluation setup.

Then measure total EV deficit at departures in this oracle rollout.

Oracle outputs

oracle_deficit_kwh_total
This is simulator-unavoidable deficit (feasibility minimum).

Policy vs Oracle gap

Given:

policy_deficit_kwh_total

oracle_deficit_kwh_total

Compute:

avoidable_wrt_oracle_kwh = max(0, policy_deficit - oracle_deficit)

avoidable_wrt_oracle_percent = avoidable_wrt_oracle_kwh / policy_deficit

This tells you:
✅ “How much of the policy’s EV deficit is avoidable in principle?”

What to call it in your thesis

Use these terms:

Simulator-unavoidable (Oracle) = deficit in oracle rollout

Avoidable wrt Oracle = policy deficit − oracle deficit

This is your clearest “best-case” benchmark.

Which to use when (quick rules)
Training (Safe RL reward / constraint)

Use V3 controllable deficit:

It tells the agent what it can improve.

It is step-wise and action-linked.

✅ Use:

ev_departure_safe_cost_v3 (uses V3 controllable)

Evaluation (thesis plots / comparison / professor report)

Use Oracle + gap:

It tells you feasibility minimum and how close your policy is to best-case.

✅ Use:

run_policy_and_oracle_rollouts

summarize_oracle_gap

How to interpret your real results (template)

Example interpretation you can reuse:

Oracle rollout (EV full when connected) produces
X kWh deficit across Y deficit departures → simulator-unavoidable minimum.

Policy rollout produces
A kWh deficit across B deficit departures.

Therefore,
A − X kWh (~Z%) of the policy’s deficit is avoidable relative to the oracle best-case.

Naming cheat-sheet (never confuse again)
V3 (Training)

“avoidable” = agent-responsible (action < 1)

“unavoidable” = V3 residual (not blamed on EV actions)

Oracle (Evaluation)

“unavoidable” = simulator-unavoidable minimum (best-case rollout)

Common confusion warning

If you see:

V3 residual-unavoidable in a bad policy > oracle-unavoidable

That is normal, because:

V3 residual is not a feasibility minimum.

Oracle is the feasibility minimum.

They answer different questions.