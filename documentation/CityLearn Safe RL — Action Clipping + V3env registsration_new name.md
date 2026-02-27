CityLearn Safe RL — Action Clipping + V3 Environment Registration Notes
Goal

Fix the problem where the policy outputs actions outside the environment bounds (e.g., -6 to +5) and ensure:

Actions passed into the simulator are always valid

KPI CSV logs reflect the bounded actions

A V3 environment (action-based EV deficit logic) is available via a new registered env_id without breaking reproducibility of older runs.

Key Facts About Action Bounds

The action space is 26-dimensional and not uniform:

25 dimensions: bounds are [-1, 1]

1 dimension (action_2): bounds are [0, 1] (lower bound = 0)

So a “global check” of just [-1,1] is not enough. You must enforce per-dimension bounds, especially for action_2.

Why We Needed Explicit Action Clipping

We observed that actions from the policy could be far outside the action bounds (e.g., -5…+5). Without clipping:

The underlying env may silently accept raw actions or internally saturate them.

Even if internal clipping exists somewhere, the policy/environment interaction becomes inconsistent (policy thinks it applied a_raw, env uses clip(a_raw)).

KPI logs can become misleading if you log the raw action rather than what should be applied.

Decision: Register a New Env Instead of Overwriting the Old One
Why register a new env_id?

The original training configs use env_id like CityLearnSafety-SoC-v0 (or similar). Changing what that id points to would:

Break reproducibility: old results would no longer correspond to the same environment implementation.

Make comparisons confusing (same env_id but different behavior/cost/reward definitions).

Risk hidden bugs if other scripts assume old wrapper behavior.

✅ Therefore, we registered a new env_id that explicitly points to V3.

Files Involved
1) Environment Registration

File: scripts/register_env.py

This file defines _thunk() and registers env ids with gym.register(...).
We updated it to register a new V3 env.

Example of what was added/changed conceptually:

Simple-v0 → old wrapper CityLearnSafetyEnv

SimpleV3-v0 → new wrapper CityLearnSafetyEnvV3

✅ New environment id name: SimpleV3-v0
(You can rename it to something clearer like CityLearnSafetyV3-SoC-v0 by changing the id="..." string in gym.register.)

Important: Gym registration is per Python process. Any training/eval script must import/run the registration code in that process.

2) V3 Safety Wrapper

File: citylearn_safe/safety_env_v3.py

This is the implementation of:

Action-based EV deficit decomposition (V3 extractor)

Bill-based reward logic

Grid peak and ramp costs (if enabled)

KPI logging

✅ Action clipping is implemented inside CityLearnSafetyEnvV3.step(), using per-dimension bounds:

Compute action_raw = np.asarray(action).ravel()

Use low = action_space.low, high = action_space.high

Compute action_arr = np.clip(action_raw, low, high)

Pass action_arr into self.base.step(action_arr)

Log action_0..25 from action_arr

This ensures:

No out-of-bounds actions reach the base environment.

action_2 is never negative (enforced to [0,1]).

How to Rename the New Env

In scripts/register_env.py, find the registration line:

gym.register(id="SimpleV3-v0", entry_point=_thunk_v3)


Change "SimpleV3-v0" to something clearer, e.g.:

gym.register(id="CityLearnSafetyV3-SoC-v0", entry_point=_thunk_v3)


Also update any registry.pop("...") overwrite lines and training configs that reference the env_id.

Verification: Tests That Prove It Works
A) Definitive “Action forwarded to inner env is clipped” test

We used a capture hook on the inner env’s .step() to verify the action actually passed downstream is clipped.

Expected result:

Forwarded action min/max should be within [-1,1]

action_2 must be within [0,1]

Violations count must be 0

✅ Result achieved: PASS

B) 1-epoch-length KPI CSV bound verification

We ran a full rollout length (8759 steps) and verified in the KPI CSV:

No action_i values outside their per-dimension bounds

No negative action_2

Expected output:

TOTAL OOB = 0

action_2 negatives = 0

✅ Result achieved: SUCCESS

Important Caveat About scripts/register_env.py

Your registration script had a __main__ sanity-check section that calls:

env = gym.make("Simple-v0")


In this repo, gym.make can crash with:

AssertionError: env.spec is not None

So when registering inside tests, we often used a safe approach:

Temporarily patch gym.make to prevent the sanity check from running

Then register envs and create envs via spec.entry_point(...) directly

For long-term cleanliness:
✅ Update the __main__ sanity checks to avoid gym.make() and call the _thunk_*() directly instead.

What To Tell a New Chat Assistant (Summary)

Action space: 26 dims; action_2 is [0,1], others mostly [-1,1].

The old env_id pipeline (CityLearnSafety-SoC-v0) used CityLearnSafetyEnv and did not clip actions at the boundary.

V3 wrapper (CityLearnSafetyEnvV3 in citylearn_safe/safety_env_v3.py) was updated to clip actions inside step().

A new env_id SimpleV3-v0 was registered in scripts/register_env.py to point to V3 without breaking old results.

Verified via:

forwarded-action capture test (inner env receives clipped actions)

8759-step rollout + KPI CSV per-dimension bounds check (OOB=0, action_2 negatives=0)