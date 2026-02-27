
Custom Safe-RL Policy Brain for CityLearn V2G

This is the full “A to Z” spec of the model architectures we will build and plug into OmniSafe PPO-Lagrangian. It includes Approach 1 (no attention) and Approach 2 (light attention), plus how the safety shield fits in without breaking PPO.

0) What we are trying to achieve
Primary target

Reliable safety: keep CMDP cost under your budget (e.g., < 350) with high consistency across seeds.

Secondary target

Better reward (lower bill) than RBC / current PPO-Lag where possible, especially under robustness tests.

Key constraint

Must remain plug-and-play with OmniSafe PPO-Lag (no custom training loop rewrite).

So: we only change the “brain” (policy + critics), keep everything else intact.

1) What stays unchanged in your pipeline

Environment wrapper (CityLearnSafetyEnvV3), cost computation, logging.

OmniSafe’s PPO-Lag training loop: rollout collection, GAE, optimization steps, lambda update.

Action space: 26 dims (17 battery + 8 EV + 1 washer), each in [-1, 1].

2) Critical interface contract with OmniSafe PPO-Lag

Your policy network must behave like a standard PPO Gaussian policy:

Actor must provide

mean action: shape [batch, 26]

log_std: shape [26] (usually a learnable parameter)

OmniSafe will compute:

action distribution

logprob / entropy

PPO clipped objective

etc.

Critics must provide

reward value: scalar per obs

cost value: scalar per obs

Important: We keep policy feed-forward (no LSTM hidden state).
If you want temporal patterns, we use explicit time features / rolling features, not recurrent memory.

3) Observation design principle (the real “brain upgrade”)

Your observation is currently a flat 153-dim vector. We will impose structure:

Why

Your system has device heterogeneity + coupled constraints.

A flat MLP can learn it, but it’s inefficient and unstable.

Structured encoding gives a strong inductive bias and improves learning stability.

How

We create an InputProcessor that slices the observation into semantic groups.

Note: Your exact slicing must match your actual obs layout. The template below shows the structure we aim for, and you’ll adjust indices once you confirm the real layout from your env.

APPROACH 1 (Recommended baseline)
“Structured Encoders + MLP trunk + Separate Heads + Hard Safety Shield (eval-time)”

This is the highest probability of good results with minimal risk.

A1. High-level architecture diagram
Obs(153)
  |
  v
[InputProcessor]
  |---------------------------|-----------------------------|
  v                           v                             v
Global features           Battery features              EV features
(time/price/grid)         (17 x d_b_in)                 (8 x d_ev_in)
  |                           |                             |
  v                           v                             v
[Global MLP]              [Battery Encoder]             [EV Encoder]
  |                           |                             |
  |                           v                             v
  |                     Battery embeddings             EV embeddings
  |                     (17 x d_dev)                   (8 x d_dev)
  |                           \                           /
  |                            \                         /
  |                             v                       v
  |                       [Pooling / Flatten]
  |                               |
  v                               v
         concat(global_emb, all_device_emb_flat)
                      |
                      v
                [Shared Trunk MLP]
                      |
          |-----------|------------|
          v                        v
   [Battery Heads]           [EV Heads]         [Washer Head]
    17 outputs                8 outputs            1 output
          \-----------|------------/
                      v
              mean_action (26)
                      |
           (training: output mean only)
           (eval: mean -> HARD SHIELD -> env)

A2. Modules in detail
A2.1 InputProcessor (structured slicing)

Goal: map obs [batch,153] → dict of tensors.

Outputs we want:

global_ctx: time/price/carbon/global grid signals (shape [batch, d_g])

battery_state: per-building features (shape [batch, 17, d_b_in])

ev_state: per-EV features (shape [batch, 8, d_ev_in])

washer_state: washer features (shape [batch, d_w_in])

optionally constraint_ctx: engineered margins (see A2.5)

Engineered features we strongly recommend adding (if not already in obs):

peak_margin = peak_threshold - current_import

ramp_margin = ramp_threshold - abs(current_signal - prev_signal)

hour_sin, hour_cos (if not already)

EV urgency: time_to_departure, energy_needed (if available or derivable)

These features are extremely useful and low-risk.

A2.2 Global Encoder (MLP)

Input: global_ctx [batch, d_g]
Output: global_emb [batch, d_global]

Recommended:

d_global = 32 or 64

2-layer MLP with LayerNorm

A2.3 Battery Encoder (shared across 17 buildings)

Input: battery_state [batch, 17, d_b_in]
Output: battery_emb [batch, 17, d_dev]

Use the same small MLP for each building:

d_dev = 16 or 32 (start with 16)

Reason:

parameter sharing → generalization and stability

A2.4 EV Encoder (shared across 8 EVs)

Input: ev_state [batch, 8, d_ev_in]
Output: ev_emb [batch, 8, d_dev]

Same idea: small MLP shared across EVs.

A2.5 Constraint Context Vector (must-have addition)

This is a small vector appended everywhere so the policy knows how close it is to limits.

Define:

constraint_ctx = [peak_margin, ramp_margin, current_import, last_import, price, hour_sin, hour_cos, ...]
Shape: [batch, d_c] (small, like 6–12 dims)

We feed it into:

shared trunk input

and optionally each head input

This improves “constraint awareness” massively.

A2.6 Fusion + Shared Trunk

We now fuse everything into one vector:

flatten device embeddings:

battery_emb_flat = reshape([batch, 17*d_dev])

ev_emb_flat = reshape([batch, 8*d_dev])

concatenate:

z = concat(global_emb, constraint_ctx, battery_emb_flat, ev_emb_flat, washer_emb(optional))

Pass through trunk MLP:

output shared_latent [batch, d_shared] where d_shared = 64 (start 64)

A2.7 Heads (device-specific outputs)

We generate actions with specialized heads:

Battery head (shared weights applied 17 times)

For building i:

input: concat(shared_latent, battery_emb[:,i,:], constraint_ctx)

output: scalar action in [-1,1]

EV head (shared weights applied 8 times)

For EV j:

input: concat(shared_latent, ev_emb[:,j,:], constraint_ctx)

output: scalar action in [-1,1]

Washer head

input: concat(shared_latent, washer_emb, constraint_ctx)

output: scalar action in [-1,1]

Concatenate into mean action [batch, 26].

A3. Critics (reward critic + cost critic)

We keep critics simpler than the actor to reduce risk.

Two options:

Option C1 (simplest, recommended)

Both critics are standard MLP over raw obs (153 → 64 → 64 → 1)

Option C2 (structured critic)

Same structured encoders as policy, then a value head

C2 is more consistent but adds complexity.
If you want stability: C1 is fine.

A4. Safety Shield (hard, verifiable) — evaluation/deployment only

This is crucial: it gives you guaranteed peak/ramp control if you design it using real env signals.

Where it lives

NOT inside policy forward during training

It lives in:

evaluation script, OR

environment wrapper for evaluation runs

What it does

Given:

mean actions a ∈ [-1,1]^26

current grid_import, current signal, previous signal

thresholds peak_threshold, ramp_threshold

It modifies actions so that:

predicted/observed import does not exceed peak threshold

ramp change does not exceed ramp threshold

EV urgency rules prevent controllable deficits

Important: The shield must use real measurable signals from env info (kpis) not a naive linear predictor.

A5. Expected behavior

Approach 1 should improve over a plain MLP PPO policy because:

device specialization reduces learning noise

constraint context improves coupling handling

shield makes safety reliable

APPROACH 2 (Optional upgrade)
“Approach 1 + Light Cross-Device Attention”

This adds coordination capacity while staying PPO-compatible.

B1. What changes vs Approach 1

We insert a single self-attention block over device embeddings:

tokens = 26 devices (17 batteries + 8 EV + washer)

embed size = d_dev (16 or 32)

layers = 1 (max 2)

heads = 2–4

No temporal transformer. No LSTM. No history.

B2. Architecture diagram
Battery embeddings (17 x d_dev)   EV embeddings (8 x d_dev)   Washer emb (1 x d_dev)
            \                         |                          /
             \                        |                         /
              v                       v                        v
           concat -> device_tokens: [batch, 26, d_dev]
                             |
                             v
               [Self-Attention block (1 layer)]
                             |
                             v
        coordinated_tokens: [batch, 26, d_dev]
                             |
                             v
flatten + concat(global_emb + constraint_ctx)
                             |
                             v
                   [Shared Trunk MLP]
                             |
                      action heads

B3. Why this helps (when it helps)

This directly targets the “coupling” you care about:

Which buildings should reduce charging when district peak is tight?

Which batteries should discharge to make room for urgent EV charge?

Coordination across devices without hand rules.

But:

it’s not guaranteed to beat Approach 1

it adds tuning sensitivity

so you build it only after A1 is solid

B4. Attention interpretability bonus

You can log attention weights and show:

which devices attend to which others under peak/ramp stress

This is a nice thesis section.

4) Training plan for both approaches
Step 1: Start with Approach 1

Train PPO-Lag with your existing configs

Compare with:

RBC

PPO-Lag baseline MLP

Approach 1 brain

Step 2: Add shield for evaluation

Report:

“raw policy” vs “policy + shield”
This is extremely defensible.

Step 3: Build Approach 2 (attention)

Same training setup

Evaluate whether attention improves:

reward

safety stability

robustness

5) What NOT to include (because it breaks plug-and-play)

LSTM history buffers inside policy (recurrent PPO requirement)

Full transformer over time sequences

Differentiable projection based on a rough import predictor (fragile + misleading safety claim)

If you want temporal awareness, use engineered time features (hour/day/month) + rolling features.

6) Minimal hyperparameter starting point

Use these as defaults:

d_dev = 16

d_global = 32

d_shared = 64

Trunk depth: 2 layers

Heads: small (32 → 1)

Attention (Approach 2):

1 layer

2 or 4 heads

dropout 0.0–0.1

7) What should be logged for thesis

For each model:

Mean ± std across seeds of:

total reward

total cost

peak violation rate

ramp violation rate

EV controllable deficit

Pareto plot: reward vs cost

Robustness tests:

price noise / solar noise / load noise

For Approach 2:

sample attention heatmaps in peak-stress hours

8) Summary: What the other LLM should understand

We are not inventing a new algorithm.

We are only upgrading the policy brain (and optionally attention), staying compatible with OmniSafe PPO-Lag.

We use a hard safety shield in evaluation/deployment to guarantee operational safety.

Approach 1 is the stable, high-probability baseline.

Approach 2 adds coordination via light attention as an ablation/upgrade.