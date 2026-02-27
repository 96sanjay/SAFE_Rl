
V3 Logic Documentation: Agent-Controllable vs Uncontrollable Deficits
Simple Definition
When an EV departs without reaching its required State of Charge (SOC), we have a DEFICIT.
V3 classifies this deficit into two categories:
Agent-Controllable (Avoidable)
Deficit caused by the agent choosing suboptimal actions

The agent COULD have prevented this
The agent had control but made a bad choice

Uncontrollable (Unavoidable)
Deficit caused by factors outside the agent's control

The agent COULD NOT have prevented this
Even with perfect actions, the deficit would still occur


The V3 Question
V3 asks ONE simple question:

"Did the agent request FULL POWER at every controllable timestep?"


NO (action < 1.0) → Agent's fault → CONTROLLABLE
YES (action = 1.0) → Not agent's fault → UNCONTROLLABLE

That's it!

V3 Logic in Code
pythonagent_controllable = 0.0

# Loop through connection window
for each timestep t when EV was connected:
    
    # Skip first timestep (agent reaction delay)
    if t == arrival_timestep:
        continue
    
    # Check agent's action
    if agent_action[t] < 1.0:
        # Agent didn't request full power
        # Calculate missed charging opportunity
        missed_power = (1.0 - agent_action[t]) × P_max
        missed_energy = missed_power × timestep_duration
        agent_controllable += missed_energy / battery_capacity

# Everything else is uncontrollable
uncontrollable = total_deficit - agent_controllable
```

---

## Key Principle

**V3 only looks at what the agent REQUESTED (actions), not what was actually DELIVERED (electricity_consumption).**

### Example:
```
Agent sends: action = 1.0 (requests 10 kW)
Battery delivers: 2 kW (due to physics taper)

V3 verdict: UNCONTROLLABLE
Why? Agent requested full power. Physics limited delivery.
```

---

## What Makes a Deficit "Agent-Controllable"?

### Scenario 1: Charging at Less Than Full Power
```
Required SOC: 80%
Agent action: 0.5 (charges at 50% power)

Result: EV reaches only 60%
Deficit: 20%

V3 Classification:
  - Agent chose action=0.5 instead of 1.0
  - Missed opportunity: (1.0 - 0.5) = 50% of power
  - CONTROLLABLE ✓
```

### Scenario 2: Discharging When Should Charge
```
Required SOC: 80%
Agent action: -0.3 (discharges at 30% power)

Result: EV reaches only 40%  
Deficit: 40%

V3 Classification:
  - Agent chose to DISCHARGE when should CHARGE
  - Should have chosen action=1.0
  - Missed: (1.0 - (-0.3)) = 130% of opportunity
  - CONTROLLABLE ✓
```

### Scenario 3: Idle When Should Charge
```
Required SOC: 80%
Agent action: 0.0 (idle, no charging)

Result: EV stays at arrival SOC
Deficit: Large

V3 Classification:
  - Agent chose to do nothing
  - Should have chosen action=1.0
  - Missed: (1.0 - 0.0) = 100% of opportunity
  - CONTROLLABLE ✓
```

---

## What Makes a Deficit "Uncontrollable"?

### General Rule:
**If agent sent action=1.0 (full power request) at every timestep, but still had deficit → UNCONTROLLABLE**

### The uncontrollable category includes ALL of these factors:

#### 1. Short Connection Time
```
EV connected: 1 hour
Energy needed: 30 kWh
Maximum possible: 10 kW × 1h = 10 kWh

Agent action: 1.0 (full power)
Result: Can only deliver 10 kWh, need 30 kWh

Uncontrollable because: NOT ENOUGH TIME
```

#### 2. Battery Physics (Charging Taper)
```
Battery SOC: 95%
Agent action: 1.0 (requests 10 kW)
Battery delivers: 2 kW (physics limit)

Uncontrollable because: BATTERY CHEMISTRY
Lithium-ion batteries charge slower near 100%
```

#### 3. Power Delivery Constraints
```
Agent requests: 10 kW (action=1.0)
Grid limit: 7 kW available
Battery receives: 7 kW

Uncontrollable because: POWER INFRASTRUCTURE LIMIT
```

#### 4. High Required SOC
```
Arrival: 10%
Required: 95%
Connection: 2 hours
Needs: 42.5 kWh
Maximum possible: 10 kW × 2h = 20 kWh

Uncontrollable because: REQUIREMENT TOO HIGH FOR TIME AVAILABLE
```

#### 5. SOC Limits
```
Battery at: 99.8%
Required: 100%
Agent action: 1.0
System limit: Cannot exceed 100%

Uncontrollable because: PHYSICAL SOC LIMIT
```

---

## What V3 Does NOT Do

**V3 does NOT decompose uncontrollable into subcategories.**

It does NOT tell you:
- X% was due to short time
- Y% was due to battery physics  
- Z% was due to power limits

**V3 simply says:**
> "Agent requested full power, but 20.81 SOC units of deficit still occurred due to factors outside agent control."

**All uncontrollable factors are lumped into ONE number.**

---

## Worked Examples

### Example 1: Perfect Agent (RBC)
```
Agent strategy: action = 1.0 at ALL timesteps

Result after full episode:
  Total departures with deficit: 228
  Total deficit: 41.58 SOC units
  
V3 Classification:
  Agent-controllable: 0.00 (0%)
  Uncontrollable: 41.58 (100%)

Interpretation:
  Agent did its best at every timestep
  All 41.58 SOC units are due to:
    - Short connection times
    - Battery physics
    - Power limits
    - High requirements
  NOT the agent's fault ✓
```

### Example 2: Terrible Agent (Time-based V2G)
```
Agent strategy:
  Hours 0-5:   action = 0.8 (charges at 80%)
  Hours 6-9:   action = -0.5 (DISCHARGES!)
  Hours 10-16: action = 1.0 (full power)
  Hours 17-21: action = -0.3 (DISCHARGES!)
  Hours 22-23: action = 1.0 (full power)

Result after full episode:
  Total departures with deficit: 3,614
  Total deficit: 626.48 SOC units
  
V3 Classification:
  Agent-controllable: 605.67 (96.68%)
  Uncontrollable: 20.81 (3.32%)

Interpretation:
  96.68% of deficits are agent's fault:
    - Charging at 80% instead of 100%
    - DISCHARGING when should charge
    
  Only 3.32% are uncontrollable:
    - Timesteps where agent DID send action=1.0
    - But physics/time/limits prevented reaching required SOC
    
  Agent made deficits 15× worse! ✓
```

### Example 3: Single Departure Breakdown
```
Arrival time: Hour 3
Departure time: Hour 9
Connection: 6 hours

Arrival SOC: 20%
Required SOC: 85%
Needs: 65% = 32.5 kWh

Agent actions:
  Hour 3: action = 0.8 → delivered 8 kWh
  Hour 4: action = 0.8 → delivered 8 kWh  
  Hour 5: action = 0.8 → delivered 8 kWh
  Hour 6: action = -0.5 → delivered -5 kWh (discharged!)
  Hour 7: action = -0.5 → delivered -5 kWh
  Hour 8: action = 1.0 → delivered 9 kWh (taper started)

Total energy: 8+8+8-5-5+9 = 23 kWh = 46%
Final SOC: 20% + 46% = 66%
Deficit: 85% - 66% = 19%

V3 Calculation:

Hour 3-5: action=0.8
  Missed per hour: (1.0-0.8)×10kW×1h = 2 kWh
  Total missed: 6 kWh = 12%
  → CONTROLLABLE

Hour 6-7: action=-0.5
  Missed per hour: (1.0-(-0.5))×10kW×1h = 15 kWh
  Total missed: 30 kWh = 60%
  → CONTROLLABLE (capped at remaining deficit)

Hour 8: action=1.0
  Agent requested full power
  But battery physics caused taper (9 kW instead of 10 kW)
  → UNCONTROLLABLE (small amount)

Final V3 verdict:
  Controllable: ~18% (agent chose action<1.0)
  Uncontrollable: ~1% (physics taper at hour 8)
```

---

## Why V3 is Better Than V2

### V2 Problem:
```
Agent: action = 1.0 (requests full power)
Battery: Delivers 2 kW (physics taper)

V2 sees: Low electricity_consumption (2 kW)
V2 thinks: "Could have charged more"
V2 verdict: AVOIDABLE ❌ WRONG!
```

### V3 Solution:
```
Agent: action = 1.0 (requests full power)
Battery: Delivers 2 kW (physics taper)

V3 sees: Agent action = 1.0
V3 thinks: "Agent requested full power"
V3 verdict: UNCONTROLLABLE ✓ CORRECT!
```

**V3 correctly recognizes that battery physics prevented charging, not the agent.**

---

## Mathematical Formulation

### For Each Departure:
```
deficit_total = required_soc - actual_soc

agent_controllable = 0
for t in connection_window:
    if agent_action[t] < 1.0:
        missed_energy_kwh = (1.0 - agent_action[t]) × P_max × dt
        agent_controllable += missed_energy_kwh / capacity_kwh

agent_controllable = min(agent_controllable, deficit_total)
uncontrollable = deficit_total - agent_controllable
```

### Action Space:
```
action = +1.0  →  Charge at maximum power
action =  0.0  →  Idle (no charge/discharge)
action = -1.0  →  Discharge at maximum power

Any action < 1.0 when charging needed → CONTROLLABLE
```

---

## Key Assumptions

### V3 Assumes:

1. **action=1.0 means "request maximum available power"**
   - Agent cannot request more than P_max
   - action=1.0 is the best the agent can do

2. **Agent has no control over battery physics**
   - Charging taper is automatic
   - Agent cannot override physics

3. **Agent has no control over connection time**
   - EV arrival/departure times are fixed
   - Agent cannot delay departure

4. **Agent has no control over power infrastructure**
   - Grid limits are external constraints
   - Agent cannot increase available power

5. **First timestep has reaction delay**
   - Agent sees observation at time t
   - Action takes effect at time t+1
   - First timestep not blamed on agent

---

## Limitations

### What V3 Cannot Do:

1. **Cannot decompose uncontrollable**
   - Doesn't separate time vs physics vs power limits
   - All lumped into one "uncontrollable" value

2. **Cannot detect optimal policy**
   - Can only say "agent requested full power"
   - Cannot say if that was the RIGHT time to charge
   - Example: Charging during expensive peak hours

3. **Single-EV perspective**
   - Doesn't consider multi-building resource allocation
   - Can't detect if agent should have prioritized different EV

4. **Assumes action space is accurate**
   - If action=1.0 doesn't actually request full power, V3 breaks
   - Depends on correct action space definition

---

## When to Use V3

### Use V3 for:

✅ **Safe RL training**
- Penalize agent for controllable deficits
- Don't penalize for uncontrollable deficits

✅ **Policy evaluation**
- Compare different policies fairly
- RBC baseline should show 0% controllable
- Bad policies should show high % controllable

✅ **Debugging agents**
- Identify if agent is causing deficits
- Distinguish bad policy from difficult scenarios

### Don't use V3 for:

❌ **Root cause analysis**
- V3 doesn't tell you WHY uncontrollable
- Need separate analysis for that

❌ **Multi-objective optimization**
- V3 only checks EV departure constraint
- Doesn't optimize for grid stability, cost, etc.

❌ **Optimal planning**
- V3 doesn't compute optimal policy
- Only checks if agent requested full power

---

## Quick Reference Card
```
╔════════════════════════════════════════════════════════════╗
║                    V3 QUICK REFERENCE                      ║
╠════════════════════════════════════════════════════════════╣
║ QUESTION: Did agent request full power?                   ║
║                                                            ║
║ IF action[t] < 1.0:                                        ║
║   → AGENT-CONTROLLABLE (avoidable)                         ║
║   → Agent chose suboptimal action                          ║
║   → Agent's fault                                          ║
║                                                            ║
║ IF action[t] = 1.0:                                        ║
║   → UNCONTROLLABLE (unavoidable)                           ║
║   → Agent did its best                                     ║
║   → System limits prevented reaching required SOC          ║
║   → Includes: time, physics, power, SOC limits             ║
║                                                            ║
║ EXAMPLE RESULTS:                                           ║
║   RBC (action=1.0):     0% controllable ✓                  ║
║   Bad policy:          97% controllable ✓                  ║
╚════════════════════════════════════════════════════════════╝