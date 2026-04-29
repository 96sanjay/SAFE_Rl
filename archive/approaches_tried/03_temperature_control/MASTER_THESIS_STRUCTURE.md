# Master Thesis Structure

## Working Title
Safe Reinforcement Learning for Building Energy Management: Constraint-Aware Control in CityLearn

## Recommended Thesis Arc

### 1. Introduction
- Motivate building energy management as a safety-critical sequential decision problem.
- Define the practical tension between energy efficiency, occupant comfort, and grid-support objectives.
- State the thesis problem: standard RL can optimize reward but does not reliably satisfy operational constraints.
- State the thesis aim: design, adapt, and evaluate safe RL methods for realistic building-control tasks in CityLearn.
- List thesis contributions at a high level.

### 2. Background and Related Work
- Building energy management systems and demand-response control.
- CityLearn as an experimental platform.
- Reinforcement learning fundamentals.
- Constrained Markov decision processes.
- Safe RL methods: Lagrangian, trust-region, projection, barrier-based, and curriculum/state-augmentation approaches.
- Gap statement:
  current safe RL methods are usually validated on toy control tasks or with limited realism relative to building operations.

### 3. SafeCityLearn Framework
- Present the software framework and environment stack.
- Explain the wrapper architecture, reward shaping, observation handling, and evaluation pipeline.
- Clarify why a unified infrastructure is necessary for fair comparison.
- This chapter should justify the methodological credibility of all later experiments.

### 4. Case Study I: Multi-Constraint V2G / District Energy Management
- Present the richer multi-constraint CMDP.
- Explain the task-specific architecture, constraint definitions, and algorithmic adaptations.
- Evaluate multiple safe RL approaches.
- Use this chapter to establish the general thesis theme:
  safe RL in realistic building-energy settings requires environment-aware algorithm design.

### 5. Case Study II: Temperature Control Benchmark
- Use the temperature-control chapter as the focused, cleaner benchmark chapter.
- Emphasize why this task is a good controlled comparison:
  single comfort constraint, cooling-only action space, standardized evaluation.
- Position CSAC-LB as the main algorithmic focus and compare it against representative baselines.
- This chapter should answer:
  which safe RL mechanism gives the best comfort-efficiency trade-off in this simplified but realistic setting?

### 6. Cross-Case Synthesis
- Compare what carries across the V2G and temperature tasks.
- Distinguish task-specific findings from general findings.
- Candidate synthesis points:
  off-policy methods benefit from long-horizon sample reuse,
  constraint mechanism choice matters more than nominal algorithm family,
  single-seed evaluation is insufficient,
  distribution shift exposes weaknesses not visible in nominal evaluation.

### 7. Limitations and Future Work
- Simulation realism limits.
- Dependence on reward shaping and wrapper design.
- Constraint specification and metric sensitivity.
- Missing real-building deployment.
- Future work:
  multi-objective comfort constraints,
  adaptive barrier schedules,
  robust/distributionally aware evaluation,
  transfer to real BEMS data.

### 8. Conclusion
- Restate the research question.
- Summarize the main empirical and methodological findings.
- State the final practical implication:
  safe RL for building energy management is promising, but only when the constraint-handling method is matched to the task and evaluated rigorously.

## How the Temperature Chapter Should Function

- It should not try to carry the full thesis alone.
- It should function as the cleanest comparative benchmark chapter in the thesis.
- It should demonstrate:
  a well-defined CMDP,
  a transparent algorithmic comparison,
  strong evaluation practice,
  and a defensible interpretation of trade-offs.

## Recommended Chapter-Level Framing for the Temperature Case Study

- Research question:
  how do different safe RL mechanisms behave on a realistic cooling-control CMDP when evaluated on comfort safety, efficiency, robustness, and seed sensitivity?
- Methodological contribution:
  adaptation of CSAC-LB and a unified benchmark setup in SafeCityLearn.
- Empirical contribution:
  multi-algorithm, multi-seed comparison plus heatwave robustness analysis.
- Practical takeaway:
  average-case safety-efficiency and worst-case robustness need not be optimized by the same method.

## Writing Guidance Across the Thesis

- Keep chapter introductions short and purpose-driven.
- Separate framework contribution from algorithm contribution from empirical finding.
- Avoid mixing benchmark-report language with thesis argument language.
- When making claims, tie each claim to:
  the task setup,
  the compared methods,
  the metric,
  and the evaluation regime.
- Reserve strong wording such as "best", "robust", or "safe" for cases where the comparison class and metric are explicitly stated.
