import os

print("="*80)
print("PRE-FLIGHT CHECKLIST")
print("="*80)

checks = []

# 1. Extractor
with open("citylearn_safe/extractors_v3.py") as f:
    ext = f.read()
c1 = '"deficit_violation_count"' in ext and 'deficit_kwh > 0.01' in ext
checks.append(("✓" if c1 else "✗", "Extractor counts violations with 0.01 threshold"))

# 2. Wrapper
with open("citylearn_safe/safety_env_v3.py") as f:
    wrap = f.read()
c2 = 'ev_departure_violation_count_deficit' in wrap
checks.append(("✓" if c2 else "✗", "Wrapper exposes violation count"))

# 3. Evaluator
with open("evaluation/core/evaluator.py") as f:
    ev = f.read()
c3 = 'ev_departure_violation_count_deficit' in ev
checks.append(("✓" if c3 else "✗", "Evaluator uses correct field name"))

# 4. Environment variables
vars_ok = all([
    os.environ.get("CITYLEARN_SCHEMA"),
    os.environ.get("CITYLEARN_EV_DENSE_COST_SCALE") == "1.0",
    os.environ.get("CITYLEARN_STEMS_P_GRID_MAX") == "27.127751",
])
checks.append(("✓" if vars_ok else "✗", "Environment variables set"))

for status, msg in checks:
    print(f"  {status} {msg}")

all_pass = all(status == "✓" for status, _ in checks)

print("\n" + "="*80)
if all_pass:
    print("✓✓✓ ALL CHECKS PASSED - READY TO RUN ✓✓✓")
    print("\nRun: python evaluate_ppolag_only.py")
else:
    print("✗ SOME CHECKS FAILED - SEE ABOVE")
print("="*80)
