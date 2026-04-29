import os
import sys

print("="*80)
print("VERIFYING ACTUAL CODE LOGIC")
print("="*80)

# Read the safety wrapper code
wrapper_file = "citylearn_safe/safety_env_v3.py"

print(f"\nReading: {wrapper_file}")
print("-"*80)

with open(wrapper_file, 'r') as f:
    code = f.read()

# Find the EV cost computation section
print("\n1. SEARCHING FOR EV COST COMPUTATION")
print("-"*80)

# Look for where ev_cost_for_cmdp is computed
import re

# Find the cost computation section
ev_cost_pattern = r'(ev_cost_for_cmdp.*?=.*?\n)'
matches = re.findall(ev_cost_pattern, code)

if matches:
    print("Found ev_cost_for_cmdp computation:")
    for match in matches:
        print(f"  {match.strip()}")
else:
    print("❌ ev_cost_for_cmdp not found")

print("\n2. SEARCHING FOR V3 CONTROLLABLE LOGIC")
print("-"*80)

# Look for agent_control_v3 or controllable_v3
v3_patterns = [
    r'(.*?agent.*?control.*?v3.*?\n)',
    r'(.*?controllable.*?v3.*?\n)',
    r'(.*?ev_agent_control.*?\n)'
]

found_v3 = False
for pattern in v3_patterns:
    matches = re.findall(pattern, code, re.IGNORECASE)
    if matches:
        found_v3 = True
        print(f"Found V3 logic (pattern: {pattern}):")
        for match in matches[:5]:  # First 5 matches
            print(f"  {match.strip()}")

if not found_v3:
    print("❌ No V3 controllable logic found")

print("\n3. SEARCHING FOR AVOIDABLE/UNAVOIDABLE LOGIC")
print("-"*80)

# Look for avoidable/unavoidable
avoidable_pattern = r'(.*?avoidable.*?\n)'
matches = re.findall(avoidable_pattern, code, re.IGNORECASE)

if matches:
    print("Found avoidable/unavoidable references:")
    for match in matches[:10]:  # First 10
        print(f"  {match.strip()}")
else:
    print("❌ No avoidable/unavoidable logic found")

print("\n4. CHECKING WHAT GETS LOGGED TO KPI")
print("-"*80)

# Look for info["cost_ev_departure"]
cost_ev_log_pattern = r'(info\[.*?cost_ev.*?\].*?=.*?\n)'
matches = re.findall(cost_ev_log_pattern, code)

if matches:
    print("Found EV cost logging:")
    for match in matches:
        print(f"  {match.strip()}")

print("\n5. CHECKING KPI LOGGER FOR V3 COLUMNS")
print("-"*80)

kpi_logger_file = "citylearn_safe/kpi_logger.py"
if os.path.exists(kpi_logger_file):
    with open(kpi_logger_file, 'r') as f:
        kpi_code = f.read()
    
    # Look for V3 controllable columns
    v3_kpi_pattern = r'(.*?controllable.*?v3.*?\n)'
    matches = re.findall(v3_kpi_pattern, kpi_code, re.IGNORECASE)
    
    if matches:
        print("Found V3 controllable columns in KPI logger:")
        for match in matches[:5]:
            print(f"  {match.strip()}")
    else:
        print("❌ No V3 controllable columns in KPI logger")
else:
    print(f"❌ {kpi_logger_file} not found")

print("\n6. CONCLUSION")
print("-"*80)

# Now check what the actual code does
lines = code.split('\n')
for i, line in enumerate(lines):
    if 'ev_cost_for_cmdp' in line and '=' in line:
        print(f"\nLine {i+1}: {line.strip()}")
        # Show context (5 lines before and after)
        context_start = max(0, i-5)
        context_end = min(len(lines), i+6)
        print("\nContext:")
        for j in range(context_start, context_end):
            marker = " >>> " if j == i else "     "
            print(f"{marker}{lines[j]}")

print("="*80)
