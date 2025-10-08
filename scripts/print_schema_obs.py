# scripts/print_schema_obs.py
from __future__ import annotations
import os, json
from typing import Any, Dict, List

def obs_names_from_building_schema(bconf: Dict[str, Any]) -> List[str]:
    """Return the observation variable names for one building."""
    candidates = ["observations", "observation_variables", "observation"]
    obs = None
    for k in candidates:
        if k in bconf:
            obs = bconf[k]
            break
    if obs is None:
        return []
    names: List[str] = []
    for item in obs:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            n = item.get("name") or item.get("variable") or item.get("id")
            if n:
                names.append(n)
    return names

def main():
    schema_path = os.environ.get("CITYLEARN_SCHEMA")
    if not schema_path or not os.path.exists(schema_path):
        raise SystemExit("Set CITYLEARN_SCHEMA to your local schema.json first.")
    schema = json.load(open(schema_path, "r"))
    buildings = schema.get("buildings", {})

    print(f"Schema: {schema_path}")
    flat: List[str] = []
    print("\nPer-building observation names:")
    offset = 0
    for i, (bname, bconf) in enumerate(buildings.items()):
        names = obs_names_from_building_schema(bconf)
        print(f"  [{i:02d}] {bname} ({len(names)} vars)")
        for j, n in enumerate(names):
            print(f"     idx {offset+j:03d}: {n}")
        flat.extend(names)
        offset += len(names)

    print("\nAny names containing 'soc' (case-insensitive):")
    hits = [(i, n) for i, n in enumerate(flat) if "soc" in n.lower()]
    if not hits:
        print("  (none found)")
    else:
        for i, n in hits:
            print(f"  idx {i:03d}: {n}")

if __name__ == "__main__":
    main()
