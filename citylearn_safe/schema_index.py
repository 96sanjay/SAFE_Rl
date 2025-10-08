# citylearn_safe/schema_index.py
from __future__ import annotations
import json, os
from typing import List, Tuple, Dict, Any

def _active_shared_lists(schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (shared_active_names, per_building_active_names) in schema order."""
    obs = schema.get("observations", {})
    shared, per_b = [], []
    for name, cfg in obs.items():
        if not isinstance(cfg, dict):
            continue
        if not cfg.get("active", False):
            continue
        if cfg.get("shared_in_central_agent", False):
            shared.append(name)
        else:
            per_b.append(name)
    return shared, per_b

def _num_included_buildings(schema: Dict[str, Any]) -> int:
    bdict: Dict[str, Any] = schema.get("buildings", {})
    n = 0
    for _, bconf in bdict.items():
        include = bconf.get("include", True)
        if include:
            n += 1
    return n

def soc_indices_from_schema_and_obs_dim(
    schema_path: str,
    obs_dim: int,
    soc_name: str = "electrical_storage_soc",
) -> Tuple[List[int], List[str]]:
    """
    Compute indices of `soc_name` in the central-agent flattened observation:
      [shared...] + [per-building vars for B1] + ... + [for BN]
    We don't assume how many extra features wrappers add to 'shared'; instead:
      start_of_per_building = obs_dim - N_buildings * len(per_building_active)
    """
    if not os.path.exists(schema_path):
        raise FileNotFoundError(schema_path)
    schema = json.load(open(schema_path, "r"))

    shared, per_b = _active_shared_lists(schema)
    n_b = _num_included_buildings(schema)
    if n_b <= 0:
        return [], per_b

    per_b_len = len(per_b)
    if per_b_len == 0:
        return [], per_b

    # position of soc within the per-building block (e.g., 0..per_b_len-1)
    try:
        soc_pos = per_b.index(soc_name)
    except ValueError:
        return [], per_b  # soc not present in per-building list

    # where the per-building blocks start, given the *actual* obs length
    start = obs_dim - n_b * per_b_len
    if start < 0:
        # schema mismatch; fail gracefully
        return [], per_b

    idxs = [start + b * per_b_len + soc_pos for b in range(n_b)]
    return idxs, per_b
