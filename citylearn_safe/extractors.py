# citylearn_safe/extractors.py
from __future__ import annotations
from typing import Any, Dict, List

def battery_soc_list(base_env: Any) -> List[float]:
    """Return SoC (0..1) for every electricity storage across buildings, if any."""
    socs: List[float] = []
    for b in getattr(base_env, "buildings", []):
        es = getattr(b, "electricity_storage", None)
        if es is not None and hasattr(es, "soc"):
            try:
                socs.append(float(es.soc))
            except Exception:
                pass
    return socs

def battery_soc_stats(base_env: Any) -> Dict[str, float]:
    socs = battery_soc_list(base_env)
    if not socs:
        return {"soc_mean": 0.5, "soc_min_obs": 0.5, "soc_max_obs": 0.5, "num_storages": 0.0}
    return {
        "soc_mean": sum(socs) / len(socs),
        "soc_min_obs": min(socs),
        "soc_max_obs": max(socs),
        "num_storages": float(len(socs)),
    }
