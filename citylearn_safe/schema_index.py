
# citylearn_safe/schema_index.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Deterministic indexing for EV-enabled schema + NormalizedObservationWrapper
# Source of truth: env.base.observation_names  (length must equal obs_dim == 153)
# =============================================================================

_EV_NAME_RE = re.compile(
    r"^(?P<prefix>electric_vehicle_charger|connected_electric_vehicle_at_charger|incoming_electric_vehicle_at_charger)"
    r"_charger_(?P<bld>\d+)_(?P<slot>\d+)_"
    r"(?P<feat>connected_state|departure_time|required_soc_departure|soc|battery_capacity|incoming_state|estimated_arrival_time)$"
)


@dataclass(frozen=True)
class ObsIndex:
    """Deterministic observation indexer built from env.base.observation_names."""
    obs_dim: int
    names: Tuple[str, ...]

    # per-building (length == n_buildings == 17 in your case)
    non_shiftable_load: Tuple[int, ...]
    solar_generation: Tuple[int, ...]
    electrical_storage_soc: Tuple[int, ...]
    net_electricity_consumption: Tuple[int, ...]

    # shared scalar indices (single index)
    month_cos: int
    month_sin: int
    day_type_cos: int
    day_type_sin: int
    hour_cos: int
    hour_sin: int
    carbon_intensity: int
    electricity_pricing: int
    electricity_pricing_predicted_1: int
    electricity_pricing_predicted_2: int
    electricity_pricing_predicted_3: int

    # EV mapping:
    # ev["charger_15_2"]["soc"] -> index
    ev: Dict[str, Dict[str, int]]

    # misc (optional)
    washing_machine_1_start_time_step: Optional[int]
    washing_machine_1_end_time_step: Optional[int]


# Cache per-process
_CACHE: Optional[ObsIndex] = None


def _get_names_env_base(env: Any) -> List[str]:
    """
    Deterministically retrieve normalized observation names.
    In your env: top is SingleAgentListAdapter, and env.base is NormalizedObservationWrapper.
    """
    # Primary deterministic path for your repo
    if hasattr(env, "base") and getattr(env, "base") is not None:
        base = getattr(env, "base")
        if hasattr(base, "observation_names"):
            names = getattr(base, "observation_names")
            return _flatten_names(names)

    # Small fallback (still deterministic failure if not found)
    cur = env
    for _ in range(10):
        for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
            if hasattr(cur, attr) and getattr(cur, attr) is not None:
                cur = getattr(cur, attr)
                if hasattr(cur, "observation_names"):
                    return _flatten_names(getattr(cur, "observation_names"))
                break
        else:
            break

    raise RuntimeError(
        "Could not find `observation_names` on env.base or any common wrapper attribute. "
        "Re-run your name-dump script to locate where observation_names lives."
    )


def _flatten_names(names_obj: Any) -> List[str]:
    """
    CityLearn often stores names as list-of-list for central agent.
    """
    if names_obj is None:
        raise RuntimeError("observation_names is None")

    if isinstance(names_obj, (list, tuple)) and len(names_obj) == 1 and isinstance(names_obj[0], (list, tuple)):
        names = list(names_obj[0])
    elif isinstance(names_obj, (list, tuple)):
        names = list(names_obj)
    else:
        raise RuntimeError(f"Unsupported observation_names type: {type(names_obj)}")

    return [str(n) for n in names]


def _obs_dim(env: Any) -> int:
    sp = env.observation_space[0] if isinstance(env.observation_space, (list, tuple)) else env.observation_space
    return int(sp.shape[0])


def _indices_of(names: List[str], target: str) -> List[int]:
    return [i for i, n in enumerate(names) if n == target]


def _index_of_unique(names: List[str], target: str) -> int:
    idxs = _indices_of(names, target)
    if len(idxs) != 1:
        raise RuntimeError(f"Expected exactly one '{target}', got {len(idxs)} at {idxs}")
    return idxs[0]


def build_index(env: Any, *, expected_buildings: int = 17) -> ObsIndex:
    """
    Build deterministic indices from observation_names.
    Validates expected counts so nothing fails silently.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    names = _get_names_env_base(env)
    dim = _obs_dim(env)

    if len(names) != dim:
        raise RuntimeError(f"len(observation_names)={len(names)} != obs_dim={dim}")

    # ---- per-building repeated signals (must be exactly 17 each in your output)
    nsl = _indices_of(names, "non_shiftable_load")
    sol = _indices_of(names, "solar_generation")
    soc = _indices_of(names, "electrical_storage_soc")
    net = _indices_of(names, "net_electricity_consumption")

    for label, idxs in [
        ("non_shiftable_load", nsl),
        ("solar_generation", sol),
        ("electrical_storage_soc", soc),
        ("net_electricity_consumption", net),
    ]:
        if len(idxs) != expected_buildings:
            raise RuntimeError(
                f"Expected {expected_buildings} occurrences of '{label}' but got {len(idxs)}.\n"
                f"indices={idxs}\n"
                "This means the env layout changed; re-run obs name dump and update mapping."
            )

    # ---- shared
    month_cos = _index_of_unique(names, "month_cos")
    month_sin = _index_of_unique(names, "month_sin")
    day_type_cos = _index_of_unique(names, "day_type_cos")
    day_type_sin = _index_of_unique(names, "day_type_sin")
    hour_cos = _index_of_unique(names, "hour_cos")
    hour_sin = _index_of_unique(names, "hour_sin")

    carbon_intensity = _index_of_unique(names, "carbon_intensity")
    electricity_pricing = _index_of_unique(names, "electricity_pricing")
    electricity_pricing_predicted_1 = _index_of_unique(names, "electricity_pricing_predicted_1")
    electricity_pricing_predicted_2 = _index_of_unique(names, "electricity_pricing_predicted_2")
    electricity_pricing_predicted_3 = _index_of_unique(names, "electricity_pricing_predicted_3")

    # ---- washing machine (optional)
    wm_start = _indices_of(names, "washing_machine_1_start_time_step")
    wm_end = _indices_of(names, "washing_machine_1_end_time_step")
    washing_machine_1_start_time_step = wm_start[0] if wm_start else None
    washing_machine_1_end_time_step = wm_end[0] if wm_end else None

    # ---- EV mapping: build a deterministic dict charger_id -> {feature -> index}
    ev: Dict[str, Dict[str, int]] = {}
    for i, n in enumerate(names):
        m = _EV_NAME_RE.match(n)
        if not m:
            continue
        bld = m.group("bld")
        slot = m.group("slot")
        feat = m.group("feat")
        charger_id = f"charger_{bld}_{slot}"
        ev.setdefault(charger_id, {})[feat] = i

    # Validate EV entries have complete feature sets (7 each)
    required_feats = {
        "connected_state",
        "departure_time",
        "required_soc_departure",
        "soc",
        "battery_capacity",
        "incoming_state",
        "estimated_arrival_time",
    }
    for charger_id, feats in ev.items():
        missing = required_feats - set(feats.keys())
        if missing:
            raise RuntimeError(f"EV charger '{charger_id}' missing features: {sorted(missing)}. Found: {sorted(feats)}")

    idx = ObsIndex(
        obs_dim=dim,
        names=tuple(names),

        non_shiftable_load=tuple(nsl),
        solar_generation=tuple(sol),
        electrical_storage_soc=tuple(soc),
        net_electricity_consumption=tuple(net),

        month_cos=month_cos,
        month_sin=month_sin,
        day_type_cos=day_type_cos,
        day_type_sin=day_type_sin,
        hour_cos=hour_cos,
        hour_sin=hour_sin,
        carbon_intensity=carbon_intensity,
        electricity_pricing=electricity_pricing,
        electricity_pricing_predicted_1=electricity_pricing_predicted_1,
        electricity_pricing_predicted_2=electricity_pricing_predicted_2,
        electricity_pricing_predicted_3=electricity_pricing_predicted_3,

        ev=ev,

        washing_machine_1_start_time_step=washing_machine_1_start_time_step,
        washing_machine_1_end_time_step=washing_machine_1_end_time_step,
    )

    _CACHE = idx
    return idx


# =============================================================================
# Convenience API (what your code should call)
# =============================================================================

def soc_indices(env: Any) -> List[int]:
    """Indices (len 17) for electrical_storage_soc across buildings."""
    return list(build_index(env).electrical_storage_soc)


def building_feature_indices(env: Any, feature_name: str) -> List[int]:
    """
    Return indices (len 17) for repeated per-building signals.
    Supported: non_shiftable_load, solar_generation, electrical_storage_soc, net_electricity_consumption
    """
    idx = build_index(env)
    if feature_name == "non_shiftable_load":
        return list(idx.non_shiftable_load)
    if feature_name == "solar_generation":
        return list(idx.solar_generation)
    if feature_name == "electrical_storage_soc":
        return list(idx.electrical_storage_soc)
    if feature_name == "net_electricity_consumption":
        return list(idx.net_electricity_consumption)
    raise KeyError(f"Unsupported per-building feature_name='{feature_name}'")


def ev_charger_ids(env: Any) -> List[str]:
    """e.g. ['charger_1_1', 'charger_4_1', ..., 'charger_15_2']"""
    return sorted(build_index(env).ev.keys())


def ev_feature_index(env: Any, charger_id: str, feature: str) -> int:
    """
    Example:
      ev_feature_index(env, 'charger_15_2', 'departure_time') -> 139
    """
    idx = build_index(env)
    if charger_id not in idx.ev:
        raise KeyError(f"Unknown charger_id='{charger_id}'. Known: {sorted(idx.ev.keys())}")
    if feature not in idx.ev[charger_id]:
        raise KeyError(f"Unknown feature='{feature}' for {charger_id}. Known: {sorted(idx.ev[charger_id].keys())}")
    return idx.ev[charger_id][feature]


def ev_feature_indices(env: Any, feature: str) -> Dict[str, int]:
    """
    Return {charger_id: index} for a given EV feature across all chargers.
    """
    idx = build_index(env)
    out: Dict[str, int] = {}
    for cid, feats in idx.ev.items():
        if feature not in feats:
            raise KeyError(f"Feature '{feature}' missing for charger {cid}")
        out[cid] = feats[feature]
    return out


# =============================================================================
# Legacy function (schema-only). Not valid for your EV-enabled normalized layout.
# =============================================================================

def soc_indices_from_schema_and_obs_dim(*args, **kwargs):
    raise RuntimeError(
        "soc_indices_from_schema_and_obs_dim is NOT valid for this EV-enabled + NormalizedObservationWrapper layout.\n"
        "Your env provides deterministic observation_names; use soc_indices(env) instead."
    )
