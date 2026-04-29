import os
import numpy as np

# Turn on extra debug fields from safety_env.py
os.environ["CITYLEARN_DEBUG_OBS_VS_STATE"] = "1"
# Include EV cost into CMDP cost if you want to test combined cost
os.environ["CITYLEARN_INCLUDE_EV_COST"] = "1"
# Optional: binary/hinge
# os.environ["CITYLEARN_COST_MODE"] = "hinge"

def unwrap_citylearn(env):
    """Best-effort unwrap down to CityLearnEnv (has .buildings)."""
    cur = env
    for _ in range(12):
        if hasattr(cur, "buildings"):
            return cur
        for a in ("base", "env", "unwrapped", "_env", "raw_env"):
            if hasattr(cur, a) and getattr(cur, a) is not None:
                cur = getattr(cur, a)
                break
        else:
            break
    return None

def fmt_list(xs, n=5):
    xs = list(xs)
    head = xs[:n]
    return " ".join([f"{x:.3f}" for x in head]) + (" ..." if len(xs) > n else "")

def main():
    # ---- IMPORT your env creation exactly as you do in training ----
    # You must replace this with your real builder.
    #
    # Example patterns:
    #   from citylearn_safe.make_env import make_env
    #   env = make_env()
    #
    # or:
    #   from citylearn_safe.env_factory import build_env
    #   env = build_env(...)
    #
    # ----------------------------------------------------------------
    from citylearn_safe.make_env import make_env  # <-- CHANGE THIS LINE IF NEEDED
    env = make_env()

    print("="*90)
    print("TOP ENV:", type(env).__name__)
    try:
        od = env.observation_space.shape[0]
    except Exception:
        od = env.observation_space[0].shape[0]
    try:
        ad = env.action_space.shape[0]
    except Exception:
        ad = env.action_space[0].shape[0]
    print("obs_dim:", od, "act_dim:", ad)
    print("="*90)

    city = unwrap_citylearn(env)
    print("[UNWRAP] CityLearnEnv found:", bool(city))

    obs, info = env.reset()
    obs = np.asarray(obs, dtype=float)

    # Try to read obs SoC indices from env if exposed (your wrapper has _soc_idx_obs)
    soc_idx_obs = getattr(env, "_soc_idx_obs", None)
    if soc_idx_obs is None:
        soc_idx_obs = getattr(env, "_soc_idx", None)  # older name
    print("\n[IDX] obs soc indices:", soc_idx_obs)

    print("\n[RESET]")
    print("  info.cost               :", info.get("cost"))
    print("  cost_building_soc       :", info.get("cost_building_soc"))
    print("  cost_ev_departure       :", info.get("cost_ev_departure"))
    print("  ev_departure_kwh(total) :", info.get("ev_departure_deficit_kwh"))
    print("  debug_obs_soc_mean      :", info.get("debug_obs_soc_mean"))
    print("  debug_state_soc_mean    :", info.get("debug_state_soc_mean"))

    # Rollout
    T = 48
    print("\nstep | obs_soc(min/max/mean) | state_soc(min/max/mean) | build_cost | ev(total/avoid/unavoid,deps) | info.cost | OK?")
    print("-"*140)

    for t in range(1, T+1):
        # Use zero action by default (change to random to test responsiveness)
        try:
            a = np.zeros(env.action_space.shape, dtype=float)
        except Exception:
            a = np.zeros(ad, dtype=float)

        obs, rew, term, trunc, info = env.step(a)
        obs = np.asarray(obs, dtype=float)

        # ---- OBS SoC stats (may be zeros in your normalized wrapper) ----
        if soc_idx_obs:
            obs_soc = obs[np.array(soc_idx_obs, dtype=int)]
            obs_min, obs_max, obs_mean = float(np.min(obs_soc)), float(np.max(obs_soc)), float(np.mean(obs_soc))
        else:
            obs_min = obs_max = obs_mean = float("nan")

        # ---- INTERNAL STATE SoC stats from info debug fields ----
        st_min = float(info.get("debug_state_soc_min", float("nan")))
        st_max = float(info.get("debug_state_soc_max", float("nan")))
        st_mean = float(info.get("debug_state_soc_mean", float("nan")))

        # ---- Costs ----
        c_build = float(info.get("cost_building_soc", 0.0))
        c_ev = float(info.get("cost_ev_departure", 0.0))
        c_ev_a = float(info.get("cost_ev_departure_avoidable", 0.0))
        c_ev_u = float(info.get("cost_ev_departure_unavoidable", 0.0))
        deps = int(info.get("ev_departure_departures", 0))
        c_total = float(info.get("cost", 0.0))

        # ---- Expected total cost check ----
        # If CITYLEARN_INCLUDE_EV_COST=1, expect total == build + ev_total
        include_ev = bool(int(os.environ.get("CITYLEARN_INCLUDE_EV_COST", "1")))
        expected = c_build + (c_ev if include_ev else 0.0)
        ok = np.isclose(c_total, expected, atol=1e-8)

        print(
            f"{t:4d} | "
            f"{obs_min:6.3f}/{obs_max:6.3f}/{obs_mean:6.3f} | "
            f"{st_min:6.3f}/{st_max:6.3f}/{st_mean:6.3f} | "
            f"{c_build:9.6f} | "
            f"{c_ev:7.4f}/{c_ev_a:7.4f}/{c_ev_u:7.4f},{deps:2d} | "
            f"{c_total:8.6f} | "
            f"{'OK' if ok else 'MISMATCH'}"
        )

        if term or trunc:
            print("\n[EPISODE END] term=", term, "trunc=", trunc)
            break

    print("\n[OK] Debug rollout complete.")

if __name__ == "__main__":
    main()
