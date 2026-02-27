#!/usr/bin/env python3
"""
PSF Smoke Test: verify full pipeline works.
    CityLearnEnv -> CityLearnSafetyEnvV3 -> PredictiveSafetyFilterWrapper
Usage: python3 scripts/run_psf_smoke_test.py
"""
import os, sys, json, numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

SCHEMA_PATH = os.environ.get("CITYLEARN_SCHEMA",
    os.path.join(PROJECT_ROOT, "data",
                 "citylearn_challenge_2022_phase_all_plus_evs", "schema.json"))

def normalize_schema(path):
    with open(path) as f: schema = json.load(f)
    base = os.path.dirname(path)
    root = schema.get("root_directory", "")
    if not root or root in (".", "./"): root = base
    elif not os.path.isabs(root): root = os.path.normpath(os.path.join(base, root))
    schema["root_directory"] = root
    return schema

def main():
    print("=" * 60)
    print("PSF SMOKE TEST")
    print("=" * 60)
    os.environ.setdefault("CITYLEARN_KPI_RUN_NAME", "psf_smoke_test")
    os.environ.setdefault("PSF_VERBOSE", "1")

    from citylearn.citylearn import CityLearnEnv
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.psf.psf_wrapper import PredictiveSafetyFilterWrapper

    schema = normalize_schema(SCHEMA_PATH)
    base_env = CityLearnEnv(schema=schema)
    safety_env = CityLearnSafetyEnvV3(base_env)
    psf_env = PredictiveSafetyFilterWrapper(safety_env, horizon=24,
                                             correction_mode="heuristic", verbose=2)

    # 1. Reset
    obs, info = psf_env.reset(seed=42)
    print(f"\n✅ reset ok | type={type(obs).__name__}")

    # 2. Run 50 steps
    n_steps = 50
    rewards, interventions, errors = [], 0, []
    for i in range(n_steps):
        action = [sp.sample() for sp in psf_env.action_space] \
            if isinstance(psf_env.action_space, list) else psf_env.action_space.sample()
        try:
            obs, r, term, trunc, info = psf_env.step(action)
            rewards.append(r)
            if info.get("psf_any_intervention"): interventions += 1
            if term or trunc: obs, info = psf_env.reset(seed=42 + i)
        except Exception as e:
            errors.append((i, str(e))); break

    if errors:
        print(f"\n❌ step FAILED at {errors[0][0]}: {errors[0][1]}"); sys.exit(1)
    print(f"\n✅ step ok ({n_steps} steps) | avg_r={np.mean(rewards):.4f} | "
          f"interventions={interventions}/{n_steps}")

    # 3. ActionMap consistency
    am = psf_env._get_action_map()
    inner_ev = sorted(getattr(safety_env, "_ev_charger_action_indices", []))
    psf_ev = sorted(am.ev_gidx)
    match = (inner_ev == psf_ev)
    print(f"\n{'✅' if match else '❌'} ActionMap consistent | inner={inner_ev} psf={psf_ev}")

    # 4. EV states
    ev_states = psf_env._extract_ev_states()
    connected = [s for s in ev_states if s.connected]
    print(f"\n✅ EV states extracted: {len(ev_states)} chargers, {len(connected)} connected")

    # 5. Constraint prediction
    flat = psf_env._flatten_action(
        [sp.sample() for sp in psf_env.action_space]
        if isinstance(psf_env.action_space, list) else psf_env.action_space.sample())
    pred = psf_env._predict_constraints(flat)
    print(f"\n✅ Constraint prediction works | ev_viol={len(pred.ev_violations)} "
          f"batt_viol={len(pred.battery_soc_violations)}")

    # 6. Roundtrip
    if isinstance(psf_env.action_space, list):
        orig = [np.random.randn(*sp.shape).astype(np.float32) for sp in psf_env.action_space]
        flat = psf_env._flatten_action(orig)
        unflat = psf_env._unflatten_action(flat)
        rt_ok = len(unflat) == len(orig) and all(
            np.allclose(a, b, atol=1e-5) for a, b in zip(orig, unflat))
    else:
        orig = np.random.randn(*psf_env.action_space.shape).astype(np.float32)
        flat = psf_env._flatten_action(orig)
        unflat = psf_env._unflatten_action(flat)
        rt_ok = np.allclose(orig, unflat, atol=1e-5)
    print(f"\n{'✅' if rt_ok else '❌'} Flatten/unflatten roundtrip: {rt_ok}")

    all_ok = match and rt_ok and not errors
    print(f"\n{'=' * 60}")
    print(f"{'✅' if all_ok else '❌'} PSF smoke test {'PASSED' if all_ok else 'FAILED'}")
    print(f"{'=' * 60}")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__": main()
