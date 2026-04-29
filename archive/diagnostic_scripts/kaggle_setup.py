"""Run this first on Kaggle to set up dependencies."""
import sys, os

# Add vendor deps to path (bundled omnisafe, citylearn, cvxpylayers)
vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor_deps")
if os.path.isdir(vendor):
    sys.path.insert(0, vendor)
    print(f"Added vendor_deps to path: {vendor}")

# Add project root to path
root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

# Verify imports
try:
    import omnisafe; print(f"  omnisafe OK ({omnisafe.__version__})")
except Exception as e: print(f"  omnisafe FAIL: {e}")
try:
    import citylearn; print(f"  citylearn OK")
except Exception as e: print(f"  citylearn FAIL: {e}")
try:
    import cvxpylayers; print(f"  cvxpylayers OK")
except Exception as e: print(f"  cvxpylayers FAIL: {e}")
try:
    import scs; print(f"  scs OK")
except Exception as e: print(f"  scs FAIL: {e}")

print("\nSetup complete. Ready to train.")
