# vendor_deps/

Vendored (locally patched) dependencies. These are checked into the repository
because the project requires specific modifications that are not available
upstream.

## Contents

### omnisafe/

Patched OmniSafe framework, based on OmniSafe 0.4.x. Custom modifications
include:

- Multi-constraint Lagrangian support (per-constraint lambda, cost critics)
- Device-mismatch fixes for mixed CPU/GPU training
- Custom algorithm registrations (PPOLagMulti, PPOLagGradS)

### citylearn/

CityLearn smart grid environment with custom patches for V2G (vehicle-to-grid)
action spaces, EV charger modeling, and multi-building central agent mode.

### cvxpylayers/

Differentiable convex optimization layers. Used in the experimental SE-RL
action projection method (safety-constrained action clipping via convex
programs). Kept for reference; not required for standard PPOLagMulti training.

### scs/

Splitting Conic Solver -- the numerical backend for CVXPyLayers. Required only
when `cvxpylayers` is active.

## Why Vendored?

Standard `pip install` versions of these packages do not include the patches
this project depends on. Installing upstream versions will break multi-constraint
training, device handling, or environment registration. Always use the vendored
copies by ensuring `vendor_deps/` is on `sys.path` (handled automatically by
the training scripts).
