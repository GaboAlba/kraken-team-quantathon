---
name: scipy
description: Scientific computing and classical optimization with SciPy. Use for scipy.optimize.minimize to tune QAOA angles, sparse linear algebra, and reference (exact/brute-force) Max-Cut baselines.
---

# scipy

General scientific computing. In this project its main role is the **classical
outer loop** of QAOA (optimizing `gamma`/`beta` angles) and computing exact
baselines to score the quantum results.

## Optimizing QAOA angles

Wrap the "build circuit → run → compute expected cut value" pipeline in a scalar
objective and minimize its negative (SciPy minimizes):

```python
from scipy.optimize import minimize

def neg_cut_value(params):
    gamma, beta = params
    counts = run_qaoa(gamma, beta)      # your pytket/guppy runner
    return -expected_cut(counts, edges) # maximize cut => minimize negative

res = minimize(
    neg_cut_value,
    x0=[0.7, 0.8],
    method="COBYLA",     # gradient-free; robust for noisy shot-based objectives
    options={"maxiter": 100},
)
best_gamma, best_beta = res.x
```

- Prefer **gradient-free** methods (`COBYLA`, `Nelder-Mead`, `Powell`) for
  shot-noisy objectives — finite-difference gradients are unreliable with sampling.
- For smooth analytic objectives use `L-BFGS-B` and pass `jac`.
- Inspect `res.success`, `res.fun`, `res.nfev`; set `maxiter` to bound backend calls.

## Other useful modules

- `scipy.sparse` / `scipy.sparse.linalg` — sparse adjacency/Laplacian for larger grids.
- `scipy.spatial.distance` — geographic distances from node `x`/`y` coordinates.

## Gotchas

- `minimize` **minimizes**; negate any value you want to maximize (like a cut).
- The objective must be deterministic enough to converge — average over enough
  shots or fix a seed, or the optimizer chases noise.
