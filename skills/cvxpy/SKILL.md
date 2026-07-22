---
name: cvxpy
description: Convex optimization modeling with CVXPY. Use for the Goemans-Williamson SDP relaxation of Max-Cut (a classical benchmark for the QAOA results) and other convex formulations.
---

# cvxpy

Python modeling language for convex optimization. In this project its primary use
is the **SDP relaxation of Max-Cut** (Goemans–Williamson), which provides a
classical upper bound / benchmark to compare against the QAOA cut values.

## Max-Cut SDP relaxation

Given the weighted Laplacian-style formulation, maximize `(1/4) * sum_{ij} w_ij (1 - X_ij)`
over a PSD matrix `X` with unit diagonal:

```python
import cvxpy as cp
import numpy as np

# W: symmetric weight matrix (n x n) from grid_cr.json edges
n = W.shape[0]
X = cp.Variable((n, n), symmetric=True)

constraints = [X >> 0]                       # X positive semidefinite
constraints += [X[i, i] == 1 for i in range(n)]

objective = cp.Maximize(0.25 * cp.sum(cp.multiply(W, (1 - X))))
prob = cp.Problem(objective, constraints)
prob.solve()                                 # add solver=cp.SCS for large/PSD problems

print(prob.value)                            # SDP upper bound on the max cut
```

Then round `X` (e.g. random hyperplane on its Cholesky/eigenvector factors) to get
an actual cut to compare with QAOA.

## Key patterns

- Build the weight matrix `W` from `grid_cr.json`: `W[i, j] = W[j, i] = edge weight`
  using a consistent `node_id -> index` map (sort node ids for determinism, as the
  rest of the codebase does).
- `X >> 0` expresses the PSD (semidefinite) constraint; the SDP needs a conic solver.
- Check `prob.status == "optimal"` before trusting `prob.value`.

## Gotchas

- The default solver may not handle SDPs well; pass `solver=cp.SCS` (or `cp.CVXOPT`)
  for PSD constraints, and `pip install` the solver if missing.
- CVXPY enforces convexity at construction — a "DCP" error means the expression
  isn't provably convex; reformulate rather than fighting it.
