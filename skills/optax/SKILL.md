---
name: optax
description: Gradient-based optimization with Optax (JAX). Use for gradient-descent tuning of QAOA/variational parameters when an analytic or autodiff gradient of the objective is available.
---

# optax

Composable gradient-processing / optimization library built on JAX. Use it when
the variational objective (e.g. a simulated, differentiable QAOA expectation) has
gradients available via JAX autodiff — an alternative to gradient-free
`scipy.optimize` (see the `scipy` skill).

## Standard training loop

```python
import jax
import jax.numpy as jnp
import optax

def objective(params):          # must be JAX-differentiable; returns a scalar
    return -expected_cut(params) # minimize negative to maximize the cut

optimizer = optax.adam(learning_rate=1e-2)
params = jnp.array([0.7, 0.8])          # e.g. [gamma, beta]
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state):
    loss, grads = jax.value_and_grad(objective)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for _ in range(200):
    params, opt_state, loss = step(params, opt_state)
```

## Key patterns

- Optimizers are pure `(init, update)` transforms: `init(params) -> state`,
  `update(grads, state, params) -> (updates, new_state)`.
- **Always** apply updates with `optax.apply_updates(params, updates)` — `update`
  returns updates to add, not the new params directly.
- Compose transforms with `optax.chain(...)` (e.g. `clip_by_global_norm` + `adam`).
- Schedules (`optax.exponential_decay`, etc.) can replace a scalar learning rate.

## When NOT to use

If the objective is only measurable through hardware/simulator **shots** (noisy,
non-differentiable), Optax's gradients don't apply — use gradient-free
`scipy.optimize.minimize` instead. Optax fits differentiable state-vector simulation.

## Gotchas

- Objective and params must be JAX arrays / traceable; Python side effects break `jit`.
- Optax minimizes along the negative gradient; negate any quantity you want to maximize.
