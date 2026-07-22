---
name: guppylang
description: Write hybrid quantum-classical programs in Guppy (guppylang), a Python-embedded language that compiles to HUGR. Use when expressing quantum kernels with the @guppy decorator, linear qubit types, mid-circuit measurement, and adaptive control.
---

# guppylang (Guppy)

Python-embedded quantum programming language from Quantinuum. Functions marked
`@guppy` are **compiled** (to the HUGR IR), not run by the Python interpreter.
Guppy enforces **linear types**: a qubit has a single owner and cannot be silently
copied or dropped.

## Basic kernel

```python
from guppylang import guppy
from guppylang.std.quantum import qubit, h, cx, measure, x, z

@guppy
def teleport(src: "qubit @ owned", tgt: "qubit") -> None:
    tmp = qubit()          # allocate a fresh qubit
    h(tmp)
    cx(tmp, tgt)
    cx(src, tmp)
    h(src)
    if measure(src):       # classical control from a mid-circuit measurement
        z(tgt)
    if measure(tmp):
        x(tgt)
```

- Gates and `qubit`/`measure` come from `guppylang.std.quantum`.
- `q @ owned` means the function **consumes** the qubit (takes ownership); a plain
  `qubit` parameter is borrowed and must remain valid for the caller.
- `measure(q)` performs a real measurement and returns a `bool`; use it in `if`
  for adaptive / branching circuits.

## Type-check and compile

```python
teleport.check()      # static type-check only, no execution
hugr = teleport.compile()   # compile to HUGR for optimization / emulation / hardware
```

Run type-checking early and often — most Guppy bugs are linearity/ownership errors
caught statically, not at runtime.

## Useful decorators & flags

- `@guppy.comptime` — compile-time (classical) function.
- `@guppy.struct` / `@guppy.enum` — user-defined types usable inside kernels.
- `@guppy(...)` protocol flags for gate definitions: `unitary`, `control`, `dagger`,
  `power` auto-derive adjoint/controlled/powered variants.

## When to use in this project

Guppy is the alternative to raw pytket circuits when the Max-Cut/QAOA experiment
needs **mid-circuit measurement or classical feedback** (e.g. adaptive mixers).
For plain fixed-angle QAOA layers, pytket `Circuit` is simpler — see the `pytket` skill.

## Gotchas

- Never reuse a qubit after it's been consumed (`@owned`) or measured — the type
  checker rejects it; restructure to allocate a fresh `qubit()`.
- The API is pre-1.0 and moves quickly; confirm imports against the installed
  version (`import guppylang; guppylang.__version__`) before assuming a symbol exists.
