---
name: selene
description: Emulate hybrid quantum-classical programs locally with Selene (selene-sim), Quantinuum's emulation framework. Use when running Guppy/HUGR programs without cloud access, choosing a simulator engine (Quest, Stim), adding noise models, or debugging shot results in this Quantathon project.
---

# Selene (selene-sim)

Quantinuum's local emulation framework for hybrid programs. It runs compiled
Guppy/HUGR programs (see the `guppylang` skill) on pluggable simulator engines —
no network or Nexus account needed. Package: `selene-sim`, module `selene_sim`.

## High-level path: emulate a Guppy kernel

```python
from guppylang import guppy
from guppylang.std.quantum import qubit, h, cx, measure
from guppylang.std.builtins import result

@guppy
def main() -> None:
    q0 = qubit()
    q1 = qubit()
    h(q0)
    cx(q0, q1)
    result("c0", measure(q0))    # result(tag, value) is how shots report output
    result("c1", measure(q1))

res = main.emulator(n_qubits=2).with_seed(7).run()   # EmulatorResult
```

Use `result("tag", value)` inside the kernel for every value you want back;
unmeasured/unreported values are lost.

## Direct path: build + run_shots (more control)

```python
from selene_sim import build, Quest
from hugr.qsystem.result import QsysResult

runner = build(main.compile())                  # HUGR -> executable emulation
qres = QsysResult(runner.run_shots(Quest(), n_qubits=2, n_shots=100))
counts = qres.collated_counts()                 # Counter of tagged outcomes
# e.g. {(('c0','0'), ('c1','0')): 52, (('c0','1'), ('c1','1')): 48}
```

## Simulator engines

- `Quest()` — statevector; exact, any gate set; cost grows 2^n (fine for ≤12 qubits here).
- `Stim()` — stabilizer; huge circuits but **Clifford-only** (no arbitrary QAOA angles).
- `Coinflip()` — no quantum state, random measurement bits; for control-flow tests.
- `ClassicalReplay(measurements=[...])` — force measurement outcomes per shot to
  drive a specific branch; for debugging adaptive circuits.

## Noise and runtimes

`selene_sim` also exposes error models (`DepolarizingErrorModel`,
`SimpleLeakageErrorModel`, `IdealErrorModel` default) and runtimes
(`SimpleRuntime`, `SoftRZRuntime`) to approximate hardware behavior.

## When to use in this project

Selene is the local runner for Guppy kernels (adaptive QAOA mixers,
mid-circuit measurement). For plain pytket circuits use `AerBackend` (pytket
skill); to validate on Quantinuum's cloud emulators/hardware use the `qnexus`
skill.

## Gotchas

- Guppy kernels must live in a **real .py file** — compiling from a REPL/stdin
  fails with `OSError: could not get source code`.
- `n_qubits` must cover every allocation in the kernel or the run fails.
- Fix seeds (`.with_seed(n)`) — this repo's convention is deterministic,
  reproducible runs.
- Quest counts are keyed by tuples of `(tag, value)` pairs, not bitstrings;
  aggregate with `QsysResult.collated_counts()` / `register_counts()`.
