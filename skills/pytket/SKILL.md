---
name: pytket
description: Build, compile, and run quantum circuits with pytket (TKET). Use when constructing QAOA/Max-Cut circuits, compiling for a backend, or reading measurement counts in this Quantathon project.
---

# pytket (TKET)

Quantum SDK from Quantinuum. In this repo it consumes the weighted subgraph in
`data/grid_cr.json` (built by `src/graph.py`) to build **Max-Cut / QAOA** circuits.
Requires pytket ≥ 2.x on Python 3.10+.

## Core objects

- `Circuit(n_qubits, n_bits)` — the unit of work. Qubits/bits are indexed from 0.
- `Backend` — a device or simulator. You **compile** a circuit for a backend, then run.
- `BackendResult` — returned from a run; read counts / distributions from it.

## Build a circuit

```python
from pytket import Circuit

circ = Circuit(n)                 # n qubits, no classical bits
circ.H(q)                         # single-qubit gates: H, X, Rx(angle,q), Rz(angle,q)
circ.CX(control, target)          # two-qubit
circ.Rz(2 * gamma, j)             # angles are in half-turns (multiples of pi)
circ.measure_all()                # add a classical bit per qubit and measure
```

Gate angles in pytket are in **half-turns** (units of pi), not radians — `Rz(0.5, q)`
is a pi/2 rotation.

## QAOA Max-Cut layer (maps directly to this project's graph)

Iterate over `grid_cr.json` edges; each node is a qubit. The ZZ interaction per
edge is the standard `CX–Rz–CX` decomposition, weighted by the edge `weight`:

```python
def qaoa_maxcut_layer(circ, edges, gamma, beta, n):
    for q in range(n):
        circ.H(q)
    for (i, j, w) in edges:                 # w = edge "weight" from grid_cr.json
        circ.CX(i, j)
        circ.Rz(2 * gamma * w, j)
        circ.CX(i, j)
    for q in range(n):
        circ.Rx(2 * beta, q)
    return circ
```

Map the string node ids from `grid_cr.json` to integer qubit indices first
(e.g. `idx = {node_id: i for i, node_id in enumerate(sorted(nodes))}`).

## Compile and run

```python
from pytket.extensions.qiskit import AerBackend   # local simulator

backend = AerBackend()
compiled = backend.get_compiled_circuit(circ)      # ALWAYS compile before running
result = backend.run_circuit(compiled, n_shots=1000)
counts = result.get_counts()                       # {(0,1,1,...): 384, ...}
```

- **Always** call `backend.get_compiled_circuit(...)` (or `backend.compile_circuit`)
  before `run_circuit` — raw circuits may use gates the backend can't execute.
- Backends live in extensions installed separately: `pip install pytket-qiskit`
  (Aer simulator), `pip install pytket-quantinuum` (Quantinuum compile/emulate).
- For Quantinuum **hardware** submission the newer path is the `qnexus` package;
  `pytket-quantinuum` (≥0.56) is compile/emulation only.

## Gotchas

- Counts are keyed by tuples of measured bit values ordered by the circuit's bits.
- Keep the qubit-count small — the default subgraph is ≤12 nodes for a reason.
- Convert QAOA angles consistently; a wrong factor of 2 or pi silently gives bad cuts.
