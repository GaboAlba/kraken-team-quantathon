# Kraken Team Quantathon Project

## Team Members

* Gabriel Alba Romero
* Juan C Lara

## Challenge
We are solving Challenge #1

## Documentation

Design docs for each major feature (most under [`docs/`](docs)):

* [Data pipeline & graph model](AGENTS.md) — how the ICE grid snapshot becomes the
  weighted subgraph in [`data/grid_cr.json`](data/grid_cr.json) (Task A; see the
  Architecture section of [`AGENTS.md`](AGENTS.md) and [`src/graph.py`](src/graph.py)).
* [QUBO / Max-Cut formulation (Task B)](docs/qubo.md) — design decisions behind
  `src/qubo.py`: the cut objective (minimize-cut), the sign-inverted
  `generation` weight scheme, and the generator-spread and balance penalties.
* [Cost Hamiltonian `H_C`](docs/hamiltonian.md) — the `QUBO → Ising → H_C` chain,
  the `CostHamiltonian` structure, its diagonal property, and its classical and
  quantum consumers.
* [QAOA solver (Task C)](docs/qaoa.md) — design decisions behind `src/qaoa.py`:
  what is fixed by the Graph + QUBO vs. the QAOA hyperparameters, the weighted
  Guppy 0.21 phase/mixer kernel, and the SciPy (COBYLA) optimizer.
* [Optimizers & comparative evaluation](docs/optimizers.md) — every optimizer
  (brute force, greedy, Goemans-Williamson, QAOA `p = 1…6`) and the parallel
  benchmark harness (`src/benchmark.py`, `notebooks/evaluation.ipynb`): growing
  grids, replicate runs, metrics, and figures.

Each doc includes rendered flow diagrams ([`docs/diagrams/`](docs/diagrams)).

## Pipeline at a glance

The project models the Costa Rican grid as a weighted graph and solves a
fault-zone Max-Cut on it, classically and with QAOA:

```
ICE ArcGIS API ─> data/raw/*.geojson ─> national graph ─> subregion ─> data/grid_cr.json
   ice_data.py        (snapshot)           graph.py         graph.py        graph.py
        │
        └─> QUBO + Ising + H_C ─> QAOA (Selene / Nexus Helios)  vs.  classical baselines
                qubo.py               qaoa.py / qaoa_nexus.py          classical_baselines.py
                                          └──────── comparative evaluation ────────┘
                                                          benchmark.py
```

| Stage | Module | Entry point |
| ----- | ------ | ----------- |
| Data snapshot | `src/ice_data.py` | `python -m src.ice_data` |
| Graph model (Task A) | `src/graph.py`, `src/weights.py` | `python -m src.graph` |
| QUBO / Ising / `H_C` (Task B) | `src/qubo.py` | `python -m src.qubo` |
| QAOA (Task C) | `src/qaoa.py` | `python -m src.qaoa` |
| QAOA on Nexus | `src/qaoa_nexus.py` | `notebooks/nexus_optimization.ipynb` |
| Classical baselines | `src/classical_baselines.py` | `python -m src.classical_baselines` |
| Comparative evaluation | `src/benchmark.py` | `notebooks/evaluation.ipynb` |
| Figures | `src/visualize.py` | `python -m src.visualize` |

Run the test suite with `pytest`. See [`AGENTS.md`](AGENTS.md) for the full
command reference and conventions.

## Setup

This project uses a Python virtual environment (`.venv`) so everyone develops against the same set of dependencies, listed in [`requirements.txt`](requirements.txt).

### Prerequisites
* Python 3.10+ installed and available on your PATH.

### Create the virtual environment

**Windows (PowerShell):**
```powershell
.\setup_venv.ps1
```
If script execution is blocked by your PowerShell policy, run:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
```

**macOS / Linux (bash):**
```bash
./setup_venv.sh
```

Either script will:
1. Create a `.venv` folder in the repo root (if it doesn't already exist).
2. Upgrade `pip`.
3. Install every package listed in `requirements.txt`.

### Activate the environment

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash):**
```bash
source .venv/bin/activate
```

### Adding new dependencies

When you need a new package, install it inside the activated venv, then add it to `requirements.txt` so the rest of the team picks it up next time they run the setup script.

### Key dependencies

* [`pytket`](https://tket.quantinuum.com/) — quantum SDK / circuit compilation
* [`guppylang`](https://github.com/CQCL/guppylang) — quantum programming language (Guppy)
* [`qnexus`](https://docs.quantinuum.com/nexus/) — Quantinuum Nexus client (cloud emulator / hardware)
* [`selene-sim`](https://github.com/CQCL/selene) — Selene quantum emulator
* [`scipy`](https://scipy.org/) — scientific computing (COBYLA optimizer)
* [`optax`](https://optax.readthedocs.io/) — gradient-based optimization
* [`cvxpy`](https://www.cvxpy.org/) — convex optimization (Goemans-Williamson SDP)
* [`networkx`](https://networkx.org/) — graph algorithms
* [`matplotlib`](https://matplotlib.org/) — figures and partition plots

