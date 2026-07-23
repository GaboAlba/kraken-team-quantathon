# AGENTS.md

Guidance for AI coding agents working in this repository. This is a Quantathon
project (Challenge #1) that models the Costa Rican national electrical grid as a
weighted graph and prepares Max-Cut / QAOA instances from it.

## Environment & commands

The project pins dependencies in `requirements.txt` and expects a local `.venv`.

- Create the environment: `.\setup_venv.ps1` (Windows) or `./setup_venv.sh` (macOS/Linux).
  Both create `.venv`, upgrade pip, and install `requirements.txt`.
- Activate: `.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate`.
- Run the full test suite: `pytest` (from the repo root).
- Run a single test file / test: `pytest tests/test_graph.py`,
  `pytest tests/test_graph.py::test_build_national_graph_basico`.
- Run the pipeline end-to-end: `python -m src.graph` (writes `data/grid_cr.json`).
- Build the Max-Cut QUBO: `python -m src.qubo` (reads `data/grid_cr.json`, writes `data/qubo_cr.json`).
- Refresh the ICE data snapshot: `python -m src.ice_data`.
- Regenerate figures: `python -m src.visualize` (writes to `figures/`).

When adding a dependency, install it in the activated venv **and** add it to
`requirements.txt` so teammates pick it up on next setup.

## Architecture

The data pipeline (Task A) is a linear flow across `src/`:

```
ICE ArcGIS API ──> data/raw/*.geojson ──> national NetworkX graph ──> subregion ──> data/grid_cr.json
   ice_data.py         (snapshot)              graph.py                graph.py         graph.py
```

- `src/ice_data.py` — downloads two ICE ArcGIS layers (`Subestaciones` → nodes,
  `LineasDeTransmision` → edges) and writes a static **snapshot** to `data/raw/`,
  plus `data/raw/source.json` for provenance. The rest of the pipeline reads the
  snapshot, never the live service, so results are reproducible.
- `src/graph.py` — the core. Parses substations into nodes and derives edges from
  each line's `Circuito` field (`"SubA-SubB"`). Builds the national graph
  (`build_national_graph`), extracts a small connected subregion
  (`extract_subregion`, default ≤12 nodes), and serializes to `grid_cr.json`
  (`to_json` / `save_graph`). `build()` orchestrates the whole flow.
- `src/weights.py` — interchangeable edge-weight schemes; see conventions below.
- `src/qubo.py` — Task B. Reads `data/grid_cr.json`, recomputes edge weights with
  the `generation_inverted` scheme (sign-inverted generation weights: critical
  lines become the largest positive), and builds the QUBO (`build_qubo`) with a
  **minimize-cut** objective so the fault-zone boundary avoids critical lines,
  plus a quadratic **generator-spread** penalty (keeps generators on both sides
  of the cut) and a **balance** penalty, registered in `qubo.PENALTIES` (same
  registry convention as `weights.SCHEMES`). Emits both QUBO and Ising forms to
  `data/qubo_cr.json` (`to_json` / `save_qubo`); `build()` orchestrates it. Pass
  `maximize_cut=True` (e.g. with `kv`) for the classic max-cut sense.
- `src/visualize.py` — renders the national grid and the chosen subregion to PNGs.

Edge weight = how critical a transmission line is to cut for fault-zone
partitioning; the subgraph is fed to Max-Cut / QAOA, which is why cycle count
(non-triviality) matters throughout.

## Skills

Package-specific usage guidance lives in `skills/<package>/SKILL.md`, one per key
dependency: `pytket`, `guppylang`, `qnexus`, `selene` (quantum), and `scipy`,
`optax`, `cvxpy`, `networkx` (scientific/optimization). Consult the relevant
skill before writing code against that library.

## Key conventions

- **Language: English only.** Write everything in English — docstrings, comments,
  identifiers, and JSON/metadata keys — so the code is readable by anyone.
- **Weight schemes are a registry:** every scheme is a pure function
  `fn(voltage, length_m, gens_u=(), gens_v=()) -> float`, registered in
  `weights.SCHEMES` with a string key; `weights.DEFAULT_SCHEME` selects the
  default (`"generation_inverted"`, generator-aware and mostly positive). Add new
  schemes by adding a function and a `SCHEMES` entry — don't hardcode weights elsewhere.
- **Reproducibility & determinism:** the pipeline only reads the static snapshot;
  subregion selection uses deterministic tie-breaks (alphabetical node order, BFS
  from the highest weighted-degree seed, fixed layout seeds) so runs are repeatable.
  Preserve this determinism when modifying selection or ordering.
- **Name normalization:** `graph.normalize_name` (lowercase, strip accents,
  drop `(...)` suffixes and a trailing bay digit) plus the `_ALIASES` map reconcile
  circuit endpoints with substation names. Route new name matching through it.
- **Border nodes:** circuit endpoints with no matching substation are added as
  `border=True` nodes (international ties, SIEPAC, industrial loads).
  The default "connectivity" subregion mode excludes them.
- **Parallel lines** between the same pair are collapsed: weights summed, highest
  `voltage` kept.
- **Imports:** modules import each other as `from src import graph` / `weights`;
  tests insert the repo root into `sys.path` before importing `src`.
- **Tests:** synthetic-data tests cover pure logic; tests decorated with the local
  `@real` marker (`pytest.mark.skipif`) run against the ICE snapshot and are skipped
  automatically when `data/raw/` is absent. Prefer synthetic fixtures for new logic.
- **Keep the validation notebook current:** whenever new code changes the pipeline
  (new/edited functions, weight formulas, node/edge attributes, serialization),
  update `notebooks/validation.ipynb` so each step can still be manually validated,
  then re-execute it end-to-end (`python -m jupyter nbconvert --to notebook --execute
  --inplace notebooks/validation.ipynb`) to confirm all cells and assertions pass.
