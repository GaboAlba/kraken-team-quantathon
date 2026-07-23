---
name: networkx
description: Graph construction and algorithms with NetworkX. Use when working with the electrical-grid graph in src/graph.py and src/visualize.py, or reading/analyzing data/grid_cr.json.
---

# networkx

Graph library underpinning this project's data model. `src/graph.py` builds a
weighted `nx.Graph` of the Costa Rican grid and serializes it to `data/grid_cr.json`;
`src/visualize.py` draws it. Follow the existing patterns when extending them.

## Project conventions (match these)

- The grid is an **undirected** `nx.Graph`. Nodes are substation ids (normalized,
  lowercase names); edges carry `weight`, `voltaje`, `circuito`.
- Node attributes used across the codebase: `nombre`, `provincia`, `canton`,
  `x`, `y` (lon/lat), `frontera` (bool — border/interconnection node).
- **Determinism is required.** Any node ordering, seed, or tie-break must be
  reproducible: sort node ids, use fixed `seed=` in layouts, break ties alphabetically
  (`max(..., key=lambda n: (metric, n))`). This is a hard convention here.

## Common operations in this codebase

```python
import networkx as nx

G.add_node(node_id, **attrs)
G.add_edge(u, v, weight=w, voltaje=v_kv, circuito=name)

G.has_edge(u, v)                       # collapse parallel lines: sum weights
G.degree(n, weight="weight")           # weighted degree (used for BFS seed)

# largest connected component (deterministic tie-break)
comp = max(nx.connected_components(H), key=lambda c: (len(c), min(c)))
H = H.subgraph(comp).copy()            # ALWAYS .copy() a subgraph before mutating

# connected BFS-limited growth from a seed
for _, node in nx.bfs_edges(H, seed): ...

# independent cycles (Max-Cut non-triviality indicator)
cycles = H.number_of_edges() - H.number_of_nodes() + nx.number_connected_components(H)

nx.is_connected(H)                     # used in tests to validate subgraphs
```

## For Max-Cut / QAOA

- Iterate `G.edges(data=True)` to emit the weighted ZZ terms (see the `pytket` skill).
- Map string node ids to qubit indices via a sorted, stable index map.
- More independent cycles ⇒ a more interesting (non-tree) Max-Cut instance; the
  default subgraph mode is chosen specifically to preserve cycles.

## Gotchas

- `G.subgraph(...)` returns a **read-only view**; call `.copy()` before adding/
  changing anything (the codebase always does).
- `nx.spring_layout` is random — pass `seed=` (the code uses `seed=42`) for repeatable figures.
