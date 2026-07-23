"""Classical Max-Cut baselines adapted to this repository's graph format.

The project stores the electrical subgraph in ``data/grid_cr.json`` as a
NetworkX graph with edge attributes including ``weight``. The baselines here
use that representation directly, so they can be evaluated against the real
pipeline data instead of an external toy implementation.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Dict, Hashable, Tuple

import cvxpy as cp
import networkx as nx
import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src import qubo

Partition = Dict[Hashable, int]


def maxcut_value(G: nx.Graph, partition: Partition) -> float:
    """Return the weighted Max-Cut value for a partition assignment.

    A cut edge is one whose endpoints land on different sides of the partition.
    The returned value is the total weight of all cut edges.
    """
    value = 0.0
    for u, v, data in G.edges(data=True):
        if partition.get(u, 0) != partition.get(v, 0):
            value += float(data.get("weight", 1.0))
    return float(value)


def load_project_graph(path: str | Path = qubo.DEFAULT_INPUT) -> nx.Graph:
    """Load the repository's grid graph for experiments using the project API."""
    return qubo.load_graph(Path(path))


def brute_force_maxcut(G: nx.Graph) -> Tuple[Partition, float]:
    """Exact Max-Cut for small graphs via exhaustive enumeration.

    The repository's project graph is small enough for this to remain practical.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n > 22:
        raise ValueError(
            f"Fuerza bruta no es práctica para {n} nodos (2^{n} particiones)."
        )

    best_partition: Partition = {}
    best_value = float("-inf")
    for bits in itertools.product([0, 1], repeat=max(n - 1, 0)):
        assignment = (0,) + bits if n > 0 else ()
        partition = {nodes[i]: int(assignment[i]) for i in range(n)}
        value = maxcut_value(G, partition)
        if value > best_value:
            best_value = value
            best_partition = partition
    return best_partition, best_value


def greedy_maxcut(G: nx.Graph, seed: int = 0) -> Tuple[Partition, float]:
    """Greedy deterministic heuristic for Max-Cut.

    Nodes are visited in a reproducible random order. Each node is assigned to
    the side that maximizes the immediate increase in cut weight relative to the
    already assigned neighbors.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    order = list(nodes)
    rng.shuffle(order)

    partition: Partition = {}
    for node in order:
        gain0 = 0.0
        gain1 = 0.0
        for neighbor in G.neighbors(node):
            if neighbor in partition:
                weight = float(G[node][neighbor].get("weight", 1.0))
                if partition[neighbor] == 0:
                    gain1 += weight
                else:
                    gain0 += weight
        partition[node] = 0 if gain0 >= gain1 else 1

    value = maxcut_value(G, partition)
    return partition, value


def goemans_williamson(
    G: nx.Graph, n_rounding_trials: int = 50, seed: int = 42
) -> Tuple[Partition, float]:
    """Goemans-Williamson SDP-based baseline adapted to the project graph.

    The solver uses a positive semidefinite matrix relaxation and randomized
    hyperplane rounding to approximate the Max-Cut optimum.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    W = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        i, j = idx[u], idx[v]
        W[i, j] = w
        W[j, i] = w

    X = cp.Variable((n, n), PSD=True)
    objective_terms = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] != 0:
                objective_terms.append(W[i, j] * (1 - X[i, j]) / 2)
    objective = cp.Maximize(cp.sum(objective_terms))
    constraints = [cp.diag(X) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"La solución SDP no es óptima: estado={problem.status}.")
    if X.value is None:
        raise RuntimeError("La solución SDP no se obtuvo correctamente.")

    X_val = np.asarray(X.value, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(X_val)
    eigvals = np.clip(eigvals, 0, None)
    V = eigvecs @ np.diag(np.sqrt(eigvals))

    rng = np.random.default_rng(seed)
    best_partition: Partition = {}
    best_value = float("-inf")
    for _ in range(n_rounding_trials):
        r = rng.normal(size=n)
        norm_r = np.linalg.norm(r)
        if norm_r == 0:
            continue
        r /= norm_r
        signs = np.sign(V @ r)
        signs[signs == 0] = 1
        partition = {nodes[i]: int(signs[i] > 0) for i in range(n)}
        value = maxcut_value(G, partition)
        if value > best_value:
            best_value = value
            best_partition = partition

    return best_partition, best_value


if __name__ == "__main__":
    G = load_project_graph()

    bf_partition, bf_value = brute_force_maxcut(G)
    print(f"Fuerza bruta (óptimo):      corte = {bf_value:.3f}")

    gr_partition, gr_value = greedy_maxcut(G, seed=0)
    print(f"Greedy:                     corte = {gr_value:.3f}")

    gw_partition, gw_value = goemans_williamson(G, n_rounding_trials=50, seed=42)
    print(f"Goemans-Williamson:         corte = {gw_value:.3f}")
