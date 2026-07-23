"""
classical_baselines.py
=======================
Métodos clásicos de referencia para Max-Cut, usados como línea base
para evaluar honestamente la calidad de QAOA:

1. `brute_force_maxcut`   -> óptimo exacto (solo viable para grafos pequeños,
                             usado como "E_optimo" en la razón de aproximación).
2. `greedy_maxcut`        -> heurística voraz local, muy rápida, cota inferior
                             de referencia simple.
3. `goemans_williamson`   -> relajación SDP (Goemans & Williamson, 1995) +
                             redondeo aleatorizado por hiperplano, con
                             garantía teórica de razón de aproximación
                             esperada >= 0.878 * OPT.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Dict, Hashable, Tuple

import cvxpy as cp
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import graph

Partition = Dict[Hashable, int]


def brute_force_maxcut(G: nx.Graph) -> Tuple[Partition, float]:
    """Encuentra el corte máximo exacto probando todas las 2^n particiones.

    Solo recomendado para n <= ~20 nodos por el costo exponencial.

    Parameters
    ----------
    G : networkx.Graph
        Grafo de entrada para Max-Cut.

    Returns
    -------
    tuple[Partition, float]
        Partición óptima y el valor del corte.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n > 22:
        raise ValueError(
            f"Fuerza bruta no es práctica para {n} nodos (2^{n} particiones). "
            "Usa esto solo para validación en grafos pequeños."
        )

    best_partition = {}
    best_value = float("-inf")
    # Por simetría del corte, basta fijar el primer nodo en 0
    for bits in itertools.product([0, 1], repeat=n - 1):
        assignment = (0,) + bits
        partition = {nodes[i]: assignment[i] for i in range(n)}
        value = maxcut_value(G, partition)
        if value > best_value:
            best_value, best_partition = value, partition

    return best_partition, best_value


def greedy_maxcut(G: nx.Graph, seed: int = 0) -> Tuple[Partition, float]:
    """Heurística voraz para Max-Cut.

    Procesa los nodos en un orden aleatorio reproducible y asigna cada uno al
    lado (0/1) que maximiza el aumento marginal del corte con respecto a los
    vecinos ya asignados.

    Parameters
    ----------
    G : networkx.Graph
        Grafo de entrada para Max-Cut.
    seed : int, optional
        Semilla para la aleatoriedad en el orden de procesado.

    Returns
    -------
    tuple[Partition, float]
        Partición construida y el valor del corte.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    order = list(nodes)
    rng.shuffle(order)

    partition: Partition = {}
    for node in order:
        gain0 = gain1 = 0.0
        for neighbor in G.neighbors(node):
            if neighbor in partition:
                w = G[node][neighbor].get("weight", 1.0)
                if partition[neighbor] == 0:
                    gain1 += w  # ponerlo en 1 corta la arista
                else:
                    gain0 += w
        partition[node] = 0 if gain0 >= gain1 else 1

    value = maxcut_value(G, partition)
    return partition, value


def maxcut_value(G: nx.Graph, partition: Partition) -> float:
    """Calcula el valor del Max-Cut para una partición dada."""
    value = 0.0
    for u, v, data in G.edges(data=True):
        if partition.get(u) != partition.get(v):
            value += data.get("weight", 1.0)
    return float(value)


def load_project_graph(path: str | Path | None = None) -> nx.Graph:
    """Carga el grafo de la subred del proyecto desde JSON.

    Busca `data/grid_cr.json` primero; si no existe, reconstruye la subred
    desde los snapshots locales raw usando `src.graph.build()`.
    """
    if path is None:
        path = ROOT / "data" / "grid_cr.json"
    path = Path(path)
    if path.exists():
        return graph.load_graph(path)
    return graph.build()


def goemans_williamson(
    G: nx.Graph, n_rounding_trials: int = 50, seed: int = 42
) -> Tuple[Partition, float]:
    """Algoritmo de Goemans-Williamson para Max-Cut.

    El algoritmo resuelve una relajación semidefinida positiva (SDP) y luego
    redondea el resultado con hiperplanos aleatorios.

    Parameters
    ----------
    G : networkx.Graph
        Grafo de entrada para Max-Cut.
    n_rounding_trials : int, optional
        Número de hiperplanos aleatorios a probar.
    seed : int, optional
        Semilla para la generación de hiperplanos aleatorios.

    Returns
    -------
    tuple[Partition, float]
        Partición obtenida y el valor del corte.
    """
   
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    W = np.zeros((n, n), dtype=float)
    for u, v, w in G.edges(data="weight"):
        w = float(w) if w is not None else 1.0
        i, j = idx[u], idx[v]
        W[i, j] = w
        W[j, i] = w

    # --- Paso 1: relajación SDP ---
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
        raise RuntimeError(
            f"La solución SDP no es óptima: estado={problem.status}."
        )

    if X.value is None:
        raise RuntimeError("La solución SDP no se obtuvo correctamente.")

    X_val = np.asarray(X.value, dtype=float)
    # Asegurar PSD numérica (puede haber pequeños errores numéricos)
    eigvals, eigvecs = np.linalg.eigh(X_val)
    eigvals = np.clip(eigvals, 0, None)
    V = eigvecs @ np.diag(np.sqrt(eigvals))
    # Cada fila de V corresponde al vector asociado a un nodo.

    # --- Paso 2: redondeo aleatorizado por hiperplano ---
    rng = np.random.default_rng(seed)
    best_partition = {}
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
            best_value, best_partition = value, partition

    return best_partition, best_value


if __name__ == "__main__":
    G = load_project_graph()
    print(f"Grafo cargado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

    bf_part, bf_val = brute_force_maxcut(G)
    print(f"Fuerza bruta (óptimo):      corte = {bf_val:.3f}")

    gr_part, gr_val = greedy_maxcut(G, seed=0)
    print(f"Greedy:                     corte = {gr_val:.3f}  (ratio={gr_val/bf_val:.3f})")

    gw_part, gw_val = goemans_williamson(G, n_rounding_trials=100, seed=42)
    print(f"Goemans-Williamson:         corte = {gw_val:.3f}  (ratio={gw_val/bf_val:.3f})")
