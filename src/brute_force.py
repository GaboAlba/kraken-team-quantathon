"""Single vectorized brute-force cut enumerator over a NetworkX graph.

One exact-enumeration implementation, shared by every brute-force call site in
the project:

- the classical grid-graph Max-Cut baseline (:mod:`src.classical_baselines`),
- the Ising ``H_C`` spectrum / ground-state helpers (:mod:`src.qaoa`), and
- the timeout-guarded benchmark baseline (:mod:`src.benchmark`).

All ``2^n`` spin assignments are enumerated in NumPy chunks, tracking the
extremal weighted cut(s). One node is pinned to a fixed side -- global-flip
symmetry leaves every cut invariant -- so only ``2^(n-1)`` assignments are
visited, and the per-edge accumulation keeps peak memory proportional to the
chunk size (not ``chunk x n_edges``); that is what lets it scale to the
~26-node grids. An optional wall-clock ``timeout`` aborts a long enumeration so
callers can fall back to an approximate baseline.

Mapping a weighted cut to an Ising energy: for a graph whose edge weights are
the fields/couplings of ``H_C`` (see :func:`src.qubo.augmented_ising_graph`),
``E = offset + total_weight - 2 * cut``. The **maximum** cut is therefore the
Ising **ground state** and the **minimum** cut its highest-energy state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Hashable

import networkx as nx
import numpy as np


@dataclass
class CutSpectrum:
    """Extremal weighted cuts found by :func:`enumerate_cut_spectrum`.

    ``min_bits``/``max_bits`` give the side (``0`` or ``1``) of each node in
    :attr:`nodes` order; the first node is always pinned to side ``0``.
    ``total_weight`` is the sum ``K`` of all edge weights, handy for mapping a
    cut to an Ising energy via ``E = offset + K - 2 * cut``.
    """

    nodes: list[Hashable]
    min_value: float
    min_bits: list[int]
    max_value: float
    max_bits: list[int]
    total_weight: float
    n_evaluated: int
    time_s: float
    timed_out: bool


def _edge_arrays(G: nx.Graph, weight: str):
    """Return node order plus parallel ``(i, j, weight)`` arrays for the edges."""
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    ii: list[int] = []
    jj: list[int] = []
    ww: list[float] = []
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        ii.append(index[u])
        jj.append(index[v])
        ww.append(float(data.get(weight, 1.0)))
    return (
        nodes,
        np.asarray(ii, dtype=np.int64),
        np.asarray(jj, dtype=np.int64),
        np.asarray(ww, dtype=np.float64),
    )


def _side_bits(state: int, free: int, n: int) -> list[int]:
    """Decode an enumerated ``state`` into per-node sides (node 0 pinned to 0)."""
    bits = [0] * n
    for b in range(free):
        bits[b + 1] = int((state >> b) & 1)
    return bits


def enumerate_cut_spectrum(
    G: nx.Graph,
    weight: str = "weight",
    chunk_bits: int = 18,
    timeout: float | None = None,
    max_nodes: int = 26,
    dtype: type = np.float64,
) -> CutSpectrum:
    """Exhaustively enumerate the weighted cut spectrum of ``G``.

    Tracks the minimum- and maximum-weight cuts (and one optimal partition for
    each). Raises ``ValueError`` when ``G`` has more than ``max_nodes`` nodes,
    since the enumeration is ``O(2^n)``. Pass ``timeout`` (seconds) to abort
    early and inspect :attr:`CutSpectrum.timed_out`; ``dtype`` (default
    ``float64``) trades accuracy for the memory/speed of the chunk arithmetic.
    """
    n = G.number_of_nodes()
    if n > max_nodes:
        raise ValueError(
            f"brute force refused for {n} nodes (> {max_nodes}); 2^{n} assignments"
        )

    nodes, ii, jj, ww = _edge_arrays(G, weight)
    total_weight = float(ww.sum())
    w = ww.astype(dtype)
    k = dtype(total_weight)

    # Pin node 0 to side 0 and enumerate the other n-1 nodes (global-flip gauge).
    free = max(n - 1, 0)
    total = 1 << free
    chunk = min(1 << chunk_bits, total)
    bit_pos = np.arange(free, dtype=np.int64)

    min_value = np.inf
    max_value = -np.inf
    min_bits = [0] * n
    max_bits = [0] * n
    evaluated = 0
    start = time.perf_counter()
    timed_out = False

    for s0 in range(0, total, chunk):
        idx = np.arange(s0, min(s0 + chunk, total), dtype=np.int64)
        # z in {+1, -1}: column 0 is the pinned node (+1); the rest come from idx.
        z = np.ones((idx.shape[0], n), dtype=dtype)
        if free:
            z[:, 1:] = 1 - 2 * ((idx[:, None] >> bit_pos) & 1).astype(dtype)
        # cut = (K - sum_k w_k z_i z_j) / 2, accumulated per edge to bound memory.
        agree = np.zeros(idx.shape[0], dtype=dtype)
        for edge in range(ii.size):
            agree += w[edge] * (z[:, ii[edge]] * z[:, jj[edge]])
        cut = 0.5 * (k - agree)

        evaluated += int(idx.shape[0])
        lo = int(cut.argmin())
        if cut[lo] < min_value:
            min_value = float(cut[lo])
            min_bits = _side_bits(int(idx[lo]), free, n)
        hi = int(cut.argmax())
        if cut[hi] > max_value:
            max_value = float(cut[hi])
            max_bits = _side_bits(int(idx[hi]), free, n)

        if timeout is not None and time.perf_counter() - start > timeout:
            timed_out = True
            break

    if n <= 1:
        min_value = max_value = 0.0

    return CutSpectrum(
        nodes=nodes,
        min_value=float(min_value),
        min_bits=min_bits,
        max_value=float(max_value),
        max_bits=max_bits,
        total_weight=total_weight,
        n_evaluated=evaluated,
        time_s=time.perf_counter() - start,
        timed_out=timed_out,
    )


def brute_force_max_cut(G: nx.Graph, weight: str = "weight", **kwargs):
    """Maximum-weight cut: return ``({node: side}, value)``."""
    spectrum = enumerate_cut_spectrum(G, weight=weight, **kwargs)
    partition = dict(zip(spectrum.nodes, spectrum.max_bits))
    return partition, spectrum.max_value


def brute_force_min_cut(G: nx.Graph, weight: str = "weight", **kwargs):
    """Minimum-weight cut: return ``({node: side}, value)``.

    The min-cut is the natural objective for the grid, where higher edge weights
    mark more critical transmission lines that the fault-zone boundary should
    avoid cutting.
    """
    spectrum = enumerate_cut_spectrum(G, weight=weight, **kwargs)
    partition = dict(zip(spectrum.nodes, spectrum.min_bits))
    return partition, spectrum.min_value
