"""Tests for the Max-Cut QUBO formulation (fault-zone partition).

Synthetic cases use a small triangle graph where the optimum is known by hand;
the real-snapshot test locks the optimal cut of the official Guanacaste North
subgrid.
"""

import sys
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import qubo


def _triangle() -> nx.Graph:
    G = nx.Graph()
    G.add_edge("a", "b", weight=2.0)
    G.add_edge("b", "c", weight=3.0)
    G.add_edge("a", "c", weight=4.0)
    return G


def test_to_qubo_labels_are_sorted_node_ids():
    Q, labels = qubo.to_qubo(_triangle())
    assert labels == ["a", "b", "c"]
    assert Q.shape == (3, 3)


def test_to_qubo_matrix_entries():
    Q, _ = qubo.to_qubo(_triangle())
    # Diagonal: minus the weighted degree of each node.
    assert Q[0][0] == -6.0   # a: 2 + 4
    assert Q[1][1] == -5.0   # b: 2 + 3
    assert Q[2][2] == -7.0   # c: 3 + 4
    # Upper triangle: +2w per edge; lower triangle stays zero.
    assert Q[0][1] == 4.0 and Q[1][0] == 0.0
    assert Q[0][2] == 8.0 and Q[2][0] == 0.0
    assert Q[1][2] == 6.0 and Q[2][1] == 0.0


def test_energy_is_minus_cut_weight():
    Q, _ = qubo.to_qubo(_triangle())
    assert qubo.energy(Q, [0, 0, 0]) == 0.0
    assert qubo.energy(Q, [0, 1, 1]) == -6.0   # cut = ab + ac = 2 + 4
    assert qubo.energy(Q, [0, 0, 1]) == -7.0   # cut = bc + ac = 3 + 4


def test_brute_force_finds_triangle_optimum():
    Q, _ = qubo.to_qubo(_triangle())
    best_cut, x = qubo.brute_force(Q)
    assert best_cut == 7.0                     # c isolated: 3 + 4
    assert x[2] != x[0] and x[0] == x[1]


def test_brute_force_rejects_large_instances():
    import numpy as np
    with pytest.raises(ValueError):
        qubo.brute_force(np.zeros((25, 25)))


# --------------------------------------------------------------------------
# Real snapshot: the official subgrid's QUBO
# --------------------------------------------------------------------------

RAW = ROOT / "data" / "raw"
has_snapshot = (RAW / "substations.geojson").exists() and (RAW / "lines.geojson").exists()
real = pytest.mark.skipif(not has_snapshot, reason="ICE snapshot not available")


@real
def test_real_guanacaste_north_optimal_cut():
    import json
    from src import graph
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, nodes=graph.GUANACASTE_NORTH)
    Q, labels = qubo.to_qubo(sub)
    best_cut, x = qubo.brute_force(Q)
    # Odd ring: 8 of the 9 lines (all 230 kV) can be cut, one must stay inside.
    assert best_cut == 8 * 230
    assert len(labels) == 9
