"""Tests for the QUBO / Max-Cut formulation module (Task B).

All cases use small synthetic graphs so the QUBO can be validated exactly,
including a brute-force enumeration over all bit assignments.
"""

import sys
from itertools import product
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import qubo


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _triangle(weights=(230.0, 138.0, 230.0), gens=(1, 1, 0)):
    """A 3-node triangle a-b-c with given edge voltages and generator counts.

    Edge weights are set directly so tests can pin the objective; the ``kv``
    scheme recomputes weight = voltage, so ``voltage`` is set to match.
    """
    G = nx.Graph()
    for name, ng in zip(("a", "b", "c"), gens):
        G.add_node(name, n_generators=ng)
    wab, wbc, wca = weights
    G.add_edge("a", "b", voltage=wab, weight=wab)
    G.add_edge("b", "c", voltage=wbc, weight=wbc)
    G.add_edge("c", "a", voltage=wca, weight=wca)
    return G


def _brute_force_min(q: qubo.QUBO):
    """Return (min_energy, [minimizing assignments as bit tuples])."""
    n = len(q.variables)
    best_e = None
    best = []
    for bits in product((0, 1), repeat=n):
        e = q.energy(list(bits))
        if best_e is None or e < best_e - 1e-9:
            best_e, best = e, [bits]
        elif abs(e - best_e) <= 1e-9:
            best.append(bits)
    return best_e, best


# --------------------------------------------------------------------------
# Objective (Max-Cut cut value)
# --------------------------------------------------------------------------

def test_maxcut_objective_matches_cut_value():
    G = _triangle()
    q = qubo.QUBO(variables=sorted(G.nodes), linear={}, quadratic={})
    qubo.add_maxcut_objective(q, G)
    idx = q.index
    # Cut {a} | {b, c}: edges a-b and c-a are cut (230 + 230 = 460).
    assignment = {"a": 0, "b": 1, "c": 1}
    # Objective is -CutValue, so energy should be -460.
    assert q.energy(assignment) == pytest.approx(-460.0)


def test_maxcut_all_same_partition_is_zero_cut():
    G = _triangle()
    q = qubo.QUBO(variables=sorted(G.nodes), linear={}, quadratic={})
    qubo.add_maxcut_objective(q, G)
    assert q.energy({"a": 0, "b": 0, "c": 0}) == pytest.approx(0.0)
    assert q.energy({"a": 1, "b": 1, "c": 1}) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Generator spread penalty
# --------------------------------------------------------------------------

def test_generator_nodes_selection_and_pairs():
    G = _triangle(gens=(1, 1, 0))
    assert qubo.generator_nodes(G) == ["a", "b"]


def test_generator_spread_penalizes_same_partition():
    # Two generator nodes a, b. Penalty coefficient P applied per pair.
    G = _triangle(gens=(1, 1, 0))
    P = 100.0
    q = qubo.QUBO(variables=sorted(G.nodes), linear={}, quadratic={})
    qubo.add_generator_spread_penalty(q, G, P)
    # Same partition -> full penalty P; split -> 0.
    assert q.energy({"a": 0, "b": 0, "c": 0}) == pytest.approx(P)
    assert q.energy({"a": 1, "b": 1, "c": 0}) == pytest.approx(P)
    assert q.energy({"a": 0, "b": 1, "c": 0}) == pytest.approx(0.0)
    assert q.energy({"a": 1, "b": 0, "c": 0}) == pytest.approx(0.0)


def test_generator_spread_is_symmetric_all0_and_all1():
    G = _triangle(gens=(1, 1, 1))
    P = 50.0
    q = qubo.QUBO(variables=sorted(G.nodes), linear={}, quadratic={})
    qubo.add_generator_spread_penalty(q, G, P)
    # 3 generator pairs, all same partition either way -> 3 * P.
    assert q.energy({"a": 0, "b": 0, "c": 0}) == pytest.approx(3 * P)
    assert q.energy({"a": 1, "b": 1, "c": 1}) == pytest.approx(3 * P)


# --------------------------------------------------------------------------
# Balance penalty
# --------------------------------------------------------------------------

def test_balance_penalty_expansion():
    G = _triangle()
    lam = 10.0
    q = qubo.QUBO(variables=sorted(G.nodes), linear={}, quadratic={})
    qubo.add_balance_penalty(q, G, lam)
    n = 3
    # lambda * (sum x - n/2)^2 for each assignment.
    for bits in product((0, 1), repeat=n):
        expected = lam * (sum(bits) - n / 2.0) ** 2
        assert q.energy(list(bits)) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Weight scheme recomputation
# --------------------------------------------------------------------------

def test_apply_weight_scheme_uses_kv_voltage():
    G = _triangle(weights=(230.0, 138.0, 230.0))
    # Corrupt the stored weight; kv scheme must recompute from voltage.
    for _, _, d in G.edges(data=True):
        d["weight"] = -999.0
    H = qubo.apply_weight_scheme(G, "kv")
    for _, _, d in H.edges(data=True):
        assert d["weight"] == pytest.approx(d["voltage"])
    # Original graph untouched (copy semantics).
    assert all(d["weight"] == -999.0 for _, _, d in G.edges(data=True))


# --------------------------------------------------------------------------
# Ising round-trip
# --------------------------------------------------------------------------

def test_qubo_ising_roundtrip():
    G = _triangle(gens=(1, 1, 0))
    q = qubo.build_qubo(G)
    ising = qubo.qubo_to_ising(q)
    n = len(q.variables)
    for bits in product((0, 1), repeat=n):
        z = [1 - 2 * b for b in bits]  # x = (1 - z)/2  ->  z = 1 - 2x
        e_ising = ising["offset"]
        for i, hi in enumerate(ising["h"]):
            e_ising += hi * z[i]
        for (i, j), Jij in ising["J"].items():
            e_ising += Jij * z[i] * z[j]
        assert e_ising == pytest.approx(q.energy(list(bits)))


# --------------------------------------------------------------------------
# End-to-end optimum splits the generators
# --------------------------------------------------------------------------

def test_strong_generator_penalty_forces_split():
    # Square ring a-b-c-d-a, generators on opposite corners a and c. The plain
    # Max-Cut optimum groups opposite corners ({a,c}|{b,d}); a strong generator
    # spread penalty must override that and split the generators instead.
    G = nx.Graph()
    for name, ng in (("a", 1), ("b", 0), ("c", 1), ("d", 0)):
        G.add_node(name, n_generators=ng)
    for u, v in (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")):
        G.add_edge(u, v, voltage=230.0, weight=230.0)
    q = qubo.build_qubo(G, weight_scheme="kv", maximize_cut=True,
                        gen_penalty_factor=10.0)
    _, minimizers = _brute_force_min(q)
    idx = q.index
    for bits in minimizers:
        assert bits[idx["a"]] != bits[idx["c"]]


def test_soft_generator_penalty_does_not_dominate():
    # With the default soft penalty, the strong Max-Cut objective wins on a
    # 4-cycle and groups the opposite-corner generators together.
    G = nx.Graph()
    for name, ng in (("a", 1), ("b", 0), ("c", 1), ("d", 0)):
        G.add_node(name, n_generators=ng)
    for u, v in (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")):
        G.add_edge(u, v, voltage=230.0, weight=230.0)
    q = qubo.build_qubo(G, weight_scheme="kv", maximize_cut=True)
    _, minimizers = _brute_force_min(q)
    idx = q.index
    assert all(bits[idx["a"]] == bits[idx["c"]] for bits in minimizers)


def test_build_qubo_metadata_kv_maximize():
    G = _triangle(gens=(1, 1, 0))
    q = qubo.build_qubo(G, weight_scheme="kv", maximize_cut=True,
                        gen_penalty_factor=0.5, balance_penalty_factor=0.15)
    md = q.metadata
    assert md["weight_scheme"] == "kv"
    assert md["maximize_cut"] is True
    assert md["n_variables"] == 3
    assert md["generator_nodes"] == ["a", "b"]
    assert md["n_generator_pairs"] == 1
    assert md["max_edge_weight"] == pytest.approx(230.0)
    assert md["penalties"]["generator_spread"]["coefficient"] == pytest.approx(115.0)
    assert md["penalties"]["balance"]["coefficient"] == pytest.approx(34.5)


def test_default_scheme_is_inverted_generation_minimize():
    G = _triangle(gens=(1, 1, 0))
    q = qubo.build_qubo(G)
    assert q.metadata["weight_scheme"] == "generation_inverted"
    assert q.metadata["maximize_cut"] is False


def test_inverted_generation_negates_generation():
    G = _triangle(gens=(1, 1, 0))
    plain = qubo.apply_weight_scheme(G, "generation")
    inverted = qubo.apply_weight_scheme(G, "generation_inverted")
    for (u, v, dp), (_, _, di) in zip(plain.edges(data=True),
                                      inverted.edges(data=True)):
        assert di["weight"] == pytest.approx(-dp["weight"])


def test_minimize_cut_spares_critical_line():
    # Path a-b-c with one heavy (critical) line and one light line. Minimizing
    # the cut should place the single cut on the light edge, sparing the heavy
    # one. Weights are supplied via a custom scheme-free graph, using 'kv' so
    # weight == voltage, and forcing exactly one cut via a strong balance push
    # is unnecessary here: a-b heavy, b-c light, 2 generators a and c force a
    # split so the boundary lands between them on the cheaper edge.
    G = nx.Graph()
    for name, ng in (("a", 1), ("b", 0), ("c", 1)):
        G.add_node(name, n_generators=ng)
    G.add_edge("a", "b", voltage=230.0)   # heavy / critical
    G.add_edge("b", "c", voltage=138.0)   # light
    q = qubo.build_qubo(G, weight_scheme="kv", maximize_cut=False,
                        gen_penalty_factor=1.0, balance_penalty_factor=0.0)
    _, minimizers = _brute_force_min(q)
    idx = q.index
    # Generators a and c must end up split, and the cut edge is the light b-c
    # one, i.e. b stays with a (heavy edge a-b uncut).
    for bits in minimizers:
        assert bits[idx["a"]] != bits[idx["c"]]
        assert bits[idx["a"]] == bits[idx["b"]]


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def test_to_json_structure_and_energy_offset():
    G = _triangle(gens=(1, 1, 0))
    q = qubo.build_qubo(G)
    doc = qubo.to_json(q)
    assert doc["variables"] == ["a", "b", "c"]
    assert "linear" in doc["qubo"] and "quadratic" in doc["qubo"]
    assert "h" in doc["ising"] and "J" in doc["ising"]
    assert doc["qubo"]["offset"] == pytest.approx(q.offset)


def test_build_reads_and_writes(tmp_path):
    # Minimal grid_cr.json-like document.
    grid = {
        "metadata": {"region": "unit_test"},
        "nodes": [
            {"id": "a", "n_generators": 1},
            {"id": "b", "n_generators": 1},
            {"id": "c", "n_generators": 0},
        ],
        "edges": [
            {"u": "a", "v": "b", "voltage": 230.0},
            {"u": "b", "v": "c", "voltage": 138.0},
        ],
    }
    import json
    in_path = tmp_path / "grid_cr.json"
    out_path = tmp_path / "qubo_cr.json"
    in_path.write_text(json.dumps(grid), encoding="utf-8")
    q = qubo.build(input_path=in_path, output=out_path)
    assert out_path.exists()
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["metadata"]["region"] == "unit_test"
    assert doc["variables"] == ["a", "b", "c"]
