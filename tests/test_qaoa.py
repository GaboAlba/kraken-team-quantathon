"""Tests for the QAOA solver (Task C).

Split into two groups:

- Pure-classical tests (loading, energy evaluation, brute force) that run with
  no quantum dependency.
- Emulator-backed tests (kernel type-check/compile, a short seeded QAOA run)
  that are skipped automatically when ``guppylang``/``selene`` are unavailable.
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import qaoa
from src.qubo import CostHamiltonian, PauliZTerm

# Skip the emulator-backed tests gracefully if the quantum stack is missing.
try:  # pragma: no cover - import guard
    import guppylang  # noqa: F401
    import selene_sim  # noqa: F401

    HAS_QUANTUM = True
except Exception:  # pragma: no cover - import guard
    HAS_QUANTUM = False

needs_quantum = pytest.mark.skipif(
    not HAS_QUANTUM, reason="guppylang/selene not installed"
)


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

def _zz_hamiltonian() -> CostHamiltonian:
    """H_C = Z0 Z1 on 2 qubits. Ground states are the anti-aligned 01/10."""
    return CostHamiltonian(
        n_qubits=2,
        variables=["a", "b"],
        terms=[PauliZTerm(coefficient=1.0, qubits=(0, 1))],
        offset=0.0,
    )


def _field_hamiltonian() -> CostHamiltonian:
    """H_C = 0.5*Z0 - 0.5*Z1 + Z0 Z1. Unique ground state at bits (1, 0)."""
    return CostHamiltonian(
        n_qubits=2,
        variables=["a", "b"],
        terms=[
            PauliZTerm(coefficient=0.5, qubits=(0,)),
            PauliZTerm(coefficient=-0.5, qubits=(1,)),
            PauliZTerm(coefficient=1.0, qubits=(0, 1)),
        ],
        offset=0.0,
    )


class _FakeResult:
    """Minimal stand-in exposing ``register_counts()`` like a QsysResult."""

    def __init__(self, counts: dict[str, int]):
        self._counts = Counter(counts)

    def register_counts(self):
        return {"c": self._counts}


# --------------------------------------------------------------------------
# Pure-classical tests
# --------------------------------------------------------------------------

def test_brute_force_zz():
    ch = _zz_hamiltonian()
    energy, bits = qaoa.brute_force_ground_state(ch)
    assert energy == pytest.approx(-1.0)
    assert bits in ([0, 1], [1, 0])


def test_brute_force_field_unique():
    ch = _field_hamiltonian()
    energy, bits = qaoa.brute_force_ground_state(ch)
    # z = 1 - 2*bits -> (1,0): z=(−1,1); 0.5*(-1) -0.5*(1) + (-1)(1) = -2.0
    assert bits == [1, 0]
    assert energy == pytest.approx(-2.0)


def test_brute_force_refuses_large():
    ch = CostHamiltonian(n_qubits=30, variables=[str(i) for i in range(30)],
                         terms=[], offset=0.0)
    with pytest.raises(ValueError):
        qaoa.brute_force_ground_state(ch, max_qubits=22)


def test_energy_bounds_zz():
    ch = _zz_hamiltonian()  # spectrum is {-1, +1}
    e_min, e_max = qaoa.energy_bounds(ch)
    assert e_min == pytest.approx(-1.0)
    assert e_max == pytest.approx(1.0)


def test_energy_bounds_refuses_large():
    ch = CostHamiltonian(n_qubits=30, variables=[str(i) for i in range(30)],
                         terms=[], offset=0.0)
    with pytest.raises(ValueError):
        qaoa.energy_bounds(ch, max_qubits=22)


def test_approximation_ratio_endpoints():
    # r = 1 at the optimum (E_min), 0 at the worst (E_max), 0.5 midway.
    assert qaoa.approximation_ratio(-1.0, -1.0, 1.0) == pytest.approx(1.0)
    assert qaoa.approximation_ratio(1.0, -1.0, 1.0) == pytest.approx(0.0)
    assert qaoa.approximation_ratio(0.0, -1.0, 1.0) == pytest.approx(0.5)


def test_approximation_ratio_degenerate_spectrum():
    # E_max == E_min: no spread, defined as a perfect ratio of 1.0.
    assert qaoa.approximation_ratio(3.0, 3.0, 3.0) == pytest.approx(1.0)


def test_qaoa_result_approximation_ratio():
    ch = _zz_hamiltonian()  # spectrum {-1, +1}; ground states 01/10
    # Most-likely bitstring is the ground state '01' (E=-1) -> ratio 1.0.
    res = qaoa.QAOAResult(
        energy=0.0,  # midway <H_C> -> ratio 0.5
        cost_angles=np.zeros(1),
        mixer_angles=np.zeros(1),
        result=_FakeResult({"01": 90, "00": 10}),
        ch=ch,
    )
    assert res.approximation_ratio() == pytest.approx(1.0)
    assert res.approximation_ratio(expectation=True) == pytest.approx(0.5)
    # Passing precomputed bounds avoids re-enumerating and gives the same result.
    assert res.approximation_ratio(bounds=(-1.0, 1.0)) == pytest.approx(1.0)


def test_energy_from_result_matches_hamiltonian():
    ch = _zz_hamiltonian()
    # 60% on '01' (E=-1), 40% on '00' (E=+1): expectation = -1*0.6 + 1*0.4 = -0.2
    res = _FakeResult({"01": 60, "00": 40})
    energy = qaoa.energy_from_result(ch, res, n_shots=100)
    assert energy == pytest.approx(-0.2)


def test_energy_from_result_uses_actual_total():
    ch = _zz_hamiltonian()
    # Robust to a mismatched declared n_shots: uses the real total (2).
    res = _FakeResult({"01": 1, "10": 1})
    energy = qaoa.energy_from_result(ch, res, n_shots=1000)
    assert energy == pytest.approx(-1.0)


def test_load_cost_hamiltonian_roundtrip(tmp_path):
    import json

    ch = _field_hamiltonian()
    doc = {
        "variables": ch.variables,
        "cost_hamiltonian": {
            "n_qubits": ch.n_qubits,
            "terms": [
                {"qubits": list(t.qubits), "coeff": t.coefficient}
                for t in ch.terms
            ],
            "offset": ch.offset,
        },
    }
    path = tmp_path / "qubo.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    loaded = qaoa.load_cost_hamiltonian(path)
    assert loaded.n_qubits == ch.n_qubits
    assert loaded.variables == ch.variables
    assert loaded.z_terms == ch.z_terms
    assert loaded.zz_terms == ch.zz_terms


# --------------------------------------------------------------------------
# Emulator-backed tests
# --------------------------------------------------------------------------

@needs_quantum
def test_kernel_type_checks():
    ch = _field_hamiltonian()
    instance = qaoa.build_qaoa_instance(ch, n_layers=2)
    instance.check()  # static type-check; raises on linearity/type errors


@needs_quantum
def test_eval_qaoa_energy_runs():
    ch = _zz_hamiltonian()
    energy, result = qaoa.eval_qaoa_energy(
        [0.3, 0.4], [0.2, 0.1], ch, seed=1, shots=200, n_layers=2,
    )
    assert isinstance(energy, float)
    counts = result.register_counts()["c"]
    assert sum(counts.values()) == 200
    # Every measured bitstring has length n_qubits.
    assert all(len(bits) == ch.n_qubits for bits in counts)


@needs_quantum
def test_qaoa_finds_zz_ground_state():
    """On the trivial 2-qubit anti-alignment problem QAOA should reach E=-1."""
    ch = _zz_hamiltonian()
    res = qaoa.solve_scipy(ch, p_value=2, n_shots=500, seed=7, maxiter=30)
    assert res.most_likely_energy() == pytest.approx(-1.0)
    assert res.most_likely_bits() in ([0, 1], [1, 0])


@needs_quantum
def test_qaoa_result_partition_keys():
    ch = _field_hamiltonian()
    res = qaoa.solve_scipy(ch, p_value=1, n_shots=200, seed=5, maxiter=10)
    partition = res.most_likely_partition()
    assert set(partition) == set(ch.variables)
    assert all(v in (0, 1) for v in partition.values())


def test_plot_partition_writes_file(tmp_path):
    import matplotlib

    matplotlib.use("Agg")  # headless
    import networkx as nx

    from src import visualize

    sub = nx.Graph()
    sub.add_node("a", n_generators=1)
    sub.add_node("b", n_generators=0)
    sub.add_node("c", n_generators=1)
    sub.add_edge("a", "b", weight=2.0, voltage=230)
    sub.add_edge("b", "c", weight=1.0, voltage=138)
    sub.add_edge("a", "c", weight=3.0, voltage=230)
    partition = {"a": 0, "b": 1, "c": 0}  # cut edges: a-b and b-c

    out = tmp_path / "partition.png"
    result = visualize.plot_partition(sub, partition, out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
