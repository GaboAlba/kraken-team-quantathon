"""QAOA solver for the grid fault-zone QUBO, implemented in Guppy (Task C).

Reads the diagonal cost Hamiltonian ``H_C`` produced by :mod:`src.qubo` and runs
the Quantum Approximate Optimization Algorithm (QAOA) on it with `guppylang`
kernels executed on the Selene emulator. The reference for the Guppy/Selene API
and the variational-loop structure is Quantinuum's
``qaoa_maxcut_example.ipynb`` (branch ``guppylang-0.21``).

What is fixed by the Graph + QUBO (nothing to decide):

- **# qubits** = number of substation nodes, one binary variable per node.
- **Cost Hamiltonian** ``H_C = offset*I + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j``
  (:class:`src.qubo.CostHamiltonian`), including the *sign* convention: the QUBO
  is built with ``maximize_cut=False``, so QAOA **minimizes** ``<H_C>`` (unlike
  the plain max-cut example, which maximizes an unweighted energy). Because the
  problem is *weighted* and has *single-Z fields*, the phase-separation layer is
  not the example's bare ``zz_phase`` per edge; it follows ``docs/qubo.md`` (and
  ``docs/qaoa.md``):

  =====================  ======================================
  Cost-Hamiltonian term  Circuit fragment
  =====================  ======================================
  ``h_i Z_i``            ``rz(2*gamma*h_i, i)``
  ``J_ij Z_i Z_j``       ``cx(i, j); rz(2*gamma*J_ij, j); cx(i, j)``
  mixer                  ``rx(2*beta, i)`` per qubit
  =====================  ======================================

What the *algorithm* still needs (hyperparameters -- not derivable):

- ``p`` (number of cost/mixer layers), ``n_shots``, ``seed``, and the classical
  optimizer. Two optimizers are provided: :func:`solve_naive` (random-sampling
  baseline, keeps the *lowest* energy) and :func:`solve_scipy` (COBYLA, the main
  path), both minimizing ``<H_C>``.

Guppy angle unit gotcha: ``angle(x)`` is ``x`` **half-turns** (``x * pi``
radians), and ``pi == angle(1)``. To keep the literal ``rz(2*gamma*h_i)``
*radian* convention above, the per-term coefficient ``2*h_i`` is folded (at
compile time) with a ``1/pi`` factor into half-turns, so ``gamma`` (``beta``) is
a plain multiplier in half-turn units.

Run end-to-end (loads ``data/qubo_cr.json``, or rebuilds it from
``data/grid_cr.json`` if missing)::

    python -m src.qaoa
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src import qubo
from src.qubo import CostHamiltonian, PauliZTerm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUBO = ROOT / "data" / "qubo_cr.json"
DEFAULT_GRAPH = ROOT / "data" / "grid_cr.json"
DEFAULT_FIGURE = ROOT / "figures" / "qaoa_partition.png"

# Default QAOA hyperparameters (the decisions that are *not* fixed by the QUBO).
DEFAULT_LAYERS = 2
DEFAULT_SHOTS = 1000
DEFAULT_SEED = 7
DEFAULT_ITERATIONS = 40  # naive baseline: number of random parameter samples
DEFAULT_MAXITER = 40  # scipy: maximum optimizer iterations


# --------------------------------------------------------------------------
# Loading the cost Hamiltonian
# --------------------------------------------------------------------------

def load_cost_hamiltonian(path: Path = DEFAULT_QUBO) -> CostHamiltonian:
    """Reconstruct :class:`CostHamiltonian` from a ``qubo_cr.json`` document.

    Reads the serialized ``cost_hamiltonian`` section (and ``variables``) so the
    QAOA solver consumes the same static artifact the rest of the pipeline
    produces, without recomputing the QUBO.
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    ch_doc = doc["cost_hamiltonian"]
    variables = list(doc["variables"])
    terms = [
        PauliZTerm(coefficient=t["coeff"], qubits=tuple(t["qubits"]))
        for t in ch_doc["terms"]
    ]
    return CostHamiltonian(
        n_qubits=ch_doc["n_qubits"],
        variables=variables,
        terms=terms,
        offset=ch_doc.get("offset", 0.0),
    )


def cost_hamiltonian_from_graph(path: Path = DEFAULT_GRAPH) -> CostHamiltonian:
    """Build ``H_C`` directly from ``grid_cr.json`` (rebuilds the QUBO)."""
    G = qubo.load_graph(path)
    return qubo.qubo_to_cost_hamiltonian(qubo.build_qubo(G))


# --------------------------------------------------------------------------
# Guppy QAOA kernel
# --------------------------------------------------------------------------

def build_qaoa_instance(ch: CostHamiltonian, n_layers: int):
    """Build a Guppy QAOA kernel for the cost Hamiltonian ``ch``.

    Returns a ``GuppyFunctionDefinition`` taking two ``frozenarray[float, p]``
    arguments (the per-layer cost angles ``gamma`` and mixer angles ``beta``) and
    returning the ``n_qubits`` qubits after the alternating cost/mixer layers.

    The single-``Z`` fields ``h_i`` and ``Z_i Z_j`` couplings ``J_ij`` are baked
    in as compile-time constants. Each field/coupling coefficient is converted
    from the intended ``rz(2*gamma*coeff)`` *radian* rotation to Guppy's
    *half-turn* angle unit by folding a ``2/pi`` factor at compile time; likewise
    the mixer uses ``rx(2*beta)`` radians via a ``2/pi`` factor.
    """
    # guppy is imported lazily so the classical helpers in this module (loading,
    # energy evaluation, brute force) stay importable even if guppy/selene are
    # unavailable.
    from guppylang import guppy
    from guppylang.std.angles import angle
    from guppylang.std.builtins import array, comptime, frozenarray
    from guppylang.std.quantum import cx, h, qubit, rx, rz

    n_qubits = ch.n_qubits
    if n_qubits < 2:
        raise ValueError("QAOA kernel needs at least 2 qubits")
    half_turn = 2.0 / math.pi  # radians -> half-turns, times the QAOA factor 2

    # Fold the factor 2 and the radians->half-turns conversion into each
    # coefficient so the kernel applies angle(gamma * coeff_i) == 2*gamma*h_i rad.
    z_coeffs = [(q, half_turn * hi) for (q, hi) in ch.z_terms]
    zz_coeffs = [(i, j, half_turn * jij) for (i, j, jij) in ch.zz_terms]
    # Guppy cannot type-infer an *empty* comptime list, so a Hamiltonian with no
    # field (or no coupling) terms would fail to compile. Inject a single
    # zero-coefficient placeholder: rz(0) is a no-op and cx; rz(0); cx == cx; cx
    # == identity, so the circuit is unchanged.
    if not z_coeffs:
        z_coeffs = [(0, 0.0)]
    if not zz_coeffs:
        zz_coeffs = [(0, 1, 0.0)]

    @guppy
    def qaoa_instance(
        cost_angles: frozenarray[float, comptime(n_layers)],
        mixer_angles: frozenarray[float, comptime(n_layers)],
    ) -> array[qubit, comptime(n_qubits)]:
        qs = array(qubit() for _ in range(comptime(n_qubits)))
        n = len(qs)

        # Uniform superposition |+>^{⊗n}.
        for i in range(n):
            h(qs[i])

        for layer in range(comptime(n_layers)):
            gamma = cost_angles[layer]

            # Cost layer: single-Z fields then ZZ couplings.
            for qi, ci in comptime(z_coeffs):
                rz(qs[qi], angle(gamma * ci))
            for i, j, cij in comptime(zz_coeffs):
                cx(qs[i], qs[j])
                rz(qs[j], angle(gamma * cij))
                cx(qs[i], qs[j])

            # Mixer layer: rx(2*beta) on every qubit.
            beta = mixer_angles[layer]
            for i in range(n):
                rx(qs[i], angle(beta * comptime(half_turn)))

        return qs

    return qaoa_instance


# --------------------------------------------------------------------------
# Energy evaluation
# --------------------------------------------------------------------------

def energy_from_result(ch: CostHamiltonian, result, n_shots: int) -> float:
    """Expectation value ``<H_C>`` from a Selene ``QsysResult``.

    Aggregates the ``"c"`` register bitstrings (index ``i`` is qubit/variable
    ``i``) and evaluates :meth:`CostHamiltonian.energy` on each, weighted by its
    shot probability. Uses the actual total count so it is robust to shot loss.
    """
    dist = result.register_counts()["c"]
    total = sum(dist.values()) or n_shots
    energy = 0.0
    for meas, count in dist.items():
        bits = [int(c) for c in meas]
        energy += ch.energy(bits) * (count / total)
    return energy


def eval_qaoa_energy(
    cost_angles,
    mixer_angles,
    ch: CostHamiltonian,
    seed: int,
    shots: int,
    instance=None,
    n_layers: int | None = None,
):
    """Run one QAOA forward pass and return ``(<H_C>, QsysResult)``.

    A fresh Guppy ``main`` entrypoint is compiled for each parameter set (Guppy
    execution entrypoints cannot take runtime arguments). Pass a prebuilt
    ``instance`` (from :func:`build_qaoa_instance`) to avoid rebuilding the inner
    kernel every call; otherwise ``n_layers`` must be given.
    """
    from guppylang import guppy
    from guppylang.std.builtins import comptime, result as guppy_result
    from guppylang.std.quantum import measure_array

    if instance is None:
        if n_layers is None:
            n_layers = len(cost_angles)
        instance = build_qaoa_instance(ch, n_layers)

    cost = [float(x) for x in cost_angles]
    mixer = [float(x) for x in mixer_angles]

    @guppy
    def main() -> None:
        qs = instance(comptime(cost), comptime(mixer))
        guppy_result("c", measure_array(qs))

    qaoa_result = (
        main.emulator(n_qubits=ch.n_qubits)
        .with_shots(shots)
        .with_seed(seed)
        .run()
    )
    return energy_from_result(ch, qaoa_result, shots), qaoa_result


# --------------------------------------------------------------------------
# Result decoding
# --------------------------------------------------------------------------

@dataclass
class QAOAResult:
    """Outcome of a QAOA solve."""

    energy: float
    cost_angles: NDArray[np.float64]
    mixer_angles: NDArray[np.float64]
    result: object  # QsysResult from the best evaluation
    ch: CostHamiltonian
    metadata: dict = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        """The measured bitstring counts of the best evaluation."""
        return dict(self.result.register_counts()["c"])

    def most_likely_bits(self) -> list[int]:
        """The most frequently sampled bitstring as a list of bits."""
        dist = self.result.register_counts()["c"]
        best = max(dist.items(), key=lambda kv: kv[1])[0]
        return [int(c) for c in best]

    def most_likely_partition(self) -> dict[str, int]:
        """The most likely assignment as ``node id -> {0, 1}``."""
        return dict(zip(self.ch.variables, self.most_likely_bits()))

    def most_likely_energy(self) -> float:
        """``H_C`` energy of the most likely bitstring."""
        return self.ch.energy(self.most_likely_bits())


# --------------------------------------------------------------------------
# Classical brute-force reference (small n)
# --------------------------------------------------------------------------

def brute_force_ground_state(
    ch: CostHamiltonian, max_qubits: int = 22
) -> tuple[float, list[int]]:
    """Exact minimum-energy assignment by enumerating all ``2^n`` bitstrings.

    Only feasible for small ``n`` (the grid subgraph is <= 12 qubits); raises for
    ``n > max_qubits`` to avoid an intractable enumeration.
    """
    if ch.n_qubits > max_qubits:
        raise ValueError(
            f"brute force refused for {ch.n_qubits} qubits (> {max_qubits})"
        )
    best_energy = math.inf
    best_bits: list[int] = []
    for bits in product((0, 1), repeat=ch.n_qubits):
        e = ch.energy(list(bits))
        if e < best_energy:
            best_energy = e
            best_bits = list(bits)
    return best_energy, best_bits


# --------------------------------------------------------------------------
# Variational loops (both minimize <H_C>)
# --------------------------------------------------------------------------

def solve_naive(
    ch: CostHamiltonian,
    iterations: int = DEFAULT_ITERATIONS,
    p_value: int = DEFAULT_LAYERS,
    n_shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
) -> QAOAResult:
    """Baseline optimizer: sample random angles, keep the *lowest* ``<H_C>``.

    Mirrors the Quantinuum example's naive loop but minimizes (our cost is a
    minimize-cut objective) instead of maximizing. Angles are sampled in
    ``[0, 1)`` half-turn units. Deterministic given ``seed``.
    """
    instance = build_qaoa_instance(ch, p_value)
    rng = np.random.default_rng(seed)

    best_energy = math.inf
    best_cost = np.zeros(p_value)
    best_mixer = np.zeros(p_value)
    best_result = None
    evaluations = 0

    for _ in range(iterations):
        guess_cost = rng.uniform(0.0, 1.0, p_value)
        guess_mixer = rng.uniform(0.0, 1.0, p_value)
        energy, res = eval_qaoa_energy(
            guess_cost, guess_mixer, ch, seed=seed, shots=n_shots,
            instance=instance,
        )
        evaluations += 1
        if energy < best_energy:
            best_energy = energy
            best_cost = guess_cost
            best_mixer = guess_mixer
            best_result = res

    return QAOAResult(
        energy=best_energy,
        cost_angles=best_cost,
        mixer_angles=best_mixer,
        result=best_result,
        ch=ch,
        metadata={
            "optimizer": "naive",
            "p": p_value,
            "iterations": iterations,
            "n_shots": n_shots,
            "seed": seed,
            "evaluations": evaluations,
        },
    )


def solve_scipy(
    ch: CostHamiltonian,
    p_value: int = DEFAULT_LAYERS,
    n_shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
    maxiter: int = DEFAULT_MAXITER,
    method: str = "COBYLA",
) -> QAOAResult:
    """Main optimizer: minimize ``<H_C>`` with a SciPy optimizer (COBYLA).

    Parameters are the flattened ``[cost(p), mixer(p)]`` half-turn multipliers.
    The initial guess is drawn from a ``seed``-seeded RNG and every emulator run
    uses the same fixed ``seed`` so the objective is deterministic.
    """
    from scipy.optimize import minimize

    instance = build_qaoa_instance(ch, p_value)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 1.0, 2 * p_value)

    history: list[float] = []

    def objective(params: NDArray[np.float64]) -> float:
        cost = params[:p_value]
        mixer = params[p_value:]
        energy, _ = eval_qaoa_energy(
            cost, mixer, ch, seed=seed, shots=n_shots, instance=instance,
        )
        history.append(float(energy))
        return energy

    opt = minimize(
        objective, x0, method=method, options={"maxiter": maxiter},
    )

    best_cost = np.asarray(opt.x[:p_value], dtype=float)
    best_mixer = np.asarray(opt.x[p_value:], dtype=float)
    best_energy, best_result = eval_qaoa_energy(
        best_cost, best_mixer, ch, seed=seed, shots=n_shots, instance=instance,
    )

    return QAOAResult(
        energy=best_energy,
        cost_angles=best_cost,
        mixer_angles=best_mixer,
        result=best_result,
        ch=ch,
        metadata={
            "optimizer": f"scipy:{method}",
            "p": p_value,
            "n_shots": n_shots,
            "seed": seed,
            "maxiter": maxiter,
            "n_evaluations": len(history),
            "converged": bool(opt.success),
        },
    )


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

def plot_partition(
    result: QAOAResult,
    graph_path: Path = DEFAULT_GRAPH,
    out: Path = DEFAULT_FIGURE,
    label: str = "QAOA fault-zone partition",
) -> Path:
    """Render the subgrid painted by the QAOA most-likely partition.

    Loads the subgraph from ``graph_path`` (its nodes match ``result.ch``'s
    variables), assigns each node its fault-zone side, and delegates to
    :func:`src.visualize.plot_partition`, which highlights the cut lines.
    Returns the path of the written PNG.
    """
    from src import visualize

    sub = qubo.load_graph(graph_path)
    partition = result.most_likely_partition()
    return visualize.plot_partition(sub, partition, Path(out), label=label)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build(
    qubo_path: Path = DEFAULT_QUBO,
    graph_path: Path = DEFAULT_GRAPH,
    p_value: int = DEFAULT_LAYERS,
    n_shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
    maxiter: int = DEFAULT_MAXITER,
) -> QAOAResult:
    """Load ``H_C`` and solve the QUBO with QAOA (SciPy path).

    Prefers the serialized ``data/qubo_cr.json``; if it is missing, rebuilds the
    cost Hamiltonian from ``data/grid_cr.json``.
    """
    if Path(qubo_path).exists():
        ch = load_cost_hamiltonian(qubo_path)
    else:
        ch = cost_hamiltonian_from_graph(graph_path)
    return solve_scipy(
        ch, p_value=p_value, n_shots=n_shots, seed=seed, maxiter=maxiter,
    )


if __name__ == "__main__":
    ch = (
        load_cost_hamiltonian(DEFAULT_QUBO)
        if DEFAULT_QUBO.exists()
        else cost_hamiltonian_from_graph(DEFAULT_GRAPH)
    )
    print(
        f"Cost Hamiltonian: {ch.n_qubits} qubits, "
        f"{len(ch.z_terms)} single-Z, {len(ch.zz_terms)} ZZ terms"
    )

    qaoa = solve_scipy(ch)
    partition = qaoa.most_likely_partition()

    print(f"\nQAOA ({qaoa.metadata['optimizer']}, p={qaoa.metadata['p']}, "
          f"{qaoa.metadata['n_shots']} shots):")
    print(f"  <H_C>            = {qaoa.energy:.4f}")
    print(f"  most-likely E    = {qaoa.most_likely_energy():.4f}")
    print(f"  cost angles      = {np.round(qaoa.cost_angles, 3)}")
    print(f"  mixer angles     = {np.round(qaoa.mixer_angles, 3)}")

    gs_energy, gs_bits = brute_force_ground_state(ch)
    gs_partition = dict(zip(ch.variables, gs_bits))
    print(f"\nBrute-force optimum: E = {gs_energy:.4f}")
    match = qaoa.most_likely_bits() == gs_bits
    print(f"  QAOA most-likely matches optimum: {match}")

    print("\nMost-likely partition (node -> side):")
    for node, side in partition.items():
        marker = "" if partition.get(node) == gs_partition.get(node) else "  (*)"
        print(f"  {node}: {side}{marker}")

    figure = plot_partition(qaoa)
    print(f"\nPainted partition figure: {figure}")
