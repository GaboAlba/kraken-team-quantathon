"""QAOA solver for the grid fault-zone QUBO, implemented in Guppy (Task C).

Reads the diagonal cost Hamiltonian ``H_C`` from :mod:`src.qubo` and runs QAOA
with `guppylang` kernels on the Selene emulator. The QUBO is built with
``maximize_cut=False``, so QAOA **minimizes** ``<H_C>``. The phase-separation
layer (``rz`` per single-``Z`` field, ``cx; rz; cx`` per ``Z_i Z_j`` coupling)
and the ``rx`` mixer follow ``docs/qaoa.md`` / ``docs/qubo.md``.

The graph + QUBO fix everything but the hyperparameters (``p``, ``n_shots``,
``seed``). :func:`solve_scipy` (COBYLA) minimizes ``<H_C>``.

Guppy angle unit: ``angle(x)`` is ``x`` half-turns (``x * pi`` radians). Each
``2*coeff`` factor is folded with ``1/pi`` at compile time so ``gamma``/``beta``
stay plain half-turn multipliers.

Run end-to-end (loads ``data/qubo_cr.json``, or rebuilds from
``data/grid_cr.json`` if missing)::

    python -m src.qaoa
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src import qubo
from src.brute_force import enumerate_cut_spectrum
from src.qubo import CostHamiltonian, PauliZTerm, augmented_ising_graph

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUBO = ROOT / "data" / "qubo_cr.json"
DEFAULT_GRAPH = ROOT / "data" / "grid_cr.json"
DEFAULT_FIGURE = ROOT / "figures" / "qaoa_partition.png"

# Default QAOA hyperparameters (the decisions that are *not* fixed by the QUBO).
DEFAULT_LAYERS = 2
DEFAULT_SHOTS = 1000
DEFAULT_SEED = 7
DEFAULT_MAXITER = 40  # scipy: maximum optimizer iterations

# Guppy's compiler mutates global/module state while lowering a kernel, so
# compiling (or running, which lowers lazily) from several threads at once
# corrupts it -- surfacing as spurious KeyError/InternalGuppyError/type errors.
# The benchmark runs QAOA tasks in a ThreadPoolExecutor, so every Guppy
# definition + emulator run is serialized behind this re-entrant lock.
_GUPPY_LOCK = threading.RLock()


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

    Returns a ``GuppyFunctionDefinition`` taking the per-layer cost angles
    ``gamma`` and mixer angles ``beta`` (each ``frozenarray[float, p]``) and
    returning the ``n_qubits`` qubits after the alternating cost/mixer layers.
    Fields ``h_i`` and couplings ``J_ij`` are baked in as compile-time constants,
    with the radians->half-turns conversion folded per coefficient (see the
    module docstring and ``docs/qaoa.md``).
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

    # Serialize the Guppy compile (see _GUPPY_LOCK); ``@guppy`` == ``guppy(fn)``.
    with _GUPPY_LOCK:
        return guppy(qaoa_instance)


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

    def main() -> None:
        qs = instance(comptime(cost), comptime(mixer))
        guppy_result("c", measure_array(qs))

    # Serialize the Guppy compile + emulator run (see _GUPPY_LOCK). The RLock
    # allows the ``build_qaoa_instance`` call above (when instance is None) to
    # re-enter safely.
    with _GUPPY_LOCK:
        main = guppy(main)
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

    def approximation_ratio(
        self, bounds: tuple[float, float] | None = None, expectation: bool = False
    ) -> float:
        """Normalized approximation ratio of this solve vs. the exact optimum.

        Rescales the solution energy onto ``[0, 1]`` using the exact spectrum
        bounds (``1.0`` == optimal). By default it scores the most-likely
        bitstring's energy; pass ``expectation=True`` to score the expectation
        value ``<H_C>`` instead. ``bounds`` may be a precomputed
        ``(min_energy, max_energy)`` pair (from :func:`energy_bounds`) to avoid
        re-enumerating the spectrum; otherwise it is computed on the fly.
        """
        if bounds is None:
            bounds = energy_bounds(self.ch)
        e = self.energy if expectation else self.most_likely_energy()
        return approximation_ratio(e, bounds[0], bounds[1])


# --------------------------------------------------------------------------
# Classical brute-force reference (small n)
# --------------------------------------------------------------------------

def _cut_spectrum(ch: CostHamiltonian, max_qubits: int):
    """Exact cut spectrum of ``H_C`` via the shared vectorized enumerator.

    Refuses ``n > max_qubits`` (guarding the ``2^n`` enumeration) before building
    the augmented Ising graph, whose max/min cut map to the ``H_C`` energy
    bounds through ``E = offset + total_weight - 2 * cut``.
    """
    if ch.n_qubits > max_qubits:
        raise ValueError(
            f"brute force refused for {ch.n_qubits} qubits (> {max_qubits})"
        )
    graph = augmented_ising_graph(ch)
    return enumerate_cut_spectrum(graph, max_nodes=ch.n_qubits + 1)


def brute_force_ground_state(
    ch: CostHamiltonian, max_qubits: int = 26
) -> tuple[float, list[int]]:
    """Exact minimum-energy assignment by enumerating all ``2^n`` bitstrings.

    Only feasible for small ``n``; raises for ``n > max_qubits``. Delegates to
    the shared vectorized cut enumerator (:func:`src.brute_force.enumerate_cut_spectrum`);
    the ground state is the augmented graph's maximum cut.
    """
    spectrum = _cut_spectrum(ch, max_qubits)
    energy = ch.offset + spectrum.total_weight - 2.0 * spectrum.max_value
    partition = dict(zip(spectrum.nodes, spectrum.max_bits))
    return energy, qubo.bits_from_partition(ch, partition)


def energy_bounds(ch: CostHamiltonian, max_qubits: int = 26) -> tuple[float, float]:
    """Exact ``(min_energy, max_energy)`` of ``H_C`` over all ``2^n`` assignments.

    The minimum is the ground state (best fault-zone partition); the maximum the
    worst assignment. Both are needed for the normalized approximation ratio
    (:func:`approximation_ratio`). Only feasible for small ``n``. The augmented
    graph's max cut gives ``min_energy`` and its min cut gives ``max_energy``.
    """
    spectrum = _cut_spectrum(ch, max_qubits)
    e_min = ch.offset + spectrum.total_weight - 2.0 * spectrum.max_value
    e_max = ch.offset + spectrum.total_weight - 2.0 * spectrum.min_value
    return e_min, e_max


def approximation_ratio(
    energy: float, min_energy: float, max_energy: float
) -> float:
    """Normalized approximation ratio in ``[0, 1]`` for a *minimization* objective.

    ``r = (E_max - E) / (E_max - E_min)`` -- ``1`` at the ground state ``E_min``,
    ``0`` at the worst assignment ``E_max`` (higher is better). A degenerate
    spectrum (``E_max == E_min``) returns ``1.0``. See ``docs/optimizers.md``.
    """
    span = max_energy - min_energy
    if span == 0.0:
        return 1.0
    return (max_energy - energy) / span


# --------------------------------------------------------------------------
# Variational loop (minimizes <H_C>)
# --------------------------------------------------------------------------

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
    e_min, e_max = energy_bounds(ch)
    print(f"\nBrute-force optimum: E = {gs_energy:.4f}")
    match = qaoa.most_likely_bits() == gs_bits
    print(f"  QAOA most-likely matches optimum: {match}")

    print("\nApproximation ratio (1.0 == classical optimum, higher is better):")
    print(f"  most-likely bitstring = {qaoa.approximation_ratio((e_min, e_max)):.4f}")
    print(
        "  expectation <H_C>     = "
        f"{qaoa.approximation_ratio((e_min, e_max), expectation=True):.4f}"
    )

    print("\nMost-likely partition (node -> side):")
    for node, side in partition.items():
        marker = "" if partition.get(node) == gs_partition.get(node) else "  (*)"
        print(f"  {node}: {side}{marker}")

    figure = plot_partition(qaoa)
    print(f"\nPainted partition figure: {figure}")
