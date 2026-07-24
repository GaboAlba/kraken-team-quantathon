"""Run the grid QAOA on Quantinuum Nexus (cloud) instead of the local emulator.

The Guppy kernel in :mod:`src.qaoa` is **backend-agnostic**:
:func:`build_qaoa_instance` compiles to HUGR, and Nexus executes it on its hosted
Selene emulator (:class:`qnexus.SeleneConfig` with a ``StatevectorSimulator``).
A HUGR execute job returns a ``QsysResult`` -- the same type the local
``main.emulator(...).run()`` returns -- so :func:`src.qaoa.energy_from_result`
and :class:`src.qaoa.QAOAResult` decode the counts unchanged. Only the *run* call
differs: ``qnx.hugr.upload`` -> ``qnx.start_execute_job`` -> ``wait_for`` ->
``download_result`` in place of ``main.emulator(...).run()``.

This is an **experiment script**, not part of the reproducible pipeline: it needs
network access, an interactive ``qnx.login()``, and submits real cloud jobs (one
per objective evaluation). Kept out of the test suite; import it only from the
Nexus notebook / experiment code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.qaoa import (
    DEFAULT_GRAPH,
    DEFAULT_LAYERS,
    DEFAULT_MAXITER,
    DEFAULT_QUBO,
    DEFAULT_SEED,
    DEFAULT_SHOTS,
    QAOAResult,
    build_qaoa_instance,
    cost_hamiltonian_from_graph,
    energy_from_result,
    load_cost_hamiltonian,
)
from src.qubo import CostHamiltonian

DEFAULT_PROJECT = "kraken-quantathon"


# --------------------------------------------------------------------------
# Nexus session helpers
# --------------------------------------------------------------------------

def login() -> None:
    """Open a browser to authenticate with Quantinuum Nexus (token lasts ~30 days)."""
    import qnexus as qnx

    qnx.login()


def get_project(name: str = DEFAULT_PROJECT):
    """Get-or-create the Nexus project and make it the active context.

    Required before uploading programs or submitting jobs.
    """
    import qnexus as qnx

    project = qnx.projects.get_or_create(name=name)
    qnx.context.set_active_project(project)
    return project


def selene_config(n_qubits: int, seed: int = DEFAULT_SEED, error_model=None):
    """Nexus-hosted Selene emulator config.

    Uses a ``StatevectorSimulator`` (exact statevector, the cloud twin of the local
    ``Quest`` engine) seeded with ``seed`` so runs are reproducible. Pass an
    ``error_model`` (e.g. ``DepolarizingErrorModel``) to approximate hardware noise;
    the default is noiseless.
    """
    import qnexus as qnx
    from quantinuum_schemas.models.emulator_config import StatevectorSimulator

    kwargs = {
        "n_qubits": n_qubits,
        "simulator": StatevectorSimulator(seed=seed),
    }
    if error_model is not None:
        kwargs["error_model"] = error_model
    return qnx.SeleneConfig(**kwargs)


# --------------------------------------------------------------------------
# Compile the Guppy kernel to HUGR for a given parameter set
# --------------------------------------------------------------------------

def compile_instance(instance, cost_angles, mixer_angles):
    """Compile a Guppy ``main`` entrypoint (for one parameter set) to a HUGR package.

    Guppy execution entrypoints cannot take runtime arguments, so the concrete
    ``cost``/``mixer`` angles are baked in via ``comptime`` and a fresh ``main`` is
    compiled per evaluation -- exactly as the local path does before ``.emulator()``.
    Returns the ``hugr.package.Package`` accepted by :func:`qnexus.hugr.upload`.
    """
    from guppylang import guppy
    from guppylang.std.builtins import comptime, result as guppy_result
    from guppylang.std.quantum import measure_array

    cost = [float(x) for x in cost_angles]
    mixer = [float(x) for x in mixer_angles]

    @guppy
    def main() -> None:
        qs = instance(comptime(cost), comptime(mixer))
        guppy_result("c", measure_array(qs))

    return main.compile()


# --------------------------------------------------------------------------
# One forward pass on Nexus
# --------------------------------------------------------------------------

def eval_qaoa_energy_nexus(
    cost_angles,
    mixer_angles,
    ch: CostHamiltonian,
    config,
    shots: int,
    instance=None,
    n_layers: int | None = None,
    job_name: str = "qaoa-eval",
):
    """Submit one QAOA circuit to Nexus and return ``(<H_C>, QsysResult)``.

    Blocks on ``qnx.jobs.wait_for`` until the cloud job finishes, then decodes the
    returned ``QsysResult`` with the same :func:`energy_from_result` used locally.
    """
    import qnexus as qnx

    if instance is None:
        instance = build_qaoa_instance(ch, n_layers or len(cost_angles))

    pkg = compile_instance(instance, cost_angles, mixer_angles)
    ref = qnx.hugr.upload(hugr_package=pkg, name=job_name)
    job = qnx.start_execute_job(
        programs=[ref], n_shots=[shots], backend_config=config, name=job_name,
    )
    qnx.jobs.wait_for(job)
    qsys = qnx.jobs.results(job)[0].download_result()
    return energy_from_result(ch, qsys, shots), qsys


# --------------------------------------------------------------------------
# Variational loop against the Nexus emulator
# --------------------------------------------------------------------------

def solve_scipy_nexus(
    ch: CostHamiltonian,
    config=None,
    p_value: int = DEFAULT_LAYERS,
    n_shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
    maxiter: int = DEFAULT_MAXITER,
    method: str = "COBYLA",
    project_name: str = DEFAULT_PROJECT,
    progress=None,
) -> QAOAResult:
    """Minimize ``<H_C>`` with SciPy (COBYLA), evaluating every point on Nexus.

    Mirrors :func:`src.qaoa.solve_scipy` but each objective evaluation submits a real
    cloud execute job to the Nexus Selene emulator. Ensures the project context is set,
    reuses one compiled kernel across parameter sets, and returns the standard
    :class:`QAOAResult` (so ``plot_partition`` etc. work with the Nexus outcome).

    ``progress(iteration, energy)`` is called after each evaluation for live monitoring.
    """
    from scipy.optimize import minimize

    get_project(project_name)
    if config is None:
        config = selene_config(ch.n_qubits, seed)

    instance = build_qaoa_instance(ch, p_value)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 1.0, 2 * p_value)

    history: list[float] = []

    def objective(params: NDArray[np.float64]) -> float:
        cost = params[:p_value]
        mixer = params[p_value:]
        energy, _ = eval_qaoa_energy_nexus(
            cost, mixer, ch, config, shots=n_shots, instance=instance,
            job_name=f"qaoa-p{p_value}-it{len(history)}",
        )
        history.append(float(energy))
        if progress is not None:
            progress(len(history), float(energy))
        return energy

    opt = minimize(objective, x0, method=method, options={"maxiter": maxiter})

    best_cost = np.asarray(opt.x[:p_value], dtype=float)
    best_mixer = np.asarray(opt.x[p_value:], dtype=float)
    best_energy, best_result = eval_qaoa_energy_nexus(
        best_cost, best_mixer, ch, config, shots=n_shots, instance=instance,
        job_name=f"qaoa-p{p_value}-final",
    )

    return QAOAResult(
        energy=best_energy,
        cost_angles=best_cost,
        mixer_angles=best_mixer,
        result=best_result,
        ch=ch,
        metadata={
            "backend": "nexus:selene",
            "optimizer": f"scipy:{method}",
            "project": project_name,
            "p": p_value,
            "n_shots": n_shots,
            "seed": seed,
            "maxiter": maxiter,
            "n_evaluations": len(history),
            "converged": bool(opt.success),
            "history": history,
        },
    )


# --------------------------------------------------------------------------
# Convenience loader (mirrors src.qaoa.build)
# --------------------------------------------------------------------------

def load_hamiltonian(
    qubo_path: Path = DEFAULT_QUBO, graph_path: Path = DEFAULT_GRAPH
) -> CostHamiltonian:
    """Load ``H_C`` from ``qubo_cr.json`` (or rebuild from ``grid_cr.json`` if absent)."""
    if Path(qubo_path).exists():
        return load_cost_hamiltonian(qubo_path)
    return cost_hamiltonian_from_graph(graph_path)
