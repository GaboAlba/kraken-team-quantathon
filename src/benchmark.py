"""Comparative evaluation framework: QAOA (Nexus/Helios) vs. classical baselines.

This module powers ``notebooks/evaluation.ipynb``. It benchmarks the QAOA solver
(run on the **Quantinuum Nexus Helios-1E-lite emulator**) against the classical
Max-Cut baselines (**brute force**, **greedy**, **Goemans-Williamson**) on the
*same* fault-zone QUBO, across a family of grids of growing size and a
configurable ``shots`` x ``max_iter`` hyperparameter sweep, executed in parallel.

Like :mod:`src.qaoa_nexus`, this is **experiment support code**: it needs network
access + an interactive ``qnx.login()`` for the QAOA path, and it is intentionally
kept out of the reproducible pipeline and the test suite. The *offline* pieces
(graph growth, vectorized brute force, classical samplers, metrics) are pure and
unit-testable without Nexus.

Design overview
---------------

* **Grids** grow from the 9-node ``graph.GUANACASTE_NORTH`` baseline by BFS-adding
  adjacent real substations (deterministic) up to each target size
  (:func:`grow_cost_hamiltonians`).
* Every method is scored on the full QUBO objective. The classical Max-Cut
  baselines run on the *augmented Ising graph* (a ``FIELD`` node tied to every
  variable with weight ``h_i`` plus ``J_ij`` couplings): maximizing that graph's
  cut is exactly minimizing ``<H_C>`` (see :func:`augmented_ising_graph`).
* **Brute force** enumerates the whole spectrum once per grid with a fast
  vectorized, **timeout-guarded** routine; on timeout GW becomes the baseline.
* **QAOA** runs COBYLA on Helios; every iteration submits one cloud job whose
  reference is saved, so per-iteration shot distributions are recoverable from the
  Nexus job results (:func:`solve_scipy_helios`).
* :func:`run_all` executes all tasks in parallel and **awaits every task** before
  the notebook does any analysis; :func:`summarize` computes the metrics.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from src import graph as gmod
from src import qaoa as qmod_qaoa
from src import qubo as qmod
from src.qubo import CostHamiltonian

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "experiments" / "results"
REFS_DIR = ROOT / "experiments" / "refs"

FIELD = "__field__"

# --------------------------------------------------------------------------
# Configurable experiment matrix (editable from the notebook)
# --------------------------------------------------------------------------
GRAPH_SIZES = [9, 15, 26]
P_VALUES = [1, 3, 6]
SHOTS_LIST = [5000]
MAXITER_LIST = [100]
N_RUNS = 3  # independent replicate runs per config, for run-level mean/std

DEVICE = "Helios-1E-lite"
PROJECT = "kraken-quantathon"
SEED = 7
BRUTE_FORCE_TIMEOUT_S = 300.0  # always-on catch-all; on timeout GW is the baseline
MAX_WORKERS = 8


# --------------------------------------------------------------------------
# Grid construction: grow from the 9-node baseline
# --------------------------------------------------------------------------

def load_national_graph(raw_dir: Path = RAW_DIR) -> nx.Graph:
    """Build the national NetworkX graph from the static ICE snapshot."""
    subs = json.loads((raw_dir / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((raw_dir / "lines.geojson").read_text(encoding="utf-8"))
    plants_path = raw_dir / "plants.geojson"
    plants = (
        json.loads(plants_path.read_text(encoding="utf-8"))
        if plants_path.exists()
        else None
    )
    G, _ = gmod.build_national_graph(subs, lines, plants_geojson=plants)
    return G


def real_node_graph(G: nx.Graph) -> nx.Graph:
    """Induced subgraph over real substations only (drops border nodes)."""
    real = [n for n, d in G.nodes(data=True) if not d.get("border")]
    return G.subgraph(real).copy()


def grow_subregion(
    H: nx.Graph, baseline: list[str], target_size: int
) -> nx.Graph:
    """Grow a connected subgraph from ``baseline`` up to ``target_size`` nodes.

    Starts from the baseline node set and repeatedly adds the adjacent real node
    with the highest weighted degree (deterministic alphabetical tie-break), which
    keeps the subgraph connected and reproducible. If the connected component of
    the baseline is smaller than ``target_size`` the growth stops early.
    """
    selected = list(dict.fromkeys(n for n in baseline if n in H))
    if not selected:
        raise ValueError("no baseline node is present in the graph")
    chosen = set(selected)
    while len(selected) < target_size:
        frontier = sorted(
            {nb for n in selected for nb in H.neighbors(n) if nb not in chosen},
            key=lambda nb: (-H.degree(nb, weight="weight"), nb),
        )
        if not frontier:
            break
        nxt = frontier[0]
        selected.append(nxt)
        chosen.add(nxt)
    return H.subgraph(selected).copy()


def grow_cost_hamiltonians(
    sizes: list[int] = GRAPH_SIZES,
    baseline: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
) -> dict[int, CostHamiltonian]:
    """Build one :class:`CostHamiltonian` per target grid size.

    Returns ``{size: CostHamiltonian}``. The 9-node baseline is
    ``graph.GUANACASTE_NORTH``; larger grids extend it via :func:`grow_subregion`.
    """
    if baseline is None:
        baseline = gmod.GUANACASTE_NORTH
    H = real_node_graph(load_national_graph(raw_dir))
    out: dict[int, CostHamiltonian] = {}
    for size in sizes:
        sub = grow_subregion(H, baseline, size)
        ch = qmod.qubo_to_cost_hamiltonian(qmod.build_qubo(sub))
        out[size] = ch
    return out


# --------------------------------------------------------------------------
# Augmented Ising graph for the classical Max-Cut baselines
# --------------------------------------------------------------------------

def augmented_ising_graph(ch: CostHamiltonian) -> nx.Graph:
    """Graph whose weighted Max-Cut equals minimizing ``<H_C>``.

    A ``FIELD`` node is connected to variable ``i`` with weight ``h_i`` (its side
    fixes the ``z = +1`` gauge) and variables ``i``/``j`` are connected with weight
    ``J_ij``. Maximizing this graph's cut minimizes ``sum h_i z_i + sum J_ij z_i z_j``.
    """
    H = nx.Graph()
    H.add_node(FIELD)
    for i, coeff in ch.z_terms:
        if coeff:
            H.add_edge(FIELD, ch.variables[i], weight=float(coeff))
    for i, j, coeff in ch.zz_terms:
        if coeff:
            H.add_edge(ch.variables[i], ch.variables[j], weight=float(coeff))
    return H


def bits_from_partition(ch: CostHamiltonian, partition: dict) -> list[int]:
    """Map an augmented-graph partition to QUBO bits (``x = 0`` on the FIELD side)."""
    side = partition[FIELD]
    return [0 if partition[name] == side else 1 for name in ch.variables]


# --------------------------------------------------------------------------
# Vectorized, timeout-guarded brute force (spectrum bounds + ground state)
# --------------------------------------------------------------------------

def _ch_arrays(ch: CostHamiltonian):
    n = ch.n_qubits
    h = np.zeros(n, dtype=np.float64)
    for q, c in ch.z_terms:
        h[q] += c
    zz = ch.zz_terms
    ii = np.array([t[0] for t in zz], dtype=np.int64)
    jj = np.array([t[1] for t in zz], dtype=np.int64)
    Jc = np.array([t[2] for t in zz], dtype=np.float64)
    return n, h, ii, jj, Jc, float(ch.offset)


@dataclass
class Baseline:
    """Approximation-ratio reference for one grid size."""

    size: int
    e_min: float
    e_max: float
    best_bits: list[int]
    time_s: float
    source: str  # "brute_force" or "goemans_williamson"
    timed_out: bool


def brute_force_baseline(
    ch: CostHamiltonian,
    timeout: float = BRUTE_FORCE_TIMEOUT_S,
    chunk_bits: int = 18,
    size: int | None = None,
) -> Baseline:
    """Exact spectrum bounds by vectorized enumeration, guarded by ``timeout``.

    Enumerates all ``2^n`` assignments in chunks, tracking the global min/max
    energy and the ground-state bits. The per-chunk energy is accumulated with a
    Python loop over the (few) Hamiltonian terms rather than a single dense
    ``(chunk x n_terms)`` product, which keeps the peak memory proportional to the
    chunk size (not ``chunk x n_terms``) so it scales to the 26-qubit grid. If the
    elapsed time exceeds ``timeout`` the enumeration aborts and the caller should
    fall back to a GW baseline (:func:`gw_baseline`). Returns a :class:`Baseline`
    with ``timed_out`` set.
    """
    n, h, ii, jj, Jc, offset = _ch_arrays(ch)
    total = 1 << n
    chunk = min(1 << chunk_bits, total)
    bit_pos = np.arange(n, dtype=np.int64)
    h = h.astype(np.float32)
    Jc = Jc.astype(np.float32)

    e_min = np.inf
    e_max = -np.inf
    best_bits: list[int] = []
    start = time.perf_counter()
    timed_out = False

    for s0 in range(0, total, chunk):
        idx = np.arange(s0, min(s0 + chunk, total), dtype=np.int64)
        # z_i in {+1, -1} as float32 columns; index i is qubit i.
        z = (1 - 2 * ((idx[:, None] >> bit_pos) & 1)).astype(np.float32)
        e = np.full(idx.shape[0], np.float32(offset), dtype=np.float32)
        e += z @ h
        for k in range(ii.size):
            e += Jc[k] * (z[:, ii[k]] * z[:, jj[k]])
        cmin = int(e.argmin())
        if e[cmin] < e_min:
            e_min = float(e[cmin])
            best_bits = ((idx[cmin] >> np.arange(n)) & 1).astype(int).tolist()
        e_max = max(e_max, float(e.max()))
        if time.perf_counter() - start > timeout:
            timed_out = True
            break

    return Baseline(
        size=size if size is not None else n,
        e_min=e_min,
        e_max=e_max,
        best_bits=best_bits,
        time_s=time.perf_counter() - start,
        source="brute_force",
        timed_out=timed_out,
    )


def gw_baseline(ch: CostHamiltonian, n_trials: int = 200, seed: int = SEED,
                size: int | None = None) -> Baseline:
    """Fallback baseline from Goemans-Williamson when brute force times out.

    Uses the best GW energy as an ``e_min`` proxy and the worst sampled energy as
    an ``e_max`` proxy, so the approximation ratio stays defined (but approximate).
    """
    start = time.perf_counter()
    energies, bits_list = gw_samples(ch, n_trials, seed=seed)
    order = int(np.argmin(energies))
    return Baseline(
        size=size if size is not None else ch.n_qubits,
        e_min=float(np.min(energies)),
        e_max=float(np.max(energies)),
        best_bits=bits_list[order],
        time_s=time.perf_counter() - start,
        source="goemans_williamson",
        timed_out=True,
    )


def compute_baseline(
    ch: CostHamiltonian,
    size: int,
    timeout: float = BRUTE_FORCE_TIMEOUT_S,
    seed: int = SEED,
) -> Baseline:
    """Exact brute-force baseline, or a GW fallback if it times out."""
    bf = brute_force_baseline(ch, timeout=timeout, size=size)
    if bf.timed_out:
        return gw_baseline(ch, seed=seed, size=size)
    return bf


# --------------------------------------------------------------------------
# Classical samplers (each produces n_shots solution energies)
# --------------------------------------------------------------------------

def greedy_samples(
    ch: CostHamiltonian, n_shots: int, seed: int = SEED
) -> tuple[list[float], list[list[int]]]:
    """``n_shots`` seeded greedy restarts on the augmented Ising graph."""
    from src.classical_baselines import greedy_maxcut

    H = augmented_ising_graph(ch)
    energies: list[float] = []
    bits_list: list[list[int]] = []
    for s in range(n_shots):
        partition, _ = greedy_maxcut(H, seed=seed + s)
        x = bits_from_partition(ch, partition)
        bits_list.append(x)
        energies.append(float(ch.energy(x)))
    return energies, bits_list


def gw_samples(
    ch: CostHamiltonian, n_shots: int, seed: int = SEED
) -> tuple[list[float], list[list[int]]]:
    """One SDP solve + ``n_shots`` hyperplane roundings on the augmented graph."""
    import cvxpy as cp

    H = augmented_ising_graph(ch)
    nodes = list(H.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    W = np.zeros((n, n), dtype=float)
    for u, v, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        W[idx[u], idx[v]] = W[idx[v], idx[u]] = w

    X = cp.Variable((n, n), PSD=True)
    terms = [
        W[i, j] * (1 - X[i, j]) / 2
        for i in range(n)
        for j in range(i + 1, n)
        if W[i, j] != 0
    ]
    prob = cp.Problem(cp.Maximize(cp.sum(terms)), [cp.diag(X) == 1])
    prob.solve(solver=cp.SCS)
    X_val = np.asarray(X.value, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(X_val)
    V = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0, None)))

    rng = np.random.default_rng(seed)
    energies: list[float] = []
    bits_list: list[list[int]] = []
    for _ in range(n_shots):
        r = rng.normal(size=n)
        norm = np.linalg.norm(r)
        if norm == 0:
            r = np.ones(n)
            norm = np.linalg.norm(r)
        r /= norm
        signs = np.sign(V @ r)
        signs[signs == 0] = 1
        partition = {nodes[i]: int(signs[i] > 0) for i in range(n)}
        x = bits_from_partition(ch, partition)
        bits_list.append(x)
        energies.append(float(ch.energy(x)))
    return energies, bits_list


# --------------------------------------------------------------------------
# QAOA on the Nexus Helios-1E-lite emulator
# --------------------------------------------------------------------------

def helios_config(n_qubits: int, device: str = DEVICE):
    """Backend config for the Nexus Helios emulator."""
    from qnexus.models import HeliosConfig
    from quantinuum_schemas.models.backend_config import HeliosEmulatorConfig

    return HeliosConfig(
        system_name=device,
        emulator_config=HeliosEmulatorConfig(n_qubits=n_qubits),
    )


def _energies_from_counts(ch: CostHamiltonian, counts: dict[str, int]):
    """Expand ``{bitstring: count}`` into per-shot energies (repeated by count)."""
    energies: list[float] = []
    for meas, count in counts.items():
        e = ch.energy([int(c) for c in meas])
        energies.extend([e] * int(count))
    return energies


def solve_scipy_helios(
    ch: CostHamiltonian,
    p_value: int,
    n_shots: int,
    max_iter: int,
    size: int,
    seed: int = SEED,
    device: str = DEVICE,
    project_name: str = PROJECT,
    save_refs: bool = True,
) -> dict:
    """Run COBYLA-QAOA on Helios, logging every iteration's job + shot energies.

    Each objective evaluation compiles the kernel for the current angles, uploads
    the HUGR, submits an execute job to the Nexus ``device`` emulator, waits, and
    decodes ``<H_C>``. Every iteration's job reference is retained (and optionally
    saved under ``experiments/refs/``) so the per-shot distributions can be
    re-fetched from the Nexus job results. Returns a plain dict (JSON-friendly)
    with the per-iteration history, per-iteration shot counts/energies, the best
    solution, and timing.
    """
    import qnexus as qnx
    from scipy.optimize import minimize

    from src.qaoa import build_qaoa_instance, energy_from_result
    from src.qaoa_nexus import compile_instance, get_project

    get_project(project_name)
    config = helios_config(ch.n_qubits, device=device)
    instance = build_qaoa_instance(ch, p_value)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 1.0, 2 * p_value)

    tag = f"g{size}-p{p_value}-s{n_shots}-m{max_iter}-seed{seed}"
    history: list[float] = []
    iter_counts: list[dict[str, int]] = []
    job_ids: list[str] = []
    job_refs = []
    start = time.perf_counter()

    def submit(params: np.ndarray, label: str):
        cost = [float(v) for v in params[:p_value]]
        mixer = [float(v) for v in params[p_value:]]
        pkg = compile_instance(instance, cost, mixer)
        ref = qnx.hugr.upload(hugr_package=pkg, name=f"qaoa-{tag}-{label}")
        job = qnx.start_execute_job(
            programs=[ref], n_shots=[n_shots], backend_config=config,
            name=f"qaoa-{tag}-{label}",
        )
        qnx.jobs.wait_for(job)
        result = qnx.jobs.results(job)[0].download_result()
        counts = dict(result.register_counts()["c"])
        energy = energy_from_result(ch, result, n_shots)
        job_refs.append(job)
        job_ids.append(getattr(job, "id", str(job)))
        return energy, counts

    def objective(params: np.ndarray) -> float:
        energy, counts = submit(params, f"it{len(history)}")
        history.append(float(energy))
        iter_counts.append(counts)
        return energy

    opt = minimize(objective, x0, method="COBYLA", options={"maxiter": max_iter})

    best_cost = np.asarray(opt.x[:p_value], dtype=float)
    best_energy, best_counts = submit(opt.x, "final")
    # Most-likely bitstring by shot frequency (ties -> lexicographic).
    ml = max(best_counts.items(), key=lambda kv: kv[1])[0]
    best_bits = [int(c) for c in ml]

    if save_refs:
        REFS_DIR.mkdir(parents=True, exist_ok=True)
        for k, job in enumerate(job_refs):
            try:
                qnx.filesystem.save(ref=job, path=REFS_DIR / tag / f"job_{k}",
                                    mkdir=True)
            except Exception:  # ref persistence is best-effort
                pass

    return {
        "backend": f"nexus:{device}",
        "optimizer": "scipy:COBYLA",
        "history": history,
        "iter_counts": iter_counts,
        "final_counts": best_counts,
        "job_ids": job_ids,
        "best_energy": float(best_energy),
        "best_bits": best_bits,
        "cost_angles": best_cost.tolist(),
        "mixer_angles": np.asarray(opt.x[p_value:], dtype=float).tolist(),
        "n_evaluations": len(history),
        "converged": bool(opt.success),
        "time_s": time.perf_counter() - start,
    }


# --------------------------------------------------------------------------
# Evaluation tasks + parallel driver
# --------------------------------------------------------------------------

@dataclass
class EvalRecord:
    """One optimizer run under one configuration (one replicate)."""

    optimizer: str  # "greedy" | "goemans_williamson" | f"qaoa_p{p}"
    size: int
    n_shots: int
    max_iter: int | None
    p: int | None
    sample_energies: list[float]  # per-sample / per-shot energies (best iter for QAOA)
    time_s: float
    run: int = 0  # replicate index (independent seed) for run-level mean/std
    extra: dict = field(default_factory=dict)


# Seed offset between replicate runs so each run is statistically independent.
RUN_SEED_STRIDE = 10_000


def _run_seed(seed: int, run: int) -> int:
    return seed + run * RUN_SEED_STRIDE


def _classical_record(kind: str, ch: CostHamiltonian, size: int, n_shots: int,
                      seed: int, run: int = 0) -> EvalRecord:
    t0 = time.perf_counter()
    rseed = _run_seed(seed, run)
    if kind == "greedy":
        energies, _ = greedy_samples(ch, n_shots, seed=rseed)
    else:
        energies, _ = gw_samples(ch, n_shots, seed=rseed)
    return EvalRecord(
        optimizer=kind, size=size, n_shots=n_shots, max_iter=None, p=None,
        sample_energies=energies, time_s=time.perf_counter() - t0, run=run,
        extra={"best_energy": float(min(energies)) if energies else None},
    )


def _qaoa_record(ch: CostHamiltonian, size: int, p: int, n_shots: int,
                 max_iter: int, seed: int, device: str, run: int = 0) -> EvalRecord:
    out = solve_scipy_helios(ch, p_value=p, n_shots=n_shots, max_iter=max_iter,
                             size=size, seed=_run_seed(seed, run), device=device)
    # QAOA "samples": per-shot energies of the best (lowest <H_C>) iteration.
    if out["history"]:
        best_it = int(np.argmin(out["history"]))
        counts = out["iter_counts"][best_it]
    else:
        counts = out["final_counts"]
    sample_energies = _energies_from_counts(ch, counts)
    return EvalRecord(
        optimizer=f"qaoa_p{p}", size=size, n_shots=n_shots, max_iter=max_iter,
        p=p, sample_energies=sample_energies, time_s=out["time_s"], run=run,
        extra=out,
    )


def build_tasks(
    hamiltonians: dict[int, CostHamiltonian],
    p_values: list[int] = P_VALUES,
    shots_list: list[int] = SHOTS_LIST,
    maxiter_list: list[int] = MAXITER_LIST,
    n_runs: int = N_RUNS,
    seed: int = SEED,
    device: str = DEVICE,
    include_qaoa: bool = True,
) -> list[tuple[str, callable]]:
    """Build the parallel task list (label, thunk) for the full cross-product.

    Classical greedy/GW depend only on ``(size, shots)`` (not ``max_iter``), so
    each is scheduled once per ``(size, shots)`` per run. QAOA is scheduled per
    ``(size, p, shots, max_iter)`` per run. Each of the ``n_runs`` replicates uses
    an independent seed, so run-level mean/std (via :func:`aggregate_runs`) reflect
    the stochasticity of the solvers -- QAOA in particular needs several runs for a
    meaningful mean/std.
    """
    tasks: list[tuple[str, callable]] = []
    for size, ch in hamiltonians.items():
        for shots in shots_list:
            for run in range(n_runs):
                for kind in ("greedy", "goemans_williamson"):
                    tasks.append((
                        f"{kind}-g{size}-s{shots}-r{run}",
                        (lambda k=kind, c=ch, s=size, sh=shots, r=run:
                            _classical_record(k, c, s, sh, seed, r)),
                    ))
                if include_qaoa:
                    for p in p_values:
                        for mit in maxiter_list:
                            tasks.append((
                                f"qaoa_p{p}-g{size}-s{shots}-m{mit}-r{run}",
                                (lambda c=ch, s=size, pp=p, sh=shots, mm=mit, r=run:
                                    _qaoa_record(c, s, pp, sh, mm, seed, device, r)),
                            ))
    return tasks


def compute_baselines(
    hamiltonians: dict[int, CostHamiltonian],
    timeout: float = BRUTE_FORCE_TIMEOUT_S,
    seed: int = SEED,
    max_workers: int = MAX_WORKERS,
) -> dict[int, Baseline]:
    """Compute every grid's approximation-ratio baseline in parallel."""
    baselines: dict[int, Baseline] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(compute_baseline, ch, size, timeout, seed): size
            for size, ch in hamiltonians.items()
        }
        wait(futs)
        for fut, size in futs.items():
            baselines[size] = fut.result()
    return baselines


def run_all(
    tasks: list[tuple[str, callable]],
    max_workers: int = MAX_WORKERS,
) -> list[EvalRecord]:
    """Run every task in parallel and **await all** before returning records."""
    records: list[EvalRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(thunk): label for label, thunk in tasks}
        wait(futs)  # block until every task has finished
        for fut, label in futs.items():
            try:
                records.append(fut.result())
            except Exception as exc:  # keep going; surface the failure in the record
                print(f"[run_all] task {label} failed: {exc!r}")
    return records


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def approximation_ratios(record: EvalRecord, baseline: Baseline) -> list[float]:
    """Per-sample approximation ratios in ``[0, 1]`` (1 = optimum)."""
    return [
        qmod_qaoa.approximation_ratio(e, baseline.e_min, baseline.e_max)
        for e in record.sample_energies
    ]


def summarize(
    records: list[EvalRecord], baselines: dict[int, Baseline]
) -> list[dict]:
    """Per-record (per-run) metrics as tidy rows.

    One row per replicate run. Each row has: optimizer, size, n_shots, max_iter,
    p, run, mean/std of the sampled energies, MSE of the sampled energies vs. the
    baseline optimum ``e_min`` (``None`` when no baseline exists), mean/best
    approximation ratio, and the execution time. Use :func:`aggregate_runs` for the
    run-level mean/std across replicates.
    """
    rows: list[dict] = []
    for rec in records:
        base = baselines.get(rec.size)
        energies = np.asarray(rec.sample_energies, dtype=float)
        ratios = approximation_ratios(rec, base) if base else []
        if base is not None:
            mse = float(np.mean((energies - base.e_min) ** 2)) if energies.size else None
        else:
            mse = None
        rows.append({
            "optimizer": rec.optimizer,
            "size": rec.size,
            "n_shots": rec.n_shots,
            "max_iter": rec.max_iter,
            "p": rec.p,
            "run": rec.run,
            "mean_energy": float(energies.mean()) if energies.size else None,
            "std_energy": float(energies.std()) if energies.size else None,
            "mse_vs_optimum": mse,
            "mean_approx_ratio": float(np.mean(ratios)) if ratios else None,
            "best_approx_ratio": float(np.max(ratios)) if ratios else None,
            "time_s": rec.time_s,
            "baseline_source": base.source if base else None,
        })
    return rows


def _config_key(rec: EvalRecord) -> tuple:
    return (rec.optimizer, rec.size, rec.n_shots, rec.max_iter, rec.p)


def aggregate_runs(
    records: list[EvalRecord], baselines: dict[int, Baseline]
) -> list[dict]:
    """Run-level mean/std across replicate runs, grouped by configuration.

    Groups records that share ``(optimizer, size, n_shots, max_iter, p)`` and
    reduces the ``n_runs`` replicates to a single row reporting, across runs, the
    **mean and std of the execution time**, of the **best energy found**, of the
    **MSE vs. the baseline optimum**, and of the **approximation ratio** (both the
    per-run best and the per-run mean-over-samples). This is the table that gives a
    statistically meaningful QAOA mean/std, which a single run cannot.
    """
    groups: dict[tuple, list[EvalRecord]] = {}
    for rec in records:
        groups.setdefault(_config_key(rec), []).append(rec)

    def stats(values: list[float]) -> tuple[float | None, float | None]:
        arr = np.asarray([v for v in values if v is not None], dtype=float)
        if not arr.size:
            return None, None
        return float(arr.mean()), float(arr.std())

    rows: list[dict] = []
    for (optimizer, size, n_shots, max_iter, p), recs in groups.items():
        base = baselines.get(size)
        best_energies, mses, mean_ars, best_ars, times = [], [], [], [], []
        for rec in recs:
            e = np.asarray(rec.sample_energies, dtype=float)
            times.append(rec.time_s)
            if not e.size:
                continue
            best_energies.append(float(e.min()))
            if base is not None:
                mses.append(float(np.mean((e - base.e_min) ** 2)))
                ratios = [
                    qmod_qaoa.approximation_ratio(x, base.e_min, base.e_max)
                    for x in e
                ]
                mean_ars.append(float(np.mean(ratios)))
                best_ars.append(float(np.max(ratios)))
        mt, st = stats(times)
        mbe, sbe = stats(best_energies)
        mmse, smse = stats(mses)
        mmar, smar = stats(mean_ars)
        mbar, sbar = stats(best_ars)
        rows.append({
            "optimizer": optimizer, "size": size, "n_shots": n_shots,
            "max_iter": max_iter, "p": p, "n_runs": len(recs),
            "mean_time_s": mt, "std_time_s": st,
            "mean_best_energy": mbe, "std_best_energy": sbe,
            "mean_mse_vs_optimum": mmse, "std_mse_vs_optimum": smse,
            "mean_approx_ratio": mmar, "std_approx_ratio": smar,
            "mean_best_approx_ratio": mbar, "std_best_approx_ratio": sbar,
            "baseline_source": base.source if base else None,
        })
    rows.sort(key=lambda r: (r["size"], r["n_shots"], str(r["max_iter"]),
                             r["optimizer"]))
    return rows


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_results(
    records: list[EvalRecord],
    baselines: dict[int, Baseline],
    rows: list[dict],
    aggregated: list[dict] | None = None,
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """Persist raw records, baselines, per-run and run-aggregated tables to JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"evaluation_{stamp}.json"
    doc = {
        "config": {
            "graph_sizes": GRAPH_SIZES, "p_values": P_VALUES,
            "shots_list": SHOTS_LIST, "maxiter_list": MAXITER_LIST,
            "n_runs": N_RUNS, "device": DEVICE, "seed": SEED,
        },
        "baselines": {
            str(size): {
                "e_min": b.e_min, "e_max": b.e_max, "source": b.source,
                "timed_out": b.timed_out, "time_s": b.time_s,
                "best_bits": b.best_bits,
            }
            for size, b in baselines.items()
        },
        "records": [
            {
                "optimizer": r.optimizer, "size": r.size, "n_shots": r.n_shots,
                "max_iter": r.max_iter, "p": r.p, "run": r.run, "time_s": r.time_s,
                "sample_energies": r.sample_energies,
                "history": r.extra.get("history"),
                "job_ids": r.extra.get("job_ids"),
            }
            for r in records
        ],
        "summary": rows,
        "aggregated": aggregated if aggregated is not None else [],
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path
