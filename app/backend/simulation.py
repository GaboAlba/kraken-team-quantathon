"""Staged simulation runs: classical solvers + injected quantum runner.

Run-everything policy: every stage executes whenever it is computationally
possible, however long it takes; only the physically impossible is skipped
(with the time estimate that justifies it).

- Brute force: vectorized enumeration up to n = 40 (2^40 states, with live
  progress/ETA); beyond that 2^n would take years-to-eons and is skipped,
  leaving the SDP certified lower bound as the quality reference.
- QAOA angles: exact statevector search up to n = 22 (memory/time bound);
  heuristic untuned angles beyond, clearly labeled.
- Quantum job: ALWAYS attempted -- Nexus enforces its own limits.
"""
from __future__ import annotations

import sys
import threading
import time
import uuid
from typing import Callable

import networkx as nx
import numpy as np

from grid_service import (BRUTE_FORCE_MAX_N, EXACT_ANGLES_MAX_N,
                          REPO, national, tier_for)

sys.path.insert(0, str(REPO))
from src import classical_baselines as cb
from src import graph as graph_mod
from src import qubo as qubo_mod

STAGES = ["build_qubo", "brute_force", "greedy", "goemans_williamson",
          "qaoa_angles", "nexus_job", "analysis"]
FIELD = "__field__"
HEURISTIC_BETA = 0.3927  # pi/8
BF_BATCH = 1 << 18       # states per vectorized brute-force batch
BF_SAMPLE_MAX = 20000    # energies shipped to the frontend chart

QuantumRunner = Callable[..., tuple[list[list[int]], str, dict]]


def build_augmented(q: qubo_mod.QUBO, ising: dict) -> nx.Graph:
    """Augmented Ising graph; field node FIRST so brute force pins z=+1."""
    H = nx.Graph()
    H.add_node(FIELD)
    for i, name in enumerate(q.variables):
        if ising["h"][i]:
            H.add_edge(FIELD, name, weight=ising["h"][i])
    for (i, j), J in ising["J"].items():
        if J:
            H.add_edge(q.variables[i], q.variables[j], weight=J)
    return H


def x_of_partition(q: qubo_mod.QUBO, partition: dict) -> list[int]:
    side = partition[FIELD]
    return [0 if partition[name] == side else 1 for name in q.variables]


def optimize_angles(energies: np.ndarray) -> tuple[float, float]:
    """QAOA p=1 grid search on the diagonal spectrum (statevector, exact)."""
    n = int(np.log2(len(energies)))

    def expectation(gamma: float, beta: float) -> float:
        psi = np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=complex)
        psi *= np.exp(-1j * gamma * energies)
        c, s = np.cos(beta), -1j * np.sin(beta)
        psi = psi.reshape((2,) * n)
        for ax in range(n):
            a = np.take(psi, 0, axis=ax)
            b = np.take(psi, 1, axis=ax)
            psi = np.stack((c * a + s * b, s * a + c * b), axis=ax)
        p = np.abs(psi.reshape(-1)) ** 2
        return float(p @ energies)

    best = (np.inf, 0.0, 0.0)
    for g in np.linspace(0.02, 1.2, 24):
        for b in np.linspace(0.05, np.pi / 2, 16):
            e = expectation(g, b)
            if e < best[0]:
                best = (e, float(g), float(b))
    return best[1], best[2]


def qubo_matrix(q: qubo_mod.QUBO) -> tuple[np.ndarray, float]:
    """Dense upper-triangular QUBO matrix (linear terms on the diagonal)."""
    n = len(q.variables)
    Q = np.zeros((n, n))
    for i, c in q.linear.items():
        Q[i, i] += c
    for (i, j), c in q.quadratic.items():
        Q[i, j] += c
    return Q, float(q.offset)


def heuristic_angles(q: qubo_mod.QUBO) -> tuple[float, float]:
    """Untuned p=1 angles for sizes where the statevector search is impossible.

    gamma scales inversely with the weight magnitude so the phase per term
    stays O(1); beta = pi/8 is a common single-layer default.
    """
    w_max = float(q.metadata.get("max_edge_weight") or 1.0)
    return 0.5 / max(w_max, 1e-9), HEURISTIC_BETA


def sdp_reference(H: nx.Graph, ising_offset: float) -> float:
    """Certified lower bound on the QUBO optimum from the Max-Cut SDP.

    ``E(z) = offset + sum_J - 2 * cut`` over the augmented graph, and the SDP
    upper-bounds the cut, so ``E_min >= offset + sum_J - 2 * SDP``.
    """
    import cvxpy as cp

    nodes = list(H.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    m = len(nodes)
    W = np.zeros((m, m))
    for u, v, d in H.edges(data=True):
        W[idx[u], idx[v]] = W[idx[v], idx[u]] = float(d.get("weight", 0.0))
    X = cp.Variable((m, m), PSD=True)
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(W, 1 - X)) / 4),
                      [cp.diag(X) == 1])
    prob.solve(solver=cp.SCS)
    sum_J = float(W.sum() / 2)
    return float(ising_offset + sum_J - 2 * float(prob.value))


class RunManager:
    def __init__(self) -> None:
        self._runs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def launch(self, nodes: list[str], shots: int,
               quantum_runner: QuantumRunner) -> str:
        run_id = uuid.uuid4().hex[:12]
        run = {
            "id": run_id, "status": "running",
            "started_ts": time.time(), "finished_ts": None,
            "stages": [{"name": s, "state": "pending", "detail": "",
                        "started_ts": None, "duration_s": None}
                       for s in STAGES],
            "log": [], "results": None, "progress_pct": 0,
        }
        with self._lock:
            self._runs[run_id] = run
        t = threading.Thread(target=self._execute,
                             args=(run, nodes, shots, quantum_runner),
                             daemon=True)
        t.start()
        return run_id

    def get(self, run_id: str) -> dict | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            now = time.time()
            stages = []
            for st in run["stages"]:
                elapsed = None
                if st["state"] == "running" and st["started_ts"] is not None:
                    elapsed = now - st["started_ts"]
                elif st["duration_s"] is not None:
                    elapsed = st["duration_s"]
                stages.append({"name": st["name"], "state": st["state"],
                               "detail": st["detail"], "elapsed_s": elapsed})
            end = run["finished_ts"] if run["finished_ts"] is not None else now
            out = {k: v for k, v in run.items()
                   if k not in ("stages", "log", "started_ts", "finished_ts")}
            out["stages"] = stages
            out["log"] = list(run["log"])
            out["elapsed_s"] = end - run["started_ts"]
            return out

    # ------------------------------------------------------------------
    def _log(self, run: dict, msg: str) -> None:
        with self._lock:
            run["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _stage(self, run: dict, idx: int, state: str, detail: str = "") -> None:
        with self._lock:
            st = run["stages"][idx]
            now = time.time()
            if state == "running" and st["started_ts"] is None:
                st["started_ts"] = now
            if state in ("done", "error", "skipped"):
                st["duration_s"] = (now - st["started_ts"]
                                    if st["started_ts"] is not None else 0.0)
            st["state"] = state
            st["detail"] = detail
            finished = sum(1 for s in run["stages"]
                           if s["state"] in ("done", "skipped"))
            run["progress_pct"] = int(100 * finished / len(STAGES))

    def _finish(self, run: dict) -> None:
        with self._lock:
            run["finished_ts"] = time.time()

    def _execute(self, run: dict, nodes: list[str], shots: int,
                 quantum_runner: QuantumRunner) -> None:
        results: dict = {"tier": None, "optimum": None, "reference": None,
                         "methods": {"brute_force": None, "greedy": None,
                                     "goemans_williamson": None, "qaoa": None}}
        try:
            # 1 -- QUBO
            self._stage(run, 0, "running")
            G = national()
            sub = graph_mod.extract_subregion(G, nodes=nodes,
                                              max_nodes=len(nodes))
            q = qubo_mod.build_qubo(sub)
            ising = qubo_mod.qubo_to_ising(q)
            n = len(q.variables)
            tier = tier_for(n)
            results["tier"] = tier
            self._log(run, f"QUBO built: {n} variables, "
                           f"{len(q.quadratic)} couplings — tier: {tier}")
            self._stage(run, 0, "done", f"{n} vars · {tier}")

            # 2 -- brute force: run whenever computationally reachable.
            if n <= BRUTE_FORCE_MAX_N:
                self._stage(run, 1, "running", f"2^{n} states")
                t0 = time.perf_counter()
                Q, offset = qubo_matrix(q)
                total = 1 << n
                keep_full = n <= EXACT_ANGLES_MAX_N
                all_E = np.empty(total) if keep_full else None
                stride = max(total // BF_SAMPLE_MAX, 1)
                sample: list[float] = []
                best_E, best_idx = np.inf, 0
                arange_n = np.arange(n, dtype=np.int64)
                for start in range(0, total, BF_BATCH):
                    idx = np.arange(start, min(start + BF_BATCH, total),
                                    dtype=np.int64)
                    S = ((idx[:, None] >> arange_n) & 1).astype(np.float64)
                    E = np.einsum("bi,ij,bj->b", S, Q, S) + offset
                    if keep_full:
                        all_E[start:start + len(E)] = E
                    sample.extend(E[::stride].tolist())
                    b = int(np.argmin(E))
                    if E[b] < best_E:
                        best_E, best_idx = float(E[b]), int(idx[b])
                    done_states = start + len(E)
                    if done_states < total:
                        elapsed = time.perf_counter() - t0
                        eta = elapsed / done_states * (total - done_states)
                        self._stage(run, 1, "running",
                                    f"{100 * done_states / total:.0f}% of 2^{n} "
                                    f"· ETA {eta:.0f}s")
                E_opt = best_E
                x_opt = [(best_idx >> i) & 1 for i in range(n)]
                bf_ms = (time.perf_counter() - t0) * 1000
                results["optimum"] = {
                    "energy": E_opt,
                    "partition": {
                        "A": sorted(v for v, b in zip(q.variables, x_opt) if b == 0),
                        "B": sorted(v for v, b in zip(q.variables, x_opt) if b == 1)}}
                results["reference"] = {"type": "exact", "energy": E_opt}
                results["methods"]["brute_force"] = {
                    "best_energy": E_opt, "time_ms": bf_ms,
                    "energies": sample[:BF_SAMPLE_MAX], "n_states": total}
                self._log(run, f"Brute force: optimum {E_opt:.4f} "
                               f"({total} states, {bf_ms / 1000:.1f} s)")
                self._stage(run, 1, "done", f"E={E_opt:.3f}")
            else:
                all_E = None
                years = (1 << n) / 5e6 / 3.15e7
                self._log(run, f"Brute force skipped: 2^{n} states would take "
                               f"~{years:.0f} years — computationally out of reach")
                self._stage(run, 1, "skipped", f"2^{n} ≈ {years:.0f} years")

            H = build_augmented(q, ising)

            def score(partition: dict) -> float:
                return float(q.energy(x_of_partition(q, partition)))

            # 3 -- greedy
            self._stage(run, 2, "running")
            t0 = time.perf_counter()
            g_energies = [score(cb.greedy_maxcut(H, seed=s)[0])
                          for s in range(20)]
            g_ms = (time.perf_counter() - t0) * 1000
            results["methods"]["greedy"] = {
                "best_energy": min(g_energies), "time_ms": g_ms,
                "energies": g_energies}
            self._log(run, f"Greedy: best {min(g_energies):.4f} over 20 restarts")
            self._stage(run, 2, "done", f"E={min(g_energies):.3f}")

            # 4 -- GW + certified SDP bound
            self._stage(run, 3, "running")
            t0 = time.perf_counter()
            gw_best_e = score(cb.goemans_williamson(
                H, n_rounding_trials=50, seed=42)[0])
            gw_trials = [score(cb.goemans_williamson(
                H, n_rounding_trials=1, seed=s)[0]) for s in range(10)]
            bound = sdp_reference(H, ising["offset"])
            gw_ms = (time.perf_counter() - t0) * 1000
            results["methods"]["goemans_williamson"] = {
                "best_energy": gw_best_e, "time_ms": gw_ms,
                "energies": gw_trials}
            results["sdp_bound_energy"] = bound
            if results["reference"] is None:
                results["reference"] = {"type": "sdp_bound", "energy": bound}
            self._log(run, f"Goemans-Williamson: best {gw_best_e:.4f}; "
                           f"SDP certified lower bound {bound:.4f}")
            self._stage(run, 3, "done", f"E={gw_best_e:.3f}")

            # 5 -- angles
            self._stage(run, 4, "running")
            if all_E is not None:
                gamma, beta = optimize_angles(all_E)
                self._log(run, f"QAOA angles (statevector): "
                               f"gamma={gamma:.4f}, beta={beta:.4f}")
                self._stage(run, 4, "done", f"g={gamma:.3f} b={beta:.3f}")
            else:
                gamma, beta = heuristic_angles(q)
                self._log(run, f"QAOA angles (heuristic, untuned — statevector "
                               f"impossible at n={n}): gamma={gamma:.4f}, "
                               f"beta={beta:.4f}")
                self._stage(run, 4, "done",
                            f"g={gamma:.3f} b={beta:.3f} (heuristic)")

            # 6 -- quantum: always attempted; Nexus decides its own limits.
            if True:
                self._stage(run, 5, "running", "submitting")
                try:
                    def q_log(m: str) -> None:
                        self._log(run, m)
                        if m.startswith("Job status:"):
                            self._stage(run, 5, "running",
                                        m.removeprefix("Job status:").strip())

                    bits, job_id, timing = quantum_runner(
                        ising=ising, gamma=gamma, beta=beta, n=n, shots=shots,
                        log=q_log)
                    qaoa_E = [float(q.energy(x)) for x in bits]
                    results["methods"]["qaoa"] = {
                        "best_energy": min(qaoa_E),
                        "mean_energy": float(np.mean(qaoa_E)),
                        "shots": len(qaoa_E), "energies": qaoa_E,
                        "gamma": gamma, "beta": beta, "job_id": job_id,
                        "queued_s": float(timing.get("queued_s", 0.0)),
                        "running_s": float(timing.get("running_s", 0.0))}
                    self._log(run, f"Quantum job {job_id}: best {min(qaoa_E):.4f}")
                    self._stage(run, 5, "done", f"E={min(qaoa_E):.3f}")
                except Exception as exc:                   # noqa: BLE001
                    self._log(run, f"Quantum stage failed: {exc}")
                    self._stage(run, 5, "error", str(exc)[:120])
                    with self._lock:
                        run["status"] = "error"

            # 7 -- analysis: comparative metrics against the final reference
            self._stage(run, 6, "running")
            ref = results["reference"]
            ref_E = float(ref["energy"]) if ref else 0.0
            exact = ref is not None and ref["type"] == "exact"

            def gap(e: float) -> float:
                return 100.0 * (e - ref_E) / max(abs(ref_E), 1e-9)

            for name, method in results["methods"].items():
                if method is None:
                    continue
                best = method["best_energy"]
                method["gap_pct"] = gap(best)
                method["found_optimum"] = (
                    bool(best <= ref_E + 1e-9) if exact else None)
            qaoa = results["methods"]["qaoa"]
            if qaoa is not None:
                if exact:
                    hits = sum(1 for e in qaoa["energies"] if e <= ref_E + 1e-9)
                    qaoa["p_optimal"] = hits / len(qaoa["energies"])
                    qaoa["first_optimal_shot"] = next(
                        (i + 1 for i, e in enumerate(qaoa["energies"])
                         if e <= ref_E + 1e-9), None)
                else:
                    qaoa["p_optimal"] = None
                    qaoa["first_optimal_shot"] = None
            self._log(run, "Comparison ready "
                           f"(reference: {ref['type'] if ref else 'none'})")
            self._stage(run, 6, "done")
            with self._lock:
                run["results"] = results
                if run["status"] != "error":
                    run["status"] = "done"
                run["finished_ts"] = time.time()
        except Exception as exc:                           # noqa: BLE001
            self._log(run, f"Run failed: {exc}")
            with self._lock:
                for s in run["stages"]:
                    if s["state"] == "running":
                        s["state"] = "error"
                        s["detail"] = str(exc)[:120]
                run["status"] = "error"
                run["results"] = results
                run["finished_ts"] = time.time()
