"""Staged simulation runs: classical solvers + injected quantum runner."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Callable

import networkx as nx
import numpy as np

from grid_service import REPO, national  # noqa: F401  (REPO used by callers)
import sys

sys.path.insert(0, str(REPO))
from src import classical_baselines as cb
from src import graph as graph_mod
from src import qubo as qubo_mod

STAGES = ["build_qubo", "brute_force", "greedy", "goemans_williamson",
          "qaoa_angles", "nexus_job", "analysis"]
FIELD = "__field__"

QuantumRunner = Callable[..., tuple[list[list[int]], str]]


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


class RunManager:
    def __init__(self) -> None:
        self._runs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def launch(self, nodes: list[str], shots: int,
               quantum_runner: QuantumRunner) -> str:
        run_id = uuid.uuid4().hex[:12]
        run = {
            "id": run_id, "status": "running",
            "stages": [{"name": s, "state": "pending", "detail": ""}
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
            return None if run is None else {**run,
                                             "stages": [dict(s) for s in run["stages"]],
                                             "log": list(run["log"])}

    # ------------------------------------------------------------------
    def _log(self, run: dict, msg: str) -> None:
        with self._lock:
            run["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _stage(self, run: dict, idx: int, state: str, detail: str = "") -> None:
        with self._lock:
            run["stages"][idx]["state"] = state
            run["stages"][idx]["detail"] = detail
            done = sum(1 for s in run["stages"] if s["state"] == "done")
            run["progress_pct"] = int(100 * done / len(STAGES))

    def _execute(self, run: dict, nodes: list[str], shots: int,
                 quantum_runner: QuantumRunner) -> None:
        results: dict = {"optimum": None, "methods": {
            "brute_force": None, "greedy": None, "goemans_williamson": None,
            "qaoa": None}}
        try:
            # 1 -- QUBO
            self._stage(run, 0, "running")
            G = national()
            sub = graph_mod.extract_subregion(G, nodes=nodes,
                                              max_nodes=len(nodes))
            q = qubo_mod.build_qubo(sub)
            ising = qubo_mod.qubo_to_ising(q)
            n = len(q.variables)
            self._log(run, f"QUBO built: {n} variables, "
                           f"{len(q.quadratic)} couplings")
            self._stage(run, 0, "done", f"{n} vars")

            # 2 -- brute force
            self._stage(run, 1, "running")
            t0 = time.perf_counter()
            states = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
            all_E = np.array([q.energy(list(s)) for s in states])
            E_opt = float(all_E.min())
            x_opt = states[int(np.argmin(all_E))].tolist()
            bf_ms = (time.perf_counter() - t0) * 1000
            part_opt = {
                "A": sorted(v for v, b in zip(q.variables, x_opt) if b == 0),
                "B": sorted(v for v, b in zip(q.variables, x_opt) if b == 1)}
            results["optimum"] = {"energy": E_opt, "partition": part_opt}
            results["methods"]["brute_force"] = {
                "best_energy": E_opt, "gap_pct": 0.0, "time_ms": bf_ms,
                "found_optimum": True, "energies": all_E.tolist()}
            self._log(run, f"Brute force: optimum {E_opt:.4f} "
                           f"({2 ** n} states, {bf_ms:.1f} ms)")
            self._stage(run, 1, "done", f"E={E_opt:.3f}")

            H = build_augmented(q, ising)

            def score(partition: dict) -> float:
                return float(q.energy(x_of_partition(q, partition)))

            # 3 -- greedy
            self._stage(run, 2, "running")
            t0 = time.perf_counter()
            g_energies = [score(cb.greedy_maxcut(H, seed=s)[0])
                          for s in range(20)]
            g_ms = (time.perf_counter() - t0) * 1000
            g_best = min(g_energies)
            results["methods"]["greedy"] = {
                "best_energy": g_best,
                "gap_pct": 100 * (g_best - E_opt) / abs(E_opt),
                "time_ms": g_ms,
                "found_optimum": bool(g_best <= E_opt + 1e-9),
                "energies": g_energies}
            self._log(run, f"Greedy: best {g_best:.4f} over 20 restarts")
            self._stage(run, 2, "done", f"E={g_best:.3f}")

            # 4 -- GW
            self._stage(run, 3, "running")
            t0 = time.perf_counter()
            gw_best_e = score(cb.goemans_williamson(
                H, n_rounding_trials=50, seed=42)[0])
            gw_trials = [score(cb.goemans_williamson(
                H, n_rounding_trials=1, seed=s)[0]) for s in range(10)]
            gw_ms = (time.perf_counter() - t0) * 1000
            results["methods"]["goemans_williamson"] = {
                "best_energy": gw_best_e,
                "gap_pct": 100 * (gw_best_e - E_opt) / abs(E_opt),
                "time_ms": gw_ms,
                "found_optimum": bool(gw_best_e <= E_opt + 1e-9),
                "energies": gw_trials}
            self._log(run, f"Goemans-Williamson: best {gw_best_e:.4f}")
            self._stage(run, 3, "done", f"E={gw_best_e:.3f}")

            # 5 -- angles
            self._stage(run, 4, "running")
            gamma, beta = optimize_angles(all_E)
            self._log(run, f"QAOA angles: gamma={gamma:.4f}, beta={beta:.4f}")
            self._stage(run, 4, "done", f"g={gamma:.3f} b={beta:.3f}")

            # 6 -- quantum
            self._stage(run, 5, "running", "submitting")
            try:
                bits, job_id = quantum_runner(
                    ising=ising, gamma=gamma, beta=beta, n=n, shots=shots,
                    log=lambda m: self._log(run, m))
                qaoa_E = [float(q.energy(x)) for x in bits]
                q_best = min(qaoa_E)
                hits = sum(1 for e in qaoa_E if e <= E_opt + 1e-9)
                first = next((i + 1 for i, e in enumerate(qaoa_E)
                              if e <= E_opt + 1e-9), None)
                results["methods"]["qaoa"] = {
                    "best_energy": q_best,
                    "gap_pct": 100 * (q_best - E_opt) / abs(E_opt),
                    "mean_energy": float(np.mean(qaoa_E)),
                    "found_optimum": bool(q_best <= E_opt + 1e-9),
                    "p_optimal": hits / len(qaoa_E),
                    "first_optimal_shot": first,
                    "shots": len(qaoa_E), "energies": qaoa_E,
                    "gamma": gamma, "beta": beta, "job_id": job_id}
                self._log(run, f"Quantum job {job_id}: best {q_best:.4f}, "
                               f"P(opt)={hits / len(qaoa_E):.3f}")
                self._stage(run, 5, "done", f"E={q_best:.3f}")
            except Exception as exc:                       # noqa: BLE001
                self._log(run, f"Quantum stage failed: {exc}")
                self._stage(run, 5, "error", str(exc)[:120])
                with self._lock:
                    run["status"] = "error"

            # 7 -- analysis (always runs; partial if quantum failed)
            self._stage(run, 6, "running")
            self._log(run, "Comparison ready")
            self._stage(run, 6, "done")
            with self._lock:
                run["results"] = results
                if run["status"] != "error":
                    run["status"] = "done"
        except Exception as exc:                           # noqa: BLE001
            self._log(run, f"Run failed: {exc}")
            with self._lock:
                run["status"] = "error"
                run["results"] = results
