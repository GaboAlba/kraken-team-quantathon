import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grid_service
import simulation


def fake_quantum(ising, gamma, beta, n, shots, log):
    # Return shots that include the optimum pattern plus noise-like variety.
    base = [0] * n
    alt = [1] * n
    shots_out = [base if i % 3 else alt for i in range(shots)]
    log("fake quantum runner used")
    return shots_out, "fake-job-id", {"queued_s": 1.5, "running_s": 3.2}


def _wait(mgr, run_id, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        run = mgr.get(run_id)
        if run["status"] in ("done", "error"):
            return run
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def _grow_selection(target: int) -> list[str]:
    selection = list(grid_service.INITIAL_NODES)
    while len(selection) < target:
        info = grid_service.subgrid_info(selection)
        selection.append(info["adjacent"][0])
    return selection


def test_full_run_with_stubbed_quantum():
    mgr = simulation.RunManager()
    run_id = mgr.launch(list(grid_service.INITIAL_NODES), shots=30,
                        quantum_runner=fake_quantum)
    run = _wait(mgr, run_id)
    assert run["status"] == "done", run
    assert [s["state"] for s in run["stages"]] == ["done"] * 7
    res = run["results"]
    assert res["optimum"]["energy"] <= res["methods"]["greedy"]["best_energy"] + 1e-9
    assert res["methods"]["brute_force"]["found_optimum"] is True
    assert len(res["methods"]["qaoa"]["energies"]) == 30
    assert res["methods"]["qaoa"]["job_id"] == "fake-job-id"
    assert res["methods"]["qaoa"]["queued_s"] == 1.5
    assert res["methods"]["qaoa"]["running_s"] == 3.2
    assert any("fake quantum" in line for line in run["log"])


def test_stage_names_and_progress():
    assert simulation.STAGES == [
        "build_qubo", "brute_force", "greedy", "goemans_williamson",
        "qaoa_angles", "nexus_job", "analysis",
    ]


def test_failed_quantum_still_reports_classical():
    def boom(ising, gamma, beta, n, shots, log):
        raise RuntimeError("nexus down")
    mgr = simulation.RunManager()
    run_id = mgr.launch(list(grid_service.INITIAL_NODES), shots=10,
                        quantum_runner=boom)
    run = _wait(mgr, run_id)
    assert run["status"] == "error"
    states = {s["name"]: s["state"] for s in run["stages"]}
    assert states["nexus_job"] == "error"
    assert states["goemans_williamson"] == "done"
    assert run["results"]["methods"]["qaoa"] is None
    assert run["results"]["methods"]["greedy"]["best_energy"] is not None


def test_classical_stage_failure_marks_stage_error(monkeypatch):
    monkeypatch.setattr(simulation, "optimize_angles",
                        lambda energies: (_ for _ in ()).throw(RuntimeError("boom")))
    mgr = simulation.RunManager()
    run_id = mgr.launch(list(grid_service.INITIAL_NODES), shots=5,
                        quantum_runner=fake_quantum)
    run = _wait(mgr, run_id)
    assert run["status"] == "error"
    states = {s["name"]: s["state"] for s in run["stages"]}
    assert states["qaoa_angles"] == "error"
    assert states["goemans_williamson"] == "done"


def test_heuristic_tier_runs_brute_force_with_heuristic_angles():
    # 24 nodes: brute force still RUNS (vectorized, 2^24 states) and stays the
    # exact reference; only the angle optimization degrades to heuristic.
    mgr = simulation.RunManager()
    run_id = mgr.launch(_grow_selection(24), shots=10,
                        quantum_runner=fake_quantum)
    run = _wait(mgr, run_id, timeout=600.0)
    assert run["status"] == "done", run
    states = {s["name"]: s["state"] for s in run["stages"]}
    assert states["brute_force"] == "done"
    assert states["nexus_job"] == "done"
    res = run["results"]
    assert res["methods"]["brute_force"] is not None
    assert res["reference"]["type"] == "exact"
    assert res["optimum"] is not None
    angles_detail = next(s["detail"] for s in run["stages"]
                         if s["name"] == "qaoa_angles")
    assert "heuristic" in angles_detail
    # exact reference -> optimality is verifiable again
    assert res["methods"]["brute_force"]["found_optimum"] is True
    # payload ships a bounded sample, not all 2^24 energies
    assert len(res["methods"]["brute_force"]["energies"]) <= 20000
    assert res["methods"]["brute_force"]["n_states"] == 2 ** 24


def test_impossible_brute_force_is_skipped_but_quantum_still_attempted():
    # 45 nodes: 2^45 states is out of computational reach -> brute force
    # skipped with an estimate; the quantum job is still ATTEMPTED.
    mgr = simulation.RunManager()
    run_id = mgr.launch(_grow_selection(45), shots=10,
                        quantum_runner=fake_quantum)
    run = _wait(mgr, run_id, timeout=600.0)
    assert run["status"] == "done", run
    states = {s["name"]: s["state"] for s in run["stages"]}
    assert states["brute_force"] == "skipped"
    assert states["nexus_job"] == "done"
    res = run["results"]
    assert res["methods"]["brute_force"] is None
    assert res["reference"]["type"] == "sdp_bound"
    assert res["methods"]["qaoa"] is not None
    assert res["methods"]["qaoa"]["p_optimal"] is None


def test_run_and_stages_report_elapsed_time():
    mgr = simulation.RunManager()
    run_id = mgr.launch(list(grid_service.INITIAL_NODES), shots=10,
                        quantum_runner=fake_quantum)
    run = _wait(mgr, run_id)
    assert run["status"] == "done"
    assert run["elapsed_s"] > 0
    for s in run["stages"]:
        if s["state"] in ("done", "error"):
            assert s["elapsed_s"] is not None and s["elapsed_s"] >= 0
        if s["state"] == "skipped":
            assert s["elapsed_s"] is None or s["elapsed_s"] >= 0
