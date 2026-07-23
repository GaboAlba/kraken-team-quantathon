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
    return shots_out, "fake-job-id"


def _wait(mgr, run_id, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        run = mgr.get(run_id)
        if run["status"] in ("done", "error"):
            return run
        time.sleep(0.2)
    raise AssertionError("run did not finish")


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
