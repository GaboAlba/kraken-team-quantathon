import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import grid_service
import main


client = TestClient(main.app)


def test_grid_endpoint():
    r = client.get("/api/grid")
    assert r.status_code == 200
    doc = r.json()
    assert len(doc["nodes"]) >= 60 and len(doc["plants"]) >= 30


def test_subgrid_endpoint_valid_and_invalid():
    nodes = ",".join(grid_service.INITIAL_NODES)
    r = client.get(f"/api/subgrid?nodes={nodes}")
    assert r.status_code == 200 and r.json()["valid"] is True
    r = client.get("/api/subgrid?nodes=arenal,canas")
    assert r.status_code == 200 and r.json()["valid"] is False


def test_simulate_requires_session(monkeypatch):
    monkeypatch.setattr(main.nexus_stage, "check_session", lambda: None)
    r = client.post("/api/simulate",
                    json={"nodes": list(grid_service.INITIAL_NODES)})
    assert r.status_code == 409
    assert "login" in r.json()["detail"].lower()


def test_simulate_and_poll(monkeypatch):
    monkeypatch.setattr(main.nexus_stage, "check_session", lambda: "tester")

    def fake_runner(ising, gamma, beta, n, shots, log):
        return [[0] * n for _ in range(shots)], "job-x", {"queued_s": 0.1, "running_s": 0.2}
    monkeypatch.setattr(main.nexus_stage, "run_quantum", fake_runner)

    r = client.post("/api/simulate",
                    json={"nodes": list(grid_service.INITIAL_NODES),
                          "shots": 10})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    for _ in range(600):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert run["status"] == "done"
    assert run["results"]["methods"]["qaoa"]["job_id"] == "job-x"


def test_unknown_run_404():
    assert client.get("/api/runs/nope").status_code == 404
