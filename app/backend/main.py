"""FastAPI app for the grid simulator frontend."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import grid_service
import nexus_stage
import simulation

app = FastAPI(title="Grid simulator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

manager = simulation.RunManager()


class SimulateRequest(BaseModel):
    nodes: list[str]
    shots: int = 500


@app.get("/api/grid")
def get_grid() -> dict:
    return grid_service.grid_payload()


@app.get("/api/subgrid")
def get_subgrid(nodes: str) -> dict:
    return grid_service.subgrid_info([n for n in nodes.split(",") if n])


@app.post("/api/simulate")
def simulate(req: SimulateRequest) -> dict:
    info = grid_service.subgrid_info(req.nodes)
    if not info["valid"]:
        raise HTTPException(status_code=400, detail=info["reason"])
    if nexus_stage.check_session() is None:
        raise HTTPException(
            status_code=409,
            detail="No Nexus session. Login with: "
                   ".venv/bin/python -c 'import qnexus; qnexus.login()'")
    run_id = manager.launch(req.nodes, req.shots,
                            quantum_runner=nexus_stage.run_quantum)
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run
