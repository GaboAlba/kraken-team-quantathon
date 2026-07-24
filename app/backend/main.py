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


@app.get("/api/config")
def get_config() -> dict:
    """Static algorithm configuration, read from the science modules."""
    import sys
    sys.path.insert(0, str(grid_service.REPO))
    from src import qubo as qubo_mod
    from src import weights as weights_mod

    return {
        "qaoa": {
            "layers": 1,
            "default_shots": 500,
            "angle_search": "statevector grid search (24x16, gamma 0.02-1.2, "
                            "beta 0.05-pi/2) up to n=22; heuristic "
                            "(gamma=0.5/w_max, beta=pi/8) beyond",
            "circuit": "per ZZ term: CX-Rz(2*gamma*J)-CX; per Z term: "
                       "Rz(2*gamma*h); mixer Rx(2*beta)",
        },
        "qubo": {
            "weight_scheme": getattr(qubo_mod, "QUBO_WEIGHT_SCHEME",
                                     weights_mod.DEFAULT_SCHEME),
            "objective": "minimize cut"
                         if not getattr(qubo_mod, "MAXIMIZE_CUT", False)
                         else "maximize cut",
            "generator_spread_factor": getattr(
                qubo_mod, "DEFAULT_GEN_PENALTY_FACTOR", None),
            "balance_factor": getattr(
                qubo_mod, "DEFAULT_BALANCE_PENALTY_FACTOR", None),
            "reference_voltage_kv": getattr(weights_mod, "_KV_REF", 230.0),
        },
        "classical": {
            "greedy_restarts": 20,
            "gw_rounding_trials": 50,
            "gw_seed": 42,
            "sdp_solver": "SCS",
        },
        "nexus": {
            "device": nexus_stage.DEVICE,
            "project": nexus_stage.PROJECT,
            "poll_timeout_s": nexus_stage.POLL_TIMEOUT_S,
            "emulator": "statevector with QSystemErrorModel (device-realistic noise)",
        },
        "scaling": {
            "exact_angles_max_n": grid_service.EXACT_ANGLES_MAX_N,
            "brute_force_max_n": grid_service.BRUTE_FORCE_MAX_N,
            "initial_nodes": grid_service.INITIAL_NODES,
        },
    }


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


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    if not manager.cancel(run_id):
        raise HTTPException(status_code=404, detail="run not found or not running")
    return {"status": "cancelling"}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run
