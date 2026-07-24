"""Quantum stage: Guppy kernel codegen + Nexus (Helios emulator) execution.

Isolated from the run manager so tests can stub `run_quantum` entirely.
Uploads the RAW Guppy HUGR (no local TKET lowering: Nexus rejects lowered
programs) and uses HeliosConfig with an explicit HeliosEmulatorConfig.
"""
from __future__ import annotations

import importlib.util
import math
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

DEVICE = "Helios-1E-lite"
PROJECT = "kraken-quantathon"
POLL_TIMEOUT_S = 1800.0


def check_session() -> str | None:
    try:
        import qnexus as qnx
        me = qnx.users.get_self()
        return getattr(me, "display_name", "ok")
    except Exception:                                      # noqa: BLE001
        return None


def _ang(theta: float) -> str:
    return f"({theta / math.pi:.12f} * pi)"


def generate_kernel_source(h: list[float], J: dict, gamma: float,
                           beta: float, n: int) -> str:
    lines = [
        "from guppylang import guppy",
        "from guppylang.std.builtins import result",
        "from guppylang.std.quantum import cx, h, measure, pi, qubit, rx, rz",
        "", "",
        "@guppy",
        "def main() -> None:",
    ]
    for i in range(n):
        lines.append(f"    q{i} = qubit()")
    for i in range(n):
        lines.append(f"    h(q{i})")
    for i, hi in enumerate(h):
        if hi:
            lines.append(f"    rz(q{i}, {_ang(2 * gamma * hi)})")
    for (i, j), Jij in sorted(J.items()):
        if Jij:
            lines.append(f"    cx(q{i}, q{j})")
            lines.append(f"    rz(q{j}, {_ang(2 * gamma * Jij)})")
            lines.append(f"    cx(q{i}, q{j})")
    for i in range(n):
        lines.append(f"    rx(q{i}, {_ang(2 * beta)})")
    for i in range(n):
        lines.append(f'    result("x{i}", measure(q{i}))')
    return "\n".join(lines) + "\n"


def load_kernel(source: str):
    """Write the kernel to a real file (guppy needs source on disk), import it."""
    tmp = Path(tempfile.mkdtemp()) / f"qaoa_kernel_{uuid.uuid4().hex[:8]}.py"
    tmp.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(tmp.stem, tmp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def run_quantum(ising: dict, gamma: float, beta: float, n: int, shots: int,
                log: Callable[[str], None],
                should_cancel: Callable[[], bool] | None = None,
                ) -> tuple[list[list[int]], str, dict]:
    import qnexus as qnx
    from qnexus.models import HeliosConfig
    from quantinuum_schemas.models.backend_config import HeliosEmulatorConfig

    source = generate_kernel_source(ising["h"], ising["J"], gamma, beta, n)
    kernel = load_kernel(source)
    package = kernel.compile()
    log("Kernel compiled to raw HUGR")

    qnx.context.set_active_project(qnx.projects.get_or_create(name=PROJECT))
    ref = qnx.hugr.upload(hugr_package=package,
                          name=f"app-qaoa-{uuid.uuid4().hex[:6]}")
    log(f"HUGR uploaded ({ref.id})")

    job = qnx.start_execute_job(
        programs=[ref], n_shots=[shots],
        backend_config=HeliosConfig(
            system_name=DEVICE,
            emulator_config=HeliosEmulatorConfig(n_qubits=n)),
        name=f"app-qaoa-run-{uuid.uuid4().hex[:6]}")
    log(f"Job submitted to {DEVICE} ({shots} shots)")

    deadline = time.monotonic() + POLL_TIMEOUT_S
    submitted_at = time.monotonic()
    first_running_at: float | None = None
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while True:
        if should_cancel is not None and should_cancel():
            try:
                qnx.jobs.cancel(job)
                log("Nexus job cancelled remotely")
            except Exception as exc:                       # noqa: BLE001
                log(f"Nexus job cancel request failed: {exc}")
            raise RuntimeError("cancelled by user")
        st = qnx.jobs.status(job)
        name = getattr(getattr(st, "status", st), "name", str(st))
        log(f"Job status: {name}")
        if name == "RUNNING" and first_running_at is None:
            first_running_at = time.monotonic()
        if name == "COMPLETED":
            break
        if name in ("ERROR", "CANCELLED", "TERMINATED", "DEPLETED"):
            raise RuntimeError(f"Nexus job ended in {name}")
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Nexus job timed out after {POLL_TIMEOUT_S:.0f}s (last status: {name})")
        time.sleep(5)

    completed_at = time.monotonic()
    # Phase split at 5 s poll granularity; if RUNNING was never observed the
    # whole wait counts as queued.
    run_start = first_running_at if first_running_at is not None else completed_at
    timing = {"queued_s": round(run_start - submitted_at, 1),
              "running_s": round(completed_at - run_start, 1)}

    result = qnx.jobs.results(job)[0].download_result()
    bits_out: list[list[int]] = []
    for shot in result.results:
        entries = dict(shot.entries)
        bits_out.append([int(entries[f"x{i}"]) for i in range(n)])
    log(f"Downloaded {len(bits_out)} shots")
    return bits_out, str(job.id), timing
