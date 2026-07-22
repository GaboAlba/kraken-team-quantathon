# Guppy → TKET → Selene → Nexus Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A staged, kernel-agnostic pipeline (`python -m src.pipeline`) that takes any Guppy kernel through TKET compilation, Selene emulation, and gated Nexus execution.

**Architecture:** One module `src/pipeline.py` with a plain function per stage (`build`, `emulate`, `submit`, `fetch_results`) communicating only through JSON artifacts in `data/results/<name>/`; a thin argparse CLI on top. An example Bell kernel in `src/kernels/bell.py` makes the pipeline testable end-to-end without QAOA.

**Tech Stack:** guppylang (kernels → HUGR), `tket` 0.13 (`NormalizeGuppy`, `QSystemPass` HUGR passes), `selene-sim` (local emulation, Quest/Stim/Coinflip), `qnexus` 0.46 (`qnx.hugr.upload` + execute jobs), pytest.

**Spec:** `docs/superpowers/specs/2026-07-21-guppy-tket-selene-nexus-pipeline-design.md`

## Global Constraints

- English only: docstrings, comments, identifiers, JSON keys (AGENTS.md).
- Determinism: default seed 42 everywhere; runs must be repeatable.
- Tests are offline: no network, no Nexus calls, no live services.
- Heavy imports (guppylang, selene_sim, tket, qnexus) go **inside** stage functions so `--help` and gate tests never touch them.
- Stage functions take `results_dir: Path | None = None` and resolve `results_dir or RESULTS_DIR` at call time (same testability pattern as `raw_dir` in `src/ice_data.py`).
- Modules import as `from src import pipeline`; tests insert repo root in `sys.path` (repo convention).
- Commit messages: plain, no AI/tool attribution.
- All commands below assume the activated venv (`source .venv/bin/activate`) at the repo root.

**Verified API facts (do not re-derive):** `kernel.compile()` returns `hugr.package.Package`; TKET passes run in-place per module (`NormalizeGuppy().then(QSystemPass()).run(pkg.modules[0])`); serialization round-trips via `package.to_str()` / `Package.from_str(s)` (HUGR envelope — `to_json` is deprecated); Selene: `build(package).run_shots(Quest(random_seed=s), n_qubits=n, n_shots=k)` wrapped in `QsysResult` gives `collated_counts()`; all three simulator classes accept `random_seed`; `qnx.users.get_self()` raises `qnexus.exceptions.AuthenticationError` when logged out.

---

### Task 1: Kernel loader and example Bell kernel

**Files:**
- Create: `src/kernels/__init__.py` (empty)
- Create: `src/kernels/bell.py`
- Create: `src/pipeline.py` (module skeleton + `load_kernel`)
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `pipeline.load_kernel(ref: str) -> GuppyFunctionDefinition` — accepts `"path/to/file.py:function"`, raises `ValueError` (bad format), `FileNotFoundError` (missing file), `AttributeError` (missing function). `src/kernels/bell.py:main` — a `@guppy` kernel with 2 qubits reporting tags `c0`, `c1`. Module constants `ROOT`, `RESULTS_DIR`, `DEFAULT_SEED = 42`, `DEFAULT_SHOTS = 100`, `NEXUS_PROJECT = "kraken-quantathon"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
"""Tests for the Guppy -> TKET -> Selene -> Nexus pipeline.

All tests are offline and deterministic: Selene runs locally with fixed
seeds; no Nexus/network calls (repo convention).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import pipeline

BELL = "src/kernels/bell.py:main"


# --------------------------------------------------------------------------
# Kernel loading
# --------------------------------------------------------------------------

def test_load_kernel_requires_file_colon_function():
    with pytest.raises(ValueError):
        pipeline.load_kernel("src/kernels/bell.py")
    with pytest.raises(ValueError):
        pipeline.load_kernel("src/kernels/bell.py:")


def test_load_kernel_missing_file():
    with pytest.raises(FileNotFoundError):
        pipeline.load_kernel("src/kernels/nope.py:main")


def test_load_kernel_missing_function():
    with pytest.raises(AttributeError):
        pipeline.load_kernel("src/kernels/bell.py:nope")


def test_load_kernel_bell():
    kernel = pipeline.load_kernel(BELL)
    assert hasattr(kernel, "compile")   # a @guppy definition, not a plain function
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'src.pipeline'` (collection error is fine).

- [ ] **Step 3: Write the kernel and the loader**

Create `src/kernels/__init__.py` empty.

Create `src/kernels/bell.py`:

```python
"""Example Guppy kernel: Bell state.

Smallest end-to-end test case for the pipeline. Expected outcome: only the
correlated results ``c0=0 c1=0`` and ``c0=1 c1=1``, roughly 50/50.
"""

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit


@guppy
def main() -> None:
    q0 = qubit()
    q1 = qubit()
    h(q0)
    cx(q0, q1)
    result("c0", measure(q0))
    result("c1", measure(q1))
```

Create `src/pipeline.py`:

```python
"""Staged pipeline: Guppy kernel -> TKET -> Selene emulation -> Nexus execution.

Stages (each writes artifacts under ``data/results/<name>/``):

    build     kernel.py:func -> program.json    (HUGR after TKET passes)
    emulate   program.json   -> emulation.json  (Selene counts, local)
    submit    program.json   -> job_ref         (Nexus execute job, gated)
    results   job_ref        -> hardware.json   (downloaded counts)

Usage:
    python -m src.pipeline build src/kernels/bell.py:main --qubits 2
    python -m src.pipeline emulate bell --simulator quest --shots 100
    python -m src.pipeline submit bell                       # H2-Emulator default
    python -m src.pipeline submit bell --device H2-1 --yes   # real hardware
    python -m src.pipeline results bell

Heavy imports (guppylang, tket, selene_sim, qnexus) are deferred into the
stage functions so ``--help`` and the offline tests stay fast and networkless.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
NEXUS_PROJECT = "kraken-quantathon"
DEFAULT_SEED = 42
DEFAULT_SHOTS = 100


def load_kernel(ref: str):
    """Load a ``@guppy`` function from a ``"path/to/file.py:function"`` reference.

    Only file-based references are accepted: Guppy compilation needs the real
    source on disk (kernels defined in a REPL fail with ``OSError``).
    """
    path_str, sep, func_name = ref.partition(":")
    if not sep or not func_name:
        raise ValueError(f"Kernel reference must look like 'file.py:function', got '{ref}'")
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Kernel file not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, func_name):
        raise AttributeError(f"No function '{func_name}' in {path}")
    return getattr(module, func_name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `pytest` — expected: all pass (19 existing + 4 new).

```bash
git add src/kernels src/pipeline.py tests/test_pipeline.py
git commit -m "Add pipeline kernel loader and example Bell kernel"
```

---

### Task 2: build stage (Guppy → TKET → program.json)

**Files:**
- Modify: `src/pipeline.py` (append after `load_kernel`)
- Modify: `tests/test_pipeline.py` (append)

**Interfaces:**
- Consumes: `load_kernel` from Task 1.
- Produces: `pipeline.build(kernel_ref: str, n_qubits: int, name: str | None = None, results_dir: Path | None = None) -> Path` — writes `<results_dir>/<name>/program.json` with schema `{"metadata": {"kernel", "name", "n_qubits", "passes", "build_date"}, "hugr": "<envelope str>"}`; `name` defaults to the kernel file stem. Returns the artifact path.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
# --------------------------------------------------------------------------
# build stage
# --------------------------------------------------------------------------

def test_build_writes_program_artifact(tmp_path):
    out = pipeline.build(BELL, n_qubits=2, results_dir=tmp_path)
    assert out == tmp_path / "bell" / "program.json"
    import json
    doc = json.loads(out.read_text(encoding="utf-8"))
    meta = doc["metadata"]
    assert meta["kernel"] == BELL
    assert meta["n_qubits"] == 2
    assert meta["passes"] == ["NormalizeGuppy", "QSystemPass"]
    assert isinstance(doc["hugr"], str) and doc["hugr"]


def test_build_hugr_round_trips(tmp_path):
    out = pipeline.build(BELL, n_qubits=2, results_dir=tmp_path)
    import json
    from hugr.package import Package
    doc = json.loads(out.read_text(encoding="utf-8"))
    pkg = Package.from_str(doc["hugr"])
    assert len(pkg.modules) >= 1


def test_build_honors_custom_name(tmp_path):
    out = pipeline.build(BELL, n_qubits=2, name="my-experiment", results_dir=tmp_path)
    assert out.parent.name == "my-experiment"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v -k build`
Expected: FAIL with `AttributeError: module 'src.pipeline' has no attribute 'build'`.

- [ ] **Step 3: Implement build**

Append to `src/pipeline.py`:

```python
def build(
    kernel_ref: str,
    n_qubits: int,
    name: str | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Compile a Guppy kernel to HUGR, apply TKET passes, write ``program.json``.

    ``n_qubits`` must cover every qubit allocation in the kernel; Selene and
    Nexus both need it later. TKET's ``NormalizeGuppy`` cleans up the Guppy
    output and ``QSystemPass`` lowers it to the Quantinuum QSystem gate set.
    """
    from tket.passes import NormalizeGuppy, QSystemPass

    kernel = load_kernel(kernel_ref)
    package = kernel.compile()

    passes = NormalizeGuppy().then(QSystemPass())
    for module in package.modules:
        passes.run(module)

    name = name or Path(kernel_ref.partition(":")[0]).stem
    out_dir = (results_dir or RESULTS_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = {
        "metadata": {
            "kernel": kernel_ref,
            "name": name,
            "n_qubits": n_qubits,
            "passes": ["NormalizeGuppy", "QSystemPass"],
            "build_date": date.today().isoformat(),
        },
        "hugr": package.to_str(),
    }
    out = out_dir / "program.json"
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "Add pipeline build stage: Guppy to HUGR with TKET passes"
```

---

### Task 3: emulate stage (program.json → Selene → emulation.json)

**Files:**
- Modify: `src/pipeline.py` (append)
- Modify: `tests/test_pipeline.py` (append)

**Interfaces:**
- Consumes: `program.json` schema from Task 2.
- Produces: `pipeline.emulate(name: str, simulator: str = "quest", shots: int = DEFAULT_SHOTS, seed: int = DEFAULT_SEED, results_dir: Path | None = None) -> Path` — writes `emulation.json` with schema `{"metadata": {"name", "simulator", "shots", "seed", "date"}, "counts": {"c0=0 c1=0": 21, ...}}`. Raises `FileNotFoundError` with the build command hint if `program.json` is missing. Helper `pipeline._counts_to_json(counts) -> dict[str, int]` (shared later by `fetch_results`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
# --------------------------------------------------------------------------
# emulate stage
# --------------------------------------------------------------------------

def test_emulate_requires_program_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="pipeline build"):
        pipeline.emulate("bell", results_dir=tmp_path)


def test_emulate_bell_gives_only_correlated_counts(tmp_path):
    pipeline.build(BELL, n_qubits=2, results_dir=tmp_path)
    out = pipeline.emulate("bell", shots=50, seed=7, results_dir=tmp_path)
    import json
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["metadata"]["simulator"] == "quest"
    assert set(doc["counts"]) <= {"c0=0 c1=0", "c0=1 c1=1"}
    assert sum(doc["counts"].values()) == 50


def test_emulate_is_deterministic_with_fixed_seed(tmp_path):
    pipeline.build(BELL, n_qubits=2, results_dir=tmp_path)
    import json
    a = json.loads(pipeline.emulate("bell", shots=30, seed=7,
                                    results_dir=tmp_path).read_text(encoding="utf-8"))
    b = json.loads(pipeline.emulate("bell", shots=30, seed=7,
                                    results_dir=tmp_path).read_text(encoding="utf-8"))
    assert a["counts"] == b["counts"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v -k emulate`
Expected: FAIL with `AttributeError: module 'src.pipeline' has no attribute 'emulate'`.

- [ ] **Step 3: Implement emulate**

Append to `src/pipeline.py`:

```python
def _load_program(name: str, results_dir: Path | None) -> dict:
    """Read ``program.json`` for ``name`` or fail with the command to run first."""
    path = (results_dir or RESULTS_DIR) / name / "program.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.pipeline build <kernel.py:func> "
            f"--qubits N --name {name}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _counts_to_json(counts) -> dict[str, int]:
    """Serialize QsysResult counts (keyed by ((tag, value), ...)) to JSON keys.

    ``{(('c0', '0'), ('c1', '1')): 5}`` becomes ``{"c0=0 c1=1": 5}``.
    """
    return {
        " ".join(f"{tag}={val}" for tag, val in outcome): n
        for outcome, n in sorted(counts.items())
    }


def emulate(
    name: str,
    simulator: str = "quest",
    shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
    results_dir: Path | None = None,
) -> Path:
    """Run the built program on Selene and write ``emulation.json``.

    Simulators: ``quest`` (statevector, exact), ``stim`` (stabilizer,
    Clifford-only), ``coinflip`` (no quantum state; control-flow checks).
    """
    from hugr.package import Package
    from hugr.qsystem.result import QsysResult
    from selene_sim import Coinflip, Quest, Stim
    from selene_sim import build as selene_build

    simulators = {"quest": Quest, "stim": Stim, "coinflip": Coinflip}
    if simulator not in simulators:
        raise ValueError(f"Unknown simulator '{simulator}'; pick from {sorted(simulators)}")

    doc = _load_program(name, results_dir)
    package = Package.from_str(doc["hugr"])

    runner = selene_build(package)
    qres = QsysResult(
        runner.run_shots(
            simulators[simulator](random_seed=seed),
            n_qubits=doc["metadata"]["n_qubits"],
            n_shots=shots,
        )
    )

    out_dir = (results_dir or RESULTS_DIR) / name
    out = out_dir / "emulation.json"
    out.write_text(
        json.dumps(
            {
                "metadata": {
                    "name": name,
                    "simulator": simulator,
                    "shots": shots,
                    "seed": seed,
                    "date": date.today().isoformat(),
                },
                "counts": _counts_to_json(qres.collated_counts()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 10 PASS (Selene tests take a few seconds each — it compiles to LLVM).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "Add pipeline emulate stage: Selene run with counts artifact"
```

---

### Task 4: submit and results stages (Nexus, hardware-gated)

**Files:**
- Modify: `src/pipeline.py` (append)
- Modify: `tests/test_pipeline.py` (append)

**Interfaces:**
- Consumes: `_load_program`, `_counts_to_json` from Task 3.
- Produces: `pipeline.needs_confirmation(device: str) -> bool` (True for real hardware); `pipeline.submit(name, device="H2-Emulator", shots=DEFAULT_SHOTS, yes=False, results_dir=None)` — returns the job ref, or `None` when gated; persists the ref at `<results_dir>/<name>/job_ref`; `pipeline.fetch_results(name, results_dir=None) -> Path | None` — writes `hardware.json` (schema `{"metadata": {"name", "status", "date"}, "counts": {...}}`) when the job is COMPLETED, otherwise prints the status and returns `None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py` (offline: the gate fires **before** any qnexus import):

```python
# --------------------------------------------------------------------------
# submit gate (offline — the gate must fire before any qnexus import)
# --------------------------------------------------------------------------

def test_needs_confirmation_only_for_real_hardware():
    assert pipeline.needs_confirmation("H2-1") is True
    assert pipeline.needs_confirmation("H1-1") is True
    assert pipeline.needs_confirmation("H2-Emulator") is False
    assert pipeline.needs_confirmation("H1-Emulator") is False
    assert pipeline.needs_confirmation("H2-1SC") is False   # syntax checker


def test_submit_missing_program_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="pipeline build"):
        pipeline.submit("bell", results_dir=tmp_path)


def test_submit_real_hardware_aborts_without_yes(tmp_path, capsys):
    (tmp_path / "bell").mkdir(parents=True)
    (tmp_path / "bell" / "program.json").write_text(
        '{"metadata": {"name": "bell", "n_qubits": 2}, "hugr": "stub"}',
        encoding="utf-8",
    )
    job = pipeline.submit("bell", device="H2-1", yes=False, results_dir=tmp_path)
    assert job is None
    assert "--yes" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v -k "confirmation or submit"`
Expected: FAIL with `AttributeError` (`needs_confirmation` / `submit` missing).

- [ ] **Step 3: Implement submit and fetch_results**

Append to `src/pipeline.py`:

```python
def needs_confirmation(device: str) -> bool:
    """True when the device is real hardware (spends HQC credits and queue time).

    Emulators end in ``-Emulator``; syntax checkers end in ``SC``. Anything
    else (``H1-1``, ``H2-1``, ...) is a physical machine.
    """
    return not device.endswith(("-Emulator", "SC"))


def submit(
    name: str,
    device: str = "H2-Emulator",
    shots: int = DEFAULT_SHOTS,
    yes: bool = False,
    results_dir: Path | None = None,
):
    """Upload the built HUGR to Nexus and start an execute job.

    Real hardware requires ``yes=True`` (CLI ``--yes``); without it the call
    prints a summary and does nothing. The job ref is saved next to the
    artifacts so ``results`` can pick it up in a later session.
    """
    doc = _load_program(name, results_dir)

    emu_path = (results_dir or RESULTS_DIR) / name / "emulation.json"
    if not emu_path.exists():
        print(f"Warning: no emulation.json for '{name}' — consider running "
              f"'python -m src.pipeline emulate {name}' before spending queue time.")

    if needs_confirmation(device) and not yes:
        print(f"'{device}' is REAL HARDWARE (kernel '{name}', {shots} shots).\n"
              f"Re-run with --yes to confirm the submission.")
        return None

    import qnexus as qnx
    from hugr.package import Package
    from qnexus.exceptions import AuthenticationError

    try:
        qnx.users.get_self()
    except AuthenticationError:
        raise SystemExit(
            "No active Nexus session. Run: python -c 'import qnexus; qnexus.login()'"
        )

    project = qnx.projects.get_or_create(name=NEXUS_PROJECT)
    qnx.context.set_active_project(project)

    package = Package.from_str(doc["hugr"])
    hugr_ref = qnx.hugr.upload(hugr_package=package, name=f"{name}-program")
    job = qnx.start_execute_job(
        programs=[hugr_ref],
        n_shots=[shots],
        backend_config=qnx.QuantinuumConfig(device_name=device),
        name=f"{name}-execute",
    )
    qnx.filesystem.save(ref=job, path=(results_dir or RESULTS_DIR) / name / "job_ref",
                        mkdir=True)
    print(f"Submitted '{name}' to {device} ({shots} shots). "
          f"Check with: python -m src.pipeline results {name}")
    return job


def fetch_results(name: str, results_dir: Path | None = None) -> Path | None:
    """Fetch the Nexus job result and write ``hardware.json`` (non-blocking).

    If the job hasn't finished, print its status and return ``None`` — the ref
    stays on disk, so just re-run later.
    """
    import qnexus as qnx

    ref_path = (results_dir or RESULTS_DIR) / name / "job_ref"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"{ref_path} not found. Run: python -m src.pipeline submit {name}"
        )
    job = qnx.filesystem.load(path=ref_path)

    status = qnx.jobs.status(job)
    status_name = getattr(getattr(status, "status", status), "name", str(status))
    if status_name != "COMPLETED":
        print(f"Job '{name}' status: {status_name} — try again later.")
        return None

    result = qnx.jobs.results(job)[0].download_result()
    if hasattr(result, "collated_counts"):          # QsysResult (HUGR programs)
        counts = _counts_to_json(result.collated_counts())
    elif hasattr(result, "get_counts"):             # pytket BackendResult
        counts = {"".join(map(str, k)): v for k, v in result.get_counts().items()}
    else:
        counts = {"raw": str(result)}

    out = (results_dir or RESULTS_DIR) / name / "hardware.json"
    out.write_text(
        json.dumps(
            {
                "metadata": {"name": name, "status": status_name,
                             "date": date.today().isoformat()},
                "counts": counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 13 PASS, no network access attempted.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "Add pipeline submit and results stages with hardware gate"
```

---

### Task 5: CLI, docs, and manual Nexus validation

**Files:**
- Modify: `src/pipeline.py` (append CLI at the end)
- Modify: `tests/test_pipeline.py` (append)
- Modify: `AGENTS.md` ("Environment & commands" list and "Architecture" section)
- Modify: `skills/selene/SKILL.md`, `skills/qnexus/SKILL.md` (one-line pointer each)

**Interfaces:**
- Consumes: all stage functions from Tasks 2–4.
- Produces: `pipeline.main(argv: list[str] | None = None) -> None` and `python -m src.pipeline <stage>` entry point.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_build_and_emulate(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    pipeline.main(["build", BELL, "--qubits", "2"])
    assert (tmp_path / "bell" / "program.json").exists()
    pipeline.main(["emulate", "bell", "--shots", "20", "--seed", "7"])
    assert (tmp_path / "bell" / "emulation.json").exists()


def test_cli_requires_stage():
    with pytest.raises(SystemExit):
        pipeline.main([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v -k cli`
Expected: FAIL with `AttributeError: module 'src.pipeline' has no attribute 'main'`.

- [ ] **Step 3: Implement the CLI**

Append to `src/pipeline.py`:

```python
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Guppy -> TKET -> Selene -> Nexus staged pipeline.",
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("build", help="compile a Guppy kernel with TKET passes")
    p.add_argument("kernel", help="kernel reference, e.g. src/kernels/bell.py:main")
    p.add_argument("--qubits", type=int, required=True,
                   help="qubits the kernel allocates (Selene/Nexus need it)")
    p.add_argument("--name", default=None, help="artifact folder name (default: file stem)")

    p = sub.add_parser("emulate", help="run the built program on Selene locally")
    p.add_argument("name")
    p.add_argument("--simulator", choices=["quest", "stim", "coinflip"], default="quest")
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p = sub.add_parser("submit", help="execute on Quantinuum via Nexus (gated)")
    p.add_argument("name")
    p.add_argument("--device", default="H2-Emulator",
                   help="H2-Emulator (default), H1-Emulator, or real hardware like H2-1")
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    p.add_argument("--yes", action="store_true", help="confirm real-hardware submission")

    p = sub.add_parser("results", help="fetch the Nexus job result (non-blocking)")
    p.add_argument("name")

    args = parser.parse_args(argv)
    if args.stage == "build":
        print(f"Wrote {build(args.kernel, n_qubits=args.qubits, name=args.name)}")
    elif args.stage == "emulate":
        print(f"Wrote {emulate(args.name, simulator=args.simulator, shots=args.shots, seed=args.seed)}")
    elif args.stage == "submit":
        submit(args.name, device=args.device, shots=args.shots, yes=args.yes)
    elif args.stage == "results":
        fetch_results(args.name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full suite**

Run: `pytest`
Expected: all pass (19 pre-existing + 15 pipeline tests).

- [ ] **Step 5: Update docs**

In `AGENTS.md`, add to the "Environment & commands" bullet list (after the figures line):

```markdown
- Pipeline (Guppy → TKET → Selene → Nexus): `python -m src.pipeline build
  src/kernels/bell.py:main --qubits 2`, then `emulate <name>`, `submit <name>`
  (real hardware needs `--device H2-1 --yes`), `results <name>`. Artifacts land
  in `data/results/<name>/`.
```

In `AGENTS.md` "Architecture", append after the `src/visualize.py` bullet:

```markdown
- `src/pipeline.py` — staged runner for Guppy kernels (`src/kernels/`):
  TKET passes → Selene emulation → gated Nexus execution, stages linked by
  JSON artifacts in `data/results/<name>/`.
```

In `skills/selene/SKILL.md`, add at the end of the "When to use in this project" section:

```markdown
The project's canonical flow is `python -m src.pipeline emulate <name>` (see
`src/pipeline.py`), which wraps this API.
```

In `skills/qnexus/SKILL.md`, add at the end of the "When to use in this project" section:

```markdown
The project's canonical flow is `python -m src.pipeline submit <name>` (see
`src/pipeline.py`), which wraps this API with the hardware gate.
```

- [ ] **Step 6: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py AGENTS.md skills/selene/SKILL.md skills/qnexus/SKILL.md
git commit -m "Add pipeline CLI and document the staged workflow"
```

- [ ] **Step 7: Manual Nexus validation (once, needs login — not in tests)**

```bash
python -m src.pipeline build src/kernels/bell.py:main --qubits 2
python -m src.pipeline emulate bell
python -m src.pipeline submit bell            # goes to H2-Emulator
python -m src.pipeline results bell           # repeat until COMPLETED
```

Expected: `hardware.json` appears in `data/results/bell/` with only `c0=0 c1=0` /
`c0=1 c1=1` outcomes. If `submit` exits with "No active Nexus session", run
`python -c 'import qnexus; qnexus.login()'` and retry. If `qnx.hugr.upload` or
the execute job rejects the program, capture the error and revisit Task 4 —
the emulator device name may need to change (check the team's Nexus console
for available devices).

---

## Self-review notes

- Spec coverage: flow/artifacts (Tasks 2–4), example kernel (Task 1), CLI (Task 5), error handling (each stage's missing-artifact hints, auth check, gate), testing (offline, deterministic), docs updates (Task 5). Out-of-scope items from the spec are not implemented anywhere. ✓
- Names used across tasks are consistent: `load_kernel`, `build`, `emulate`, `submit`, `fetch_results`, `needs_confirmation`, `_load_program`, `_counts_to_json`, `RESULTS_DIR`, `DEFAULT_SEED`, `DEFAULT_SHOTS`, `NEXUS_PROJECT`, `main`. ✓
- Counts of tests per step verified cumulatively: 4 → 7 → 10 → 13 → 15. ✓
