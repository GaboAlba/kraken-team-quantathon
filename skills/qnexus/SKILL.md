---
name: qnexus
description: Submit circuits to Quantinuum Nexus (cloud platform) with the qnexus package. Use when compiling or executing pytket circuits on Quantinuum emulators/hardware, managing Nexus projects and jobs, or retrieving cloud execution results in this Quantathon project.
---

# qnexus (Quantinuum Nexus)

Cloud platform client for Quantinuum. This is the path to run this project's
QAOA/Max-Cut circuits (built per the `pytket` skill) on Quantinuum **emulators
or H-series hardware** instead of a local simulator. Package: `qnexus`,
imported as `qnx`.

## Authentication and project context

```python
import qnexus as qnx

qnx.login()                      # opens a browser to log in (token lasts ~30 days)
# or: qnx.login_with_credentials()   # prompts in the terminal

project = qnx.projects.get_or_create(name="kraken-quantathon")
qnx.context.set_active_project(project)   # required before uploading/submitting
```

## Upload a circuit

Nexus works with **references** to remote objects, not local objects:

```python
from pytket import Circuit

circ = Circuit(12)               # QAOA circuit from grid_cr.json (see pytket skill)
ref = qnx.circuits.upload(circuit=circ, name="qaoa-maxcut-p1")
circ_again = ref.download_circuit()   # refs download, they don't hold the data
```

## Compile and execute

```python
config = qnx.QuantinuumConfig(device_name="H2-Emulator")   # emulator, no queue cost

compile_job = qnx.start_compile_job(
    programs=[ref], backend_config=config,
    optimisation_level=2, name="compile-qaoa",
)
qnx.jobs.wait_for(compile_job)
compiled_ref = qnx.jobs.results(compile_job)[0].get_output()

execute_job = qnx.start_execute_job(
    programs=[compiled_ref], n_shots=[1000],   # one shots entry PER program
    backend_config=config, name="run-qaoa",
)
qnx.jobs.wait_for(execute_job)
backend_result = qnx.jobs.results(execute_job)[0].download_result()
counts = backend_result.get_counts()           # same BackendResult as pytket
```

- Job control: `qnx.jobs.status(job)`, `qnx.jobs.cancel(job)`, `qnx.jobs.delete(job)`.
- Persist refs across sessions: `qnx.filesystem.save(path=..., ref=job, mkdir=True)`
  and `qnx.filesystem.load(path=...)`.

## When to use in this project

Use the local `AerBackend` (pytket skill) or Selene (`selene` skill) for
iteration; switch to qnexus only to validate results on a Quantinuum emulator
(`H2-Emulator`) or to make a real hardware submission (`H1-1`, `H2-1`).

## Gotchas

- Everything is asynchronous: `start_*_job` returns immediately; always
  `qnx.jobs.wait_for(...)` before reading results.
- `n_shots` is a **list** parallel to `programs`, not an int.
- Requires network + login; never put Nexus calls in the test suite or the
  reproducible pipeline — keep them in separate experiment scripts.
- Hardware queues are shared and slow; batch circuits into one job where possible.
