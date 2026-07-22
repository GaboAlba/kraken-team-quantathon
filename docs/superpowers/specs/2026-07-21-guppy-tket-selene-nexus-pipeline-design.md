# Guppy → TKET → Selene → Nexus pipeline — design

**Date:** 2026-07-21
**Status:** approved (design), pending implementation

## Goal

A staged, kernel-agnostic pipeline that takes any Guppy kernel in the repo
through the real Quantinuum flow: compile/optimize with TKET, test locally on
the Selene emulator, and execute on Quantinuum cloud (Nexus) — with hardware
submission explicitly gated. QAOA/Max-Cut is out of scope for now; the QAOA
module will be the first consumer of this pipeline once it exists.

## Verified technical basis

Checked against the installed packages (see `requirements.txt`):

- `tket` 0.13 (the HUGR-native TKET) provides `tket.passes.NormalizeGuppy` and
  `tket.passes.QSystemPass` to lower Guppy/HUGR to the Quantinuum QSystem gate set.
- `qnexus` 0.46 accepts HUGR programs directly (`qnx.hugr`); no pytket
  `Circuit` conversion is needed.
- `selene-sim` 0.2.x runs compiled HUGR via `build(...)` + `run_shots(...)`
  (verified end-to-end with a Bell kernel; Quest yields only `00`/`11`).

## Flow and artifacts

```
kernel Guppy ──> build ──> emulate ──> submit ──> results
(module:func)   (TKET)     (Selene)    (qnexus)   (counts)
```

Each kernel gets its own artifact directory `data/results/<kernel-name>/` so
teammates don't collide. Every artifact carries metadata (source kernel, config,
date) following the repo's provenance conventions.

| Stage | Input | Output |
|---|---|---|
| `build` | `module.py:function` reference to a `@guppy` kernel | `program.json` — serialized HUGR after TKET passes + metadata |
| `emulate` | `program.json` | `emulation.json` — collated counts, simulator, seed, shots |
| `submit` | `program.json` (warns if `emulation.json` is missing, but proceeds) | Nexus job ref persisted locally (`qnx.filesystem`) |
| `results` | saved job ref | `hardware.json` — downloaded counts + job/device metadata |

- **Example kernel:** `src/kernels/bell.py` (Bell state) so the pipeline is
  testable end-to-end without any QAOA code.
- **`build`:** import the kernel, `compile()` to HUGR, apply
  `NormalizeGuppy` → `QSystemPass`, serialize.
- **`emulate`:** Selene with `--simulator quest|stim|coinflip`, `--shots N`,
  fixed default seed (repo determinism convention).
- **`submit`:** upload HUGR via `qnx.hugr`, start execute job. Default device
  `H2-Emulator`. Real hardware requires `--device H2-1 --yes`; without `--yes`
  it prints a summary (device, shots, kernel) and aborts.
- **`results`:** non-blocking; if the job is still queued it reports status and
  exits (resumable — the ref is on disk).

## Module structure

```
src/
  kernels/
    __init__.py
    bell.py          # example @guppy kernel
  pipeline.py        # stage functions + argparse CLI (~200 lines)
data/results/<kernel>/   # program.json, emulation.json, hardware.json, job ref
```

`pipeline.py` exposes one plain function per stage (`build()`, `emulate()`,
`submit()`, `fetch_results()`); the CLI (`python -m src.pipeline <stage> ...`)
is a thin layer over them so other modules can import the functions directly.
No global state: stages communicate only through on-disk artifacts.

## Error handling

- Each stage validates its input artifact exists and fails with the exact
  command to run first.
- `submit` verifies an active Nexus login before uploading and points to
  `qnx.login()` if missing.
- Hardware gate: any device name that is not an emulator/syntax checker
  requires `--yes`.
- Kernel references only accept `module.py:function` in a real file (avoids
  Selene's `OSError: could not get source code` on REPL-defined kernels).

## Testing

`tests/test_pipeline.py`, synthetic and offline (repo convention):

- `build` + `emulate` on the Bell kernel with Quest and a fixed seed: counts
  contain only `00`/`11` and are identical across runs.
- `module:function` parsing errors and missing-artifact errors.
- Hardware gate logic (function level, no CLI process spawning).
- No Nexus calls in tests; `submit`/`results` validated once manually against
  `H2-Emulator`.

## Documentation updates

- `AGENTS.md`: add pipeline commands to "Environment & commands".
- `skills/selene/SKILL.md` and `skills/qnexus/SKILL.md`: short pointer to
  `src/pipeline.py` as the project's canonical flow.

## Out of scope (for now)

- QAOA kernel builder, angle optimization (scipy loop), cut evaluation and
  classical-optimum comparison — future `src/qaoa.py` consuming this pipeline.
- Noise models / error models in emulation (easy to add later via a flag).
- Batch submission of multiple programs per job.
