"""Run the validation notebook's QAOA (cost Hamiltonian) on Nexus, then
compare against the exact brute-force solution of the same QUBO.

Guppy kernel (from qaoa_kernel.py, same H_C as notebooks/validation.ipynb
section 10.3) -> raw HUGR upload -> execute job on a Helios emulator ->
tagged shots -> QUBO energies -> side-by-side comparison with exhaustive
enumeration (2^9 = 512 states).
"""
import time
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

from src import qubo as qmod
from qaoa_kernel import main as kernel

SHOTS = 500
DEVICE = "Helios-1E-lite"

cfg = json.loads((SCRATCH / "qaoa_angles.json").read_text())
q = qmod.build_qubo(qmod.load_graph())
n = len(q.variables)

# Guppy -> HUGR. Upload the RAW package: Nexus compiles server-side and its
# validator rejects locally lowered (QSystemPass) programs.
package = kernel.compile()
print("Kernel compiled to HUGR (raw; Nexus lowers it server-side)")

import qnexus as qnx
from qnexus.models import HeliosConfig
from quantinuum_schemas.models.backend_config import HeliosEmulatorConfig

project = qnx.projects.get_or_create(name="kraken-quantathon")
qnx.context.set_active_project(project)

hugr_ref = qnx.hugr.upload(hugr_package=package, name="qaoa-guanacaste-north-p1")
print("HUGR uploaded:", hugr_ref.id)

# Helios devices use HeliosConfig (QuantinuumConfig is for the H1/H2 series).
# Emulation requires an explicit emulator_config; the defaults give a
# statevector simulator with the device-realistic QSystemErrorModel.
job = qnx.start_execute_job(
    programs=[hugr_ref],
    n_shots=[SHOTS],
    backend_config=HeliosConfig(
        system_name=DEVICE,
        emulator_config=HeliosEmulatorConfig(n_qubits=n),
    ),
    name="qaoa-guanacaste-north-run",
)
qnx.filesystem.save(ref=job, path=SCRATCH / "refs" / "qaoa_job", mkdir=True)
print(f"Execute job submitted to {DEVICE} ({SHOTS} shots), waiting...")
qnx.jobs.wait_for(job)

result = qnx.jobs.results(job)[0].download_result()
print("Result type:", type(result).__name__)

# Tagged shots -> bits -> QUBO energies
shots = getattr(result, "results", None)
energies, hits = [], 0
best_shot_e, best_shot_x = float("inf"), None
for shot in shots:
    bits = dict(shot.entries)
    x = [int(bits[f"x{i}"]) for i in range(n)]
    e = q.energy(x)
    energies.append(e)
    if e < best_shot_e:
        best_shot_e, best_shot_x = e, x
    if e <= cfg["E_min"] + 1e-9:
        hits += 1

E_mean = float(np.mean(energies))
quality = (cfg["E_mean_random"] - E_mean) / (cfg["E_mean_random"] - cfg["E_min"])
p_opt = hits / len(energies)

# ---- Brute force: exact optimum by exhaustive enumeration (2^n states) ----
t0 = time.perf_counter()
states = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
all_E = np.array([q.energy(list(s)) for s in states])
bf_seconds = time.perf_counter() - t0
bf_best = int(np.argmin(all_E))
bf_E, bf_x = float(all_E[bf_best]), states[bf_best].tolist()


def partition(x):
    return {"A": sorted(v for v, b in zip(q.variables, x) if b == 0),
            "B": sorted(v for v, b in zip(q.variables, x) if b == 1)}


print(f"\n{'=' * 62}")
print(f"COMPARISON: QAOA on {DEVICE} vs classical brute force")
print(f"{'=' * 62}")
print(f"{'':24}{'QAOA (quantum)':>18}{'Brute force':>18}")
print(f"{'best energy found':24}{best_shot_e:>18.4f}{bf_E:>18.4f}")
print(f"{'mean energy':24}{E_mean:>18.4f}{float(all_E.mean()):>18.4f}")
print(f"{'evaluations':24}{len(energies):>18}{2 ** n:>18}")
print(f"{'wall time':24}{'(queue + run)':>18}{bf_seconds:>17.3f}s")
print(f"{'guarantee':24}{'probabilistic':>18}{'exact optimum':>18}")

found_opt = abs(best_shot_e - bf_E) <= 1e-9
print(f"\nQAOA best shot matches the exact optimum: {'YES' if found_opt else 'NO'}")
if not found_opt:
    gap = (best_shot_e - bf_E) / abs(bf_E)
    print(f"  relative gap: {100 * gap:.2f}%")
print(f"QAOA quality (0=random, 1=optimal): {quality:.3f}   "
      f"P(optimal shot) = {p_opt:.3f}  (random baseline: {1 / 2 ** n:.4f})")
print(f"\nOptimal fault-zone partition (brute force):")
print(f"  {partition(bf_x)}")
if found_opt:
    print("QAOA best-shot partition: identical")
else:
    print(f"QAOA best-shot partition:\n  {partition(best_shot_x)}")
print(f"\nNote: brute force wins at n={n} (2^{n} = {2 ** n} states), but it "
      f"doubles per added substation; at n=60 it is ~10^18 states while the "
      f"QAOA circuit only grows polynomially.")
