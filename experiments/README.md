# Experiments

The comparative evaluation of **QAOA (Quantinuum Nexus / Helios-1E-lite)** against
the classical Max-Cut baselines (**brute force**, **greedy**, **Goemans-Williamson**)
now lives in two places:

- **`src/benchmark.py`** — the supporting module: grid growth (9 → 15 → 26 nodes),
  the vectorized timeout-guarded brute-force baseline (GW fallback on timeout), the
  classical samplers, the QAOA-on-Helios runner (one cloud job per COBYLA
  iteration, job refs saved for per-iteration log extraction), the parallel driver,
  and the metrics.
- **`notebooks/evaluation.ipynb`** — the orchestration notebook: configure the
  `shots` × `max_iter` sweep, build the grids, compute baselines, run everything in
  parallel (awaiting all tasks), tabulate the metrics (mean execution time, MSE,
  std dev, approximation ratio), and render the figures (energy histograms,
  approximation-ratio box plots per combination, approximation-ratio convergence).

Generated artifacts are written here at run time:

- `experiments/results/evaluation_<timestamp>.json` — raw records, baselines, and
  the summary table.
- `experiments/refs/<config>/` — saved Nexus job references, so each QAOA
  iteration's shot distribution can be re-fetched from the Nexus job results.

These directories are created on demand and are not committed.
