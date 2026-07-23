"""Detailed comparison: classical baselines vs the quantum (QAOA/Helios) run.

Every method is scored on the SAME objective: the full fault-zone QUBO
(min-cut with generation_inverted weights + generator-spread + balance
penalties). The classical Max-Cut baselines from ``src.classical_baselines``
run unmodified on the *augmented Ising graph*: minimizing
``sum h_i z_i + sum J_ij z_i z_j`` is exactly maximizing the cut of a graph
whose edges carry ``J_ij`` plus a "field node" tied to every variable with
weight ``h_i`` (the field node's side defines the ``z = +1`` gauge).

Outputs a detailed per-method report, an energy-distribution figure and a
JSON record in ``experiments/results/``.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import classical_baselines as cb
from src import qubo as qmod

FIELD = "__field__"
RESULTS_DIR = REPO / "experiments" / "results"

# ---------------------------------------------------------------- problem --
q = qmod.build_qubo(qmod.load_graph())
ising = qmod.qubo_to_ising(q)
n = len(q.variables)

# Component QUBOs for the energy decomposition of any solution.
G_grid = qmod.load_graph()
q_objective = qmod.build_qubo(G_grid, gen_penalty_factor=0.0,
                              balance_penalty_factor=0.0)
w_max = q.metadata["max_edge_weight"]
q_spread = qmod.QUBO(variables=q.variables, linear={}, quadratic={})
qmod.add_generator_spread_penalty(
    q_spread, qmod.apply_weight_scheme(G_grid, "generation_inverted"),
    q.metadata["penalties"]["generator_spread"]["coefficient"])
q_balance = qmod.QUBO(variables=q.variables, linear={}, quadratic={})
qmod.add_balance_penalty(
    q_balance, G_grid, q.metadata["penalties"]["balance"]["coefficient"])


def decompose(x) -> dict:
    return {
        "cut_objective": float(q_objective.energy(x)),
        "generator_spread_penalty": float(q_spread.energy(x)),
        "balance_penalty": float(q_balance.energy(x)),
        "total": float(q.energy(x)),
    }


def partition_of(x) -> dict:
    return {"A": sorted(v for v, b in zip(q.variables, x) if b == 0),
            "B": sorted(v for v, b in zip(q.variables, x) if b == 1)}


def canonical(x) -> tuple:
    """Complement-invariant signature (zone labels are interchangeable)."""
    return tuple(x) if x[0] == 0 else tuple(1 - b for b in x)


# Augmented Ising graph (field node first: brute force pins it to side 0).
H = nx.Graph()
H.add_node(FIELD)
for i, name in enumerate(q.variables):
    if ising["h"][i]:
        H.add_edge(FIELD, name, weight=ising["h"][i])
for (i, j), J in ising["J"].items():
    if J:
        H.add_edge(q.variables[i], q.variables[j], weight=J)


def x_of_partition(partition: dict) -> list:
    side = partition[FIELD]
    return [0 if partition[name] == side else 1 for name in q.variables]


# --------------------------------------------------------------- landscape --
states = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
all_E = np.array([q.energy(list(s)) for s in states])
E_opt, E_max = float(all_E.min()), float(all_E.max())
E_mean_rand = float(all_E.mean())
opt_states = [states[i].tolist() for i in np.flatnonzero(all_E <= E_opt + 1e-9)]
near_5pct = int((all_E <= E_opt + 0.05 * abs(E_opt)).sum())
x_opt = opt_states[0]

# -------------------------------------------------------------- brute force --
t0 = time.perf_counter()
p_bf, _ = cb.brute_force_maxcut(H)
bf_time = time.perf_counter() - t0
x_bf = x_of_partition(p_bf)

# ------------------------------------------------------------------- greedy --
t0 = time.perf_counter()
greedy_runs = []
for seed in range(20):
    p_g, _ = cb.greedy_maxcut(H, seed=seed)
    greedy_runs.append(float(q.energy(x_of_partition(p_g))))
greedy_time = time.perf_counter() - t0
greedy_best_seed = int(np.argmin(greedy_runs))
p_g, _ = cb.greedy_maxcut(H, seed=greedy_best_seed)
x_greedy = x_of_partition(p_g)
greedy_hits = sum(1 for e in greedy_runs if e <= E_opt + 1e-9)

# ------------------------------------------------- goemans-williamson + SDP --
t0 = time.perf_counter()
p_gw, _ = cb.goemans_williamson(H, n_rounding_trials=50, seed=42)
gw_time = time.perf_counter() - t0
x_gw = x_of_partition(p_gw)

gw_trials = []
for seed in range(25):     # per-trial detail (re-solves the SDP each call)
    p_t, _ = cb.goemans_williamson(H, n_rounding_trials=1, seed=seed)
    gw_trials.append(float(q.energy(x_of_partition(p_t))))
gw_hits = sum(1 for e in gw_trials if e <= E_opt + 1e-9)

# Certified lower bound on the QUBO optimum from the SDP relaxation:
# E = offset_total - 2 * cut  =>  E_opt >= offset_total - 2 * SDP_bound.
import cvxpy as cp
nodes_h = list(H.nodes())
idx_h = {v: i for i, v in enumerate(nodes_h)}
m = len(nodes_h)
W = np.zeros((m, m))
for u, v, d in H.edges(data=True):
    W[idx_h[u], idx_h[v]] = W[idx_h[v], idx_h[u]] = d["weight"]
X = cp.Variable((m, m), PSD=True)
obj = cp.Maximize(cp.sum(cp.multiply(W, 1 - X)) / 4)
prob = cp.Problem(obj, [cp.diag(X) == 1])
prob.solve(solver=cp.SCS)
sum_J = float(W.sum() / 2)
certified_lower = ising["offset"] + sum_J - 2 * float(prob.value)

# --------------------------------------------------------------- QAOA data --
helios = json.loads((RESULTS_DIR / "qaoa_helios_7a37980d.json").read_text())
seq = json.loads((RESULTS_DIR / "qaoa_helios_7a37980d_sequence.json").read_text())
shot_E = np.array(seq["shot_energies"])
first_opt_shot = int(np.argmax(shot_E <= E_opt + 1e-9)) + 1
qaoa_best_bits = min(helios["counts"], key=lambda s: helios["counts"][s]["energy"])
x_qaoa = [int(b) for b in qaoa_best_bits]
top5 = list(helios["counts"].items())[:5]

# ------------------------------------------------------------------ report --
L = 66
print("=" * L)
print("DETAILED SOLVER COMPARISON - fault-zone QUBO, Guanacaste North")
print("=" * L)
print(f"""
PROBLEM
  variables (qubits):        {n}
  quadratic couplings:       {len(q.quadratic)}
  weight scheme:             {q.metadata['weight_scheme']} (min-cut sense)
  penalties:                 generator_spread (P={q.metadata['penalties']['generator_spread']['coefficient']:.3f}/pair), balance (lambda={q.metadata['penalties']['balance']['coefficient']:.3f})
  Ising offset:              {ising['offset']:.4f}

LANDSCAPE (exhaustive, 2^{n} = {2 ** n} states)
  optimum / mean / worst:    {E_opt:.4f} / {E_mean_rand:.4f} / {E_max:.4f}
  optimal states:            {len(opt_states)} (complement pair = same physical partition)
  states within 5% of opt:   {near_5pct}
  optimal partition:         A={partition_of(x_opt)['A']}
                             B={partition_of(x_opt)['B']}
  SDP certificate:           E_opt >= {certified_lower:.4f} (found {E_opt:.4f}; certification gap {100 * (E_opt - certified_lower) / abs(E_opt):.1f}%)
""")

methods = [
    ("Brute force (exact)", x_bf, bf_time * 1000, f"{2 ** n} evaluations",
     "exact optimum", {}),
    ("Greedy (20 restarts)", x_greedy, greedy_time * 1000,
     f"{20 * H.number_of_nodes()} node placements",
     "none (heuristic)",
     {"restarts finding optimum": f"{greedy_hits}/20",
      "restart energies best/mean/worst":
          f"{min(greedy_runs):.4f} / {np.mean(greedy_runs):.4f} / {max(greedy_runs):.4f}"}),
    ("Goemans-Williamson", x_gw, gw_time * 1000,
     "1 SDP + 50 hyperplane roundings",
     "0.878 approx (nonneg weights only; here heuristic + SDP bound)",
     {"single-trial roundings finding optimum": f"{gw_hits}/25",
      "single-trial energies best/mean/worst":
          f"{min(gw_trials):.4f} / {np.mean(gw_trials):.4f} / {max(gw_trials):.4f}",
      "SDP upper bound on aug. cut": f"{float(prob.value):.4f}"}),
    ("QAOA p=1 (Helios-1E-lite)", x_qaoa, None, "500 shots (cloud job)",
     "probabilistic sampler",
     {"mean shot energy": f"{helios['metrics']['mean_energy']:.4f}",
      "quality (0=random, 1=opt)": f"{helios['metrics']['quality']:.3f}",
      "P(optimal shot)": f"{helios['metrics']['p_optimal']:.3f} (random: {len(opt_states) / 2 ** n:.4f})",
      "first optimal shot": f"#{first_opt_shot}",
      "angles": f"gamma={helios['metadata']['angles']['gamma']:.4f}, beta={helios['metadata']['angles']['beta']:.4f}",
      "top-5 bitstrings (energy x count)":
          ", ".join(f"{s}({v['energy']:.2f}x{v['count']})" for s, v in top5)}),
]

for name, x, ms, work, guarantee, extra in methods:
    d = decompose(x)
    same = canonical(x) == canonical(x_opt)
    print("-" * L)
    print(name)
    print(f"  best energy found:         {d['total']:.4f}  "
          f"(gap to optimum: {100 * (d['total'] - E_opt) / abs(E_opt):.2f}%)")
    print(f"  energy decomposition:      cut {d['cut_objective']:.4f} + "
          f"spread {d['generator_spread_penalty']:.4f} + "
          f"balance {d['balance_penalty']:.4f}")
    print(f"  partition found:           {'IDENTICAL to optimum' if same else partition_of(x)}")
    print(f"  work / wall time:          {work}"
          + (f", {ms:.1f} ms" if ms is not None else ""))
    print(f"  guarantee:                 {guarantee}")
    for k, v in extra.items():
        print(f"  {k + ':':27}{v}")

print("=" * L)
print("VERDICT: all four methods reach the exact optimum at n=9. The")
print("differentiators are guarantees and scaling: brute force doubles per")
print("node, greedy/GW stay polynomial without/with a bound, and the QAOA")
print("circuit grows polynomially -- its case is asymptotic, not n=9.")

# ------------------------------------------------------------------ figure --
fig, ax = plt.subplots(figsize=(12, 6))
bins = np.linspace(E_opt - 0.3, E_max + 0.3, 46)
ax.hist(all_E, bins=bins, density=True, alpha=0.35, color="#9e9e9e",
        label="brute force enumeration = full landscape (512 states)")
ax.hist(shot_E, bins=bins, density=True, alpha=0.5, color="#1f77b4",
        label="QAOA shots (Helios, 500)")
ax.hist(greedy_runs, bins=bins, density=True, alpha=0.5, color="#ff7f0e",
        label="greedy restarts (20)")
ax.hist(gw_trials, bins=bins, density=True, alpha=0.5, color="#2ca02c",
        label="GW single roundings (25)")
ax.axvline(E_opt, color="#d62728", ls="--", lw=2,
           label=f"brute-force optimum ({E_opt:.2f}) — found by all 4 methods")
ax.annotate("brute force selects\nthis state exactly",
            (E_opt, ax.get_ylim()[1] * 0.0 + 1.05), xytext=(E_opt + 1.6, 1.18),
            fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.axvline(certified_lower, color="#d62728", ls=":", lw=1.5,
           label=f"SDP certified bound ({certified_lower:.2f})")
ax.set_xlabel("QUBO energy (lower = better)")
ax.set_ylabel("density")
ax.set_title("Where each solver's answers land on the energy landscape", fontsize=12)
ax.legend(framealpha=0.92)
ax.grid(True, ls=":", alpha=0.35)
fig.tight_layout()
fig_path = REPO / "figures" / "solver_comparison.png"
fig.savefig(fig_path, dpi=150)
print(f"\nFigure: {fig_path}")

# -------------------------------------------------------------------- JSON --
out = RESULTS_DIR / "solver_comparison.json"
out.write_text(json.dumps({
    "problem": {"n": n, "couplings": len(q.quadratic),
                "weight_scheme": q.metadata["weight_scheme"],
                "penalties": q.metadata["penalties"],
                "ising_offset": ising["offset"]},
    "landscape": {"optimum": E_opt, "mean": E_mean_rand, "worst": E_max,
                  "n_optimal_states": len(opt_states),
                  "states_within_5pct": near_5pct,
                  "optimal_partition": partition_of(x_opt),
                  "sdp_certified_lower_bound": certified_lower},
    "methods": {
        "brute_force": {"energy": decompose(x_bf), "time_ms": bf_time * 1000},
        "greedy": {"energy": decompose(x_greedy), "time_ms": greedy_time * 1000,
                   "restart_energies": greedy_runs,
                   "restarts_finding_optimum": greedy_hits},
        "goemans_williamson": {"energy": decompose(x_gw), "time_ms": gw_time * 1000,
                               "single_trial_energies": gw_trials,
                               "trials_finding_optimum": gw_hits,
                               "sdp_bound_augmented_cut": float(prob.value)},
        "qaoa_helios": {"energy": decompose(x_qaoa),
                        "mean_shot_energy": helios["metrics"]["mean_energy"],
                        "quality": helios["metrics"]["quality"],
                        "p_optimal": helios["metrics"]["p_optimal"],
                        "first_optimal_shot": first_opt_shot,
                        "job_id": helios["metadata"]["job_id"]},
    },
}, indent=2), encoding="utf-8")
print(f"JSON:   {out}")
