"""Max-Cut QUBO formulation of the fault-zone partition.

Each substation gets a binary variable (0 = zone A, 1 = zone B). Maximizing
the weight of the lines *between* zones (where protection elements go) is the
Max-Cut of the grid graph, written as the QUBO

    minimize  f(x) = x^T Q x,   x in {0, 1}^n

with ``Q[i][i] = -weighted_degree(i)`` and ``Q[i][j] = +2 w_ij`` (upper
triangle). Then ``f(x) = -cut(x)``, so the QUBO minimum is the maximum cut.

This matrix is the direct input for QAOA: via ``x_i = (1 - z_i) / 2`` each
quadratic term becomes a weighted ZZ interaction (see ``skills/pytket``).

Usage:
    python -m src.qubo    # build the official subgrid QUBO, solve, plot
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

# Exact enumeration guard: 2^n states; 20 qubits ~ 1M evaluations.
_BRUTE_FORCE_MAX_N = 20


def to_qubo(G: nx.Graph, weight: str = "weight") -> tuple[np.ndarray, list[str]]:
    """Build the Max-Cut QUBO matrix of ``G``.

    Returns ``(Q, labels)`` where ``labels`` is the sorted node list (stable
    qubit index map, per the repo determinism convention) and ``Q`` is upper
    triangular with the linear terms on the diagonal.
    """
    labels = sorted(G.nodes)
    idx = {n: i for i, n in enumerate(labels)}
    Q = np.zeros((len(labels), len(labels)))
    for u, v, d in G.edges(data=True):
        w = float(d[weight])
        i, j = sorted((idx[u], idx[v]))
        Q[i][j] += 2.0 * w
        Q[i][i] -= w
        Q[j][j] -= w
    return Q, labels


def energy(Q: np.ndarray, x) -> float:
    """QUBO objective ``x^T Q x`` (equals minus the cut weight)."""
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x)


def brute_force(Q: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact optimum by enumeration: returns ``(best_cut, x)``.

    Ground truth for QAOA on small instances. Raises ``ValueError`` beyond
    ``_BRUTE_FORCE_MAX_N`` variables (2^n states).
    """
    n = Q.shape[0]
    if n > _BRUTE_FORCE_MAX_N:
        raise ValueError(f"brute_force is exponential; refusing n={n} > {_BRUTE_FORCE_MAX_N}")
    # All bitstrings as an (2^n, n) 0/1 matrix; row-wise energy in one shot.
    states = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    energies = np.einsum("si,ij,sj->s", states, Q, states)
    best = int(np.argmin(energies))
    return float(-energies[best]), states[best].astype(int)


def plot_qubo(Q: np.ndarray, labels: list[str], sub: nx.Graph,
              x: np.ndarray, out: Path) -> Path:
    """Two-panel figure: the Q matrix heatmap and the optimal partition."""
    import matplotlib.pyplot as plt

    from src.visualize import _positions

    cut = -energy(Q, x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Q as a heatmap (diagonal negative, couplings positive).
    full = Q + np.triu(Q, 1).T          # symmetrize for display only
    vmax = np.abs(full).max()
    im = ax1.imshow(full, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if full[i][j]:
                ax1.text(j, i, f"{full[i][j]:.0f}", ha="center", va="center",
                         fontsize=7)
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    short = [f"x{i}·{n}" for i, n in enumerate(labels)]
    ax1.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(short, fontsize=8)
    ax1.set_title("QUBO matrix Q (symmetrized view)\n"
                  "diagonal = -weighted degree · off-diagonal = +2·w(i,j)",
                  fontsize=11)
    fig.colorbar(im, ax=ax1, shrink=0.8)

    # Panel 2: optimal partition on the geographic subgrid.
    pos = _positions(sub)
    zone = {n: x[i] for i, n in enumerate(labels)}
    for u, v, d in sub.edges(data=True):
        in_cut = zone[u] != zone[v]
        ax2.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                 color="#d62728" if in_cut else "#999999",
                 ls="--" if in_cut else "-",
                 lw=2.6 if in_cut else 1.6, alpha=0.9, zorder=1)
    for value, color, label in ((0, "#1f77b4", "Zone A"), (1, "#ff7f0e", "Zone B")):
        ns = [n for n in sub.nodes if zone[n] == value]
        ax2.scatter([pos[n][0] for n in ns], [pos[n][1] for n in ns],
                    s=650, c=color, edgecolors="black", linewidths=0.8,
                    zorder=2, label=label)
    for n, d in sub.nodes(data=True):
        ax2.annotate(d.get("name", n), pos[n], fontsize=8, ha="center",
                     va="center", zorder=3)
    ax2.plot([], [], color="#d62728", ls="--", lw=2.6, label="Cut line (protection)")
    total = sum(d["weight"] for _, _, d in sub.edges(data=True))
    ax2.set_title(f"Optimal partition (brute force)\n"
                  f"cut = {cut:.0f} of {total:.0f} total weight "
                  f"({100 * cut / total:.0f}%)", fontsize=11)
    ax2.legend(loc="lower left", framealpha=0.9)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.axis("off")

    fig.suptitle("Max-Cut QUBO of the Guanacaste North subgrid "
                 "(minimize xᵀQx, x ∈ {0,1}⁹)", fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    import json

    from src import graph

    subs = json.loads((graph.RAW_DIR / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((graph.RAW_DIR / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, nodes=graph.GUANACASTE_NORTH,
                                  max_nodes=len(graph.GUANACASTE_NORTH))

    Q, labels = to_qubo(sub)
    best_cut, x = brute_force(Q)
    zone_a = sorted(n for i, n in enumerate(labels) if x[i] == 0)
    zone_b = sorted(n for i, n in enumerate(labels) if x[i] == 1)
    print(f"QUBO: {len(labels)} variables; optimal cut = {best_cut:.0f}")
    print("Zone A:", ", ".join(zone_a))
    print("Zone B:", ", ".join(zone_b))
    print("Figure:", plot_qubo(Q, labels, sub, x, FIG_DIR / "qubo_maxcut.png"))
