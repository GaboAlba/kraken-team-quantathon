"""Visualization of the electrical grid graph.

Generates two figures from the ICE snapshot:

  1. The full national grid, with substations placed at their real geographic
     coordinates (lon/lat) and lines colored by voltage level. The chosen
     subgrid (northern Guanacaste) is highlighted.
  2. The chosen subgrid alone, with node labels and edge weights.

Usage:
    python -m src.visualize
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx

from src import classical_baselines, graph

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "figures"

# Colors by voltage level (kV).
VOLT_COLOR = {230: "#d62728", 138: "#1f77b4"}
VOLT_LABEL = {230: "230 kV", 138: "138 kV"}


def _edge_color(voltage) -> str:
    return VOLT_COLOR.get(voltage, "#7f7f7f")


def _positions(G: nx.Graph) -> dict:
    """Geographic coordinates (lon, lat) of each node, when available."""
    return {
        n: (d["x"], d["y"])
        for n, d in G.nodes(data=True)
        if d.get("x") is not None and d.get("y") is not None
    }


def plot_national(G: nx.Graph, highlight: set[str], out: Path,
                  highlight_label: str = "Chosen subgrid") -> Path:
    """Plot the national grid using the real coordinates."""
    pos = _positions(G)
    G = G.subgraph(pos).copy()  # nodes with coordinates only
    highlight = highlight & set(G.nodes)

    fig, ax = plt.subplots(figsize=(11, 13))

    # Edges colored by voltage.
    for u, v, d in G.edges(data=True):
        if u in pos and v in pos:
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, color=_edge_color(d.get("voltage")), lw=1.4, alpha=0.7,
                    zorder=1)
            ax.text((x[0] + x[1]) / 2, (y[0] + y[1]) / 2,
                    f"{d.get('weight', 0.0):.1f}", fontsize=5.5,
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.65))

    # Nodes: highlighted (Valle Central) vs. the rest.
    rest = [n for n in G.nodes if n not in highlight]
    ax.scatter([pos[n][0] for n in rest], [pos[n][1] for n in rest],
               s=40, c="#444444", zorder=2, label="Substation")
    if highlight:
        ax.scatter([pos[n][0] for n in highlight], [pos[n][1] for n in highlight],
                   s=90, c="#2ca02c", edgecolors="black", linewidths=0.6,
                   zorder=3, label=highlight_label)
    generator_nodes = [n for n, d in G.nodes(data=True) if d.get("n_generators", 0) > 0]
    if generator_nodes:
        ax.scatter([pos[n][0] for n in generator_nodes],
                   [pos[n][1] for n in generator_nodes],
                   s=130, facecolors="none", edgecolors="#ffbf00", linewidths=1.0,
                   zorder=5, label="Generation")

    # Voltage legend.
    for kv, color in VOLT_COLOR.items():
        ax.plot([], [], color=color, lw=2, label=VOLT_LABEL[kv])

    ax.set_title("Costa Rica electrical transmission grid (ICE)\n"
                 f"{G.number_of_nodes()} substations · {G.number_of_edges()} lines",
                 fontsize=13)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_subgraph(sub: nx.Graph, out: Path,
                  label: str = "Chosen subgrid") -> Path:
    """Plot the subgrid with node labels and edge weights."""
    pos = _positions(sub)
    if len(pos) < sub.number_of_nodes():
        pos = nx.spring_layout(sub, seed=42, weight="weight")

    fig, ax = plt.subplots(figsize=(10, 9))

    edge_colors = [_edge_color(d.get("voltage")) for _, _, d in sub.edges(data=True)]
    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color=edge_colors, width=2.2,
                           alpha=0.8)
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=650, node_color="#2ca02c",
                           edgecolors="black", linewidths=0.8)
    labels = {n: d.get("name", n) for n, d in sub.nodes(data=True)}
    nx.draw_networkx_labels(sub, pos, labels, ax=ax, font_size=8)

    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in sub.edges(data=True)}
    nx.draw_networkx_edge_labels(sub, pos, edge_labels, ax=ax, font_size=7,
                                 label_pos=0.5, bbox=dict(boxstyle="round,pad=0.15",
                                 fc="white", ec="none", alpha=0.7))

    for kv, color in VOLT_COLOR.items():
        ax.plot([], [], color=color, lw=2.2, label=VOLT_LABEL[kv])

    cycles = sub.number_of_edges() - sub.number_of_nodes() + \
        nx.number_connected_components(sub)
    ax.set_title(f"{label} (subgraph for Max-Cut / QAOA)\n"
                 f"{sub.number_of_nodes()} nodes · {sub.number_of_edges()} edges · "
                 f"{cycles} cycles · weight = generation",
                 fontsize=13)
    ax.legend(loc="best", framealpha=0.9)
    ax.axis("off")
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_result_bars(results: dict[str, float], out: Path) -> Path:
    """Grafica una comparación clara de valores de corte entre algoritmos."""
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(results.keys())
    values = [results[name] for name in names]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    ax.barh(names, values, color=colors[: len(names)], edgecolor="#333333", alpha=0.9)
    ax.set_xlabel("Valor de Max-Cut")
    ax.set_title("Comparación de algoritmos clásicos para Max-Cut")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for i, value in enumerate(values):
        ax.text(value + max(values) * 0.01, i, f"{value:.2f}", va="center", fontsize=10)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cut_partition(G: nx.Graph, partition: dict[str, int], out: Path) -> Path:
    """Grafica el corte para una partición de Max-Cut usando un diseño claro."""
    pos = _positions(G)
    if len(pos) < G.number_of_nodes():
        pos = nx.spring_layout(G, seed=42, weight="weight")

    cut_edges = [(u, v) for u, v in G.edges() if partition.get(u) != partition.get(v)]
    same_edges = [(u, v) for u, v in G.edges() if partition.get(u) == partition.get(v)]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=240, node_color=["#1f77b4" if partition.get(n) == 0 else "#ff7f0e" for n in G.nodes()], edgecolors="#222222", linewidths=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=same_edges, edge_color="#bbbbbb", alpha=0.6, width=1.2)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=cut_edges, edge_color="#d62728", alpha=0.95, width=2.3)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="#222222")

    ax.set_title("Partición de Max-Cut: bordes en rojo representan el corte", fontsize=13)
    ax.axis("off")
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    subs = json.loads((RAW_DIR / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW_DIR / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, nodes=graph.GUANACASTE_NORTH,
                                  max_nodes=len(graph.GUANACASTE_NORTH))

    p1 = plot_national(G, highlight=set(sub.nodes), out=FIG_DIR / "red_nacional.png",
                       highlight_label="Guanacaste North subgrid")
    p2 = plot_subgraph(sub, out=FIG_DIR / "subred_valle_central.png",
                       label="Guanacaste North subgrid")

    results = {}
    results["Greedy"] = classical_baselines.greedy_maxcut(sub, seed=0)[1]
    results["Goemans-Williamson"] = classical_baselines.goemans_williamson(sub, n_rounding_trials=100, seed=42)[1]

    p3 = plot_result_bars(results, out=FIG_DIR / "comparacion_algoritmos.png")

    # Usar la partición del mejor algoritmo para mostrar el corte
    best_algo = max(results, key=results.get)
    if best_algo == "Greedy":
        best_partition = classical_baselines.greedy_maxcut(sub, seed=0)[0]
    else:
        best_partition = classical_baselines.goemans_williamson(sub, n_rounding_trials=100, seed=42)[0]

    p4 = plot_cut_partition(sub, best_partition, out=FIG_DIR / "maxcut_partition.png")

    print("Figuras generadas:")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)
    print(" -", p4)


if __name__ == "__main__":
    main()
