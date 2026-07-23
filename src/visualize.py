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

import matplotlib.pyplot as plt
import networkx as nx

from src import graph

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


# Partition colors (fault-zone side 0 / 1) for the QAOA result.
PART_COLOR = {0: "#1f77b4", 1: "#d62728"}  # blue / red
CUT_COLOR = "#ff7f0e"  # highlighted cut lines


def plot_partition(sub: nx.Graph, partition: dict, out: Path,
                   label: str = "QAOA fault-zone partition",
                   cut_value: float | None = None) -> Path:
    """Plot the subgrid painted by a QAOA partition.

    ``partition`` maps each node id to its fault-zone side ``{0, 1}``. Nodes are
    colored by side, edges whose endpoints fall on *different* sides (the cut
    lines) are highlighted and dashed, and generator nodes are ringed. When
    ``cut_value`` is omitted it is computed as the total ``weight`` of the cut
    edges.
    """
    pos = _positions(sub)
    if len(pos) < sub.number_of_nodes():
        pos = nx.spring_layout(sub, seed=42, weight="weight")

    cut_edges = [(u, v) for u, v in sub.edges()
                 if partition.get(u) != partition.get(v)]
    kept_edges = [(u, v) for u, v in sub.edges()
                  if partition.get(u) == partition.get(v)]
    if cut_value is None:
        cut_value = sum(sub[u][v].get("weight", 0.0) for u, v in cut_edges)

    fig, ax = plt.subplots(figsize=(10, 9))

    nx.draw_networkx_edges(sub, pos, edgelist=kept_edges, ax=ax,
                           edge_color="#bbbbbb", width=1.8, alpha=0.8)
    nx.draw_networkx_edges(sub, pos, edgelist=cut_edges, ax=ax,
                           edge_color=CUT_COLOR, width=2.6, style="dashed")

    for side, color in PART_COLOR.items():
        nodes = [n for n in sub.nodes if partition.get(n) == side]
        if nodes:
            nx.draw_networkx_nodes(sub, pos, nodelist=nodes, ax=ax,
                                   node_size=650, node_color=color,
                                   edgecolors="black", linewidths=0.8,
                                   label=f"Zone {side}")

    gens = [n for n, d in sub.nodes(data=True) if d.get("n_generators", 0) > 0]
    if gens:
        nx.draw_networkx_nodes(sub, pos, nodelist=gens, ax=ax, node_size=980,
                               node_color="none", edgecolors="#ffbf00",
                               linewidths=1.6)
        ax.plot([], [], marker="o", mfc="none", mec="#ffbf00", ls="none",
                label="Generation")

    labels = {n: d.get("name", n) for n, d in sub.nodes(data=True)}
    nx.draw_networkx_labels(sub, pos, labels, ax=ax, font_size=8,
                            font_color="white")
    edge_labels = {(u, v): f"{d['weight']:.1f}"
                   for u, v, d in sub.edges(data=True)}
    nx.draw_networkx_edge_labels(sub, pos, edge_labels, ax=ax, font_size=7,
                                 label_pos=0.5, bbox=dict(boxstyle="round,pad=0.15",
                                 fc="white", ec="none", alpha=0.7))

    ax.plot([], [], color=CUT_COLOR, lw=2.6, ls="dashed", label="Cut line")
    ax.set_title(f"{label}\n{len(cut_edges)} cut lines · "
                 f"cut weight = {cut_value:.1f}", fontsize=13)
    ax.legend(loc="best", framealpha=0.9)
    ax.axis("off")
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    subs = json.loads((RAW_DIR / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW_DIR / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, nodes=graph.GUANACASTE_NORTH,
                                  max_nodes=len(graph.GUANACASTE_NORTH))

    p1 = plot_national(G, highlight=set(sub.nodes), out=FIG_DIR / "national_grid.png",
                       highlight_label="Guanacaste North subgrid")
    p2 = plot_subgraph(sub, out=FIG_DIR / "guanacaste_north_subgrid.png",
                       label="Guanacaste North subgrid")
    print("Figures generated:")
    print(" -", p1)
    print(" -", p2)


if __name__ == "__main__":
    main()
