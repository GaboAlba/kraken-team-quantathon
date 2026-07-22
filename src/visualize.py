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

    # Nodes: highlighted (Valle Central) vs. the rest.
    rest = [n for n in G.nodes if n not in highlight]
    ax.scatter([pos[n][0] for n in rest], [pos[n][1] for n in rest],
               s=40, c="#444444", zorder=2, label="Substation")
    if highlight:
        ax.scatter([pos[n][0] for n in highlight], [pos[n][1] for n in highlight],
                   s=90, c="#2ca02c", edgecolors="black", linewidths=0.6,
                   zorder=3, label=highlight_label)

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

    edge_labels = {(u, v): f"{int(d['weight'])}" for u, v, d in sub.edges(data=True)}
    nx.draw_networkx_edge_labels(sub, pos, edge_labels, ax=ax, font_size=7,
                                 label_pos=0.5, bbox=dict(boxstyle="round,pad=0.15",
                                 fc="white", ec="none", alpha=0.7))

    for kv, color in VOLT_COLOR.items():
        ax.plot([], [], color=color, lw=2.2, label=VOLT_LABEL[kv])

    cycles = sub.number_of_edges() - sub.number_of_nodes() + \
        nx.number_connected_components(sub)
    ax.set_title(f"{label} (subgraph for Max-Cut / QAOA)\n"
                 f"{sub.number_of_nodes()} nodes · {sub.number_of_edges()} edges · "
                 f"{cycles} cycles · weight = kV",
                 fontsize=13)
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
