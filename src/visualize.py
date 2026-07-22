"""Visualization of the electrical grid graph.

Generates four figures from the ICE snapshot:

  1. The full national grid, with substations placed at their real geographic
     coordinates (lon/lat) and lines colored by voltage level. The Valle
     Central cluster used as the default subgrid is highlighted.
  2. The Valle Central subgrid alone, with node labels and edge weights.
  3. The national grid overlaid with the generation plants layer
     (``Plantas_NGICE``), colored by technology and sized by MW, with
     connectors from each plant to its matched substation node.
  4. A per-technology summary of how much generation the graph captures
     (matched vs. unmatched MW and plant counts).

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

# Colors by generation technology (ICE ``Tecnologia`` values).
TECH_COLOR = {
    "Hidroeléctrico": "#1f77b4",
    "Geotérmico": "#d62728",
    "Eólico": "#2ca02c",
    "Solar": "#ff7f0e",
    "Térmico": "#8c564b",
}


def _edge_color(voltage) -> str:
    return VOLT_COLOR.get(voltage, "#7f7f7f")


def _positions(G: nx.Graph) -> dict:
    """Geographic coordinates (lon, lat) of each node, when available."""
    return {
        n: (d["x"], d["y"])
        for n, d in G.nodes(data=True)
        if d.get("x") is not None and d.get("y") is not None
    }


def plot_national(G: nx.Graph, highlight: set[str], out: Path) -> Path:
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
                   zorder=3, label="Valle Central subgrid")

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


def plot_subgraph(sub: nx.Graph, out: Path) -> Path:
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
    ax.set_title("Valle Central subgrid (subgraph for Max-Cut / QAOA)\n"
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


def plot_plants_overlay(G: nx.Graph, plants_geojson: dict, report: dict,
                        out: Path) -> Path:
    """Overlay the generation plants on the national grid.

    Matched plants are connected to their substation node with a dashed line;
    unmatched ones are drawn with an ``x``. Marker area scales with MW.
    """
    pos = _positions(G)
    matched_nodes = {m["plant"]: m["node"] for m in report["matched"]}

    fig, ax = plt.subplots(figsize=(11, 13))

    # Grid as light-gray background so the technology colors stand out.
    for u, v, _ in G.edges(data=True):
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="#bbbbbb", lw=1.0, alpha=0.8, zorder=1)
    ax.scatter([xy[0] for xy in pos.values()], [xy[1] for xy in pos.values()],
               s=25, c="#555555", zorder=2, label="Substation")

    seen_techs = set()
    for feat in plants_geojson.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        plant, tech = props.get("Planta"), props.get("Tecnologia")
        if not plant or not coords:
            continue
        mw = props.get("PotenciaEfectivaMW") or 0.0
        color = TECH_COLOR.get(tech, "#7f7f7f")
        seen_techs.add(tech)

        node = matched_nodes.get(plant)
        if node is not None and node in pos:
            ax.plot([coords[0], pos[node][0]], [coords[1], pos[node][1]],
                    color=color, lw=1.0, ls="--", alpha=0.9, zorder=3)
            marker = "o"
        else:
            marker = "x"
        ax.scatter([coords[0]], [coords[1]], s=25 + 1.2 * mw, c=color,
                   marker=marker, alpha=0.85, edgecolors="black" if marker == "o" else None,
                   linewidths=0.5, zorder=4)

    # Legend via proxies so every technology shows a uniform round marker.
    for tech in sorted(seen_techs):
        ax.scatter([], [], s=60, c=TECH_COLOR.get(tech, "#7f7f7f"),
                   edgecolors="black", linewidths=0.5, label=tech)

    n_matched = len(report["matched"])
    n_total = n_matched + len(report["unmatched"])
    ax.set_title("Generation plants over the transmission grid (ICE)\n"
                 f"{n_matched}/{n_total} plants matched to a substation node · "
                 "marker area = MW · x = unmatched",
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


def plot_plants_coverage(report: dict, out: Path) -> Path:
    """Bar chart: matched vs. unmatched generation (MW) per technology."""
    techs = sorted({p["technology"] for p in report["matched"] + report["unmatched"]})
    matched_mw = {t: sum(p["mw"] for p in report["matched"] if p["technology"] == t)
                  for t in techs}
    unmatched_mw = {t: sum(p["mw"] for p in report["unmatched"] if p["technology"] == t)
                    for t in techs}

    fig, ax = plt.subplots(figsize=(9, 6))
    x = range(len(techs))
    ax.bar(x, [matched_mw[t] for t in techs], width=0.6,
           color=[TECH_COLOR.get(t, "#7f7f7f") for t in techs])
    ax.bar(x, [unmatched_mw[t] for t in techs], width=0.6,
           bottom=[matched_mw[t] for t in techs],
           color=[TECH_COLOR.get(t, "#7f7f7f") for t in techs],
           alpha=0.35, hatch="//")

    # Neutral legend proxies (the bars themselves are colored per technology).
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#666666", label="Matched to a graph node"),
        Patch(facecolor="#666666", alpha=0.35, hatch="//", label="Unmatched"),
    ], framealpha=0.9)
    for i, t in enumerate(techs):
        total = matched_mw[t] + unmatched_mw[t]
        pct = 100.0 * matched_mw[t] / total if total else 0.0
        ax.text(i, total, f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(techs, rotation=15)
    ax.set_ylabel("Effective power (MW)")
    ax.set_title("How much ICE generation the grid graph captures\n"
                 "solid = plant matched to a substation node · hatched = unmatched",
                 fontsize=12)
    ax.grid(True, axis="y", ls=":", alpha=0.3)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    subs = json.loads((RAW_DIR / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW_DIR / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region=None, max_nodes=12)

    paths = [
        plot_national(G, highlight=set(sub.nodes), out=FIG_DIR / "national_grid.png"),
        plot_subgraph(sub, out=FIG_DIR / "valle_central_subgrid.png"),
    ]

    plants_path = RAW_DIR / "plants.geojson"
    if plants_path.exists():
        plants = json.loads(plants_path.read_text(encoding="utf-8"))
        report = graph.annotate_generators(G, plants)
        paths.append(plot_plants_overlay(G, plants, report,
                                         out=FIG_DIR / "national_grid_plants.png"))
        paths.append(plot_plants_coverage(report,
                                          out=FIG_DIR / "generation_coverage.png"))

    print("Figures generated:")
    for p in paths:
        print(" -", p)


if __name__ == "__main__":
    main()
