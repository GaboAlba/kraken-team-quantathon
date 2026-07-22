"""Construction of the electrical grid graph from the ICE data.

Pipeline (Task A):

    ICE snapshot (GeoJSON)  ->  national graph (NetworkX)  ->  subgrid  ->  grid_cr.json

- Nodes: substations (``Subestaciones`` layer).
- Edges: transmission lines (``LineasDeTransmision`` layer). Connectivity is
  derived from the ``Circuito`` field with format ``"SubstationA-SubstationB"``.
- Weight: interchangeable function from ``src.weights`` (default ``kv``).

See ``Docs/desiciones.md`` for the rationale behind each decision.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from pathlib import Path

import networkx as nx

from src import weights

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "data" / "grid_cr.json"

# Known aliases: name in the Circuito field -> actual substation name.
# Applied after the general normalization.
_ALIASES = {
    "garita": "la garita",
}


def normalize_name(s: str | None) -> str:
    """Normalize a name so circuits can be matched against substations.

    - lowercases and strips accents;
    - collapses whitespace;
    - drops the parenthesized suffix (e.g. ``"(SIEPAC)"``);
    - drops a trailing bay/circuit digit (``"Colima2"`` -> ``"colima"``).
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)          # drops "(siepac)" and the like
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*\d+$", "", s)           # drops trailing digit ("colima2")
    return s.strip()


def _canonical(name: str) -> str:
    """Apply normalization + aliases to obtain the node identifier."""
    n = normalize_name(name)
    return _ALIASES.get(n, n)


def parse_circuit(circuit: str | None) -> tuple[str, str] | None:
    """Return the two normalized endpoints of an ``"A-B"`` circuit.

    Returns ``None`` if the format is not exactly two parts separated by a
    hyphen (e.g. ``"SIEPAC"`` or empty strings).
    """
    if not circuit:
        return None
    parts = circuit.split("-")
    if len(parts) != 2:
        return None
    a, b = _canonical(parts[0]), _canonical(parts[1])
    if not a or not b:
        return None
    return a, b


def parse_substations(geojson: dict) -> dict[str, dict]:
    """Extract the nodes (substations) from the GeoJSON as id -> attributes."""
    nodes: dict[str, dict] = {}
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        name = props.get("Subestacio")
        if not name:
            continue
        node_id = _canonical(name)
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None]
        nodes[node_id] = {
            "id": node_id,
            "name": name,
            "province": props.get("Provincia"),
            "canton": props.get("Canton"),
            "x": coords[0],
            "y": coords[1],
            "border": False,
        }
    return nodes


def build_national_graph(
    subs_geojson: dict,
    lines_geojson: dict,
    weight_scheme: str = weights.DEFAULT_SCHEME,
) -> tuple[nx.Graph, dict]:
    """Build the weighted national graph.

    Circuit endpoints that do not match a known substation are added as
    *border* nodes (``border=True``): international interconnections, SIEPAC
    or industrial loads. Parallel lines between the same pair are collapsed by
    summing their weights.

    Returns ``(graph, report)`` where ``report`` documents what was not
    recognized.
    """
    weight_fn = weights.SCHEMES[weight_scheme]
    G = nx.Graph()

    nodes = parse_substations(subs_geojson)
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)

    unrecognized_endpoints: set[str] = set()
    ignored_circuits: list[str] = []

    for feat in lines_geojson.get("features", []):
        props = feat.get("properties", {})
        endpoints = parse_circuit(props.get("Circuito"))
        if endpoints is None:
            ignored_circuits.append(props.get("Circuito"))
            continue
        u, v = endpoints
        if u == v:
            continue

        # Unknown endpoints -> border nodes.
        for ep in (u, v):
            if ep not in G:
                G.add_node(ep, id=ep, name=ep.title(), province=None,
                           canton=None, x=None, y=None, border=True)
                unrecognized_endpoints.add(ep)

        voltage = props.get("Voltaje")
        length_m = props.get("Shape__Length")
        w = weight_fn(voltage=voltage, length_m=length_m)

        if G.has_edge(u, v):
            # Parallel line: sum the weights and keep the highest voltage.
            G[u][v]["weight"] += w
            G[u][v]["voltage"] = max(G[u][v]["voltage"], voltage)
        else:
            G.add_edge(u, v, weight=w, voltage=voltage,
                       circuit=props.get("Circuito"))

    report = {
        "unrecognized_endpoints": sorted(unrecognized_endpoints),
        "ignored_circuits": ignored_circuits,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }
    return G, report


def extract_subregion(
    G: nx.Graph,
    region: str | None = None,
    nodes: list[str] | None = None,
    max_nodes: int = 12,
) -> nx.Graph:
    """Extract a connected subgrid of at most ``max_nodes`` nodes.

    Selection modes (mutually exclusive):
      - ``nodes``: explicit list of substation ids;
      - ``region``: province (matched loosely);
      - neither (**connectivity mode**, default): all real substations, which
        preserves the national mesh and its cycles.

    In all cases the induced subgraph is taken, the **largest connected
    component** is kept and, if it exceeds ``max_nodes``, it is trimmed with a
    BFS-style growth from the highest weighted-degree node to keep it connected.
    The seed node and tie-breaks are deterministic (alphabetical order) to
    guarantee reproducibility.

    Note: filtering by a single province yields almost radial subgraphs (trees),
    whose Max-Cut is trivial; connectivity mode is the default in the
    experiments because it preserves cycles (see ``Docs/desiciones.md``).
    """
    if nodes is not None:
        selected = [n for n in nodes if n in G]
    elif region is not None:
        r = normalize_name(region)
        selected = [
            n for n, d in G.nodes(data=True)
            if d.get("province") and normalize_name(d["province"]) == r
        ]
    else:
        # Connectivity mode: real substations only (no border nodes).
        selected = [n for n, d in G.nodes(data=True) if not d.get("border")]

    H = G.subgraph(selected).copy()
    if H.number_of_nodes() == 0:
        return H

    # Largest connected component (deterministic tie-break by minimum node).
    largest = max(nx.connected_components(H), key=lambda c: (len(c), min(c)))
    H = H.subgraph(largest).copy()

    if H.number_of_nodes() <= max_nodes:
        return H

    # Connected trim: BFS from the highest weighted-degree node (deterministic).
    seed = max(H.nodes, key=lambda n: (H.degree(n, weight="weight"), n))
    keep = [seed]
    for _, node in nx.bfs_edges(H, seed):
        if len(keep) >= max_nodes:
            break
        keep.append(node)
    return H.subgraph(keep).copy()


def to_json(G: nx.Graph, metadata: dict | None = None) -> dict:
    """Serialize the graph to the ``grid_cr.json`` schema."""
    nodes = [
        {
            "id": n,
            "name": d.get("name"),
            "province": d.get("province"),
            "canton": d.get("canton"),
            "x": d.get("x"),
            "y": d.get("y"),
            "border": d.get("border", False),
        }
        for n, d in G.nodes(data=True)
    ]
    edges = [
        {
            "u": u,
            "v": v,
            "weight": d.get("weight"),
            "voltage": d.get("voltage"),
            "circuit": d.get("circuit"),
        }
        for u, v, d in G.edges(data=True)
    ]
    return {"metadata": metadata or {}, "nodes": nodes, "edges": edges}


def save_graph(G: nx.Graph, path: Path, metadata: dict | None = None) -> Path:
    """Write the graph to disk in JSON format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = to_json(G, metadata)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build(
    region: str | None = None,
    nodes: list[str] | None = None,
    weight_scheme: str = weights.DEFAULT_SCHEME,
    max_nodes: int = 12,
    label: str = "valle_central",
    output: Path = DEFAULT_OUTPUT,
    raw_dir: Path = RAW_DIR,
) -> nx.Graph:
    """Orchestrate the full pipeline and write ``data/grid_cr.json``.

    By default (``region=None``) it uses connectivity mode, which yields the
    meshed Valle Central cluster; ``region="Guanacaste"`` selects that province.
    """
    subs = json.loads((raw_dir / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((raw_dir / "lines.geojson").read_text(encoding="utf-8"))
    source_info = {}
    source_path = raw_dir / "source.json"
    if source_path.exists():
        source_info = json.loads(source_path.read_text(encoding="utf-8"))

    G, report = build_national_graph(subs, lines, weight_scheme=weight_scheme)
    sub = extract_subregion(G, region=region, nodes=nodes, max_nodes=max_nodes)

    # Number of independent cycles: indicator of how "interesting" the Max-Cut is.
    cycles = (sub.number_of_edges() - sub.number_of_nodes()
              + nx.number_connected_components(sub)) if sub.number_of_nodes() else 0

    metadata = {
        "source": source_info.get("source"),
        "urls": source_info.get("urls"),
        "download_date": source_info.get("download_date"),
        "build_date": date.today().isoformat(),
        "weight_scheme": weight_scheme,
        "region": region if region is not None else label,
        "selection_mode": "nodes" if nodes else ("province" if region else "connectivity"),
        "max_nodes": max_nodes,
        "n_nodes": sub.number_of_nodes(),
        "n_edges": sub.number_of_edges(),
        "independent_cycles": cycles,
        "national_report": report,
    }
    save_graph(sub, output, metadata)
    return sub


if __name__ == "__main__":
    g = build()
    print(f"Subgrid: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
          f"-> {DEFAULT_OUTPUT}")
    print("Nodes:", ", ".join(sorted(g.nodes)))
