"""Construction of the electrical grid graph from the ICE data.

Pipeline (Task A):

    ICE snapshot (GeoJSON)  ->  national graph (NetworkX)  ->  subgrid  ->  grid_cr.json

- Nodes: substations (``Subestaciones`` layer). Generation plants
  (``Plantas_NGICE`` layer) within 20 km are attached to each substation node.
- Edges: transmission lines (``LineasDeTransmision`` layer). Connectivity is
  derived from the ``Circuito`` field with format ``"SubstationA-SubstationB"``.
- Weight: interchangeable function from ``src.weights`` (default
  ``generation_inverted``, which is generator-aware; see ``docs/qubo.md``).

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

# Subgrid chosen for the experiments: northern Guanacaste. Guanacaste
# substations with lat >= 10.4, minus the leaves filadelfia/sandillal/tejona
# and the isolated tanque. Forms the 230 kV ring Liberia-Pailas-Mogote-
# Miravalles-Arenal-Corobici-Canas-Liberia (1 cycle) plus the
# Papagayo-Nuevo Colon branch.
GUANACASTE_NORTH = [
    "arenal", "canas", "corobici", "liberia", "miravalles",
    "mogote", "nuevo colon", "pailas", "papagayo",
]


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
            # Projected coordinates (CRTM05, metres) for distance queries; kept
            # internal (not serialized) and used to assign nearby generators.
            "px": props.get("PuntoX"),
            "py": props.get("PuntoY"),
            "border": False,
        }
    return nodes


def _normalize_technology(technology: str | None) -> str:
    """Lowercase, accent-stripped technology label (``"Térmico"`` -> ``"termico"``)."""
    if not technology:
        return ""
    t = unicodedata.normalize("NFKD", technology).encode("ascii", "ignore").decode()
    return t.lower().strip()


def is_thermal(technology: str | None) -> bool:
    """Return ``True`` for fossil thermal plants (``"Térmico"``/``"Térmica"``).

    Geothermal (``"Geotérmico"``) is explicitly excluded: only fossil thermal
    generation is treated as non-sustainable for the weight penalty.
    """
    return _normalize_technology(technology) in {"termico", "termica"}


def parse_generators(geojson: dict) -> list[dict]:
    """Extract active generation plants from the ICE ``Plantas_NGICE`` GeoJSON.

    Only active plants (``EstAct == "Activo"``) with projected coordinates are
    kept. Each generator is ``{plant, technology, thermal, power_mw, x, y}``
    where ``x``/``y`` are the CRTM05 projected coordinates (metres).
    """
    generators: list[dict] = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        if props.get("EstAct") != "Activo":
            continue
        x, y = props.get("XCoord"), props.get("YCoord")
        if x is None or y is None:
            continue
        technology = props.get("Tecnologia")
        generators.append({
            "plant": props.get("Planta"),
            "technology": technology,
            "thermal": is_thermal(technology),
            "power_mw": props.get("PotenciaEfectivaMW"),
            "x": float(x),
            "y": float(y),
        })
    return generators


def _distance_m(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in metres between two CRTM05 projected points."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def assign_generators(
    G: nx.Graph,
    generators: list[dict],
    radius_m: float = 10000.0,
) -> nx.Graph:
    """Attach generators within ``radius_m`` of each substation to its node.

    Populates three node attributes on every node:
      - ``generators``: list of ``{plant, technology, thermal, power_mw,
        power_norm, dist_m}`` for plants within the radius, sorted by
        ``(dist_m, plant)`` for determinism;
      - ``n_generators``: number of associated generators;
      - ``n_thermal``: number of associated fossil-thermal generators.

    ``power_norm`` is the generator's ``power_mw`` divided by the largest
    ``power_mw`` among all generators attached anywhere in the graph, so the
    single biggest generator scores 1.0 and the rest score proportionally. The
    graph-global maximum is stored on ``G.graph['max_generator_power_mw']``.

    Border nodes and substations without projected coordinates receive empty
    lists. Mutates and returns ``G``.
    """
    for _, d in G.nodes(data=True):
        px, py = d.get("px"), d.get("py")
        nearby: list[dict] = []
        if px is not None and py is not None:
            for gen in generators:
                dist = _distance_m(float(px), float(py), gen["x"], gen["y"])
                if dist <= radius_m:
                    nearby.append({
                        "plant": gen["plant"],
                        "technology": gen["technology"],
                        "thermal": gen["thermal"],
                        "power_mw": gen["power_mw"],
                        "dist_m": round(dist, 1),
                    })
            nearby.sort(key=lambda g: (g["dist_m"], g["plant"] or ""))
        d["generators"] = nearby
        d["n_generators"] = len(nearby)
        d["n_thermal"] = sum(1 for g in nearby if g["thermal"])

    # Normalize generator sizes by the biggest generator attached to the graph
    # so the largest one contributes 1.0 to the weight and the rest scale down.
    powers = [g["power_mw"] for _, d in G.nodes(data=True)
              for g in d["generators"]
              if g["power_mw"] is not None and g["power_mw"] > 0]
    max_power = max(powers) if powers else 0.0
    G.graph["max_generator_power_mw"] = max_power
    for _, d in G.nodes(data=True):
        for g in d["generators"]:
            power = g["power_mw"]
            g["power_norm"] = (float(power) / max_power
                               if max_power > 0 and power else 0.0)
    return G


def build_national_graph(
    subs_geojson: dict,
    lines_geojson: dict,
    plants_geojson: dict | None = None,
    weight_scheme: str = weights.DEFAULT_SCHEME,
    radius_m: float = 20000.0,
) -> tuple[nx.Graph, dict]:
    """Build the weighted national graph.

    Circuit endpoints that do not match a known substation are added as
    *border* nodes (``border=True``): international interconnections, SIEPAC
    or industrial loads. Parallel lines between the same pair are collapsed by
    summing their weights.

    When ``plants_geojson`` is given, active generation plants within
    ``radius_m`` of each substation are attached to its node (see
    ``assign_generators``); the generator-aware weight schemes use this context.

    Returns ``(graph, report)`` where ``report`` documents what was not
    recognized.
    """
    weight_fn = weights.SCHEMES[weight_scheme]
    G = nx.Graph()

    nodes = parse_substations(subs_geojson)
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)

    # Assign generators to substation nodes before computing edge weights so
    # generator-aware schemes see the endpoint context.
    generators = parse_generators(plants_geojson) if plants_geojson else []
    assign_generators(G, generators, radius_m=radius_m)

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

        # Unknown endpoints -> border nodes (no coordinates, no generators).
        for ep in (u, v):
            if ep not in G:
                G.add_node(ep, id=ep, name=ep.title(), province=None,
                           canton=None, x=None, y=None, px=None, py=None,
                           border=True, generators=[], n_generators=0,
                           n_thermal=0)
                unrecognized_endpoints.add(ep)

        voltage = props.get("Voltaje")
        length_m = props.get("Shape__Length")
        w = weight_fn(voltage=voltage, length_m=length_m,
                      gens_u=G.nodes[u].get("generators", []),
                      gens_v=G.nodes[v].get("generators", []))

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
            "generators": d.get("generators", []),
            "n_generators": d.get("n_generators", 0),
            "n_thermal": d.get("n_thermal", 0),
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
    plants = None
    plants_path = raw_dir / "plants.geojson"
    if plants_path.exists():
        plants = json.loads(plants_path.read_text(encoding="utf-8"))
    source_info = {}
    source_path = raw_dir / "source.json"
    if source_path.exists():
        source_info = json.loads(source_path.read_text(encoding="utf-8"))

    G, report = build_national_graph(subs, lines, plants_geojson=plants,
                                     weight_scheme=weight_scheme)
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
        "max_generator_power_mw": G.graph.get("max_generator_power_mw"),
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
    g = build(nodes=GUANACASTE_NORTH, label="guanacaste_north",
              max_nodes=len(GUANACASTE_NORTH))
    print(f"Subgrid: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
          f"-> {DEFAULT_OUTPUT}")
    print("Nodes:", ", ".join(sorted(g.nodes)))
