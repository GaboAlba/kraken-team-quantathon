"""Read-only grid data for the app: national graph, plants, subgrid rules."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import networkx as nx

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src import graph as graph_mod

RAW = REPO / "data" / "raw"
INITIAL_NODES: list[str] = list(graph_mod.GUANACASTE_NORTH)
PLANT_RADIUS_M = 2000.0
EXACT_ANGLES_MAX_N = 22   # statevector angle search: memory/time bound
BRUTE_FORCE_MAX_N = 40    # beyond this, 2^n enumeration takes days-to-eons


def tier_for(n: int) -> str:
    """exact: fully exact pipeline; heuristic: brute force still runs but
    QAOA angles are untuned; classical: exhaustive search is out of reach."""
    if n <= EXACT_ANGLES_MAX_N:
        return "exact"
    if n <= BRUTE_FORCE_MAX_N:
        return "heuristic"
    return "classical"


@lru_cache(maxsize=1)
def _snapshots() -> tuple[dict, dict, dict]:
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    plants = json.loads((RAW / "plants.geojson").read_text(encoding="utf-8"))
    return subs, lines, plants


@lru_cache(maxsize=1)
def national() -> nx.Graph:
    subs, lines, plants = _snapshots()
    G, _ = graph_mod.build_national_graph(subs, lines, plants_geojson=plants)
    return G


def grid_payload() -> dict:
    G = national()
    _, _, plants_geo = _snapshots()

    nodes = [
        {"id": n, "name": d.get("name"), "lat": d["y"], "lon": d["x"],
         "is_initial": n in INITIAL_NODES}
        for n, d in G.nodes(data=True)
        if not d.get("border") and d.get("x") is not None
    ]
    coords = {n["id"] for n in nodes}
    edges = [
        {"u": u, "v": v, "voltage": d.get("voltage"),
         "weight": round(float(d.get("weight", 0.0)), 4)}
        for u, v, d in G.edges(data=True) if u in coords and v in coords
    ]

    plants = []
    for feat in plants_geo.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry") or {}
        lonlat = geom.get("coordinates")
        if not props.get("Planta") or not lonlat:
            continue
        px, py = props.get("XCoord"), props.get("YCoord")
        substation = None
        if px is not None and py is not None:
            best, best_d = None, PLANT_RADIUS_M
            for n, d in G.nodes(data=True):
                if d.get("px") is None:
                    continue
                dist = graph_mod._distance_m(float(px), float(py),
                                             float(d["px"]), float(d["py"]))
                if dist <= best_d:
                    best, best_d = n, dist
            substation = best
        plants.append({
            "name": props["Planta"],
            "technology": props.get("Tecnologia"),
            "mw": props.get("PotenciaEfectivaMW") or 0.0,
            "lat": lonlat[1], "lon": lonlat[0],
            "substation": substation,
        })
    return {"nodes": nodes, "edges": edges, "plants": plants}


def subgrid_info(node_ids: list[str]) -> dict:
    G = national()
    selection = list(dict.fromkeys(node_ids))
    unknown = [n for n in selection if n not in G]
    missing = [n for n in INITIAL_NODES if n not in selection]

    reason: str | None = None
    if unknown:
        reason = f"Unknown nodes: {', '.join(unknown)}"
    elif missing:
        reason = f"Initial nodes cannot be removed: {', '.join(missing)}"
    else:
        H = G.subgraph(selection)
        if H.number_of_nodes() and not nx.is_connected(H):
            reason = "Selection is not connected"

    H = G.subgraph([n for n in selection if n in G])
    edges = [{"u": u, "v": v, "voltage": d.get("voltage"),
              "weight": round(float(d.get("weight", 0.0)), 4)}
             for u, v, d in H.edges(data=True)]
    adjacent = sorted({
        nb for n in selection if n in G for nb in G.neighbors(n)
        if nb not in selection and not G.nodes[nb].get("border")
        and G.nodes[nb].get("x") is not None
    })
    return {"valid": reason is None, "reason": reason,
            "nodes": selection, "edges": edges, "adjacent": adjacent,
            "tier": tier_for(len(selection))}
