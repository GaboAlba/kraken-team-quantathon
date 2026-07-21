"""Construcción del grafo de la red eléctrica a partir de los datos del ICE.

Pipeline (Tarea A):

    snapshot ICE (GeoJSON)  ->  grafo nacional (NetworkX)  ->  subred  ->  grid_cr.json

- Nodos: subestaciones (capa ``Subestaciones``).
- Aristas: líneas de transmisión (capa ``LineasDeTransmision``). La conectividad
  se deriva del campo ``Circuito`` con formato ``"SubestaciónA-SubestaciónB"``.
- Peso: función intercambiable de ``src.weights`` (default ``kv``).

Ver ``Docs/desiciones.md`` para la justificación de cada decisión.
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

# Alias conocidos: nombre en el campo Circuito -> nombre real de la subestación.
# Se aplican tras la normalización general.
_ALIASES = {
    "garita": "la garita",
}


def normalize_name(s: str | None) -> str:
    """Normaliza un nombre para poder comparar circuitos con subestaciones.

    - pasa a minúsculas y elimina acentos;
    - colapsa espacios;
    - quita el sufijo entre paréntesis (p.ej. ``"(SIEPAC)"``);
    - quita un dígito final de bahía/circuito (``"Colima2"`` -> ``"colima"``).
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)          # quita "(siepac)" y similares
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*\d+$", "", s)           # quita dígito final ("colima2")
    return s.strip()


def _canonical(name: str) -> str:
    """Aplica normalización + alias para obtener el identificador de nodo."""
    n = normalize_name(name)
    return _ALIASES.get(n, n)


def parse_circuit(circuito: str | None) -> tuple[str, str] | None:
    """Devuelve los dos extremos normalizados de un circuito ``"A-B"``.

    Devuelve ``None`` si el formato no es exactamente dos partes separadas por
    un guion (p.ej. ``"SIEPAC"`` o cadenas vacías).
    """
    if not circuito:
        return None
    parts = circuito.split("-")
    if len(parts) != 2:
        return None
    a, b = _canonical(parts[0]), _canonical(parts[1])
    if not a or not b:
        return None
    return a, b


def parse_substations(geojson: dict) -> dict[str, dict]:
    """Extrae los nodos (subestaciones) del GeoJSON como id -> atributos."""
    nodes: dict[str, dict] = {}
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        nombre = props.get("Subestacio")
        if not nombre:
            continue
        node_id = _canonical(nombre)
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None]
        nodes[node_id] = {
            "id": node_id,
            "nombre": nombre,
            "provincia": props.get("Provincia"),
            "canton": props.get("Canton"),
            "x": coords[0],
            "y": coords[1],
            "frontera": False,
        }
    return nodes


def build_national_graph(
    subs_geojson: dict,
    lines_geojson: dict,
    weight_scheme: str = weights.DEFAULT_SCHEME,
) -> tuple[nx.Graph, dict]:
    """Construye el grafo nacional ponderado.

    Los extremos de circuito que no correspondan a una subestación conocida se
    agregan como nodos de *frontera* (``frontera=True``): interconexiones
    internacionales, SIEPAC o cargas industriales. Las líneas paralelas entre el
    mismo par se colapsan sumando sus pesos.

    Devuelve ``(grafo, reporte)`` donde ``reporte`` documenta lo no reconocido.
    """
    weight_fn = weights.SCHEMES[weight_scheme]
    G = nx.Graph()

    nodes = parse_substations(subs_geojson)
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)

    extremos_no_reconocidos: set[str] = set()
    circuitos_ignorados: list[str] = []

    for feat in lines_geojson.get("features", []):
        props = feat.get("properties", {})
        endpoints = parse_circuit(props.get("Circuito"))
        if endpoints is None:
            circuitos_ignorados.append(props.get("Circuito"))
            continue
        u, v = endpoints
        if u == v:
            continue

        # Extremos desconocidos -> nodos de frontera.
        for ep in (u, v):
            if ep not in G:
                G.add_node(ep, id=ep, nombre=ep.title(), provincia=None,
                           canton=None, x=None, y=None, frontera=True)
                extremos_no_reconocidos.add(ep)

        voltaje = props.get("Voltaje")
        length_m = props.get("Shape__Length")
        w = weight_fn(voltaje=voltaje, length_m=length_m)

        if G.has_edge(u, v):
            # Línea paralela: sumamos peso y conservamos la mayor tensión.
            G[u][v]["weight"] += w
            G[u][v]["voltaje"] = max(G[u][v]["voltaje"], voltaje)
        else:
            G.add_edge(u, v, weight=w, voltaje=voltaje,
                       circuito=props.get("Circuito"))

    report = {
        "extremos_no_reconocidos": sorted(extremos_no_reconocidos),
        "circuitos_ignorados": circuitos_ignorados,
        "n_nodos": G.number_of_nodes(),
        "n_aristas": G.number_of_edges(),
    }
    return G, report


def extract_subregion(
    G: nx.Graph,
    region: str | None = None,
    nodes: list[str] | None = None,
    max_nodes: int = 12,
) -> nx.Graph:
    """Extrae una subred conexa de a lo sumo ``max_nodes`` nodos.

    Modos de selección (excluyentes):
      - ``nodes``: lista explícita de ids de subestación;
      - ``region``: provincia (comparada de forma laxa);
      - ninguno (**modo conectividad**, default): todas las subestaciones reales,
        que preserva la malla nacional y sus ciclos.

    En todos los casos se toma el subgrafo inducido, se conserva la **componente
    conexa más grande** y, si excede ``max_nodes``, se recorta con un crecimiento
    tipo BFS desde el nodo de mayor grado ponderado para mantener la conexidad.
    El nodo semilla y los desempates son deterministas (orden alfabético) para
    garantizar reproducibilidad.

    Nota: filtrar por una sola provincia produce subgrafos casi radiales (árboles),
    cuyo Max-Cut es trivial; el modo conectividad se usa por defecto en los
    experimentos porque conserva ciclos (ver ``Docs/desiciones.md``).
    """
    if nodes is not None:
        selected = [n for n in nodes if n in G]
    elif region is not None:
        r = normalize_name(region)
        selected = [
            n for n, d in G.nodes(data=True)
            if d.get("provincia") and normalize_name(d["provincia"]) == r
        ]
    else:
        # Modo conectividad: solo subestaciones reales (sin nodos de frontera).
        selected = [n for n, d in G.nodes(data=True) if not d.get("frontera")]

    H = G.subgraph(selected).copy()
    if H.number_of_nodes() == 0:
        return H

    # Componente conexa más grande (desempate determinista por nodo mínimo).
    largest = max(nx.connected_components(H), key=lambda c: (len(c), min(c)))
    H = H.subgraph(largest).copy()

    if H.number_of_nodes() <= max_nodes:
        return H

    # Recorte conexo: BFS desde el nodo de mayor grado ponderado (determinista).
    seed = max(H.nodes, key=lambda n: (H.degree(n, weight="weight"), n))
    keep = [seed]
    for _, node in nx.bfs_edges(H, seed):
        if len(keep) >= max_nodes:
            break
        keep.append(node)
    return H.subgraph(keep).copy()


def to_json(G: nx.Graph, metadata: dict | None = None) -> dict:
    """Serializa el grafo al esquema de ``grid_cr.json``."""
    nodes = [
        {
            "id": n,
            "nombre": d.get("nombre"),
            "provincia": d.get("provincia"),
            "canton": d.get("canton"),
            "x": d.get("x"),
            "y": d.get("y"),
            "frontera": d.get("frontera", False),
        }
        for n, d in G.nodes(data=True)
    ]
    edges = [
        {
            "u": u,
            "v": v,
            "weight": d.get("weight"),
            "voltaje": d.get("voltaje"),
            "circuito": d.get("circuito"),
        }
        for u, v, d in G.edges(data=True)
    ]
    return {"metadata": metadata or {}, "nodes": nodes, "edges": edges}


def save_graph(G: nx.Graph, path: Path, metadata: dict | None = None) -> Path:
    """Escribe el grafo a disco en formato JSON."""
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
    """Orquesta el pipeline completo y escribe ``data/grid_cr.json``.

    Por defecto (``region=None``) usa el modo conectividad, que produce el cluster
    mallado del Valle Central; ``region="Guanacaste"`` selecciona esa provincia.
    """
    subs = json.loads((raw_dir / "subestaciones.geojson").read_text(encoding="utf-8"))
    lines = json.loads((raw_dir / "lineas.geojson").read_text(encoding="utf-8"))
    fuente = {}
    fuente_path = raw_dir / "fuente.json"
    if fuente_path.exists():
        fuente = json.loads(fuente_path.read_text(encoding="utf-8"))

    G, report = build_national_graph(subs, lines, weight_scheme=weight_scheme)
    sub = extract_subregion(G, region=region, nodes=nodes, max_nodes=max_nodes)

    # Número de ciclos independientes: indicador de cuán "interesante" es el Max-Cut.
    ciclos = (sub.number_of_edges() - sub.number_of_nodes()
              + nx.number_connected_components(sub)) if sub.number_of_nodes() else 0

    metadata = {
        "fuente": fuente.get("fuente"),
        "urls": fuente.get("urls"),
        "fecha_descarga": fuente.get("fecha_descarga"),
        "fecha_construccion": date.today().isoformat(),
        "esquema_peso": weight_scheme,
        "region": region if region is not None else label,
        "modo_seleccion": "nodos" if nodes else ("provincia" if region else "conectividad"),
        "max_nodes": max_nodes,
        "n_nodos": sub.number_of_nodes(),
        "n_aristas": sub.number_of_edges(),
        "ciclos_independientes": ciclos,
        "reporte_nacional": report,
    }
    save_graph(sub, output, metadata)
    return sub


if __name__ == "__main__":
    g = build()
    print(f"Subred: {g.number_of_nodes()} nodos, {g.number_of_edges()} aristas "
          f"-> {DEFAULT_OUTPUT}")
    print("Nodos:", ", ".join(sorted(g.nodes)))
