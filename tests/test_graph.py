"""Pruebas del módulo de construcción del grafo (Tarea A).

Combinan casos sintéticos pequeños (para lógica pura y determinista) con
verificaciones contra el snapshot real del ICE en ``data/raw/``.
"""

import sys
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import graph
from src import weights


# --------------------------------------------------------------------------
# Normalización de nombres
# --------------------------------------------------------------------------

def test_normalize_quita_acentos_y_minusculas():
    assert graph.normalize_name("Liberón") == "liberon"
    assert graph.normalize_name("  Río  Macho ") == "rio macho"


def test_normalize_quita_digito_final_de_bahia():
    # Los circuitos numerados (colima1, colima2) apuntan a la misma subestación.
    assert graph.normalize_name("Colima1") == graph.normalize_name("Colima2")
    assert graph.normalize_name("Colima2") == "colima"


def test_normalize_quita_sufijo_siepac():
    assert graph.normalize_name("Jaco (SIEPAC)") == "jaco"


# --------------------------------------------------------------------------
# Parseo del circuito
# --------------------------------------------------------------------------

def test_parse_circuit_dos_extremos():
    assert graph.parse_circuit("Liberia-Papagayo") == ("liberia", "papagayo")


def test_parse_circuit_formato_invalido_devuelve_none():
    assert graph.parse_circuit("SIEPAC") is None
    assert graph.parse_circuit("") is None
    assert graph.parse_circuit(None) is None


# --------------------------------------------------------------------------
# Funciones de peso
# --------------------------------------------------------------------------

def test_peso_kv_es_el_voltaje():
    assert weights.SCHEMES["kv"](voltaje=230, length_m=1000) == 230
    assert weights.SCHEMES["kv"](voltaje=138, length_m=5000) == 138


def test_peso_kv_normalizado_entre_0_y_1():
    w = weights.SCHEMES["kv_normalized"](voltaje=230, length_m=1000)
    assert w == pytest.approx(1.0)
    assert weights.SCHEMES["kv_normalized"](voltaje=138, length_m=1) == pytest.approx(138 / 230)


def test_peso_todos_los_esquemas_son_positivos():
    for name, fn in weights.SCHEMES.items():
        assert fn(voltaje=138, length_m=2500) > 0, name


# --------------------------------------------------------------------------
# Construcción del grafo con datos sintéticos
# --------------------------------------------------------------------------

def _fake_geojson_subs(nombres_provincia):
    feats = []
    for i, (nombre, prov) in enumerate(nombres_provincia):
        feats.append({
            "properties": {"Subestacio": nombre, "Provincia": prov, "Canton": "X"},
            "geometry": {"type": "Point", "coordinates": [float(i), float(i)]},
        })
    return {"features": feats}


def _fake_geojson_lines(circuitos_voltaje):
    feats = []
    for circ, volt in circuitos_voltaje:
        feats.append({
            "properties": {"Circuito": circ, "Voltaje": volt, "Shape__Length": 1000.0},
            "geometry": {"type": "MultiLineString", "coordinates": [[[0, 0], [1, 1]]]},
        })
    return {"features": feats}


def test_build_national_graph_basico():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste"),
                               ("Canas", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230), ("Canas-Liberia", 138)])
    G, report = graph.build_national_graph(subs, lines)
    assert set(G.nodes) == {"liberia", "papagayo", "canas"}
    assert G.number_of_edges() == 2
    assert G["liberia"]["papagayo"]["weight"] == 230
    assert G.nodes["liberia"]["provincia"] == "Guanacaste"


def test_lineas_paralelas_se_colapsan_sumando_peso():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230), ("Liberia-Papagayo", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    assert G.number_of_edges() == 1
    assert G["liberia"]["papagayo"]["weight"] == 460


def test_extremo_no_reconocido_se_marca_como_frontera():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Frontera Nicaragua", 230)])
    G, report = graph.build_national_graph(subs, lines)
    assert G.nodes["liberia"]["frontera"] is False
    assert G.nodes["frontera nicaragua"]["frontera"] is True
    assert "frontera nicaragua" in report["extremos_no_reconocidos"]


# --------------------------------------------------------------------------
# Extracción de subred
# --------------------------------------------------------------------------

def test_extract_subregion_por_provincia():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste"),
                               ("Colima", "Alajuela")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230), ("Papagayo-Colima", 138)])
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="Guanacaste")
    assert set(sub.nodes) == {"liberia", "papagayo"}


def test_extract_subregion_toma_componente_conexa_mas_grande():
    subs = _fake_geojson_subs([("A", "G"), ("B", "G"), ("C", "G"), ("D", "G")])
    # A-B-C conexos; D aislado (sin aristas dentro de la región).
    lines = _fake_geojson_lines([("A-B", 230), ("B-C", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="G", max_nodes=12)
    assert "d" not in sub.nodes
    assert set(sub.nodes) == {"a", "b", "c"}


def test_extract_subregion_respeta_max_nodes():
    subs = _fake_geojson_subs([(c, "G") for c in "ABCDE"])
    lines = _fake_geojson_lines([("A-B", 230), ("B-C", 230), ("C-D", 230), ("D-E", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="G", max_nodes=3)
    assert sub.number_of_nodes() == 3


# --------------------------------------------------------------------------
# Serialización
# --------------------------------------------------------------------------

def test_to_json_estructura(tmp_path):
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    doc = graph.to_json(G, metadata={"region": "Guanacaste", "esquema_peso": "kv"})
    assert doc["metadata"]["region"] == "Guanacaste"
    assert len(doc["nodes"]) == 2
    assert len(doc["edges"]) == 1
    e = doc["edges"][0]
    assert {"u", "v", "weight", "voltaje", "circuito"} <= set(e)


# --------------------------------------------------------------------------
# Verificación contra el snapshot real del ICE
# --------------------------------------------------------------------------

RAW = ROOT / "data" / "raw"
has_snapshot = (RAW / "subestaciones.geojson").exists() and (RAW / "lineas.geojson").exists()
real = pytest.mark.skipif(not has_snapshot, reason="snapshot del ICE no disponible")


@real
def test_snapshot_real_grafo_nacional():
    import json
    subs = json.loads((RAW / "subestaciones.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lineas.geojson").read_text(encoding="utf-8"))
    G, report = graph.build_national_graph(subs, lines)
    # Debe incluir al menos las 70 subestaciones reales como nodos.
    reales = [n for n, d in G.nodes(data=True) if not d["frontera"]]
    assert len(reales) >= 70
    assert G.number_of_edges() > 60


@real
def test_snapshot_real_subred_guanacaste_es_valida():
    import json
    subs = json.loads((RAW / "subestaciones.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lineas.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="Guanacaste", max_nodes=12)
    assert 6 <= sub.number_of_nodes() <= 12
    assert nx.is_connected(sub)
    assert sub.number_of_edges() >= sub.number_of_nodes() - 1


@real
def test_snapshot_real_modo_conectividad_conserva_ciclos():
    # El default (conectividad) debe dar instancias no triviales: con >=8 nodos
    # el Max-Cut deja de ser un árbol (al menos un ciclo independiente).
    import json
    subs = json.loads((RAW / "subestaciones.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lineas.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    for n in (8, 10, 12):
        sub = graph.extract_subregion(G, region=None, max_nodes=n)
        assert nx.is_connected(sub)
        ciclos = sub.number_of_edges() - sub.number_of_nodes() + 1
        assert ciclos >= 1, f"N={n} salió sin ciclos"


@real
def test_snapshot_real_conectividad_es_determinista():
    import json
    subs = json.loads((RAW / "subestaciones.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lineas.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    a = graph.extract_subregion(G, region=None, max_nodes=10)
    b = graph.extract_subregion(G, region=None, max_nodes=10)
    assert set(a.nodes) == set(b.nodes)
