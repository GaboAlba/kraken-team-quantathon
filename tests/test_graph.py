"""Tests for the graph construction module (Task A).

They combine small synthetic cases (for pure, deterministic logic) with checks
against the real ICE snapshot in ``data/raw/``.
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
# Name normalization
# --------------------------------------------------------------------------

def test_normalize_strips_accents_and_lowercases():
    assert graph.normalize_name("Liberón") == "liberon"
    assert graph.normalize_name("  Río  Macho ") == "rio macho"


def test_normalize_drops_trailing_bay_digit():
    # Numbered circuits (colima1, colima2) point to the same substation.
    assert graph.normalize_name("Colima1") == graph.normalize_name("Colima2")
    assert graph.normalize_name("Colima2") == "colima"


def test_normalize_drops_siepac_suffix():
    assert graph.normalize_name("Jaco (SIEPAC)") == "jaco"


# --------------------------------------------------------------------------
# Circuit parsing
# --------------------------------------------------------------------------

def test_parse_circuit_two_endpoints():
    assert graph.parse_circuit("Liberia-Papagayo") == ("liberia", "papagayo")


def test_parse_circuit_invalid_format_returns_none():
    assert graph.parse_circuit("SIEPAC") is None
    assert graph.parse_circuit("") is None
    assert graph.parse_circuit(None) is None


# --------------------------------------------------------------------------
# Weight functions
# --------------------------------------------------------------------------

def test_weight_kv_is_the_voltage():
    assert weights.SCHEMES["kv"](voltage=230, length_m=1000) == 230
    assert weights.SCHEMES["kv"](voltage=138, length_m=5000) == 138


def test_weight_kv_normalized_between_0_and_1():
    w = weights.SCHEMES["kv_normalized"](voltage=230, length_m=1000)
    assert w == pytest.approx(1.0)
    assert weights.SCHEMES["kv_normalized"](voltage=138, length_m=1) == pytest.approx(138 / 230)


def test_weight_all_schemes_are_positive():
    for name, fn in weights.SCHEMES.items():
        assert fn(voltage=138, length_m=2500) > 0, name


# --------------------------------------------------------------------------
# Graph construction with synthetic data
# --------------------------------------------------------------------------

def _fake_geojson_subs(names_provinces):
    feats = []
    for i, (name, prov) in enumerate(names_provinces):
        feats.append({
            "properties": {"Subestacio": name, "Provincia": prov, "Canton": "X"},
            "geometry": {"type": "Point", "coordinates": [float(i), float(i)]},
        })
    return {"features": feats}


def _fake_geojson_lines(circuits_voltages):
    feats = []
    for circ, volt in circuits_voltages:
        feats.append({
            "properties": {"Circuito": circ, "Voltaje": volt, "Shape__Length": 1000.0},
            "geometry": {"type": "MultiLineString", "coordinates": [[[0, 0], [1, 1]]]},
        })
    return {"features": feats}


def test_build_national_graph_basic():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste"),
                               ("Canas", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230), ("Canas-Liberia", 138)])
    G, report = graph.build_national_graph(subs, lines)
    assert set(G.nodes) == {"liberia", "papagayo", "canas"}
    assert G.number_of_edges() == 2
    assert G["liberia"]["papagayo"]["weight"] == 230
    assert G.nodes["liberia"]["province"] == "Guanacaste"


def test_parallel_lines_are_collapsed_summing_weight():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230), ("Liberia-Papagayo", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    assert G.number_of_edges() == 1
    assert G["liberia"]["papagayo"]["weight"] == 460


def test_unrecognized_endpoint_is_marked_as_border():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Frontera Nicaragua", 230)])
    G, report = graph.build_national_graph(subs, lines)
    assert G.nodes["liberia"]["border"] is False
    assert G.nodes["frontera nicaragua"]["border"] is True
    assert "frontera nicaragua" in report["unrecognized_endpoints"]


# --------------------------------------------------------------------------
# Subregion extraction
# --------------------------------------------------------------------------

def test_extract_subregion_by_province():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste"),
                               ("Colima", "Alajuela")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230), ("Papagayo-Colima", 138)])
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="Guanacaste")
    assert set(sub.nodes) == {"liberia", "papagayo"}


def test_extract_subregion_keeps_largest_connected_component():
    subs = _fake_geojson_subs([("A", "G"), ("B", "G"), ("C", "G"), ("D", "G")])
    # A-B-C connected; D isolated (no edges within the region).
    lines = _fake_geojson_lines([("A-B", 230), ("B-C", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="G", max_nodes=12)
    assert "d" not in sub.nodes
    assert set(sub.nodes) == {"a", "b", "c"}


def test_extract_subregion_respects_max_nodes():
    subs = _fake_geojson_subs([(c, "G") for c in "ABCDE"])
    lines = _fake_geojson_lines([("A-B", 230), ("B-C", 230), ("C-D", 230), ("D-E", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="G", max_nodes=3)
    assert sub.number_of_nodes() == 3


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def test_to_json_structure(tmp_path):
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    doc = graph.to_json(G, metadata={"region": "Guanacaste", "weight_scheme": "kv"})
    assert doc["metadata"]["region"] == "Guanacaste"
    assert len(doc["nodes"]) == 2
    assert len(doc["edges"]) == 1
    e = doc["edges"][0]
    assert {"u", "v", "weight", "voltage", "circuit"} <= set(e)


# --------------------------------------------------------------------------
# Checks against the real ICE snapshot
# --------------------------------------------------------------------------

RAW = ROOT / "data" / "raw"
has_snapshot = (RAW / "substations.geojson").exists() and (RAW / "lines.geojson").exists()
real = pytest.mark.skipif(not has_snapshot, reason="ICE snapshot not available")


@real
def test_real_snapshot_national_graph():
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, report = graph.build_national_graph(subs, lines)
    # It must include at least the 70 real substations as nodes.
    real_nodes = [n for n, d in G.nodes(data=True) if not d["border"]]
    assert len(real_nodes) >= 70
    assert G.number_of_edges() > 60


@real
def test_real_snapshot_guanacaste_subgrid_is_valid():
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="Guanacaste", max_nodes=12)
    assert 6 <= sub.number_of_nodes() <= 12
    assert nx.is_connected(sub)
    assert sub.number_of_edges() >= sub.number_of_nodes() - 1


@real
def test_real_snapshot_connectivity_mode_preserves_cycles():
    # The default (connectivity) must yield non-trivial instances: with >=8
    # nodes the Max-Cut stops being a tree (at least one independent cycle).
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    for n in (8, 10, 12):
        sub = graph.extract_subregion(G, region=None, max_nodes=n)
        assert nx.is_connected(sub)
        cycles = sub.number_of_edges() - sub.number_of_nodes() + 1
        assert cycles >= 1, f"N={n} came out without cycles"


GUANACASTE_NORTH_EXPECTED = {
    "arenal", "canas", "corobici", "liberia", "miravalles",
    "mogote", "nuevo colon", "pailas", "papagayo",
}


@real
def test_real_snapshot_guanacaste_north_selection():
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, nodes=graph.GUANACASTE_NORTH)
    assert set(sub.nodes) == GUANACASTE_NORTH_EXPECTED
    assert nx.is_connected(sub)
    # The ring Liberia-Pailas-Mogote-Miravalles-Arenal-Corobici-Canas-Liberia.
    assert sub.number_of_edges() - sub.number_of_nodes() + 1 == 1


@real
def test_real_snapshot_build_writes_guanacaste_north(tmp_path):
    out = tmp_path / "grid.json"
    sub = graph.build(nodes=graph.GUANACASTE_NORTH, label="guanacaste_north",
                      max_nodes=len(graph.GUANACASTE_NORTH), output=out)
    import json
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["metadata"]["region"] == "guanacaste_north"
    assert doc["metadata"]["selection_mode"] == "nodes"
    assert {n["id"] for n in doc["nodes"]} == GUANACASTE_NORTH_EXPECTED
    assert sub.number_of_nodes() == 9


@real
def test_real_snapshot_connectivity_is_deterministic():
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    a = graph.extract_subregion(G, region=None, max_nodes=10)
    b = graph.extract_subregion(G, region=None, max_nodes=10)
    assert set(a.nodes) == set(b.nodes)
