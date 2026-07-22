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


def test_normalize_plant_name_drops_roman_numerals():
    assert graph.normalize_plant_name("Miravalles I") == "miravalles"
    assert graph.normalize_plant_name("Miravalles III") == "miravalles"
    assert graph.normalize_plant_name("Toro II") == "toro"
    assert graph.normalize_plant_name("Moin II") == "moin"


def test_normalize_plant_name_keeps_non_roman_words():
    # "Solar" and "Pozo" are ordinary words, not bay numerals.
    assert graph.normalize_plant_name("Miravalles Solar") == "miravalles solar"
    assert graph.normalize_plant_name("Boca Pozo") == "boca pozo"
    assert graph.normalize_plant_name("Cachí") == "cachi"


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
# Generator annotation from the plants layer
# --------------------------------------------------------------------------

def _fake_geojson_plants(entries):
    feats = []
    for name, tech, mw in entries:
        feats.append({
            "properties": {"Planta": name, "Tecnologia": tech,
                           "PotenciaEfectivaMW": mw, "EstAct": "Activo"},
            "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
        })
    return {"features": feats}


def test_annotate_generators_direct_match():
    subs = _fake_geojson_subs([("Arenal", "Guanacaste"), ("Liberia", "Guanacaste")])
    lines = _fake_geojson_lines([("Arenal-Liberia", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    plants = _fake_geojson_plants([("Arenal", "Hidroeléctrico", 166)])
    report = graph.annotate_generators(G, plants)
    assert G.nodes["arenal"]["generator"] is True
    assert G.nodes["arenal"]["generation_mw"] == 166
    assert G.nodes["arenal"]["technologies"] == ["Hidroeléctrico"]
    assert G.nodes["liberia"]["generator"] is False
    assert report["matched"] == [{"plant": "Arenal", "node": "arenal",
                                  "technology": "Hidroeléctrico", "mw": 166}]
    assert report["unmatched"] == []


def test_annotate_generators_roman_suffix_aggregates_on_one_node():
    subs = _fake_geojson_subs([("Miravalles", "Guanacaste"), ("Liberia", "Guanacaste")])
    lines = _fake_geojson_lines([("Miravalles-Liberia", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    plants = _fake_geojson_plants([("Miravalles I", "Geotérmico", 55),
                                   ("Miravalles II", "Geotérmico", 55)])
    graph.annotate_generators(G, plants)
    assert G.nodes["miravalles"]["generator"] is True
    assert G.nodes["miravalles"]["generation_mw"] == 110
    assert G.nodes["miravalles"]["technologies"] == ["Geotérmico"]


def test_annotate_generators_reports_unmatched():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Canas", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Canas", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    plants = _fake_geojson_plants([("El General", "Hidroeléctrico", 39)])
    report = graph.annotate_generators(G, plants)
    assert report["matched"] == []
    assert report["unmatched"] == [{"plant": "El General",
                                    "technology": "Hidroeléctrico", "mw": 39}]


def test_to_json_includes_generator_fields():
    subs = _fake_geojson_subs([("Arenal", "Guanacaste"), ("Liberia", "Guanacaste")])
    lines = _fake_geojson_lines([("Arenal-Liberia", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    plants = _fake_geojson_plants([("Arenal", "Hidroeléctrico", 166)])
    graph.annotate_generators(G, plants)
    doc = graph.to_json(G)
    by_id = {n["id"]: n for n in doc["nodes"]}
    assert by_id["arenal"]["generator"] is True
    assert by_id["arenal"]["generation_mw"] == 166
    assert by_id["arenal"]["technologies"] == ["Hidroeléctrico"]
    assert by_id["liberia"]["generator"] is False


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


def test_ice_data_declares_plants_layer():
    from src import ice_data
    assert "plants" in ice_data.LAYERS


has_plants = (RAW / "plants.geojson").exists()
real_plants = pytest.mark.skipif(not (has_snapshot and has_plants),
                                 reason="ICE plants snapshot not available")


@real_plants
def test_real_snapshot_plants_annotation():
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    plants = json.loads((RAW / "plants.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    report = graph.annotate_generators(G, plants)
    # Direct + roman-suffix matches: at least 20 of the 32 ICE plants.
    assert len(report["matched"]) >= 20
    techs = {m["technology"] for m in report["matched"]}
    assert {"Hidroeléctrico", "Geotérmico", "Eólico"} <= techs


@real
def test_real_snapshot_connectivity_is_deterministic():
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    a = graph.extract_subregion(G, region=None, max_nodes=10)
    b = graph.extract_subregion(G, region=None, max_nodes=10)
    assert set(a.nodes) == set(b.nodes)
